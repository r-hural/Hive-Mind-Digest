"""Model monitor (IDEA-302): parsers of the published pages, findings, fetch failures, cache.

The three pages under fixtures/ are small synthetic stand-ins in the published pages' table shapes."""
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from pipeline import db
from pipeline.config import Config
from pipeline.models_monitor import (
    Finding, ModelStatus, PageFormatError, cached_findings, check_models, evaluate, parse_deprecations,
    parse_overview, parse_price_list, stage_usage, write_cache,
)
from pipeline.pricing import PRICES, Price

FIX = Path(__file__).parent / "fixtures"
PRICING = (FIX / "pricing_page.md").read_text(encoding="utf-8")
DEPRECATIONS = (FIX / "deprecations_page.md").read_text(encoding="utf-8")
OVERVIEW = (FIX / "models_overview_page.md").read_text(encoding="utf-8")
TODAY = date(2026, 10, 15)


# --- parsers ---------------------------------------------------------------------------------

def test_price_list_parses_current_models():
    pub = parse_price_list(PRICING)
    assert pub["claude-sonnet-5-5"] == Price(2.0, 10.0, cache_read_x=0.05)
    assert pub["claude-fable-5-1"].cache_read_x == 0.025
    assert pub["claude-sonnet-4-6"] == Price(3.0, 15.0)
    assert "claude-opus-4-1" not in pub                    # retired rows are skipped


def test_price_list_keeps_haiku_5_5_tiers_like_the_table():
    assert parse_price_list(PRICING)["claude-haiku-5-5"] == PRICES["claude-haiku-5-5"]


def test_price_list_without_table_is_a_format_error():
    with pytest.raises(PageFormatError):
        parse_price_list("# Pricing\n\nNo table here.\n")


def test_deprecations_parse_states_and_dates():
    st = parse_deprecations(DEPRECATIONS)
    assert st["claude-sonnet-4-5-20250929"] == ModelStatus("Deprecated", date(2026, 8, 3), date(2026, 12, 1), False)
    assert st["claude-haiku-4-5-20251001"] == ModelStatus("Active", None, date(2026, 11, 2), True)
    assert st["claude-mythos-preview"].retires_on is None
    with pytest.raises(PageFormatError):
        parse_deprecations("nothing")


def test_overview_lists_current_then_legacy_ids():
    ids = parse_overview(OVERVIEW)
    assert ids[:4] == ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
    assert "claude-sonnet-4-6" in ids and "claude-haiku-4-5" in ids
    assert len(ids) == len(set(ids))
    with pytest.raises(PageFormatError):
        parse_overview("nothing")


# --- findings ----------------------------------------------------------------------------------

def _eval(models, provider="anthropic_api", *, published=None, statuses=None, listed=None, account=None,
          usage=None, today=TODAY):
    return evaluate(models, provider, PRICES,
                    parse_price_list(PRICING) if published is None else published,
                    parse_deprecations(DEPRECATIONS) if statuses is None else statuses,
                    parse_overview(OVERVIEW) if listed is None else listed,
                    account, usage or {}, today)


def _levels(fs, level):
    return [f for f in fs if f.level == level]


def test_cheaper_newer_model_is_suggested_with_notebook_cost():
    fs = _eval({"merge": "claude-sonnet-4-6"}, usage={"merge": (1_000_000.0, 100_000.0)})
    s = [f for f in _levels(fs, "suggest") if f.stage == "merge"]
    assert len(s) == 1
    # now: 1M * $3 + 0.1M * $15 = $4.50; Sonnet 5.5 with the 1.3 tokenizer factor: 1.3M * $2 + 0.13M * $10 = $3.90
    assert "claude-sonnet-5-5" in s[0].text and "$3.90" in s[0].text and "$4.50" in s[0].text
    assert "tokenizer" in s[0].text and "--merge-model claude-sonnet-5-5" in s[0].text


def test_dearer_newer_model_is_a_warning():
    prices = dict(PRICES, **{"claude-sonnet-5-5": Price(9.0, 45.0)})
    pub = dict(parse_price_list(PRICING), **{"claude-sonnet-5-5": Price(9.0, 45.0)})
    fs = evaluate({"merge": "claude-sonnet-4-6"}, "anthropic_api", prices, pub, parse_deprecations(DEPRECATIONS),
                  parse_overview(OVERVIEW), None, {"merge": (1_000_000.0, 100_000.0)}, TODAY)
    w = [f for f in _levels(fs, "warn") if "claude-sonnet-5-5" in f.text]
    assert w and "more" in w[0].text


def test_no_usage_history_says_so():
    fs = _eval({"analyze": "claude-haiku-4-5-20251001"})
    s = [f for f in fs if f.stage == "analyze" and "claude-haiku-5-5" in f.text]
    assert s and "no usage history" in s[0].text and s[0].level == "suggest"


def test_current_models_have_no_candidate():
    fs = _eval({"analyze": "claude-haiku-5-5", "merge": "claude-sonnet-5-5"})
    assert not _levels(fs, "suggest") and not _levels(fs, "warn")
    assert _levels(fs, "ok")


def test_deprecated_and_soon_retiring_models_warn():
    fs = _eval({"merge": "claude-sonnet-4-5-20250929"})
    assert any("deprecated" in f.text and "2026-12-01" in f.text for f in _levels(fs, "warn"))
    statuses = {"claude-haiku-4-5-20251001": ModelStatus("Active", None, date(2026, 11, 1), False)}
    fs = _eval({"analyze": "claude-haiku-4-5-20251001"}, statuses=statuses)
    assert any("retires on 2026-11-01" in f.text for f in _levels(fs, "warn"))


def test_not_sooner_than_date_alone_is_no_warning():
    # the fixture's "Not sooner than November 2, 2026" is within RETIRING_SOON_DAYS of TODAY
    fs = _eval({"analyze": "claude-haiku-4-5-20251001"})
    assert not any("retire" in f.text for f in _levels(fs, "warn"))


def test_retired_model_warns():
    fs = _eval({"merge": "claude-sonnet-4-20250514"})
    assert any("retired" in f.text for f in _levels(fs, "warn"))


def test_published_price_difference_is_reported():
    pub = dict(parse_price_list(PRICING), **{"claude-sonnet-5-5": Price(2.5, 12.0, cache_read_x=0.05)})
    fs = _eval({"merge": "claude-sonnet-5-5"}, published=pub)
    w = [f for f in _levels(fs, "warn") if "price list differs" in f.text]
    assert w and "claude-sonnet-5-5" in w[0].text and "pipeline/pricing.py" in w[0].text


def test_matching_price_list_reports_no_difference():
    fs = _eval({"analyze": "claude-haiku-5-5", "merge": "claude-sonnet-5-5"})
    assert not any("differs" in f.text for f in fs)


def test_old_price_table_warns():
    fs = _eval({"merge": "claude-sonnet-5-5"}, today=date(2027, 1, 1))
    assert any("price table" in f.text and "days old" in f.text for f in _levels(fs, "warn"))


def test_unpriced_configured_model_warns_on_the_api():
    fs = evaluate({"merge": "claude-sonnet-9-9"}, "anthropic_api", PRICES, None, None, None, None, {}, TODAY)
    assert any("no price" in f.text for f in _levels(fs, "warn"))


def test_claude_cli_gets_newer_models_without_prices():
    fs = _eval({"merge": "claude-sonnet-4-6"}, provider="claude_cli",
               usage={"merge": (1_000_000.0, 100_000.0)}, today=date(2027, 6, 1))
    s = [f for f in _levels(fs, "suggest") if f.stage == "merge"]
    assert s and "claude-sonnet-5-5" in s[0].text and "$" not in s[0].text
    assert not any("price" in f.text for f in fs)             # no price table age, no list differences


def test_candidate_missing_from_the_account_is_reported():
    fs = _eval({"merge": "claude-sonnet-4-6"}, account={"claude-sonnet-4-6"})
    assert any("not available to this account" in f.text for f in fs if f.stage == "merge")


def test_unreadable_inputs_are_skipped_not_fatal():
    fs = evaluate({"merge": "claude-sonnet-4-6"}, "anthropic_api", PRICES, None, None, None, None, {}, TODAY)
    sk = _levels(fs, "skipped")
    assert len(sk) == 1 and "price list" in sk[0].text and "model status" in sk[0].text
    assert not _levels(fs, "ok")                               # never "up to date" when unchecked


# --- fetch, usage, cache -------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    c = Config(home=str(tmp_path), llm={"max_parallel": 2}, run={"budget_usd": 0})
    db.init_db(c.db_path)
    return c


def _pages(url, timeout=10):
    return {"pricing": PRICING, "deprecations": DEPRECATIONS, "overview": OVERVIEW}[
        "pricing" if "pricing" in url else "deprecations" if "deprecations" in url else "overview"]


def test_check_models_offline_never_raises(cfg):
    def down(url, timeout=10):
        raise OSError("network is unreachable")
    fs = check_models(cfg, fetch=down, api_key=None, today=TODAY)
    assert _levels(fs, "skipped") and not _levels(fs, "ok")


def test_check_models_without_key_notes_the_account_check(cfg):
    fs = check_models(cfg, fetch=_pages, api_key=None, today=TODAY)
    assert any(f.level == "note" and "account" in f.text for f in fs)


def test_check_models_with_key_uses_the_account_list(cfg):
    seen = []
    fs = check_models(cfg, fetch=_pages, api_key="k", list_account=lambda key: seen.append(key) or {"x"},
                      today=TODAY)
    assert seen == ["k"] and not any(f.level == "note" for f in fs)


def test_stage_usage_averages_recent_runs(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    for _ in range(2):
        rid = db.create_run(cfg.db_path, nid, provider="anthropic_api", budget_usd=0, stages="merge")
        db.add_usage_event(cfg.db_path, run_id=rid, notebook_id=nid, source_id=None, stage="merge:section",
                     provider="anthropic_api", model="m", input_tokens=1000, output_tokens=100,
                     cache_read_tokens=0, cache_write_tokens=0, cost_usd=0.0, duration_ms=1)
        db.add_usage_event(cfg.db_path, run_id=rid, notebook_id=nid, source_id=None, stage="merge:check",
                     provider="anthropic_api", model="m", input_tokens=500, output_tokens=50,
                     cache_read_tokens=0, cache_write_tokens=0, cost_usd=0.0, duration_ms=1)
    assert stage_usage(cfg.db_path, "merge") == (1500.0, 150.0, 2.0)
    assert stage_usage(cfg.db_path, "analyze") is None


def test_cache_round_trip_and_expiry(cfg):
    fs = [Finding("suggest", "merge", "x"), Finding("warn", None, "y")]
    now = datetime(2026, 10, 15, 12, 0)
    write_cache(cfg, fs, now=now)
    assert cached_findings(cfg, now=now + timedelta(hours=23)) == fs
    assert cached_findings(cfg, now=now + timedelta(hours=25)) is None
    assert json.loads((Path(cfg.home) / "cache" / "model-monitor.json").read_text())["findings"][0]["level"] == "suggest"


def test_cache_missing_or_corrupt_is_none(cfg):
    assert cached_findings(cfg) is None
    p = Path(cfg.home) / "cache" / "model-monitor.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    assert cached_findings(cfg) is None


def test_stage_usage_ignores_claude_cli_runs(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    for provider, tokens in (("anthropic_api", 1000), ("claude_cli", 99_000)):
        rid = db.create_run(cfg.db_path, nid, provider=provider, budget_usd=0, stages="analyze")
        db.add_usage_event(cfg.db_path, run_id=rid, notebook_id=nid, source_id=None, stage="analyze",
                           provider=provider, model="m", input_tokens=tokens, output_tokens=10,
                           cache_read_tokens=0, cache_write_tokens=0, cost_usd=0.0, duration_ms=1)
    assert stage_usage(cfg.db_path, "analyze") == (1000.0, 10.0, 1.0)


# --- final review fixes ------------------------------------------------------------------------

def test_http_protocol_errors_are_skipped_not_raised(cfg):
    import http.client
    def cut(url, timeout=10):
        raise http.client.IncompleteRead(b"partial")
    fs = check_models(cfg, fetch=cut, api_key=None, today=TODAY)
    assert _levels(fs, "skipped")


def test_long_prompt_tier_is_chosen_per_request_not_per_notebook():
    # 420k input over 40 calls is ~10k per request: Haiku 5.5 base tier, not the >100k tier
    fs = _eval({"analyze": "claude-haiku-4-5-20251001"}, usage={"analyze": (420_000.0, 60_000.0, 40.0)})
    s = [f for f in fs if f.stage == "analyze" and "claude-haiku-5-5" in f.text][0]
    # 1.3 * (420k * $0.10 + 60k * $0.50) / 1M = $0.0936
    assert "$0.09 per notebook" in s.text


def test_per_mtok_comparison_applies_the_tokenizer_factor():
    fs = _eval({"merge": "claude-opus-4-6"})            # no usage history; Opus 5.5 $4/$20 x1.3 > $5/$25
    f = [f for f in fs if f.stage == "merge" and "claude-opus-5-5" in f.text][0]
    assert f.level == "warn" and "tokenizer" in f.text


def test_prices_match_only_when_every_model_is_listed():
    pub = {k: v for k, v in parse_price_list(PRICING).items() if k != "claude-sonnet-5-5"}
    fs = _eval({"analyze": "claude-haiku-5-5", "merge": "claude-sonnet-5-5"}, published=pub)
    assert not any("prices match" in f.text for f in fs)
    assert any("claude-sonnet-5-5" in f.text and "not in the published price list" in f.text for f in fs)


def test_cache_is_ignored_when_the_models_changed(cfg):
    write_cache(cfg, [Finding("suggest", "merge", "x")])
    assert cached_findings(cfg) is not None
    cfg.llm.models["merge"] = "claude-sonnet-9-9"
    assert cached_findings(cfg) is None


def test_set_stage_model_keeps_a_utf8_bom(tmp_path):
    from pipeline.models_monitor import set_stage_model
    p = tmp_path / "config.yaml"
    p.write_bytes("﻿llm:\n  models:\n    analyze: claude-haiku-4-5\n    merge: claude-sonnet-4-6\n".encode("utf-8"))
    assert set_stage_model(str(p), "analyze", "claude-haiku-5-5") == "claude-haiku-4-5"
    assert p.read_bytes().decode("utf-8") == "﻿llm:\n  models:\n    analyze: claude-haiku-5-5\n    merge: claude-sonnet-4-6\n"


def test_stage_usage_counts_batched_calls(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    rid = db.create_run(cfg.db_path, nid, provider="anthropic_api", budget_usd=0, stages="analyze")
    db.add_usage_event(cfg.db_path, run_id=rid, notebook_id=nid, source_id=None, stage="analyze",
                       provider="anthropic_batch", model="m", input_tokens=800, output_tokens=80,
                       cache_read_tokens=0, cache_write_tokens=0, cost_usd=0.0, duration_ms=1)
    assert stage_usage(cfg.db_path, "analyze") == (800.0, 80.0, 1.0)
