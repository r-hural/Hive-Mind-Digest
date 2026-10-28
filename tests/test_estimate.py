import json

import pytest

from pipeline import db
from pipeline.config import load_config
from pipeline.estimate import DEFAULTS, CHARS_PER_SECOND, calibrate, estimate_notebook


def _cfg(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text(f"home: {tmp_path.as_posix()}/h\n", encoding="utf-8")
    cfg = load_config(str(p)); db.init_db(cfg.db_path)
    return cfg


def test_estimate_without_history_or_transcripts(tmp_path):
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    db.add_source(cfg.db_path, nid, "aaaaaaaaaaa", "u", duration_seconds=600)
    db.add_source(cfg.db_path, nid, "bbbbbbbbbbb", "u2")                       # no duration, no transcript
    e = estimate_notebook(cfg, nid, analyze_model="claude-haiku-5-5", merge_model="claude-sonnet-5-5", batch=True)
    assert e.sources == 2 and e.sources_to_analyze == 2 and e.estimated_chars
    assert e.chars == int(600 * CHARS_PER_SECOND)                                 # the undated source adds 0
    assert e.from_duration and e.unknown_sources == 1
    assert e.calibration == DEFAULTS and 0 < e.low <= e.analyze_cost + e.merge_cost <= e.high


def _transcribed(cfg, nid, tmp_path, vid, chars, analyzed=False):
    sid = db.add_source(cfg.db_path, nid, vid, "u-" + vid, duration_seconds=600)
    p = tmp_path / f"{vid}.json"
    p.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "x" * (chars - 1)}]}), encoding="utf-8")
    db.update_transcribe_status(cfg.db_path, sid, "done", transcript_path=str(p))
    if analyzed:
        db.update_analyze_status(cfg.db_path, sid, "done", analysis_json='{"summary": "s", "key_claims": []}')
    return sid


def test_analyzed_sources_cost_no_analyze_but_still_merge(tmp_path):
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    _transcribed(cfg, nid, tmp_path, "aaaaaaaaaaa", 10000, analyzed=True)
    e = estimate_notebook(cfg, nid, analyze_model="claude-haiku-5-5", merge_model="claude-sonnet-5-5", batch=False)
    assert e.sources_to_analyze == 0 and e.analyze_cost == 0 and e.merge_cost > 0 and not e.estimated_chars
    assert e.chars == 10000


def _seed_merge(cfg, nid, run, merged_sources):
    source_map = {f"S{i + 1}": i + 1 for i in range(merged_sources)}
    db.create_merge(cfg.db_path, nid, run, "m", "{}", json.dumps(source_map), "{}", "r.md")


A_MODEL, M_MODEL = "claude-haiku-5-5", "claude-sonnet-5-5"


def _seed_run(cfg, nid, *, sources=14, analyze_out=56000, section=(250000, 40000), direct=(36000, 3000),
              provider="anthropic_api", analyze_model=A_MODEL, merge_model=M_MODEL, merged_sources=None,
              status="done", with_merge=True):
    """A finished run: `sources` analyze calls (one per source), one section, check, outline and verify call."""
    run = db.create_run(cfg.db_path, nid, provider=provider, budget_usd=5, stages="analyze,merge")
    db.finish_run(cfg.db_path, run, status)
    if with_merge:
        _seed_merge(cfg, nid, run, sources if merged_sources is None else merged_sources)
    ev = dict(run_id=run, notebook_id=nid, provider=provider, cache_read_tokens=0, cache_write_tokens=0,
              cost_usd=0.0, duration_ms=1)
    for _ in range(sources):
        db.add_usage_event(cfg.db_path, source_id=None, stage="analyze", model=analyze_model,
                           input_tokens=1000, output_tokens=analyze_out // sources, **ev)
    for stage, (i, o) in (("merge:section", (section[0] // 2, section[1] // 2)),
                          ("merge:check", (section[0] - section[0] // 2, section[1] - section[1] // 2)),
                          ("merge:outline", (direct[0] // 2, direct[1] // 2)),
                          ("merge:verify", (direct[0] - direct[0] // 2, direct[1] - direct[1] // 2))):
        db.add_usage_event(cfg.db_path, source_id=None, stage=stage, model=merge_model,
                           input_tokens=i, output_tokens=o, **ev)
    return run


def _tokens(cal_merge, analyze_out, sources):
    from pipeline.estimate import merge_tokens
    return merge_tokens(cal_merge, analyze_out, sources)


def test_recent_runs_split_merge_calls_and_name_provider_and_models(tmp_path):
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    _seed_run(cfg, nid, section=(1000, 200), direct=(300, 40))
    r = db.recent_runs_with_merge(cfg.db_path, 5)[0]
    assert (r["section_in"], r["section_out"], r["direct_in"], r["direct_out"]) == (1000, 200, 300, 40)
    assert (r["merge_in"], r["merge_out"]) == (1300, 240)
    assert (r["provider"], r["analyze_model"], r["merge_model"]) == ("anthropic_api", A_MODEL, M_MODEL)


def test_calibrate_without_matching_history_is_the_built_in(tmp_path):
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    assert calibrate(cfg.db_path, analyze_model=A_MODEL, merge_model=M_MODEL) == DEFAULTS
    _seed_run(cfg, nid, provider="claude_cli", section=(10, 900000))            # subscription: tokens as cache writes
    _seed_run(cfg, nid, merge_model="claude-sonnet-4-5", section=(10, 900000))   # another merge model
    _seed_run(cfg, nid, analyze_model="claude-haiku-4-5", section=(10, 900000))  # another analyze model
    _seed_run(cfg, nid, status="failed", section=(10, 900000))
    _seed_run(cfg, nid, with_merge=False, section=(10, 900000))                 # merge failed, no stored row
    _seed_run(cfg, nid, sources=2, merged_sources=15, section=(10, 900000))     # notebook add + resume
    assert calibrate(cfg.db_path, analyze_model=A_MODEL, merge_model=M_MODEL) == DEFAULTS


def test_calibrate_blends_a_few_runs_with_the_built_in(tmp_path):
    """One to three matching runs scale each built-in line by the median actual/built-in ratio,
    weighted by runs/4: two runs at twice the built-in tokens give 1.5x."""
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    for sources, a_out in ((14, 56000), (6, 20000)):
        t = _tokens(DEFAULTS.merge, a_out // sources * sources, sources)
        _seed_run(cfg, nid, sources=sources, analyze_out=a_out,
                  section=(round(2 * t["section_in"]), round(2 * t["section_out"])),
                  direct=(round(2 * t["direct_in"]), round(2 * t["direct_out"])))
    cal = calibrate(cfg.db_path, analyze_model=A_MODEL, merge_model=M_MODEL)
    assert cal.from_history and cal.history_runs == 2 and cal.spread == DEFAULTS.spread
    got, want = _tokens(cal.merge, 30000, 10), _tokens(DEFAULTS.merge, 30000, 10)
    for k in want:
        assert got[k] == pytest.approx(1.5 * want[k], rel=1e-3), k


def test_calibrate_refits_on_four_or_more_runs(tmp_path):
    """Four matching runs on an exact line: least squares recovers it, and the range is the
    leave-one-out spread of actual/estimate, never narrower than the built-in one."""
    import math
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    for sources, a_out in ((14, 56000), (6, 20000), (42, 168000), (13, 80000)):
        x = a_out / math.sqrt(sources)
        _seed_run(cfg, nid, sources=sources, analyze_out=a_out, section=(round(50000 + 15 * x), round(8000 + 2 * x)),
                  direct=(round(1000 + 3 * x), round(3000 + 0.1 * x)))
    cal = calibrate(cfg.db_path, analyze_model=A_MODEL, merge_model=M_MODEL)
    assert cal.from_history and cal.history_runs == 4
    assert cal.merge.section_in == pytest.approx((50000, 15), rel=1e-3)
    assert cal.merge.direct_out == pytest.approx((3000, 0.1), rel=1e-2)
    assert cal.spread == DEFAULTS.spread


def test_merge_tokens_grow_with_the_square_root_of_the_source_count():
    """Same analyze output per source: 4x the sources is about 2x the per-call content, not 4x."""
    small, big = _tokens(DEFAULTS.merge, 10 * 4000, 10), _tokens(DEFAULTS.merge, 40 * 4000, 40)
    assert 1.2 < big["section_in"] / small["section_in"] < 2.0
    assert all(v >= 0 for v in _tokens(DEFAULTS.merge, 10, 1).values())


def test_batch_halves_only_the_batched_calls(tmp_path):
    """Analyze and the section/check merge calls go through Message Batches; outline and verify run
    direct at full price."""
    from pipeline.pricing import BATCH_DISCOUNT
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    _transcribed(cfg, nid, tmp_path, "aaaaaaaaaaa", 20000)
    kw = dict(analyze_model=A_MODEL, merge_model=M_MODEL)
    full = estimate_notebook(cfg, nid, batch=False, **kw)
    half = estimate_notebook(cfg, nid, batch=True, **kw)
    assert half.analyze_cost == pytest.approx(BATCH_DISCOUNT * full.analyze_cost)
    assert BATCH_DISCOUNT * full.merge_cost < half.merge_cost < full.merge_cost
    assert half.minutes == DEFAULTS.minutes_batch and full.minutes == DEFAULTS.minutes_direct_per_source * 1 + 5


def test_range_brackets_the_estimate(tmp_path):
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    _transcribed(cfg, nid, tmp_path, "aaaaaaaaaaa", 20000)
    e = estimate_notebook(cfg, nid, analyze_model=A_MODEL, merge_model=M_MODEL, batch=True)
    total = e.analyze_cost + e.merge_cost
    assert 0.7 * total < e.low < total < e.high < 1.5 * total


def test_force_estimates_a_full_reanalysis(tmp_path):
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    _transcribed(cfg, nid, tmp_path, "aaaaaaaaaaa", 10000, analyzed=True)
    kw = dict(analyze_model=A_MODEL, merge_model=M_MODEL, batch=False)
    plain, forced = estimate_notebook(cfg, nid, **kw), estimate_notebook(cfg, nid, force=True, **kw)
    assert plain.sources_to_analyze == 0 and plain.analyze_cost == 0
    assert forced.sources_to_analyze == 1 and forced.analyze_cost > 0 and forced.merge_cost == plain.merge_cost


def test_long_prompt_tier_is_chosen_per_source(tmp_path):
    """Two sources of 150k chars each stay under the 100k-token tier; one 300k-char source does not."""
    from pipeline.pricing import estimate_cost
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    _transcribed(cfg, nid, tmp_path, "aaaaaaaaaaa", 150_000)
    _transcribed(cfg, nid, tmp_path, "bbbbbbbbbbb", 150_000)
    e = estimate_notebook(cfg, nid, analyze_model="claude-haiku-5-5", merge_model="claude-sonnet-5-5", batch=False)
    c = DEFAULTS
    one = estimate_cost("claude-haiku-5-5", int(c.analyze_in_base + c.analyze_in_per_char * 150_000),
                        int(c.analyze_out_base + c.analyze_out_per_char * 150_000))
    assert abs(e.analyze_cost - 2 * one) < 1e-3


def test_unreadable_transcript_falls_back_to_duration(tmp_path):
    from pipeline.estimate import transcript_chars
    bad = tmp_path / "bad.json"; bad.write_text("{not json", encoding="utf-8")
    odd = tmp_path / "odd.json"; odd.write_text("[1, 2]", encoding="utf-8")
    seg = tmp_path / "seg.json"; seg.write_text('{"segments": [{"text": 5}]}', encoding="utf-8")
    assert transcript_chars(str(bad)) is None and transcript_chars(str(odd)) is None
    assert transcript_chars(str(seg)) is None and transcript_chars(str(tmp_path)) is None   # a directory
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    sid = db.add_source(cfg.db_path, nid, "aaaaaaaaaaa", "u", duration_seconds=600)
    db.update_transcribe_status(cfg.db_path, sid, "done", transcript_path=str(bad))
    e = estimate_notebook(cfg, nid, analyze_model="claude-haiku-5-5", merge_model="claude-sonnet-5-5", batch=False)
    assert e.estimated_chars and e.chars == int(600 * CHARS_PER_SECOND)


def test_estimate_notebook_has_no_sources_parameter():
    import inspect
    assert "sources" not in inspect.signature(estimate_notebook).parameters


def test_retried_analyze_calls_count_each_source_once(tmp_path):
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    sids = [db.add_source(cfg.db_path, nid, f"v{i:010d}", f"u{i}") for i in range(3)]
    run = db.create_run(cfg.db_path, nid, provider="anthropic_api", budget_usd=5, stages="analyze,merge")
    db.finish_run(cfg.db_path, run, "done")
    _seed_merge(cfg, nid, run, 3)
    for sid in sids + [sids[0], sids[0]]:                      # two retries on the first source
        db.add_usage_event(cfg.db_path, run_id=run, notebook_id=nid, source_id=sid, stage="analyze",
                           provider="anthropic_api", model="m", input_tokens=1000, output_tokens=1000,
                           cache_read_tokens=0, cache_write_tokens=0, cost_usd=0.0, duration_ms=1)
    db.add_usage_event(cfg.db_path, run_id=run, notebook_id=nid, source_id=None, stage="merge:section",
                       provider="anthropic_api", model="m", input_tokens=20000, output_tokens=3000,
                       cache_read_tokens=0, cache_write_tokens=0, cost_usd=0.0, duration_ms=1)
    assert db.recent_runs_with_merge(cfg.db_path, 5)[0]["analyze_calls"] == 3


def test_untranscribed_documents_are_sized_by_kind(tmp_path):
    from pipeline.estimate import PDF_BYTES_PER_CHAR, WEB_PAGE_CHARS
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    notes = tmp_path / "notes.md"; notes.write_bytes(b"x" * 5000)
    paper = tmp_path / "paper.pdf"; paper.write_bytes(b"x" * 100_000)
    db.add_source(cfg.db_path, nid, "text-0000000000000001", notes.resolve().as_uri(), kind="text")
    db.add_source(cfg.db_path, nid, "pdf-0000000000000002", paper.resolve().as_uri(), kind="pdf")
    db.add_source(cfg.db_path, nid, "web-0000000000000003", "https://e.org/a", kind="web")
    e = estimate_notebook(cfg, nid, analyze_model="claude-haiku-5-5", merge_model="claude-sonnet-5-5", batch=False)
    assert e.chars == 5000 + 100_000 // PDF_BYTES_PER_CHAR + WEB_PAGE_CHARS
    assert e.estimated_chars and not e.from_duration and e.unknown_sources == 0
    assert e.analyze_cost > 0 and e.sources_to_analyze == 3


def test_sources_of_unknown_size_are_counted(tmp_path):
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    db.add_source(cfg.db_path, nid, "media-0000000000000001", (tmp_path / "talk.mp3").as_uri(), kind="media")
    db.add_source(cfg.db_path, nid, "pdf-0000000000000002", "https://e.org/paper.pdf", kind="pdf")   # remote
    db.add_source(cfg.db_path, nid, "text-0000000000000003", (tmp_path / "gone.md").as_uri(), kind="text")
    e = estimate_notebook(cfg, nid, analyze_model="claude-haiku-5-5", merge_model="claude-sonnet-5-5", batch=False)
    assert e.unknown_sources == 3 and e.chars == 0 and e.analyze_cost == 0 and not e.from_duration


def test_reruns_of_one_notebook_blend_instead_of_fitting_a_slope(tmp_path):
    """Four re-runs of one notebook vary only a few percent in x: a slope fitted to that is noise,
    so the built-in lines are blended toward them instead."""
    from pipeline.estimate import blend_merge
    cfg = _cfg(tmp_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    for a_out, sec_in in ((54000, 200000), (56000, 260000), (57000, 230000), (58000, 300000)):
        _seed_run(cfg, nid, sources=14, analyze_out=a_out, section=(sec_in, 40000))
    cal = calibrate(cfg.db_path, analyze_model=A_MODEL, merge_model=M_MODEL)
    assert cal.history_runs == 4 and cal.spread == DEFAULTS.spread
    assert cal.merge == blend_merge(db.recent_runs_with_merge(cfg.db_path, 10))


def test_fit_needs_two_sizes_and_a_30_percent_spread_in_x():
    from pipeline.estimate import fit_merge
    def run(n, a_out):
        return dict(analyze_out=a_out, merged_sources=n, section_in=a_out * 5, section_out=a_out,
                    direct_in=a_out, direct_out=1000)
    assert fit_merge([run(14, 50000), run(14, 80000), run(14, 60000), run(14, 70000)]) is None  # one size
    assert fit_merge([run(14, 50000), run(13, 52000), run(14, 51000), run(13, 50000)]) is None  # x within 30%
    assert fit_merge([run(14, 50000), run(6, 20000), run(42, 170000), run(13, 52000)]) is not None
