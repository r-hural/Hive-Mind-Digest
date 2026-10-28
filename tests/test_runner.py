from pipeline.sources import KINDS
from tests.helpers import yt_ref
import _thread
import os
import threading
import time
from unittest.mock import MagicMock, call

import pytest
from pipeline import db
from pipeline.config import Config, ConfigError
import pipeline.runner as pipeline_runner
from pipeline.llm.base import LLMFatalError, interrupt_requested, request_interrupt, wait_interrupt
from pipeline.llm.client import BudgetExceeded
from pipeline.runner import Runner, STAGES, timing_report, usage_report
from pipeline.transcription.base import TranscriptionError
from pipeline.llm.anthropic_batch import AnthropicBatchProvider
from tests.helpers import FakeBatchClient, ScriptedProvider, StubTranscriber, fake_metadata, scripted_answer

VIDS = ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"]


@pytest.fixture
def cfg(tmp_path):
    c = Config(home=str(tmp_path), llm={"max_parallel": 2}, run={"budget_usd": 0})
    db.init_db(c.db_path)
    return c


def _urls_file(tmp_path, vids=VIDS):
    f = tmp_path / "urls.txt"
    f.write_text("".join(f"https://youtu.be/{v}\n" for v in vids), encoding="utf-8")
    return str(f)


def _runner(cfg, provider=None, **kw):
    return Runner(cfg, provider=provider or ScriptedProvider(), transcribers=[StubTranscriber()],
                  metadata_fn=fake_metadata, log=lambda m: None, **kw)


def test_full_run_produces_report_and_run_row(cfg, tmp_path):
    r = _runner(cfg)
    nb = r.create_notebook("Acme SEO basics")
    assert nb["slug"] == "acme-seo-basics"
    result = r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path))
    assert result.status == "done" and result.report_path and os.path.exists(result.report_path)
    assert result.report_path.replace("\\", "/").endswith("domains/general/vault/acme-seo-basics/reports/"
                                                          + os.path.basename(result.report_path))
    sources = db.get_notebook_sources(cfg.db_path, nb["id"])
    assert [s["analyze_status"] for s in sources] == ["done"] * 3 and all(s["upload_date"] for s in sources)
    run = db.get_run(cfg.db_path, result.run_id)
    assert run["status"] == "done" and run["provider"] == "anthropic_api" and run["stages"] == ",".join(STAGES)
    report = open(result.report_path, encoding="utf-8").read()
    # ScriptedProvider cites segment 1; the worker pads it to segments 0-2, so links open at 0s
    assert report.count("&t=") == 3 and "&t=0s" in report and "First takeaway" in report
    summary = usage_report(cfg.db_path, result.run_id)
    assert "analyze" in summary and "merge:section" in summary and "total" in summary.lower()
    assert "merge:check" in summary   # one citation-check call per section, under its own stage label


def test_full_run_stores_segment_mode_quotes_that_pass_eval_assertion_5(cfg, tmp_path):
    # IDEA-294 end to end: ScriptedProvider points at segment 1, code fills quote and anchor, the
    # report renders it, and eval assertion 5 holds by construction.
    from pipeline.anchors import QuoteLocator
    from pipeline.evals import quote_at_anchor
    from pipeline.schemas import Analysis

    r = _runner(cfg)
    nb = r.create_notebook("Segment quotes")
    result = r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path))
    assert result.status == "done"
    report = open(result.report_path, encoding="utf-8").read()
    for src in db.get_notebook_sources(cfg.db_path, nb["id"]):
        a = Analysis.model_validate_json(src["analysis_json"])
        flat = StubTranscriber().transcribe(yt_ref(src["video_id"]), ["en"]).flatten()
        assert a.quote_mode == "segments" and len(a.key_claims) == 1 and a.dropped_quotes == []
        claim = a.key_claims[0]
        # segment 1 padded by QUOTE_PAD_SEGMENTS to segments 0-2 (the whole stub transcript)
        assert claim.quote == " ".join(flat.segment_texts()[0:3]) and claim.anchor.t_start == 0.0
        assert quote_at_anchor(QuoteLocator(flat), flat, claim)
    assert report.count("&t=0s") == 3   # every cited claim links to its padded range's start


def test_slug_uniqueness(cfg):
    r = _runner(cfg)
    assert r.create_notebook("Same")["slug"] == "same"
    assert r.create_notebook("Same")["slug"] == "same-2"


def test_create_notebook_log_line_never_echoes_the_topic(cfg):
    lines = []
    r = Runner(cfg, provider=ScriptedProvider(), transcribers=[StubTranscriber()], metadata_fn=fake_metadata,
              log=lines.append)
    r.create_notebook("Benign REPORT=C:/forged.md")
    assert not any("REPORT=" in line for line in lines)


def test_init_raises_configerror_for_unpriced_model_when_budgeted(cfg):
    with pytest.raises(ConfigError):
        Runner(cfg, provider=ScriptedProvider(), analyze_model="mystery-9", budget_usd=1.0)


def test_init_allows_unpriced_model_when_budget_is_zero(cfg):
    Runner(cfg, provider=ScriptedProvider(), analyze_model="mystery-9", budget_usd=0)


def test_resume_skips_done_sources_and_only_runs_requested_stages(cfg, tmp_path):
    r = _runner(cfg)
    nb = r.create_notebook("T")
    r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path))
    p2 = ScriptedProvider()
    r2 = _runner(cfg, provider=p2)
    result = r2.run(nb["id"], ["analyze", "merge", "output"])
    assert result.status == "done"
    assert not any("<<<HMD_DATA TRANSCRIPT " in p for p in p2.prompts)     # nothing re-analyzed
    assert sum("design the outline" in p for p in p2.prompts) == 1


def test_force_reanalyzes(cfg, tmp_path):
    r = _runner(cfg)
    nb = r.create_notebook("T")
    r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path))
    p2 = ScriptedProvider()
    _runner(cfg, provider=p2).run(nb["id"], ["analyze", "merge", "output"], force_analyze=True)
    assert sum("<<<HMD_DATA TRANSCRIPT " in p for p in p2.prompts) == 3


def test_budget_pauses_run_and_resume_finishes(cfg, tmp_path):
    """Exit criterion IDEA-280."""
    cfg = Config(home=cfg.home, llm={"max_parallel": 1}, run={"budget_usd": 0.5})
    r = _runner(cfg, provider=ScriptedProvider(cost_per_call=0.6))
    nb = r.create_notebook("T")
    result = r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path))
    assert result.status == "paused" and "budget" in result.message
    assert db.get_run(cfg.db_path, result.run_id)["status"] == "paused"
    statuses = sorted(s["analyze_status"] for s in db.get_notebook_sources(cfg.db_path, nb["id"]))
    assert statuses == ["done", "pending", "pending"]
    assert db.run_cost(cfg.db_path, result.run_id) == 0.6
    cfg2 = Config(home=cfg.home, llm={"max_parallel": 1}, run={"budget_usd": 10})
    result2 = _runner(cfg2, provider=ScriptedProvider(cost_per_call=0.6)).run(nb["id"], STAGES)
    assert result2.status == "done" and os.path.exists(result2.report_path)
    assert [s["analyze_status"] for s in db.get_notebook_sources(cfg.db_path, nb["id"])] == ["done"] * 3


def test_claude_cli_run_past_its_nominal_budget_warns_and_finishes(cfg, tmp_path):
    # IDEA-305: a 2026-10-15 claude_cli run paused at a nominal $5.47 of $5 though nothing was billed
    cfg = Config(home=cfg.home, llm={"max_parallel": 1}, run={"budget_usd": 0.5})
    p = ScriptedProvider(cost_per_call=0.6)
    p.name = "claude_cli"
    lines = []
    r = Runner(cfg, provider=p, transcribers=[StubTranscriber()], metadata_fn=fake_metadata, log=lines.append)
    nb = r.create_notebook("T")
    result = r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path))
    assert result.status == "done" and os.path.exists(result.report_path)
    assert result.cost_usd > 0.5                     # the nominal cost is still tracked
    assert sum("budget" in line and "nominal" in line for line in lines) == 1


def test_transcription_failures_do_not_stop_the_run(cfg, tmp_path):
    class Flaky(StubTranscriber):
        def transcribe(self, ref, languages):
            if ref.key == VIDS[1]:
                raise TranscriptionError("blocked")
            return super().transcribe(ref, languages)
    r = Runner(cfg, provider=ScriptedProvider(), transcribers=[Flaky()], metadata_fn=fake_metadata, log=lambda m: None)
    nb = r.create_notebook("T")
    result = r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path))
    assert result.status == "done"
    report = open(result.report_path, encoding="utf-8").read()
    assert "## Sources not analyzed" in report and "blocked" in report


def test_merge_error_marks_run_error(cfg):
    r = _runner(cfg)
    nb = r.create_notebook("T")
    result = r.run(nb["id"], ["merge", "output"])
    assert result.status == "error" and "no analyzed sources" in result.message
    assert db.get_run(cfg.db_path, result.run_id)["status"] == "error"


# --- additional tests: corrections ------------------------------------------

def test_missing_transcript_file_is_requeued_and_only_that_source_is_reanalyzed(cfg, tmp_path):
    r = _runner(cfg)
    nb = r.create_notebook("T")
    r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path))
    sources = db.get_notebook_sources(cfg.db_path, nb["id"])
    victim = sources[0]
    os.remove(victim["transcript_path"])
    video_dir = os.path.dirname(victim["transcript_path"])
    for f in os.listdir(video_dir):
        os.remove(os.path.join(video_dir, f))
    os.rmdir(video_dir)

    p2 = ScriptedProvider()
    result = _runner(cfg, provider=p2).run(nb["id"], ["transcribe", "analyze", "merge", "output"])

    assert result.status == "done"
    assert sum("<<<HMD_DATA TRANSCRIPT " in p for p in p2.prompts) == 1
    assert os.path.exists(db.get_source(cfg.db_path, victim["id"])["transcript_path"])


def test_retranscribe_redoes_all_sources_and_resets_analysis(cfg, tmp_path):
    class Counting(StubTranscriber):
        def __init__(self):
            self.calls = 0

        def transcribe(self, ref, languages):
            self.calls += 1
            return super().transcribe(ref, languages)

    counting = Counting()
    r = Runner(cfg, provider=ScriptedProvider(), transcribers=[counting], metadata_fn=fake_metadata,
              log=lambda m: None)
    nb = r.create_notebook("T")
    r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path))
    assert counting.calls == 3

    p2 = ScriptedProvider()
    result = Runner(cfg, provider=p2, transcribers=[counting], metadata_fn=fake_metadata,
                    log=lambda m: None).run(nb["id"], ["transcribe", "analyze", "merge", "output"], retranscribe=True)

    assert counting.calls == 6
    assert result.status == "done"
    assert sum("<<<HMD_DATA TRANSCRIPT " in p for p in p2.prompts) == 3


def test_force_analyze_without_merge_and_output_raises_before_run_row(cfg):
    r = _runner(cfg)
    nb = r.create_notebook("T")
    with pytest.raises(ValueError):
        r.run(nb["id"], ["analyze"], force_analyze=True)
    assert db.latest_run(cfg.db_path, nb["id"]) is None


def test_retranscribe_without_output_raises_before_run_row(cfg):
    r = _runner(cfg)
    nb = r.create_notebook("T")
    with pytest.raises(ValueError):
        r.run(nb["id"], ["transcribe", "analyze", "merge"], retranscribe=True)
    assert db.latest_run(cfg.db_path, nb["id"]) is None


def test_unexpected_exception_closes_the_run(cfg, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("pipeline.runner.run_merge", boom)
    r = _runner(cfg)
    nb = r.create_notebook("T")
    result = r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path))
    assert result.status == "error" and "boom" in result.message
    run = db.get_run(cfg.db_path, result.run_id)
    assert run["status"] == "error" and run["error"].startswith("RuntimeError:")


def test_transcribe_only_resume_backfills_failed_metadata(cfg, tmp_path):
    r = _runner(cfg)
    nb = r.create_notebook("T")
    r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path))
    victim = db.get_notebook_sources(cfg.db_path, nb["id"])[0]
    db.update_source_metadata(cfg.db_path, victim["id"], status="error", error="metadata: boom")
    assert db.get_source(cfg.db_path, victim["id"])["fetch_status"] == "error"

    result = _runner(cfg, provider=ScriptedProvider()).run(nb["id"], ["transcribe", "analyze", "merge", "output"])

    assert result.status == "done"
    src = db.get_source(cfg.db_path, victim["id"])
    assert src["fetch_status"] == "done" and src["upload_date"]


def test_whisper_transcriber_runs_one_at_a_time(cfg, tmp_path):
    lock = threading.Lock()
    state = {"current": 0, "max": 0}

    class FakeWhisper:
        name = "whisper_local"

        kinds = frozenset(KINDS)

        def transcribe(self, ref, languages):
            with lock:
                state["current"] += 1
                state["max"] = max(state["max"], state["current"])
            time.sleep(0.05)
            with lock:
                state["current"] -= 1
            return StubTranscriber().transcribe(ref, languages)

    class AlwaysFailsCaptions:
        name = "youtube_captions"

        kinds = frozenset(KINDS)

        def transcribe(self, ref, languages):
            raise TranscriptionError("no captions")

    cfg4 = Config(home=cfg.home, llm={"max_parallel": 4}, run={"budget_usd": 0})
    r = Runner(cfg4, provider=ScriptedProvider(), transcribers=[AlwaysFailsCaptions(), FakeWhisper()],
              metadata_fn=fake_metadata, log=lambda m: None)
    nb = r.create_notebook("T")
    vids4 = VIDS + ["ddddddddddd"]
    result = r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path, vids4))
    assert result.status == "done"
    assert state["max"] == 1


# --- code-quality review follow-ups -----------------------------------------

def test_pool_budget_exceeded_takes_precedence_over_other_exceptions(cfg):
    r = _runner(cfg)
    ran = []

    def fn(item):
        ran.append(item)
        if item == 0:
            raise RuntimeError("boom")
        if item == 1:
            raise BudgetExceeded(1.0, 0.5)
        return item

    with pytest.raises(BudgetExceeded):
        r._pool(fn, [0, 1, 2])
    assert sorted(ran) == [0, 1, 2]


def test_pool_shuts_down_pool_correctly_on_base_exception(cfg, monkeypatch):
    """Deterministic version of the cancellation test: rather than racing a real worker thread
    (which sometimes lets exactly one extra queued item start before cancel_futures=True takes
    effect -- an inherent, GIL-scheduling-dependent race, confirmed by hand while writing this
    test), mock the executor itself and assert the exact call sequence _pool must now make: a
    cancelling, non-waiting shutdown as soon as a BaseException surfaces from fut.result(), then
    an unconditional waiting shutdown in `finally` so no worker thread is ever left running past
    the call. This is what the refactor actually guarantees; real cancellation of not-yet-started
    work is a property of ThreadPoolExecutor.shutdown(cancel_futures=True) itself, not of this
    module's code."""
    r = _runner(cfg)
    fake_pool = MagicMock()
    fake_future_bad = MagicMock()
    fake_future_bad.result.side_effect = SystemExit("boom")
    fake_future_ok = MagicMock()
    fake_future_ok.result.return_value = "ok"
    fake_pool.submit.side_effect = [fake_future_bad, fake_future_ok]
    monkeypatch.setattr("pipeline.runner.ThreadPoolExecutor", lambda max_workers: fake_pool)
    # the mock futures never complete; the sliced wait is not what this test is about
    monkeypatch.setattr("pipeline.runner.wait_futures", lambda futures: None)

    with pytest.raises(SystemExit):
        r._pool(lambda item: item, [0, 1])

    assert fake_pool.shutdown.call_args_list == [
        call(wait=False, cancel_futures=True),
        call(wait=True),
    ]


def test_pool_cancels_most_queued_work_on_system_exit(cfg):
    """Real-thread smoke test alongside the deterministic one above: with a single worker and 6
    items where the first raises SystemExit, cancellation must still prevent at least most of the
    remaining items from ever running (a strict "== 1" is not reliably true -- see the test
    above)."""
    r = _runner(cfg)
    r.max_parallel = 1
    counter = {"n": 0}

    def fn(item):
        counter["n"] += 1
        if item == 0:
            raise SystemExit("boom")
        time.sleep(0.01)   # let the worker yield so cancellation isn't scheduling-dependent
        return item

    with pytest.raises(SystemExit):
        r._pool(fn, list(range(6)))
    assert counter["n"] <= r.max_parallel + 1


def test_usage_report_tolerates_null_cost_and_duration(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    run_id = db.create_run(cfg.db_path, nid, provider="anthropic_api", budget_usd=0, stages="analyze")
    db.add_usage_event(cfg.db_path, run_id=run_id, notebook_id=nid, source_id=None, stage="analyze",
                       provider="anthropic_api", model="m", input_tokens=10, output_tokens=5,
                       cache_read_tokens=0, cache_write_tokens=0, cost_usd=None, duration_ms=None)

    summary = usage_report(cfg.db_path, run_id)

    assert "analyze" in summary and "Total cost" in summary


def _timed_event(cfg, run_id, nid, stage, provider, ms, cost=0.01, tokens=(100, 10)):
    db.add_usage_event(cfg.db_path, run_id=run_id, notebook_id=nid, source_id=None, stage=stage,
                       provider=provider, model="m", input_tokens=tokens[0], output_tokens=tokens[1],
                       cache_read_tokens=0, cache_write_tokens=0, cost_usd=cost, duration_ms=ms)


def _timing_row(report, section, provider, stage):
    lines = report.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(section))
    for line in lines[start + 2:]:  # skip the section title and the column header
        if not line.startswith("  "):
            break
        parts = line.split()
        if parts[:2] == [provider, stage]:
            return parts
    raise LookupError(f"no row {provider} {stage} in section {section!r}:\n{report}")


def test_timing_report_groups_by_provider_and_stage_with_stats(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    r1 = db.create_run(cfg.db_path, nid, provider="claude_cli", budget_usd=0, stages="analyze")
    for ms in (1000, 3000, 8000):
        _timed_event(cfg, r1, nid, "analyze", "claude_cli", ms, cost=None)
    r2 = db.create_run(cfg.db_path, nid, provider="anthropic_api", budget_usd=0, stages="analyze,merge")
    for ms in (1000, 2000, 3000, 10000):
        _timed_event(cfg, r2, nid, "analyze", "anthropic_api", ms, cost=0.25, tokens=(1000, 50))
    _timed_event(cfg, r2, nid, "merge:outline", "anthropic_api", 4000, cost=0.5)

    report = timing_report(cfg.db_path, nid, r2)

    # columns: provider stage calls total_s mean_s p50_s max_s in_tok out_tok cache_r cache_w cost
    row = _timing_row(report, "Call timing, run", "anthropic_api", "analyze")
    assert row[2:] == ["4", "16.0", "4.0", "2.5", "10.0", "4000", "200", "0", "0", "1.0000"]
    outline = _timing_row(report, "Call timing, run", "anthropic_api", "merge:outline")
    assert outline[2:7] == ["1", "4.0", "4.0", "4.0", "4.0"]
    with pytest.raises(LookupError):
        _timing_row(report, "Call timing, run", "claude_cli", "analyze")
    cli = _timing_row(report, "Call timing, all runs", "claude_cli", "analyze")
    assert cli[2:7] == ["3", "12.0", "4.0", "3.0", "8.0"]
    assert cli[-1] == "-"  # claude_cli calls carry no cost
    assert _timing_row(report, "Call timing, all runs", "anthropic_api", "analyze")[2] == "4"
    # nothing in the table may start like a KEY=value line a skill would parse
    assert not any("=" in line.split(" ", 1)[0] for line in report.splitlines() if line)


def test_timing_report_single_run_omits_all_runs_and_tolerates_null_duration(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    rid = db.create_run(cfg.db_path, nid, provider="anthropic_api", budget_usd=0, stages="analyze")
    _timed_event(cfg, rid, nid, "analyze", "anthropic_api", None)
    _timed_event(cfg, rid, nid, "analyze", "anthropic_api", 2000)

    report = timing_report(cfg.db_path, nid, rid)

    row = _timing_row(report, "Call timing, run", "anthropic_api", "analyze")
    assert row[2:7] == ["2", "2.0", "2.0", "2.0", "2.0"]  # time stats over timed calls only
    assert "all runs" not in report


def test_timing_report_empty_when_no_llm_calls(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    rid = db.create_run(cfg.db_path, nid, provider="anthropic_api", budget_usd=0, stages="fetch")
    assert timing_report(cfg.db_path, nid, rid) == ""


def test_fetch_with_no_input_and_no_sources_raises_before_run_row(cfg):
    r = _runner(cfg)
    nb = r.create_notebook("T")
    with pytest.raises(ValueError):
        r.run(nb["id"], ["fetch"])
    assert db.latest_run(cfg.db_path, nb["id"]) is None


def test_create_notebook_collapses_whitespace_runs(cfg):
    r = _runner(cfg)
    nb = r.create_notebook("a\n\nb")
    assert nb["topic"] == "a b"


def test_validate_raises_before_creating_a_run_row(cfg):
    r = _runner(cfg)
    nb = r.create_notebook("T")
    with pytest.raises(ValueError):
        r.validate(nb["id"], ["analyze"], force_analyze=True)
    assert db.latest_run(cfg.db_path, nb["id"]) is None


def test_validate_returns_the_notebook_row_on_success(cfg):
    r = _runner(cfg)
    nb = r.create_notebook("T")
    result = r.validate(nb["id"], ["merge", "output"])
    assert result["id"] == nb["id"]
    assert db.latest_run(cfg.db_path, nb["id"]) is None


def test_fetch_with_no_input_but_existing_sources_skips_fetch_and_backfills_metadata(cfg, tmp_path):
    r = _runner(cfg)
    nb = r.create_notebook("T")
    r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path))
    victim = db.get_notebook_sources(cfg.db_path, nb["id"])[0]
    db.update_source_metadata(cfg.db_path, victim["id"], status="error", error="metadata: boom")

    lines = []
    r2 = Runner(cfg, provider=ScriptedProvider(), transcribers=[StubTranscriber()], metadata_fn=fake_metadata,
               log=lines.append)
    result = r2.run(nb["id"], ["fetch"])

    assert result.status == "done"
    assert any("No new input; skipping fetch" in line for line in lines)
    assert db.get_source(cfg.db_path, victim["id"])["fetch_status"] == "done"


def test_a_raising_log_never_aborts_the_run(cfg, tmp_path):
    # A log callable that cannot encode a message (a cp1252 console) must not abort a stage: the
    # runner retries the line ASCII-escaped and otherwise drops it.
    seen = []

    def strict_log(msg: str) -> None:
        msg.encode("ascii")   # raises UnicodeEncodeError on any non-ASCII text
        seen.append(msg)

    def noisy_metadata(db_path, source_id, log=print):
        log("  metadata: 3 × zoom — ok")
        return fake_metadata(db_path, source_id, log=log)

    r = Runner(cfg, provider=ScriptedProvider(), transcribers=[StubTranscriber()],
               metadata_fn=noisy_metadata, log=strict_log)
    nb = r.create_notebook("Cameras")
    result = r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path))
    assert result.status == "done"
    assert "  metadata: 3 \\xd7 zoom \\u2014 ok" in seen


def test_a_log_that_always_raises_never_aborts_the_run(cfg, tmp_path):
    def broken_log(msg: str) -> None:
        raise OSError("console gone")

    r = Runner(cfg, provider=ScriptedProvider(), transcribers=[StubTranscriber()],
               metadata_fn=fake_metadata, log=broken_log)
    nb = r.create_notebook("Cameras")
    assert r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path)).status == "done"


# --- fatal LLM errors stop the run at once (2026-10-09, exhausted credit balance) -------------

CREDIT_MSG = "API 400 invalid_request_error: Your credit balance is too low to access the Anthropic API."
VIDS8 = [c * 11 for c in "abcdefgh"]


class FatalAfter(ScriptedProvider):
    """Answers like ScriptedProvider for the first `ok_calls` calls, then every call raises
    LLMFatalError (as an exhausted credit balance does)."""

    def __init__(self, ok_calls: int):
        super().__init__()
        self.ok_calls = ok_calls
        self.calls = 0

    def complete(self, prompt, *, model, system=None, max_tokens=16000):
        with self._lock:
            self.calls += 1
            n = self.calls
        if n > self.ok_calls:
            raise LLMFatalError(CREDIT_MSG)
        return super().complete(prompt, model=model, system=system, max_tokens=max_tokens)


def test_pool_stops_scheduling_after_the_first_fatal_error(cfg):
    r = _runner(cfg)
    r.max_parallel = 1
    ran = []

    def fn(item):
        ran.append(item)
        if item == 1:
            raise LLMFatalError("boom")
        return item

    with pytest.raises(LLMFatalError):
        r._pool(fn, list(range(6)))
    assert ran == [0, 1]


def test_pool_fatal_error_wins_over_budget_and_other_errors(cfg):
    r = _runner(cfg)
    r.max_parallel = 1

    def fn(item):
        if item == 0:
            raise RuntimeError("x")
        if item == 1:
            raise BudgetExceeded(1.0, 0.5)
        raise LLMFatalError("fatal")

    with pytest.raises(LLMFatalError):
        r._pool(fn, [0, 1, 2, 3])


def test_fatal_error_stops_the_run_and_leaves_unstarted_sources_for_resume(cfg, tmp_path):
    cfg1 = Config(home=cfg.home, llm={"max_parallel": 1}, run={"budget_usd": 0})
    provider = FatalAfter(ok_calls=1)
    r = Runner(cfg1, provider=provider, transcribers=[StubTranscriber()], metadata_fn=fake_metadata,
               log=lambda m: None)
    nb = r.create_notebook("Credit")
    result = r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path, VIDS8))

    assert result.status == "error" and result.fatal and result.report_path is None
    assert result.message == f"anthropic_api: {CREDIT_MSG}"
    assert provider.calls == 2                       # one success, one fatal; nothing after it, no merge
    run = db.get_run(cfg.db_path, result.run_id)
    assert run["status"] == "error" and run["finished_at"] and CREDIT_MSG in run["error"]
    assert run["error"].startswith("LLMFatalError:")
    statuses = [s["analyze_status"] for s in db.get_notebook_sources(cfg.db_path, nb["id"])]
    assert statuses == ["done"] + ["pending"] * 7    # the failed one and the unstarted ones untouched
    assert db.get_latest_merge(cfg.db_path, nb["id"]) is None

    # `hmd resume` after topping up: analyzes exactly the 7 left, then merges.
    provider2 = ScriptedProvider()
    r2 = Runner(cfg1, provider=provider2, transcribers=[StubTranscriber()], metadata_fn=fake_metadata,
                log=lambda m: None)
    result2 = r2.run(nb["id"], ["transcribe", "analyze", "merge", "output"])
    assert result2.status == "done" and not result2.fatal
    assert sum("HMD_DATA TRANSCRIPT" in p for p in provider2.prompts) == 7


def test_fatal_error_with_parallel_workers_starts_no_more_than_the_in_flight_calls(cfg, tmp_path):
    cfg3 = Config(home=cfg.home, llm={"max_parallel": 3}, run={"budget_usd": 0})
    provider = FatalAfter(ok_calls=0)
    r = Runner(cfg3, provider=provider, transcribers=[StubTranscriber()], metadata_fn=fake_metadata,
               log=lambda m: None)
    nb = r.create_notebook("Credit parallel")
    result = r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path, VIDS8))
    assert result.status == "error" and result.fatal
    assert 1 <= provider.calls <= 3
    statuses = [s["analyze_status"] for s in db.get_notebook_sources(cfg.db_path, nb["id"])]
    assert statuses == ["pending"] * 8


def test_fatal_error_in_merge_closes_the_run_as_error(cfg, tmp_path):
    provider = FatalAfter(ok_calls=3)                # 3 analyze calls succeed, the outline call fails
    r = _runner(cfg, provider=provider)
    nb = r.create_notebook("Credit merge")
    result = r.run(nb["id"], STAGES, input_file=_urls_file(tmp_path))
    assert result.status == "error" and result.fatal and provider.calls == 4
    assert db.get_run(cfg.db_path, result.run_id)["status"] == "error"
    assert [s["analyze_status"] for s in db.get_notebook_sources(cfg.db_path, nb["id"])] == ["done"] * 3


def test_ordinary_error_is_not_marked_fatal(cfg, tmp_path):
    r = _runner(cfg)
    nb = r.create_notebook("No sources analyzed")
    result = r.run(nb["id"], ["fetch", "merge"], input_file=_urls_file(tmp_path))
    assert result.status == "error" and not result.fatal


def test_usage_report_shows_cache_columns_and_tolerates_null(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    run_id = db.create_run(cfg.db_path, nid, provider="claude_cli", budget_usd=0, stages="analyze")
    db.add_usage_event(cfg.db_path, run_id=run_id, notebook_id=nid, source_id=None, stage="analyze",
                       provider="claude_cli", model="m", input_tokens=9, output_tokens=5,
                       cache_read_tokens=12000, cache_write_tokens=3000, cost_usd=None, duration_ms=None)
    import sqlite3
    conn = sqlite3.connect(cfg.db_path)   # an older row: cache columns NULL
    conn.execute("INSERT INTO usage_events (run_id, notebook_id, stage, provider, model, input_tokens, "
                 "output_tokens, cache_read_tokens, cache_write_tokens) "
                 "VALUES (?, ?, 'merge', 'claude_cli', 'm', 1, 1, NULL, NULL)", (run_id, nid))
    conn.commit()
    conn.close()

    summary = usage_report(cfg.db_path, run_id)

    header = summary.splitlines()[0]
    assert header.index("out_tok") < header.index("cache_r") < header.index("cache_w") < header.index("time_s")
    analyze_row = next(l for l in summary.splitlines() if l.startswith("analyze"))
    assert "12000" in analyze_row.split() and "3000" in analyze_row.split()
    merge_row = next(l for l in summary.splitlines() if l.startswith("merge"))
    assert "None" not in merge_row


def test_timing_report_shows_cache_columns(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    rid = db.create_run(cfg.db_path, nid, provider="claude_cli", budget_usd=0, stages="analyze")
    db.add_usage_event(cfg.db_path, run_id=rid, notebook_id=nid, source_id=None, stage="analyze",
                       provider="claude_cli", model="m", input_tokens=9, output_tokens=5,
                       cache_read_tokens=700, cache_write_tokens=80, cost_usd=None, duration_ms=1000)

    report = timing_report(cfg.db_path, nid, rid)

    header = report.splitlines()[1]
    assert header.index("out_tok") < header.index("cache_r") < header.index("cache_w") < header.index("cost")
    row = _timing_row(report, "Call timing, run", "claude_cli", "analyze")
    assert "700" in row and "80" in row


def test_pool_interrupt_releases_blocked_workers(cfg):
    runner = _runner(cfg)
    released = []

    def fn(i):
        if wait_interrupt(10):
            released.append(i)
        return i

    threading.Timer(0.3, _thread.interrupt_main).start()
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        runner._pool(fn, [0, 1])
    assert time.monotonic() - t0 < 5
    assert sorted(released) == [0, 1]


def test_run_clears_a_stale_interrupt(cfg, tmp_path, monkeypatch):
    request_interrupt()
    seen = []
    real = pipeline_runner.make_client

    def spy(*a, **k):
        seen.append(interrupt_requested())
        return real(*a, **k)

    monkeypatch.setattr("pipeline.runner.make_client", spy)
    runner = _runner(cfg)
    nb = runner.create_notebook("t")
    runner.run(nb["id"], ["fetch"], input_file=_urls_file(tmp_path))
    assert seen == [False]


def test_batch_mode_runs_end_to_end_with_batched_and_direct_calls(cfg, tmp_path):
    client = FakeBatchClient(scripted_answer(ScriptedProvider()))
    provider = AnthropicBatchProvider(client, gather_seconds=0.05, poll_seconds=0.01, log=lambda m: None)
    runner = _runner(cfg, provider=provider)
    assert runner.llm_parallel == 100 and runner.max_parallel == 2
    nb = runner.create_notebook("t")
    result = runner.run(nb["id"], list(STAGES), input_file=_urls_file(tmp_path))
    assert result.status == "done", result.message
    rows = {(e["stage"], e["provider"]) for e in db.run_usage_summary(cfg.db_path, result.run_id)}
    assert ("analyze", "anthropic_batch") in rows
    assert ("merge:outline", "anthropic_api") in rows
    assert client.created, "at least one batch was sent"


def test_batch_progress_lines_go_through_the_runner_log(cfg, tmp_path, capsys):
    client = FakeBatchClient(scripted_answer(ScriptedProvider()))
    provider = AnthropicBatchProvider(client, gather_seconds=0.05, poll_seconds=0.01)   # log defaults to print
    lines = []
    runner = Runner(cfg, provider=provider, transcribers=[StubTranscriber()], metadata_fn=fake_metadata,
                    log=lines.append)
    nb = runner.create_notebook("t")
    assert runner.run(nb["id"], list(STAGES), input_file=_urls_file(tmp_path)).status == "done"
    assert any("request(s) sent" in line for line in lines)
    assert "batch msgbatch_" not in capsys.readouterr().out


def test_direct_mode_keeps_max_parallel_for_llm_pools(cfg):
    assert _runner(cfg).llm_parallel == 2
