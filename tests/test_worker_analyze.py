import json
import sqlite3

import pytest
from pipeline import db
from pipeline.config import Config
from pipeline.llm.base import LLMError, LLMFatalError, LLMTruncatedError
from pipeline.llm.client import BudgetExceeded
from pipeline.prompts.analyze import number_segments
from pipeline.prompts.common import data_block
from pipeline.schemas import Analysis, AnalysisDraft, strip_code_filled
from pipeline.transcription.base import Segment, Transcript
from pipeline.transcription.cache import TranscriptCache
from pipeline.workers.analyze import analyze_source


class FakeLLM:
    def __init__(self, result=None, exc=None): self.result, self.exc, self.prompts = result, exc, []
    def call_json(self, prompt, schema, *, model, stage, source_id=None, **kw):
        self.prompts.append((prompt, schema, model, stage, source_id))
        if self.exc: raise self.exc
        # as LLMClient._parse does: code-filled keys are stripped from the wire before validation
        return schema.model_validate(strip_code_filled(self.result))


# The env transcript has two segments: [0] "Acme's search weighs recency about thirty percent",
# [1] "more." -- the second claim points past the end and is dropped.
GOOD = {"summary": "s", "topics": ["t"], "key_claims": [
    {"claim": "Recency matters", "segment_from": 0, "segment_to": 0, "confidence": "high"},
    {"claim": "Made up", "segment_from": 5, "segment_to": 6}]}


@pytest.fixture
def env(tmp_path):
    cfg = Config(home=str(tmp_path))
    db.init_db(cfg.db_path)
    nid = db.create_notebook(cfg.db_path, "Acme SEO", "acme-seo")
    sid = db.add_source(cfg.db_path, nid, "abcdefghijk", "u", title="T", channel="C")
    cache = TranscriptCache(cfg.transcript_cache_dir)
    cache.save(Transcript("abcdefghijk", "youtube_captions", "en",
                          [Segment(0, 4, "Acme's search weighs recency about thirty percent"), Segment(4, 8, "more.")]))
    db.update_transcribe_status(cfg.db_path, sid, "done", transcript_path=cache.path("abcdefghijk", "youtube_captions"),
                                method="youtube_captions", language="en")
    return cfg, sid, cache


def test_analyze_validates_anchors_and_stores(env):
    cfg, sid, cache = env
    llm = FakeLLM(GOOD)
    status = analyze_source(cfg.db_path, sid, "Acme SEO", llm, "claude-haiku-4-5-20251001", cache, log=lambda m: None)
    src = db.get_source(cfg.db_path, sid)
    assert status == "done" and src["analyze_status"] == "done"
    a = Analysis.model_validate_json(src["analysis_json"])
    assert [c.claim for c in a.key_claims] == ["Recency matters"] and a.key_claims[0].anchor.t_start == 0.0
    # the analyze worker pads kept ranges by QUOTE_PAD_SEGMENTS (pipeline/anchors.py)
    assert a.key_claims[0].quote == "Acme's search weighs recency about thirty percent more."
    assert a.quote_mode == "segments"
    assert a.dropped_quotes[0]["claim"] == "Made up"
    assert a.dropped_quotes[0]["reason"] == "segment index out of range"
    prompt, schema, model, stage, source_id = llm.prompts[0]
    assert schema is AnalysisDraft and stage == "analyze" and source_id == sid
    flat = cache.load("abcdefghijk", "youtube_captions").flatten()
    assert data_block("TRANSCRIPT", number_segments(flat.segment_texts())) in prompt


def test_no_anchored_claim_is_an_error(env):
    cfg, sid, cache = env
    llm = FakeLLM({"summary": "s", "key_claims": [{"claim": "x", "quote": "Acme's search weighs recency"}]})
    assert analyze_source(cfg.db_path, sid, "T", llm, "m", cache, log=lambda m: None) == "error"
    src = db.get_source(cfg.db_path, sid)
    assert "no claim had a usable segment range" in src["error_message"]
    assert "segment index missing or not an integer" in src["error_message"]


def test_no_anchored_claim_records_reason_breakdown_and_persists_analysis(env):
    cfg, sid, cache = env
    llm = FakeLLM({"summary": "s", "key_claims": [
        {"claim": "a", "segment_from": 0, "segment_to": 9},
        {"claim": "b", "segment_from": 1, "segment_to": 0},
        {"claim": "c", "segment_from": 1, "segment_to": 1}]})   # "more." is too short
    assert analyze_source(cfg.db_path, sid, "T", llm, "m", cache, log=lambda m: None) == "error"
    src = db.get_source(cfg.db_path, sid)
    msg = src["error_message"]
    assert "no claim had a usable segment range" in msg and "3 dropped" in msg
    assert "transcript_method=youtube_captions" in msg
    assert "segment index out of range" in msg and "segment_from after segment_to" in msg and "too short" in msg
    # dropped_quotes evidence must survive for debugging even on the error path
    a = Analysis.model_validate_json(src["analysis_json"])
    assert len(a.dropped_quotes) == 3


def test_model_returns_no_claims_at_all_is_a_distinct_error(env):
    cfg, sid, cache = env
    llm = FakeLLM({"summary": "s", "key_claims": []})
    assert analyze_source(cfg.db_path, sid, "T", llm, "m", cache, log=lambda m: None) == "error"
    src = db.get_source(cfg.db_path, sid)
    assert src["error_message"] == "the model returned no claims for this transcript"


def test_llm_error_marks_error(env):
    cfg, sid, cache = env
    assert analyze_source(cfg.db_path, sid, "T", FakeLLM(exc=LLMError("boom")), "m", cache, log=lambda m: None) == "error"
    src = db.get_source(cfg.db_path, sid)
    assert src["error_message"].startswith("LLMError:") and "boom" in src["error_message"]


def test_llm_truncated_error_marks_error_with_type_prefix(env):
    cfg, sid, cache = env
    status = analyze_source(cfg.db_path, sid, "T", FakeLLM(exc=LLMTruncatedError("cut")), "m", cache,
                            log=lambda m: None)
    assert status == "error"
    assert db.get_source(cfg.db_path, sid)["error_message"].startswith("LLMTruncatedError:")


def test_unexpected_llm_exception_marks_error_with_unexpected_prefix(env):
    cfg, sid, cache = env
    status = analyze_source(cfg.db_path, sid, "T", FakeLLM(exc=RuntimeError("bug")), "m", cache, log=lambda m: None)
    assert status == "error"
    assert db.get_source(cfg.db_path, sid)["error_message"].startswith("unexpected RuntimeError")


def test_budget_exceeded_propagates(env):
    cfg, sid, cache = env
    with pytest.raises(BudgetExceeded):
        analyze_source(cfg.db_path, sid, "T", FakeLLM(exc=BudgetExceeded(1, 1)), "m", cache, log=lambda m: None)
    assert db.get_source(cfg.db_path, sid)["analyze_status"] == "pending"


def test_fatal_llm_error_propagates_and_leaves_the_source_untouched(env):
    cfg, sid, cache = env
    with pytest.raises(LLMFatalError):
        analyze_source(cfg.db_path, sid, "T", FakeLLM(exc=LLMFatalError("credit balance is too low")), "m", cache,
                       log=lambda m: None)
    src = db.get_source(cfg.db_path, sid)
    assert src["analyze_status"] == "pending" and not src["error_message"]


def test_interrupted_call_propagates_and_leaves_the_source_pending(env):
    # Ctrl+C is not the source's fault: `hmd resume` redoes it without it showing as an error
    from pipeline.llm.base import LLMInterrupted
    cfg, sid, cache = env
    with pytest.raises(LLMInterrupted):
        analyze_source(cfg.db_path, sid, "T", FakeLLM(exc=LLMInterrupted("interrupted")), "m", cache,
                       log=lambda m: None)
    src = db.get_source(cfg.db_path, sid)
    assert src["analyze_status"] == "pending" and not src["error_message"]


def test_done_is_skipped_unless_force(env):
    cfg, sid, cache = env
    analyze_source(cfg.db_path, sid, "T", FakeLLM(GOOD), "m", cache, log=lambda m: None)
    assert analyze_source(cfg.db_path, sid, "T", FakeLLM(GOOD), "m", cache, log=lambda m: None) == "skipped"
    assert analyze_source(cfg.db_path, sid, "T", FakeLLM(GOOD), "m", cache, force=True, log=lambda m: None) == "done"


def test_force_retry_after_error_clears_stale_error_message(env):
    cfg, sid, cache = env
    status = analyze_source(cfg.db_path, sid, "T", FakeLLM(exc=LLMError("boom")), "m", cache, log=lambda m: None)
    src = db.get_source(cfg.db_path, sid)
    assert status == "error" and src["error_message"]

    status = analyze_source(cfg.db_path, sid, "T", FakeLLM(GOOD), "m", cache, force=True, log=lambda m: None)
    src = db.get_source(cfg.db_path, sid)
    assert status == "done" and src["analyze_status"] == "done" and src["error_message"] is None


def test_untranscribed_source_is_error(env):
    cfg, sid, cache = env
    db.update_transcribe_status(cfg.db_path, sid, "pending")
    status = analyze_source(cfg.db_path, sid, "T", FakeLLM(GOOD), "m", cache, log=lambda m: None)
    assert status == "error"
    assert "transcribe_status=pending" in db.get_source(cfg.db_path, sid)["error_message"]


def test_log_message_pluralization(env):
    cfg, sid, cache = env
    lines = []
    analyze_source(cfg.db_path, sid, "T", FakeLLM(GOOD), "m", cache, log=lines.append)
    assert "  [abcdefghijk] 1 claim anchored, 1 dropped" in lines


def test_model_supplied_quote_anchor_and_mode_are_ignored(env):
    cfg, sid, cache = env
    llm = FakeLLM({"summary": "s", "key_claims": [
        {"claim": "Recency matters", "segment_from": 0, "segment_to": 0, "quote": "a forged quote, never said",
         "anchor": {"char_start": 0, "char_end": 5, "t_start": 99.0, "score": 100.0}}],
        "quote_mode": "text", "dropped_quotes": [{"claim": "forged"}],
        "salvaged_quotes": [{"claim": "forged"}]})
    assert analyze_source(cfg.db_path, sid, "T", llm, "m", cache, log=lambda m: None) == "done"
    a = Analysis.model_validate_json(db.get_source(cfg.db_path, sid)["analysis_json"])
    # the analyze worker pads kept ranges by QUOTE_PAD_SEGMENTS (pipeline/anchors.py)
    assert a.key_claims[0].quote == "Acme's search weighs recency about thirty percent more."
    assert a.key_claims[0].anchor.t_start == 0.0 and a.key_claims[0].anchor.char_end == 55
    assert a.quote_mode == "segments" and a.dropped_quotes == []
    assert "salvaged_quotes" not in json.loads(db.get_source(cfg.db_path, sid)["analysis_json"])


def test_prompt_size_is_logged(env):
    cfg, sid, cache = env
    lines = []
    analyze_source(cfg.db_path, sid, "T", FakeLLM(GOOD), "m", cache, log=lines.append)
    assert any(l.startswith("  [abcdefghijk] analyze prompt: 2 segments, ") and l.endswith(" chars") for l in lines)


# --- corrections ------------------------------------------------------------

def test_corrupt_cache_entry_is_an_error(env):
    cfg, sid, cache = env
    path = cache.path("abcdefghijk", "youtube_captions")
    with open(path, "w", encoding="utf-8") as f:
        f.write("not json at all {{{")
    status = analyze_source(cfg.db_path, sid, "T", FakeLLM(GOOD), "m", cache, log=lambda m: None)
    assert status == "error"
    assert "unreadable" in db.get_source(cfg.db_path, sid)["error_message"]


def test_null_segment_text_is_an_error(env):
    cfg, sid, cache = env
    path = cache.path("abcdefghijk", "youtube_captions")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["segments"][0]["text"] = None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    llm = FakeLLM(GOOD)
    status = analyze_source(cfg.db_path, sid, "T", llm, "m", cache, log=lambda m: None)
    assert status == "error"
    assert "unreadable" in db.get_source(cfg.db_path, sid)["error_message"]
    assert llm.prompts == []


def test_bad_video_id_or_method_is_an_error(env):
    cfg, sid, cache = env
    # A source whose video_id can't form a valid cache path (ValueError from TranscriptCache.path)
    nid = db.get_source(cfg.db_path, sid)["notebook_id"]
    bad_sid = db.add_source(cfg.db_path, nid, "not-11-chars", "u", title="T", channel="C")
    db.update_transcribe_status(cfg.db_path, bad_sid, "done", transcript_path="x",
                                method="youtube_captions", language="en")
    status = analyze_source(cfg.db_path, bad_sid, "T", FakeLLM(GOOD), "m", cache, log=lambda m: None)
    assert status == "error"
    assert "unreadable" in db.get_source(cfg.db_path, bad_sid)["error_message"]


def test_unexpected_exception_after_llm_call_marks_error(env, monkeypatch):
    cfg, sid, cache = env

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("pipeline.workers.analyze.resolve_segment_claims", boom)
    status = analyze_source(cfg.db_path, sid, "T", FakeLLM(GOOD), "m", cache, log=lambda m: None)
    src = db.get_source(cfg.db_path, sid)
    assert status == "error"
    assert "kaboom" in src["error_message"] and "RuntimeError" in src["error_message"]


def test_db_error_on_done_write_marks_error_not_raises(env, monkeypatch):
    cfg, sid, cache = env
    real_update = db.update_analyze_status

    def flaky_update(db_path, source_id, status, **kwargs):
        if status == "done":
            raise sqlite3.OperationalError("locked")
        return real_update(db_path, source_id, status, **kwargs)

    monkeypatch.setattr("pipeline.workers.analyze.db.update_analyze_status", flaky_update)
    status = analyze_source(cfg.db_path, sid, "T", FakeLLM(GOOD), "m", cache, log=lambda m: None)
    assert status == "error"
    src = db.get_source(cfg.db_path, sid)
    assert src["analyze_status"] == "error"
    assert "locked" in src["error_message"]
    assert "OperationalError" in src["error_message"]


def test_empty_transcript_is_an_error_without_calling_llm(env):
    cfg, sid, cache = env
    cache.save(Transcript("abcdefghijk", "youtube_captions", "en", [Segment(0, 1, "   "), Segment(1, 2, "")]))
    llm = FakeLLM(GOOD)
    status = analyze_source(cfg.db_path, sid, "T", llm, "m", cache, log=lambda m: None)
    assert status == "error"
    assert "empty" in db.get_source(cfg.db_path, sid)["error_message"]
    assert llm.prompts == []


def test_missing_source_raises(env):
    cfg, sid, cache = env
    with pytest.raises(ValueError):
        analyze_source(cfg.db_path, sid + 999, "T", FakeLLM(GOOD), "m", cache, log=lambda m: None)


def test_main_exits_nonzero_on_error(env, tmp_path, monkeypatch):
    cfg, sid, cache = env
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"home: {cfg.home}\n", encoding="utf-8")

    from pipeline.workers import analyze as worker_module

    monkeypatch.setattr(worker_module, "make_provider", lambda cfg_arg: object())
    monkeypatch.setattr(worker_module, "make_client", lambda *a, **k: object())
    monkeypatch.setattr(worker_module, "analyze_source", lambda *a, **k: "error")

    with pytest.raises(SystemExit) as excinfo:
        worker_module.main(["--source-id", str(sid), "--topic", "T", "--config", str(config_path)])
    assert excinfo.value.code == 1


def test_main_missing_source_prints_to_stderr_and_exits(env, tmp_path, capsys):
    cfg, sid, cache = env
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"home: {cfg.home}\n", encoding="utf-8")

    from pipeline.workers import analyze as worker_module

    with pytest.raises(SystemExit) as excinfo:
        worker_module.main(["--source-id", str(sid + 999), "--topic", "T", "--config", str(config_path)])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert f"source {sid + 999} not found" in captured.err


def test_main_budget_exceeded_prints_to_stderr_and_exits(env, tmp_path, monkeypatch, capsys):
    cfg, sid, cache = env
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"home: {cfg.home}\n", encoding="utf-8")

    from pipeline.workers import analyze as worker_module

    monkeypatch.setattr(worker_module, "make_provider", lambda cfg_arg: object())
    monkeypatch.setattr(worker_module, "make_client", lambda *a, **k: object())

    def raise_budget(*a, **k):
        raise BudgetExceeded(5, 5)

    monkeypatch.setattr(worker_module, "analyze_source", raise_budget)

    with pytest.raises(SystemExit) as excinfo:
        worker_module.main(["--source-id", str(sid), "--topic", "T", "--config", str(config_path)])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "run budget reached" in captured.err


def test_analyze_passes_the_source_kind_to_the_prompt(env):
    cfg, sid, cache = env
    nid = db.create_notebook(cfg.db_path, "Docs", "docs")
    key = "pdf-0123456789abcdef"
    pid = db.add_source(cfg.db_path, nid, key, "file:///x.pdf", kind="pdf", title="A paper", channel="")
    cache.save(Transcript(key, "pdf_text", "en", [Segment(0, 1, "Recency matters a lot."), Segment(1, 2, "More.")]))
    db.update_transcribe_status(cfg.db_path, pid, "done", transcript_path=cache.path(key, "pdf_text"),
                                method="pdf_text", language="en")
    llm = FakeLLM({"summary": "s", "topics": ["t"], "key_claims": [
        {"claim": "Recency matters", "segment_from": 0, "segment_to": 0, "confidence": "high"}]})
    analyze_source(cfg.db_path, pid, "Acme SEO", llm, "m", cache, log=lambda m: None)
    assert '"kind": "PDF document"' in llm.prompts[0][0]
