from pipeline.sources import KINDS
import os
import sqlite3

import pytest
from pipeline import db
from pipeline.config import Config
from pipeline.transcription.base import Segment, Transcript, TranscriptionError
from pipeline.transcription.cache import TranscriptCache
from pipeline.workers.transcribe import transcribe_source


class Stub:
    name = "youtube_captions"
    def __init__(self, ok=True): self.ok = ok
    kinds = frozenset(KINDS)

    def transcribe(self, ref, languages):
        if not self.ok: raise TranscriptionError("nope")
        return Transcript(ref.key, self.name, "en", [Segment(0, 1, "text")])


@pytest.fixture
def env(tmp_path):
    cfg = Config(home=str(tmp_path))
    db.init_db(cfg.db_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    sid = db.add_source(cfg.db_path, nid, "abcdefghijk", "https://www.youtube.com/watch?v=abcdefghijk")
    return cfg, sid


def test_success_records_path_method_language(env):
    cfg, sid = env
    status = transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub()], log=lambda m: None)
    src = db.get_source(cfg.db_path, sid)
    assert status == "done" and src["transcribe_status"] == "done"
    assert src["transcript_method"] == "youtube_captions" and src["transcript_language"] == "en"
    assert os.path.exists(src["transcript_path"])
    assert src["transcript_path"].replace("\\", "/").endswith("cache/transcripts/abcdefghijk/youtube_captions.json")


def test_failure_marks_error(env):
    cfg, sid = env
    status = transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub(ok=False)], log=lambda m: None)
    src = db.get_source(cfg.db_path, sid)
    assert status == "error" and src["transcribe_status"] == "error" and "nope" in src["error_message"]


def test_failure_then_retry_success_clears_error(env):
    cfg, sid = env
    status = transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub(ok=False)], log=lambda m: None)
    src = db.get_source(cfg.db_path, sid)
    assert status == "error" and src["transcribe_status"] == "error" and src["error_message"]

    status = transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub()], log=lambda m: None)
    src = db.get_source(cfg.db_path, sid)
    assert status == "done" and src["transcribe_status"] == "done" and src["error_message"] is None


def test_failure_log_message_format(env):
    cfg, sid = env
    lines: list[str] = []
    transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub(ok=False)], log=lines.append)
    assert "  [abcdefghijk] ERROR: all transcribers failed for abcdefghijk: nope" in lines


def test_success_roundtrips_through_cache(env):
    cfg, sid = env
    transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub()], log=lambda m: None)
    src = db.get_source(cfg.db_path, sid)
    transcript = TranscriptCache(cfg.transcript_cache_dir).load("abcdefghijk", src["transcript_method"])
    assert transcript.segments[0].text == "text"


def test_db_error_on_done_write_marks_error_not_raises(env, monkeypatch):
    cfg, sid = env
    real_update = db.update_transcribe_status

    def flaky_update(db_path, source_id, status, **kwargs):
        if status == "done":
            raise sqlite3.OperationalError("locked")
        return real_update(db_path, source_id, status, **kwargs)

    monkeypatch.setattr("pipeline.workers.transcribe.db.update_transcribe_status", flaky_update)
    status = transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub()], log=lambda m: None)
    assert status == "error"
    src = db.get_source(cfg.db_path, sid)
    assert src["transcribe_status"] == "error"
    assert "locked" in src["error_message"]
    assert "OperationalError" in src["error_message"]


def test_done_source_with_existing_file_is_skipped(env):
    cfg, sid = env
    transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub()], log=lambda m: None)
    assert transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub(ok=False)], log=lambda m: None) == "skipped"


def test_force_reruns_chain_on_a_done_source(env):
    cfg, sid = env
    transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub()], log=lambda m: None)
    path_before = db.get_source(cfg.db_path, sid)["transcript_path"]
    assert os.path.exists(path_before)

    calls = []

    class Counting(Stub):
        def transcribe(self, ref, languages):
            calls.append(ref.key)
            return super().transcribe(ref, languages)

    status = transcribe_source(cfg.db_path, sid, cfg, transcribers=[Counting()], log=lambda m: None, force=True)
    assert status == "done" and calls == ["abcdefghijk"]
    assert os.path.exists(db.get_source(cfg.db_path, sid)["transcript_path"])


def test_force_reruns_chain_resets_analyze_status_atomically(env):
    cfg, sid = env
    transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub()], log=lambda m: None)
    db.update_analyze_status(cfg.db_path, sid, "done", analysis_json='{"ok": true}')
    assert db.get_source(cfg.db_path, sid)["analyze_status"] == "done"

    status = transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub()], log=lambda m: None, force=True)

    assert status == "done"
    src = db.get_source(cfg.db_path, sid)
    assert src["analyze_status"] == "pending" and src["analysis_json"] is None


def test_done_source_with_missing_file_is_redone(env):
    cfg, sid = env
    transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub()], log=lambda m: None)
    os.remove(db.get_source(cfg.db_path, sid)["transcript_path"])
    assert transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub()], log=lambda m: None) == "done"


# --- agreed deviations -------------------------------------------------------

def test_malformed_video_id_marks_error_not_raises(env):
    cfg, sid = env
    nid = db.get_source(cfg.db_path, sid)["notebook_id"]
    bad_sid = db.add_source(cfg.db_path, nid, "bad id!", "https://www.youtube.com/watch?v=bad")
    status = transcribe_source(cfg.db_path, bad_sid, cfg, transcribers=[Stub()], log=lambda m: None)
    src = db.get_source(cfg.db_path, bad_sid)
    assert status == "error" and src["transcribe_status"] == "error"
    assert "video id" in src["error_message"] or "bad id!" in src["error_message"]


def test_unexpected_exception_marks_error_not_raises(env, monkeypatch):
    cfg, sid = env

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("pipeline.workers.transcribe.transcribe_with_chain", boom)
    status = transcribe_source(cfg.db_path, sid, cfg, transcribers=[Stub()], log=lambda m: None)
    src = db.get_source(cfg.db_path, sid)
    assert status == "error" and src["transcribe_status"] == "error"
    assert "disk full" in src["error_message"]
    assert "OSError" in src["error_message"]


def test_missing_source_raises(env):
    cfg, sid = env
    with pytest.raises(ValueError):
        transcribe_source(cfg.db_path, sid + 999, cfg, transcribers=[Stub()], log=lambda m: None)


def test_main_exits_nonzero_on_error(env, tmp_path):
    cfg, sid = env
    nid = db.get_source(cfg.db_path, sid)["notebook_id"]
    bad_sid = db.add_source(cfg.db_path, nid, "bad id!", "https://www.youtube.com/watch?v=bad")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"home: {cfg.home}\n", encoding="utf-8")

    from pipeline.workers import transcribe as worker_module

    with pytest.raises(SystemExit) as excinfo:
        worker_module.main(["--source-id", str(bad_sid), "--config", str(config_path)])
    assert excinfo.value.code == 1


def test_worker_hands_the_chain_a_source_ref_with_the_row_kind(env):
    cfg, _ = env
    nid = db.create_notebook(cfg.db_path, "T2", "t2")
    key = "video-0123456789abcdef"
    sid = db.add_source(cfg.db_path, nid, key, "https://vimeo.com/1", kind="video")
    seen = []

    class Recording(Stub):
        def transcribe(self, ref, languages):
            seen.append(ref)
            return super().transcribe(ref, languages)

    assert transcribe_source(cfg.db_path, sid, cfg, transcribers=[Recording()], log=lambda m: None) == "done"
    assert [(r.key, r.kind, r.url) for r in seen] == [(key, "video", "https://vimeo.com/1")]


def test_worker_reports_a_source_no_transcriber_handles(env):
    cfg, _ = env
    nid = db.create_notebook(cfg.db_path, "T3", "t3")
    sid = db.add_source(cfg.db_path, nid, "media-0123456789abcdef", "file:///x.mp3", kind="media")

    class YouTubeOnly(Stub):
        kinds = frozenset({"youtube"})

    lines = []
    assert transcribe_source(cfg.db_path, sid, cfg, transcribers=[YouTubeOnly()], log=lines.append) == "error"
    assert any("no transcriber handles media sources" in l for l in lines)
