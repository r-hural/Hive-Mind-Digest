# tests/test_db.py
import sqlite3
import threading
import pytest
from pipeline import db


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "state.sqlite")
    db.init_db(path)
    return path


def test_init_creates_tables_and_wal(db_path):
    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"meta", "notebooks", "sources", "runs", "usage_events", "merges"} <= tables
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == db.SCHEMA_VERSION


def test_init_is_idempotent(db_path):
    db.init_db(db_path)
    db.init_db(db_path)


def test_legacy_mvp_database_is_refused(tmp_path):
    path = str(tmp_path / "hmd.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE projects (id INTEGER PRIMARY KEY, topic TEXT)")
    conn.commit(); conn.close()
    with pytest.raises(db.LegacyDatabaseError, match="MILESTONE-0"):
        db.init_db(path)


def test_schema_version_mismatch_is_refused(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE meta SET value='0' WHERE key='schema_version'")
    conn.commit(); conn.close()
    with pytest.raises(db.LegacyDatabaseError, match="0"):
        db.init_db(db_path)


def test_connection_busy_timeout_is_30s(db_path):
    conn = db._conn(db_path)
    try:
        # sqlite3 exposes the timeout only through behaviour; check the attribute we set
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    finally:
        conn.close()


def test_notebook_crud(db_path):
    nid = db.create_notebook(db_path, "Acme SEO", "acme-seo")
    assert nid == 1
    assert db.get_notebook(db_path, 1)["slug"] == "acme-seo"
    assert db.get_notebook(db_path, "acme-seo")["id"] == 1
    assert db.get_notebook(db_path, "missing") is None
    assert [n["slug"] for n in db.list_notebooks(db_path)] == ["acme-seo"]
    assert db.notebook_slugs(db_path) == {"acme-seo"}


def test_get_notebook_looks_up_slug_before_id(db_path):
    nid = db.create_notebook(db_path, "Year Review", "2026")
    found = db.get_notebook(db_path, "2026")
    assert found["id"] == nid and found["slug"] == "2026"


def test_add_source_idempotent_and_metadata_update(db_path):
    nid = db.create_notebook(db_path, "T", "t")
    sid = db.add_source(db_path, nid, "abcdefghijk", "https://www.youtube.com/watch?v=abcdefghijk")
    assert db.add_source(db_path, nid, "abcdefghijk", "https://www.youtube.com/watch?v=abcdefghijk") == sid
    src = db.get_source(db_path, sid)
    assert src["fetch_status"] == "pending" and src["title"] is None
    db.update_source_metadata(db_path, sid, title="Title", channel="Chan", duration_seconds=90, upload_date="20260101")
    src = db.get_source(db_path, sid)
    assert src["fetch_status"] == "done" and src["upload_date"] == "20260101"


def test_add_source_same_video_id_different_notebooks_is_two_rows(db_path):
    n1 = db.create_notebook(db_path, "A", "a")
    n2 = db.create_notebook(db_path, "B", "b")
    s1 = db.add_source(db_path, n1, "v", "u")
    s2 = db.add_source(db_path, n2, "v", "u")
    assert s1 != s2


def test_add_usage_event_concurrent_writers(db_path):
    nid = db.create_notebook(db_path, "T", "t")
    rid = db.create_run(db_path, nid, provider="p", budget_usd=0, stages="x")
    errors = []

    def worker():
        try:
            for _ in range(25):
                db.add_usage_event(db_path, run_id=rid, notebook_id=nid, source_id=None, stage="analyze",
                                   provider="p", model="m", input_tokens=1, output_tokens=1,
                                   cache_read_tokens=0, cache_write_tokens=0, cost_usd=0.01, duration_ms=1)
        except Exception as exc:  # pragma: no cover - surfaced via assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM usage_events WHERE run_id=?", (rid,)).fetchone()[0]
    finally:
        conn.close()
    assert count == 100


def test_status_updates_only_set_given_data_columns(db_path):
    nid = db.create_notebook(db_path, "T", "t")
    sid = db.add_source(db_path, nid, "v", "u")

    # error followed by success clears error_message
    db.update_transcribe_status(db_path, sid, "error", error="boom")
    db.update_transcribe_status(db_path, sid, "done", transcript_path="p", method="youtube_captions", language="en")
    src = db.get_source(db_path, sid)
    assert src["transcribe_status"] == "done" and src["error_message"] is None
    assert src["transcript_path"] == "p" and src["transcript_language"] == "en"

    # transcribe error keeps transcript_path/transcript_method
    db.update_transcribe_status(db_path, sid, "error", error="oops")
    src = db.get_source(db_path, sid)
    assert src["transcribe_status"] == "error" and src["error_message"] == "oops"
    assert src["transcript_path"] == "p" and src["transcript_method"] == "youtube_captions"

    # analyze error after a successful analysis keeps the old analysis_json
    db.update_analyze_status(db_path, sid, "done", analysis_json='{"ok": true}')
    db.update_analyze_status(db_path, sid, "error", error="analyze boom")
    src = db.get_source(db_path, sid)
    assert src["analyze_status"] == "error" and src["error_message"] == "analyze boom"
    assert src["analysis_json"] == '{"ok": true}'

    # a subsequent non-error status clears error_message again
    db.update_analyze_status(db_path, sid, "done")
    src = db.get_source(db_path, sid)
    assert src["analyze_status"] == "done" and src["error_message"] is None
    assert src["analysis_json"] == '{"ok": true}'


def test_status_updates_and_reset_analyze(db_path):
    nid = db.create_notebook(db_path, "T", "t")
    sid = db.add_source(db_path, nid, "v", "u")
    db.update_analyze_status(db_path, sid, "error", error="boom")
    db.update_analyze_status(db_path, sid, "done", analysis_json="{}")
    assert db.get_source(db_path, sid)["analyze_status"] == "done"
    db.reset_analyze_status(db_path, nid)
    src = db.get_source(db_path, sid)
    assert src["analyze_status"] == "pending" and src["error_message"] is None


def test_reset_analyze_status_only_clears_error_for_analyze_errors(db_path):
    nid = db.create_notebook(db_path, "T", "t")
    sid = db.add_source(db_path, nid, "v", "u")
    # a transcribe error (not an analyze error) should survive reset_analyze_status
    db.update_transcribe_status(db_path, sid, "error", error="transcribe boom")
    db.reset_analyze_status(db_path, nid)
    src = db.get_source(db_path, sid)
    assert src["analyze_status"] == "pending" and src["analysis_json"] is None
    assert src["transcribe_status"] == "error" and src["error_message"] == "transcribe boom"


def test_reset_analyze_status_with_source_ids_only_resets_those(db_path):
    nid = db.create_notebook(db_path, "T", "t")
    sid1 = db.add_source(db_path, nid, "aaaaaaaaaaa", "u1")
    sid2 = db.add_source(db_path, nid, "bbbbbbbbbbb", "u2")
    db.update_analyze_status(db_path, sid1, "done", analysis_json="{}")
    db.update_analyze_status(db_path, sid2, "done", analysis_json="{}")

    db.reset_analyze_status(db_path, nid, source_ids=[sid1])

    src1, src2 = db.get_source(db_path, sid1), db.get_source(db_path, sid2)
    assert src1["analyze_status"] == "pending" and src1["analysis_json"] is None
    assert src2["analyze_status"] == "done" and src2["analysis_json"] == "{}"


def test_reset_analyze_status_with_empty_source_ids_is_a_noop(db_path):
    nid = db.create_notebook(db_path, "T", "t")
    sid = db.add_source(db_path, nid, "v", "u")
    db.update_analyze_status(db_path, sid, "done", analysis_json="{}")

    db.reset_analyze_status(db_path, nid, source_ids=[])

    src = db.get_source(db_path, sid)
    assert src["analyze_status"] == "done" and src["analysis_json"] == "{}"


def test_update_source_metadata_error_then_success_clears_error(db_path):
    nid = db.create_notebook(db_path, "T", "t")
    sid = db.add_source(db_path, nid, "v", "u")
    db.update_source_metadata(db_path, sid, status="error", error="fetch boom")
    src = db.get_source(db_path, sid)
    assert src["fetch_status"] == "error" and src["error_message"] == "fetch boom"
    db.update_source_metadata(db_path, sid, title="Title", channel="Chan",
                              duration_seconds=1, upload_date="20260101")
    src = db.get_source(db_path, sid)
    assert src["fetch_status"] == "done" and src["error_message"] is None


def test_add_source_other_integrity_errors_still_raise(db_path):
    nid = db.create_notebook(db_path, "T", "t")
    with pytest.raises(sqlite3.IntegrityError):
        db.add_source(db_path, nid, None, "u")


def test_source_queries(db_path):
    nid = db.create_notebook(db_path, "T", "t")
    a = db.add_source(db_path, nid, "a", "ua")
    b = db.add_source(db_path, nid, "b", "ub")
    db.update_analyze_status(db_path, a, "done", analysis_json="{}")
    assert [s["id"] for s in db.get_notebook_sources(db_path, nid)] == [a, b]
    assert [s["id"] for s in db.get_analyzed_sources(db_path, nid)] == [a]
    assert [s["id"] for s in db.get_notebook_sources(db_path, nid, analyze_status="pending")] == [b]


def test_runs_and_usage(db_path):
    nid = db.create_notebook(db_path, "T", "t")
    rid = db.create_run(db_path, nid, provider="anthropic_api", budget_usd=2.0, stages="fetch,transcribe")
    assert db.get_run(db_path, rid)["status"] == "running"
    db.add_usage_event(db_path, run_id=rid, notebook_id=nid, source_id=None, stage="analyze",
                       provider="anthropic_api", model="m", input_tokens=10, output_tokens=5,
                       cache_read_tokens=0, cache_write_tokens=3, cost_usd=0.25, duration_ms=12)
    db.add_usage_event(db_path, run_id=rid, notebook_id=nid, source_id=None, stage="merge",
                       provider="anthropic_api", model="m", input_tokens=1, output_tokens=1,
                       cache_read_tokens=0, cache_write_tokens=0, cost_usd=None, duration_ms=1)
    assert db.run_cost(db_path, rid) == 0.25
    summary = db.run_usage_summary(db_path, rid)
    assert {r["stage"] for r in summary} == {"analyze", "merge"}
    analyze_row = next(r for r in summary if r["stage"] == "analyze")
    assert analyze_row["cache_write_tokens"] == 3
    db.finish_run(db_path, rid, "paused", error="budget")
    run = db.get_run(db_path, rid)
    assert run["status"] == "paused" and run["error"] == "budget" and run["finished_at"]
    assert db.latest_run(db_path, nid)["id"] == rid


def _ev(db_path, rid, nid, stage, provider, ms):
    db.add_usage_event(db_path, run_id=rid, notebook_id=nid, source_id=None, stage=stage,
                       provider=provider, model="m", input_tokens=100, output_tokens=10,
                       cache_read_tokens=0, cache_write_tokens=0, cost_usd=0.01, duration_ms=ms)


def test_notebook_usage_events_scopes_by_notebook_and_run(db_path):
    nid = db.create_notebook(db_path, "T", "t")
    other = db.create_notebook(db_path, "U", "u")
    r1 = db.create_run(db_path, nid, provider="claude_cli", budget_usd=0, stages="analyze")
    r2 = db.create_run(db_path, nid, provider="anthropic_api", budget_usd=0, stages="analyze")
    r3 = db.create_run(db_path, other, provider="claude_cli", budget_usd=0, stages="analyze")
    _ev(db_path, r1, nid, "analyze", "claude_cli", 1000)
    _ev(db_path, r2, nid, "analyze", "anthropic_api", 2000)
    _ev(db_path, r2, nid, "merge:outline", "anthropic_api", 3000)
    _ev(db_path, r3, other, "analyze", "claude_cli", 9000)

    all_rows = db.notebook_usage_events(db_path, nid)
    assert sorted(r["duration_ms"] for r in all_rows) == [1000, 2000, 3000]
    assert {"run_id", "stage", "provider", "input_tokens", "output_tokens", "cost_usd"} <= set(all_rows[0])
    run_rows = db.notebook_usage_events(db_path, nid, run_id=r2)
    assert sorted(r["stage"] for r in run_rows) == ["analyze", "merge:outline"]
    assert db.notebook_usage_events(db_path, other, run_id=r2) == []


def test_merges(db_path):
    nid = db.create_notebook(db_path, "T", "t")
    rid = db.create_run(db_path, nid, provider="p", budget_usd=0, stages="merge")
    mid = db.create_merge(db_path, nid, rid, "model", '{"tldr":[]}', '{"S1": 1}',
                          '{"merge": "2026-09-06"}', "vault/t/reports/x.md")
    m = db.get_latest_merge(db_path, nid)
    assert m["id"] == mid and m["source_map_json"] == '{"S1": 1}'


def test_get_previous_merge_returns_the_second_newest(tmp_path):
    from pipeline import db
    p = str(tmp_path / "s.sqlite"); db.init_db(p)
    nid = db.create_notebook(p, "T", "t")
    assert db.get_previous_merge(p, nid) is None
    for n in range(3):
        db.create_merge(p, nid, None, "m", f'{{"n": {n}}}', "{}", "{}", None)
    assert db.get_previous_merge(p, nid)["merged_json"] == '{"n": 1}'
