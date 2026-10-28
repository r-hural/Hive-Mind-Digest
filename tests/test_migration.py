# tests/test_migration.py
import json
import os
import sqlite3
from datetime import date

import pytest

from pipeline import db
from pipeline.stages.output import render_markdown, report_path_for, run_output

V1_SOURCES = """CREATE TABLE sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT, notebook_id INTEGER NOT NULL REFERENCES notebooks(id),
    video_id TEXT NOT NULL, url TEXT NOT NULL, title TEXT, channel TEXT, duration_seconds INTEGER,
    upload_date TEXT, fetch_status TEXT DEFAULT 'pending', transcribe_status TEXT DEFAULT 'pending',
    analyze_status TEXT DEFAULT 'pending', transcript_path TEXT, transcript_method TEXT,
    transcript_language TEXT, analysis_json TEXT, error_message TEXT, UNIQUE(notebook_id, video_id))"""


def _make_v1(path):
    """A schema-v1 database: today's SCHEMA with the v1 sources table and schema_version 1."""
    db.init_db(path)                                   # creates v2
    c = sqlite3.connect(path)
    try:
        with c:
            c.execute("DROP TABLE sources")
            c.execute(V1_SOURCES)
            c.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
    finally:
        c.close()


def _version(p):
    c = sqlite3.connect(p)
    try:
        return c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    finally:
        c.close()


def _columns(p):
    c = sqlite3.connect(p)
    try:
        return [r[1] for r in c.execute("PRAGMA table_info(sources)")]
    finally:
        c.close()


def _insert_v1_source(p):
    c = sqlite3.connect(p)
    try:
        with c:
            c.execute("INSERT INTO notebooks (slug, topic) VALUES ('t', 'T')")
            c.execute("INSERT INTO sources (notebook_id, video_id, url) VALUES (1, 'aaaaaaaaaaa', 'u')")
    finally:
        c.close()


def test_v1_database_is_migrated_with_a_backup(tmp_path):
    p = str(tmp_path / "state.sqlite")
    _make_v1(p)
    _insert_v1_source(p)
    db.init_db(p)
    assert os.path.exists(p + ".v1-backup")
    assert _version(p) == "2"
    c = sqlite3.connect(p)
    try:
        assert c.execute("SELECT kind FROM sources").fetchone()[0] == "youtube"
    finally:
        c.close()
    db.init_db(p)                                       # idempotent: no second backup
    assert not os.path.exists(p + ".v1-backup.2")


def test_backup_is_a_v1_copy_including_wal_content(tmp_path):
    p = str(tmp_path / "state.sqlite")
    _make_v1(p)
    # keep a connection open so the rows stay in the WAL, unmerged
    holder = sqlite3.connect(p)
    try:
        holder.execute("PRAGMA journal_mode=WAL")
        holder.execute("PRAGMA wal_autocheckpoint=0")
        with holder:
            holder.execute("INSERT INTO notebooks (slug, topic) VALUES ('t', 'T')")
            holder.execute("INSERT INTO sources (notebook_id, video_id, url) VALUES (1, 'aaaaaaaaaaa', 'u')")
        db.init_db(p)
    finally:
        holder.close()
    b = sqlite3.connect(p + ".v1-backup")
    try:
        assert b.execute("SELECT video_id FROM sources").fetchone()[0] == "aaaaaaaaaaa"
        assert b.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "1"
        assert "kind" not in [r[1] for r in b.execute("PRAGMA table_info(sources)")]
    finally:
        b.close()


def test_existing_backup_is_never_overwritten(tmp_path):
    p = str(tmp_path / "state.sqlite")
    _make_v1(p)
    with open(p + ".v1-backup", "wb") as f:
        f.write(b"precious")
    db.init_db(p)
    with open(p + ".v1-backup", "rb") as f:
        assert f.read() == b"precious"
    assert os.path.exists(p + ".v1-backup.2")
    # a third migration attempt picks the next free name
    _make_v1(p)
    db.init_db(p)
    assert os.path.exists(p + ".v1-backup.3")


def test_failing_migration_leaves_the_database_at_version_1(tmp_path, monkeypatch):
    p = str(tmp_path / "state.sqlite")
    _make_v1(p)
    _insert_v1_source(p)
    cols_before = _columns(p)
    monkeypatch.setitem(db.MIGRATIONS, "1", (
        "ALTER TABLE sources ADD COLUMN kind TEXT NOT NULL DEFAULT 'youtube'",
        "ALTER TABLE no_such_table ADD COLUMN x TEXT"))
    with pytest.raises(db.LegacyDatabaseError):
        db.init_db(p)
    assert _version(p) == "1"
    assert _columns(p) == cols_before
    c = sqlite3.connect(p)
    try:
        assert c.execute("SELECT count(*) FROM sources").fetchone()[0] == 1
    finally:
        c.close()


def test_pending_migration_reports_without_changing_anything(tmp_path):
    p = str(tmp_path / "state.sqlite")
    _make_v1(p)
    c = sqlite3.connect(p)
    try:
        assert db.pending_migration(c) == "1 -> 2"
        assert db.legacy_reason(c) is None
    finally:
        c.close()
    assert _version(p) == "1" and not os.path.exists(p + ".v1-backup")
    db.init_db(p)
    c = sqlite3.connect(p)
    try:
        assert db.pending_migration(c) is None
    finally:
        c.close()


def test_v1_database_is_migrated_and_still_renders(tmp_path):
    from tests.test_output import DOC, _analysis
    p = str(tmp_path / "state.sqlite")
    _make_v1(p)
    c = sqlite3.connect(p)
    try:
        with c:
            c.execute("INSERT INTO notebooks (slug, topic) VALUES ('ai-research', 'AI Research')")
            for vid, t0 in (("aaaaaaaaaaa", 754.0), ("bbbbbbbbbbb", 30.0)):
                c.execute("INSERT INTO sources (notebook_id, video_id, url, title, channel, analyze_status, analysis_json) "
                          "VALUES (1, ?, ?, ?, ?, 'done', ?)",
                          (vid, f"https://www.youtube.com/watch?v={vid}", f"Video {vid[0].upper()}", f"Chan {vid[0].upper()}",
                           json.dumps(_analysis(t0))))
    finally:
        c.close()
    sources_before = {r["id"]: dict(r) for r in _rows(p)}
    from pipeline.schemas import MergedDocument
    before = render_markdown("AI Research", MergedDocument.model_validate(DOC), sources_before, today=date(2026, 9, 6))
    db.init_db(p)
    db.create_merge(p, 1, None, "m", json.dumps(DOC), json.dumps(DOC["source_map"]), "{}",
                    report_path_for(str(tmp_path / "v"), "ai-research", date(2026, 9, 6)))
    out = run_output(p, 1, str(tmp_path / "v"))
    assert open(out, encoding="utf-8").read() == before


def _rows(p):
    c = sqlite3.connect(p)
    c.row_factory = sqlite3.Row
    try:
        return c.execute("SELECT * FROM sources").fetchall()
    finally:
        c.close()


def test_unknown_future_version_is_still_refused(tmp_path):
    p = str(tmp_path / "state.sqlite")
    db.init_db(p)
    c = sqlite3.connect(p)
    try:
        with c:
            c.execute("UPDATE meta SET value='9' WHERE key='schema_version'")
    finally:
        c.close()
    with pytest.raises(db.LegacyDatabaseError, match="schema_version 9"):
        db.init_db(p)
    assert not os.path.exists(p + ".v1-backup")


def test_add_source_stores_the_kind(tmp_path):
    p = str(tmp_path / "s.sqlite")
    db.init_db(p)
    nid = db.create_notebook(p, "T", "t")
    sid = db.add_source(p, nid, "web-0123456789abcdef", "https://e.org/a", kind="web")
    assert db.get_source(p, sid)["kind"] == "web"
    sid2 = db.add_source(p, nid, "aaaaaaaaaaa", "https://y/aaaaaaaaaaa")
    assert db.get_source(p, sid2)["kind"] == "youtube"


def test_already_migrated_database_writes_no_backup(tmp_path):
    """Another process migrated between our version check and our lock: no stale v2 backup."""
    p = str(tmp_path / "state.sqlite")
    _make_v1(p)
    stale = sqlite3.connect(p)            # the caller's view: still sees version 1
    try:
        db.init_db(p)                     # "another process" migrates; makes .v1-backup
        for f in os.listdir(tmp_path):
            if "backup" in f:
                os.remove(tmp_path / f)
        db._migrate(stale, p)
    finally:
        stale.close()
    assert [f for f in os.listdir(tmp_path) if "backup" in f] == []
    assert _version(p) == "2"


def test_read_only_v1_database_fails_cleanly_without_a_backup(tmp_path):
    import stat
    p = str(tmp_path / "state.sqlite")
    _make_v1(p)
    c = sqlite3.connect(p)
    try:
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        c.close()
    os.chmod(p, stat.S_IREAD)
    try:
        with pytest.raises(db.LegacyDatabaseError) as e:
            db.init_db(p)
    finally:
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
    assert p in str(e.value) and "schema migration 1 -> 2 failed" in str(e.value)
    assert [f for f in os.listdir(tmp_path) if "backup" in f] == []
    assert _version(p) == "1"


def test_failing_statement_error_names_the_migration(tmp_path, monkeypatch):
    p = str(tmp_path / "state.sqlite")
    _make_v1(p)
    monkeypatch.setitem(db.MIGRATIONS, "1", ("ALTER TABLE no_such_table ADD COLUMN x TEXT",))
    with pytest.raises(db.LegacyDatabaseError, match="schema migration 1 -> 2 failed"):
        db.init_db(p)
    assert _version(p) == "1"


def test_missing_version_row_with_kindless_sources_is_version_1(tmp_path):
    p = str(tmp_path / "state.sqlite")
    _make_v1(p)
    c = sqlite3.connect(p)
    try:
        with c:
            c.execute("DELETE FROM meta WHERE key='schema_version'")
    finally:
        c.close()
    db.init_db(p)
    assert "kind" in _columns(p)
    assert _version(p) == "2"
