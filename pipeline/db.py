"""SQLite run state: notebooks, sources, runs, usage_events, merges.

One connection per call, 30 s busy timeout (IDEA-075), always closed. WAL is set once in
init_db, never per connection. No migration from the MILESTONE-0 `projects` schema
(ARCHITECTURE §10): a legacy file, or a file with an incompatible schema_version, is
refused with a message.
"""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from typing import Optional

SCHEMA_VERSION = "2"
BUSY_TIMEOUT_SECONDS = 30
# from-version -> statements that bring it to the next version (DECISION-031). Additive only.
MIGRATIONS: dict[str, tuple[str, ...]] = {
    "1": ("ALTER TABLE sources ADD COLUMN kind TEXT NOT NULL DEFAULT 'youtube'",),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS notebooks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    topic TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    notebook_id INTEGER NOT NULL REFERENCES notebooks(id),
    video_id TEXT NOT NULL,
    url TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'youtube',
    title TEXT,
    channel TEXT,
    duration_seconds INTEGER,
    upload_date TEXT,
    fetch_status TEXT DEFAULT 'pending',
    transcribe_status TEXT DEFAULT 'pending',
    analyze_status TEXT DEFAULT 'pending',
    transcript_path TEXT,
    transcript_method TEXT,
    transcript_language TEXT,
    analysis_json TEXT,
    error_message TEXT,
    UNIQUE(notebook_id, video_id)
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    notebook_id INTEGER NOT NULL REFERENCES notebooks(id),
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    status TEXT DEFAULT 'running',
    provider TEXT,
    budget_usd REAL,
    stages TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    notebook_id INTEGER,
    source_id INTEGER,
    stage TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    cost_usd REAL,
    duration_ms INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS merges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    notebook_id INTEGER NOT NULL REFERENCES notebooks(id),
    run_id INTEGER REFERENCES runs(id),
    model_used TEXT,
    merged_json TEXT NOT NULL,
    source_map_json TEXT NOT NULL,
    prompt_versions TEXT,
    report_path TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runs_notebook ON runs(notebook_id);
CREATE INDEX IF NOT EXISTS idx_usage_run ON usage_events(run_id);
"""


class LegacyDatabaseError(Exception):
    pass


def _conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _stored_version(conn: sqlite3.Connection) -> Optional[str]:
    """meta.schema_version; a meta table with no version row but a `sources` table lacking
    `kind` is a version-1 database. None when there is nothing to migrate."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "meta" not in tables:
        return None
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is not None:
        return row[0]
    if "sources" in tables and "kind" not in [r[1] for r in conn.execute("PRAGMA table_info(sources)")]:
        return "1"
    return None


def _version_path(version: str) -> list[str]:
    """Versions to step through from `version` to SCHEMA_VERSION, or [] if there is no path."""
    steps, v = [], version
    while v != SCHEMA_VERSION:
        if v not in MIGRATIONS:
            return []
        steps.append(v)
        v = str(int(v) + 1)
    return steps


def pending_migration(conn: sqlite3.Connection) -> Optional[str]:
    """`"1 -> 2"` when the database is at an older version that has a migration path, else None."""
    version = _stored_version(conn)
    if version is None or version == SCHEMA_VERSION or not _version_path(version):
        return None
    return f"{version} -> {SCHEMA_VERSION}"


def legacy_reason(conn: sqlite3.Connection) -> Optional[str]:
    """Inspect an already-open connection's tables and return a human-readable reason the
    database is incompatible with this code's schema, or None if it isn't. Read-only: never
    creates a table. Covers both rejection cases `init_db` refuses: a MILESTONE-0 database
    (table `projects`, no `meta`) and a `meta.schema_version` with no migration path to SCHEMA_VERSION.
    Shared by `init_db` (below) and `hmd doctor` (pipeline/doctor.py) so the rejection rules
    live in exactly one place."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "projects" in tables and "meta" not in tables:
        return ("is a MILESTONE-0 database (table `projects`). There is no migration: rename "
                "or delete it and re-run the notebook.")
    if "meta" in tables:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is not None and row[0] != SCHEMA_VERSION and not _version_path(row[0]):
            return (f"has schema_version {row[0]}, but this code expects {SCHEMA_VERSION}. "
                    "There is no migration: rename or delete it and re-run the notebook.")
    return None


def _reserve_backup(db_path: str, from_version: str) -> str:
    """Atomically create the first free `<db>.v<N>-backup`, `.2`, `.3`, ... (O_EXCL): an
    existing backup is never overwritten and two processes never pick the same name."""
    base = f"{db_path}.v{from_version}-backup"
    n = 1
    while True:
        path = base if n == 1 else f"{base}.{n}"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            n += 1
            continue
        os.close(fd)
        return path


def _backup_under_lock(db_path: str, from_version: str) -> str:
    """Copy the file with the sqlite backup API through a SECOND connection (the migrating one
    holds the write lock; in WAL mode a reader sees exactly the committed pre-migration state)."""
    path = _reserve_backup(db_path, from_version)
    try:
        src = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_SECONDS)
        try:
            dst = sqlite3.connect(path)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
    except BaseException:
        _discard(path)
        raise
    return path


def _discard(path: str) -> None:
    with contextlib.suppress(OSError):
        os.remove(path)


def _migrate(conn: sqlite3.Connection, db_path: str) -> None:
    """Take the write lock, re-read the version, and only if a step is still needed back the
    file up (while the lock is held, so the copy is exactly the pre-migration state) and run
    every step's statements plus the version bump in ONE transaction; a failure rolls back."""
    target = f"{_stored_version(conn)} -> {SCHEMA_VERSION}"
    backup = None
    conn.isolation_level = None          # explicit transaction control: DDL must roll back too
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            version = _stored_version(conn)      # re-read under the lock: another process may have migrated
            steps = _version_path(version) if version is not None else []
            if steps:
                target = f"{version} -> {SCHEMA_VERSION}"
                backup = _backup_under_lock(db_path, version)
                for step in steps:
                    for statement in MIGRATIONS[step]:
                        conn.execute(statement)
                    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                                 (str(int(step) + 1),))
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
            if backup is not None:               # nothing changed: do not leave an orphan backup
                _discard(backup)
            raise
    except (sqlite3.Error, OSError) as e:
        raise LegacyDatabaseError(f"{db_path}: schema migration {target} failed: {e}") from e
    finally:
        conn.isolation_level = ""


def init_db(db_path: str) -> None:
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with contextlib.closing(_conn(db_path)) as conn, conn:
        reason = legacy_reason(conn)
        if reason is not None:
            raise LegacyDatabaseError(f"{db_path} {reason}")
        if pending_migration(conn) is not None:
            _migrate(conn, db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,))


# --- notebooks -------------------------------------------------------------

def create_notebook(db_path: str, topic: str, slug: str) -> int:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        cur = conn.execute("INSERT INTO notebooks (topic, slug) VALUES (?, ?)", (topic, slug))
        return cur.lastrowid


def get_notebook(db_path: str, ref: int | str) -> Optional[dict]:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        row = None
        if isinstance(ref, str):
            row = conn.execute("SELECT * FROM notebooks WHERE slug=?", (ref,)).fetchone()
        if row is None and (isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit())):
            row = conn.execute("SELECT * FROM notebooks WHERE id=?", (int(ref),)).fetchone()
        return dict(row) if row else None


def list_notebooks(db_path: str) -> list[dict]:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        return [dict(r) for r in conn.execute("SELECT * FROM notebooks ORDER BY id")]


def notebook_slugs(db_path: str) -> set[str]:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        return {r[0] for r in conn.execute("SELECT slug FROM notebooks")}


# --- sources ---------------------------------------------------------------

def add_source(db_path: str, notebook_id: int, video_id: str, url: str, *,
               kind: str = "youtube", title: str | None = None, channel: str | None = None,
               duration_seconds: int | None = None, upload_date: str | None = None) -> int:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        conn.execute(
            """INSERT INTO sources (notebook_id, video_id, url, kind, title, channel, duration_seconds, upload_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(notebook_id, video_id) DO NOTHING""",
            (notebook_id, video_id, url, kind, title, channel, duration_seconds, upload_date))
        row = conn.execute("SELECT id FROM sources WHERE notebook_id=? AND video_id=?",
                           (notebook_id, video_id)).fetchone()
        if row is None:
            raise RuntimeError(
                f"add_source: no row found for notebook_id={notebook_id} video_id={video_id!r} "
                "after insert (unexpected)")
        return row["id"]


def update_source_metadata(db_path: str, source_id: int, *, title: str | None = None, channel: str | None = None,
                           duration_seconds: int | None = None, upload_date: str | None = None,
                           status: str = "done", error: str | None = None) -> None:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        fields = "title=COALESCE(?, title), channel=COALESCE(?, channel), " \
                 "duration_seconds=COALESCE(?, duration_seconds), upload_date=COALESCE(?, upload_date), fetch_status=?"
        params: list = [title, channel, duration_seconds, upload_date, status]
        if status == "error":
            if error is not None:
                fields += ", error_message=?"
                params.append(error)
        else:
            fields += ", error_message=NULL"
        params.append(source_id)
        conn.execute(f"UPDATE sources SET {fields} WHERE id=?", params)


def get_source(db_path: str, source_id: int) -> Optional[dict]:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        row = conn.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
        return dict(row) if row else None


def get_notebook_sources(db_path: str, notebook_id: int, **filters: str) -> list[dict]:
    """Sources of a notebook ordered by id. filters: fetch_status=, transcribe_status=, analyze_status=."""
    allowed = {"fetch_status", "transcribe_status", "analyze_status"}
    bad = set(filters) - allowed
    if bad:
        raise ValueError(f"unknown filters {bad}")
    where = "notebook_id=?"
    params: list = [notebook_id]
    for k, v in filters.items():
        where += f" AND {k}=?"
        params.append(v)
    with contextlib.closing(_conn(db_path)) as conn, conn:
        rows = conn.execute(f"SELECT * FROM sources WHERE {where} ORDER BY id", params).fetchall()
        return [dict(r) for r in rows]


def get_analyzed_sources(db_path: str, notebook_id: int) -> list[dict]:
    return get_notebook_sources(db_path, notebook_id, analyze_status="done")


def update_transcribe_status(db_path: str, source_id: int, status: str, *, transcript_path: str | None = None,
                             method: str | None = None, language: str | None = None,
                             error: str | None = None) -> None:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        fields = ["transcribe_status=?"]
        params: list = [status]
        if transcript_path is not None:
            fields.append("transcript_path=?")
            params.append(transcript_path)
        if method is not None:
            fields.append("transcript_method=?")
            params.append(method)
        if language is not None:
            fields.append("transcript_language=?")
            params.append(language)
        if status == "error":
            if error is not None:
                fields.append("error_message=?")
                params.append(error)
        else:
            fields.append("error_message=NULL")
        params.append(source_id)
        conn.execute(f"UPDATE sources SET {', '.join(fields)} WHERE id=?", params)


def update_analyze_status(db_path: str, source_id: int, status: str, *, analysis_json: str | None = None,
                          error: str | None = None) -> None:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        fields = ["analyze_status=?"]
        params: list = [status]
        if analysis_json is not None:
            fields.append("analysis_json=?")
            params.append(analysis_json)
        if status == "error":
            if error is not None:
                fields.append("error_message=?")
                params.append(error)
        else:
            fields.append("error_message=NULL")
        params.append(source_id)
        conn.execute(f"UPDATE sources SET {', '.join(fields)} WHERE id=?", params)


def reset_analyze_status(db_path: str, notebook_id: int, source_ids: Optional[list] = None) -> None:
    """Notebook-wide by default; pass source_ids to reset only those sources (e.g. the ones the
    runner just (re)transcribed in this run -- see pipeline/runner.py)."""
    if source_ids is not None and not source_ids:
        return
    where = "notebook_id=?"
    params: list = [notebook_id]
    if source_ids is not None:
        where += f" AND id IN ({','.join('?' for _ in source_ids)})"
        params.extend(source_ids)
    with contextlib.closing(_conn(db_path)) as conn, conn:
        conn.execute(
            f"UPDATE sources SET analyze_status='pending', analysis_json=NULL, "
            f"error_message = CASE WHEN analyze_status='error' THEN NULL ELSE error_message END "
            f"WHERE {where}",
            params)


# --- runs and usage --------------------------------------------------------

def create_run(db_path: str, notebook_id: int, *, provider: str, budget_usd: float, stages: str) -> int:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        cur = conn.execute("INSERT INTO runs (notebook_id, provider, budget_usd, stages) VALUES (?, ?, ?, ?)",
                           (notebook_id, provider, budget_usd, stages))
        return cur.lastrowid


def finish_run(db_path: str, run_id: int, status: str, error: str | None = None) -> None:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        conn.execute("UPDATE runs SET status=?, error=?, finished_at=CURRENT_TIMESTAMP WHERE id=?",
                     (status, error, run_id))


def get_run(db_path: str, run_id: int) -> Optional[dict]:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None


def latest_run(db_path: str, notebook_id: int) -> Optional[dict]:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        row = conn.execute("SELECT * FROM runs WHERE notebook_id=? ORDER BY id DESC LIMIT 1",
                           (notebook_id,)).fetchone()
        return dict(row) if row else None


def add_usage_event(db_path: str, *, run_id: int | None, notebook_id: int | None, source_id: int | None,
                    stage: str, provider: str, model: str, input_tokens: int, output_tokens: int,
                    cache_read_tokens: int, cache_write_tokens: int, cost_usd: float | None,
                    duration_ms: int) -> int:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        cur = conn.execute(
            """INSERT INTO usage_events (run_id, notebook_id, source_id, stage, provider, model, input_tokens,
               output_tokens, cache_read_tokens, cache_write_tokens, cost_usd, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, notebook_id, source_id, stage, provider, model, input_tokens, output_tokens,
             cache_read_tokens, cache_write_tokens, cost_usd, duration_ms))
        return cur.lastrowid


def run_cost(db_path: str, run_id: int) -> float:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM usage_events WHERE run_id=?", (run_id,)).fetchone()
        return float(row[0])


def run_usage_summary(db_path: str, run_id: int) -> list[dict]:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        rows = conn.execute(
            """SELECT stage, provider, model, COUNT(*) AS calls, SUM(input_tokens) AS input_tokens,
                      SUM(output_tokens) AS output_tokens, SUM(cache_read_tokens) AS cache_read_tokens,
                      SUM(cache_write_tokens) AS cache_write_tokens, SUM(cost_usd) AS cost_usd,
                      SUM(duration_ms) AS duration_ms
               FROM usage_events WHERE run_id=? GROUP BY stage, provider, model ORDER BY stage""",
            (run_id,)).fetchall()
        return [dict(r) for r in rows]


def notebook_usage_events(db_path: str, notebook_id: int, run_id: int | None = None) -> list[dict]:
    """Per-call usage rows of a notebook's runs (one run when run_id is given), oldest first."""
    sql = ("SELECT run_id, stage, provider, model, input_tokens, output_tokens, cache_read_tokens, "
           "cache_write_tokens, cost_usd, duration_ms "
           "FROM usage_events WHERE run_id IN (SELECT id FROM runs WHERE notebook_id=?)")
    params: list = [notebook_id]
    if run_id is not None:
        sql += " AND run_id=?"
        params.append(run_id)
    with contextlib.closing(_conn(db_path)) as conn, conn:
        return [dict(r) for r in conn.execute(sql + " ORDER BY id", params).fetchall()]


# --- merges ----------------------------------------------------------------

def create_merge(db_path: str, notebook_id: int, run_id: int | None, model_used: str, merged_json: str,
                 source_map_json: str, prompt_versions: str, report_path: str) -> int:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        cur = conn.execute(
            """INSERT INTO merges (notebook_id, run_id, model_used, merged_json, source_map_json, prompt_versions,
               report_path) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (notebook_id, run_id, model_used, merged_json, source_map_json, prompt_versions, report_path))
        return cur.lastrowid


def get_latest_merge(db_path: str, notebook_id: int) -> Optional[dict]:
    with contextlib.closing(_conn(db_path)) as conn, conn:
        row = conn.execute("SELECT * FROM merges WHERE notebook_id=? ORDER BY id DESC LIMIT 1",
                           (notebook_id,)).fetchone()
        return dict(row) if row else None


def get_previous_merge(db_path: str, notebook_id: int) -> Optional[dict]:
    """The merge before the latest one (the light diff's baseline, IDEA-012), or None."""
    with contextlib.closing(_conn(db_path)) as conn, conn:
        row = conn.execute("SELECT * FROM merges WHERE notebook_id=? ORDER BY id DESC LIMIT 1 OFFSET 1",
                           (notebook_id,)).fetchone()
        return dict(row) if row else None


def recent_runs_with_merge(db_path: str, limit: int = 5) -> list[dict]:
    """Token totals of the newest finished runs that have both analyze and merge usage and a stored
    merge (the dry-run calibration, IDEA-003), newest first: analyze_out, analyze_calls, merge_in,
    merge_out, the merge split into section_in/section_out (section and check calls, the ones a
    batch run sends through Message Batches) and direct_in/direct_out (outline and verify, always
    direct), merged_sources (the size of that run's merge source map), the run's provider and its
    analyze_model / merge_model (None when a stage used more than one model). A run whose stored
    source map cannot be read is left out."""
    with contextlib.closing(_conn(db_path)) as conn, conn:
        rows = conn.execute(
            """SELECT r.id, r.provider,
                      CASE WHEN MIN(CASE WHEN u.stage = 'analyze' THEN u.model END)
                                = MAX(CASE WHEN u.stage = 'analyze' THEN u.model END)
                           THEN MIN(CASE WHEN u.stage = 'analyze' THEN u.model END) END AS analyze_model,
                      CASE WHEN MIN(CASE WHEN u.stage LIKE 'merge%' THEN u.model END)
                                = MAX(CASE WHEN u.stage LIKE 'merge%' THEN u.model END)
                           THEN MIN(CASE WHEN u.stage LIKE 'merge%' THEN u.model END) END AS merge_model,
                      SUM(CASE WHEN u.stage IN ('merge:outline', 'merge:verify') THEN u.input_tokens ELSE 0 END)
                          AS direct_in,
                      SUM(CASE WHEN u.stage IN ('merge:outline', 'merge:verify') THEN u.output_tokens ELSE 0 END)
                          AS direct_out,
                      SUM(CASE WHEN u.stage = 'analyze' THEN u.output_tokens ELSE 0 END) AS analyze_out,
                      -- distinct sources, so a retried analyze call is counted once; events without a
                      -- source id (older or synthetic rows) fall back to counting the calls
                      CASE WHEN COUNT(DISTINCT CASE WHEN u.stage = 'analyze' THEN u.source_id END) > 0
                           THEN COUNT(DISTINCT CASE WHEN u.stage = 'analyze' THEN u.source_id END)
                           ELSE SUM(CASE WHEN u.stage = 'analyze' THEN 1 ELSE 0 END) END AS analyze_calls,
                      SUM(CASE WHEN u.stage LIKE 'merge%' THEN u.input_tokens ELSE 0 END) AS merge_in,
                      SUM(CASE WHEN u.stage LIKE 'merge%' THEN u.output_tokens ELSE 0 END) AS merge_out,
                      (SELECT m.source_map_json FROM merges m WHERE m.run_id = r.id
                       ORDER BY m.id DESC LIMIT 1) AS source_map_json
               FROM runs r JOIN usage_events u ON u.run_id = r.id
               WHERE r.status = 'done'
               GROUP BY r.id
               HAVING analyze_out > 0 AND merge_in > 0 AND source_map_json IS NOT NULL
               ORDER BY r.id DESC LIMIT ?""", (limit,)).fetchall()
    out = []
    for r in rows:
        row = dict(r)
        try:
            source_map = json.loads(row.pop("source_map_json"))
        except (TypeError, ValueError):
            continue
        if not isinstance(source_map, dict):
            continue
        row["merged_sources"] = len(source_map)
        row["section_in"] = row["merge_in"] - row["direct_in"]
        row["section_out"] = row["merge_out"] - row["direct_out"]
        out.append(row)
    return out
