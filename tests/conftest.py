"""Shared pytest options."""
from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--update-fingerprints",
        action="store_true",
        default=False,
        help="tests/test_prompt_versions.py: record the fingerprint of each CURRENT PROMPT_VERSION "
             "in tests/fixtures/prompt_fingerprints.json (refused when that version is already "
             "recorded with a different fingerprint -- bump PROMPT_VERSION first).",
    )


import pytest


@pytest.fixture(autouse=True)
def _no_model_monitor_network(monkeypatch):
    """The model monitor (pipeline/models_monitor.py) fetches public pages and the Models API; no test
    may reach the network, so both fail fast unless a test injects its own fetcher."""
    def _offline(*args, **kwargs):
        raise OSError("network disabled in tests")
    monkeypatch.setattr("pipeline.models_monitor.fetch_text", _offline)
    monkeypatch.setattr("pipeline.models_monitor.list_account_models", _offline)


@pytest.fixture(autouse=True)
def _clear_llm_interrupt():
    """The interrupt flag (pipeline/llm/base.py) is process-wide; no test may leak it into the next."""
    from pipeline.llm.base import clear_interrupt
    clear_interrupt()
    yield
    clear_interrupt()


@pytest.fixture(autouse=True)
def _isolate_config_discovery(monkeypatch):
    """pipeline.cli.main records the config in use for the `.env` lookup (pipeline/envfile.py) and
    honours HMD_CONFIG / HMD_PLUGIN_ROOT (IDEA-310): no test may inherit them from the shell or
    from an earlier test."""
    from pipeline import envfile
    monkeypatch.setattr(envfile, "_config_path", None)
    monkeypatch.delenv("HMD_CONFIG", raising=False)
    monkeypatch.delenv("HMD_PLUGIN_ROOT", raising=False)


# --- Test isolation (Evidence-Echo lesson) ---------------------------------------------------------
# Every test runs with a throwaway home, and the session fails if the repo's real data/ tree or the
# real ~/.claude changed while it ran: a stray Config() default or subprocess cannot touch real data
# silently.

from pathlib import Path

_REPO_DATA = Path(__file__).resolve().parent.parent / "data"
_REAL_CLAUDE_DIR = Path.home() / ".claude"
_snapshots: dict[str, object] = {}


def snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    """Relative path -> (size, mtime_ns) of every file under `root`; a missing root is `{}`."""
    root = Path(root)
    if not root.is_dir():
        return {}
    snap = {}
    for path in root.rglob("*"):
        if path.is_file():
            st = path.stat()
            snap[path.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    return snap


def _top_level_names(root: Path) -> set[str]:
    # Names only: the running Claude Code session itself writes inside ~/.claude while tests run.
    return {p.name for p in root.iterdir()} if root.is_dir() else set()


def _describe(before: dict, after: dict) -> str:
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    return ", ".join(changed[:10]) + (" ..." if len(changed) > 10 else "")


def pytest_sessionstart(session):
    _snapshots["data"] = snapshot_tree(_REPO_DATA)
    _snapshots["claude"] = _top_level_names(_REAL_CLAUDE_DIR)


def pytest_sessionfinish(session, exitstatus):
    if not _snapshots:
        return
    data_after = snapshot_tree(_REPO_DATA)
    claude_after = _top_level_names(_REAL_CLAUDE_DIR)
    problems = []
    if data_after != _snapshots["data"]:
        problems.append(f"data/: {_describe(_snapshots['data'], data_after)}")
    if claude_after != _snapshots["claude"]:
        problems.append(f"~/.claude: {sorted(claude_after ^ _snapshots['claude'])}")
    if problems:
        pytest.exit("tests modified real data: " + "; ".join(problems), returncode=3)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """HOME/USERPROFILE and the Claude config/plugin-data dirs point into a per-test temp dir."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(home / "plugin-data"))
    return home
