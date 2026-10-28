"""Test isolation (Evidence-Echo lesson): no test may see the developer's real home or touch the
repo's real data/ (tests/conftest.py)."""
from __future__ import annotations

import os


def test_home_is_isolated():
    assert "pytest" in os.environ["HOME"].lower() or "tmp" in os.environ["HOME"].lower()
    assert os.path.expanduser("~") == os.environ["HOME"] or os.name == "nt"
    assert os.environ["USERPROFILE"] == os.environ["HOME"]


def test_claude_dirs_point_into_the_isolated_home():
    home = os.environ["HOME"]
    assert os.environ["CLAUDE_CONFIG_DIR"].startswith(home)
    assert os.environ["CLAUDE_PLUGIN_DATA"].startswith(home)


def test_data_snapshot_detects_change(tmp_path):
    from tests.conftest import snapshot_tree
    (tmp_path / "a.txt").write_text("x")
    before = snapshot_tree(tmp_path)
    (tmp_path / "a.txt").write_text("xy")
    assert snapshot_tree(tmp_path) != before


def test_snapshot_of_missing_dir_is_empty(tmp_path):
    from tests.conftest import snapshot_tree
    assert snapshot_tree(tmp_path / "absent") == {}


def test_session_guard_fails_when_real_data_changes(tmp_path, monkeypatch):
    import pytest
    from tests import conftest
    data = tmp_path / "data"
    data.mkdir()
    (data / "state.sqlite").write_text("x")
    monkeypatch.setattr(conftest, "_REPO_DATA", data)
    monkeypatch.setattr(conftest, "_REAL_CLAUDE_DIR", tmp_path / "claude")
    monkeypatch.setattr(conftest, "_snapshots", {})
    conftest.pytest_sessionstart(None)
    conftest.pytest_sessionfinish(None, 0)  # unchanged: no exit
    (data / "state.sqlite").write_text("changed")
    with pytest.raises(pytest.exit.Exception) as exc:
        conftest.pytest_sessionfinish(None, 0)
    assert exc.value.returncode == 3
    assert "state.sqlite" in str(exc.value)
