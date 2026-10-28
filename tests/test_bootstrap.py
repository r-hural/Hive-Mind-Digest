"""pipeline/bootstrap.py (IDEA-310): the plugin venv /hmd:init creates. No real venv, no real pip:
every subprocess goes through a fake runner, and every path is under tmp_path."""
from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
import sys

import pytest

from pipeline import bootstrap

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _plugin_root(tmp_path, *, version="0.1.0", requirements="pydantic==2.13.5\n", pyproject_version="0.1.0.dev0",
                 requires=">=3.10", manifest=True):
    root = tmp_path / "plugin-root"
    (root / "pipeline").mkdir(parents=True)
    (root / "pipeline" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pipeline" / "__pycache__").mkdir()
    (root / "pipeline" / "__pycache__" / "x.pyc").write_bytes(b"junk")
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "hive-mind-digest"\nversion = "{pyproject_version}"\n'
        f'requires-python = "{requires}"\n', encoding="utf-8")
    (root / "requirements.txt").write_bytes(requirements.encode("utf-8"))     # exact line endings
    if manifest:
        (root / ".claude-plugin").mkdir()
        (root / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "hmd", "version": version}),
                                                            encoding="utf-8")
    return root


class FakeRunner:
    """Records commands; `-m venv <dir>` creates the venv's hmd executable like the real thing."""

    def __init__(self, fail_on=None):
        self.calls, self.fail_on, self.package_dirs = [], fail_on, []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        if self.fail_on and self.fail_on in " ".join(cmd):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ERROR: boom")
        if cmd[1:3] == ["-m", "venv"]:
            hmd = pathlib.Path(bootstrap.venv_hmd(cmd[3]))
            hmd.parent.mkdir(parents=True)
            hmd.write_text("exe", encoding="utf-8")
        elif "install" in cmd and "-r" not in cmd:
            pkg = pathlib.Path(cmd[-1])
            self.package_dirs.append(sorted(p.relative_to(pkg).as_posix() for p in pkg.rglob("*")))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def _ensure(root, data, runner, version_info=(3, 13, 1), lines=None):
    lines = [] if lines is None else lines
    code = bootstrap.ensure(str(root), str(data), runner=runner, python="/usr/bin/python3",
                            version_info=version_info, log=lines.append)
    return code, lines


# --- pure functions ----------------------------------------------------------------------------

def test_min_python_comes_from_pyproject(tmp_path):
    assert bootstrap.min_python(str(_plugin_root(tmp_path, requires=">=3.12"))) == (3, 12)
    assert bootstrap.min_python(str(tmp_path / "nowhere")) == bootstrap.DEFAULT_MIN_PYTHON


def test_min_python_of_this_repo_matches_pyproject():
    assert bootstrap.min_python(str(REPO_ROOT)) == (3, 12)


def test_plugin_version_prefers_the_plugin_manifest(tmp_path):
    root = _plugin_root(tmp_path, version="0.2.0", pyproject_version="0.1.0.dev0")
    assert bootstrap.plugin_version(str(root)) == "0.2.0"
    (root / ".claude-plugin" / "plugin.json").unlink()
    assert bootstrap.plugin_version(str(root)) == "0.1.0.dev0"


def test_requirements_hash_ignores_line_endings(tmp_path):
    a = _plugin_root(tmp_path / "a", requirements="x==1\ny==2\n")
    b = _plugin_root(tmp_path / "b", requirements="x==1\r\ny==2\r\n")
    c = _plugin_root(tmp_path / "c", requirements="x==1\ny==3\n")
    assert bootstrap.requirements_sha256(str(a)) == bootstrap.requirements_sha256(str(b))
    assert bootstrap.requirements_sha256(str(a)) != bootstrap.requirements_sha256(str(c))


def test_stale_reason():
    want = {"plugin_version": "0.2.0", "requirements_sha256": "abc"}
    assert bootstrap.stale_reason(None, want) == "not installed"
    assert bootstrap.stale_reason({"plugin_version": "0.2.0", "requirements_sha256": "abc"}, want) is None
    assert bootstrap.stale_reason({"plugin_version": "0.1.0", "requirements_sha256": "abc"}, want) == \
        "plugin version 0.1.0 -> 0.2.0"
    assert bootstrap.stale_reason({"plugin_version": "0.2.0", "requirements_sha256": "old"}, want) == \
        "requirements.txt changed"


def test_read_marker_tolerates_garbage(tmp_path):
    assert bootstrap.read_marker(str(tmp_path)) is None
    (tmp_path / bootstrap.MARKER_NAME).write_text("not json", encoding="utf-8")
    assert bootstrap.read_marker(str(tmp_path)) is None
    (tmp_path / bootstrap.MARKER_NAME).write_text("[1]", encoding="utf-8")
    assert bootstrap.read_marker(str(tmp_path)) is None


def test_launcher_runs_the_venv_hmd_with_forward_slashes(tmp_path):
    text = bootstrap.launcher_text(str(tmp_path / "venv"))
    assert text.startswith("#!/bin/sh\n")
    target = bootstrap.venv_hmd(str((tmp_path / "venv").resolve())).replace("\\", "/")
    assert f'exec "{target}" "$@"' in text
    assert "\\" not in text


# --- ensure() ----------------------------------------------------------------------------------

def test_too_old_python_stops_before_anything_runs(tmp_path):
    runner = FakeRunner()
    code, lines = _ensure(_plugin_root(tmp_path, requires=">=3.12"), tmp_path / "data", runner,
                          version_info=(3, 11, 9))
    assert code == bootstrap.EXIT_USAGE and runner.calls == []
    assert "BOOTSTRAP=fail" in lines
    assert any("needs Python 3.12 or newer; this is 3.11" in line for line in lines)
    assert not (tmp_path / "data").exists()


def test_not_a_plugin_root_is_refused(tmp_path):
    runner = FakeRunner()
    (tmp_path / "empty").mkdir()
    code, lines = _ensure(tmp_path / "empty", tmp_path / "data", runner)
    assert code == bootstrap.EXIT_USAGE and runner.calls == [] and "BOOTSTRAP=fail" in lines


def test_fresh_install_creates_venv_installs_and_records_the_marker(tmp_path):
    root, data = _plugin_root(tmp_path, version="0.1.0"), tmp_path / "data"
    runner = FakeRunner()
    code, lines = _ensure(root, data, runner)
    assert code == 0 and "BOOTSTRAP=installed" in lines
    venv = data / "venv" / bootstrap.env_digest(str(root))
    vpy = bootstrap.venv_python(str(venv))
    assert runner.calls[0] == ["/usr/bin/python3", "-m", "venv", str(venv)]
    assert runner.calls[1][:4] == [vpy, "-m", "pip", "install"]
    assert runner.calls[1][-2:] == ["-r", str(root / "requirements.txt")]
    assert runner.calls[2][:4] == [vpy, "-m", "pip", "install"] and "-r" not in runner.calls[2]
    # the package installs non-editable from a temp copy: never from (or into) the plugin root
    assert "-e" not in runner.calls[2] and runner.calls[2][-1] != str(root)
    assert runner.package_dirs == [["pipeline", "pipeline/__init__.py", "pyproject.toml"]]
    assert not os.path.exists(runner.calls[2][-1])                    # temp copy cleaned up
    marker = json.loads((venv / bootstrap.MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["plugin_version"] == "0.1.0"
    assert marker["requirements_sha256"] == bootstrap.requirements_sha256(str(root))
    assert marker["python"] == "3.13.1" and marker["installed_at"]
    launcher = data / bootstrap.LAUNCHER_NAME
    assert launcher.read_bytes().startswith(b"#!/bin/sh\n") and b"\r\n" not in launcher.read_bytes()
    assert f"VENV={venv}" in lines and f"HMD={launcher}" in lines
    assert f"CONFIG={data / 'config.yaml'} (not written yet)" in lines
    assert f"SUGGESTED_HOME={os.path.abspath(os.path.expanduser(os.path.join('~', 'HiveMindDigest')))}" in lines
    assert sorted(p.name for p in root.iterdir()) == [".claude-plugin", "pipeline", "pyproject.toml",
                                                      "requirements.txt"]


def test_up_to_date_install_is_a_no_op(tmp_path):
    root, data = _plugin_root(tmp_path), tmp_path / "data"
    _ensure(root, data, FakeRunner())
    (data / "config.yaml").write_text("home: x\n", encoding="utf-8")
    runner = FakeRunner()
    code, lines = _ensure(root, data, runner)
    assert code == 0 and runner.calls == [] and "BOOTSTRAP=ok" in lines
    assert f"CONFIG={data / 'config.yaml'}" in lines


@pytest.mark.parametrize("change", ["version", "requirements"])
def test_a_plugin_update_reinstalls_into_a_fresh_venv(tmp_path, change):
    root, data = _plugin_root(tmp_path), tmp_path / "data"
    _ensure(root, data, FakeRunner())
    old = data / "venv" / bootstrap.env_digest(str(root))
    (old / "leftover.txt").write_text("old", encoding="utf-8")
    (data / "config.yaml").write_text("home: x\n", encoding="utf-8")
    (data / ".env").write_text("ANTHROPIC_API_KEY=k\n", encoding="utf-8")
    if change == "version":
        (root / ".claude-plugin" / "plugin.json").write_text('{"version": "0.2.0"}', encoding="utf-8")
    else:
        (root / "requirements.txt").write_text("pydantic==2.14.0\n", encoding="utf-8")
    assert bootstrap.install_status(str(root), str(data)) is not None
    runner = FakeRunner()
    code, lines = _ensure(root, data, runner)
    assert code == 0 and "BOOTSTRAP=installed" in lines and len(runner.calls) == 3
    assert (old / "leftover.txt").exists()                          # the old venv stays (in use?)
    assert not (data / "venv" / bootstrap.env_digest(str(root)) / "leftover.txt").exists()
    assert (data / "config.yaml").is_file() and (data / ".env").is_file()   # user files kept
    assert bootstrap.install_status(str(root), str(data)) is None


def test_missing_launcher_is_rewritten(tmp_path):
    root, data = _plugin_root(tmp_path), tmp_path / "data"
    _ensure(root, data, FakeRunner())
    (data / bootstrap.LAUNCHER_NAME).unlink()
    code, lines = _ensure(root, data, FakeRunner())
    assert code == 0 and (data / bootstrap.LAUNCHER_NAME).is_file()


def test_pip_failure_reports_and_records_no_marker(tmp_path):
    root, data = _plugin_root(tmp_path), tmp_path / "data"
    code, lines = _ensure(root, data, FakeRunner(fail_on="requirements.txt"))
    assert code == bootstrap.EXIT_ERROR and "BOOTSTRAP=fail" in lines
    assert any("ERROR: boom" in line for line in lines)
    assert bootstrap.read_marker(str(data / "venv" / bootstrap.env_digest(str(root)))) is None
    assert not (data / bootstrap.LAUNCHER_NAME).exists()


def test_windows_long_path_failure_gets_a_hint(tmp_path):
    class LongPath(FakeRunner):
        def __call__(self, cmd, **kw):
            if "-r" in cmd:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=(
                    "ERROR: Could not install packages due to an OSError\nHINT: This error might have "
                    "occurred since this system does not have Windows Long Path support enabled."))
            return super().__call__(cmd, **kw)
    code, lines = _ensure(_plugin_root(tmp_path), tmp_path / "data", LongPath())
    assert code == bootstrap.EXIT_ERROR
    assert any("260-character path limit" in line for line in lines)


def test_main_defaults_the_root_to_the_folder_holding_pipeline(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(bootstrap, "ensure", lambda root, data: seen.update(root=root, data=data) or 0)
    assert bootstrap.main([str(tmp_path)]) == 0
    assert pathlib.Path(seen["root"]) == REPO_ROOT and seen["data"] == str(tmp_path)


# --- runs as a plain script before any requirement is installed --------------------------------

def _imports(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    return names


def test_bootstrap_imports_the_standard_library_only():
    names = _imports(REPO_ROOT / "pipeline" / "bootstrap.py")
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is None:
        pytest.skip("sys.stdlib_module_names needs Python 3.10+")
    assert names <= set(stdlib), names - set(stdlib)


def test_no_pipeline_module_shadows_a_bootstrap_import():
    # Run as a script, pipeline/ is sys.path[0]: a pipeline/json.py would replace the stdlib json.
    modules = {p.stem for p in (REPO_ROOT / "pipeline").glob("*.py")} | \
        {p.name for p in (REPO_ROOT / "pipeline").iterdir() if p.is_dir()}
    assert not (modules & _imports(REPO_ROOT / "pipeline" / "bootstrap.py"))


def test_bootstrap_runs_as_a_script_and_refuses_an_unknown_flag():
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "pipeline" / "bootstrap.py"), "--help"],
                          capture_output=True, text=True, cwd=str(REPO_ROOT.parent))
    assert proc.returncode == 0 and "data_dir" in proc.stdout


# --- one venv per requirements digest, lock, idle pruning (Evidence-Echo lesson) -----------------

def _digest_venv(root, data):
    return data / "venv" / bootstrap.env_digest(str(root))


def test_env_digest_ignores_line_endings_and_tracks_every_input(tmp_path):
    a = _plugin_root(tmp_path / "a", requirements="x==1\ny==2\n")
    b = _plugin_root(tmp_path / "b", requirements="x==1\r\ny==2\r\n")
    digest = bootstrap.env_digest(str(a))
    assert len(digest) == 12 and int(digest, 16) >= 0
    assert digest == bootstrap.env_digest(str(b))
    (a / "constraints.txt").write_text("z==3\n", encoding="utf-8")
    assert bootstrap.env_digest(str(a)) != digest
    c = _plugin_root(tmp_path / "c", requirements="x==1\ny==2\n", version="0.2.0")
    assert bootstrap.env_digest(str(c)) != digest


def test_new_digest_builds_beside_old_venv(tmp_path):
    root, data = _plugin_root(tmp_path), tmp_path / "data"
    _ensure(root, data, FakeRunner())
    old = _digest_venv(root, data)
    (old / "in-use.txt").write_text("a running hmd", encoding="utf-8")
    (root / "requirements.txt").write_text("pydantic==2.14.0\n", encoding="utf-8")
    new = _digest_venv(root, data)
    assert new != old
    runner = FakeRunner()
    code, lines = _ensure(root, data, runner)
    assert code == 0 and "BOOTSTRAP=installed" in lines
    assert runner.calls[0] == ["/usr/bin/python3", "-m", "venv", str(new)]
    assert (old / "in-use.txt").is_file() and bootstrap.read_marker(str(old)) is not None
    launcher = (data / bootstrap.LAUNCHER_NAME).read_text(encoding="utf-8")
    assert bootstrap.venv_hmd(str(new.resolve())).replace("\\", "/") in launcher
    assert f"VENV={new}" in lines


def test_same_digest_is_ok_without_install(tmp_path):
    root, data = _plugin_root(tmp_path), tmp_path / "data"
    _ensure(root, data, FakeRunner())
    runner = FakeRunner()
    code, lines = _ensure(root, data, runner)
    assert code == 0 and runner.calls == [] and "BOOTSTRAP=ok" in lines


def test_failed_install_keeps_the_launcher_on_the_old_venv(tmp_path):
    root, data = _plugin_root(tmp_path), tmp_path / "data"
    _ensure(root, data, FakeRunner())
    before = (data / bootstrap.LAUNCHER_NAME).read_text(encoding="utf-8")
    (root / "requirements.txt").write_text("pydantic==2.14.0\n", encoding="utf-8")
    code, _ = _ensure(root, data, FakeRunner(fail_on="requirements.txt"))
    assert code == bootstrap.EXIT_ERROR
    assert (data / bootstrap.LAUNCHER_NAME).read_text(encoding="utf-8") == before


def test_lock_held_waits_then_fails_with_exit_2(tmp_path):
    import time
    root, data = _plugin_root(tmp_path), tmp_path / "data"
    (data / "venv").mkdir(parents=True)
    (data / "venv" / bootstrap.LOCK_NAME).write_text("other", encoding="utf-8")   # fresh: held
    slept = []
    start = time.time()
    runner, lines = FakeRunner(), []
    code = bootstrap.ensure(str(root), str(data), runner=runner, python="/usr/bin/python3",
                            version_info=(3, 13, 1), log=lines.append,
                            sleep=slept.append, clock=lambda: start + sum(slept))
    assert code == bootstrap.EXIT_ERROR and runner.calls == []
    assert slept and sum(slept) >= bootstrap.LOCK_WAIT_SECONDS
    assert "BOOTSTRAP=fail" in lines and any("another /hmd:init" in line for line in lines)
    assert (data / "venv" / bootstrap.LOCK_NAME).is_file()             # not ours: left alone


def test_stale_lock_is_taken_over_and_released(tmp_path):
    root, data = _plugin_root(tmp_path), tmp_path / "data"
    lock = data / "venv" / bootstrap.LOCK_NAME
    lock.parent.mkdir(parents=True)
    lock.write_text("dead", encoding="utf-8")
    old = os.path.getmtime(lock) - bootstrap.LOCK_STALE_SECONDS - 5
    os.utime(lock, (old, old))
    code, lines = _ensure(root, data, FakeRunner())
    assert code == 0 and "BOOTSTRAP=installed" in lines
    assert not lock.exists()


def _aged_venv(data, digest, last_used):
    venv = data / "venv" / digest
    venv.mkdir(parents=True)
    (venv / bootstrap.MARKER_NAME).write_text(json.dumps({"last_used": last_used.isoformat()}),
                                              encoding="utf-8")
    return venv


def test_prune_keeps_recent_and_current(tmp_path):
    import datetime as dt
    now = dt.datetime(2026, 10, 26, 12, tzinfo=dt.timezone.utc)
    data = tmp_path / "data"
    current = _aged_venv(data, "aaaaaaaaaaaa", now - dt.timedelta(days=30))
    recent = _aged_venv(data, "bbbbbbbbbbbb", now - dt.timedelta(hours=2))
    removed = bootstrap.prune_old_venvs(str(data), keep="aaaaaaaaaaaa", now=now)
    assert removed == [] and current.is_dir() and recent.is_dir()


def test_prune_removes_idle_other_digest(tmp_path):
    import datetime as dt
    now = dt.datetime(2026, 10, 26, 12, tzinfo=dt.timezone.utc)
    data = tmp_path / "data"
    _aged_venv(data, "aaaaaaaaaaaa", now)
    idle = _aged_venv(data, "cccccccccccc", now - dt.timedelta(days=2))
    (data / "venv" / bootstrap.LOCK_NAME).write_text("x", encoding="utf-8")
    removed = bootstrap.prune_old_venvs(str(data), keep="aaaaaaaaaaaa", now=now)
    assert removed == [str(idle)] and not idle.exists()
    assert (data / "venv" / bootstrap.LOCK_NAME).is_file()             # only digest dirs are pruned


def test_touch_last_used_updates_the_marker(tmp_path):
    import datetime as dt
    venv = _aged_venv(tmp_path, "dddddddddddd", dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
    now = dt.datetime(2026, 10, 26, tzinfo=dt.timezone.utc)
    bootstrap.touch_last_used(str(venv), now=now)
    assert bootstrap.read_marker(str(venv))["last_used"] == now.isoformat()
    bootstrap.touch_last_used(str(tmp_path / "no-venv"), now=now)        # no marker: no error


def test_data_dir_of_a_digest_venv(tmp_path):
    data = tmp_path / "data"
    assert bootstrap.data_dir_of(str(data / "venv" / "0123456789ab")) == str(data)
    assert bootstrap.data_dir_of(str(data / "venv")) == str(data)       # pre-0.1 flat layout


def test_pre_01_flat_venv_is_replaced_by_a_digest_venv(tmp_path):
    root, data = _plugin_root(tmp_path), tmp_path / "data"
    flat = data / "venv"
    (flat / "Lib").mkdir(parents=True)
    (flat / bootstrap.MARKER_NAME).write_text('{"plugin_version": "0.0.9"}', encoding="utf-8")
    code, lines = _ensure(root, data, FakeRunner())
    assert code == 0 and "BOOTSTRAP=installed" in lines
    assert sorted(p.name for p in flat.iterdir()) == [bootstrap.env_digest(str(root))]


def test_launcher_pointing_at_another_venv_is_repointed(tmp_path):
    root, data = _plugin_root(tmp_path), tmp_path / "data"
    _ensure(root, data, FakeRunner())
    (data / bootstrap.LAUNCHER_NAME).write_text("#!/bin/sh\nexec elsewhere\n", encoding="utf-8")
    runner = FakeRunner()
    code, lines = _ensure(root, data, runner)
    assert code == 0 and runner.calls == [] and "BOOTSTRAP=ok" in lines
    assert (data / bootstrap.LAUNCHER_NAME).read_text(encoding="utf-8") == \
        bootstrap.launcher_text(str(_digest_venv(root, data)))


def test_requirements_install_with_the_constraints_when_present(tmp_path):
    root, data = _plugin_root(tmp_path), tmp_path / "data"
    (root / "constraints.txt").write_text("pydantic==2.13.5\n", encoding="utf-8")
    runner = FakeRunner()
    _ensure(root, data, runner)
    assert runner.calls[1][-4:] == ["-r", str(root / "requirements.txt"), "-c", str(root / "constraints.txt")]


def test_marker_write_does_not_share_a_temp_name_with_another_process(tmp_path):
    # Two hmd starts touch the marker at once; a shared temp name lets one clobber the other's
    # half-written file (a corrupt marker drops plugin mode). Simulate the other writer by
    # occupying the old shared name.
    import datetime as dt
    venv = _aged_venv(tmp_path, "eeeeeeeeeeee", dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
    (venv / (bootstrap.MARKER_NAME + ".tmp")).mkdir()
    now = dt.datetime(2026, 10, 26, tzinfo=dt.timezone.utc)
    bootstrap.touch_last_used(str(venv), now=now)
    assert bootstrap.read_marker(str(venv))["last_used"] == now.isoformat()
    assert not [p for p in venv.iterdir() if p.name.endswith(".tmp") and p.is_file()]
