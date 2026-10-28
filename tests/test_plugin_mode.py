"""IDEA-310: how hmd finds its config, `.env` and home when it runs from the plugin venv /hmd:init
created (pipeline/bootstrap.py), from any working directory; and that a development checkout
behaves as before. A plugin venv is simulated by a temp dir holding the install marker, with
sys.prefix pointed at it."""
from __future__ import annotations

import json
import os
import sys

import pytest

from pipeline import bootstrap, config, envfile
from pipeline.config import CONFIG_PATH, Config, resolve_config_path


def _plugin_data(tmp_path, version="0.1.0", root=None):
    """A fake ${CLAUDE_PLUGIN_DATA} with venv/hmd-plugin.json; returns (data dir, venv dir)."""
    data = tmp_path / "plugin-data"
    venv = data / "venv"
    venv.mkdir(parents=True)
    want = bootstrap.desired_state(str(root)) if root else {"plugin_version": version,
                                                            "requirements_sha256": "x"}
    (venv / bootstrap.MARKER_NAME).write_text(json.dumps(want), encoding="utf-8")
    return data, venv


def _plugin_root(tmp_path, version="0.1.0"):
    root = tmp_path / "plugin-root"
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps({"version": version}), encoding="utf-8")
    (root / "pyproject.toml").write_text('[project]\nversion = "0.1.0.dev0"\n', encoding="utf-8")
    (root / "requirements.txt").write_text("pydantic==2.13.5\n", encoding="utf-8")
    return root


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    """Plugin mode: sys.prefix is a venv with the marker; the cwd is an unrelated folder."""
    root = _plugin_root(tmp_path)
    data, venv = _plugin_data(tmp_path, root=root)
    monkeypatch.setattr(sys, "prefix", str(venv))
    elsewhere = tmp_path / "some-project"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return root, data, venv


def _main(argv):
    from pipeline.cli import main
    return main(argv)


def _write_config(path, home):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"home: {json.dumps(str(home))}\n", encoding="utf-8")


# --- resolve_config_path: precedence -------------------------------------------------------------

def test_development_default_is_the_cwd_relative_config(tmp_path):
    assert resolve_config_path(None, environ={}, prefix=str(tmp_path)) == (CONFIG_PATH, "default")
    assert config.run_mode(prefix=str(tmp_path)) == "development"     # this checkout has pyproject


def test_hmd_config_env_beats_the_default(tmp_path):
    env = {"HMD_CONFIG": str(tmp_path / "c.yaml")}
    assert resolve_config_path(None, environ=env, prefix=str(tmp_path)) == (str(tmp_path / "c.yaml"), "HMD_CONFIG")
    assert resolve_config_path(None, environ={"HMD_CONFIG": "  "}, prefix=str(tmp_path))[1] == "default"


def test_plugin_venv_defaults_to_the_config_in_the_plugin_data_dir(tmp_path):
    data, venv = _plugin_data(tmp_path)
    assert config.plugin_data_dir(str(venv)) == str(data)
    assert config.run_mode(prefix=str(venv)) == "plugin"
    assert resolve_config_path(None, environ={}, prefix=str(venv)) == (str(data / "config.yaml"), "plugin")


def test_precedence_is_flag_then_env_then_plugin_then_default(tmp_path):
    _, venv = _plugin_data(tmp_path)
    env = {"HMD_CONFIG": "from-env.yaml"}
    assert resolve_config_path("flag.yaml", environ=env, prefix=str(venv)) == ("flag.yaml", "--config")
    assert resolve_config_path(None, environ=env, prefix=str(venv)) == ("from-env.yaml", "HMD_CONFIG")
    assert resolve_config_path(None, environ={}, prefix=str(venv))[1] == "plugin"


def test_home_expands_the_user_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert Config(home="~/HiveMindDigest").home == os.path.join(str(tmp_path), "HiveMindDigest")
    assert Config(home="data").home == "data"                          # relative stays relative


# --- the CLI in plugin mode, from an unrelated cwd -----------------------------------------------

def test_plugin_mode_uses_the_data_dir_config_from_any_cwd(plugin, tmp_path, capsys):
    _, data, _ = plugin
    home = tmp_path / "HiveMindDigest"
    _write_config(data / "config.yaml", home)
    assert _main(["notebook", "list"]) == 0
    assert "No notebooks yet" in capsys.readouterr().out
    assert (home / "domains" / "general" / "state.sqlite").is_file()
    assert not (tmp_path / "some-project" / "data").exists()          # nothing in the cwd


def test_plugin_mode_without_a_config_says_run_init(plugin, capsys):
    assert _main(["notebook", "list"]) == 4
    err = capsys.readouterr().err
    assert "config file not found" in err and "/hmd:init" in err


def test_missing_hmd_config_file_is_an_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HMD_CONFIG", str(tmp_path / "missing.yaml"))
    assert _main(["notebook", "list"]) == 4
    assert "(from HMD_CONFIG)" in capsys.readouterr().err


def test_hmd_config_is_used(monkeypatch, tmp_path, capsys):
    cfg = tmp_path / "elsewhere" / "config.yaml"
    _write_config(cfg, tmp_path / "home")
    monkeypatch.setenv("HMD_CONFIG", str(cfg))
    monkeypatch.chdir(tmp_path)
    assert _main(["notebook", "list"]) == 0
    assert (tmp_path / "home" / "domains" / "general" / "state.sqlite").is_file()


def test_stale_plugin_venv_refuses_to_run(plugin, tmp_path, monkeypatch, capsys):
    root, data, _ = plugin
    _write_config(data / "config.yaml", tmp_path / "home")
    monkeypatch.setenv("HMD_PLUGIN_ROOT", str(root))
    assert _main(["notebook", "list"]) == 0                             # marker matches the root
    (root / ".claude-plugin" / "plugin.json").write_text('{"version": "0.2.0"}', encoding="utf-8")
    assert _main(["notebook", "list"]) == 4
    err = capsys.readouterr().err
    assert "out of date (plugin version 0.1.0 -> 0.2.0)" in err and "/hmd:init" in err


def test_staleness_is_ignored_outside_plugin_mode(tmp_path):
    root = _plugin_root(tmp_path, version="9.9.9")
    assert config.plugin_stale_reason({"HMD_PLUGIN_ROOT": str(root)}, prefix=str(tmp_path)) is None
    assert config.plugin_stale_reason({}, prefix=str(tmp_path)) is None


def test_staleness_ignores_a_root_that_is_not_a_plugin_root(tmp_path):
    _, venv = _plugin_data(tmp_path)
    assert config.plugin_stale_reason({"HMD_PLUGIN_ROOT": str(tmp_path / "nope")}, prefix=str(venv)) is None


# --- hmd init in plugin mode ---------------------------------------------------------------------

def test_init_in_plugin_mode_writes_the_data_dir_config_and_an_absolute_home(plugin, monkeypatch,
                                                                             tmp_path, capsys):
    _, data, _ = plugin
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "user"))
    assert _main(["init", "--provider", "claude_cli"]) == 0
    out = capsys.readouterr().out
    assert f"CONFIG={data / 'config.yaml'}" in out
    assert f"HOME={tmp_path / 'user' / 'HiveMindDigest'}" in out
    assert f"ENV={data / '.env.example'}" in out
    cfg = config.load_config(str(data / "config.yaml"))
    assert cfg.home == str(tmp_path / "user" / "HiveMindDigest")
    assert not (tmp_path / "some-project" / "config").exists()


def test_init_in_plugin_mode_stores_a_relative_home_as_absolute(plugin, tmp_path, capsys):
    _, data, _ = plugin
    assert _main(["init", "--home", "reports"]) == 0
    assert config.load_config(str(data / "config.yaml")).home == str(tmp_path / "some-project" / "reports")


def test_init_in_development_keeps_the_relative_default(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    assert _main(["init"]) == 0
    assert config.load_config(str(tmp_path / "config" / "config.yaml")).home == "data"
    assert f"HOME={tmp_path / 'data'}" in capsys.readouterr().out


# --- .env next to the config ---------------------------------------------------------------------

@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.setattr(envfile, "DOTENV_KEYS", ("HMD_ENVFILE_TEST",))
    # setenv then delenv so monkeypatch restores the "absent" state even after load_dotenv writes it.
    monkeypatch.setenv("HMD_ENVFILE_TEST", "placeholder")
    monkeypatch.delenv("HMD_ENVFILE_TEST")


def test_dotenv_next_to_the_config_is_loaded(monkeypatch, tmp_path, no_key):
    monkeypatch.setattr(envfile, "PROJECT_ROOT", tmp_path / "no-checkout")
    (tmp_path / ".env").write_text("HMD_ENVFILE_TEST=plugin\n", encoding="utf-8")
    envfile.set_config_path(str(tmp_path / "config.yaml"))
    assert envfile.dotenv_paths() == [str(tmp_path / ".env")]
    assert envfile.load_project_dotenv() == str(tmp_path / ".env")
    assert os.environ["HMD_ENVFILE_TEST"] == "plugin"


def test_config_dotenv_wins_over_the_checkout_dotenv_and_the_shell_wins_over_both(monkeypatch, tmp_path,
                                                                                    no_key):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("", encoding="utf-8")
    (checkout / ".env").write_text("HMD_ENVFILE_TEST=checkout\n", encoding="utf-8")
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / ".env").write_text("HMD_ENVFILE_TEST=config\n", encoding="utf-8")
    monkeypatch.setattr(envfile, "PROJECT_ROOT", checkout)
    envfile.set_config_path(str(cfg_dir / "config.yaml"))
    assert envfile.dotenv_paths() == [str(cfg_dir / ".env"), str(checkout / ".env")]
    envfile.load_project_dotenv()
    assert os.environ["HMD_ENVFILE_TEST"] == "config"
    monkeypatch.setenv("HMD_ENVFILE_TEST", "shell")
    envfile.load_project_dotenv()
    assert os.environ["HMD_ENVFILE_TEST"] == "shell"


def test_the_same_dotenv_is_listed_once(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text("X=1\n", encoding="utf-8")
    monkeypatch.setattr(envfile, "PROJECT_ROOT", tmp_path)
    envfile.set_config_path(str(tmp_path / "config.yaml"))
    assert envfile.dotenv_paths() == [str(tmp_path / ".env")]


def test_cli_records_the_config_for_the_dotenv_lookup(plugin, tmp_path, capsys):
    _, data, _ = plugin
    _write_config(data / "config.yaml", tmp_path / "home")
    _main(["notebook", "list"])
    assert envfile._config_path == str(data / "config.yaml")


def test_doctor_credits_the_plugin_dotenv_and_never_prints_the_key(plugin, monkeypatch, tmp_path, capsys):
    _, data, _ = plugin
    _write_config(data / "config.yaml", tmp_path / "home")
    (data / ".env").write_text("ANTHROPIC_API_KEY=sk-very-secret-test\n", encoding="utf-8")
    monkeypatch.setattr(envfile, "PROJECT_ROOT", tmp_path / "no-checkout")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setattr("pipeline.doctor._model_section", lambda cfg, args: [])
    monkeypatch.setattr("pipeline.doctor._check_claude_binary",
                        lambda provider: __import__("pipeline.doctor", fromlist=["Check"]).Check(
                            "claude_binary", False, False, "stubbed"))
    _main(["doctor", "--no-model-check"])
    out = capsys.readouterr().out
    assert f"set (from {data / '.env'})" in out
    assert "sk-very-secret-test" not in out


# --- doctor: mode, config path, home -------------------------------------------------------------

def _location(cfg, **ctx):
    from pipeline.doctor import _check_location
    return {c.name: c for c in _check_location(cfg, ctx)}


def test_doctor_reports_development_mode_config_and_home(tmp_path):
    c = _location(Config(home=str(tmp_path / "h")), mode="development", config_path=CONFIG_PATH,
                  config_source="default", config_missing=False)
    assert c["mode"].ok and c["mode"].detail.startswith("development checkout (")
    assert c["home"].ok and c["home"].detail == str(tmp_path / "h")
    assert c["config_file"].ok


def test_doctor_reports_plugin_mode_and_fails_a_missing_config(plugin, tmp_path):
    _, data, venv = plugin
    c = _location(Config(home=str(tmp_path / "h")), mode="plugin", config_path=str(data / "config.yaml"),
                  config_source="plugin", config_missing=True)
    assert c["mode"].detail == f"plugin venv ({venv})"
    assert not c["config_file"].ok and c["config_file"].required
    assert c["home"].ok and c["home"].detail.startswith("not set yet")
    assert "plugin data dir" in c["config_file"].detail and "/hmd:init" in c["config_file"].detail


def test_doctor_warns_when_home_is_inside_the_plugin_data_dir(plugin):
    _, data, _ = plugin
    c = _location(Config(home=str(data / "home")), mode="plugin", config_path=str(data / "config.yaml"),
                  config_source="plugin", config_missing=False)
    assert not c["home"].ok and not c["home"].required and "uninstall" in c["home"].detail


def test_doctor_fails_a_stale_plugin_venv(plugin, tmp_path):
    c = _location(Config(home=str(tmp_path)), mode="plugin", stale="requirements.txt changed")
    assert not c["plugin_env"].ok and c["plugin_env"].required and "/hmd:init" in c["plugin_env"].detail


def test_doctor_runs_without_a_plugin_config_and_says_run_init(plugin, monkeypatch, capsys):
    monkeypatch.setattr("pipeline.doctor.run_checks", lambda cfg: [])
    monkeypatch.setattr("pipeline.doctor._model_section", lambda cfg, args: [])
    assert _main(["doctor"]) == 2
    out = capsys.readouterr().out
    assert "[FAIL] config_file" in out and "[ok]   mode" in out and "DOCTOR=fail" in out


# --- one venv per requirements digest (pipeline/bootstrap.py) ----------------------------------

def test_a_digest_venv_belongs_to_its_plugin_data_dir(tmp_path):
    data = tmp_path / "plugin-data"
    venv = data / "venv" / "0123456789ab"
    venv.mkdir(parents=True)
    (venv / bootstrap.MARKER_NAME).write_text('{"plugin_version": "0.1.0"}', encoding="utf-8")
    assert config.plugin_data_dir(str(venv)) == str(data)
    assert resolve_config_path(None, environ={}, prefix=str(venv)) == (str(data / "config.yaml"), "plugin")


def test_hmd_in_plugin_mode_records_last_used(plugin, tmp_path):
    _, data, venv = plugin
    _write_config(data / "config.yaml", tmp_path / "home")
    assert "last_used" not in bootstrap.read_marker(str(venv))
    assert _main(["notebook", "list"]) == 0
    assert bootstrap.read_marker(str(venv))["last_used"]


def _install_digest_venv(data, root):
    """A complete venv for `root` under data/venv/<digest>, as /hmd:init builds it."""
    venv = data / "venv" / bootstrap.env_digest(str(root))
    hmd = __import__("pathlib").Path(bootstrap.venv_hmd(str(venv)))
    hmd.parent.mkdir(parents=True)
    hmd.write_text("exe", encoding="utf-8")
    (venv / bootstrap.MARKER_NAME).write_text(json.dumps(bootstrap.desired_state(str(root))), encoding="utf-8")
    return hmd


def test_stale_hmd_hands_off_to_the_venv_of_its_plugin_root(plugin, tmp_path, monkeypatch):
    # Two sessions on different plugin versions share the launcher <data>/hmd: after /hmd:init in
    # the newer one, the older session's call lands in the newer venv. It must run the venv built
    # for its own root (kept beside) instead of refusing.
    root, data, _ = plugin
    _write_config(data / "config.yaml", tmp_path / "home")
    other_root = _plugin_root(tmp_path / "other", version="0.2.0")
    target = _install_digest_venv(data, other_root)
    monkeypatch.setenv("HMD_PLUGIN_ROOT", str(other_root))
    calls = []
    monkeypatch.setattr("pipeline.cli._run_delegate", lambda cmd: calls.append(cmd) or 7)
    assert _main(["notebook", "list"]) == 7
    assert calls == [[str(target), "notebook", "list"]]


def test_stale_hmd_without_a_matching_venv_still_refuses(plugin, tmp_path, monkeypatch, capsys):
    root, data, _ = plugin
    _write_config(data / "config.yaml", tmp_path / "home")
    other_root = _plugin_root(tmp_path / "other", version="0.2.0")
    monkeypatch.setenv("HMD_PLUGIN_ROOT", str(other_root))
    monkeypatch.setattr("pipeline.cli._run_delegate", lambda cmd: pytest.fail("must not delegate"))
    assert _main(["notebook", "list"]) == 4
    assert "/hmd:init" in capsys.readouterr().err
