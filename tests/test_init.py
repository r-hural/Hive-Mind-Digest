import os
import pathlib
from types import SimpleNamespace

import pytest
import yaml

from pipeline.config import Config, ConfigError, load_config
from pipeline.init_cmd import ENV_EXAMPLE, cmd_init, render_config

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _flatten_keys(d, prefix=""):
    """Dotted-path key set of a nested dict; lists are leaves (not expanded), matching the shape
    of both yaml.safe_load's output and Config().model_dump()."""
    keys = set()
    if isinstance(d, dict):
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else str(k)
            keys.add(full)
            keys |= _flatten_keys(v, full)
    return keys


def _args(**kw):
    base = dict(config="config/config.yaml", provider="anthropic_api", budget=5.0, home="data", languages="en",
                analyze_model="claude-haiku-5-5", merge_model="claude-sonnet-5-5", force=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_rendered_config_loads_and_reflects_choices(tmp_path):
    text = render_config(_args(provider="claude_cli", budget=2.5, languages="en, de", home="mydata"))
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.llm.provider == "claude_cli" and cfg.run.budget_usd == 2.5
    assert cfg.transcription.languages == ["en", "de"] and cfg.home == "mydata"


def test_init_writes_config_and_env_template_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("config")
    assert cmd_init(_args()) == 0
    assert os.path.exists("config/config.yaml") and os.path.exists(".env.example")
    assert not os.path.exists(".env")
    open("config/config.yaml", "a", encoding="utf-8").write("# edited\n")
    assert cmd_init(_args()) == 4                      # refuses without --force
    assert "# edited" in open("config/config.yaml", encoding="utf-8").read()
    assert cmd_init(_args(force=True)) == 0
    assert "# edited" not in open("config/config.yaml", encoding="utf-8").read()


def test_windows_home_with_backslash_and_space_round_trips(tmp_path):
    home = r"C:\Hive Mind\data"
    text = render_config(_args(home=home))
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.home == home


def test_rendered_config_matches_defaults_field_for_field(tmp_path):
    text = render_config(_args())
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(str(p))
    default = Config()
    dumped = cfg.model_dump()
    default_dumped = default.model_dump()
    # only the values the default args explicitly set may differ from Config()'s own defaults,
    # and for the default args they are the same values anyway.
    assert dumped == default_dumped


def test_render_config_rejects_invalid_choices():
    with pytest.raises(ConfigError):
        render_config(_args(provider="nope"))
    with pytest.raises(ConfigError):
        render_config(_args(budget=-1))
    with pytest.raises(ConfigError):
        render_config(_args(budget="not-a-number"))


def test_cmd_init_maps_invalid_choice_to_exit_4(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = cmd_init(_args(provider="nope"))
    out = capsys.readouterr()
    assert code == 4
    assert out.err.startswith("hmd: ")
    assert "Traceback" not in out.err


def test_cmd_init_creates_parent_directory_of_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cmd_init(_args(config="cfg/sub/c.yaml")) == 0
    assert os.path.exists("cfg/sub/c.yaml")


# --- code-quality review follow-ups ------------------------------------------


def test_cmd_init_maps_oserror_to_exit_4(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # a regular file where a directory component is expected: os.makedirs(parent) raises OSError
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    cfg_path = str(blocker / "config.yaml")
    code = cmd_init(_args(config=cfg_path))
    out = capsys.readouterr()
    assert code == 4
    assert out.err.startswith("hmd: ")
    assert cfg_path in out.err
    assert "Traceback" not in out.err


def test_non_bmp_characters_round_trip(tmp_path):
    home = "data \U0001F600"
    text = render_config(_args(home=home))
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.home == home


def test_rendered_config_matches_shipped_config_semantically():
    rendered = yaml.safe_load(render_config(_args()))
    shipped = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert rendered == shipped


def test_rendered_config_key_set_matches_config_model_dump():
    # Catches a key silently dropped from render_config that the plain equality test above
    # cannot: a missing key that happens to coincide with a pydantic default would still compare
    # equal dict-to-dict.
    rendered = yaml.safe_load(render_config(_args()))
    default_dump = Config().model_dump()
    assert _flatten_keys(rendered) == _flatten_keys(default_dump)


def test_tracked_env_example_matches_env_example_constant():
    tracked = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert tracked == ENV_EXAMPLE


# --- additional tests: residual fixes ----------------------------------------

def test_cmd_init_uses_shared_exit_code_constants():
    # Regression guard: init_cmd must import its exit codes from
    # pipeline.exit_codes rather than hardcoding literal 4/0 -- if it ever reverts to literals
    # these module attributes disappear and this test fails with an AttributeError.
    from pipeline import exit_codes, init_cmd
    assert init_cmd.EXIT_USAGE == exit_codes.EXIT_USAGE == 4
    assert init_cmd.EXIT_OK == exit_codes.EXIT_OK == 0


def test_env_example_path_for_config_subdirectory_layout(tmp_path):
    from pipeline.init_cmd import env_example_path_for
    config_path = tmp_path / "root" / "config" / "config.yaml"
    assert env_example_path_for(str(config_path)) == str(tmp_path / "root" / ".env.example")


def test_env_example_path_for_config_without_config_parent(tmp_path):
    from pipeline.init_cmd import env_example_path_for
    config_path = tmp_path / "somewhere" / "c.yaml"
    assert env_example_path_for(str(config_path)) == str(tmp_path / "somewhere" / ".env.example")


def test_env_example_path_for_config_parent_case_insensitive(tmp_path):
    from pipeline.init_cmd import env_example_path_for
    config_path = tmp_path / "root" / "Config" / "config.yaml"
    assert env_example_path_for(str(config_path)) == str(tmp_path / "root" / ".env.example")


def test_init_writes_output_html_true(tmp_path):
    text = render_config(_args())
    assert "\n  html: true" in text.split("output:", 1)[1]
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    assert load_config(str(p)).output.html is True
