"""pipeline.envfile: .env comes from this checkout's project root only, never an ancestor."""
import os
from pathlib import Path

import pytest

from pipeline import envfile


def _nested_project(tmp_path, *, project_key=None):
    parent = tmp_path / "parent"
    project = parent / ".claude" / "worktrees" / "wt"
    project.mkdir(parents=True)
    (parent / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (parent / ".env").write_text("HMD_ENVFILE_TEST=parent\n", encoding="utf-8")
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    if project_key is not None:
        (project / ".env").write_text(f"HMD_ENVFILE_TEST={project_key}\n", encoding="utf-8")
    return project


@pytest.fixture
def no_test_var(monkeypatch):
    monkeypatch.setattr(envfile, "DOTENV_KEYS", ("HMD_ENVFILE_TEST",))
    # setenv then delenv so monkeypatch restores the "absent" state even after load_dotenv writes it.
    monkeypatch.setenv("HMD_ENVFILE_TEST", "placeholder")
    monkeypatch.delenv("HMD_ENVFILE_TEST")


def test_parent_checkout_dotenv_is_never_found(monkeypatch, tmp_path, no_test_var):
    project = _nested_project(tmp_path)
    monkeypatch.setattr(envfile, "PROJECT_ROOT", project)
    monkeypatch.chdir(project)
    assert envfile.project_dotenv_path() == ""
    assert envfile.load_project_dotenv() == ""
    assert "HMD_ENVFILE_TEST" not in os.environ


def test_project_root_dotenv_is_found_and_loaded(monkeypatch, tmp_path, no_test_var):
    project = _nested_project(tmp_path, project_key="project")
    monkeypatch.setattr(envfile, "PROJECT_ROOT", project)
    assert envfile.project_dotenv_path() == str(project / ".env")
    assert envfile.load_project_dotenv() == str(project / ".env")
    assert os.environ["HMD_ENVFILE_TEST"] == "project"


def test_load_never_overrides_existing_environment(monkeypatch, tmp_path):
    project = _nested_project(tmp_path, project_key="project")
    monkeypatch.setattr(envfile, "PROJECT_ROOT", project)
    monkeypatch.setenv("HMD_ENVFILE_TEST", "shell")
    envfile.load_project_dotenv()
    assert os.environ["HMD_ENVFILE_TEST"] == "shell"


def test_no_pyproject_means_no_dotenv(monkeypatch, tmp_path):
    root = tmp_path / "site"
    root.mkdir()
    (root / ".env").write_text("HMD_ENVFILE_TEST=stray\n", encoding="utf-8")
    monkeypatch.setattr(envfile, "PROJECT_ROOT", root)
    assert envfile.project_dotenv_path() == ""


def test_project_root_is_this_checkout():
    assert Path(envfile.PROJECT_ROOT).is_absolute()
    assert (envfile.PROJECT_ROOT / "pyproject.toml").is_file()
    assert (envfile.PROJECT_ROOT / "pipeline" / "envfile.py").samefile(envfile.__file__)


def test_envfile_has_no_llm_imports():
    import inspect
    src = inspect.getsource(envfile)
    assert "import anthropic" not in src and "pipeline.llm" not in src


def _dotenv_dir(tmp_path, text):
    d = tmp_path / "cfgdir"
    d.mkdir(exist_ok=True)
    (d / ".env").write_text(text, encoding="utf-8")
    return d


def test_dotenv_sets_only_the_api_keys(monkeypatch, tmp_path):
    d = _dotenv_dir(tmp_path, "ANTHROPIC_BASE_URL=https://attacker.example\nANTHROPIC_API_KEY=k\n"
                              "HTTPS_PROXY=http://evil\nSUPADATA_API_KEY=s\n")
    for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "HTTPS_PROXY", "SUPADATA_API_KEY"):
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)
    monkeypatch.setattr(envfile, "PROJECT_ROOT", tmp_path / "no-checkout")
    envfile.set_config_path(str(d / "config.yaml"))
    envfile.load_project_dotenv()
    assert os.environ["ANTHROPIC_API_KEY"] == "k" and os.environ["SUPADATA_API_KEY"] == "s"
    assert "ANTHROPIC_BASE_URL" not in os.environ and "HTTPS_PROXY" not in os.environ


def test_implicit_default_config_does_not_load_its_directory_dotenv(monkeypatch, tmp_path):
    d = _dotenv_dir(tmp_path, "ANTHROPIC_API_KEY=k\n")
    monkeypatch.setattr(envfile, "PROJECT_ROOT", tmp_path / "no-checkout")
    envfile.set_config_path(str(d / "config.yaml"), explicit=False)
    assert envfile.config_dotenv_path() == "" and envfile.dotenv_paths() == []
    envfile.set_config_path(str(d / "config.yaml"), explicit=True)
    assert envfile.dotenv_paths() == [str(d / ".env")]


def test_cli_passes_whether_the_config_is_explicit(monkeypatch, tmp_path):
    from pipeline import cli
    seen = []
    monkeypatch.setattr(envfile, "set_config_path", lambda p, explicit=True: seen.append(explicit))
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"home: {tmp_path.as_posix()}/h\n", encoding="utf-8")
    cli.main(["--config", str(cfg), "notebook", "list"])
    monkeypatch.chdir(tmp_path)
    cli.main(["notebook", "list"])
    assert seen[0] is True and seen[1] is False
