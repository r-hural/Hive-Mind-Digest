import subprocess

import dotenv
import pytest

from pipeline.config import Config
from pipeline.doctor import Check, format_report, gpu_report, prewarm_whisper, run_checks


@pytest.fixture(autouse=True)
def _no_real_dotenv(monkeypatch):
    """Never let the developer's real .env leak into os.environ during this suite. Tests that
    need ANTHROPIC_API_KEY set do so explicitly with monkeypatch.setenv; tests about the
    dotenv-source detail override project_dotenv_path again themselves."""
    monkeypatch.setattr("pipeline.doctor.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("pipeline.doctor.project_dotenv_path", lambda *a, **k: "")


def _by_name(checks):
    return {c.name: c for c in checks}


def test_api_key_required_only_for_anthropic_api(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = _by_name(run_checks(Config(home=str(tmp_path))))
    assert c["anthropic_api_key"].ok is False and c["anthropic_api_key"].required is True
    c2 = _by_name(run_checks(Config(home=str(tmp_path), llm={"provider": "claude_cli"})))
    assert c2["anthropic_api_key"].required is False


def test_anthropic_api_key_detail_names_dotenv_source(monkeypatch, tmp_path):
    # The key must come from load_dotenv() actually reading the file, not from a pre-set
    # environment variable -- run_checks snapshots os.environ before calling load_dotenv, so a
    # key set beforehand (by the test or the real shell) would always be credited to
    # "environment" instead (see test_anthropic_api_key_detail_prefers_pre_existing_environment).
    # Un-stub load_dotenv back to the real function for this one test so that side effect happens.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("ANTHROPIC_API_KEY=sk-test\n", encoding="utf-8")
    monkeypatch.setattr("pipeline.doctor.project_dotenv_path", lambda *a, **k: str(env_path))
    monkeypatch.setattr("pipeline.doctor.load_dotenv", dotenv.load_dotenv)
    # The real load_dotenv() writes ANTHROPIC_API_KEY=sk-test straight into os.environ. A plain
    # monkeypatch.delenv above records no undo entry for a variable that was absent, so nothing
    # would restore it after this test -- setenv first (to any placeholder) so monkeypatch snapshots
    # the "absent" state, then delenv it right back off before the real load_dotenv() call: teardown
    # then correctly deletes whatever load_dotenv() wrote, in both directions.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    c = _by_name(run_checks(Config(home=str(tmp_path))))
    assert c["anthropic_api_key"].ok is True
    assert str(env_path) in c["anthropic_api_key"].detail


def test_anthropic_api_key_detail_falls_back_to_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr("pipeline.doctor.project_dotenv_path", lambda *a, **k: "")
    c = _by_name(run_checks(Config(home=str(tmp_path))))
    assert c["anthropic_api_key"].ok is True
    assert "environment" in c["anthropic_api_key"].detail


def test_anthropic_api_key_detail_prefers_pre_existing_environment(monkeypatch, tmp_path):
    # The key is already in os.environ (e.g. set by the real shell) before run_checks ever calls
    # load_dotenv, which never overrides an existing environment value -- the detail must credit
    # the environment, never a .env file, even when that file also happens to name the same key.
    env_path = tmp_path / ".env"
    env_path.write_text("ANTHROPIC_API_KEY=sk-test-from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-shell")
    monkeypatch.setattr("pipeline.doctor.project_dotenv_path", lambda *a, **k: str(env_path))
    c = _by_name(run_checks(Config(home=str(tmp_path))))
    assert c["anthropic_api_key"].ok is True
    assert "environment" in c["anthropic_api_key"].detail
    assert str(env_path) not in c["anthropic_api_key"].detail


def test_claude_binary_check(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.doctor.shutil.which", lambda name: None)
    c = _by_name(run_checks(Config(home=str(tmp_path), llm={"provider": "claude_cli"})))
    assert c["claude_binary"].ok is False and c["claude_binary"].required is True
    monkeypatch.setattr("pipeline.doctor.shutil.which", lambda name: "C:/x/claude.exe")
    monkeypatch.setattr("pipeline.doctor.subprocess.run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="2.1.263 (Claude Code)\n", stderr=""))
    c = _by_name(run_checks(Config(home=str(tmp_path), llm={"provider": "claude_cli"})))
    assert c["claude_binary"].ok and "2.1.263" in c["claude_binary"].detail


def test_claude_binary_shim_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.doctor.shutil.which", lambda name: "C:/x/claude.cmd")
    c = _by_name(run_checks(Config(home=str(tmp_path), llm={"provider": "claude_cli"})))
    assert c["claude_binary"].ok is False and c["claude_binary"].required is True
    assert "shim" in c["claude_binary"].detail.lower()


def test_ffmpeg_and_whisper_are_optional_unless_configured(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.doctor.shutil.which", lambda name: None)
    # default config: whisper_local is one of three fallback providers -> warn, not required
    c = _by_name(run_checks(Config(home=str(tmp_path))))
    assert c["ffmpeg"].required is False and c["ffmpeg"].ok is False
    # whisper_local is the sole configured provider -> hmd run cannot work at all without it
    c = _by_name(run_checks(Config(home=str(tmp_path), transcription={"providers": ["whisper_local"]})))
    assert c["ffmpeg"].required is True and c["ffmpeg"].ok is False


def test_ffmpeg_not_needed_detail_when_whisper_local_not_configured(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.doctor.shutil.which", lambda name: None)
    c = _by_name(run_checks(Config(home=str(tmp_path), transcription={"providers": ["youtube_captions"]})))
    assert c["ffmpeg"].required is False
    assert "not needed" in c["ffmpeg"].detail


def test_prices_known_for_configured_models(tmp_path):
    c = _by_name(run_checks(Config(home=str(tmp_path), llm={"models": {"analyze": "mystery-1", "merge": "claude-sonnet-4-6"}})))
    assert c["model_prices"].ok is False and "mystery-1" in c["model_prices"].detail


def test_state_dir_writable_and_legacy_detected(tmp_path):
    import sqlite3, os
    home = tmp_path / "home"; os.makedirs(home / "domains" / "general")
    conn = sqlite3.connect(str(home / "domains" / "general" / "state.sqlite"))
    conn.execute("CREATE TABLE projects (id INTEGER)"); conn.commit(); conn.close()
    c = _by_name(run_checks(Config(home=str(home))))
    assert c["state_db"].ok is False and "MILESTONE-0" in c["state_db"].detail


def test_state_db_schema_version_mismatch_detected(tmp_path):
    import sqlite3
    from pipeline import db as db_mod
    home = tmp_path / "home"
    db_path = home / "domains" / "general" / "state.sqlite"
    db_mod.init_db(str(db_path))
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE meta SET value='0' WHERE key='schema_version'")
    conn.commit(); conn.close()
    c = _by_name(run_checks(Config(home=str(home))))
    assert c["state_db"].ok is False
    assert "0" in c["state_db"].detail


def test_state_db_pending_migration_is_reported_ok(tmp_path):
    import sqlite3
    from pipeline import db as db_mod
    home = tmp_path / "home"
    db_path = home / "domains" / "general" / "state.sqlite"
    db_mod.init_db(str(db_path))
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
    conn.commit(); conn.close()
    c = _by_name(run_checks(Config(home=str(home))))
    assert c["state_db"].ok is True
    assert c["state_db"].detail == "migrates 1 -> 2 on next run (backup kept)"
    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "1"
    conn.close()


def test_state_db_corrupted_file_reported_not_raised(tmp_path):
    import os
    home = tmp_path / "home"
    os.makedirs(home / "domains" / "general")
    db_path = home / "domains" / "general" / "state.sqlite"
    db_path.write_bytes(b"not a sqlite database, just junk bytes")
    checks = run_checks(Config(home=str(home)))
    c = _by_name(checks)
    assert c["state_db"].ok is False
    assert c["state_db"].required is True


def test_state_db_and_vault_dir_do_not_create_directories(tmp_path):
    import os
    home = tmp_path / "home"
    run_checks(Config(home=str(home)))
    assert not os.path.exists(home / "domains")


def test_gpu_report_without_torch(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name in ("torch", "whisper"):
            raise ImportError(name)
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    ok, detail = gpu_report()
    assert ok is False and "whisper" in detail.lower()


def test_format_report_and_exit(tmp_path):
    checks = [Check("a", True, True, "fine"), Check("b", False, False, "meh"), Check("c", False, True, "bad")]
    text, code = format_report(checks)
    assert "[ok]   a" in text and "[warn] b" in text and "[FAIL] c" in text and code == 2
    assert format_report(checks[:2])[1] == 0


# --- additional tests: corrections ------------------------------------------


def test_whisper_model_check_reports_ok_when_cached(monkeypatch, tmp_path):
    import types
    fake_whisper = types.ModuleType("whisper")
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)
    cache_dir = tmp_path / "whisper-cache"
    cache_dir.mkdir()
    (cache_dir / "base.pt").write_text("fake", encoding="utf-8")
    monkeypatch.setattr("pipeline.doctor._whisper_cache_dir", lambda: str(cache_dir))
    c = _by_name(run_checks(Config(home=str(tmp_path), transcription={"providers": ["whisper_local"]})))
    assert c["whisper_model"].ok is True and c["whisper_model"].required is False


def test_whisper_model_check_warns_when_missing(monkeypatch, tmp_path):
    import types
    fake_whisper = types.ModuleType("whisper")
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)
    cache_dir = tmp_path / "whisper-cache-empty"
    monkeypatch.setattr("pipeline.doctor._whisper_cache_dir", lambda: str(cache_dir))
    # whisper_local is a fallback among several providers here (default config) -> warn
    c = _by_name(run_checks(Config(home=str(tmp_path))))
    assert c["whisper_model"].ok is False and c["whisper_model"].required is False
    assert "hmd doctor --prewarm-whisper" in c["whisper_model"].detail


def test_whisper_model_check_required_when_sole_provider_and_missing(monkeypatch, tmp_path):
    import types
    fake_whisper = types.ModuleType("whisper")
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)
    cache_dir = tmp_path / "whisper-cache-empty"
    monkeypatch.setattr("pipeline.doctor._whisper_cache_dir", lambda: str(cache_dir))
    c = _by_name(run_checks(Config(home=str(tmp_path), transcription={"providers": ["whisper_local"]})))
    assert c["whisper_model"].ok is False and c["whisper_model"].required is True


def test_whisper_model_check_skipped_when_not_importable(monkeypatch, tmp_path):
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == "whisper":
            raise ImportError(name)
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    c = _by_name(run_checks(Config(home=str(tmp_path), transcription={"providers": ["whisper_local"]})))
    assert "whisper_model" not in c


def test_whisper_model_check_not_required_when_whisper_local_not_configured(monkeypatch, tmp_path):
    import types
    fake_whisper = types.ModuleType("whisper")
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake_whisper)
    cache_dir = tmp_path / "whisper-cache"
    monkeypatch.setattr("pipeline.doctor._whisper_cache_dir", lambda: str(cache_dir))
    c = _by_name(run_checks(Config(home=str(tmp_path), transcription={"providers": ["youtube_captions"]})))
    assert "whisper_model" in c
    assert c["whisper_model"].required is False


def test_whisper_model_filename_uses_real_mapping(monkeypatch, tmp_path):
    import sys, types
    fake_whisper = types.ModuleType("whisper")
    fake_whisper._MODELS = {"large": "https://example.com/large-v3.pt"}
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    cache_dir = tmp_path / "whisper-cache"
    monkeypatch.setattr("pipeline.doctor._whisper_cache_dir", lambda: str(cache_dir))
    c = _by_name(run_checks(Config(home=str(tmp_path), transcription={"whisper": {"model_size": "large"}})))
    assert "large-v3.pt" in c["whisper_model"].detail


def test_prewarm_whisper_calls_load_model(monkeypatch, tmp_path):
    import sys, types
    calls = []

    class FakeWhisper(types.ModuleType):
        def load_model(self, size, device=None, download_root=None):
            calls.append((size, device, download_root))
            return object()

    fake = FakeWhisper("whisper")
    monkeypatch.setitem(sys.modules, "whisper", fake)
    monkeypatch.setattr("pipeline.doctor._whisper_cache_dir", lambda: str(tmp_path / "cache"))
    cfg = Config(home=str(tmp_path), transcription={"whisper": {"model_size": "small"}})
    logged = []
    ok = prewarm_whisper(cfg, log=logged.append)
    assert ok is True
    assert calls == [("small", "cpu", str(tmp_path / "cache"))]
    assert any("small" in m for m in logged)


def test_prewarm_whisper_returns_false_when_not_importable(monkeypatch, tmp_path):
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == "whisper":
            raise ImportError(name)
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = Config(home=str(tmp_path))
    logged = []
    ok = prewarm_whisper(cfg, log=logged.append)
    assert ok is False
    assert any("install" in m.lower() or "whisper" in m.lower() for m in logged)


def test_prewarm_whisper_returns_false_when_load_model_raises(monkeypatch, tmp_path):
    import sys, types

    class FakeWhisper(types.ModuleType):
        def load_model(self, size, device=None, download_root=None):
            raise RuntimeError("network unreachable")

    fake = FakeWhisper("whisper")
    monkeypatch.setitem(sys.modules, "whisper", fake)
    monkeypatch.setattr("pipeline.doctor._whisper_cache_dir", lambda: str(tmp_path / "cache"))
    cfg = Config(home=str(tmp_path))
    logged = []
    ok = prewarm_whisper(cfg, log=logged.append)
    assert ok is False
    assert any("network unreachable" in m for m in logged)


def test_cmd_doctor_prints_report_and_status(monkeypatch, tmp_path, capsys):
    from pipeline.doctor import cmd_doctor

    class FakeArgs:
        prewarm_whisper = False

    monkeypatch.setattr("pipeline.doctor.run_checks", lambda cfg: [Check("a", True, True, "fine")])
    code = cmd_doctor(Config(home=str(tmp_path)), FakeArgs())
    out = capsys.readouterr().out
    assert code == 0
    assert "DOCTOR=ok" in out


def test_cmd_doctor_fail_exit_code(monkeypatch, tmp_path, capsys):
    from pipeline.doctor import cmd_doctor

    class FakeArgs:
        prewarm_whisper = False

    monkeypatch.setattr("pipeline.doctor.run_checks", lambda cfg: [Check("a", False, True, "bad")])
    code = cmd_doctor(Config(home=str(tmp_path)), FakeArgs())
    out = capsys.readouterr().out
    assert code == 2
    assert "DOCTOR=fail" in out


def test_cmd_doctor_prewarm_runs_before_checks(monkeypatch, tmp_path, capsys):
    from pipeline.doctor import cmd_doctor

    class FakeArgs:
        prewarm_whisper = True

    calls = []
    monkeypatch.setattr("pipeline.doctor.prewarm_whisper", lambda cfg, log=print: calls.append("prewarm") or True)
    monkeypatch.setattr("pipeline.doctor.run_checks", lambda cfg: calls.append("checks") or [Check("a", True, True, "fine")])
    code = cmd_doctor(Config(home=str(tmp_path)), FakeArgs())
    assert code == 0
    assert calls == ["prewarm", "checks"]


def test_cmd_doctor_logs_note_when_prewarm_fails(monkeypatch, tmp_path, capsys):
    from pipeline.doctor import cmd_doctor

    class FakeArgs:
        prewarm_whisper = True

    monkeypatch.setattr("pipeline.doctor.prewarm_whisper", lambda cfg, log=print: False)
    monkeypatch.setattr("pipeline.doctor.run_checks", lambda cfg: [Check("a", True, True, "fine")])
    code = cmd_doctor(Config(home=str(tmp_path)), FakeArgs())
    out = capsys.readouterr().out
    assert code == 0
    assert "prewarm failed" in out.lower()


def test_run_checks_never_makes_an_llm_call():
    import inspect
    import pipeline.doctor as doctor_mod
    source = inspect.getsource(doctor_mod)
    assert "make_client" not in source
    assert "make_provider" not in source
    assert "call_json" not in source
    assert "pipeline.llm.anthropic_api" not in source
    assert "from pipeline.llm.factory" not in source
    assert "from pipeline.llm.client" not in source
    assert "ClaudeCliProvider" not in source
    assert "AnthropicApiProvider" not in source
    assert "complete(" not in source


def test_egress_check_lists_configured_providers(tmp_path):
    c = _by_name(run_checks(Config(home=str(tmp_path),
                                   transcription={"providers": ["youtube_captions", "ytdlp_subs"]},
                                   llm={"provider": "anthropic_api"})))
    egress = c["egress"]
    assert egress.required is False and egress.ok is True
    assert "youtube_captions" in egress.detail
    assert "ytdlp_subs" in egress.detail
    assert "whisper_local" not in egress.detail
    assert "api.anthropic.com" in egress.detail


def test_run_checks_never_raises_on_missing_tools(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.doctor.shutil.which", lambda name: None)
    checks = run_checks(Config(home=str(tmp_path), transcription={"providers": ["whisper_local"]}))
    assert isinstance(checks, list) and len(checks) > 0


# --- additional tests: residual fixes ----------------------------------------

def test_doctor_uses_shared_exit_code_constants():
    # Regression guard: doctor must import its exit codes from pipeline.exit_codes
    # rather than hardcoding literal 2/0 -- if it ever reverts to literals these module attributes
    # disappear and this test fails with an AttributeError.
    from pipeline import doctor, exit_codes
    assert doctor.EXIT_ERROR == exit_codes.EXIT_ERROR == 2
    assert doctor.EXIT_OK == exit_codes.EXIT_OK == 0


def test_doctor_uses_the_shared_envfile_lookup():
    # Provenance must match the provider: both use pipeline.envfile, the project-root .env only,
    # never a .env found by an upward search (which reaches a parent checkout from a worktree).
    import inspect
    from pipeline import doctor
    src = inspect.getsource(doctor)
    assert "from pipeline.envfile import project_dotenv_path" in src
    assert "find_dotenv" not in src


def test_doctor_ignores_parent_checkout_dotenv(monkeypatch, tmp_path):
    from pipeline import envfile
    parent = tmp_path / "parent"
    project = parent / "wt"
    project.mkdir(parents=True)
    (parent / "pyproject.toml").write_text("", encoding="utf-8")
    (parent / ".env").write_text("ANTHROPIC_API_KEY=sk-test-parent\n", encoding="utf-8")
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(envfile, "PROJECT_ROOT", project)
    monkeypatch.setattr("pipeline.doctor.project_dotenv_path", envfile.project_dotenv_path)
    monkeypatch.setattr("pipeline.doctor.load_dotenv", dotenv.load_dotenv)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.chdir(project)
    c = _by_name(run_checks(Config(home=str(tmp_path))))
    assert c["anthropic_api_key"].ok is False


# --- supadata (DECISION-026) --------------------------------------------------

_WITH_SUPADATA = ["youtube_captions", "ytdlp_subs", "supadata", "whisper_local"]


def test_supadata_key_check_absent_when_not_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
    c = _by_name(run_checks(Config(home=str(tmp_path), transcription={"providers": ["youtube_captions"]})))
    assert "supadata_api_key" not in c
    assert "supadata" not in c["egress"].detail


def test_supadata_key_missing_is_a_warning_among_fallbacks(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
    c = _by_name(run_checks(Config(home=str(tmp_path), transcription={"providers": _WITH_SUPADATA})))
    k = c["supadata_api_key"]
    assert k.ok is False and k.required is False and "SUPADATA_API_KEY" in k.detail


def test_supadata_key_missing_fails_when_sole_provider(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
    c = _by_name(run_checks(Config(home=str(tmp_path), transcription={"providers": ["supadata"]})))
    assert c["supadata_api_key"].ok is False and c["supadata_api_key"].required is True


def test_supadata_key_set_is_ok_and_never_printed(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPADATA_API_KEY", "sd-very-secret")
    c = _by_name(run_checks(Config(home=str(tmp_path), transcription={"providers": _WITH_SUPADATA})))
    k = c["supadata_api_key"]
    assert k.ok is True and "set" in k.detail
    text, _ = format_report(list(c.values()))
    assert "sd-very-secret" not in text


def test_supadata_key_detail_names_dotenv_source(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("SUPADATA_API_KEY=sd-test-from-file\n", encoding="utf-8")
    monkeypatch.setattr("pipeline.doctor.project_dotenv_path", lambda *a, **k: str(env_path))
    monkeypatch.setattr("pipeline.doctor.load_dotenv", dotenv.load_dotenv)
    # setenv-then-delenv registers the variable with monkeypatch, so the value load_dotenv
    # writes into os.environ is removed again at teardown.
    monkeypatch.setenv("SUPADATA_API_KEY", "placeholder")
    monkeypatch.delenv("SUPADATA_API_KEY")
    c = _by_name(run_checks(Config(home=str(tmp_path), transcription={"providers": _WITH_SUPADATA})))
    assert c["supadata_api_key"].ok is True
    assert str(env_path) in c["supadata_api_key"].detail
    assert "sd-test-from-file" not in c["supadata_api_key"].detail


def test_egress_names_supadata_as_third_party(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
    c = _by_name(run_checks(Config(home=str(tmp_path), transcription={"providers": _WITH_SUPADATA})))
    assert "transcribe: supadata -> api.supadata.ai (video ids sent to a third party)" in c["egress"].detail


def test_doctor_never_calls_the_supadata_api(monkeypatch, tmp_path):
    import inspect

    import pipeline.doctor as doctor_mod
    import pipeline.transcription.supadata as sd

    def boom(*a, **k):
        raise AssertionError("doctor must not call the Supadata API")

    monkeypatch.setattr(sd, "_http_get", boom)
    monkeypatch.setattr(sd._OPENER, "open", boom)
    monkeypatch.setenv("SUPADATA_API_KEY", "sd-x")
    run_checks(Config(home=str(tmp_path), transcription={"providers": _WITH_SUPADATA}))
    assert "pipeline.transcription.supadata" not in inspect.getsource(doctor_mod)


def test_cmd_doctor_prints_model_section_without_changing_the_exit_code(monkeypatch, tmp_path, capsys):
    from pipeline.doctor import cmd_doctor
    from pipeline.models_monitor import Finding, cached_findings

    class FakeArgs:
        prewarm_whisper = False
        no_model_check = False

    monkeypatch.setattr("pipeline.doctor.run_checks", lambda cfg: [Check("a", True, True, "fine")])
    monkeypatch.setattr("pipeline.models_monitor.check_models", lambda cfg, **kw: [
        Finding("suggest", "merge", "newer model claude-sonnet-9-9 ..."), Finding("warn", None, "price list differs")])
    cfg = Config(home=str(tmp_path))
    code = cmd_doctor(cfg, FakeArgs())
    out = capsys.readouterr().out
    assert code == 0 and "DOCTOR=ok" in out
    assert "Models:" in out and "[suggest] merge: newer model claude-sonnet-9-9" in out
    assert "[warn] price list differs" in out
    assert out.index("Models:") < out.index("DOCTOR=")
    assert [f.level for f in cached_findings(cfg)] == ["suggest", "warn"]


def test_cmd_doctor_no_model_check_skips_the_network(monkeypatch, tmp_path, capsys):
    from pipeline.doctor import cmd_doctor

    class FakeArgs:
        prewarm_whisper = False
        no_model_check = True

    monkeypatch.setattr("pipeline.doctor.run_checks", lambda cfg: [Check("a", True, True, "fine")])
    monkeypatch.setattr("pipeline.models_monitor.check_models",
                        lambda cfg, **kw: (_ for _ in ()).throw(AssertionError("network used")))
    cmd_doctor(Config(home=str(tmp_path)), FakeArgs())
    assert "Models: not checked (--no-model-check)" in capsys.readouterr().out


def test_cmd_doctor_offline_reports_skipped(monkeypatch, tmp_path, capsys):
    from pipeline.doctor import cmd_doctor

    class FakeArgs:
        prewarm_whisper = False
        no_model_check = False

    monkeypatch.setattr("pipeline.doctor.run_checks", lambda cfg: [Check("a", True, True, "fine")])
    code = cmd_doctor(Config(home=str(tmp_path)), FakeArgs())     # conftest makes every fetch fail
    out = capsys.readouterr().out
    assert code == 0 and "[skipped] could not read:" in out


def test_egress_lists_the_model_monitor(tmp_path):
    c = _by_name(run_checks(Config(home=str(tmp_path))))
    assert "platform.claude.com" in c["egress"].detail


def test_warns_that_local_media_needs_whisper_local_when_not_configured(tmp_path):
    c = _by_name(run_checks(Config(home=str(tmp_path), transcription={"providers": ["youtube_captions"]})))
    chk = c["local_media"]
    assert chk.ok is False and chk.required is False
    assert "local audio/video needs whisper_local and ffmpeg" in chk.detail
    assert "[warn]" in format_report([chk])[0]


def test_no_local_media_warning_when_whisper_local_configured(tmp_path):
    c = _by_name(run_checks(Config(home=str(tmp_path))))
    assert "local_media" not in c
