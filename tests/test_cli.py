from tests.helpers import yt_ref
import json
import os
import pytest
from pipeline import db
from pipeline.config import Config
from pipeline.transcription.cache import TranscriptCache
from tests.helpers import ScriptedProvider, StubTranscriber, fake_metadata


def _scripted_provider(cfg, name=None, **kw):
    # Mirrors real make_provider: the returned provider's .name reflects what was requested, so
    # the Runner-level price gate (which keys on the actual provider's .name) behaves the same
    # under test as it does in production.
    provider = ScriptedProvider()
    if name:
        provider.name = name
    return provider


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    # claude_cli's default rate limit (6 rpm, pipeline/config.py) is a real sliding window: a
    # scripted run makes 8 calls (2 analyze, outline, 2 sections, 2 checks, verify), so the 7th
    # would sleep ~60 s for real. Tests never reach a real endpoint, so the limit is switched off.
    cfg_path.write_text(f"home: {str(tmp_path / 'home')!r}\nrun:\n  budget_usd: 0\n"
                        "llm:\n  rate_limits:\n    claude_cli: {rpm: 0}\n", encoding="utf-8")
    monkeypatch.setattr("pipeline.cli.make_provider", _scripted_provider)
    monkeypatch.setattr("pipeline.runner.build_transcribers", lambda cfg: [StubTranscriber()])
    monkeypatch.setattr("pipeline.runner.fetch_source_metadata", fake_metadata)
    urls = tmp_path / "urls.txt"
    urls.write_text("https://youtu.be/aaaaaaaaaaa\nhttps://youtu.be/bbbbbbbbbbb\n", encoding="utf-8")
    return str(cfg_path), str(urls)


def _main(argv):
    from pipeline.cli import main
    return main(argv)


def test_run_creates_notebook_and_report(env, capsys):
    cfg_path, urls = env
    code = _main(["--config", cfg_path, "run", "--topic", "Acme SEO", "--input", urls])
    out = capsys.readouterr().out
    assert code == 0 and "REPORT=" in out and "NOTEBOOK=acme-seo" in out
    assert "RUN_ID=" in out and "COST=" in out
    report_line = next(l for l in out.splitlines() if l.startswith("REPORT="))
    report = report_line.split("REPORT=", 1)[1].strip()
    assert os.path.exists(report)


def test_run_requires_one_input(env, capsys):
    cfg_path, urls = env
    assert _main(["--config", cfg_path, "run", "--topic", "T"]) == 4
    assert "one of --input" in capsys.readouterr().err


def test_topic_file(env, tmp_path):
    cfg_path, urls = env
    tf = tmp_path / "topic.txt"; tf.write_text("It's \"quoted\" $(topic)\n", encoding="utf-8")
    assert _main(["--config", cfg_path, "run", "--topic-file", str(tf), "--input", urls]) == 0
    cfg = Config(home=str(tmp_path / "home"))
    assert db.list_notebooks(cfg.db_path)[0]["topic"] == 'It\'s "quoted" $(topic)'


def test_resume_and_analyze_and_status(env, capsys):
    cfg_path, urls = env
    _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls, "--stages", "fetch,transcribe"])
    assert _main(["--config", cfg_path, "resume", "t"]) == 0
    assert "NOTEBOOK=t" in capsys.readouterr().out
    assert _main(["--config", cfg_path, "analyze", "t", "--force"]) == 0
    assert "NOTEBOOK=t" in capsys.readouterr().out
    assert _main(["--config", cfg_path, "status"]) == 0
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.strip()]
    assert any(line.split()[0] == "t" for line in lines) and "done" in out
    assert _main(["--config", cfg_path, "status", "t"]) == 0
    out = capsys.readouterr().out
    assert "aaaaaaaaaaa" in out and "Total cost" in out and "REPORT=" in out
    # resume and analyze both made LLM calls, so both timing sections appear
    assert "Call timing, run" in out and "Call timing, all runs" in out and "p50_s" in out
    assert _main(["--config", cfg_path, "notebook", "list"]) == 0


def test_unknown_notebook_is_exit_4(env, capsys):
    cfg_path, urls = env
    assert _main(["--config", cfg_path, "resume", "nope"]) == 4
    assert "not found" in capsys.readouterr().err


def test_budget_pause_exit_code_3(env, monkeypatch, capsys):
    cfg_path, urls = env
    monkeypatch.setattr("pipeline.cli.make_provider", lambda cfg, name=None, **kw: ScriptedProvider(cost_per_call=0.6))
    code = _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls, "--budget", "0.5", "--max-parallel", "1"])
    assert code == 3
    assert "hmd resume" in capsys.readouterr().out


def test_legacy_db_is_exit_4(tmp_path, capsys):
    import sqlite3
    home = tmp_path / "home"; (home / "domains" / "general").mkdir(parents=True)
    conn = sqlite3.connect(str(home / "domains" / "general" / "state.sqlite"))
    conn.execute("CREATE TABLE projects (id INTEGER)"); conn.commit(); conn.close()
    cfg_path = tmp_path / "c.yaml"; cfg_path.write_text(f"home: {str(home)!r}\n", encoding="utf-8")
    assert _main(["--config", str(cfg_path), "status"]) == 4
    assert "MILESTONE-0" in capsys.readouterr().err


# --- additional tests: analyze and resume -----------------------------------

def test_analyze_force_without_merge_output_is_usage_error(env, capsys):
    cfg_path, urls = env
    _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls, "--stages", "fetch,transcribe"])
    code = _main(["--config", cfg_path, "analyze", "t", "--force", "--stages", "analyze"])
    err = capsys.readouterr().err
    assert code == 4 and "merge" in err


def test_resume_retranscribe_reruns_transcription(env, monkeypatch, capsys):
    cfg_path, urls = env

    class CountingTranscriber(StubTranscriber):
        calls = 0

        def transcribe(self, ref, languages):
            type(self).calls += 1
            return super().transcribe(ref, languages)

    monkeypatch.setattr("pipeline.runner.build_transcribers", lambda cfg: [CountingTranscriber()])
    assert _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls]) == 0
    assert CountingTranscriber.calls == 2
    assert _main(["--config", cfg_path, "resume", "t", "--retranscribe"]) == 0
    assert CountingTranscriber.calls == 4


def test_run_unknown_stages_is_usage_error(env, capsys):
    cfg_path, urls = env
    code = _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls, "--stages", "bogus"])
    assert code == 4
    assert "unknown stages" in capsys.readouterr().err


def test_unpriced_model_refused_before_notebook_created(env, capsys):
    cfg_path, urls = env
    code = _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls,
                 "--analyze-model", "mystery-9", "--budget", "1"])
    err = capsys.readouterr().err
    assert code == 4 and "no price" in err
    from pipeline.config import load_config
    real_cfg = load_config(cfg_path)
    assert db.list_notebooks(real_cfg.db_path) == []


def test_unpriced_model_ok_with_claude_cli_provider(env, capsys):
    cfg_path, urls = env
    code = _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls,
                 "--provider", "claude_cli", "--analyze-model", "mystery-9", "--budget", "1"])
    assert code == 0


def test_status_on_empty_database_prints_hint(env, capsys):
    cfg_path, urls = env
    code = _main(["--config", cfg_path, "status"])
    out = capsys.readouterr().out
    assert code == 0 and "No notebooks yet" in out


def test_stage_error_maps_to_exit_2(env, capsys):
    cfg_path, urls = env
    _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls, "--stages", "fetch"])
    code = _main(["--config", cfg_path, "analyze", "t", "--stages", "merge,output"])
    out = capsys.readouterr().out
    assert code == 2 and "STATUS=error" in out
    from pipeline.config import load_config
    real_cfg = load_config(cfg_path)
    nb = db.get_notebook(real_cfg.db_path, "t")
    run = db.latest_run(real_cfg.db_path, nb["id"])
    assert run["status"] == "error"


def test_fatal_llm_error_prints_one_clear_line_and_exits_2(env, capsys, monkeypatch):
    from pipeline.llm.base import LLMFatalError

    class Broke(ScriptedProvider):
        calls = 0

        def complete(self, prompt, **kw):
            Broke.calls += 1
            raise LLMFatalError("API 400 invalid_request_error: Your credit balance is too low\n"
                                "to access the Anthropic API.")

    monkeypatch.setattr("pipeline.cli.make_provider", lambda cfg, name=None, **kw: Broke())
    cfg_path, urls = env
    code = _main(["--config", cfg_path, "run", "--topic", "Credit", "--input", urls, "--max-parallel", "1"])
    out = capsys.readouterr().out
    assert code == 2 and Broke.calls == 1
    lines = out.splitlines()
    assert "STATUS=error" in lines
    error_lines = [l for l in lines if l.startswith("ERROR: ")]
    assert error_lines == [
        "ERROR: anthropic_api: API 400 invalid_request_error: Your credit balance is too low to access the "
        "Anthropic API. -- the run stopped; fix the account/key and run hmd resume credit"]


def test_keyboard_interrupt_is_exit_130(env, monkeypatch, capsys):
    cfg_path, urls = env

    class InterruptingProvider(ScriptedProvider):
        def complete(self, *a, **kw):
            raise KeyboardInterrupt()

    monkeypatch.setattr("pipeline.cli.make_provider", lambda cfg, name=None, **kw: InterruptingProvider())
    code = _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls])
    assert code == 130
    assert "hmd: interrupted" in capsys.readouterr().err

    from pipeline.config import load_config
    real_cfg = load_config(cfg_path)
    nb = db.get_notebook(real_cfg.db_path, "t")
    run = db.latest_run(real_cfg.db_path, nb["id"])
    assert run["status"] == "error" and "interrupted" in (run["error"] or "")


def test_eval_no_goldens_is_exit_0(env, golden_root, capsys):
    cfg_path, urls = env
    code = _main(["--config", cfg_path, "eval"])
    err = capsys.readouterr().err
    assert code == 0
    assert "no golden" in err


def test_eval_no_goldens_names_absolute_repo_path_on_stderr(env, tmp_path, monkeypatch, capsys):
    # GOLDEN_ROOT is deliberately NOT patched here: this exercises the real, package-relative
    # path so it must be an absolute path under the repo root, found regardless of the CWD. The
    # real evals/golden/ holds the recorded goldens when present (not shipped publicly,
    # DECISION-019), so only the listing is emptied -- otherwise this would replay every golden.
    from pipeline.evals import GOLDEN_ROOT
    cfg_path, urls = env
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pipeline.evals.list_goldens", lambda root=None: [])
    code = _main(["--config", cfg_path, "eval"])
    err = capsys.readouterr().err
    assert code == 0
    assert GOLDEN_ROOT in err
    assert os.path.isabs(GOLDEN_ROOT)


# --- code-quality review follow-ups -----------------------------------------

def test_topic_with_newline_cannot_forge_report_line(env, capsys):
    cfg_path, urls = env
    code = _main(["--config", cfg_path, "run", "--topic", "Benign\nREPORT=C:/fake.md", "--input", urls])
    out = capsys.readouterr().out
    assert code == 0
    report_lines = [l for l in out.splitlines() if l.startswith("REPORT=")]
    assert len(report_lines) == 1
    assert os.path.exists(report_lines[0].split("REPORT=", 1)[1])
    from pipeline.config import load_config
    real_cfg = load_config(cfg_path)
    topic = db.list_notebooks(real_cfg.db_path)[0]["topic"]
    assert "\n" not in topic


def test_status_handles_null_source_columns(env, capsys):
    cfg_path, urls = env
    _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls, "--stages", "fetch"])
    import sqlite3
    from pipeline.config import load_config
    real_cfg = load_config(cfg_path)
    nb = db.get_notebook(real_cfg.db_path, "t")
    conn = sqlite3.connect(real_cfg.db_path)
    conn.execute(
        "UPDATE sources SET fetch_status=NULL, transcribe_status=NULL, analyze_status=NULL "
        "WHERE notebook_id=?", (nb["id"],))
    conn.commit()
    conn.close()
    assert _main(["--config", cfg_path, "status", "t"]) == 0


def test_missing_explicit_config_is_usage_error(tmp_path, capsys):
    missing = str(tmp_path / "nope.yaml")
    code = _main(["--config", missing, "status"])
    assert code == 4
    assert f"config file not found: {missing}" in capsys.readouterr().err


def test_missing_default_config_falls_back_to_defaults(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = _main(["status"])
    out = capsys.readouterr().out
    assert code == 0 and "No notebooks yet" in out


def test_stages_empty_string_uses_default(env, capsys):
    cfg_path, urls = env
    code = _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls, "--stages", ""])
    assert code == 0


def test_stages_whitespace_only_uses_default(env, capsys):
    cfg_path, urls = env
    code = _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls, "--stages", "   "])
    assert code == 0


def test_stages_commas_only_is_usage_error(env, capsys):
    cfg_path, urls = env
    code = _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls, "--stages", ",,,"])
    assert code == 4
    assert "--stages must not be empty" in capsys.readouterr().err


def test_negative_budget_is_usage_error(env, capsys):
    cfg_path, urls = env
    code = _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls, "--budget", "-1"])
    assert code == 4
    assert "--budget must be >= 0" in capsys.readouterr().err


def test_optional_command_returns_none_when_module_itself_missing(monkeypatch):
    from pipeline.cli import _optional_command

    def fake_import(name, *a, **kw):
        raise ModuleNotFoundError("boom", name=name)

    monkeypatch.setattr("pipeline.cli.importlib.import_module", fake_import)
    assert _optional_command("pipeline.init_cmd", "init") is None


def test_optional_command_reraises_unrelated_missing_module(monkeypatch):
    from pipeline.cli import _optional_command

    def fake_import(name, *a, **kw):
        raise ModuleNotFoundError("boom", name="totally_different_dependency")

    monkeypatch.setattr("pipeline.cli.importlib.import_module", fake_import)
    with pytest.raises(ModuleNotFoundError):
        _optional_command("pipeline.init_cmd", "init")


def test_hmd_debug_reraises_usage_errors(env, monkeypatch):
    cfg_path, urls = env
    monkeypatch.setenv("HMD_DEBUG", "1")
    from pipeline.config import ConfigError
    with pytest.raises(ConfigError):
        _main(["--config", cfg_path, "run", "--topic", "T"])


def test_stage_canonicalization_normalizes_order(env, capsys):
    cfg_path, urls = env
    assert _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls]) == 0
    assert _main(["--config", cfg_path, "analyze", "t", "--stages", "output,merge"]) == 0
    from pipeline.config import load_config
    real_cfg = load_config(cfg_path)
    nb = db.get_notebook(real_cfg.db_path, "t")
    run = db.latest_run(real_cfg.db_path, nb["id"])
    assert run["stages"] == "merge,output"


# --- additional tests: hmd init ----------------------------------------------

def test_init_bypasses_explicit_config_existence_check(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # so init's cwd-relative .env.example write can never touch the repo
    new_config = tmp_path / "new.yaml"
    home = tmp_path / "home"
    assert not new_config.exists()
    code = _main(["--config", str(new_config), "init", "--home", str(home)])
    assert code == 0
    assert new_config.exists()


def test_init_with_no_explicit_config_writes_default_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("config")
    assert _main(["init"]) == 0
    assert os.path.exists("config/config.yaml")


# --- whole-phase review follow-ups -------------------------------------------

def test_topic_file_that_is_a_directory_is_usage_error(env, tmp_path, capsys):
    cfg_path, urls = env
    bad_dir = tmp_path / "not_a_file"
    bad_dir.mkdir()
    code = _main(["--config", cfg_path, "run", "--topic-file", str(bad_dir), "--input", urls])
    assert code == 4
    assert "hmd:" in capsys.readouterr().err


def test_status_topic_with_report_substring_does_not_forge_report_line(env, capsys):
    cfg_path, urls = env
    assert _main(["--config", cfg_path, "run", "--topic", "x REPORT=C:/forged.md y", "--input", urls]) == 0
    capsys.readouterr()   # discard run's own stdout; run no longer echoes the topic at all
    from pipeline.config import load_config
    real_cfg = load_config(cfg_path)
    slug = db.list_notebooks(real_cfg.db_path)[0]["slug"]
    assert _main(["--config", cfg_path, "status", slug]) == 0
    out = capsys.readouterr().out
    assert "x REPORT=C:/forged.md y" in out    # the topic is shown, but never at line-start
    report_lines = [l for l in out.splitlines() if l.startswith("REPORT=")]
    assert len(report_lines) == 1
    assert os.path.exists(report_lines[0].split("REPORT=", 1)[1])


# --- additional tests: residual fixes ----------------------------------------

def test_analyze_force_without_analyze_stage_prints_no_notebook_line(env, capsys):
    cfg_path, urls = env
    _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls, "--stages", "fetch,transcribe"])
    capsys.readouterr()   # discard the run's own output
    code = _main(["--config", cfg_path, "analyze", "t", "--force", "--stages", "merge,output"])
    out, err = capsys.readouterr()
    assert code == 4
    assert not any(l.startswith("NOTEBOOK=") for l in out.splitlines())
    assert "analyze" in err


def test_show_anchors_flag_defaults_to_none(env):
    from pipeline.cli import build_parser
    parsed = build_parser().parse_args(["run", "--topic", "t", "--url", "u"])
    assert parsed.show_anchors is None


def _report_text(out: str) -> str:
    report_line = next(l for l in out.splitlines() if l.startswith("REPORT="))
    return open(report_line.split("REPORT=", 1)[1].strip(), encoding="utf-8").read()


def test_no_show_anchors_flag_overrides_config_true(env, capsys):
    cfg_path, urls = env
    with open(cfg_path, "a", encoding="utf-8") as f:
        f.write("output:\n  show_anchors: true\n")

    code = _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls, "--no-show-anchors"])
    out = capsys.readouterr().out
    assert code == 0
    assert '  > "' not in _report_text(out)


def test_show_anchors_config_true_without_flag_renders_quotes(env, capsys):
    cfg_path, urls = env
    with open(cfg_path, "a", encoding="utf-8") as f:
        f.write("output:\n  show_anchors: true\n")

    code = _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls])
    out = capsys.readouterr().out
    assert code == 0
    assert '  > "' in _report_text(out)


def test_init_writes_env_example_next_to_config_not_cwd(tmp_path, monkeypatch, capsys):
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    project_root = tmp_path / "project"
    config_path = project_root / "config" / "config.yaml"

    code = _main(["--config", str(config_path), "init", "--home", str(project_root / "data")])
    out = capsys.readouterr().out

    assert code == 0
    expected_env_example = project_root / ".env.example"
    assert expected_env_example.exists()
    assert not (other_cwd / ".env.example").exists()
    env_lines = [l for l in out.splitlines() if l.startswith("ENV=")]
    assert len(env_lines) == 1
    assert str(expected_env_example) in env_lines[0]


# --- eval command ------------------------------------------------------------

def _scripted_replay_provider(cfg, name=None, cassette_dir=None, record=False):
    from pipeline.llm.replay import ReplayProvider
    return ReplayProvider(cassette_dir, inner=ScriptedProvider() if record else None, record=record)


def _scripted_replay_provider_costly(cfg, name=None, cassette_dir=None, record=False):
    from pipeline.llm.replay import ReplayProvider
    return ReplayProvider(cassette_dir, inner=ScriptedProvider(cost_per_call=10.0) if record else None,
                          record=record)


@pytest.fixture
def golden_root(tmp_path, monkeypatch):
    root = tmp_path / "golden"
    monkeypatch.setattr("pipeline.evals.GOLDEN_ROOT", str(root))
    monkeypatch.setattr("pipeline.evals.make_provider", _scripted_replay_provider)
    return root


def _make_cli_golden(root, name="synthetic", must_contain=()):
    vids = ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"]
    gdir = root / name
    cache = TranscriptCache(str(gdir / "transcripts"))
    for v in vids:
        cache.save(StubTranscriber().transcribe(yt_ref(v), ["en"]))
    manifest = {
        "topic": "Synthetic topic",
        "sources": [{"video_id": v, "title": f"Video {v[:4]}", "channel": "Chan", "duration_seconds": 20,
                    "upload_date": "20260101", "transcript_method": "youtube_captions"} for v in vids],
        "assertions": {"min_sections": 1, "min_points": 1, "must_contain": list(must_contain),
                       "min_contradictions": 0, "max_dropped_ratio": 0.3},
    }
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return gdir


def test_eval_replay_passes_with_recorded_cassettes(env, golden_root, capsys):
    from pipeline.evals import run_golden
    cfg_path, urls = env
    gdir = _make_cli_golden(golden_root)
    run_golden(str(gdir), provider_name="anthropic_api", record=True, home=str(golden_root / "rechome"))
    code = _main(["--config", cfg_path, "eval"])
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert code == 0
    assert "GOLDEN=synthetic" in lines
    assert "PROVIDER=anthropic_api" in lines
    assert "COST=0.0000" in lines
    assert "STATUS=pass" in lines
    assert any("0 miss(es)" in l for l in lines)


def test_eval_replay_fails_on_missing_detail(env, golden_root, capsys):
    from pipeline.evals import run_golden
    cfg_path, urls = env
    gdir = _make_cli_golden(golden_root, must_contain=("the number forty-two",))
    run_golden(str(gdir), provider_name="anthropic_api", record=True, home=str(golden_root / "rechome"))
    code = _main(["--config", cfg_path, "eval", "synthetic", "--provider", "anthropic_api"])
    lines = capsys.readouterr().out.splitlines()
    assert code == 2
    assert "STATUS=fail" in lines
    assert any("forty-two" in l for l in lines)


def test_eval_replay_skips_when_no_cassettes(env, golden_root, capsys):
    cfg_path, urls = env
    _make_cli_golden(golden_root)
    code = _main(["--config", cfg_path, "eval", "synthetic", "--provider", "anthropic_api"])
    out = capsys.readouterr().out
    assert code == 0
    assert "skip:" in out and "no cassettes" in out


def test_eval_live_without_budget_is_usage_error_and_never_calls_make_provider(env, golden_root, monkeypatch, capsys):
    cfg_path, urls = env
    _make_cli_golden(golden_root)

    def boom(*a, **kw):
        raise AssertionError("make_provider must not be called before the budget refusal")

    monkeypatch.setattr("pipeline.evals.make_provider", boom)
    code = _main(["--config", cfg_path, "eval", "--live"])
    err = capsys.readouterr().err
    assert code == 4
    assert "--budget" in err


def test_eval_negative_budget_is_usage_error(env, golden_root, capsys):
    cfg_path, urls = env
    _make_cli_golden(golden_root)
    code = _main(["--config", cfg_path, "eval", "--live", "--budget", "-1"])
    err = capsys.readouterr().err
    assert code == 4
    assert "--budget" in err


def test_eval_live_records_cassettes_and_passes(env, golden_root, capsys):
    cfg_path, urls = env
    _make_cli_golden(golden_root)
    code = _main(["--config", cfg_path, "eval", "--live", "--provider", "anthropic_api", "--budget", "1"])
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert code == 0
    assert "GOLDEN=synthetic" in lines
    assert "PROVIDER=anthropic_api" in lines
    assert "STATUS=pass" in lines
    assert any(l.startswith("REPORT=") for l in lines)
    assert any(l.startswith("HOME=") for l in lines)
    cassette_dir = golden_root / "synthetic" / "cassettes" / "anthropic_api"
    assert cassette_dir.is_dir() and len(list(cassette_dir.iterdir())) > 0


def test_eval_live_budget_pause_reports_paused_not_failed(env, golden_root, monkeypatch, capsys):
    cfg_path, urls = env
    _make_cli_golden(golden_root)
    monkeypatch.setattr("pipeline.evals.make_provider", _scripted_replay_provider_costly)
    code = _main(["--config", cfg_path, "eval", "--live", "--provider", "anthropic_api", "--budget", "5"])
    lines = capsys.readouterr().out.splitlines()
    assert code == 3
    assert "STATUS=paused" in lines
    assert not any(l == "STATUS=fail" for l in lines)


def test_eval_export_from_notebook(env, golden_root, capsys):
    cfg_path, urls = env
    _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls])
    capsys.readouterr()
    code = _main(["--config", cfg_path, "eval", "exp", "--export-from", "t"])
    out = capsys.readouterr().out
    assert code == 0
    expected_gdir = os.path.join(str(golden_root), "exp")
    assert f"GOLDEN={expected_gdir}" in out
    assert os.path.isdir(expected_gdir)
    assert os.path.exists(os.path.join(expected_gdir, "manifest.json"))


def test_eval_export_from_without_name_is_usage_error(env, golden_root, capsys):
    cfg_path, urls = env
    code = _main(["--config", cfg_path, "eval", "--export-from", "t"])
    err = capsys.readouterr().err
    assert code == 4
    assert "NAME" in err


def test_eval_name_rejects_path_traversal(env, golden_root, capsys):
    cfg_path, urls = env
    code = _main(["--config", cfg_path, "eval", "../x"])
    err = capsys.readouterr().err
    assert code == 4
    assert "../x" in err


def test_eval_export_from_rejects_live_flag(env, golden_root, capsys):
    cfg_path, urls = env
    _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls])
    capsys.readouterr()
    code = _main(["--config", cfg_path, "eval", "exp", "--export-from", "t", "--live"])
    err = capsys.readouterr().err
    assert code == 4
    assert "--export-from" in err


def test_eval_export_from_rejects_provider_flag(env, golden_root, capsys):
    cfg_path, urls = env
    _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls])
    capsys.readouterr()
    code = _main(["--config", cfg_path, "eval", "exp", "--export-from", "t", "--provider", "anthropic_api"])
    err = capsys.readouterr().err
    assert code == 4
    assert "--export-from" in err


def test_eval_export_from_rejects_unsafe_name(env, golden_root, capsys):
    cfg_path, urls = env
    _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls])
    capsys.readouterr()
    code = _main(["--config", cfg_path, "eval", "../escape", "--export-from", "t"])
    err = capsys.readouterr().err
    assert code == 4
    assert not os.path.exists(os.path.normpath(os.path.join(str(golden_root), "..", "escape")))


def test_eval_missing_fixture_is_usage_error_naming_the_file_and_leaks_no_tempdir(
        env, golden_root, tmp_path, monkeypatch, capsys):
    import shutil as _shutil
    import tempfile as _tempfile
    systmp = tmp_path / "systmp"
    systmp.mkdir()
    monkeypatch.setattr(_tempfile, "tempdir", str(systmp))
    cfg_path, urls = env
    gdir = _make_cli_golden(golden_root)
    _shutil.rmtree(gdir / "transcripts" / "aaaaaaaaaaa")
    code = _main(["--config", cfg_path, "eval", "--live", "--provider", "anthropic_api", "--budget", "1"])
    err = capsys.readouterr().err
    assert code == 4
    assert "aaaaaaaaaaa" in err
    leftover = [d for d in os.listdir(str(systmp)) if d.startswith("hmd-eval-")]
    assert leftover == []


def test_eval_manifest_missing_topic_is_usage_error_naming_the_file(env, golden_root, capsys):
    cfg_path, urls = env
    gdir = golden_root / "broken"
    gdir.mkdir(parents=True)
    (gdir / "manifest.json").write_text(json.dumps({"sources": [{"video_id": "aaaaaaaaaaa"}]}), encoding="utf-8")
    code = _main(["--config", cfg_path, "eval", "broken"])
    err = capsys.readouterr().err
    assert code == 4
    assert "manifest.json" in err
    assert "topic" in err


def test_eval_replay_names_rerecord_command_on_analyze_cassette_miss(env, golden_root, capsys):
    from pipeline.evals import run_golden
    cfg_path, urls = env
    gdir = _make_cli_golden(golden_root)
    run_golden(str(gdir), provider_name="anthropic_api", record=True, home=str(golden_root / "rechome"))
    cassette_dir = golden_root / "synthetic" / "cassettes" / "anthropic_api"
    deleted = 0
    for f in list(cassette_dir.iterdir()):
        data = json.loads(f.read_text(encoding="utf-8"))
        if "HMD_DATA TRANSCRIPT" in data.get("prompt", ""):
            f.unlink()
            deleted += 1
    assert deleted == 3   # one analyze cassette per source: all three must miss on replay
    code = _main(["--config", cfg_path, "eval", "synthetic", "--provider", "anthropic_api"])
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert code == 2
    assert "STATUS=fail" in lines
    hint = [l for l in lines if "cassette miss(es)" in l]
    assert len(hint) == 1
    assert "re-record with:" in hint[0] and "--live" in hint[0] and "--provider anthropic_api" in hint[0]


def test_eval_budget_without_live_is_usage_error(env, golden_root, capsys):
    cfg_path, urls = env
    _make_cli_golden(golden_root)
    code = _main(["--config", cfg_path, "eval", "--budget", "1"])
    err = capsys.readouterr().err
    assert code == 4
    assert "--live" in err


def test_eval_exit_precedence_error_outranks_paused(env, golden_root, monkeypatch, capsys):
    from pipeline import evals as evals_mod
    from pipeline.exit_codes import EXIT_ERROR, EXIT_PAUSED
    cfg_path, urls = env
    _make_cli_golden(golden_root, name="g1")
    _make_cli_golden(golden_root, name="g2")
    codes = iter([EXIT_PAUSED, EXIT_ERROR])
    monkeypatch.setattr(evals_mod, "_run_one", lambda *a, **kw: next(codes))
    code = _main(["--config", cfg_path, "eval", "--provider", "anthropic_api"])
    assert code == 2


# --- console encoding (2026-10-09: a billed run died on a cp1252 stdout) ---------------------

NON_CP1252 = "3 × zoom — 中文 \U0001f4f7"   # x-sign and em dash are cp1252; CJK/emoji are not


def _cp1252_streams(monkeypatch):
    import io
    import sys
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", line_buffering=True)
    err = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", line_buffering=True)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    return out, err


def _text_of(stream) -> str:
    stream.flush()
    return stream.buffer.getvalue().decode(stream.encoding, errors="replace")


def test_run_on_a_cp1252_stdout_completes_with_non_cp1252_log_lines(env, monkeypatch):
    # Model and YouTube text reaches the console through log lines (merge:check prints point text);
    # stdout redirected to a file on Windows is cp1252. A log line must never abort a run.
    cfg_path, urls = env

    def noisy_metadata(db_path, source_id, log=print):
        log(f"  metadata: {NON_CP1252}")
        return fake_metadata(db_path, source_id, log=log)

    monkeypatch.setattr("pipeline.runner.fetch_source_metadata", noisy_metadata)
    out, err = _cp1252_streams(monkeypatch)
    code = _main(["--config", cfg_path, "run", "--topic", "Cameras", "--input", urls])
    assert code == 0, _text_of(err)
    assert out.encoding.lower().replace("-", "") == "utf8" and out.line_buffering
    text = _text_of(out)
    assert "STATUS=done" in text and NON_CP1252 in text


def test_status_and_list_on_a_cp1252_stdout_print_a_non_cp1252_topic(env, monkeypatch):
    cfg_path, urls = env
    _main(["--config", cfg_path, "run", "--topic", NON_CP1252, "--input", urls, "--stages", "fetch"])
    out, err = _cp1252_streams(monkeypatch)
    assert _main(["--config", cfg_path, "notebook", "list"]) == 0
    assert NON_CP1252 in _text_of(out)


def test_console_reconfigure_skips_streams_without_reconfigure(monkeypatch):
    import io
    import sys
    from pipeline.cli import configure_console
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    configure_console()   # no AttributeError on a stream that cannot be reconfigured


def test_status_prints_quality_gate_numbers_before_report(env, capsys):
    cfg_path, urls = env
    assert _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls]) == 0
    capsys.readouterr()
    assert _main(["--config", cfg_path, "status", "t"]) == 0
    out = capsys.readouterr().out
    assert "Quality (latest analyses and merge):" in out
    assert "  claims: " in out and " points, " in out
    assert out.index("Quality (") < out.index("REPORT=")


def test_status_still_prints_report_when_a_stored_row_is_corrupt(env, capsys):
    cfg_path, urls = env
    assert _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls]) == 0
    import sqlite3
    from pipeline.config import load_config
    real_cfg = load_config(cfg_path)
    conn = sqlite3.connect(real_cfg.db_path)
    conn.execute("UPDATE sources SET analysis_json='{truncated'")
    conn.commit()
    conn.close()
    capsys.readouterr()
    assert _main(["--config", cfg_path, "status", "t"]) == 0
    out = capsys.readouterr().out
    assert "unreadable rows" in out and "REPORT=" in out


def test_init_defaults_to_the_5_5_models():
    from pipeline.cli import build_parser
    ns = build_parser().parse_args(["init"])
    assert (ns.analyze_model, ns.merge_model) == ("claude-haiku-5-5", "claude-sonnet-5-5")


def test_runs_print_the_cached_model_summary_once(env, capsys, tmp_path):
    from pipeline.config import load_config
    from pipeline.models_monitor import Finding, write_cache
    cfg_path, urls = env
    write_cache(load_config(cfg_path), [Finding("suggest", "merge", "x"), Finding("warn", None, "y"),
                                       Finding("note", None, "z")])
    _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls])
    out = capsys.readouterr().out
    assert out.count("models: 1 suggestion(s), 1 warning(s) -- see hmd doctor") == 1


def test_runs_print_nothing_without_a_fresh_cache(env, capsys, monkeypatch):
    monkeypatch.setattr("pipeline.models_monitor.check_models",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network on the run path")))
    cfg_path, urls = env
    _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls])
    assert "models:" not in capsys.readouterr().out


def test_runs_print_nothing_when_models_are_current(env, capsys):
    from pipeline.config import load_config
    from pipeline.models_monitor import Finding, write_cache
    cfg_path, urls = env
    write_cache(load_config(cfg_path), [Finding("ok", None, "current")])
    _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls])
    assert "models:" not in capsys.readouterr().out


_MODELS_CFG = ("# my config\nhome: data\n\nllm:\n  provider: anthropic_api   # keep\n  models:\n"
               "    analyze: claude-haiku-4-5-20251001   # cheap\n    merge: claude-sonnet-4-6\n"
               "  max_parallel: 4\n")


def test_models_use_changes_only_that_line(tmp_path, capsys):
    p = tmp_path / "config.yaml"
    p.write_bytes(_MODELS_CFG.replace("\n", "\r\n").encode())
    assert _main(["--config", str(p), "models", "use", "analyze", "claude-haiku-5-5"]) == 0
    assert p.read_bytes().decode() == _MODELS_CFG.replace(
        "analyze: claude-haiku-4-5-20251001", "analyze: claude-haiku-5-5").replace("\n", "\r\n")
    out = capsys.readouterr().out
    assert "analyze: claude-haiku-4-5-20251001 -> claude-haiku-5-5" in out


def test_models_use_rejects_bad_stage_and_id(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(_MODELS_CFG, encoding="utf-8")
    assert _main(["--config", str(p), "models", "use", "check", "claude-haiku-5-5"]) == 4
    assert _main(["--config", str(p), "models", "use", "merge", "gpt-9; rm -rf"]) == 4
    assert p.read_text(encoding="utf-8") == _MODELS_CFG


def test_models_use_warns_for_an_unpriced_model_on_the_api(tmp_path, capsys):
    p = tmp_path / "config.yaml"
    p.write_text(_MODELS_CFG, encoding="utf-8")
    assert _main(["--config", str(p), "models", "use", "merge", "claude-sonnet-9-9"]) == 0
    assert "no price" in capsys.readouterr().out


def test_models_use_without_a_models_block_or_config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("home: data\n", encoding="utf-8")
    assert _main(["--config", str(p), "models", "use", "merge", "claude-sonnet-5-5"]) == 4
    assert _main(["--config", str(tmp_path / "missing.yaml"), "models", "use", "merge", "claude-sonnet-5-5"]) == 4


from pipeline.cli import _runner, build_parser
from pipeline.config import Config, ConfigError


def _capture_make_provider(monkeypatch):
    seen = {}

    def fake(cfg, name=None, **kw):
        seen["name"], seen["batch"] = name, kw.get("batch")
        return ScriptedProvider()

    monkeypatch.setattr("pipeline.cli.make_provider", fake)
    return seen


def test_batch_flag_reaches_make_provider(tmp_path, monkeypatch):
    seen = _capture_make_provider(monkeypatch)
    _runner(Config(home=str(tmp_path)), build_parser().parse_args(["analyze", "x", "--batch"]))
    assert seen["batch"] is True


def test_batch_config_applies_without_a_flag_and_no_batch_overrides_it(tmp_path, monkeypatch):
    seen = _capture_make_provider(monkeypatch)
    cfg = Config(home=str(tmp_path), llm={"batch": {"enabled": True}})
    _runner(cfg, build_parser().parse_args(["analyze", "x"]))
    assert seen["batch"] is True
    _runner(cfg, build_parser().parse_args(["analyze", "x", "--no-batch"]))
    assert seen["batch"] is False


def test_batch_config_is_ignored_for_claude_cli(tmp_path, monkeypatch):
    seen = _capture_make_provider(monkeypatch)
    cfg = Config(home=str(tmp_path), llm={"batch": {"enabled": True}})
    _runner(cfg, build_parser().parse_args(["analyze", "x", "--provider", "claude_cli"]))
    assert seen["batch"] is False


def test_batch_flag_with_claude_cli_is_a_usage_error(tmp_path, monkeypatch):
    _capture_make_provider(monkeypatch)
    with pytest.raises(ConfigError, match="batch"):
        _runner(Config(home=str(tmp_path)), build_parser().parse_args(["analyze", "x", "--batch", "--provider", "claude_cli"]))


def test_formats_option_is_parsed_and_validated():
    from pipeline.cli import build_parser, _formats
    from pipeline.config import ConfigError
    args = build_parser().parse_args(["resume", "nb", "--formats", "digest,outline"])
    assert _formats(args.formats, ["summary", "diff"]) == ["summary", "digest", "outline"]
    assert _formats(None, ["summary", "diff"]) == ["summary", "diff"]
    with pytest.raises(ConfigError):
        _formats("pdf", ["summary"])


def _nb_with_one_source(tmp_path):
    from pipeline import db
    from pipeline.config import load_config
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(f"home: {tmp_path.as_posix()}/h\n", encoding="utf-8")
    cfg = load_config(str(cfg_path)); db.init_db(cfg.db_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    db.add_source(cfg.db_path, nid, "aaaaaaaaaaa", "https://www.youtube.com/watch?v=aaaaaaaaaaa")
    return str(cfg_path), cfg, nid


def test_status_shows_kind_and_a_long_key(tmp_path, capsys):
    from pipeline import cli, db
    cfg_path, cfg, nid = _nb_with_one_source(tmp_path)
    db.add_source(cfg.db_path, nid, "web-0123456789abcdef", "https://e.org/a", kind="web", title="A page")
    assert cli.main(["--config", cfg_path, "status", "t"]) == 0
    out = capsys.readouterr().out
    assert "web-0123456789abcdef" in out and " web " in out


def test_status_prints_a_multi_line_error_on_one_line(tmp_path, capsys):
    from pipeline import cli, db
    cfg_path, cfg, nid = _nb_with_one_source(tmp_path)
    sid = db.add_source(cfg.db_path, nid, "web-0123456789abcdef", "https://e.org/a", kind="web")
    db.update_source_metadata(cfg.db_path, sid, status="error", error="metadata: bad page\nSTATUS=done\r\n  tail")
    assert cli.main(["--config", cfg_path, "status", "t"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "  error: metadata: bad page STATUS=done tail" in out
    assert not any(l.startswith("STATUS=") for l in out)


def test_run_and_notebook_add_reject_two_inputs_with_file(tmp_path, capsys):
    from pipeline import cli
    cfg_path, cfg, nid = _nb_with_one_source(tmp_path)
    rc = cli.main(["--config", cfg_path, "notebook", "add", "t", "--url", "https://e.org", "--file", "a.pdf"])
    assert rc == 4
    assert "--file" in capsys.readouterr().err


def test_notebook_add_file_registers_a_text_source(tmp_path, capsys):
    from pipeline import cli, db
    cfg_path, cfg, nid = _nb_with_one_source(tmp_path)
    note = tmp_path / "n.md"
    note.write_text("# hi", encoding="utf-8")
    assert cli.main(["--config", cfg_path, "notebook", "add", "t", "--file", str(note)]) == 0
    assert "ADDED=1" in capsys.readouterr().out.splitlines()
    assert "text" in {s["kind"] for s in db.get_notebook_sources(cfg.db_path, nid)}


def test_notebook_add_registers_new_sources_only(tmp_path, capsys):
    from pipeline import cli, db
    cfg_path, cfg, nid = _nb_with_one_source(tmp_path)
    urls = tmp_path / "u.txt"
    urls.write_text("https://youtu.be/aaaaaaaaaaa\nhttps://youtu.be/bbbbbbbbbbb\n", encoding="utf-8")
    assert cli.main(["--config", cfg_path, "notebook", "add", "t", "--input", str(urls)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "NOTEBOOK=t" in out and "ADDED=1" in out and "ALREADY=1" in out
    assert "Next: hmd resume t" in out
    assert {s["video_id"] for s in db.get_notebook_sources(cfg.db_path, nid)} == {"aaaaaaaaaaa", "bbbbbbbbbbb"}


def test_notebook_add_reports_already_present(tmp_path, capsys):
    from pipeline import cli, db
    cfg_path, cfg, nid = _nb_with_one_source(tmp_path)
    assert cli.main(["--config", cfg_path, "notebook", "add", "t", "--url", "https://youtu.be/aaaaaaaaaaa"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "ADDED=0" in out and "ALREADY=1" in out and "Nothing new to process." in out
    assert len(db.get_notebook_sources(cfg.db_path, nid)) == 1


def test_notebook_add_with_no_usable_url_is_a_usage_error(tmp_path, capsys):
    from pipeline import cli
    cfg_path, cfg, nid = _nb_with_one_source(tmp_path)
    assert cli.main(["--config", cfg_path, "notebook", "add", "t", "--url", "not a url"]) == 4
    captured = capsys.readouterr()
    assert captured.err.startswith("hmd:") and "ADDED=" not in captured.out


def test_dry_run_prints_estimate_lines_and_never_builds_a_provider(tmp_path, capsys, monkeypatch):
    from pipeline import cli, db
    def no_provider(*a, **k):
        raise AssertionError("dry-run must not build a provider")
    monkeypatch.setattr(cli, "make_provider", no_provider)
    cfg_path, cfg, nid = _nb_with_one_source(tmp_path)
    sid = db.get_notebook_sources(cfg.db_path, nid)[0]["id"]
    db.update_source_metadata(cfg.db_path, sid, duration_seconds=600)
    assert cli.main(["--config", cfg_path, "dry-run", "t"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "NOTEBOOK=t" in out
    usd = next(l for l in out if l.startswith("ESTIMATE_USD="))
    assert float(usd.split("=")[1]) > 0
    assert any(l.startswith("ESTIMATE_RANGE=") for l in out) and any(l.startswith("ESTIMATE_MINUTES=") for l in out)
    assert any("estimated from the video length" in l for l in out)
    assert not any("size unknown" in l for l in out)


def test_dry_run_force_counts_analyzed_sources_as_to_analyze(tmp_path, capsys):
    """`hmd dry-run --force` estimates what `hmd analyze --force` would bill: every source re-analyzed."""
    from pipeline import cli, db
    cfg_path, cfg, nid = _nb_with_one_source(tmp_path)
    sid = db.get_notebook_sources(cfg.db_path, nid)[0]["id"]
    db.update_source_metadata(cfg.db_path, sid, duration_seconds=600)
    db.update_analyze_status(cfg.db_path, sid, "done", analysis_json='{"summary": "s", "key_claims": []}')
    assert cli.main(["--config", cfg_path, "dry-run", "t"]) == 0
    assert any("0 of 1 sources to analyze" in l for l in capsys.readouterr().out.splitlines())
    assert cli.main(["--config", cfg_path, "dry-run", "t", "--force"]) == 0
    assert any("1 of 1 sources to analyze" in l for l in capsys.readouterr().out.splitlines())


def test_dry_run_with_an_unpriced_model_is_a_config_error(tmp_path, capsys):
    """No price, no estimate: exit 4 naming the model (as a budgeted run refuses it), never $0.00."""
    from pipeline import cli
    cfg_path, cfg, nid = _nb_with_one_source(tmp_path)
    assert cli.main(["--config", cfg_path, "dry-run", "t", "--merge-model", "claude-nonesuch-9"]) == 4
    captured = capsys.readouterr()
    assert "claude-nonesuch-9" in captured.err and "pricing.py" in captured.err
    assert "ESTIMATE_USD=" not in captured.out


def test_dry_run_names_sources_of_unknown_size_instead_of_the_video_length(tmp_path, capsys):
    from pipeline import cli, db
    cfg_path, cfg, nid = _nb_with_one_source(tmp_path)          # one YouTube source, no duration
    db.add_source(cfg.db_path, nid, "web-0000000000000001", "https://e.org/a", kind="web")
    db.add_source(cfg.db_path, nid, "media-0000000000000002", (tmp_path / "talk.mp3").as_uri(), kind="media")
    assert cli.main(["--config", cfg_path, "dry-run", "t"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert ("  size unknown for 2 sources; the estimate leaves them out, so it understates the merge too"
            in out)
    assert not any("video length" in l for l in out)
    assert float(next(l for l in out if l.startswith("ESTIMATE_USD=")).split("=")[1]) > 0   # the web page


def _nb_with_stored_merge(tmp_path, monkeypatch, extra_cfg=""):
    """A notebook with two analyzed sources and a stored merge (as tests/test_output.py's `state`),
    a config on the default anthropic_api provider with a budget, and no API key anywhere."""
    from pipeline import envfile
    from pipeline.config import load_config
    from tests.test_output import DOC, _analysis
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(envfile, "project_dotenv_path", lambda: "")
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(f"home: {tmp_path.as_posix()}/h\nrun:\n  budget_usd: 5\n{extra_cfg}", encoding="utf-8")
    cfg = load_config(str(cfg_path)); db.init_db(cfg.db_path)
    nid = db.create_notebook(cfg.db_path, "AI Research", "ai-research")
    for vid, t0 in (("aaaaaaaaaaa", 754.0), ("bbbbbbbbbbb", 30.0)):
        sid = db.add_source(cfg.db_path, nid, vid, f"https://www.youtube.com/watch?v={vid}", title="V " + vid)
        db.update_analyze_status(cfg.db_path, sid, "done", analysis_json=json.dumps(_analysis(t0)))
    db.create_merge(cfg.db_path, nid, None, "m", json.dumps(DOC), json.dumps(DOC["source_map"]), "{}", None)
    return str(cfg_path), cfg, nid


def test_resume_output_only_needs_no_api_key(tmp_path, monkeypatch, capsys):
    """`hmd resume <slug> --stages output` is free: no key, no provider client, and the report is written."""
    from pipeline import cli
    cfg_path, cfg, nid = _nb_with_stored_merge(tmp_path, monkeypatch)
    assert cli.main(["--config", cfg_path, "resume", "ai-research", "--stages", "output"]) == 0
    out = capsys.readouterr().out.splitlines()
    report = next(l for l in out if l.startswith("REPORT="))[len("REPORT="):]
    assert os.path.isfile(report) and "STATUS=done" in out and "COST=0.0000" in out
    run = db.latest_run(cfg.db_path, nid)
    assert run["status"] == "done" and run["provider"] == "anthropic_api"


def test_output_only_skips_the_batch_check_and_the_price_gate(tmp_path, monkeypatch, capsys):
    from pipeline import cli
    cfg_path, cfg, nid = _nb_with_stored_merge(tmp_path, monkeypatch)
    assert cli.main(["--config", cfg_path, "resume", "ai-research", "--stages", "output",
                     "--provider", "claude_cli", "--batch"]) == 0
    assert cli.main(["--config", cfg_path, "resume", "ai-research", "--stages", "output",
                     "--analyze-model", "no-such-model"]) == 0
    assert cli.main(["--config", cfg_path, "resume", "ai-research", "--stages", "fetch,transcribe,output",
                     "--batch"]) == 0


def test_offline_provider_refuses_any_call():
    from pipeline.llm.base import LLMError
    from pipeline.llm.factory import OfflineProvider
    p = OfflineProvider("anthropic_api")
    assert p.name == "anthropic_api" and not getattr(p, "batching", False)
    with pytest.raises(LLMError, match="no LLM stage"):
        p.complete("x", model="m")


def test_runner_with_offline_provider_refuses_llm_stages(tmp_path, monkeypatch):
    from pipeline.llm.factory import OfflineProvider
    from pipeline.runner import Runner
    cfg_path, cfg, nid = _nb_with_stored_merge(tmp_path, monkeypatch)
    runner = Runner(cfg, provider=OfflineProvider("anthropic_api"), analyze_model="no-such-model")
    with pytest.raises(ValueError, match="analyze"):
        runner.validate(nid, ["analyze", "merge", "output"])


def test_dry_run_batch_with_claude_cli_is_a_usage_error(tmp_path, capsys):
    from pipeline import cli
    cfg_path, cfg, nid = _nb_with_one_source(tmp_path)
    assert cli.main(["--config", cfg_path, "dry-run", "t", "--provider", "claude_cli", "--batch"]) == 4
    captured = capsys.readouterr()
    assert "batch" in captured.err and "ESTIMATE_USD=" not in captured.out


def test_keep_home_leaves_notebook_for_rendering(env, golden_root, tmp_path, capsys):
    from pipeline.evals import run_golden
    cfg_path, urls = env
    gdir = _make_cli_golden(golden_root)
    run_golden(str(gdir), provider_name="anthropic_api", record=True, home=str(golden_root / "rechome"))
    keep = tmp_path / "kept"
    code = _main(["--config", cfg_path, "eval", "synthetic", "--provider", "anthropic_api",
                  "--keep-home", str(keep)])
    lines = capsys.readouterr().out.splitlines()
    assert code == 0 and "STATUS=pass" in lines and "COST=0.0000" in lines
    assert f"HOME={keep}" in lines
    assert (keep / "domains" / "general" / "state.sqlite").is_file()
    assert list((keep / "domains" / "general" / "vault" / "synthetic-topic" / "reports").glob("*-summary.md"))


def test_keep_home_needs_one_golden_and_one_provider(env, golden_root, tmp_path, capsys):
    cfg_path, urls = env
    _make_cli_golden(golden_root)
    assert _main(["--config", cfg_path, "eval", "synthetic", "--keep-home", str(tmp_path / "k")]) == 4
    assert _main(["--config", cfg_path, "eval", "--provider", "anthropic_api",
                  "--keep-home", str(tmp_path / "k")]) == 4
    assert _main(["--config", cfg_path, "eval", "synthetic", "--provider", "anthropic_api", "--live",
                  "--budget", "1", "--keep-home", str(tmp_path / "k")]) == 4
    assert "--keep-home" in capsys.readouterr().err


def test_html_flags_parse_and_override_config(env, monkeypatch):
    from pipeline.cli import build_parser
    from pipeline.config import Config
    from pipeline.runner import Runner
    parse = build_parser().parse_args
    assert parse(["run", "--topic", "t", "--url", "u"]).html is None
    assert parse(["run", "--topic", "t", "--url", "u", "--html"]).html is True
    assert parse(["resume", "x", "--no-html"]).html is False
    assert parse(["analyze", "x", "--no-html"]).html is False
    on, off = Config(output={"html": True}), Config(output={"html": False})
    assert Runner(on, provider=ScriptedProvider(), html=False).html is False
    assert Runner(off, provider=ScriptedProvider(), html=True).html is True
    assert Runner(on, provider=ScriptedProvider()).html is True


def _lines(out):
    return out.splitlines()


def test_run_prints_report_html_after_report(env, capsys):
    cfg_path, urls = env
    assert _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls]) == 0
    lines = _lines(capsys.readouterr().out)
    i = next(k for k, l in enumerate(lines) if l.startswith("REPORT="))
    assert lines[i + 1].startswith("REPORT_HTML=") and lines[i + 1].endswith("-summary.html")
    assert os.path.exists(lines[i + 1].split("=", 1)[1])


def test_no_html_prints_no_report_html(env, capsys):
    cfg_path, urls = env
    assert _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls, "--no-html"]) == 0
    assert not any(l.startswith("REPORT_HTML=") for l in _lines(capsys.readouterr().out))


def test_same_day_rerender_without_html_prints_no_report_html(env, capsys):
    cfg_path, urls = env
    assert _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls]) == 0
    capsys.readouterr()
    assert _main(["--config", cfg_path, "resume", "t", "--stages", "output", "--no-html"]) == 0
    assert not any(l.startswith("REPORT_HTML=") for l in _lines(capsys.readouterr().out))


def test_status_skips_an_html_twin_older_than_its_report(env, capsys):
    cfg_path, urls = env
    assert _main(["--config", cfg_path, "run", "--topic", "T", "--input", urls]) == 0
    out = capsys.readouterr().out
    md = next(l for l in out.splitlines() if l.startswith("REPORT=")).split("=", 1)[1]
    html_path = md[:-3] + ".html"
    assert _main(["--config", cfg_path, "status", "t"]) == 0
    assert f"REPORT_HTML={html_path}" in capsys.readouterr().out.splitlines()
    st = os.stat(md)
    os.utime(html_path, (st.st_atime, st.st_mtime - 60))     # a same-day --no-html re-render left it behind
    assert _main(["--config", cfg_path, "status", "t"]) == 0
    assert not any(l.startswith("REPORT_HTML=") for l in capsys.readouterr().out.splitlines())


def test_implicit_cwd_config_does_not_lift_the_private_host_check(tmp_path, monkeypatch, capsys):
    from pipeline.transcription import web_text
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text("transcription:\n  allow_private_hosts: true\n",
                                                     encoding="utf-8")
    monkeypatch.setattr(web_text, "_allow_private_hosts", False)
    code = _main(["status"])
    assert code == 0
    assert web_text._allow_private_hosts is False
    assert "allow_private_hosts" in capsys.readouterr().err


def test_run_refuses_an_input_list_naming_a_file_outside_its_folder(env, tmp_path, capsys):
    cfg_path, _ = env
    (tmp_path / "secret.txt").write_text("private", encoding="utf-8")
    (tmp_path / "lists").mkdir()
    lst = tmp_path / "lists" / "in.txt"
    lst.write_text("https://youtu.be/aaaaaaaaaaa\n../secret.txt\n", encoding="utf-8")
    code = _main(["--config", cfg_path, "run", "--topic", "T", "--input", str(lst)])
    captured = capsys.readouterr()
    assert code == 4
    assert "NOTEBOOK=" not in captured.out                 # refused before the run starts
    assert "outside the input list's folder" in captured.err and "--file" in captured.err


def test_usage_error_text_is_printed_without_control_characters(tmp_path, capsys):
    missing = str(tmp_path / "evil\x1b[2J.yaml")
    assert _main(["--config", missing, "status"]) == 4
    err = capsys.readouterr().err
    assert "\x1b" not in err and "evil?[2J.yaml" in err


def test_finish_prints_the_run_message_without_control_characters(capsys):
    from types import SimpleNamespace
    from pipeline.cli import _finish
    result = SimpleNamespace(status="error", message="stage failed: \x1b]8;;http://x\x07click", fatal=False,
                             report_path=None, report_html_path=None, cost_usd=0.0, run_id=1)
    _finish(result, "slug")
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x07" not in out


def test_status_prints_a_stored_remote_error_without_control_characters(env, capsys):
    from pipeline import db
    from pipeline.config import load_config
    cfg_path, _ = env
    cfg = load_config(cfg_path)
    db.init_db(cfg.db_path)
    nb = db.create_notebook(cfg.db_path, "T", "t")
    sid = db.add_source(cfg.db_path, nb, "web-0123456789abcdef", "https://example.org/", kind="web")
    db.update_source_metadata(cfg.db_path, sid, status="error", error="HTTP 500: \x1b[2Jgone")
    assert _main(["--config", cfg_path, "status", "t"]) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out and "?[2Jgone" in out
