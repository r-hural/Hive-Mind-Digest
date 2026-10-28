import json, os, shutil, subprocess
import pytest
from pipeline.llm.base import LLMError, LLMFatalError, LLMTransientError
from pipeline.llm.claude_cli import ClaudeCliProvider, build_argv, FIXED_ARGS, _child_env, which_on_path


@pytest.fixture(autouse=True)
def _claude_on_path(monkeypatch):
    # the provider refuses a claude it cannot find on PATH; CI has none installed
    monkeypatch.setattr(shutil, "which", lambda n: "C:/x/claude.exe")


def test_claude_not_on_path_is_refused_not_run_by_bare_name(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda n: None)
    with pytest.raises(LLMError, match="not found on PATH"):
        ClaudeCliProvider(binary="claude")


def test_claude_cli_argv_is_sandboxed():
    """Exit criterion (DECISION-003 amendment): tools off, restricted, no MCP, no session, model only."""
    argv = build_argv("claude", "claude-sonnet-4-6")
    assert argv[0] == "claude" and argv[1] == "-p"
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--tools") + 1] == ""
    for flag in ("--restricted", "--safe-mode", "--disable-slash-commands",
                 "--strict-mcp-config", "--no-session-persistence"):
        assert flag in argv
    assert argv[argv.index("--permission-prompts") + 1] == "none"
    assert argv[argv.index("--model") + 1] == "claude-sonnet-4-6"
    forbidden = {"--dangerously-skip-permissions", "--add-dir", "--mcp-config", "--bare",
                 "--allowedTools", "--permission-mode", "--settings", "--plugin-dir", "--agents"}
    assert not forbidden & set(argv)
    assert argv[1:1 + len(FIXED_ARGS)] == FIXED_ARGS


def test_system_prompt_is_passed_as_flag():
    argv = build_argv("claude", "claude-sonnet-4-6", system="rules")
    assert argv[argv.index("--system-prompt") + 1] == "rules"


def test_no_system_prompt_flag_when_system_is_none():
    argv = build_argv("claude", "claude-sonnet-4-6")
    assert "--system-prompt" not in argv


def test_system_prompt_leading_dash_is_rejected():
    with pytest.raises(LLMError):
        build_argv("claude", "claude-sonnet-4-6", system="-x")


def test_system_prompt_too_long_is_rejected():
    with pytest.raises(LLMError):
        build_argv("claude", "claude-sonnet-4-6", system="a" * 8001)


def test_model_id_is_validated_before_argv():
    with pytest.raises(LLMError):
        build_argv("claude", "sonnet; rm -rf /")
    with pytest.raises(LLMError):
        build_argv("claude", "--dangerously-skip-permissions")


def test_child_env_drops_anthropic_and_claude_vars_but_keeps_config_dir(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_MESSAGING_TOKEN", "t")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "a")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "X")
    monkeypatch.setenv("UNRELATED", "1")
    env = _child_env()
    assert "CLAUDE_CODE_MESSAGING_TOKEN" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "CLAUDECODE" not in env
    assert env["CLAUDE_CONFIG_DIR"] == "X"
    assert env["UNRELATED"] == "1"
    assert env


def test_child_env_drops_the_supadata_key(monkeypatch):
    monkeypatch.setenv("SUPADATA_API_KEY", "sd-test-key")
    assert "SUPADATA_API_KEY" not in _child_env()


def test_which_on_path_skips_a_binary_found_only_in_the_current_directory(monkeypatch, tmp_path):
    # Windows' shutil.which looks in the CWD first and reports that hit as a relative path
    real = tmp_path / "bin"
    real.mkdir()
    exe = real / ("claude.exe" if os.name == "nt" else "claude")
    exe.write_bytes(b"")
    exe.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda n: os.path.join(".", "claude.EXE"))
    monkeypatch.setenv("PATH", os.pathsep.join([".", "", str(real)]))
    monkeypatch.setenv("PATHEXT", ".COM;.EXE")
    assert os.path.normcase(which_on_path("claude")) == os.path.normcase(str(exe))
    monkeypatch.setenv("PATH", ".")
    assert which_on_path("claude") is None


def test_which_on_path_keeps_an_absolute_hit(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda n: "C:/x/claude.exe")
    assert which_on_path("claude") == "C:/x/claude.exe"


def test_cmd_shim_binary_is_rejected(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda n: "C:/x/claude.cmd")
    with pytest.raises(LLMError):
        ClaudeCliProvider(binary="claude")


def test_bat_shim_binary_is_rejected_case_insensitively(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda n: "C:/x/CLAUDE.BAT")
    with pytest.raises(LLMError):
        ClaudeCliProvider(binary="claude")


def test_prompt_goes_to_stdin_in_empty_cwd_without_api_key(monkeypatch):
    captured = {}

    def fake_run(argv, **kw):
        captured["argv"] = argv
        captured.update(kw)
        captured["listing"] = os.listdir(kw["cwd"])
        out = {"type": "result", "is_error": False, "result": '{"a": 1}',
               "usage": {"input_tokens": 10, "output_tokens": 5}, "total_cost_usd": 0.001}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(out), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-must-not-reach-the-child")
    r = ClaudeCliProvider(binary="claude").complete("THE PROMPT", model="claude-sonnet-4-6", system="S")
    assert captured["input"] == "THE PROMPT"
    assert captured["shell"] is False
    assert "THE PROMPT" not in " ".join(captured["argv"])
    assert captured["listing"] == []
    assert os.path.basename(captured["cwd"]).startswith("hmd-cli-")
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert r.text == '{"a": 1}' and r.cost_usd == 0.001 and r.input_tokens == 10 and r.provider == "claude_cli"


def test_nonzero_exit_with_rate_limit_text_is_transient(monkeypatch):
    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="API Error: rate limit exceeded, retrying")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LLMTransientError):
        ClaudeCliProvider(binary="claude").complete("p", model="claude-sonnet-4-6")


@pytest.mark.parametrize("text", [
    "You have hit your usage limit, try again at 3pm",       # was transient before 2026-10-09
    "Claude AI usage limit reached|1760018400",
    "You've hit your limit · resets 3pm (Europe/Kyiv)",
    "Not logged in · Please run /login",
    "Invalid API key · Please run /login",
    "OAuth token has expired. Please obtain a new token or refresh your existing token.",
    "Credit balance is too low",
])
@pytest.mark.parametrize("where", ["stderr", "is_error"])
def test_auth_and_plan_limit_messages_are_fatal(monkeypatch, text, where):
    def fake_run(argv, **kw):
        if where == "stderr":
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr=text)
        return subprocess.CompletedProcess(argv, 1 if "OAuth" in text else 0,
                                           stdout=json.dumps({"is_error": True, "result": text}), stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LLMFatalError) as ei:
        ClaudeCliProvider(binary="claude").complete("p", model="claude-sonnet-4-6")
    assert not isinstance(ei.value, LLMTransientError)
    assert text.split(" ")[0] in str(ei.value)


@pytest.mark.parametrize("text", ["overloaded_error", "API Error: 529 capacity", "rate_limit_error"])
def test_overload_and_rate_limit_are_transient_not_fatal(monkeypatch, text):
    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({"is_error": True, "result": text}), stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LLMTransientError) as ei:
        ClaudeCliProvider(binary="claude").complete("p", model="claude-sonnet-4-6")
    assert not isinstance(ei.value, LLMFatalError)


def test_a_model_answer_mentioning_login_is_not_fatal(monkeypatch):
    # The fatal markers are looked at only on a failed call, never in a successful result's text.
    def fake_run(argv, **kw):
        out = {"is_error": False, "result": '{"note": "Please run /login; usage limit reached"}', "usage": {}}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(out), stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    r = ClaudeCliProvider(binary="claude").complete("p", model="claude-sonnet-4-6")
    assert "/login" in r.text


def test_nonzero_exit_otherwise_is_error(monkeypatch):
    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(argv, 2, stdout="", stderr="unknown flag")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LLMError) as ei:
        ClaudeCliProvider(binary="claude").complete("p", model="claude-sonnet-4-6")
    assert not isinstance(ei.value, LLMTransientError)


def test_is_error_result_raises(monkeypatch):
    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({"is_error": True, "result": "boom"}), stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LLMError, match="boom"):
        ClaudeCliProvider(binary="claude").complete("p", model="claude-sonnet-4-6")


def test_timeout_is_transient(monkeypatch):
    def fake_run(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LLMTransientError):
        ClaudeCliProvider(binary="claude", timeout_seconds=1).complete("p", model="claude-sonnet-4-6")


def test_file_not_found_raises_llm_error(monkeypatch):
    def fake_run(argv, **kw):
        raise FileNotFoundError("no such file")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LLMError):
        ClaudeCliProvider(binary="claude").complete("p", model="claude-sonnet-4-6")


def test_non_dict_json_result_raises(monkeypatch):
    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps([1, 2, 3]), stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LLMError):
        ClaudeCliProvider(binary="claude").complete("p", model="claude-sonnet-4-6")


def test_error_subtype_raises(monkeypatch):
    def fake_run(argv, **kw):
        out = {"type": "result", "is_error": False, "subtype": "error_during_execution",
               "result": "", "usage": {}, "total_cost_usd": 0.0}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(out), stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LLMError, match="error_during_execution"):
        ClaudeCliProvider(binary="claude").complete("p", model="claude-sonnet-4-6")


def test_non_string_result_is_json_dumped(monkeypatch):
    def fake_run(argv, **kw):
        out = {"type": "result", "is_error": False, "result": {"a": 1}, "usage": {}}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(out), stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    r = ClaudeCliProvider(binary="claude").complete("p", model="claude-sonnet-4-6")
    assert r.text == json.dumps({"a": 1})


def test_non_json_stdout_error_includes_stderr(monkeypatch):
    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(argv, 0, stdout="not json", stderr="some stderr detail")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LLMError, match="some stderr detail"):
        ClaudeCliProvider(binary="claude").complete("p", model="claude-sonnet-4-6")


def test_cost_usd_is_none_when_missing(monkeypatch):
    def fake_run(argv, **kw):
        out = {"type": "result", "is_error": False, "result": "ok", "usage": {}}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(out), stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    r = ClaudeCliProvider(binary="claude").complete("p", model="claude-sonnet-4-6")
    assert r.cost_usd is None


def test_cache_tokens_are_mapped(monkeypatch):
    def fake_run(argv, **kw):
        out = {"type": "result", "is_error": False, "result": "ok",
               "usage": {"input_tokens": 1, "output_tokens": 2,
                         "cache_read_input_tokens": 30, "cache_creation_input_tokens": 40}}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(out), stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    r = ClaudeCliProvider(binary="claude").complete("p", model="claude-sonnet-4-6")
    assert r.cache_read_tokens == 30 and r.cache_write_tokens == 40


def test_transient_markers_no_longer_include_bare_429_or_try_again(monkeypatch):
    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="error 429, please try again")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LLMError) as ei:
        ClaudeCliProvider(binary="claude").complete("p", model="claude-sonnet-4-6")
    assert not isinstance(ei.value, LLMTransientError)
