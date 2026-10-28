from types import SimpleNamespace
from unittest.mock import MagicMock
import anthropic
import httpx
import pytest
from pipeline.llm.base import LLMError, LLMFatalError, LLMProvider, LLMTransientError
from pipeline.llm.anthropic_api import AnthropicApiProvider, STREAM_THRESHOLD


def _msg(text="ok", stop="end_turn", model="claude-haiku-4-5-20251001", cache_read=0):
    usage = SimpleNamespace(input_tokens=11, output_tokens=7, cache_read_input_tokens=cache_read,
                            cache_creation_input_tokens=0)
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop,
                           model=model, usage=usage)


def _provider(create_return=None, stream_return=None):
    client = MagicMock()
    client.messages.create.return_value = create_return or _msg()
    stream_cm = MagicMock()
    stream_cm.__enter__.return_value.get_final_message.return_value = stream_return or _msg()
    stream_cm.__exit__.return_value = False
    client.messages.stream.return_value = stream_cm
    return AnthropicApiProvider(client=client), client


def test_is_llm_provider():
    p, _ = _provider()
    assert isinstance(p, LLMProvider) and p.name == "anthropic_api"


def test_small_max_tokens_uses_create_and_maps_usage():
    p, client = _provider(create_return=_msg(cache_read=3))
    r = p.complete("hi", model="claude-haiku-4-5-20251001", system="sys", max_tokens=1000)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["system"] == "sys" and kwargs["max_tokens"] == 1000
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert r.text == "ok" and r.input_tokens == 11 and r.output_tokens == 7 and r.cache_read_tokens == 3
    assert r.stop_reason == "end_turn" and r.provider == "anthropic_api" and r.cost_usd is None
    client.messages.stream.assert_not_called()


def test_large_max_tokens_streams():
    p, client = _provider()
    p.complete("hi", model="m", max_tokens=STREAM_THRESHOLD + 1)
    client.messages.stream.assert_called_once()
    client.messages.create.assert_not_called()


def test_stop_reason_max_tokens_is_reported_not_hidden():
    p, _ = _provider(create_return=_msg(stop="max_tokens"))
    assert p.complete("hi", model="m", max_tokens=10).stop_reason == "max_tokens"


def _status_error(status):
    resp = httpx.Response(status, request=httpx.Request("POST", "https://x"))
    return anthropic.APIStatusError("err", response=resp, body=None)


def test_rate_limit_is_transient():
    p, client = _provider()
    resp = httpx.Response(429, request=httpx.Request("POST", "https://x"))
    client.messages.create.side_effect = anthropic.RateLimitError("rl", response=resp, body=None)
    with pytest.raises(LLMTransientError):
        p.complete("hi", model="m", max_tokens=10)


@pytest.mark.parametrize("status", [500, 529, 408])
def test_server_errors_are_transient(status):
    p, client = _provider()
    client.messages.create.side_effect = _status_error(status)
    with pytest.raises(LLMTransientError):
        p.complete("hi", model="m", max_tokens=10)


def test_bad_request_is_not_transient():
    p, client = _provider()
    client.messages.create.side_effect = _status_error(400)
    with pytest.raises(LLMError) as ei:
        p.complete("hi", model="m", max_tokens=10)
    assert not isinstance(ei.value, LLMTransientError)


def test_connection_error_is_transient():
    p, client = _provider()
    client.messages.create.side_effect = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    with pytest.raises(LLMTransientError):
        p.complete("hi", model="m", max_tokens=10)


def test_refusal_stop_reason_raises():
    p, _ = _provider(create_return=_msg(stop="refusal"))
    with pytest.raises(LLMError) as ei:
        p.complete("hi", model="m", max_tokens=10)
    assert not isinstance(ei.value, LLMTransientError)
    assert "refusal" in str(ei.value)


def test_empty_content_non_max_tokens_raises():
    empty = _msg(stop="end_turn")
    empty.content = []
    p, _ = _provider(create_return=empty)
    with pytest.raises(LLMError) as ei:
        p.complete("hi", model="m", max_tokens=10)
    assert not isinstance(ei.value, LLMTransientError)
    assert "end_turn" in str(ei.value)


def test_api_response_validation_error_is_not_transient():
    p, client = _provider()
    resp = httpx.Response(200, request=httpx.Request("POST", "https://x"))
    client.messages.create.side_effect = anthropic.APIResponseValidationError(response=resp, body=None)
    with pytest.raises(LLMError) as ei:
        p.complete("hi", model="m", max_tokens=10)
    assert not isinstance(ei.value, LLMTransientError)


def test_bare_value_error_is_not_transient():
    p, client = _provider()
    client.messages.create.side_effect = ValueError("x")
    with pytest.raises(LLMError) as ei:
        p.complete("hi", model="m", max_tokens=10)
    assert not isinstance(ei.value, LLMTransientError)


def test_stream_threshold_boundary_streams_at_default():
    p, client = _provider()
    p.complete("hi", model="m", max_tokens=STREAM_THRESHOLD)
    client.messages.stream.assert_called_once()
    client.messages.create.assert_not_called()


def test_below_stream_threshold_uses_create():
    p, client = _provider()
    p.complete("hi", model="m", max_tokens=STREAM_THRESHOLD - 1)
    client.messages.create.assert_called_once()
    client.messages.stream.assert_not_called()


def test_rate_limit_carries_retry_after():
    p, client = _provider()
    resp = httpx.Response(429, request=httpx.Request("POST", "https://x"),
                          headers={"retry-after": "7"})
    client.messages.create.side_effect = anthropic.RateLimitError("rl", response=resp, body=None)
    with pytest.raises(LLMTransientError) as ei:
        p.complete("hi", model="m", max_tokens=10)
    assert ei.value.retry_after == 7.0


def test_real_sdk_message_maps_correctly():
    msg = anthropic.types.Message(
        id="msg_1", type="message", role="assistant", model="claude-haiku-4-5-20251001",
        content=[anthropic.types.TextBlock(type="text", text="hello")],
        stop_reason="end_turn", stop_sequence=None,
        usage=anthropic.types.Usage(input_tokens=1, output_tokens=1),
    )
    p, _ = _provider(create_return=msg)
    r = p.complete("hi", model="claude-haiku-4-5-20251001", max_tokens=10)
    assert r.text == "hello" and r.model == "claude-haiku-4-5-20251001"
    assert r.cache_read_tokens == 0 and r.cache_write_tokens == 0




# --- .env: the provider loads the project-root file only (lookup itself: tests/test_envfile.py) ---

def _nested_project(tmp_path, *, project_key=None):
    """A parent checkout with its own .env and a nested worktree-like project with pyproject.toml."""
    parent = tmp_path / "parent"
    project = parent / ".claude" / "worktrees" / "wt"
    project.mkdir(parents=True)
    (parent / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (parent / ".env").write_text("ANTHROPIC_API_KEY=sk-test-parent\n", encoding="utf-8")
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    if project_key is not None:
        (project / ".env").write_text(f"ANTHROPIC_API_KEY={project_key}\n", encoding="utf-8")
    return project


def _isolate_key(monkeypatch):
    # setenv then delenv so monkeypatch restores the "absent" state even after load_dotenv writes it.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    monkeypatch.delenv("ANTHROPIC_API_KEY")


def test_provider_never_loads_parent_checkout_dotenv(monkeypatch, tmp_path):
    import os
    from pipeline import envfile
    project = _nested_project(tmp_path)
    _isolate_key(monkeypatch)
    monkeypatch.setattr(envfile, "PROJECT_ROOT", project)
    monkeypatch.chdir(project)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY is not set"):
        AnthropicApiProvider()
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_provider_loads_project_root_dotenv(monkeypatch, tmp_path):
    from pipeline import envfile
    from pipeline.llm import anthropic_api
    project = _nested_project(tmp_path, project_key="sk-project")
    _isolate_key(monkeypatch)
    monkeypatch.setattr(envfile, "PROJECT_ROOT", project)
    seen = {}
    monkeypatch.setattr(anthropic_api.anthropic, "Anthropic", lambda **kw: seen.update(kw) or MagicMock())
    AnthropicApiProvider()
    assert seen["api_key"] == "sk-project"


def test_provider_dotenv_never_overrides_existing_environment(monkeypatch, tmp_path):
    from pipeline import envfile
    from pipeline.llm import anthropic_api
    project = _nested_project(tmp_path, project_key="sk-project")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-shell")
    monkeypatch.setattr(envfile, "PROJECT_ROOT", project)
    seen = {}
    monkeypatch.setattr(anthropic_api.anthropic, "Anthropic", lambda **kw: seen.update(kw) or MagicMock())
    AnthropicApiProvider()
    assert seen["api_key"] == "sk-shell"


# --- fatal errors: fail the whole run at once (2026-10-09, exhausted credit balance) ----------

CREDIT_BODY = {"type": "error", "error": {
    "type": "invalid_request_error",
    "message": "Your credit balance is too low to access the Anthropic API. "
               "Please go to Plans & Billing to upgrade or purchase credits."},
    "request_id": "req_011CTx0000000000000000"}


def _sdk_error(cls, status, body):
    """Build the exception exactly as the SDK does (_base_client._make_status_error_from_response)."""
    resp = httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"), json=body)
    return cls(f"Error code: {status} - {body}", response=resp, body=body)


def _raises(exc, *, stream=False):
    p, client = _provider()
    if stream:
        client.messages.stream.side_effect = exc
    else:
        client.messages.create.side_effect = exc
    with pytest.raises(LLMError) as ei:
        p.complete("hi", model="m", max_tokens=STREAM_THRESHOLD if stream else 10)
    return ei.value


@pytest.mark.parametrize("stream", [False, True])
def test_credit_balance_400_is_fatal_and_keeps_the_message(stream):
    err = _raises(_sdk_error(anthropic.BadRequestError, 400, CREDIT_BODY), stream=stream)
    assert isinstance(err, LLMFatalError) and not isinstance(err, LLMTransientError)
    assert "Your credit balance is too low to access the Anthropic API" in str(err)
    assert "400" in str(err)


@pytest.mark.parametrize("cls,status,etype", [
    (anthropic.AuthenticationError, 401, "authentication_error"),
    (anthropic.PermissionDeniedError, 403, "permission_error"),
])
def test_auth_and_permission_errors_are_fatal(cls, status, etype):
    body = {"type": "error", "error": {"type": etype, "message": "invalid x-api-key"}}
    err = _raises(_sdk_error(cls, status, body))
    assert isinstance(err, LLMFatalError) and "invalid x-api-key" in str(err) and str(status) in str(err)


def test_auth_error_without_a_body_is_still_fatal():
    err = _raises(_status_error(401))
    assert isinstance(err, LLMFatalError)


def test_billing_error_402_is_fatal():
    body = {"type": "error", "error": {"type": "billing_error", "message": "payment required"}}
    err = _raises(_sdk_error(anthropic.APIStatusError, 402, body))
    assert isinstance(err, LLMFatalError) and "payment required" in str(err)


def test_ordinary_bad_request_is_not_fatal():
    body = {"type": "error", "error": {"type": "invalid_request_error",
                                       "message": "max_tokens: 99999999 > 64000, the maximum allowed"}}
    err = _raises(_sdk_error(anthropic.BadRequestError, 400, body))
    assert not isinstance(err, (LLMFatalError, LLMTransientError))


@pytest.mark.parametrize("cls,status,etype", [
    (anthropic.RateLimitError, 429, "rate_limit_error"),
    (anthropic.OverloadedError, 529, "overloaded_error"),
])
def test_rate_limit_and_overload_stay_transient_not_fatal(cls, status, etype):
    body = {"type": "error", "error": {"type": etype, "message": "slow down"}}
    err = _raises(_sdk_error(cls, status, body))
    assert isinstance(err, LLMTransientError) and not isinstance(err, LLMFatalError)
