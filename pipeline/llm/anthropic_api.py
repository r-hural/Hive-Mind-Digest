"""LLMProvider over the Anthropic SDK, billed per token to ANTHROPIC_API_KEY."""
from __future__ import annotations

import os
import time

import anthropic

from pipeline.envfile import load_project_dotenv
from pipeline.llm.base import LLMError, LLMFatalError, LLMResult, LLMTransientError

STREAM_THRESHOLD = 16000   # at or above this the SDK needs streaming to avoid HTTP timeouts
TRANSIENT_STATUSES = {408, 409, 429}
# Errors that fail every call of the run (LLMFatalError): 401 authentication, 402 billing,
# 403 permission -- by status or by the API's error.type -- and a 400 invalid_request_error whose
# message mentions the credit balance (that is how an exhausted prepaid balance is reported,
# seen live 2026-10-09). Any other 400 stays an ordinary per-call LLMError.
FATAL_STATUSES = {401, 402, 403}
FATAL_ERROR_TYPES = {"authentication_error", "billing_error", "permission_error"}
FATAL_MESSAGE_MARKERS = ("credit balance",)


def _error_detail(e: anthropic.APIStatusError) -> tuple[str, str]:
    """(error.type, error.message) from the API's JSON error body, or ("", "") when absent."""
    body = getattr(e, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        return "", ""
    return str(err.get("type") or ""), str(err.get("message") or "")


def _status_error(e: anthropic.APIStatusError) -> LLMError:
    status = e.status_code
    if status >= 500 or status in TRANSIENT_STATUSES:
        return LLMTransientError(f"API {status}: {e}")
    etype, emsg = _error_detail(e)
    text = (emsg or str(e)).lower()
    if (status in FATAL_STATUSES or etype in FATAL_ERROR_TYPES
            or (status == 400 and any(m in text for m in FATAL_MESSAGE_MARKERS))):
        detail = f"{etype}: {emsg}" if emsg else str(e)
        return LLMFatalError(f"API {status} {detail}")
    return LLMError(f"API {status}: {e}")


def message_result(msg, model: str, provider: str, duration_ms: int) -> LLMResult:
    """An SDK Message as an LLMResult; a refusal, or no text without truncation, is an LLMError.
    Shared by the direct provider and the batch provider (pipeline/llm/anthropic_batch.py)."""
    if msg.stop_reason == "refusal":
        raise LLMError(f"model {model} refused this request (stop_reason=refusal)")
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    if not text and msg.stop_reason != "max_tokens":
        raise LLMError(f"model {model} returned no text (stop_reason={msg.stop_reason})")
    u = msg.usage
    return LLMResult(
        text=text, model=getattr(msg, "model", model), provider=provider,
        input_tokens=int(u.input_tokens or 0), output_tokens=int(u.output_tokens or 0),
        cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        stop_reason=msg.stop_reason or "end_turn",
        duration_ms=duration_ms,
    )


class AnthropicApiProvider:
    name = "anthropic_api"

    def __init__(self, api_key: str | None = None, client: anthropic.Anthropic | None = None):
        if client is None:
            load_project_dotenv()
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError("ANTHROPIC_API_KEY is not set; add it to .env or choose llm.provider: claude_cli")
            client = anthropic.Anthropic(api_key=key, max_retries=0)   # the client retries, not the SDK
        self._client = client

    def complete(self, prompt: str, *, model: str, system: str | None = None,
                 max_tokens: int = 16000) -> LLMResult:
        kwargs: dict = dict(model=model, max_tokens=max_tokens,
                            messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        t0 = time.monotonic()
        try:
            if max_tokens >= STREAM_THRESHOLD:
                with self._client.messages.stream(**kwargs) as stream:
                    msg = stream.get_final_message()
            else:
                msg = self._client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after"))
            except (TypeError, ValueError, AttributeError):
                pass
            raise LLMTransientError(f"rate limited: {e}", retry_after=retry_after) from e
        except anthropic.APIStatusError as e:
            raise _status_error(e) from e
        except anthropic.APIConnectionError as e:   # includes APITimeoutError
            raise LLMTransientError(f"connection error: {e}") from e
        except anthropic.AnthropicError as e:
            raise LLMError(f"anthropic SDK error: {e}") from e
        except ValueError as e:   # the SDK raises a bare ValueError for non-streaming max_tokens above a model cap
            raise LLMError(f"invalid request: {e}") from e

        return message_result(msg, model, self.name, int((time.monotonic() - t0) * 1000))
