"""The LLM provider boundary (DECISION-003, DECISION-012).

Providers return raw text plus usage; they never parse JSON, retry or log. That is the
client's job (pipeline/llm/client.py), so both providers and the replay provider behave
identically above this line.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, wait
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None        # known to the provider (claude_cli); else priced by the client
    stop_reason: str = "end_turn"        # "max_tokens" means truncated: the client raises
    duration_ms: int = 0


class LLMError(Exception):
    """Non-retryable provider failure."""


class LLMTransientError(LLMError):
    """Retryable: 429, 5xx, overloaded, network, process timeout."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMFatalError(LLMError):
    """Fails every call of the run, so the run stops at once instead of trying each source:
    authentication/permission failure, exhausted credit balance or billing, a subscription usage
    limit that resets only hours later. Never retried, never schema-repaired; the runner stops
    scheduling new work and closes the run as an error; `hmd resume` continues it once fixed.
    The message is the provider's own, minus anything secret (providers never include a key)."""


class LLMTruncatedError(LLMError):
    """Response stopped at max_tokens (IDEA-211)."""


class LLMSchemaError(LLMError):
    """Response was not valid JSON for the requested schema (IDEA-230)."""


class LLMInterrupted(LLMError):
    """The call stopped because the user interrupted the run (Ctrl+C): a retry backoff cut short, a batch
    cancelled or never sent. Not the source's fault, so the analyze worker leaves it pending for
    `hmd resume`, like BudgetExceeded."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, prompt: str, *, model: str, system: str | None = None,
                 max_tokens: int = 16000) -> LLMResult: ...


# Set when the user interrupts a run (Ctrl+C). A provider that blocks for long (the batch provider waits
# minutes to hours for a batch) watches it: it stops waiting, cancels what it sent and fails its callers,
# so the pools can shut down. Runner.run clears it when a run starts.
_INTERRUPT = threading.Event()


def request_interrupt() -> None:
    _INTERRUPT.set()


def clear_interrupt() -> None:
    _INTERRUPT.clear()


def interrupt_requested() -> bool:
    return _INTERRUPT.is_set()


def wait_interrupt(timeout: float) -> bool:
    """Sleep up to `timeout` seconds; True as soon as an interrupt is requested."""
    return _INTERRUPT.wait(timeout)


def wait_futures(futures: Iterable[Future], poll: float = 0.5) -> None:
    """concurrent.futures.wait in short slices: an untimed wait does not see Ctrl+C on Windows until it
    returns, and a batch can keep it waiting for an hour."""
    pending = set(futures)
    while pending:
        _, pending = wait(pending, timeout=poll)
