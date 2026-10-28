"""Build providers and clients from Config. The only place provider names are switched on."""
from __future__ import annotations

from pipeline.config import PROVIDERS, Config, ConfigError
from pipeline.llm.anthropic_api import AnthropicApiProvider
from pipeline.llm.anthropic_batch import AnthropicBatchProvider, BatchJournal
from pipeline.llm.base import LLMError, LLMProvider, LLMResult
from pipeline.llm.claude_cli import ClaudeCliProvider
from pipeline.llm.client import LLMClient
from pipeline.llm.ratelimit import RateLimiter
from pipeline.llm.replay import ReplayProvider


LLM_STAGES = ("analyze", "merge")


class OfflineProvider:
    """Stands in for the configured provider on a run with no LLM stage (`hmd resume <slug>
    --stages output` re-renders from the stored merge): it needs no key or binary and fails any
    call it is ever given. `name` is the configured provider's, so the run row reads as usual."""
    offline = True

    def __init__(self, name: str):
        self.name = name

    def complete(self, prompt: str, *, model: str, system: str | None = None,
                 max_tokens: int = 16000) -> LLMResult:
        raise LLMError(f"no LLM stage was requested, so this run has no {self.name} provider; "
                       "add analyze or merge to --stages")


def make_provider(cfg: Config, name: str | None = None, *, cassette_dir: str | None = None,
                  record: bool = False, batch: bool = False, log=print) -> LLMProvider:
    name = name or cfg.llm.provider
    if name not in PROVIDERS:
        raise ConfigError(f"unknown llm.provider {name!r}; use one of {PROVIDERS}")
    if record and not cassette_dir:
        raise ConfigError("record=True needs a cassette_dir")
    if batch and (name != "anthropic_api" or cassette_dir):
        raise ConfigError("batch mode needs the anthropic_api provider and no cassettes "
                          "(claude_cli has no batch API; hmd eval never batches)")
    if cassette_dir and not record:
        return ReplayProvider(cassette_dir)          # replay-only: no network, no key needed
    if batch:
        b = cfg.llm.batch
        return AnthropicBatchProvider(min_requests=b.min_requests, max_wave=b.max_wave,
                                      gather_seconds=b.gather_seconds, poll_seconds=b.poll_seconds,
                                      max_wait_seconds=b.max_wait_hours * 3600,
                                      journal=BatchJournal(cfg.batch_journal_path), log=log)
    inner: LLMProvider = AnthropicApiProvider() if name == "anthropic_api" else ClaudeCliProvider()
    if cassette_dir:
        return ReplayProvider(cassette_dir, inner=inner, record=True)
    return inner


def make_client(cfg: Config, provider: LLMProvider, *, db_path: str, run_id: int | None,
                notebook_id: int | None, budget_usd: float, log=print) -> LLMClient:
    replay_only = isinstance(provider, ReplayProvider) and provider.inner is None
    # a batch is one request per wave; the per-minute limits do not apply
    limits = None if replay_only or getattr(provider, "batching", False) else cfg.llm.rate_limits.get(provider.name)
    limiter = RateLimiter(limits.rpm, limits.tpm) if limits else None
    return LLMClient(provider, db_path=db_path, run_id=run_id, notebook_id=notebook_id, budget_usd=budget_usd,
                     rate_limiter=limiter, retry_attempts=cfg.llm.retry_attempts,
                     retry_base_delay=cfg.llm.retry_base_delay_seconds,
                     max_output_tokens=cfg.llm.max_output_tokens, log=log)
