"""Provider-agnostic LLM client: rate limit, retry with backoff, stop_reason check, JSON extraction,
schema validation with one corrective retry, usage log and per-run budget cap.

IDEA-075 (retry/backoff), IDEA-082 (usage log), IDEA-211 (truncation is an error),
IDEA-230 (schema validation; model output never re-enters the instruction region), IDEA-280 (budget).

Only LLMTransientError is retried and only LLMSchemaError gets the corrective round-trip; an
LLMFatalError (auth, billing, credit balance, subscription limit) propagates from the first call
untouched, so the runner can stop the whole run (pipeline/runner.py).

Budget semantics: `cost_usd` is the API-equivalent price of each call. For `claude_cli` (subscription)
it is nominal, not money spent (IDEA-305): the cost is still tracked and recorded, and crossing the cap
logs one warning line, but the run never pauses on it.

The budget check happens before each provider call, not atomically across it: with N workers sharing
one `LLMClient` (and one run's budget), the cap can be overshot by up to one call per worker, since
each worker checks `spent` before its own call rather than reserving budget ahead of time.
With the batch provider (pipeline/llm/anthropic_batch.py) a caller checks the budget before it joins a
wave, so the cap can be overshot by up to one wave's cost.
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
import threading
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from pipeline import db
from pipeline.llm.base import (LLMInterrupted, LLMProvider, LLMResult, LLMSchemaError, LLMTransientError,
                               LLMTruncatedError, interrupt_requested, wait_interrupt)
from pipeline.llm.ratelimit import RateLimiter
from pipeline.pricing import estimate_cost
from pipeline.prompts.common import SYSTEM_PROMPT, repair_prompt
from pipeline.schemas import strip_code_filled

T = TypeVar("T", bound=BaseModel)
# Any language tag (or none), case-insensitive; the fence is a preference, never the only branch.
FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*\s*(.*?)```", re.S | re.I)
_BRACE_RE = re.compile(r"[{\[]")
# Providers whose cost_usd is nominal (subscription, nothing billed per call): the cap warns, never pauses.
NOMINAL_COST_PROVIDERS = {"claude_cli"}


class BudgetExceeded(Exception):
    def __init__(self, spent: float, budget: float, *, batching: bool = False):
        overshoot = ("a batch already sent may overshoot the cap by up to one whole batch" if batching
                     else "in-flight calls may overshoot the cap by up to one call per worker")
        super().__init__(f"run budget reached: spent ${spent:.4f} of ${budget:.2f}; "
                         f"resume with `hmd resume <notebook>` (a new run gets a fresh budget) "
                         f"({overshoot})")
        self.spent = spent
        self.budget = budget


def extract_json(text: str) -> object:
    """Try every fenced code block in order (any language tag, or none) with a plain `json.loads`;
    if none of them parses, fall back to a raw scan that tries `JSONDecoder().raw_decode` at every
    '{' or '[' in the whole text, in order, until one succeeds. Raises LLMSchemaError only once both
    have failed (the message never echoes the response text)."""
    for fence in FENCE_RE.finditer(text):
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    for brace in _BRACE_RE.finditer(text):
        try:
            return json.JSONDecoder().raw_decode(text, brace.start())[0]
        except json.JSONDecodeError:
            continue
    raise LLMSchemaError(f"not JSON: no fenced block or JSON value found (response length {len(text)})")


def _format_validation_error(schema: type[BaseModel], e: ValidationError) -> str:
    parts = []
    for err in e.errors()[:5]:
        loc = ".".join(str(p) for p in err["loc"])
        msg = err["msg"][:200]
        parts.append(f"{loc}: {err['type']}: {msg}")
    return f"JSON does not match {schema.__name__}: " + "; ".join(parts)


class LLMClient:
    def __init__(self, provider: LLMProvider, *, db_path: str, run_id: int | None, notebook_id: int | None,
                 budget_usd: float = 0.0, rate_limiter: RateLimiter | None = None,
                 retry_attempts: int = 5, retry_base_delay: float = 2.0, max_output_tokens: int = 64000,
                 sleep: Callable[[float], object] = wait_interrupt, log: Callable[[str], None] = print):
        self._provider = provider
        self._db_path = db_path
        self._run_id = run_id
        self._notebook_id = notebook_id
        self._budget_usd = budget_usd
        self._rate_limiter = rate_limiter
        self._retry_attempts = max(1, retry_attempts)
        self._retry_base_delay = retry_base_delay
        self._max_output_tokens = max_output_tokens
        self._sleep = sleep
        self._emit = log
        self._lock = threading.Lock()
        self._warned_unpriced_models: set[str] = set()
        self._nominal_cost = getattr(provider, "name", "") in NOMINAL_COST_PROVIDERS
        self._warned_budget = False
        self.spent = db.run_cost(db_path, run_id) if run_id is not None else 0.0
        self.calls = 0             # provider calls made; a schema repair makes this 2 per call_json
        self.waited_seconds = 0.0  # time spent inside rate_limiter.acquire() only, not backoff sleeps

    def _check_budget(self) -> None:
        if self._budget_usd > 0 and self.spent >= self._budget_usd and not self._nominal_cost:
            raise BudgetExceeded(self.spent, self._budget_usd, batching=getattr(self._provider, "batching", False))

    def _check_truncation(self, result: LLMResult, stage: str, max_tokens: int) -> None:
        if result.stop_reason == "max_tokens":
            raise LLMTruncatedError(
                f"{stage}: response truncated at {max_tokens} output tokens (model {result.model}); "
                "raise llm.max_output_tokens or reduce the input of this call")

    def _record_usage(self, result: LLMResult, stage: str, source_id: int | None) -> None:
        cost = result.cost_usd
        if cost is None:
            cost = estimate_cost(result.model, result.input_tokens, result.output_tokens,
                                 result.cache_read_tokens, result.cache_write_tokens)
            if cost is None and self._budget_usd > 0 and result.model not in self._warned_unpriced_models:
                self._warned_unpriced_models.add(result.model)
                self._emit(f"warning: no price for model {result.model}; "
                          f"its cost is not counted against the budget")
        try:
            db.add_usage_event(
                self._db_path, run_id=self._run_id, notebook_id=self._notebook_id, source_id=source_id,
                stage=stage, provider=result.provider, model=result.model,
                input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                cache_read_tokens=result.cache_read_tokens, cache_write_tokens=result.cache_write_tokens,
                cost_usd=cost, duration_ms=result.duration_ms)
        except sqlite3.Error as e:
            self._emit(f"warning: usage event not recorded: {e}")
        with self._lock:
            self.spent += cost or 0.0
            self.calls += 1
            warn = (self._nominal_cost and self._budget_usd > 0 and self.spent >= self._budget_usd
                    and not self._warned_budget)
            if warn:
                self._warned_budget = True
        if warn:
            self._emit(f"warning: nominal cost ${self.spent:.2f} reached the ${self._budget_usd:.2f} budget; "
                       f"{self._provider.name} is billed by subscription, so the run continues")

    def _complete_with_retry(self, prompt: str, *, model: str, system: str | None,
                             max_tokens: int) -> LLMResult:
        estimated_tokens = len(prompt) // 4
        for attempt in range(self._retry_attempts):
            waited = self._rate_limiter.acquire(estimated_tokens) if self._rate_limiter is not None else 0.0
            with self._lock:
                self.waited_seconds += waited
            try:
                return self._provider.complete(prompt, model=model, system=system, max_tokens=max_tokens)
            except LLMTransientError as e:
                if attempt == self._retry_attempts - 1:
                    raise
                delay = min(self._retry_base_delay * 2 ** attempt + random.uniform(0, 1), 60)
                if e.retry_after is not None:
                    delay = max(delay, e.retry_after)
                    if self._rate_limiter is not None:
                        self._rate_limiter.penalize(e.retry_after)
                # wait_interrupt (the default) returns as soon as Ctrl+C raises the interrupt flag, so a
                # worker in a backoff of up to 60 s does not hold up the pool's shutdown
                self._sleep(delay)
                if interrupt_requested():
                    raise LLMInterrupted("interrupted during the retry backoff") from e

    def _parse(self, text: str, schema: type[T], stage: str) -> T:
        data = extract_json(text)
        stripped = strip_code_filled(data)
        if stripped != data:
            self._emit(f"{stage}: dropped code-filled keys fabricated by the model")
        data = stripped
        if isinstance(data, dict):
            ignored = set(data) - set(schema.model_fields)
            if ignored:
                self._emit(f"{stage}: ignored keys from model output: {sorted(ignored)}")
        try:
            return schema.model_validate(data)
        except ValidationError as e:
            raise LLMSchemaError(_format_validation_error(schema, e)) from None

    def call_json(self, prompt: str, schema: type[T], *, model: str, stage: str, source_id: int | None = None,
                  system: str | None = SYSTEM_PROMPT, max_tokens: int | None = None) -> T:
        max_tokens = max_tokens if max_tokens is not None else self._max_output_tokens
        self._check_budget()
        result = self._complete_with_retry(prompt, model=model, system=system, max_tokens=max_tokens)
        self._record_usage(result, stage, source_id)
        self._check_truncation(result, stage, max_tokens)
        try:
            return self._parse(result.text, schema, stage)
        except LLMSchemaError as e:
            repair = repair_prompt(prompt, str(e))
            self._check_budget()
            result2 = self._complete_with_retry(repair, model=model, system=system, max_tokens=max_tokens)
            self._record_usage(result2, stage, source_id)
            self._check_truncation(result2, stage, max_tokens)
            return self._parse(result2.text, schema, stage)
