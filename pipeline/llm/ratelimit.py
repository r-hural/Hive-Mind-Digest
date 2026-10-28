"""Sliding-window rate limiter shared by all worker threads of one provider (IDEA-075)."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

from pipeline.llm.base import LLMInterrupted, interrupt_requested, wait_interrupt


class RateLimiter:
    def __init__(self, rpm: int, tpm: int = 0, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], object] = wait_interrupt):
        self.rpm = rpm
        self.tpm = tpm
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._events: deque[tuple[float, int]] = deque()   # (timestamp, estimated tokens)
        self._barrier = 0.0

    def acquire(self, estimated_tokens: int = 0) -> float:
        """Block until one more request fits in the last 60 s. Returns seconds waited.

        Token accounting is estimate-only and best-effort under contention: `estimated_tokens`
        is the caller's guess (e.g. len(prompt) // 4), never the provider's actual usage, and
        concurrent callers can race past each other between the check and the append, so a large
        request may end up waiting behind a burst of small ones rather than being queued fairly.
        """
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                if now < self._barrier:
                    delay = self._barrier - now
                else:
                    while self._events and now - self._events[0][0] >= 60:
                        self._events.popleft()
                    count = len(self._events)
                    used = sum(t for _, t in self._events)
                    rpm_ok = self.rpm <= 0 or count < self.rpm
                    tpm_ok = self.tpm <= 0 or count == 0 or used + estimated_tokens <= self.tpm
                    if rpm_ok and tpm_ok:
                        self._events.append((now, estimated_tokens))
                        return waited
                    delay = max(0.05, 60 - (now - self._events[0][0]))
            self._sleep(delay)     # the default returns at once on Ctrl+C (pipeline/llm/base.py)
            if interrupt_requested():
                raise LLMInterrupted("interrupted while waiting for the rate limit")
            waited += delay

    def penalize(self, seconds: float) -> None:
        """Push the shared barrier out by `seconds` from now (e.g. after a 429 with Retry-After),
        so every acquire() across threads waits that long before trying again."""
        with self._lock:
            self._barrier = max(self._barrier, self._clock() + seconds)
