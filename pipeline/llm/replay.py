"""Cassettes at the provider boundary (IDEA-283): (model, system, prompt) -> LLMResult.

Recorded by `hmd eval --live`, replayed by tests and CI. Because the key is the exact prompt
text, a prompt-version change invalidates its cassettes and needs one manual live run.

`max_tokens` is deliberately not part of the key: changing the output cap must not invalidate
existing cassettes. The tradeoff is that replay can never reproduce a truncation that a
different `max_tokens` would have caused. Replayed results always zero `duration_ms` (and
`cost_usd`) since reading a cassette off disk is not model time.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
import threading
from dataclasses import asdict

from pipeline.llm.base import LLMError, LLMProvider, LLMResult

CASSETTE_VERSION = 1

_RESULT_FIELDS = {f.name for f in dataclasses.fields(LLMResult)}


class CassetteMissing(LLMError):
    pass


def cassette_key(model: str, system: str | None, prompt: str) -> str:
    h = hashlib.sha256()
    for part in (model, system or "", prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class ReplayProvider:
    def __init__(self, cassette_dir: str, inner: LLMProvider | None = None, record: bool = False):
        if record and inner is None:
            raise ValueError("record=True needs an inner provider")
        self.cassette_dir = cassette_dir
        self.inner = inner
        self.record = record
        # the directory is named after the provider that recorded it, e.g. cassettes/anthropic_api
        self.name = inner.name if inner is not None else os.path.basename(os.path.normpath(cassette_dir))
        self.hits = 0
        self.misses = 0
        self.recorded = 0
        # `_lock` is held only briefly: it guards the counters and the per-key lock table.
        # `_key_locks` holds one lock per cassette key ever seen; entries are never evicted, which
        # is fine for eval-sized runs (one small Lock per distinct prompt) but would grow unbounded
        # in a long-lived process recording an open-ended stream of prompts.
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}

    def _path(self, key: str) -> str:
        return os.path.join(self.cassette_dir, key + ".json")

    def complete(self, prompt: str, *, model: str, system: str | None = None,
                 max_tokens: int = 16000) -> LLMResult:
        key = cassette_key(model, system, prompt)
        path = self._path(key)

        # Find-or-record runs under a per-key lock: calls for different keys reach the inner
        # provider in parallel, while concurrent calls for the same key serialize so only one
        # writer ever reaches disk (the loser replays what the winner just wrote, as a hit).
        with self._key_lock(key):
            if os.path.exists(path):
                self._count("hits")
                return self._load(path)

            self._count("misses")
            if self.inner is None or not self.record:
                raise CassetteMissing(
                    f"no cassette {key[:12]}... for model {model} in {self.cassette_dir}; "
                    f"record it with `hmd eval --live --provider {self.name}`")

            result = self.inner.complete(prompt, model=model, system=system, max_tokens=max_tokens)
            self._write(path, model, system, prompt, result)
            self._count("recorded")
            return result

    def _key_lock(self, key: str) -> threading.Lock:
        with self._lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = self._key_locks[key] = threading.Lock()
            return lock

    def _count(self, counter: str) -> None:
        with self._lock:
            setattr(self, counter, getattr(self, counter) + 1)

    def _load(self, path: str) -> LLMResult:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            response = {k: v for k, v in data["response"].items() if k in _RESULT_FIELDS}
            result = LLMResult(**response)
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
            raise LLMError(
                f"corrupt or outdated cassette {path}: {e}; delete it and re-record with "
                f"`hmd eval --live --provider {self.name}`") from e
        result.provider = f"replay:{self.name}"
        result.cost_usd = 0.0
        result.duration_ms = 0
        return result

    def _write(self, path: str, model: str, system: str | None, prompt: str, result: LLMResult) -> None:
        os.makedirs(self.cassette_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=self.cassette_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"cassette_version": CASSETTE_VERSION, "model": model, "system": system,
                           "prompt": prompt, "response": asdict(result)},
                          f, indent=1, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
