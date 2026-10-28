"""Message Batches API (IDEA-304): half the token price, results in minutes to hours instead of seconds.

This module maps one request to a batch line and one result line back to the LLMResult (or LLMError) a
direct call would give, and submits and waits for a batch. AnthropicBatchProvider (below) gathers the
calls a step sends together into one batch. Batched results carry provider "anthropic_batch" and are
priced here at the batch rate, so the client records the discounted cost.

Waves: the stages send a step's calls from a thread pool, so the calls that arrive within
`gather_seconds` of each other (up to `max_wave`) become one batch, and each caller blocks until its own
line is in. A wave smaller than `min_requests` (merge outline, verification, a lone schema repair) goes
to the direct API at full price instead of waiting for a batch of one. Identical requests in a wave
(same model, system, prompt, max_tokens) are sent once; one caller carries the tokens and cost, the others get a
zero-usage copy. A wave runs on its own daemon thread, so a later wave (a schema repair) is not held up
by an earlier one. A wave whose requests would exceed the API's request-size limit goes out as several
batches (split_requests), sent together and waited for in turn.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Iterable

import anthropic
import httpx

from pipeline.llm.anthropic_api import (FATAL_ERROR_TYPES, FATAL_MESSAGE_MARKERS, AnthropicApiProvider,
                                        _status_error, message_result)
from pipeline.llm.base import (LLMError, LLMFatalError, LLMInterrupted, LLMProvider, LLMResult,
                               LLMTransientError, interrupt_requested, request_interrupt, wait_interrupt)
from pipeline.llm.replay import cassette_key
from pipeline.pricing import estimate_cost

BATCH_PROVIDER = "anthropic_batch"      # usage_events.provider of a batched call
TRANSIENT_ERROR_TYPES = {"api_error", "overloaded_error", "rate_limit_error", "timeout_error"}
PROGRESS_LOG_SECONDS = 300              # one progress line per batch every 5 minutes
# Message Batches limits: 100,000 requests or 256 MB per batch. A wave is at most llm.batch.max_wave
# requests (pipeline/config.py), far under the count; the size is checked here on the serialized
# requests, with a margin for the request envelope.
MAX_BATCH_BYTES = 230_000_000


def request_key(model: str, system: str | None, prompt: str, max_tokens: int) -> str:
    """A request's custom_id, dedup key and journal key: the cassette key plus `max_tokens`, so two calls
    asking for different output limits are never merged. 64 hex characters, the API's custom_id limit."""
    return hashlib.sha256(f"{cassette_key(model, system, prompt)}|{max_tokens}".encode("ascii")).hexdigest()


def build_request(custom_id: str, prompt: str, *, model: str, system: str | None, max_tokens: int) -> dict:
    params: dict = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if system:
        params["system"] = system
    return {"custom_id": custom_id, "params": params}


def split_requests(requests: list[dict], max_bytes: int) -> list[list[dict]]:
    """Consecutive chunks whose serialized size stays within max_bytes. A single request over the
    limit goes alone (the API then rejects it, failing only its own caller's batch)."""
    chunks: list[list[dict]] = []
    size = 0
    for r in requests:
        n = len(json.dumps(r).encode("utf-8")) + 1
        if chunks and size + n <= max_bytes:
            chunks[-1].append(r)
            size += n
        else:
            chunks.append([r])
            size = n
    return chunks


def entry_result(entry, model: str, duration_ms: int) -> LLMResult:
    """One line of a batch's results as the LLMResult a direct call returns, or the LLMError it raises:
    an errored line maps like the direct provider's errors (auth, billing, credit balance -> fatal;
    overloaded / api / rate-limit -> transient, so the client retries it in a later wave)."""
    res = entry.result
    kind = getattr(res, "type", "")
    if kind == "succeeded":
        r = message_result(res.message, model, BATCH_PROVIDER, duration_ms)
        r.cost_usd = estimate_cost(r.model, r.input_tokens, r.output_tokens, r.cache_read_tokens,
                                   r.cache_write_tokens, batch=True)
        return r
    if kind == "errored":
        err = getattr(getattr(res, "error", None), "error", None)
        etype, emsg = str(getattr(err, "type", "") or ""), str(getattr(err, "message", "") or "")
        detail = f"batch request errored: {etype}: {emsg}"
        if etype in FATAL_ERROR_TYPES or any(m in emsg.lower() for m in FATAL_MESSAGE_MARKERS):
            raise LLMFatalError(detail)
        if etype in TRANSIENT_ERROR_TYPES:
            raise LLMTransientError(detail)
        raise LLMError(detail)
    if kind == "expired":
        raise LLMError("batch request expired before it was processed (24 h limit); resume the notebook")
    raise LLMError(f"batch request {kind or 'returned no result'}")      # canceled


def _api_call(fn, what: str):
    try:
        return fn()
    except anthropic.APIStatusError as e:
        err = _status_error(e)
        err.status = e.status_code      # read by AnthropicBatchProvider._reuse (404 gone, 403 not ours)
        raise err from e
    except anthropic.APIConnectionError as e:
        raise LLMTransientError(f"connection error ({what}): {e}") from e
    except anthropic.AnthropicError as e:
        raise LLMError(f"anthropic SDK error ({what}): {e}") from e
    except httpx.HTTPError as e:
        # the results are streamed: a dropped connection mid-read surfaces as a bare httpx error, not an
        # SDK one. Transient, so the client retries; the retry reuses the batch through the journal.
        raise LLMTransientError(f"network error ({what}): {type(e).__name__}: {e}") from e


def submit_batch(client, requests: list[dict]) -> str:
    return _api_call(lambda: client.messages.batches.create(requests=requests), "batch create").id


def _cancel(client, batch_id: str, log: Callable[[str], None]) -> None:
    try:
        client.messages.batches.cancel(batch_id)
    except anthropic.AnthropicError as e:
        log(f"  warning: batch {batch_id} could not be cancelled: {e}")


def wait_batch(client, batch_id: str, *, poll_seconds: float, max_wait_seconds: float,
               log: Callable[[str], None], clock: Callable[[], float] = time.monotonic,
               sleep: Callable[[float], object] = wait_interrupt) -> dict[str, object]:
    """Polls until the batch ends and returns its result lines by custom_id. A transient poll error is
    polled again. On an interrupt, or after max_wait_seconds, the batch is cancelled (best effort; the
    requests it already finished are still billed) and LLMInterrupted or LLMError is raised."""
    start = last_log = clock()
    while True:
        try:
            b = _api_call(lambda: client.messages.batches.retrieve(batch_id), "batch poll")
        except LLMTransientError:
            b = None
        if b is not None and b.processing_status == "ended":
            break
        if interrupt_requested():
            _cancel(client, batch_id, log)
            raise LLMInterrupted(f"interrupted; batch {batch_id} cancelled")
        if clock() - start >= max_wait_seconds:
            _cancel(client, batch_id, log)
            raise LLMError(f"batch {batch_id} not done after {max_wait_seconds / 3600:g} h; cancelled")
        if b is not None and clock() - last_log >= PROGRESS_LOG_SECONDS:
            c = b.request_counts
            log(f"  batch {batch_id}: {c.succeeded + c.errored} done, {c.processing} processing "
                f"after {(clock() - start) / 60:.0f} min")
            last_log = clock()
        sleep(poll_seconds)
    c = b.request_counts
    log(f"  batch {batch_id}: ended after {(clock() - start) / 60:.1f} min ({c.succeeded} succeeded, "
        f"{c.errored} errored, {c.expired} expired, {c.canceled} canceled)")
    return _api_call(lambda: {e.custom_id: e for e in client.messages.batches.results(batch_id)},
                     "batch results")


@dataclass(eq=False)
class _Request:
    key: str
    prompt: str
    model: str
    system: str | None
    max_tokens: int
    done: threading.Event = field(default_factory=threading.Event)
    direct: bool = False
    result: LLMResult | None = None
    error: BaseException | None = None


class AnthropicBatchProvider:
    """LLMProvider that sends the calls a step makes together as one Message Batch (see module doc)."""

    name = "anthropic_api"      # same key, prices, run row and price gate as the direct API
    batching = True             # read by the factory (no rate limiter) and the runner (wide pools)

    def __init__(self, client=None, *, direct: LLMProvider | None = None, min_requests: int = 2,
                 max_wave: int = 100, gather_seconds: float = 2.0, poll_seconds: float = 30.0,
                 max_wait_seconds: float = 86_400.0, journal=None, log: Callable[[str], None] = print):
        if client is None:
            direct = direct or AnthropicApiProvider()
            client = direct._client
        self._client = client
        self._direct = direct if direct is not None else AnthropicApiProvider(client=client)
        self._min_requests = max(1, min_requests)
        self._max_wave = max(1, max_wave)
        self._gather = gather_seconds
        self._poll = poll_seconds
        self._max_wait = max_wait_seconds
        self._journal = journal
        self.log = log
        self._cond = threading.Condition()
        self._pending: list[_Request] = []
        self._last_arrival = 0.0
        self._dispatcher: threading.Thread | None = None

    def complete(self, prompt: str, *, model: str, system: str | None = None,
                 max_tokens: int = 16000) -> LLMResult:
        req = _Request(request_key(model, system, prompt, max_tokens), prompt, model, system, max_tokens)
        with self._cond:
            self._pending.append(req)
            self._last_arrival = time.monotonic()
            if self._dispatcher is None:
                self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True,
                                                    name="hmd-batch-dispatch")
                self._dispatcher.start()
            self._cond.notify_all()
        try:
            while not req.done.wait(0.5):    # sliced: a caller on the main thread still sees Ctrl+C
                if interrupt_requested():
                    with self._cond:
                        if req in self._pending:       # not yet dispatched: never send it
                            self._pending.remove(req)
                            raise LLMInterrupted("interrupted before the batch was sent")
        except BaseException:
            request_interrupt()
            raise
        if req.direct:
            return self._direct.complete(prompt, model=model, system=system, max_tokens=max_tokens)
        if req.error is not None:
            raise req.error
        return req.result

    def _journal_call(self, fn, what: str) -> None:
        try:
            fn()
        except OSError as e:
            self.log(f"  warning: batch journal not updated ({what}): {e}")

    def _reuse(self, keys: list[str]) -> dict[str, object]:
        """Succeeded result lines for `keys` from batches an earlier run sent and never delivered."""
        found: dict[str, object] = {}
        for bid in self._journal.batches_for(keys):
            missing = [k for k in keys if k not in found]
            if not missing:
                break
            try:
                entries = wait_batch(self._client, bid, poll_seconds=self._poll,
                                     max_wait_seconds=self._max_wait, log=self.log)
            except LLMError as e:
                if interrupt_requested():
                    raise           # Ctrl+C while waiting on an old batch: send nothing new
                status = getattr(e, "status", None)
                if status == 404:   # gone (or another workspace's): drop it, or every wave polls it again
                    self._journal_call(lambda: self._journal.drop(bid), "drop")
                    self.log(f"  batch {bid}: no longer reachable (404); dropped from the journal")
                    continue
                if status == 403:   # this key may not read it: not worth stopping the run for
                    self.log(f"  warning: batch {bid}: earlier results not readable with this key (403); "
                             f"sending those requests again")
                    continue
                if isinstance(e, LLMFatalError):
                    raise
                self.log(f"  batch {bid}: earlier results not reusable ({e})")
                continue
            hits = {k: entries[k] for k in missing
                    if k in entries and getattr(entries[k].result, "type", "") == "succeeded"}
            if hits:
                self.log(f"  batch {bid}: reused {len(hits)} result(s) an earlier run paid for")
                found.update(hits)
        return found

    def _dispatch_loop(self) -> None:
        while True:
            with self._cond:
                while not self._pending:
                    self._cond.wait()
                while len(self._pending) < self._max_wave:
                    quiet = time.monotonic() - self._last_arrival
                    if quiet >= self._gather:
                        break
                    self._cond.wait(self._gather - quiet)
                wave, self._pending = self._pending[:self._max_wave], self._pending[self._max_wave:]
            try:
                if interrupt_requested():
                    for r in wave:
                        r.error = LLMInterrupted("interrupted before the batch was sent")
                        r.done.set()
                elif len(wave) < self._min_requests:
                    for r in wave:
                        r.direct = True
                        r.done.set()
                else:
                    threading.Thread(target=self._run_wave, args=(wave,), daemon=True,
                                     name="hmd-batch-wave").start()
            except Exception as e:      # the dispatcher must survive, or every later caller waits forever
                for r in wave:
                    if not r.done.is_set():
                        r.error = LLMError(f"batch dispatch failed: {e}")
                        r.done.set()

    def _send_and_wait(self, requests: list[dict]) -> dict[str, object]:
        """Sends the requests as one batch, or several when they exceed MAX_BATCH_BYTES, and returns
        every result line. On a failure, the batches sent and not yet waited for are cancelled, except on a
        transient one: the callers retry, and the retry reuses those batches through the journal."""
        sent: list[str] = []
        waiting = False
        entries: dict[str, object] = {}
        try:
            for chunk in split_requests(requests, MAX_BATCH_BYTES):
                batch_id = submit_batch(self._client, chunk)
                sent.append(batch_id)
                keys = [r["custom_id"] for r in chunk]
                if self._journal is not None:
                    self._journal_call(lambda: self._journal.record(batch_id, keys), "record")
                self.log(f"  batch {batch_id}: {len(chunk)} request(s) sent; results can take minutes to hours")
            waiting = True
            while sent:
                entries.update(wait_batch(self._client, sent[0], poll_seconds=self._poll,
                                          max_wait_seconds=self._max_wait, log=self.log))
                sent.pop(0)
        except LLMTransientError:
            raise
        except BaseException:
            # wait_batch already cancelled the batch it was waiting on (interrupt, time-out)
            for batch_id in sent[1:] if waiting else sent:
                _cancel(self._client, batch_id, self.log)
            raise
        return entries

    def _run_wave(self, wave: list[_Request]) -> None:
        by_key: dict[str, list[_Request]] = {}
        for r in wave:
            by_key.setdefault(r.key, []).append(r)
        t0 = time.monotonic()
        try:
            entries: dict[str, object] = self._reuse(list(by_key)) if self._journal is not None else {}
            todo = [k for k in by_key if k not in entries]
            if todo and interrupt_requested():
                raise LLMInterrupted("interrupted before the batch was sent")
            if todo:
                requests = [build_request(k, by_key[k][0].prompt, model=by_key[k][0].model,
                                          system=by_key[k][0].system, max_tokens=by_key[k][0].max_tokens)
                            for k in todo]
                entries.update(self._send_and_wait(requests))
            ms = int((time.monotonic() - t0) * 1000)
            for key, reqs in by_key.items():
                entry = entries.get(key)
                for i, r in enumerate(reqs):
                    try:
                        if entry is None:
                            raise LLMError("batch returned no result for this request")
                        res = entry_result(entry, r.model, ms)
                        if i:      # an identical request rides along: sent and billed once
                            res = dataclasses.replace(res, input_tokens=0, output_tokens=0, cache_read_tokens=0,
                                                      cache_write_tokens=0, cost_usd=0.0)
                        r.result = res
                    except LLMError as e:
                        r.error = e
            if self._journal is not None:
                self._journal_call(lambda: self._journal.forget(by_key), "forget")
        except Exception as e:
            for reqs in by_key.values():
                for r in reqs:
                    if r.result is None and r.error is None:
                        r.error = e
        finally:
            for reqs in by_key.values():
                for r in reqs:
                    if r.result is None and r.error is None:
                        r.error = LLMError("batch wave ended without a result")
                    r.done.set()


RESULTS_DAYS = 29          # the API keeps a batch's results this long
LOCK_STALE_SECONDS = 60    # a journal lock file older than this was left by a crashed process
LOCK_TIMEOUT_SECONDS = 10


class BatchJournal:
    """Batches sent from this home (`<home>/cache/batches.json`): [{"batch_id", "created", "keys"}].
    A key is a request's custom_id; it is removed once its result has been delivered, so the journal
    only ever offers results nobody has used yet (a run interrupted or crashed after it paid for them).
    A missing or unreadable file is an empty journal; a failed write is logged by the caller, never fatal.

    Two hmd processes can share one home, so every read and read-modify-write holds a lock file
    (`batches.json.lock`, created exclusively, which works the same on Windows) on top of the thread
    lock: each change re-reads the file under the lock, merges, and replaces it atomically."""

    def __init__(self, path: str, now: Callable[[], datetime] = datetime.now):
        self.path = path
        self._now = now
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def _locked(self):
        lock = self.path + ".lock"
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
        with self._lock:
            while True:
                try:
                    os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                    break
                except (FileExistsError, PermissionError):   # Windows: PermissionError while it is deleted
                    try:
                        if time.time() - os.path.getmtime(lock) > LOCK_STALE_SECONDS:
                            os.remove(lock)
                            continue
                    except OSError:
                        pass
                    if time.monotonic() >= deadline:
                        raise OSError(f"batch journal is locked by another process ({lock})") from None
                    time.sleep(0.02)
            try:
                yield
            finally:
                try:
                    os.remove(lock)
                except OSError:
                    pass

    def _load(self) -> list[dict]:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return []
        cutoff = self._now() - timedelta(days=RESULTS_DAYS)
        out = []
        for e in data if isinstance(data, list) else []:
            try:
                if datetime.fromisoformat(e["created"]) >= cutoff and e["keys"]:
                    out.append({"batch_id": str(e["batch_id"]), "created": e["created"], "keys": list(e["keys"])})
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def _save(self, entries: list[dict]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=1)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def record(self, batch_id: str, keys: list[str]) -> None:
        with self._locked():
            entries = self._load()
            entries.append({"batch_id": batch_id, "created": self._now().isoformat(), "keys": list(keys)})
            self._save(entries)

    def batches_for(self, keys: Iterable[str]) -> list[str]:
        wanted = set(keys)
        try:
            with self._locked():
                entries = self._load()
        except OSError:
            return []        # an unreadable journal is an empty one: nothing to reuse
        return [e["batch_id"] for e in reversed(entries) if wanted.intersection(e["keys"])]

    def drop(self, batch_id: str) -> None:
        with self._locked():
            self._save([e for e in self._load() if e["batch_id"] != batch_id])

    def forget(self, keys: Iterable[str]) -> None:
        done = set(keys)
        with self._locked():
            entries = self._load()
            for e in entries:
                e["keys"] = [k for k in e["keys"] if k not in done]
            self._save([e for e in entries if e["keys"]])
