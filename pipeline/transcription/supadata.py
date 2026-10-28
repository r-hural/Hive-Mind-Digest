"""Server-side transcripts via the Supadata API (DECISION-026).

Fetches an *existing* caption track through Supadata's servers, so it still works when YouTube
blocks this machine's IP. Off unless `supadata` is listed in `transcription.providers`; intended
order youtube_captions -> ytdlp_subs -> supadata -> whisper_local.

- Always `mode=native`: never an AI-generated transcript (trust rule "transcripts come from the
  source"). The response does not say whether the track is manual or auto-generated, so
  `extra["is_generated"]` is None.
- Key from `SUPADATA_API_KEY` only (environment, or the project-root `.env` via
  `pipeline.envfile`, never overriding). It is sent only as the `x-api-key` header and never put in
  a message; redirects are refused so the header cannot follow one to another host.
- Each configured language is requested in order. The API falls back to the first available
  language when the requested one is missing, so a response in another language counts as
  "unavailable" for the requested one (it is reused only if it is the next preferred available
  language, which saves a credit). "No track" / "not found" / auth / limit errors stop at once.
- HTTP 202 means an async job: poll it, bounded by JOB_TIMEOUT_SECONDS.
- Process-wide throttle (free tier: 1 request/second): request starts are spaced by
  MIN_REQUEST_INTERVAL_SECONDS across all threads, job polls included; an HTTP 429 is retried once
  after RETRY_429_BACKOFF_SECONDS.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from pipeline.envfile import load_project_dotenv
from pipeline.sources import SourceRef
from pipeline.transcription.base import Segment, Transcript, TranscriptionError, watch_url

API_BASE = "https://api.supadata.ai/v1/transcript"
REQUEST_TIMEOUT_SECONDS = 30
JOB_TIMEOUT_SECONDS = 120
POLL_INTERVAL_SECONDS = 2.0
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
PENDING_STATUSES = {"queued", "active", "pending", "processing"}
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}\Z")
KEY_ENV = "SUPADATA_API_KEY"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: urllib would otherwise re-send the x-api-key header to the target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect())


def _http_get(url: str, api_key: str, timeout: float) -> tuple[int, dict | None]:
    """GET `url`; return (HTTP status, parsed JSON object or None). Network failure -> TranscriptionError."""
    req = urllib.request.Request(url, headers={
        "x-api-key": api_key, "Accept": "application/json", "User-Agent": "hive-mind-digest"})
    try:
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                status, raw = resp.status, resp.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as e:
            status = e.code
            try:
                raw = e.read(MAX_RESPONSE_BYTES + 1)
            finally:
                e.close()
    except (urllib.error.URLError, OSError, ValueError) as e:
        # str(URLError) never carries the request headers, so the key cannot appear here.
        raise TranscriptionError(f"supadata: network error: {type(e).__name__}: {e}") from e
    if len(raw) > MAX_RESPONSE_BYTES:
        raise TranscriptionError(f"supadata: response over {MAX_RESPONSE_BYTES} bytes")
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return status, None
    return status, body if isinstance(body, dict) else None


# Process-wide throttle: the free tier allows 1 request/second, and transcription runs up to
# llm.max_parallel workers. Every request (including job polls) takes the lock, waits until
# MIN_REQUEST_INTERVAL_SECONDS have passed since the previous request started, and is sent while
# the lock is held, so request starts are spaced across all threads.
MIN_REQUEST_INTERVAL_SECONDS = 1.1
RETRY_429_BACKOFF_SECONDS = 3.0
_throttle_lock = threading.Lock()
_last_request_at: float | None = None


def _throttled_get(url: str, api_key: str) -> tuple[int, dict | None]:
    global _last_request_at
    with _throttle_lock:
        if _last_request_at is not None:
            wait = _last_request_at + MIN_REQUEST_INTERVAL_SECONDS - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        _last_request_at = time.monotonic()
        return _http_get(url, api_key, REQUEST_TIMEOUT_SECONDS)


def _request(url: str, api_key: str) -> tuple[int, dict | None]:
    """A throttled GET; an HTTP 429 is retried once after RETRY_429_BACKOFF_SECONDS."""
    status, body = _throttled_get(url, api_key)
    if status == 429:
        time.sleep(RETRY_429_BACKOFF_SECONDS)
        status, body = _throttled_get(url, api_key)
    return status, body


def _lang_matches(requested: str, got: Any) -> bool:
    """Same language: equal, or one is a regional variant of the other (en ~ en-US), case-insensitive."""
    if not isinstance(got, str) or not got:
        return False
    a, b = requested.lower(), got.lower()
    return a == b or b.startswith(a + "-") or a.startswith(b + "-")


class SupadataProvider:
    name = "supadata"
    kinds = frozenset({"youtube"})

    def transcribe(self, ref: SourceRef, languages: list[str]) -> Transcript:
        if ref.kind not in self.kinds:
            raise TranscriptionError(f"supadata: cannot handle {ref.kind} sources")
        video_id = ref.key
        load_project_dotenv()
        key = os.environ.get(KEY_ENV, "").strip()
        if not key:
            raise TranscriptionError(
                f"supadata: {KEY_ENV} is not set (add it to .env or the environment, "
                f"or remove supadata from transcription.providers)")
        try:
            return self._transcribe(video_id, list(languages), key)
        except TranscriptionError as e:
            msg = str(e)
            if key in msg:   # belt and braces: an API message must never echo the key back
                raise TranscriptionError(msg.replace(key, "***")) from None
            raise

    def _transcribe(self, video_id: str, languages: list[str], key: str) -> Transcript:
        if not languages:
            raise TranscriptionError("supadata: no transcription.languages configured")
        remaining = list(languages)
        misses: list[str] = []
        while remaining:
            lang = remaining.pop(0)
            data = self._fetch(video_id, lang, key)
            got = data.get("lang")
            if _lang_matches(lang, got):
                return self._build(video_id, data)
            misses.append(f"{lang} unavailable (API returned {got!r})")
            avail = data.get("availableLangs")
            if isinstance(avail, list):
                remaining = [x for x in remaining if any(_lang_matches(x, a) for a in avail)]
            if remaining and _lang_matches(remaining[0], got):
                return self._build(video_id, data)
        raise TranscriptionError(f"supadata: no track in {languages} for {video_id}: " + "; ".join(misses))

    def _fetch(self, video_id: str, lang: str, key: str) -> dict:
        query = urllib.parse.urlencode({"url": watch_url(video_id), "lang": lang, "mode": "native"})
        status, body = _request(f"{API_BASE}?{query}", key)
        if status == 202:
            job_id = (body or {}).get("jobId")
            if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
                raise TranscriptionError(f"supadata: HTTP 202 without a usable job id for {video_id}")
            return self._poll(video_id, job_id, key)
        return self._check(video_id, status, body)

    def _poll(self, video_id: str, job_id: str, key: str) -> dict:
        url = f"{API_BASE}/{urllib.parse.quote(job_id, safe='')}"
        deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
        while True:
            if time.monotonic() >= deadline:
                raise TranscriptionError(
                    f"supadata: transcript job for {video_id} timed out after {JOB_TIMEOUT_SECONDS}s")
            time.sleep(POLL_INTERVAL_SECONDS)
            status, body = _request(url, key)
            if status == 202:
                continue
            body = self._check(video_id, status, body, job=True)
            if "content" in body:
                return body
            job_status = body.get("status")
            if job_status in PENDING_STATUSES:
                continue
            if job_status == "failed" or "error" in body:
                err = body.get("error")
                detail = err.get("message") if isinstance(err, dict) else err
                raise TranscriptionError(f"supadata: transcript job for {video_id} failed: {str(detail)[:200]}")
            raise TranscriptionError(
                f"supadata: transcript job for {video_id} returned unknown status {str(job_status)[:50]!r}")

    def _check(self, video_id: str, status: int, body: dict | None, *, job: bool = False) -> dict:
        """Map HTTP status / error body to TranscriptionError; return the body when usable."""
        error = (body or {}).get("error") if isinstance((body or {}).get("error"), str) else None
        if status == 206 or error == "transcript-unavailable":
            raise TranscriptionError(f"supadata: no caption track for {video_id} (transcript-unavailable)")
        if status == 404:
            raise TranscriptionError(f"supadata: HTTP 404: video {video_id} not found")
        if status in (401, 403):
            raise TranscriptionError(
                f"supadata: HTTP {status}: {KEY_ENV} was rejected (check the key and that the plan allows this)")
        if status == 402:
            raise TranscriptionError(f"supadata: HTTP 402: payment required, the Supadata plan has no credits left")
        if status == 429:
            raise TranscriptionError(
                f"supadata: HTTP 429: rate or monthly credit limit exceeded ({error or 'limit-exceeded'})")
        if status != 200:
            message = (body or {}).get("message")
            detail = f": {error}" if error else ""
            if isinstance(message, str):
                detail += f": {message[:200]}"
            raise TranscriptionError(f"supadata: HTTP {status}{detail}")
        if body is None:
            raise TranscriptionError(f"supadata: HTTP 200 with a non-JSON body for {video_id}")
        if error and not job:
            raise TranscriptionError(f"supadata: {error[:100]}")
        return body

    def _build(self, video_id: str, data: dict) -> Transcript:
        content = data.get("content")
        if not isinstance(content, list):
            raise TranscriptionError(f"supadata: unexpected content shape for {video_id}")
        segments: list[Segment] = []
        try:
            for chunk in content:
                text = chunk["text"]
                if not isinstance(text, str):
                    raise TypeError("chunk text is not a string")
                text = " ".join(text.split())
                if not text:
                    continue
                start = float(chunk["offset"]) / 1000.0
                end = start + float(chunk["duration"]) / 1000.0
                segments.append(Segment(start, end, text))
        except (KeyError, TypeError, ValueError) as e:
            raise TranscriptionError(f"supadata: malformed chunk for {video_id}: {type(e).__name__}: {e}") from e
        if not segments:
            raise TranscriptionError(f"supadata: empty track for {video_id}")
        return Transcript(video_id=video_id, method=self.name, language=str(data.get("lang") or "und"),
                          segments=segments, extra={"is_generated": None, "via": "supadata"})
