"""SupadataProvider (DECISION-026): all HTTP is faked through `_http_get`; no network, no key."""
from __future__ import annotations
from pipeline.sources import KINDS
from tests.helpers import yt_ref

import io
import urllib.error
import urllib.parse

import pytest

from pipeline.transcription import supadata as sd
from pipeline.transcription.base import Segment, Transcriber, Transcript, TranscriptionError
from pipeline.transcription.cache import TranscriptCache
from pipeline.transcription.chain import transcribe_with_chain
from pipeline.transcription.supadata import SupadataProvider

VID = "86j97plANCs"
KEY = "sd-secret-key-123"


class FakeHttp:
    """Scripted `_http_get`: each call pops the next (status, body) and records the request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, str, float]] = []

    def __call__(self, url, api_key, timeout):
        self.calls.append((url, api_key, timeout))
        if not self.responses:
            raise AssertionError(f"unexpected request {url}")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def queries(self):
        return [urllib.parse.parse_qs(urllib.parse.urlsplit(u).query) for u, _, _ in self.calls]


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s

    def monotonic(self):
        return self.now


def _ok(lang="en", content=None, avail=None):
    if content is None:
        content = [{"text": "Hello there", "offset": 0, "duration": 1500, "lang": lang},
                   {"text": "general Kenobi", "offset": 1500, "duration": 2250, "lang": lang}]
    return 200, {"lang": lang, "availableLangs": avail or [lang], "content": content}


@pytest.fixture(autouse=True)
def _fresh_throttle(monkeypatch):
    # The throttle is process-wide state; every test starts as if no request was ever sent.
    monkeypatch.setattr(sd, "_last_request_at", None)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(sd, "load_project_dotenv", lambda: "")
    monkeypatch.setenv("SUPADATA_API_KEY", KEY)
    clock = FakeClock()
    monkeypatch.setattr(sd.time, "sleep", clock.sleep)
    monkeypatch.setattr(sd.time, "monotonic", clock.monotonic)
    return clock


def _use(monkeypatch, fake):
    monkeypatch.setattr(sd, "_http_get", fake)
    return fake


def test_maps_chunks_ms_to_seconds(env, monkeypatch):
    fake = _use(monkeypatch, FakeHttp(_ok()))
    p = SupadataProvider()
    tr = p.transcribe(yt_ref(VID), ["en"])
    assert isinstance(p, Transcriber)
    assert tr.method == "supadata" and tr.video_id == VID and tr.language == "en"
    assert [(s.start, s.end, s.text) for s in tr.segments] == [
        (0.0, 1.5, "Hello there"), (1.5, 3.75, "general Kenobi")]
    assert tr.extra == {"is_generated": None, "via": "supadata"}
    url, key, timeout = fake.calls[0]
    assert url.startswith("https://api.supadata.ai/v1/transcript?")
    assert key == KEY and timeout > 0


def test_request_always_native_with_canonical_url_and_language(env, monkeypatch):
    fake = _use(monkeypatch, FakeHttp(_ok("de", avail=["de"])))
    SupadataProvider().transcribe(yt_ref(VID), ["de"])
    q = fake.queries()[0]
    assert q["mode"] == ["native"]
    assert q["lang"] == ["de"]
    assert q["url"] == [f"https://www.youtube.com/watch?v={VID}"]


def test_empty_and_whitespace_chunks_are_skipped_and_all_empty_is_an_error(env, monkeypatch):
    content = [{"text": "  ", "offset": 0, "duration": 100}, {"text": "hi", "offset": 100, "duration": 100}]
    _use(monkeypatch, FakeHttp(_ok(content=content)))
    tr = SupadataProvider().transcribe(yt_ref(VID), ["en"])
    assert [s.text for s in tr.segments] == ["hi"]
    _use(monkeypatch, FakeHttp(_ok(content=[{"text": " ", "offset": 0, "duration": 1}])))
    with pytest.raises(TranscriptionError, match="empty"):
        SupadataProvider().transcribe(yt_ref(VID), ["en"])


def test_malformed_chunk_is_a_transcription_error(env, monkeypatch):
    _use(monkeypatch, FakeHttp(_ok(content=[{"text": "x", "offset": None, "duration": 1}])))
    with pytest.raises(TranscriptionError, match="supadata"):
        SupadataProvider().transcribe(yt_ref(VID), ["en"])
    _use(monkeypatch, FakeHttp((200, {"lang": "en", "content": "plain text"})))
    with pytest.raises(TranscriptionError, match="supadata"):
        SupadataProvider().transcribe(yt_ref(VID), ["en"])


# --- 202 async jobs -----------------------------------------------------------

def test_202_job_is_polled_until_content(env, monkeypatch):
    fake = _use(monkeypatch, FakeHttp(
        (202, {"jobId": "job_ABC-123"}),
        (200, {"status": "queued"}),
        (200, {"status": "active"}),
        _ok(),
    ))
    tr = SupadataProvider().transcribe(yt_ref(VID), ["en"])
    assert len(tr.segments) == 2
    assert [c[0] for c in fake.calls[1:]] == ["https://api.supadata.ai/v1/transcript/job_ABC-123"] * 3
    assert all(c[1] == KEY for c in fake.calls)
    assert env.sleeps and all(s > 0 for s in env.sleeps)


def test_202_job_times_out(env, monkeypatch):
    pending = [(200, {"status": "active"})] * 1000
    _use(monkeypatch, FakeHttp((202, {"jobId": "job1"}), *pending))
    with pytest.raises(TranscriptionError, match="timed out"):
        SupadataProvider().transcribe(yt_ref(VID), ["en"])
    assert env.now <= sd.JOB_TIMEOUT_SECONDS + sd.POLL_INTERVAL_SECONDS


def test_202_job_failed(env, monkeypatch):
    _use(monkeypatch, FakeHttp((202, {"jobId": "job1"}),
                               (200, {"status": "failed", "error": {"message": "boom"}})))
    with pytest.raises(TranscriptionError, match="failed"):
        SupadataProvider().transcribe(yt_ref(VID), ["en"])


def test_202_job_unknown_status_is_an_error(env, monkeypatch):
    _use(monkeypatch, FakeHttp((202, {"jobId": "job1"}), (200, {"status": "weird"})))
    with pytest.raises(TranscriptionError, match="weird"):
        SupadataProvider().transcribe(yt_ref(VID), ["en"])


def test_202_without_valid_job_id_is_an_error(env, monkeypatch):
    _use(monkeypatch, FakeHttp((202, {"jobId": "../../etc?x=1"})))
    with pytest.raises(TranscriptionError, match="job id"):
        SupadataProvider().transcribe(yt_ref(VID), ["en"])
    _use(monkeypatch, FakeHttp((202, {})))
    with pytest.raises(TranscriptionError, match="job id"):
        SupadataProvider().transcribe(yt_ref(VID), ["en"])


def test_202_job_poll_error_status_is_mapped(env, monkeypatch):
    _use(monkeypatch, FakeHttp((202, {"jobId": "job1"}), (429, {"error": "limit-exceeded"}),
                               (429, {"error": "limit-exceeded"})))
    with pytest.raises(TranscriptionError, match="limit"):
        SupadataProvider().transcribe(yt_ref(VID), ["en"])


# --- throttle and 429 retry ---------------------------------------------------

def test_consecutive_requests_are_spaced_by_the_minimum_interval(env, monkeypatch):
    times: list[float] = []
    fake = FakeHttp((202, {"jobId": "job1"}), (200, {"status": "active"}), _ok())
    monkeypatch.setattr(sd, "POLL_INTERVAL_SECONDS", 0.0)   # only the throttle spaces the polls

    def timed(url, api_key, timeout):
        times.append(env.monotonic())
        return fake(url, api_key, timeout)

    _use(monkeypatch, timed)
    SupadataProvider().transcribe(yt_ref(VID), ["en"])
    assert len(times) == 3
    assert all(b - a >= sd.MIN_REQUEST_INTERVAL_SECONDS - 1e-9 for a, b in zip(times, times[1:]))


def test_concurrent_requests_from_threads_are_spaced(env, monkeypatch):
    import threading

    times: list[float] = []
    record_lock = threading.Lock()

    def timed(url, api_key, timeout):
        with record_lock:
            times.append(env.monotonic())
        return _ok()

    _use(monkeypatch, timed)
    errors: list[BaseException] = []

    def work():
        try:
            SupadataProvider().transcribe(yt_ref(VID), ["en"])
        except BaseException as e:   # surfaced below
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors and len(times) == 6
    times.sort()
    assert all(b - a >= sd.MIN_REQUEST_INTERVAL_SECONDS - 1e-9 for a, b in zip(times, times[1:]))


def test_429_then_200_succeeds_with_exactly_one_retry(env, monkeypatch):
    fake = _use(monkeypatch, FakeHttp((429, {"error": "limit-exceeded"}), _ok()))
    tr = SupadataProvider().transcribe(yt_ref(VID), ["en"])
    assert tr.method == "supadata" and len(fake.calls) == 2
    assert fake.calls[0][0] == fake.calls[1][0]          # the same request, retried
    assert sd.RETRY_429_BACKOFF_SECONDS in env.sleeps


def test_429_twice_raises_the_limit_message(env, monkeypatch):
    fake = _use(monkeypatch, FakeHttp((429, {"error": "limit-exceeded"}), (429, {"error": "limit-exceeded"})))
    with pytest.raises(TranscriptionError, match="HTTP 429: rate or monthly credit limit exceeded"):
        SupadataProvider().transcribe(yt_ref(VID), ["en"])
    assert len(fake.calls) == 2


def test_429_on_a_job_poll_is_retried_once(env, monkeypatch):
    fake = _use(monkeypatch, FakeHttp((202, {"jobId": "job1"}), (429, {"error": "limit-exceeded"}), _ok()))
    tr = SupadataProvider().transcribe(yt_ref(VID), ["en"])
    assert len(tr.segments) == 2 and len(fake.calls) == 3


# --- errors -------------------------------------------------------------------

@pytest.mark.parametrize("status,body", [
    (206, {"error": "transcript-unavailable", "message": "No transcript"}),
    (404, {"error": "not-found", "message": "Video not found"}),
    (200, {"error": "transcript-unavailable"}),
])
def test_no_track_is_a_transcription_error_and_stops(env, monkeypatch, status, body):
    fake = _use(monkeypatch, FakeHttp((status, body)))
    with pytest.raises(TranscriptionError, match="no caption track|not found"):
        SupadataProvider().transcribe(yt_ref(VID), ["en", "de"])
    assert len(fake.calls) == 1     # no further credits spent on other languages


@pytest.mark.parametrize("status,needle", [
    (401, "SUPADATA_API_KEY"),
    (402, "plan"),
    (403, "SUPADATA_API_KEY"),
    (429, "limit"),
])
def test_auth_plan_limit_errors_never_leak_the_key(env, monkeypatch, status, needle):
    body = {"error": "x", "message": f"bad key {KEY}", "details": KEY}
    _use(monkeypatch, FakeHttp(*[(status, body)] * (2 if status == 429 else 1)))   # 429 is retried once
    with pytest.raises(TranscriptionError) as ei:
        SupadataProvider().transcribe(yt_ref(VID), ["en"])
    assert needle in str(ei.value)
    assert KEY not in str(ei.value)
    assert f"HTTP {status}" in str(ei.value)


def test_other_http_error_includes_status_without_key(env, monkeypatch):
    _use(monkeypatch, FakeHttp((500, {"error": "internal-error", "message": f"oops {KEY}"})))
    with pytest.raises(TranscriptionError) as ei:
        SupadataProvider().transcribe(yt_ref(VID), ["en"])
    assert "HTTP 500" in str(ei.value) and KEY not in str(ei.value)


def test_network_error_is_a_transcription_error(env, monkeypatch):
    _use(monkeypatch, FakeHttp(TranscriptionError("supadata: network error: URLError")))
    with pytest.raises(TranscriptionError, match="network"):
        SupadataProvider().transcribe(yt_ref(VID), ["en"])


def test_missing_key_is_a_transcription_error_without_any_request(monkeypatch):
    monkeypatch.setattr(sd, "load_project_dotenv", lambda: "")
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
    fake = _use(monkeypatch, FakeHttp())
    with pytest.raises(TranscriptionError, match="SUPADATA_API_KEY is not set"):
        SupadataProvider().transcribe(yt_ref(VID), ["en"])
    assert fake.calls == []


def test_key_is_loaded_through_the_project_dotenv_loader(monkeypatch):
    loaded = []
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)

    def fake_load():
        loaded.append(True)
        monkeypatch.setenv("SUPADATA_API_KEY", KEY)
        return "x/.env"

    monkeypatch.setattr(sd, "load_project_dotenv", fake_load)
    fake = _use(monkeypatch, FakeHttp(_ok()))
    SupadataProvider().transcribe(yt_ref(VID), ["en"])
    assert loaded and fake.calls[0][1] == KEY


# --- language handling --------------------------------------------------------

def test_language_mismatch_is_unavailable_not_accepted(env, monkeypatch):
    fake = _use(monkeypatch, FakeHttp(_ok("fr", avail=["fr"])))
    with pytest.raises(TranscriptionError, match="en"):
        SupadataProvider().transcribe(yt_ref(VID), ["en"])
    assert len(fake.calls) == 1


def test_language_mismatch_tries_next_configured_language(env, monkeypatch):
    # en missing; the API falls back to fr; de is available and next in preference -> request de.
    fake = _use(monkeypatch, FakeHttp(_ok("fr", avail=["fr", "de"]), _ok("de", avail=["fr", "de"])))
    tr = SupadataProvider().transcribe(yt_ref(VID), ["en", "de", "fr"])
    assert tr.language == "de"
    assert [q["lang"] for q in fake.queries()] == [["en"], ["de"]]


def test_fallback_response_reused_when_it_is_the_next_preferred_language(env, monkeypatch):
    # en missing; the fallback is de, which is the next available configured language: no second credit.
    fake = _use(monkeypatch, FakeHttp(_ok("de", avail=["de"])))
    tr = SupadataProvider().transcribe(yt_ref(VID), ["en", "it", "de"])
    assert tr.language == "de" and len(fake.calls) == 1


def test_languages_not_in_available_langs_are_skipped(env, monkeypatch):
    fake = _use(monkeypatch, FakeHttp(_ok("fr", avail=["fr"])))
    with pytest.raises(TranscriptionError, match="unavailable"):
        SupadataProvider().transcribe(yt_ref(VID), ["en", "de"])
    assert len(fake.calls) == 1


def test_regional_variant_matches_base_language(env, monkeypatch):
    _use(monkeypatch, FakeHttp(_ok("en-US", avail=["en-US"])))
    tr = SupadataProvider().transcribe(yt_ref(VID), ["en"])
    assert tr.language == "en-US"


def test_lang_matches():
    m = sd._lang_matches
    assert m("en", "en") and m("en", "EN") and m("en", "en-GB") and m("en-GB", "en")
    assert not m("en", "fr") and not m("en", None) and not m("en", "eng") and not m("en-US", "en-GB")


# --- the real HTTP layer, with the opener mocked ------------------------------

class _Resp(io.BytesIO):
    def __init__(self, status, payload):
        super().__init__(payload)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_http_get_sends_key_header_and_timeout(monkeypatch):
    seen = {}

    def fake_open(req, timeout):
        seen["req"], seen["timeout"] = req, timeout
        return _Resp(200, b'{"lang": "en", "content": []}')

    monkeypatch.setattr(sd._OPENER, "open", fake_open)
    status, body = sd._http_get("https://api.supadata.ai/v1/transcript?x=1", KEY, 7)
    assert status == 200 and body == {"lang": "en", "content": []}
    assert seen["req"].get_header("X-api-key") == KEY and seen["timeout"] == 7
    assert seen["req"].full_url.startswith("https://api.supadata.ai/")


def test_http_get_reads_error_bodies(monkeypatch):
    def fake_open(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {},
                                     io.BytesIO(b'{"error": "unauthorized"}'))

    monkeypatch.setattr(sd._OPENER, "open", fake_open)
    assert sd._http_get("https://api.supadata.ai/v1/transcript", KEY, 7) == (401, {"error": "unauthorized"})


def test_http_get_non_json_body_and_network_errors(monkeypatch):
    monkeypatch.setattr(sd._OPENER, "open", lambda req, timeout: _Resp(502, b"<html>bad gateway"))
    assert sd._http_get("https://api.supadata.ai/v1/transcript", KEY, 7) == (502, None)

    def boom(req, timeout):
        raise urllib.error.URLError("dns failure")

    monkeypatch.setattr(sd._OPENER, "open", boom)
    with pytest.raises(TranscriptionError, match="network"):
        sd._http_get("https://api.supadata.ai/v1/transcript", KEY, 7)


def test_http_get_refuses_redirects():
    # The key header must never follow a redirect to another host.
    handler = sd._NoRedirect()
    assert handler.redirect_request(None, None, 302, "Found", {}, "https://evil.example/") is None


def test_200_without_json_is_an_error(env, monkeypatch):
    _use(monkeypatch, FakeHttp((200, None)))
    with pytest.raises(TranscriptionError, match="supadata"):
        SupadataProvider().transcribe(yt_ref(VID), ["en"])


# --- chain and cache ----------------------------------------------------------

class Stub:
    def __init__(self, name, ok=True):
        self.name, self.ok, self.calls = name, ok, 0

    kinds = frozenset(KINDS)

    def transcribe(self, ref, languages):
        self.calls += 1
        if not self.ok:
            raise TranscriptionError(f"{self.name} failed")
        return Transcript(ref.key, self.name, "en", [Segment(0, 1, f"from {self.name}")])


def test_chain_falls_through_to_supadata(env, monkeypatch, tmp_path):
    _use(monkeypatch, FakeHttp(_ok()))
    cache = TranscriptCache(str(tmp_path))
    a, b, w = Stub("youtube_captions", False), Stub("ytdlp_subs", False), Stub("whisper_local")
    tr = transcribe_with_chain(yt_ref(VID), [a, b, SupadataProvider(), w], ["en"], cache, log=lambda m: None)
    assert tr.method == "supadata" and w.calls == 0
    assert cache.has(VID, "supadata")


def test_chain_falls_through_supadata_to_whisper(env, monkeypatch, tmp_path):
    _use(monkeypatch, FakeHttp((206, {"error": "transcript-unavailable"})))
    cache = TranscriptCache(str(tmp_path))
    w = Stub("whisper_local")
    logs: list[str] = []
    tr = transcribe_with_chain(yt_ref(VID), [Stub("youtube_captions", False), SupadataProvider(), w], ["en"],
                               cache, log=logs.append)
    assert tr.method == "whisper_local" and w.calls == 1
    assert any("supadata failed" in line for line in logs)


def test_chain_missing_key_falls_through_to_whisper(monkeypatch, tmp_path):
    monkeypatch.setattr(sd, "load_project_dotenv", lambda: "")
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
    cache = TranscriptCache(str(tmp_path))
    tr = transcribe_with_chain(yt_ref(VID), [SupadataProvider(), Stub("whisper_local")], ["en"], cache,
                               log=lambda m: None)
    assert tr.method == "whisper_local"


def test_cache_round_trip_with_supadata_method(env, monkeypatch, tmp_path):
    _use(monkeypatch, FakeHttp(_ok()))
    cache = TranscriptCache(str(tmp_path))
    tr = SupadataProvider().transcribe(yt_ref(VID), ["en"])
    cache.save(tr)
    back = cache.load(VID, "supadata")
    assert back.method == "supadata" and back.language == "en"
    assert [(s.start, s.end, s.text) for s in back.segments] == [(s.start, s.end, s.text) for s in tr.segments]
    assert back.extra == {"is_generated": None, "via": "supadata"}
    assert cache.load_best(VID, ["youtube_captions", "supadata"]).method == "supadata"
