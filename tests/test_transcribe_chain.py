from pipeline.sources import KINDS
from tests.helpers import yt_ref
import pytest
from pipeline.config import DEFAULT_TRANSCRIBERS, TRANSCRIBERS, Config
from pipeline.transcription.base import Segment, Transcript, TranscriptionError
from pipeline.transcription.cache import TranscriptCache
from pipeline.transcription.chain import build_transcribers, transcribe_with_chain


class Stub:
    def __init__(self, name, ok=True):
        self.name, self.ok, self.calls = name, ok, 0
    kinds = frozenset(KINDS)

    def transcribe(self, ref, languages):
        self.calls += 1
        if not self.ok:
            raise TranscriptionError(f"{self.name} failed")
        return Transcript(ref.key, self.name, "en", [Segment(0, 1, f"from {self.name}")])


class BuggyStub:
    def __init__(self, name):
        self.name, self.calls = name, 0
    kinds = frozenset(KINDS)

    def transcribe(self, ref, languages):
        self.calls += 1
        raise RuntimeError("bug")


def test_build_transcribers_follows_config_order():
    cfg = Config.model_validate({"transcription": {"providers": ["whisper_local", "youtube_captions"],
                                                    "whisper": {"model_size": "tiny", "timeout_seconds": 7}}})
    ts = build_transcribers(cfg)
    assert [t.name for t in ts] == ["whisper_local", "youtube_captions", "web_text", "pdf_text", "local_text"]
    assert ts[0].model_size == "tiny" and ts[0].timeout_seconds == 7


def test_first_success_wins_and_is_cached(tmp_path):
    cache = TranscriptCache(str(tmp_path))
    a, b = Stub("youtube_captions", ok=False), Stub("ytdlp_subs")
    tr = transcribe_with_chain(yt_ref("abcdefghijk"), [a, b], ["en"], cache, log=lambda m: None)
    assert tr.method == "ytdlp_subs" and cache.has("abcdefghijk", "ytdlp_subs")
    assert a.calls == 1 and b.calls == 1


def test_cache_hit_skips_providers(tmp_path):
    cache = TranscriptCache(str(tmp_path))
    cache.save(Transcript("abcdefghijk", "whisper_local", "en", [Segment(0, 1, "cached")]))
    a = Stub("youtube_captions")
    tr = transcribe_with_chain(yt_ref("abcdefghijk"), [a, Stub("whisper_local")], ["en"], cache, log=lambda m: None)
    assert tr.method == "whisper_local" and tr.segments[0].text == "cached" and a.calls == 0


def test_all_fail_raises_with_every_reason(tmp_path):
    cache = TranscriptCache(str(tmp_path))
    with pytest.raises(TranscriptionError) as ei:
        transcribe_with_chain(yt_ref("abcdefghijk"), [Stub("youtube_captions", False), Stub("whisper_local", False)],
                              ["en"], cache, log=lambda m: None)
    assert "youtube_captions failed" in str(ei.value) and "whisper_local failed" in str(ei.value)


def test_unexpected_exception_is_contained_and_chain_continues(tmp_path):
    cache = TranscriptCache(str(tmp_path))
    logs: list[str] = []
    buggy, ok = BuggyStub("youtube_captions"), Stub("ytdlp_subs")
    tr = transcribe_with_chain(yt_ref("abcdefghijk"), [buggy, ok], ["en"], cache, log=logs.append)
    assert tr.method == "ytdlp_subs"
    assert buggy.calls == 1 and ok.calls == 1
    assert any("unexpected RuntimeError" in line for line in logs)


def test_cache_write_failure_is_logged_and_raised(tmp_path, monkeypatch):
    cache = TranscriptCache(str(tmp_path))
    logs: list[str] = []

    def boom(transcript):
        raise TranscriptionError("disk")

    monkeypatch.setattr(cache, "save", boom)
    with pytest.raises(TranscriptionError, match="disk"):
        transcribe_with_chain(yt_ref("abcdefghijk"), [Stub("youtube_captions")], ["en"], cache, log=logs.append)
    assert any("cache write failed" in line for line in logs)


def test_cache_write_os_error_propagates_and_is_logged(tmp_path, monkeypatch):
    cache = TranscriptCache(str(tmp_path))
    logs: list[str] = []

    def boom(transcript):
        raise OSError("disk full")

    monkeypatch.setattr(cache, "save", boom)
    with pytest.raises(OSError, match="disk full"):
        transcribe_with_chain(yt_ref("abcdefghijk"), [Stub("youtube_captions")], ["en"], cache, log=logs.append)
    assert any("cache write failed" in line for line in logs)


def test_no_transcribers_configured_raises():
    cache = TranscriptCache("unused")
    with pytest.raises(TranscriptionError, match="no transcribers configured"):
        transcribe_with_chain(yt_ref("abcdefghijk"), [], ["en"], cache, log=lambda m: None)


def test_build_transcribers_default_is_the_three_local_providers():
    # supadata (a third-party API, DECISION-026) is off unless listed in transcription.providers.
    assert [t.name for t in build_transcribers(Config())] == [*DEFAULT_TRANSCRIBERS, "web_text", "pdf_text", "local_text"]
    assert "supadata" not in DEFAULT_TRANSCRIBERS


def test_build_transcribers_covers_every_known_provider():
    cfg = Config.model_validate({"transcription": {"providers": list(TRANSCRIBERS)}})
    assert [t.name for t in build_transcribers(cfg)] == [*TRANSCRIBERS, "web_text", "pdf_text", "local_text"]
    assert TRANSCRIBERS == ("youtube_captions", "ytdlp_subs", "supadata", "whisper_local")


def test_success_log_lines_are_exact(tmp_path):
    cache = TranscriptCache(str(tmp_path))
    logs: list[str] = []
    a, b = Stub("youtube_captions", ok=False), Stub("ytdlp_subs")
    transcribe_with_chain(yt_ref("abcdefghijk"), [a, b], ["en"], cache, log=logs.append)
    assert logs == [
        "  [abcdefghijk] youtube_captions failed: youtube_captions failed",
        "  [abcdefghijk] ytdlp_subs: 1 segments, 2 words",
    ]


def test_cache_hit_log_line_is_exact(tmp_path):
    cache = TranscriptCache(str(tmp_path))
    cache.save(Transcript("abcdefghijk", "whisper_local", "en", [Segment(0, 1, "cached")]))
    logs: list[str] = []
    tr = transcribe_with_chain(yt_ref("abcdefghijk"), [Stub("whisper_local")], ["en"], cache, log=logs.append)
    assert tr.method == "whisper_local"
    assert logs == ["  [abcdefghijk] cached transcript (whisper_local)"]


class Truncated:
    """A provider that 'succeeds' with a track covering only the first few seconds (seen live:
    supadata returned 4 segments, 0:06-0:15, for a 17:23 talk)."""
    def __init__(self, name, end=15.0):
        self.name, self.end, self.calls = name, end, 0
    kinds = frozenset(KINDS)

    def transcribe(self, ref, languages):
        self.calls += 1
        return Transcript(ref.key, self.name, "en", [Segment(6.0, self.end, "only the opening words")])


def test_truncated_transcript_is_rejected_and_chain_continues(tmp_path):
    cache = TranscriptCache(str(tmp_path))
    logs: list[str] = []
    short, full = Truncated("supadata"), Stub("whisper_local")
    full.transcribe = lambda r, l: Transcript(r.key, "whisper_local", "en", [Segment(0, 1040, "the whole talk")])
    tr = transcribe_with_chain(yt_ref("abcdefghijk"), [short, full], ["en"], cache, log=logs.append,
                               expected_duration=1043)
    assert tr.method == "whisper_local"
    assert not cache.has("abcdefghijk", "supadata")          # a truncated track is never cached
    assert any("supadata" in m and "covers only" in m for m in logs)


def test_truncated_transcript_from_the_only_provider_fails_the_source(tmp_path):
    cache = TranscriptCache(str(tmp_path))
    with pytest.raises(TranscriptionError, match="covers only 15s of 1043s"):
        transcribe_with_chain(yt_ref("abcdefghijk"), [Truncated("supadata")], ["en"], cache, log=lambda m: None,
                              expected_duration=1043)


def test_coverage_is_not_checked_without_a_known_duration(tmp_path):
    cache = TranscriptCache(str(tmp_path))
    tr = transcribe_with_chain(yt_ref("abcdefghijk"), [Truncated("supadata")], ["en"], cache, log=lambda m: None)
    assert tr.method == "supadata"                            # unknown duration: accepted as before


def test_coverage_tolerates_trailing_silence(tmp_path):
    cache = TranscriptCache(str(tmp_path))
    tr = transcribe_with_chain(yt_ref("abcdefghijk"), [Truncated("supadata", end=300.0)], ["en"], cache,
                               log=lambda m: None, expected_duration=500)
    assert tr.method == "supadata"                            # 60% coverage passes the 50% floor


def test_truncated_cached_transcript_is_ignored(tmp_path):
    cache = TranscriptCache(str(tmp_path))
    cache.save(Transcript("abcdefghijk", "supadata", "en", [Segment(6, 15, "stale truncated copy")]))
    full = Stub("youtube_captions")
    full.transcribe = lambda r, l: Transcript(r.key, "youtube_captions", "en", [Segment(0, 1040, "full")])
    tr = transcribe_with_chain(yt_ref("abcdefghijk"), [full, Truncated("supadata")], ["en"], cache, log=lambda m: None,
                               expected_duration=1043)
    assert tr.method == "youtube_captions"


def test_multiline_provider_errors_are_logged_as_one_line(tmp_path):
    class Chatty:
        name = "youtube_captions"
        kinds = frozenset(KINDS)

        def transcribe(self, ref, languages):
            raise TranscriptionError("youtube_captions: IpBlocked: \nCould not retrieve a transcript!\n\nlong advice")
    cache = TranscriptCache(str(tmp_path))
    logs: list[str] = []
    transcribe_with_chain(yt_ref("abcdefghijk"), [Chatty(), Stub("ytdlp_subs")], ["en"], cache, log=logs.append)
    assert logs[0] == "  [abcdefghijk] youtube_captions failed: youtube_captions: IpBlocked: ..."


# --- source kinds (MILESTONE-2 Track B) ------------------------------------------

class _K:
    def __init__(self, name, kinds):
        self.name, self.kinds, self.calls = name, frozenset(kinds), []

    def transcribe(self, ref, languages):
        self.calls.append(ref.key)
        return Transcript(ref.key, self.name, "en", [Segment(0, 1, "x")])


def test_chain_skips_transcribers_for_other_kinds(tmp_path):
    from pipeline.sources import SourceRef
    yt, sub = _K("youtube_captions", {"youtube"}), _K("ytdlp_subs", {"youtube", "video"})
    ref = SourceRef("video-0123456789abcdef", "video", "https://vimeo.com/1")
    t = transcribe_with_chain(ref, [yt, sub], ["en"], TranscriptCache(str(tmp_path)), log=lambda m: None)
    assert t.method == "ytdlp_subs" and yt.calls == [] and sub.calls == [ref.key]


def test_no_transcriber_for_kind_is_a_clear_error(tmp_path):
    from pipeline.sources import SourceRef
    ref = SourceRef("media-0123456789abcdef", "media", "file:///x.mp3")
    with pytest.raises(TranscriptionError, match="whisper_local") as e:
        transcribe_with_chain(ref, [_K("youtube_captions", {"youtube"})], ["en"], TranscriptCache(str(tmp_path)),
                              log=lambda m: None)
    assert "no transcriber handles media sources; configured: youtube_captions" in str(e.value)


def test_no_transcriber_for_web_kind_has_no_whisper_hint(tmp_path):
    from pipeline.sources import SourceRef
    ref = SourceRef("web-0123456789abcdef", "web", "https://example.com/a")
    with pytest.raises(TranscriptionError) as e:
        transcribe_with_chain(ref, [_K("youtube_captions", {"youtube"})], ["en"], TranscriptCache(str(tmp_path)),
                              log=lambda m: None)
    assert "no transcriber handles web sources" in str(e.value) and "whisper_local" not in str(e.value)


def test_provider_kinds():
    from pipeline.transcription.supadata import SupadataProvider
    from pipeline.transcription.whisper_local import WhisperLocalProvider
    from pipeline.transcription.youtube_captions import YouTubeCaptionsProvider
    from pipeline.transcription.ytdlp_subs import YtDlpSubsProvider
    assert YouTubeCaptionsProvider.kinds == SupadataProvider.kinds == frozenset({"youtube"})
    assert YtDlpSubsProvider.kinds == frozenset({"youtube", "video"})
    assert WhisperLocalProvider.kinds == frozenset({"youtube", "video", "media"})
