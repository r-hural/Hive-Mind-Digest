from tests.helpers import yt_ref
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from pipeline.transcription.base import TranscriptionError, Transcriber
from pipeline.transcription.youtube_captions import YouTubeCaptionsProvider


def _fetched(snippets, lang="en", generated=False):
    return SimpleNamespace(snippets=[SimpleNamespace(text=t, start=s, duration=d) for t, s, d in snippets],
                           language_code=lang, is_generated=generated, video_id="abcdefghijk")


def test_returns_segments_with_times_and_language():
    p = YouTubeCaptionsProvider()
    with patch("pipeline.transcription.youtube_captions.YouTubeTranscriptApi.fetch",
               return_value=_fetched([("Hello", 0.0, 1.5), ("world", 1.5, 2.0)], "en", True)) as fetch:
        tr = p.transcribe(yt_ref("abcdefghijk"), ["en", "de"])
    assert fetch.call_args.args == ("abcdefghijk",)
    assert fetch.call_args.kwargs["languages"] == ["en", "de"]
    assert [(s.start, s.end, s.text) for s in tr.segments] == [(0.0, 1.5, "Hello"), (1.5, 3.5, "world")]
    assert tr.method == "youtube_captions" and tr.language == "en" and tr.extra["is_generated"] is True
    assert isinstance(p, Transcriber)


def test_any_library_error_becomes_transcription_error():
    from youtube_transcript_api._errors import TranscriptsDisabled
    p = YouTubeCaptionsProvider()
    with patch("pipeline.transcription.youtube_captions.YouTubeTranscriptApi.fetch",
               side_effect=TranscriptsDisabled("abcdefghijk")):
        with pytest.raises(TranscriptionError, match="youtube_captions"):
            p.transcribe(yt_ref("abcdefghijk"), ["en"])
    with patch("pipeline.transcription.youtube_captions.YouTubeTranscriptApi.fetch",
               side_effect=RuntimeError("ip blocked")):
        with pytest.raises(TranscriptionError):
            p.transcribe(yt_ref("abcdefghijk"), ["en"])


def test_empty_track_is_an_error():
    p = YouTubeCaptionsProvider()
    with patch("pipeline.transcription.youtube_captions.YouTubeTranscriptApi.fetch", return_value=_fetched([])):
        with pytest.raises(TranscriptionError, match="empty"):
            p.transcribe(yt_ref("abcdefghijk"), ["en"])


def test_whitespace_only_track_is_an_error():
    p = YouTubeCaptionsProvider()
    with patch("pipeline.transcription.youtube_captions.YouTubeTranscriptApi.fetch",
               return_value=_fetched([("\n", 0.0, 1.0), ("  ", 1.0, 1.0)])):
        with pytest.raises(TranscriptionError, match="empty"):
            p.transcribe(yt_ref("abcdefghijk"), ["en"])


def test_malformed_snippet_becomes_transcription_error():
    p = YouTubeCaptionsProvider()
    with patch("pipeline.transcription.youtube_captions.YouTubeTranscriptApi.fetch",
               return_value=_fetched([("Hello", 0.0, None)])):
        with pytest.raises(TranscriptionError, match="youtube_captions"):
            p.transcribe(yt_ref("abcdefghijk"), ["en"])


def test_old_library_shape_becomes_transcription_error():
    p = YouTubeCaptionsProvider()
    with patch("pipeline.transcription.youtube_captions.YouTubeTranscriptApi.fetch",
               return_value=[{"text": "Hello", "start": 0.0, "duration": 1.0}]):
        with pytest.raises(TranscriptionError, match="youtube_captions"):
            p.transcribe(yt_ref("abcdefghijk"), ["en"])


def test_missing_language_code_defaults_to_und():
    p = YouTubeCaptionsProvider()
    with patch("pipeline.transcription.youtube_captions.YouTubeTranscriptApi.fetch",
               return_value=_fetched([("Hello", 0.0, 1.0)], lang=None)):
        tr = p.transcribe(yt_ref("abcdefghijk"), ["en"])
    assert tr.language == "und"


def test_non_youtube_ref_is_refused():
    from pipeline.sources import SourceRef
    with pytest.raises(TranscriptionError, match="cannot handle video"):
        YouTubeCaptionsProvider().transcribe(SourceRef("video-0123456789abcdef", "video", "https://vimeo.com/1"), ["en"])
