from tests.helpers import yt_ref
import json
import os
import re
from unittest.mock import patch

import pytest

from pipeline.transcription.base import Transcriber, TranscriptionError
from pipeline.transcription.ytdlp_subs import YtDlpSubsProvider, parse_json3

VIDEO_ID = "abcdefghijk"

JSON3 = {"events": [
    {"tStartMs": 0, "dDurationMs": 1200, "segs": [{"utf8": "Hello "}, {"utf8": "world"}]},
    {"tStartMs": 1200, "dDurationMs": 800, "segs": [{"utf8": "\n"}]},          # empty: skipped
    {"tStartMs": 2000, "dDurationMs": 1000},                                     # no segs: skipped
    {"tStartMs": 3000, "dDurationMs": 500, "segs": [{"utf8": "bye"}]},
]}


def test_parse_json3():
    segs = parse_json3(JSON3)
    assert [(s.start, s.end, s.text) for s in segs] == [(0.0, 1.2, "Hello world"), (3.0, 3.5, "bye")]


def test_is_a_transcriber():
    assert isinstance(YtDlpSubsProvider(), Transcriber)


def make_fake_ydl(manual=None, auto=None, calls=None, download_error=None, payload=None):
    """Factory returning a fresh FakeYDL class (and its call log) per test -- no shared
    mutable class state. Mimics the real YoutubeDL.extract_info(process=False) / mutable
    .params / process_ie_result(download=True) sequence used by the provider."""
    manual = dict(manual or {})
    auto = dict(auto or {})
    calls = calls if calls is not None else []
    payload = payload if payload is not None else JSON3

    class FakeYDL:
        def __init__(self, opts):
            self.params = opts          # same object the provider mutates via ydl.params[...]
            calls.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False, process=True):
            return {"id": VIDEO_ID, "subtitles": dict(manual), "automatic_captions": dict(auto)}

        def process_ie_result(self, info, download=True):
            if download_error is not None:
                raise download_error
            patterns = [re.compile(p, re.I) for p in self.params.get("subtitleslangs", [])]
            d = os.path.dirname(self.params["outtmpl"])
            for key in {**manual, **auto}:
                if any(p.fullmatch(key) for p in patterns):
                    with open(os.path.join(d, f"{info.get('id', VIDEO_ID)}.{key}.json3"), "w", encoding="utf-8") as f:
                        json.dump(payload, f)

    return FakeYDL, calls


def test_manual_track_preferred_over_auto_original():
    FakeYDL, calls = make_fake_ydl(manual={"en": [{"ext": "json3"}]}, auto={"en-orig": [{"ext": "json3"}]})
    with patch("pipeline.transcription.ytdlp_subs.yt_dlp.YoutubeDL", FakeYDL):
        tr = YtDlpSubsProvider().transcribe(yt_ref(VIDEO_ID), ["en"])
    assert tr.method == "ytdlp_subs"
    assert tr.language == "en"
    assert tr.extra["is_generated"] is False
    assert tr.extra["track"] == "en"
    assert len(tr.segments) == 2
    opts = calls[0]
    assert opts["skip_download"] is True and opts["writesubtitles"] and opts["writeautomaticsub"]
    assert opts["subtitlesformat"] == "json3"
    assert opts["subtitleslangs"] == ["en"]
    assert opts["ignore_no_formats_error"] is True


def test_manual_prefix_match_when_exact_missing():
    FakeYDL, _ = make_fake_ydl(manual={"en-US": [{"ext": "json3"}]})
    with patch("pipeline.transcription.ytdlp_subs.yt_dlp.YoutubeDL", FakeYDL):
        tr = YtDlpSubsProvider().transcribe(yt_ref(VIDEO_ID), ["en"])
    assert tr.language == "en-US"
    assert tr.extra["is_generated"] is False
    assert tr.extra["track"] == "en-US"


def test_auto_original_track_when_no_manual():
    FakeYDL, _ = make_fake_ydl(auto={"en-orig": [{"ext": "json3"}]})
    with patch("pipeline.transcription.ytdlp_subs.yt_dlp.YoutubeDL", FakeYDL):
        tr = YtDlpSubsProvider().transcribe(yt_ref(VIDEO_ID), ["en"])
    assert tr.language == "en"
    assert tr.extra["is_generated"] is True
    assert tr.extra["track"] == "en-orig"


def test_never_selects_a_plain_auto_translation():
    # German video: original auto track de-orig, auto de, and a machine-translated auto en.
    FakeYDL, _ = make_fake_ydl(auto={"de-orig": [{}], "de": [{}], "en": [{}]})
    with patch("pipeline.transcription.ytdlp_subs.yt_dlp.YoutubeDL", FakeYDL):
        with pytest.raises(TranscriptionError, match="no subtitle track"):
            YtDlpSubsProvider().transcribe(yt_ref(VIDEO_ID), ["en"])


def test_plain_auto_track_accepted_when_asr_is_not_translatable():
    # No "-orig" key exists anywhere in auto: the ASR track wasn't translatable, so yt-dlp
    # registered only the plain "en" key -- safe to use as the original track.
    FakeYDL, _ = make_fake_ydl(auto={"en": [{"ext": "json3"}]})
    with patch("pipeline.transcription.ytdlp_subs.yt_dlp.YoutubeDL", FakeYDL):
        tr = YtDlpSubsProvider().transcribe(yt_ref(VIDEO_ID), ["en"])
    assert tr.language == "en"
    assert tr.extra["is_generated"] is True
    assert tr.extra["track"] == "en"


def test_falls_through_language_preference_order():
    FakeYDL, _ = make_fake_ydl(manual={"de": [{"ext": "json3"}]})
    with patch("pipeline.transcription.ytdlp_subs.yt_dlp.YoutubeDL", FakeYDL):
        tr = YtDlpSubsProvider().transcribe(yt_ref(VIDEO_ID), ["en", "de"])
    assert tr.language == "de"


def test_no_track_at_all_is_an_error():
    FakeYDL, _ = make_fake_ydl()
    with patch("pipeline.transcription.ytdlp_subs.yt_dlp.YoutubeDL", FakeYDL):
        with pytest.raises(TranscriptionError, match="no subtitle track"):
            YtDlpSubsProvider().transcribe(yt_ref(VIDEO_ID), ["en"])


def test_ytdlp_exception_is_transcription_error():
    FakeYDL, _ = make_fake_ydl(manual={"en": [{"ext": "json3"}]}, download_error=RuntimeError("HTTP 403"))
    with patch("pipeline.transcription.ytdlp_subs.yt_dlp.YoutubeDL", FakeYDL):
        with pytest.raises(TranscriptionError, match="403"):
            YtDlpSubsProvider().transcribe(yt_ref(VIDEO_ID), ["en"])


def test_invalid_json_file_is_a_transcription_error():
    class FakeYDL:
        def __init__(self, opts):
            self.params = opts

        def __enter__(self): return self
        def __exit__(self, *a): return False

        def extract_info(self, url, download=False, process=True):
            return {"id": VIDEO_ID, "subtitles": {"en": [{}]}, "automatic_captions": {}}

        def process_ie_result(self, info, download=True):
            d = os.path.dirname(self.params["outtmpl"])
            with open(os.path.join(d, f"{VIDEO_ID}.en.json3"), "w", encoding="utf-8") as f:
                f.write("{not valid json")

    with patch("pipeline.transcription.ytdlp_subs.yt_dlp.YoutubeDL", FakeYDL):
        with pytest.raises(TranscriptionError):
            YtDlpSubsProvider().transcribe(yt_ref(VIDEO_ID), ["en"])


def test_non_numeric_timestamp_is_a_transcription_error():
    bad_json3 = {"events": [{"tStartMs": None, "dDurationMs": 1000, "segs": [{"utf8": "hi"}]}]}
    FakeYDL, _ = make_fake_ydl(manual={"en": [{"ext": "json3"}]}, payload=bad_json3)
    with patch("pipeline.transcription.ytdlp_subs.yt_dlp.YoutubeDL", FakeYDL):
        with pytest.raises(TranscriptionError):
            YtDlpSubsProvider().transcribe(yt_ref(VIDEO_ID), ["en"])


def test_no_file_written_is_a_transcription_error_naming_the_chosen_key():
    class FakeYDL:
        def __init__(self, opts):
            self.params = opts

        def __enter__(self): return self
        def __exit__(self, *a): return False

        def extract_info(self, url, download=False, process=True):
            return {"id": VIDEO_ID, "subtitles": {"en": [{}]}, "automatic_captions": {}}

        def process_ie_result(self, info, download=True):
            pass  # selects "en" but writes nothing -- simulates a silent no-op download

    with patch("pipeline.transcription.ytdlp_subs.yt_dlp.YoutubeDL", FakeYDL):
        with pytest.raises(TranscriptionError, match=r"no subtitle track written.*\(en\)"):
            YtDlpSubsProvider().transcribe(yt_ref(VIDEO_ID), ["en"])


def test_temp_directory_is_cleaned_up_after_success():
    captured_dir = {}

    class FakeYDL:
        def __init__(self, opts):
            self.params = opts
            captured_dir["path"] = os.path.dirname(opts["outtmpl"])

        def __enter__(self): return self
        def __exit__(self, *a): return False

        def extract_info(self, url, download=False, process=True):
            return {"id": VIDEO_ID, "subtitles": {"en": [{}]}, "automatic_captions": {}}

        def process_ie_result(self, info, download=True):
            d = os.path.dirname(self.params["outtmpl"])
            with open(os.path.join(d, f"{VIDEO_ID}.en.json3"), "w", encoding="utf-8") as f:
                json.dump(JSON3, f)

    with patch("pipeline.transcription.ytdlp_subs.yt_dlp.YoutubeDL", FakeYDL):
        YtDlpSubsProvider().transcribe(yt_ref(VIDEO_ID), ["en"])
    assert captured_dir["path"]
    assert not os.path.isdir(captured_dir["path"])


def test_options_never_expand_playlists():
    FakeYDL, calls = make_fake_ydl(manual={"en": [{"ext": "json3"}]}, auto={})
    seen = []
    orig_init = FakeYDL.__init__
    def init(self, opts, *a, **k):
        seen.append(opts)
        orig_init(self, opts, *a, **k)
    FakeYDL.__init__ = init
    with patch("pipeline.transcription.ytdlp_subs.yt_dlp.YoutubeDL", FakeYDL):
        YtDlpSubsProvider().transcribe(yt_ref(VIDEO_ID), ["en"])
    assert seen and all(o.get("noplaylist") is True for o in seen)
