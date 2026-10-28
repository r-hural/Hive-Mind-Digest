from unittest.mock import patch

import pytest

from pipeline import db
from pipeline.stages.fetch import (
    canonical_url,
    fetch_source_metadata,
    fetch_stage,
    parse_playlist_url,
    parse_youtube_url,
    read_url_file,
    register_sources,
    resolve_playlist,
)

VID = "dQw4w9WgXcQ"
KELVIN = "K"   # Kelvin sign; lowercases to plain "k"


@pytest.mark.parametrize("url", [
    f"https://youtu.be/{VID}", f"https://www.youtube.com/watch?v={VID}", f"https://youtube.com/watch?v={VID}&t=30",
    f"https://m.youtube.com/watch?feature=share&v={VID}", f"https://www.youtube.com/embed/{VID}",
    f"https://www.youtube.com/shorts/{VID}", f"https://www.youtube.com/live/{VID}?si=abc", f"  https://youtu.be/{VID}  ",
    f"https://www.youtube.com/watch/?v={VID}",   # trailing slash on /watch
    f"https://www.youtube-nocookie.com/embed/{VID}",
    f"https://youtube-nocookie.com/embed/{VID}",   # bare nocookie host, no www.
    f"https://WWW.YOUTUBE.COM/watch?v={VID}",    # host is case-insensitive
    f"https://music.youtube.com/watch?v={VID}",
])
def test_parse_valid(url):
    assert parse_youtube_url(url) == VID


@pytest.mark.parametrize("url", [
    f"https://evil.example/youtube.com/watch?v={VID}",          # substring attack (IDEA-206)
    f"https://youtube.com.evil.example/watch?v={VID}",
    f"ftp://www.youtube.com/watch?v={VID}",
    "https://www.youtube.com/watch?v=short", "https://www.youtube.com/playlist?list=PL123", "not a url",
    f"https://youtube.com@evil.example/watch?v={VID}",          # userinfo trick: real host is evil.example
    "https://www.youtube.com/watch",                             # no ?v= at all
    f"https://www.youtube.com/watch?v={VID}%0A",                 # trailing control char breaks the id shape
    f"https://www.youtube-nocoo{KELVIN}ie.com/embed/{VID}",      # Kelvin sign lowercases into the allowlist
    f"https://youtube-nocoo{KELVIN}ie.com/embed/{VID}",          # same trick against the bare nocookie host
    f"https://youtube-nocookie.com.evil.example/embed/{VID}",    # lookalike: suffix after the bare host
    f"https://evilyoutube-nocookie.com/embed/{VID}",             # lookalike: prefix glued onto the bare host
    f"https://youtube-nocookie.co/embed/{VID}",                  # lookalike: wrong TLD
])
def test_parse_invalid(url):
    with pytest.raises(ValueError):
        parse_youtube_url(url)


def test_canonical_url():
    assert canonical_url(VID) == f"https://www.youtube.com/watch?v={VID}"


def test_canonical_url_validates_video_id():
    with pytest.raises(ValueError):
        canonical_url("../evil")


def test_parse_playlist_url():
    assert parse_playlist_url("https://www.youtube.com/playlist?list=PLabc_DEF-123") == "PLabc_DEF-123"
    assert parse_playlist_url(f"https://www.youtube.com/watch?v={VID}&list=PLabc_DEF-123") == "PLabc_DEF-123"
    with pytest.raises(ValueError):
        parse_playlist_url("https://evil.example/playlist?list=PLabc")


def test_parse_playlist_url_rejects_non_ascii_host():
    with pytest.raises(ValueError):
        parse_playlist_url(f"https://www.youtube-nocoo{KELVIN}ie.com/playlist?list=PLabcdefghij")


def test_read_url_file(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text(f"# comment\n\nhttps://youtu.be/{VID}\n  https://youtu.be/{VID}  \n", encoding="utf-8")
    assert read_url_file(str(f)) == [f"https://youtu.be/{VID}"] * 2


def test_read_url_file_tolerates_bom_and_crlf(tmp_path):
    f = tmp_path / "u.txt"
    f.write_bytes((f"https://youtu.be/{VID}\r\n# comment\r\n").encode("utf-8-sig"))
    assert read_url_file(str(f)) == [f"https://youtu.be/{VID}"]


class FakeYDL:
    info = {}
    calls = []
    last_url = None
    def __init__(self, opts): self.opts = opts; FakeYDL.calls.append(opts)
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def extract_info(self, url, download=False): FakeYDL.last_url = url; return FakeYDL.info


@pytest.fixture(autouse=True)
def _reset_fake_ydl():
    FakeYDL.info = {}
    FakeYDL.calls = []
    FakeYDL.last_url = None
    yield


def test_resolve_playlist_is_flat_and_canonical():
    FakeYDL.info = {"entries": [
        {"id": VID, "title": "One", "uploader": "Chan", "duration": 61},
        {"id": "aaaaaaaaaaa", "title": "Two", "channel": "Chan2", "duration": None},
        {"id": None}, {"id": "bad"}, None,
    ]}
    with patch("pipeline.stages.fetch.yt_dlp.YoutubeDL", FakeYDL):
        entries = resolve_playlist("https://www.youtube.com/playlist?list=PLabcdefghij&si=tracking")
    opts = FakeYDL.calls[0]
    assert opts["extract_flat"] is True and opts["playlist_items"] == "1:200"
    assert "playlistend" not in opts and "playliststart" not in opts   # deprecated in yt-dlp
    assert FakeYDL.last_url == "https://www.youtube.com/playlist?list=PLabcdefghij"   # rebuilt, no tracking params
    assert entries == [
        {"video_id": VID, "title": "One", "channel": "Chan", "duration_seconds": 61},
        {"video_id": "aaaaaaaaaaa", "title": "Two", "channel": "Chan2", "duration_seconds": None},
    ]


def test_resolve_playlist_caps_entries_even_if_ytdlp_ignores_the_limit():
    FakeYDL.info = {"entries": [{"id": f"{i:011d}"} for i in range(5)]}
    with patch("pipeline.stages.fetch.yt_dlp.YoutubeDL", FakeYDL):
        entries = resolve_playlist("https://www.youtube.com/playlist?list=PLabcdefghij", limit=3)
    assert FakeYDL.calls[0]["playlist_items"] == "1:3"
    assert [e["video_id"] for e in entries] == [f"{i:011d}" for i in range(3)]


def test_parse_playlist_url_accepts_bare_nocookie_host():
    assert parse_playlist_url("https://youtube-nocookie.com/playlist?list=PLabcdefghij") == "PLabcdefghij"


def test_resolve_playlist_raises_on_no_entries():
    FakeYDL.info = None
    with patch("pipeline.stages.fetch.yt_dlp.YoutubeDL", FakeYDL):
        with pytest.raises(ValueError, match="PLabcdefghij"):
            resolve_playlist("https://www.youtube.com/playlist?list=PLabcdefghij")


def test_resolve_playlist_raises_when_no_usable_entries():
    FakeYDL.info = {"entries": [{"id": None}, {"id": "bad"}, None, "not-a-dict"]}
    with patch("pipeline.stages.fetch.yt_dlp.YoutubeDL", FakeYDL):
        with pytest.raises(ValueError, match="PLabcdefghij"):
            resolve_playlist("https://www.youtube.com/playlist?list=PLabcdefghij")


def test_resolve_playlist_passes_a_logger_that_can_error():
    FakeYDL.info = {"entries": [{"id": VID}]}
    with patch("pipeline.stages.fetch.yt_dlp.YoutubeDL", FakeYDL):
        resolve_playlist("https://www.youtube.com/playlist?list=PLabcdefghij")
    opts = FakeYDL.calls[0]
    assert "logger" in opts and hasattr(opts["logger"], "error")
    opts["logger"].error("boom")   # must not raise


@pytest.fixture
def state(tmp_path):
    path = str(tmp_path / "s.sqlite"); db.init_db(path)
    return path, db.create_notebook(path, "T", "t")


def test_register_sources_stores_canonical_url_and_is_idempotent(state):
    path, nid = state
    ids = register_sources(path, nid, [{"video_id": VID, "title": "One", "channel": "C", "duration_seconds": 5},
                                       {"video_id": VID}])
    assert ids == [1, 1]
    src = db.get_source(path, 1)
    assert src["url"] == canonical_url(VID) and src["title"] == "One" and src["fetch_status"] == "pending"


def test_register_sources_does_not_overwrite_existing_title(state):
    path, nid = state
    register_sources(path, nid, [{"video_id": VID, "title": "Original", "channel": "C", "duration_seconds": 5}])
    ids = register_sources(path, nid, [{"video_id": VID, "title": "Changed", "channel": "C2",
                                        "duration_seconds": 99}])
    assert ids == [1]
    src = db.get_source(path, 1)
    assert src["title"] == "Original" and src["channel"] == "C" and src["duration_seconds"] == 5


def test_fetch_stage_from_file_skips_bad_lines(state, tmp_path):
    path, nid = state
    f = tmp_path / "u.txt"
    f.write_text(f"https://youtu.be/{VID}\nnot-a-source\n", encoding="utf-8")
    msgs = []
    ids = fetch_stage(path, nid, input_file=str(f), log=msgs.append)
    assert ids == [1] and any("Skipping" in m for m in msgs)


def test_fetch_stage_single_url(state):
    path, nid = state
    assert fetch_stage(path, nid, url=f"https://youtu.be/{VID}", log=lambda m: None) == [1]


def test_fetch_stage_dedupes_returned_ids(state, tmp_path):
    path, nid = state
    f = tmp_path / "u.txt"
    f.write_text(f"https://youtu.be/{VID}\nhttps://www.youtube.com/watch?v={VID}\n", encoding="utf-8")
    ids = fetch_stage(path, nid, input_file=str(f), log=lambda m: None)
    assert ids == [1]


def test_fetch_stage_requires_an_input(state):
    path, nid = state
    with pytest.raises(ValueError):
        fetch_stage(path, nid, log=lambda m: None)


def test_fetch_stage_rejects_multiple_inputs(state, tmp_path):
    path, nid = state
    f = tmp_path / "u.txt"
    f.write_text(f"https://youtu.be/{VID}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        fetch_stage(path, nid, input_file=str(f), url=f"https://youtu.be/{VID}", log=lambda m: None)


def test_fetch_stage_raises_when_nothing_usable(state, tmp_path):
    path, nid = state
    f = tmp_path / "u.txt"
    f.write_text("not-a-source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no usable sources"):
        fetch_stage(path, nid, input_file=str(f), log=lambda m: None)


def test_metadata_worker_stores_upload_date(state):
    path, nid = state
    sid = register_sources(path, nid, [{"video_id": VID}])[0]
    FakeYDL.info = {"title": "Full", "uploader": "Chan", "duration": 300, "upload_date": "20260115"}
    with patch("pipeline.stages.fetch.yt_dlp.YoutubeDL", FakeYDL):
        assert fetch_source_metadata(path, sid, log=lambda m: None) == "done"
    assert FakeYDL.last_url == canonical_url(VID)          # never the user's string
    src = db.get_source(path, sid)
    assert src["title"] == "Full" and src["upload_date"] == "20260115" and src["fetch_status"] == "done"


def test_metadata_worker_logger_swallows_errors_to_avoid_double_logging(state):
    path, nid = state
    sid = register_sources(path, nid, [{"video_id": VID}])[0]
    FakeYDL.info = {"title": "Full", "uploader": "Chan", "duration": 300}
    msgs = []
    with patch("pipeline.stages.fetch.yt_dlp.YoutubeDL", FakeYDL):
        fetch_source_metadata(path, sid, log=msgs.append)
    opts = FakeYDL.calls[0]
    assert "logger" in opts and hasattr(opts["logger"], "error")
    opts["logger"].error("boom")   # must not raise, and must not reach our log (would double-log)
    assert msgs == []


def test_metadata_worker_treats_none_info_as_error(state):
    path, nid = state
    sid = register_sources(path, nid, [{"video_id": VID}])[0]
    FakeYDL.info = None
    with patch("pipeline.stages.fetch.yt_dlp.YoutubeDL", FakeYDL):
        assert fetch_source_metadata(path, sid, log=lambda m: None) == "error"
    assert db.get_source(path, sid)["error_message"]


def test_metadata_worker_skips_complete_rows_and_marks_errors(state):
    path, nid = state
    sid = register_sources(path, nid, [{"video_id": VID, "title": "T", "channel": "C", "duration_seconds": 1}])[0]
    db.update_source_metadata(path, sid, title=None, channel=None, duration_seconds=None, upload_date="20260101")
    assert fetch_source_metadata(path, sid, log=lambda m: None) == "skipped"
    sid2 = register_sources(path, nid, [{"video_id": "aaaaaaaaaaa"}])[0]
    class Boom(FakeYDL):
        def extract_info(self, url, download=False): raise RuntimeError("private video")
    with patch("pipeline.stages.fetch.yt_dlp.YoutubeDL", Boom):
        assert fetch_source_metadata(path, sid2, log=lambda m: None) == "error"
    assert "private video" in db.get_source(path, sid2)["error_message"]


def test_metadata_worker_error_does_not_clobber_existing_title(state):
    path, nid = state
    sid = register_sources(path, nid, [{"video_id": VID, "title": "Existing", "channel": "C",
                                        "duration_seconds": 1}])[0]
    class Boom(FakeYDL):
        def extract_info(self, url, download=False): raise RuntimeError("boom")
    with patch("pipeline.stages.fetch.yt_dlp.YoutubeDL", Boom):
        assert fetch_source_metadata(path, sid, log=lambda m: None) == "error"
    src = db.get_source(path, sid)
    assert src["title"] == "Existing" and src["channel"] == "C" and src["duration_seconds"] == 1
