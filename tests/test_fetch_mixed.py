import pytest

from pipeline import db
from pipeline.stages.fetch import fetch_stage


def _nb(tmp_path):
    p = str(tmp_path / "s.sqlite")
    db.init_db(p)
    return p, db.create_notebook(p, "T", "t")


def test_input_file_with_mixed_lines(tmp_path):
    p, nid = _nb(tmp_path)
    (tmp_path / "notes.md").write_text("# N\n\ntext", encoding="utf-8")
    lst = tmp_path / "in.txt"
    lst.write_text("https://youtu.be/dQw4w9WgXcQ\nhttps://example.org/post\nnotes.md\n"
                   "https://example.org/post/?utm_source=x\nnot-a-thing\n", encoding="utf-8")
    logs = []
    ids = fetch_stage(p, nid, input_file=str(lst), log=logs.append)
    kinds = sorted(s["kind"] for s in db.get_notebook_sources(p, nid))
    assert kinds == ["text", "web", "youtube"] and len(ids) == 3
    assert any("Skipping 'not-a-thing'" in line for line in logs)


def test_single_file_option(tmp_path):
    p, nid = _nb(tmp_path)
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4\n")
    fetch_stage(p, nid, file_path=str(tmp_path / "a.pdf"), log=lambda m: None)
    (s,) = db.get_notebook_sources(p, nid)
    assert s["kind"] == "pdf" and s["url"].startswith("file:") and s["video_id"].startswith("pdf-")


def test_more_than_one_input_is_rejected(tmp_path):
    p, nid = _nb(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        fetch_stage(p, nid, url="https://e.org", file_path="x.pdf")


def test_no_input_is_rejected(tmp_path):
    p, nid = _nb(tmp_path)
    with pytest.raises(ValueError, match="no input"):
        fetch_stage(p, nid)


def _fake_ydl(seen, info):
    class _Ydl:
        def __init__(self, opts):
            seen["opts"] = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            seen["url"] = url
            return info
    return _Ydl


def test_video_site_metadata_uses_the_real_url(tmp_path, monkeypatch):
    from pipeline.stages import fetch
    seen = {}
    monkeypatch.setattr(fetch.yt_dlp, "YoutubeDL", _fake_ydl(
        seen, {"title": "A Vimeo talk", "uploader": "Studio", "duration": 600, "upload_date": "20260201"}))
    p = str(tmp_path / "s.sqlite"); db.init_db(p); nid = db.create_notebook(p, "T", "t")
    sid = db.add_source(p, nid, "video-0123456789abcdef", "https://vimeo.com/1", kind="video")
    assert fetch.fetch_source_metadata(p, sid, log=lambda m: None) == "done"
    s = db.get_source(p, sid)
    assert seen["url"] == "https://vimeo.com/1" and s["title"] == "A Vimeo talk"
    assert seen["opts"]["noplaylist"] is True and seen["opts"]["playlist_items"] == "1"
    assert s["channel"] == "Studio" and s["duration_seconds"] == 600 and s["upload_date"] == "20260201"


def test_local_media_metadata_is_the_file_stem_without_network(tmp_path, monkeypatch):
    from pipeline.stages import fetch

    def boom(*a, **k):
        raise AssertionError("media metadata must not touch the network")
    monkeypatch.setattr(fetch.yt_dlp, "YoutubeDL", boom)
    f = tmp_path / "my talk.mp3"; f.write_bytes(b"\x00")
    p = str(tmp_path / "s.sqlite"); db.init_db(p); nid = db.create_notebook(p, "T", "t")
    sid = db.add_source(p, nid, "media-0123456789abcdef", f.resolve().as_uri(), kind="media")
    assert fetch.fetch_source_metadata(p, sid, log=lambda m: None) == "done"
    s = db.get_source(p, sid)
    assert s["title"] == "my talk" and s["duration_seconds"] is None


def test_playlist_shaped_metadata_is_an_error_not_stored(tmp_path, monkeypatch):
    from pipeline.stages import fetch
    seen = {}
    monkeypatch.setattr(fetch.yt_dlp, "YoutubeDL", _fake_ydl(
        seen, {"_type": "playlist", "title": "An album", "entries": [{"id": "a"}]}))
    p = str(tmp_path / "s.sqlite"); db.init_db(p); nid = db.create_notebook(p, "T", "t")
    sid = db.add_source(p, nid, "video-0123456789abcdef", "https://vimeo.com/album/1", kind="video")
    assert fetch.fetch_source_metadata(p, sid, log=lambda m: None) == "error"
    s = db.get_source(p, sid)
    assert s["title"] is None and "URL is a playlist, not a single video" in s["error_message"]


def test_relative_file_and_url_options_resolve_against_the_working_directory(tmp_path, monkeypatch):
    p, nid = _nb(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.pdf").write_bytes(b"%PDF-1.4\n")
    (work / "notes.md").write_text("# N\n\ntext", encoding="utf-8")
    monkeypatch.chdir(work)
    fetch_stage(p, nid, file_path="a.pdf", log=lambda m: None)
    fetch_stage(p, nid, url="notes.md", log=lambda m: None)
    urls = sorted(s["url"] for s in db.get_notebook_sources(p, nid))
    assert urls == sorted([(work / "a.pdf").resolve().as_uri(), (work / "notes.md").resolve().as_uri()])


# A downloaded list must not pull in any local file (release security review, 2026-10-28): a local
# path in an --input list has to resolve inside the list's own folder; --file stays unrestricted.
def test_input_list_refuses_a_local_file_outside_its_folder(tmp_path):
    p, nid = _nb(tmp_path)
    (tmp_path / "secret.txt").write_text("private", encoding="utf-8")
    lists = tmp_path / "lists"
    lists.mkdir()
    (lists / "ok.md").write_text("# N\n\ntext", encoding="utf-8")
    for line in ("../secret.txt", str(tmp_path / "secret.txt")):
        lst = lists / "in.txt"
        lst.write_text(f"ok.md\n{line}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="outside the input list's folder"):
            fetch_stage(p, nid, input_file=str(lst), log=lambda m: None)
    assert db.get_notebook_sources(p, nid) == []          # nothing registered, not even ok.md


def test_input_list_accepts_local_files_in_its_folder_and_below(tmp_path):
    p, nid = _nb(tmp_path)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.md").write_text("# A\n\ntext", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\ntext", encoding="utf-8")
    lst = tmp_path / "in.txt"
    lst.write_text(f"sub/a.md\n{tmp_path / 'b.md'}\n", encoding="utf-8")
    assert len(fetch_stage(p, nid, input_file=str(lst), log=lambda m: None)) == 2


def test_file_option_may_name_any_local_file(tmp_path):
    p, nid = _nb(tmp_path)
    (tmp_path / "elsewhere.md").write_text("# E\n\ntext", encoding="utf-8")
    assert len(fetch_stage(p, nid, file_path=str(tmp_path / "elsewhere.md"), log=lambda m: None)) == 1
