import os
import pytest
from pipeline.sources import (SourceRef, classify_input, hashed_key, is_network_path, is_source_key,
                              normalize_url)


def test_youtube_url_keeps_its_video_id():
    ref = classify_input("https://youtu.be/dQw4w9WgXcQ")
    assert (ref.kind, ref.key, ref.url) == ("youtube", "dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_other_urls_by_host_and_extension():
    assert classify_input("https://vimeo.com/123456").kind == "video"
    assert classify_input("https://example.org/paper.pdf").kind == "pdf"
    ref = classify_input("https://example.org/blog/post/")
    assert ref.kind == "web" and ref.url == "https://example.org/blog/post" and is_source_key(ref.key)
    assert ref.key.startswith("web-") and len(ref.key) == 20


def test_same_page_twice_is_one_source():
    a = classify_input("https://Example.org/blog/post/?utm_source=x#intro")
    b = classify_input("https://example.org/blog/post")
    assert a.key == b.key and a.url == b.url


def test_local_files_by_extension(tmp_path):
    (tmp_path / "notes.md").write_text("# Notes\n\nSome text.", encoding="utf-8")
    (tmp_path / "talk.mp3").write_bytes(b"\x00" * 10)
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    md = classify_input("notes.md", base_dir=str(tmp_path))
    assert md.kind == "text" and md.url.startswith("file:") and md.path == str((tmp_path / "notes.md").resolve())
    assert classify_input(str(tmp_path / "talk.mp3")).kind == "media"
    assert classify_input(str(tmp_path / "paper.pdf")).kind == "pdf"


def test_same_file_content_is_one_source(tmp_path):
    (tmp_path / "a.txt").write_text("same", encoding="utf-8")
    (tmp_path / "b.txt").write_text("same", encoding="utf-8")
    assert classify_input(str(tmp_path / "a.txt")).key == classify_input(str(tmp_path / "b.txt")).key


@pytest.mark.parametrize("bad", ["ftp://example.org/x", "javascript:alert(1)", "file:///etc/passwd",
                                 "not a url or file", "missing.pdf"])
def test_unclassifiable_input_is_a_value_error(bad, tmp_path):
    with pytest.raises(ValueError):
        classify_input(bad, base_dir=str(tmp_path))


def test_unsupported_extension(tmp_path):
    (tmp_path / "x.docx").write_bytes(b"PK")
    with pytest.raises(ValueError, match="unsupported file type"):
        classify_input(str(tmp_path / "x.docx"))


def test_key_rules():
    assert is_source_key("dQw4w9WgXcQ") and is_source_key(hashed_key("pdf", "x"))
    assert not is_source_key("web-XYZ") and not is_source_key("../../etc") and not is_source_key("blog-0123456789abcdef")


def test_source_ref_from_row_defaults_to_youtube():
    ref = SourceRef.from_row({"video_id": "dQw4w9WgXcQ", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
    assert ref.kind == "youtube" and ref.path is None


def test_normalize_url_keeps_meaningful_query():
    assert normalize_url("https://e.org/a?id=3&utm_medium=m&fbclid=z") == "https://e.org/a?id=3"


def test_unc_uri_has_no_local_path():
    # a file: URL naming a server (stored before network shares were refused) is never opened
    import pathlib
    unc = chr(92) * 2 + "srv" + chr(92) + "share" + chr(92) + "a b.pdf"
    uri = pathlib.PureWindowsPath(unc).as_uri()
    assert SourceRef("pdf-0123456789abcdef", "pdf", uri).path is None
    # the server smuggled into the path part, with an empty netloc
    assert SourceRef("pdf-0123456789abcdef", "pdf", "file:////srv/share/a.pdf").path is None


B = chr(92)


@pytest.mark.parametrize("path,network", [
    (B * 2 + "srv" + B + "share" + B + "a.pdf", True),
    ("//srv/share/a.pdf", True),
    (B * 2 + "srv@SSL" + B + "DavWWWRoot" + B + "a.pdf", True),
    (B * 2 + "?" + B + "UNC" + B + "srv" + B + "share" + B + "a.pdf", True),
    (B * 2 + "?" + B + "C:" + B + "notes" + B + "a.md", True),       # device paths: refused too
    (B * 2 + "." + B + "C:" + B + ".." + B + "UNC" + B + "srv" + B + "s" + B + "a.pdf", True),
    (B + "??" + B + "UNC" + B + "srv" + B + "share" + B + "a.pdf", True),   # NT object path
    (B + "notes" + B + "a.md", True),                                      # rooted, no drive
    ("C:" + B + "notes" + B + "a.md", False),
    ("c:/notes/a.md", False),
])
def test_is_network_path_on_windows_allows_only_drive_paths(path, network):
    assert is_network_path(path, windows=True) is network


@pytest.mark.parametrize("path,network", [
    ("//srv/share/a.pdf", True),
    ("/srv/notes/a.md", False),
    ("/srv/notes/" + B + B + "odd name.md", False),     # a backslash is an ordinary character here
])
def test_is_network_path_elsewhere_refuses_a_leading_double_slash(path, network):
    assert is_network_path(path, windows=False) is network


def test_network_share_input_is_refused_before_any_file_access(monkeypatch):
    def no_access(*_a, **_k):
        raise AssertionError("the share was touched")
    monkeypatch.setattr("pipeline.sources.os.path.isfile", no_access)
    monkeypatch.setattr("pipeline.sources.os.path.getsize", no_access)
    texts = ["//attacker.example/s/doc.pdf"]
    if os.name == "nt":     # elsewhere a backslash is an ordinary file-name character
        texts.append(B * 2 + "attacker.example" + B + "s" + B + "doc.pdf")
    for text in texts:
        with pytest.raises(ValueError, match="network share"):
            classify_input(text)


def test_relative_path_from_a_list_on_a_share_is_refused(monkeypatch):
    monkeypatch.setattr("pipeline.sources.os.path.isfile", lambda *_a: True)
    with pytest.raises(ValueError, match="network share"):
        classify_input("doc.pdf", base_dir=B * 2 + "srv" + B + "share")


@pytest.mark.parametrize("a,b", [
    ("https://e.org:443/a", "https://e.org/a"),
    ("http://e.org:80/a", "http://e.org/a"),
    ("https://e.org/a%2fb", "https://e.org/a%2Fb"),
    ("https://U:Pw@E.org/a", "https://e.org/a"),
])
def test_normalize_url_one_form_per_page(a, b):
    assert normalize_url(a) == b


def test_normalize_url_keeps_non_default_port():
    assert normalize_url("https://e.org:8443/a") == "https://e.org:8443/a"


def test_video_host_matches_with_port():
    assert classify_input("https://vimeo.com:443/1").kind == "video"


@pytest.mark.parametrize("q", ['"{}"', "'{}'"])
def test_quoted_line_is_unquoted(q, tmp_path):
    f = tmp_path / "n.txt"
    f.write_text("x", encoding="utf-8")
    assert classify_input(q.format(f)).kind == "text"
    assert classify_input(q.format("https://example.org/a")).kind == "web"


def test_drive_letter_path_is_a_file(tmp_path):
    f = tmp_path / "n.txt"
    f.write_text("x", encoding="utf-8")
    assert classify_input(str(f).replace("/", os.sep)).kind == "text"


def test_cache_rejects_traversal(tmp_path):
    from pipeline.transcription.cache import TranscriptCache
    with pytest.raises(ValueError):
        TranscriptCache(str(tmp_path)).path("../x", "m")


@pytest.mark.parametrize("url", ["https://www.youtube.com/playlist?list=PL0123456789abcdef",
                                 "https://www.youtube.com/@somechannel", "https://youtu.be/",
                                 "https://www.youtube.com/watch?v=tooShort", "https://m.youtube.com/channel/UC0123"])
def test_youtube_urls_that_are_not_one_video_are_refused(url):
    with pytest.raises(ValueError, match="not a single YouTube video; use --playlist for playlists"):
        classify_input(url)


@pytest.mark.parametrize("name, kind", [("big.txt", "text"), ("big.pdf", "pdf"), ("big.mp3", "media")])
def test_file_over_its_kind_cap_is_refused_before_hashing(tmp_path, monkeypatch, name, kind):
    import pipeline.sources as sources
    p = tmp_path / name
    with open(p, "wb") as f:
        f.truncate(sources.MAX_FILE_BYTES[kind] + 1)        # sparse where the filesystem allows it

    def no_hash(path):
        raise AssertionError("hashed an oversized file")
    monkeypatch.setattr(sources, "file_identity", no_hash)
    with pytest.raises(ValueError, match="larger than"):
        classify_input(str(p))


def test_file_at_its_cap_is_accepted(tmp_path, monkeypatch):
    import pipeline.sources as sources
    monkeypatch.setitem(sources.MAX_FILE_BYTES, "text", 10)
    (tmp_path / "a.txt").write_bytes(b"x" * 10)
    assert classify_input(str(tmp_path / "a.txt")).kind == "text"
    (tmp_path / "b.txt").write_bytes(b"x" * 11)
    with pytest.raises(ValueError, match="larger than 10 bytes"):
        classify_input(str(tmp_path / "b.txt"))


def test_relative_path_resolves_against_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / "notes.md").write_text("# Notes\n\nSome text.", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    ref = classify_input("notes.md")
    assert ref.kind == "text" and ref.path == str((tmp_path / "notes.md").resolve())
