import pytest

from pipeline import db
from pipeline.sources import SourceRef
from pipeline.stages import fetch
from pipeline.transcription import local_text
from pipeline.transcription.base import TranscriptionError
from pipeline.transcription.local_text import LocalTextExtractor

KEY = "text-0123456789abcdef"


def _run(f):
    return LocalTextExtractor().transcribe(SourceRef(KEY, "text", f.resolve().as_uri()), ["en"])


def test_markdown_headings_start_segments(tmp_path):
    f = tmp_path / "n.md"
    f.write_text("# Habits\n\n" + "Tiny actions add up. " * 30 + "\n\n## Caveat\n\nIt takes months.\n", encoding="utf-8")
    t = _run(f)
    texts = [s.text for s in t.segments]
    assert texts[0] == "# Habits" and "## Caveat" in texts and t.extra["title"] == "Habits"
    assert all(len(s.text.split()) <= 40 for s in t.segments)
    assert t.extra["kind"] == "text" and t.method == "local_text"
    assert all(s.page is None and s.start == s.end == 0.0 for s in t.segments)


def test_code_fence_is_one_segment_and_hides_headings(tmp_path):
    f = tmp_path / "n.md"
    f.write_text("Intro words " * 10 + "\n\n```py\n# not a heading\n\nx = 1\n```\n\n" + "More words here. " * 5 + "\n",
                 encoding="utf-8")
    t = _run(f)
    fences = [s.text for s in t.segments if s.text.startswith("```")]
    assert len(fences) == 1 and "# not a heading" in fences[0] and "x = 1" in fences[0]
    assert t.extra["title"] == "n"


def test_plain_text_has_no_heading_handling_and_stem_title(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("# not markdown here\nline two of the first paragraph\n\n" + "Second paragraph. " * 10, encoding="utf-8")
    t = _run(f)
    assert [s.text for s in t.segments][0] == "# not markdown here line two of the first paragraph"
    assert len(t.segments) == 2 and t.extra["title"] == "notes"


def test_cp1252_fallback_and_bom(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(("caf\xe9 " * 30).encode("cp1252"))
    assert "caf\xe9" in _run(f).segments[0].text
    g = tmp_path / "b.txt"
    g.write_bytes(b"\xef\xbb\xbf" + ("word " * 30).encode())
    assert _run(g).segments[0].text.startswith("word")


def test_almost_empty_file_is_an_error(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("just a few words", encoding="utf-8")
    with pytest.raises(TranscriptionError, match="almost no text"):
        _run(f)


def test_oversized_and_irregular_files_are_refused(tmp_path, monkeypatch):
    f = tmp_path / "a.txt"
    f.write_text("word " * 30, encoding="utf-8")
    monkeypatch.setattr(local_text, "MAX_TEXT_BYTES", 10)
    with pytest.raises(TranscriptionError, match="larger than"):
        _run(f)
    with pytest.raises(TranscriptionError, match="not a regular file"):
        _run(tmp_path)
    with pytest.raises(TranscriptionError, match="not a regular file"):
        _run(tmp_path / "missing.txt")


def test_symlink_is_refused(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("word " * 30, encoding="utf-8")
    link = tmp_path / "l.txt"
    try:
        link.symlink_to(f)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(TranscriptionError, match="not a regular file"):
        local_text.read_text_file(str(link))


def test_text_metadata_title_is_first_heading_or_stem(tmp_path):
    path = str(tmp_path / "state.sqlite")
    db.init_db(path)
    nb = db.create_notebook(path, "T", "t")
    f = tmp_path / "n.md"
    f.write_text("# Habits\n\nbody words\n", encoding="utf-8")
    sid = db.add_source(path, nb, KEY, f.resolve().as_uri(), kind="text")
    fetch.fetch_source_metadata(path, sid, log=lambda m: None)
    assert db.get_source(path, sid)["title"] == "Habits"
    g = tmp_path / "plain.txt"
    g.write_text("body words\n", encoding="utf-8")
    sid2 = db.add_source(path, nb, "text-1123456789abcdef", g.resolve().as_uri(), kind="text")
    fetch.fetch_source_metadata(path, sid2, log=lambda m: None)
    assert db.get_source(path, sid2)["title"] == "plain"


def test_word_cap_is_refused(tmp_path, monkeypatch):
    f = tmp_path / "a.txt"
    f.write_text("word " * 30, encoding="utf-8")
    monkeypatch.setattr(local_text, "MAX_SOURCE_WORDS", 10)
    with pytest.raises(TranscriptionError, match="more than 10 words"):
        _run(f)


def test_unterminated_fence_keeps_text_and_inner_fence_does_not_close(tmp_path):
    f = tmp_path / "n.md"
    f.write_text("````\ncode ``` still inside\n" + "word " * 30 + "\n", encoding="utf-8")
    t = _run(f)
    assert len(t.segments) >= 1 and "still inside" in t.segments[0].text and t.segments[0].text.startswith("````")
    g = tmp_path / "m.md"
    g.write_text("```\nx = 1 ```\n```\n\nAfter the fence " + "word " * 25 + "\n", encoding="utf-8")
    texts = [s.text for s in _run(g).segments]
    assert texts[0] == "``` x = 1 ``` ```" and texts[1].startswith("After the fence")


def test_heading_parse_is_linear_on_a_run_of_hashes():
    import time
    t0 = time.perf_counter()
    local_text.parse_text("# " + "#" * 200_000 + "x", markdown=True)
    assert time.perf_counter() - t0 < 2.0


def test_heading_closing_sequence_and_levels():
    segs, first = local_text.parse_text("## Title ##\n\nbody words\n\n####### seven\n\n#nospace", markdown=True)
    assert first == "Title"
    assert segs[0] == "## Title"
    assert not any(s.startswith("####### ") and s != "####### seven" for s in segs)
