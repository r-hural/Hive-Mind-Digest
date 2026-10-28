import pytest

from pipeline import db
from pipeline.sources import SourceRef
from pipeline.stages import fetch
from pipeline.transcription import pdf_text, web_text
from pipeline.transcription.base import TranscriptionError
from pipeline.transcription.pdf_text import PdfTextExtractor
from tests.pdf_fixtures import write_blank_pdf, write_text_pdf

KEY = "pdf-0123456789abcdef"


def _ref(p):
    return SourceRef(KEY, "pdf", p.resolve().as_uri())


def test_pages_become_segments_with_page_numbers(tmp_path):
    p = write_text_pdf(tmp_path / "a.pdf", ["Page one says habits compound. " * 10, "Page two adds a caveat. " * 10],
                       info={"Title": "Habit notes", "CreationDate": "D:20240102030405Z"})
    t = PdfTextExtractor().transcribe(_ref(p), ["en"])
    assert {s.page for s in t.segments} == {1, 2} and t.extra["pages"] == 2
    assert t.method == "pdf_text" and all(s.start == s.end == 0.0 for s in t.segments)
    assert "habits compound" in t.segments[0].text and "caveat" in t.segments[-1].text
    assert all(len(s.text.split()) <= 40 for s in t.segments)
    assert t.extra == {"kind": "pdf", "pages": 2, "title": "Habit notes", "date": "20240102"}


def test_missing_metadata_is_none(tmp_path):
    p = write_text_pdf(tmp_path / "a.pdf", ["Words about habits. " * 20])
    t = PdfTextExtractor().transcribe(_ref(p), ["en"])
    assert t.extra["title"] is None and t.extra["date"] is None


def test_scanned_pdf_is_a_clear_error(tmp_path):
    p = write_blank_pdf(tmp_path / "scan.pdf")
    with pytest.raises(TranscriptionError, match="OCR is not supported"):
        PdfTextExtractor().transcribe(_ref(p), ["en"])


def test_malformed_pdf_is_a_transcription_error(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.4\nthis is not a pdf at all\n")
    with pytest.raises(TranscriptionError, match="pdf_text: could not read PDF"):
        PdfTextExtractor().transcribe(_ref(p), ["en"])


def test_encrypted_pdf_is_a_clear_error(tmp_path, monkeypatch):
    p = write_text_pdf(tmp_path / "a.pdf", ["Words about habits. " * 20])

    class _R:
        is_encrypted = True
        pages: list = []

        def __init__(self, *a, **k):
            pass

        def decrypt(self, pw):
            return 0

    monkeypatch.setattr(pdf_text.pypdf, "PdfReader", _R)
    monkeypatch.setattr(pdf_text, "_extract_bounded", pdf_text._extract_in_process)   # the patch is in this process
    with pytest.raises(TranscriptionError, match="encrypted PDF"):
        PdfTextExtractor().transcribe(_ref(p), ["en"])


def test_missing_local_file_is_refused(tmp_path):
    with pytest.raises(TranscriptionError, match="not a regular file"):
        PdfTextExtractor().transcribe(_ref(tmp_path / "nope.pdf"), ["en"])


def test_oversized_local_pdf_is_refused(tmp_path, monkeypatch):
    p = write_text_pdf(tmp_path / "a.pdf", ["Words about habits. " * 20])
    monkeypatch.setattr(pdf_text, "MAX_PDF_BYTES", 100)
    with pytest.raises(TranscriptionError, match="larger than"):
        PdfTextExtractor().transcribe(_ref(p), ["en"])


class _Resp:
    """A streaming response: read(n) returns the next n bytes, like http.client's."""

    def __init__(self, ctype, body):
        self.headers, self._b, self._pos = {"Content-Type": ctype}, body, 0

    def read(self, n=-1):
        end = len(self._b) if n < 0 else self._pos + n
        chunk, self._pos = self._b[self._pos:end], min(end, len(self._b))
        return chunk

    def geturl(self):
        return "https://e.org/paper.pdf"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_remote_pdf_uses_the_guarded_opener(tmp_path, monkeypatch):
    data = write_text_pdf(tmp_path / "a.pdf", ["Remote words about habits. " * 20]).read_bytes()
    seen = []
    monkeypatch.setattr(web_text, "_open", lambda req, timeout: seen.append(req.full_url) or _Resp("application/pdf", data))
    t = PdfTextExtractor().transcribe(SourceRef(KEY, "pdf", "https://e.org/paper.pdf"), ["en"])
    assert seen == ["https://e.org/paper.pdf"] and t.extra["pages"] == 1


def test_octet_stream_is_accepted_only_for_pdf_urls(tmp_path, monkeypatch):
    data = write_text_pdf(tmp_path / "a.pdf", ["Remote words about habits. " * 20]).read_bytes()
    monkeypatch.setattr(web_text, "_open", lambda req, timeout: _Resp("application/octet-stream", data))
    assert PdfTextExtractor().transcribe(SourceRef(KEY, "pdf", "https://e.org/paper.pdf"), ["en"]).segments
    with pytest.raises(TranscriptionError, match="not a PDF"):
        PdfTextExtractor().transcribe(SourceRef(KEY, "pdf", "https://e.org/download?id=3"), ["en"])


def test_remote_html_is_refused(monkeypatch):
    monkeypatch.setattr(web_text, "_open", lambda req, timeout: _Resp("text/html", b"<html>"))
    with pytest.raises(TranscriptionError, match="not a PDF"):
        PdfTextExtractor().transcribe(SourceRef(KEY, "pdf", "https://e.org/paper.pdf"), ["en"])


def test_pdf_metadata_local_and_remote(tmp_path, monkeypatch):
    path = str(tmp_path / "state.sqlite")
    db.init_db(path)
    nb = db.create_notebook(path, "T", "t")
    p = write_text_pdf(tmp_path / "a.pdf", ["Words about habits. " * 20], info={"Title": "Habit notes"})
    sid = db.add_source(path, nb, KEY, p.resolve().as_uri(), kind="pdf")
    assert fetch.fetch_source_metadata(path, sid, log=lambda m: None) == "done"
    row = db.get_source(path, sid)
    assert row["title"] == "Habit notes" and row["channel"] is None
    data = write_text_pdf(tmp_path / "b.pdf", ["Words about habits. " * 20]).read_bytes()
    monkeypatch.setattr(web_text, "_open", lambda req, timeout: _Resp("application/pdf", data))
    sid2 = db.add_source(path, nb, "pdf-1123456789abcdef", "https://e.org/files/My%20Paper.pdf", kind="pdf")
    fetch.fetch_source_metadata(path, sid2, log=lambda m: None)
    row = db.get_source(path, sid2)
    assert row["title"] == "My Paper.pdf" and row["channel"] == "e.org"


def test_pdf_and_text_transcribers_are_registered():
    from pipeline.config import Config
    from pipeline.transcription.chain import build_transcribers
    names = {t.name: t for t in build_transcribers(Config())}
    assert names["pdf_text"].kinds == frozenset({"pdf"}) and names["local_text"].kinds == frozenset({"text"})


def test_too_many_pages_is_refused(tmp_path, monkeypatch):
    p = write_text_pdf(tmp_path / "a.pdf", ["Words about habits. " * 20] * 3)
    monkeypatch.setattr(pdf_text, "MAX_PDF_PAGES", 2)
    with pytest.raises(TranscriptionError, match="PDF has 3 pages; the limit is 2"):
        PdfTextExtractor().transcribe(_ref(p), ["en"])


def test_word_cap_stops_extraction(tmp_path, monkeypatch):
    p = write_text_pdf(tmp_path / "a.pdf", ["Words about habits. " * 20] * 5)
    monkeypatch.setattr(pdf_text, "MAX_SOURCE_WORDS", 100)
    with pytest.raises(TranscriptionError, match="more than 100 words"):
        PdfTextExtractor().transcribe(_ref(p), ["en"])


def test_creation_date_without_prefix(tmp_path):
    p = write_text_pdf(tmp_path / "a.pdf", ["Words about habits. " * 20], info={"CreationDate": "20230405120000"})
    assert PdfTextExtractor().transcribe(_ref(p), ["en"]).extra["date"] == "20230405"


class _BrokenResp(_Resp):
    def read(self, n=-1):
        import http.client
        raise http.client.IncompleteRead(b"%PDF-1.4 partial")


def test_broken_remote_pdf_download_keeps_the_basename_title(monkeypatch, tmp_path):
    path = str(tmp_path / "state.sqlite")
    db.init_db(path)
    nb = db.create_notebook(path, "T", "t")
    monkeypatch.setattr(web_text, "_open", lambda req, timeout: _BrokenResp("application/pdf", b""))
    sid = db.add_source(path, nb, KEY, "https://e.org/files/Paper.pdf", kind="pdf")
    assert fetch.fetch_source_metadata(path, sid, log=lambda m: None) == "done"
    row = db.get_source(path, sid)
    assert (row["title"], row["channel"], row["fetch_status"]) == ("Paper.pdf", "e.org", "done")


def test_any_document_metadata_failure_falls_back_to_the_basename(monkeypatch, tmp_path):
    path = str(tmp_path / "state.sqlite")
    db.init_db(path)
    nb = db.create_notebook(path, "T", "t")

    def boom(ref):
        raise RuntimeError("parser bug")
    monkeypatch.setattr(pdf_text, "pdf_title", boom)
    sid = db.add_source(path, nb, KEY, "https://e.org/files/Paper.pdf", kind="pdf")
    logs = []
    assert fetch.fetch_source_metadata(path, sid, log=logs.append) == "done"
    assert db.get_source(path, sid)["title"] == "Paper.pdf"
    assert any("metadata fallback" in line and "parser bug" in line for line in logs)


def test_text_metadata_failure_falls_back_to_the_file_stem(monkeypatch, tmp_path):
    from pipeline.transcription import local_text
    path = str(tmp_path / "state.sqlite")
    db.init_db(path)
    nb = db.create_notebook(path, "T", "t")

    def boom(p):
        raise RuntimeError("reader bug")
    monkeypatch.setattr(local_text, "text_title", boom)
    f = tmp_path / "my notes.md"
    f.write_text("# N\n\ntext", encoding="utf-8")
    sid = db.add_source(path, nb, "text-0123456789abcdef", f.resolve().as_uri(), kind="text")
    assert fetch.fetch_source_metadata(path, sid, log=lambda m: None) == "done"
    assert db.get_source(path, sid)["title"] == "my notes"


# pypdf runs in a child process with a wall-clock limit (release security review, 2026-10-28): a
# crafted PDF that makes the parser loop is killed, not left spinning in the run's worker thread.
def test_extraction_runs_in_a_child_without_api_keys(tmp_path, monkeypatch):
    seen = {}
    real_run = pdf_text.subprocess.run

    def spy(argv, **kw):
        seen["argv"], seen["env"] = argv, kw["env"]
        return real_run(argv, **kw)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(pdf_text.subprocess, "run", spy)
    p = write_text_pdf(tmp_path / "a.pdf", ["Words about habits. " * 20], info={"Title": "Habit notes"})
    t = PdfTextExtractor().transcribe(_ref(p), ["en"])
    assert t.extra["title"] == "Habit notes" and t.segments[0].page == 1
    assert seen["argv"][1] == "-P" and "pipeline.transcription.pdf_text" in seen["argv"]
    assert "ANTHROPIC_API_KEY" not in seen["env"]


def test_extraction_over_the_time_limit_is_a_transcription_error(tmp_path, monkeypatch):
    import sys
    p = write_text_pdf(tmp_path / "a.pdf", ["Words about habits. " * 20])
    monkeypatch.setattr(pdf_text, "PDF_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(pdf_text, "_child_argv", lambda: [sys.executable, "-c", "import time; time.sleep(30)"])
    with pytest.raises(TranscriptionError, match="took longer than 1 s"):
        PdfTextExtractor().transcribe(_ref(p), ["en"])
    assert pdf_text.pdf_title(_ref(p)) is None             # the metadata step gives up the same way


def test_a_crashing_child_is_a_transcription_error(tmp_path, monkeypatch):
    import sys
    p = write_text_pdf(tmp_path / "a.pdf", ["Words about habits. " * 20])
    monkeypatch.setattr(pdf_text, "_child_argv", lambda: [sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(TranscriptionError, match="pdf_text: could not read PDF"):
        PdfTextExtractor().transcribe(_ref(p), ["en"])
