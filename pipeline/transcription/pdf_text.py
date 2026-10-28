"""PDFs as sources (IDEA-005): the text layer of each page, split into paragraphs and then into
sentence-group segments of at most 40 words (textseg.py) that carry their 1-based page number
(start = end = 0.0: a document has no timeline).

PDFs are untrusted: 50 MB, 2000-page and 200k-word caps, local files must be regular files (a
symlink the user names is followed: SourceRef.path resolves it), remote ones through web_text's guarded
opener; the file is parsed with pypdf, never executed. pypdf can raise almost anything on a
malformed file, so every reading error becomes a TranscriptionError. No OCR: a scanned PDF fails
with a clear message.

pypdf runs in a child process (`python -P -m pipeline.transcription.pdf_text`, the PDF on stdin,
JSON on stdout, no API keys in its environment) killed after PDF_TIMEOUT_SECONDS: a crafted file
that makes the parser loop cannot hold a run's worker thread (release security review, 2026-10-28)."""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

import pypdf

from pipeline.sources import MAX_FILE_BYTES, SourceRef
from pipeline.transcription import web_text
from pipeline.transcription.base import Segment, Transcript, TranscriptionError
from pipeline.transcription.textseg import MAX_SEGMENT_WORDS, MAX_SOURCE_WORDS, split_paragraphs

MAX_PDF_BYTES = MAX_FILE_BYTES["pdf"]          # remote downloads too
MIN_WORDS = 50
MAX_PDF_PAGES = 2000
PDF_TIMEOUT_SECONDS = 120       # wall clock for one PDF's parse in the child
CHILD_ENV_DROP = frozenset({"ANTHROPIC_API_KEY", "SUPADATA_API_KEY"})
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PDF_TYPES = ("application/pdf", "application/x-pdf")
_PDF_DATE = re.compile(r"^(?:D:)?(\d{4})(\d{2})(\d{2})")


def read_pdf_bytes(ref: SourceRef) -> bytes:
    """The PDF's bytes: a regular local file, or a guarded download (max 50 MB)."""
    path = ref.path
    if path is not None:
        if not os.path.isfile(path) or os.path.islink(path):
            raise TranscriptionError(f"pdf_text: {path} is not a regular file")
        try:
            with open(path, "rb") as fh:
                data = fh.read(MAX_PDF_BYTES + 1)
        except OSError as e:
            raise TranscriptionError(f"pdf_text: could not read {path}: {e}") from e
        if len(data) > MAX_PDF_BYTES:
            raise TranscriptionError(f"pdf_text: file is larger than {MAX_PDF_BYTES} bytes")
        return data
    url_path = urlsplit(ref.url).path.lower()

    def type_ok(ctype: str) -> bool:
        return ctype in _PDF_TYPES or (ctype == "application/octet-stream" and url_path.endswith(".pdf"))

    _final, body, _ctype, _cs = web_text.fetch_bytes(
        ref.url, accept="application/pdf", type_ok=type_ok, what="a PDF", max_bytes=MAX_PDF_BYTES,
        timeout=web_text.FETCH_TIMEOUT, label="pdf_text")
    return body


def _open_reader(data: bytes) -> pypdf.PdfReader:
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception as e:      # e.g. an AES PDF without the optional crypto library
                raise TranscriptionError(f"pdf_text: encrypted PDF (cannot decrypt: {e})") from e
            if not ok:
                raise TranscriptionError("pdf_text: encrypted PDF (password protected); decrypt it first")
        return reader
    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(f"pdf_text: could not read PDF: {type(e).__name__}: {e}") from e


def _metadata(reader: pypdf.PdfReader) -> tuple[str | None, str | None]:
    """(title, date YYYYMMDD) from the document information; None when absent or unreadable."""
    try:
        info = reader.metadata
        title = str(info.get("/Title") or "").strip() or None if info else None
        m = _PDF_DATE.match(str(info.get("/CreationDate") or "")) if info else None
        return title, ("".join(m.groups()) if m else None)
    except Exception:
        return None, None


def _page_paragraphs(text: str) -> list[str]:
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    paragraphs = [p for p in paragraphs if p]
    out: list[str] = []
    for p in paragraphs:
        out.extend(split_paragraphs(p, MAX_SEGMENT_WORDS))
    return out


def _extract(data: bytes, *, text: bool, max_pages: int, max_words: int) -> dict:
    """Parse the PDF: {"title", "date", "pages"} and, with `text`, "paragraphs" as [page, text]
    pairs. TranscriptionError for an unreadable, encrypted or over-cap file. Runs in the child."""
    reader = _open_reader(data)
    paragraphs: list[list] = []
    try:
        pages = reader.pages
        n = len(pages)
        if text:
            if n > max_pages:
                raise TranscriptionError(f"pdf_text: PDF has {n} pages; the limit is {max_pages}")
            total = 0
            for i in range(n):
                for p in _page_paragraphs(pages[i].extract_text() or ""):
                    paragraphs.append([i + 1, p])
                    total += len(p.split())
                if total > max_words:
                    raise TranscriptionError(f"pdf_text: PDF has more than {max_words} words")
    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(f"pdf_text: could not read PDF: {type(e).__name__}: {e}") from e
    title, date = _metadata(reader)
    return {"title": title, "date": date, "pages": n, "paragraphs": paragraphs}


def _extract_in_process(data: bytes, *, text: bool) -> dict:
    return _extract(data, text=text, max_pages=MAX_PDF_PAGES, max_words=MAX_SOURCE_WORDS)


def _child_argv() -> list[str]:
    # -P and a neutral cwd: the user's working directory is not on the child's sys.path.
    return [sys.executable, "-P", "-m", "pipeline.transcription.pdf_text"]


def _extract_bounded(data: bytes, *, text: bool) -> dict:
    """_extract in a child process, killed after PDF_TIMEOUT_SECONDS."""
    argv = _child_argv() + ["--text" if text else "--meta", str(MAX_PDF_PAGES), str(MAX_SOURCE_WORDS)]
    env = {k: v for k, v in os.environ.items() if k not in CHILD_ENV_DROP}
    env["PYTHONPATH"] = os.pathsep.join(x for x in [_REPO_ROOT, os.environ.get("PYTHONPATH", "")] if x)
    try:
        proc = subprocess.run(argv, input=data, capture_output=True, timeout=PDF_TIMEOUT_SECONDS, env=env,
                              cwd=tempfile.gettempdir(), shell=False)
    except subprocess.TimeoutExpired as e:
        raise TranscriptionError(f"pdf_text: reading the PDF took longer than {PDF_TIMEOUT_SECONDS} s") from e
    try:
        out = json.loads(proc.stdout.decode("utf-8"))
    except ValueError:
        out = None
    if not isinstance(out, dict):
        tail = proc.stderr.decode("utf-8", "replace").strip()[-300:]
        raise TranscriptionError(f"pdf_text: could not read PDF: the reader exited with {proc.returncode}: {tail}")
    if "error" in out:
        raise TranscriptionError(str(out["error"]))
    return out


def _child_main(argv: list[str]) -> int:
    mode, max_pages, max_words = argv[0], int(argv[1]), int(argv[2])
    data = sys.stdin.buffer.read()
    try:
        out = _extract(data, text=mode == "--text", max_pages=max_pages, max_words=max_words)
    except TranscriptionError as e:
        out = {"error": str(e)}
    sys.stdout.write(json.dumps(out, ensure_ascii=True))
    return 0


def pdf_title(ref: SourceRef) -> str | None:
    """The PDF's /Title, or None (any failure: the transcribe step reports the real error)."""
    try:
        return _extract_bounded(read_pdf_bytes(ref), text=False)["title"]
    except TranscriptionError:
        return None


class PdfTextExtractor:
    name = "pdf_text"
    kinds = frozenset({"pdf"})

    def transcribe(self, ref: SourceRef, languages: list[str]) -> Transcript:
        if ref.kind not in self.kinds:
            raise TranscriptionError(f"pdf_text: cannot handle {ref.kind} sources")
        out = _extract_bounded(read_pdf_bytes(ref), text=True)
        try:
            segments = [Segment(0.0, 0.0, str(text), page=int(page)) for page, text in out["paragraphs"]]
        except (KeyError, TypeError, ValueError) as e:
            raise TranscriptionError(f"pdf_text: the reader's output is not parseable: {e}") from e
        words = sum(len(s.text.split()) for s in segments)
        if words < MIN_WORDS:
            raise TranscriptionError(f"pdf_text: no extractable text (a scanned PDF?); OCR is not supported "
                                     f"({words} words found, need {MIN_WORDS})")
        return Transcript(video_id=ref.key, method=self.name, language=languages[0] if languages else "",
                          segments=segments, created_at="",
                          extra={"kind": "pdf", "pages": out["pages"], "title": out["title"], "date": out["date"]})


def url_basename(url: str) -> str:
    return unquote(os.path.basename(urlsplit(url).path.rstrip("/"))) or url


if __name__ == "__main__":
    sys.exit(_child_main(sys.argv[1:]))
