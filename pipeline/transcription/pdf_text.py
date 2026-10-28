"""PDFs as sources (IDEA-005): the text layer of each page, split into paragraphs and then into
sentence-group segments of at most 40 words (textseg.py) that carry their 1-based page number
(start = end = 0.0: a document has no timeline).

PDFs are untrusted: 50 MB, 2000-page and 200k-word caps, local files must be regular files (a
symlink the user names is followed: SourceRef.path resolves it), remote ones through web_text's guarded
opener; the file is parsed with pypdf, never executed. pypdf can raise almost anything on a
malformed file, so every reading error becomes a TranscriptionError. No OCR: a scanned PDF fails
with a clear message."""
from __future__ import annotations

import io
import os
import re
from urllib.parse import unquote, urlsplit

import pypdf

from pipeline.sources import MAX_FILE_BYTES, SourceRef
from pipeline.transcription import web_text
from pipeline.transcription.base import Segment, Transcript, TranscriptionError
from pipeline.transcription.textseg import MAX_SEGMENT_WORDS, MAX_SOURCE_WORDS, split_paragraphs

MAX_PDF_BYTES = MAX_FILE_BYTES["pdf"]          # remote downloads too
MIN_WORDS = 50
MAX_PDF_PAGES = 2000
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


def pdf_title(ref: SourceRef) -> str | None:
    """The PDF's /Title, or None (any failure: the transcribe step reports the real error)."""
    try:
        return _metadata(_open_reader(read_pdf_bytes(ref)))[0]
    except TranscriptionError:
        return None


def _page_paragraphs(text: str) -> list[str]:
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    paragraphs = [p for p in paragraphs if p]
    out: list[str] = []
    for p in paragraphs:
        out.extend(split_paragraphs(p, MAX_SEGMENT_WORDS))
    return out


class PdfTextExtractor:
    name = "pdf_text"
    kinds = frozenset({"pdf"})

    def transcribe(self, ref: SourceRef, languages: list[str]) -> Transcript:
        if ref.kind not in self.kinds:
            raise TranscriptionError(f"pdf_text: cannot handle {ref.kind} sources")
        reader = _open_reader(read_pdf_bytes(ref))
        segments: list[Segment] = []
        try:
            pages = reader.pages
            if len(pages) > MAX_PDF_PAGES:
                raise TranscriptionError(f"pdf_text: PDF has {len(pages)} pages; the limit is {MAX_PDF_PAGES}")
            total = 0
            for i in range(len(pages)):
                for p in _page_paragraphs(pages[i].extract_text() or ""):
                    segments.append(Segment(0.0, 0.0, p, page=i + 1))
                    total += len(p.split())
                if total > MAX_SOURCE_WORDS:
                    raise TranscriptionError(f"pdf_text: PDF has more than {MAX_SOURCE_WORDS} words")
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(f"pdf_text: could not read PDF: {type(e).__name__}: {e}") from e
        words = sum(len(s.text.split()) for s in segments)
        if words < MIN_WORDS:
            raise TranscriptionError(f"pdf_text: no extractable text (a scanned PDF?); OCR is not supported "
                                     f"({words} words found, need {MIN_WORDS})")
        title, date = _metadata(reader)
        return Transcript(video_id=ref.key, method=self.name, language=languages[0] if languages else "",
                          segments=segments, created_at="",
                          extra={"kind": "pdf", "pages": len(reader.pages), "title": title, "date": date})


def url_basename(url: str) -> str:
    return unquote(os.path.basename(urlsplit(url).path.rstrip("/"))) or url
