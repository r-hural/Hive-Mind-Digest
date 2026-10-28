"""Tiny hand-built PDFs for the PDF tests (no binary files are committed)."""
from __future__ import annotations

import pathlib
import textwrap


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build(pages_streams: list[bytes], info: dict[str, str] | None = None) -> bytes:
    """Objects: 1 catalog, 2 pages, 3 font, then per page a page object and a content stream."""
    n = len(pages_streams)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Count {n} /Kids [{kids}] >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for i, stream in enumerate(pages_streams):
        objs[4 + 2 * i] = (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                           b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % (5 + 2 * i))
        objs[5 + 2 * i] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
    trailer_extra = b""
    if info:
        num = 4 + 2 * n
        objs[num] = ("<< " + " ".join(f"/{k} ({_esc(v)})" for k, v in info.items()) + " >>").encode("latin-1")
        trailer_extra = b" /Info %d 0 R" % num
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += b"%d 0 obj\n" % num + objs[num] + b"\nendobj\n"
    xref = len(out)
    size = max(objs) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for num in range(1, size):
        out += b"%010d 00000 n \n" % offsets[num]
    out += b"trailer\n<< /Size %d /Root 1 0 R%s >>\nstartxref\n%d\n%%%%EOF\n" % (size, trailer_extra, xref)
    return bytes(out)


def write_text_pdf(path, pages: list[str], info: dict[str, str] | None = None) -> pathlib.Path:
    """One page per string; the text is wrapped into 60-character lines (Helvetica 12 pt)."""
    streams = []
    for text in pages:
        lines = textwrap.wrap(text, 60) or [""]
        body = "BT /F1 12 Tf 14 TL 50 740 Td " + " T* ".join(f"({_esc(l)}) Tj" for l in lines) + " ET"
        streams.append(body.encode("latin-1"))
    p = pathlib.Path(path)
    p.write_bytes(_build(streams, info))
    return p


def write_blank_pdf(path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.write_bytes(_build([b""]))
    return p
