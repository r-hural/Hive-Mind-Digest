"""Local .txt / .md files as sources (IDEA-005): paragraph segments (start = end = 0.0).

Markdown files (.md, .markdown): a heading line starts a new paragraph and is kept as its own
short segment; a fenced code block is one paragraph (whitespace collapsed). Plain .txt files are
split into paragraphs on blank lines only. Each paragraph becomes sentence groups of at most 40 words
(pipeline/transcription/textseg.py). Files are untrusted: regular files only (a
symlink the user names is followed: SourceRef.path resolves it), 10 MB and 200k-word caps; the content is read, never executed."""
from __future__ import annotations

import os
import pathlib
import re

from pipeline.sources import MAX_FILE_BYTES, SourceRef
from pipeline.transcription.base import Segment, Transcript, TranscriptionError
from pipeline.transcription.textseg import MAX_SEGMENT_WORDS, MAX_SOURCE_WORDS, split_paragraphs

MAX_TEXT_BYTES = MAX_FILE_BYTES["text"]
MIN_WORDS = 20
_MARKDOWN_SUFFIXES = (".md", ".markdown")
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*)$")   # closing run stripped in parse_text (linear)
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def read_text_file(path: str) -> str:
    if not os.path.isfile(path) or os.path.islink(path):
        raise TranscriptionError(f"local_text: {path} is not a regular file")
    try:
        with open(path, "rb") as fh:
            data = fh.read(MAX_TEXT_BYTES + 1)
    except OSError as e:
        raise TranscriptionError(f"local_text: could not read {path}: {e}") from e
    if len(data) > MAX_TEXT_BYTES:
        raise TranscriptionError(f"local_text: file is larger than {MAX_TEXT_BYTES} bytes")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _is_markdown(path: str) -> bool:
    return path.lower().endswith(_MARKDOWN_SUFFIXES)


def parse_text(text: str, markdown: bool) -> tuple[list[str], str | None]:
    """(segment texts, first heading or None)."""
    out: list[str] = []
    first_heading: str | None = None
    para: list[str] = []
    fence: list[str] | None = None
    fence_mark = ""

    def flush() -> None:
        if para:
            out.extend(split_paragraphs(" ".join(" ".join(para).split()), MAX_SEGMENT_WORDS))
            para.clear()

    for line in text.splitlines():
        if fence is not None:
            fence.append(line)
            s = line.strip()
            if len(s) >= len(fence_mark) and set(s) == {fence_mark[0]}:
                out.extend(split_paragraphs(" ".join(" ".join(fence).split()), MAX_SEGMENT_WORDS))
                fence = None
            continue
        if markdown:
            m = _FENCE.match(line)
            if m:
                flush()
                fence, fence_mark = [line], m.group(1)
                continue
            h = _HEADING.match(line)
            title = h.group(2).rstrip(" 	#") if h else ""
            if title:
                flush()
                if first_heading is None:
                    first_heading = title
                out.extend(split_paragraphs(f"{h.group(1)} {title}", MAX_SEGMENT_WORDS))
                continue
        if line.strip():
            para.append(line.strip())
        else:
            flush()
    if fence is not None:       # unterminated fence: keep what there is
        out.extend(split_paragraphs(" ".join(" ".join(fence).split()), MAX_SEGMENT_WORDS))
    flush()
    return out, first_heading


def text_title(path: str) -> str:
    """First Markdown heading, else the file stem (any read failure: the stem)."""
    stem = pathlib.Path(path).stem
    if not _is_markdown(path):
        return stem
    try:
        return parse_text(read_text_file(path), True)[1] or stem
    except TranscriptionError:
        return stem


class LocalTextExtractor:
    name = "local_text"
    kinds = frozenset({"text"})

    def transcribe(self, ref: SourceRef, languages: list[str]) -> Transcript:
        if ref.kind not in self.kinds:
            raise TranscriptionError(f"local_text: cannot handle {ref.kind} sources")
        path = ref.path
        if path is None:
            raise TranscriptionError(f"local_text: not a local file: {ref.url!r}")
        content = read_text_file(path)
        if len(content.split()) > MAX_SOURCE_WORDS:
            raise TranscriptionError(f"local_text: file has more than {MAX_SOURCE_WORDS} words")
        texts, heading = parse_text(content, _is_markdown(path))
        if sum(len(t.split()) for t in texts) < MIN_WORDS:
            raise TranscriptionError("local_text: file has almost no text")
        return Transcript(video_id=ref.key, method=self.name, language=languages[0] if languages else "",
                          segments=[Segment(0.0, 0.0, t) for t in texts], created_at="",
                          extra={"kind": "text", "title": heading or pathlib.Path(path).stem})
