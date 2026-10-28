"""Shared segmentation for the text extractors (web pages, PDFs, local text files): each paragraph
is split into sentence groups of at most MAX_SEGMENT_WORDS words, never merging across paragraphs
(the callers split per paragraph, and a PDF per page too), so a quote of 1-4 segments is about the
length of a transcript quote and of the rendered excerpt."""
from __future__ import annotations

import re

MAX_SEGMENT_WORDS = 40
MAX_SOURCE_WORDS = 200_000      # per source: more than this floods analyze, so extraction refuses
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_paragraphs(text: str, max_words: int = MAX_SEGMENT_WORDS) -> list[str]:
    """Split one paragraph at sentence ends (`.`, `!` or `?` followed by whitespace) into groups of
    whole sentences of at most max_words words; a single sentence longer than that is cut every
    max_words words. Text with no more than max_words words comes back whole."""
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for sentence in _SENTENCE_END.split(text.strip()):
        words = sentence.split()
        if not words:
            continue
        while len(words) > max_words:
            if cur:
                chunks.append(" ".join(cur))
                cur, n = [], 0
            chunks.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if n + len(words) > max_words and cur:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(" ".join(words))
        n += len(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks
