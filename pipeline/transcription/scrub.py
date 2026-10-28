"""Scrub secret-shaped strings from transcripts and extracted text before they are cached
(IDEA-074): a pasted key in a web page or a slide must never land in the cache, a report or a
model prompt. Conservative patterns only; ordinary words like "token" or "secret" stay."""
from __future__ import annotations

import re

from pipeline.transcription.base import Segment

REDACTED = "[REDACTED]"
# Key shapes need a left boundary, or hyphenated words (task-oriented-...) lose their tail.
_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9_-])sk-(?:ant-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9_-])AKIA[0-9A-Z]{16}"),
    re.compile(r"(?<![A-Za-z0-9_-])gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"(?<![A-Za-z0-9_-])xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{35}"),
    # the whole block, body included; with no END line (a truncated paste), to the end of the text
    re.compile(r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"),
]
# Lookahead bounded (256) and the value possessive: a long run of labels stays linear, the whole
# value goes. The value must contain a digit: "The secret: consistency beats intensity" is prose, not a credential.
_LABELLED = re.compile(r"(?i)\b(api[_-]?key|secret|token|password)(\s*[:=]\s*)(?=\S{0,256}\d)\S{8,}+")


def scrub_text(text: str) -> tuple[str, int]:
    count = 0
    for pat in _PATTERNS:
        text, n = pat.subn(REDACTED, text)
        count += n
    text, n = _LABELLED.subn(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    return text, count + n


def scrub_segments(segments: list[Segment]) -> int:
    total = 0
    for s in segments:
        s.text, n = scrub_text(s.text)
        total += n
    return total
