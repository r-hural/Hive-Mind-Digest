"""Transcriber Protocol (DECISION-012) and the transcript data model (DECISION-016).

Every provider returns segments with timestamps; flat text is derived, never stored as the
primary form, so IDEA-001 deep links are always possible.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pipeline.sources import SourceRef


class TranscriptionError(Exception):
    """Any provider failure; the chain moves on to the next transcriber."""


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


@dataclass
class Segment:
    start: float
    end: float
    text: str
    page: int | None = None      # 1-based page for PDF sources; None elsewhere


@dataclass
class FlatTranscript:
    """Segments joined by single spaces, with the char offset and start time of each segment."""
    text: str
    seg_starts: list[int]
    seg_times: list[float]
    seg_pages: list[int | None] = field(default_factory=list)

    def time_at(self, char_offset: int) -> float:
        if not self.seg_starts:
            return 0.0
        i = bisect_right(self.seg_starts, char_offset) - 1
        return self.seg_times[max(i, 0)]

    def segment_end(self, i: int) -> int:
        """Char offset just past segment i's text (the joining space is not part of it)."""
        return self.seg_starts[i + 1] - 1 if i + 1 < len(self.seg_starts) else len(self.text)

    def segment_texts(self) -> list[str]:
        """The non-empty segments' texts, in order; index i is segment i of the analyze prompt's
        numbering (IDEA-294) and of seg_starts/seg_times."""
        return [self.text[s:self.segment_end(i)] for i, s in enumerate(self.seg_starts)]


def flatten(segments: list[Segment]) -> FlatTranscript:
    parts: list[str] = []
    starts: list[int] = []
    times: list[float] = []
    pages: list[int | None] = []
    pos = 0
    for seg in segments:
        text = " ".join(seg.text.split())
        if not text:
            continue
        if parts:
            pos += 1                      # the joining space
        starts.append(pos)
        times.append(float(seg.start))
        pages.append(seg.page)
        parts.append(text)
        pos += len(text)
    return FlatTranscript(" ".join(parts), starts, times, pages)


@dataclass
class Transcript:
    video_id: str               # the source key (SourceRef.key)
    method: str                 # youtube_captions | ytdlp_subs | supadata | whisper_local
    language: str
    segments: list[Segment]
    created_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)   # is_generated, whisper model, ...

    @property
    def duration(self) -> float:
        # Rolling auto-caption cues overlap, so the last segment isn't always the latest-ending one.
        return float(max(s.end for s in self.segments)) if self.segments else 0.0

    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.segments)

    def flatten(self) -> FlatTranscript:
        return flatten(self.segments)


@runtime_checkable
class Transcriber(Protocol):
    # `name` is a non-method member, so only isinstance() works here — issubclass() raises
    # TypeError on a data-member Protocol. isinstance() checks attribute presence, not signatures.
    name: str
    kinds: frozenset[str]     # source kinds this transcriber handles (pipeline.sources.KINDS)

    def transcribe(self, ref: SourceRef, languages: list[str]) -> Transcript: ...
