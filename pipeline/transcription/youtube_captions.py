"""Captions via youtube-transcript-api. Unofficial, blocked from datacenter IPs, so it is an
optimisation, never a guarantee (DECISION-016 amendment). Any failure falls through the chain."""
from __future__ import annotations

from youtube_transcript_api import YouTubeTranscriptApi

from pipeline.sources import SourceRef
from pipeline.transcription.base import Segment, Transcript, TranscriptionError


class YouTubeCaptionsProvider:
    name = "youtube_captions"
    kinds = frozenset({"youtube"})

    def transcribe(self, ref: SourceRef, languages: list[str]) -> Transcript:
        if ref.kind not in self.kinds:
            raise TranscriptionError(f"youtube_captions: cannot handle {ref.kind} sources")
        video_id = ref.key
        try:
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
            segments = []
            for s in fetched.snippets:
                start = float(s.start)
                segments.append(Segment(start, start + float(s.duration), s.text))
            language = fetched.language_code or "und"
            is_generated = bool(fetched.is_generated)
        except Exception as e:   # the library raises a dozen classes; all mean "next provider"
            raise TranscriptionError(f"youtube_captions: {type(e).__name__}: {e}") from e
        if not any(s.text.strip() for s in segments):
            raise TranscriptionError(f"youtube_captions: empty track for {video_id}")
        return Transcript(video_id=video_id, method=self.name, language=language,
                          segments=segments, extra={"is_generated": is_generated})
