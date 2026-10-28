"""Provider chain: configured order, first success is cached and returned (DECISION-016 amendment).

Provider imports are deferred into `build_transcribers` (lazy) so this module -- and
`transcribe_with_chain` in particular -- is importable without `yt_dlp` on the path, saving
~200ms on CLI start paths that never transcribe.
"""
from __future__ import annotations

from typing import Callable

from pipeline.config import Config
from pipeline.sources import SourceRef
from pipeline.transcription.base import Transcriber, Transcript, TranscriptionError
from pipeline.transcription.cache import TranscriptCache


def build_transcribers(cfg: Config) -> list[Transcriber]:
    from pipeline.transcription.local_text import LocalTextExtractor
    from pipeline.transcription.pdf_text import PdfTextExtractor
    from pipeline.transcription.supadata import SupadataProvider
    from pipeline.transcription.web_text import WebTextExtractor
    from pipeline.transcription.whisper_local import WhisperLocalProvider
    from pipeline.transcription.youtube_captions import YouTubeCaptionsProvider
    from pipeline.transcription.ytdlp_subs import YtDlpSubsProvider

    w = cfg.transcription.whisper
    from pipeline.transcription import web_text
    web_text.set_allow_private_hosts(cfg.transcription.allow_private_hosts)
    registry: dict[str, Callable[[], Transcriber]] = {
        "youtube_captions": YouTubeCaptionsProvider,
        "ytdlp_subs": YtDlpSubsProvider,
        "supadata": SupadataProvider,
        "whisper_local": lambda: WhisperLocalProvider(w.model_size, w.timeout_seconds),
    }
    transcribers = [registry[name]() for name in cfg.transcription.providers]
    # web pages, PDFs and local text files need no configuration: always present, last
    transcribers += [WebTextExtractor(), PdfTextExtractor(), LocalTextExtractor()]
    return transcribers


# A transcript whose last segment ends before this share of the known video length is treated as
# a provider failure, not a success: seen live 2026-10-09, supadata returned 4 segments (0:06-0:15)
# for a 17:23 talk, and analyze then found no claims. 0.5 leaves room for long music/silence tails.
MIN_COVERAGE = 0.5


def _one_line(message: str) -> str:
    """First line of a provider error for the log; libraries such as youtube-transcript-api put
    several paragraphs of advice in theirs, which buries every other line of a run."""
    lines = [l.strip() for l in str(message).splitlines() if l.strip()]
    if not lines:
        return ""
    return lines[0] + (" ..." if len(lines) > 1 else "")


def _coverage_problem(transcript: Transcript, expected_duration: float | None) -> str | None:
    if not expected_duration or expected_duration <= 0 or not transcript.segments:
        return None
    end = max(seg.end for seg in transcript.segments)
    if end < MIN_COVERAGE * expected_duration:
        return f"{transcript.method}: transcript covers only {end:.0f}s of {expected_duration:.0f}s"
    return None


def transcribe_with_chain(ref: SourceRef, transcribers: list[Transcriber], languages: list[str],
                          cache: TranscriptCache, log: Callable[[str], None] = print, *,
                          force: bool = False, expected_duration: float | None = None) -> Transcript:
    """expected_duration (seconds, from the source metadata) enables the coverage check: a
    transcript ending before MIN_COVERAGE of it is rejected, never cached, and the chain moves on.
    Without a known duration every non-empty transcript is accepted, as before."""
    video_id = ref.key
    if not transcribers:
        raise TranscriptionError(f"no transcribers configured for {video_id}")
    handlers = [t for t in transcribers if ref.kind in t.kinds]
    if not handlers:
        names = ", ".join(t.name for t in transcribers)
        hint = ("; add whisper_local to transcription.providers"
                if ref.kind == "media" and not any(t.name == "whisper_local" for t in transcribers) else "")
        raise TranscriptionError(f"no transcriber handles {ref.kind} sources; configured: {names}{hint}")
    if not force:
        cached = cache.load_best(video_id, [t.name for t in handlers])
        if cached is not None:
            problem = _coverage_problem(cached, expected_duration)
            if problem is None:
                log(f"  [{video_id}] cached transcript ({cached.method})")
                return cached
            log(f"  [{video_id}] ignoring cached transcript: {problem}")
    errors: list[str] = []
    for t in handlers:
        try:
            transcript = t.transcribe(ref, languages)
        except TranscriptionError as e:
            errors.append(str(e))
            log(f"  [{video_id}] {t.name} failed: {_one_line(e)}")
            continue
        except Exception as e:
            msg = f"{t.name}: unexpected {type(e).__name__}: {e}"
            errors.append(msg)
            log(f"  [{video_id}] {_one_line(msg)}")
            continue
        problem = _coverage_problem(transcript, expected_duration)
        if problem is not None:
            errors.append(problem)
            log(f"  [{video_id}] {t.name} rejected: {problem}")
            continue
        try:
            cache.save(transcript)
        except Exception as e:
            log(f"  [{video_id}] {t.name} transcribed but cache write failed: {e}")
            raise
        log(f"  [{video_id}] {t.name}: {len(transcript.segments)} segments, {transcript.word_count()} words")
        return transcript
    raise TranscriptionError(f"all transcribers failed for {video_id}: " + " | ".join(errors))
