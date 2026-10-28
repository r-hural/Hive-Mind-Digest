"""Second transcriber: yt-dlp subtitle download (manual and automatic tracks) as json3,
which carries timestamps (IDEA-204). Uses the yt-dlp Python API, so no binary on PATH.

Track selection is deliberate. yt-dlp treats each `subtitleslangs` entry as a full-match
regex (`YoutubeDL.process_subtitles` -> `orderedSet_from_options(..., use_regex=True)`), so a
plain language code like "en" matches the literal key "en" only -- it misses a manual
"en-US"/"en-GB" track. And a plain automatic_captions "en" key can be a machine translation of
a non-English video; yt-dlp labels the original-language ASR track "<lang>-orig" (see
yt_dlp/extractor/youtube/_video.py, the `subtitles`/`automatic_captions` construction around
lines 4283-4330). So we fetch metadata first (no download), choose one exact track key
ourselves -- manual exact, then manual prefix, then auto original, never a plain auto key --
and only then download that one key.
"""
from __future__ import annotations

import json
import os
import re
import tempfile

import yt_dlp

from pipeline.sources import SourceRef
from pipeline.transcription.base import Segment, Transcript, TranscriptionError


def parse_json3(data: dict) -> list[Segment]:
    segments: list[Segment] = []
    for event in data.get("events", []):
        parts = event.get("segs") or []
        text = " ".join("".join(p.get("utf8", "") for p in parts).split())
        if not text:
            continue
        start = float(event.get("tStartMs", 0)) / 1000.0
        end = start + float(event.get("dDurationMs", 0)) / 1000.0
        segments.append(Segment(start, end, text))
    return segments


def _select_track(manual: dict, auto: dict, languages: list[str]) -> str | None:
    """First present of, per language in order: manual exact, manual prefix (e.g. en-US for
    en), auto original (en-orig). A plain auto key (possibly a translation) is never chosen --
    except as a last resort: when an ASR track is not translatable, yt-dlp registers only the
    plain "<lang>" key with no "-orig" twin, so a plain auto key is safe to use only when no
    "-orig" key exists anywhere in auto (otherwise plain keys are translations)."""
    for lang in languages:
        if lang in manual:
            return lang
        prefix_matches = sorted(k for k in manual if k.lower().startswith(f"{lang.lower()}-"))
        if prefix_matches:
            return prefix_matches[0]
        orig_key = f"{lang}-orig"
        if orig_key in auto:
            return orig_key
    if not any(k.endswith("-orig") for k in auto):
        for lang in languages:
            if lang in auto:
                return lang
    return None


class YtDlpSubsProvider:
    name = "ytdlp_subs"
    kinds = frozenset({"youtube", "video"})

    def transcribe(self, ref: SourceRef, languages: list[str]) -> Transcript:
        if ref.kind not in self.kinds:
            raise TranscriptionError(f"ytdlp_subs: cannot handle {ref.kind} sources")
        video_id, url = ref.key, ref.url
        try:
            with tempfile.TemporaryDirectory(prefix="hmd-subs-", ignore_cleanup_errors=True) as tmp:
                opts = {
                    "skip_download": True, "writesubtitles": True, "writeautomaticsub": True,
                    "subtitlesformat": "json3",
                    "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"), "quiet": True, "no_warnings": True,
                    "socket_timeout": 30, "retries": 3, "extractor_retries": 2,
                    # We skip the actual video download, so missing/unplayable formats must
                    # not fail caption extraction.
                    "ignore_no_formats_error": True,
                    "noplaylist": True,       # a playlist/channel URL must never fan out
                }
                with yt_dlp.YoutubeDL(opts) as ydl:
                    # Metadata only, no download yet: raw subtitle/caption tracks available.
                    info = ydl.extract_info(url, download=False, process=False) or {}
                    manual = info.get("subtitles") or {}
                    auto = info.get("automatic_captions") or {}
                    chosen_key = _select_track(manual, auto, languages)
                    if chosen_key is None:
                        raise TranscriptionError(
                            f"ytdlp_subs: no subtitle track for {video_id} in {languages}; "
                            f"manual={sorted(manual)}, auto={sorted(auto)}"
                        )
                    # Full-match regex: escape so only this exact key is ever selected.
                    ydl.params["subtitleslangs"] = [re.escape(chosen_key)]
                    # download=True re-processes the already-fetched info with the narrowed
                    # subtitleslangs and writes the file; skip_download=True means the video
                    # itself is never fetched. process_ie_result raises DownloadError (via
                    # report_error/ignoreerrors=False by default) on failure, so we don't need
                    # to inspect its return value.
                    ydl.process_ie_result(info, download=True)
                # yt-dlp resolves the subtitle path via prepare_filename(info, "temp") (falls
                # back to the default outtmpl, so no ".part" suffix) then subtitles_filename(),
                # giving exactly "<id>.<key>.json3"; skip_download=True makes the final
                # temp -> destination move for the video itself a no-op.
                video_ref = info.get("id") or video_id
                filename = f"{video_ref}.{chosen_key}.json3"
                if filename not in os.listdir(tmp):
                    raise TranscriptionError(
                        f"ytdlp_subs: no subtitle track written for {video_id} ({chosen_key})"
                    )
                with open(os.path.join(tmp, filename), encoding="utf-8") as f:
                    data = json.load(f)
                segments = parse_json3(data)
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(f"ytdlp_subs: {type(e).__name__}: {e}") from e
        if not segments:
            raise TranscriptionError(f"ytdlp_subs: empty track for {video_id}")
        is_generated = chosen_key not in manual
        language = chosen_key[: -len("-orig")] if chosen_key.endswith("-orig") else chosen_key
        return Transcript(video_id=video_id, method=self.name, language=language, segments=segments,
                          extra={"is_generated": is_generated, "track": chosen_key})
