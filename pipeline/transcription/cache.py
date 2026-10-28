"""Transcript cache: cache/transcripts/<source key>/<method>.json (IDEA-084, DECISION-016).

Immutable, global across notebooks and domains, never inside the vault. Delete a file to
force re-transcription with that method.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone

from pipeline.sources import is_source_key
from pipeline.transcription.base import Segment, Transcript, TranscriptionError
from pipeline.transcription.scrub import scrub_segments

METHOD_RE = re.compile(r"^[a-z_]{1,32}\Z")


def _segment_dict(s: Segment) -> dict:
    d = {"start": float(s.start), "end": float(s.end), "text": s.text}
    if s.page is not None:
        d["page"] = s.page
    return d


class TranscriptCache:
    def __init__(self, root: str):
        self.root = root

    def path(self, key: str, method: str) -> str:
        if not is_source_key(key):
            raise ValueError(f"not a source key: {key!r}")
        if not METHOD_RE.match(method):
            raise ValueError(f"not a method name: {method!r}")
        return os.path.join(self.root, key, f"{method}.json")

    def has(self, key: str, method: str) -> bool:
        return os.path.exists(self.path(key, method))

    def available(self, key: str) -> list[str]:
        if not is_source_key(key):
            raise ValueError(f"not a source key: {key!r}")
        d = os.path.join(self.root, key)
        if not os.path.isdir(d):
            return []
        return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))

    def save(self, transcript: Transcript) -> str:
        """Write transcript to cache/<key>/<method>.json, atomically.

        Secret-shaped strings are scrubbed from the segments first (in place; the count goes in
        extra["redacted"]). Stamps transcript.created_at in place if it is empty, and returns the final path.
        """
        path = self.path(transcript.video_id, transcript.method)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        redacted = scrub_segments(transcript.segments)
        if redacted:
            transcript.extra["redacted"] = redacted
        if not transcript.created_at:
            transcript.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        data = {
            "schema": 1,
            "video_id": transcript.video_id,
            "method": transcript.method,
            "language": transcript.language,
            "created_at": transcript.created_at,
            "extra": transcript.extra,
            "segments": [_segment_dict(s) for s in transcript.segments],
        }
        kind = transcript.extra.get("kind")
        if kind and kind != "youtube":
            data["kind"] = kind
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=f"{transcript.method}.", suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        try:
            os.replace(tmp, path)
        except OSError as e:
            # On Windows, two writers racing on the same (key, method) can make
            # os.replace raise PermissionError because the other writer holds the
            # destination; the file itself is never corrupted. If a valid cache entry
            # is present after that, the other writer won the race and we're done.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            if os.path.exists(path):
                try:
                    self.load(transcript.video_id, transcript.method)
                    return path
                except TranscriptionError:
                    pass
            raise TranscriptionError(f"cache: cannot write {path}: {e}") from e
        return path

    def load(self, key: str, method: str) -> Transcript:
        path = self.path(key, method)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            segments = []
            for s in data["segments"]:
                text = s["text"]
                if not isinstance(text, str):
                    raise TypeError(f"segment text must be a string, got {type(text).__name__}")
                page = s.get("page")
                segments.append(Segment(float(s["start"]), float(s["end"]), text, int(page) if page is not None else None))
            language = data.get("language", "und")
            created_at = data.get("created_at", "")
            extra = data.get("extra", {})
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise TranscriptionError(f"cache: unreadable {path}: {e}") from e
        return Transcript(video_id=key, method=method, language=language,
                          segments=segments, created_at=created_at, extra=extra)

    def load_best(self, key: str, preference: list[str]) -> Transcript | None:
        if not is_source_key(key):
            raise ValueError(f"not a source key: {key!r}")
        present = set(self.available(key))
        for method in preference:
            if method in present:
                try:
                    return self.load(key, method)
                except TranscriptionError:
                    continue
        return None
