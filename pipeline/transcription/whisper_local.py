"""Last-resort transcriber: local Whisper.

- Audio is downloaded as `bestaudio` in its original container; Whisper decodes it through
  ffmpeg itself, so no transcode to mp3 (IDEA-272). ffmpeg must be on PATH for Whisper.
- Transcription runs in a child process (`python -m pipeline.transcription.whisper_local`)
  so the per-notebook-source timeout can kill it (IDEA-271). The child needs `openai-whisper`
  (requirements-whisper.txt); `hmd doctor` reports GPU availability.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import yt_dlp

from pipeline.sources import MAX_FILE_BYTES, SourceRef
from pipeline.transcription import media_guard
from pipeline.transcription.base import Segment, Transcript, TranscriptionError

# The child is launched as `python -m pipeline.transcription.whisper_local`, which requires
# `pipeline` to be importable. That only holds automatically when the parent's CWD is the
# repo root; stage scripts assume that convention (see the README), but a caller that runs
# from elsewhere would otherwise hit "ModuleNotFoundError: No module named 'pipeline'" in the
# child. Prepending the repo root to the child's PYTHONPATH makes the import work regardless.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHILD_ENV_DROP = frozenset({"ANTHROPIC_API_KEY", "SUPADATA_API_KEY"})

# Bound on the downloaded audio file so a single runaway (very long or misidentified) video can't
# tie up the whole run: yt-dlp's own "max_filesize" option skips a download over this size, which
# then surfaces here as the ordinary "produced no file" TranscriptionError below. Best-effort only:
# yt-dlp enforces max_filesize in its HTTP downloader when the server sends a Content-Length header
# up front, but a fragmented format (DASH/HLS, common for YouTube) or an external downloader can
# still exceed it -- there is no universal pre-download size check across every code path.
MAX_AUDIO_BYTES = MAX_FILE_BYTES["media"]


class WhisperLocalProvider:
    name = "whisper_local"
    kinds = frozenset({"youtube", "video", "media"})

    def __init__(self, model_size: str = "base", timeout_seconds: int = 1800):
        self.model_size = model_size
        self.timeout_seconds = timeout_seconds

    def download_audio(self, url: str, tmpdir: str) -> str:
        opts = {
            "format": "bestaudio/best", "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
            "quiet": True, "no_warnings": True, "postprocessors": [],
            "socket_timeout": 30, "retries": 3, "extractor_retries": 2,
            "max_filesize": MAX_AUDIO_BYTES,
            "noplaylist": True, "playlist_items": "1",   # one video, never a whole playlist/channel
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                media_guard.install(ydl)           # format and manifest hosts must be public
                ydl.download([url])
        except Exception as e:
            raise TranscriptionError(f"whisper_local: audio download failed: {type(e).__name__}: {e}") from e
        # yt-dlp leaves ".part"/".ytdl"/".temp" siblings behind for in-progress or resumed
        # downloads; only a fully written file is a valid pick.
        # Also skip fragment leftovers (".part-FragN", ".f137.mp4" format pieces); the largest
        # remaining candidate is the merged file.
        files = sorted(
            (f for f in glob.glob(os.path.join(tmpdir, "audio.*"))
             if not f.endswith((".ytdl", ".temp")) and ".part" not in os.path.basename(f)
             and not re.search(r"\.f\d+\.", os.path.basename(f))),
            key=lambda f: (-os.path.getsize(f), f))
        if not files:
            raise TranscriptionError(
                f"whisper_local: audio download produced no file for {url} "
                f"(it may exceed the {MAX_AUDIO_BYTES // (1024 * 1024)} MB cap)")
        return files[0]

    @staticmethod
    def _stage_local_file(ref: SourceRef, tmpdir: str) -> str:
        """Copy a local media file into tmpdir so the worker's whisper.json never lands beside
        the user's file; the size cap is checked first."""
        path = ref.path
        if not path:
            raise TranscriptionError(f"whisper_local: {ref.url} is not a local file")
        if not os.path.isfile(path) or os.path.islink(path):
            raise TranscriptionError(f"whisper_local: {path} is not a regular file")
        try:
            size = os.path.getsize(path)
        except OSError as e:
            raise TranscriptionError(f"whisper_local: cannot read {path}: {e}") from e
        if size > MAX_AUDIO_BYTES:
            raise TranscriptionError(
                f"whisper_local: {path} is {size // (1024 * 1024)} MB, over the "
                f"{MAX_AUDIO_BYTES // (1024 * 1024)} MB cap")
        dest = os.path.join(tmpdir, "audio" + os.path.splitext(path)[1].lower())
        try:
            shutil.copyfile(path, dest)
            copied = os.path.getsize(dest)
        except OSError as e:
            raise TranscriptionError(f"whisper_local: cannot read {path}: {e}") from e
        if copied > MAX_AUDIO_BYTES:      # the file grew between the check and the copy
            os.remove(dest)
            raise TranscriptionError(
                f"whisper_local: {path} grew past the {MAX_AUDIO_BYTES // (1024 * 1024)} MB cap while copying")
        return dest

    def transcribe_file(self, audio_path: str) -> tuple[list[Segment], str]:
        out_path = os.path.join(os.path.dirname(audio_path), "whisper.json")
        # -P and cwd=<our temp dir>: the user's working directory is neither on the child's
        # sys.path (a planted pipeline/ or json.py there would run) nor where Windows looks first
        # for the ffmpeg Whisper starts. The child needs no API key.
        argv = [sys.executable, "-P", "-m", "pipeline.transcription.whisper_local",
                "--audio", audio_path, "--model", self.model_size, "--out", out_path]
        env = {k: v for k, v in os.environ.items() if k not in CHILD_ENV_DROP}
        env["PYTHONPATH"] = os.pathsep.join(p for p in [_REPO_ROOT, os.environ.get("PYTHONPATH", "")] if p)
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  timeout=self.timeout_seconds, env=env, cwd=os.path.dirname(audio_path),
                                  shell=False)
        except subprocess.TimeoutExpired as e:
            # Windows note: subprocess.run's timeout path calls Popen.kill(), which only
            # terminates the direct child (this worker's Python process) -- it does not signal
            # the ffmpeg process Whisper spawns underneath it. In practice ffmpeg exits on its
            # own shortly after, once the pipes it was writing to are closed underneath it.
            raise TranscriptionError(f"whisper_local: exceeded {self.timeout_seconds}s on {audio_path}") from e
        if proc.returncode != 0:
            raise TranscriptionError(f"whisper_local: worker failed: {proc.stderr.strip()[-500:]}")
        stderr_tail = proc.stderr.strip()[-500:]
        try:
            with open(out_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            raise TranscriptionError(f"whisper_local: worker wrote no usable output: {stderr_tail}") from e
        if not isinstance(data, dict):
            raise TranscriptionError(f"whisper_local: worker output not a JSON object: {stderr_tail}")
        try:
            segments = [Segment(float(s["start"]), float(s["end"]), s["text"]) for s in data.get("segments", [])]
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise TranscriptionError(f"whisper_local: worker output not parseable: {data!r}"[:500]) from e
        return segments, data.get("language") or "und"

    def transcribe(self, ref: SourceRef, languages: list[str]) -> Transcript:
        # `languages` is unused: Whisper auto-detects language from the audio itself. By the
        # time we've fallen through to this last-resort provider, forcing a configured language
        # onto a video whose language we don't actually know would risk garbage output, so we
        # trust Whisper's own detection -- the resulting Transcript.language may end up outside
        # the caller's configured `languages` list.
        try:
            with tempfile.TemporaryDirectory(prefix="hmd-whisper-", ignore_cleanup_errors=True) as tmp:
                if ref.kind == "media":
                    audio = self._stage_local_file(ref, tmp)
                else:
                    audio = self.download_audio(ref.url, tmp)
                segments, language = self.transcribe_file(audio)
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(f"whisper_local: {type(e).__name__}: {e}") from e
        if not any(s.text.strip() for s in segments):
            raise TranscriptionError(f"whisper_local: empty transcript for {ref.key}")
        return Transcript(video_id=ref.key, method=self.name, language=language, segments=segments,
                          extra={"model": self.model_size, "is_generated": True})


def _worker(argv: list[str] | None = None) -> None:
    """Child-process entry point. Writes {"language": ..., "segments": [...]} as JSON to --out."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="base")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    import whisper   # only in the child; the parent never imports torch
    model = whisper.load_model(args.model)
    result = model.transcribe(args.audio)
    payload = {"language": result.get("language"),
               "segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in result["segments"]]}
    # The parent reads this file with open(..., encoding="utf-8"); ensure_ascii=True keeps the
    # bytes on disk plain ASCII regardless, so the encoding round-trip never depends on locale.
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)


if __name__ == "__main__":
    _worker()
