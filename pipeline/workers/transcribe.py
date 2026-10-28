"""Per-source transcription worker, called by the runner's thread pool or by hand:

    python -m pipeline.workers.transcribe --source-id 3
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

from pipeline import db
from pipeline.config import Config, load_config
from pipeline.sources import SourceRef
from pipeline.transcription.base import Transcriber, TranscriptionError
from pipeline.transcription.cache import TranscriptCache
from pipeline.transcription.chain import build_transcribers, transcribe_with_chain


def transcribe_source(db_path: str, source_id: int, cfg: Config, *, transcribers: list[Transcriber] | None = None,
                      cache: TranscriptCache | None = None, log: Callable[[str], None] = print,
                      force: bool = False) -> str:
    """Returns 'done', 'error' or 'skipped'. force=True re-transcribes even a 'done' source with
    an existing transcript file, bypassing the cache too; used by the runner's --retranscribe
    path."""
    source = db.get_source(db_path, source_id)
    if not source:
        raise ValueError(f"source {source_id} not found")
    if (not force and source["transcribe_status"] == "done" and source["transcript_path"]
            and os.path.exists(source["transcript_path"])):
        return "skipped"
    transcribers = transcribers if transcribers is not None else build_transcribers(cfg)
    cache = cache or TranscriptCache(cfg.transcript_cache_dir)
    ref = SourceRef.from_row(source)
    video_id = ref.key
    try:
        transcript = transcribe_with_chain(ref, transcribers, cfg.transcription.languages, cache, log=log,
                                           force=force, expected_duration=source["duration_seconds"])
        # cache.path() cannot raise here: transcript.method was already validated by
        # cache.save()/cache.load() inside the chain.
        db.update_transcribe_status(db_path, source_id, "done", transcript_path=cache.path(video_id, transcript.method),
                                    method=transcript.method, language=transcript.language)
        # A (re)transcription invalidates any existing analysis of this source -- do this here,
        # atomically with the transcribe status update, rather than as a separate post-pool step
        # in the runner, so the two can never drift apart (e.g. a crash between the two, or a
        # caller that calls transcribe_source directly without going through the runner's pool).
        db.reset_analyze_status(db_path, source["notebook_id"], source_ids=[source_id])
    except (TranscriptionError, ValueError) as e:
        db.update_transcribe_status(db_path, source_id, "error", error=str(e))
        log(f"  [{video_id}] ERROR: {e}")
        return "error"
    except Exception as e:
        msg = f"unexpected {type(e).__name__}: {e}"
        db.update_transcribe_status(db_path, source_id, "error", error=msg)
        log(f"  [{video_id}] ERROR: {msg}")
        return "error"
    return "done"


def main(argv: list[str] | None = None) -> None:
    from pipeline.cli import configure_console   # deferred: cli imports the runner, which imports this module
    configure_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id", type=int, required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    status = transcribe_source(cfg.db_path, args.source_id, cfg)
    print(status)
    if status == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
