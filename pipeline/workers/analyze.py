"""Per-source analysis worker: transcript -> prompt -> schema-validated JSON -> anchored claims,
called by the runner's thread pool or by hand:

    python -m pipeline.workers.analyze --source-id 3 --topic "Acme SEO"
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import Callable

from pipeline import db
from pipeline.anchors import QUOTE_PAD_SEGMENTS, resolve_segment_claims
from pipeline.config import CONFIG_PATH, load_config
from pipeline.llm.base import LLMError, LLMFatalError, LLMInterrupted
from pipeline.llm.client import BudgetExceeded, LLMClient
from pipeline.llm.factory import make_client, make_provider
from pipeline.prompts.analyze import claim_target, get_analyze_prompt
from pipeline.schemas import AnalysisDraft
from pipeline.sources import kind_of
from pipeline.transcription.base import TranscriptionError
from pipeline.transcription.cache import TranscriptCache


def analyze_source(db_path: str, source_id: int, topic: str, llm: LLMClient, model: str, cache: TranscriptCache, *,
                   force: bool = False, log: Callable[[str], None] = print) -> str:
    """Returns 'done', 'error' or 'skipped'. BudgetExceeded propagates so the runner can pause;
    LLMFatalError propagates so the runner can stop the run; LLMInterrupted (Ctrl+C) propagates too.
    All three leave the source untouched."""
    source = db.get_source(db_path, source_id)
    if not source:
        raise ValueError(f"source {source_id} not found")
    if source["analyze_status"] == "done" and source["analysis_json"] and not force:
        return "skipped"
    video_id = source["video_id"]
    if source["transcribe_status"] != "done" or not source["transcript_method"]:
        db.update_analyze_status(
            db_path, source_id, "error",
            error=f"source is not transcribed (transcribe_status={source['transcribe_status']})")
        return "error"
    try:
        transcript = cache.load(video_id, source["transcript_method"])
        flat = transcript.flatten()
        lo, hi = claim_target(transcript.word_count())
    except (TranscriptionError, ValueError) as e:
        db.update_analyze_status(db_path, source_id, "error", error=f"transcript unreadable: {e}")
        return "error"

    if not flat.text.strip():
        db.update_analyze_status(db_path, source_id, "error", error="transcript is empty")
        return "error"

    segments = flat.segment_texts()
    prompt = get_analyze_prompt(topic, source["title"] or "", source["channel"] or "", segments, lo, hi,
                                 kind=kind_of(source))
    # Numbering adds roughly 10-15% input per call (DECISION-027); logged so long sources show it.
    log(f"  [{video_id}] analyze prompt: {len(segments)} segments, {len(prompt)} chars")
    try:
        draft = llm.call_json(prompt, AnalysisDraft, model=model, stage="analyze", source_id=source_id)
    except (BudgetExceeded, LLMFatalError, LLMInterrupted):
        # None is this source's fault: the source keeps its status so `hmd resume` retries it.
        raise
    except LLMError as e:
        db.update_analyze_status(db_path, source_id, "error", error=f"{type(e).__name__}: {e}")
        log(f"  [{video_id}] ERROR: {e}")
        return "error"
    except Exception as e:   # unexpected provider failures the client didn't wrap
        msg = f"unexpected {type(e).__name__}: {e}"
        db.update_analyze_status(db_path, source_id, "error", error=msg)
        log(f"  [{video_id}] ERROR: {msg}")
        return "error"

    try:
        analysis = resolve_segment_claims(draft, flat, pad=QUOTE_PAD_SEGMENTS)
        if not analysis.key_claims:
            if not analysis.dropped_quotes:
                error = "the model returned no claims for this transcript"
            else:
                reasons = Counter(d.get("reason", "unknown") for d in analysis.dropped_quotes)
                reason_str = ", ".join(f"{n} {r}" for r, n in reasons.most_common())
                error = (f"no claim had a usable segment range ({len(analysis.dropped_quotes)} dropped: "
                         f"{reason_str}; transcript_method={source['transcript_method']})")
            db.update_analyze_status(db_path, source_id, "error", error=error,
                                     analysis_json=analysis.model_dump_json())
            return "error"
        db.update_analyze_status(db_path, source_id, "done", analysis_json=analysis.model_dump_json())
    except Exception as e:
        msg = f"unexpected {type(e).__name__}: {e}"
        db.update_analyze_status(db_path, source_id, "error", error=msg)
        log(f"  [{video_id}] ERROR: {msg}")
        return "error"
    n = len(analysis.key_claims)
    log(f"  [{video_id}] {n} claim{'' if n == 1 else 's'} anchored, {len(analysis.dropped_quotes)} dropped")
    return "done"


def main(argv: list[str] | None = None) -> None:
    from pipeline.cli import configure_console   # deferred: cli imports the runner, which imports this module
    configure_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id", type=int, required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config or CONFIG_PATH, explicit=args.config is not None)
    source = db.get_source(cfg.db_path, args.source_id)
    if not source:
        print(f"source {args.source_id} not found", file=sys.stderr)
        sys.exit(1)
    provider = make_provider(cfg)
    llm = make_client(cfg, provider, db_path=cfg.db_path, run_id=None, notebook_id=source["notebook_id"],
                      budget_usd=cfg.run.budget_usd)
    cache = TranscriptCache(cfg.transcript_cache_dir)
    try:
        status = analyze_source(cfg.db_path, args.source_id, args.topic, llm, cfg.llm.models["analyze"], cache,
                                force=args.force)
    except (BudgetExceeded, LLMFatalError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    print(status)
    if status == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
