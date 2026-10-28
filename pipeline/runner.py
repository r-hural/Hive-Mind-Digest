"""Orchestrator (DECISION-003, IDEA-052): stages in order, a thread pool for per-source work,
run rows and a usage summary, pause on the per-run budget (IDEA-280).

Merge error policy: a section, outline or verification call that fails on a non-transient error
(anything `LLMClient` doesn't already retry) discards the whole merge -- see
pipeline/stages/merge.py's module docstring for the all-or-nothing invariant. The runner does not
retry a whole merge on top of that: `LLMClient` already retries transient provider errors up to
`llm.retry_attempts` times and performs one schema-repair round-trip per call, so a failure that
still reaches the runner is a deterministic one (bad prompt, broken schema, exhausted budget) that
a bare retry would not fix -- it would only double the spend. The run is simply marked "error" (or
"paused" for a budget stop) and its spend and usage are recorded exactly as they stand; a human
re-runs `hmd resume <notebook>` after fixing the underlying cause.

Fatal LLM errors (LLMFatalError: auth/permission, billing or credit balance, subscription usage
limit) stop the run at once instead: the pool stops scheduling (items not yet started never run and
keep their status), in-flight items finish, no later stage runs, and the run row is closed "error"
with the provider's message; RunResult.fatal tells the CLI to print its one-line ERROR hint.

Pipeline order for a requested set of `stages`: fetch -> per-source metadata (whenever "fetch" or
"transcribe" is requested: it fills in what fetch just registered, but also backfills any source
whose yt-dlp metadata call failed in an earlier run, since `hmd resume` defaults to
transcribe,analyze,merge,output and such a source would otherwise never get a title/upload_date --
the pool's own todo filter makes the extra check cheap when nothing is pending) -> transcribe pool
-> analyze pool -> one `run_merge` call -> `run_output`. Every LLM-calling step shares one
`LLMClient` bound to this run's row and budget, so `db.run_cost`/`run_usage_summary` return exactly
what this run spent.
"""
from __future__ import annotations

import os
import sqlite3
import statistics
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from pipeline import db
from pipeline.config import Config, ConfigError
from pipeline.llm.base import LLMFatalError, LLMProvider, clear_interrupt, request_interrupt, wait_futures
from pipeline.llm.client import BudgetExceeded
from pipeline.llm.factory import make_client, make_provider
from pipeline.pricing import price_for
from pipeline.slug import slugify, unique_slug
from pipeline.termtext import terminal_safe
from pipeline.stages.fetch import check_input_file, fetch_source_metadata, fetch_stage
from pipeline.stages.merge import run_merge
from pipeline.stages.output import render_reports, report_path_for
from pipeline.transcription.base import Transcriber
from pipeline.transcription.cache import TranscriptCache
from pipeline.transcription.chain import build_transcribers
from pipeline.workers.analyze import analyze_source
from pipeline.workers.transcribe import transcribe_source

STAGES = ["fetch", "transcribe", "analyze", "merge", "output"]


@dataclass
class RunResult:
    run_id: int
    notebook_id: int
    status: str   # done | paused | error
    message: str = ""
    report_path: str | None = None
    cost_usd: float = 0.0
    report_html_path: str | None = None   # the summary's HTML twin, when this run wrote it
    fatal: bool = False   # status "error" from an LLMFatalError: the account/key must be fixed first


class _NotStarted(Exception):
    """A pool item skipped because an LLMFatalError had already stopped the pool."""


class _SerializedTranscriber:
    """Wraps a whisper_local transcriber so at most one Whisper job runs at a time (its CPU/GPU
    decode is heavy and single-machine-bound), while other transcribers in the same chain still
    run in parallel across the pool. `name` is preserved so the wrapped instance still satisfies
    the Transcriber protocol and transcribe_with_chain's cache lookup (which keys on t.name)."""

    def __init__(self, inner: Transcriber, lock: threading.Lock):
        self.name = inner.name
        self.kinds = inner.kinds
        self._inner = inner
        self._lock = lock

    def transcribe(self, ref, languages):
        with self._lock:
            return self._inner.transcribe(ref, languages)


def usage_report(db_path: str, run_id: int) -> str:
    rows = db.run_usage_summary(db_path, run_id)
    if not rows:
        return "No LLM calls in this run."
    header = (f"{'stage':<16}{'provider':<16}{'model':<32}{'calls':>6}{'in_tok':>10}{'out_tok':>10}"
              f"{'cache_r':>10}{'cache_w':>10}{'time_s':>9}{'cost':>10}")
    lines = [header]
    total_cost = 0.0
    total_seconds = 0.0
    for r in rows:
        seconds = (r["duration_ms"] or 0) / 1000.0
        cost = r["cost_usd"] or 0.0
        total_cost += cost
        total_seconds += seconds
        lines.append(
            f"{r['stage']:<16}{r['provider']:<16}{r['model']:<32}{r['calls']:>6}"
            f"{r['input_tokens'] or 0:>10}{r['output_tokens'] or 0:>10}"
            f"{r['cache_read_tokens'] or 0:>10}{r['cache_write_tokens'] or 0:>10}{seconds:>9.1f}{cost:>10.4f}")
    lines.append(f"Total cost: ${total_cost:.4f}")
    lines.append(f"Total time in LLM calls: {total_seconds:.1f}s")
    return "\n".join(lines)


def _timing_table(events: list[dict]) -> list[str]:
    """Rows grouped by provider x stage: calls, wall-time stats per call, tokens (including cache
    reads/writes), cost.

    Time stats cover only calls with a recorded duration; cost is '-' when no call in the group
    has one (claude_cli reports none). Every row is indented so none can start a KEY=value line.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for e in events:
        groups.setdefault((e["provider"], e["stage"]), []).append(e)
    lines = [f"  {'provider':<16}{'stage':<16}{'calls':>6}{'total_s':>9}{'mean_s':>8}{'p50_s':>8}{'max_s':>8}"
             f"{'in_tok':>10}{'out_tok':>9}{'cache_r':>10}{'cache_w':>9}{'cost':>10}"]
    for (provider, stage), rows in sorted(groups.items()):
        secs = [r["duration_ms"] / 1000.0 for r in rows if r["duration_ms"] is not None]
        if secs:
            timing = (f"{sum(secs):>9.1f}{sum(secs) / len(secs):>8.1f}{statistics.median(secs):>8.1f}"
                      f"{max(secs):>8.1f}")
        else:
            timing = f"{'-':>9}{'-':>8}{'-':>8}{'-':>8}"
        costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
        cost = f"{sum(costs):>10.4f}" if costs else f"{'-':>10}"
        in_tok = sum(r["input_tokens"] or 0 for r in rows)
        out_tok = sum(r["output_tokens"] or 0 for r in rows)
        cache_r = sum(r["cache_read_tokens"] or 0 for r in rows)
        cache_w = sum(r["cache_write_tokens"] or 0 for r in rows)
        lines.append(f"  {provider:<16}{stage:<16}{len(rows):>6}{timing}{in_tok:>10}{out_tok:>9}"
                     f"{cache_r:>10}{cache_w:>9}{cost}")
    return lines


def timing_report(db_path: str, notebook_id: int, run_id: int) -> str:
    """Per-call wall time by provider x stage for one run, plus an all-runs total when the notebook
    has LLM calls in more than one run (DECISION-024 data). Empty string when there are no calls."""
    events = db.notebook_usage_events(db_path, notebook_id)
    run_events = [e for e in events if e["run_id"] == run_id]
    lines: list[str] = []
    if run_events:
        lines.append(f"Call timing, run {run_id} (provider x stage):")
        lines += _timing_table(run_events)
    run_ids = {e["run_id"] for e in events}
    if len(run_ids) > 1:
        lines.append(f"Call timing, all runs ({len(run_ids)} runs with LLM calls):")
        lines += _timing_table(events)
    return "\n".join(lines)


def safe_log(log: Callable[[str], None]) -> Callable[[str], None]:
    """Wrap a log callable so a logging failure can never abort a stage (2026-10-09: a cp1252
    console raised UnicodeEncodeError on a point quoted by merge:check and killed a billed run at
    the end). A line that raises is retried once ASCII-escaped ("\\xd7" for the multiplication
    sign); if that raises too, the line is dropped. KeyboardInterrupt and other BaseExceptions are
    not caught. Control characters are replaced first (pipeline/termtext.py). Every stage, worker
    and merge call receives the runner's log, so wrapping here covers every call site."""
    def _log(msg: str) -> None:
        msg = terminal_safe(msg)       # error text and titles come from remote pages and APIs
        try:
            log(msg)
        except Exception:
            try:
                log(str(msg).encode("ascii", "backslashreplace").decode("ascii"))
            except Exception:
                pass
    return _log


class Runner:
    def __init__(self, cfg: Config, *, provider: LLMProvider | None = None,
                 transcribers: list[Transcriber] | None = None,
                 metadata_fn: Callable[..., str] | None = None,
                 log: Callable[[str], None] = print, analyze_model: str | None = None,
                 merge_model: str | None = None, budget_usd: float | None = None,
                 max_parallel: int | None = None, show_anchors: bool | None = None,
                 formats: list[str] | None = None, html: bool | None = None):
        self.cfg = cfg
        self.provider = provider or make_provider(cfg)
        # None means "use the module-global default at call time" (see the module docstring):
        # never resolved here, so a test's monkeypatch of pipeline.runner.build_transcribers or
        # pipeline.runner.fetch_source_metadata is still honoured.
        self.transcribers = transcribers
        self.metadata_fn = metadata_fn
        self.log = safe_log(log)
        self.analyze_model = analyze_model if analyze_model is not None else cfg.llm.models["analyze"]
        self.merge_model = merge_model if merge_model is not None else cfg.llm.models["merge"]
        self.budget_usd = cfg.run.budget_usd if budget_usd is None else budget_usd
        # The price gate lives here, and only here (pipeline/cli.py::_runner no longer duplicates
        # it), so any caller -- a future `hmd eval`, an MCP tool, a test harness --
        # that constructs a Runner directly cannot start a budgeted anthropic_api run against a
        # model pipeline/pricing.py cannot price (its spend would never be counted against the
        # budget). This still rejects a bad model before a notebook exists in `hmd run`, because
        # the CLI constructs the Runner before calling create_notebook.
        # An OfflineProvider (a run with no LLM stage, see pipeline/cli.py::_runner) prices nothing.
        self.offline = bool(getattr(self.provider, "offline", False))
        if self.provider.name == "anthropic_api" and self.budget_usd > 0 and not self.offline:
            for model in (self.analyze_model, self.merge_model):
                if price_for(model) is None:
                    raise ConfigError(
                        f"no price for model {model!r} in pipeline/pricing.py; add it or run with --budget 0")
        self.max_parallel = max(1, max_parallel if max_parallel is not None else cfg.llm.max_parallel)
        # A batch gathers the calls that are in flight together, so the LLM pools must be as wide as a
        # wave; transcription keeps max_parallel.
        self.batching = bool(getattr(self.provider, "batching", False))
        if self.batching:
            # the batch provider logs from its own threads (sent, progress, ended): route those lines
            # through the runner's log like every other stage line, not straight to stdout
            self.provider.log = self.log
        self.llm_parallel = max(self.max_parallel, cfg.llm.batch.max_wave) if self.batching else self.max_parallel
        self.show_anchors = cfg.output.show_anchors if show_anchors is None else show_anchors
        self.html = cfg.output.html if html is None else html
        self.formats = list(formats) if formats else list(cfg.output.formats)
        self.cache = TranscriptCache(cfg.transcript_cache_dir)
        self._whisper_lock = threading.Lock()

    # --- notebooks -----------------------------------------------------------

    def create_notebook(self, topic: str) -> dict:
        # Collapse whitespace runs here (not only in the CLI -- see pipeline/cli.py::cmd_run) so
        # the KEY=value forging guard (a topic containing a raw "\n" could otherwise start a fake
        # KEY=value line where the topic is echoed back, e.g. by `status`/`notebook list`) does
        # not depend on going through the CLI: any caller of create_notebook is protected.
        topic = " ".join(topic.split())
        slug = unique_slug(slugify(topic), db.notebook_slugs(self.cfg.db_path))
        nid = db.create_notebook(self.cfg.db_path, topic, slug)
        # The topic itself is untrusted, user-supplied text -- never echo it here: this log line
        # can reach stdout, and a topic containing e.g. "REPORT=" must not be able to forge a
        # KEY=value line a skill parses (see pipeline/cli.py's module docstring).
        self.log(f"Created notebook {nid} ({slug!r})")
        return db.get_notebook(self.cfg.db_path, nid)

    # --- thread pool ----------------------------------------------------------

    def _pool(self, fn: Callable, items: list, workers: int | None = None) -> list:
        """Run fn(item) for every item with up to self.max_parallel workers. Every future is
        waited on (so every billed call is logged) before any exception is raised: the first
        BudgetExceeded seen wins over any other exception, so the runner always pauses rather
        than errors when the budget is what actually stopped the run -- except LLMFatalError,
        which wins over both (a resume after a budget pause would only hit it again). The first
        LLMFatalError also stops the pool: items not yet started are skipped (never run), items
        already running finish.

        Built without the `with ThreadPoolExecutor(...)` context manager on purpose: that context
        manager's __exit__ always calls shutdown(wait=True), which -- on a KeyboardInterrupt or
        other BaseException raised while a caller is blocked in fut.result() -- would still sit
        there waiting for every already-queued item to run (and bill) before the interrupt is
        allowed to propagate. Here, a BaseException instead cancels every not-yet-started item
        immediately (cancel_futures=True) and only *then* waits (in `finally`) for whatever was
        already running to actually finish, so we never leave an orphaned worker thread touching
        the DB or filesystem behind us.

        Waits in short slices (wait_futures) so Ctrl+C is seen at once on Windows, and raises the
        interrupt flag first so a worker blocked on a batch is released."""
        if not items:
            return []
        results: list = [None] * len(items)
        fatal_exc: LLMFatalError | None = None
        budget_exc: BudgetExceeded | None = None
        other_exc: Exception | None = None
        stop = threading.Event()

        def run_one(item):
            # The first LLMFatalError sets `stop`; every item not yet started is then skipped
            # without running (its source keeps its status for `hmd resume`), while items already
            # running finish on their own.
            if stop.is_set():
                raise _NotStarted()
            try:
                return fn(item)
            except LLMFatalError:
                stop.set()
                raise

        pool = ThreadPoolExecutor(max_workers=workers or self.max_parallel)
        try:
            futures = {pool.submit(run_one, item): i for i, item in enumerate(items)}
            for fut, i in futures.items():
                wait_futures([fut])       # in item order, as fut.result() did, but sliced
                try:
                    results[i] = fut.result()
                except _NotStarted:
                    pass
                except LLMFatalError as e:
                    if fatal_exc is None:
                        fatal_exc = e
                except BudgetExceeded as e:
                    if budget_exc is None:
                        budget_exc = e
                except Exception as e:
                    if other_exc is None:
                        other_exc = e
        except BaseException:
            request_interrupt()           # a batch-blocked worker stops waiting, so shutdown can finish
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            pool.shutdown(wait=True)
        if fatal_exc is not None:
            raise fatal_exc
        if budget_exc is not None:
            raise budget_exc
        if other_exc is not None:
            raise other_exc
        return results

    def _log_counts(self, statuses: list[str]) -> None:
        c = Counter(statuses)
        parts = ", ".join(f"{c[k]} {k}" for k in ("done", "error", "skipped") if c[k])
        self.log(f"  {parts or 'nothing to do'}")

    def _resolve_transcribers(self) -> list:
        transcribers = self.transcribers if self.transcribers is not None else build_transcribers(self.cfg)
        return [_SerializedTranscriber(t, self._whisper_lock) if getattr(t, "name", None) == "whisper_local" else t
                for t in transcribers]

    # --- per-stage pools --------------------------------------------------------

    def _run_metadata_pool(self, notebook_id: int) -> None:
        sources = db.get_notebook_sources(self.cfg.db_path, notebook_id)
        todo = [s["id"] for s in sources
                if s["fetch_status"] != "done" or not s["upload_date"] or not s["title"]]
        if not todo:
            return
        self.log(f"Fetching metadata for {len(todo)} sources...")
        fn = self.metadata_fn if self.metadata_fn is not None else fetch_source_metadata
        statuses = self._pool(lambda sid: fn(self.cfg.db_path, sid, log=self.log), todo)
        self._log_counts(statuses)

    def _run_transcribe_pool(self, notebook_id: int, retranscribe: bool) -> None:
        sources = db.get_notebook_sources(self.cfg.db_path, notebook_id)
        todo = [s["id"] for s in sources
                if retranscribe or s["transcribe_status"] != "done" or not s["transcript_path"]
                or not os.path.exists(s["transcript_path"])]
        if not todo:
            return
        self.log(f"Transcribing {len(todo)} sources...")
        transcribers = self._resolve_transcribers()
        # transcribe_source itself resets analyze_status for whatever it (re)transcribes,
        # atomically with its own transcribe_status update (pipeline/workers/transcribe.py) --
        # nothing to do here after the pool returns.
        statuses = self._pool(
            lambda sid: transcribe_source(self.cfg.db_path, sid, self.cfg, transcribers=transcribers,
                                          cache=self.cache, log=self.log, force=retranscribe),
            todo)
        self._log_counts(statuses)

    def _run_analyze_pool(self, notebook_id: int, topic: str, llm, force_analyze: bool) -> None:
        sources = db.get_notebook_sources(self.cfg.db_path, notebook_id)
        todo = [s["id"] for s in sources
                if s["transcribe_status"] == "done" and (force_analyze or s["analyze_status"] != "done")]
        if not todo:
            return
        self.log(f"Analyzing {len(todo)} sources with {self.analyze_model}...")
        statuses = self._pool(
            lambda sid: analyze_source(self.cfg.db_path, sid, topic, llm, self.analyze_model, self.cache,
                                       force=force_analyze, log=self.log),
            todo, workers=self.llm_parallel)
        self._log_counts(statuses)

    # --- run ------------------------------------------------------------------

    def _safe_finish_run(self, run_id: int, status: str, error: str | None) -> None:
        try:
            db.finish_run(self.cfg.db_path, run_id, status, error=error)
        except sqlite3.Error as e:
            self.log(f"warning: could not finalize run {run_id}: {e}")

    def validate(self, notebook_id: int, stages: list[str], *, input_file: str | None = None,
                 url: str | None = None, playlist_url: str | None = None, file_path: str | None = None,
                 force_analyze: bool = False, retranscribe: bool = False) -> dict:
        """Pre-run validation, callable on its own so a caller -- the CLI in
        particular -- can validate a usage error *before* committing to any visible side effect
        (e.g. printing `NOTEBOOK=`), without duplicating these checks. No run row exists yet when
        this runs, so a failing call here costs nothing. `run` calls this first and reuses the
        notebook row it returns. Raises ValueError on any usage conflict; returns the notebook row
        on success."""
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            raise ValueError(f"unknown stage(s) {unknown}; choose from {STAGES}")
        notebook = db.get_notebook(self.cfg.db_path, notebook_id)
        if not notebook:
            raise ValueError(f"notebook {notebook_id} not found")
        if force_analyze and not all(s in stages for s in ("analyze", "merge", "output")):
            raise ValueError(
                "--force-analyze requires 'analyze', 'merge' and 'output' to all be in stages "
                "(run_output refuses a merge whose sources were re-analyzed after it ran)")
        if retranscribe and not all(s in stages for s in ("transcribe", "analyze", "merge", "output")):
            raise ValueError(
                "--retranscribe requires 'transcribe', 'analyze', 'merge' and 'output' to all be "
                "in stages (a retranscribed source's analysis is reset, so analyze/merge/output "
                "must run again in the same call)")
        if self.offline and any(s in ("analyze", "merge") for s in stages):
            raise ValueError("this runner has no LLM provider (it was built for a run without "
                             "analyze or merge); build it with a real provider")
        has_input = bool(input_file or url or playlist_url or file_path)
        if "fetch" in stages and not has_input and not db.get_notebook_sources(self.cfg.db_path, notebook_id):
            raise ValueError(
                "fetch requested with no input and the notebook has no sources; "
                "give --input, --url, --playlist or --file")
        if "fetch" in stages and input_file:
            check_input_file(input_file)     # a local path outside the list's folder refuses the list
        return notebook

    def run(self, notebook_id: int, stages: list[str], *, input_file: str | None = None, url: str | None = None,
            playlist_url: str | None = None, file_path: str | None = None, force_analyze: bool = False,
            retranscribe: bool = False) -> RunResult:
        clear_interrupt()   # a previous interrupted run in this process must not stop this one
        # --- pre-run validation: no run row exists yet, so a bad call here costs nothing ---
        notebook = self.validate(notebook_id, stages, input_file=input_file, url=url, playlist_url=playlist_url,
                                 file_path=file_path, force_analyze=force_analyze, retranscribe=retranscribe)
        has_input = bool(input_file or url or playlist_url or file_path)

        run_id = db.create_run(self.cfg.db_path, notebook_id, provider=self.provider.name,
                               budget_usd=self.budget_usd, stages=",".join(stages))

        report_path: str | None = None
        report_html_path: str | None = None
        status, message, error_for_db = "done", "", None
        fatal = False
        try:
            try:
                # make_client is inside this try (not above, with create_run) so a failure here
                # -- e.g. a bad provider/config -- still closes the run row as an ordinary error
                # rather than leaving it stuck "running" forever.
                llm = make_client(self.cfg, self.provider, db_path=self.cfg.db_path, run_id=run_id,
                                  notebook_id=notebook_id, budget_usd=self.budget_usd, log=self.log)
                if self.batching:
                    self.log("Batch mode: each step's calls go out as one Message Batch (half price); "
                             "a step can take minutes to hours")
                if "fetch" in stages:
                    # A resume call may legitimately request "fetch" (it's part of STAGES) without
                    # giving new input -- e.g. re-running the full pipeline after a budget pause.
                    # fetch_stage itself requires exactly one input, so only call it when one was
                    # actually given (pre-run validation above already ruled out the case where
                    # there's neither new input nor any existing source to fall back on).
                    if has_input:
                        fetch_stage(self.cfg.db_path, notebook_id, input_file=input_file, url=url,
                                   playlist_url=playlist_url, file_path=file_path, log=self.log)
                    else:
                        self.log("No new input; skipping fetch (metadata backfill only)")
                if "fetch" in stages or "transcribe" in stages:
                    # Run once even when both stages are requested. Also needed on a
                    # transcribe-only resume (the default `hmd resume` stage list): a source
                    # whose yt-dlp metadata call failed in the first run must be retried here,
                    # not just left stuck on fetch_status='error' forever.
                    self._run_metadata_pool(notebook_id)
                if "transcribe" in stages:
                    self._run_transcribe_pool(notebook_id, retranscribe)
                if "analyze" in stages:
                    self._run_analyze_pool(notebook_id, notebook["topic"], llm, force_analyze)
                if "merge" in stages:
                    rp = report_path_for(self.cfg.vault_dir, notebook["slug"])
                    run_merge(self.cfg.db_path, notebook_id, notebook["topic"], llm, self.merge_model,
                             run_id=run_id, report_path=rp, max_parallel=self.llm_parallel, log=self.log)
                if "output" in stages:
                    files = render_reports(self.cfg.db_path, notebook_id, self.cfg.vault_dir,
                                           show_anchors=self.show_anchors, formats=self.formats, html=self.html,
                                           log=self.log)
                    report_path, report_html_path = files.report_path, files.html_paths.get("summary")
            except BudgetExceeded as e:
                status, message, error_for_db = "paused", str(e), str(e)
            except LLMFatalError as e:
                # Every further call would fail the same way: the pool / merge already stopped
                # scheduling work, and the later stages never ran. One line, provider-prefixed.
                detail = " ".join(str(e).split())
                status, message, error_for_db = "error", f"{self.provider.name}: {detail}", f"LLMFatalError: {detail}"
                fatal = True
            except Exception as e:
                status, message, error_for_db = "error", str(e), f"{type(e).__name__}: {e}"
        except BaseException:
            self._safe_finish_run(run_id, "error", "interrupted")
            # Never let a failure in the usage report itself (or in self.log) replace the
            # propagating KeyboardInterrupt/SystemExit with something less informative.
            try:
                self.log(usage_report(self.cfg.db_path, run_id))
            except Exception:
                pass
            raise

        self._safe_finish_run(run_id, status, error_for_db)
        cost = db.run_cost(self.cfg.db_path, run_id)
        self.log(usage_report(self.cfg.db_path, run_id))
        return RunResult(run_id=run_id, notebook_id=notebook_id, status=status, message=message,
                         report_path=report_path, cost_usd=cost, fatal=fatal,
                         report_html_path=report_html_path)
