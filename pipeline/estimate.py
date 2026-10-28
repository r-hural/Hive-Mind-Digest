"""Cost and time estimate before a run (IDEA-003): token counts from the cached transcripts (or, when
a source is not transcribed yet, its video duration, its file size or a per-kind default), turned into
cost with the price table and the user's own usage history. Never calls a provider and never bills.

Merge is almost all of the cost (94-98% on the default models). It is modelled per call group, each
group's input and output tokens as a line `base + per_x * x` with `x = analyze_out / sqrt(N)` (the
predicted analyze output per source times sqrt(N), N the number of merged sources): the merge makes a
roughly fixed number of section and check calls (about 12 each) that see at most
MAX_SOURCES_PER_SECTION sources' claims each (pipeline/prompts/merge.py, pipeline/stages/merge.py), so
it grows sub-linearly with N; a flat ratio to the analyze output under-states small notebooks and
over-states large ones."""
from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass, replace

from pipeline import db
from pipeline.pricing import estimate_cost
from pipeline.sources import SourceRef, kind_of

CHARS_PER_SECOND = 17.2          # median transcript density over 83 cached transcripts
WEB_PAGE_CHARS = 8_000           # a typical article's extracted text; the page is not fetched to size it
# A rough text ratio: a PDF's bytes are mostly fonts, images and structure; its text layer is
# typically a tenth of the file or less (the estimate rather over- than under-states the text).
PDF_BYTES_PER_CHAR = 10

# The merge's call groups. "section" is the section and check calls (stages merge:section and
# merge:check; Message Batches with --batch), "direct" the outline and verify calls (always direct,
# full price). Each group's tokens are priced as this many equal calls, so a per-request
# long-prompt tier is checked on the group's average call, not on the group's total (outline and
# verify are not modelled apart).
SECTION_CALLS = 24
DIRECT_CALLS = 2
DIRECT_STAGES = ("merge:outline", "merge:verify")
SERIES = ("section_in", "section_out", "direct_in", "direct_out")
REFIT_MIN_RUNS = 4               # fewer matching runs blend with the built-in line instead of a refit
REFIT_MIN_X_RATIO = 1.3          # ... as do runs whose x (max / min) spreads less: re-runs of one notebook
HISTORY_RUNS = 20


@dataclass(frozen=True)
class MergeFit:
    """(base, per_x) token lines for each merge call group: tokens = base + per_x * analyze_out / sqrt(N)."""
    section_in: tuple[float, float]
    section_out: tuple[float, float]
    direct_in: tuple[float, float]
    direct_out: tuple[float, float]


@dataclass(frozen=True)
class Calibration:
    analyze_in_base: float
    analyze_in_per_char: float
    analyze_out_base: float
    analyze_out_per_char: float
    merge: MergeFit
    # (low, high) multipliers of the estimate (analyze + merge) for the printed range: the spread
    # of actual / estimate over a leave-one-out backtest.
    spread: tuple[float, float]
    minutes_batch: float
    minutes_direct_per_source: float
    from_history: bool = False
    history_runs: int = 0

    def with_merge(self, fit: MergeFit, spread: tuple[float, float] | None = None, runs: int = 0) -> "Calibration":
        return replace(self, merge=fit, spread=spread or self.spread,
                       from_history=runs > 0, history_runs=runs)


# Merge lines fitted 2026-10-18 by least squares on the ten runs on the default models
# (claude-haiku-5-5 analyze, claude-sonnet-5-5 merge, anthropic_api, 2026-10-15..16: the three
# 13-14 video golden notebooks twice or three times each, a 42-video notebook and a 6-source mixed
# notebook; the fixture is tests/test_estimate_backtest.py). Spread: actual / estimate of the total
# cost over the leave-one-out backtest on those runs (estimate within -26%..+20%), rounded outward.
# Analyze per-char fit 2026-10-16 on runs 32-37 (99 calls, stable within 4%); batch runs took
# 10-18 minutes.
BUILTIN_MERGE = MergeFit(section_in=(43645, 15.577), section_out=(7878, 2.485),
                         direct_in=(-6222, 2.927), direct_out=(3658, 0.004))
DEFAULTS = Calibration(1153, 0.4261, 2272, 0.1392, BUILTIN_MERGE, (0.83, 1.35), 12.0, 1.0)


def transcript_chars(path: str | None) -> int | None:
    """Characters in a cached transcript, or None when it is missing, unreadable or malformed (the
    caller then estimates the size from the video length)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return sum(len(s.get("text", "")) + 1 for s in data.get("segments", []))
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _x(analyze_out: float, sources: int) -> float:
    return analyze_out / math.sqrt(max(sources, 1))


def merge_tokens(fit: MergeFit, analyze_out: float, sources: int) -> dict[str, float]:
    """Predicted merge tokens per call group (never negative) for a notebook of `sources` merged
    sources whose analyze output totals `analyze_out` tokens."""
    x = _x(analyze_out, sources)
    return {k: max(0.0, getattr(fit, k)[0] + getattr(fit, k)[1] * x) for k in SERIES}


def merge_cost(tokens: dict[str, float], model: str, *, batch: bool) -> float:
    """USD for predicted merge tokens: section/check calls at batch prices when `batch`, outline and
    verify always at direct prices."""
    def group(prefix: str, calls: int, batched: bool) -> float:
        one = estimate_cost(model, int(tokens[prefix + "_in"] / calls), int(tokens[prefix + "_out"] / calls),
                            batch=batched)
        # None only for an unpriced model, which `hmd dry-run` refuses before estimating (exit 4);
        # a group clamped to 0 tokens costs 0 here, which is right.
        return calls * (one or 0.0)
    return group("section", SECTION_CALLS, batch) + group("direct", DIRECT_CALLS, False)


def _line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares (base, slope); the caller makes sure the x values spread."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - slope * mx, slope


def fit_merge(runs: list[dict]) -> MergeFit | None:
    """Least-squares merge lines over runs as db.recent_runs_with_merge reports them, or None when
    the runs do not spread in size: fewer than two distinct merged source counts, or x (max / min)
    under REFIT_MIN_X_RATIO. Re-runs of one notebook vary a few percent in x, and a slope fitted to
    that is noise."""
    xs = [_x(r["analyze_out"], r["merged_sources"]) for r in runs]
    if (len({r["merged_sources"] for r in runs}) < 2 or min(xs) <= 0
            or max(xs) / min(xs) < REFIT_MIN_X_RATIO):
        return None
    return MergeFit(**{k: _line(xs, [float(r[k]) for r in runs]) for k in SERIES})


def blend_merge(runs: list[dict], base: MergeFit = BUILTIN_MERGE) -> MergeFit:
    """The built-in lines scaled, per group, toward the median actual / built-in ratio of the runs,
    weighted by len(runs) / REFIT_MIN_RUNS (one run moves the built-in a quarter of the way)."""
    weight = min(len(runs), REFIT_MIN_RUNS) / REFIT_MIN_RUNS
    scaled = {}
    for k in SERIES:
        ratios = [r[k] / t for r in runs
                  if (t := merge_tokens(base, r["analyze_out"], r["merged_sources"])[k]) > 0]
        f = 1 + weight * (statistics.median(ratios) - 1) if ratios else 1.0
        b, p = getattr(base, k)
        scaled[k] = (b * f, p * f)
    return MergeFit(**scaled)


def _merge_fit_for(runs: list[dict]) -> MergeFit:
    fit = fit_merge(runs) if len(runs) >= REFIT_MIN_RUNS else None
    return fit or blend_merge(runs)


def calibrate(db_path: str, *, analyze_model: str, merge_model: str, batch: bool = True,
              limit: int = HISTORY_RUNS) -> Calibration:
    """The merge lines from the newest `limit` comparable runs: finished, on the anthropic_api
    provider (claude_cli records its input as cache writes), with the same analyze and merge models
    as the estimate, and not a `notebook add` + `resume` run (one that analyzed fewer sources than it
    merged: its analyze output understates the notebook). With REFIT_MIN_RUNS or more such runs (and
    a spread in size, fit_merge) the lines are refitted by least squares and the range is their
    leave-one-out spread of the merge cost, never narrower than the built-in one (a history run's
    leave-one-out estimate uses its actual analyze output, so it misses the analyze-prediction
    error); otherwise the built-in lines are blended toward them (blend_merge) and keep the
    built-in range; with none, DEFAULTS."""
    runs = [r for r in db.recent_runs_with_merge(db_path, limit * 5)
            if r["analyze_out"] and r["analyze_calls"] >= r["merged_sources"]
            and r.get("provider") == "anthropic_api"
            and r.get("analyze_model") == analyze_model and r.get("merge_model") == merge_model][:limit]
    if not runs:
        return DEFAULTS
    fit = fit_merge(runs) if len(runs) >= REFIT_MIN_RUNS else None
    if fit is None:
        return DEFAULTS.with_merge(blend_merge(runs), runs=len(runs))
    ratios = []
    for i, r in enumerate(runs):
        rest = _merge_fit_for(runs[:i] + runs[i + 1:])
        est = merge_cost(merge_tokens(rest, r["analyze_out"], r["merged_sources"]), merge_model, batch=batch)
        act = merge_cost({k: float(r[k]) for k in SERIES}, merge_model, batch=batch)
        if est > 0:
            ratios.append(act / est)
    spread = (math.floor(100 * min(min(ratios, default=1.0), DEFAULTS.spread[0])) / 100,
              math.ceil(100 * max(max(ratios, default=1.0), DEFAULTS.spread[1])) / 100)
    return DEFAULTS.with_merge(fit, spread, runs=len(runs))


def untranscribed_chars(src: dict) -> tuple[int | None, bool]:
    """(estimated characters or None when the size cannot be estimated, whether the video duration
    was used) for a source without a readable transcript: the duration for videos and recordings,
    the file size for local text files and PDFs, a per-kind default for web pages."""
    kind = kind_of(src)
    if kind == "web":
        return WEB_PAGE_CHARS, False
    if kind in ("text", "pdf"):
        path = SourceRef.from_row(src).path
        try:
            size = os.path.getsize(path) if path else None
        except OSError:
            size = None
        if size is None:
            return None, False
        return (size if kind == "text" else size // PDF_BYTES_PER_CHAR), False
    if src.get("duration_seconds"):
        return int(src["duration_seconds"] * CHARS_PER_SECOND), True
    return None, False


@dataclass(frozen=True)
class Costs:
    analyze: float
    merge: float
    low: float
    high: float


def estimate_sources(chars: list[int], todo: list[bool], cal: Calibration, *, analyze_model: str,
                     merge_model: str, batch: bool) -> Costs:
    """Cost of analyzing the sources flagged in `todo` and merging all of them, from each source's
    character count (0 = unknown size, left out)."""
    analyze = a_out_all = 0.0
    sized = 0
    for ch, pending in zip(chars, todo):
        if not ch:
            continue
        sized += 1
        out_tokens = cal.analyze_out_base + cal.analyze_out_per_char * ch
        a_out_all += out_tokens
        if pending:
            # one price lookup per source: the long-prompt tier is chosen per request
            analyze += estimate_cost(analyze_model, int(cal.analyze_in_base + cal.analyze_in_per_char * ch),
                                     int(out_tokens), batch=batch) or 0.0
    merge = merge_cost(merge_tokens(cal.merge, a_out_all, sized), merge_model, batch=batch) if sized else 0.0
    # N counts only the sources of known size: an unsized source is left out of the merge too,
    # which the CLI says (Estimate.unknown_sources).
    lo, hi = cal.spread
    return Costs(analyze, merge, (analyze + merge) * lo, (analyze + merge) * hi)


@dataclass(frozen=True)
class Estimate:
    sources: int
    sources_to_analyze: int
    chars: int
    estimated_chars: bool              # some source is not transcribed: its size is estimated or unknown
    analyze_cost: float
    merge_cost: float
    low: float
    high: float
    minutes: float
    batch: bool
    calibration: Calibration
    from_duration: bool = False        # some size was estimated from a video duration
    unknown_sources: int = 0           # sources whose size could not be estimated (left out)


def estimate_notebook(cfg, notebook_id: int, *, analyze_model: str, merge_model: str, batch: bool,
                      force: bool = False) -> Estimate:
    """The next run's estimate; `force` counts every source as to analyze (as `hmd analyze --force`)."""
    sources = db.get_notebook_sources(cfg.db_path, notebook_id)
    cal = calibrate(cfg.db_path, analyze_model=analyze_model, merge_model=merge_model, batch=batch)
    chars, todo = [], []
    estimated, from_duration, unknown = False, False, 0
    for s in sources:
        ch = transcript_chars(s.get("transcript_path"))
        if ch is None:
            estimated = True
            ch, used_duration = untranscribed_chars(s)
            from_duration |= used_duration
            if ch is None:
                unknown += 1
                ch = 0
        chars.append(ch)
        # counted even when its size is unknown (no transcript, no duration)
        todo.append(force or s.get("analyze_status") != "done")
    c = estimate_sources(chars, todo, cal, analyze_model=analyze_model, merge_model=merge_model, batch=batch)
    n_todo = sum(todo)
    minutes = cal.minutes_batch if batch else cal.minutes_direct_per_source * n_todo + 5
    return Estimate(len(sources), n_todo, sum(chars), estimated, c.analyze, c.merge,
                    c.low, c.high, minutes, batch, cal, from_duration, unknown)
