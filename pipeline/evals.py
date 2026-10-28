"""Golden notebooks (IDEA-080) with cassettes at the provider boundary (IDEA-283).

Layout (real transcripts and recordings, kept out of the public release; DECISION-019):
    evals/golden/<name>/manifest.json                      topic, sources (metadata), assertions
    evals/golden/<name>/transcripts/<video_id>/<method>.json   transcript fixtures (copied from the cache)
    evals/golden/<name>/cassettes/<provider>/<key>.json     recorded LLM responses per provider

`hmd eval <name> --live --provider X` records; `hmd eval` and CI replay with no network.

`run_golden` seeds a fresh notebook straight from the manifest -- no fetch, no transcription: it
writes each source row directly (db.add_source) and copies the golden's transcript fixture into
the run's own transcript cache, then runs only analyze -> merge -> output through the real
`Runner`. That keeps a golden's LLM behaviour exercised end to end while its inputs (metadata,
transcripts) stay fixed forever.

`run_golden` always builds its own fresh `Config` (models, max_output_tokens, retries: the
built-in defaults) rather than using whatever `config.yaml` the caller loaded -- the only value
config.yaml can influence is `run.budget_usd`, and only via `cmd_eval`'s own default for `--live`.
A cassette's key is `(model, system, prompt)`, so the model used to record one MUST be the same
model used to replay it; letting a golden float with the user's configured model would silently
invalidate every cassette the moment someone edited config.yaml. In practice this means a `--live`
recording run always bills the default `analyze`/`merge` models (pipeline/config.py's
`LLMConfig.models`), never a model chosen via config.yaml.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pipeline import db
from pipeline.anchors import QuoteLocator, normalize
from pipeline.config import PROVIDERS, Config, ConfigError
from pipeline.exit_codes import EXIT_ERROR, EXIT_OK, EXIT_PAUSED
from pipeline.llm.factory import make_provider
from pipeline.runner import Runner, usage_report
from pipeline.schemas import Analysis, MergedDocument
from pipeline.slug import slugify
from pipeline.stages.output import SourceLabeler, _md_text
from pipeline.sources import KINDS, is_source_key, kind_of
from pipeline.transcription.base import TranscriptionError, watch_url
from pipeline.transcription.cache import TranscriptCache

# Package-relative, not CWD-relative: pipeline/evals.py lives at <repo_root>/pipeline/evals.py, so
# its own parent's parent is the repo root regardless of where `hmd` (or pytest) is invoked from.
# A CWD-relative "evals/golden" would silently report "no golden notebooks" from any directory
# other than the repo root, and would make `--export-from` write wherever the shell happened to be.
GOLDEN_ROOT = str(Path(__file__).resolve().parents[1] / "evals" / "golden")
DEFAULT_ASSERTIONS = {"min_sections": 2, "min_points": 10, "must_contain": [], "min_contradictions": 0,
                      "max_dropped_ratio": 0.3, "must_not_contain": [], "forbidden_contradictions": []}
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+\Z")


@dataclass
class GoldenResult:
    name: str
    status: str
    message: str
    report_path: str | None
    doc: MergedDocument | None
    sources: list[dict]
    analyses: dict[int, Analysis]
    transcripts: dict[str, object]
    report_text: str
    cost: float
    provider: object
    run_id: int
    db_path: str


def list_goldens(root: str | None = None) -> list[str]:
    root = root or GOLDEN_ROOT
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "manifest.json")))


def has_cassettes(golden_dir: str, provider: str) -> bool:
    """True if `golden_dir`/cassettes/`provider` exists and holds at least one recorded cassette
    (a `*.json` file). A stray `.tmp` left behind by a killed `--live` run does not count -- it is
    a partial write, not a usable recording."""
    cassette_dir = os.path.join(golden_dir, "cassettes", provider)
    if not os.path.isdir(cassette_dir):
        return False
    return any(f.endswith(".json") for f in os.listdir(cassette_dir))


def discover_cases(root: str | None = None) -> list[tuple[str, str]]:
    """(golden_name, provider) pairs for every golden under `root` that has cassettes for that
    provider -- i.e. every case a no-network replay run can actually exercise. Sorted for stable
    test IDs and reproducible CI output."""
    root = root or GOLDEN_ROOT
    cases = [(name, provider) for name in list_goldens(root) for provider in PROVIDERS
             if has_cassettes(os.path.join(root, name), provider)]
    return sorted(cases)


def load_manifest(golden_dir: str) -> dict:
    path = os.path.join(golden_dir, "manifest.json")
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ConfigError(f"golden manifest {path}: {e}") from e
    if not isinstance(m, dict) or not m.get("topic"):
        raise ConfigError(f"golden manifest {path}: missing required key 'topic'")
    sources = m.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ConfigError(f"golden manifest {path}: 'sources' must be a non-empty list")
    for i, s in enumerate(sources):
        if not isinstance(s, dict) or not s.get("video_id"):
            raise ConfigError(f"golden manifest {path}: sources[{i}] is missing 'video_id'")
        if not isinstance(s["video_id"], str):
            raise ConfigError(f"golden manifest {path}: sources[{i}] 'video_id' must be a string, "
                              f"not {s['video_id']!r}")
        if not is_source_key(s["video_id"]):
            raise ConfigError(f"golden manifest {path}: sources[{i}] has an invalid source key {s['video_id']!r}")
        kind = s.get("kind", "youtube")
        if kind not in KINDS:
            raise ConfigError(f"golden manifest {path}: sources[{i}] has unknown kind {kind!r}")
        if kind != "youtube" and not (isinstance(s.get("url"), str) and s["url"]):
            raise ConfigError(f"golden manifest {path}: sources[{i}] (kind {kind!r}) is missing 'url'")
    m["assertions"] = {**copy.deepcopy(DEFAULT_ASSERTIONS), **m.get("assertions", {})}
    _check_negative_assertions(path, m["assertions"], {s["video_id"] for s in sources})
    return m


def _check_negative_assertions(path: str, a: dict, video_ids: set[str]) -> None:
    """Validate the two negative assertions up front: a malformed entry, or a pair naming a video
    that is not a source of this golden, would make the check pass vacuously forever."""
    mnc = a["must_not_contain"]
    if not isinstance(mnc, list) or not all(isinstance(x, str) and x.strip() for x in mnc):
        raise ConfigError(f"golden manifest {path}: must_not_contain must be a list of non-empty strings")
    pairs = a["forbidden_contradictions"]
    if not isinstance(pairs, list):
        raise ConfigError(f"golden manifest {path}: forbidden_contradictions must be a list of "
                          f"[video_id, video_id] pairs")
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(v, str) for v in pair)
                or pair[0] == pair[1]):
            raise ConfigError(f"golden manifest {path}: forbidden_contradictions entry {pair!r} must be "
                              f"two different video ids")
        unknown = [v for v in pair if v not in video_ids]
        if unknown:
            raise ConfigError(f"golden manifest {path}: forbidden_contradictions entry {pair!r} names "
                              f"{unknown}, not a source of this golden")


def register_manifest_sources(cfg: Config, notebook_id: int, golden_dir: str, manifest: dict,
                              cache: TranscriptCache) -> list[int]:
    """Register every manifest source as a finished-transcript row and copy its fixture into `cache`."""
    name = os.path.basename(os.path.normpath(golden_dir))
    fixtures = TranscriptCache(os.path.join(golden_dir, "transcripts"))
    ids = []
    for s in manifest["sources"]:
        video_id = s["video_id"]
        kind = s.get("kind", "youtube")
        url = s.get("url") or watch_url(video_id)
        sid = db.add_source(cfg.db_path, notebook_id, video_id, url, kind=kind, title=s.get("title"),
                            channel=s.get("channel"), duration_seconds=s.get("duration_seconds"),
                            upload_date=s.get("upload_date"))
        db.update_source_metadata(cfg.db_path, sid, status="done")
        method = s.get("transcript_method") or (fixtures.available(video_id) or ["youtube_captions"])[0]
        try:
            fixture_path = fixtures.path(video_id, method)
        except ValueError as e:
            raise ConfigError(f"golden {name!r}: source {video_id!r}: {e}") from e
        try:
            transcript = fixtures.load(video_id, method)
        except (TranscriptionError, ValueError, OSError) as e:
            raise ConfigError(
                f"golden {name!r}: transcript fixture for source {video_id!r} method {method!r} "
                f"is missing or corrupt (expected {fixture_path}): {e}") from e
        cache.save(transcript)
        db.update_transcribe_status(cfg.db_path, sid, "done", transcript_path=cache.path(video_id, method),
                                    method=method, language=transcript.language or "en")
        ids.append(sid)
    return ids


def run_golden(golden_dir: str, *, provider_name: str, record: bool, home: str,
               budget_usd: float = 0.0, log: Callable[[str], None] = lambda m: None) -> GoldenResult:
    name = os.path.basename(os.path.normpath(golden_dir))
    manifest = load_manifest(golden_dir)
    # See the module docstring: a fresh, built-in-defaults Config -- never the caller's
    # config.yaml -- so a cassette's (model, system, prompt) key never drifts under the user.
    cfg = Config(home=home, llm={"provider": provider_name}, run={"budget_usd": budget_usd})
    db.init_db(cfg.db_path)
    cache = TranscriptCache(cfg.transcript_cache_dir)

    slug = slugify(manifest["topic"])
    nid = db.create_notebook(cfg.db_path, manifest["topic"], slug)
    register_manifest_sources(cfg, nid, golden_dir, manifest, cache)

    cassette_dir = os.path.join(golden_dir, "cassettes", provider_name)
    provider = make_provider(cfg, provider_name, cassette_dir=cassette_dir, record=record)
    runner = Runner(cfg, provider=provider, log=log)
    result = runner.run(nid, ["analyze", "merge", "output"])

    sources = db.get_notebook_sources(cfg.db_path, nid)
    analyses = {s["id"]: Analysis.model_validate_json(s["analysis_json"]) for s in sources if s["analysis_json"]}
    transcripts = {s["video_id"]: cache.load(s["video_id"], s["transcript_method"])
                  for s in sources if s["transcript_method"]}
    merge_row = db.get_latest_merge(cfg.db_path, nid)
    doc = MergedDocument.model_validate_json(merge_row["merged_json"]) if merge_row else None
    report_text = ""
    if result.report_path and os.path.exists(result.report_path):
        with open(result.report_path, encoding="utf-8") as f:
            report_text = f.read()
    return GoldenResult(name=name, status=result.status, message=result.message, report_path=result.report_path,
                        doc=doc, sources=sources, analyses=analyses, transcripts=transcripts,
                        report_text=report_text, cost=db.run_cost(cfg.db_path, result.run_id), provider=provider,
                        run_id=result.run_id, db_path=cfg.db_path)


# A link that locates the claim inside its source (pipeline/links.py): a YouTube or Vimeo
# timestamp, a PDF page or a web text fragment. A bare source URL is not a locator.
LOCATOR_RE = re.compile(r"(?:[&#]t=\d+s|#page=\d+|#:~:text=[^)\s]+)\)")
# "(confirmed by N sources)", or "(N sources)" for a partly supported or disputed joint point (IDEA-307b).
CONFIRMED_RE = re.compile(r"\((?:confirmed by )?(\d+) sources\)")


def _report_bullets(report_text: str) -> list[str]:
    """Every '- ' bullet line from the first numbered section heading up to (not including) the
    '## Sources' heading -- i.e. every rendered point bullet, excluding the TL;DR (which comes
    before "## 1.") and the sources table / contradictions table (which come at or after
    "## Sources"). Line-based on purpose: a substring split on "## 1." also matches "### 1.2"
    (subsection headings), and a substring split on "## Sources" would also match a section or
    point whose text happens to contain that literal string; comparing whole lines avoids both."""
    lines = report_text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("## 1.")), None)
    if start is None:
        return []
    end = next((i for i in range(start, len(lines)) if lines[i] == "## Sources"), len(lines))
    return [l for l in lines[start:end] if l.startswith("- ")]


def quote_at_anchor(locator: QuoteLocator, flat, claim) -> bool:
    """Assertion 5 for one claim: it has an anchor, its quote still locates in the transcript, and
    the anchored span starts with the quote's first 20 normalised chars. Segment-mode claims
    (IDEA-294: quote copied by code from the transcript span it is anchored at,
    pipeline/anchors.py::resolve_segment_claims) pass this by construction; text-mode claims
    (older stored analyses) are checked for real."""
    if claim.anchor is None:
        return False
    span = flat.text[claim.anchor.char_start:claim.anchor.char_end]
    return locator.locate(claim.quote) is not None and normalize(span)[0][:20] == normalize(claim.quote)[0][:20]


def check_golden(res: GoldenResult, manifest: dict) -> list[str]:
    """Ten regression assertions plus two negative ones (11, 12; empty lists by default). Returns a
    list of failure messages (empty = pass). Manifests passed in directly may omit 11 and 12."""
    a = manifest["assertions"]
    fails: list[str] = []
    # 1. the run completed with a document stamped by every current renderer/prompt version (also
    # catches truncation: the client raises and the run errors, which surfaces here as no doc).
    if res.status != "done" or res.doc is None:
        return [f"run status {res.status}: {res.message}"]
    doc = res.doc
    if set(doc.prompt_versions) != {"analyze", "merge", "check", "retract", "render"}:
        fails.append(f"prompt_versions keys {sorted(doc.prompt_versions)} != analyze/merge/check/retract/render")
    points = [(sec, p) for sec in doc.sections for ss in sec.subsections for p in ss.points]
    # 2. structure
    if len(doc.sections) < a["min_sections"]:
        fails.append(f"sections {len(doc.sections)} < {a['min_sections']}")
    if len(points) < a["min_points"]:
        fails.append(f"points {len(points)} < {a['min_points']}")
    # 3. TL;DR size: 3-5 bullets; a merge with no points (contradictions only) renders no TL;DR and
    # skips verification by design, so its TL;DR must be empty.
    if not points:
        if doc.tldr:
            fails.append(f"tldr has {len(doc.tldr)} bullets, want none for a merge with no points")
    elif not 3 <= len(doc.tldr) <= 5:
        fails.append(f"tldr has {len(doc.tldr)} bullets, want 3-5")
    # 4. every rendered bullet (after the TL;DR) links to a locator -- also guards assertion 7
    # below, which reads the same `bullets` list: an empty report must not let either pass
    # vacuously.
    if not res.report_text.strip():
        fails.append("report_text is empty; cannot check bullet timestamps or confirmed_by counts")
        bullets: list[str] = []
    else:
        bullets = _report_bullets(res.report_text)
        missing = [l for l in bullets if not LOCATOR_RE.search(l)]
        if missing:
            fails.append(f"{len(missing)} bullets without a source locator link (timestamp, page or "
                         f"text fragment), e.g. {missing[0][:80]!r}")
    # 5. every anchored quote is found in its transcript at the stored offset
    bad_quotes = 0
    for sid, an in res.analyses.items():
        src = next(s for s in res.sources if s["id"] == sid)
        transcript = res.transcripts.get(src["video_id"])
        if transcript is None:
            bad_quotes += len(an.key_claims)
            continue
        flat = transcript.flatten()
        locator = QuoteLocator(flat)
        bad_quotes += sum(1 for c in an.key_claims if not quote_at_anchor(locator, flat, c))
    if bad_quotes:
        fails.append(f"{bad_quotes} anchored quotes not found at their offset")
    # 6. every citation resolves to a real source
    ids = {s["id"] for s in res.sources}
    unresolved = [p.id for _, p in points for t in p.sources if doc.source_map.get(t) not in ids]
    if unresolved:
        fails.append(f"unresolved citations in points {unresolved[:5]}")
    # 7. confirmed_by is computed from the distinct sources. Matched against the same collapsed
    # form the renderer itself writes (_collapse_newlines joins whitespace runs into one space)
    # so a point whose text contains a run of whitespace is not silently skipped here. A point
    # whose bullet cannot be located at all (as opposed to found but missing a timestamp, which
    # assertion 4 already catches) is itself a failure, not silently ignored.
    # The renderer replaces leaked source tokens in point text (SourceLabeler), so the stored
    # text is labelled the same way before it is looked up.
    # Points are rendered in document order, so the search resumes after the last matched bullet:
    # two points sharing their first 40 characters (a story repeated across sections) each meet
    # their own bullet, never the first one with that prefix.
    label = SourceLabeler(doc.source_map, {s["id"]: s for s in res.sources})
    not_found = 0
    pos = 0
    for _, p in points:
        n = len(set(p.sources))
        collapsed = _md_text(label(p.text))[:40]   # as the renderer escapes it
        idx = next((i for i in range(pos, len(bullets)) if bullets[i].startswith(f"- {collapsed}")), None)
        if idx is None:
            not_found += 1
            continue
        pos = idx + 1
        line = bullets[idx]
        m = CONFIRMED_RE.search(line)
        if (n > 1) != bool(m) or (m and int(m.group(1)) != n):
            fails.append(f"confirmed_by mismatch for {p.id}: sources {n}, rendered {m.group(1) if m else 'none'}")
            break
    if not_found:
        fails.append(f"{not_found} points not found in the rendered report")
    # 8. detail retention (IDEA-210): a must_contain string is checked against the rendered
    # report only -- that's the whole point of the assertion (a merge that quietly drops a
    # number or qualifier from the sources must be caught here). The owner picks these strings
    # by reading the finished report of a real recorded run and copying them from it, so they
    # are already present at record time; the assertion exists to catch a later prompt or merge
    # change that stops reproducing them.
    low = res.report_text.lower()
    for needle in a["must_contain"]:
        if needle.lower() not in low:
            fails.append(f"report lacks expected detail {needle!r}")
    # 9. contradictions
    n_contra = sum(len(sec.contradictions) for sec in doc.sections)
    if n_contra < a["min_contradictions"]:
        fails.append(f"contradictions {n_contra} < {a['min_contradictions']}")
    # 10. anchor quality: dropped quotes per source
    for sid, an in res.analyses.items():
        total = len(an.key_claims) + len(an.dropped_quotes)
        if total and len(an.dropped_quotes) / total > a["max_dropped_ratio"]:
            fails.append(f"source {sid}: {len(an.dropped_quotes)}/{total} quotes dropped")
            break
    # 11. text the report must NOT contain (a known wrong claim, e.g. a deadline a source never
    # stated as universal); same case-insensitive match against the rendered report as assertion 8.
    for needle in a.get("must_not_contain", []):
        if needle.lower() in low:
            fails.append(f"report contains forbidden text {needle!r}")
    # 12. false contradictions: no contradiction may set these two sources against each other, in
    # either order (a source that only looks like it disagrees, e.g. a caveated "21 days" title).
    forbidden = {frozenset(p) for p in a.get("forbidden_contradictions", [])}
    if forbidden:
        video_of = {s["id"]: s["video_id"] for s in res.sources}
        for sec in doc.sections:
            for c in sec.contradictions:
                pair = frozenset({video_of.get(doc.source_map.get(c.source_a)),
                                  video_of.get(doc.source_map.get(c.source_b))})
                if pair in forbidden:
                    a_id, b_id = sorted(pair)
                    fails.append(f"forbidden contradiction between {a_id} and {b_id} in section {sec.title!r}")
    return fails


def export_golden(cfg: Config, notebook_ref: str, root: str, name: str, *,
                  log: Callable[[str], None] = lambda m: None) -> str:
    """Turn a notebook that already ran into a golden dir: manifest + transcript fixtures. Assertions
    start at the defaults; the owner adds must_contain strings after reading the report. A source
    with no finished transcript is silently excluded from the golden -- `log` is told which ones,
    so a caller (`cmd_eval`) can tell the owner rather than leaving them to notice a smaller
    `sources` list than expected."""
    if not NAME_RE.match(name):
        raise ConfigError(f"golden name {name!r} must match {NAME_RE.pattern} (no path separators)")
    nb = db.get_notebook(cfg.db_path, notebook_ref)
    if not nb:
        raise ValueError(f"notebook {notebook_ref!r} not found")
    gdir = os.path.join(root, name)
    os.makedirs(os.path.join(gdir, "transcripts"), exist_ok=True)
    cache = TranscriptCache(cfg.transcript_cache_dir)
    sources = []
    skipped = []
    for s in db.get_notebook_sources(cfg.db_path, nb["id"]):
        if s["transcribe_status"] != "done" or not s["transcript_method"]:
            skipped.append(s["video_id"])
            continue
        src_path = cache.path(s["video_id"], s["transcript_method"])
        dst = os.path.join(gdir, "transcripts", s["video_id"], f"{s['transcript_method']}.json")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src_path, dst)
        entry = {"video_id": s["video_id"], "title": s["title"], "channel": s["channel"],
                 "duration_seconds": s["duration_seconds"], "upload_date": s["upload_date"],
                 "transcript_method": s["transcript_method"]}
        if kind_of(s) != "youtube":
            entry["kind"] = kind_of(s)
            entry["url"] = s["url"]
        sources.append(entry)
    if skipped:
        log(f"skipped {len(skipped)} source(s) without a finished transcript: {skipped}")
    manifest = {"topic": nb["topic"], "sources": sources, "assertions": copy.deepcopy(DEFAULT_ASSERTIONS)}
    with open(os.path.join(gdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    return gdir


def _cassette_summary(provider) -> str:
    hits = getattr(provider, "hits", None)
    misses = getattr(provider, "misses", None)
    recorded = getattr(provider, "recorded", None)
    if hits is None and misses is None and recorded is None:
        return ""
    return f"  cassette: {hits or 0} hit(s), {misses or 0} miss(es), {recorded or 0} recorded"


def _run_one(gdir: str, name: str, provider_name: str, manifest: dict, *, live: bool,
            budget_usd: float | None, cfg: Config, keep_home: str | None = None) -> int:
    """Runs (replay or live) one golden/provider case, prints its KEY=value block, and returns an
    exit-code contribution: EXIT_OK (passed, or nothing to replay), EXIT_ERROR (assertions failed
    or the run errored) or EXIT_PAUSED (a live run paused on its budget -- a replay run can never
    pause, since replay always runs at budget_usd=0)."""
    if not live and not has_cassettes(gdir, provider_name):
        print(f"skip: {name}/{provider_name} has no cassettes; "
             f"record with: hmd eval {name} --live --provider {provider_name}")
        return EXIT_OK

    if live:
        home = tempfile.mkdtemp(prefix=f"hmd-eval-{name}-{provider_name}-")
        budget = budget_usd if budget_usd is not None else cfg.run.budget_usd
        try:
            res = run_golden(gdir, provider_name=provider_name, record=True, home=home, budget_usd=budget,
                             log=print)
        except Exception:
            # A ConfigError here (bad manifest, missing/corrupt fixture) means the run never got
            # to do anything with `home` -- don't leave the mkdtemp behind for the caller to clean
            # up by hand. A *completed* run's home is kept on purpose (see below).
            shutil.rmtree(home, ignore_errors=True)
            raise
        print(usage_report(res.db_path, res.run_id))
    elif keep_home:
        home = os.path.abspath(keep_home)
        res = run_golden(gdir, provider_name=provider_name, record=False, home=home, budget_usd=0.0)
    else:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as home:
            res = run_golden(gdir, provider_name=provider_name, record=False, home=home, budget_usd=0.0)

    print(f"GOLDEN={res.name}")
    print(f"PROVIDER={provider_name}")
    print(f"COST={res.cost:.4f}")

    if res.status == "paused":
        print("STATUS=paused")
        print(f"  - {res.message}")
        exit_code = EXIT_PAUSED
    else:
        failures = check_golden(res, manifest)
        print(f"STATUS={'pass' if not failures else 'fail'}")
        for msg in failures:
            print(f"  - {msg}")
        exit_code = EXIT_OK if not failures else EXIT_ERROR
        # A prompt-version bump invalidates every cassette a stage needs; the symptom the operator
        # actually sees is an opaque pipeline failure (every source erroring out of analyze, then
        # merge refusing with "no analyzed sources") with no mention of cassettes anywhere in that
        # message. Name the likely cause and the fix explicitly whenever replay actually missed.
        if not live and getattr(res.provider, "misses", 0) > 0:
            print(f"  {res.provider.misses} cassette miss(es); prompts changed? re-record with: "
                 f"hmd eval {name} --live --provider {provider_name}")

    summary = _cassette_summary(res.provider)
    if summary:
        print(summary)
    if live or keep_home:
        if res.report_path:
            print(f"REPORT={res.report_path}")
        print(f"HOME={home}")
    return exit_code


def cmd_eval(cfg: Config, args) -> int:
    if args.budget is not None and args.budget < 0:
        raise ConfigError("--budget must be >= 0")
    if args.budget is not None and not args.live:
        raise ConfigError("--budget only applies to --live; replay never bills")
    # `name` becomes a path component (os.path.join(GOLDEN_ROOT, name)) below and in export_golden
    # -- validate it up front on every path, not only --export-from's, so e.g. `hmd eval ../../x`
    # can never make that join walk outside GOLDEN_ROOT.
    if args.name and not NAME_RE.match(args.name):
        raise ConfigError(f"golden name {args.name!r} must match {NAME_RE.pattern} (no path separators)")

    keep_home = getattr(args, "keep_home", None)
    if keep_home and (args.live or not args.name or not args.provider):
        raise ConfigError("--keep-home needs one golden NAME and one --provider, and replay only "
                          "(--live already keeps its home)")

    if args.export_from:
        if not args.name:
            raise ConfigError("hmd eval NAME --export-from REF: NAME is required")
        if args.live or args.provider:
            raise ConfigError("--export-from does not record; run hmd eval <name> --live afterwards")
        gdir = export_golden(cfg, args.export_from, GOLDEN_ROOT, args.name, log=print)
        print(f"GOLDEN={gdir}")
        print("Edit its manifest.json: add 3-5 'assertions.must_contain' strings from the "
             "rendered report, and raise 'assertions.min_points' to about 80% of what it has.")
        return EXIT_OK

    names = [args.name] if args.name else list_goldens(GOLDEN_ROOT)
    if not names:
        # Not a KEY line (see pipeline/cli.py's module docstring) and not something a skill
        # parses either way, so stderr: stdout stays reserved for GOLDEN=/PROVIDER=/etc.
        print(f"hmd eval: no golden notebooks under {GOLDEN_ROOT}", file=sys.stderr)
        return EXIT_OK
    for name in names:
        if not os.path.exists(os.path.join(GOLDEN_ROOT, name, "manifest.json")):
            raise ConfigError(f"golden {name!r} not found under {GOLDEN_ROOT}")

    if args.live and args.budget is None and cfg.run.budget_usd == 0:
        raise ConfigError(
            "hmd eval --live refuses to run with no budget: pass --budget N, or --budget 0 to "
            "run uncapped on purpose, or set run.budget_usd in config.yaml")

    providers = [args.provider] if args.provider else list(PROVIDERS)

    codes: list[int] = []
    for name in names:
        gdir = os.path.join(GOLDEN_ROOT, name)
        manifest = load_manifest(gdir)
        for provider_name in providers:
            codes.append(_run_one(gdir, name, provider_name, manifest, live=args.live,
                                  budget_usd=args.budget, cfg=cfg, keep_home=keep_home))

    # An assertion failure/error outranks a budget pause: a CI gate keyed on the exit code must
    # still see the regression (2) even if some other golden/provider case in the same invocation
    # also paused (3) -- pausing is a "come back and finish this" signal, not a pass. Each case's
    # own STATUS=paused/fail line already says which is which.
    if EXIT_ERROR in codes:
        return EXIT_ERROR
    if EXIT_PAUSED in codes:
        return EXIT_PAUSED
    return EXIT_OK
