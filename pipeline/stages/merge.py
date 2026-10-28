"""Merge stage without an output ceiling (IDEA-211): outline -> one call per section (parallel) ->
citation check per section (parallel) -> verification. Source tokens are code-generated and the token->id map is stored with the
document (IDEA-212). Every trust-relevant field is computed here or in the renderer, never
copied from the model.

Invariant: a Point's `sources` is never taken from the model's `sources` list -- it is always
`list(dict.fromkeys(r.source for r in claim_refs))`, i.e. the distinct sources actually backing
its claim_refs, in the order those refs appear. This holds both when a point is first built (in
_clean_points) and after cross-section duplicate folding (in _apply_duplicates), so a
consumer's `len(set(point.sources))` ("confirmed by N sources") is always counting validated
evidence, never a source the model merely listed.

Claim refs and contradictions in a section's response are validated against that section's own
token set (the sources actually sent in that section's prompt), never the notebook-wide token
set -- a chunked section call only ever saw its own sources, so a token it did not receive must
never be trusted just because it happens to exist elsewhere in the notebook.

Duplicates across sections: points are folded by the verification call (the model names the
groups, code does the folding); contradictions are folded by code alone (_dedupe_contradictions),
with no LLM call, when they pit the same two sources against each other with the same positions or
on a near-identical topic (TOPIC_FOLD_THRESHOLD), and dropped when both positions are the same text.

Citation check (merge:check, IDEA-295): after the section calls and before verification -- so the
TL;DR and duplicate groups are built from checked points only and folding moves only refs that
survived the check; every applied fold group is still logged -- one call per section judges each
point over all of its cited quotes together (supported / partial / unsupported), each cited quote's
relevance to its point (related / unrelated) and each contradiction (real + a label, IDEA-013 /
not_a_contradiction + reason). Code applies the verdicts (_apply_check) and never takes text from that call: unrelated
refs are removed and `sources` recomputed from the refs left; a point judged unsupported, or left
with no ref, is dropped; a partial point is kept with support="partial"; a contradiction judged
not_a_contradiction is dropped; a section left empty is dropped. Missing verdicts fail closed (a
point without one is marked partial, a contradiction without one is dropped, a ref without one is
kept) and are counted in the log; verdicts for unknown ids are ignored. A real contradiction also
gets the check's label (same_conditions / may_differ_by_edition_or_date, IDEA-013), copied by code.

Retraction (merge:retract + merge:recheck, IDEA-297): after the check and before verification,
each point the check judged partial gets one call asking to delete its unsupported part. Code takes
that text only when it is a strict word-level deletion of the original (_is_strict_deletion) and
the unchanged check prompt then finds the shorter text supported with every quote related; the
point keeps its "(partly supported)" mark and its original text in `retracted_from`. Otherwise the
point is left as it was. Counts are stored on the document (`retraction`).
"""
from __future__ import annotations

import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from pydantic import ValidationError
from rapidfuzz import fuzz

from pipeline import db
from pipeline.llm.base import LLMFatalError, LLMSchemaError, LLMTruncatedError, request_interrupt, wait_futures
from pipeline.llm.client import LLMClient
from pipeline.prompts import analyze as analyze_prompts
from pipeline.prompts import check as check_prompts
from pipeline.prompts import retract as retract_prompts
from pipeline.prompts.merge import (
    MAX_SOURCES_PER_SECTION,
    PROMPT_VERSION,
    get_outline_prompt,
    get_section_prompt,
    get_verification_prompt,
    normalize_upload_date,
    source_digest,
)
from pipeline.schemas import (
    Analysis,
    CitationCheck,
    Contradiction,
    MergedDocument,
    MergedSection,
    Outline,
    OutlineSection,
    Point,
    Retraction,
    SectionBody,
    Subsection,
    Verification,
    claims_hash,
)
from pipeline.stages.output import RENDER_VERSION


# Share of a section's points + contradictions the citation check must give a verdict for
# (see _apply_check; ref relevances do not count).
MIN_CHECK_COVERAGE = 0.5


class MergeError(Exception):
    pass


class _NotStarted(Exception):
    """A parallel call skipped because an LLMFatalError had already stopped the batch."""


def _parallel_calls(fn: Callable, items: list, max_parallel: int) -> list:
    """fn(item) for every item, up to max_parallel at once, results in item order. As
    `pool.map` did: every started call is waited for, then the first exception in item order is
    raised. One difference: the first LLMFatalError (auth, billing, credit balance) stops the
    batch -- calls not yet started are skipped, never sent -- and is raised in preference to any
    other exception, since every further call would fail the same way."""
    stop = threading.Event()

    def run_one(item):
        if stop.is_set():
            raise _NotStarted()
        try:
            return fn(item)
        except LLMFatalError:
            stop.set()
            raise

    pool = ThreadPoolExecutor(max_workers=max(1, max_parallel))
    try:
        futures = [pool.submit(run_one, item) for item in items]
        wait_futures(futures)
    except BaseException:
        request_interrupt()   # a call blocked on a batch stops waiting, so shutdown can finish
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True)
    fatal = next((f.exception() for f in futures if isinstance(f.exception(), LLMFatalError)), None)
    if fatal is not None:
        raise fatal
    return [f.result() for f in futures]


def build_source_map(analyzed: list[dict]) -> dict[str, int]:
    return {f"S{i}": src["id"] for i, src in enumerate(sorted(analyzed, key=lambda s: s["id"]), 1)}


def _chunk_section(sec: OutlineSection, max_size: int) -> list[OutlineSection]:
    """Split a section into balanced chunks of at most max_size sources, in order.

    Tokens are deduped (order-preserving) first so the size arithmetic sees unique tokens, not
    raw (possibly repeated) outline output. A section already within the bound comes back as a
    single-element list, titled the same as it went in. An oversized section uses the minimum
    number of chunks (ceil(n / max_size)) with sizes as even as possible -- e.g. 25 sources at
    max_size 10 become chunks of 9/8/8, not 10/10/5 -- so no chunk is needlessly starved. Each
    chunk is an independent section call: points are only folded across a split by the
    verification pass, contradictions only by _dedupe_contradictions (same sources with the same
    positions or a near-identical topic).
    """
    tokens = list(dict.fromkeys(sec.sources))
    n = len(tokens)
    if n <= max_size:
        return [OutlineSection(title=sec.title, scope=sec.scope, sources=tokens)]
    num_chunks = -(-n // max_size)   # ceil division
    base, extra = divmod(n, num_chunks)
    sizes = [base + 1] * extra + [base] * (num_chunks - extra)
    chunks = []
    start = 0
    for i, size in enumerate(sizes, 1):
        chunk = tokens[start:start + size]
        start += size
        chunks.append(OutlineSection(title=sec.title if i == 1 else f"{sec.title} ({i})",
                                     scope=sec.scope, sources=chunk))
    return chunks


def _clean_points(subsections: list[Subsection], allowed: dict[str, tuple[str, str, Analysis]],
                  log: Callable[[str], None]) -> list[Subsection]:
    """Validate points against `allowed` -- the analyses of only the sources a section's prompt
    actually contained -- and rebuild `sources` purely from the claim_refs that survive, never
    from the model's own `sources` list (see module docstring)."""
    out = []
    for sub in subsections:
        kept = []
        for p in sub.points:
            refs = [r for r in p.claim_refs
                    if r.source in allowed and 0 <= r.claim < len(allowed[r.source][2].key_claims)]
            if not refs:
                log(f"  merge: dropped point without a valid claim ref: {p.text[:80]!r}")
                continue
            sources = list(dict.fromkeys(r.source for r in refs))
            kept.append(Point(text=p.text, sources=sources, claim_refs=refs))
        if kept:
            out.append(Subsection(title=sub.title, points=kept))
    return out


def _apply_duplicates(sections: list[MergedSection], groups: list[list[str]], log: Callable[[str], None]) -> None:
    index: dict[str, Point] = {p.id: p for s in sections for ss in s.subsections for p in ss.points}
    used: set[str] = set()
    remove: set[str] = set()
    ignored = 0
    for group in groups:
        # Dedupe within the group (order-preserving) so a repeated id in one group can never
        # fold a point into itself, then drop unknown ids and ids already consumed by an
        # earlier group (an id in two groups: the first group wins).
        ids = [i for i in dict.fromkeys(group) if i in index and i not in used]
        if len(ids) < 2:
            ignored += 1
            continue
        keep = index[ids[0]]
        before = list(keep.sources)
        folded = []
        for pid in ids[1:]:
            other = index[pid]
            seen = {(r.source, r.claim) for r in keep.claim_refs}
            added = [r for r in other.claim_refs if (r.source, r.claim) not in seen]
            keep.claim_refs.extend(added)
            if added:
                # these refs were checked against the folded point's text, not the kept one's: the
                # renderer does not call the point "confirmed" (IDEA-307b, review 2026-10-18)
                keep.refs_from_fold = True
            remove.add(pid)
            folded.append(f"{pid} [{', '.join(other.sources)}]")
        keep.sources = list(dict.fromkeys(r.source for r in keep.claim_refs))
        used.update(ids)
        # One line per applied group, so a fold that attached unrelated evidence can be traced in
        # the run log (folding runs after the citation check, so the refs it moves were checked
        # against the folded point, never re-checked against the kept point's text).
        log(f"  merge: fold group: kept {keep.id} [{', '.join(before)}], folded {'; '.join(folded)} "
            f"-> {keep.id} cites [{', '.join(keep.sources)}]")
    if remove:
        log(f"  merge: folded {len(remove)} duplicate points across sections")
    if ignored:
        log(f"  merge: dropped {ignored} duplicate group(s) with fewer than two still-unused points")
    for s in sections:
        for ss in s.subsections:
            ss.points = [p for p in ss.points if p.id not in remove]
        s.subsections = [ss for ss in s.subsections if ss.points]


_NON_WORD_RE = re.compile(r"[^\w\s]")


def _norm_position(text: str) -> str:
    """Casefold, turn punctuation into spaces and collapse whitespace, so trivially reworded
    copies ("Five years." / "five  years") of one position compare equal."""
    return " ".join(_NON_WORD_RE.sub(" ", text.casefold()).split())


def _contradiction_key(c: Contradiction) -> frozenset[tuple[str, str]]:
    # Unordered: S1-says-X vs S2-says-Y is the same disagreement as S2-says-Y vs S1-says-X. The
    # topic label is left out on purpose -- two sections word the same disagreement differently.
    return frozenset({(c.source_a, _norm_position(c.position_a)), (c.source_b, _norm_position(c.position_b))})


# Near-duplicate contradictions (same unordered source pair, reworded topic) are folded when the
# normalised topics score at least this on rapidfuzz's token_set_ratio. Calibrated on the three
# golden step-1 merges (2026-10-09), every same-pair topic pair scored: the reworded copies of one
# disagreement 82.1-100 ("Fitbit Air battery life rating" / "... figure" 86.8, "Timeline to habit
# automaticity" / "How long it takes for a habit to reach automaticity" 82.4, "Magnitude of
# compounding from 1% daily improvement" / "Magnitude of 1% daily compounding over one year" 82.1),
# genuinely different subjects at most 65.9 ("Garmin subscription requirement for core features" /
# "Garmin Connect subscription model value"); synthetic "Battery life" / "Battery charging time"
# scores 73.7. 80 sits in that gap. tests/test_merge.py pins these examples.
TOPIC_FOLD_THRESHOLD = 80.0

# token_set_ratio scores a topic whose words are a subset of the other's at 100 ("Battery life" vs
# "Battery life of the Pixel Watch 3"), is blind to a changed number ("Zoom at 10x" vs "Zoom at
# 30x" scores 94.7) and to one swapped criterion ("Photo quality in low light" vs "Video quality in
# low light" scores 87), so two topics are only folded when their identity-bearing tokens -- any
# token with a digit, edition words, and photo/video-style criterion words -- are the same set:
# different models, generations, numbers or criteria are never folded into one contradiction.
_EDITION_WORDS = frozenset({"pro", "max", "ultra", "plus", "mini", "lite", "fe", "se", "edge", "fold", "flip"})
_CRITERION_WORDS = {"photo": "photo", "photos": "photo", "photography": "photo", "video": "video",
                    "videos": "video", "day": "day", "daylight": "day", "night": "night", "indoor": "indoor",
                    "indoors": "indoor", "outdoor": "outdoor", "outdoors": "outdoor", "selfie": "selfie",
                    "selfies": "selfie", "front": "front", "rear": "rear"}


def _identity_tokens(norm_topic: str) -> frozenset[str]:
    out = set()
    for t in norm_topic.split():
        if t in _EDITION_WORDS or any(ch.isdigit() for ch in t):
            out.add(t)
        elif t in _CRITERION_WORDS:
            out.add(_CRITERION_WORDS[t])
    return frozenset(out)


def _same_disagreement(a: Contradiction, b: Contradiction) -> bool:
    """Same unordered source pair and a near-identical topic (see TOPIC_FOLD_THRESHOLD)."""
    if frozenset({a.source_a, a.source_b}) != frozenset({b.source_a, b.source_b}):
        return False
    ta, tb = _norm_position(a.topic), _norm_position(b.topic)
    if _identity_tokens(ta) != _identity_tokens(tb):
        return False
    return fuzz.token_set_ratio(ta, tb) >= TOPIC_FOLD_THRESHOLD


def _dedupe_contradictions(sections: list[MergedSection], log: Callable[[str], None]) -> list[MergedSection]:
    """Deterministic, code-only counterpart of the verification call's point folding. In document
    order, within and across sections, a contradiction is dropped when:
      - its two positions normalise to the same text (the sources agree; not a disagreement);
      - an earlier one has the same contradiction key (same sources, same normalised positions);
      - an earlier one pits the same unordered source pair against each other on a near-identical
        topic (_same_disagreement: token_set_ratio >= TOPIC_FOLD_THRESHOLD and the same digit /
        edition tokens) -- the first one is kept, the later reworded copy is folded away.
    A section left with neither points nor contradictions is dropped. No LLM call judges this."""
    seen: set[frozenset[tuple[str, str]]] = set()
    kept_all: list[Contradiction] = []
    folded = agreeing = 0
    out = []
    for s in sections:
        kept = []
        for c in s.contradictions:
            if _norm_position(c.position_a) == _norm_position(c.position_b):
                agreeing += 1
                continue
            key = _contradiction_key(c)
            if key in seen or any(_same_disagreement(prev, c) for prev in kept_all):
                folded += 1
                continue
            seen.add(key)
            kept.append(c)
            kept_all.append(c)
        s.contradictions = kept
        if not s.subsections and not s.contradictions:
            log(f"  merge: dropped section {s.title!r}: only duplicate contradictions were left")
            continue
        out.append(s)
    if agreeing:
        log(f"  merge: dropped {agreeing} contradiction(s) whose two positions are the same")
    if folded:
        log(f"  merge: folded {folded} duplicate contradiction(s)")
    return out


def _check_inputs(section: MergedSection, analyses: dict[str, tuple[str, str, Analysis]],
                  upload_dates: dict[str, str]
                  ) -> tuple[list[dict], list[dict], dict[str, tuple[Point, int]], dict[str, Contradiction]]:
    """What the check call sees for one section, built from stored data only: each point's text
    and, per claim ref, the stored quote, source title and upload date; each contradiction's
    positions with the quotes of its claims_a/claims_b. Returns (points, contradictions, ref id ->
    (point, index into point.claim_refs), contradiction id -> contradiction). Ref ids (R1, ...) and
    contradiction ids (C1, ...) are code-generated per call; points keep the merge's own P-ids."""
    def title(tok: str) -> str:
        return analyses[tok][0]

    def date_of(tok: str) -> str:
        return upload_dates.get(tok) or normalize_upload_date(None)

    def quote(tok: str, idx: int) -> str | None:
        claims = analyses[tok][2].key_claims
        return claims[idx].quote if 0 <= idx < len(claims) else None

    def side(position: str, tok: str, idxs: list[int]) -> dict:
        quotes = [q for q in (quote(tok, i) for i in dict.fromkeys(idxs)) if q is not None]
        return {"position": position, "source_title": title(tok), "upload_date": date_of(tok), "quotes": quotes}

    points: list[dict] = []
    ref_ids: dict[str, tuple[Point, int]] = {}
    for ss in section.subsections:
        for p in ss.points:
            refs = []
            for i, r in enumerate(p.claim_refs):
                rid = f"R{len(ref_ids) + 1}"
                ref_ids[rid] = (p, i)
                refs.append({"ref": rid, "source_title": title(r.source), "upload_date": date_of(r.source),
                             "quote": quote(r.source, r.claim) or ""})
            points.append({"point": p.id, "text": p.text, "refs": refs})
    contradictions: list[dict] = []
    contra_ids: dict[str, Contradiction] = {}
    for c in section.contradictions:
        cid = f"C{len(contra_ids) + 1}"
        contra_ids[cid] = c
        contradictions.append({"contradiction": cid, "topic": c.topic,
                               "side_a": side(c.position_a, c.source_a, c.claims_a),
                               "side_b": side(c.position_b, c.source_b, c.claims_b)})
    return points, contradictions, ref_ids, contra_ids


def _apply_check(section: MergedSection, check: CitationCheck, ref_ids: dict[str, tuple[Point, int]],
                 contra_ids: dict[str, Contradiction], stats: Counter,
                 log: Callable[[str], None], partials: list[Point] | None = None) -> MergedSection | None:
    """Apply one section's check verdicts in code; the response carries ids and verdicts only, so
    no text is ever taken from it. Returns the section, or None when nothing is left in it.

    Shape (check-2026-10-09.2): one verdict per point, judged over all of its quotes together
    (supported / partial / unsupported); one relevance per ref (related / unrelated); one verdict
    per contradiction. Applied as: an unrelated ref is removed and `sources` recomputed from the
    refs left; a point judged unsupported, or left with no ref, is dropped; a partial point is kept
    with support="partial"; a supported point is left unmarked; a not_a_contradiction is dropped.

    Ids: a point verdict is matched by the point's P-id, a ref relevance by its R-id wherever it is
    listed (R-ids are unique per call), a contradiction by its C-id. The first verdict for an id
    wins; a verdict for an id not in this section is ignored and counted.

    Fails closed (security review 2026-10-09, adapted): a point without a verdict is marked partial,
    never treated as supported; a ref without a relevance is kept; a contradiction without a verdict
    is dropped; and a response that gives verdicts for fewer than MIN_CHECK_COVERAGE of the
    section's items -- its points plus its contradictions; ref relevances do not count -- is a
    MergeError, so a truncated or steered check cannot wave a section through.

    `partials`, when given, collects every kept point the check judged "partial" (not one marked
    partial only for lack of a verdict): the candidates for retraction (_retract_partials)."""
    point_ids = {p.id: p for ss in section.subsections for p in ss.points}
    point_verdicts: dict[str, str] = {}
    ref_relevance: dict[str, str] = {}
    for pv in check.points:
        if pv.point not in point_ids:
            stats["unknown_ids"] += 1
        elif pv.verdict is not None and pv.point not in point_verdicts:
            point_verdicts[pv.point] = pv.verdict
        for rv in pv.refs:
            if rv.ref not in ref_ids:
                stats["unknown_ids"] += 1
            elif rv.relevance is not None and rv.ref not in ref_relevance:
                ref_relevance[rv.ref] = rv.relevance
    contra_verdicts: dict[str, tuple[str, str | None, str | None]] = {}
    for v in check.contradictions:
        if v.contradiction not in contra_ids:
            stats["unknown_ids"] += 1
        elif v.contradiction not in contra_verdicts:
            contra_verdicts[v.contradiction] = (v.verdict, v.reason, v.label)

    total = len(point_ids) + len(contra_ids)
    answered = len(point_verdicts) + len(contra_verdicts)
    if total and answered < MIN_CHECK_COVERAGE * total:
        raise MergeError(f"citation check answered only {answered} of {total} items in section "
                         f"{section.title!r}")

    by_ref: dict[tuple[int, int], str | None] = {(id(p), i): ref_relevance.get(rid)
                                                 for rid, (p, i) in ref_ids.items()}
    for ss in section.subsections:
        kept = []
        for p in ss.points:
            refs = []
            for i, r in enumerate(p.claim_refs):
                rel = by_ref.get((id(p), i))
                if rel == "unrelated":
                    stats["refs_removed"] += 1
                    log(f"  merge:check: removed {p.id}'s ref {r.source} claim {r.claim} (unrelated)")
                    continue
                if rel is None:
                    stats["refs_unchecked"] += 1
                refs.append(r)
            verdict = point_verdicts.get(p.id)
            if not refs:
                stats["points_dropped"] += 1
                log(f"  merge:check: dropped {p.id} (no related quote left): {p.text[:80]!r}")
                continue
            if verdict == "unsupported":
                stats["points_dropped"] += 1
                log(f"  merge:check: dropped {p.id} (unsupported by its quotes): {p.text[:80]!r}")
                continue
            p.claim_refs = refs
            p.sources = list(dict.fromkeys(r.source for r in refs))
            if verdict == "supported":
                p.support = None
            else:
                note = ""
                if verdict is None:
                    stats["points_unchecked"] += 1
                    note = " (no verdict)"
                p.support = "partial"
                stats["points_partial"] += 1
                log(f"  merge:check: {p.id} is only partly supported{note}: {p.text[:80]!r}")
                if verdict == "partial" and partials is not None:
                    partials.append(p)
            kept.append(p)
        ss.points = kept
    section.subsections = [ss for ss in section.subsections if ss.points]

    by_contra = {id(c): contra_verdicts.get(cid) for cid, c in contra_ids.items()}
    kept_c = []
    for c in section.contradictions:
        v = by_contra.get(id(c))
        if v is None:
            stats["contradictions_unchecked"] += 1
            log(f"  merge:check: dropped contradiction {c.topic[:80]!r} (no verdict)")
            continue
        if v[0] == "not_a_contradiction":
            stats[f"contradiction_dropped:{v[1]}"] += 1
            log(f"  merge:check: dropped contradiction {c.topic[:80]!r} ({v[1]})")
            continue
        # IDEA-013: the check's label for a real contradiction, copied by code (None: no label)
        c.check_label = v[2]
        if v[2] == "may_differ_by_edition_or_date":
            stats["contradictions_may_differ"] += 1
        kept_c.append(c)
    section.contradictions = kept_c

    if not section.subsections and not section.contradictions:
        stats["sections_dropped"] += 1
        log(f"  merge:check: dropped section {section.title!r}: nothing survived the citation check")
        return None
    return section


def _log_check_stats(stats: Counter, log: Callable[[str], None]) -> None:
    by_reason = {k.split(":", 1)[1]: n for k, n in stats.items() if k.startswith("contradiction_dropped:")}
    reasons = ", ".join(f"{r} {n}" for r, n in sorted(by_reason.items())) or "none"
    log(f"  merge:check: {stats['refs_removed']} ref(s) removed, {stats['points_dropped']} point(s) dropped, "
        f"{stats['points_partial']} point(s) partly supported, "
        f"{sum(by_reason.values())} contradiction(s) dropped (by reason: {reasons}), "
        f"{stats['sections_dropped']} section(s) dropped")
    if stats["contradictions_may_differ"]:
        log(f"  merge:check: {stats['contradictions_may_differ']} contradiction(s) may differ by edition or date")
    if stats["points_unchecked"] or stats["refs_unchecked"] or stats["contradictions_unchecked"]:
        log(f"  merge:check: no verdict for {stats['points_unchecked']} point(s), {stats['refs_unchecked']} ref(s) "
            f"and {stats['contradictions_unchecked']} contradiction(s): points marked partly supported, "
            f"refs kept, contradictions dropped")
    if stats["unknown_ids"]:
        log(f"  merge:check: ignored {stats['unknown_ids']} verdict(s) for unknown ids")


# --- retraction of partly supported points (merge:retract + merge:recheck, IDEA-297) ---------------

# Punctuation a deletion may leave behind or remove around a word. Never "$", "%", "-", "+" or a
# digit separator inside a token: "6,000" stays one word, and "-5" or "$5" keep their sign or unit.
_EDGE_PUNCT = ".,;:!?…()[]{}\"'“”‘’«»—–"
# Word separators besides whitespace: em/en dash and slash, so "days—not 3" or "A/not B" cannot
# hide a word inside a token. A dash also ends a clause; a slash does not.
_SPLIT_RE = re.compile(r"(\s+|[—–/])")
_CLAUSE_END = ",;:.!?"
# A deleted run may start at one of these (a clause it introduces), as well as after clause punctuation.
_CLAUSE_WORDS = frozenset({"and", "but", "or", "while", "whereas", "although", "though", "because", "since",
                           "which", "who", "where", "when", "so", "yet", "as"})
# A deleted clause may not start with one of these (nor with a number): removing a negation, a
# hedge, a comparison or a limiting condition ("unless ...", "except ...") changes what the rest
# says instead of only removing an unsupported clause.
_PROTECTED_WORDS = frozenset({"not", "no", "never", "none", "nor", "nothing", "neither", "nobody", "without",
                              "cannot", "hardly", "rarely", "seldom", "except", "excluding", "unless", "if",
                              "only", "than", "few", "fewer", "less", "more", "most", "least", "about",
                              "around", "roughly", "approximately", "nearly", "almost", "up", "under", "over"})
MIN_RETRACTED_WORDS = 3


def _tokens(text: str) -> list[tuple[str, bool]]:
    """(word, ends_clause) per word: whitespace, em/en dash and slash separate words; edge
    punctuation is stripped from the word; ends_clause is True when the word is followed by clause
    punctuation (, ; : . ! ?) or a dash."""
    out: list[tuple[str, bool]] = []
    for piece in _SPLIT_RE.split(text):
        if not piece or piece.isspace():
            continue
        if piece in "—–":
            if out:
                out[-1] = (out[-1][0], True)
            continue
        if piece == "/":
            continue
        word = piece.strip(_EDGE_PUNCT)
        tail = piece.rstrip(")]}\"'“”‘’«»")
        ends = bool(tail) and tail[-1] in _CLAUSE_END
        if word:
            out.append((word, ends))
        elif out and ends:                      # a lone "," or ";" token
            out[-1] = (out[-1][0], True)
    return out


def _protected(word: str) -> bool:
    w = word.casefold()
    return (w in _PROTECTED_WORDS or w.endswith("n't") or w.endswith("n’t")
            or any(ch.isdigit() for ch in w))


def _is_strict_deletion(original: str, candidate: str | None) -> bool:
    """True when `candidate` is `original` with whole clauses deleted and nothing else changed.

    Words (_tokens) of the candidate must be a subsequence of the original's, in order and
    case-sensitive; the first word must be the original's first word (only its case may differ), so
    nothing is ever cut from the start. At least one word is gone and MIN_RETRACTED_WORDS are left.
    Every deleted run must be a whole clause: it starts after clause punctuation or a dash, or at a
    clause word (and, but, which, ...), and it ends at clause punctuation, a dash or the end of the
    text; and it must not start with a protected word (negation, hedge, comparison, limiting
    condition) or a number. Conservative on purpose: a refused retraction only leaves the point
    marked partly supported."""
    if not candidate or not candidate.strip():
        return False
    orig, cand = _tokens(original), [w for w, _ in _tokens(candidate)]
    if len(cand) < MIN_RETRACTED_WORDS or len(cand) >= len(orig):
        return False
    if cand[0].casefold() != orig[0][0].casefold():
        return False
    kept = [0]
    i = 1
    for w in cand[1:]:
        while i < len(orig) and orig[i][0] != w:
            i += 1
        if i == len(orig):
            return False
        kept.append(i)
        i += 1
    kept_set = set(kept)
    j = 0
    while j < len(orig):
        if j in kept_set:
            j += 1
            continue
        start = j
        while j < len(orig) and j not in kept_set:
            j += 1
        first = orig[start][0]
        starts_ok = orig[start - 1][1] or first.casefold() in _CLAUSE_WORDS
        ends_ok = j == len(orig) or orig[j - 1][1]
        if not (starts_ok and ends_ok) or _protected(first):
            return False
    return True


def _retract_one(topic: str, section_title: str, p: Point, analyses: dict[str, tuple[str, str, Analysis]],
                 upload_dates: dict[str, str], llm: LLMClient, model: str) -> tuple[str | None, str]:
    """One retraction attempt: (accepted shortened text or None, reason). Only the retraction's and
    the re-check's malformed answers (schema or truncation errors) are caught -- counted as a
    rejection, the original kept; every other error (fatal, budget, interrupt, cassette miss)
    propagates like any other merge call."""
    probe = MergedSection(title=section_title, subsections=[Subsection(title="-", points=[p])])
    pts, _, ref_ids, _ = _check_inputs(probe, analyses, upload_dates)
    quotes = [{k: r[k] for k in ("source_title", "upload_date", "quote")} for r in pts[0]["refs"]]
    try:
        answer = llm.call_json(retract_prompts.get_retract_prompt(topic, section_title, p.text, quotes),
                               Retraction, model=model, stage="merge:retract")
    except (LLMSchemaError, LLMTruncatedError) as e:
        return None, f"retraction answer unusable ({type(e).__name__})"
    if answer.text is None:
        return None, "nothing to delete"
    if not _is_strict_deletion(p.text, answer.text):
        return None, "not a strict deletion"
    short = " ".join(answer.text.split())
    recheck_points = [{**pts[0], "text": short}]   # same point id and refs, the shortened text
    try:
        check = llm.call_json(check_prompts.get_check_prompt(topic, section_title, recheck_points, []), CitationCheck,
                              model=model, stage="merge:recheck")
    except (LLMSchemaError, LLMTruncatedError) as e:
        return None, f"re-check answer unusable ({type(e).__name__})"
    verdict = next((pv.verdict for pv in check.points if pv.point == p.id and pv.verdict is not None), None)
    unrelated = any(rv.relevance == "unrelated" for pv in check.points for rv in pv.refs if rv.ref in ref_ids)
    if verdict != "supported" or unrelated:
        why = "a quote judged unrelated" if verdict == "supported" else f"verdict {verdict or 'missing'}"
        return None, f"re-check did not find it supported ({why})"
    return short, "accepted"


def _retract_partials(topic: str, partials: list[tuple[str, Point]], analyses: dict[str, tuple[str, str, Analysis]],
                      upload_dates: dict[str, str], llm: LLMClient, model: str, max_parallel: int,
                      log: Callable[[str], None]) -> dict[str, int]:
    """IDEA-297: for each point the check judged partial, ask for a deletion of its unsupported part
    (merge:retract); code accepts it only as a strict word-level deletion (_is_strict_deletion) that
    the unchanged citation check then finds supported with every quote related (merge:recheck). An
    accepted point gets the shorter text and keeps its original in `retracted_from`; it still renders
    "(partly supported)" (owner decision 2026-10-18). Anything else keeps the point unchanged.
    Calls run in parallel like the other merge calls (so they share a Message Batch wave)."""
    results = _parallel_calls(
        lambda item: _retract_one(topic, item[0], item[1], analyses, upload_dates, llm, model),
        partials, max_parallel)
    counts = {"attempted": len(partials), "accepted": 0, "rejected": 0}
    for (_, p), (short, why) in zip(partials, results):
        if short is None:
            counts["rejected"] += 1
            log(f"  merge:retract: {p.id} kept as is ({why}): {p.text[:80]!r}")
            continue
        counts["accepted"] += 1
        log(f"  merge:retract: {p.id} shortened: {p.text[:80]!r} -> {short[:80]!r}")
        p.retracted_from, p.text = p.text, short
    if partials:
        log(f"  merge:retract: {counts['attempted']} attempted, {counts['accepted']} accepted, "
            f"{counts['rejected']} rejected")
    return counts


def run_merge(db_path: str, notebook_id: int, topic: str, llm: LLMClient, model: str, *, run_id: int | None,
              report_path: str, max_parallel: int = 4, log: Callable[[str], None] = print) -> int:
    analyzed = db.get_analyzed_sources(db_path, notebook_id)
    if not analyzed:
        raise MergeError(f"no analyzed sources in notebook {notebook_id}")
    source_map = build_source_map(analyzed)
    by_id = {s["id"]: s for s in analyzed}
    analyses: dict[str, tuple[str, str, Analysis]] = {}
    # Upload dates go into every merge prompt (inside the data blocks) so a change over time is
    # not mistaken for a contradiction and an old status is never rendered as current.
    upload_dates = {tok: normalize_upload_date(by_id[sid].get("upload_date")) for tok, sid in source_map.items()}
    for tok, sid in source_map.items():
        row = by_id[sid]
        try:
            analysis = Analysis.model_validate_json(row["analysis_json"])
        except ValidationError as e:
            raise MergeError(
                f"source {sid} ({row['title']!r}) has an unreadable analysis; re-run analyze for it: {e}"
            ) from None
        analyses[tok] = (row["title"] or "", row["channel"] or "", analysis)
    log(f"Merging {len(analyses)} analyzed sources...")

    # 1. outline
    digests = [source_digest(tok, *analyses[tok], upload_date=upload_dates[tok]) for tok in source_map]
    outline = llm.call_json(
        get_outline_prompt(topic, digests, max_sources_per_section=MAX_SOURCES_PER_SECTION),
        Outline, model=model, stage="merge:outline")

    sections: list[OutlineSection] = []
    assigned: set[str] = set()
    for sec in outline.sections:
        tokens = list(dict.fromkeys(sec.sources))
        known = [t for t in tokens if t in analyses]
        unknown = [t for t in tokens if t not in analyses]
        if unknown:
            log(f"  merge: outline section {sec.title!r} cited unknown source(s), dropped: {unknown}")
        if not known:
            log(f"  merge: dropped outline section {sec.title!r}: no known sources")
            continue
        assigned.update(known)
        chunks = _chunk_section(OutlineSection(title=sec.title, scope=sec.scope, sources=known),
                                MAX_SOURCES_PER_SECTION)
        if len(chunks) > 1:
            log(f"  merge: split outline section {sec.title!r} into {len(chunks)} calls "
               f"(> {MAX_SOURCES_PER_SECTION} sources); the section calls do not see each other's "
               f"points or contradictions")
        sections.extend(chunks)

    orphans = [t for t in source_map if t not in assigned]
    if orphans:
        log(f"  merge: {len(orphans)} source(s) not placed by the outline, added to 'Other findings': {orphans}")
        sections.extend(_chunk_section(
            OutlineSection(title="Other findings", scope="sources the outline did not place", sources=orphans),
            MAX_SOURCES_PER_SECTION))

    # Every token in source_map is either assigned by the outline or an orphan swept into
    # "Other findings" above, and source_map is non-empty (checked via `analyzed` above), so
    # sections can never come out empty here -- unreachable in practice, but a hard error is
    # cheaper than a confusing downstream failure if it ever does.
    if not sections:
        raise MergeError("outline produced no usable section")
    log(f"  outline: {len(sections)} section call(s)")

    # 2. sections, in parallel
    def write_section(sec: OutlineSection) -> SectionBody:
        return llm.call_json(get_section_prompt(topic, sec, analyses, upload_dates), SectionBody, model=model,
                             stage="merge:section")

    bodies = _parallel_calls(write_section, sections, max_parallel)

    merged_sections: list[MergedSection] = []
    counter = 0
    for sec, body in zip(sections, bodies):
        # A section call only ever saw its own sources; validate its response against exactly
        # that set, never the notebook-wide `analyses` (a chunked call for S1..S10 must not be
        # trusted if it cites S27, even though S27 exists elsewhere in the notebook).
        allowed = {t: analyses[t] for t in sec.sources}
        subsections = _clean_points(body.subsections, allowed, log)
        for ss in subsections:
            for p in ss.points:
                counter += 1
                p.id = f"P{counter}"
        contradictions = []
        dropped_contradictions = dropped_claim_idx = 0
        for c in body.contradictions:
            if c.source_a not in allowed or c.source_b not in allowed or c.source_a == c.source_b:
                dropped_contradictions += 1
                continue
            # claims_a/claims_b are evidence pointers for the citation check: an index that does
            # not exist for that side's source is dropped, the contradiction itself is kept.
            n_a = len(allowed[c.source_a][2].key_claims)
            n_b = len(allowed[c.source_b][2].key_claims)
            valid_a = list(dict.fromkeys(i for i in c.claims_a if i < n_a))
            valid_b = list(dict.fromkeys(i for i in c.claims_b if i < n_b))
            dropped_claim_idx += len(set(c.claims_a)) - len(valid_a) + len(set(c.claims_b)) - len(valid_b)
            c.claims_a, c.claims_b = valid_a, valid_b
            contradictions.append(c)
        if dropped_contradictions:
            log(f"  merge: dropped {dropped_contradictions} contradiction(s) citing unknown or identical sources")
        if dropped_claim_idx:
            log(f"  merge: dropped {dropped_claim_idx} contradiction claim index(es) that do not exist")
        n_points = sum(len(ss.points) for ss in subsections)
        log(f"  section {sec.title!r}: {n_points} points, {len(contradictions)} contradictions")
        if not subsections and not contradictions:
            log(f"  merge: dropped section {sec.title!r}: no points or contradictions survived")
            continue
        merged_sections.append(MergedSection(title=sec.title, subsections=subsections, contradictions=contradictions))
    merged_sections = _dedupe_contradictions(merged_sections, log)

    if not merged_sections:
        raise MergeError("no point or contradiction survived claim-ref validation")

    # 3. citation check, one call per section in parallel, before verification: the TL;DR and the
    # duplicate groups are then built only from checked points, and folding only ever moves refs
    # that survived the check. All-or-nothing like the other merge calls.
    check_inputs = [_check_inputs(s, analyses, upload_dates) for s in merged_sections]

    def check_section(item: tuple[MergedSection, tuple]) -> CitationCheck:
        sec, (pts, contras, _, _) = item
        return llm.call_json(check_prompts.get_check_prompt(topic, sec.title, pts, contras), CitationCheck,
                             model=model, stage="merge:check")

    checks = _parallel_calls(check_section, list(zip(merged_sections, check_inputs)), max_parallel)
    stats: Counter = Counter()
    partials: list[tuple[str, Point]] = []
    checked = []
    for s, chk, inp in zip(merged_sections, checks, check_inputs):
        found: list[Point] = []
        checked.append(_apply_check(s, chk, inp[2], inp[3], stats, log, found))
        partials.extend((s.title, p) for p in found)
    merged_sections = [s for s in checked if s is not None]
    _log_check_stats(stats, log)
    if not merged_sections:
        raise MergeError("no point or contradiction survived the citation check")

    # 3b. retraction (IDEA-297), before verification so the TL;DR and duplicate groups see the
    # shortened text: one merge:retract call per point the check judged partial, plus a merge:recheck
    # call for each strict deletion.
    retraction = _retract_partials(topic, partials, analyses, upload_dates, llm, model, max_parallel, log)

    # 4. verification: TL;DR and cross-section duplicates, from the checked points. A merge where
    # no point survived (claim-ref validation or the check) but contradictions did is still a
    # valid (contradictions-only) document: the verification call only ever sees points, so it is
    # skipped and the TL;DR is left empty -- the renderer says so in place of bullets.
    listing = [(s.title, ss.title, p) for s in merged_sections for ss in s.subsections for p in ss.points]
    tldr: list[str] = []
    if listing:
        verification = llm.call_json(get_verification_prompt(topic, listing, upload_dates), Verification, model=model,
                                     stage="merge:verify")
        _apply_duplicates(merged_sections, verification.duplicates, log)
        tldr = verification.tldr
    else:
        log("  merge: no point survived claim-ref validation and the citation check; storing contradictions "
            "only, no TL;DR (verification call skipped)")

    # Pin the document to the exact analyses it resolved against: the renderer refuses it if any
    # source's claims_hash no longer matches (a re-analysis since this merge, same count or not).
    claim_counts = {tok: len(a.key_claims) for tok, (_, _, a) in analyses.items()}
    claim_hashes = {tok: claims_hash(a) for tok, (_, _, a) in analyses.items()}
    # The recorded "render" version is the renderer current at merge time, not necessarily the one
    # that eventually produces the report: run_output always re-renders the *latest* merge, so a
    # report generated later by a newer renderer is never re-stamped here (MILESTONE-1 runs merge
    # and output together, so in practice the two stay in step).
    doc = MergedDocument(tldr=tldr, sections=merged_sections, source_map=source_map,
                         claim_counts=claim_counts, claim_hashes=claim_hashes, retraction=retraction,
                         prompt_versions={"analyze": analyze_prompts.PROMPT_VERSION, "merge": PROMPT_VERSION,
                                          "check": check_prompts.PROMPT_VERSION,
                                          "retract": retract_prompts.PROMPT_VERSION, "render": RENDER_VERSION})
    merge_id = db.create_merge(db_path, notebook_id, run_id, model, doc.model_dump_json(),
                               json.dumps(source_map), json.dumps(doc.prompt_versions), report_path)
    final_points = sum(len(ss.points) for s in merged_sections for ss in s.subsections)
    log(f"Merge complete -> merge_id={merge_id}, {final_points} points, {len(merged_sections)} sections")
    return merge_id
