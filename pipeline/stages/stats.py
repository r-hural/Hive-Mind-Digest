"""Facts about one stored merge, computed by code (never by a model): which sources each point
cites, which points a contradiction disputes, and per-source and per-section counts. The summary
renderer (scorecard, coverage, badges, per-section sources) and the extra formats read these, so
every number in a report has one definition (IDEA-010, IDEA-015, IDEA-017)."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

from pipeline.schemas import MergedDocument, Point

# A source is "worth watching in full" when it contributes at least this many points nobody else
# makes and is in the top third of sources by that count (IDEA-010).
WORTH_WATCHING_MIN_UNIQUE = 3


@dataclass(frozen=True)
class PointStat:
    section: int
    subsection: int
    point: Point
    source_ids: tuple[int, ...]
    disputed: bool


@dataclass
class SourceStat:
    source_id: int
    token: str
    claims: int = 0
    claims_cited: int = 0
    points: int = 0
    unique_points: int = 0
    shared_points: int = 0
    contradictions: int = 0


@dataclass
class NotebookStats:
    points: list[PointStat]
    sources: dict[int, SourceStat]
    section_sources: list[list[tuple[int, int]]] = field(default_factory=list)


@dataclass(frozen=True)
class Coverage:
    points: int
    single_source: int
    sections: list[tuple[int, int, int]]


def notebook_stats(doc: MergedDocument, claim_counts: dict[str, int]) -> NotebookStats:
    sid_of = doc.source_map
    sources = {sid: SourceStat(sid, tok, claims=claim_counts.get(tok, 0)) for tok, sid in sid_of.items()}
    disputed_keys: set[tuple[str, int]] = set()
    for sec in doc.sections:
        for c in sec.contradictions:
            disputed_keys.update((c.source_a, i) for i in c.claims_a)
            disputed_keys.update((c.source_b, i) for i in c.claims_b)
            for tok in {c.source_a, c.source_b}:
                if tok in sid_of:
                    sources[sid_of[tok]].contradictions += 1
    cited: dict[int, set[int]] = {sid: set() for sid in sources}
    points: list[PointStat] = []
    section_sources: list[list[tuple[int, int]]] = []
    for i, sec in enumerate(doc.sections):
        per_section: dict[int, int] = {}
        for j, sub in enumerate(sec.subsections):
            for p in sub.points:
                ids: list[int] = []
                for ref in p.claim_refs:
                    sid = sid_of.get(ref.source)
                    if sid is None:
                        continue
                    cited[sid].add(ref.claim)
                    if sid not in ids:
                        ids.append(sid)
                disputed = any((ref.source, ref.claim) in disputed_keys for ref in p.claim_refs)
                points.append(PointStat(i, j, p, tuple(ids), disputed))
                for sid in ids:
                    s = sources[sid]
                    s.points += 1
                    if len(ids) == 1:
                        s.unique_points += 1
                    else:
                        s.shared_points += 1
                    per_section[sid] = per_section.get(sid, 0) + 1
        section_sources.append(sorted(per_section.items(), key=lambda kv: (-kv[1], kv[0])))
    for sid, claims in cited.items():
        sources[sid].claims_cited = len(claims)
    return NotebookStats(points, sources, section_sources)


def worth_watching(stats: NotebookStats) -> set[int]:
    ranked = sorted(stats.sources.values(), key=lambda s: -s.unique_points)
    if len(ranked) < 3:
        return set()
    cutoff = ranked[ceil(len(ranked) / 3) - 1].unique_points
    return {s.source_id for s in ranked
            if s.unique_points >= max(cutoff, WORTH_WATCHING_MIN_UNIQUE)}


def coverage(stats: NotebookStats) -> Coverage:
    single = sum(1 for p in stats.points if len(p.source_ids) == 1)
    sections = []
    for i, srcs in enumerate(stats.section_sources):
        pts = [p for p in stats.points if p.section == i]
        sections.append((len(srcs), len(pts), sum(1 for p in pts if len(p.source_ids) == 1)))
    return Coverage(len(stats.points), single, sections)
