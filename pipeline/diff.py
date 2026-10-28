"""Light diff (IDEA-012, DECISION-031): what changed between a notebook's two latest merges,
matched by the claims their points cite -- never by point text or section titles, which vary from
merge to merge. A source's claims are comparable only when its claims_hash is the same in both
merges (it was not re-analyzed in between); everything else is reported as not compared."""
from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.schemas import Contradiction, MergedDocument, Point

ClaimKey = tuple[int, int]


@dataclass
class DiffResult:
    new_points: list[str] = field(default_factory=list)
    from_new_sources: set[str] = field(default_factory=set)
    confirmed: dict[str, list[int]] = field(default_factory=dict)
    removed: list[Point] = field(default_factory=list)
    new_contradictions: list[Contradiction] = field(default_factory=list)
    new_sources: list[int] = field(default_factory=list)
    silent_sources: list[int] = field(default_factory=list)
    not_compared: list[int] = field(default_factory=list)
    matched: dict[str, str] = field(default_factory=dict)
    restructured: list[str] = field(default_factory=list)
    not_compared_points: list[str] = field(default_factory=list)
    dropped_sources: list[int] = field(default_factory=list)


def point_key(p: Point, index: int) -> str:
    return p.id or f"#{index}"


def _points(doc: MergedDocument) -> list[tuple[str, Point]]:
    flat = [p for sec in doc.sections for sub in sec.subsections for p in sub.points]
    return [(point_key(p, i), p) for i, p in enumerate(flat)]


def _keys(doc: MergedDocument, p: Point) -> set[ClaimKey]:
    return {(doc.source_map[r.source], r.claim) for r in p.claim_refs if r.source in doc.source_map}


def _sources(doc: MergedDocument, p: Point) -> set[int]:
    return {sid for sid, _ in _keys(doc, p)}


def _hash_by_sid(doc: MergedDocument) -> dict[int, str]:
    return {sid: doc.claim_hashes[tok] for tok, sid in doc.source_map.items() if tok in doc.claim_hashes}


def diff_merges(old: MergedDocument, new: MergedDocument) -> DiffResult:
    res = DiffResult()
    old_h, new_h = _hash_by_sid(old), _hash_by_sid(new)
    old_sids, new_sids = set(old.source_map.values()), set(new.source_map.values())
    comparable = {sid for sid in old_sids & new_sids if sid in old_h and old_h[sid] == new_h.get(sid)}
    res.new_sources = sorted(new_sids - old_sids)
    res.not_compared = sorted((old_sids & new_sids) - comparable)
    res.dropped_sources = sorted(old_sids - new_sids)
    dropped = set(res.dropped_sources)

    old_pts, new_pts = _points(old), _points(new)
    old_keys = {k: {x for x in _keys(old, p) if x[0] in comparable} for k, p in old_pts}
    new_keys = {k: {x for x in _keys(new, p) if x[0] in comparable} for k, p in new_pts}
    old_union = set().union(*old_keys.values()) if old_keys else set()
    new_union = set().union(*new_keys.values()) if new_keys else set()
    pairs = []
    for ni, (nk, _np) in enumerate(new_pts):
        for oi, (ok, _op) in enumerate(old_pts):
            overlap = len(new_keys[nk] & old_keys[ok])
            if overlap:
                pairs.append((-overlap, ni, oi, nk, ok))
    used_old, used_new = set(), set()
    for _, _, _, nk, ok in sorted(pairs):
        if nk in used_new or ok in used_old:
            continue
        used_new.add(nk)
        used_old.add(ok)
        res.matched[nk] = ok
    old_by_key = dict(old_pts)
    allowed_new = comparable | set(res.new_sources)
    for nk, np_ in new_pts:
        srcs = _sources(new, np_)
        if nk in res.matched:
            gained = sorted(srcs - _sources(old, old_by_key[res.matched[nk]]))
            if gained:
                res.confirmed[nk] = gained
        elif not srcs <= allowed_new:
            res.not_compared_points.append(nk)
        elif not _keys(new, np_) or any(x not in old_union for x in _keys(new, np_)):
            res.new_points.append(nk)
            if srcs and srcs <= set(res.new_sources):
                res.from_new_sources.add(nk)
        else:
            res.restructured.append(nk)
    for ok, op in old_pts:
        if ok in used_old or not _sources(old, op) <= comparable | dropped:
            continue
        if not _keys(old, op) or any(x not in new_union for x in _keys(old, op)):
            res.removed.append(op)
    in_new_points = {sid for nk, np_ in new_pts if nk in res.new_points for sid in _sources(new, np_)}
    res.silent_sources = [sid for sid in res.new_sources if sid not in in_new_points]
    res.new_contradictions = [c for c in _contradictions(new)
                              if _checkable(new, c, allowed_new) and not _seen(old, new, c)]
    return res


def _contradictions(doc: MergedDocument) -> list[Contradiction]:
    return [c for sec in doc.sections for c in sec.contradictions]


def _checkable(doc: MergedDocument, c: Contradiction, allowed: set[int]) -> bool:
    return all(doc.source_map.get(t) in allowed for t in (c.source_a, c.source_b))


def _side(doc: MergedDocument, c: Contradiction) -> tuple[frozenset, frozenset]:
    a, b = doc.source_map.get(c.source_a), doc.source_map.get(c.source_b)
    keys = {(a, i) for i in c.claims_a} | {(b, i) for i in c.claims_b}
    return frozenset({a, b}), frozenset(keys)


def _seen(old: MergedDocument, new: MergedDocument, c: Contradiction) -> bool:
    pair, keys = _side(new, c)
    for oc in _contradictions(old):
        opair, okeys = _side(old, oc)
        if opair == pair and ((not keys and not okeys) or keys & okeys):
            return True
    return False
