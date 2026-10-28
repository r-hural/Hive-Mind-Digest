"""Merge-quality gate numbers for one notebook, read from the state DB (never bills).

The numbers the merge-quality loop gates on: claims kept vs dropped
by the analyze stage, with drop reasons; points in the latest merge and how many the citation check
marked partly supported; contradictions kept; retractions of partly supported points attempted /
accepted / rejected (IDEA-297). Rows are read as plain JSON, not through the pydantic
schemas, so analyses and merges stored by older prompt versions still count. A row that cannot be
read (corrupt JSON, unexpected shape) is counted as unreadable, never raised: this report runs inside
`hmd status`, which must still print its REPORT= line.
"""
from __future__ import annotations

import json
from collections import Counter

from pipeline import db

_UNREADABLE = (ValueError, TypeError, AttributeError, KeyError)


def _analysis_counts(analysis_json: str) -> tuple[int, list[str], int]:
    a = json.loads(analysis_json)
    return (len(a.get("key_claims") or []), [str(d.get("reason", "unknown")) for d in a.get("dropped_quotes") or []],
            len(a.get("trimmed_quotes") or []))


def _merge_counts(merged_json: str) -> tuple[int, int, int, dict[str, int]]:
    doc = json.loads(merged_json)
    sections = doc.get("sections") or []
    points = [p for sec in sections for ss in sec.get("subsections") or [] for p in ss.get("points") or []]
    # retraction counts (IDEA-297); a merge stored before retraction existed has none
    retraction = doc.get("retraction") or {}
    retract = {k: int(retraction.get(k) or 0) for k in ("attempted", "accepted", "rejected")}
    return (len(points), sum(1 for p in points if p.get("support") == "partial"),
            sum(len(sec.get("contradictions") or []) for sec in sections), retract)


def notebook_quality(db_path: str, notebook_id: int) -> dict:
    kept = 0
    trimmed = 0
    unreadable = 0
    reasons: Counter[str] = Counter()
    for s in db.get_analyzed_sources(db_path, notebook_id):
        if not s.get("analysis_json"):
            continue
        try:
            n, drops, n_trimmed = _analysis_counts(s["analysis_json"])
        except _UNREADABLE:
            unreadable += 1
            continue
        kept += n
        trimmed += n_trimmed
        reasons.update(drops)
    q = {"claims_kept": kept, "claims_dropped": sum(reasons.values()), "claims_trimmed": trimmed,
         "drop_reasons": dict(reasons.most_common()),
         "merge_id": None, "points": 0, "partial": 0, "contradictions": 0,
         "retract_attempted": 0, "retract_accepted": 0, "retract_rejected": 0, "unreadable": unreadable}
    merge = db.get_latest_merge(db_path, notebook_id)
    if merge:
        try:
            points, partial, contradictions, retract = _merge_counts(merge["merged_json"])
        except _UNREADABLE:
            q["unreadable"] += 1
        else:
            q.update(merge_id=merge["id"], points=points, partial=partial, contradictions=contradictions,
                     **{f"retract_{k}": n for k, n in retract.items()})
    return q


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "0.0%"


def quality_report(db_path: str, notebook_id: int) -> str:
    """Indented lines for `hmd status <slug>`; empty when nothing is analyzed or merged yet."""
    q = notebook_quality(db_path, notebook_id)
    if q["merge_id"] is None and not q["unreadable"] and not db.get_analyzed_sources(db_path, notebook_id):
        return ""
    total = q["claims_kept"] + q["claims_dropped"]
    trimmed = f" ({q['claims_trimmed']} trimmed to fit)" if q["claims_trimmed"] else ""
    claims = (f"  claims: {q['claims_kept']} kept{trimmed}, {q['claims_dropped']} dropped "
              f"({_pct(q['claims_dropped'], total)})")
    if q["drop_reasons"]:
        # reasons are code-generated literals (pipeline/anchors.py); collapsed anyway so a stored
        # value can never start a line of its own
        claims += " -- " + ", ".join(f"{' '.join(r.split())} {n}" for r, n in q["drop_reasons"].items())
    lines = ["Quality (latest analyses and merge):", claims]
    if q["merge_id"] is not None:
        lines.append(f"  merge {q['merge_id']}: {q['points']} points, {q['partial']} partly supported "
                     f"({_pct(q['partial'], q['points'])}), {q['contradictions']} contradictions")
        if q["retract_attempted"]:
            lines.append(f"  retraction of partly supported points: {q['retract_attempted']} attempted, "
                         f"{q['retract_accepted']} accepted, {q['retract_rejected']} rejected")
    if q["unreadable"]:
        lines.append(f"  {q['unreadable']} unreadable rows (analysis_json or merged_json)")
    return "\n".join(lines)
