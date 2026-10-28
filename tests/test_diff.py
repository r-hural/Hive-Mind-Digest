import hashlib
from pipeline.schemas import MergedDocument
from pipeline.diff import diff_merges

H = {n: hashlib.sha256(n.encode()).hexdigest() for n in "abcd"}


def _doc(source_map, hashes, points, contradictions=()):
    return MergedDocument.model_validate({
        "tldr": ["t"], "source_map": source_map, "claim_hashes": hashes,
        "sections": [{"title": "S", "subsections": [{"title": "s", "points": points}],
                      "contradictions": list(contradictions)}]})


def _p(pid, refs):
    return {"id": pid, "text": pid, "sources": sorted({r[0] for r in refs}),
            "claim_refs": [{"source": s, "claim": c} for s, c in refs]}


OLD = _doc({"S1": 1, "S2": 2}, {"S1": H["a"], "S2": H["b"]},
           [_p("P1", [("S1", 0)]), _p("P2", [("S2", 0)]), _p("P3", [("S1", 1), ("S2", 1)])])


def test_redundant_source_only_confirms_existing_points():
    new = _doc({"S1": 1, "S2": 2, "S3": 3}, {"S1": H["a"], "S2": H["b"], "S3": H["c"]},
               [_p("P1", [("S1", 0), ("S3", 0)]), _p("P2", [("S2", 0)]), _p("P3", [("S1", 1), ("S2", 1)])])
    d = diff_merges(OLD, new)
    assert d.new_points == [] and d.removed == []
    assert d.confirmed == {"P1": [3]}
    assert d.new_sources == [3] and d.silent_sources == [3]


def test_new_point_from_a_new_source():
    new = _doc({"S1": 1, "S2": 2, "S3": 3}, {"S1": H["a"], "S2": H["b"], "S3": H["c"]},
               [_p("P1", [("S1", 0)]), _p("P2", [("S2", 0)]), _p("P3", [("S1", 1), ("S2", 1)]), _p("P4", [("S3", 0)])])
    d = diff_merges(OLD, new)
    assert d.new_points == ["P4"] and d.from_new_sources == {"P4"} and d.silent_sources == []


def test_point_ids_are_renumbered_but_matched_by_claims():
    new = _doc({"S1": 1, "S2": 2}, {"S1": H["a"], "S2": H["b"]},
               [_p("P9", [("S2", 0)]), _p("P8", [("S1", 1), ("S2", 1)]), _p("P7", [("S1", 0)])])
    d = diff_merges(OLD, new)
    assert d.matched == {"P9": "P2", "P8": "P3", "P7": "P1"} and not d.new_points and not d.removed


def test_removed_point_is_reported():
    new = _doc({"S1": 1, "S2": 2}, {"S1": H["a"], "S2": H["b"]}, [_p("P1", [("S1", 0)]), _p("P2", [("S2", 0)])])
    assert [p.id for p in diff_merges(OLD, new).removed] == ["P3"]


def test_diff_skips_reanalyzed_sources():
    new = _doc({"S1": 1, "S2": 2}, {"S1": H["a"], "S2": H["d"]},          # S2 re-analyzed
               [_p("P1", [("S1", 0)]), _p("P5", [("S2", 3)])])
    d = diff_merges(OLD, new)
    assert d.not_compared == [2]
    assert [p.id for p in d.removed] == []                                # P2, P3 involve S2: not judged
    assert d.new_points == [] and d.not_compared_points == ["P5"]


def test_new_contradiction_detected_by_source_pair_and_claims():
    c_old = {"topic": "x", "position_a": "a", "source_a": "S1", "claims_a": [0], "position_b": "b", "source_b": "S2", "claims_b": [0]}
    old = _doc({"S1": 1, "S2": 2}, {"S1": H["a"], "S2": H["b"]}, [_p("P1", [("S1", 0)])], [c_old])
    c_same = dict(c_old, topic="reworded")
    c_new = {"topic": "y", "position_a": "a", "source_a": "S1", "claims_a": [2], "position_b": "b", "source_b": "S2", "claims_b": [2]}
    new = _doc({"S1": 1, "S2": 2}, {"S1": H["a"], "S2": H["b"]}, [_p("P1", [("S1", 0)])], [c_same, c_new])
    assert [c.topic for c in diff_merges(old, new).new_contradictions] == ["y"]


def test_split_point_is_restructured_not_new():
    old = _doc({"S1": 1}, {"S1": H["a"]}, [_p("P1", [("S1", 0), ("S1", 1)])])
    new = _doc({"S1": 1}, {"S1": H["a"]}, [_p("P1", [("S1", 0)]), _p("P2", [("S1", 1)])])
    d = diff_merges(old, new)
    assert d.new_points == [] and d.restructured == ["P2"] and d.removed == []


def test_merged_points_are_not_removed():
    old = _doc({"S1": 1}, {"S1": H["a"]}, [_p("P1", [("S1", 0)]), _p("P2", [("S1", 1)])])
    new = _doc({"S1": 1}, {"S1": H["a"]}, [_p("P1", [("S1", 0), ("S1", 1)])])
    d = diff_merges(old, new)
    assert d.removed == [] and d.new_points == []


def test_legacy_merge_without_hashes_compares_nothing():
    hashed = _doc({"S1": 1, "S2": 2}, {"S1": H["a"], "S2": H["b"]},
                  [_p("P1", [("S1", 0)]), _p("P2", [("S2", 0)])])
    legacy = _doc({"S1": 1, "S2": 2}, {}, [_p("P1", [("S1", 0)]), _p("P2", [("S2", 0)])])
    for old, new in ((legacy, hashed), (hashed, legacy)):
        d = diff_merges(old, new)
        assert d.new_points == [] and d.removed == [] and d.not_compared == [1, 2]


def test_dropped_source_point_is_removed():
    old = _doc({"S1": 1, "S2": 2}, {"S1": H["a"], "S2": H["b"]}, [_p("P1", [("S1", 0)]), _p("P2", [("S2", 0)])])
    new = _doc({"S1": 1}, {"S1": H["a"]}, [_p("P1", [("S1", 0)])])
    d = diff_merges(old, new)
    assert d.dropped_sources == [2] and [p.id for p in d.removed] == ["P2"]


def test_contradiction_on_reanalyzed_source_is_not_new():
    c = {"topic": "y", "position_a": "a", "source_a": "S1", "claims_a": [2], "position_b": "b", "source_b": "S2", "claims_b": [2]}
    new = _doc({"S1": 1, "S2": 2}, {"S1": H["a"], "S2": H["d"]}, [_p("P1", [("S1", 0)])], [c])
    assert diff_merges(OLD, new).new_contradictions == []


def test_tie_break_uses_old_document_order_not_id_string():
    old = _doc({"S1": 1}, {"S1": H["a"]}, [_p("P10", [("S1", 0)]), _p("P2", [("S1", 0)])])
    new = _doc({"S1": 1}, {"S1": H["a"]}, [_p("P1", [("S1", 0)])])
    assert diff_merges(old, new).matched == {"P1": "P10"}
