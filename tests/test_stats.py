# tests/test_stats.py
from pipeline.schemas import MergedDocument
from pipeline.stages.stats import notebook_stats, worth_watching, coverage

DOC = MergedDocument.model_validate({
    "tldr": ["t"], "source_map": {"S1": 11, "S2": 12, "S3": 13},
    "sections": [
        {"title": "A", "subsections": [{"title": "a", "points": [
            {"id": "P1", "text": "joint", "sources": ["S1", "S2"],
             "claim_refs": [{"source": "S1", "claim": 0}, {"source": "S2", "claim": 0}, {"source": "S1", "claim": 1}]},
            {"id": "P2", "text": "only one", "sources": ["S1"], "claim_refs": [{"source": "S1", "claim": 2}]}]}],
         "contradictions": [{"topic": "x", "position_a": "a", "source_a": "S1", "claims_a": [2],
                             "position_b": "b", "source_b": "S3", "claims_b": [0]}]},
        {"title": "B", "subsections": [{"title": "b", "points": [
            {"id": "P3", "text": "s2 alone", "sources": ["S2"], "claim_refs": [{"source": "S2", "claim": 1}]}]}]},
    ]})
COUNTS = {"S1": 4, "S2": 2, "S3": 3}


def test_point_sources_are_distinct_cited_sources_in_ref_order():
    st = notebook_stats(DOC, COUNTS)
    assert [p.source_ids for p in st.points] == [(11, 12), (11,), (12,)]
    assert [(p.section, p.subsection) for p in st.points] == [(0, 0), (0, 0), (1, 0)]


def test_point_is_disputed_when_a_cited_claim_is_a_contradiction_side():
    st = notebook_stats(DOC, COUNTS)
    assert [p.disputed for p in st.points] == [False, True, False]


def test_source_counts():
    s = notebook_stats(DOC, COUNTS).sources
    assert (s[11].claims, s[11].claims_cited, s[11].points, s[11].unique_points, s[11].shared_points, s[11].contradictions) == (4, 3, 2, 1, 1, 1)
    assert (s[12].claims_cited, s[12].unique_points, s[12].shared_points) == (2, 1, 1)
    assert (s[13].points, s[13].contradictions, s[13].claims_cited) == (0, 1, 0)


def test_section_sources_sorted_by_point_count():
    st = notebook_stats(DOC, COUNTS)
    assert st.section_sources == [[(11, 2), (12, 1)], [(12, 1)]]


def test_coverage_counts_single_source_points():
    cov = coverage(notebook_stats(DOC, COUNTS))
    assert (cov.points, cov.single_source) == (3, 2)
    assert cov.sections == [(2, 2, 1), (1, 1, 1)]


def test_stats_tolerate_points_without_ids():
    raw = DOC.model_dump()
    for sec in raw["sections"]:
        for sub in sec["subsections"]:
            for p in sub["points"]:
                p["id"] = None
    st = notebook_stats(MergedDocument.model_validate(raw), {})
    assert len(st.points) == 3 and st.sources[11].claims == 0


def test_worth_watching_needs_three_unique_points_and_top_third():
    raw = {"tldr": ["t"], "source_map": {"S1": 1, "S2": 2, "S3": 3}, "sections": [{"title": "A", "subsections": [{"title": "a", "points": []}]}]}
    pts = []
    for i in range(4):
        pts.append({"text": f"a{i}", "sources": ["S1"], "claim_refs": [{"source": "S1", "claim": i}]})
    for i in range(3):
        pts.append({"text": f"b{i}", "sources": ["S2"], "claim_refs": [{"source": "S2", "claim": i}]})
    pts.append({"text": "c", "sources": ["S3"], "claim_refs": [{"source": "S3", "claim": 0}]})
    raw["sections"][0]["subsections"][0]["points"] = pts
    st = notebook_stats(MergedDocument.model_validate(raw), {"S1": 5, "S2": 5, "S3": 5})
    assert worth_watching(st) == {1}
