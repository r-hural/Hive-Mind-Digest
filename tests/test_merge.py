import _thread
import json
import re
import threading
import time

import pytest
from pipeline import db
from pipeline.llm.base import LLMError, LLMFatalError, wait_interrupt
from pipeline.llm.client import BudgetExceeded
from pipeline.schemas import (Analysis, CitationCheck, MergedDocument, Outline, Retraction, SectionBody, Verification,
                              claims_hash, strip_code_filled)
from pipeline.stages.merge import _parallel_calls, run_merge, build_source_map, MergeError
from pipeline.prompts.merge import SOURCE_BLOCK_RE
from tests.helpers import all_supported, check_payload, retract_payload


def _analysis(n_claims=3):
    return {"summary": "s", "topics": ["t"], "key_claims": [
        {"claim": f"c{i}", "quote": f"q{i}", "confidence": "high",
         "anchor": {"char_start": i, "char_end": i + 5, "t_start": 10.0 * i, "score": 100}} for i in range(n_claims)]}


class FakeLLM:
    """Answers by stage; records prompts. Sections cite claim 0 of each of their sources."""
    def __init__(self, outline, verification, section_fn=None, check_fn=None, retract_fn=None, recheck_fn=None):
        self.outline, self.verification, self.section_fn = outline, verification, section_fn
        # check_fn(payload of the CHECK data block) -> raw check answer; default: all supported/real
        self.check_fn = check_fn
        # retract_fn(payload of the RETRACT data block) -> raw answer; default {"text": None} (nothing
        # deleted). recheck_fn(CHECK payload) answers merge:recheck; default: all supported.
        self.retract_fn, self.recheck_fn = retract_fn, recheck_fn
        self.calls = []

    def call_json(self, prompt, schema, *, model, stage, source_id=None, **kw):
        self.calls.append((stage, prompt))
        if schema is Outline:
            return Outline.model_validate(self.outline)
        if schema is SectionBody:
            tokens = self.outline_tokens_in(prompt)
            body = self.section_fn(tokens) if self.section_fn else {
                "subsections": [{"title": "Sub", "points": [
                    {"text": f"point from {t}", "sources": [t], "claim_refs": [{"source": t, "claim": 0}]} for t in tokens]}],
                "contradictions": []}
            return SectionBody.model_validate(body)
        if schema is Verification:
            return Verification.model_validate(self.verification)
        if schema is CitationCheck and stage == "merge:recheck":
            raw = self.recheck_fn(check_payload(prompt)) if self.recheck_fn else all_supported(prompt)
            return CitationCheck.model_validate(strip_code_filled(raw))
        if schema is CitationCheck:
            raw = self.check_fn(check_payload(prompt)) if self.check_fn else all_supported(prompt)
            return CitationCheck.model_validate(strip_code_filled(raw))
        if schema is Retraction:
            raw = self.retract_fn(retract_payload(prompt)) if self.retract_fn else {"text": None}
            return Retraction.model_validate(strip_code_filled(raw))
        raise AssertionError(schema)

    @staticmethod
    def outline_tokens_in(prompt):
        return SOURCE_BLOCK_RE.findall(prompt)


@pytest.fixture
def state(tmp_path):
    path = str(tmp_path / "s.sqlite"); db.init_db(path)
    nid = db.create_notebook(path, "T", "t")
    rid = db.create_run(path, nid, provider="p", budget_usd=0, stages="merge")
    return path, nid, rid


def _seed(path, nid, n, analyzed=True):
    ids = []
    for i in range(n):
        sid = db.add_source(path, nid, f"vid{i:08d}", "u", title=f"Title {i}", channel=f"Chan {i % 3}")
        if analyzed:
            db.update_analyze_status(path, sid, "done", analysis_json=json.dumps(_analysis()))
        ids.append(sid)
    return ids


def test_build_source_map_is_in_id_order_and_skips_failed(state):
    path, nid, _ = state
    ids = _seed(path, nid, 3)
    db.update_analyze_status(path, ids[1], "error", error="x")
    assert build_source_map(db.get_analyzed_sources(path, nid)) == {"S1": ids[0], "S2": ids[2]}


def test_merge_no_analyzed_raises(state):
    path, nid, rid = state
    _seed(path, nid, 2, analyzed=False)
    with pytest.raises(MergeError):
        run_merge(path, nid, "T", FakeLLM({}, {}), "m", run_id=rid, report_path="r.md", log=lambda m: None)


def test_merge_three_steps_and_stored_document(state):
    path, nid, rid = state
    ids = _seed(path, nid, 4)
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}, {"title": "B", "sources": ["S3", "S4", "S9"]}]},
                  {"tldr": ["one", "two"], "duplicates": [["P1", "P3"]]})
    mid = run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=2, report_path="r.md", log=lambda m: None)
    stages = [s for s, _ in llm.calls]
    assert stages.count("merge:outline") == 1 and stages.count("merge:section") == 2 and stages.count("merge:verify") == 1
    # each section prompt holds only its own sources
    sec_prompts = [p for s, p in llm.calls if s == "merge:section"]
    assert sorted(len(FakeLLM.outline_tokens_in(p)) for p in sec_prompts) == [2, 2]     # S9 unknown, dropped
    row = db.get_latest_merge(path, nid)
    doc = MergedDocument.model_validate_json(row["merged_json"])
    assert doc.tldr == ["one", "two"] and doc.source_map == {"S1": ids[0], "S2": ids[1], "S3": ids[2], "S4": ids[3]}
    assert doc.claim_counts == {"S1": 3, "S2": 3, "S3": 3, "S4": 3}
    expected_hash = claims_hash(Analysis.model_validate(_analysis()))
    assert doc.claim_hashes == {t: expected_hash for t in ("S1", "S2", "S3", "S4")}
    assert json.loads(row["source_map_json"]) == {"S1": ids[0], "S2": ids[1], "S3": ids[2], "S4": ids[3]}
    assert doc.prompt_versions["merge"].startswith("merge-") and row["report_path"] == "r.md"
    assert set(doc.prompt_versions) == {"analyze", "merge", "check", "retract", "render"}
    # P1 and P3 merged: P1 keeps both sources, P3 gone
    pts = [p for s in doc.sections for ss in s.subsections for p in ss.points]
    assert [p.id for p in pts] == ["P1", "P2", "P4"]
    assert pts[0].sources == ["S1", "S3"] and len(pts[0].claim_refs) == 2
    # the fold added refs checked against P3's text, so P1 must not render "confirmed" (review 2026-10-18)
    assert pts[0].refs_from_fold is True and pts[1].refs_from_fold is False


def test_points_without_valid_claim_refs_are_dropped(state):
    path, nid, rid = state
    _seed(path, nid, 2)
    def section(tokens):
        return {"subsections": [{"title": "s", "points": [
            {"text": "ok", "sources": ["S1"], "claim_refs": [{"source": "S1", "claim": 0}]},
            {"text": "bad index", "sources": ["S1"], "claim_refs": [{"source": "S1", "claim": 99}]},
            {"text": "unknown source", "sources": ["S7"], "claim_refs": [{"source": "S7", "claim": 0}]},
            {"text": "mixed", "sources": ["S1", "S7"], "claim_refs": [{"source": "S7", "claim": 0}, {"source": "S1", "claim": 1}]},
        ]}], "contradictions": [
            {"topic": "t", "position_a": "a", "source_a": "S1", "position_b": "b", "source_b": "S2"},
            {"topic": "t", "position_a": "a", "source_a": "S1", "position_b": "b", "source_b": "S8"}]}
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}]}, {"tldr": ["x"], "duplicates": []}, section)
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, report_path="r.md", log=msgs.append)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    pts = doc.sections[0].subsections[0].points
    assert [p.text for p in pts] == ["ok", "mixed"]
    assert pts[1].sources == ["S1"] and [(r.source, r.claim) for r in pts[1].claim_refs] == [("S1", 1)]
    assert len(doc.sections[0].contradictions) == 1
    assert any("dropped" in m for m in msgs)


def test_forty_sources_never_put_everything_in_one_call(state):
    path, nid, rid = state
    _seed(path, nid, 40)
    sections = [{"title": f"Sec {k}", "sources": [f"S{i}" for i in range(1, 41) if i % 8 == k]} for k in range(8)]
    llm = FakeLLM({"sections": sections}, {"tldr": ["x"], "duplicates": []})
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=4, report_path="r.md", log=lambda m: None)
    outline_prompt = next(p for s, p in llm.calls if s == "merge:outline")
    assert '"quote"' not in outline_prompt                       # digests only
    for s, p in llm.calls:
        if s == "merge:section":
            assert len(FakeLLM.outline_tokens_in(p)) == 5      # 40 / 8
    verify_prompt = next(p for s, p in llm.calls if s == "merge:verify")
    assert verify_prompt.count("\nP") == 40 and "<<<HMD_DATA SOURCE" not in verify_prompt


def test_large_outline_section_is_split_into_chunks(state):
    path, nid, rid = state
    _seed(path, nid, 25)
    llm = FakeLLM({"sections": [{"title": "Big", "sources": [f"S{i}" for i in range(1, 26)]}]},
                  {"tldr": ["x"], "duplicates": []})
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=4, report_path="r.md", log=lambda m: None)
    sec_prompts = [p for s, p in llm.calls if s == "merge:section"]
    assert len(sec_prompts) == 3
    sizes = [len(SOURCE_BLOCK_RE.findall(p)) for p in sec_prompts]
    assert sizes == [9, 8, 8]                                        # balanced, in order
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    assert [s.title for s in doc.sections] == ["Big", "Big (2)", "Big (3)"]


def test_orphan_sources_get_other_findings_section(state):
    path, nid, rid = state
    _seed(path, nid, 3)
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}]}, {"tldr": ["x"], "duplicates": []})
    run_merge(path, nid, "T", llm, "m", run_id=rid, report_path="r.md", log=lambda m: None)
    sec_prompts = [p for s, p in llm.calls if s == "merge:section"]
    assert len(sec_prompts) == 2
    assert SOURCE_BLOCK_RE.findall(sec_prompts[1]) == ["S3"]
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    assert doc.sections[-1].title == "Other findings"


def test_section_llm_error_propagates_and_no_merge_row_created(state):
    path, nid, rid = state
    _seed(path, nid, 2)

    class FailingLLM(FakeLLM):
        def call_json(self, prompt, schema, *, model, stage, source_id=None, **kw):
            if schema is SectionBody:
                raise LLMError("boom")
            return super().call_json(prompt, schema, model=model, stage=stage, source_id=source_id, **kw)

    llm = FailingLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}]}, {"tldr": ["x"], "duplicates": []})
    with pytest.raises(LLMError):
        run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=lambda m: None)
    assert db.get_latest_merge(path, nid) is None


def test_duplicate_groups_edge_cases(state):
    path, nid, rid = state
    _seed(path, nid, 3)
    def section(tokens):
        return {"subsections": [{"title": "s", "points": [
            {"text": f"point from {t}", "sources": [t], "claim_refs": [{"source": t, "claim": 0}]} for t in tokens]}],
            "contradictions": []}
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2", "S3"]}]},
                  {"tldr": ["x"], "duplicates": [["P9"], ["P1", "P99"], ["P2", "P3"], ["P1", "P2"]]}, section)
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, report_path="r.md", log=msgs.append)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    pts = [p for s in doc.sections for ss in s.subsections for p in ss.points]
    assert [p.id for p in pts] == ["P1", "P2"]
    assert pts[1].sources == ["S2", "S3"]
    assert any("dropped" in m and "duplicate group" in m for m in msgs)


def test_duplicate_group_with_repeated_id_does_not_self_fold(state):
    path, nid, rid = state
    _seed(path, nid, 1)
    def section(tokens):
        return {"subsections": [{"title": "s", "points": [
            {"text": "only", "sources": ["S1"], "claim_refs": [{"source": "S1", "claim": 0}]}]}],
            "contradictions": []}
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1"]}]},
                  {"tldr": ["x"], "duplicates": [["P1", "P1"]]}, section)
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, report_path="r.md", log=msgs.append)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    pts = [p for s in doc.sections for ss in s.subsections for p in ss.points]
    assert [p.id for p in pts] == ["P1"]


def test_contradiction_with_same_source_both_sides_is_dropped(state):
    path, nid, rid = state
    _seed(path, nid, 2)
    def section(tokens):
        return {"subsections": [{"title": "s", "points": [
            {"text": "ok", "sources": ["S1"], "claim_refs": [{"source": "S1", "claim": 0}]}]}],
            "contradictions": [
                {"topic": "t", "position_a": "a", "source_a": "S1", "position_b": "b", "source_b": "S1"},
                {"topic": "t", "position_a": "a", "source_a": "S1", "position_b": "b", "source_b": "S2"}]}
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}]}, {"tldr": ["x"], "duplicates": []}, section)
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, report_path="r.md", log=msgs.append)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    assert len(doc.sections[0].contradictions) == 1
    assert any("dropped 1 contradiction" in m for m in msgs)


def test_section_with_no_surviving_points_is_dropped(state):
    path, nid, rid = state
    _seed(path, nid, 3)
    def section(tokens):
        if tokens == ["S1"]:
            return {"subsections": [{"title": "s", "points": [
                {"text": "bad", "sources": ["S1"], "claim_refs": [{"source": "S1", "claim": 99}]}]}],
                "contradictions": []}
        return {"subsections": [{"title": "s", "points": [
            {"text": f"point from {t}", "sources": [t], "claim_refs": [{"source": t, "claim": 0}]} for t in tokens]}],
            "contradictions": []}
    llm = FakeLLM({"sections": [{"title": "Empty", "sources": ["S1"]}, {"title": "Good", "sources": ["S2", "S3"]}]},
                  {"tldr": ["x"], "duplicates": []}, section)
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, report_path="r.md", log=msgs.append)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    assert [s.title for s in doc.sections] == ["Good"]
    assert any("dropped section" in m for m in msgs)


def test_claim_refs_and_contradictions_validated_against_section_not_notebook(state):
    """A chunked/section call for S1,S2 must not be trusted if it cites S3 or S4, even though
    those tokens exist elsewhere in the notebook (they were never sent in this section's prompt)."""
    path, nid, rid = state
    _seed(path, nid, 4)
    def section(tokens):
        if tokens == ["S1", "S2"]:
            return {"subsections": [{"title": "s", "points": [
                {"text": "cross-section citation", "sources": ["S1", "S3"],
                 "claim_refs": [{"source": "S3", "claim": 0}]},
                {"text": "valid", "sources": ["S1"], "claim_refs": [{"source": "S1", "claim": 0}]},
            ]}], "contradictions": [
                {"topic": "t", "position_a": "a", "source_a": "S1", "position_b": "b", "source_b": "S3"}]}
        return {"subsections": [{"title": "s", "points": [
            {"text": f"point from {t}", "sources": [t], "claim_refs": [{"source": t, "claim": 0}]} for t in tokens]}],
            "contradictions": []}
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}, {"title": "B", "sources": ["S3", "S4"]}]},
                  {"tldr": ["x"], "duplicates": []}, section)
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, report_path="r.md", log=msgs.append)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    sec_a = next(s for s in doc.sections if s.title == "A")
    texts = [p.text for ss in sec_a.subsections for p in ss.points]
    assert texts == ["valid"]
    assert sec_a.contradictions == []
    assert any("dropped" in m for m in msgs)


def test_point_sources_are_derived_purely_from_claim_refs(state):
    path, nid, rid = state
    _seed(path, nid, 2)
    def section(tokens):
        return {"subsections": [{"title": "s", "points": [
            {"text": "p", "sources": ["S1", "S2"], "claim_refs": [{"source": "S1", "claim": 0}]}]}],
            "contradictions": []}
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}]}, {"tldr": ["x"], "duplicates": []}, section)
    run_merge(path, nid, "T", llm, "m", run_id=rid, report_path="r.md", log=lambda m: None)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    pts = [p for s in doc.sections for ss in s.subsections for p in ss.points]
    assert pts[0].sources == ["S1"]


def test_budget_exceeded_in_one_of_many_sections_propagates_no_merge_row(state):
    path, nid, rid = state
    _seed(path, nid, 6)

    class FailingLLM(FakeLLM):
        def call_json(self, prompt, schema, *, model, stage, source_id=None, **kw):
            if schema is SectionBody and "S4" in self.outline_tokens_in(prompt):
                raise BudgetExceeded(1.0, 1.0)
            return super().call_json(prompt, schema, model=model, stage=stage, source_id=source_id, **kw)

    sections = [{"title": f"Sec {i}", "sources": [f"S{i}"]} for i in range(1, 7)]
    llm = FailingLLM({"sections": sections}, {"tldr": ["x"], "duplicates": []})
    with pytest.raises(BudgetExceeded):
        run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=4, report_path="r.md", log=lambda m: None)
    assert db.get_latest_merge(path, nid) is None


class _FatalOn(FakeLLM):
    """Raises LLMFatalError on the n-th call (1-based) of `fail_schema`, recording every call."""
    def __init__(self, fail_schema, n, *a, **kw):
        super().__init__(*a, **kw)
        self.fail_schema, self.n, self.seen = fail_schema, n, 0

    def call_json(self, prompt, schema, *, model, stage, source_id=None, **kw):
        if schema is self.fail_schema:
            self.seen += 1
            if self.seen >= self.n:
                self.calls.append((stage, prompt))
                raise LLMFatalError("API 401 authentication_error: invalid x-api-key")
        return super().call_json(prompt, schema, model=model, stage=stage, source_id=source_id, **kw)


def test_fatal_error_in_a_section_call_stops_queued_section_calls(state):
    path, nid, rid = state
    _seed(path, nid, 6)
    sections = [{"title": f"Sec {i}", "sources": [f"S{i}"]} for i in range(1, 7)]
    llm = _FatalOn(SectionBody, 2, {"sections": sections}, {"tldr": ["x"], "duplicates": []})
    with pytest.raises(LLMFatalError):
        run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=lambda m: None)
    stages = [s for s, _ in llm.calls]
    assert stages == ["merge:outline", "merge:section", "merge:section"]
    assert db.get_latest_merge(path, nid) is None


def test_fatal_error_in_a_check_call_stops_queued_check_calls(state):
    path, nid, rid = state
    _seed(path, nid, 6)
    sections = [{"title": f"Sec {i}", "sources": [f"S{i}"]} for i in range(1, 7)]
    llm = _FatalOn(CitationCheck, 1, {"sections": sections}, {"tldr": ["x"], "duplicates": []})
    with pytest.raises(LLMFatalError):
        run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=lambda m: None)
    stages = [s for s, _ in llm.calls]
    assert stages.count("merge:section") == 6 and stages.count("merge:check") == 1
    assert "merge:verify" not in stages and db.get_latest_merge(path, nid) is None


def test_fatal_error_wins_over_an_earlier_ordinary_section_error(state):
    path, nid, rid = state
    _seed(path, nid, 3)

    class Mixed(FakeLLM):
        def call_json(self, prompt, schema, *, model, stage, source_id=None, **kw):
            if schema is SectionBody:
                toks = self.outline_tokens_in(prompt)
                if "S1" in toks:
                    raise LLMError("schema broke")
                if "S2" in toks:
                    raise LLMFatalError("credit balance is too low")
            return super().call_json(prompt, schema, model=model, stage=stage, source_id=source_id, **kw)

    sections = [{"title": f"Sec {i}", "sources": [f"S{i}"]} for i in range(1, 4)]
    llm = Mixed({"sections": sections}, {"tldr": ["x"], "duplicates": []})
    with pytest.raises(LLMFatalError):
        run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=lambda m: None)


def test_unreadable_analysis_json_raises_merge_error(state):
    path, nid, rid = state
    ids = _seed(path, nid, 2)
    db.update_analyze_status(path, ids[0], "done", analysis_json="{}")
    with pytest.raises(MergeError):
        run_merge(path, nid, "T", FakeLLM({}, {}), "m", run_id=rid, report_path="r.md", log=lambda m: None)


def _sequential_sections(*bodies):
    """section_fn answering the n-th section call with bodies[n] (run with max_parallel=1)."""
    calls = iter(bodies)
    return lambda tokens: next(calls)


def _ok_points(*tokens):
    return [{"title": "s", "points": [
        {"text": f"point from {t}", "sources": [t], "claim_refs": [{"source": t, "claim": 0}]} for t in tokens]}]


def test_contradictions_are_deduplicated_across_sections_keeping_the_first(state):
    # Two outline sections both given S1 and S2 report the same disagreement; code folds it to
    # the first occurrence. Matching ignores side order, case, punctuation, whitespace and the
    # topic label (sections word it differently); a different position is a different contradiction.
    path, nid, rid = state
    _seed(path, nid, 2)
    first = {"topic": "Timeline", "position_a": "Five years.", "source_a": "S1",
             "position_b": "twenty years", "source_b": "S2"}
    swapped = {"topic": "When it ships", "position_a": "Twenty  Years", "source_a": "S2",
               "position_b": "five years", "source_b": "S1"}
    other = {"topic": "Cost", "position_a": "cheap", "source_a": "S1", "position_b": "expensive", "source_b": "S2"}
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}, {"title": "B", "sources": ["S1", "S2"]}]},
                  {"tldr": ["x"], "duplicates": []},
                  _sequential_sections({"subsections": _ok_points("S1", "S2"), "contradictions": [first, first]},
                                       {"subsections": _ok_points("S1", "S2"), "contradictions": [swapped, other]}))
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=msgs.append)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    by_title = {s.title: s for s in doc.sections}
    assert [c.topic for c in by_title["A"].contradictions] == ["Timeline"]
    assert [c.topic for c in by_title["B"].contradictions] == ["Cost"]
    assert any("folded 2 duplicate contradiction" in m for m in msgs)


def test_contradictions_between_different_sources_are_not_folded(state):
    path, nid, rid = state
    _seed(path, nid, 3)
    c12 = {"topic": "t", "position_a": "a", "source_a": "S1", "position_b": "b", "source_b": "S2"}
    c13 = {"topic": "t", "position_a": "a", "source_a": "S1", "position_b": "b", "source_b": "S3"}
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2", "S3"]}]}, {"tldr": ["x"], "duplicates": []},
                  _sequential_sections({"subsections": _ok_points("S1"), "contradictions": [c12, c13]}))
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=lambda m: None)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    assert len(doc.sections[0].contradictions) == 2


def test_section_left_empty_by_contradiction_dedup_is_dropped(state):
    path, nid, rid = state
    _seed(path, nid, 2)
    c = {"topic": "t", "position_a": "a", "source_a": "S1", "position_b": "b", "source_b": "S2"}
    bad_points = [{"title": "s", "points": [
        {"text": "bad", "sources": ["S1"], "claim_refs": [{"source": "S1", "claim": 99}]}]}]
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}, {"title": "B", "sources": ["S1", "S2"]}]},
                  {"tldr": ["x"], "duplicates": []},
                  _sequential_sections({"subsections": _ok_points("S1"), "contradictions": [c]},
                                       {"subsections": bad_points, "contradictions": [c]}))
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=lambda m: None)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    assert [s.title for s in doc.sections] == ["A"]


def test_contradictions_only_merge_is_a_valid_document_that_renders(state, tmp_path):
    # Every point fails claim-ref validation but a contradiction survives: that is a real
    # finding, stored as a contradictions-only document with no TL;DR (the verification call
    # only ever sees points, so it is skipped), and it renders end to end.
    from pipeline.stages.output import report_path_for, run_output
    path, nid, rid = state
    _seed(path, nid, 2)
    c = {"topic": "Timeline", "position_a": "soon", "source_a": "S1", "position_b": "never", "source_b": "S2"}
    bad_points = [{"title": "s", "points": [
        {"text": "bad", "sources": ["S1"], "claim_refs": [{"source": "S1", "claim": 99}]}]}]
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}]}, {"tldr": ["x"], "duplicates": []},
                  _sequential_sections({"subsections": bad_points, "contradictions": [c]}))
    vault = str(tmp_path / "vault")
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path=report_path_for(vault, "t"),
              log=msgs.append)
    assert "merge:verify" not in [s for s, _ in llm.calls]
    assert any("contradictions only" in m for m in msgs)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    assert doc.tldr == [] and len(doc.sections) == 1
    assert doc.sections[0].subsections == [] and len(doc.sections[0].contradictions) == 1
    out = run_output(path, nid, vault, log=lambda m: None)
    with open(out, encoding="utf-8") as f:
        md = f.read()
    assert "## 1. A" in md and "**Contradiction — Timeline:**" in md and "## Contradictions" in md
    tldr_block = md.split("## TL;DR")[1].split("## 1.")[0]
    assert "no point survived" in tldr_block and "- " not in tldr_block


def test_merge_with_neither_points_nor_contradictions_still_raises(state):
    path, nid, rid = state
    _seed(path, nid, 1)
    bad_points = [{"title": "s", "points": [
        {"text": "bad", "sources": ["S1"], "claim_refs": [{"source": "S1", "claim": 99}]}]}]
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1"]}]}, {"tldr": ["x"], "duplicates": []},
                  _sequential_sections({"subsections": bad_points, "contradictions": []}))
    with pytest.raises(MergeError, match="no point or contradiction survived"):
        run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=lambda m: None)
    assert db.get_latest_merge(path, nid) is None


def test_upload_dates_reach_every_merge_prompt_inside_data_blocks(state):
    # The merge must see when each source was published, so a change over time is not read as a
    # contradiction. Dates are normalised to YYYY-MM-DD ("unknown" when missing) and travel
    # inside the data blocks (metadata is untrusted data), never in the instruction region.
    path, nid, rid = state
    ids = _seed(path, nid, 2)
    db.update_source_metadata(path, ids[0], upload_date="20220315")
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}]}, {"tldr": ["x"], "duplicates": []})
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=lambda m: None)
    prompts = dict(llm.calls)
    for stage in ("merge:outline", "merge:section"):
        p = prompts[stage]
        assert '"upload_date": "2022-03-15"' in p and '"upload_date": "unknown"' in p
        block = re.search(r"<<<HMD_DATA SOURCE_S1 ([0-9a-f]{8})>>>(.*?)<<<END_HMD_DATA SOURCE_S1 \1>>>", p, re.S)
        assert block and "2022-03-15" in block.group(2)
    dates = re.search(r"<<<HMD_DATA UPLOAD_DATES ([0-9a-f]{8})>>>\n(.*?)\n<<<END_HMD_DATA UPLOAD_DATES \1>>>",
                      prompts["merge:verify"], re.S)
    assert dates and json.loads(dates.group(2)) == {"S1": "2022-03-15", "S2": "unknown"}


def _contra(topic, a=None, sa="S1", b=None, sb="S2"):
    # Positions default to topic-specific text: identical positions for the same pair are folded
    # by the exact contradiction key whatever the topic says.
    return {"topic": topic, "position_a": a or f"{topic}: yes", "source_a": sa,
            "position_b": b or f"{topic}: no", "source_b": sb}


def _run_with_contradictions(state, first, second, n_sources=2):
    path, nid, rid = state
    _seed(path, nid, n_sources)
    toks = [f"S{i}" for i in range(1, n_sources + 1)]
    llm = FakeLLM({"sections": [{"title": "A", "sources": toks}, {"title": "B", "sources": toks}]},
                  {"tldr": ["x"], "duplicates": []},
                  _sequential_sections({"subsections": _ok_points(*toks), "contradictions": first},
                                       {"subsections": _ok_points(*toks), "contradictions": second}))
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=msgs.append)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    return {s.title: [c.topic for c in s.contradictions] for s in doc.sections}, msgs


def test_near_duplicate_contradiction_across_sections_is_folded_keeping_the_first(state):
    # Same unordered source pair, reworded topic and positions: the later copy is folded.
    first = [_contra("Is Daily Readiness behind the paywall", "it needs Premium", "S1", "it is free", "S2")]
    second = [_contra("Daily Readiness paywall status", "Free for everyone", "S2", "Premium only", "S1")]
    by_title, msgs = _run_with_contradictions(state, first, second)
    assert by_title == {"A": ["Is Daily Readiness behind the paywall"], "B": []}
    assert any("folded 1 duplicate contradiction" in m for m in msgs)


def test_distinct_contradictions_for_the_same_pair_are_kept(state):
    # Different subject, a changed number, a different edition, or a different source pair: all kept.
    first = [_contra("Battery life"), _contra("Zoom quality at 10x"), _contra("Find X9 Pro low light")]
    second = [_contra("Battery charging time"), _contra("Zoom quality at 30x"),
              _contra("Find X9 Ultra low light"), _contra("Battery life", sb="S3")]
    by_title, _ = _run_with_contradictions(state, first, second, n_sources=3)
    assert by_title["A"] == ["Battery life", "Zoom quality at 10x", "Find X9 Pro low light"]
    assert by_title["B"] == ["Battery charging time", "Zoom quality at 30x", "Find X9 Ultra low light", "Battery life"]


def test_topic_fold_threshold_separates_rewordings_from_different_subjects():
    # Pins TOPIC_FOLD_THRESHOLD against the examples quoted next to it in pipeline/stages/merge.py.
    from pipeline.schemas import Contradiction
    from pipeline.stages.merge import TOPIC_FOLD_THRESHOLD, _same_disagreement

    def c(topic):
        return Contradiction(topic=topic, position_a="x", source_a="S1", position_b="y", source_b="S2")
    assert TOPIC_FOLD_THRESHOLD == 80.0
    same = [  # reworded copies of one disagreement, from the golden step-1 merges (82.1-100)
            ("Fitbit Air battery life rating", "Fitbit Air battery life figure"),
            ("Fitbit account migration deadline", "Fitbit account migration deadline date"),
            ("Timeline to habit automaticity", "How long it takes for a habit to reach automaticity"),
            ("Magnitude of compounding from 1% daily improvement", "Magnitude of 1% daily compounding over one year"),
            ("Is Daily Readiness behind the paywall", "Daily Readiness paywall status"),
            ("Heart rate accuracy during workouts", "Workout heart rate accuracy")]
    different = [("Garmin subscription requirement for core features", "Garmin Connect subscription model value"),
                 ("Samsung Galaxy S26 Ultra ultrawide low-light performance versus iPhone",
                  "Samsung ultra-wide sharpness advantage over iPhone"),
                 ("Battery life", "Battery charging time"),
                 ("Low-light photo quality", "Low light video quality"),
                 ("Photo quality in low light", "Video quality in low light"),
                 ("Heart rate accuracy", "Sleep tracking accuracy"),
                 ("Zoom quality at 10x", "Zoom quality at 30x"),
                 ("Battery life", "Battery life of the Pixel Watch 3"),
                 ("Pixel 10 thermals", "Pixel 11 thermals"),
                 ("Find X9 Pro camera", "Find X9 Pro Max camera")]
    assert all(_same_disagreement(c(a), c(b)) for a, b in same)
    assert not any(_same_disagreement(c(a), c(b)) for a, b in different)


def test_contradiction_whose_positions_are_the_same_is_dropped(state):
    # Two sources that say the same thing agree; that is never a contradiction.
    first = [_contra("Price", "$299.", "S1", "$299", "S2"), _contra("Weight", "30 g", "S1", "45 g", "S2")]
    by_title, msgs = _run_with_contradictions(state, first, [])
    assert by_title["A"] == ["Weight"]
    assert any("positions are the same" in m for m in msgs)


# --- citation check (merge:check, IDEA-295) ----------------------------------------------------

def _verdicts(point_map=None, ref_map=None, contra_map=None, default_point="supported", default_ref="related",
              default_contra="real", reason=None):
    """check_fn answering in the per-point shape: point_map {point id: verdict}, ref_map {point id:
    [relevance per ref]}, contra_map {topic: (verdict, reason)}; anything not listed gets the
    defaults (None = omit that verdict)."""
    def fn(payload):
        points, contras = [], []
        for p in payload["points"]:
            listed = (ref_map or {}).get(p["point"])
            refs = []
            for i, r in enumerate(p["refs"]):
                v = listed[i] if listed else default_ref
                refs.append({"ref": r["ref"]} if v is None else {"ref": r["ref"], "relevance": v})
            entry = {"point": p["point"], "refs": refs}
            v = (point_map or {}).get(p["point"], default_point)
            if v is not None:
                entry["verdict"] = v
            points.append(entry)
        for c in payload["contradictions"]:
            v, why = (contra_map or {}).get(c["topic"], (default_contra, reason))
            if v is not None:
                contras.append({"contradiction": c["contradiction"], "verdict": v, "reason": why})
        return {"points": points, "contradictions": contras}
    return fn


def _two_ref_points():
    # P1 cites S1 c0 + S2 c1; P2 cites S1 c1 + S2 c0; P3 cites S2 c2
    return {"subsections": [{"title": "Sub", "points": [
        {"text": "point one", "sources": ["S1", "S2"],
         "claim_refs": [{"source": "S1", "claim": 0}, {"source": "S2", "claim": 1}]},
        {"text": "point two", "sources": ["S1", "S2"],
         "claim_refs": [{"source": "S1", "claim": 1}, {"source": "S2", "claim": 0}]},
        {"text": "point three", "sources": ["S2"], "claim_refs": [{"source": "S2", "claim": 2}]},
    ]}], "contradictions": [
        _contra("Battery life", sa="S1", sb="S2") | {"claims_a": [0], "claims_b": [2, 99]},
        _contra("Price of the base model", sa="S1", sb="S2"),
    ]}


def _run_check(state, check_fn, body=None, verification=None):
    path, nid, rid = state
    _seed(path, nid, 2)
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}]},
                  verification or {"tldr": ["x"], "duplicates": []},
                  lambda tokens: body or _two_ref_points(), check_fn)
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=msgs.append)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    pts = {p.id: p for s in doc.sections for ss in s.subsections for p in ss.points}
    return doc, pts, llm, msgs


def _summary(msgs):
    return next(m for m in msgs if "ref(s) removed" in m)


def test_check_prompt_carries_stored_quotes_titles_dates_and_contradiction_quotes(state):
    _, _, llm, _ = _run_check(state, None)
    prompt = next(p for s, p in llm.calls if s == "merge:check")
    payload = check_payload(prompt)
    assert payload["section"] == "A"
    p1 = payload["points"][0]
    assert p1["point"] == "P1" and p1["text"] == "point one"
    assert [(r["ref"], r["source_title"], r["quote"], r["upload_date"]) for r in p1["refs"]] == [
        ("R1", "Title 0", "q0", "unknown"), ("R2", "Title 1", "q1", "unknown")]
    assert [r["ref"] for p in payload["points"] for r in p["refs"]] == ["R1", "R2", "R3", "R4", "R5"]
    c1 = payload["contradictions"][0]
    assert c1["contradiction"] == "C1" and c1["topic"] == "Battery life"
    assert c1["side_a"]["quotes"] == ["q0"] and c1["side_b"]["quotes"] == ["q2"]   # index 99 dropped
    assert payload["contradictions"][1]["side_a"]["quotes"] == []
    # the claim text (a model paraphrase) is not shown as evidence, only the stored quote
    assert '"c0"' not in prompt


def test_check_supported_point_combining_several_quotes_stays_unmarked(state):
    # Round-3 calibration: one verdict per point over all its quotes; a point built from two quotes
    # that together support it is "supported", not "partial" twice.
    doc, pts, _, msgs = _run_check(state, _verdicts())
    assert all(p.support is None for p in pts.values())
    assert [(r.source, r.claim) for r in pts["P1"].claim_refs] == [("S1", 0), ("S2", 1)]
    assert "0 ref(s) removed, 0 point(s) dropped, 0 point(s) partly supported" in _summary(msgs)


def test_check_removes_unrelated_refs_and_recomputes_sources(state):
    doc, pts, _, msgs = _run_check(state, _verdicts(ref_map={"P1": ["related", "unrelated"]}))
    p1 = pts["P1"]
    assert [(r.source, r.claim) for r in p1.claim_refs] == [("S1", 0)] and p1.sources == ["S1"]
    assert p1.support is None and p1.text == "point one"
    assert any("removed P1's ref S2 claim 1 (unrelated)" in m for m in msgs)
    assert "1 ref(s) removed" in _summary(msgs)


def test_check_drops_point_with_only_unrelated_refs(state):
    doc, pts, _, msgs = _run_check(state, _verdicts(ref_map={"P2": ["unrelated", "unrelated"]}))
    assert "P2" not in pts and set(pts) == {"P1", "P3"}
    assert any("dropped P2" in m and "no related quote" in m and "point two" in m for m in msgs)


def test_check_drops_point_judged_unsupported(state):
    doc, pts, _, msgs = _run_check(state, _verdicts(point_map={"P3": "unsupported"}))
    assert set(pts) == {"P1", "P2"}
    assert any("dropped P3" in m and "unsupported" in m and "point three" in m for m in msgs)
    assert "1 point(s) dropped" in _summary(msgs)


def test_check_keeps_partial_point_and_marks_it(state):
    doc, pts, _, msgs = _run_check(state, _verdicts(point_map={"P2": "partial", "P3": "partial"},
                                                    ref_map={"P2": ["related", "unrelated"]}))
    assert pts["P2"].support == "partial" and pts["P2"].sources == ["S1"]
    assert pts["P3"].support == "partial"
    assert pts["P1"].support is None and pts["P1"].sources == ["S1", "S2"]
    assert "2 point(s) partly supported" in _summary(msgs)
    assert any("P2 is only partly supported" in m for m in msgs)


def test_check_drops_contradictions_judged_not_real_and_logs_reasons(state):
    doc, _, _, msgs = _run_check(state, _verdicts(contra_map={"Battery life": ("not_a_contradiction", "absence")}))
    assert [c.topic for s in doc.sections for c in s.contradictions] == ["Price of the base model"]
    assert "1 contradiction(s) dropped (by reason: absence 1)" in _summary(msgs)


def test_check_label_on_real_contradictions_is_stored(state):
    # IDEA-013 check-side label: code copies the check's label onto the kept contradiction
    def fn(payload):
        out = _verdicts()(payload)
        for c, raw in zip(out["contradictions"], payload["contradictions"]):
            if raw["topic"] == "Battery life":
                c["label"] = "may_differ_by_edition_or_date"
        return out
    doc, _, _, msgs = _run_check(state, fn)
    labels = {c.topic: c.check_label for s in doc.sections for c in s.contradictions}
    assert labels == {"Battery life": "may_differ_by_edition_or_date", "Price of the base model": None}
    assert any("1 contradiction(s) may differ by edition or date" in m for m in msgs)


def test_check_not_a_contradiction_without_reason_counts_as_other(state):
    doc, _, _, msgs = _run_check(state, _verdicts(default_contra="not_a_contradiction"))
    assert not [c for s in doc.sections for c in s.contradictions]
    assert any("(by reason: other 2)" in m for m in msgs)


def test_check_drops_section_left_empty(state):
    path, nid, rid = state
    _seed(path, nid, 2)
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1"]}, {"title": "B", "sources": ["S2"]}]},
                  {"tldr": ["x"], "duplicates": []},
                  check_fn=lambda payload: _verdicts(default_point="unsupported")(payload)
                  if payload["section"] == "B" else _verdicts()(payload))
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=msgs.append)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    assert [s.title for s in doc.sections] == ["A"]
    assert any("dropped section 'B'" in m for m in msgs)
    assert "1 section(s) dropped" in _summary(msgs)


def test_check_dropping_everything_is_a_merge_error_and_stores_nothing(state):
    path, nid, rid = state
    with pytest.raises(MergeError, match="citation check"):
        _run_check(state, _verdicts(default_point="unsupported", default_contra="not_a_contradiction"))
    assert db.get_latest_merge(path, nid) is None


def test_check_missing_verdicts_fail_closed_and_are_logged(state):
    # Security review 2026-10-09, adapted to the per-point shape: a point without a verdict is
    # marked partial (never treated as supported); a ref without a relevance verdict is kept; a
    # contradiction without a verdict is dropped.
    doc, pts, _, msgs = _run_check(state, _verdicts(point_map={"P1": None}, ref_map={"P2": ["related", None]},
                                                    contra_map={"Battery life": (None, None)}))
    assert pts["P1"].support == "partial" and len(pts["P1"].claim_refs) == 2
    assert pts["P2"].support is None and len(pts["P2"].claim_refs) == 2   # unjudged ref kept
    assert [c.topic for s in doc.sections for c in s.contradictions] == ["Price of the base model"]
    line = next(m for m in msgs if "no verdict for" in m)
    assert "no verdict for 1 point(s), 1 ref(s) and 1 contradiction(s)" in line
    assert "points marked partly supported" in line and "refs kept" in line and "contradictions dropped" in line
    assert "1 point(s) partly supported" in _summary(msgs)


def test_point_left_without_a_verdict_and_without_refs_answered_is_marked_partial(state):
    doc, pts, _, _ = _run_check(state, _verdicts(point_map={"P3": None}, ref_map={"P3": [None]}))
    assert pts["P3"].support == "partial" and len(pts["P3"].claim_refs) == 1


def test_check_answering_under_half_of_a_section_is_a_merge_error(state):
    path, nid, rid = state
    # 3 points + 2 contradictions = 5 items; answer only P1 and P2 (2 of 5)
    def fn(payload):
        return {"points": [{"point": "P1", "verdict": "supported"}, {"point": "P2", "verdict": "supported"}],
                "contradictions": []}
    with pytest.raises(MergeError, match="citation check answered only 2 of 5"):
        _run_check(state, fn)
    assert db.get_latest_merge(path, nid) is None


def test_ref_relevance_alone_does_not_count_as_coverage(state):
    # Items are points (one verdict each) and contradictions; ref relevance verdicts never count,
    # so an answer that judges every ref but no point or contradiction is a MergeError.
    def fn(payload):
        return {"points": [{"point": p["point"], "refs": [{"ref": r["ref"], "relevance": "related"}
                                                          for r in p["refs"]]} for p in payload["points"]],
                "contradictions": []}
    with pytest.raises(MergeError, match="citation check answered only 0 of 5"):
        _run_check(state, fn)


def test_check_unknown_ids_are_ignored_and_first_verdict_wins(state):
    def fn(payload):
        ans = _verdicts()(payload)
        ans["points"] = ([{"point": "P9", "verdict": "unsupported",
                           "refs": [{"ref": "R1", "relevance": "unrelated"}, {"ref": "R99", "relevance": "unrelated"}]},
                          {"point": "P2", "verdict": "partial", "refs": []}]
                         + ans["points"])
        ans["contradictions"].append({"contradiction": "C9", "verdict": "not_a_contradiction"})
        return ans
    doc, pts, _, msgs = _run_check(state, fn)
    # R1's first relevance verdict (unrelated, given under the unknown P9) wins: ref ids are unique
    # per call, so a ref is matched by its id wherever it is listed.
    assert [(r.source, r.claim) for r in pts["P1"].claim_refs] == [("S2", 1)]
    assert pts["P2"].support == "partial" and pts["P1"].support is None
    assert len([c for s in doc.sections for c in s.contradictions]) == 2
    assert any("ignored 3 verdict(s) for unknown ids" in m for m in msgs)


def test_check_never_edits_text(state):
    def fn(payload):
        ans = _verdicts()(payload)
        for p in ans["points"]:
            p["text"] = "REWRITTEN"
            p["support"] = "partial"
            for r in p["refs"]:
                r["quote"] = r["text"] = "REWRITTEN"
        ans["support"] = "partial"
        for c in ans["contradictions"]:
            c["topic"] = c["position_a"] = "REWRITTEN"
        return ans
    doc, pts, _, _ = _run_check(state, fn)
    assert "REWRITTEN" not in doc.model_dump_json()
    assert [p.text for p in pts.values()] == ["point one", "point two", "point three"]
    assert all(p.support is None for p in pts.values())


def test_check_support_flag_from_section_call_is_never_trusted(state):
    body = _two_ref_points()
    body["subsections"][0]["points"][0]["support"] = "partial"
    _, pts, _, _ = _run_check(state, None, body=body)
    assert pts["P1"].support is None


def test_check_runs_once_per_section_before_verification(state):
    path, nid, rid = state
    _seed(path, nid, 4)
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}, {"title": "B", "sources": ["S3", "S4"]}]},
                  {"tldr": ["x"], "duplicates": []})
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=2, report_path="r.md", log=lambda m: None)
    stages = [s for s, _ in llm.calls]
    assert stages == ["merge:outline", "merge:section", "merge:section", "merge:check", "merge:check",
                      "merge:verify"]


def _four_source_llm(check_fn, duplicates):
    # Section A: P1 (S1), P2 (S2). Section B: P3 citing S3 claim 0 and S4 claim 0.
    def section(tokens):
        if tokens == ["S1", "S2"]:
            return {"subsections": _ok_points("S1", "S2"), "contradictions": []}
        return {"subsections": [{"title": "s", "points": [
            {"text": "point from S3 and S4", "sources": ["S3", "S4"],
             "claim_refs": [{"source": "S3", "claim": 0}, {"source": "S4", "claim": 0}]}]}], "contradictions": []}
    return FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}, {"title": "B", "sources": ["S3", "S4"]}]},
                   {"tldr": ["x"], "duplicates": duplicates}, section, check_fn)


def test_folding_uses_only_refs_that_survived_the_check_and_is_logged(state):
    path, nid, rid = state
    _seed(path, nid, 4)
    llm = _four_source_llm(_verdicts(ref_map={"P3": ["related", "unrelated"]}), [["P1", "P3"]])
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=msgs.append)
    fold_line = "  merge: fold group: kept P1 [S1], folded P3 [S3] -> P1 cites [S1, S3]"
    assert fold_line in msgs
    assert msgs.index(fold_line) > max(i for i, m in enumerate(msgs) if m.startswith("  merge:check:"))
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    p1 = doc.sections[0].subsections[0].points[0]
    assert p1.id == "P1" and p1.sources == ["S1", "S3"]   # S4, removed by the check, is never folded in
    assert [(r.source, r.claim) for r in p1.claim_refs] == [("S1", 0), ("S3", 0)]


def test_point_dropped_by_the_check_never_reaches_the_verification_listing(state):
    path, nid, rid = state
    _seed(path, nid, 4)
    llm = _four_source_llm(_verdicts(point_map={"P2": "unsupported", "P3": "partial"},
                                     ref_map={"P3": ["related", "unrelated"]}), [["P1", "P2"]])
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=msgs.append)
    verify_prompt = next(p for s, p in llm.calls if s == "merge:verify")
    listing = re.search(r"<<<HMD_DATA POINTS ([0-9a-f]{8})>>>\n(.*?)\n<<<END_HMD_DATA POINTS \1>>>",
                        verify_prompt, re.S).group(2)
    # The TL;DR is written from the checked points only: P2 is gone, P3 lists only its kept source.
    assert "P2" not in listing and "point from S2" not in listing
    assert "P1 [S1] point from S1" in listing and "P3 [S3] point from S3 and S4" in listing
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    pts = {p.id: p for s in doc.sections for ss in s.subsections for p in ss.points}
    assert set(pts) == {"P1", "P3"} and pts["P1"].sources == ["S1"]   # group with dropped P2 ignored
    assert pts["P3"].support == "partial" and doc.tldr == ["x"]


def test_check_leaving_only_contradictions_skips_verification(state):
    path, nid, rid = state
    _seed(path, nid, 2)
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}]}, {"tldr": ["x"], "duplicates": []},
                  lambda tokens: _two_ref_points(), _verdicts(default_ref="unrelated"))
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=1, report_path="r.md", log=msgs.append)
    assert "merge:verify" not in [s for s, _ in llm.calls]
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    assert doc.tldr == [] and not doc.sections[0].subsections and len(doc.sections[0].contradictions) == 2
    assert any("verification call skipped" in m for m in msgs)


def test_check_error_propagates_and_no_merge_row(state):
    path, nid, rid = state
    _seed(path, nid, 2)

    class Failing(FakeLLM):
        def call_json(self, prompt, schema, *, model, stage, source_id=None, **kw):
            if stage == "merge:check":
                raise LLMError("boom")
            return super().call_json(prompt, schema, model=model, stage=stage, source_id=source_id, **kw)
    llm = Failing({"sections": [{"title": "A", "sources": ["S1", "S2"]}]}, {"tldr": ["x"], "duplicates": []})
    with pytest.raises(LLMError):
        run_merge(path, nid, "T", llm, "m", run_id=rid, report_path="r.md", log=lambda m: None)
    assert db.get_latest_merge(path, nid) is None


def test_contradiction_claim_indices_are_validated_against_their_own_source(state):
    doc, _, _, msgs = _run_check(state, None)
    c = next(c for s in doc.sections for c in s.contradictions if c.topic == "Battery life")
    assert c.claims_a == [0] and c.claims_b == [2]
    assert any("dropped 1 contradiction claim index(es)" in m for m in msgs)


def test_check_usage_events_are_recorded_under_merge_check(state):
    from pipeline.llm.client import LLMClient
    from tests.helpers import ScriptedProvider
    path, nid, rid = state
    _seed(path, nid, 3)
    client = LLMClient(ScriptedProvider(cost_per_call=0.01), db_path=path, run_id=rid, notebook_id=nid,
                       log=lambda m: None)
    run_merge(path, nid, "T", client, "m", run_id=rid, report_path="r.md", log=lambda m: None)
    stages = [e["stage"] for e in db.notebook_usage_events(path, nid, rid)]
    assert stages.count("merge:check") == 2 and stages.count("merge:section") == 2


def test_check_schema_repair_path(state):
    """A malformed check answer gets the client's one corrective retry, like every merge call."""
    from pipeline.llm.client import LLMClient
    from tests.helpers import ScriptedProvider
    path, nid, rid = state
    _seed(path, nid, 1)

    class BadOnce(ScriptedProvider):
        def _answer(self, prompt):
            if "You are checking the citations" in prompt and "PREVIOUS_ERROR" not in prompt:
                return {"points": [{"point": "P1", "verdict": "maybe"}]}
            return super()._answer(prompt)
    provider = BadOnce()
    client = LLMClient(provider, db_path=path, run_id=rid, notebook_id=nid, log=lambda m: None)
    run_merge(path, nid, "T", client, "m", run_id=rid, report_path="r.md", log=lambda m: None)
    stages = [e["stage"] for e in db.notebook_usage_events(path, nid, rid)]
    assert stages.count("merge:check") == 2   # the invalid answer and the repair
    assert any("PREVIOUS_ERROR" in p for p in provider.prompts)


def test_parallel_calls_interrupt_releases_blocked_calls():
    released = []

    def fn(i):
        if wait_interrupt(10):
            released.append(i)
        return i

    threading.Timer(0.3, _thread.interrupt_main).start()
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _parallel_calls(fn, [0, 1], 2)
    assert time.monotonic() - t0 < 5
    assert sorted(released) == [0, 1]


# --- retraction of partly supported points (merge:retract, IDEA-297) ------------------------------

def _partial_body(text="Battery lasts 2 days and charges in 30 minutes"):
    return {"subsections": [{"title": "Sub", "points": [
        {"text": text, "sources": ["S1", "S2"],
         "claim_refs": [{"source": "S1", "claim": 0}, {"source": "S2", "claim": 1}]},
        {"text": "point two", "sources": ["S2"], "claim_refs": [{"source": "S2", "claim": 0}]},
    ]}], "contradictions": []}


def _run_retract(state, retract_fn, recheck_fn=None, check_fn=None, body=None):
    path, nid, rid = state
    _seed(path, nid, 2)
    llm = FakeLLM({"sections": [{"title": "A", "sources": ["S1", "S2"]}]}, {"tldr": ["x"], "duplicates": []},
                  lambda tokens: body or _partial_body(),
                  check_fn or _verdicts(point_map={"P1": "partial"}), retract_fn, recheck_fn)
    msgs = []
    run_merge(path, nid, "T", llm, "m", run_id=rid, max_parallel=2, report_path="r.md", log=msgs.append)
    doc = MergedDocument.model_validate_json(db.get_latest_merge(path, nid)["merged_json"])
    pts = {p.id: p for s in doc.sections for ss in s.subsections for p in ss.points}
    return doc, pts, llm, msgs


def test_retraction_accepted_shortens_text_and_keeps_the_partial_badge(state):
    doc, pts, llm, msgs = _run_retract(state, lambda pl: {"text": "Battery lasts 2 days."})
    p1 = pts["P1"]
    assert p1.text == "Battery lasts 2 days." and p1.support == "partial"          # owner decision: badge kept
    assert p1.retracted_from == "Battery lasts 2 days and charges in 30 minutes"
    assert pts["P2"].text == "point two" and pts["P2"].retracted_from is None
    stages = [s for s, _ in llm.calls]
    assert stages.count("merge:retract") == 1 and stages.count("merge:recheck") == 1
    assert stages.index("merge:recheck") < stages.index("merge:verify")            # TL;DR sees the short text
    assert doc.retraction == {"attempted": 1, "accepted": 1, "rejected": 0}
    from pipeline.prompts import retract as retract_prompts
    assert doc.prompt_versions["retract"] == retract_prompts.PROMPT_VERSION
    # the re-check judged the shortened text over the same refs
    recheck = check_payload(next(p for s, p in llm.calls if s == "merge:recheck"))
    assert [p["text"] for p in recheck["points"]] == ["Battery lasts 2 days."] and recheck["contradictions"] == []
    assert [r["quote"] for r in recheck["points"][0]["refs"]] == ["q0", "q1"]
    verify = next(p for s, p in llm.calls if s == "merge:verify")
    assert "P1 [S1, S2] Battery lasts 2 days." in verify
    assert any("merge:retract: P1 shortened" in m for m in msgs)
    assert any("merge:retract: 1 attempted, 1 accepted, 0 rejected" in m for m in msgs)


def test_retract_prompt_carries_point_and_its_quotes(state):
    seen = []
    _run_retract(state, lambda pl: seen.append(pl) or {"text": None})
    assert seen == [{"section": "A", "point": "Battery lasts 2 days and charges in 30 minutes", "quotes": [
        {"source_title": "Title 0", "upload_date": "unknown", "quote": "q0"},
        {"source_title": "Title 1", "upload_date": "unknown", "quote": "q1"}]}]


@pytest.mark.parametrize("answer", [
    {"text": None},
    {"text": "Battery lasts 3 days"},                       # a changed number
    {"text": "Battery easily lasts 2 days"},                # an added word
    {"text": "Lasts battery 2 days"},                       # reordered
    {"text": "Battery lasts 2 days and charges in 30 minutes"},   # nothing deleted
    {"text": "Battery"},                                    # too little left
])
def test_retraction_that_is_not_a_strict_deletion_is_rejected_without_recheck(state, answer):
    doc, pts, llm, msgs = _run_retract(state, lambda pl: answer)
    assert pts["P1"].text == "Battery lasts 2 days and charges in 30 minutes" and pts["P1"].retracted_from is None
    assert pts["P1"].support == "partial"
    assert "merge:recheck" not in [s for s, _ in llm.calls]
    assert doc.retraction == {"attempted": 1, "accepted": 0, "rejected": 1}


@pytest.mark.parametrize("recheck", [
    _verdicts(default_point="partial"),
    _verdicts(default_point="unsupported"),
    _verdicts(ref_map={"P1": ["related", "unrelated"]}),     # a quote no longer bears on the short text
    _verdicts(default_point=None),                           # no verdict: fails closed
])
def test_retraction_whose_recheck_fails_keeps_the_original(state, recheck):
    doc, pts, llm, msgs = _run_retract(state, lambda pl: {"text": "Battery lasts 2 days"}, recheck)
    assert pts["P1"].text == "Battery lasts 2 days and charges in 30 minutes" and pts["P1"].retracted_from is None
    assert [(r.source, r.claim) for r in pts["P1"].claim_refs] == [("S1", 0), ("S2", 1)]
    assert doc.retraction == {"attempted": 1, "accepted": 0, "rejected": 1}
    assert any("merge:retract: P1 kept" in m and "re-check" in m for m in msgs)


def test_retraction_skips_points_marked_partial_only_for_lack_of_a_verdict(state):
    doc, pts, llm, _ = _run_retract(state, lambda pl: {"text": "Battery lasts 2 days"},
                                    check_fn=_verdicts(point_map={"P1": None}))
    assert pts["P1"].support == "partial" and "merge:retract" not in [s for s, _ in llm.calls]
    assert doc.retraction == {"attempted": 0, "accepted": 0, "rejected": 0}


def test_retraction_schema_error_counts_as_rejected_and_the_merge_goes_on(state):
    from pipeline.llm.base import LLMSchemaError

    def bad(pl):
        raise LLMSchemaError("not json")
    doc, pts, _, msgs = _run_retract(state, bad)
    assert pts["P1"].text.startswith("Battery lasts 2 days and") and doc.retraction["rejected"] == 1


def test_retraction_fatal_error_stops_the_merge(state):
    path, nid, rid = state

    def fatal(pl):
        raise LLMFatalError("credit balance is too low")
    with pytest.raises(LLMFatalError):
        _run_retract(state, fatal)
    assert db.get_latest_merge(path, nid) is None


def test_old_merge_without_retraction_fields_still_loads():
    doc = MergedDocument.model_validate({"tldr": ["a"], "source_map": {"S1": 1}, "sections": [
        {"title": "A", "subsections": [{"title": "s", "points": [
            {"text": "t", "sources": ["S1"], "claim_refs": [{"source": "S1", "claim": 0}]}]}]}]})
    assert doc.retraction == {} and doc.sections[0].subsections[0].points[0].retracted_from is None


@pytest.mark.parametrize("original,candidate,ok", [
    ("Battery lasts 2 days and charges in 30 minutes", "Battery lasts 2 days.", True),
    ("the battery lasts 2 days, and it charges fast", "The battery lasts 2 days", True),   # first word capitalised
    ("It holds 6,000 mAh and weighs 200 g", "It holds 6 mAh", False),                    # a number cut
    ("It holds 6,000 mAh and weighs 200 g", "It holds 6,000 mAh", True),
    ("The update does not fix the bug in the camera app", "The update does fix the bug", False),  # lone "not"
    ("It lasts about 2 days on a charge with AOD on", "It lasts 2 days on a charge", False),     # lone hedge
    ("It lasts 2 days, but not with the always-on display", "It lasts 2 days", True),   # clause with "not"
    ("Battery lasts 2 days", "Battery lasts 2 days", False),
    ("Battery lasts 2 days", "", False),
    ("Battery lasts 2 days and more", "battery Lasts 2 days", False),                    # case changed mid-text
    # review 2026-10-18: only whole clauses may go
    ("Sales increased 5% in 2023 but fell 8% in 2024", "Sales increased 8% in 2024", False),   # splice
    ("Sales did not significantly rise in 2024, analysts said", "Sales rise in 2024, analysts said", False),
    ("Sales did not rise in 2024 and the margin fell", "Sales did rise in 2024", False),       # 3-word run with "not"
    ("The battery, as Tech Spurt measured, lasts 2 days", "The battery lasts 2 days", True),  # middle clause
    ("Battery lasts 2 days; it charges in 30 minutes", "Battery lasts 2 days", True),          # trailing clause
    ("In 2023, the battery lasted 2 days", "The battery lasted 2 days", False),                # first word kept
    ("It lasts 2 days unless the always-on display is on", "It lasts 2 days", False),          # a limiting clause
    ("It lasts 2 days, not 3 days, on one charge", "It lasts 2 days, on one charge", False),   # clause led by "not"
    ("It lasts 2 days—not 3—on one charge", "It lasts 2 days on one charge", False),  # dash join
    ("Battery life is long; battery lasts 2 days", "battery lasts 2 days", False),            # case bound to word 1
])
def test_is_strict_deletion(original, candidate, ok):
    from pipeline.stages.merge import _is_strict_deletion
    assert _is_strict_deletion(original, candidate) is ok
