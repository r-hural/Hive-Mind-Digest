"""Proves tests/helpers.py's ScriptedProvider answers every real prompt shape correctly, using
the real prompt builders (pipeline/prompts/*) and real schemas (pipeline/schemas.py) -- no
mocking of our own code, only the LLM boundary itself."""
from tests.helpers import yt_ref
import json

import pytest

from pipeline.anchors import resolve_segment_claims
from pipeline.llm.base import LLMProvider
from pipeline.prompts.analyze import get_analyze_prompt
from pipeline.prompts.common import repair_prompt
from pipeline.prompts.merge import get_outline_prompt, get_section_prompt, get_verification_prompt, source_digest
from pipeline.schemas import Analysis, AnalysisDraft, ClaimRef, Outline, OutlineSection, Point, SectionBody, Verification
from pipeline.transcription.base import Transcriber
from tests.helpers import ScriptedProvider, StubTranscriber, fake_metadata


def _analysis(claim="c", quote="q"):
    return Analysis.model_validate({
        "summary": "s", "topics": ["t"],
        "key_claims": [{"claim": claim, "quote": quote, "confidence": "high"}],
    })


def test_analyze_prompt_keeps_the_claim_through_anchor_validation():
    transcript = StubTranscriber().transcribe(yt_ref("aaaaaaaaaaa"), ["en"])
    flat = transcript.flatten()
    prompt = get_analyze_prompt("Topic", "Title", "Channel", flat.segment_texts(), 1, 5)

    result = ScriptedProvider().complete(prompt, model="m")
    raw = json.loads(result.text)
    assert "quote" not in raw["key_claims"][0]
    assert (raw["key_claims"][0]["segment_from"], raw["key_claims"][0]["segment_to"]) == (1, 1)

    analysis = resolve_segment_claims(AnalysisDraft.model_validate(raw), flat)
    assert analysis.key_claims
    assert analysis.dropped_quotes == []
    assert "aaaaaaaaaaa" in analysis.key_claims[0].claim
    # The scripted range is StubTranscriber's second segment (t=5s) rather than char offset 0 --
    # a real, informative deep-link timestamp; the quote is that segment's text, filled by code.
    assert analysis.key_claims[0].anchor.t_start == 5.0
    assert analysis.key_claims[0].quote == flat.segment_texts()[1]


def test_scripted_range_extends_to_eight_words_and_falls_back_to_segment_0():
    from pipeline.transcription.base import Segment, flatten
    flat = flatten([Segment(0, 1, "only one segment here but long enough to quote")])
    raw = json.loads(ScriptedProvider().complete(
        get_analyze_prompt("T", "t", "c", flat.segment_texts(), 1, 5), model="m").text)
    assert (raw["key_claims"][0]["segment_from"], raw["key_claims"][0]["segment_to"]) == (0, 0)
    flat = flatten([Segment(0, 1, "zero"), Segment(1, 2, "one two"), Segment(2, 3, "three four five"),
                    Segment(3, 4, "six seven eight"), Segment(4, 5, "nine")])
    raw = json.loads(ScriptedProvider().complete(
        get_analyze_prompt("T", "t", "c", flat.segment_texts(), 1, 5), model="m").text)
    assert (raw["key_claims"][0]["segment_from"], raw["key_claims"][0]["segment_to"]) == (1, 3)


def test_outline_prompt_places_every_token_in_some_section():
    a1, a2 = _analysis("c1", "q1"), _analysis("c2", "q2")
    digests = [source_digest("S1", "t1", "c1", a1), source_digest("S2", "t2", "c2", a2)]
    prompt = get_outline_prompt("Topic", digests)

    result = ScriptedProvider().complete(prompt, model="m")
    outline = Outline.model_validate(json.loads(result.text))

    all_tokens = {tok for sec in outline.sections for tok in sec.sources}
    assert all_tokens == {"S1", "S2"}


def test_section_prompt_cites_only_the_tokens_it_was_given():
    a1, a2 = _analysis("c1", "q1"), _analysis("c2", "q2")
    analyses = {"S1": ("t1", "c1", a1), "S2": ("t2", "c2", a2)}
    section = OutlineSection(title="Sec", scope="scope", sources=["S2"])
    prompt = get_section_prompt("Topic", section, analyses)

    result = ScriptedProvider().complete(prompt, model="m")
    body = SectionBody.model_validate(json.loads(result.text))

    cited = {ref.source for sub in body.subsections for p in sub.points for ref in p.claim_refs}
    assert cited == {"S2"}
    for sub in body.subsections:
        for p in sub.points:
            for ref in p.claim_refs:
                assert ref.claim == 0


def test_verification_prompt_returns_three_tldr_items():
    point = Point(text="x", sources=["S1"], claim_refs=[ClaimRef(source="S1", claim=0)], id="P1")
    prompt = get_verification_prompt("Topic", [("Sec", "Sub", point)])

    result = ScriptedProvider().complete(prompt, model="m")
    verification = Verification.model_validate(json.loads(result.text))

    assert len(verification.tldr) == 3
    assert verification.duplicates == []


def test_unrecognised_prompt_raises_assertion_error():
    with pytest.raises(AssertionError, match=r"unrecognised prompt.*nothing recognisable here"):
        ScriptedProvider().complete("nothing recognisable here", model="m")


def test_cost_per_call_is_used_as_is():
    # Use a real, recognised prompt so complete() doesn't raise before returning the LLMResult.
    a = _analysis()
    prompt = get_outline_prompt("Topic", [source_digest("S1", "t", "c", a)])
    result = ScriptedProvider(cost_per_call=0.6).complete(prompt, model="m")
    assert result.cost_usd == 0.6


def test_fake_metadata_fills_source_row(tmp_path):
    from pipeline import db

    db_path = str(tmp_path / "s.sqlite")
    db.init_db(db_path)
    nid = db.create_notebook(db_path, "T", "t")
    sid = db.add_source(db_path, nid, "aaaaaaaaaaa", "u")

    status = fake_metadata(db_path, sid, log=lambda m: None)

    src = db.get_source(db_path, sid)
    assert status == "done"
    assert src["title"] == "Video aaaa"
    assert src["channel"] == "Channel"
    assert src["duration_seconds"] == 20
    assert src["upload_date"] == "20260101"


def test_fake_metadata_skips_when_already_done(tmp_path):
    from pipeline import db

    db_path = str(tmp_path / "s.sqlite")
    db.init_db(db_path)
    nid = db.create_notebook(db_path, "T", "t")
    sid = db.add_source(db_path, nid, "aaaaaaaaaaa", "u")
    fake_metadata(db_path, sid, log=lambda m: None)   # first call fills title/upload_date

    assert fake_metadata(db_path, sid, log=lambda m: None) == "skipped"


def test_fake_metadata_missing_source_raises(tmp_path):
    from pipeline import db

    db_path = str(tmp_path / "s.sqlite")
    db.init_db(db_path)

    with pytest.raises(ValueError, match="not found"):
        fake_metadata(db_path, 999, log=lambda m: None)


def test_complete_echoes_the_requested_model_not_the_constructor_default():
    a = _analysis()
    prompt = get_outline_prompt("Topic", [source_digest("S1", "t", "c", a)])
    result = ScriptedProvider(model="constructor-default").complete(prompt, model="claude-sonnet-4-6")
    assert result.model == "claude-sonnet-4-6"


def test_outline_prompt_is_deterministic():
    a1, a2 = _analysis("c1", "q1"), _analysis("c2", "q2")
    digests = [source_digest("S1", "t1", "c1", a1), source_digest("S2", "t2", "c2", a2)]
    prompt = get_outline_prompt("Topic", digests)
    p = ScriptedProvider()

    assert p.complete(prompt, model="m").text == p.complete(prompt, model="m").text


def test_single_source_outline_yields_one_section():
    a = _analysis()
    prompt = get_outline_prompt("Topic", [source_digest("S1", "t", "c", a)])
    result = ScriptedProvider().complete(prompt, model="m")
    outline = Outline.model_validate(json.loads(result.text))
    assert len(outline.sections) == 1
    assert outline.sections[0].sources == ["S1"]


def _repair_wrapped(prompt: str) -> str:
    """The exact repair-retry prompt LLMClient.call_json sends (pipeline/prompts/common.py), so the
    recogniser is proven to work on the actual wrapped prompt it will see in production."""
    return repair_prompt(prompt, "boom")


def test_repair_wrapped_analyze_prompt_still_parses():
    transcript = StubTranscriber().transcribe(yt_ref("aaaaaaaaaaa"), ["en"])
    flat = transcript.flatten()
    prompt = get_analyze_prompt("Topic", "Title", "Channel", flat.segment_texts(), 1, 5)
    p = ScriptedProvider()

    original = AnalysisDraft.model_validate(json.loads(p.complete(prompt, model="m").text))
    repaired = AnalysisDraft.model_validate(json.loads(p.complete(_repair_wrapped(prompt), model="m").text))

    assert original.key_claims and repaired.key_claims
    assert original.key_claims[0].claim == repaired.key_claims[0].claim


def test_repair_wrapped_section_prompt_still_parses():
    a1, a2 = _analysis("c1", "q1"), _analysis("c2", "q2")
    analyses = {"S1": ("t1", "c1", a1), "S2": ("t2", "c2", a2)}
    section = OutlineSection(title="Sec", scope="scope", sources=["S2"])
    prompt = get_section_prompt("Topic", section, analyses)
    p = ScriptedProvider()

    original = SectionBody.model_validate(json.loads(p.complete(prompt, model="m").text))
    repaired = SectionBody.model_validate(json.loads(p.complete(_repair_wrapped(prompt), model="m").text))

    for body in (original, repaired):
        cited = {ref.source for sub in body.subsections for pt in sub.points for ref in pt.claim_refs}
        assert cited == {"S2"}


def test_scripted_provider_and_stub_transcriber_satisfy_their_protocols():
    assert isinstance(ScriptedProvider(), LLMProvider)
    assert isinstance(StubTranscriber(), Transcriber)


def test_scripted_provider_answers_the_dated_prompt_shapes():
    # The merge prompts now carry upload dates (inside the source blocks and an UPLOAD_DATES block
    # in the verification prompt); ScriptedProvider must still recognise each shape and must not
    # mistake the UPLOAD_DATES block for a source.
    a1, a2 = _analysis("c1", "q1"), _analysis("c2", "q2")
    dates = {"S1": "20220101", "S2": "20250601"}
    provider = ScriptedProvider()
    digests = [source_digest("S1", "t1", "c1", a1, upload_date=dates["S1"]),
               source_digest("S2", "t2", "c2", a2, upload_date=dates["S2"])]
    outline = Outline.model_validate(json.loads(provider.complete(get_outline_prompt("Topic", digests), model="m").text))
    assert {t for s in outline.sections for t in s.sources} == {"S1", "S2"}
    section = OutlineSection(title="Sec", scope="scope", sources=["S1", "S2"])
    prompt = get_section_prompt("Topic", section, {"S1": ("t1", "c1", a1), "S2": ("t2", "c2", a2)}, dates)
    body = SectionBody.model_validate(json.loads(provider.complete(prompt, model="m").text))
    assert {r.source for sub in body.subsections for p in sub.points for r in p.claim_refs} == {"S1", "S2"}
    point = Point(text="x", sources=["S1"], claim_refs=[ClaimRef(source="S1", claim=0)], id="P1")
    v = Verification.model_validate(json.loads(
        provider.complete(get_verification_prompt("Topic", [("Sec", "Sub", point)], dates), model="m").text))
    assert len(v.tldr) == 3
