"""Segment-pointer quotes (IDEA-294, DECISION-027): the analyze model returns a segment range per
claim, code fills the quote and the anchor (pipeline/anchors.py::resolve_segment_claims)."""
from __future__ import annotations

import json

import pytest

from pipeline.anchors import (
    MAX_QUOTE_SEGMENTS, MAX_QUOTE_SPAN_WORDS, QuoteLocator, resolve_segment_claims, validate_analysis,
)
from pipeline.evals import quote_at_anchor
from pipeline.schemas import (
    CODE_FILLED_KEYS, Analysis, AnalysisDraft, claims_hash, strip_code_filled,
)
from pipeline.transcription.base import Segment, flatten

SEGS = [
    Segment(0.0, 3.0, "Welcome back to the channel everybody"),
    Segment(3.0, 7.5, "  today we compare   the two flagship phones  "),   # extra whitespace
    Segment(7.5, 9.0, ""),                                                    # empty: not numbered
    Segment(9.0, 12.0, "the ’Find X9 Pro’ has the best main camera"),
    Segment(12.0, 15.0, "I have tested this year by a clear margin"),
    Segment(15.0, 18.0, "ok"),
    Segment(18.0, 21.0, "ﬁnal thoughts: battery life is about 30% better"),
    Segment(21.0, 25.0, "这款手机的电池续航非常好"),
]
FLAT = flatten(SEGS)


def _draft(*claims: dict, **extra) -> AnalysisDraft:
    return AnalysisDraft.model_validate({"summary": "s", "topics": ["phones"], "key_claims": list(claims), **extra})


def _one(seg_from, seg_to, flat=FLAT):
    return resolve_segment_claims(_draft({"claim": "c", "segment_from": seg_from, "segment_to": seg_to}), flat)


def test_segment_texts_are_numbered_like_seg_starts():
    texts = FLAT.segment_texts()
    assert len(texts) == len(FLAT.seg_starts) == 7          # the empty segment is skipped
    assert texts[1] == "today we compare the two flagship phones"
    assert " ".join(texts) == FLAT.text
    for i, t in enumerate(texts):
        assert FLAT.text[FLAT.seg_starts[i]:FLAT.segment_end(i)] == t


def test_quote_anchor_and_t_start_are_filled_by_code_and_exact():
    out = _one(2, 3)
    assert out.quote_mode == "segments" and out.dropped_quotes == []
    c = out.key_claims[0]
    texts = FLAT.segment_texts()
    assert c.quote == texts[2] + " " + texts[3]
    assert c.quote == "the ’Find X9 Pro’ has the best main camera I have tested this year by a clear margin"
    assert c.anchor.char_start == FLAT.seg_starts[2]
    assert c.anchor.char_end == FLAT.seg_starts[3] + len(texts[3])
    assert FLAT.text[c.anchor.char_start:c.anchor.char_end] == c.quote
    assert c.anchor.t_start == 9.0 and c.anchor.score == 100.0
    assert c.claim == "c" and c.confidence == "medium"


def test_single_segment_range_and_whitespace_normalised_quote():
    c = _one(1, 1).key_claims[0]
    assert c.quote == "today we compare the two flagship phones"
    assert c.anchor.t_start == 3.0


def test_other_fields_and_confidence_carry_over():
    out = resolve_segment_claims(_draft(
        {"claim": "Best camera", "segment_from": 2, "segment_to": 3, "confidence": "high"},
        insights=["i"], facts=["f"], caveats=["c"], references=["r"]), FLAT)
    assert out.topics == ["phones"] and out.summary == "s"
    assert (out.insights, out.facts, out.caveats, out.references) == (["i"], ["f"], ["c"], ["r"])
    assert out.key_claims[0].confidence == "high" and out.key_claims[0].claim == "Best camera"


@pytest.mark.parametrize("seg_from,seg_to", [
    (None, 3), (2, None), ("2", 3), (2, "3"), (2.0, 3), (2, 3.5), (True, 3), (2, [3]),
])
def test_missing_or_non_integer_index_is_dropped(seg_from, seg_to):
    out = _one(seg_from, seg_to)
    assert out.key_claims == []
    assert out.dropped_quotes == [{"claim": "c", "segment_from": seg_from, "segment_to": seg_to,
                                   "reason": "segment index missing or not an integer"}]


def test_claim_without_range_keys_is_dropped():
    out = resolve_segment_claims(_draft({"claim": "no range"}), FLAT)
    assert out.dropped_quotes[0]["reason"] == "segment index missing or not an integer"


@pytest.mark.parametrize("seg_from,seg_to", [(-1, 2), (0, 7), (7, 7), (99, 100)])
def test_out_of_range_index_is_dropped(seg_from, seg_to):
    assert _one(seg_from, seg_to).dropped_quotes[0]["reason"] == "segment index out of range"


def test_reversed_range_is_dropped():
    assert _one(3, 2).dropped_quotes[0]["reason"] == "segment_from after segment_to"


def _long_flat(n_segments: int, words_per_segment: int):
    return flatten([Segment(float(i), float(i + 1), " ".join(f"w{i}x{j}" for j in range(words_per_segment)))
                    for i in range(n_segments)])


def test_span_over_max_segments_is_dropped_and_max_is_kept():
    flat = _long_flat(30, 2)
    ok = _one(0, MAX_QUOTE_SEGMENTS - 1, flat)
    assert len(ok.key_claims) == 1 and ok.dropped_quotes == []
    out = _one(0, MAX_QUOTE_SEGMENTS, flat)
    assert out.key_claims == []
    assert out.dropped_quotes[0]["reason"] == f"span over {MAX_QUOTE_SEGMENTS} segments"
    assert MAX_QUOTE_SEGMENTS == 24


def test_span_over_max_words_is_dropped_and_max_is_kept():
    assert MAX_QUOTE_SPAN_WORDS == 130
    flat = _long_flat(6, 26)                       # 26 words per segment
    assert len(_one(0, 4, flat).key_claims) == 1   # exactly 130 words
    out = _one(0, 5, flat)                         # 156 words
    assert out.key_claims == []
    d = out.dropped_quotes[0]
    assert d["reason"] == f"span over {MAX_QUOTE_SPAN_WORDS} words"
    assert d["quote"] == flat.text                 # the text is recorded for debugging


def test_span_below_the_floors_is_dropped():
    out = _one(4, 4)                               # "ok": under MIN_QUOTE_CHARS
    assert out.dropped_quotes[0]["reason"] == "too short" and out.dropped_quotes[0]["quote"] == "ok"
    flat = flatten([Segment(0, 1, "five plain words right here"), Segment(1, 2, "x")])
    assert len(_one(0, 0, flat).key_claims) == 1    # exactly MIN_QUOTE_WORDS is kept


def test_word_floor_and_mixed_script_rule_apply():
    flat = flatten([Segment(0, 1, "only four words here"),                       # 4 words, 20 chars
                    Segment(1, 2, "Claude Codeを使う"),                # 2 + ceil(3/2) = 4 words
                    Segment(2, 3, "Claude Codeを使うと開発が速い")])  # 2 + 5 = 7
    assert _one(0, 0, flat).dropped_quotes[0]["reason"] == "too short"
    assert _one(1, 1, flat).dropped_quotes[0]["reason"] == "too short"
    assert len(_one(2, 2, flat).key_claims) == 1
    # a pure-CJK segment clears the floors on its own
    assert len(_one(6, 6).key_claims) == 1


def test_mixed_kept_and_dropped_claims_keep_order():
    out = resolve_segment_claims(_draft(
        {"claim": "a", "segment_from": 0, "segment_to": 1},
        {"claim": "b", "segment_from": 9, "segment_to": 9},
        {"claim": "c", "segment_from": 5, "segment_to": 5}), FLAT)
    assert [c.claim for c in out.key_claims] == ["a", "c"]
    assert [d["claim"] for d in out.dropped_quotes] == ["b"]


def test_eval_assertion_5_passes_by_construction():
    # every valid range over a transcript with curly quotes, a ligature, punctuation and CJK
    n = len(FLAT.seg_starts)
    claims = [{"claim": f"{a}-{b}", "segment_from": a, "segment_to": b}
              for a in range(n) for b in range(a, min(n, a + MAX_QUOTE_SEGMENTS))]
    out = resolve_segment_claims(_draft(*claims), FLAT)
    assert out.key_claims and all(d["reason"] == "too short" for d in out.dropped_quotes)
    locator = QuoteLocator(FLAT)
    assert all(quote_at_anchor(locator, FLAT, c) for c in out.key_claims)


def test_model_supplied_quote_anchor_and_mode_never_survive():
    for key in ("quote", "anchor", "quote_mode", "dropped_quotes", "trimmed_quotes"):
        assert key in CODE_FILLED_KEYS
    raw = {"summary": "s", "quote_mode": "text", "dropped_quotes": [{"x": 1}],
           "key_claims": [{"claim": "c", "segment_from": 2, "segment_to": 3, "quote": "forged words",
                           "anchor": {"char_start": 0, "char_end": 3, "t_start": 0, "score": 100}}]}
    stripped = strip_code_filled(raw)
    assert "quote" not in stripped["key_claims"][0] and "anchor" not in stripped["key_claims"][0]
    assert "quote_mode" not in stripped and "dropped_quotes" not in stripped
    # even unstripped, the wire schema has no place for them and code fills its own
    out = resolve_segment_claims(AnalysisDraft.model_validate(raw), FLAT)
    assert out.key_claims[0].quote.startswith("the ’Find X9 Pro’")
    assert out.key_claims[0].anchor.char_start == FLAT.seg_starts[2]
    assert out.quote_mode == "segments" and out.dropped_quotes == []


# --- stored analyses -----------------------------------------------------------------------

# An analysis as the text-quote prompt (analyze-2026-10-09.1) stored it: no quote_mode, a
# salvaged_quotes record from the retired salvage path.
OLD_STORED = {
    "summary": "s", "topics": ["t"],
    "key_claims": [{"claim": "Recency matters", "quote": "weighs recency about thirty percent", "confidence": "high",
                    "anchor": {"char_start": 14, "char_end": 49, "t_start": 0.0, "score": 100.0}}],
    "dropped_quotes": [{"claim": "x", "reason": "no match", "quote": "q"}],
    "salvaged_quotes": [{"claim": "c", "quote": "a ... b", "kept": "a", "method": "split"}],
}
# claims_hash of OLD_STORED computed by the code before this change (d7450d5): merges stored
# before the segment-pointer prompt must keep matching their sources.
OLD_STORED_HASH = "dcc70cad32a2314b275540e72436d60d023e1324c41298a9fea87bc554a2bd9a"


def test_stored_text_mode_analysis_still_loads_with_an_unchanged_claims_hash():
    a = Analysis.model_validate_json(json.dumps(OLD_STORED))
    assert a.quote_mode == "text"
    assert a.key_claims[0].quote == "weighs recency about thirty percent"
    assert claims_hash(a) == OLD_STORED_HASH
    assert "salvaged_quotes" not in a.model_dump()


def test_stored_text_mode_analysis_still_validates_with_the_locator():
    flat = flatten([Segment(0, 4, "Acme's search weighs recency about thirty percent"), Segment(4, 8, "more.")])
    a = Analysis.model_validate(OLD_STORED)
    for c in a.key_claims:
        c.anchor = None
    out = validate_analysis(a, flat)
    assert out.quote_mode == "text" and len(out.key_claims) == 1 and out.dropped_quotes == []
    assert out.key_claims[0].anchor.char_start == 14
    assert quote_at_anchor(QuoteLocator(flat), flat, out.key_claims[0])


def test_quote_mode_is_not_part_of_claims_hash():
    seg = _one(2, 3)
    text = Analysis.model_validate({**seg.model_dump(), "quote_mode": "text"})
    assert claims_hash(seg) == claims_hash(text)


def test_segment_mode_analysis_round_trips_through_storage():
    out = _one(2, 3)
    back = Analysis.model_validate_json(out.model_dump_json())
    assert back.quote_mode == "segments" and back == out and claims_hash(back) == claims_hash(out)


# --- padding (round 5, 2026-10-14) -------------------------------------------------------------

def _padded(seg_from, seg_to, pad=1, flat=FLAT):
    draft = _draft({"claim": "c", "segment_from": seg_from, "segment_to": seg_to})
    return resolve_segment_claims(draft, flat, pad=pad)


def test_padding_adds_one_segment_on_each_side():
    texts = FLAT.segment_texts()
    c = _padded(2, 3).key_claims[0]
    assert c.quote == " ".join(texts[1:5])
    assert c.anchor.char_start == FLAT.seg_starts[1] and c.anchor.char_end == FLAT.segment_end(4)
    assert c.anchor.t_start == FLAT.seg_times[1]
    assert FLAT.text[c.anchor.char_start:c.anchor.char_end] == c.quote


def test_padding_clamps_at_transcript_edges():
    texts = FLAT.segment_texts()
    assert _padded(0, 1).key_claims[0].quote == " ".join(texts[0:3])
    assert _padded(5, 6).key_claims[0].quote == " ".join(texts[4:7])


def test_padding_never_exceeds_the_segment_limit():
    segs = [Segment(float(i), float(i + 1), f"word{i} filler text here") for i in range(40)]
    flat = flatten(segs)
    c = _padded(5, 5 + MAX_QUOTE_SEGMENTS - 2, flat=flat).key_claims[0]   # 15 segments
    assert c.anchor.char_end == flat.segment_end(5 + MAX_QUOTE_SEGMENTS - 1)   # end padded
    assert c.anchor.char_start == flat.seg_starts[5]                            # start not
    full = _padded(5, 5 + MAX_QUOTE_SEGMENTS - 1, flat=flat).key_claims[0]     # 16 already
    assert full.anchor.char_start == flat.seg_starts[5]
    assert full.anchor.char_end == flat.segment_end(5 + MAX_QUOTE_SEGMENTS - 1)


def test_padding_never_exceeds_the_word_limit():
    words = MAX_QUOTE_SPAN_WORDS - 2
    segs = [Segment(0.0, 1.0, "lead in words here"),
            Segment(1.0, 2.0, " ".join(["w"] * words)),
            Segment(2.0, 3.0, "tail out words here")]
    flat = flatten(segs)
    c = _padded(1, 1, flat=flat).key_claims[0]
    assert c.quote == " ".join(["w"] * words)          # either neighbour would pass 100 words


def test_padding_never_rescues_a_dropped_claim():
    out_short = _padded(4, 4)                            # "ok": too short on its own range
    assert out_short.key_claims == [] and out_short.dropped_quotes[0]["reason"] == "too short"
    assert _padded(3, 2).dropped_quotes[0]["reason"] == "segment_from after segment_to"


def test_pad_zero_is_the_default_and_keeps_exact_ranges():
    assert _padded(2, 3, pad=0).key_claims[0].quote == _one(2, 3).key_claims[0].quote


def test_stored_segment_analysis_is_not_re_padded_on_load():
    a = _padded(2, 3, pad=0)
    again = Analysis.model_validate_json(a.model_dump_json())
    assert again.key_claims[0].quote == a.key_claims[0].quote


# --- trimming an over-limit range (round 6) -------------------------------------------------

def _filler_flat(n: int, special: dict[int, str]):
    """n segments of distinct filler words; `special` replaces chosen segments' text."""
    return flatten([Segment(float(i), float(i + 1), special.get(i, f"filler{i} padding{i} text{i} here{i}"))
                    for i in range(n)])


def _claim(claim, seg_from, seg_to, flat):
    return resolve_segment_claims(_draft({"claim": claim, "segment_from": seg_from, "segment_to": seg_to}),
                                  flat, pad=1)


VENU = "the garmin venue tracks sets reps and weight for you"


def test_over_limit_range_is_trimmed_to_the_window_holding_the_claim():
    flat = _filler_flat(40, {30: VENU})
    out = _claim("Garmin Venue tracks sets, reps and weight", 0, 39, flat)
    assert out.dropped_quotes == []
    c = out.key_claims[0]
    texts = flat.segment_texts()
    assert c.quote == " ".join(texts[29:32])                 # segment 30, padded one per side
    assert c.anchor.t_start == flat.seg_times[29]
    assert out.trimmed_quotes == [{"claim": "Garmin Venue tracks sets, reps and weight", "segment_from": 0,
                                   "segment_to": 39, "kept_from": 30, "kept_to": 30, "coverage": 1.0}]


def test_over_word_limit_range_is_trimmed():
    long_seg = " ".join(f"word{j}" for j in range(125))
    flat = flatten([Segment(0.0, 1.0, long_seg), Segment(1.0, 2.0, VENU), Segment(2.0, 3.0, long_seg)])
    out = _claim("Garmin Venue tracks sets, reps and weight", 0, 2, flat)   # 3 segments, over 130 words
    assert out.dropped_quotes == [] and out.key_claims[0].quote == VENU   # padding would pass 130 words
    assert out.trimmed_quotes[0]["kept_from"] == out.trimmed_quotes[0]["kept_to"] == 1


def test_trim_needs_half_the_claim_words():
    flat = _filler_flat(40, {30: "the garmin venue is a watch"})
    out = _claim("Garmin Venue tracks sets, reps and weight", 0, 39, flat)   # 2 of 6 content words
    assert out.key_claims == [] and out.trimmed_quotes == []
    assert out.dropped_quotes[0]["reason"] == f"span over {MAX_QUOTE_SEGMENTS} segments"


def test_claim_without_content_words_is_never_trimmed():
    out = _claim("it is", 0, 39, _filler_flat(40, {}))
    assert out.key_claims == [] and out.dropped_quotes[0]["reason"] == f"span over {MAX_QUOTE_SEGMENTS} segments"


def test_trim_never_leaves_the_models_range():
    flat = _filler_flat(60, {50: VENU})
    out = _claim("Garmin Venue tracks sets, reps and weight", 0, 39, flat)   # the words sit at 50
    assert out.key_claims == [] and out.trimmed_quotes == []


def test_trimmed_window_too_short_is_dropped_as_too_short():
    flat = _filler_flat(40, {30: "garmin venue"})
    out = _claim("Garmin Venue", 0, 39, flat)
    assert out.key_claims == [] and out.dropped_quotes[0]["reason"] == "too short"


def test_trim_tie_prefers_fewest_segments_then_earliest():
    flat = _filler_flat(40, {10: VENU, 30: VENU, 20: "garmin venue tracks", 21: "sets reps and weight"})
    t = _claim("Garmin Venue tracks sets, reps and weight", 0, 39, flat).trimmed_quotes[0]
    assert (t["kept_from"], t["kept_to"]) == (10, 10)


def test_trim_is_fast_on_an_absurd_range():
    import time
    flat = _filler_flat(3000, {2500: VENU})
    t0 = time.perf_counter()
    out = _claim("Garmin Venue tracks sets, reps and weight", 0, 2999, flat)
    assert time.perf_counter() - t0 < 3.0
    assert out.trimmed_quotes[0]["kept_from"] == 2500


def test_trim_prefers_the_window_keeping_a_negation():
    flat = _filler_flat(40, {10: "the watch does track sets reps and weight",
                             30: "the watch does not track sets reps and weight"})
    t = _claim("The watch does not track sets, reps and weight", 0, 39, flat).trimmed_quotes[0]
    assert (t["kept_from"], t["kept_to"]) == (30, 30)


def test_segment_anchor_records_segment_index_and_page():
    a = _one(2, 2).key_claims[0].anchor
    assert (a.segment, a.page) == (2, None)             # YouTube-like flat: no pages
    paged = flatten(SEGS)
    paged.seg_pages = [1, 1, 2, 3, 3, 3, 3]
    b = _one(2, 2, paged).key_claims[0].anchor
    assert (b.segment, b.page) == (2, 2)
    c = _one(3, 3, paged).key_claims[0].anchor
    assert (c.segment, c.page) == (3, 3)
