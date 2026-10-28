import unicodedata

from pipeline.anchors import normalize, QuoteLocator, validate_analysis, MIN_QUOTE_CHARS
from pipeline.schemas import Analysis, Claim
from pipeline.transcription.base import Segment, Transcript

TRANSCRIPT = Transcript("abcdefghijk", "youtube_captions", "en", [
    Segment(0.0, 4.0, "So the first thing you need to know is that"),
    Segment(4.0, 9.5, "Acme's search, uh, weighs recency about thirty percent"),
    Segment(9.5, 14.0, "more than it used to, at least for shops under one hundred sales."),
    Segment(754.0, 760.0, "And that's why I always say: test one listing at a time!"),
])
FLAT = TRANSCRIPT.flatten()


def test_normalize_keeps_index_map():
    norm, idx = normalize("Hello,   World! it's")
    assert norm == "hello world it's"
    assert "Hello,   World! it's"[idx[6]] == "W"


def test_exact_match_after_punctuation_and_case():
    loc = QuoteLocator(FLAT)
    m = loc.locate("ACME'S search... uh, weighs recency about thirty percent")
    assert m is not None and m.score == 100.0
    assert FLAT.text[m.char_start:m.char_end] == "Acme's search, uh, weighs recency about thirty percent"


def test_fuzzy_match_with_small_error():
    loc = QuoteLocator(FLAT)
    m = loc.locate("Acme's search weighs recency about thirty percent more than it used to")   # "uh," dropped
    assert m is not None and m.score >= 90
    assert FLAT.time_at(m.char_start) == 4.0


def test_no_match_and_too_short():
    loc = QuoteLocator(FLAT)
    assert loc.locate("completely unrelated sentence about cooking pasta al dente") is None
    assert len("Acme") < MIN_QUOTE_CHARS
    assert loc.locate("Acme") is None       # shorter than MIN_QUOTE_CHARS


def test_validate_analysis_attaches_anchor_and_drops_unmatched():
    a = Analysis(summary="s", key_claims=[
        Claim(claim="Recency weighs ~30% more", quote="weighs recency about thirty percent more than it used to"),
        Claim(claim="Test one listing at a time", quote="test one listing at a time"),
        Claim(claim="Hallucinated", quote="Acme penalises shops that use more than five tags per listing"),
    ])
    out = validate_analysis(a, FLAT)
    assert [c.claim for c in out.key_claims] == ["Recency weighs ~30% more", "Test one listing at a time"]
    assert out.key_claims[0].anchor.t_start == 4.0
    assert out.key_claims[1].anchor.t_start == 754.0
    assert out.dropped_quotes == [{"claim": "Hallucinated", "reason": "no match",
                                   "quote": "Acme penalises shops that use more than five tags per listing"}]


# --- edge cases (review notes) ----------------------------------------------

def test_curly_apostrophe_normalizes_like_straight_apostrophe():
    norm, idx = normalize("it’s")
    assert norm == "it's"
    assert len(norm) == len(idx)


def test_needle_longer_than_haystack_scoring_low_is_rejected():
    # Needle contains every word of the (single-segment) haystack, interspersed with enough
    # extra/junk words that the alignment score drops below MIN_SCORE -- this exercises the
    # "needle longer than haystack" swap in partial_ratio_alignment, which still finds a
    # full-haystack-width window (never an IndexError), just with a low score.
    tiny = Transcript("x", "youtube_captions", "en", TRANSCRIPT.segments[:1]).flatten()
    loc = QuoteLocator(tiny)
    hay_words = normalize(tiny.text)[0].split()
    junky = " ".join(w + " zz" for w in hay_words) + \
        " extra long unrelated tail words to pad this out significantly more and more"
    assert loc.locate(junky) is None


def test_needle_longer_than_haystack_scoring_high_still_matches():
    # Needle is the whole haystack plus a couple of trailing words: partial_ratio finds the
    # haystack as a perfect (100%) window at the start, so this must still match, and the
    # resulting char_end must never exceed the transcript's own text length.
    tiny = Transcript("x", "youtube_captions", "en", TRANSCRIPT.segments[:1]).flatten()
    loc = QuoteLocator(tiny)
    m = loc.locate("So the first thing you need to know is that too")
    assert m is not None and m.score >= 90
    assert m.char_end <= len(tiny.text)


def test_match_at_very_end_of_haystack():
    loc = QuoteLocator(FLAT)
    m = loc.locate("test one listing at a time")
    assert m is not None
    # the transcript's final char is "!" (stripped by normalize, so not part of the match)
    assert m.char_end == len(FLAT.text) - 1
    assert FLAT.text[m.char_start:m.char_end] == "test one listing at a time"


def test_empty_transcript_never_matches():
    empty = Transcript("x", "youtube_captions", "en", []).flatten()
    loc = QuoteLocator(empty)
    assert loc.locate("any quote long enough to pass the minimum length check") is None


def test_fuzzy_match_snaps_to_word_boundary_across_a_big_time_gap():
    # A naive fuzzy alignment lands at dest_start=159, inside "sales" (segment 3, t=9.5), even
    # though the quote's real content ("And that's why...") belongs to segment 4 (t=754.0).
    # Without snapping to the following word boundary this resolves to the wrong segment.
    loc = QuoteLocator(FLAT)
    m = loc.locate("so And that's why I always say test one listing at a time")
    assert m is not None
    assert FLAT.time_at(m.char_start) == 754.0


def test_fuzzy_match_snaps_off_the_segment_joining_space():
    # The single space joining segment 1 and segment 2 belongs to segment 1 for `time_at`
    # purposes (flatten() records seg_starts *after* accounting for the joining space). A raw
    # alignment that starts exactly on that space must snap forward onto segment 2's first word,
    # not resolve to segment 1's time.
    loc = QuoteLocator(FLAT)
    m = loc.locate("xk Acme's search weighs recency about thirty percent")
    assert m is not None
    assert FLAT.time_at(m.char_start) == 4.0


def test_quote_shorter_than_min_words_is_dropped_as_too_short():
    a = Analysis(summary="s", key_claims=[
        Claim(claim="Too short", quote="weighs recency more"),   # 3 words: under MIN_QUOTE_WORDS
    ])
    out = validate_analysis(a, FLAT)
    assert out.key_claims == []
    assert out.dropped_quotes == [{"claim": "Too short", "reason": "too short", "quote": "weighs recency more"}]


def test_min_score_threshold_rejects_a_match_that_would_otherwise_pass():
    # The default MIN_SCORE (90) accepts this quote (see test_fuzzy_match_with_small_error,
    # score ~95.7); a locator configured with a stricter threshold must reject it.
    loc = QuoteLocator(FLAT, min_score=99)
    quote = "Acme's search weighs recency about thirty percent more than it used to"
    assert loc.locate(quote) is None


def test_repeated_quote_resolves_to_first_occurrence():
    tr = Transcript("x", "youtube_captions", "en", [
        Segment(0.0, 5.0, "test one listing at a time and remember this always"),
        Segment(100.0, 105.0, "test one listing at a time again for good measure"),
    ])
    flat = tr.flatten()
    loc = QuoteLocator(flat)
    m = loc.locate("test one listing at a time")
    assert m is not None
    assert flat.time_at(m.char_start) == 0.0
    assert flat.text[m.char_start:m.char_end] == "test one listing at a time"


def test_modifier_letter_apostrophe_normalizes_like_straight_apostrophe():
    norm, idx = normalize("itʼs")   # U+02BC MODIFIER LETTER APOSTROPHE, as Whisper emits
    assert norm == "it's"
    assert len(norm) == len(idx)


# --- CJK / space-less scripts (regression: MIN_QUOTE_WORDS rejected every quote) ------------

CJK_TRANSCRIPT = Transcript("cjk0000001", "whisper_local", "zh", [
    Segment(0.0, 5.0, "这是第一段文字内容用于测试分词功能是否正常"),
    Segment(5.0, 10.0, "这是第二段的重要内容包含关键信息和细节描述"),
    Segment(10.0, 15.0, "这是第三段总结性的文字用于收尾陈述和结论"),
])
CJK_FLAT = CJK_TRANSCRIPT.flatten()


def test_cjk_quote_without_spaces_locates_with_correct_timestamp():
    # Chinese (and Japanese/Thai) has no inter-word spaces, so normalize() produces a single
    # unbroken "word" for an entire segment -- the old word-count floor rejected every such
    # quote as "too short" regardless of length, which would fail the whole source in the analyze worker.
    quote = CJK_TRANSCRIPT.segments[1].text[:12]   # >= MIN_QUOTE_CHARS, still a single token
    assert len(quote.split()) == 1
    loc = QuoteLocator(CJK_FLAT)
    m = loc.locate(quote)
    assert m is not None and m.score == 100.0
    assert CJK_FLAT.time_at(m.char_start) == 5.0
    assert CJK_FLAT.text[m.char_start:m.char_end] == quote


def test_single_long_latin_token_still_requires_a_real_match():
    # The word-count floor exemption for single tokens isn't CJK-specific -- a lone 15-char
    # English word also skips it -- but it must still fail on the fuzzy-match step rather than
    # being accepted just for being long and absent.
    loc = QuoteLocator(FLAT)
    assert loc.locate("extraordinarily") is None



# --- NFKC: Unicode-equivalent but differently encoded text ---------------------------------

def test_normalize_applies_nfkc_compatibility_forms():
    # ligature, full-width Latin, NBSP and ideographic space all fold to their plain forms
    text = "ﬁnal ＣＬＡＵＤＥ　code"   # "ﬁnal ＣＬＡＵＤＥ　code"
    norm, idx = normalize(text)
    assert norm == "final claude code"
    assert len(idx) == len(norm)
    # both "f" and "i" come from the single ligature char at offset 0
    assert idx[0] == 0 and idx[1] == 0
    assert text[idx[norm.index("claude")]] == "Ｃ"


def test_nfkc_ascii_quote_matches_fullwidth_and_ligature_transcript():
    # (flatten() already folds an NBSP inside a segment, so the NBSP case is on the quote side)
    original = ("the ﬁrst ＩＤＥ plugin really changed my ofﬁce "
                "workﬂow")   # "the ﬁrst ＩＤＥ plugin really changed my ofﬁce workﬂow"
    tr = Transcript("x", "youtube_captions", "en", [
        Segment(0.0, 5.0, "intro words here"),
        Segment(5.0, 9.0, original),
    ])
    flat = tr.flatten()
    m = QuoteLocator(flat).locate("the first IDE plugin really changed my office workflow")
    assert m is not None and m.score == 100.0
    assert flat.time_at(m.char_start) == 5.0
    # the span covers the original text in full, including the trailing ligature char
    assert flat.text[m.char_start:m.char_end] == original


def test_nfkc_length_change_before_quote_keeps_original_offsets():
    # Every char in segment 0 expands under NFKC (½ -> 1⁄2, ㎏ -> kg, ㍻ -> 平成, ﬃ -> ffi,
    # … -> ..., ㎒ -> MHz, ¼ -> 1⁄4): a normaliser that ran NFKC over the whole string first and
    # kept a per-char index map would shift every offset after it. The returned span must still
    # index the ORIGINAL transcript text.
    seg0 = "½ ㎏ ㍻ ﬃ … ㎒ ¼"
    assert len(unicodedata.normalize("NFKC", seg0)) > len(seg0)
    tr = Transcript("x", "youtube_captions", "en", [
        Segment(0.0, 3.0, seg0),
        Segment(3.0, 8.0, "so weigh the flour before you start mixing anything"),
        Segment(8.0, 12.0, "and then rest the dough for an hour"),
    ])
    flat = tr.flatten()
    loc = QuoteLocator(flat)
    m = loc.locate("weigh the flour before you start mixing")
    assert m is not None
    assert flat.text[m.char_start:m.char_end] == "weigh the flour before you start mixing"
    assert flat.time_at(m.char_start) == 3.0
    m2 = loc.locate("rest the dough for an hour")
    assert m2 is not None
    assert flat.text[m2.char_start:m2.char_end] == "rest the dough for an hour"
    assert flat.time_at(m2.char_start) == 8.0


def test_nfkc_composes_combining_sequences_and_span_keeps_the_mark():
    # transcript has "e" + U+0301 COMBINING ACUTE; the quote has precomposed "é" (U+00E9)
    tr = Transcript("x", "youtube_captions", "fr", [
        Segment(0.0, 4.0, "nous avons parle du café"),
    ])
    flat = tr.flatten()
    m = QuoteLocator(flat).locate("nous avons parle du café")
    assert m is not None and m.score == 100.0
    assert flat.text[m.char_start:m.char_end] == "nous avons parle du café"


# --- score floor: short fuzzy matches must be omission-only (see SHORT_NEEDLE_CHARS) ---------

def test_short_near_miss_with_a_wrong_number_is_dropped():
    # 34 normalised chars; the transcript says "thirty", the model wrote "forty". This scores
    # ~91.2 -- above MIN_SCORE -- but it changes a word, so it's a misquote and must be dropped.
    loc = QuoteLocator(FLAT)
    m, reason = loc.locate_with_reason("weighs recency about forty percent")
    assert m is None and reason == "no match"


def test_short_quote_with_one_dropped_filler_still_lands():
    # 34 chars; the transcript has "Acme's search, uh, weighs recency about". The quote only
    # omits the filler; it scores ~91.2 too -- the same as the misquote above, which is why a
    # higher score floor can't separate the two -- and must land on segment 2.
    loc = QuoteLocator(FLAT)
    m = loc.locate("Acme's search weighs recency about")
    assert m is not None and 90 <= m.score < 95
    assert FLAT.time_at(m.char_start) == 4.0
    assert FLAT.text[m.char_start:].startswith("Acme's search")


def test_short_quote_with_an_added_word_is_dropped():
    # omission-only cuts both ways: an inserted word ("really") is not in the transcript
    loc = QuoteLocator(FLAT)
    assert loc.locate("test one listing really at a time") is None


def test_long_quote_with_several_small_deviations_keeps_the_90_floor():
    # 125 chars with "that" and "uh" missing and the ends trimmed mid-word by the aligner: ~93.6.
    # Long needles are judged on score alone, so this still lands.
    loc = QuoteLocator(FLAT)
    m = loc.locate("the first thing you need to know is Acme's search weighs recency about thirty "
                   "percent more than it used to at least for shops")
    assert m is not None and 90 <= m.score < 95
    assert FLAT.time_at(m.char_start) == 0.0


# --- mixed CJK + Latin quotes ----------------------------------------------------------------
# Rule (pipeline/anchors.py, _too_short): when a needle contains space-less-script chars, its
# word count is (space-delimited words of other scripts) + ceil(space-less chars / 2); the
# 5-word floor applies to that count. A pure-CJK needle that clears the 12-char floor always
# counts >= 6, so pure CJK behaves as before; a needle with no space-less chars keeps the old
# rule (a single token is exempt, 2+ tokens need 5).

JA_TRANSCRIPT = Transcript("ja00000001", "youtube_captions", "ja", [
    Segment(0.0, 5.0, "今日はClaude Codeの新しい機能について話します"),
    Segment(5.0, 10.0, "Claude Codeを使うと開発がとても速くなります"),
    Segment(10.0, 15.0, "ｉＰｈｏｎｅのカメラは本当にすごいです"),   # full-width "ｉＰｈｏｎｅ"
])
JA_FLAT = JA_TRANSCRIPT.flatten()


def test_mixed_japanese_quote_with_english_product_name_lands():
    # "claude" + "codeを使うと..." is two space-delimited tokens; the old floor counted 2 words
    # and rejected it as too short despite 16 Japanese chars.
    loc = QuoteLocator(JA_FLAT)
    m, reason = loc.locate_with_reason("Claude Codeを使うと開発がとても速くなります")
    assert reason is None and m.score == 100.0
    assert JA_FLAT.time_at(m.char_start) == 5.0
    assert JA_FLAT.text[m.char_start:m.char_end] == "Claude Codeを使うと開発がとても速くなります"


def test_short_mixed_quote_is_too_short():
    # 2 Latin words + 3 Japanese chars (ceil(3/2) = 2) = 4 word-equivalents: under the floor,
    # although it clears the 12-char floor
    loc = QuoteLocator(JA_FLAT)
    m, reason = loc.locate_with_reason("Claude Codeを使う")
    assert m is None and reason == "too short"


def test_mixed_quote_matches_fullwidth_latin_in_transcript():
    loc = QuoteLocator(JA_FLAT)
    m = loc.locate("iPhoneのカメラは本当にすごいです")
    assert m is not None and m.score == 100.0
    assert JA_FLAT.time_at(m.char_start) == 10.0
    assert JA_FLAT.text[m.char_start:m.char_end] == JA_TRANSCRIPT.segments[2].text


def test_mixed_quote_with_one_changed_kanji_is_dropped():
    # 速 (fast) -> 遅 (slow): one char, a high fuzzy score, opposite meaning. Short needle, so the
    # omission-only rule applies per Japanese char and the misquote is dropped.
    loc = QuoteLocator(JA_FLAT)
    m, reason = loc.locate_with_reason("Claude Codeを使うと開発がとても遅くなります")
    assert m is None and reason == "no match"


def test_mixed_quote_that_omits_a_word_still_lands():
    # the particle が left out: omission-only, so it still anchors on segment 2 (leaving out
    # とても, three chars of a 30-char needle, already scores under MIN_SCORE)
    loc = QuoteLocator(JA_FLAT)
    m = loc.locate("Claude Codeを使うと開発とても速くなります")
    assert m is not None and m.score >= 90
    assert JA_FLAT.time_at(m.char_start) == 5.0
