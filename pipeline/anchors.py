"""Anchor validation (IDEA-205): every claim's quote is tied to an exact place in its transcript;
the offset becomes a timestamp (IDEA-001). Claims that cannot be tied are dropped and recorded.

Two modes, recorded on the analysis as `Analysis.quote_mode`:

- "segments" (IDEA-294, DECISION-027; every new analysis): the analyze model returns a contiguous
  segment range per claim and `resolve_segment_claims` copies the quote out of the transcript and
  sets the anchor at the first segment's start -- exact by construction, no matching involved.
- "text" (analyses stored before the segment-pointer prompt): the model's own quote is located
  with `QuoteLocator` by `validate_analysis`. QuoteLocator also backs eval assertion 5.

Matching is on a normalised form (NFKC, lowercase, punctuation stripped, whitespace folded) with
start/end maps back to the original text so offsets stay exact even where NFKC changes length.
The fragment salvage of stitched quotes (added 2026-10-09) is retired: a segment range cannot
be stitched, and stored text-mode analyses already carry their salvaged spans as quotes.
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

from pipeline.schemas import Analysis, AnalysisDraft, Anchor, Claim
from pipeline.transcription.base import FlatTranscript

MIN_SCORE = 90.0
MIN_QUOTE_CHARS = 12
# The analyze prompt asks for 8-40 word quotes; a handful of normalised characters clears
# MIN_QUOTE_CHARS by chance in a long transcript (e.g. "and in that is" scored 92.9 against
# random text), so word count is the primary floor and MIN_QUOTE_CHARS a secondary one. Scripts
# without inter-word spaces (zh/ja/th), alone or mixed with Latin, are counted as described in
# _too_short.
MIN_QUOTE_WORDS = 5

# Segment-pointer quotes (IDEA-294): a claim's range may span at most MAX_QUOTE_SEGMENTS segments
# and MAX_QUOTE_SPAN_WORDS words (counted by _word_count). Raised from 8 / 60 on 2026-10-09: on
# real runs auto-caption segments were 2-3 s, 8 segments dropped 14-27% of claims as too long, and
# the dropped spans had a median of 10 segments (max 24). 16 segments is still well under a minute
# of speech; the renderer shows at most MAX_QUOTE_WORDS (40) of the quote either way. Both values
# are rendered into the analyze prompt (pipeline/prompts/analyze.py), so changing one changes the
# prompt fingerprint. Raised again to 24 / 130 in round 6 (2026-10-14): the round-5 API runs dropped
# 5-12% of claims as too long, most at 17-24 segments or 101-133 words.
MAX_QUOTE_SEGMENTS = 24
MAX_QUOTE_SPAN_WORDS = 130
# Round 5 (2026-10-14): kept segment ranges are widened by this many segments per side, within the
# limits above, because models cut ranges mid-sentence or just after the segment naming the subject.
QUOTE_PAD_SEGMENTS = 1
# Round 6 (2026-10-14): a range over the limits is trimmed to the window inside it that holds the most
# of the claim's content words (_trim_range) and dropped only when that window holds under this share.
# Calibrated on the 38 round-5 drops: replayed with the raised limits, 33 are kept (most trimmed, a few
# now fit outright); kept claims' own quotes hold a median 0.88-0.94 of their claim's content words
# (p10 0.56-0.73).
TRIM_MIN_COVERAGE = 0.5
# Words that say nothing about which part of a transcript states a claim; with words under 3 chars they
# are left out of the coverage count. Negations ("not", "never", "without") are content words on
# purpose: a negated claim must favour the window that holds the negation.
_STOPWORDS = frozenset(
    "the and for that this with are was were you your they them their from have has had but all can "
    "its it's than then there what when which who will would could should about into over just like also "
    "very more most some such been being our out one two get got does did".split())

# Score floor review (2026-10-08). MIN_SCORE stays 90: against a synthetic 20k-word transcript
# of high-frequency spoken English, random quotes of 9+ words never reached 90 (max 85.7), so
# long needles don't land on wrong spans by chance, and a long legitimate quote routinely loses a
# few points to dropped fillers ("uh", "you know") and ASR spelling. Short needles are different:
# random 5-7 word quotes reached 90 in ~10% of trials (max 94.4), and a 34-char misquote
# ("about forty percent" for "about thirty percent") scores 91.2 -- exactly what a legitimate
# 34-char quote that merely drops an "uh" scores. No score floor separates those two, so instead
# of raising MIN_SCORE a fuzzy match on a needle shorter than SHORT_NEEDLE_CHARS must also be
# omission-only: every needle unit (word, or single char of a space-less script) appears, in
# order, in the matched transcript span. The model may leave transcript words out of a short
# quote; it may not add or change any. With that rule none of the random >= 90 hits survived.
# 40 chars is about 8 words, the shortest quote the text-quote analyze prompt asked for.
SHORT_NEEDLE_CHARS = 40

# Transcripts (YouTube captions) commonly use curly quotes; the model emitting quotes for
# matching typically uses straight ones. Map the common curly variants onto their straight
# equivalents before the rest of normalisation runs. U+02BC (MODIFIER LETTER APOSTROPHE) is
# what Whisper emits for contractions; it is alnum, so without this mapping it would silently
# survive normalisation as a "letter" instead of being treated like every other apostrophe.
_QUOTE_MAP = {"’": "'", "‘": "'", "ʼ": "'"}

# Scripts written without spaces between words: Thai, Lao, Myanmar, Khmer, Hiragana, Katakana
# (incl. the prolonged sound mark), CJK ideographs (unified, extension A, compatibility, and the
# supplementary-plane extensions). Hangul is written with spaces and is deliberately not here.
_UNSPACED_RANGES = (
    (0x0E00, 0x0EFF), (0x1000, 0x109F), (0x1780, 0x17FF), (0x3040, 0x30FF), (0x31F0, 0x31FF),
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x3FFFF),
)


def _is_unspaced(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _UNSPACED_RANGES)


def _units(norm: str) -> list[str]:
    """Splits normalised text into matching units: a space-delimited run of other scripts is one
    unit (a word); each char of a space-less script is a unit of its own."""
    out: list[str] = []
    for token in norm.split():
        run = ""
        for ch in token:
            if _is_unspaced(ch):
                if run:
                    out.append(run)
                    run = ""
                out.append(ch)
            else:
                run += ch
        if run:
            out.append(run)
    return out


def _word_count(norm: str) -> int:
    """Words in a normalised text: space-delimited units of other scripts count one each; chars of
    a space-less script (zh/ja/th) count half a word each, rounded up (a Chinese/Japanese word
    averages ~1.5-2 chars)."""
    units = _units(norm)
    unspaced = sum(1 for u in units if len(u) == 1 and _is_unspaced(u))
    return len(units) - unspaced + math.ceil(unspaced / 2)


def _too_short(needle: str) -> bool:
    """The length floors (MIN_QUOTE_CHARS, then MIN_QUOTE_WORDS) on a normalised needle.

    Words are counted by _word_count: "Claude Codeを使う" is 2 + ceil(3/2) = 4 words, too short,
    while "Claude Codeを使うと開発がとても速くなります" is 2 + 8 = 10. A pure-CJK needle that clears
    MIN_QUOTE_CHARS always counts >= 6. A needle with no space-less chars keeps the plain rule: a
    lone token long enough to clear MIN_QUOTE_CHARS is exempt (it is distinctive enough not to
    match by chance, and covers space-less scripts not in _UNSPACED_RANGES), 2+ tokens need
    MIN_QUOTE_WORDS.
    """
    if len(needle) < MIN_QUOTE_CHARS:
        return True
    units = _units(needle)
    if len(units) == 1 and not _is_unspaced(units[0][0]):
        return False
    return _word_count(needle) < MIN_QUOTE_WORDS


def _is_subsequence(needle: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(any(u == h for h in it) for u in needle)


@dataclass
class QuoteMatch:
    """A located quote: the char span in the transcript's original (unnormalised) text, and the
    rapidfuzz match score (100.0 for an exact normalised substring match)."""
    char_start: int
    char_end: int
    score: float


def _clusters(text: str):
    """Yields (start, end) spans of `text`: one base char plus any combining marks after it.

    NFKC can change length (a ligature or "½" expands, a base + combining mark composes into one
    char), so it is applied per cluster rather than to the whole string: every normalised char
    then still knows which span of the ORIGINAL text it came from. A cluster is the unit NFKC
    composition works on for everything except conjoining Hangul jamo, which captions don't use.
    """
    n = len(text)
    i = 0
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(text[j]):
            j += 1
        yield i, j
        i = j


def normalize_spans(text: str) -> tuple[str, list[int], list[int]]:
    """Returns (normalised text, starts, ends): normalised char k came from `text[starts[k]:ends[k]]`.

    Normalisation is NFKC, then curly/modifier apostrophes folded to "'", lowercase, every char
    that is not alphanumeric or "'" dropped, whitespace runs folded to one space and trimmed. Chars
    that NFKC expands (e.g. the ligature "ﬁ" -> "fi") all map to the same original span, so a match
    that starts or ends inside an expansion still covers the whole original char.
    """
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    prev_space = True
    for i, j in _clusters(text):
        chunk = text[i:j]
        if j - i > 1 or not chunk.isascii():
            chunk = unicodedata.normalize("NFKC", chunk)
        for ch in chunk:
            ch = _QUOTE_MAP.get(ch, ch)
            c = ch.lower()
            if len(c) != 1:
                c = ch
            if c.isspace():
                if prev_space:
                    continue
                out.append(" ")
                starts.append(i)
                ends.append(j)
                prev_space = True
                continue
            if not (c.isalnum() or c == "'"):
                continue
            out.append(c)
            starts.append(i)
            ends.append(j)
            prev_space = False
    while out and out[-1] == " ":
        out.pop()
        starts.pop()
        ends.pop()
    return "".join(out), starts, ends


def normalize(text: str) -> tuple[str, list[int]]:
    """Returns (normalised text, index_map) where index_map[i] is the offset in `text` of the
    (start of the) original char that normalised char i came from. See `normalize_spans`."""
    norm, starts, _ends = normalize_spans(text)
    return norm, starts


class QuoteLocator:
    """Locates quotes against one transcript's normalised text.

    The haystack is normalised once at construction; each `locate()` call normalises only the
    quote, so a locator is meant to be reused across every claim in an `Analysis`.
    """

    def __init__(self, flat: FlatTranscript, min_score: float = MIN_SCORE):
        self.flat = flat
        self.min_score = min_score
        self.hay, self.idx, self.idx_end = normalize_spans(flat.text)

    def locate(self, quote: str) -> QuoteMatch | None:
        """Returns the best match for `quote`, or None if it's too short or has no good match.

        Use `locate_with_reason` instead when the drop reason is also needed (e.g. from
        `validate_analysis`) to avoid normalising the quote a second time.
        """
        match, _reason = self.locate_with_reason(quote)
        return match

    def locate_with_reason(self, quote: str) -> tuple[QuoteMatch | None, str | None]:
        """Core matching logic. Returns (match, drop_reason); drop_reason is None on success."""
        needle, _ = normalize(quote)
        if _too_short(needle):
            return None, "too short"
        if not self.hay:
            return None, "no match"
        pos = self.hay.find(needle)
        if pos >= 0:
            # An exact hit on a needle that already cleared the length floor (5+ words, or a
            # single token of 12+ chars) is specific enough that landing mid-word is negligible;
            # no snapping needed here (unlike the fuzzy branch below).
            n_start, n_end, score = pos, pos + len(needle), 100.0
        else:
            al = fuzz.partial_ratio_alignment(needle, self.hay, score_cutoff=self.min_score)
            if al is None:
                return None, "no match"
            n_start, n_end, score = al.dest_start, al.dest_end, float(al.score)
            if len(needle) < SHORT_NEEDLE_CHARS and not self._omission_only(needle, n_start, n_end):
                return None, "no match"
            # partial_ratio_alignment returns a raw char window that can start (or end) mid-word
            # -- e.g. straddling the single space that joins two transcript segments, or landing
            # inside an unrelated word entirely. Snap onto whole words so the offset resolves to
            # the segment the quote's first word actually belongs to (`FlatTranscript.time_at`
            # attributes a boundary space to the segment before it).
            # Exception: when the window cut into the quote's own first word (the aligner's window
            # is only as long as the needle, so a filler the quote omits pushes it right, e.g.
            # "cme's search uh ..." for "acme's search ..."), snap back to that word's start.
            if n_start > 0 and self.hay[n_start - 1] != " ":
                word_start = self.hay.rfind(" ", 0, n_start) + 1
                word_end = self.hay.find(" ", n_start)
                word_end = len(self.hay) if word_end < 0 else word_end
                if self.hay[word_start:word_end] == needle.split(" ", 1)[0]:
                    n_start = word_start
                else:
                    nxt = self.hay.find(" ", n_start, n_end)
                    if 0 <= nxt < n_end - 1:
                        n_start = nxt + 1
            while n_start < n_end and self.hay[n_start] == " ":
                n_start += 1
            while n_end > n_start and self.hay[n_end - 1] == " ":
                n_end -= 1
            if n_end <= n_start:
                return None, "no match"
        start = self.idx[n_start]
        # n_end <= len(hay) == len(self.idx_end), so n_end - 1 is valid; idx_end (not idx + 1) so a
        # span ending in a combining mark or an NFKC-expanded char covers it in full.
        end = self.idx_end[n_end - 1]
        return QuoteMatch(start, end, score), None

    def _omission_only(self, needle: str, n_start: int, n_end: int) -> bool:
        """True if every unit of `needle` appears, in order, in the haystack window
        [n_start, n_end) widened to whole words (see SHORT_NEEDLE_CHARS).

        The raw aligner window can cut a word at either end, so it is widened over adjacent
        space-delimited word chars -- but never across a space-less-script char, where widening
        "to the word boundary" would swallow a whole caption segment and make the check lax.
        """
        hay = self.hay

        def widenable(ch: str) -> bool:
            return ch != " " and not _is_unspaced(ch)

        while n_start > 0 and widenable(hay[n_start - 1]) and widenable(hay[n_start]):
            n_start -= 1
        while n_end < len(hay) and widenable(hay[n_end]) and widenable(hay[n_end - 1]):
            n_end += 1
        return _is_subsequence(_units(needle), _units(hay[n_start:n_end]))


def validate_analysis(analysis: Analysis, flat: FlatTranscript, min_score: float = MIN_SCORE) -> Analysis:
    """Text mode (analyses made with the old text-quote prompt): locates every claim's quote in
    `flat` and attaches the resulting Anchor; claims whose quote can't be located are removed from
    `key_claims` and recorded in `dropped_quotes` instead. Sets `quote_mode = "text"`. New analyses
    use `resolve_segment_claims` instead.

    Mutates and returns the same `analysis` object (it overwrites `key_claims` and
    `dropped_quotes`), so call it exactly once per analysis -- a second call would validate the
    already-filtered `key_claims` and silently drop the record of the first pass's rejects.
    """
    locator = QuoteLocator(flat, min_score)
    kept: list[Claim] = []
    dropped: list[dict] = []
    for claim in analysis.key_claims:
        m, reason = locator.locate_with_reason(claim.quote)
        if m is None:
            dropped.append({"claim": claim.claim, "reason": reason, "quote": claim.quote})
            continue
        claim.anchor = Anchor(char_start=m.char_start, char_end=m.char_end,
                              t_start=flat.time_at(m.char_start), score=m.score)
        kept.append(claim)
    analysis.key_claims = kept
    analysis.dropped_quotes = dropped
    analysis.quote_mode = "text"
    return analysis


def _segment_index(value: object) -> int | None:
    """A segment index as the model returned it: an int (bool excluded), else None."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _pad_range(flat: FlatTranscript, lo: int, hi: int, pad: int) -> tuple[int, int]:
    """Widen [lo, hi] by up to `pad` segments per side, end first, each step only while the range
    stays inside the transcript, within MAX_QUOTE_SEGMENTS and within MAX_QUOTE_SPAN_WORDS."""
    for _ in range(pad):
        if _fits(flat, lo, hi + 1):
            hi += 1
        if _fits(flat, lo - 1, hi):
            lo -= 1
    return lo, hi


def _content_words(text: str) -> set[str]:
    return {w for w in normalize(text)[0].split() if len(w) >= 3 and w not in _STOPWORDS}


def _fits(flat: FlatTranscript, a: int, b: int) -> bool:
    """[a, b] is inside the transcript and within MAX_QUOTE_SEGMENTS and MAX_QUOTE_SPAN_WORDS."""
    if a < 0 or b >= len(flat.seg_starts) or b - a + 1 > MAX_QUOTE_SEGMENTS:
        return False
    return _word_count(normalize(flat.text[flat.seg_starts[a]:flat.segment_end(b)])[0]) <= MAX_QUOTE_SPAN_WORDS


def _trim_range(flat: FlatTranscript, lo: int, hi: int, claim: str) -> tuple[int, int, float] | None:
    """The window [a, b] inside the model's over-limit range [lo, hi] that fits the limits and holds the
    largest share of the claim's content words, with that share; None when the claim has no content
    words, no window fits, or the best share is under TRIM_MIN_COVERAGE.

    Every start is tried: the window is extended as far as the limits allow, and that start's candidate
    is the shortest window from it reaching the same share. Ties go to the fewest segments, then the
    earliest start, so the result is deterministic. The window never leaves [lo, hi]: the quote stays where the model pointed.
    """
    wanted = _content_words(claim)
    if not wanted:
        return None
    # Each segment is normalised once. Units never cross the single space that joins two segments, so a
    # window's _word_count follows from prefix sums of units and space-less units (rounded up only at
    # the end): fitting a window is O(1), and an absurd range (0..N-1 of a long video) stays cheap.
    seg_words: list[set[str]] = []
    units = [0]
    unspaced = [0]
    for i in range(lo, hi + 1):
        norm = normalize(flat.text[flat.seg_starts[i]:flat.segment_end(i)])[0]
        seg_words.append({w for w in norm.split() if len(w) >= 3 and w not in _STOPWORDS})
        us = _units(norm)
        units.append(units[-1] + len(us))
        unspaced.append(unspaced[-1] + sum(1 for u in us if len(u) == 1 and _is_unspaced(u)))

    def fits(a: int, b: int) -> bool:
        if b - a + 1 > MAX_QUOTE_SEGMENTS:
            return False
        n, cjk = units[b - lo + 1] - units[a - lo], unspaced[b - lo + 1] - unspaced[a - lo]
        return n - cjk + math.ceil(cjk / 2) <= MAX_QUOTE_SPAN_WORDS

    def share(a: int, b: int) -> float:
        return len(wanted & set().union(*seg_words[a - lo:b - lo + 1])) / len(wanted)

    best: tuple[float, int, int, int] | None = None   # (-share, size, a, b): the smallest wins
    for a in range(lo, hi + 1):
        if not fits(a, a):
            continue
        b = a
        while b < hi and fits(a, b + 1):
            b += 1
        s = share(a, b)
        y = a
        while share(a, y) < s:
            y += 1
        cand = (-s, y - a + 1, a, y)
        if best is None or cand < best:
            best = cand
    if best is None or -best[0] < TRIM_MIN_COVERAGE:
        return None
    return best[2], best[3], -best[0]


def resolve_segment_claims(draft: AnalysisDraft, flat: FlatTranscript, pad: int = 0) -> Analysis:
    """Segment mode (IDEA-294): turns the model's segment ranges into quoted, anchored claims.

    For each claim, `segment_from`..`segment_to` (inclusive; the numbers rendered in the analyze
    prompt, i.e. indices into `flat.seg_starts`) is checked, then code fills `quote` -- the
    segments' texts joined by single spaces, which is exactly `flat.text` over the span because
    flatten() joins whitespace-normalised segments with one space -- and `anchor`: that char span,
    t_start = the first segment's start time, score 100.0. A claim is dropped into
    `dropped_quotes` ({claim, segment_from, segment_to, reason[, quote]}) with one of:

    - "segment index missing or not an integer"
    - "segment index out of range"
    - "segment_from after segment_to"
    - "span over 24 segments" (MAX_QUOTE_SEGMENTS) or "span over 130 words" (MAX_QUOTE_SPAN_WORDS,
      counted by _word_count), only when trimming fails: a range over either limit is first trimmed to
      its best-matching window (`_trim_range`); a claim kept that way is recorded in `trimmed_quotes`
      ({claim, segment_from, segment_to, kept_from, kept_to, coverage}) and checked further as if the
      model had returned the window
    - "too short" (the joined text is under the floors: MIN_QUOTE_CHARS, MIN_QUOTE_WORDS and the
      space-less-script counting of _too_short)

    `pad` (the analyze worker passes QUOTE_PAD_SEGMENTS) widens each KEPT range afterwards, see
    `_pad_range`; it never changes which claims are kept or dropped.

    The claim text is never changed. Returns a new Analysis with `quote_mode = "segments"`.
    """
    n_segments = len(flat.seg_starts)
    kept: list[Claim] = []
    dropped: list[dict] = []
    trimmed: list[dict] = []
    for c in draft.key_claims:
        lo, hi = _segment_index(c.segment_from), _segment_index(c.segment_to)
        rec = {"claim": c.claim, "segment_from": c.segment_from, "segment_to": c.segment_to}
        if lo is None or hi is None:
            dropped.append({**rec, "reason": "segment index missing or not an integer"})
            continue
        if not (0 <= lo < n_segments and 0 <= hi < n_segments):
            dropped.append({**rec, "reason": "segment index out of range"})
            continue
        if lo > hi:
            dropped.append({**rec, "reason": "segment_from after segment_to"})
            continue
        if not _fits(flat, lo, hi):
            window = _trim_range(flat, lo, hi, c.claim)
            if window is None:
                if hi - lo + 1 > MAX_QUOTE_SEGMENTS:
                    dropped.append({**rec, "reason": f"span over {MAX_QUOTE_SEGMENTS} segments"})
                else:
                    dropped.append({**rec, "reason": f"span over {MAX_QUOTE_SPAN_WORDS} words",
                                    "quote": flat.text[flat.seg_starts[lo]:flat.segment_end(hi)]})
                continue
            lo, hi, coverage = window
            trimmed.append({**rec, "kept_from": lo, "kept_to": hi, "coverage": round(coverage, 2)})
        start, end = flat.seg_starts[lo], flat.segment_end(hi)
        quote = flat.text[start:end]
        norm = normalize(quote)[0]
        if _too_short(norm):
            dropped.append({**rec, "reason": "too short", "quote": quote})
            continue
        if pad:
            lo, hi = _pad_range(flat, lo, hi, pad)
            start, end = flat.seg_starts[lo], flat.segment_end(hi)
            quote = flat.text[start:end]
        anchor = Anchor(char_start=start, char_end=end, t_start=flat.seg_times[lo], score=100.0,
                        segment=lo, page=flat.seg_pages[lo] if flat.seg_pages else None)
        kept.append(Claim(claim=c.claim, quote=quote, confidence=c.confidence, anchor=anchor))
    body = draft.model_dump(exclude={"key_claims"})
    return Analysis(**body, key_claims=kept, dropped_quotes=dropped, trimmed_quotes=trimmed,
                    quote_mode="segments")
