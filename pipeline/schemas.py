"""JSON contracts for every LLM response and for the stored merge document.

Every LLM response is validated against one of these before anything downstream reads it
(IDEA-230). Extra keys are ignored; missing or malformed required keys are an error.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

SOURCE_TOKEN_RE = re.compile(r"^S[1-9]\d*\Z")
POINT_ID_RE = re.compile(r"^P[1-9]\d*\Z")
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}\Z")

# Keys that code fills in after the LLM responds (or reads back from the database), and that
# the model itself must never be allowed to fabricate on the wire. `quote` is code-filled since
# the segment-pointer analyze prompt (IDEA-294): the model returns a segment range and code copies
# the quote out of the transcript (pipeline/anchors.py::resolve_segment_claims). `support` is set
# by the citation check (IDEA-295, pipeline/stages/merge.py), never by a model; so is a
# contradiction's `check_label` (IDEA-013), copied from the check's own `label` key, and a point's
# `retracted_from` (IDEA-297), the original text code keeps when it accepts a retraction, and its
# `refs_from_fold` (set when cross-section folding adds refs).
CODE_FILLED_KEYS = frozenset({"anchor", "dropped_quotes", "trimmed_quotes", "quote", "quote_mode", "id", "support",
                              "check_label", "retracted_from", "refs_from_fold"})

# A string that must be non-empty after trimming surrounding whitespace.
NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")


def strip_code_filled(obj: Any) -> Any:
    """Recursively remove CODE_FILLED_KEYS from raw (dict/list) JSON before validation.

    Call this on wire output from the LLM client before handing it to a schema, so a model
    cannot smuggle in a fabricated `anchor`, `dropped_quotes`, `quote`, `quote_mode`, or `id`. The schemas themselves
    keep accepting those fields because the same models are re-loaded from the database, where
    code has already populated them legitimately.
    """
    if isinstance(obj, dict):
        return {k: strip_code_filled(v) for k, v in obj.items() if k not in CODE_FILLED_KEYS}
    if isinstance(obj, list):
        return [strip_code_filled(v) for v in obj]
    return obj


# --- analyze ---------------------------------------------------------------

class Anchor(_Lenient):
    """Filled by code (pipeline/anchors.py), never by the model."""
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    t_start: float = Field(ge=0)
    score: float = Field(ge=0, le=100)   # rapidfuzz scores are 0-100
    # code-filled for segment-range quotes: the segment the quote starts in, and its PDF page
    # (IDEA-202). Left out of claims_hash while None, so analyses stored before they existed hash as before.
    segment: int | None = Field(default=None, ge=0)
    page: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _char_range(self) -> "Anchor":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        return self


class Claim(_Lenient):
    """A stored claim. `quote` and `anchor` are filled by code (IDEA-294): copied from the
    transcript segments the model pointed at, or, for an analysis made with the old text-quote
    prompt (`Analysis.quote_mode == "text"`), the model's quote located by QuoteLocator."""
    claim: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    confidence: Literal["high", "medium", "low"] = "medium"
    anchor: Anchor | None = None


class _AnalysisBody(_Lenient):
    topics: list[NonEmptyStr] = Field(default_factory=list)
    insights: list[NonEmptyStr] = Field(default_factory=list)
    facts: list[NonEmptyStr] = Field(default_factory=list)
    caveats: list[NonEmptyStr] = Field(default_factory=list)
    references: list[NonEmptyStr] = Field(default_factory=list)
    summary: str = Field(min_length=1)


class ClaimDraft(_Lenient):
    """A claim as the analyze model returns it: a segment range, no quote. The indices are typed
    loosely on purpose -- a missing, non-integer, out-of-range or reversed range drops only that
    claim (into `Analysis.dropped_quotes`, pipeline/anchors.py), not the whole response."""
    claim: str = Field(min_length=1)
    segment_from: Any = None
    segment_to: Any = None
    confidence: Literal["high", "medium", "low"] = "medium"


class AnalysisDraft(_AnalysisBody):
    """The analyze response on the wire (segment-pointer prompt). Code turns it into an
    `Analysis` with `resolve_segment_claims` (pipeline/anchors.py)."""
    key_claims: list[ClaimDraft]


class Analysis(_AnalysisBody):
    key_claims: list[Claim]
    dropped_quotes: list[dict] = Field(default_factory=list)   # code-filled: claims whose quote failed
    # code-filled (round 6): claims kept by trimming an over-limit segment range to the window inside it
    # that best matches the claim, see pipeline/anchors.py::_trim_range. Empty on older analyses.
    trimmed_quotes: list[dict] = Field(default_factory=list)
    # code-filled: how the claims' quotes were produced. "segments": copied by code from the segment
    # range the model chose (IDEA-294, every analysis since analyze-2026-10-09.2). "text": the
    # model's own quote, fuzzy-located (older stored analyses; the default, so they load as such).
    # Not part of claims_hash, which covers key_claims only. Stored analyses may still carry a
    # `salvaged_quotes` key from the retired salvage path; it is ignored on load.
    quote_mode: Literal["segments", "text"] = "text"


def claims_hash(analysis: Analysis) -> str:
    """SHA-256 over a canonical JSON dump of `analysis.key_claims` (claim, quote, confidence,
    anchor) -- everything a merge's claim_refs resolve against and the renderer reads. Computed
    by code at merge time (pipeline/stages/merge.py) and re-checked by the renderer, so any
    re-analysis that changes claim content is caught, not only one that changes the count."""
    payload = [c.model_dump(mode="json") for c in analysis.key_claims]
    for item in payload:
        anchor = item.get("anchor")
        if anchor:
            for key in ("segment", "page"):
                if anchor.get(key) is None:
                    anchor.pop(key, None)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --- merge -----------------------------------------------------------------

def _check_tokens(v: list[str]) -> list[str]:
    bad = [t for t in v if not SOURCE_TOKEN_RE.match(t)]
    if bad:
        raise ValueError(f"source tokens must look like S1, S2, ...; got {bad}")
    return v


class OutlineSection(_Lenient):
    title: str = Field(min_length=1)
    scope: str = ""
    sources: list[str] = Field(min_length=1)

    @field_validator("sources")
    @classmethod
    def _tokens(cls, v: list[str]) -> list[str]:
        return _check_tokens(v)


class Outline(_Lenient):
    sections: list[OutlineSection] = Field(min_length=1, max_length=20)


class ClaimRef(_Lenient):
    source: str
    claim: int = Field(ge=0)

    @field_validator("source")
    @classmethod
    def _token(cls, v: str) -> str:
        return _check_tokens([v])[0]

    @field_validator("claim", mode="before")
    @classmethod
    def _no_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("claim must be an int, not a bool")
        return v


class Point(_Lenient):
    text: str = Field(min_length=1)
    sources: list[str] = Field(min_length=1)
    claim_refs: list[ClaimRef] = Field(min_length=1)
    id: str | None = None   # assigned by code before the verification call
    # code-filled by the citation check (pipeline/stages/merge.py::_apply_check): "partial" when the
    # check judged the point partly supported by its quotes together, or gave it no verdict; the
    # renderer marks such a point "(partly supported)". None = supported, or merged before the check.
    support: Literal["partial"] | None = None
    # code-filled by the retraction step (IDEA-297, pipeline/stages/merge.py::_retract_partials):
    # the point's text before code accepted a strict deletion of its unsupported part. None = not
    # shortened, or merged before retraction existed.
    retracted_from: str | None = None
    # code-filled by cross-section folding (pipeline/stages/merge.py::_apply_duplicates): True when a
    # fold added refs that the citation check judged against another point's text; the renderer then
    # says "(N sources)", never "(confirmed by N sources)". False on merges stored before it existed.
    refs_from_fold: bool = False

    @field_validator("sources")
    @classmethod
    def _tokens(cls, v: list[str]) -> list[str]:
        return _check_tokens(v)

    @field_validator("id")
    @classmethod
    def _id_token(cls, v: str | None) -> str | None:
        if v is not None and not POINT_ID_RE.match(v):
            raise ValueError(f"id must look like P1, P2, ...; got {v!r}")
        return v


class Subsection(_Lenient):
    title: str = Field(min_length=1)
    points: list[Point] = Field(default_factory=list)


# IDEA-013 / IDEA-309: for a contradiction the citation check keeps as real, whether the two claims
# were made under the same conditions or may differ by edition, region or date (never "outdated": a
# change over time is not a contradiction at all, see the check's different_time reason).
ContradictionLabel = Literal["same_conditions", "may_differ_by_edition_or_date"]


class Contradiction(_Lenient):
    topic: str = Field(min_length=1)
    position_a: str = Field(min_length=1)
    source_a: str
    position_b: str = Field(min_length=1)
    source_b: str
    # claim_index values of source_a / source_b whose claims state each position (may be empty).
    # Validated against the section's own sources in pipeline/stages/merge.py (invalid indices are
    # dropped, the contradiction is kept); the citation check shows their quotes as evidence.
    claims_a: list[int] = Field(default_factory=list)
    claims_b: list[int] = Field(default_factory=list)
    # code-filled by the citation check (IDEA-013, pipeline/stages/merge.py::_apply_check) from the
    # check's `label` for a contradiction it keeps as real; the renderer notes "may differ by edition
    # or date". None = no label given, or merged before the label existed.
    check_label: ContradictionLabel | None = None

    @field_validator("source_a", "source_b")
    @classmethod
    def _tokens(cls, v: str) -> str:
        return _check_tokens([v])[0]

    @field_validator("claims_a", "claims_b", mode="before")
    @classmethod
    def _claim_indices(cls, v: Any) -> Any:
        # Lenient: a non-list, a bool, or a negative/non-int entry is dropped here rather than
        # failing the whole section -- these are optional evidence pointers, not the finding.
        if not isinstance(v, list):
            return []
        return [i for i in v if isinstance(i, int) and not isinstance(i, bool) and i >= 0]


class SectionBody(_Lenient):
    subsections: list[Subsection] = Field(min_length=1)
    contradictions: list[Contradiction] = Field(default_factory=list)


class Verification(_Lenient):
    tldr: list[NonEmptyStr] = Field(min_length=1, max_length=7)
    duplicates: list[list[str]] = Field(default_factory=list)

    @field_validator("duplicates")
    @classmethod
    def _point_ids(cls, v: list[list[str]]) -> list[list[str]]:
        for group in v:
            bad = [p for p in group if not POINT_ID_RE.match(p)]
            if bad:
                raise ValueError(f"duplicate groups must hold point ids like P3; got {bad}")
        return v


# --- citation check (merge:check, IDEA-295) ----------------------------------

PointVerdictValue = Literal["supported", "partial", "unsupported"]
RefRelevanceValue = Literal["related", "unrelated"]
ContradictionReason = Literal["same_position", "different_subject", "different_time", "absence", "other"]


class RefRelevance(_Lenient):
    """Whether one cited quote bears on its point at all; `ref` is the code-generated ref id (R1,
    R2, ...) of the call. A missing relevance is allowed (None: the ref is kept); an invalid one is
    a validation error (the client's repair round-trip)."""
    ref: str = Field(min_length=1)
    relevance: RefRelevanceValue | None = None


class PointVerdict(_Lenient):
    """One verdict per point, judged over all of its cited quotes together (round-3 calibration,
    2026-10-09: judging each quote alone marked most multi-quote points partial). `point` is the
    merge's own P-id. A missing verdict is allowed (None: the point is marked partial, never
    treated as supported); an invalid one is a validation error."""
    point: str = Field(min_length=1)
    verdict: PointVerdictValue | None = None
    refs: list[RefRelevance] = Field(default_factory=list)


class ContradictionVerdict(_Lenient):
    """`contradiction` is the code-generated id (C1, C2, ...) of the call. A not_a_contradiction
    verdict without a (valid) reason is kept as reason "other"; a real one carries no reason."""
    contradiction: str = Field(min_length=1)
    verdict: Literal["real", "not_a_contradiction"]
    reason: ContradictionReason | None = None
    # A real verdict's label (IDEA-013); lenient: a missing or unknown label is None (no note).
    label: ContradictionLabel | None = None

    @field_validator("reason", mode="before")
    @classmethod
    def _lenient_reason(cls, v: Any) -> Any:
        return v if v in ("same_position", "different_subject", "different_time", "absence", "other") else None

    @field_validator("label", mode="before")
    @classmethod
    def _lenient_label(cls, v: Any) -> Any:
        return v if v in ("same_conditions", "may_differ_by_edition_or_date") else None

    @model_validator(mode="after")
    def _reason_matches_verdict(self) -> "ContradictionVerdict":
        if self.verdict == "real":
            self.reason = None
        else:
            self.label = None
            if self.reason is None:
                self.reason = "other"
        return self


class CitationCheck(_Lenient):
    """The merge:check response. Holds ids, verdicts and reason codes only -- never text."""
    points: list[PointVerdict] = Field(default_factory=list)
    contradictions: list[ContradictionVerdict] = Field(default_factory=list)


class Retraction(_Lenient):
    """The merge:retract response (IDEA-297): the point with its unsupported words deleted, or None
    when nothing can be deleted. Code accepts it only as a strict word-level deletion of the
    original (pipeline/stages/merge.py::_is_strict_deletion), then re-checks it."""
    text: str | None = None


class MergedSection(_Lenient):
    title: str = Field(min_length=1)
    subsections: list[Subsection]
    contradictions: list[Contradiction] = Field(default_factory=list)


class MergedDocument(_Lenient):
    """What merges.merged_json stores. source_map: token -> sources.id."""
    tldr: list[NonEmptyStr]
    sections: list[MergedSection]
    source_map: dict[str, int]
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    # token -> len(Analysis.key_claims) at merge time. For a hashed merge it only feeds the error
    # message ("N claims then, M now"); for a legacy merge with no claim_hashes it is the only
    # (count-only) check the renderer can still make.
    claim_counts: dict[str, int] = Field(default_factory=dict)
    # token -> claims_hash(Analysis) at merge time, for every source_map token. Lets the renderer
    # refuse a merge whose sources were re-analyzed after it ran (`analyze --force`), where
    # claim_refs would otherwise resolve against different claims with no error -- including a
    # re-analysis that keeps the same claim count. Defaulted so a document stored before this
    # field existed still loads; the renderer falls back to claim_counts for it and run_output
    # warns that a same-count re-analysis cannot be detected.
    claim_hashes: dict[str, str] = Field(default_factory=dict)
    # Retraction counts of this merge (IDEA-297): {"attempted", "accepted", "rejected"}; empty for a
    # merge stored before retraction existed.
    retraction: dict[str, int] = Field(default_factory=dict)

    @field_validator("source_map")
    @classmethod
    def _source_map(cls, v: dict[str, int]) -> dict[str, int]:
        bad_keys = [k for k in v if not SOURCE_TOKEN_RE.match(k)]
        if bad_keys:
            raise ValueError(f"source_map keys must look like S1, S2, ...; got {bad_keys}")
        bad_vals = {k: val for k, val in v.items() if val < 1}
        if bad_vals:
            raise ValueError(f"source_map values must be >= 1; got {bad_vals}")
        return v

    @field_validator("claim_counts")
    @classmethod
    def _claim_counts(cls, v: dict[str, int]) -> dict[str, int]:
        bad_keys = [k for k in v if not SOURCE_TOKEN_RE.match(k)]
        if bad_keys:
            raise ValueError(f"claim_counts keys must look like S1, S2, ...; got {bad_keys}")
        bad_vals = {k: val for k, val in v.items() if val < 0}
        if bad_vals:
            raise ValueError(f"claim_counts values must be >= 0; got {bad_vals}")
        return v

    @field_validator("claim_hashes")
    @classmethod
    def _claim_hashes(cls, v: dict[str, str]) -> dict[str, str]:
        bad_keys = [k for k in v if not SOURCE_TOKEN_RE.match(k)]
        if bad_keys:
            raise ValueError(f"claim_hashes keys must look like S1, S2, ...; got {bad_keys}")
        bad_vals = {k: val for k, val in v.items() if not SHA256_HEX_RE.match(val)}
        if bad_vals:
            raise ValueError(f"claim_hashes values must be lowercase SHA-256 hex digests; got {bad_vals}")
        return v
