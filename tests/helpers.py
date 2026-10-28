"""Scripted LLMProvider and stub transcriber for end-to-end tests.

`ScriptedProvider` answers every one of the pipeline's five prompt shapes (analyze, outline,
section, verification, citation check -- every point "supported", every ref "related", every
contradiction "real") by
recognising markers/phrases the real prompt builders always emit -- no network, no mocking of our
own code. It cites claim 0 of every source token it sees. For the
analyze prompt it reads the numbered segments ("[n] text", IDEA-294) out of the TRANSCRIPT data
block and points its one claim at a segment range (segment_from/segment_to), never at quote text:
segment 1 when the transcript has 2+ segments (so the anchor lands away from t=0 -- with
StubTranscriber that is t=5s), else segment 0, extended over following segments until the range
holds at least 8 words, so the anchor floors (pipeline/anchors.py) keep the claim.

`StubTranscriber` returns a short, video-id-flavoured transcript so claims/quotes differ per
source. `fake_metadata` is a drop-in replacement for
`pipeline.stages.fetch.fetch_source_metadata` that needs no network.
"""
from __future__ import annotations

import contextlib
import json
import re
import threading
from types import SimpleNamespace

from pipeline.llm.base import LLMResult
from pipeline.prompts.merge import SOURCE_BLOCK_RE
from pipeline.sources import KINDS, SourceRef
from pipeline.transcription.base import Segment, Transcript, watch_url

# pipeline/prompts/common.py::data_block wraps content as
# "<<<HMD_DATA LABEL <8 hex nonce>>>>\n<content>\n<<<END_HMD_DATA LABEL <nonce>>>>" -- the nonce
# is derived from the content itself, so it must be captured and echoed back via backreference
# rather than guessed.
_TRANSCRIPT_RE = re.compile(
    r"<<<HMD_DATA TRANSCRIPT ([0-9a-f]{8})>>>\n(.*?)\n<<<END_HMD_DATA TRANSCRIPT \1>>>", re.S
)

_SEGMENT_LINE_RE = re.compile(r"^\[(\d+)\] (.*)$")

# The exact phrases pipeline/prompts/merge.py uses to introduce each of its three prompt shapes.
_OUTLINE_MARKER = "You will design the outline of a research document"
_SECTION_MARKER = "The section to write follows as data"
_VERIFICATION_MARKER = "Below is every point of a research document"
# pipeline/prompts/check.py's opening phrase and its single data block.
_CHECK_MARKER = "You are checking the citations of one section"
_CHECK_RE = re.compile(r"<<<HMD_DATA CHECK ([0-9a-f]{8})>>>\n(.*?)\n<<<END_HMD_DATA CHECK \1>>>", re.S)


def check_payload(prompt: str) -> dict:
    """The JSON inside a citation-check prompt's CHECK data block: {"section", "points": [{"point",
    "text", "refs": [{"ref", "source_title", "upload_date", "quote"}]}], "contradictions": [...]}."""
    m = _CHECK_RE.search(prompt)
    if m is None:
        raise AssertionError("check prompt carries no CHECK data block")
    return json.loads(m.group(2))


_RETRACT_RE = re.compile(r"<<<HMD_DATA RETRACT ([0-9a-f]{8})>>>\n(.*?)\n<<<END_HMD_DATA RETRACT \1>>>", re.S)


def retract_payload(prompt: str) -> dict:
    """The JSON inside a retraction prompt's RETRACT data block: {"section", "point", "quotes":
    [{"source_title", "upload_date", "quote"}]} (pipeline/prompts/retract.py)."""
    m = _RETRACT_RE.search(prompt)
    if m is None:
        raise AssertionError("retract prompt carries no RETRACT data block")
    return json.loads(m.group(2))


def all_supported(prompt: str) -> dict:
    """A check answer that judges every point "supported", every ref "related" and every
    contradiction "real" (the per-point shape of pipeline/schemas.py::CitationCheck)."""
    payload = check_payload(prompt)
    return {"points": [{"point": p["point"], "refs": [{"ref": r["ref"], "relevance": "related"} for r in p["refs"]],
                        "verdict": "supported"} for p in payload["points"]],
            "contradictions": [{"contradiction": c["contradiction"], "verdict": "real"}
                               for c in payload["contradictions"]]}


class ScriptedProvider:
    """Deterministic stand-in for a real LLMProvider (pipeline/llm/base.py's Protocol)."""

    name = "anthropic_api"

    def __init__(self, cost_per_call: float | None = None, model: str = "claude-haiku-4-5-20251001"):
        self.cost_per_call = cost_per_call
        self.model = model
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def complete(self, prompt: str, *, model: str, system: str | None = None,
                 max_tokens: int = 16000) -> LLMResult:
        with self._lock:
            self.prompts.append(prompt)
        body = self._answer(prompt)
        return LLMResult(
            text=json.dumps(body),
            model=model or self.model,
            provider=self.name,
            input_tokens=100,
            output_tokens=50,
            cost_usd=self.cost_per_call,
            stop_reason="end_turn",
            duration_ms=1,
        )

    def _answer(self, prompt: str) -> dict:
        m = _TRANSCRIPT_RE.search(prompt)
        if m is not None:
            return self._analyze_answer(m.group(2))
        if _OUTLINE_MARKER in prompt:
            return self._outline_answer(prompt)
        if _SECTION_MARKER in prompt:
            return self._section_answer(prompt)
        if _VERIFICATION_MARKER in prompt:
            return {"tldr": ["First takeaway", "Second takeaway", "Third takeaway"], "duplicates": []}
        if _CHECK_MARKER in prompt:
            # Every point is supported, every cited quote related, every contradiction real, so the
            # end-to-end report keeps everything the section calls produced.
            return all_supported(prompt)
        if _RETRACT_RE.search(prompt):
            # Never reached while every point is supported; answer "nothing to delete" if it is.
            return {"text": None}
        raise AssertionError(f"unrecognised prompt (first 200 chars): {prompt[:200]!r}")

    @staticmethod
    def _analyze_answer(transcript_block: str) -> dict:
        # Each line of the block is "[n] text" (pipeline/prompts/analyze.py::number_segments).
        segments = [m.group(2) for m in map(_SEGMENT_LINE_RE.match, transcript_block.splitlines()) if m]
        if not segments:
            raise AssertionError("analyze prompt carries no numbered segments")
        first = 1 if len(segments) >= 2 else 0
        last = first
        while last + 1 < len(segments) and len(" ".join(segments[first:last + 1]).split()) < 8:
            last += 1
        text = " ".join(segments[first:last + 1])
        claim = f"The video reports: {text}"
        return {
            "topics": ["topic"],
            "summary": f"Summary covering: {text}",
            "key_claims": [{"claim": claim, "segment_from": first, "segment_to": last, "confidence": "high"}],
            "insights": ["an insight"],
            "facts": ["a fact"],
            "caveats": [],
            "references": [],
        }

    @staticmethod
    def _outline_answer(prompt: str) -> dict:
        tokens = SOURCE_BLOCK_RE.findall(prompt)
        if not tokens:
            raise AssertionError("outline prompt carries no source tokens")
        if len(tokens) == 1:
            # A single source gets one section, not two identical halves -- a duplicate "Part
            # two" citing the same lone token would make a one-source merge produce two
            # identical points downstream.
            return {"sections": [{"title": "Part one", "scope": "the only source", "sources": tokens}]}
        half = (len(tokens) + 1) // 2
        part_one, part_two = tokens[:half], tokens[half:]
        return {
            "sections": [
                {"title": "Part one", "scope": "first half of the sources", "sources": part_one},
                {"title": "Part two", "scope": "second half of the sources", "sources": part_two},
            ]
        }

    @staticmethod
    def _section_answer(prompt: str) -> dict:
        tokens = SOURCE_BLOCK_RE.findall(prompt)
        if not tokens:
            raise AssertionError("section prompt carries no source tokens")
        return {
            "subsections": [
                {
                    "title": "Findings",
                    "points": [
                        {
                            "text": f"a point supported by {tok}",
                            "sources": [tok],
                            "claim_refs": [{"source": tok, "claim": 0}],
                        }
                        for tok in tokens
                    ],
                }
            ],
            "contradictions": [],
        }


def yt_ref(video_id: str) -> SourceRef:
    """The SourceRef of a YouTube source, as transcribe_source builds it from a row."""
    return SourceRef(video_id, "youtube", watch_url(video_id))


class StubTranscriber:
    """Returns a short transcript whose text mentions the video id, so quotes/claims differ
    per source. Not network-backed; a drop-in for any real pipeline.transcription.* Transcriber."""

    name = "youtube_captions"

    kinds = frozenset(KINDS)

    def transcribe(self, ref: SourceRef, languages: list[str]) -> Transcript:
        return Transcript(
            ref.key,
            self.name,
            "en",
            [
                Segment(0.0, 5.0, f"This is the opening of video {ref.key} about the research topic"),
                Segment(5.0, 12.0, f"and here the speaker of {ref.key} makes the main point with a number of 30 percent"),
                Segment(12.0, 20.0, "finally a closing remark that wraps everything up nicely for the audience"),
            ],
        )


def fake_metadata(db_path: str, source_id: int, log=print) -> str:
    """Drop-in replacement for pipeline.stages.fetch.fetch_source_metadata: no network.

    Mirrors that function's shape (pipeline/stages/fetch.py:203-210): raises if the source row
    doesn't exist, and returns "skipped" rather than re-fetching when the row already looks done.
    """
    from pipeline import db

    src = db.get_source(db_path, source_id)
    if not src:
        raise ValueError(f"source {source_id} not found")
    if src["fetch_status"] == "done" and src["title"] and src["upload_date"]:
        return "skipped"
    db.update_source_metadata(
        db_path,
        source_id,
        title=f"Video {src['video_id'][:4]}",
        channel="Channel",
        duration_seconds=20,
        upload_date="20260101",
    )
    return "done"


def fake_message(text: str, model: str = "claude-haiku-5-5", input_tokens: int = 10, output_tokens: int = 5,
                 stop: str = "end_turn"):
    """An anthropic Message as the SDK returns it, for the fields the providers read."""
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                            cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop, model=model,
                           usage=usage)


def scripted_answer(provider):
    """FakeBatchClient answer from a ScriptedProvider: the same JSON it gives a direct call."""
    def answer(params: dict):
        r = provider.complete(params["messages"][0]["content"], model=params["model"],
                              system=params.get("system"), max_tokens=params["max_tokens"])
        return fake_message(r.text, model=r.model, input_tokens=r.input_tokens, output_tokens=r.output_tokens)
    return answer


class FakeBatchClient:
    """Stands in for anthropic.Anthropic in batch tests. `answer(params)` returns a message (see
    fake_message) or a result object whose `type` is "errored" / "expired" / "canceled". Direct calls
    (messages.create / messages.stream) answer at once; a batch ends after `polls_until_end` retrieve
    calls, or at once after a cancel."""

    def __init__(self, answer, polls_until_end: int = 1):
        self.answer = answer
        self.polls_until_end = polls_until_end
        self.created: list[list[dict]] = []
        self.cancelled: list[str] = []
        self.direct_calls = 0
        self._polls: dict[str, int] = {}
        self._lock = threading.Lock()
        self.messages = SimpleNamespace(
            create=self._create, stream=self._stream,
            batches=SimpleNamespace(create=self._batch_create, retrieve=self._retrieve,
                                    results=self._results, cancel=self._cancel))

    def _create(self, **params):
        with self._lock:
            self.direct_calls += 1
        return self.answer(params)

    @contextlib.contextmanager
    def _stream(self, **params):
        yield SimpleNamespace(get_final_message=lambda: self._create(**params))

    def _counts(self, n: int, ended: bool):
        return SimpleNamespace(processing=0 if ended else n, succeeded=n if ended else 0, errored=0,
                               canceled=0, expired=0)

    def _batch_create(self, *, requests):
        with self._lock:
            self.created.append(list(requests))
            bid = f"msgbatch_{len(self.created)}"
            self._polls[bid] = 0
        return SimpleNamespace(id=bid, processing_status="in_progress",
                               request_counts=self._counts(len(requests), False))

    def _requests(self, bid: str) -> list[dict]:
        return self.created[int(bid.split("_")[1]) - 1]

    def _retrieve(self, bid: str):
        with self._lock:
            self._polls[bid] += 1
            ended = self._polls[bid] >= self.polls_until_end or bid in self.cancelled
        return SimpleNamespace(id=bid, processing_status="ended" if ended else "in_progress",
                               request_counts=self._counts(len(self._requests(bid)), ended))

    def _results(self, bid: str):
        out = []
        for req in self._requests(bid):
            if bid in self.cancelled:
                res = SimpleNamespace(type="canceled")
            else:
                ans = self.answer(req["params"])
                kind = getattr(ans, "type", None)
                res = ans if kind in ("errored", "expired", "canceled") else SimpleNamespace(type="succeeded", message=ans)
            out.append(SimpleNamespace(custom_id=req["custom_id"], result=res))
        return iter(out)

    def _cancel(self, bid: str):
        with self._lock:
            self.cancelled.append(bid)
        return SimpleNamespace(id=bid, processing_status="canceling")


def skill_files() -> dict[str, "pathlib.Path"]:
    """Short skill name ("run", "init", ...) -> its SKILL.md, in either layout: this repo keeps
    `.claude/skills/hmd-<name>/`, the public plugin export ships
    `skills/<name>/` so the plugin named hmd serves them as /hmd:<name>."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    private = sorted((root / ".claude" / "skills").glob("hmd-*/SKILL.md"))
    if private:
        return {p.parent.name[len("hmd-"):]: p for p in private}
    return {p.parent.name: p for p in sorted((root / "skills").glob("*/SKILL.md"))}
