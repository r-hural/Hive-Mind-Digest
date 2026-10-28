"""Output stage: render the merged document into the vault shape (IDEA-216):

    <vault>/<notebook-slug>/_notebook.md
    <vault>/<notebook-slug>/reports/<YYYY-MM-DD>-summary.md

Trust rules: source tokens resolve through the stored map (unknown token = error), confirmed_by
is computed (= number of distinct sources with an anchored claim reference on the point, never
copied from the model's `sources` list -- see pipeline/stages/merge.py's invariant), every point
links to the timestamp of a validated, anchored claim (IDEA-001), anchors are rendered on request
(IDEA-002). Titles and other LLM/YouTube-sourced text are untrusted (IDEA-230) and are escaped
before landing in a Markdown link label or table cell. Merge-internal source tokens (S1, S3, ...)
that leak into model-written text (TL;DR, point text, contradiction topic/positions) are replaced
with the source's channel or short title (SourceLabeler). A point the merge's citation check marked partly supported renders with
"(partly supported)" after its text. Output structure changes only here.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date
from typing import Callable

from pydantic import ValidationError

from pipeline import db
from pipeline.config import FORMATS
from pipeline.diff import DiffResult, diff_merges, point_key
from pipeline.links import kind_noun, locator_label, safe_url, source_link
from pipeline.schemas import Analysis, Anchor, MergedDocument, Point, claims_hash
from pipeline.sources import MEDIA_KINDS, YOUTUBE_ID_RE as VIDEO_ID_RE, kind_of
from pipeline.stages.stats import WORTH_WATCHING_MIN_UNIQUE, coverage, notebook_stats, worth_watching

# Bump this whenever the rendered Markdown structure changes (headings, bullet shape, table
# columns, ...); recorded into a merge's prompt_versions (pipeline/stages/merge.py) alongside the
# analyze/merge prompt versions so a golden notebook can detect a renderer change too.
RENDER_VERSION = "render-2026-10-18.3"

# Two sources' publication dates at least this far apart mean a contradiction may be about different times or editions.
DATE_GAP_DAYS = 90

# Rendered after a contradiction's topic when the citation check labelled it
# may_differ_by_edition_or_date (IDEA-013); same_conditions, or no label, renders unmarked.
MAY_DIFFER = "(may differ by edition or date)"

# Rendered right after the text of a point the citation check marked support="partial".
PARTLY_SUPPORTED = "(partly supported)"
DISPUTED = "(disputed)"
NEW_BADGE = "(new)"
BADGE_LEGEND = ("_Badges: (confirmed by N sources) = N sources make the point and its quotes support all of "
                "it; (N sources) = N sources are cited but the point is partly supported, disputed or "
                "merged from several points; "
                "(disputed) = another source contradicts a claim it rests on; (partly supported) = its "
                "quotes support only part of it; (new) = not in the previous report._")

# Longest transcript excerpt a rendered quote line may carry. The analyze prompt asks for 8-40
# word quotes (pipeline/prompts/analyze.py) but nothing validates it, and the recorded sample
# digest (2026-09-07) came back with 53 of 119 quotes over 40 words, the longest 93. A shipped
# report must keep excerpts short (IDEA-042 release checklist), so the cap is enforced where the
# excerpt is rendered, never on the stored claim: the anchor still resolves to the full quote.
MAX_QUOTE_WORDS = 40



class RenderError(Exception):
    pass


def report_path_for(vault_dir: str, slug: str, day: date | None = None, fmt: str = "summary") -> str:
    day = day or date.today()
    return os.path.join(vault_dir, slug, "reports", f"{day.isoformat()}-{fmt}.md")


def deep_link(video_id: str, t_start: float) -> str:
    if not VIDEO_ID_RE.match(video_id):
        raise RenderError(f"not a YouTube video id: {video_id!r}")
    return f"https://www.youtube.com/watch?v={video_id}&t={int(t_start)}s"


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _fmt_date(upload_date: str | None) -> str:
    if upload_date and len(upload_date) == 8 and upload_date.isdigit():
        return f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
    return upload_date or ""


def _parse_upload(upload_date: str | None) -> date | None:
    if upload_date and len(upload_date) == 8 and upload_date.isdigit():
        try:
            return date(int(upload_date[:4]), int(upload_date[4:6]), int(upload_date[6:]))
        except ValueError:
            return None
    return None


def _date_gap_note(a: dict, b: dict) -> str:
    da, dbb = _parse_upload(a.get("upload_date")), _parse_upload(b.get("upload_date"))
    if not da or not dbb or abs((da - dbb).days) < DATE_GAP_DAYS:
        return ""
    months = round(abs((da - dbb).days) / 30.4)
    return f" — published {months} months apart; one may be outdated or describe a different edition"


def _iso_date(src: dict) -> str:
    """A source's publication date as YYYY-MM-DD when it parses, else '' (the stored value is
    YouTube metadata: anything else is never echoed into a report)."""
    d = _parse_upload(src.get("upload_date"))
    return d.isoformat() if d else ""


def _dated(src: dict) -> str:
    """' (2026-01-02)' after a source link when its publication date parses, else ''."""
    d = _iso_date(src)
    return f" ({d})" if d else ""


def _collapse_newlines(text: str) -> str:
    """Collapse any run of whitespace (including bare \\r, tabs, multiple newlines) to a
    single space, so untrusted model/YouTube text can never inject a fake heading, table
    row, or bullet by embedding a newline."""
    return " ".join(text.split())


def _cap_quote(quote: str) -> str:
    """First MAX_QUOTE_WORDS words of a whitespace-collapsed quote, with an ellipsis glued to
    the last kept word when anything was cut (so the rendered line still counts as at most
    MAX_QUOTE_WORDS words -- tests/test_examples.py parses that shape). The quote is untrusted
    source text, so it is escaped like other inline prose (_md_inline)."""
    words = _md_inline(quote).split()
    if len(words) <= MAX_QUOTE_WORDS:
        return " ".join(words)
    return " ".join(words[:MAX_QUOTE_WORDS]) + "…"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


_HTML_RE = re.compile(r"[<>]|&(?=#?\w+;)")
_HTML_ENTITY = {"<": "&lt;", ">": "&gt;", "&": "&amp;"}


def _html_safe(text: str) -> str:
    """Neutralise inline HTML and entities in untrusted text: `<` and `>` always, `&` only where it
    would start an entity (`R&D` stays as written), so a title like `<a href=...>` is never a live
    tag in any renderer."""
    return _HTML_RE.sub(lambda m: _HTML_ENTITY[m.group(0)], text)


def _md_text(text: str) -> str:
    """For model-written or fetched prose that is not a link label or a table cell (a point, a
    TL;DR bullet, a heading, a position, a topic): collapse whitespace, then escape backslash, `[`
    and `]` (so `[x](u)` and `![](u)` are text, never a link or an image) and inline HTML."""
    text = _collapse_newlines(text).replace("\\", "\\\\")
    return _html_safe(text.replace("[", "\\[").replace("]", "\\]"))


def _md_escape(text: str) -> str:
    """For table cells and inline (non-link-label) text: collapse whitespace, then escape
    backslashes BEFORE pipes. Escaping `|` alone would turn a literal `\\|` in the source text
    into `\\\\|` -- CommonMark reads that as an escaped backslash followed by an *unescaped*
    pipe, which still splits the table cell. Escaping backslashes first keeps the encoding
    unambiguous."""
    text = _collapse_newlines(text).replace("\\", "\\\\")
    return _html_safe(text).replace("|", "\\|")


_BLOCK_MARKER_RE = re.compile(r"\A(?:([#>*+-])|(\d+)([.)]))")


def _md_inline(text: str) -> str:
    """For untrusted inline prose that is not a link label (a source label, a heading or paragraph
    made of a title or a model summary): collapse whitespace, escape backslash, `[`, `]` and `|`,
    and a leading block marker (`#`, `-`, `*`, `+`, `>` or a `1.`/`1)` ordinal), so the text can
    never open a link, split a table cell, or start a heading, list or quote."""
    text = _collapse_newlines(text).replace("\\", "\\\\")
    text = text.replace("[", "\\[").replace("]", "\\]").replace("|", "\\|")
    m = _BLOCK_MARKER_RE.match(text)      # judged before the HTML escape, which would hide a leading ">"
    if m and m.group(1):
        return "\\" + text[0] + _html_safe(text[1:])
    if m:
        return f"{m.group(2)}\\{m.group(3)}{_html_safe(text[m.end():])}"
    return _html_safe(text)


def _md_link_text(text: str) -> str:
    """For `[text](url)` labels: collapse whitespace, then escape backslashes BEFORE `[`/`]`.
    Escaping `]` alone would turn a literal `\\]` in the source text into `\\\\]` -- CommonMark
    reads the doubled backslash as an escaped backslash and leaves the `]` unescaped, closing
    the link label early and letting whatever follows (e.g. `(javascript:...)`) become the new
    link's target. Escaping backslashes first keeps the encoding unambiguous. Titles come from
    YouTube and are untrusted (IDEA-230)."""
    text = _collapse_newlines(text).replace("\\", "\\\\")
    return _html_safe(text.replace("[", "\\[").replace("]", "\\]"))


# A bare merge-internal source token ("S3", never "S3x", "MS3" or "S3_"). Only tokens present in
# the merge's source_map are ever replaced; anything else (S10 in a 3-source notebook) is left alone.
_LEAKED_TOKEN_RE = re.compile(r"(?<![\w])S[1-9]\d*(?![\w])")
# A product name that looks like a token is left alone: "Galaxy S25", "Samsung S24", "S25 Ultra",
# "S24+", "S25 FE" are phones, not sources.
_PRODUCT_BEFORE_RE = re.compile(r"(?:galaxy|samsung|note|tab)\s+\Z", re.I)
_PRODUCT_AFTER_RE = re.compile(r"\A(?:\+|\s+(?:ultra|plus|edge|fe|lite|pro|max|mini)\b)", re.I)
SHORT_TITLE_CHARS = 40


def _short_title(title: str) -> str:
    title = _collapse_newlines(title)
    if len(title) <= SHORT_TITLE_CHARS:
        return title
    cut = title[:SHORT_TITLE_CHARS].rsplit(" ", 1)[0] or title[:SHORT_TITLE_CHARS]
    return cut + "…"


class SourceLabeler:
    """Replaces leaked merge-internal source tokens in model-written text with a human label: the
    source's channel when no other source in the merge shares it, else its short title (first
    SHORT_TITLE_CHARS characters on a word boundary), else its video id. The replacement runs on
    the raw text, before the caller's own escaping (_collapse_newlines/_md_escape), so a label is
    escaped exactly like the text around it; a single re.sub pass means a label that itself
    contains "S2" is never re-substituted."""

    def __init__(self, source_map: dict[str, int], sources_by_id: dict[int, dict]):
        rows = {tok: sources_by_id[sid] for tok, sid in source_map.items() if sid in sources_by_id}
        channels = [(r.get("channel") or "").strip() for r in rows.values()]
        self.labels: dict[str, str] = {}
        for tok, r in rows.items():
            channel = (r.get("channel") or "").strip()
            if channel and channels.count(channel) == 1:
                label = _collapse_newlines(channel)
            elif r.get("title"):
                label = _short_title(r["title"])
            else:
                label = r.get("video_id") or tok
            self.labels[tok] = label

    def __call__(self, text: str) -> str:
        def sub(m: re.Match) -> str:
            tok = m.group(0)
            if tok not in self.labels:
                return tok
            if _PRODUCT_BEFORE_RE.search(text, 0, m.start()) or _PRODUCT_AFTER_RE.match(text[m.end():]):
                return tok
            return self.labels[tok]
        return _LEAKED_TOKEN_RE.sub(sub, text)


class _Resolver:
    def __init__(self, doc: MergedDocument, sources_by_id: dict[int, dict], slug: str | None = None):
        self.map = doc.source_map
        self.sources_by_id = sources_by_id
        self.claim_counts = doc.claim_counts
        self.claim_hashes = doc.claim_hashes
        # A merge stored before claim_hashes existed carries none at all: it falls back to the
        # claim_counts check (see analysis()) instead of being refused, which would force a billed
        # `hmd analyze` just to re-render. A document with *some* hashes is held to all of them.
        self.legacy = not doc.claim_hashes
        self.rerun = f"re-run `hmd analyze {slug or '<notebook>'}`"
        self._analyses: dict[str, Analysis] = {}

    def source(self, token: str) -> dict:
        sid = self.map.get(token)
        if sid is None or sid not in self.sources_by_id:
            raise RenderError(f"merge document cites unknown source token {token}")
        return self.sources_by_id[sid]

    def analysis(self, token: str) -> Analysis:
        """The source's analysis, refused unless its claims_hash still matches the one stored at
        merge time; a hashed merge missing the token's hash is corrupt and refused too. A legacy
        merge (no claim_hashes at all) is only refused on a claim_counts mismatch, as before
        hashes existed -- a same-count re-analysis goes undetected there (run_output warns)."""
        if token not in self._analyses:
            raw = self.source(token).get("analysis_json")
            if not raw:
                raise RenderError(f"source {token} has no analysis")
            try:
                analysis = Analysis.model_validate_json(raw)
            except ValidationError as e:
                raise RenderError(
                    f"source {token} has an unreadable analysis; re-run analyze for it: {e}") from None
            then, now = self.claim_counts.get(token), len(analysis.key_claims)
            if self.legacy:
                if then is not None and then != now:
                    raise RenderError(
                        f"source {token} was re-analyzed after the merge ({then} claims then, "
                        f"{now} now); {self.rerun}")
            else:
                stored = self.claim_hashes.get(token)
                if stored is None:
                    raise RenderError(
                        f"the stored merge has no claim hash for source {token} (it is corrupt), so "
                        f"it cannot be checked against the current analyses; {self.rerun} to "
                        f"rebuild the merge")
                if stored != claims_hash(analysis):
                    detail = (f"{then} claims then, {now} now" if then is not None and then != now
                              else "claim content changed")
                    raise RenderError(
                        f"source {token} was re-analyzed after the merge ({detail}); {self.rerun}")
            self._analyses[token] = analysis
        return self._analyses[token]

    def anchored_refs(self, point: Point) -> list[tuple[dict, Anchor, str]]:
        """(source row, anchor, quote) for the first ref of each distinct cited source, in
        order. Every claim_ref on a rendered point is expected to be a validated, anchored
        claim (the merge stage validates indices; the analyze stage strips unanchored claims
        before merge ever sees them), so an out-of-range index or a missing anchor here means the stored document is corrupt
        -- that is a RenderError, never a silent skip."""
        out = []
        seen: set[str] = set()
        for ref in point.claim_refs:
            if ref.source in seen:
                continue
            claims = self.analysis(ref.source).key_claims
            if not (0 <= ref.claim < len(claims)):
                raise RenderError(
                    f"point {point.id or ''} cites {ref.source} claim {ref.claim} which does not exist")
            if claims[ref.claim].anchor is None:
                raise RenderError(
                    f"point {point.id or ''} cites {ref.source} claim {ref.claim} which is not anchored")
            seen.add(ref.source)
            out.append((self.source(ref.source), claims[ref.claim].anchor, claims[ref.claim].quote))
        if not out:
            raise RenderError(f"point {point.id or ''} {point.text[:60]!r} has no anchored claim reference")
        return out


_ANCHOR_DROP_RE = re.compile(r"[^\w\- ]", re.UNICODE)


def _heading_anchor(heading: str) -> str:
    """GitHub-style heading id: lowercase, drop punctuation, spaces to hyphens. "1. Basics &
    Setup" -> "1-basics--setup". Obsidian also resolves these for ASCII headings."""
    return _ANCHOR_DROP_RE.sub("", _collapse_newlines(heading).lower()).replace(" ", "-")


def count_badge(p: Point, n_refs: int, *, disputed: bool) -> tuple[str, bool] | None:
    """(badge text without parentheses, confirmed?) for a point citing more than one source, else None.
    IDEA-307b: "confirmed" only for a joint point the citation check found fully supported and that no
    contradiction disputes; otherwise the sources are only counted, since a joint point whose sources
    say different things must not read as replication. Refs a cross-section fold added were checked
    against another point's text, so such a point is not "confirmed" either."""
    if n_refs <= 1:
        return None
    confirmed = p.support is None and not disputed and not p.refs_from_fold
    return (f"confirmed by {n_refs} sources" if confirmed else f"{n_refs} sources"), confirmed


def flag_badges(p: Point, *, disputed: bool, new: bool) -> list[str]:
    """The point's flag badges in render order, e.g. ["(partly supported)", "(new)"]. PARTLY_SUPPORTED
    is code-filled by the citation check (merge:check): no cited quote fully supports the point, so the
    reader is told before the links, never silently."""
    return [b for b, on in ((PARTLY_SUPPORTED, p.support == "partial"), (DISPUTED, disputed),
                            (NEW_BADGE, new)) if on]


def _point_line(p: Point, res: "_Resolver", label: SourceLabeler, *, disputed: bool, new: bool):
    """The bullet line of a point and its anchored refs (for show_anchors quote lines)."""
    refs = res.anchored_refs(p)
    links = ", ".join(
        f"[{_md_link_text(src['title'] or src['video_id'])} {locator_label(src, a)}]({source_link(src, a, quote)})"
        for src, a, quote in refs)
    count = count_badge(p, len(refs), disputed=disputed)
    suffix = f" ({count[0]})" if count else ""
    badges = "".join(f" {b}" for b in flag_badges(p, disputed=disputed, new=new))
    return f"- {_md_text(label(p.text))}{badges} — {links}{suffix}", refs


def _contradiction_note(c) -> str:
    return f" {MAY_DIFFER}" if getattr(c, "check_label", None) == "may_differ_by_edition_or_date" else ""


def _contradiction_line(c, res: "_Resolver", label: SourceLabeler) -> str:
    a, b = res.source(c.source_a), res.source(c.source_b)
    return (f"**Contradiction — {_md_text(label(c.topic))}{_contradiction_note(c)}:** "
            f"[{_md_link_text(a['title'] or a['video_id'])}]({safe_url(a['url'])}){_dated(a)} claims \"{_md_text(label(c.position_a))}\" "
            f"while [{_md_link_text(b['title'] or b['video_id'])}]({safe_url(b['url'])}){_dated(b)} claims \"{_md_text(label(c.position_b))}\""
            + _date_gap_note(a, b))


@dataclass
class SummaryPrep:
    """What the summary renderers (Markdown and HTML) compute before writing a line."""
    res: "_Resolver"
    stats: object
    disputed: set
    tok_of: dict
    label: SourceLabeler
    counts_text: str
    channels: set
    missing: list


def prepare_summary(doc: MergedDocument, sources_by_id: dict[int, dict], slug: str | None = None) -> SummaryPrep:
    """Resolve and check every source of the merge, compute the stats and the header counts."""
    res = _Resolver(doc, sources_by_id, slug)
    analyzed_ids = set(doc.source_map.values())
    # Resolve every source_map token up front, the same way every other citation is resolved:
    # an unknown token is a RenderError here too, never a silently-dropped count.
    analyzed = [res.source(t) for t in doc.source_map]
    # Check every source's claim hash up front, not only the ones a point cites: the outline,
    # TL;DR and contradictions were written from all of them, so any re-analysis since the merge
    # makes the whole document stale. A legacy (pre-hash) merge keeps its old lazy, cited-only
    # check so a document that rendered before still renders.
    if not res.legacy:
        for t in doc.source_map:
            res.analysis(t)
    counts: dict[str, int] = {}
    for t in doc.source_map:
        try:
            counts[t] = len(res.analysis(t).key_claims)
        except RenderError:
            if not res.legacy:
                raise
            counts[t] = doc.claim_counts.get(t, 0)   # legacy merge: keep its lazy, cited-only check
    stats = notebook_stats(doc, counts)
    disputed = {id(ps.point) for ps in stats.points if ps.disputed}
    tok_of = {sid: tok for tok, sid in doc.source_map.items()}
    channels = {s["channel"] for s in analyzed if s.get("channel")}
    kinds: dict[str, int] = {}
    for s in analyzed:
        k = kind_of(s)
        k = "video" if k == "youtube" else k   # YouTube and other video sites share one noun
        kinds[k] = kinds.get(k, 0) + 1
    noun_order = [k for k in ("video", "media", "web", "pdf", "text") if k in kinds]
    noun_order += [k for k in kinds if k not in noun_order]   # an unexpected kind counts as a plain "source"
    counts_text = ", ".join(f"{n} {kind_noun(k, n)}" for k, n in ((k, kinds[k]) for k in noun_order))
    label = SourceLabeler(doc.source_map, sources_by_id)
    missing = [s for sid, s in sorted(sources_by_id.items()) if sid not in analyzed_ids]
    return SummaryPrep(res=res, stats=stats, disputed=disputed, tok_of=tok_of, label=label,
                       counts_text=counts_text, channels=channels, missing=missing)


def render_markdown(topic: str, doc: MergedDocument, sources_by_id: dict[int, dict], *,
                    show_anchors: bool = False, today: date | None = None, slug: str | None = None,
                    new_point_ids: set[str] | None = None,
                    extra_before_sources: list[str] | None = None,
                    title_suffix: str = "Research Summary") -> str:
    """`slug` only names the notebook in a re-run hint when the merge is refused."""
    today = today or date.today()
    prep = prepare_summary(doc, sources_by_id, slug)
    res, stats, disputed, tok_of, label = prep.res, prep.stats, prep.disputed, prep.tok_of, prep.label
    counts_text, channels, missing = prep.counts_text, prep.channels, prep.missing
    new_ids = new_point_ids or set()
    flat_i = -1   # index over all points in document order: the key the diff uses for id-less points
    lines = [f"# {_md_text(topic)} — {title_suffix}",
             f"Generated: {today.isoformat()} | Sources: {counts_text or '0 videos'}, "
             f"{_plural(len(channels), 'channel')}", "",
             "## TL;DR"]
    lines += [f"- {_md_text(label(b))}" for b in doc.tldr]
    if not doc.tldr:
        # A contradictions-only merge (pipeline/stages/merge.py) has no TL;DR. Not a "- " bullet,
        # so eval/bullet parsers never count it as a finding.
        lines.append("_No TL;DR: no point survived claim validation; only contradictions are reported below._")
    lines.append("")
    lines += [BADGE_LEGEND, "", "## Contents", ""]
    for i, section in enumerate(doc.sections, 1):
        heading = f"{i}. {_collapse_newlines(section.title)}"
        lines.append(f"{i}. [{_md_link_text(section.title)}](#{_heading_anchor(heading)})")
    tail = ["Sources"] + (["Sources not analyzed"] if missing else []) + ["Source scorecard", "Coverage"]
    if any(sec.contradictions for sec in doc.sections):
        tail.append("Contradictions")
    for k, name in enumerate(tail, len(doc.sections) + 1):
        lines.append(f"{k}. [{name}](#{_heading_anchor(name)})")
    lines.append("")

    for i, section in enumerate(doc.sections, 1):
        lines.append(f"## {i}. {_md_text(section.title)}")
        lines.append("")
        srcs = stats.section_sources[i - 1]
        if srcs:
            names = ", ".join(f"{_md_inline(label.labels.get(tok_of[sid], tok_of[sid]))} ({n})" for sid, n in srcs)
            lines += [f"_Sources: {names}_", ""]
        for j, sub in enumerate(section.subsections, 1):
            lines.append(f"### {i}.{j} {_md_text(sub.title)}")
            for p in sub.points:
                flat_i += 1
                line, refs = _point_line(p, res, label, disputed=id(p) in disputed,
                                         new=point_key(p, flat_i) in new_ids)
                lines.append(line)
                if show_anchors:
                    for idx, (src, anchor, quote) in enumerate(refs):
                        if idx > 0:
                            lines.append("  >")   # separates quotes so they don't merge into one paragraph
                        title = _md_inline(src["title"] or src["video_id"])
                        lines.append(f'  > "{_cap_quote(quote)}" — {title} {locator_label(src, anchor)}')
            lines.append("")
        for c in section.contradictions:
            lines.append(_contradiction_line(c, res, label))
            lines.append("")

    lines += extra_before_sources or []
    lines += ["## Sources", "", "| # | Kind | Title | Channel | Published | Length | URL |", "|---|---|---|---|---|---|---|"]
    for token, sid in doc.source_map.items():
        s = res.source(token)
        dur = fmt_time(s["duration_seconds"]) if s.get("duration_seconds") and kind_of(s) in MEDIA_KINDS else ""
        lines.append(f"| {token} | {kind_of(s)} | {_md_inline(s['title'] or '')} | {_md_inline(s['channel'] or '')} | "
                     f"{_fmt_date(s.get('upload_date'))} | {dur} | {safe_url(s['url'])} |")
    lines.append("")

    if missing:
        lines += ["## Sources not analyzed", ""]
        for s in missing:
            reason = s.get("error_message") or f"{s.get('transcribe_status')}/{s.get('analyze_status')}"
            lines.append(f"- [{_md_link_text(s['title'] or s['video_id'])}]({safe_url(s['url'])}) — {_md_inline(reason)}")
        lines.append("")

    stars = worth_watching(stats)
    lines += ["## Source scorecard", "",
              "| Source | Points | Unique | Shared | Claims used | Contradictions | Worth watching |",
              "|---|---|---|---|---|---|---|"]
    for token, sid in doc.source_map.items():
        s, src_row = stats.sources[sid], res.source(token)
        lines.append(f"| {_md_inline(src_row['title'] or src_row['video_id'])} | {s.points} | {s.unique_points} | "
                     f"{s.shared_points} | {s.claims_cited} of {s.claims} | {s.contradictions} | "
                     f"{'★' if sid in stars else ''} |")
    lines += ["", "_Unique: points no other source makes. Worth watching: at least "
              f"{WORTH_WATCHING_MIN_UNIQUE} unique points and in the top third of sources by them._", ""]
    cov = coverage(stats)
    pct = round(100 * cov.single_source / cov.points) if cov.points else 0
    lines += ["## Coverage", "", f"- Single-source points: {cov.single_source} of {cov.points} ({pct}%)"]
    thin = [f"{i}. {_collapse_newlines(sec.title)}" for i, (sec, (n_src, _, _)) in
            enumerate(zip(doc.sections, cov.sections), 1) if n_src == 1]
    if thin:
        lines.append(f"- Sections drawn from one source: {_md_inline('; '.join(thin))}")
    lines += ["", "| Section | Sources | Points | Single-source |", "|---|---|---|---|"]
    for i, (sec, (n_src, n_pts, n_single)) in enumerate(zip(doc.sections, cov.sections), 1):
        lines.append(f"| {i}. {_md_inline(sec.title)} | {n_src} | {n_pts} | {n_single} |")
    lines.append("")

    all_c = [c for sec in doc.sections for c in sec.contradictions]
    if all_c:
        lines += ["## Contradictions", "", "| Topic | Position A | Source | Position B | Source | Dates |", "|---|---|---|---|---|---|"]
        for c in all_c:
            a, b = res.source(c.source_a), res.source(c.source_b)
            lines.append(f"| {_md_inline(label(c.topic) + _contradiction_note(c))} | {_md_inline(label(c.position_a))} | {_md_inline(a['title'] or '')} | "
                         f"{_md_inline(label(c.position_b))} | {_md_inline(b['title'] or '')} | "
                         f"{_iso_date(a) or '?'} / {_iso_date(b) or '?'} |")
        lines.append("")
    return "\n".join(lines)


def write_notebook_stub(vault_dir: str, notebook: dict, sources: list[dict], report_paths: list[str],
                        html_paths: list[str] | None = None) -> str:
    """Machine-owned stub; the knowledge layer (MILESTONE-3) adds edit preservation (DECISION-022)."""
    slug = notebook["slug"]
    # id/topic are quoted JSON string literals (valid double-quoted YAML) so a slug or topic
    # that looks like a YAML scalar -- "2026", "no", "null" -- is never misparsed as an int/bool.
    lines = ["---", f"id: {json.dumps(slug, ensure_ascii=False)}", "type: notebook",
             f"topic: {json.dumps(notebook['topic'], ensure_ascii=False)}",
             f"created: {str(notebook.get('created_at', ''))[:10]}", f"sources: {len(sources)}", "---", "",
             f"# {_md_text(notebook['topic'])}", "", "## Sources", ""]
    for s in sources:
        meta = _md_inline(", ".join(x for x in (s.get("channel"), _fmt_date(s.get("upload_date"))) if x))
        title = _md_link_text(s["title"] or s["video_id"])
        lines.append(f"- [{title}]({safe_url(s['url'])})" + (f" — {meta}" if meta else ""))
    lines += ["", "## Reports", ""]
    for rp in report_paths:
        lines.append(f"- [[reports/{os.path.splitext(os.path.basename(rp))[0]}]]")
    if html_paths:
        # the HTML twins of this render only (a file left by an earlier render is not listed)
        lines += ["", "## HTML reports", ""]
        for hp in html_paths:
            name = os.path.basename(hp)
            lines.append(f"- [{name}](reports/{name})")
    path = os.path.join(vault_dir, slug, "_notebook.md")
    _atomic_write(path, "\n".join(lines) + "\n")
    return path


def _atomic_write(path: str, content: str) -> None:
    # No fsync and no cross-writer race handling here (contrast with
    # pipeline/transcription/cache.py): output is produced by a single stage run, never by
    # concurrent writers racing on the same path.
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


_REPORT_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(summary|digest|hybrid|outline|quotes|diff)\.md\Z")


def _date_from_report_path(report_path: str) -> date:
    """Recover the report's own date from its filename, so re-rendering an older merge keeps
    that report dated the day it was generated instead of relabelling it with today's date."""
    m = _REPORT_FILENAME_RE.match(os.path.basename(report_path))
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            pass
    return date.today()


def _under_vault(vault_dir: str, path: str) -> bool:
    vault_abs = os.path.normcase(os.path.abspath(vault_dir))
    path_abs = os.path.normcase(os.path.abspath(path))
    try:
        return os.path.commonpath([vault_abs, path_abs]) == vault_abs
    except ValueError:   # different drives on Windows
        return False


@dataclass(frozen=True)
class RenderContext:
    """Everything a format renderer needs, built once per run_output call."""
    topic: str
    doc: MergedDocument
    sources_by_id: dict[int, dict]
    today: date
    slug: str
    show_anchors: bool
    db_path: str
    notebook_id: int
    merge_id: int | None = None
    diff: DiffResult | None = None
    previous: dict | None = None


# fmt -> fn(ctx) -> markdown, or None to skip the file. "summary" is written by run_output itself.
FORMAT_RENDERERS: dict[str, Callable[[RenderContext], str | None]] = {}


# Labels of the HTML twins in the links row, in format order.
HTML_LABELS = {"summary": "Summary", "digest": "Digest", "hybrid": "Hybrid", "outline": "Outline",
               "quotes": "Quotes", "diff": "What changed"}


def html_twin_path(md_path: str) -> str:
    """The `.html` twin of a `reports/<date>-<format>.md` report."""
    return os.path.splitext(md_path)[0] + ".html"


@dataclass
class OutputFiles:
    report_path: str                 # the summary Markdown
    html_paths: dict[str, str]       # format -> HTML twin written by this render


def run_output(db_path: str, notebook_id: int, vault_dir: str, *, show_anchors: bool = False,
               formats: list[str] | None = None, html: bool = False, log: Callable[[str], None] = print) -> str:
    return render_reports(db_path, notebook_id, vault_dir, show_anchors=show_anchors, formats=formats, html=html,
                          log=log).report_path


def _write_html_twins(ctx: RenderContext, written: dict[str, str], log: Callable[[str], None]) -> dict[str, str]:
    """Render and write the HTML twin of every Markdown report written by this render. A twin is never
    essential: one that fails prints a WARNING and is skipped, and the others link only the twins
    that rendered."""
    from pipeline.stages import html_render
    pages = {}
    for fmt in HTML_LABELS:
        if fmt not in written:
            continue
        try:
            page = html_render.render_summary_html(ctx) if fmt == "summary" else html_render.HTML_RENDERERS[fmt](ctx)
        except Exception as e:
            log(f"WARNING: the {fmt} HTML report failed ({type(e).__name__}: {e}); skipped")
            continue
        if page is not None:
            pages[fmt] = page
    out: dict[str, str] = {}
    for fmt, page in pages.items():
        target = html_twin_path(written[fmt])
        links = [(HTML_LABELS[f], os.path.basename(html_twin_path(written[f])), f == fmt) for f in pages]
        try:
            _atomic_write(target, html_render.render_page(page, links))
        except Exception as e:
            log(f"WARNING: the {fmt} HTML report could not be written ({type(e).__name__}: {e}); skipped")
            continue
        out[fmt] = target
        log(f"{HTML_LABELS[fmt]} HTML written to: {target}")
    return out


def render_reports(db_path: str, notebook_id: int, vault_dir: str, *, show_anchors: bool = False,
                   formats: list[str] | None = None, html: bool = False,
                   log: Callable[[str], None] = print) -> OutputFiles:
    """Write the summary and the requested formats as Markdown, then (html=True) their HTML twins."""
    formats = formats or ["summary"]
    for fmt in formats:
        if fmt not in FORMATS:
            raise RenderError(f"unknown report format {fmt!r}")
    merge_row = db.get_latest_merge(db_path, notebook_id)
    if not merge_row:
        raise RenderError(f"no merge found for notebook {notebook_id}")
    notebook = db.get_notebook(db_path, notebook_id)
    sources = db.get_notebook_sources(db_path, notebook_id)
    doc = MergedDocument.model_validate_json(merge_row["merged_json"])

    report_path = merge_row["report_path"]
    if report_path and not _under_vault(vault_dir, report_path):
        log(f"WARNING: stored report_path {report_path!r} is outside vault_dir {vault_dir!r}; "
            "using the default path instead")
        report_path = None
    report_path = report_path or report_path_for(vault_dir, notebook["slug"])

    today = _date_from_report_path(report_path)
    if not doc.claim_hashes:
        log(f"WARNING: this merge predates claim hashes, so a re-analysis that kept a source's claim "
            f"count cannot be detected; run `hmd analyze {notebook['slug']}` to re-pin it")
    sources_by_id = {s["id"]: s for s in sources}
    prev = db.get_previous_merge(db_path, notebook_id)
    diff = None
    if prev:
        try:
            diff = diff_merges(MergedDocument.model_validate_json(prev["merged_json"]), doc)
        except ValidationError:
            log("WARNING: the previous merge could not be read; no diff this time")
            prev = None
        except Exception as e:   # the diff is an optional extra: it never fails the summary
            log(f"WARNING: the diff against the previous merge failed ({type(e).__name__}: {e}); "
                "no diff and no (new) badges this time")
            prev = None
    content = render_markdown(notebook["topic"], doc, sources_by_id,
                              show_anchors=show_anchors, today=today, slug=notebook["slug"],
                              new_point_ids=set(diff.new_points) if diff else None)
    _atomic_write(report_path, content)
    log(f"Output written to: {report_path}")
    written = {"summary": report_path}
    ctx = RenderContext(topic=notebook["topic"], doc=doc, sources_by_id=sources_by_id, today=today,
                        slug=notebook["slug"], show_anchors=show_anchors, db_path=db_path,
                        notebook_id=notebook_id, diff=diff, previous=prev, merge_id=merge_row["id"])
    for fmt in formats:
        if fmt == "summary":
            continue
        fn = FORMAT_RENDERERS.get(fmt)
        if fn is None:
            log(f"format {fmt} not available yet; skipped")
            continue
        # An optional format never fails the run: the summary is already written and the
        # notebook stub must still list it.
        try:
            extra = fn(ctx)
            if extra is None:
                continue
            p = report_path_for(vault_dir, notebook["slug"], today, fmt)
            _atomic_write(p, extra)
        except Exception as e:
            log(f"WARNING: the {fmt} report failed ({type(e).__name__}: {e}); skipped")
            continue
        written[fmt] = p
        log(f"{fmt.capitalize()} written to: {p}")
    html_paths = _write_html_twins(ctx, written, log) if html else {}
    reports_dir = os.path.dirname(report_path)
    report_paths = sorted(
        os.path.join(reports_dir, f) for f in os.listdir(reports_dir) if _REPORT_FILENAME_RE.match(f))
    stub_path = write_notebook_stub(vault_dir, notebook, sources, report_paths, list(html_paths.values()))
    log(f"Notebook stub written to: {stub_path}")
    return OutputFiles(report_path=report_path, html_paths=html_paths)


from pipeline.stages import formats as _formats  # noqa: E402,F401  (registers digest/hybrid/outline/quotes)
