"""Extra report renderings of one stored merge (DECISION-031): digest (IDEA-018), hybrid
(IDEA-110), outline and quote bank (light IDEA-025 / IDEA-016). Pure rendering from the merge and
the analyses -- no LLM call. Same trust rules as the summary (pipeline/stages/output.py):
untrusted text escaped, leaked tokens labelled, quotes capped, every link to a validated anchor."""
from __future__ import annotations

from pipeline.diff import point_key
from pipeline.schemas import MergedDocument
from pipeline.sources import MEDIA_KINDS, kind_of
from pipeline.stages import output as out
from pipeline.stages.stats import notebook_stats


def _resolver(ctx):
    return out._Resolver(ctx.doc, ctx.sources_by_id, ctx.slug)


def _header(ctx, kind: str) -> list[str]:
    return [f"# {out._md_text(ctx.topic)} — {kind}",
            f"Generated: {ctx.today.isoformat()} | Sources: {out._plural(len(ctx.doc.source_map), 'video')}", ""]


def digest_chapters(ctx, level: int) -> list[str]:
    res = _resolver(ctx)
    counts = {t: len(res.analysis(t).key_claims) for t in ctx.doc.source_map}
    stats = notebook_stats(ctx.doc, counts)
    h = "#" * level
    lines: list[str] = []
    ordered = sorted(ctx.doc.source_map.items(), key=lambda kv: kv[1])
    for n, (tok, sid) in enumerate(ordered, 1):
        src, an = res.source(tok), res.analysis(tok)
        day = out._parse_upload(src.get("upload_date"))
        meta = " · ".join(x for x in (out._md_inline(src.get("channel") or ""), day.isoformat() if day else "",
                                      out.fmt_time(src["duration_seconds"])
                                      if src.get("duration_seconds") and kind_of(src) in MEDIA_KINDS else "",
                                      f"[{'watch' if kind_of(src) in MEDIA_KINDS else 'open'}]({out.safe_url(src['url'])})") if x)
        lines += [f"{h} {n}. {out._md_inline(src['title'] or src['video_id'])}", "", meta, "",
                  out._md_inline(an.summary), "", f"{h}# Key claims", ""]
        for c in sorted((c for c in an.key_claims if c.anchor), key=lambda c: (c.anchor.t_start, c.anchor.segment or 0)):
            lines.append(f"- {out._md_inline(c.claim)} — "
                         f"[{out.locator_label(src, c.anchor)}]({out.source_link(src, c.anchor, c.quote)})")
            if ctx.show_anchors:
                lines.append(f'  > "{out._cap_quote(c.quote)}"')
        s = stats.sources[sid]
        lines += ["", f"_In the summary: {s.points} points, {s.unique_points} only from this source._", ""]
    return lines


def render_digest(ctx) -> str:
    return "\n".join(_header(ctx, "Digest") + digest_chapters(ctx, level=2))


def render_hybrid(ctx) -> str:
    extra = ["## Per-source chapters", ""] + digest_chapters(ctx, level=3)
    return out.render_markdown(ctx.topic, ctx.doc, ctx.sources_by_id, show_anchors=ctx.show_anchors,
                               today=ctx.today, slug=ctx.slug, extra_before_sources=extra, title_suffix="Hybrid",
                               new_point_ids=set(ctx.diff.new_points) if ctx.diff else None)


def render_outline(ctx) -> str:
    label = out.SourceLabeler(ctx.doc.source_map, ctx.sources_by_id)
    lines = _header(ctx, "Outline") + ["## TL;DR"] + [f"- {out._md_text(label(b))}" for b in ctx.doc.tldr] + [""]
    for i, sec in enumerate(ctx.doc.sections, 1):
        lines += [f"## {i}. {out._md_text(sec.title)}", ""]
        for j, sub in enumerate(sec.subsections, 1):
            lines.append(f"### {i}.{j} {out._md_text(sub.title)}")
            for p in sub.points:
                names = []
                for ref in p.claim_refs:
                    name = label.labels.get(ref.source, ref.source)
                    if name not in names:
                        names.append(name)
                lines.append(f"- {out._md_text(label(p.text))} [{out._md_inline(', '.join(names))}]")
            lines.append("")
    return "\n".join(lines)


def render_quotes(ctx) -> str:
    res = _resolver(ctx)
    label = out.SourceLabeler(ctx.doc.source_map, ctx.sources_by_id)
    lines = _header(ctx, "Quote bank")
    for i, sec in enumerate(ctx.doc.sections, 1):
        lines += [f"## {i}. {out._md_text(sec.title)}", ""]
        for j, sub in enumerate(sec.subsections, 1):
            lines += [f"### {i}.{j} {out._md_text(sub.title)}", ""]
            for p in sub.points:
                lines.append(f"- **{out._md_text(label(p.text))}**")
                seen = set()
                for ref in p.claim_refs:
                    if (ref.source, ref.claim) in seen:
                        continue
                    seen.add((ref.source, ref.claim))
                    src, claims = res.source(ref.source), res.analysis(ref.source).key_claims
                    if not (0 <= ref.claim < len(claims)):
                        raise out.RenderError(
                            f"point {p.id or ''} cites {ref.source} claim {ref.claim} which does not exist")
                    claim = claims[ref.claim]
                    if claim.anchor is None:
                        raise out.RenderError(
                            f"point {p.id or ''} cites {ref.source} claim {ref.claim} which is not anchored")
                    lines.append(f'  > "{out._cap_quote(claim.quote)}" — '
                                 f"[{out._md_link_text(src['title'] or src['video_id'])} {out.locator_label(src, claim.anchor)}]"
                                 f"({out.source_link(src, claim.anchor, claim.quote)})")
                lines.append("")
    return "\n".join(lines)


def _title(ctx, sid: int) -> str:
    """A source's title as inline Markdown (escaped: it is YouTube text)."""
    src = ctx.sources_by_id.get(sid)
    return out._md_inline((src and (src.get("title") or src.get("video_id"))) or f"source {sid}")


def _cited_sids(doc, p) -> set[int]:
    return {doc.source_map[r.source] for r in p.claim_refs if r.source in doc.source_map}


def render_diff(ctx) -> str | None:
    """What changed against the previous merge (IDEA-012). None when there is no diff."""
    d, prev = ctx.diff, ctx.previous
    if d is None or prev is None:
        return None
    old = MergedDocument.model_validate_json(prev["merged_json"])
    res = _resolver(ctx)
    label = out.SourceLabeler(ctx.doc.source_map, ctx.sources_by_id)
    old_label = out.SourceLabeler(old.source_map, ctx.sources_by_id)
    flat = [p for sec in ctx.doc.sections for sub in sec.subsections for p in sub.points]
    pts = {point_key(p, i): p for i, p in enumerate(flat)}
    # the same (disputed) flag as the summary, so a disputed point never reads "confirmed" here
    stats = notebook_stats(ctx.doc, {t: len(res.analysis(t).key_claims) for t in ctx.doc.source_map})
    disputed = {id(ps.point) for ps in stats.points if ps.disputed}
    day =out._md_escape(str(prev.get("created_at") or "")[:10] or "an earlier date")
    lines = [f"# {out._md_text(ctx.topic)} — What changed",
             f"Compared with the report of {day} (merge {prev['id']} -> {ctx.merge_id}).", ""]
    body: list[str] = []
    n_re = len(d.restructured)
    regrouped = (f"_{out._plural(n_re, 'point')} {'was' if n_re == 1 else 'were'} regrouped "
                 "from claims already in the previous report._"
                 if d.restructured else None)

    def confirms(sid: int) -> int:
        return sum(1 for gained in d.confirmed.values() if sid in gained)

    if d.new_sources:
        body += ["## New sources", "", "| Source | New points | Points it now confirms |", "|---|---|---|"]
        for sid in d.new_sources:
            n_new = sum(1 for k in d.new_points if sid in _cited_sids(ctx.doc, pts[k]))
            body.append(f"| {_title(ctx, sid)} | {n_new} | {confirms(sid)} |")
        body.append("")
        for sid in d.silent_sources:
            body.append(f"- {_title(ctx, sid)}: added nothing new; it confirms "
                        f"{out._plural(confirms(sid), 'point')} already in the report.")
        if d.silent_sources:
            body.append("")

    if d.new_points:
        body += ["## New points", ""]
        body += [out._point_line(pts[k], res, label, disputed=id(pts[k]) in disputed, new=True)[0]
                 for k in d.new_points]
        body.append("")
    if regrouped:
        body += [regrouped, ""]

    if d.confirmed:
        body += ["## Newly confirmed", ""]
        for k, gained in d.confirmed.items():
            line = out._point_line(pts[k], res, label, disputed=id(pts[k]) in disputed, new=False)[0]
            body.append(f"{line} — now also: {', '.join(_title(ctx, sid) for sid in gained)}")
        body.append("")

    if d.new_contradictions:
        body += ["## New contradictions", ""]
        for c in d.new_contradictions:
            body += [out._contradiction_line(c, res, label), ""]

    if d.removed:
        body += ["## No longer in the report", ""]
        for p in d.removed:
            names = []
            for r in p.claim_refs:
                name = old_label.labels.get(r.source, r.source)
                if name not in names:
                    names.append(name)
            body.append(f"- {out._md_text(old_label(p.text))} (from {out._md_inline(', '.join(names))})")
        body.append("")

    if d.dropped_sources:
        body += ["## Dropped sources", ""] + [f"- {_title(ctx, sid)}" for sid in d.dropped_sources] + [""]

    if d.not_compared:
        body += ["## Not compared", ""]
        why = ("the previous report predates claim hashes; sources were not compared" if not old.claim_hashes
               else "re-analyzed since the previous report; its points were not compared")
        body += [f"- {_title(ctx, sid)}: {why}." for sid in d.not_compared]
        body += ["", f"{out._plural(len(d.not_compared_points), 'point')} not compared.", ""]

    if not (d.new_points or d.confirmed or d.removed or d.new_contradictions or d.new_sources
            or d.dropped_sources or d.not_compared):
        body = ["No changes: the same points, sources and contradictions as the previous report."]
        if regrouped:
            body.append(regrouped)
    return "\n".join(lines + body)


out.FORMAT_RENDERERS.update({"digest": render_digest, "hybrid": render_hybrid,
                             "outline": render_outline, "quotes": render_quotes, "diff": render_diff})
