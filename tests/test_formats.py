import json
import os
from datetime import date
from pipeline import db
from pipeline.schemas import MergedDocument
from pipeline.stages.output import RenderContext
from pipeline.stages.formats import render_digest, render_hybrid, render_outline, render_quotes
from tests.test_output import DOC, state  # noqa: F401  (fixture reuse)


def _ctx(state, **kw):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    base = dict(topic="AI Research", doc=MergedDocument.model_validate(DOC), sources_by_id=sources,
                today=date(2026, 9, 6), slug="ai-research", show_anchors=False, db_path=path, notebook_id=nid)
    base.update(kw)
    return RenderContext(**base)


def test_digest_has_a_chapter_per_analyzed_source_in_source_order(state):
    md = render_digest(_ctx(state))
    assert md.startswith("# AI Research — Digest")
    assert md.index("## 1. Video A") < md.index("## 2. Video B")
    assert "Video C" not in md                                    # not analyzed
    assert "- c0 — [12:34](https://www.youtube.com/watch?v=aaaaaaaaaaa&t=754s)" in md
    assert "_In the summary: 1 points, 0 only from this source._" in md


def test_hybrid_is_the_summary_plus_chapters_before_sources(state):
    md = render_hybrid(_ctx(state))
    lines = md.splitlines()
    assert lines[0] == "# AI Research — Hybrid"
    assert lines.index("## 1. Fundamentals") < lines.index("## Per-source chapters") < lines.index("## Sources")
    assert "### 1. Video A" in lines


def test_outline_has_no_links(state):
    md = render_outline(_ctx(state))
    assert "- Both agree [Chan A, Chan B]" in md and "](http" not in md


def test_quote_bank_lists_each_cited_quote_capped(state):
    md = render_quotes(_ctx(state))
    assert "- **Both agree**" in md
    assert md.count('> "the exact quote') == 4                     # P1 cites 3 claims, P2 one
    assert "[Video A 12:34](https://www.youtube.com/watch?v=aaaaaaaaaaa&t=754s)" in md


def _diff_text(rp):
    return open(rp.replace("-summary.md", "-diff.md"), encoding="utf-8").read()


def test_first_report_has_no_diff(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    run_output(path, nid, vault, formats=["summary", "diff"])
    assert not os.path.exists(rp.replace("-summary.md", "-diff.md"))


def test_redundant_video_diff_says_it_added_nothing(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    doc2 = json.loads(json.dumps(DOC))
    db.create_merge(path, nid, None, "m", json.dumps(doc2), json.dumps(doc2["source_map"]), "{}", rp)
    run_output(path, nid, vault, formats=["summary", "diff"])
    assert "No changes" in _diff_text(rp)


def _second_merge(state, doc2):
    path, nid, vault, rp = state
    from tests.test_output import _rehash
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    _rehash(doc2, sources)
    db.create_merge(path, nid, None, "m", json.dumps(doc2), json.dumps(doc2["source_map"]), "{}", rp)


def _add_analyzed_d(state):
    path, nid, vault, rp = state
    from tests.test_output import _analysis
    d = db.add_source(path, nid, "ddddddddddd", "https://www.youtube.com/watch?v=ddddddddddd", title="Video D", channel="Chan D")
    db.update_analyze_status(path, d, "done", analysis_json=json.dumps(_analysis(100.0)))
    return d


def test_new_source_with_only_confirmations_is_called_out(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    d = _add_analyzed_d(state)
    doc2 = json.loads(json.dumps(DOC))
    doc2["source_map"]["S3"] = d
    p1 = doc2["sections"][0]["subsections"][0]["points"][0]
    p1["sources"].append("S3"); p1["claim_refs"].append({"source": "S3", "claim": 0})
    _second_merge(state, doc2)
    run_output(path, nid, vault, formats=["summary", "diff"])
    diff = _diff_text(rp)
    assert "Video D: added nothing new; it confirms 1 point already in the report." in diff
    assert "## New points" not in diff
    summary = open(rp, encoding="utf-8").read()
    assert "(new)" not in summary.split("## Sources")[0].replace("(new) = not in the previous report", "")


def test_new_point_gets_new_badge_in_summary(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    d = _add_analyzed_d(state)
    doc2 = json.loads(json.dumps(DOC))
    doc2["source_map"]["S3"] = d
    doc2["sections"][0]["subsections"].append({"title": "More", "points": [
        {"id": "P3", "text": "A fresh point", "sources": ["S3"], "claim_refs": [{"source": "S3", "claim": 0}]}]})
    _second_merge(state, doc2)
    run_output(path, nid, vault, formats=["summary", "diff"])
    summary = open(rp, encoding="utf-8").read()
    assert "(new)" in next(l for l in summary.splitlines() if l.startswith("- A fresh point"))
    diff = _diff_text(rp)
    assert "## New points" in diff and "A fresh point" in diff
    assert "## New sources" in diff and "| Video D | 1 | 0 |" in diff


def test_regrouped_point_gets_no_badge_and_a_regrouped_line(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    doc2 = json.loads(json.dumps(DOC))
    doc2["sections"][0]["subsections"].append({"title": "More", "points": [
        {"id": "P3", "text": "A regrouped point", "sources": ["S1"], "claim_refs": [{"source": "S1", "claim": 0}]}]})
    _second_merge(state, doc2)
    run_output(path, nid, vault, formats=["summary", "diff"])
    summary = open(rp, encoding="utf-8").read()
    assert "(new)" not in next(l for l in summary.splitlines() if l.startswith("- A regrouped point"))
    diff = _diff_text(rp)
    assert "## New points" not in diff
    assert "_1 point was regrouped from claims already in the previous report._" in diff


def test_dropped_source_is_listed(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    d = _add_analyzed_d(state)
    doc1 = json.loads(json.dumps(DOC))
    doc1["source_map"]["S3"] = d
    _second_merge(state, doc1)          # previous merge knows D ...
    _second_merge(state, json.loads(json.dumps(DOC)))   # ... the latest does not
    run_output(path, nid, vault, formats=["summary", "diff"])
    diff = _diff_text(rp)
    assert "## Dropped sources" in diff and "Video D" in diff
    assert "No changes" not in diff


def test_newly_confirmed_lists_the_gained_source(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    d = _add_analyzed_d(state)
    doc2 = json.loads(json.dumps(DOC))
    doc2["source_map"]["S3"] = d
    p2 = doc2["sections"][0]["subsections"][0]["points"][1]
    p2["sources"].append("S3"); p2["claim_refs"].append({"source": "S3", "claim": 0})
    _second_merge(state, doc2)
    run_output(path, nid, vault, formats=["summary", "diff"])
    diff = _diff_text(rp)
    assert "## Newly confirmed" in diff
    line = next(l for l in diff.splitlines() if l.startswith("- Only one"))
    assert line.endswith(" — now also: Video D")


def test_diff_marks_disputed_points_like_the_summary(state):
    # review 2026-10-18: the diff passed disputed=False, so a disputed joint point read "confirmed"
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    d = _add_analyzed_d(state)
    doc2 = json.loads(json.dumps(DOC))
    doc2["source_map"]["S3"] = d
    doc2["sections"][0]["subsections"][0]["points"].append(
        {"id": "P3", "text": "New joint point", "sources": ["S1", "S3"],
         "claim_refs": [{"source": "S1", "claim": 0}, {"source": "S3", "claim": 0}]})
    doc2["sections"][0]["contradictions"][0]["claims_a"] = [0]      # S1 claim 0: disputes P1 and P3
    _second_merge(state, doc2)
    run_output(path, nid, vault, formats=["summary", "diff"])
    line = next(l for l in _diff_text(rp).splitlines() if l.startswith("- New joint point"))
    assert "(disputed)" in line and line.endswith(" (2 sources)") and "confirmed by" not in line


def test_removed_point_is_listed_with_its_old_sources(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    doc2 = json.loads(json.dumps(DOC))
    del doc2["sections"][0]["subsections"][0]["points"][1]       # P2 (S2 claim 0) is gone
    _second_merge(state, doc2)
    run_output(path, nid, vault, formats=["summary", "diff"])
    diff = _diff_text(rp)
    assert "## No longer in the report" in diff
    assert "- Only one (from Chan B)" in diff


def test_new_contradiction_uses_the_summary_format(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    doc2 = json.loads(json.dumps(DOC))
    doc2["sections"][0]["contradictions"].append(
        {"topic": "Cost", "position_a": "cheap", "source_a": "S1", "claims_a": [1],
         "position_b": "dear", "source_b": "S2", "claims_b": [0]})
    _second_merge(state, doc2)
    run_output(path, nid, vault, formats=["summary", "diff"])
    diff = _diff_text(rp)
    assert "## New contradictions" in diff
    section = diff.split("## New contradictions")[1]
    assert '**Contradiction — Cost:** [Video A]' in section and 'claims "cheap" while [Video B]' in section
    assert "Timeline" not in section
    summary = open(rp, encoding="utf-8").read()
    assert next(l for l in section.splitlines() if l.startswith("**Contradiction")) in summary


def test_reanalyzed_source_is_not_compared(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    from tests.test_output import _analysis
    a = db.get_notebook_sources(path, nid)[0]["id"]
    changed = _analysis(754.0)
    changed["key_claims"][0]["claim"] = "c0 revised"
    db.update_analyze_status(path, a, "done", analysis_json=json.dumps(changed))
    doc2 = json.loads(json.dumps(DOC))
    doc2["sections"][0]["subsections"].append({"title": "More", "points": [
        {"id": "P3", "text": "From the re-analyzed video", "sources": ["S1"],
         "claim_refs": [{"source": "S1", "claim": 0}]}]})
    _second_merge(state, doc2)
    run_output(path, nid, vault, formats=["summary", "diff"])
    diff = _diff_text(rp)
    assert "## Not compared" in diff
    assert "- Video A: re-analyzed since the previous report; its points were not compared." in diff
    assert "1 point not compared." in diff
    assert "## New points" not in diff


def test_idless_new_point_is_badged_in_the_summary_and_listed(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    d = _add_analyzed_d(state)
    doc2 = json.loads(json.dumps(DOC))
    doc2["source_map"]["S3"] = d
    doc2["sections"][0]["subsections"].append({"title": "More", "points": [
        {"text": "No id here", "sources": ["S3"], "claim_refs": [{"source": "S3", "claim": 0}]}]})
    _second_merge(state, doc2)
    run_output(path, nid, vault, formats=["summary", "diff"])
    summary = open(rp, encoding="utf-8").read()
    assert "(new)" in next(l for l in summary.splitlines() if l.startswith("- No id here"))
    assert "No id here" in _diff_text(rp).split("## New points")[1]


def test_diff_header_names_both_merge_ids(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    _second_merge(state, json.loads(json.dumps(DOC)))
    run_output(path, nid, vault, formats=["summary", "diff"])
    assert "(merge 1 -> 2)." in _diff_text(rp)


def test_diff_against_a_merge_without_claim_hashes_says_so(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    old = json.loads(json.dumps(DOC)); old.pop("claim_hashes")
    db.create_merge(path, nid, None, "m", json.dumps(old), json.dumps(old["source_map"]), "{}", rp)
    _second_merge(state, json.loads(json.dumps(DOC)))       # the latest merge, hashed
    run_output(path, nid, vault, formats=["summary", "diff"])
    diff = _diff_text(rp)
    assert "- Video A: the previous report predates claim hashes; sources were not compared." in diff
    assert "re-analyzed since the previous report" not in diff


def test_a_failing_optional_format_does_not_fail_the_run(state, monkeypatch):
    path, nid, vault, rp = state
    from pipeline.stages import output
    def boom(ctx):
        raise RuntimeError("renderer bug")
    monkeypatch.setitem(output.FORMAT_RENDERERS, "digest", boom)
    logs = []
    assert output.run_output(path, nid, vault, formats=["summary", "digest", "outline"], log=logs.append) == rp
    assert os.path.exists(rp) and os.path.exists(rp.replace("-summary.md", "-outline.md"))
    assert not os.path.exists(rp.replace("-summary.md", "-digest.md"))
    warn = next(l for l in logs if l.startswith("WARNING:") and "digest" in l)
    assert "renderer bug" in warn
    assert os.path.exists(os.path.join(vault, "ai-research", "_notebook.md"))


def test_a_failing_diff_computation_does_not_fail_the_run(state, monkeypatch):
    path, nid, vault, rp = state
    from pipeline.stages import output
    _second_merge(state, json.loads(json.dumps(DOC)))
    def boom(old, new):
        raise KeyError("diff bug")
    monkeypatch.setattr(output, "diff_merges", boom)
    logs = []
    output.run_output(path, nid, vault, formats=["summary", "diff"], log=logs.append)
    assert os.path.exists(rp) and not os.path.exists(rp.replace("-summary.md", "-diff.md"))
    assert any(l.startswith("WARNING:") and "diff" in l for l in logs)
    assert os.path.exists(os.path.join(vault, "ai-research", "_notebook.md"))


def _hostile_source_a(state):
    path, nid, vault, rp = state
    from tests.test_output import _analysis
    a = db.get_notebook_sources(path, nid)[0]["id"]
    db.update_source_metadata(path, a, title="x](https://evil)", channel="[c](https://evil)",
                              upload_date="[d](https://evil)")
    hostile = _analysis(754.0); hostile["summary"] = "# owned\n- also | here"
    db.update_analyze_status(path, a, "done", analysis_json=json.dumps(hostile))   # claims unchanged: hash holds


def test_untrusted_labels_render_inert_in_the_new_formats(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    _hostile_source_a(state)
    run_output(path, nid, vault, formats=["summary", "digest", "outline"])
    read = lambda fmt: open(rp.replace("-summary.md", f"-{fmt}.md"), encoding="utf-8").read()
    summary, digest, outline = open(rp, encoding="utf-8").read(), read("digest"), read("outline")
    lines = summary.splitlines()
    assert r"_Sources: Chan B (2), \[c\](https://evil) (1)_" in lines
    contradiction = next(l for l in lines if l.startswith("**Contradiction"))
    assert "[d]" not in contradiction                                  # unparseable date: not shown
    assert any(l.startswith(r"| x\](https://evil) | ") for l in lines)            # scorecard row
    assert next(l for l in lines if l.startswith("| Timeline |")).endswith("| ? / ? |")
    assert r"## 1. x\](https://evil)" in digest.splitlines()
    assert r"\# owned - also \| here" in digest.splitlines()
    meta = next(l for l in digest.splitlines() if "[watch]" in l)
    assert meta.startswith(r"\[c\](https://evil) · ") and "[d]" not in meta
    assert r"- Both agree [\[c\](https://evil), Chan B]" in outline.splitlines()


def test_md_inline_escapes_block_markers_and_link_syntax():
    from pipeline.stages.output import _md_inline
    assert _md_inline("# h") == r"\# h" and _md_inline("- b") == r"\- b" and _md_inline("> q") == r"\> q"
    assert _md_inline("* s") == r"\* s" and _md_inline("+ p") == r"\+ p" and _md_inline("12. n") == r"12\. n"
    assert _md_inline("3) n") == r"3\) n"
    assert _md_inline(r"a [b](c) | d \ e") == r"a \[b\](c) \| d \\ e"
    assert _md_inline("  one\n\ntwo ") == "one two" and _md_inline("C# - ok") == "C# - ok"


def test_diff_title_prose_is_escaped(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    d = _add_analyzed_d(state)
    db.update_source_metadata(path, d, title="# y](https://evil)")
    doc2 = json.loads(json.dumps(DOC))
    doc2["source_map"]["S3"] = d
    p1 = doc2["sections"][0]["subsections"][0]["points"][0]
    p1["sources"].append("S3"); p1["claim_refs"].append({"source": "S3", "claim": 0})
    _second_merge(state, doc2)
    run_output(path, nid, vault, formats=["summary", "diff"])
    diff = _diff_text(rp)
    assert r"- \# y\](https://evil): added nothing new" in diff
    assert r"| \# y\](https://evil) | 0 | 1 |" in diff


def test_hybrid_shows_new_badges(state):
    path, nid, vault, rp = state
    from pipeline.stages.output import run_output
    d = _add_analyzed_d(state)
    doc2 = json.loads(json.dumps(DOC))
    doc2["source_map"]["S3"] = d
    doc2["sections"][0]["subsections"].append({"title": "More", "points": [
        {"id": "P3", "text": "A fresh point", "sources": ["S3"], "claim_refs": [{"source": "S3", "claim": 0}]}]})
    _second_merge(state, doc2)
    run_output(path, nid, vault, formats=["summary", "hybrid"])
    hybrid = open(rp.replace("-summary.md", "-hybrid.md"), encoding="utf-8").read()
    assert "(new)" in next(l for l in hybrid.splitlines() if l.startswith("- A fresh point"))


def test_digest_and_quotes_link_non_youtube_kinds_per_kind(state):
    path, nid, vault, rp = state
    web = db.add_source(path, nid, "web-0123456789abcdef", "https://e.org/a", kind="web", title="An article")
    db.update_analyze_status(path, web, "done", analysis_json=json.dumps(
        {"summary": "s", "key_claims": [{"claim": "wc", "quote": "Small changes compound over months",
         "anchor": {"char_start": 0, "char_end": 30, "t_start": 0.0, "score": 100, "segment": 2}}]}))
    doc = json.loads(json.dumps(DOC))
    doc["source_map"]["S3"] = web
    doc["sections"][0]["subsections"][0]["points"].append(
        {"id": "P9", "text": "Compounding", "sources": ["S3"], "claim_refs": [{"source": "S3", "claim": 0}]})
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    from tests.test_output import _rehash
    _rehash(doc, sources)
    ctx = _ctx(state, doc=MergedDocument.model_validate(doc), sources_by_id=sources)
    digest = render_digest(ctx)
    assert "- wc — [¶ 3](https://e.org/a#:~:text=Small%20changes%20compound%20over%20months)" in digest
    assert "[open](https://e.org/a)" in digest and "[watch](https://www.youtube.com/watch?v=aaaaaaaaaaa)" in digest
    quotes = render_quotes(ctx)
    assert "[An article ¶ 3](https://e.org/a#:~:text=Small%20changes%20compound%20over%20months)" in quotes


def test_digest_keeps_document_order_for_text_sources(state):
    path, nid, vault, rp = state
    web = db.add_source(path, nid, "web-0123456789abcdef", "https://e.org/a", kind="web", title="An article")
    claims = [{"claim": f"wclaim{i}", "quote": f"quote number {i}",
               "anchor": {"char_start": 0, "char_end": 5, "t_start": 0.0, "score": 100, "segment": seg}}
              for i, seg in enumerate((5, 1, 3))]
    db.update_analyze_status(path, web, "done", analysis_json=json.dumps({"summary": "s", "key_claims": claims}))
    doc = json.loads(json.dumps(DOC))
    doc["source_map"]["S3"] = web
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    from tests.test_output import _rehash
    _rehash(doc, sources)
    md = render_digest(_ctx(state, doc=MergedDocument.model_validate(doc), sources_by_id=sources))
    assert md.index("- wclaim1 ") < md.index("- wclaim2 ") < md.index("- wclaim0 ")


def test_model_written_and_untrusted_text_cannot_inject_links_images_or_html_in_any_format(state):
    import re
    path, nid, vault, rp = state
    from pipeline.stages.formats import render_diff
    from pipeline.stages.output import render_markdown
    a = db.get_notebook_sources(path, nid)[0]["id"]
    db.update_source_metadata(path, a, title='<a href="javascript:alert(1)">x</a>',
                              channel="![](https://evil.example/b.png)")
    evil = "![](https://evil.example/y) [click](https://evil.example) <img src=x onerror=alert(1)>"
    doc = json.loads(json.dumps(DOC))
    doc["tldr"] = [evil]
    sec = doc["sections"][0]
    sec["title"] = evil
    sec["subsections"][0]["title"] = evil
    sec["subsections"][0]["points"][0]["text"] = evil + " S1"          # S1 is labelled with the channel
    sec["contradictions"][0].update(topic=evil, position_a=evil + " S1", position_b=evil)
    ctx = _ctx(state, doc=MergedDocument.model_validate(doc), show_anchors=True)
    texts = [render_markdown(ctx.topic, ctx.doc, ctx.sources_by_id, show_anchors=True, slug=ctx.slug),
             render_digest(ctx), render_hybrid(ctx), render_outline(ctx), render_quotes(ctx)]
    for text in texts:
        assert "<a " not in text and "<img" not in text and "![" not in text.replace("!\\[", "")
        assert re.search(r"(?<!\\)\]\(https://evil", text) is None
        assert re.search(r"(?<!\\)\[click", text) is None
    assert "&lt;a href=" in texts[0]
