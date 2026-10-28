"""HTML report twins (pipeline/stages/html_render.py): self-contained pages, escaping, safe links,
unique ids (IDEA-312)."""
from __future__ import annotations

import re

from pipeline.stages.html_render import HtmlPage, IdMaker, esc, link, render_page

_FORBIDDEN = ("<script", "<link ", "<iframe", "<img", "@import", "url(")


def assert_self_contained(page: str) -> None:
    low = page.lower()
    for bad in _FORBIDDEN:
        assert bad not in low, bad
    # the only absolute URLs are citation links
    for m in re.finditer(r'(src|href)="(https?:[^"]*)"', page):
        assert m.group(1) == "href", m.group(0)


def assert_anchors_resolve(page: str) -> None:
    ids = re.findall(r'\sid="([^"]+)"', page)
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    for target in re.findall(r'href="#([^"]+)"', page):
        assert target in ids, target


def test_page_is_self_contained():
    page = render_page(HtmlPage("T", ["<p>x</p>"], [(1, "a", "A")], '<h2 id="a">A</h2>'), [])
    assert page.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in page and 'href="#a"' in page
    assert_self_contained(page)
    assert_anchors_resolve(page)


def test_esc_and_title_escape_markup():
    page = render_page(HtmlPage('<b>"x"</b>', [], [], ""), [])
    assert "&lt;b&gt;&quot;x&quot;&lt;/b&gt;" in page and "<b>" not in page
    assert esc("a\n  <b>") == "a &lt;b&gt;"


def test_sidebar_label_is_escaped():
    page = render_page(HtmlPage("T", [], [(1, "a", "<img src=x>")], '<h2 id="a">x</h2>'), [])
    assert "<img" not in page and "&lt;img src=x&gt;" in page


def test_link_refuses_other_schemes():
    assert link("javascript:alert(1)", "x") == "x"
    assert link("data:text/html,1", "<x>") == "&lt;x&gt;"
    assert 'href="https://e.org/?q=&quot;&lt;a&gt;"' in link('https://e.org/?q="<a>', "t")
    assert link("file:///C:/My%20Docs/a.pdf", "a").startswith("<a ")
    assert link("HTTPS://E.ORG/", "a").startswith("<a ")


def test_ids_unique_on_repeats():
    m = IdMaker()
    assert [m("Setup"), m("Setup"), m("setup!"), m("")] == ["setup", "setup-2", "setup-3", "section"]
    assert IdMaker()("1.2 Ça va: <ok>") == "1-2-a-va-ok"


def test_links_row_marks_current():
    page = render_page(HtmlPage("T", [], [], ""), [("Summary", "2026-10-26-summary.html", True),
                                                   ("Digest", "2026-10-26-digest.html", False)])
    assert page.count('aria-current="page"') == 1
    assert 'href="2026-10-26-digest.html"' in page
    assert 'href="2026-10-26-summary.html"' not in page        # the current page is not a link


def test_theme_is_one_replaceable_block():
    from pipeline.stages.html_render import THEME_END, THEME_START
    page = render_page(HtmlPage("T", ["<p>x</p>"], [(1, "a", "A")], '<h2 id="a">A</h2>', kind="digest"), [])
    assert page.count("<style") == 1 and page.count(THEME_START) == 1 and page.count(THEME_END) == 1
    block = page[page.index(THEME_START):page.index(THEME_END) + len(THEME_END)]
    assert '<style id="hmd-theme">' in block and "</style>" in block
    assert page.index("</head>") > page.index(THEME_END)
    assert not re.search(r"\sstyle=", page)                     # markup carries classes only


def test_page_hooks_for_themes():
    from pipeline.stages.html_render import THEME_END
    page = render_page(HtmlPage("T", [], [(1, "a", "A")], '<h2 id="a">A</h2>', kind="quotes"),
                       [("Summary", "s.html", True)])
    assert '<body class="report-quotes">' in page
    assert '<a class="skip-link" href="#content">' in page and '<main id="content">' in page
    assert '<nav class="sidebar" id="contents" aria-label="Contents">' in page
    assert '<a class="contents-link" href="#contents">Contents</a>' in page
    assert page.index('<main id="content">') < page.index('<nav class="report-links"') \
        < page.index('<header class="report-header">')                 # the tabs come first, above the h1
    assert_anchors_resolve(page)
    bare = render_page(HtmlPage("T", [], [], ""), [("Summary", "s.html", True)])
    markup = bare[bare.index(THEME_END):]                        # the theme's own comments name the hooks
    assert "contents-link" not in markup and 'id="contents"' not in markup   # no sidebar, no link to it
    assert "<body>" in markup


def test_table_marks_numeric_columns_and_scrolls():
    from pipeline.stages.html_render import table
    t = table(["Source", "Points", "Claims used", "Length", "Worth"],
              [["A", "7", "10 of 10", "15:04", "★"], ["B", "12", "9 of 9", "1:02:03", ""]])
    assert t.startswith('<div class="table-wrap"><table>') and t.endswith("</table></div>")
    assert "<th>Source</th>" in t and "<td>A</td>" in t
    for h in ("Points", "Claims used", "Length", "Worth"):
        assert f'<th class="num">{h}</th>' in t, h
    assert '<td class="num">10 of 10</td>' in t and '<td class="num"></td>' in t


# --- report twins: a rich fixture shared by the summary and format tests ------------------------

import json  # noqa: E402
from datetime import date  # noqa: E402

from pipeline import db  # noqa: E402
from pipeline.diff import diff_merges  # noqa: E402
from pipeline.schemas import MergedDocument  # noqa: E402
from pipeline.stages.output import RenderContext  # noqa: E402
from tests.test_output import DOC, _rehash, state  # noqa: E402,F401  (fixture reuse)

EVIL = "![](https://attacker.example/y) [click](https://attacker.example) <img src=x onerror=alert(1)>"
EVIL_QUOTE = "</details><script>alert(1)</script> said so"


def rich_ctx(state, *, show_anchors=False, hostile=False):
    """YouTube A/B, a web page (S3) and a PDF (S4); P1 joint and disputed, P2 partly supported, P4 joint and
    confirmed and new since the previous merge (DOC), P5 regrouped; a contradiction labelled may-differ; Video C
    not analyzed."""
    path, nid, vault, rp = state
    web = db.add_source(path, nid, "web-0123456789abcdef", "https://e.org/a", kind="web", title="An article",
                        channel="E Org")
    pdf = db.add_source(path, nid, "pdf-0123456789abcdef", "https://e.org/p.pdf", kind="pdf", title="A paper",
                        channel="Journal")
    db.update_analyze_status(path, web, "done", analysis_json=json.dumps({"summary": "web summary", "key_claims": [
        {"claim": "wc", "quote": EVIL_QUOTE if hostile else "Small changes compound over months",
         "anchor": {"char_start": 0, "char_end": 30, "t_start": 0.0, "score": 100, "segment": 2}}]}))
    db.update_analyze_status(path, pdf, "done", analysis_json=json.dumps({"summary": "pdf summary", "key_claims": [
        {"claim": "pc", "quote": "Automaticity plateaued after a median of 66 days",
         "anchor": {"char_start": 0, "char_end": 30, "t_start": 0.0, "score": 100, "segment": 4, "page": 7}}]}))
    if hostile:
        a = db.get_notebook_sources(path, nid)[0]["id"]
        db.update_source_metadata(path, a, title='<a href="javascript:alert(1)">x</a>',
                                  channel="![](https://attacker.example/b.png)")
    doc = json.loads(json.dumps(DOC))
    doc["source_map"].update({"S3": web, "S4": pdf})
    sub = doc["sections"][0]["subsections"][0]
    sub["points"][1]["support"] = "partial"
    sub["points"].append({"id": "P4", "text": "Web and PDF agree", "sources": ["S3", "S4"],
                          "claim_refs": [{"source": "S3", "claim": 0}, {"source": "S4", "claim": 0}]})
    doc["sections"][0]["subsections"].append({"title": "Later", "points": [
        {"id": "P5", "text": "A fresh point", "sources": ["S2"], "claim_refs": [{"source": "S2", "claim": 1}]}]})
    c = doc["sections"][0]["contradictions"][0]
    c["claims_a"] = [0]                                       # S1 claim 0: disputes P1
    c["check_label"] = "may_differ_by_edition_or_date"
    if hostile:
        doc["tldr"] = [EVIL]
        doc["sections"][0]["title"] = EVIL
        sub["title"] = EVIL
        sub["points"][0]["text"] = EVIL + " S1"
        c.update(topic=EVIL, position_a=EVIL + " S1", position_b=EVIL)
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    _rehash(doc, sources)
    merged = MergedDocument.model_validate(doc)
    prev = {"id": 1, "created_at": "2026-09-01 10:00:00", "merged_json": json.dumps(DOC)}
    return RenderContext(topic=EVIL if hostile else "AI Research", doc=merged, sources_by_id=sources,
                         today=date(2026, 9, 6), slug="ai-research", show_anchors=show_anchors, db_path=path,
                         notebook_id=nid, merge_id=2, previous=prev,
                         diff=diff_merges(MergedDocument.model_validate(DOC), merged))


def full(page) -> str:
    return render_page(page, [])


def assert_inert(text: str) -> None:
    low = text.lower()
    assert "<script" not in low and "<img" not in low and 'href="javascript:' not in low
    assert 'href="https://attacker.example' not in low


def test_summary_html_has_every_element(state):
    from pipeline.stages.html_render import render_summary_html
    page = full(render_summary_html(rich_ctx(state)))
    assert "<title>AI Research" in page and "Generated: 2026-09-06" in page
    assert "Bullet one" in page and 'class="legend"' in page
    for cls in ("badge-confirmed", "badge-sources", "badge-partly", "badge-disputed", "badge-new"):
        assert f'class="badge {cls}"' in page, cls
    assert 'class="sources-line"' in page
    assert "&amp;t=754s" in page and "#:~:text=" in page and "#page=7" in page
    for heading in ("Sources", "Sources not analyzed", "Source scorecard", "Coverage", "Contradictions"):
        assert f">{heading}</h2>" in page, heading
    assert "captions unavailable" in page and "Worth watching" in page
    assert "(may differ by edition or date)" in page
    assert "<th>Position A</th>" in page
    assert_self_contained(page)
    assert_anchors_resolve(page)


def test_summary_html_quotes_only_with_anchors(state):
    from pipeline.stages.html_render import render_summary_html
    assert 'class="quote"' not in full(render_summary_html(rich_ctx(state)))


def test_summary_html_quotes_with_anchors_are_folded_and_capped(state):
    from pipeline.stages.html_render import render_summary_html
    path, nid, vault, rp = state
    long_quote = " ".join(f"w{i}" for i in range(60))
    b = db.get_notebook_sources(path, nid)[1]["id"]
    analysis = json.loads(db.get_source(path, b)["analysis_json"])
    analysis["key_claims"][0]["quote"] = long_quote
    db.update_analyze_status(path, b, "done", analysis_json=json.dumps(analysis))
    page = full(render_summary_html(rich_ctx(state, show_anchors=True)))
    assert '<details class="quotes">' in page and "<summary>Quotes (" in page
    quotes = re.findall(r'<q class="quote">(.*?)</q>', page)
    assert quotes and max(len(q.split()) for q in quotes) <= 40
    assert any(q.endswith("w39…") for q in quotes)


def test_summary_html_escapes_hostile_text(state):
    from pipeline.stages.html_render import render_summary_html
    page = full(render_summary_html(rich_ctx(state, show_anchors=True, hostile=True)))
    assert_inert(page)
    assert "&lt;img src=x" in page and "&lt;/details&gt;&lt;script&gt;" in page
    assert_anchors_resolve(page)


def test_contradictions_only_merge_renders(state):
    from pipeline.stages.html_render import render_summary_html
    ctx = rich_ctx(state)
    doc = json.loads(ctx.doc.model_dump_json())
    doc["tldr"] = []
    for sec in doc["sections"]:
        sec["subsections"] = []
    import dataclasses
    ctx = dataclasses.replace(ctx, doc=MergedDocument.model_validate(doc), diff=None)
    page = full(render_summary_html(ctx))
    assert "No TL;DR" in page and "Contradictions" in page
    assert_anchors_resolve(page)


# --- the other formats --------------------------------------------------------------------------

def _render(fmt, ctx):
    from pipeline.stages.html_render import HTML_RENDERERS
    return full(HTML_RENDERERS[fmt](ctx))


def test_digest_html_has_chapters(state):
    page = _render("digest", rich_ctx(state))
    assert "<title>AI Research — Digest</title>" in page
    assert page.index(">1. Video A</h2>") < page.index(">2. Video B</h2>") < page.index(">3. An article</h2>")
    assert "Video C" not in page                                   # not analyzed
    assert "web summary" in page and "#page=7" in page and "#:~:text=" in page
    assert "In the summary: 1 points, 0 only from this source." in page
    assert 'class="quote"' not in page
    with_q = _render("digest", rich_ctx(state, show_anchors=True))
    assert '<q class="quote">' in with_q
    assert_self_contained(page)
    assert_anchors_resolve(page)


def test_hybrid_html_is_summary_plus_chapters(state):
    page = _render("hybrid", rich_ctx(state))
    assert "<title>AI Research — Hybrid</title>" in page
    assert page.index(">1. Fundamentals</h2>") < page.index(">Per-source chapters</h2>") < page.index(">Sources</h2>")
    assert ">1. Video A</h3>" in page and 'class="badge badge-new"' in page
    assert_anchors_resolve(page)


def test_outline_html_lists_points_with_sources(state):
    page = _render("outline", rich_ctx(state))
    assert "<title>AI Research — Outline</title>" in page
    assert "Both agree" in page and "[Chan A, Chan B]" in page
    assert 'class="cite"' not in page                              # the outline has no links
    assert_anchors_resolve(page)


def test_quotes_html_groups_quotes_by_point(state):
    page = _render("quotes", rich_ctx(state))
    assert "<title>AI Research — Quote bank</title>" in page
    assert page.count('<q class="quote">the exact quote') == 5     # P1 cites 3 claims, P2 one, P5 one
    assert "<strong>Both agree</strong>" in page and "#page=7" in page
    assert_anchors_resolve(page)


def test_diff_html_sections(state):
    page = _render("diff", rich_ctx(state))
    assert "<title>AI Research — What changed</title>" in page
    assert "Compared with the report of 2026-09-01 (merge 1 -&gt; 2)." in page
    assert ">New sources</h2>" in page and ">New points</h2>" in page and "Web and PDF agree" in page
    assert "regrouped from claims already in the previous report" in page   # P5 reuses an old claim
    assert 'class="badge badge-new"' in page
    assert_anchors_resolve(page)


def test_report_kind_and_citations_are_marked_up(state):
    from pipeline.stages.html_render import render_summary_html
    ctx = rich_ctx(state, show_anchors=True)
    summary = full(render_summary_html(ctx))
    assert '<h1>AI Research <span class="report-kind">— Research Summary</span></h1>' in summary
    assert '<body class="report-summary">' in summary
    assert '<span class="cites"><a class="cite"' in summary and '<span class="cites">—' not in summary
    assert '</q><cite class="source">' in summary and "</q> — " not in summary
    digest = _render("digest", ctx)
    assert '<span class="cites"><a class="cite"' in digest and "</q> — " not in digest
    quotes = _render("quotes", ctx)
    assert '</q><cite class="source"><a class="cite"' in quotes
    diff = _render("diff", ctx)
    assert "— now also" not in diff          # the "Now also" branch is checked on the sample diff
    for page in (summary, digest, quotes, diff):
        assert_self_contained(page)
        assert_anchors_resolve(page)


def test_diff_html_is_none_without_a_previous_merge(state):
    import dataclasses
    from pipeline.stages.html_render import HTML_RENDERERS
    assert HTML_RENDERERS["diff"](dataclasses.replace(rich_ctx(state), diff=None, previous=None)) is None


def test_all_formats_escape_hostile_text(state):
    ctx = rich_ctx(state, show_anchors=True, hostile=True)
    for fmt in ("digest", "hybrid", "outline", "quotes", "diff"):
        page = _render(fmt, ctx)
        assert_inert(page)
        assert_self_contained(page)
        assert_anchors_resolve(page)


def test_every_untrusted_field_is_inert_in_every_format(state):
    """Beyond rich_ctx(hostile=True): a hostile digest summary and claim, a hostile error message and a
    javascript: URL on a source that was not analyzed (security scan F6/F9)."""
    import dataclasses
    import sqlite3
    from pipeline.stages.html_render import HTML_RENDERERS, render_summary_html
    path, nid, vault, rp = state
    ctx = rich_ctx(state, show_anchors=True, hostile=True)
    web = next(s for s in ctx.sources_by_id.values() if s["video_id"].startswith("web-"))
    analysis = json.loads(web["analysis_json"])
    analysis["summary"] = EVIL
    analysis["key_claims"][0]["claim"] = EVIL
    db.update_analyze_status(path, web["id"], "done", analysis_json=json.dumps(analysis))
    missing = next(s for s in ctx.sources_by_id.values() if s["video_id"] == "ccccccccccc")
    db.update_analyze_status(path, missing["id"], "error", error="<script>alert(1)</script> " + EVIL)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE sources SET url=? WHERE id=?", ("javascript:alert(1)", missing["id"]))
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    doc = _rehash(json.loads(ctx.doc.model_dump_json()), sources)      # the claim edit changes its hash
    ctx = dataclasses.replace(ctx, sources_by_id=sources, doc=MergedDocument.model_validate(doc))
    pages = [full(render_summary_html(ctx))] + [full(fn(ctx)) for fn in HTML_RENDERERS.values()]
    for page in pages:
        assert_inert(page)
        assert_self_contained(page)
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in pages[0]          # the error message, as text
    digest = full(HTML_RENDERERS["digest"](ctx))
    assert digest.count("&lt;img src=x onerror=alert(1)&gt;") >= 2     # summary and claim, as text
