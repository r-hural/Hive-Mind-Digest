import json, os, re
from datetime import date
import pytest
from pipeline import db
from pipeline.stages.output import (render_markdown, run_output, report_path_for, deep_link, fmt_time,
                                    write_notebook_stub, RenderError)
from pipeline.schemas import Analysis, MergedDocument, claims_hash


def _analysis(t0):
    return {"summary": "s", "key_claims": [
        {"claim": "c0", "quote": "the exact quote zero", "anchor": {"char_start": 0, "char_end": 5, "t_start": t0, "score": 100}},
        {"claim": "c1", "quote": "the exact quote one", "anchor": {"char_start": 9, "char_end": 12, "t_start": t0 + 60, "score": 95}}]}


DOC = {"tldr": ["Bullet one", "Bullet two"], "source_map": {"S1": 1, "S2": 2}, "prompt_versions": {"merge": "m"},
       "sections": [{"title": "Fundamentals", "subsections": [{"title": "Basics", "points": [
           {"id": "P1", "text": "Both agree", "sources": ["S1", "S2", "S1"],
            "claim_refs": [{"source": "S1", "claim": 0}, {"source": "S2", "claim": 1}, {"source": "S1", "claim": 1}]},
           {"id": "P2", "text": "Only one", "sources": ["S2"], "claim_refs": [{"source": "S2", "claim": 0}]}]}],
           "contradictions": [{"topic": "Timeline", "position_a": "5 years", "source_a": "S1", "position_b": "20 years", "source_b": "S2"}]}]}
# What run_merge would have stored for the analyses seeded by the `state` fixture below.
DOC["claim_hashes"] = {"S1": claims_hash(Analysis.model_validate(_analysis(754.0))),
                       "S2": claims_hash(Analysis.model_validate(_analysis(30.0)))}


def _rehash(doc, sources):
    """Re-pin doc's claim_hashes to the analyses as they stand now (for tests that edit an
    analysis on purpose and are not about the re-analysis check)."""
    doc["claim_hashes"] = {tok: claims_hash(Analysis.model_validate_json(sources[sid]["analysis_json"]))
                           for tok, sid in doc["source_map"].items()}
    return doc


@pytest.fixture
def state(tmp_path):
    path = str(tmp_path / "s.sqlite"); db.init_db(path)
    nid = db.create_notebook(path, "AI Research", "ai-research")
    a = db.add_source(path, nid, "aaaaaaaaaaa", "https://www.youtube.com/watch?v=aaaaaaaaaaa", title="Video A", channel="Chan A",
                      duration_seconds=754, upload_date="20260102")
    b = db.add_source(path, nid, "bbbbbbbbbbb", "https://www.youtube.com/watch?v=bbbbbbbbbbb", title="Video B", channel="Chan B")
    c = db.add_source(path, nid, "ccccccccccc", "https://www.youtube.com/watch?v=ccccccccccc", title="Video C", channel="Chan C")
    db.update_analyze_status(path, a, "done", analysis_json=json.dumps(_analysis(754.0)))
    db.update_analyze_status(path, b, "done", analysis_json=json.dumps(_analysis(30.0)))
    db.update_analyze_status(path, c, "error", error="captions unavailable")
    vault = str(tmp_path / "vault")
    rp = report_path_for(vault, "ai-research", date(2026, 9, 6))
    db.create_merge(path, nid, None, "m", json.dumps(DOC), json.dumps(DOC["source_map"]), "{}", rp)
    return path, nid, vault, rp


def test_helpers():
    assert fmt_time(754) == "12:34" and fmt_time(3661) == "1:01:01" and fmt_time(5.9) == "0:05"
    assert deep_link("aaaaaaaaaaa", 754.6) == "https://www.youtube.com/watch?v=aaaaaaaaaaa&t=754s"
    assert report_path_for("v", "slug", date(2026, 9, 6)).replace("\\", "/") == "v/slug/reports/2026-09-06-summary.md"


def test_render_points_with_deep_links_and_computed_confirmed_by(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    md = render_markdown("AI Research", MergedDocument.model_validate(DOC), sources, today=date(2026, 9, 6))
    assert md.startswith("# AI Research — Research Summary")
    assert "- Bullet one" in md
    assert "## 1. Fundamentals" in md and "### 1.1 Basics" in md
    line = next(l for l in md.splitlines() if l.startswith("- Both agree"))
    assert "(confirmed by 2 sources)" in line                       # len(set(S1,S2,S1)) == 2
    assert "[Video A 12:34](https://www.youtube.com/watch?v=aaaaaaaaaaa&t=754s)" in line
    assert "[Video B 1:30](https://www.youtube.com/watch?v=bbbbbbbbbbb&t=90s)" in line
    assert line.count("Video A") == 1                                # one link per source: first ref
    only = next(l for l in md.splitlines() if l.startswith("- Only one"))
    assert "confirmed by" not in only and "&t=30s" in only
    assert "**Contradiction — Timeline:** [Video A]" in md and '"5 years"' in md
    assert "| S1 | youtube | Video A | Chan A | 2026-01-02 | 12:34 |" in md
    assert "## Sources not analyzed" in md and "Video C" in md and "captions unavailable" in md
    assert "> " not in md                                            # anchors off by default
    assert "Sources: 2 videos, 2 channels" in md


def test_header_pluralization_is_singular_for_one(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    doc = json.loads(json.dumps(DOC))
    doc["source_map"] = {"S1": 1}
    doc["sections"][0]["subsections"][0]["points"] = [doc["sections"][0]["subsections"][0]["points"][0]]
    doc["sections"][0]["subsections"][0]["points"][0]["sources"] = ["S1"]
    doc["sections"][0]["subsections"][0]["points"][0]["claim_refs"] = [{"source": "S1", "claim": 0}]
    doc["sections"][0]["contradictions"] = []
    md = render_markdown("T", MergedDocument.model_validate(doc), sources)
    assert "Sources: 1 video, 1 channel" in md


def test_every_point_has_a_timestamp(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    md = render_markdown("T", MergedDocument.model_validate(DOC), sources)
    body = md.split("## Sources")[0]
    bullets = [l for l in body.splitlines() if l.startswith("- ") and not l.startswith("- Bullet")]
    assert bullets and all(re.search(r"&t=\d+s\)", l) for l in bullets)


def test_show_anchors_renders_quotes(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    md = render_markdown("T", MergedDocument.model_validate(DOC), sources, show_anchors=True)
    assert '  > "the exact quote zero" — Video A 12:34' in md


def test_show_anchors_escapes_the_source_title(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    for s in sources.values():
        s["title"] = "[x](javascript:alert(1)) | cell"
    md = render_markdown("T", MergedDocument.model_validate(DOC), sources, show_anchors=True)
    assert '  > "the exact quote zero" — \\[x\\](javascript:alert(1)) \\| cell 12:34' in md
    assert "— [x](" not in md


def test_show_anchors_escapes_the_quote_text(state):
    # The quote is untrusted source text (a web page or PDF as much as a transcript): it must not
    # open a link or split a table cell on the quote line.
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    analysis = _analysis(754.0)
    analysis["key_claims"][0]["quote"] = "see [x](javascript:alert(1)) | now"
    db.update_analyze_status(path, sources[1]["id"], "done", analysis_json=json.dumps(analysis))
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    doc = json.loads(json.dumps(DOC))
    md = render_markdown("T", MergedDocument.model_validate(_rehash(doc, sources)), sources, show_anchors=True)
    assert '  > "see \\[x\\](javascript:alert(1)) \\| now" — Video A 12:34' in md


def test_show_anchors_separates_multiple_quotes(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    md = render_markdown("T", MergedDocument.model_validate(DOC), sources, show_anchors=True)
    lines = md.splitlines()
    idx = next(i for i, l in enumerate(lines) if l.startswith("- Both agree"))
    assert lines[idx + 1].startswith('  > "the exact quote zero"')
    assert lines[idx + 2] == "  >"
    assert lines[idx + 3].startswith('  > "the exact quote one"')


def test_show_anchors_caps_quote_at_forty_words(state):
    # The analyze prompt asks for 8-40 word quotes but nothing enforces it (Haiku returned up to
    # 93 words when the sample digest was recorded on 2026-09-07). The rendered quote line is the
    # only place a transcript excerpt reaches a shipped report, so cap it here: first 40 words plus
    # an ellipsis glued to the last word, so the line still counts as 40 words and the closing
    # quote and attribution keep their shape (tests/test_examples.py parses that shape).
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    long_quote = " ".join(f"w{i}" for i in range(1, 61))
    doc = json.loads(json.dumps(DOC))
    analysis = _analysis(754.0)
    analysis["key_claims"][0]["quote"] = long_quote
    db.update_analyze_status(path, sources[1]["id"], "done", analysis_json=json.dumps(analysis))
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    md = render_markdown("T", MergedDocument.model_validate(_rehash(doc, sources)), sources, show_anchors=True)
    line = next(l for l in md.splitlines() if l.startswith('  > "w1 '))
    quote, attribution = line[4:].rsplit('" — ', 1)
    assert quote == '"' + " ".join(f"w{i}" for i in range(1, 41)) + "…"
    assert len(quote.split()) == 40
    assert attribution == "Video A 12:34"
    # A quote at or under the cap is untouched.
    assert '  > "the exact quote one" — Video B 1:30' in md


def test_unknown_token_is_an_error_not_a_wrong_title(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    bad = json.loads(json.dumps(DOC)); bad["sections"][0]["subsections"][0]["points"][1]["sources"] = ["S9"]
    bad["sections"][0]["subsections"][0]["points"][1]["claim_refs"] = [{"source": "S9", "claim": 0}]
    with pytest.raises(RenderError, match="S9"):
        render_markdown("T", MergedDocument.model_validate(bad), sources)


def test_header_source_count_validates_every_source_map_token(state):
    # A source_map token that no point ever cites must still be validated -- the header count
    # must not silently drop it.
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    doc = json.loads(json.dumps(DOC))
    doc["source_map"]["S3"] = 9999
    with pytest.raises(RenderError, match="S3"):
        render_markdown("T", MergedDocument.model_validate(doc), sources)


def test_run_output_writes_report_and_stub(state):
    path, nid, vault, rp = state
    out = run_output(path, nid, vault, log=lambda m: None)
    assert out == rp and os.path.exists(rp)
    stub = open(os.path.join(vault, "ai-research", "_notebook.md"), encoding="utf-8").read()
    assert stub.startswith('---\nid: "ai-research"\ntype: notebook\n') and "[[reports/2026-09-06-summary]]" in stub
    assert "Video A" in stub


def test_run_output_twice_lists_both_reports_in_stub(state):
    path, nid, vault, rp = state
    run_output(path, nid, vault, log=lambda m: None)
    new_rp = report_path_for(vault, "ai-research", date(2026, 9, 7))
    db.create_merge(path, nid, None, "m", json.dumps(DOC), json.dumps(DOC["source_map"]), "{}", new_rp)
    run_output(path, nid, vault, log=lambda m: None)
    stub = open(os.path.join(vault, "ai-research", "_notebook.md"), encoding="utf-8").read()
    assert "[[reports/2026-09-06-summary]]" in stub
    assert "[[reports/2026-09-07-summary]]" in stub


def test_run_output_keeps_report_own_date_from_filename(state):
    # Re-rendering an older merge must keep that report's original date, not today's.
    path, nid, vault, rp = state
    old_rp = report_path_for(vault, "ai-research", date(2020, 1, 1))
    db.create_merge(path, nid, None, "m", json.dumps(DOC), json.dumps(DOC["source_map"]), "{}", old_rp)
    out = run_output(path, nid, vault, log=lambda m: None)
    assert out == old_rp
    content = open(old_rp, encoding="utf-8").read()
    assert "Generated: 2020-01-01" in content


def test_run_output_falls_back_when_report_path_outside_vault(tmp_path, state):
    path, nid, vault, rp = state
    outside = str(tmp_path / "elsewhere" / "2026-09-06-summary.md")
    db.create_merge(path, nid, None, "m", json.dumps(DOC), json.dumps(DOC["source_map"]), "{}", outside)
    logs = []
    out = run_output(path, nid, vault, log=logs.append)
    assert out == report_path_for(vault, "ai-research")
    assert os.path.exists(out)
    assert not os.path.exists(outside)
    assert any("outside vault_dir" in m for m in logs)


def test_write_notebook_stub_directly_quotes_ambiguous_slug(tmp_path):
    path = str(tmp_path / "s.sqlite"); db.init_db(path)
    nid = db.create_notebook(path, "Topic", "2026")
    vault = str(tmp_path / "vault")
    stub_path = write_notebook_stub(vault, db.get_notebook(path, nid), [], [])
    content = open(stub_path, encoding="utf-8").read()
    assert content.startswith('---\nid: "2026"\n')


def test_write_notebook_stub_escapes_the_site_name(tmp_path):
    path = str(tmp_path / "s.sqlite"); db.init_db(path)
    nid = db.create_notebook(path, "Topic", "topic")
    db.add_source(path, nid, "web-0123456789abcdef", "https://e.org/a", kind="web", title="A page",
                  channel="[Evil](https://x.example) | Site")
    stub = write_notebook_stub(str(tmp_path / "vault"), db.get_notebook(path, nid),
                               db.get_notebook_sources(path, nid), [])
    content = open(stub, encoding="utf-8").read()
    assert "- [A page](https://e.org/a) — \\[Evil\\](https://x.example) \\| Site" in content


# --- corrections -------------------------------------------------------------

def test_out_of_range_claim_ref_is_a_render_error(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    bad = json.loads(json.dumps(DOC))
    bad["sections"][0]["subsections"][0]["points"][1]["claim_refs"] = [{"source": "S2", "claim": 99}]
    with pytest.raises(RenderError, match="S2"):
        render_markdown("T", MergedDocument.model_validate(bad), sources)


def test_unanchored_claim_ref_is_a_render_error(state):
    # A claim that has no anchor at all (anchor: None) must not be silently skipped.
    path, nid, vault, rp = state
    a = [s for s in db.get_notebook_sources(path, nid) if s["video_id"] == "aaaaaaaaaaa"][0]
    analysis = _analysis(754.0)
    analysis["key_claims"][0]["anchor"] = None
    db.update_analyze_status(path, a["id"], "done", analysis_json=json.dumps(analysis))
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    doc = json.loads(json.dumps(DOC))
    doc["sections"][0]["subsections"][0]["points"][1]["sources"] = ["S1"]
    doc["sections"][0]["subsections"][0]["points"][1]["claim_refs"] = [{"source": "S1", "claim": 0}]
    with pytest.raises(RenderError, match="S1"):
        render_markdown("T", MergedDocument.model_validate(doc), sources)


def test_untrusted_title_is_escaped_in_link_labels(state):
    path, nid, vault, rp = state
    a = [s for s in db.get_notebook_sources(path, nid) if s["video_id"] == "aaaaaaaaaaa"][0]
    db.update_source_metadata(path, a["id"], title="Evil] title\nwith newline")
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    md = render_markdown("T", MergedDocument.model_validate(DOC), sources)
    assert "Evil\\] title with newline" in md
    assert "with newline" in md and "\nwith newline" not in md


def test_escaped_backslash_in_title_cannot_open_a_new_link(state):
    # A title containing a literal `\]` must not let CommonMark read the doubled backslash
    # as an escaped backslash followed by an unescaped `]`, which would close the link label
    # early and turn whatever follows into a new link's target.
    path, nid, vault, rp = state
    a = [s for s in db.get_notebook_sources(path, nid) if s["video_id"] == "aaaaaaaaaaa"][0]
    evil = "Pwn\\](javascript:alert(1))"
    db.update_source_metadata(path, a["id"], title=evil)
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    md = render_markdown("T", MergedDocument.model_validate(DOC), sources)
    # The naive fix (escape `]`/`[` but not `\`) would leave the "]" unescaped -- CommonMark
    # reads "\\" (untouched original backslash + new escape backslash) as one literal
    # backslash, then the bare "]" closes the label early and "(javascript:alert(1))"
    # becomes a new link's destination. That vulnerable rendering must not appear.
    vulnerable_label = evil.replace("]", "\\]")
    assert f"[{vulnerable_label} 12:34](https://www.youtube.com/watch?v=aaaaaaaaaaa&t=754s)" not in md
    # The correct fix escapes the backslash first, so the "]" stays escaped and the label
    # (and the real deep-link target after it) survive intact.
    expected_label = evil.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    assert f"[{expected_label} 12:34](https://www.youtube.com/watch?v=aaaaaaaaaaa&t=754s)" in md


def test_pipe_in_title_escaped_in_sources_table(state):
    path, nid, vault, rp = state
    a = [s for s in db.get_notebook_sources(path, nid) if s["video_id"] == "aaaaaaaaaaa"][0]
    db.update_source_metadata(path, a["id"], title="Weird | Title")
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    md = render_markdown("T", MergedDocument.model_validate(DOC), sources)
    assert "| S1 | youtube | Weird \\| Title | Chan A |" in md


def test_escaped_backslash_in_title_cannot_break_table_row(state):
    path, nid, vault, rp = state
    a = [s for s in db.get_notebook_sources(path, nid) if s["video_id"] == "aaaaaaaaaaa"][0]
    evil = "Weird\\|Title"
    db.update_source_metadata(path, a["id"], title=evil)
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    md = render_markdown("T", MergedDocument.model_validate(DOC), sources)
    expected = evil.replace("\\", "\\\\").replace("|", "\\|")
    assert f"| S1 | youtube | {expected} | Chan A |" in md


def test_newline_injection_in_headings_and_tldr_is_collapsed(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    doc = json.loads(json.dumps(DOC))
    doc["tldr"] = ["Safe bullet\n## Injected TLDR"]
    doc["sections"][0]["title"] = "Fundamentals\n## Injected Section"
    md = render_markdown("Topic\n## Injected Topic", MergedDocument.model_validate(doc), sources)
    lines = md.splitlines()
    assert not any(l.strip().startswith("## Injected") for l in lines)
    assert "Injected TLDR" in md and "Injected Section" in md and "Injected Topic" in md


def test_deep_link_rejects_bad_video_id():
    with pytest.raises(RenderError):
        deep_link("not-a-video-id!!", 1.0)


def test_render_markdown_with_zero_sections_does_not_crash(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    doc = json.loads(json.dumps(DOC))
    doc["sections"] = []
    md = render_markdown("T", MergedDocument.model_validate(doc), sources)
    assert "## TL;DR" in md and "## Sources" in md


def test_contradictions_only_document_renders_without_tldr_bullets(state):
    # run_merge stores a contradictions-only document (no points survived claim-ref validation)
    # with an empty TL;DR: the renderer says so in a non-bullet line instead of an empty heading.
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    doc = json.loads(json.dumps(DOC))
    doc["tldr"] = []
    doc["sections"][0]["subsections"] = []
    md = render_markdown("T", MergedDocument.model_validate(doc), sources)
    lines = md.splitlines()
    tldr = lines[lines.index("## TL;DR") + 1]
    assert tldr.startswith("_") and "no point survived" in tldr
    assert "## 1. Fundamentals" in md and "###" not in md
    assert "**Contradiction — Timeline:**" in md and "| Timeline | 5 years |" in md


# --- claim_counts pins the merge to the analyses it saw ---------------------

def test_matching_claim_counts_render_as_before(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    doc = json.loads(json.dumps(DOC))
    doc["claim_counts"] = {"S1": 2, "S2": 2}
    md = render_markdown("T", MergedDocument.model_validate(doc), sources)
    assert "- Both agree" in md and "- Only one" in md


def test_reanalyzed_source_with_different_claim_count_is_a_render_error(state):
    path, nid, vault, rp = state
    a = [s for s in db.get_notebook_sources(path, nid) if s["video_id"] == "aaaaaaaaaaa"][0]
    reanalyzed = _analysis(754.0)
    reanalyzed["key_claims"].append(
        {"claim": "c2", "quote": "a third quote",
         "anchor": {"char_start": 20, "char_end": 25, "t_start": 900.0, "score": 90}})
    db.update_analyze_status(path, a["id"], "done", analysis_json=json.dumps(reanalyzed))
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    doc = json.loads(json.dumps(DOC))
    doc["claim_counts"] = {"S1": 2, "S2": 2}
    with pytest.raises(RenderError, match="re-analyzed"):
        render_markdown("T", MergedDocument.model_validate(doc), sources)


def test_reanalyzed_source_with_same_claim_count_is_a_render_error(state):
    # Same number of claims, different content: a count check cannot see this, the hash must.
    path, nid, vault, rp = state
    a = [s for s in db.get_notebook_sources(path, nid) if s["video_id"] == "aaaaaaaaaaa"][0]
    reanalyzed = _analysis(754.0)
    reanalyzed["key_claims"][0]["claim"] = "a different claim"
    reanalyzed["key_claims"][0]["quote"] = "a different quote"
    db.update_analyze_status(path, a["id"], "done", analysis_json=json.dumps(reanalyzed))
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    with pytest.raises(RenderError, match="S1 was re-analyzed"):
        render_markdown("T", MergedDocument.model_validate(DOC), sources)


def test_reanalysis_of_an_uncited_source_is_still_a_render_error(state):
    # Every source the merge resolved against is pinned, not only the ones a point cites: the
    # outline, TL;DR and contradictions were written from all of them.
    path, nid, vault, rp = state
    doc = json.loads(json.dumps(DOC))
    doc["sections"][0]["subsections"][0]["points"] = [doc["sections"][0]["subsections"][0]["points"][1]]   # S2 only
    a = [s for s in db.get_notebook_sources(path, nid) if s["video_id"] == "aaaaaaaaaaa"][0]
    reanalyzed = _analysis(754.0)
    reanalyzed["key_claims"][1]["claim"] = "changed"
    db.update_analyze_status(path, a["id"], "done", analysis_json=json.dumps(reanalyzed))
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    with pytest.raises(RenderError, match="S1 was re-analyzed"):
        render_markdown("T", MergedDocument.model_validate(doc), sources)


def _legacy_doc(claim_counts=None):
    """DOC as stored before claim_hashes existed (optionally with the older claim_counts)."""
    doc = json.loads(json.dumps(DOC))
    del doc["claim_hashes"]
    if claim_counts is not None:
        doc["claim_counts"] = claim_counts
    return doc


def test_legacy_merge_without_claim_hashes_renders_with_a_warning(state):
    # Refusing a pre-hash merge outright would force a billed `hmd analyze` just to re-render;
    # it falls back to the claim_counts check and run_output warns once that same-count
    # re-analysis cannot be detected.
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    md = render_markdown("T", MergedDocument.model_validate(_legacy_doc({"S1": 2, "S2": 2})), sources)
    assert "- Both agree" in md and "- Only one" in md
    db.create_merge(path, nid, None, "m", json.dumps(_legacy_doc({"S1": 2, "S2": 2})),
                    json.dumps(DOC["source_map"]), "{}", rp)
    logs = []
    run_output(path, nid, vault, log=logs.append)
    warnings = [m for m in logs if "predates claim hashes" in m]
    assert len(warnings) == 1 and "\n" not in warnings[0] and "hmd analyze ai-research" in warnings[0]


def test_legacy_merge_without_claim_counts_either_still_renders(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    md = render_markdown("T", MergedDocument.model_validate(_legacy_doc()), sources)
    assert "- Both agree" in md


def test_legacy_merge_count_mismatch_is_still_refused(state):
    path, nid, vault, rp = state
    a = [s for s in db.get_notebook_sources(path, nid) if s["video_id"] == "aaaaaaaaaaa"][0]
    reanalyzed = _analysis(754.0)
    reanalyzed["key_claims"].append(
        {"claim": "c2", "quote": "a third quote",
         "anchor": {"char_start": 20, "char_end": 25, "t_start": 900.0, "score": 90}})
    db.update_analyze_status(path, a["id"], "done", analysis_json=json.dumps(reanalyzed))
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    with pytest.raises(RenderError, match="S1 was re-analyzed after the merge \\(2 claims then, 3 now\\)"):
        render_markdown("T", MergedDocument.model_validate(_legacy_doc({"S1": 2, "S2": 2})), sources)


def test_legacy_merge_same_count_reanalysis_is_not_detected(state):
    # The known limitation the warning describes: without hashes, only a count change is caught.
    path, nid, vault, rp = state
    a = [s for s in db.get_notebook_sources(path, nid) if s["video_id"] == "aaaaaaaaaaa"][0]
    reanalyzed = _analysis(754.0)
    reanalyzed["key_claims"][0]["quote"] = "a different quote"
    db.update_analyze_status(path, a["id"], "done", analysis_json=json.dumps(reanalyzed))
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    md = render_markdown("T", MergedDocument.model_validate(_legacy_doc({"S1": 2, "S2": 2})), sources)
    assert "- Both agree" in md


def test_merge_missing_one_claim_hash_is_refused(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    doc = json.loads(json.dumps(DOC))
    del doc["claim_hashes"]["S2"]
    with pytest.raises(RenderError, match="S2"):
        render_markdown("T", MergedDocument.model_validate(doc), sources)


def test_unreadable_analysis_json_is_a_render_error(state):
    path, nid, vault, rp = state
    a = [s for s in db.get_notebook_sources(path, nid) if s["video_id"] == "aaaaaaaaaaa"][0]
    db.update_analyze_status(path, a["id"], "done", analysis_json="{}")
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    with pytest.raises(RenderError, match="unreadable"):
        render_markdown("T", MergedDocument.model_validate(DOC), sources)


# --- leaked merge-internal source tokens (2026-10-09 golden findings) ------------------------------

def test_leaked_source_tokens_are_replaced_with_channel_and_unknown_tokens_kept(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    doc = json.loads(json.dumps(DOC))
    doc["tldr"] = ["S1 and S2 disagree; S10 and S3 are not in this merge"]
    p1 = doc["sections"][0]["subsections"][0]["points"][0]
    p1["text"] = "Per S2, the Galaxy S2 and S1+ differ (S1)"
    doc["sections"][0]["contradictions"] = [
        {"topic": "S1 vs S2 timeline", "position_a": "S1: 5 years | soon", "source_a": "S1",
         "position_b": "twenty years, per S2", "source_b": "S2"}]
    md = render_markdown("T", MergedDocument.model_validate(doc), sources)
    assert "- Chan A and Chan B disagree; S10 and S3 are not in this merge" in md
    # Product names that look like tokens are kept ("Galaxy S2", "S1+"); the bare tokens are replaced.
    assert "- Per Chan B, the Galaxy S2 and S1+ differ (Chan A) — " in md
    assert '**Contradiction — Chan A vs Chan B timeline:**' in md
    assert 'claims "Chan A: 5 years | soon"' in md and 'claims "twenty years, per Chan B"' in md
    # Table cells keep their escaping after the substitution.
    assert "| Chan A vs Chan B timeline | Chan A: 5 years \\| soon | Video A | twenty years, per Chan B | Video B |" in md
    # Source tokens still label the Sources table (that column is the token itself, by design).
    assert "| S1 | youtube | Video A |" in md


def test_source_labeler_falls_back_to_short_title_when_channels_collide():
    from pipeline.stages.output import SHORT_TITLE_CHARS, SourceLabeler
    rows = {1: {"channel": "Same", "title": "A very long video title that keeps going well past the limit",
                "video_id": "aaaaaaaaaaa"},
            2: {"channel": "Same", "title": "Short [one]", "video_id": "bbbbbbbbbbb"},
            3: {"channel": "", "title": "", "video_id": "ccccccccccc"}}
    label = SourceLabeler({"S1": 1, "S2": 2, "S3": 3}, rows)
    out = label("S1, S2 and S3; S4 and xS1 and S1x untouched")
    first = out.split(",")[0]
    assert first.endswith("…") and len(first) <= SHORT_TITLE_CHARS + 1 and first.startswith("A very long video")
    assert ", Short [one] and ccccccccccc; S4 and xS1 and S1x untouched" in out
    # A label containing a token-shaped string is not substituted again (single pass).
    two = {1: {"channel": "S2 Reviews", "title": "t"}, 2: {"channel": "Other", "title": "u"}}
    assert SourceLabeler({"S1": 1, "S2": 2}, two)("S1 S2") == "S2 Reviews Other"


def test_partly_supported_point_renders_a_visible_marker(state):
    from pipeline.stages.output import PARTLY_SUPPORTED, RENDER_VERSION
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    doc = json.loads(json.dumps(DOC))
    doc["sections"][0]["subsections"][0]["points"][0]["support"] = "partial"
    md = render_markdown("AI Research", MergedDocument.model_validate(doc), sources, today=date(2026, 9, 6))
    line = next(l for l in md.splitlines() if l.startswith("- Both agree"))
    assert PARTLY_SUPPORTED == "(partly supported)"
    assert line.startswith("- Both agree (partly supported) — [Video A 12:34]")
    # IDEA-307b: a partly supported joint point is not "confirmed", it only cites N sources
    assert line.endswith(" (2 sources)") and "confirmed by" not in line
    only = next(l for l in md.splitlines() if l.startswith("- Only one"))
    assert "partly supported" not in only
    assert RENDER_VERSION == "render-2026-10-18.3"


def test_disputed_joint_point_is_not_confirmed(state):
    doc = json.loads(json.dumps(DOC))
    doc["sections"][0]["contradictions"][0]["claims_a"] = [0]       # S1 claim 0 = one of P1's claims
    line = next(l for l in _md(state, doc).splitlines() if l.startswith("- Both agree"))
    assert "(disputed)" in line and line.endswith(" (2 sources)") and "confirmed by" not in line


def test_joint_point_with_refs_from_a_fold_is_not_confirmed(state):
    doc = json.loads(json.dumps(DOC))
    doc["sections"][0]["subsections"][0]["points"][0]["refs_from_fold"] = True
    line = next(l for l in _md(state, doc).splitlines() if l.startswith("- Both agree"))
    assert line.endswith(" (2 sources)") and "confirmed by" not in line and "partly supported" not in line


def test_supported_undisputed_joint_point_stays_confirmed(state):
    line = next(l for l in _md(state).splitlines() if l.startswith("- Both agree"))
    assert line.endswith(" (confirmed by 2 sources)")


def _md(state, doc=None, **kw):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    return render_markdown("AI Research", MergedDocument.model_validate(doc or DOC), sources,
                           today=date(2026, 9, 6), **kw)


def test_contents_lists_sections_with_anchors_before_the_first_section(state):
    md = _md(state)
    lines = md.splitlines()
    assert "## Contents" in lines
    assert lines.index("## Contents") < lines.index("## 1. Fundamentals")
    assert "1. [Fundamentals](#1-fundamentals)" in lines
    assert "- [Sources](#sources)" not in md          # contents entries are numbered, never "- " bullets


def test_section_lists_its_sources_by_point_count(state):
    md = _md(state)
    lines = md.splitlines()
    i = lines.index("## 1. Fundamentals")
    assert lines[i + 2] == "_Sources: Chan B (2), Chan A (1)_"     # S2 in P1 and P2, S1 in P1 only


def test_contradiction_shows_dates_and_a_gap_note(state):
    path, nid, vault, rp = state
    db.update_source_metadata(path, 2, upload_date="20260601")      # Video B, five months after A
    md = _md(state)
    line = next(l for l in md.splitlines() if l.startswith("**Contradiction"))
    assert "(2026-01-02)" in line and "(2026-06-01)" in line
    assert "published 5 months apart; one may be outdated or describe a different edition" in line
    assert "| Timeline | 5 years | Video A | 20 years | Video B | 2026-01-02 / 2026-06-01 |" in md.splitlines()


def test_contradiction_label_may_differ_renders_a_note(state):
    doc = json.loads(json.dumps(DOC))
    doc["sections"][0]["contradictions"][0]["check_label"] = "may_differ_by_edition_or_date"
    md = _md(state, doc)
    line = next(l for l in md.splitlines() if l.startswith("**Contradiction"))
    assert line.startswith("**Contradiction — Timeline (may differ by edition or date):**")
    assert "| Timeline (may differ by edition or date) | 5 years |" in md


def test_contradiction_label_same_conditions_and_old_merges_render_unmarked(state):
    for label in ("same_conditions", None):
        doc = json.loads(json.dumps(DOC))
        if label:
            doc["sections"][0]["contradictions"][0]["check_label"] = label
        md = _md(state, doc)
        assert "**Contradiction — Timeline:**" in md and "may differ" not in md and "same conditions" not in md


def test_no_gap_note_when_a_date_is_missing(state):
    line = next(l for l in _md(state).splitlines() if l.startswith("**Contradiction"))
    assert "apart" not in line


def test_disputed_badge_marks_points_citing_a_contradiction_side(state):
    doc = json.loads(json.dumps(DOC))
    doc["sections"][0]["contradictions"][0]["claims_b"] = [0]       # S2 claim 0 = P2's claim
    md = _md(state, doc)
    line = next(l for l in md.splitlines() if l.startswith("- Only one"))
    assert "(disputed)" in line and line.index("(disputed)") < line.index(" — ")
    both = next(l for l in md.splitlines() if l.startswith("- Both agree"))
    assert "(disputed)" not in both


def test_new_badge_only_for_listed_point_ids(state):
    md = _md(state, new_point_ids={"P2"})
    assert "(new)" in next(l for l in md.splitlines() if l.startswith("- Only one"))
    assert "(new)" not in next(l for l in md.splitlines() if l.startswith("- Both agree"))


def test_badge_legend_is_not_a_bullet(state):
    md = _md(state)
    legend = next(l for l in md.splitlines() if l.startswith("_Badges:"))
    assert "(confirmed by N sources)" in legend and "(disputed)" in legend and "(partly supported)" in legend
    assert "(N sources)" in legend


def test_eval_bullets_still_only_count_points(state):
    from pipeline.evals import _report_bullets
    bullets = _report_bullets(_md(state))
    assert [b[:14] for b in bullets] == ["- Both agree —", "- Only one — ["]


def test_scorecard_table_after_sources(state):
    md = _md(state)
    lines = md.splitlines()
    assert lines.index("## Source scorecard") > lines.index("## Sources")
    assert "| Source | Points | Unique | Shared | Claims used | Contradictions | Worth watching |" in lines
    # Video A (S1): P1 shared; claims used 2 of 2; 1 contradiction
    assert "| Video A | 1 | 0 | 1 | 2 of 2 | 1 |  |" in lines
    assert "| Video B | 2 | 1 | 1 | 2 of 2 | 1 |  |" in lines


def test_single_source_notebook_has_no_star(state):
    doc = json.loads(json.dumps(DOC))
    doc["source_map"] = {"S2": 2}
    doc["claim_hashes"] = {"S2": DOC["claim_hashes"]["S2"]}
    sec = doc["sections"][0]
    sec["contradictions"] = []
    sec["subsections"][0]["points"] = [sec["subsections"][0]["points"][1]]
    md = _md(state, doc)
    assert "★" not in md
    assert "- Single-source points: 1 of 1 (100%)" in md


def test_coverage_section_counts(state):
    md = _md(state)
    assert "## Coverage" in md
    assert "- Single-source points: 1 of 2 (50%)" in md
    assert "| 1. Fundamentals | 2 | 2 | 1 |" in md.splitlines()


def test_contents_links_scorecard_and_coverage(state):
    lines = _md(state).splitlines()
    assert any(l.endswith("](#source-scorecard)") for l in lines)
    assert any(l.endswith("](#coverage)") for l in lines)


def test_report_path_for_names_the_format():
    assert report_path_for("v", "slug", date(2026, 9, 6), "digest").replace("\\", "/") == "v/slug/reports/2026-09-06-digest.md"


def test_stub_lists_every_format_once(state, tmp_path):
    path, nid, vault, rp = state
    run_output(path, nid, vault)
    reports = os.path.dirname(rp)
    for name in ("2026-09-06-digest.md", "2026-09-06-outline.md"):
        with open(os.path.join(reports, name), "w", encoding="utf-8") as f:
            f.write("x")
    run_output(path, nid, vault)
    with open(os.path.join(vault, "ai-research", "_notebook.md"), encoding="utf-8") as f:
        stub = f.read()
    assert stub.count("[[reports/2026-09-06-summary]]") == 1
    assert "[[reports/2026-09-06-digest]]" in stub and "[[reports/2026-09-06-outline]]" in stub


def test_unknown_format_is_a_render_error(state):
    path, nid, vault, rp = state
    with pytest.raises(RenderError, match="unknown report format"):
        run_output(path, nid, vault, formats=["summary", "slides"])


def test_format_without_renderer_is_logged_and_skipped(state, monkeypatch):
    from pipeline.stages.output import FORMAT_RENDERERS
    monkeypatch.delitem(FORMAT_RENDERERS, "quotes")
    path, nid, vault, rp = state
    lines = []
    assert run_output(path, nid, vault, formats=["summary", "quotes"], log=lines.append) == rp
    assert "format quotes not available yet; skipped" in lines


def test_contents_lists_sources_not_analyzed_when_rendered(state):
    path, nid, vault, rp = state
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    md = render_markdown("AI Research", MergedDocument.model_validate(DOC), sources, today=date(2026, 9, 6))
    contents = md.split("## Contents")[1].split("## 1.")[0]
    assert "3. [Sources not analyzed](#sources-not-analyzed)" in contents
    assert contents.index("[Sources]") < contents.index("[Sources not analyzed]") < contents.index("[Source scorecard]")
    analyzed_only = {sid: s for sid, s in sources.items() if sid in DOC["source_map"].values()}
    md2 = render_markdown("AI Research", MergedDocument.model_validate(DOC), analyzed_only, today=date(2026, 9, 6))
    assert "Sources not analyzed" not in md2


def test_bracket_title_cannot_become_a_link_in_the_tables(state):
    path, nid, vault, rp = state
    a = [s for s in db.get_notebook_sources(path, nid) if s["video_id"] == "aaaaaaaaaaa"][0]
    db.update_source_metadata(path, a["id"], title="x](https://evil) [y", channel="[c](https://evil)")
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    md = render_markdown("T", MergedDocument.model_validate(DOC), sources)
    tables = md.split("## Sources", 1)[1]
    assert "](https://evil)" not in tables.replace(r"\](https://evil)", "")
    assert r"| S1 | youtube | x\](https://evil) \[y | \[c\](https://evil) |" in md
    row = next(l for l in md.splitlines() if l.startswith("| Timeline |"))
    assert r"x\](https://evil) \[y" in row


def test_mixed_notebook_renders_every_kind(state):
    path, nid, vault, rp = state
    web = db.add_source(path, nid, "web-0123456789abcdef", "https://e.org/a", kind="web", title="An article", channel="Example Mag")
    pdf = db.add_source(path, nid, "pdf-0123456789abcdef", "file:///C:/docs/p.pdf", kind="pdf", title="A paper")
    for sid in (web, pdf):
        db.update_analyze_status(path, sid, "done", analysis_json=json.dumps(
            {"summary": "s", "key_claims": [{"claim": "c", "quote": "Small changes compound over months",
             "anchor": {"char_start": 0, "char_end": 30, "t_start": 0.0, "score": 100, "segment": 2, "page": 7}}]}))
    doc = json.loads(json.dumps(DOC))
    doc["source_map"].update({"S3": web, "S4": pdf})
    doc["sections"][0]["subsections"][0]["points"].append(
        {"id": "P9", "text": "Compounding", "sources": ["S3", "S4"], "claim_refs": [{"source": "S3", "claim": 0}, {"source": "S4", "claim": 0}]})
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    _rehash(doc, sources)
    md = render_markdown("AI Research", MergedDocument.model_validate(doc), sources, today=date(2026, 9, 6))
    line = next(l for l in md.splitlines() if l.startswith("- Compounding"))
    assert "[An article \u00b6 3](https://e.org/a#:~:text=Small%20changes%20compound%20over%20months)" in line
    assert "[A paper p. 7](file:///C:/docs/p.pdf#page=7)" in line
    assert "Sources: 2 videos, 1 article, 1 PDF, 3 channels" in md
    assert "| S3 | web | An article |" in md
    assert "| # | Kind | Title | Channel | Published | Length | URL |" in md


def test_contradiction_link_urls_with_parentheses_are_encoded_for_both_sources(state):
    path, nid, vault, rp = state
    wiki = "https://en.wikipedia.org/wiki/Foo_(bar)"
    s1 = db.add_source(path, nid, "web-1111111111111111", "https://en.wikipedia.org/wiki/A_(x)", kind="web", title="A")
    s2 = db.add_source(path, nid, "web-2222222222222222", wiki, kind="web", title="B")
    for sid in (s1, s2):
        db.update_analyze_status(path, sid, "done", analysis_json=json.dumps(_analysis(0.0)))
    doc = json.loads(json.dumps(DOC))
    doc["source_map"].update({"S3": s1, "S4": s2})
    c = doc["sections"][0]["contradictions"][0]
    c["source_a"], c["source_b"] = "S3", "S4"
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    _rehash(doc, sources)
    md = render_markdown("AI Research", MergedDocument.model_validate(doc), sources, today=date(2026, 9, 6))
    line = next(l for l in md.splitlines() if l.startswith("**Contradiction"))
    assert "](https://en.wikipedia.org/wiki/A_%28x%29)" in line
    assert "](https://en.wikipedia.org/wiki/Foo_%28bar%29)" in line


def test_header_counts_a_row_with_an_unknown_kind_as_a_source(state):
    path, nid, vault, rp = state
    odd = db.add_source(path, nid, "zzz-0123456789abcdef", "https://e.org/z", kind="hologram", title="Odd")
    db.update_analyze_status(path, odd, "done", analysis_json=json.dumps(_analysis(0.0)))
    doc = json.loads(json.dumps(DOC))
    doc["source_map"]["S3"] = odd
    sources = {s["id"]: s for s in db.get_notebook_sources(path, nid)}
    _rehash(doc, sources)
    md = render_markdown("AI Research", MergedDocument.model_validate(doc), sources, today=date(2026, 9, 6))
    assert "Sources: 2 videos, 1 source," in md


def test_html_and_markdown_in_untrusted_text_are_neutralised():
    from pipeline.stages.output import _md_inline, _md_link_text, _md_text
    evil = '<a href="javascript:alert(1)">x</a>'
    for fn in (_md_inline, _md_link_text, _md_text):
        assert "<a" not in fn(evil) and "&lt;a" in fn(evil)
    assert "![](" not in _md_text("see ![](https://x/y) and [click](https://x)")
    assert "](" not in _md_text("see ![](https://x/y) and [click](https://x)").replace("\\]", "").replace("\\[", "")


def test_ordinary_text_stays_readable():
    from pipeline.stages.output import _md_text
    assert _md_text("R&D at Bob's (2025) costs $5") == "R&D at Bob's (2025) costs $5"
    assert _md_text("a &lt; b") == "a &amp;lt; b"


# --- HTML twins (IDEA-312) ---------------------------------------------------------------------

def _with_previous_merge(state):
    path, nid, vault, rp = state
    db.create_merge(path, nid, None, "m", json.dumps(DOC), json.dumps(DOC["source_map"]), "{}", rp)


def test_render_reports_writes_twins_for_written_formats(state):
    from pipeline.stages.output import html_twin_path, render_reports
    path, nid, vault, rp = state
    _with_previous_merge(state)
    files = render_reports(path, nid, vault, formats=["summary", "digest", "diff"], html=True, log=lambda m: None)
    assert files.report_path == rp and sorted(files.html_paths) == ["diff", "digest", "summary"]
    assert files.html_paths["summary"] == html_twin_path(rp) == rp[:-3] + ".html"
    for fmt, p in files.html_paths.items():
        page = open(p, encoding="utf-8").read()
        assert os.path.exists(p[:-5] + ".md")
        for other in ("summary", "digest", "diff"):
            assert f"2026-09-06-{other}.html" in page or other == fmt


def test_markdown_identical_with_html_on_and_off(state):
    from pipeline.stages.output import render_reports
    path, nid, vault, rp = state
    fmts = ["summary", "digest", "hybrid", "outline", "quotes"]
    render_reports(path, nid, vault, formats=fmts, html=False, log=lambda m: None)
    off = {f: open(rp.replace("-summary.md", f"-{f}.md"), "rb").read() for f in fmts}
    render_reports(path, nid, vault, formats=fmts, html=True, log=lambda m: None)
    assert off == {f: open(rp.replace("-summary.md", f"-{f}.md"), "rb").read() for f in fmts}


def test_no_html_writes_none_and_deletes_nothing(state):
    from pipeline.stages.output import render_reports
    path, nid, vault, rp = state
    render_reports(path, nid, vault, formats=["summary"], html=True, log=lambda m: None)
    twin = rp[:-3] + ".html"
    before = open(twin, "rb").read()
    files = render_reports(path, nid, vault, formats=["summary", "digest"], html=False, log=lambda m: None)
    assert files.html_paths == {} and open(twin, "rb").read() == before
    assert not os.path.exists(rp.replace("-summary.md", "-digest.html"))


def test_failed_twin_warns_and_is_skipped(state, monkeypatch):
    from pipeline.stages import html_render
    from pipeline.stages.output import render_reports

    def boom(ctx):
        raise ValueError("bad")
    monkeypatch.setitem(html_render.HTML_RENDERERS, "digest", boom)
    path, nid, vault, rp = state
    logs = []
    files = render_reports(path, nid, vault, formats=["summary", "digest"], html=True, log=logs.append)
    assert any(l.startswith("WARNING: the digest HTML report failed (ValueError: bad); skipped") for l in logs)
    assert sorted(files.html_paths) == ["summary"]
    assert os.path.exists(rp.replace("-summary.md", "-digest.md"))
    assert "-digest.html" not in open(files.html_paths["summary"], encoding="utf-8").read()


def test_failed_summary_twin_is_not_reported(state, monkeypatch):
    from pipeline.stages import html_render
    from pipeline.stages.output import render_reports

    def boom(ctx):
        raise ValueError("bad")
    monkeypatch.setattr(html_render, "render_summary_html", boom)
    path, nid, vault, rp = state
    files = render_reports(path, nid, vault, formats=["summary"], html=True, log=lambda m: None)
    assert "summary" not in files.html_paths and os.path.exists(rp)


def test_stub_lists_html_twins_of_this_render(state):
    from pipeline.stages.output import render_reports
    path, nid, vault, rp = state
    render_reports(path, nid, vault, formats=["summary", "digest"], html=True, log=lambda m: None)
    stub = open(os.path.join(vault, "ai-research", "_notebook.md"), encoding="utf-8").read()
    assert "## HTML reports" in stub and "reports/2026-09-06-digest.html" in stub
    render_reports(path, nid, vault, formats=["summary"], html=False, log=lambda m: None)
    stub = open(os.path.join(vault, "ai-research", "_notebook.md"), encoding="utf-8").read()
    assert "## HTML reports" not in stub


def test_run_output_keeps_its_signature_and_default(state):
    path, nid, vault, rp = state
    assert run_output(path, nid, vault, log=lambda m: None) == rp
    assert not os.path.exists(rp[:-3] + ".html")                       # html defaults to off here
