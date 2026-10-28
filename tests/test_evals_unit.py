from tests.helpers import yt_ref
import json, os
import pytest
from pipeline.config import Config
from pipeline.evals import discover_cases, load_manifest, run_golden, check_golden, export_golden, list_goldens
from pipeline.llm.replay import ReplayProvider
from pipeline.transcription.cache import TranscriptCache
from tests.helpers import ScriptedProvider, StubTranscriber, fake_metadata

VIDS = ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"]


def _scripted_replay(cfg, name=None, cassette_dir=None, record=False):
    """Stand-in for make_provider: cassettes backed by the scripted provider when recording."""
    return ReplayProvider(cassette_dir, inner=ScriptedProvider() if record else None, record=record)


def _make_golden(root, name="synthetic", must_contain=("Second takeaway",)):
    gdir = root / "golden" / name
    tdir = gdir / "transcripts"
    cache = TranscriptCache(str(tdir))
    for v in VIDS:
        cache.save(StubTranscriber().transcribe(yt_ref(v), ["en"]))
    manifest = {"topic": "Synthetic topic",
                "sources": [{"video_id": v, "title": f"Video {v[:4]}", "channel": "Chan", "duration_seconds": 20,
                             "upload_date": "20260101", "transcript_method": "youtube_captions"} for v in VIDS],
                "assertions": {"min_sections": 2, "min_points": 3, "must_contain": list(must_contain),
                               "min_contradictions": 0, "max_dropped_ratio": 0.3}}
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return gdir


def test_load_manifest_and_list(tmp_path):
    gdir = _make_golden(tmp_path)
    m = load_manifest(str(gdir))
    assert m["topic"] == "Synthetic topic" and len(m["sources"]) == 3
    assert list_goldens(str(tmp_path / "golden")) == ["synthetic"]


def test_record_then_replay_and_assertions_pass(tmp_path, monkeypatch):
    gdir = _make_golden(tmp_path)
    cassettes = gdir / "cassettes" / "anthropic_api"
    monkeypatch.setattr("pipeline.evals.make_provider", _scripted_replay)
    res = run_golden(str(gdir), provider_name="anthropic_api", record=True, home=str(tmp_path / "home1"))
    assert res.status == "done" and os.path.exists(res.report_path)
    assert res.provider.misses > 0 and len(os.listdir(cassettes)) == res.provider.misses
    assert res.provider.recorded == res.provider.misses
    assert check_golden(res, load_manifest(str(gdir))) == []
    res2 = run_golden(str(gdir), provider_name="anthropic_api", record=False, home=str(tmp_path / "home2"))
    assert res2.status == "done" and res2.provider.misses == 0 and res2.provider.hits > 0
    assert res2.cost == 0.0
    assert check_golden(res2, load_manifest(str(gdir))) == []


def test_assertions_fail_on_missing_detail(tmp_path, monkeypatch):
    gdir = _make_golden(tmp_path, must_contain=("the number forty-two",))
    monkeypatch.setattr("pipeline.evals.make_provider", _scripted_replay)
    res = run_golden(str(gdir), provider_name="anthropic_api", record=True, home=str(tmp_path / "home"))
    failures = check_golden(res, load_manifest(str(gdir)))
    assert len(failures) == 1 and "forty-two" in failures[0]


def test_export_golden_from_a_run_notebook(tmp_path):
    from pipeline import db
    from pipeline.runner import Runner, STAGES
    cfg = Config(home=str(tmp_path / "home"), run={"budget_usd": 0})
    db.init_db(cfg.db_path)
    r = Runner(cfg, provider=ScriptedProvider(), transcribers=[StubTranscriber()], metadata_fn=fake_metadata, log=lambda m: None)
    nb = r.create_notebook("Exported")
    urls = tmp_path / "u.txt"; urls.write_text("".join(f"https://youtu.be/{v}\n" for v in VIDS), encoding="utf-8")
    r.run(nb["id"], STAGES, input_file=str(urls))
    gdir = export_golden(cfg, "exported", str(tmp_path / "golden"), "exp")
    m = load_manifest(gdir)
    assert m["topic"] == "Exported" and [s["video_id"] for s in m["sources"]] == VIDS
    assert all(os.path.exists(os.path.join(gdir, "transcripts", v, "youtube_captions.json")) for v in VIDS)
    assert m["assertions"]["must_contain"] == []          # owner fills these in by hand


def test_export_golden_rejects_unsafe_name(tmp_path):
    from pipeline import db as db_mod
    from pipeline.config import ConfigError
    cfg = Config(home=str(tmp_path / "home"), run={"budget_usd": 0})
    db_mod.init_db(cfg.db_path)
    db_mod.create_notebook(cfg.db_path, "T", "t")
    with pytest.raises(ConfigError) as exc_info:
        export_golden(cfg, "t", str(tmp_path / "golden"), "../escape")
    assert "../escape" in str(exc_info.value)


def test_discover_cases_finds_only_recorded_provider(tmp_path, monkeypatch):
    gdir = _make_golden(tmp_path)
    monkeypatch.setattr("pipeline.evals.make_provider", _scripted_replay)
    run_golden(str(gdir), provider_name="anthropic_api", record=True, home=str(tmp_path / "home"))
    root = str(tmp_path / "golden")
    assert discover_cases(root) == [("synthetic", "anthropic_api")]

    # An existing but empty cassette dir for another provider must not count as recorded.
    empty_dir = gdir / "cassettes" / "claude_cli"
    empty_dir.mkdir(parents=True)
    assert discover_cases(root) == [("synthetic", "anthropic_api")]

    # A stray .tmp (partial write from a killed --live run) must not count either.
    (empty_dir / "x.tmp").write_text("partial", encoding="utf-8")
    assert discover_cases(root) == [("synthetic", "anthropic_api")]


def test_discover_cases_empty_or_missing_root(tmp_path):
    assert discover_cases(str(tmp_path / "no-such-dir")) == []
    empty_root = tmp_path / "empty-golden"
    empty_root.mkdir()
    assert discover_cases(str(empty_root)) == []


def test_missing_cassette_on_replay_reports_message_with_live_hint(tmp_path, monkeypatch):
    gdir = _make_golden(tmp_path)
    monkeypatch.setattr("pipeline.evals.make_provider", _scripted_replay)
    run_golden(str(gdir), provider_name="anthropic_api", record=True, home=str(tmp_path / "home1"))
    cassette_dir = gdir / "cassettes" / "anthropic_api"
    # Delete one recorded cassette so replay must miss it.
    next(cassette_dir.iterdir()).unlink()
    res2 = run_golden(str(gdir), provider_name="anthropic_api", record=False, home=str(tmp_path / "home2"))
    assert res2.status == "error"
    failures = check_golden(res2, load_manifest(str(gdir)))
    assert len(failures) == 1
    assert "no cassette" in failures[0] and "--live" in failures[0]


def test_assertion_4_checks_bullets_in_every_subsection_of_section_1():
    """Regression: a naive `report_text.split("## 1.")[-1]` also matches the "## 1." embedded
    inside a "### 1.N" subsection heading, so with two subsections in section 1 it would keep
    only the LAST subsection's bullets and silently miss a missing timestamp in the first one."""
    from pipeline.evals import GoldenResult, check_golden
    from pipeline.schemas import MergedDocument

    report_text = (
        "# Topic - Research Summary\n"
        "Generated: 2026-01-01 | Sources: 1 video, 1 channel\n\n"
        "## TL;DR\n"
        "- a bullet\n\n"
        "## 1. Section one\n\n"
        "### 1.1 First sub\n"
        "- bullet missing its timestamp link\n\n"
        "### 1.2 Second sub\n"
        "- bullet with a timestamp (https://x/watch?v=aaaaaaaaaaa&t=5s)\n\n"
        "## Sources\n\n"
        "| # | Title | Channel | Published | Duration | URL |\n"
        "|---|---|---|---|---|---|\n"
    )
    doc = MergedDocument(tldr=["a", "b", "c"], sections=[], source_map={},
                         prompt_versions={"analyze": "a", "merge": "m", "check": "c", "retract": "x", "render": "r"})
    res = GoldenResult(name="x", status="done", message="", report_path=None, doc=doc, sources=[],
                       analyses={}, transcripts={}, report_text=report_text, cost=0.0, provider=None,
                       run_id=1, db_path=":memory:")
    manifest = {"assertions": {"min_sections": 0, "min_points": 0, "must_contain": [],
                               "min_contradictions": 0, "max_dropped_ratio": 1.0}}
    failures = check_golden(res, manifest)
    assert any("timestamp" in f for f in failures)


def test_assertion_4_accepts_every_kind_of_locator_link_and_rejects_a_bare_url():
    """A mixed notebook cites PDFs by page, web pages by text fragment and Vimeo by #t=; each of
    those is a locator. A bare URL (no locator) still fails the check."""
    from pipeline.evals import GoldenResult, check_golden
    from pipeline.schemas import MergedDocument

    def failures_for(bullets: str) -> list[str]:
        report_text = ("# T\n\n## TL;DR\n- a\n\n## 1. One\n\n### 1.1 Sub\n" + bullets + "\n## Sources\n")
        doc = MergedDocument(tldr=["a", "b", "c"], sections=[], source_map={},
                             prompt_versions={"analyze": "a", "merge": "m", "check": "c", "retract": "x", "render": "r"})
        res = GoldenResult(name="x", status="done", message="", report_path=None, doc=doc, sources=[],
                           analyses={}, transcripts={}, report_text=report_text, cost=0.0, provider=None,
                           run_id=1, db_path=":memory:")
        manifest = {"assertions": {"min_sections": 0, "min_points": 0, "must_contain": [],
                                   "min_contradictions": 0, "max_dropped_ratio": 1.0}}
        return [f for f in check_golden(res, manifest) if "locator" in f]

    assert failures_for("- yt [V 0:05](https://www.youtube.com/watch?v=aaaaaaaaaaa&t=5s)\n"
                        "- pdf [P p. 2](https://e.org/p.pdf#page=2)\n"
                        "- web [W ¶ 3](https://e.org/a#:~:text=Small%20changes)\n"
                        "- vimeo [M 1:00](https://vimeo.com/1#t=60s)\n") == []
    (fail,) = failures_for("- bare [W](https://e.org/a)\n")
    assert fail.startswith("1 bullets without a source locator link")


def test_assertion_4_and_7_fail_on_empty_report():
    from pipeline.evals import GoldenResult, check_golden
    from pipeline.schemas import MergedDocument

    doc = MergedDocument(tldr=["a", "b", "c"], sections=[], source_map={},
                         prompt_versions={"analyze": "a", "merge": "m", "check": "c", "retract": "x", "render": "r"})
    res = GoldenResult(name="x", status="done", message="", report_path=None, doc=doc, sources=[],
                       analyses={}, transcripts={}, report_text="", cost=0.0, provider=None,
                       run_id=1, db_path=":memory:")
    manifest = {"assertions": {"min_sections": 0, "min_points": 0, "must_contain": [],
                               "min_contradictions": 0, "max_dropped_ratio": 1.0}}
    failures = check_golden(res, manifest)
    assert any("empty" in f for f in failures)


def test_golden_root_is_absolute_under_repo_root():
    import os
    from pathlib import Path
    from pipeline.evals import GOLDEN_ROOT
    repo_root = Path(__file__).resolve().parents[1]
    assert os.path.isabs(GOLDEN_ROOT)
    assert GOLDEN_ROOT == str(repo_root / "evals" / "golden")


def test_export_golden_logs_skipped_sources(tmp_path):
    from pipeline import db as db_mod
    cfg = Config(home=str(tmp_path / "home"), run={"budget_usd": 0})
    db_mod.init_db(cfg.db_path)
    nid = db_mod.create_notebook(cfg.db_path, "T", "t")
    db_mod.add_source(cfg.db_path, nid, "aaaaaaaaaaa", "https://youtu.be/aaaaaaaaaaa")
    # Never transcribed: transcribe_status stays 'pending' and transcript_method stays None.
    logs = []
    gdir = export_golden(cfg, "t", str(tmp_path / "golden"), "exp", log=logs.append)
    assert len(logs) == 1
    assert "skipped 1 source(s)" in logs[0] and "aaaaaaaaaaa" in logs[0]
    # load_manifest itself requires a non-empty 'sources' list (a golden with none is useless),
    # so read the raw file here rather than asserting through it.
    with open(os.path.join(gdir, "manifest.json"), encoding="utf-8") as f:
        m = json.load(f)
    assert m["sources"] == []


def test_assertion_7_flags_points_not_found_in_report(tmp_path, monkeypatch):
    gdir = _make_golden(tmp_path)
    monkeypatch.setattr("pipeline.evals.make_provider", _scripted_replay)
    res = run_golden(str(gdir), provider_name="anthropic_api", record=True, home=str(tmp_path / "home"))
    assert check_golden(res, load_manifest(str(gdir))) == []
    # Drop one whole point bullet from the rendered report, so it can no longer be located by
    # prefix -- distinct from assertion 4's "present but missing a timestamp" case.
    lines = res.report_text.splitlines()
    bullet_idx = next(i for i, l in enumerate(lines) if l.startswith("- a point supported by"))
    del lines[bullet_idx]
    res.report_text = "\n".join(lines)
    failures = check_golden(res, load_manifest(str(gdir)))
    assert any("not found in the rendered report" in f for f in failures)


def _negative_case(report_text="", pairs=(("S1", "S2"),)):
    """A finished GoldenResult whose merge holds one contradiction per (token_a, token_b) pair."""
    from pipeline.evals import GoldenResult
    from pipeline.schemas import Contradiction, MergedDocument, MergedSection

    contradictions = [Contradiction(topic="t", position_a="x", source_a=a, position_b="y", source_b=b)
                      for a, b in pairs]
    doc = MergedDocument(tldr=["a", "b", "c"],
                         sections=[MergedSection(title="Sec", subsections=[], contradictions=contradictions)],
                         source_map={"S1": 1, "S2": 2, "S3": 3},
                         prompt_versions={"analyze": "a", "merge": "m", "check": "c", "retract": "x", "render": "r"})
    sources = [{"id": i + 1, "video_id": v} for i, v in enumerate(VIDS)]
    return GoldenResult(name="x", status="done", message="", report_path=None, doc=doc, sources=sources,
                        analyses={}, transcripts={}, report_text=report_text, cost=0.0, provider=None,
                        run_id=1, db_path=":memory:")


def _negative_manifest(**extra):
    a = {"min_sections": 0, "min_points": 0, "must_contain": [], "min_contradictions": 0,
         "max_dropped_ratio": 1.0}
    a.update(extra)
    return {"assertions": a}


def test_must_not_contain_flags_forbidden_text_case_insensitively():
    res = _negative_case(report_text="Habits form in exactly 21 DAYS for everyone.")
    failures = check_golden(res, _negative_manifest(must_not_contain=["21 days for everyone", "absent"]))
    assert len([f for f in failures if "forbidden" in f]) == 1
    assert any("21 days for everyone" in f for f in failures)


def test_forbidden_contradiction_pair_fails_in_either_order():
    res = _negative_case(pairs=[("S2", "S1")])
    failures = check_golden(res, _negative_manifest(forbidden_contradictions=[[VIDS[0], VIDS[1]]]))
    assert any("forbidden contradiction" in f and VIDS[0] in f and VIDS[1] in f for f in failures)


def test_forbidden_contradiction_pair_passes_when_the_pair_is_not_contradicted():
    res = _negative_case(pairs=[("S1", "S3")])
    failures = check_golden(res, _negative_manifest(forbidden_contradictions=[[VIDS[0], VIDS[1]]]))
    assert not any("forbidden contradiction" in f for f in failures)


def test_negative_assertions_default_to_empty(tmp_path):
    m = load_manifest(str(_make_golden(tmp_path)))
    assert m["assertions"]["must_not_contain"] == []
    assert m["assertions"]["forbidden_contradictions"] == []


@pytest.mark.parametrize("bad", [
    [[VIDS[0]]],                          # not a pair
    [[VIDS[0], VIDS[0]]],                 # same source twice
    [[VIDS[0], "zzzzzzzzzzz"]],           # not a source of this golden: the check would pass vacuously
    "aaaaaaaaaaa",                        # not a list of pairs
])
def test_load_manifest_rejects_bad_forbidden_contradictions(tmp_path, bad):
    from pipeline.config import ConfigError
    gdir = _make_golden(tmp_path)
    m = json.loads((gdir / "manifest.json").read_text(encoding="utf-8"))
    m["assertions"]["forbidden_contradictions"] = bad
    (gdir / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ConfigError, match="forbidden_contradictions"):
        load_manifest(str(gdir))


@pytest.mark.parametrize("bad", ["one string", [""], [3]])
def test_load_manifest_rejects_bad_must_not_contain(tmp_path, bad):
    from pipeline.config import ConfigError
    gdir = _make_golden(tmp_path)
    m = json.loads((gdir / "manifest.json").read_text(encoding="utf-8"))
    m["assertions"]["must_not_contain"] = bad
    (gdir / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ConfigError, match="must_not_contain"):
        load_manifest(str(gdir))


def test_assertion_7_finds_partly_supported_bullets():
    """The renderer puts "(partly supported)" after a point's text; check 7 still finds the bullet
    (prefix match on the text) and reads its confirmed_by suffix."""
    from pipeline.evals import GoldenResult, check_golden
    from pipeline.schemas import MergedDocument

    long_text = "A long point text that runs well past forty characters before the marker"
    doc = MergedDocument.model_validate({
        "tldr": ["a", "b", "c"], "source_map": {"S1": 1, "S2": 2},
        "prompt_versions": {"analyze": "a", "merge": "m", "check": "c", "retract": "x", "render": "r"},
        "sections": [{"title": "One", "subsections": [{"title": "Sub", "points": [
            {"id": "P1", "text": "Short", "sources": ["S1"], "claim_refs": [{"source": "S1", "claim": 0}],
             "support": "partial"},
            {"id": "P2", "text": long_text, "sources": ["S1", "S2"], "support": "partial",
             "claim_refs": [{"source": "S1", "claim": 0}, {"source": "S2", "claim": 0}]}]}]}]})
    report_text = (
        "## TL;DR\n- a\n\n## 1. One\n\n### 1.1 Sub\n"
        "- Short (partly supported) — [V 0:05](https://www.youtube.com/watch?v=aaaaaaaaaaa&t=5s)\n"
        f"- {long_text} (partly supported) — [V 0:05](https://www.youtube.com/watch?v=aaaaaaaaaaa&t=5s), "
        "[W 0:05](https://www.youtube.com/watch?v=bbbbbbbbbbb&t=5s) (2 sources)\n\n"
        "## Sources\n")
    sources = [{"id": 1, "video_id": "aaaaaaaaaaa", "title": "V", "channel": "C1"},
               {"id": 2, "video_id": "bbbbbbbbbbb", "title": "W", "channel": "C2"}]
    res = GoldenResult(name="x", status="done", message="", report_path=None, doc=doc, sources=sources,
                       analyses={}, transcripts={}, report_text=report_text, cost=0.0, provider=None,
                       run_id=1, db_path=":memory:")
    manifest = {"assertions": {"min_sections": 0, "min_points": 0, "must_contain": [],
                               "min_contradictions": 0, "max_dropped_ratio": 1.0}}
    failures = check_golden(res, manifest)
    # IDEA-307b: a partly supported joint point renders "(N sources)"; check 7 reads that count too
    assert not [f for f in failures if "not found" in f or "confirmed_by" in f or "timestamp" in f
                or "prompt_versions" in f], failures


def test_assertion_7_matches_points_sharing_a_text_prefix_in_render_order():
    """Two points whose first 40 characters are equal (a story repeated across sections, IDEA-307)
    must each be checked against their own bullet, not the first bullet with that prefix
    (seen in two golden-notebook recordings)."""
    from pipeline.evals import GoldenResult, check_golden
    from pipeline.schemas import MergedDocument

    head = "Newer research suggests habit formation takes"
    doc = MergedDocument.model_validate({
        "tldr": ["a", "b", "c"], "source_map": {"S1": 1, "S2": 2},
        "prompt_versions": {"analyze": "a", "merge": "m", "check": "c", "retract": "x", "render": "r"},
        "sections": [
            {"title": "One", "subsections": [{"title": "Sub", "points": [
                {"id": "P1", "text": head + " one", "sources": ["S1"],
                 "claim_refs": [{"source": "S1", "claim": 0}]}]}]},
            {"title": "Two", "subsections": [{"title": "Sub", "points": [
                {"id": "P2", "text": head + " two", "sources": ["S1", "S2"],
                 "claim_refs": [{"source": "S1", "claim": 0}, {"source": "S2", "claim": 0}]}]}]}]})
    report_text = (
        "## TL;DR\n- a\n\n## 1. One\n\n### 1.1 Sub\n"
        f"- {head} one — [V 0:05](https://www.youtube.com/watch?v=aaaaaaaaaaa&t=5s)\n\n"
        "## 2. Two\n\n### 2.1 Sub\n"
        f"- {head} two — [V 0:05](https://www.youtube.com/watch?v=aaaaaaaaaaa&t=5s), "
        "[W 0:05](https://www.youtube.com/watch?v=bbbbbbbbbbb&t=5s) (confirmed by 2 sources)\n\n"
        "## Sources\n")
    sources = [{"id": 1, "video_id": "aaaaaaaaaaa", "title": "V", "channel": "C1"},
               {"id": 2, "video_id": "bbbbbbbbbbb", "title": "W", "channel": "C2"}]
    res = GoldenResult(name="x", status="done", message="", report_path=None, doc=doc, sources=sources,
                       analyses={}, transcripts={}, report_text=report_text, cost=0.0, provider=None,
                       run_id=1, db_path=":memory:")
    manifest = {"assertions": {"min_sections": 0, "min_points": 0, "must_contain": [],
                               "min_contradictions": 0, "max_dropped_ratio": 1.0}}
    failures = check_golden(res, manifest)
    assert not [f for f in failures if "not found" in f or "confirmed_by" in f], failures


def _tldr_case(tldr, with_point):
    """A finished GoldenResult whose merge has the given TL;DR and either no points or one point."""
    from pipeline.evals import GoldenResult
    from pipeline.schemas import ClaimRef, MergedDocument, MergedSection, Point, Subsection

    sections = []
    if with_point:
        point = Point(id="P1", text="A point", sources=["S1"], claim_refs=[ClaimRef(source="S1", claim=0)])
        sections = [MergedSection(title="Sec", subsections=[Subsection(title="Sub", points=[point])])]
    doc = MergedDocument(tldr=tldr, sections=sections, source_map={"S1": 1},
                         prompt_versions={"analyze": "a", "merge": "m", "check": "c", "retract": "x", "render": "r"})
    return GoldenResult(name="x", status="done", message="", report_path=None, doc=doc,
                        sources=[{"id": 1, "video_id": VIDS[0]}], analyses={}, transcripts={},
                        report_text="", cost=0.0, provider=None, run_id=1, db_path=":memory:")


def test_tldr_assertion_accepts_empty_tldr_when_there_are_no_points():
    fails = check_golden(_tldr_case([], with_point=False), _negative_manifest())
    assert not any("tldr" in f for f in fails)


def test_tldr_assertion_rejects_a_tldr_on_a_merge_with_no_points():
    fails = check_golden(_tldr_case(["a"], with_point=False), _negative_manifest())
    assert "tldr has 1 bullets, want none for a merge with no points" in fails


def test_tldr_assertion_still_requires_three_to_five_with_points():
    fails = check_golden(_tldr_case(["a", "b"], with_point=True), _negative_manifest())
    assert "tldr has 2 bullets, want 3-5" in fails


def test_manifest_accepts_kinds_and_requires_a_url_for_them(tmp_path):
    import json, pytest
    from pipeline.config import ConfigError
    from pipeline.evals import load_manifest
    m = {"topic": "T", "sources": [{"video_id": "aaaaaaaaaaa"},
                                   {"video_id": "web-0123456789abcdef", "kind": "web", "url": "https://e.org/a"}],
         "assertions": {}}
    d = tmp_path / "g"; d.mkdir(); (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    assert [s.get("kind", "youtube") for s in load_manifest(str(d))["sources"]] == ["youtube", "web"]
    m["sources"][1].pop("url")
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ConfigError, match="url"):
        load_manifest(str(d))


def test_register_manifest_sources_handles_mixed_kinds(tmp_path):
    from pipeline import db
    from pipeline.config import Config
    from pipeline.evals import register_manifest_sources
    from pipeline.transcription.base import Segment, Transcript
    from pipeline.transcription.cache import TranscriptCache
    gdir = tmp_path / "g"
    fixtures = TranscriptCache(str(gdir / "transcripts"))
    fixtures.save(Transcript("aaaaaaaaaaa", "youtube_captions", "en", [Segment(0, 2, "a video")]))
    fixtures.save(Transcript("web-0123456789abcdef", "web_text", "de", [Segment(0, 0, "ein Artikel")], extra={"kind": "web"}))
    manifest = {"topic": "T", "sources": [
        {"video_id": "aaaaaaaaaaa", "title": "V", "transcript_method": "youtube_captions"},
        {"video_id": "web-0123456789abcdef", "kind": "web", "url": "https://e.org/a", "title": "A", "transcript_method": "web_text"}]}
    cfg = Config(home=str(tmp_path / "home")); db.init_db(cfg.db_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    ids = register_manifest_sources(cfg, nid, str(gdir), manifest, TranscriptCache(cfg.transcript_cache_dir))
    rows = {r["id"]: r for r in db.get_notebook_sources(cfg.db_path, nid)}
    web = rows[ids[1]]
    assert (web["kind"], web["url"], web["transcript_language"], web["transcribe_status"]) == ("web", "https://e.org/a", "de", "done")
    assert rows[ids[0]]["url"] == "https://www.youtube.com/watch?v=aaaaaaaaaaa"



@pytest.mark.parametrize("source, message", [
    ({"video_id": 12345678901}, "video_id"),
    ({"video_id": ["aaaaaaaaaaa"]}, "video_id"),
    ({"video_id": "aaaaaaaaaaa", "kind": "podcast"}, "unknown kind 'podcast'"),
    ({"video_id": "../../etc/passwd"}, "invalid source key"),
    ({"video_id": "blog-0123456789abcdef", "kind": "web", "url": "https://e.org/a"}, "invalid source key"),
])
def test_load_manifest_refuses_bad_sources(tmp_path, source, message):
    from pipeline.config import ConfigError
    d = tmp_path / "g"; d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"topic": "T", "sources": [source]}), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_manifest(str(d))


def test_export_golden_round_trips_a_web_source(tmp_path):
    from pipeline import db
    from pipeline.evals import register_manifest_sources
    from pipeline.transcription.base import Segment, Transcript
    cfg = Config(home=str(tmp_path / "home"))
    db.init_db(cfg.db_path)
    nid = db.create_notebook(cfg.db_path, "Mixed", "mixed")
    key, url = "web-0123456789abcdef", "https://e.org/a"
    sid = db.add_source(cfg.db_path, nid, key, url, kind="web", title="A page", channel="e.org")
    cache = TranscriptCache(cfg.transcript_cache_dir)
    cache.save(Transcript(key, "web_text", "en", [Segment(0, 0, "First paragraph."), Segment(0, 0, "Second.")],
                          extra={"kind": "web"}))
    db.update_transcribe_status(cfg.db_path, sid, "done", transcript_path=cache.path(key, "web_text"),
                                method="web_text", language="en")
    gdir = export_golden(cfg, "mixed", str(tmp_path / "golden"), "mixed")
    assert os.path.exists(os.path.join(gdir, "transcripts", key, "web_text.json"))
    m = load_manifest(gdir)
    (entry,) = m["sources"]
    assert (entry["video_id"], entry["kind"], entry["url"], entry["transcript_method"]) == (key, "web", url, "web_text")
    cfg2 = Config(home=str(tmp_path / "home2"))
    db.init_db(cfg2.db_path)
    nid2 = db.create_notebook(cfg2.db_path, "Mixed", "mixed")
    (sid2,) = register_manifest_sources(cfg2, nid2, gdir, m, TranscriptCache(cfg2.transcript_cache_dir))
    row = db.get_source(cfg2.db_path, sid2)
    assert (row["kind"], row["url"], row["title"], row["transcribe_status"]) == ("web", url, "A page", "done")
    back = TranscriptCache(cfg2.transcript_cache_dir).load(key, "web_text")
    assert [s.text for s in back.segments] == ["First paragraph.", "Second."]
