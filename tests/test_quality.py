"""Merge-quality gate numbers for `hmd status <slug>` (pipeline/quality.py)."""
import json

import pytest

from pipeline import db
from pipeline.config import Config
from pipeline.quality import notebook_quality, quality_report


@pytest.fixture
def cfg(tmp_path):
    c = Config(home=str(tmp_path), llm={"max_parallel": 2}, run={"budget_usd": 0})
    db.init_db(c.db_path)
    return c


def _source(cfg, nid, vid, analysis):
    sid = db.add_source(cfg.db_path, nid, vid, f"https://youtu.be/{vid}")
    db.update_analyze_status(cfg.db_path, sid, "done", analysis_json=json.dumps(analysis))
    return sid


def _merge(cfg, nid, supports, contradictions=0):
    points = [{"text": f"p{i}", "sources": ["S1"], **({"support": s} if s else {})}
              for i, s in enumerate(supports)]
    doc = {"tldr": ["a"], "source_map": {"S1": 1},
           "sections": [{"title": "T", "subsections": [{"title": "s", "points": points}],
                         "contradictions": [{"topic": "x"}] * contradictions}]}
    return db.create_merge(cfg.db_path, nid, None, "m", json.dumps(doc), "{}", "{}", "r.md")


def test_quality_counts_claims_drops_points_and_contradictions(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    _source(cfg, nid, "aaaaaaaaaaa", {"key_claims": [{"claim": "c"}] * 3,
                                      "dropped_quotes": [{"reason": "too short"}]})
    _source(cfg, nid, "bbbbbbbbbbb", {"key_claims": [{"claim": "c"}] * 6,
                                      "dropped_quotes": [{"reason": "too short"},
                                                         {"reason": "span over 16 segments"}]})
    mid = _merge(cfg, nid, ["partial", None, None, "partial"], contradictions=2)

    q = notebook_quality(cfg.db_path, nid)

    assert q == {"claims_kept": 9, "claims_dropped": 3, "claims_trimmed": 0,
                 "drop_reasons": {"too short": 2, "span over 16 segments": 1},
                 "merge_id": mid, "points": 4, "partial": 2, "contradictions": 2,
                 "retract_attempted": 0, "retract_accepted": 0, "retract_rejected": 0, "unreadable": 0}
    report = quality_report(cfg.db_path, nid)
    assert "claims: 9 kept, 3 dropped (25.0%) -- too short 2, span over 16 segments 1" in report
    assert f"merge {mid}: 4 points, 2 partly supported (50.0%), 2 contradictions" in report
    assert all(l.startswith("  ") for l in report.splitlines()[1:])


def test_quality_without_merge(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    _source(cfg, nid, "aaaaaaaaaaa", {"key_claims": [{"claim": "c"}], "dropped_quotes": []})
    report = quality_report(cfg.db_path, nid)
    assert "claims: 1 kept, 0 dropped (0.0%)" in report
    assert "  merge " not in report


def test_quality_tolerates_old_rows(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    _source(cfg, nid, "aaaaaaaaaaa", {"key_claims": [{"claim": "c", "quote": "q"}]})   # no dropped_quotes
    _merge(cfg, nid, [None, None])                                                      # no support keys
    q = notebook_quality(cfg.db_path, nid)
    assert (q["claims_kept"], q["claims_dropped"], q["points"], q["partial"]) == (1, 0, 2, 0)


def test_quality_zero_claims(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    _source(cfg, nid, "aaaaaaaaaaa", {"key_claims": []})
    _merge(cfg, nid, [])
    report = quality_report(cfg.db_path, nid)
    assert "claims: 0 kept, 0 dropped (0.0%)" in report
    assert "0 points, 0 partly supported (0.0%)" in report


def test_quality_report_empty_for_a_fresh_notebook(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    assert quality_report(cfg.db_path, nid) == ""


def test_quality_counts_unreadable_rows_instead_of_raising(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    _source(cfg, nid, "aaaaaaaaaaa", {"key_claims": [{"claim": "c"}] * 2, "dropped_quotes": ["not a dict"]})
    sid = db.add_source(cfg.db_path, nid, "bbbbbbbbbbb", "https://youtu.be/bbbbbbbbbbb")
    db.update_analyze_status(cfg.db_path, sid, "done", analysis_json="{truncated")
    db.create_merge(cfg.db_path, nid, None, "m", "{truncated", "{}", "{}", "r.md")

    q = notebook_quality(cfg.db_path, nid)

    # a row is read whole or not at all: the malformed dropped_quotes entry voids its two claims
    assert (q["claims_kept"], q["claims_dropped"], q["unreadable"], q["merge_id"]) == (0, 0, 3, None)
    report = quality_report(cfg.db_path, nid)
    assert "  3 unreadable rows (analysis_json or merged_json)" in report
    assert "  merge " not in report


def test_quality_counts_trimmed_claims(cfg):
    nid = db.create_notebook(cfg.db_path, "T", "t")
    _source(cfg, nid, "aaaaaaaaaaa", {"key_claims": [{"claim": "c"}] * 4, "dropped_quotes": [],
                                      "trimmed_quotes": [{"claim": "c"}, {"claim": "d"}]})
    _source(cfg, nid, "bbbbbbbbbbb", {"key_claims": [{"claim": "c"}]})          # stored before round 6
    assert notebook_quality(cfg.db_path, nid)["claims_trimmed"] == 2
    assert "claims: 5 kept (2 trimmed to fit), 0 dropped (0.0%)" in quality_report(cfg.db_path, nid)


def test_quality_counts_retractions(cfg):
    # IDEA-297: the merge stores its retraction counts; older merges have none (zeros)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    _source(cfg, nid, "aaaaaaaaaaa", {"key_claims": [{"claim": "c"}], "dropped_quotes": []})
    mid = _merge(cfg, nid, ["partial", "partial", None])
    q = notebook_quality(cfg.db_path, nid)
    assert (q["retract_attempted"], q["retract_accepted"], q["retract_rejected"]) == (0, 0, 0)
    assert "retraction" not in quality_report(cfg.db_path, nid)
    doc = json.loads(db.get_latest_merge(cfg.db_path, nid)["merged_json"])
    doc["retraction"] = {"attempted": 2, "accepted": 1, "rejected": 1}
    mid = db.create_merge(cfg.db_path, nid, None, "m", json.dumps(doc), "{}", "{}", "r.md")
    q = notebook_quality(cfg.db_path, nid)
    assert (q["merge_id"], q["retract_attempted"], q["retract_accepted"], q["retract_rejected"]) == (mid, 2, 1, 1)
    assert "  retraction of partly supported points: 2 attempted, 1 accepted, 1 rejected" in quality_report(cfg.db_path, nid)
