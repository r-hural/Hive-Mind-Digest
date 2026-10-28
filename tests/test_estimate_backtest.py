"""Backtest of the dry-run estimate (IDEA-003) against real runs: each run's estimate, calibrated on the
other runs only (leave-one-out), must land within +-30% of what the run cost.

The fixture is per-run token totals and per-source transcript sizes from the ten runs on the default
models (claude-haiku-5-5 analyze / claude-sonnet-5-5 merge, anthropic_api, batch prices for analyze and
the section/check merge calls) recorded 2026-10-15..16: no transcript text, neutral labels. nb1..nb3 are
the 13-14 video golden notebooks, nb4 a 42-video notebook and nb5 a 6-source mixed notebook (videos,
web pages, a PDF). Tuples are (input tokens, output tokens, cost USD)."""
import pytest

from pipeline.estimate import DEFAULTS, MergeFit, estimate_sources, fit_merge

RUNS = [
    dict(run="nb1-r28", merged_sources=13, chars=[15268, 2653, 9251, 2650, 10714, 7110, 1538, 16951, 11084, 16139, 11821, 28256, 23121],
         analyze=(81841, 49965, 0.01658), section=(279750, 47611, 0.51781), direct=(36917, 4726, 0.12109)),
    dict(run="nb1-r29", merged_sources=13, chars=[15268, 2653, 9251, 2650, 10714, 7110, 1538, 16951, 11084, 16139, 11821, 28256, 23121],
         analyze=(81841, 50910, 0.01682), section=(272202, 40769, 0.47605), direct=(34670, 5307, 0.12241)),
    dict(run="nb2-r30", merged_sources=14, chars=[12249, 3977, 6878, 8272, 16513, 9273, 20854, 9278, 4214, 4845, 7867, 26349, 2514, 9124],
         analyze=(77662, 47254, 0.0157), section=(238077, 47737, 0.47676), direct=(37562, 2630, 0.10142)),
    dict(run="nb3-r31", merged_sources=14, chars=[16024, 8865, 30082, 11902, 13617, 22629, 16216, 6630, 7907, 63669, 7069, 13274, 33267, 15060],
         analyze=(135337, 76160, 0.02581), section=(415021, 61247, 0.72126), direct=(47501, 2455, 0.11955)),
    dict(run="nb2-r32", merged_sources=14, chars=[12249, 3977, 6878, 8272, 16513, 9273, 20854, 9278, 4214, 4845, 7867, 26349, 2514, 9124],
         analyze=(83038, 54107, 0.01869), section=(207630, 38123, 0.39825), direct=(34233, 2476, 0.09323)),
    dict(run="nb3-r33", merged_sources=14, chars=[16024, 8865, 30082, 11902, 13617, 22629, 16216, 6630, 7907, 63669, 7069, 13274, 33267, 15060],
         analyze=(135337, 76976, 0.02601), section=(491196, 61186, 0.79713), direct=(47876, 5757, 0.15332)),
    dict(run="nb1-r34", merged_sources=13, chars=[15268, 2653, 9251, 2650, 10714, 7110, 1538, 16951, 11084, 16139, 11821, 28256, 23121],
         analyze=(81841, 53473, 0.01746), section=(244213, 34583, 0.41713), direct=(31319, 2128, 0.08392)),
    dict(run="nb2-r35", merged_sources=14, chars=[12249, 3977, 6878, 8272, 16513, 9273, 20854, 9278, 4214, 4845, 7867, 26349, 2514, 9124],
         analyze=(81185, 57702, 0.01935), section=(219830, 43909, 0.43937), direct=(35762, 2543, 0.09695)),
    dict(run="nb4-r37", merged_sources=42, chars=[11827, 8664, 7089, 2797, 20951, 11056, 10207, 4578, 24267, 8117, 31096, 17667, 44392,
                                                  14552, 9092, 20019, 29013, 21204, 28148, 18453, 15323, 18006, 12682, 12463, 7900,
                                                  3327, 26344, 25973, 14186, 28976, 19071, 10725, 5168, 9569, 7901, 4214, 23936,
                                                  15912, 17351, 16906, 5452, 10210],
         analyze=(319039, 174202, 0.0595), section=(390661, 74910, 0.76521), direct=(81990, 4149, 0.20547)),
    dict(run="nb5-r2", merged_sources=6, chars=[15268, 2653, 7110, 17496, 6194, 16958],
         analyze=(33634, 20033, 0.00669), section=(190564, 29583, 0.33848), direct=(22083, 5060, 0.09477)),
]
MODELS = dict(analyze_model="claude-haiku-5-5", merge_model="claude-sonnet-5-5")


def _history(r):
    """A run as db.recent_runs_with_merge reports it."""
    return dict(analyze_out=r["analyze"][1], merged_sources=r["merged_sources"],
                section_in=r["section"][0], section_out=r["section"][1],
                direct_in=r["direct"][0], direct_out=r["direct"][1])


def _actual(r):
    return r["analyze"][2] + r["section"][2] + r["direct"][2]


def _estimate(r, cal):
    return estimate_sources(r["chars"], [True] * len(r["chars"]), cal, batch=True, **MODELS)


@pytest.mark.parametrize("r", RUNS, ids=[r["run"] for r in RUNS])
def test_leave_one_out_estimate_is_within_30_percent(r):
    others = [_history(x) for x in RUNS if x is not r]
    cal = DEFAULTS.with_merge(fit_merge(others))
    e = _estimate(r, cal)
    assert abs((e.analyze + e.merge) / _actual(r) - 1) <= 0.30


@pytest.mark.parametrize("r", RUNS, ids=[r["run"] for r in RUNS])
def test_built_in_estimate_and_range_cover_each_run(r):
    e = _estimate(r, DEFAULTS)
    assert abs((e.analyze + e.merge) / _actual(r) - 1) <= 0.30
    assert e.low <= _actual(r) <= e.high


def test_built_in_coefficients_are_the_fit_on_the_fixture():
    fit = fit_merge([_history(r) for r in RUNS])
    for name in ("section_in", "section_out", "direct_in", "direct_out"):
        base, per_x = getattr(fit, name)
        b0, p0 = getattr(DEFAULTS.merge, name)
        assert base == pytest.approx(b0, abs=50) and per_x == pytest.approx(p0, abs=0.01), name
    assert isinstance(DEFAULTS.merge, MergeFit)


def test_built_in_range_is_the_leave_one_out_spread():
    """The printed range is the estimate times the spread of actual/estimate over the
    leave-one-out backtest above (rounded outward), not the extremes of a ratio."""
    ratios = []
    for r in RUNS:
        e = _estimate(r, DEFAULTS.with_merge(fit_merge([_history(x) for x in RUNS if x is not r])))
        ratios.append(_actual(r) / (e.analyze + e.merge))
    lo, hi = DEFAULTS.spread
    assert lo <= min(ratios) and max(ratios) <= hi
    assert lo >= min(ratios) - 0.03 and hi <= max(ratios) + 0.03


# --- the same backtest through the real path: runs seeded into a state DB, calibrate() picks them --


def _seed_fixture_runs(tmp_path, runs):
    """Each fixture run as db.recent_runs_with_merge reads it: one analyze event per merged source,
    the section/check and outline/verify totals, a stored merge with that many sources."""
    import json
    from pipeline import db
    from pipeline.config import load_config
    p = tmp_path / "c.yaml"
    p.write_text(f"home: {tmp_path.as_posix()}/h\n", encoding="utf-8")
    cfg = load_config(str(p))
    db.init_db(cfg.db_path)
    nid = db.create_notebook(cfg.db_path, "T", "t")
    for r in runs:
        n = r["merged_sources"]
        run = db.create_run(cfg.db_path, nid, provider="anthropic_api", budget_usd=5, stages="analyze,merge")
        db.finish_run(cfg.db_path, run, "done")
        db.create_merge(cfg.db_path, nid, run, "m", "{}", json.dumps({f"S{i + 1}": i + 1 for i in range(n)}), "{}", "r.md")
        ev = dict(run_id=run, notebook_id=nid, source_id=None, provider="anthropic_api", cache_read_tokens=0,
                  cache_write_tokens=0, cost_usd=0.0, duration_ms=1)
        for i in range(n):
            out = r["analyze"][1] // n + (r["analyze"][1] % n if i == 0 else 0)
            db.add_usage_event(cfg.db_path, stage="analyze", model=MODELS["analyze_model"],
                               input_tokens=1000, output_tokens=out, **ev)
        for stage, (i, o, _) in (("merge:section", r["section"]), ("merge:outline", r["direct"])):
            db.add_usage_event(cfg.db_path, stage=stage, model=MODELS["merge_model"],
                               input_tokens=i, output_tokens=o, **ev)
    return cfg


@pytest.mark.parametrize("keep", [None, 3], ids=["refit-on-9", "blend-on-3"])
@pytest.mark.parametrize("r", RUNS, ids=[r["run"] for r in RUNS])
def test_leave_one_out_through_calibrate(tmp_path, r, keep):
    """Calibrated on the other nine runs (least-squares refit and its leave-one-out spread), or on
    only the three newest others (blended with the built-in lines, built-in spread): within +-30%,
    and the actual cost inside the printed range."""
    from pipeline.estimate import calibrate
    others = [x for x in RUNS if x is not r]
    cfg = _seed_fixture_runs(tmp_path, others if keep is None else others[-keep:])
    cal = calibrate(cfg.db_path, batch=True, **MODELS)
    assert cal.from_history and cal.history_runs == (keep or 9)
    e = _estimate(r, cal)
    assert abs((e.analyze + e.merge) / _actual(r) - 1) <= 0.30
    assert e.low <= _actual(r) <= e.high
