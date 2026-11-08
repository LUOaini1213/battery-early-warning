"""Smoke test of the experiment pipelines that produce the README numbers.

The real data (8 GB) cannot run in CI, so the README-number tests only prove README == committed
results/*.json. This test runs the same orchestration code (early_prediction.run,
warning_experiment.run, failure_cases.run, sql_layer.run) on a small synthetic cohort and checks
that every results key that bew.report reads is still written. A renamed or dropped key in any
run() makes it fail, even though the committed JSON would still match the README.
"""
import json

import numpy as np
import pandas as pd
import pytest

from bew import early_prediction, failure_cases, report, sql_layer, warning_experiment
from synth import capacity_curve, qdlin_for

# (split, batch, cycle life, knee) per synthetic cell; train has 6 early failures (< 550) and
# 6 long-lived cells so the 4-fold stratified CV of the L1 classifier has enough of each class.
SPECS = (
    [("train", "2017-05-12", e, int(e * 0.65)) for e in (300, 360, 420, 470, 510, 540)]
    + [("train", "2017-06-30", e, int(e * 0.7)) for e in (620, 700, 780, 860, 950, 1050)]
    + [("primary_test", "2017-05-12", e, int(e * 0.68)) for e in (330, 450, 530, 640, 820, 990)]
    + [("secondary_test", "2018-04-12", e, int(e * 0.7)) for e in (380, 560, 680, 760, 900, 1100)]
)
PIPELINE_FILES = {report.EP, report.WN, report.FC, report.SQ}


def _cohort():
    cells, cycles, q = [], [], []
    for i, (split, batch, eol, knee) in enumerate(SPECS):
        cid = f"syn{i}"
        s = capacity_curve(eol=eol, knee=knee, n_extra=5, seed=200 + i)
        s.insert(0, "cell_id", cid)
        cycles.append(s)
        cells.append({"cell_id": cid, "order_idx": i, "batch": batch, "policy": f"{4 + i % 3}C(80%)-4C",
                      "c1": 4.0 + i % 3, "q1": 80.0, "c2": 4.0, "split": split,
                      "cycle_life": float(eol), "carried_over": False})
        q.append(qdlin_for(100, seed=i, scale=1e-3 * (1 + 300 / eol)))
    return pd.DataFrame(cells), pd.concat(cycles, ignore_index=True), np.stack(q)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    cells, cycles, qdlin = _cohort()
    mp = pytest.MonkeyPatch()
    # one or two parameter sets per detector keeps the grid search fast; the loop itself still runs
    mp.setattr(warning_experiment, "GRIDS", {
        "threshold_baseline": {"frac": [0.97, 0.95], "min_consecutive": [3]},
        "rolling_slope": {"window": [40], "rate_threshold": [3e-4, 5e-4], "min_consecutive": [3]},
        "cusum": {"kappa": [0.5], "h": [10.0], "min_consecutive": [3]},
        "forecast_quadratic": {"window": [50], "horizon": [150.0], "min_consecutive": [3]},
    })
    # 3 instead of 30 random re-splits: same code path, a tenth of the run time
    orig_splits = early_prediction.random_group_splits
    mp.setattr(early_prediction, "random_group_splits",
               lambda groups, n_splits, test_frac, seed: orig_splits(groups, 3, test_frac, seed))
    try:
        early_prediction.run(cells, cycles, qdlin, out / "early_prediction")
        warning_experiment.run(cells, cycles, out / "warning")
    finally:
        mp.undo()
    knees = pd.read_csv(out / "warning" / "knee_labels.csv", index_col=0)
    preds = pd.read_csv(out / "early_prediction" / "predictions.csv", index_col=0)
    failure_cases.run(cells, cycles, knees, preds, out / "failure_cases")
    feats = pd.read_csv(out / "early_prediction" / "features_first100.csv", index_col=0)
    sql_layer.run(cells, cycles, feats, knees, preds, out / "sql")
    return out, cells


def test_every_readme_key_is_written_by_the_pipelines(results):
    out, _ = results
    missing = [name for name, src in report.SOURCES.items()
               if src[0] in PIPELINE_FILES and report.lookup(out, src) is None]
    assert not missing, f"README keys no longer produced by the pipelines: {missing}"


def test_headline_renders_without_gaps(results):
    out, _ = results
    hl = report.build_headline(out)
    gaps = [n for n, src in report.SOURCES.items() if src[0] in PIPELINE_FILES and hl[n] == "n/a"]
    assert not gaps, gaps


def test_pipeline_outputs_are_consistent(results):
    out, cells = results
    ep = json.loads((out / report.EP).read_text())
    assert ep["n_cells"] == len(cells)
    assert ep["split_sizes"]["train"] == (cells["split"] == "train").sum()
    pt = cells[cells["split"] == "primary_test"]
    assert ep["anomalous_primary_test_cell"]["cell_id"] == pt.loc[pt["cycle_life"].idxmin(), "cell_id"]
    wn = json.loads((out / report.WN).read_text())
    n_test = int((cells["split"] != "train").sum())
    for name, d in wn["detectors"].items():
        grid = warning_experiment.GRIDS[name]
        # params were chosen from the (monkeypatched) grid on training cells and scored on the rest
        assert d["test"]["n_cells"] == n_test and d["train"]["n_cells"] == len(cells) - n_test
        assert set(d["params"]) <= set(grid) | {"min_consecutive"}
    sq = json.loads((out / report.SQ).read_text())
    assert sq["n_templates"] == len(sql_layer.TEMPLATE_KEYS)
    assert sq["knee_candidates_scored"]["all"]["n_cells"] == len(cells)
    assert sq["sql_vs_python_variance_model_max_abs_diff_cycles"] < 0.01
    for f in ("pred_vs_obs.png",):
        assert (out / "early_prediction" / f).stat().st_size > 0
    for f in ("tradeoff.png", "alarm_examples.png", "train_grid.csv", "alarms_test_cells.csv"):
        assert (out / "warning" / f).stat().st_size > 0
