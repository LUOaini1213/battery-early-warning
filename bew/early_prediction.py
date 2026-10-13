"""Experiment 1: predict cycle life / classify early failure from the first 100 cycles."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .features import DISCHARGE_FEATURES, FULL_FEATURES, feature_table
from .models import (class_metrics, fit_predict_regression, make_classifier, mean_pct_error, rmse,
                     selected_features)
from .splits import assert_disjoint, random_group_splits

# Severson et al. 2019, Table 1 and Table 2 (author copy, web.mit.edu/braatzgroup)
PAPER_TABLE1 = {
    "variance": {"rmse": [103, 138, 138, 196], "mpe": [14.1, 14.7, 13.2, 11.4]},
    "discharge": {"rmse": [76, 91, 86, 173], "mpe": [9.8, 13.0, 10.1, 8.6]},
    "full": {"rmse": [51, 118, 100, 214], "mpe": [5.6, 14.1, 7.5, 10.7]},
}  # order: train, primary test, primary test excl. anomalous cell, secondary test
PAPER_TABLE2 = {"variance_classifier_5cyc": [82.1, 78.6, 97.5], "full_classifier_5cyc": [97.4, 92.7, 97.5]}

REG_MODELS = {
    "variance": ("ols", ["log_var_dq"]),
    "discharge": ("elastic_net", DISCHARGE_FEATURES),
    "full": ("elastic_net", FULL_FEATURES),
    "random_forest_full": ("random_forest", FULL_FEATURES),
}
SETS = ["train", "primary_test", "primary_test_excl_outlier", "secondary_test"]


def _masks(cells: pd.DataFrame, outlier: str) -> dict[str, np.ndarray]:
    sp = cells["split"].to_numpy()
    return {"train": sp == "train", "primary_test": sp == "primary_test",
            "primary_test_excl_outlier": (sp == "primary_test") & (cells["cell_id"].to_numpy() != outlier),
            "secondary_test": sp == "secondary_test"}


def run(cells: pd.DataFrame, cycles: pd.DataFrame, qdlin: np.ndarray, out: Path, seed: int = 0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    F = feature_table(cells, cycles, qdlin, k=config.EARLY_CYCLES)
    F.to_csv(out / "features_first100.csv")
    y = cells.set_index("cell_id")["cycle_life"].astype(float)
    groups = cells["cell_id"].to_numpy()
    pt = cells[cells["split"] == "primary_test"]
    outlier = str(pt.loc[pt["cycle_life"].idxmin(), "cell_id"])
    m = _masks(cells, outlier)
    assert_disjoint(groups, np.where(m["train"])[0], np.where(m["primary_test"] | m["secondary_test"])[0])

    res: dict = {"n_cells": int(len(cells)), "split_sizes": {k: int(v.sum()) for k, v in m.items()},
                 "anomalous_primary_test_cell": {"cell_id": outlier, "cycle_life": float(y[outlier])},
                 "target": "log10(cycle life); metrics in cycles after back-transform",
                 "corr_log_var_dq_vs_log_life": float(np.corrcoef(F["log_var_dq"], np.log10(y))[0, 1]),
                 "regression": {}, "paper_table1": PAPER_TABLE1}
    preds = pd.DataFrame({"cycle_life": y, "split": cells["split"].to_numpy()})
    for name, (kind, cols) in REG_MODELS.items():
        yhat, model = fit_predict_regression(kind, F[cols], y, m["train"], seed=seed)
        preds[name] = yhat
        r = {"model": kind, "n_candidate_features": len(cols),
             "selected_features": selected_features(model, cols)}
        for s in SETS:
            r[f"rmse_{s}"] = rmse(y[m[s]], yhat[m[s]])
            r[f"mpe_{s}"] = mean_pct_error(y[m[s]], yhat[m[s]])
        res["regression"][name] = r
    preds.to_csv(out / "predictions.csv")

    # robustness: the fixed split is one draw; repeat random 50/50 cell splits of batches 1+2
    b12 = np.where(cells["batch"].isin(["2017-05-12", "2017-06-30"]).to_numpy())[0]
    rep = {name: [] for name in ("variance", "discharge", "full")}
    for tr_rows, te_rows in random_group_splits(groups[b12], 30, 0.5, seed=seed):
        tr_mask = np.zeros(len(cells), bool); tr_mask[b12[tr_rows]] = True
        te_mask = np.zeros(len(cells), bool); te_mask[b12[te_rows]] = True
        assert_disjoint(groups, np.where(tr_mask)[0], np.where(te_mask)[0])
        for name in rep:
            kind, cols = REG_MODELS[name]
            yhat, _ = fit_predict_regression(kind, F[cols], y, tr_mask, seed=seed, cv_repeats=2)
            rep[name].append(rmse(y[te_mask], yhat[te_mask]))
    res["random_split_rmse"] = {k: {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                                    "p90": float(np.percentile(v, 90)), "n_splits": len(v)}
                                for k, v in rep.items()}

    # classification: early failure = cycle life < 550 (paper Table 2 threshold)
    thr = config.EARLY_FAIL_THRESHOLD
    lab = (y < thr).to_numpy()
    res["classification"] = {"threshold_cycles": thr,
                             "early_failure_rate_all": float(lab.mean() * 100),
                             "paper_table2_5cycle": PAPER_TABLE2}
    cls_specs = {
        "variance_classifier_5cyc": ("logistic", ["log_var_dq_5_4"]),
        "variance_classifier_100cyc": ("logistic", ["log_var_dq"]),
        "full_classifier_100cyc": ("logistic_l1", FULL_FEATURES),
    }
    for name, (kind, cols) in cls_specs.items():
        clf = make_classifier(kind, seed)
        clf.fit(F.loc[m["train"], cols], lab[m["train"]])
        p = clf.predict(F[cols]).astype(bool)
        preds[f"cls_{name}"] = p
        res["classification"][name] = {s: class_metrics(lab[m[s]], p[m[s]])
                                        for s in ("train", "primary_test", "secondary_test")}
    preds.to_csv(out / "predictions.csv")
    _plot(preds, res, out)
    (out / "metrics.json").write_text(json.dumps(_no_nan(res), indent=2, allow_nan=False))
    return res


def _no_nan(x):
    """Strict JSON: undefined metrics (e.g. precision with no positive predictions) become null."""
    if isinstance(x, dict):
        return {k: _no_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_no_nan(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _plot(preds: pd.DataFrame, res: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    colors = {"train": "#7a7a7a", "primary_test": "#1f6fb2", "secondary_test": "#d9822b"}
    for ax, name in zip(axes, ["variance", "discharge", "full"]):
        for s, c in colors.items():
            d = preds[preds["split"] == s]
            ax.scatter(d["cycle_life"], d[name], s=14, c=c, label=s.replace("_", " "), alpha=0.85)
        ax.plot([100, 2500], [100, 2500], "k--", lw=0.8)
        r = res["regression"][name]
        ax.set_title(f"{name}: test RMSE {r['rmse_primary_test']:.0f} / {r['rmse_secondary_test']:.0f}")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("observed cycle life")
    axes[0].set_ylabel("predicted cycle life (first 100 cycles)")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "pred_vs_obs.png", dpi=120); plt.close(fig)
