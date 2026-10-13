"""Experiment 2: online knee / accelerated-fade warning. Parameters tuned on TRAIN cells only."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cleaning import clean_summary
from .knee import knee_labels
from . import warning as W

GRIDS = {
    "threshold_baseline": {"frac": [0.99, 0.98, 0.97, 0.95, 0.93], "min_consecutive": [3, 10]},
    "rolling_slope": {"window": [20, 40, 80], "rate_threshold": [1e-4, 1.5e-4, 2e-4, 3e-4, 5e-4, 8e-4],
                      "min_consecutive": [3, 10]},
    "cusum": {"kappa": [0.25, 0.5, 1.0], "h": [5.0, 10.0, 20.0, 40.0], "min_consecutive": [3, 10]},
    "forecast_quadratic": {"window": [50, 100], "horizon": [100.0, 150.0, 200.0, 300.0, 400.0],
                           "min_consecutive": [3, 10]},
}
TOL = 50.0


def detector_flags(name: str, qd: np.ndarray, cyc: np.ndarray, p: dict, cache: dict) -> np.ndarray:
    if name == "threshold_baseline":
        return W.flags_threshold(qd, cyc, frac=p["frac"])
    if name == "rolling_slope":
        key = ("slope", p["window"])
        if key not in cache:
            cache[key] = W.trailing_slope(W.causal_smooth(qd), p["window"])
        return (-cache[key] > p["rate_threshold"]) & (cyc >= W.MONITOR_START)
    if name == "cusum":
        return W.flags_cusum(qd, cyc, kappa=p["kappa"], h=p["h"])
    if name == "forecast_quadratic":
        key = ("fc", p["window"])
        if key not in cache:
            cache[key] = W.forecast_eol(qd, cyc, p["window"])
        with np.errstate(invalid="ignore"):
            return (cache[key] - cyc < p["horizon"]) & (cyc >= W.MONITOR_START)
    raise ValueError(name)


def _score(s: dict) -> tuple:
    return (s["true_warning_rate"] - 2.0 * s["false_alarm_rate"],
            np.nan_to_num(s["median_lead_to_eol"], nan=-1.0))


def run(cells: pd.DataFrame, cycles: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    meta = cells.set_index("cell_id")
    series, labels = {}, []
    for cid, g in cycles.groupby("cell_id", sort=False):
        g = clean_summary(g).sort_values("cycle")
        eol = float(meta.loc[cid, "cycle_life"])
        kl = knee_labels(g["cycle"].to_numpy(), g["qd"].to_numpy(), eol)
        labels.append({"cell_id": cid, "split": meta.loc[cid, "split"], "eol": eol, **kl})
        series[cid] = (g["qd"].to_numpy(float), g["cycle"].to_numpy(int))
    lab = pd.DataFrame(labels).set_index("cell_id")
    lab.to_csv(out / "knee_labels.csv")

    caches = {cid: {} for cid in series}

    def evaluate(name: str, p: dict, ids) -> pd.DataFrame:
        rows = []
        for cid in ids:
            qd, cyc = series[cid]
            fl = detector_flags(name, qd, cyc, p, caches[cid])
            a = W.first_alarm(fl, cyc, p["min_consecutive"])
            ev = W.evaluate_alarm(a, lab.loc[cid, "knee_onset"], lab.loc[cid, "eol"], TOL)
            rows.append({"cell_id": cid, "alarm_cycle": a, **ev})
        return pd.DataFrame(rows).set_index("cell_id")

    train_ids = lab.index[lab["split"] == "train"]
    test_ids = lab.index[lab["split"] != "train"]
    res = {"definition": {"tolerance_cycles": TOL, "monitor_start_cycle": W.MONITOR_START,
                          "selection": "train cells: maximise true-warning% - 2 x false-alarm%, tie-break median lead"},
           "knee_labels": {"median_knee_onset_frac_of_life": float((lab.knee_onset / lab.eol).median()),
                           "median_knee_point_frac_of_life": float((lab.knee_point / lab.eol).median())},
           "detectors": {}}
    per_cell, grid_rows = [], []
    for name, grid in GRIDS.items():
        keys = list(grid)
        best, best_p, tried = None, None, 0
        for combo in itertools.product(*grid.values()):
            p = dict(zip(keys, combo))
            s = W.summarize(evaluate(name, p, train_ids))
            grid_rows.append({"detector": name, "params": json.dumps(p), **s})
            tried += 1
            if best is None or _score(s) > _score(best):
                best, best_p = s, p
        ev_test = evaluate(name, best_p, test_ids)
        ev_test["detector"] = name
        per_cell.append(ev_test.reset_index())
        res["detectors"][name] = {"params": best_p, "n_param_sets_tried": tried, "train": best,
                                  "test": W.summarize(ev_test),
                                  "test_by_split": {sp: W.summarize(ev_test.loc[lab.loc[test_ids, "split"] == sp])
                                                    for sp in ("primary_test", "secondary_test")}}
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(out / "train_grid.csv", index=False)
    _plot_tradeoff(grid_df, res, out)
    pc = pd.concat(per_cell, ignore_index=True).merge(lab.reset_index(), on="cell_id")
    pc.to_csv(out / "alarms_test_cells.csv", index=False)
    _plot(series, lab, pc, out)
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def _plot(series, lab, pc, out: Path) -> None:
    ids = list(lab[lab["split"] != "train"].sort_values("eol").index[[2, 15, 40, 70]])
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    colors = {"threshold_baseline": "#999999", "rolling_slope": "#1f6fb2", "cusum": "#2a9d55",
              "forecast_quadratic": "#d9822b"}
    for ax, cid in zip(axes, ids):
        qd, cyc = series[cid]
        ax.plot(cyc, qd, ".", ms=1.5, color="#444444")
        ax.axvline(lab.loc[cid, "knee_onset"], color="k", ls=":", lw=1, label="knee onset (offline)")
        ax.axvline(lab.loc[cid, "eol"], color="k", ls="-", lw=1, label="EOL 0.88 Ah")
        for _, r in pc[pc["cell_id"] == cid].iterrows():
            if np.isfinite(r["alarm_cycle"]):
                ax.axvline(r["alarm_cycle"], color=colors[r["detector"]], lw=1.5, label=r["detector"])
        ax.set_ylim(0.85, 1.1); ax.set_title(cid); ax.set_xlabel("cycle")
    axes[0].set_ylabel("discharge capacity (Ah)")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "alarm_examples.png", dpi=120); plt.close(fig)


def _plot_tradeoff(grid: pd.DataFrame, res: dict, out: Path) -> None:
    """Train-set trade-off between false alarms and lead time for every parameter set tried."""
    colors = {"threshold_baseline": "#999999", "rolling_slope": "#1f6fb2", "cusum": "#2a9d55",
              "forecast_quadratic": "#d9822b"}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, g in grid.groupby("detector"):
        ax.scatter(g["false_alarm_rate"], g["median_lead_to_eol"], s=12, alpha=0.5, c=colors[name], label=name)
        t = res["detectors"][name]["test"]
        ax.scatter([t["false_alarm_rate"]], [t["median_lead_to_eol"]], s=90, marker="*",
                   c=colors[name], edgecolors="k")
    ax.set_xlabel("false alarms (% of cells, alarm > 50 cycles before knee onset)")
    ax.set_ylabel("median lead time to EOL (cycles)")
    ax.set_title("dots: train grid; stars: selected setting on test cells", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "tradeoff.png", dpi=120); plt.close(fig)
