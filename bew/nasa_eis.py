"""Experiment 4 (mechanism illustration): impedance growth vs capacity fade in 4 NASA PCoE lab cells.

B0005/6/7/18, 2 A CC discharge at room temperature, EIS sweep 0.1 Hz-5 kHz. The data set provides
fitted Re (electrolyte / ohmic) and Rct (charge-transfer) resistances for each impedance
operation. Four lab cells cannot establish a general law; this is a worked example of how
resistance growth and capacity fade can be related, with time-on-test as the obvious confounder.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]


def load_cell(path: Path, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (discharges, impedances) in chronological operation order."""
    ops = sio.loadmat(path, simplify_cells=True)[name]["cycle"]
    dis, imp = [], []
    n_dis = 0
    last_cap = np.nan
    for i, op in enumerate(ops):
        t = op["type"]
        if t == "discharge":
            n_dis += 1
            last_cap = float(op["data"]["Capacity"])
            dis.append({"cell": name, "op_index": i, "discharge_no": n_dis, "capacity_ah": last_cap})
        elif t == "impedance":
            d = op["data"]
            imp.append({"cell": name, "op_index": i, "discharges_before": n_dis,
                        "re_ohm": float(np.real(d["Re"])), "rct_ohm": float(np.real(d["Rct"])),
                        "capacity_prev_discharge_ah": last_cap})
    return pd.DataFrame(dis), pd.DataFrame(imp)


def partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y after removing a linear trend in rank(z)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def run(raw_dir: Path, out: Path, derived: Path | None = None) -> dict:
    """Per-operation tables go to `derived` (outside the repo: the NASA terms state no licence, so
    only aggregates are committed); metrics and the plot go to `out`."""
    out.mkdir(parents=True, exist_ok=True)
    derived = derived or raw_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    D, I = [], []
    for c in NASA_CELLS:
        d, i = load_cell(raw_dir / f"{c}.mat", c)
        D.append(d); I.append(i)
    dis = pd.concat(D, ignore_index=True); imp = pd.concat(I, ignore_index=True)
    imp.to_csv(derived / "impedance_ops.csv", index=False)
    dis.to_csv(derived / "discharges.csv", index=False)
    valid = imp.dropna(subset=["capacity_prev_discharge_ah"])
    valid = valid[(valid.re_ohm > 0) & (valid.rct_ohm > 0)]
    res = {"n_impedance_ops": int(len(imp)), "n_with_prior_discharge": int(len(valid)),
           "n_discharges": int(len(dis)), "per_cell": {},
           "caveat": "4 lab cells, single chemistry/protocol family; correlations are descriptive."}
    for c, g in valid.groupby("cell"):
        first, last = g.iloc[:5], g.iloc[-5:]
        res["per_cell"][c] = {
            "n_ops": int(len(g)),
            "re_start_mohm": float(first.re_ohm.median() * 1000), "re_end_mohm": float(last.re_ohm.median() * 1000),
            "rct_start_mohm": float(first.rct_ohm.median() * 1000), "rct_end_mohm": float(last.rct_ohm.median() * 1000),
            "cap_start_ah": float(first.capacity_prev_discharge_ah.median()),
            "cap_end_ah": float(last.capacity_prev_discharge_ah.median()),
            "spearman_re_vs_capacity": float(stats.spearmanr(g.re_ohm, g.capacity_prev_discharge_ah)[0]),
            "spearman_rct_vs_capacity": float(stats.spearmanr(g.rct_ohm, g.capacity_prev_discharge_ah)[0]),
            "partial_spearman_re_vs_capacity_given_discharge_no": partial_spearman(
                g.re_ohm, g.capacity_prev_discharge_ah, g.discharges_before),
            "partial_spearman_rct_vs_capacity_given_discharge_no": partial_spearman(
                g.rct_ohm, g.capacity_prev_discharge_ah, g.discharges_before),
        }
    # within-cell growth relative to the cell's own first 5 measurements (removes cell offsets)
    valid = valid.copy()
    for col in ("re_ohm", "rct_ohm"):
        base = valid.groupby("cell")[col].transform(lambda s: s.iloc[:5].median())
        valid["d_" + col] = valid[col] - base
    # leave-one-cell-out: capacity from (Re, Rct) vs from discharge count alone
    loco = {"re_rct": [], "delta_re_rct": [], "discharge_count": []}
    for c in valid.cell.unique():
        tr, te = valid[valid.cell != c], valid[valid.cell == c]
        for key, cols in (("re_rct", ["re_ohm", "rct_ohm"]), ("delta_re_rct", ["d_re_ohm", "d_rct_ohm"]),
                          ("discharge_count", ["discharges_before"])):
            X = np.column_stack([np.ones(len(tr))] + [tr[k] for k in cols])
            coef, *_ = np.linalg.lstsq(X, tr.capacity_prev_discharge_ah, rcond=None)
            Xt = np.column_stack([np.ones(len(te))] + [te[k] for k in cols])
            err = Xt @ coef - te.capacity_prev_discharge_ah.to_numpy()
            loco[key].append({"held_out": c, "rmse_ah": float(np.sqrt(np.mean(err ** 2)))})
    pc = res["per_cell"].values()
    res["range"] = {
        "spearman_re_min": min(c["spearman_re_vs_capacity"] for c in pc),
        "spearman_re_max": max(c["spearman_re_vs_capacity"] for c in pc),
        "spearman_rct_min": min(c["spearman_rct_vs_capacity"] for c in pc),
        "spearman_rct_max": max(c["spearman_rct_vs_capacity"] for c in pc),
        "partial_abs_max": max(max(abs(c["partial_spearman_re_vs_capacity_given_discharge_no"]),
                                   abs(c["partial_spearman_rct_vs_capacity_given_discharge_no"])) for c in pc),
    }
    res["leave_one_cell_out"] = loco
    res["loco_mean_rmse_ah"] = {k: float(np.mean([r["rmse_ah"] for r in v])) for k, v in loco.items()}
    _plot(valid, out)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    return res


def _plot(v: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for c, g in v.groupby("cell"):
        ax[0].plot(g.re_ohm * 1000, g.capacity_prev_discharge_ah, ".", ms=3, label=c)
        ax[1].plot(g.rct_ohm * 1000, g.capacity_prev_discharge_ah, ".", ms=3, label=c)
    ax[0].set_xlabel("Re (mOhm)"); ax[1].set_xlabel("Rct (mOhm)")
    ax[0].set_ylabel("capacity of preceding discharge (Ah)")
    ax[0].legend(fontsize=8)
    fig.suptitle("NASA PCoE lab cells: resistance from EIS vs capacity (4 cells)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "re_rct_vs_capacity.png", dpi=120); plt.close(fig)
