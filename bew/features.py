"""Early-life features (cycles <= k only).

Reconstruction of the candidate features listed in the main text of Severson et al. (2019);
the supplementary table was not used, so names/count differ slightly (12 discharge-only
features vs. the paper's 13, 19 in total vs. 20). Every function truncates its own input to
cycle <= k, so passing a full-life table cannot leak future data (tested in test_features.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .cleaning import clean_summary

DISCHARGE_FEATURES = [
    "log_var_dq", "log_min_dq", "log_mean_dq", "log_skew_dq", "log_kurt_dq",
    "qd_slope_2_k", "qd_int_2_k", "qd_slope_last10", "qd_int_last10",
    "qd_2", "qd_max_minus_2", "qd_k",
]
OTHER_FEATURES = [
    "chargetime_first5", "tmax_2_k", "tmin_2_k", "tavg_sum_2_k",
    "ir_2", "ir_min_2_k", "ir_k_minus_2",
]
FULL_FEATURES = DISCHARGE_FEATURES + OTHER_FEATURES


def _log_abs(x: float) -> float:
    return float(np.log10(abs(x))) if np.isfinite(x) and x != 0 else float("nan")


def delta_q(qdlin: np.ndarray, cycles: np.ndarray, hi: int, lo: int) -> np.ndarray:
    """Q_hi(V) - Q_lo(V) on the common 1000-point voltage grid; `cycles[r]` is the cycle
    number of qdlin row r."""
    rh = np.where(cycles == hi)[0]
    rl = np.where(cycles == lo)[0]
    if rh.size == 0 or rl.size == 0:
        return np.full(qdlin.shape[1], np.nan)
    return qdlin[rh[0]].astype(float) - qdlin[rl[0]].astype(float)


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return float(slope), float(intercept)


def early_features(summary: pd.DataFrame, qdlin: np.ndarray, k: int = 100,
                   lo: int = 10) -> dict[str, float]:
    """Features from cycles <= k. `summary` columns: cycle, qd, ir, tavg, tmin, tmax, chargetime.
    `qdlin` row r belongs to cycle r+1 (struct cycles array order)."""
    s = clean_summary(summary[(summary["cycle"] <= k) & (summary["cycle"] >= 1)])
    q_cycles = np.arange(1, qdlin.shape[0] + 1)
    keep = q_cycles <= k
    dq = delta_q(qdlin[keep], q_cycles[keep], k, lo)
    f: dict[str, float] = {}
    f["log_var_dq"] = _log_abs(np.nanvar(dq)) if np.isfinite(dq).any() else np.nan
    f["log_min_dq"] = _log_abs(np.nanmin(dq)) if np.isfinite(dq).any() else np.nan
    f["log_mean_dq"] = _log_abs(np.nanmean(dq)) if np.isfinite(dq).any() else np.nan
    fin = dq[np.isfinite(dq)]
    f["log_skew_dq"] = _log_abs(stats.skew(fin)) if fin.size > 3 else np.nan
    f["log_kurt_dq"] = _log_abs(stats.kurtosis(fin, fisher=False)) if fin.size > 3 else np.nan

    v = s[(s["cycle"] >= 2) & s["qd"].notna()]
    f["qd_slope_2_k"], f["qd_int_2_k"] = _fit(v["cycle"].to_numpy(float), v["qd"].to_numpy(float))
    last = v[v["cycle"] > k - 10]
    f["qd_slope_last10"], f["qd_int_last10"] = _fit(last["cycle"].to_numpy(float),
                                                    last["qd"].to_numpy(float))
    q2 = v.loc[v["cycle"] == 2, "qd"]
    qk = v.loc[v["cycle"] == k, "qd"]
    f["qd_2"] = float(q2.iloc[0]) if len(q2) else np.nan
    f["qd_k"] = float(qk.iloc[0]) if len(qk) else np.nan
    f["qd_max_minus_2"] = float(v["qd"].max() - f["qd_2"]) if len(v) else np.nan

    ct = s[(s["cycle"] >= 2) & s["chargetime"].notna()].sort_values("cycle").head(5)
    f["chargetime_first5"] = float(ct["chargetime"].mean()) if len(ct) else np.nan
    t = s[s["cycle"] >= 2]
    f["tmax_2_k"] = float(t["tmax"].max()) if len(t) else np.nan
    f["tmin_2_k"] = float(t["tmin"].min()) if len(t) else np.nan
    f["tavg_sum_2_k"] = float(t["tavg"].sum()) if len(t) else np.nan
    ir = s[(s["cycle"] >= 2) & s["ir"].notna()]
    ir2 = ir.loc[ir["cycle"] == 2, "ir"]
    irk = ir.loc[ir["cycle"] == k, "ir"]
    f["ir_2"] = float(ir2.iloc[0]) if len(ir2) else np.nan
    f["ir_min_2_k"] = float(ir["ir"].min()) if len(ir) else np.nan
    f["ir_k_minus_2"] = float(irk.iloc[0] - ir2.iloc[0]) if len(irk) and len(ir2) else np.nan
    return f


def five_cycle_variance(qdlin: np.ndarray) -> float:
    """Paper's 'variance classifier' feature: log10 var(Q5(V) - Q4(V))."""
    cyc = np.arange(1, qdlin.shape[0] + 1)
    keep = cyc <= 5
    dq = delta_q(qdlin[keep], cyc[keep], 5, 4)
    return _log_abs(np.nanvar(dq)) if np.isfinite(dq).any() else np.nan


def feature_table(cells: pd.DataFrame, cycles: pd.DataFrame, qdlin: np.ndarray,
                  k: int = 100) -> pd.DataFrame:
    rows = []
    by_cell = dict(tuple(cycles.groupby("cell_id", sort=False)))
    for i, cid in enumerate(cells["cell_id"]):
        f = early_features(by_cell[cid], qdlin[i], k=k)
        f["log_var_dq_5_4"] = five_cycle_variance(qdlin[i])
        f["cell_id"] = cid
        rows.append(f)
    return pd.DataFrame(rows).set_index("cell_id")
