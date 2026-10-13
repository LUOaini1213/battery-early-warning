"""Online early warning of accelerated capacity fade.

Every detector produces a per-cycle boolean flag series where the value at cycle k depends only
on discharge capacities at cycles <= k (trailing windows only; tested by perturbing the future).
An alarm is raised at the cycle where `min_consecutive` flags in a row are first completed.

Outcome of the first alarm, judged against offline labels (knee onset, end of life):
  true warning : knee_onset - tol <= alarm < EOL     lead time = EOL - alarm (cycles)
  false alarm  : alarm < knee_onset - tol            (trajectory still in its linear phase)
  missed       : no alarm before EOL
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONITOR_START = 101   # cycles 1..100 are the reference / early-life window


def causal_smooth(qd: np.ndarray, w: int = 5) -> np.ndarray:
    """Trailing running median over the last w valid points (NaN-tolerant, causal)."""
    return pd.Series(qd, dtype=float).rolling(w, min_periods=1).median().to_numpy()


def trailing_slope(y: np.ndarray, w: int) -> np.ndarray:
    """OLS slope (per cycle) of y over the trailing window [k-w+1, k]; NaN until enough points."""
    y = np.asarray(y, float)
    n = len(y)
    out = np.full(n, np.nan)
    t = np.arange(w, dtype=float)
    for k in range(w - 1, n):
        seg = y[k - w + 1:k + 1]
        ok = np.isfinite(seg)
        if ok.sum() >= max(3, w // 2):
            out[k] = np.polyfit(t[ok], seg[ok], 1)[0]
    return out


def flags_slope(qd: np.ndarray, cycles: np.ndarray, window: int = 40,
                rate_threshold: float = 3e-4) -> np.ndarray:
    """Detector A: trailing fade rate (Ah/cycle) exceeds a threshold."""
    s = trailing_slope(causal_smooth(qd), window)
    return (-s > rate_threshold) & (cycles >= MONITOR_START)


def cusum_statistic(qd: np.ndarray, cycles: np.ndarray, kappa: float = 0.5,
                    ref: tuple[int, int] = (10, 100)) -> np.ndarray:
    """One-sided CUSUM S_k on standardised one-cycle drops of the smoothed capacity curve;
    mean/sd come from the reference window (cycles 10..100, i.e. the past). S_k = 0 before
    monitoring starts."""
    q = causal_smooth(qd)
    d = np.diff(q, prepend=np.nan)
    rmask = (cycles >= ref[0]) & (cycles <= ref[1]) & np.isfinite(d)
    mu = float(np.mean(d[rmask])) if rmask.sum() > 5 else 0.0
    sd = float(np.std(d[rmask])) if rmask.sum() > 5 else 1e-3
    sd = max(sd, 1e-5)
    s = 0.0
    out = np.zeros(len(q))
    for k in range(len(q)):
        if cycles[k] >= MONITOR_START and np.isfinite(d[k]):
            s = max(0.0, s + (mu - d[k]) / sd - kappa)
        out[k] = s if cycles[k] >= MONITOR_START else 0.0
    return out


def flags_cusum(qd: np.ndarray, cycles: np.ndarray, kappa: float = 0.5, h: float = 8.0,
                ref: tuple[int, int] = (10, 100)) -> np.ndarray:
    """Detector B: CUSUM statistic above h."""
    return cusum_statistic(qd, cycles, kappa, ref) > h


def forecast_eol(qd: np.ndarray, cycles: np.ndarray, window: int = 100,
                 eol_ah: float = 0.88) -> np.ndarray:
    """Model-based: at each k fit a quadratic to the smoothed capacity of the trailing window and
    extrapolate the cycle at which it crosses eol_ah (NaN if it never does within 3000 cycles)."""
    q = causal_smooth(qd)
    n = len(q)
    out = np.full(n, np.nan)
    grid = np.arange(0, 3000, 1.0)
    for k in range(n):
        if cycles[k] < MONITOR_START:
            continue
        lo = max(0, k - window + 1)
        x, y = cycles[lo:k + 1].astype(float), q[lo:k + 1]
        ok = np.isfinite(y)
        if ok.sum() < 20:
            continue
        x0 = x[ok] - cycles[k]
        c = np.polyfit(x0, y[ok], 2)
        if c[0] > 0:            # convex fit = decelerating fade; fall back to the linear trend
            c = np.r_[0.0, np.polyfit(x0, y[ok], 1)]
        f = np.polyval(c, grid)
        below = np.where(f < eol_ah)[0]
        if below.size:
            out[k] = cycles[k] + grid[below[0]]
    return out


def flags_forecast(qd: np.ndarray, cycles: np.ndarray, horizon: float = 200.0,
                   window: int = 100) -> np.ndarray:
    """Detector C: forecast remaining cycles to 0.88 Ah fall below `horizon`."""
    eol_hat = forecast_eol(qd, cycles, window)
    with np.errstate(invalid="ignore"):
        return (eol_hat - cycles < horizon) & (cycles >= MONITOR_START)


def first_alarm(flags: np.ndarray, cycles: np.ndarray, min_consecutive: int = 3) -> float:
    """Cycle at which the first run of `min_consecutive` consecutive True flags is completed."""
    run = 0
    for f, c in zip(flags, cycles):
        run = run + 1 if f else 0
        if run >= min_consecutive:
            return float(c)
    return float("nan")


def evaluate_alarm(alarm: float, knee_onset: float, eol: float, tol: float = 50.0) -> dict:
    if not np.isfinite(alarm) or alarm >= eol:
        return {"outcome": "missed", "lead_to_eol": np.nan, "delay_vs_onset": np.nan}
    lead = float(eol - alarm)
    delay = float(alarm - knee_onset)
    if alarm < knee_onset - tol:
        return {"outcome": "false_alarm", "lead_to_eol": lead, "delay_vs_onset": delay}
    return {"outcome": "true_warning", "lead_to_eol": lead, "delay_vs_onset": delay}


def summarize(ev: pd.DataFrame) -> dict:
    n = len(ev)
    tw = ev[ev["outcome"] == "true_warning"]
    return {
        "n_cells": int(n),
        "true_warning": int(len(tw)), "false_alarm": int((ev["outcome"] == "false_alarm").sum()),
        "missed": int((ev["outcome"] == "missed").sum()),
        "true_warning_rate": 100.0 * len(tw) / n if n else np.nan,
        "false_alarm_rate": 100.0 * (ev["outcome"] == "false_alarm").sum() / n if n else np.nan,
        "median_lead_to_eol": float(tw["lead_to_eol"].median()) if len(tw) else np.nan,
        "p25_lead_to_eol": float(tw["lead_to_eol"].quantile(0.25)) if len(tw) else np.nan,
        "min_lead_to_eol": float(tw["lead_to_eol"].min()) if len(tw) else np.nan,
        "median_delay_vs_onset": float(tw["delay_vs_onset"].median()) if len(tw) else np.nan,
    }


def threshold_ratio(qd: np.ndarray, cycles: np.ndarray, ref: tuple[int, int] = (10, 100)) -> np.ndarray:
    """Smoothed capacity divided by the reference capacity (median of the cycle 10..100 window)."""
    q = causal_smooth(qd)
    rmask = (cycles >= ref[0]) & (cycles <= ref[1]) & np.isfinite(q)
    q0 = float(np.median(q[rmask])) if rmask.any() else np.nan
    return q / q0


def flags_threshold(qd: np.ndarray, cycles: np.ndarray, frac: float = 0.97,
                    ref: tuple[int, int] = (10, 100)) -> np.ndarray:
    """Naive baseline D: smoothed capacity below `frac` x reference capacity. This is the usual
    state-of-health threshold alarm."""
    with np.errstate(invalid="ignore"):
        return (threshold_ratio(qd, cycles, ref) < frac) & (cycles >= MONITOR_START)
