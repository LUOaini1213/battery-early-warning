"""Offline ground truth for knees (uses the WHOLE trajectory -- evaluation only, never a feature).

knee point : breakpoint of a continuous two-segment linear fit to the capacity curve from
             cycle `start` to end of life (Bacon-Watts with a sharp transition; see
             Fermin-Cueto et al. 2020, Energy and AI 1:100006).
knee onset : breakpoint of the same fit applied to the curve between `start` and the knee point.
"""
from __future__ import annotations

import numpy as np


def two_segment_breakpoint(x: np.ndarray, y: np.ndarray, min_seg: int = 10, step: int = 1) -> float:
    """Breakpoint b minimising SSE of y ~ a + b1*x + b2*max(0, x-b)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 2 * min_seg + 1:
        return float("nan")
    best, best_b = np.inf, np.nan
    for i in range(min_seg, n - min_seg, step):
        b = x[i]
        A = np.column_stack([np.ones(n), x, np.maximum(0.0, x - b)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(np.sum((A @ coef - y) ** 2))
        if sse < best:
            best, best_b = sse, b
    return float(best_b)


def smooth_centered(y: np.ndarray, w: int = 5) -> np.ndarray:
    """Centred running median (offline only)."""
    import pandas as pd
    return pd.Series(y).rolling(w, center=True, min_periods=1).median().to_numpy()


def knee_labels(cycles: np.ndarray, qd: np.ndarray, eol: float, start: int = 10) -> dict:
    """Return knee point / knee onset for one cell. `qd` must already be cleaned."""
    cycles = np.asarray(cycles, float); qd = np.asarray(qd, float)
    m = (cycles >= start) & (cycles <= eol) & np.isfinite(qd)
    x, y = cycles[m], smooth_centered(qd[m])
    step = max(1, len(x) // 400)
    kp = two_segment_breakpoint(x, y, step=step)
    if not np.isfinite(kp):
        return {"knee_point": np.nan, "knee_onset": np.nan}
    m2 = x <= kp
    ko = two_segment_breakpoint(x[m2], y[m2], step=max(1, int(m2.sum()) // 400))
    return {"knee_point": kp, "knee_onset": ko if np.isfinite(ko) else kp}
