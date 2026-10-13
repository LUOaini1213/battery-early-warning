"""Physical-range validity mask for per-cycle summary data.

Rule (fixed before any model comparison was reported, but added AFTER the first discharge-model
run blew up on two glitches -- see README): a discharge capacity is invalid if it is <= 0 Ah
(cycle-1 placeholders) or above 1.2 x nominal (1.32 Ah), which a 1.1 Ah cell cannot deliver at
4C. Non-positive IR / temperature / charge-time values are placeholders and become NaN.
Each rule only looks at the value itself, so it cannot leak information across cycles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import NOMINAL_AH

QD_MAX_VALID = 1.2 * NOMINAL_AH


def clean_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "qd" in d:
        bad = (d["qd"] <= 0) | (d["qd"] > QD_MAX_VALID)
        d.loc[bad, "qd"] = np.nan
    for col in ("qc", "ir", "tavg", "tmin", "tmax", "chargetime"):
        if col in d:
            d.loc[d[col] <= 0, col] = np.nan
    return d


def invalid_counts(df: pd.DataFrame) -> dict:
    return {
        "qd_nonpositive": int((df["qd"] <= 0).sum()),
        "qd_above_1p2_nominal": int((df["qd"] > QD_MAX_VALID).sum()),
        "ir_nonpositive": int((df["ir"] <= 0).sum()),
    }
