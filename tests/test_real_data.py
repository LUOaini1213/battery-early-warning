"""Optional checks on the real derived data (skipped in CI / when the data is absent)."""
import numpy as np
import pytest

from bew import config

pytestmark = pytest.mark.skipif(not (config.SEVERSON_DERIVED / "cells.parquet").exists(),
                                reason="real Severson data not available")


def test_real_cells_split_and_future_invariance():
    from bew.extract import load_derived
    from bew.features import FULL_FEATURES, early_features
    cells, cycles, q = load_derived()
    assert len(cells) == 124
    tr = set(cells.loc[cells.split == "train", "cell_id"])
    assert not tr & set(cells.loc[cells.split != "train", "cell_id"])
    for i in (0, 50, 100):
        cid = cells.cell_id.iloc[i]
        s = cycles[cycles.cell_id == cid]
        f0 = early_features(s, q[i])
        s2 = s.copy()
        s2.loc[s2.cycle > 100, ["qd", "ir", "tavg", "tmax", "tmin"]] = 0.5
        f1 = early_features(s2, q[i])
        assert all(np.isclose(f0[k], f1[k], equal_nan=True) for k in FULL_FEATURES)
