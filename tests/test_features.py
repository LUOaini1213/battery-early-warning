import numpy as np
import pandas as pd

from bew.cleaning import QD_MAX_VALID, clean_summary
from bew.features import FULL_FEATURES, delta_q, early_features, five_cycle_variance
from synth import capacity_curve, qdlin_for


def _cell():
    s = capacity_curve(eol=400, knee=250, n_extra=50)
    q = qdlin_for(150)
    return s, q


def test_no_future_leakage_summary_and_curves():
    s, q = _cell()
    f0 = early_features(s, q, k=100)
    s2, q2 = s.copy(), q.copy()
    after = s2["cycle"] > 100
    for col in ("qd", "qc", "ir", "tavg", "tmin", "tmax", "chargetime"):
        s2.loc[after, col] = s2.loc[after, col] * 0.5 + 0.123
    q2[100:] = q2[100:] * 3 + 1
    f1 = early_features(s2, q2, k=100)
    for k in FULL_FEATURES:
        assert (np.isnan(f0[k]) and np.isnan(f1[k])) or f0[k] == f1[k], k


def test_past_change_does_change_features():
    """Guard against a test that passes because features ignore their input."""
    s, q = _cell()
    f0 = early_features(s, q, k=100)
    q2 = q.copy(); q2[99] = q2[99] + np.linspace(0, 0.01, 1000)
    s2 = s.copy(); s2.loc[s2["cycle"] == 2, "ir"] = 0.02
    f1 = early_features(s2, q2, k=100)
    assert f1["log_var_dq"] != f0["log_var_dq"]
    assert f1["ir_2"] == 0.02


def test_delta_q_uses_cycle_100_minus_10():
    q = np.zeros((100, 1000)); cyc = np.arange(1, 101)
    q[99] = 2.0; q[9] = 0.5            # cycle 100 and cycle 10
    q[100 - 1 - 1] = 9.0; q[10] = 9.0   # neighbours must not be used
    np.testing.assert_allclose(delta_q(q, cyc, 100, 10), 1.5)


def test_known_feature_values_and_units():
    s, q = _cell()
    f = early_features(s, q, k=100)
    dq = q[99].astype(float) - q[9].astype(float)
    assert abs(f["log_var_dq"] - np.log10(np.var(dq))) < 1e-9
    assert abs(f["ir_2"] - s.loc[s.cycle == 2, "ir"].iloc[0]) < 1e-12   # ohms
    assert 0.01 < f["ir_2"] < 0.03
    assert abs(f["qd_2"] - s.loc[s.cycle == 2, "qd"].iloc[0]) < 1e-12
    assert abs(f["chargetime_first5"] - 10.0) < 1e-12                    # cycle-1 zero excluded
    assert f["qd_slope_2_k"] < 0


def test_five_cycle_variance_uses_cycles_5_and_4():
    q = np.zeros((10, 1000)); q[4] = np.linspace(0, 1, 1000); q[3] = 0
    assert abs(five_cycle_variance(q) - np.log10(np.var(np.linspace(0, 1, 1000)))) < 1e-9


def test_cleaning_masks_impossible_values():
    d = pd.DataFrame({"cycle": [1, 2, 3, 4], "qd": [0.0, 1.07, 1.5, 1.0], "ir": [0, 0.016, 0.016, 0.016]})
    c = clean_summary(d)
    assert np.isnan(c["qd"].iloc[0]) and np.isnan(c["qd"].iloc[2])
    assert c["qd"].iloc[1] == 1.07 and np.isnan(c["ir"].iloc[0])
    assert 1.3 < QD_MAX_VALID < 1.35
