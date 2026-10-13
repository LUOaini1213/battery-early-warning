import numpy as np
import pytest

from bew import warning as W
from bew.knee import knee_labels, two_segment_breakpoint
from synth import capacity_curve


def test_first_alarm_needs_consecutive_flags():
    cyc = np.arange(101, 111)
    flags = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 0], bool)
    assert W.first_alarm(flags, cyc, 3) == 107.0   # third flag of the first 3-run, at cycle 107
    assert W.first_alarm(flags, cyc, 2) == 103.0
    assert W.first_alarm(flags, cyc, 1) == 102.0
    assert np.isnan(W.first_alarm(flags, cyc, 4))


def test_evaluate_alarm_lead_time_arithmetic():
    e = W.evaluate_alarm(380.0, knee_onset=300.0, eol=500.0, tol=50)
    assert e == {"outcome": "true_warning", "lead_to_eol": 120.0, "delay_vs_onset": 80.0}
    assert W.evaluate_alarm(250.0, 300.0, 500.0, 50)["outcome"] == "true_warning"   # boundary: onset - tol
    assert W.evaluate_alarm(249.0, 300.0, 500.0, 50)["outcome"] == "false_alarm"
    assert W.evaluate_alarm(499.0, 300.0, 500.0, 50)["lead_to_eol"] == 1.0
    assert W.evaluate_alarm(500.0, 300.0, 500.0, 50)["outcome"] == "missed"         # at EOL is too late
    assert W.evaluate_alarm(np.nan, 300.0, 500.0, 50)["outcome"] == "missed"


def test_summarize_counts():
    import pandas as pd
    ev = pd.DataFrame({"outcome": ["true_warning", "true_warning", "false_alarm", "missed"],
                       "lead_to_eol": [100.0, 200.0, 400.0, np.nan], "delay_vs_onset": [10.0, 30.0, -90, np.nan]})
    s = W.summarize(ev)
    assert (s["true_warning"], s["false_alarm"], s["missed"]) == (2, 1, 1)
    assert s["median_lead_to_eol"] == 150.0 and s["false_alarm_rate"] == 25.0


DETECTORS = {
    "slope": lambda q, c: W.flags_slope(q, c, window=40, rate_threshold=2e-4),
    "cusum": lambda q, c: W.flags_cusum(q, c, kappa=0.5, h=5.0),
    "forecast": lambda q, c: W.flags_forecast(q, c, horizon=200.0, window=100),
    "threshold": lambda q, c: W.flags_threshold(q, c, frac=0.97),
}


@pytest.mark.parametrize("name", list(DETECTORS))
def test_detectors_are_causal(name):
    s = capacity_curve(eol=700, knee=450, seed=3)
    q, c = s["qd"].to_numpy(dtype=float, copy=True), s["cycle"].to_numpy(int)
    q[0] = np.nan
    base = DETECTORS[name](q, c)
    for k in (150, 300, 500):
        q2 = q.copy()
        q2[c > k] = q2[c > k] - 0.05 * np.random.default_rng(k).random((c > k).sum())
        alt = DETECTORS[name](q2, c)
        assert np.array_equal(base[c <= k], alt[c <= k]), f"{name} leaks future data at k={k}"


@pytest.mark.parametrize("name", list(DETECTORS))
def test_detectors_fire_after_knee_not_before(name):
    s = capacity_curve(eol=700, knee=450, seed=4)
    q, c = s["qd"].to_numpy(dtype=float, copy=True), s["cycle"].to_numpy(int)
    q[0] = np.nan
    a = W.first_alarm(DETECTORS[name](q, c), c, 3)
    assert 400 <= a < 700, (name, a)


def test_causal_smooth_uses_past_only():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(W.causal_smooth(x, 3), [1.0, 1.5, 2.0, 3.0, 4.0])   # trailing medians


@pytest.mark.parametrize("fn", [lambda q, c: W.causal_smooth(q),
                                lambda q, c: W.trailing_slope(W.causal_smooth(q), 40),
                                lambda q, c: W.forecast_eol(q, c, 100),
                                lambda q, c: W.cusum_statistic(q, c, 0.5),
                                lambda q, c: W.threshold_ratio(q, c)])
def test_online_statistics_ignore_the_future(fn):
    """Continuous statistics (not just thresholded flags) at cycle k must not depend on cycles > k."""
    s = capacity_curve(eol=500, knee=300, seed=7)
    q, c = s["qd"].to_numpy(dtype=float, copy=True), s["cycle"].to_numpy(int)
    q[0] = np.nan
    base = fn(q, c)
    for k in (120, 260, 400):
        q2 = q.copy()
        q2[c > k] = np.random.default_rng(k).uniform(0.5, 1.2, (c > k).sum())
        np.testing.assert_array_equal(fn(q2, c)[c <= k], base[c <= k])


def test_trailing_slope_exact_on_line():
    y = 2.0 - 0.001 * np.arange(200)
    s = W.trailing_slope(y, 20)
    assert np.isnan(s[18]) and abs(s[19] + 0.001) < 1e-12 and abs(s[-1] + 0.001) < 1e-12


def test_forecast_eol_on_known_curve():
    s = capacity_curve(eol=700, knee=450, noise=0.0)
    q, c = s["qd"].to_numpy(dtype=float, copy=True), s["cycle"].to_numpy(int)
    q[0] = np.nan
    f = W.forecast_eol(q, c, window=100)
    assert abs(f[c == 650][0] - 700) < 15


def test_knee_labels_on_synthetic_curve():
    s = capacity_curve(eol=800, knee=500, noise=1e-4)
    q = s["qd"].to_numpy(dtype=float, copy=True); q[0] = np.nan
    k = knee_labels(s["cycle"].to_numpy(), q, eol=800)
    assert k["knee_onset"] < k["knee_point"] < 800
    assert 450 <= k["knee_onset"] <= 650


def test_two_segment_breakpoint_exact():
    x = np.arange(100.0)
    y = np.where(x < 60, 1.0, 1.0 - 0.01 * (x - 60))
    assert two_segment_breakpoint(x, y) == 60.0
