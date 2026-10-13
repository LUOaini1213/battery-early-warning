import numpy as np
import pandas as pd

from bew.models import class_metrics, fit_predict_regression, mean_pct_error, rmse


def test_metrics_by_hand():
    assert rmse([100, 200], [110, 190]) == 10.0
    assert mean_pct_error([100, 200], [110, 150]) == 17.5      # (10% + 25%) / 2, relative to truth
    m = class_metrics(np.array([1, 1, 0, 0, 1], bool), np.array([1, 0, 0, 1, 1], bool))
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 1)
    assert m["accuracy"] == 60.0 and abs(m["recall_early_failure"] - 200 / 3) < 1e-9


def test_regression_back_transforms_log_target():
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, -3, 60)
    life = 10 ** (1.0 - 0.5 * x)
    X = pd.DataFrame({"f": x}); y = pd.Series(life)
    train = np.arange(60) % 2 == 0
    yhat, _ = fit_predict_regression("ols", X, y, train)
    np.testing.assert_allclose(yhat, life, rtol=1e-8)
    yhat2, m = fit_predict_regression("elastic_net", X, y, train)
    assert rmse(life[~train], yhat2[~train]) < 0.05 * life.mean()
