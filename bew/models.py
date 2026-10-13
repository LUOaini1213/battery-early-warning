"""Early-life cycle-life regression and early-failure classification."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def mean_pct_error(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(yhat - y) / y) * 100.0)


def make_regressor(kind: str, seed: int = 0, cv_repeats: int = 10):
    """All models predict log10(cycle life); cycles are recovered by 10**yhat."""
    if kind == "ols":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
    if kind == "elastic_net":
        # paper: elastic net, hyper-parameters by 4-fold CV with Monte Carlo repetition
        cv = RepeatedKFold(n_splits=4, n_repeats=cv_repeats, random_state=seed)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             ElasticNetCV(l1_ratio=L1_RATIOS, alphas=50, cv=cv, max_iter=200000, n_jobs=4,
                                          random_state=seed))
    if kind == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                                   max_features=0.5, random_state=seed, n_jobs=1))
    raise ValueError(kind)


def fit_predict_regression(kind: str, X: pd.DataFrame, y_cycles: pd.Series,
                           train_mask: np.ndarray, seed: int = 0,
                           cv_repeats: int = 10) -> tuple[np.ndarray, object]:
    model = make_regressor(kind, seed, cv_repeats)
    model.fit(X[train_mask], np.log10(y_cycles[train_mask].to_numpy(float)))
    return 10 ** model.predict(X), model


def selected_features(model, names: list[str]) -> list[str]:
    est = model[-1]
    if hasattr(est, "coef_"):
        coef = np.ravel(est.coef_)
        return [n for n, c in zip(names, coef) if abs(c) > 1e-12]
    return list(names)


def make_classifier(kind: str, seed: int = 0):
    if kind == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=10000))
    if kind == "logistic_l1":
        # L1-regularised logistic regression; C chosen by repeated stratified 4-fold CV
        cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=5, random_state=seed)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegressionCV(Cs=10, cv=cv, l1_ratios=(1.0,), solver="liblinear",
                                                  use_legacy_attributes=False,
                                                  max_iter=10000, scoring="balanced_accuracy",
                                                  n_jobs=1, random_state=seed))
    raise ValueError(kind)


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, bool); y_pred = np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum()); fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum()); tn = int((~y_true & ~y_pred).sum())
    n = len(y_true)
    return {"n": n, "n_early_failure": int(y_true.sum()), "accuracy": 100.0 * (tp + tn) / n if n else np.nan,
            "recall_early_failure": 100.0 * tp / (tp + fn) if tp + fn else np.nan,
            "precision_early_failure": 100.0 * tp / (tp + fp) if tp + fp else np.nan,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
