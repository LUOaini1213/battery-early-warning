"""Analyst layer: load tidy tables into embedded DuckDB and run the SQL templates."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import REPO

SQL_DIR = REPO / "sql"
CELL_COLS = ["cell_id", "batch", "policy", "c1", "q1", "c2", "split", "cycle_life", "carried_over"]
CYCLE_COLS = ["cell_id", "cycle", "qd", "qc", "ir", "tavg", "tmin", "tmax", "chargetime"]
# columns that identify a row of each template's output (used to align engines)
TEMPLATE_KEYS = {
    "01_cohort_fade_curves": ["batch", "cycle_bucket"],
    "02_knee_candidates": ["cell_id"],
    "03_ir_vs_capacity": ["cell_id"],
    "04_early_feature_ranking": ["feature"],
    "05_survival_curve": ["batch", "cycle_mark"],
    "06_data_quality": ["cell_id"],
    "07_policy_leaderboard": ["policy"],
    "08_early_failure_screen": ["cell_id"],
}


def tables(cells: pd.DataFrame, cycles: pd.DataFrame, features: pd.DataFrame,
           knees: pd.DataFrame) -> dict[str, pd.DataFrame]:
    f = features.reset_index().melt(id_vars="cell_id", var_name="feature", value_name="value")
    f = f[np.isfinite(f["value"])]
    k = knees.reset_index()[["cell_id", "knee_onset", "knee_point", "eol"]]
    return {"cells": cells[CELL_COLS].copy(), "cycles": cycles[CYCLE_COLS].copy(),
            "early_features": f.reset_index(drop=True), "knee_labels": k}


def build_duckdb(tbls: dict[str, pd.DataFrame], path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(path)
    con.execute("SET threads TO 4")
    con.execute((SQL_DIR / "duckdb" / "00_schema.sql").read_text())
    for name, df in tbls.items():
        con.register("tmp_df", df)
        con.execute(f"INSERT INTO {name} SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    return con


def run_templates(con: duckdb.DuckDBPyConnection, dialect_dir: Path = SQL_DIR / "duckdb") -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(dialect_dir.glob("0[1-9]_*.sql")):
        out[f.stem] = con.execute(f.read_text()).df()
    return out


def compare_frames(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], atol: float = 1e-5) -> dict:
    """Align two result sets on key columns and compare every shared column."""
    a = a.copy(); b = b.copy()
    for k in keys:
        a[k] = a[k].astype(str); b[k] = b[k].astype(str)
    m = a.merge(b, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)
    only = int((m["_merge"] != "both").sum())
    worst, mismatched = 0.0, []
    for col in [c for c in a.columns if c not in keys and c in b.columns]:
        x, y = m[col + "_a"], m[col + "_b"]
        xn = pd.to_numeric(x, errors="coerce"); yn = pd.to_numeric(y, errors="coerce")
        if xn.notna().any() or yn.notna().any():
            d = (xn.astype(float) - yn.astype(float)).abs()
            both_nan = xn.isna() & yn.isna()
            bad = (~both_nan) & ~(d <= atol)
            worst = max(worst, float(d[~both_nan].max()) if (~both_nan).any() else 0.0)
        else:
            bad = x.astype(str) != y.astype(str)
        if bad.any():
            mismatched.append({"column": col, "n_rows": int(bad.sum())})
    return {"rows_a": int(len(a)), "rows_b": int(len(b)), "rows_unmatched": only,
            "max_abs_diff": worst, "mismatched_columns": mismatched,
            "equal": only == 0 and not mismatched}


def run(cells, cycles, features, knees, preds: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    tbls = tables(cells, cycles, features, knees)
    con = build_duckdb(tbls)
    res = run_templates(con)
    summary = {"tables": {k: int(len(v)) for k, v in tbls.items()}, "templates": {}, "n_templates": len(res)}
    for name, df in res.items():
        df.to_csv(out / f"{name}.csv", index=False)
        summary["templates"][name] = {"rows": int(len(df))}
    # cross-check: SQL variance model == Python variance model predictions
    s = res["08_early_failure_screen"].set_index("cell_id")
    py = preds["variance"].reindex(s.index)
    summary["sql_vs_python_variance_model_max_abs_diff_cycles"] = float((s["predicted_life"] - py).abs().max())
    k = res["02_knee_candidates"]
    summary["knee_candidates_cells_flagged_before_eol"] = int((k["lead_to_eol"] > 0).sum())
    summary["knee_candidates_median_lead_to_eol"] = float(k.loc[k["lead_to_eol"] > 0, "lead_to_eol"].median())
    con.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
