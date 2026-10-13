"""Generate README.md from README.template.md and the JSON files in results/.

Every number in the README results sections is a {{placeholder}} resolved here from a named
results file + key path. tests/test_readme_numbers.py parses README.md back and re-checks each
value against the same files, so a stale or hand-edited README number fails the tests.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import REPO, RESULTS

EP = "early_prediction/metrics.json"
WN = "warning/metrics.json"
FC = "failure_cases/summary.json"
NA = "nasa_eis/metrics.json"
SQ = "sql/summary.json"
CH = "sql/clickhouse_check.json"
MU = "mutation_report.json"
PV = "provenance.json"

# name -> (file, key path, format). Format: 'd0' integer-rounded, 'f1' one decimal, etc.
SOURCES: dict[str, tuple[str, list, str]] = {
    "n_cells": (EP, ["n_cells"], "d0"),
    "n_train": (EP, ["split_sizes", "train"], "d0"),
    "n_primary": (EP, ["split_sizes", "primary_test"], "d0"),
    "n_secondary": (EP, ["split_sizes", "secondary_test"], "d0"),
    "outlier_id": (EP, ["anomalous_primary_test_cell", "cell_id"], "s"),
    "outlier_life": (EP, ["anomalous_primary_test_cell", "cycle_life"], "d0"),
    "corr_logvar": (EP, ["corr_log_var_dq_vs_log_life"], "f2"),
    "raw_gb": (PV, ["severson_total_gb"], "f2"),
    "rs_var": (EP, ["random_split_rmse", "variance", "median"], "d0"),
    "rs_var_p10": (EP, ["random_split_rmse", "variance", "p10"], "d0"),
    "rs_var_p90": (EP, ["random_split_rmse", "variance", "p90"], "d0"),
    "rs_full": (EP, ["random_split_rmse", "full", "median"], "d0"),
    "rs_full_p10": (EP, ["random_split_rmse", "full", "p10"], "d0"),
    "rs_full_p90": (EP, ["random_split_rmse", "full", "p90"], "d0"),
    "rs_n": (EP, ["random_split_rmse", "full", "n_splits"], "d0"),
    "ef_rate": (EP, ["classification", "early_failure_rate_all"], "f1"),
    "kn_onset_frac": (WN, ["knee_labels", "median_knee_onset_frac_of_life"], "pct0"),
    "kn_point_frac": (WN, ["knee_labels", "median_knee_point_frac_of_life"], "pct0"),
    "fc_cases": (FC, ["n_cases"], "d0"),
    "fc_screened": (FC, ["n_cells_screened"], "d0"),
    "na_ops": (NA, ["n_impedance_ops"], "d0"),
    "na_re_min": (NA, ["range", "spearman_re_min"], "f2"),
    "na_re_max": (NA, ["range", "spearman_re_max"], "f2"),
    "na_rct_min": (NA, ["range", "spearman_rct_min"], "f2"),
    "na_rct_max": (NA, ["range", "spearman_rct_max"], "f2"),
    "na_partial": (NA, ["range", "partial_abs_max"], "f2"),
    "na_loco_rr": (NA, ["loco_mean_rmse_ah", "re_rct"], "f3"),
    "na_loco_dc": (NA, ["loco_mean_rmse_ah", "discharge_count"], "f3"),
    "na_loco_drr": (NA, ["loco_mean_rmse_ah", "delta_re_rct"], "f3"),
    "sql_n_templates": (SQ, ["n_templates"], "d0"),
    "sql_py_diff": (SQ, ["sql_vs_python_variance_model_max_abs_diff_cycles"], "f3"),
    "sql_knee_flagged": (SQ, ["knee_candidates_cells_flagged_before_eol"], "d0"),
    "sql_knee_lead": (SQ, ["knee_candidates_median_lead_to_eol"], "d0"),
    "ch_equal": (CH, ["n_templates_equal"], "d0"),
    "ch_total": (CH, ["n_templates"], "d0"),
    "ch_version": (CH, ["clickhouse_version"], "s"),
    "ch_maxdiff": (CH, ["max_abs_diff_all"], "e1"),
    "mu_caught": (MU, ["caught"], "d0"),
    "mu_total": (MU, ["total"], "d0"),
    "mu_tests": (MU, ["baseline_tests_passed"], "d0"),
    "mu_prev_caught": (MU, ["history", 0, "caught"], "d0"),
    "mu_prev_total": (MU, ["history", 0, "total"], "d0"),
    "mu_prev_missed": (MU, ["history", 0, "missed", 0], "s"),
    "mu_prev2_caught": (MU, ["history", 1, "caught"], "d0"),
    "mu_prev2_total": (MU, ["history", 1, "total"], "d0"),
    "mu_prev2_missed_a": (MU, ["history", 1, "missed", 0], "s"),
    "mu_prev2_missed_b": (MU, ["history", 1, "missed", 1], "s"),
}
for model, short in (("variance", "var"), ("discharge", "dis"), ("full", "full"), ("random_forest_full", "rf")):
    for s, ss in (("train", "tr"), ("primary_test", "pt"), ("primary_test_excl_outlier", "px"),
                  ("secondary_test", "st")):
        SOURCES[f"{short}_rmse_{ss}"] = (EP, ["regression", model, f"rmse_{s}"], "d0")
        SOURCES[f"{short}_mpe_{ss}"] = (EP, ["regression", model, f"mpe_{s}"], "f1")
    SOURCES[f"{short}_nsel"] = (EP, ["regression", model, "selected_features"], "len")
    SOURCES[f"{short}_ncand"] = (EP, ["regression", model, "n_candidate_features"], "d0")
for i, s in enumerate(("tr", "pt", "px", "st")):
    for model, short in (("variance", "var"), ("discharge", "dis"), ("full", "full")):
        SOURCES[f"paper_{short}_rmse_{s}"] = (EP, ["paper_table1", model, "rmse", i], "d0")
        SOURCES[f"paper_{short}_mpe_{s}"] = (EP, ["paper_table1", model, "mpe", i], "f1")
for i, s in enumerate(("tr", "pt", "st")):
    SOURCES[f"paper_cls_var_{s}"] = (EP, ["classification", "paper_table2_5cycle", "variance_classifier_5cyc", i], "f1")
    SOURCES[f"paper_cls_full_{s}"] = (EP, ["classification", "paper_table2_5cycle", "full_classifier_5cyc", i], "f1")
for clf, short in (("variance_classifier_5cyc", "v5"), ("variance_classifier_100cyc", "v100"),
                   ("full_classifier_100cyc", "f100")):
    for s, ss in (("train", "tr"), ("primary_test", "pt"), ("secondary_test", "st")):
        SOURCES[f"cls_{short}_acc_{ss}"] = (EP, ["classification", clf, s, "accuracy"], "f1")
        SOURCES[f"cls_{short}_rec_{ss}"] = (EP, ["classification", clf, s, "recall_early_failure"], "f1")
    SOURCES[f"cls_{short}_nef_st"] = (EP, ["classification", clf, "secondary_test", "n_early_failure"], "d0")
    SOURCES[f"cls_{short}_nef_pt"] = (EP, ["classification", clf, "primary_test", "n_early_failure"], "d0")
for det, short in (("threshold_baseline", "thr"), ("rolling_slope", "slope"), ("cusum", "cusum"),
                   ("forecast_quadratic", "fcst")):
    for k, kk in (("true_warning_rate", "tw"), ("false_alarm_rate", "fa"), ("missed", "miss"),
                  ("median_lead_to_eol", "lead"), ("p25_lead_to_eol", "lead25"),
                  ("median_delay_vs_onset", "delay"), ("n_cells", "n")):
        fmt = "f1" if k.endswith("rate") else "d0"
        SOURCES[f"w_{short}_{kk}"] = (WN, ["detectors", det, "test", k], fmt)
        SOURCES[f"w_{short}_{kk}_train"] = (WN, ["detectors", det, "train", k], fmt)
for sig in ("early_knee", "life_overpredicted", "life_underpredicted", "ir_jump", "temperature_step",
            "noisy_capacity", "invalid_values"):
    SOURCES[f"fc_{sig}"] = (FC, ["signature_counts", sig], "d0")


def lookup(results: Path, src: tuple[str, list, str]):
    f, path, _ = src
    p = results / f
    if not p.exists():
        return None
    v = json.loads(p.read_text())
    for k in path:
        if isinstance(v, list):
            if not isinstance(k, int) or not -len(v) <= k < len(v):
                return None
            v = v[k]
        elif isinstance(v, dict) and k in v:
            v = v[k]
        else:
            return None
    return v


def fmt(v, kind: str) -> str:
    if v is None:
        return "n/a"
    if kind == "s":
        return str(v)
    if kind == "len":
        return str(len(v))
    if kind == "d0":
        return f"{float(v):.0f}"
    if kind == "pct0":
        return f"{100 * float(v):.0f}"
    if kind.startswith("f"):
        return f"{float(v):.{int(kind[1:])}f}"
    if kind.startswith("e"):
        return f"{float(v):.{int(kind[1:])}e}"
    raise ValueError(kind)


def build_headline(results: Path = RESULTS) -> dict[str, str]:
    return {name: fmt(lookup(results, src), src[2]) for name, src in SOURCES.items()}


PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def render(template: str, headline: dict[str, str]) -> str:
    def sub(m):
        if m.group(1) not in headline:
            raise KeyError(f"unknown placeholder {m.group(1)}")
        return headline[m.group(1)]
    return PLACEHOLDER.sub(sub, template)


def write_readme(results: Path = RESULTS) -> Path:
    tpl = (REPO / "README.template.md").read_text(encoding="utf-8")
    out = REPO / "README.md"
    out.write_text(render(tpl, build_headline(results)), encoding="utf-8")
    hl = results / "headline.json"
    hl.write_text(json.dumps(build_headline(results), indent=2, sort_keys=True))
    return out
