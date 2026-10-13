"""Qualitative statements in the README prose, checked against results/ so they cannot go stale."""
import json

from bew.config import RESULTS


def _j(p):
    return json.loads((RESULTS / p).read_text())


def test_variance_model_within_one_cycle_of_paper():
    r = _j("early_prediction/metrics.json")
    m = r["regression"]["variance"]
    paper = r["paper_table1"]["variance"]["rmse"]
    mine = [m["rmse_train"], m["rmse_primary_test"], m["rmse_primary_test_excl_outlier"], m["rmse_secondary_test"]]
    assert all(abs(a - b) <= 1.0 for a, b in zip(mine, paper))


def test_full_model_worse_than_paper_on_primary_and_rf_worse_than_linear():
    r = _j("early_prediction/metrics.json")["regression"]
    assert r["full"]["rmse_primary_test"] > 118
    for s in ("rmse_primary_test", "rmse_secondary_test"):
        assert r["random_forest_full"][s] > max(r["variance"][s], r["discharge"][s], r["full"][s])


def test_secondary_set_has_one_early_failure():
    c = _j("early_prediction/metrics.json")["classification"]
    assert c["variance_classifier_5cyc"]["secondary_test"]["n_early_failure"] == 1


def test_warning_ranking_statements():
    d = _j("warning/metrics.json")["detectors"]
    t = {k: v["test"] for k, v in d.items()}
    assert t["cusum"]["median_lead_to_eol"] == max(v["median_lead_to_eol"] for v in t.values())
    assert t["cusum"]["false_alarm"] == max(v["false_alarm"] for v in t.values()) > 0
    assert t["cusum"]["missed"] > 0 and all(v["missed"] == 0 for k, v in t.items() if k != "cusum")
    for k in ("rolling_slope", "forecast_quadratic"):
        assert t[k]["false_alarm"] == 0 and t[k]["missed"] == 0
        assert t["threshold_baseline"]["median_lead_to_eol"] > t[k]["median_lead_to_eol"]
    assert t["cusum"]["false_alarm_rate"] > d["cusum"]["train"]["false_alarm_rate"]


def test_nasa_statements():
    r = _j("nasa_eis/metrics.json")
    assert r["range"]["spearman_re_max"] < -0.5
    rct = {c: v["spearman_rct_vs_capacity"] for c, v in r["per_cell"].items()}
    assert sorted(c for c, v in rct.items() if v < -0.5) == ["B0005", "B0006", "B0007"]
    assert -0.3 < rct["B0018"] < 0
    lo = r["loco_mean_rmse_ah"]
    assert lo["re_rct"] > lo["discharge_count"] and lo["delta_re_rct"] > lo["discharge_count"]


def test_sql_and_extraction_facts():
    ex = _j("extraction_checks.json")
    assert ex["n_cells"] == 124 and ex["cycle_life_struct_vs_matlab_rule_n_equal"] == 124
    assert _j("sql/summary.json")["sql_vs_python_variance_model_max_abs_diff_cycles"] < 0.01


def test_random_split_full_beats_variance_on_average():
    r = _j("early_prediction/metrics.json")["random_split_rmse"]
    assert r["full"]["median"] < r["variance"]["median"]
