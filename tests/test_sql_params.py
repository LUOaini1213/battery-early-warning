"""The SQL knee screen copies the rolling-slope detector's tuned window and threshold, and the
README states its scored outcome. Both are pinned here to the committed results."""
import json
import re

import pandas as pd
import pytest

from bew import sql_layer
from bew.config import REPO, RESULTS


def _tuned():
    d = json.loads((RESULTS / "warning" / "metrics.json").read_text())["detectors"]
    return d["rolling_slope"]["params"]


@pytest.mark.parametrize("dialect", ["duckdb", "clickhouse"])
def test_sql_knee_screen_uses_tuned_rolling_slope_params(dialect):
    sql = (REPO / "sql" / dialect / "02_knee_candidates.sql").read_text(encoding="utf-8")
    p = _tuned()
    w, thr = int(p["window"]), float(p["rate_threshold"])
    preceding = re.findall(r"ROWS BETWEEN (\d+) PRECEDING AND CURRENT ROW", sql)
    assert preceding and all(int(x) == w - 1 for x in preceding), preceding
    n_win = re.findall(r"n_win = (\d+)", sql)
    assert n_win and all(int(x) == w for x in n_win), n_win
    rate = re.findall(r"-\(w\.cov_xy / w\.var_x\) > ([0-9.eE-]+)", sql)
    assert rate and all(float(x) == thr for x in rate), rate
    # the header comment states the same values in words
    assert f"trailing {w}-cycle" in sql
    assert f"{thr * 1000:g} mAh/cycle" in sql


def test_sql_knee_screen_score_matches_committed_csvs():
    cand = pd.read_csv(RESULTS / "sql" / "02_knee_candidates.csv")
    knees = pd.read_csv(RESULTS / "warning" / "knee_labels.csv", index_col=0)
    got = sql_layer.score_knee_screen(cand, knees)
    want = json.loads((RESULTS / "sql" / "summary.json").read_text())["knee_candidates_scored"]
    assert got == want


def test_score_knee_screen_counts_false_alarms_and_misses():
    knees = pd.DataFrame({"split": ["train", "primary_test", "secondary_test"],
                          "knee_onset": [300.0, 300.0, 300.0], "eol": [500.0, 500.0, 500.0]},
                         index=pd.Index(["a", "b", "c"], name="cell_id"))
    cand = pd.DataFrame({"cell_id": ["a", "b"], "first_candidate_cycle": [100, 320]})  # c: no candidate
    s = sql_layer.score_knee_screen(cand, knees)["all"]
    assert (s["false_alarm"], s["true_warning"], s["missed"]) == (1, 1, 1)
    assert s["median_lead_to_eol"] == 180.0
