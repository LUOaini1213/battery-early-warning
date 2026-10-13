import numpy as np
import pandas as pd
import pytest

from bew import sql_layer
from bew.features import feature_table
from bew.models import fit_predict_regression
from bew.config import REPO
from synth import qdlin_for

FIX = REPO / "tests" / "fixtures"


@pytest.fixture(scope="module")
def db():
    cells = pd.read_csv(FIX / "synthetic_cells.csv")
    cycles = pd.read_csv(FIX / "synthetic_cycles.csv")
    q = np.stack([qdlin_for(100, seed=i, scale=1e-3 * (1 + 3 / c)) for i, c in
                  enumerate(cells["cycle_life"] / 100)])
    feats = feature_table(cells, cycles, q)
    knees = pd.DataFrame({"cell_id": cells["cell_id"], "knee_onset": 100.0, "knee_point": 150.0,
                          "eol": cells["cycle_life"]}).set_index("cell_id")
    tbls = sql_layer.tables(cells, cycles, feats, knees)
    con = sql_layer.build_duckdb(tbls)
    res = sql_layer.run_templates(con)
    con.close()
    return cells, cycles, feats, res


def test_all_templates_run(db):
    _, _, _, res = db
    assert set(res) == set(sql_layer.TEMPLATE_KEYS)
    for name, df in res.items():
        assert len(df) > 0, name


def test_sql_variance_model_equals_python(db):
    cells, _, feats, res = db
    y = cells.set_index("cell_id")["cycle_life"]
    train = (cells["split"] == "train").to_numpy()
    yhat, _ = fit_predict_regression("ols", feats[["log_var_dq"]], y, train)
    s = res["08_early_failure_screen"].set_index("cell_id")["predicted_life"]
    np.testing.assert_allclose(s.loc[feats.index].to_numpy(), yhat, atol=1e-3)


def test_feature_ranking_uses_train_cells_only(db):
    cells, _, _, res = db
    n_train = int((cells["split"] == "train").sum())
    assert (res["04_early_feature_ranking"]["n_train_cells"] == n_train).all()


def test_knee_candidates_fire_before_eol(db):
    _, _, _, res = db
    k = res["02_knee_candidates"]
    assert (k["first_candidate_cycle"] >= 101).all()
    assert (k["lead_to_eol"] > 0).sum() >= 4
    # the longest-lived synthetic cell knees at 500: no candidate in its linear phase
    assert k.set_index("cell_id").loc["syn7", "first_candidate_cycle"] > 450


def test_survival_fractions(db):
    cells, _, _, res = db
    s = res["05_survival_curve"].set_index("cycle_mark")
    assert s.loc[100, "frac_surviving"] == 1.0
    assert s.loc[500, "n_surviving"] == int((cells["cycle_life"] > 500).sum())


def test_data_quality_counts_placeholders(db):
    dq = res = db[3]["06_data_quality"]
    assert (dq["n_qd_nonpositive"] == 1).all()          # every synthetic cell has a cycle-1 zero


def test_compare_frames_detects_difference():
    a = pd.DataFrame({"k": ["x", "y"], "v": [1.0, 2.0]})
    assert sql_layer.compare_frames(a, a.copy(), ["k"])["equal"]
    b = a.copy(); b.loc[1, "v"] = 2.001
    r = sql_layer.compare_frames(a, b, ["k"])
    assert not r["equal"] and r["mismatched_columns"][0]["column"] == "v"
    c = a.iloc[:1]
    assert not sql_layer.compare_frames(a, c, ["k"])["equal"]


def test_clickhouse_templates_only_differ_where_dialect_requires():
    duck = REPO / "sql" / "duckdb"; ch = REPO / "sql" / "clickhouse"
    for f in sorted(duck.glob("0[1-9]_*.sql")):
        a = f.read_text().splitlines(); b = (ch / f.name).read_text().splitlines()
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        assert len(a) == len(b)
        if f.stem == "05_survival_curve":
            assert len(diff) == 1 and "numbers(" in diff[0][1]
        else:
            assert diff == [], f.name
