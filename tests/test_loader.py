import numpy as np
import pytest

from bew import severson_loader as L
from bew.splits import assign_paper_split
from synth import write_fake_batch


@pytest.fixture(scope="module")
def batches(tmp_path_factory):
    d = tmp_path_factory.mktemp("mat")
    write_fake_batch(d / "b1.mat", 46, seed=1)
    write_fake_batch(d / "b2.mat", 48, seed=2, cycle_life_offset=1000)
    write_fake_batch(d / "b3.mat", 46, seed=3, cycle_life_offset=2000)
    return (L.read_batch(d / "b1.mat", "2017-05-12", early_cycles=5),
            L.read_batch(d / "b2.mat", "2017-06-30", early_cycles=5),
            L.read_batch(d / "b3.mat", "2018-04-12", early_cycles=5))


def test_read_batch_fields(batches):
    b1 = batches[0]
    assert len(b1) == 46 and list(b1)[:2] == ["b1c0", "b1c1"]
    c = b1["b1c3"]
    assert c.policy == "3.6C(80%)-3.6C"
    assert c.cycle_life == 103.0
    assert list(c.summary.columns) == ["cycle", "qd", "qc", "ir", "tavg", "tmin", "tmax", "chargetime"]
    np.testing.assert_allclose(c.summary["ir"].iloc[1], 0.016 + 2e-4)   # ohms, not mOhm
    assert c.qdlin.shape == (5, 1000)
    assert np.isnan(c.qdlin[0]).all()                 # cycle 1 is an empty placeholder
    np.testing.assert_allclose(c.qdlin[2, -1], 1 * 1.02 + 3, rtol=1e-6)   # row 2 = cycle 3


def test_assemble_matches_official_loader(batches):
    cells = L.assemble(*batches)
    ids = [c.cell_id for c in cells]
    assert len(cells) == 124
    assert sum(i.startswith("b1") for i in ids) == 41
    assert sum(i.startswith("b2") for i in ids) == 43
    assert sum(i.startswith("b3") for i in ids) == 40
    for k in L.B1_REMOVE + L.B2_CARRY + L.B3_REMOVE:
        assert k not in ids
    b1c0 = cells[0]
    assert b1c0.cell_id == "b1c0" and b1c0.carried_over
    assert len(b1c0.summary) == 24
    assert list(b1c0.summary["cycle"]) == list(range(1, 25))       # numbering continues
    assert b1c0.cycle_life == 100 + 662
    assert cells[4].cycle_life == 104 + 482


def test_tables_and_split(batches):
    cells = L.assemble(*batches)
    meta, cyc, q = L.to_tables(cells)
    split = assign_paper_split(meta)
    assert split.value_counts().to_dict() == {"primary_test": 43, "train": 41, "secondary_test": 40}
    assert split.iloc[0] == "primary_test" and split.iloc[1] == "train" and split.iloc[83] == "primary_test"
    assert q.shape == (124, 5, 1000)
    assert cyc.groupby("cell_id").size().loc["b1c0"] == 24


def test_parse_policy():
    assert L.parse_policy("5.4C(40%)-3.6C") == (5.4, 40.0, 3.6)
    assert L.parse_policy("5.6C(36%)-4.3C-newstructure") == (5.6, 36.0, 4.3)
    assert all(np.isnan(L.parse_policy("weird")))


def test_eol_rule_boundary():
    qd = np.array([0.0, 1.0, 0.9, 0.88, 0.87])
    cyc = np.arange(1, 6)
    assert L.eol_cycle_from_capacity(qd, cyc, 0.88) == 5.0      # strictly below 0.88
    assert L.cycle_life_matlab_rule(qd[:4], cyc[:4], 0.88) == 5.0  # never below -> n + 1
