import numpy as np
import scipy.io as sio

from bew.nasa_eis import load_cell, partial_spearman


def _op(t, data):
    return {"type": t, "ambient_temperature": 24, "time": np.zeros(6), "data": data}


def test_load_cell_pairs_impedance_with_previous_discharge(tmp_path):
    ops = [_op("charge", {"Voltage_measured": np.ones(3)}),
           _op("impedance", {"Re": 0.05, "Rct": 0.07}),
           _op("discharge", {"Capacity": 1.9}),
           _op("impedance", {"Re": 0.051, "Rct": 0.071}),
           _op("discharge", {"Capacity": 1.8}),
           _op("discharge", {"Capacity": 1.7}),
           _op("impedance", {"Re": 0.06, "Rct": 0.08})]
    arr = np.empty((1, len(ops)), dtype=object)
    for i, o in enumerate(ops):
        arr[0, i] = o
    sio.savemat(tmp_path / "B9999.mat", {"B9999": {"cycle": arr}})
    dis, imp = load_cell(tmp_path / "B9999.mat", "B9999")
    assert list(dis["capacity_ah"]) == [1.9, 1.8, 1.7]
    assert list(imp["discharges_before"]) == [0, 1, 3]
    assert np.isnan(imp["capacity_prev_discharge_ah"].iloc[0])
    assert list(imp["capacity_prev_discharge_ah"].iloc[1:]) == [1.9, 1.7]
    assert abs(imp["re_ohm"].iloc[2] - 0.06) < 1e-12      # ohms as stored


def test_partial_spearman_removes_shared_trend():
    rng = np.random.default_rng(0)
    z = np.arange(300.0)
    x = z + rng.normal(0, 5, 300); y = -z + rng.normal(0, 5, 300)
    assert abs(partial_spearman(x, y, z)) < 0.2
    assert partial_spearman(x, x + rng.normal(0, 1, 300), z) > 0.5
