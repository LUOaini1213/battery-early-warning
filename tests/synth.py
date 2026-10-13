"""Synthetic data generators for tests (no real data needed; used in CI)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def capacity_curve(eol: int = 600, knee: int = 400, n_extra: int = 20, q0: float = 1.07,
                   slow: float = 2e-5, seed: int = 0, noise: float = 2e-4) -> pd.DataFrame:
    """Linear slow fade, then quadratic acceleration after `knee`, crossing 0.88 Ah at `eol`."""
    rng = np.random.default_rng(seed)
    n = eol + n_extra
    c = np.arange(1, n + 1)
    lin = q0 - slow * c
    q_knee = q0 - slow * knee
    # choose curvature so that capacity hits 0.88 exactly at eol
    a = (q_knee - slow * (eol - knee) - 0.88) / (eol - knee) ** 2
    q = np.where(c <= knee, lin, q_knee - slow * (c - knee) - a * (c - knee) ** 2)
    q = q + rng.normal(0, noise, n)
    df = pd.DataFrame({"cycle": c, "qd": q, "qc": q + 0.001, "ir": 0.016 + 2e-6 * c,
                       "tavg": 32.0 + 0.001 * c, "tmin": 30.0, "tmax": 36.0, "chargetime": 10.0})
    df.loc[0, ["qd", "qc", "ir", "tavg", "tmin", "tmax", "chargetime"]] = 0.0  # cycle-1 placeholder
    return df


def qdlin_for(n_rows: int = 100, seed: int = 0, scale: float = 1e-3) -> np.ndarray:
    """Qdlin-like array: row r = cycle r+1; row 0 empty (NaN) as in batch 1."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 1.07, 1000)
    q = np.empty((n_rows, 1000), dtype=np.float32)
    for r in range(n_rows):
        q[r] = base - scale * (r + 1) * np.sin(np.linspace(0, 3, 1000)) + rng.normal(0, 1e-6, 1000)
    q[0] = np.nan
    return q


def write_fake_batch(path: Path, n_cells: int, n_cycles: int = 12, seed: int = 0,
                     cycle_life_offset: float = 0.0) -> None:
    """Write an HDF5 file shaped like the MATLAB v7.3 batch structs (object references)."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        b = f.create_group("batch")
        counter = [0]

        def new(data, cls=None):
            counter[0] += 1
            d = refs.create_dataset(f"r{counter[0]}", data=data)
            if cls:
                d.attrs["MATLAB_class"] = np.bytes_(cls)
            return d.ref

        def ref_col(items):
            ds = np.empty((len(items), 1), dtype=h5py.ref_dtype)
            for i, r in enumerate(items):
                ds[i, 0] = r
            return ds

        summ, cyc, cl, pol = [], [], [], []
        for i in range(n_cells):
            sg = refs.create_group(f"s{i}")
            c = np.arange(1, n_cycles + 1, dtype=float)
            sg["cycle"] = c[None, :]
            sg["QDischarge"] = (1.07 - 0.001 * c + 0.0001 * i)[None, :]
            sg["QCharge"] = (1.07 - 0.001 * c)[None, :]
            sg["IR"] = (0.016 + 1e-4 * c)[None, :]
            sg["Tavg"] = (32 + 0 * c)[None, :]
            sg["Tmin"] = (30 + 0 * c)[None, :]
            sg["Tmax"] = (36 + 0 * c)[None, :]
            sg["chargetime"] = (10 + 0 * c)[None, :]
            summ.append(sg.ref)
            cg = refs.create_group(f"c{i}")
            qd_refs = []
            for j in range(n_cycles):
                if j == 0:
                    qd_refs.append(new(np.zeros((1, 2))))       # empty cycle 1
                else:
                    qd_refs.append(new((np.linspace(0, 1, 1000) * (1 + 0.01 * j) + i)[None, :]))
            cg.create_dataset("Qdlin", data=ref_col(qd_refs))
            cyc.append(cg.ref)
            cl.append(new(np.array([[100.0 + i + cycle_life_offset]])))
            p = f"{3 + i % 3}.6C(80%)-3.6C"
            pol.append(new(np.frombuffer(p.encode(), dtype=np.uint8).astype(np.uint16)[:, None], "char"))
        b.create_dataset("summary", data=ref_col(summ))
        b.create_dataset("cycles", data=ref_col(cyc))
        b.create_dataset("cycle_life", data=ref_col(cl))
        b.create_dataset("policy_readable", data=ref_col(pol))
