"""Load the Severson et al. (2019) batch structs (MATLAB v7.3 / HDF5) into tidy tables.

Cell exclusion and the batch-1/batch-2 merge follow the official starter code
(github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation,
`Load Data.ipynb`), so that cell order -- and therefore the paper's train/test split -- matches.
Only summary data and the interpolated discharge curve Qdlin of the first `early_cycles`
cycles are read; raw within-cycle time series are never loaded (memory stays small).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SUMMARY_FIELDS = {  # struct field -> column name
    "cycle": "cycle", "QDischarge": "qd", "QCharge": "qc", "IR": "ir",
    "Tavg": "tavg", "Tmin": "tmin", "Tmax": "tmax", "chargetime": "chargetime",
}
BATCH_PREFIX = {"2017-05-12": "b1", "2017-06-30": "b2", "2018-04-12": "b3"}

# official starter code (Load Data.ipynb)
B1_REMOVE = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]       # did not reach 80 %
B2_CARRY = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]         # continuation of ...
B1_CARRY = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]           # ... these batch-1 cells
CARRY_ADD_LEN = [662, 981, 1060, 208, 482]
B3_REMOVE = ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]  # noisy / unfinished


@dataclass
class Cell:
    cell_id: str
    batch: str
    policy: str
    cycle_life: float
    summary: pd.DataFrame
    qdlin: np.ndarray            # (n_early, 1000) float32, row r = cycles-array index r
    carried_over: bool = False
    notes: list[str] = field(default_factory=list)


def _char(f: h5py.File, ref) -> str:
    return f[ref][()].astype(np.uint16).tobytes()[::2].decode("latin-1")


def read_batch(path: str | Path, batch: str, early_cycles: int = 100) -> dict[str, Cell]:
    """Read one batch file. Returns cells keyed 'b{k}c{i}' in file order."""
    prefix = BATCH_PREFIX[batch]
    out: dict[str, Cell] = {}
    with h5py.File(path, "r") as f:
        b = f["batch"]
        n = b["summary"].shape[0]
        for i in range(n):
            s = f[b["summary"][i, 0]]
            summ = pd.DataFrame({col: np.asarray(s[k][0, :], dtype=float)
                                 for k, col in SUMMARY_FIELDS.items()})
            cl = float(np.asarray(f[b["cycle_life"][i, 0]][()]).ravel()[0])
            cyc = f[b["cycles"][i, 0]]
            n_cyc = cyc["Qdlin"].shape[0]
            m = min(early_cycles, n_cyc)
            q = np.full((early_cycles, 1000), np.nan, dtype=np.float32)
            for j in range(m):
                v = np.asarray(f[cyc["Qdlin"][j, 0]][()], dtype=np.float32).ravel()
                if v.size == 1000:
                    q[j] = v
            out[f"{prefix}c{i}"] = Cell(f"{prefix}c{i}", batch, _char(f, b["policy_readable"][i, 0]),
                                        cl, summ, q)
    return out


def assemble(b1: dict[str, Cell], b2: dict[str, Cell], b3: dict[str, Cell]) -> list[Cell]:
    """Apply the official exclusions/merge and return cells in the paper's order.
    Inputs are not modified (cells are deep-copied before the batch-1/batch-2 merge)."""
    b1 = copy.deepcopy(b1); b2 = dict(b2); b3 = dict(b3)
    for k in B1_REMOVE:
        del b1[k]
    for k1, k2, add in zip(B1_CARRY, B2_CARRY, CARRY_ADD_LEN):
        c1, c2 = b1[k1], b2[k2]
        tail = c2.summary.copy()
        tail["cycle"] = tail["cycle"] + len(c1.summary)
        c1.summary = pd.concat([c1.summary, tail], ignore_index=True)
        c1.cycle_life = c1.cycle_life + add
        c1.carried_over = True
        c1.notes.append(f"continued in 2017-06-30 as {k2}; tests paused between batches")
    for k in B2_CARRY:
        del b2[k]
    for k in B3_REMOVE:
        del b3[k]
    return list(b1.values()) + list(b2.values()) + list(b3.values())


_POL = re.compile(r"^\s*([\d.]+)C\((\d+)%\)-([\d.]+)C")


def parse_policy(p: str) -> tuple[float, float, float]:
    """'5.4C(40%)-3.6C' -> (5.4, 40, 3.6); NaNs when the string has another format."""
    m = _POL.match(p)
    if not m:
        return (np.nan, np.nan, np.nan)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)))


def eol_cycle_from_capacity(qd: np.ndarray, cycles: np.ndarray, eol_ah: float) -> float:
    """First cycle whose discharge capacity is below eol_ah (MATLAB LoadData.m rule).
    Cycle-1 zero placeholders (qd == 0) are ignored. NaN if never reached."""
    ok = qd > 0
    below = np.where(ok & (qd < eol_ah))[0]
    return float(cycles[below[0]]) if below.size else float("nan")


def cycle_life_matlab_rule(qd: np.ndarray, cycles: np.ndarray, eol_ah: float) -> float:
    """LoadData.m: first cycle below eol_ah, else number of recorded cycles + 1."""
    e = eol_cycle_from_capacity(qd, cycles, eol_ah)
    return e if np.isfinite(e) else float(len(qd) + 1)


def to_tables(cells: list[Cell], eol_ah: float = 0.88) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """cells table, long cycle table, and qdlin array (n_cells, n_early, 1000)."""
    meta, rows = [], []
    for order, c in enumerate(cells):
        c1, q1, c2 = parse_policy(c.policy)
        s = c.summary
        meta.append({
            "cell_id": c.cell_id, "order_idx": order, "batch": c.batch, "policy": c.policy,
            "c1": c1, "q1": q1, "c2": c2, "cycle_life": c.cycle_life,
            "eol_cycle_qd_rule": eol_cycle_from_capacity(s["qd"].to_numpy(), s["cycle"].to_numpy(), eol_ah),
            "n_cycles_recorded": int(len(s)), "carried_over": c.carried_over,
            "cycle_life_matlab_rule": cycle_life_matlab_rule(s["qd"].to_numpy(), s["cycle"].to_numpy(), eol_ah),
            "notes": "; ".join(c.notes),
        })
        d = s.copy()
        d.insert(0, "cell_id", c.cell_id)
        rows.append(d)
    cyc = pd.concat(rows, ignore_index=True)
    cyc["cycle"] = cyc["cycle"].astype(int)
    return pd.DataFrame(meta), cyc, np.stack([c.qdlin for c in cells])
