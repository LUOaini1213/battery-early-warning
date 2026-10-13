"""Train/test splits. Every split is over CELLS (one row per cell or a group key), never rows."""
from __future__ import annotations

import numpy as np
import pandas as pd


def paper_split(n_b1: int, n_b2: int, n_b3: int) -> dict[str, np.ndarray]:
    """Severson et al. split as given in the official starter code (0-based positions):
    test = even positions of batches 1+2 plus position 83, train = odd positions up to 81,
    secondary test = all of batch 3 (generated after model development)."""
    n12 = n_b1 + n_b2
    test = np.hstack((np.arange(0, n12, 2), [83]))
    train = np.arange(1, n12 - 1, 2)
    secondary = np.arange(n12, n12 + n_b3)
    return {"train": train, "primary_test": test, "secondary_test": secondary}


def assign_paper_split(cells: pd.DataFrame) -> pd.Series:
    """Return a split label per cell, using `order_idx` and batch sizes."""
    n = cells["batch"].value_counts()
    idx = paper_split(int(n.get("2017-05-12", 0)), int(n.get("2017-06-30", 0)),
                      int(n.get("2018-04-12", 0)))
    lab = pd.Series("unassigned", index=cells.index, dtype=object)
    pos = cells["order_idx"].to_numpy()
    for name, ix in idx.items():
        lab[np.isin(pos, ix)] = name
    return lab


def random_group_splits(groups: np.ndarray, n_splits: int, test_frac: float,
                        seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Repeated random splits over unique group ids; returns row-index pairs."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    n_test = int(round(test_frac * len(uniq)))
    out = []
    for _ in range(n_splits):
        test_g = rng.choice(uniq, size=n_test, replace=False)
        te = np.isin(groups, test_g)
        out.append((np.where(~te)[0], np.where(te)[0]))
    return out


def assert_disjoint(groups: np.ndarray, train_rows: np.ndarray, test_rows: np.ndarray) -> None:
    inter = set(np.asarray(groups)[train_rows]) & set(np.asarray(groups)[test_rows])
    if inter:
        raise AssertionError(f"cells in both train and test: {sorted(inter)[:5]}")
