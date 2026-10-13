"""Raw .mat -> derived tables (stored next to the raw data, outside the repo)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .severson_loader import assemble, read_batch, to_tables
from .splits import assign_paper_split


def extract_severson(raw_dir: Path = config.SEVERSON_RAW, out_dir: Path = config.SEVERSON_DERIVED,
                     hashes: dict[str, str] | None = None, force: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = out_dir / "extract_stamp.json"
    if not force and stamp.exists() and hashes is not None:
        if json.loads(stamp.read_text()).get("raw_sha256") == hashes:
            return json.loads(stamp.read_text())
    batches = {}
    for bname, meta in config.SEVERSON_BATCHES.items():
        batches[bname] = read_batch(raw_dir / meta["name"], bname, config.EARLY_CYCLES)
    raw_counts = {k: len(v) for k, v in batches.items()}
    cells = assemble(batches["2017-05-12"], batches["2017-06-30"], batches["2018-04-12"])
    meta_df, cyc_df, qdlin = to_tables(cells, config.EOL_AH)
    meta_df["split"] = assign_paper_split(meta_df).to_numpy()

    # integrity checks recorded (not silently fixed)
    checks = {"raw_cell_counts": raw_counts, "n_cells": int(len(meta_df)),
              "cells_per_batch": meta_df["batch"].value_counts().to_dict(),
              "split_sizes": meta_df["split"].value_counts().to_dict()}
    numbering = cyc_df.groupby("cell_id")["cycle"].apply(
        lambda c: bool((c.to_numpy()[:100] == np.arange(1, min(100, len(c)) + 1)).all()))
    checks["cells_with_cycle_numbering_1_to_100"] = int(numbering.sum())
    diff = meta_df["cycle_life"] - meta_df["cycle_life_matlab_rule"]
    checks["cycle_life_struct_vs_matlab_rule_n_equal"] = int((diff == 0).sum())
    checks["cycle_life_struct_vs_matlab_rule_abs_diff_max"] = float(np.nanmax(np.abs(diff)))
    checks["n_cells_never_recorded_below_0p88"] = int(meta_df["eol_cycle_qd_rule"].isna().sum())
    checks["cycle_life_min_max"] = [float(meta_df["cycle_life"].min()), float(meta_df["cycle_life"].max())]

    meta_df.to_parquet(out_dir / "cells.parquet", index=False)
    cyc_df.to_parquet(out_dir / "cycles.parquet", index=False)
    np.savez_compressed(out_dir / "qdlin_first100.npz", cell_id=meta_df["cell_id"].to_numpy().astype("U16"),
                        qdlin=qdlin)
    payload = {"raw_sha256": hashes, "checks": checks}
    stamp.write_text(json.dumps(payload, indent=2))
    return payload


def load_derived(out_dir: Path = config.SEVERSON_DERIVED) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    cells = pd.read_parquet(out_dir / "cells.parquet")
    cycles = pd.read_parquet(out_dir / "cycles.parquet")
    z = np.load(out_dir / "qdlin_first100.npz", allow_pickle=False)
    assert list(z["cell_id"]) == list(cells["cell_id"]), "qdlin order mismatch"
    return cells, cycles, z["qdlin"]
