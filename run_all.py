"""Reproduce every result in results/ and regenerate README.md.

    python run_all.py                 # analysis + README (needs raw data under BEW_DATA_ROOT)
    python run_all.py --download      # fetch missing raw Severson files from data.matr.io first
    python run_all.py --readme-only   # only re-render README.md from existing results/

ClickHouse verification and mutation testing are separate (they need Docker / take minutes):
    python scripts/verify_clickhouse.py
    python scripts/mutation_test.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bew import config

config.limit_threads()

from bew import early_prediction, failure_cases, nasa_eis, report, sql_layer, warning_experiment  # noqa: E402
from bew.extract import extract_severson, load_derived  # noqa: E402
from bew.provenance import check_hashes, hash_files, missing_files, write_provenance  # noqa: E402

import pandas as pd  # noqa: E402


def provenance(download: bool) -> dict:
    raw = config.SEVERSON_RAW
    # check the hand-placed NASA files first, so a missing file fails before any 8 GB download/hash
    nasa_paths = [config.NASA_RAW / f for f in config.NASA_FILES]
    gone = missing_files(nasa_paths)
    if gone:
        sys.exit("missing NASA PCoE file(s): " + ", ".join(str(p) for p in gone)
                 + "; they are not downloaded automatically: place B0005/6/7/18.mat in "
                 f"{config.NASA_RAW} (see DATA_SOURCES.md section 2)")
    if download:
        from scripts.fetch_data import fetch_severson
        fetch_severson(raw)
    files = []
    for b, meta in config.SEVERSON_BATCHES.items():
        p = raw / meta["name"]
        if not p.exists():
            sys.exit(f"missing {p}; run with --download or see DATA_SOURCES.md")
        if p.stat().st_size != meta["size"]:
            sys.exit(f"size mismatch for {p}: {p.stat().st_size} != {meta['size']}")
        files.append(p)
    sev = hash_files(files, cache=config.SEVERSON_DERIVED / "hash_cache.json")
    bad = check_hashes(sev, {m["name"]: m["sha256"] for m in config.SEVERSON_BATCHES.values()})
    if bad:
        sys.exit("Severson raw file differs from the published hash: " + "; ".join(bad))
    config.SEVERSON_DERIVED.mkdir(parents=True, exist_ok=True)
    (config.SEVERSON_DERIVED / "hash_cache.json").write_text(json.dumps({"files": sev}, indent=2))
    nasa = hash_files(nasa_paths)
    bad = check_hashes(nasa, {n: m["sha256"] for n, m in config.NASA_EXPECTED.items()})
    if bad:
        sys.exit("NASA PCoE file differs from the published hash: " + "; ".join(bad))
    payload = {
        "severson2019": {
            "source": "https://data.matr.io/1/projects/5c48dd2bc625d700019f3204",
            "download_url_pattern": "https://data.matr.io/1/api/v1/file/<file_id>/download",
            "license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), stated site-wide on data.matr.io/1",
            "files": [dict(f, file_id=config.SEVERSON_BATCHES[b]["file_id"])
                      for f, b in zip(sev, config.SEVERSON_BATCHES)],
            "stored_at": "$BEW_DATA_ROOT/severson2019/raw",
        },
        "severson_total_gb": sum(f["bytes"] for f in sev) / 1e9,
        "nasa_pcoe": {
            "source": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip "
                      "(archive member '1. BatteryAgingARC-FY08Q4.zip')",
            "files": nasa, "stored_at": "$BEW_DATA_ROOT/nasa_pcoe",
        },
    }
    for d in payload["severson2019"]["files"] + payload["nasa_pcoe"]["files"]:
        d.pop("mtime", None)
    write_provenance(config.RESULTS / "provenance.json", payload)
    return {f["name"]: f["sha256"] for f in sev}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--readme-only", action="store_true")
    a = ap.parse_args()
    R = config.RESULTS
    if not a.readme_only:
        t0 = time.time()
        hashes = provenance(a.download)
        print("[1/7] provenance ok", flush=True)
        ex = extract_severson(hashes=hashes)
        (R / "extraction_checks.json").write_text(json.dumps(ex["checks"], indent=2))
        cells, cycles, qdlin = load_derived()
        print("[2/7] extracted", len(cells), "cells", flush=True)
        early_prediction.run(cells, cycles, qdlin, R / "early_prediction")
        print("[3/7] early prediction", flush=True)
        warning_experiment.run(cells, cycles, R / "warning")
        print("[4/7] warning", flush=True)
        knees = pd.read_csv(R / "warning" / "knee_labels.csv", index_col=0)
        preds = pd.read_csv(R / "early_prediction" / "predictions.csv", index_col=0)
        failure_cases.run(cells, cycles, knees, preds, R / "failure_cases")
        print("[5/7] failure cases", flush=True)
        nasa_eis.run(config.NASA_RAW, R / "nasa_eis")
        print("[6/7] nasa eis", flush=True)
        feats = pd.read_csv(R / "early_prediction" / "features_first100.csv", index_col=0)
        sql_layer.run(cells, cycles, feats, knees, preds, R / "sql")
        print(f"[7/7] sql done in {time.time() - t0:.0f}s", flush=True)
    print("README ->", report.write_readme(R))


if __name__ == "__main__":
    main()
