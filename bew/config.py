"""Paths and constants. Raw data lives outside the repo (BEW_DATA_ROOT, default D:/data)."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("BEW_RESULTS", REPO / "results"))
DATA_ROOT = Path(os.environ.get("BEW_DATA_ROOT", "D:/data"))
SEVERSON_RAW = DATA_ROOT / "severson2019" / "raw"
SEVERSON_DERIVED = DATA_ROOT / "severson2019" / "derived"
NASA_RAW = DATA_ROOT / "nasa_pcoe"

# Severson et al. 2019: nominal 1.1 Ah, end of life = 80 % of nominal = 0.88 Ah
NOMINAL_AH = 1.1
EOL_AH = 0.88
EARLY_CYCLES = 100          # only cycles 1..100 may be used for early prediction
EARLY_FAIL_THRESHOLD = 550  # cycles; same threshold as Severson et al. Table 2

SEVERSON_BATCHES = {
    "2017-05-12": {"file_id": "5c86c0b5fa2ede00015ddf66",
                   "name": "2017-05-12_batchdata_updated_struct_errorcorrect.mat", "size": 3025320241},
    "2017-06-30": {"file_id": "5c86bf13fa2ede00015ddd82",
                   "name": "2017-06-30_batchdata_updated_struct_errorcorrect.mat", "size": 2007331155},
    "2018-04-12": {"file_id": "5c86bd64fa2ede00015ddbb2",
                   "name": "2018-04-12_batchdata_updated_struct_errorcorrect.mat", "size": 3236690412},
}
NASA_FILES = ["B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"]

# keep the shared machine responsive
MAX_THREADS = 4


def limit_threads() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(MAX_THREADS))
