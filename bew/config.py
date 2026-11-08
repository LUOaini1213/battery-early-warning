"""Paths and constants. Raw data lives under BEW_DATA_ROOT (default: the git-ignored data/ folder
of this repo), never in tracked files."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("BEW_RESULTS", REPO / "results"))
DATA_ROOT = Path(os.environ.get("BEW_DATA_ROOT") or REPO / "data")
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
                   "name": "2017-05-12_batchdata_updated_struct_errorcorrect.mat", "size": 3025320241,
                   "sha256": "9d928ab978f0e3c70b31cb833a749fedd35094d01af76475d69b40aa3497f5ba"},
    "2017-06-30": {"file_id": "5c86bf13fa2ede00015ddd82",
                   "name": "2017-06-30_batchdata_updated_struct_errorcorrect.mat", "size": 2007331155,
                   "sha256": "63ab200d09ecb237fee5ef3a5c5db76e3212e3206a0bd92f769e1427fed338b8"},
    "2018-04-12": {"file_id": "5c86bd64fa2ede00015ddbb2",
                   "name": "2018-04-12_batchdata_updated_struct_errorcorrect.mat", "size": 3236690412,
                   "sha256": "62c30e413b63e6144720e016deed3661fac8468641794a5807b123fe84717998"},
}
# NASA PCoE files used in section 4: name -> expected bytes and SHA-256 (as listed in DATA_SOURCES.md)
NASA_EXPECTED = {
    "B0005.mat": {"size": 15956874, "sha256": "0eae4585baf3f200c09fe24c5ab884f1889679fc75206ca1aa19da704104f0b0"},
    "B0006.mat": {"size": 16028159, "sha256": "fa818ab4db5db8ab21e910b6dd6c3e20d3761bb9672089e1a4de8f96074616c5"},
    "B0007.mat": {"size": 16049460, "sha256": "d022afa086efaf54ab5b63f05220f5be178c8027e2fa8d68589db0e85a441a3b"},
    "B0018.mat": {"size": 8502218, "sha256": "d1e6c923a43ea1c9666b3a90bbb521757a067fd60b4d17dbfaa49c50b179da69"},
}
NASA_FILES = list(NASA_EXPECTED)

# keep the shared machine responsive
MAX_THREADS = 4


def limit_threads() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(MAX_THREADS))
