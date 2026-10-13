"""Generate the small SYNTHETIC fixture committed under tests/fixtures (no real data)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import pandas as pd  # noqa: E402

from synth import capacity_curve  # noqa: E402

LF = chr(10)  # LF line endings on every OS so CI can diff the fixture

SPECS = [  # (eol, knee) per synthetic cell
    (180, 120), (260, 170), (320, 230), (380, 250), (450, 300), (520, 360), (600, 420), (700, 500),
]


def main() -> None:
    out = ROOT / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    cells, cyc = [], []
    for i, (eol, knee) in enumerate(SPECS):
        cid = f"syn{i}"
        s = capacity_curve(eol=eol, knee=knee, n_extra=5, seed=100 + i).round(6)
        s.insert(0, "cell_id", cid)
        cyc.append(s)
        cells.append({"cell_id": cid, "order_idx": i, "batch": "synthetic", "policy": f"{4 + i % 3}C(80%)-4C",
                      "c1": 4.0 + i % 3, "q1": 80.0, "c2": 4.0, "split": "train" if i % 2 else "primary_test",
                      "cycle_life": float(eol), "carried_over": False})
    pd.DataFrame(cells).to_csv(out / "synthetic_cells.csv", index=False, lineterminator=LF)
    pd.concat(cyc, ignore_index=True).to_csv(out / "synthetic_cycles.csv", index=False, lineterminator=LF)


if __name__ == "__main__":
    main()
