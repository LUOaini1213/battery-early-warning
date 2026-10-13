"""Download the Severson et al. (2019) batch structs from the official data.matr.io API.

Files are written to BEW_DATA_ROOT/severson2019/raw (default D:/data), never into the repo.
Existing files with the expected byte size are skipped. Licence: CC BY 4.0 (see DATA_SOURCES.md).
The NASA PCoE files are not downloaded here; see DATA_SOURCES.md for their official URL.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bew import config  # noqa: E402

API = "https://data.matr.io/1/api/v1/file/{file_id}/download"


def fetch_severson(raw: Path = config.SEVERSON_RAW) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(raw).free
    need = sum(m["size"] for m in config.SEVERSON_BATCHES.values() if not (raw / m["name"]).exists())
    if need > free - 5e9:
        sys.exit(f"not enough free space on {raw}: need {need/1e9:.1f} GB + 5 GB margin")
    for b, m in config.SEVERSON_BATCHES.items():
        p = raw / m["name"]
        if p.exists() and p.stat().st_size == m["size"]:
            print("ok  ", p.name)
            continue
        print("get ", p.name, f"{m['size']/1e9:.2f} GB")
        tmp = p.with_suffix(".part")
        with urllib.request.urlopen(API.format(file_id=m["file_id"])) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh, length=8 << 20)
        if tmp.stat().st_size != m["size"]:
            sys.exit(f"incomplete download {tmp}")
        tmp.replace(p)


if __name__ == "__main__":
    fetch_severson()
