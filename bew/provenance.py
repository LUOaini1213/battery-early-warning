"""SHA-256 provenance for every raw file used."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_files(paths: list[Path], cache: Path | None = None) -> list[dict]:
    """Hash files; reuse a cached hash only when size and mtime are unchanged."""
    old = {}
    if cache and cache.exists():
        old = {d["name"]: d for d in json.loads(cache.read_text())["files"]}
    out = []
    for p in paths:
        st = os.stat(p)
        prev = old.get(p.name)
        if prev and prev["bytes"] == st.st_size and prev.get("mtime") == int(st.st_mtime):
            out.append(prev)
            continue
        out.append({"name": p.name, "bytes": st.st_size, "mtime": int(st.st_mtime),
                    "sha256": sha256(p)})
    return out


def write_provenance(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["written_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
