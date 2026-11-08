"""Raw files are checked against the published SHA-256 values, and a missing hand-placed NASA file
stops run_all.py before any download."""
import json
import re

import pytest

from bew import config
from bew.provenance import check_hashes, hash_files, missing_files, sha256

ROW = re.compile(r"^\s*\| (\S+\.mat) \|(?: \S+ \|)? ([\d,]+) \| ([0-9a-f]{64}) \|", re.M)


def test_expected_hashes_match_data_sources_and_provenance():
    doc = {m[0]: (int(m[1].replace(",", "")), m[2])
           for m in ROW.findall((config.REPO / "DATA_SOURCES.md").read_text(encoding="utf-8"))}
    expected = {m["name"]: (m["size"], m["sha256"]) for m in config.SEVERSON_BATCHES.values()}
    expected.update({n: (m["size"], m["sha256"]) for n, m in config.NASA_EXPECTED.items()})
    assert doc == expected
    prov = json.loads((config.RESULTS / "provenance.json").read_text())
    recorded = {f["name"]: (f["bytes"], f["sha256"])
                for k in ("severson2019", "nasa_pcoe") for f in prov[k]["files"]}
    assert recorded == expected


def test_check_hashes_flags_a_wrong_file(tmp_path):
    p = tmp_path / "B0005.mat"
    p.write_bytes(b"not the NASA file")
    files = hash_files([p])
    assert files[0]["sha256"] == sha256(p)
    bad = check_hashes(files, {n: m["sha256"] for n, m in config.NASA_EXPECTED.items()})
    assert len(bad) == 1 and "B0005.mat" in bad[0]
    assert check_hashes(files, {"B0005.mat": sha256(p)}) == []
    assert check_hashes(files, {}) != []


def test_missing_nasa_files_stop_run_all_before_download(tmp_path, monkeypatch):
    import run_all

    monkeypatch.setattr(config, "NASA_RAW", tmp_path / "nasa_pcoe")
    monkeypatch.setattr(config, "SEVERSON_RAW", tmp_path / "severson2019" / "raw")

    def boom(*a, **k):
        raise AssertionError("download started although NASA files are missing")

    import scripts.fetch_data as fd
    monkeypatch.setattr(fd, "fetch_severson", boom)
    with pytest.raises(SystemExit) as e:
        run_all.provenance(download=True)
    msg = str(e.value)
    assert all(n in msg for n in config.NASA_FILES)
    assert missing_files([tmp_path]) == [tmp_path]  # a directory is not a data file


def test_default_data_root_is_inside_the_repo(monkeypatch):
    import importlib

    monkeypatch.delenv("BEW_DATA_ROOT", raising=False)
    try:
        cfg = importlib.reload(config)
        assert cfg.DATA_ROOT == cfg.REPO / "data" and cfg.DATA_ROOT.is_absolute()
    finally:
        monkeypatch.undo()
        importlib.reload(config)
