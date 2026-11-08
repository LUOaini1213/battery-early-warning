"""No tracked text file may contain a local machine path (drive letters, user home folders)."""
import re
import subprocess

import pytest

from bew.config import REPO

PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]{1,2}[A-Za-z0-9_$.]|"
                     r"[\\/](?:Users|home)[\\/][A-Za-z0-9_.-]+[\\/]")
TEXT = {".py", ".md", ".json", ".csv", ".sql", ".toml", ".txt", ".yml", ".yaml", ".cfg", ".ini"}


SKIP_DIRS = {".git", ".venv", "data", "__pycache__", ".pytest_cache", ".mutation_work", "_cache"}


def tracked_files():
    """git-tracked files; outside a git checkout (e.g. the mutation-test copy) every file not in
    an ignored folder."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
        if (REPO / ".git").exists():
            return [REPO / f for f in out.splitlines() if f]
    except (OSError, subprocess.CalledProcessError):
        pass
    return [p for p in REPO.rglob("*") if p.is_file() and not SKIP_DIRS & set(p.relative_to(REPO).parts)]


def find_local_paths(paths, root=REPO):
    hits = []
    for p in paths:
        if p.suffix.lower() not in TEXT or not p.is_file():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{p.relative_to(root)}:{i}: {line.strip()[:120]}")
    return hits


def test_no_local_paths_in_tracked_files():
    me = REPO / "tests" / "test_no_local_paths.py"
    hits = find_local_paths([p for p in tracked_files() if p != me])
    assert not hits, "local machine paths in tracked files:\n" + "\n".join(hits)


@pytest.mark.parametrize("line", [r"stored at D:\data\nasa_pcoe", r'"D:\\data\\nasa_pcoe"', "default D:/data",
                                  "C:/Users/someone/x", "/home/someone/data", "see /Users/someone/x"])
def test_pattern_catches_local_paths(tmp_path, line):
    p = tmp_path / "x.md"
    p.write_text(line, encoding="utf-8")
    assert find_local_paths([p], root=tmp_path)


@pytest.mark.parametrize("line", ["https://data.matr.io/1/x", "$BEW_DATA_ROOT/nasa_pcoe", "./data",
                                  "s3://bucket/key"])
def test_pattern_ignores_urls_and_placeholders(line):
    assert not PATTERN.search(line)
