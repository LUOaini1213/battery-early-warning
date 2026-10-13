"""Run the ClickHouse DDL + templates in a throw-away container and compare with DuckDB.

Starts exactly one container (name `bew-clickhouse`, official image, --memory=1g) inside the WSL
distro, loads the same tables, runs every template, writes results/sql/clickhouse_check.json and
ALWAYS stops and removes the container at the end (also on error). It never touches other
containers.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bew import config  # noqa: E402

config.limit_threads()
import pandas as pd  # noqa: E402

from bew import sql_layer  # noqa: E402
from bew.extract import load_derived  # noqa: E402

DISTRO = "Ubuntu-24.04"
IMAGE = "clickhouse/clickhouse-server:24.8"
NAME = "bew-clickhouse"


def wsl_path(p: Path) -> str:
    w = PureWindowsPath(p.resolve())
    return "/mnt/" + w.drive[0].lower() + "/" + "/".join(w.parts[1:])


def docker(*args: str, check: bool = True, timeout: int = 600) -> subprocess.CompletedProcess:
    cmd = ["wsl", "-d", DISTRO, "--", "docker", *args]
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])} failed: {r.stderr.decode(errors='replace')[-800:]}")
    return r


def ch(query: str | None = None, file: str | None = None, fmt: str | None = None) -> str:
    args = ["exec", NAME, "clickhouse-client"]
    if fmt:
        args += ["--format", fmt]
    if query:
        args += ["--query", query]
    if file:
        args += ["--queries-file", file]
    return docker(*args).stdout.decode()


def main() -> None:
    R = config.RESULTS
    cells, cycles, _ = load_derived()
    feats = pd.read_csv(R / "early_prediction" / "features_first100.csv", index_col=0)
    knees = pd.read_csv(R / "warning" / "knee_labels.csv", index_col=0)
    tbls = sql_layer.tables(cells, cycles, feats, knees)
    export = config.SEVERSON_DERIVED / "sql_export"
    export.mkdir(parents=True, exist_ok=True)
    for name, df in tbls.items():
        df.to_parquet(export / f"{name}.parquet", index=False)
    con = sql_layer.build_duckdb(tbls)
    duck = sql_layer.run_templates(con)
    con.close()

    existing = docker("ps", "-a", "--filter", f"name=^{NAME}$", "--format", "{{.Names}}").stdout.decode().split()
    if NAME in existing:
        sys.exit(f"a container named {NAME} already exists; refusing to touch it")
    report = {"image": IMAGE, "container": NAME, "memory_limit": "1g", "templates": {}}
    started = False
    try:
        docker("pull", IMAGE, timeout=1200)
        docker("run", "-d", "--name", NAME, "--memory=1g",
               "--ulimit", "nofile=262144:262144",
               "-v", f"{wsl_path(export)}:/import:ro",
               "-v", f"{wsl_path(config.REPO / 'sql' / 'clickhouse')}:/sql:ro", IMAGE)
        started = True
        for _ in range(60):
            if docker("exec", NAME, "clickhouse-client", "--query", "SELECT 1", check=False).returncode == 0:
                break
            time.sleep(2)
        report["clickhouse_version"] = ch("SELECT version()").strip()
        ch(file="/sql/00_ddl.sql")
        for name in tbls:
            ch(f"INSERT INTO {name} FROM INFILE '/import/{name}.parquet' FORMAT Parquet")
            n = int(ch(f"SELECT count() FROM {name}").strip())
            report.setdefault("loaded_rows", {})[name] = n
            if n != len(tbls[name]):
                raise RuntimeError(f"{name}: loaded {n} rows, expected {len(tbls[name])}")
        n_equal, worst = 0, 0.0
        for f in sorted((config.REPO / "sql" / "clickhouse").glob("0[1-9]_*.sql")):
            out = ch(file=f"/sql/{f.name}", fmt="CSVWithNames")
            got = pd.read_csv(io.StringIO(out))
            got.to_csv(R / "sql" / f"clickhouse_{f.stem}.csv", index=False)
            cmp = sql_layer.compare_frames(duck[f.stem], got, sql_layer.TEMPLATE_KEYS[f.stem])
            report["templates"][f.stem] = cmp
            n_equal += int(cmp["equal"])
            worst = max(worst, cmp["max_abs_diff"])
        report["n_templates"] = len(report["templates"])
        report["n_templates_equal"] = n_equal
        report["max_abs_diff_all"] = worst
    finally:
        if started or NAME in docker("ps", "-a", "--format", "{{.Names}}", check=False).stdout.decode().split():
            docker("stop", NAME, check=False)
            docker("rm", NAME, check=False)
        left = docker("ps", "-a", "--filter", f"name=^{NAME}$", "--format", "{{.Names}}", check=False).stdout.decode().split()
        report["container_removed"] = NAME not in left
        (R / "sql").mkdir(parents=True, exist_ok=True)
        (R / "sql" / "clickhouse_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "templates"}, indent=2))
    for k, v in report["templates"].items():
        print(k, "equal" if v["equal"] else v)


if __name__ == "__main__":
    main()
