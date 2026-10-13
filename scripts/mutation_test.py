"""Mutation check of the test suite: introduce one deliberate bug at a time in a COPY of the repo
and confirm that pytest fails. Writes results/mutation_report.json. The working tree is never
modified; the temporary copy is deleted at the end.

Real-data tests are disabled (BEW_DATA_ROOT points nowhere), so the result reflects what CI sees.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (file, original snippet, mutated snippet, what the bug means)
MUTATIONS = [
    ("bew/features.py", 'summary[(summary["cycle"] <= k)', 'summary[(summary["cycle"] <= k + 50)',
     "leakage: summary features read 50 cycles past k"),
    ("bew/features.py", "dq = delta_q(qdlin[keep], q_cycles[keep], k, lo)", "dq = delta_q(qdlin, q_cycles, k + 1, lo)",
     "leakage: delta Q uses the discharge curve of cycle k+1"),
    ("bew/features.py", "q_cycles = np.arange(1, qdlin.shape[0] + 1)", "q_cycles = np.arange(0, qdlin.shape[0])",
     "off-by-one: Qdlin row r mapped to cycle r instead of r+1"),
    ("bew/features.py", "dq = delta_q(qdlin[keep], q_cycles[keep], k, lo)", "dq = delta_q(qdlin[keep], q_cycles[keep], k, lo + 1)",
     "wrong reference cycle for delta Q (11 instead of 10)"),
    ("bew/features.py", 'f["ir_2"] = float(ir2.iloc[0])', 'f["ir_2"] = float(ir2.iloc[0]) * 1000',
     "units: IR reported in mOhm where ohm is expected"),
    ("bew/cleaning.py", "QD_MAX_VALID = 1.2 * NOMINAL_AH", "QD_MAX_VALID = 2.0 * NOMINAL_AH",
     "cleaning: impossible capacities no longer masked"),
    ("bew/splits.py", "test = np.hstack((np.arange(0, n12, 2), [83]))", "test = np.hstack((np.arange(0, n12, 2), [82]))",
     "split: wrong extra test cell (duplicate / missing cell)"),
    ("bew/splits.py", "train = np.arange(1, n12 - 1, 2)", "train = np.arange(1, n12 + 1, 2)",
     "split: cell 83 in both train and test"),
    ("bew/splits.py", "te = np.isin(groups, test_g)", "te = rng.random(len(groups)) < test_frac",
     "split by row instead of by cell"),
    ("bew/warning.py", "lead = float(eol - alarm)", "lead = float(eol - alarm + 1)",
     "off-by-one in lead time"),
    ("bew/warning.py", "if run >= min_consecutive:", "if run > min_consecutive:",
     "off-by-one in consecutive-flag alarm rule"),
    ("bew/warning.py", "if alarm < knee_onset - tol:", "if alarm < knee_onset + tol:",
     "false-alarm window on the wrong side of knee onset"),
    ("bew/warning.py", "if not np.isfinite(alarm) or alarm >= eol:", "if not np.isfinite(alarm) or alarm > eol:",
     "alarm exactly at EOL counted as a warning"),
    ("bew/warning.py", "rolling(w, min_periods=1).median()", "rolling(w, min_periods=1, center=True).median()",
     "leakage: centred (non-causal) smoothing online"),
    ("bew/warning.py", "seg = y[k - w + 1:k + 1]", "seg = y[k - w + 2:k + 2]",
     "leakage: trailing slope window includes the next cycle"),
    ("bew/warning.py", "rmask = (cycles >= ref[0]) & (cycles <= ref[1]) & np.isfinite(d)", "rmask = np.isfinite(d)",
     "leakage: CUSUM reference statistics computed over the whole life"),
    ("bew/warning.py", "rmask = (cycles >= ref[0]) & (cycles <= ref[1]) & np.isfinite(q)", "rmask = np.isfinite(q)",
     "leakage: threshold baseline reference level from the whole life"),
    ("bew/severson_loader.py", 'tail["cycle"] = tail["cycle"] + len(c1.summary)', 'tail["cycle"] = tail["cycle"]',
     "loader: carried-over cycles not renumbered"),
    ("bew/severson_loader.py", "CARRY_ADD_LEN = [662, 981, 1060, 208, 482]", "CARRY_ADD_LEN = [661, 981, 1060, 208, 482]",
     "loader: wrong cycle-life extension for a carried-over cell"),
    ("bew/severson_loader.py", 'B3_REMOVE = ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]',
     'B3_REMOVE = ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42"]', "loader: a noisy batch-3 cell kept"),
    ("bew/severson_loader.py", "below = np.where(ok & (qd < eol_ah))[0]", "below = np.where(ok & (qd <= eol_ah))[0]",
     "EOL rule: <= instead of < 0.88 Ah"),
    ("bew/models.py", "return float(np.mean(np.abs(yhat - y) / y) * 100.0)", "return float(np.mean(np.abs(yhat - y) / yhat) * 100.0)",
     "metric: percentage error relative to prediction"),
    ("bew/models.py", "return 10 ** model.predict(X), model", "return model.predict(X), model",
     "model output left in log10 space"),
    ("bew/knee.py", "m2 = x <= kp", "m2 = x >= kp", "knee onset searched after the knee point"),
    ("bew/nasa_eis.py", "n_dis += 1", "n_dis += 2", "NASA: discharge counter wrong"),
    ("bew/sql_layer.py", "bad = (~both_nan) & ~(d <= atol)", "bad = (~both_nan) & (d < 0)",
     "engine comparison can never report a numeric mismatch"),
    ("sql/duckdb/04_early_feature_ranking.sql", "WHERE k.split = 'train'", "WHERE k.split <> 'none'",
     "SQL: feature ranking leaks test cells"),
    ("sql/duckdb/02_knee_candidates.sql", "-(w.cov_xy / w.var_x) > 0.0003", "(w.cov_xy / w.var_x) > 0.0003",
     "SQL: knee screen sign error"),
    ("bew/report.py", 'return f"{float(v):.0f}"', 'return f"{int(float(v))}"',
     "README: integers truncated instead of rounded"),
    ("bew/report.py", '"n_primary": (EP, ["split_sizes", "primary_test"], "d0"),',
     '"n_primary": (EP, ["split_sizes", "train"], "d0"),', "README: number pulled from the wrong result"),
]


def run_pytest(cwd: Path, env: dict) -> tuple[int, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
                       cwd=cwd, env=env, capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout[-2000:]


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="bew_mut_"))
    try:
        src = work / "repo"
        shutil.copytree(REPO, src, ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", ".pytest_cache"))
        env = dict(os.environ, BEW_DATA_ROOT=str(work / "no_data"), PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="4", BEW_RESULTS=str(src / "results"))
        code, out = run_pytest(src, env)
        if code != 0:
            sys.exit("baseline test suite fails; fix before mutation testing\n" + out)
        m = re.search(r"(\d+) passed", out)
        baseline = int(m.group(1)) if m else -1
        rows = []
        for f, old, new, why in MUTATIONS:
            p = src / f
            text = p.read_text(encoding="utf-8")
            if text.count(old) != 1:
                sys.exit(f"mutation anchor not unique/absent in {f}: {old!r}")
            p.write_text(text.replace(old, new), encoding="utf-8")
            try:
                code, out = run_pytest(src, env)
            finally:
                p.write_text(text, encoding="utf-8")
            failed = re.findall(r"FAILED (\S+)", out)
            rows.append({"file": f, "bug": why, "caught": code != 0,
                         "first_failing_test": failed[0] if failed else ("error" if code not in (0, 1) else None)})
            print(("CAUGHT  " if code != 0 else "MISSED  ") + why, flush=True)
        report = {"baseline_tests_passed": baseline, "total": len(rows),
                  "caught": sum(r["caught"] for r in rows), "mutations": rows}
        prev_path = REPO / "results" / "mutation_report.json"
        if prev_path.exists():
            prev = json.loads(prev_path.read_text())
            hist = prev.get("history", [])
            hist.append({"caught": prev["caught"], "total": prev["total"],
                         "missed": [m["bug"] for m in prev["mutations"] if not m["caught"]]})
            report["history"] = hist
        (REPO / "results" / "mutation_report.json").write_text(json.dumps(report, indent=2))
        print(f"caught {report['caught']}/{report['total']}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
