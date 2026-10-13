# battery-early-warning

Early-life cycle-life prediction, online knee warning, a failure-case library and an SQL analyst
layer, built on the public Severson et al. (2019) fast-charging dataset (124 commercial
LFP/graphite 18650 cells) plus four NASA PCoE lab cells with impedance data.

> **What this is.** A personal portfolio project built in October 2026 with AI assistance (code
> and text were drafted with an AI coding assistant and checked by running them). All data is
> public. Every result number in this README is generated from `results/` by `run_all.py`, and
> `tests/test_readme_numbers.py` parses this file and checks each one against those files
> (the remaining numbers are fixed protocol constants such as the 0.88 Ah end-of-life threshold).
>
> **What is not claimed.** No BMS or production deployment; no proprietary or field data; lab
> cells only (one cell type and one test rig for the main dataset, four cells for the impedance
> analysis). Root causes in the failure-case library are hypotheses, not diagnoses.

**中文摘要**：个人作品（2026 年 10 月，借助 AI 辅助完成），只用公开数据：Severson 2019 的
124 颗磷酸铁锂快充电芯 + NASA PCoE 4 颗实验室电芯的阻抗数据。内容包括：用前 100 圈预测寿命并
识别早衰电芯（按论文原划分复现，方差模型的误差与论文表 1 基本一致）；只用历史数据的在线拐点预警，比较
四种方法的提前量与误报；25 个异常电芯案例（根因一律写成"假设"并附文献）；DuckDB +
ClickHouse 两套 SQL 模板并核对结果一致。不涉及 BMS 部署、专有数据或现场数据。

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # or: uv venv && uv pip install -r requirements.txt
python run_all.py --download        # fetches 8.27 GB from data.matr.io into $BEW_DATA_ROOT (default D:/data);
                                     # the 4 NASA files go in $BEW_DATA_ROOT/nasa_pcoe (see DATA_SOURCES.md)
python -m pytest                     # unit tests (CI runs these on a synthetic fixture)
python scripts/verify_clickhouse.py  # optional: needs Docker; runs the ClickHouse templates and compares with DuckDB
python scripts/mutation_test.py      # optional: breaks the code on purpose and checks the tests notice
```

Raw data never enters the repo. Every downloaded file is SHA-256 hashed into
`results/provenance.json`; licences and citations are in [DATA_SOURCES.md](DATA_SOURCES.md).

## 1. Early prediction from the first 100 cycles

Protocol follows the paper and the official loader: same cell exclusions and merge, same split
(41 train / 43 primary test / 40 secondary test, by cell),
target log10(cycle life), elastic net with repeated 4-fold CV on the training cells only. Features
are my reconstruction from the paper's main text (the supplementary feature table was not used),
so the discharge and full models use 12 and 19 candidate features instead
of 13 and 20.

RMSE in cycles; in brackets: primary test without the anomalous cell b2c1 (life
148), as in the paper.

| model | train | primary test | secondary test | paper (train / primary / secondary) |
|---|---|---|---|---|
| variance (1 feature, OLS) | 104 | 138 (138) | 196 | 103 / 138 (138) / 196 |
| discharge (elastic net, 8 of 12 selected) | 73 | 92 (88) | 171 | 76 / 91 (86) / 173 |
| full (elastic net, 9 of 19 selected) | 65 | 125 (124) | 223 | 51 / 118 (100) / 214 |
| random forest (full features) | 137 | 201 (198) | 258 | not in paper |

Mean absolute percentage error (%):

| model | train | primary test | secondary test | paper |
|---|---|---|---|---|
| variance | 14.1 | 14.7 (13.2) | 11.4 | 14.1 / 14.7 (13.2) / 11.4 |
| discharge | 9.5 | 12.7 (9.6) | 9.4 | 9.8 / 13.0 (10.1) / 8.6 |
| full | 7.4 | 12.1 (9.5) | 12.4 | 5.6 / 14.1 (7.5) / 10.7 |
| random forest | 5.4 | 14.5 (10.0) | 12.7 | — |

Where I agree and differ with the paper:

- **Variance model reproduces.** Correlation of log10 var ΔQ100-10(V) with log10 cycle life is
  -0.93; the single-feature model matches Table 1 to within one cycle of RMSE.
- **Discharge model is close**, the **full model is not**: my full model is worse on the primary test
  set. Most likely cause: my 19 features are a reconstruction, not the paper's exact 20
  (e.g. I use a sum of average temperature instead of the paper's temperature integral).
- **The "stronger" model is not stronger.** A random forest on the same features is worse than the
  linear models on both test sets: with 41 training cells, trees cannot extrapolate to the
  shortest- and longest-lived cells.
- **One fixed split is one draw.** Over 30 random 50/50 cell splits of batches 1–2, test RMSE
  median (10th–90th percentile) is 133 (106–159) for the variance
  model and 100 (82–142) for the full model, so on average the full
  model does beat the variance model; the ranking on the single paper split is not stable.
- **Post-hoc cleaning rule, stated openly.** The first discharge-model run predicted ~0 cycles for
  two cells because of one impossible capacity reading in each (far above the 1.1 Ah rating). I then
  added one physical-range rule (capacity must be in (0, 1.32] Ah) and applied it everywhere. The
  variance model is unaffected by it.

**Early-failure classification** (life < 550 cycles, the paper's Table 2 threshold;
34.7% of all cells). Accuracy in %, early-failure recall in brackets:

| classifier | train | primary test | secondary test | paper (train / primary / secondary) |
|---|---|---|---|---|
| variance, cycles 4–5 only | 85.4 | 79.1 (81.0) | 97.5 (0.0) | 82.1 / 78.6 / 97.5 |
| variance, cycles 10–100 | 82.9 | 86.0 (85.7) | 97.5 (100.0) | — |
| L1 logistic, all 100-cycle features | 100.0 | 90.7 (95.2) | 95.0 (0.0) | full classifier uses 5 cycles: 97.4 / 92.7 / 97.5 |

The secondary test set contains only 1 early-failure cell, so its accuracy says
little about catching early failures (predicting "long-lived" for every cell already scores high).
The primary test set (21 early failures out of 43) is the informative one.

Files: `results/early_prediction/metrics.json`, `predictions.csv`, `pred_vs_obs.png`.

## 2. Online knee / accelerated-fade warning

At every cycle a detector sees only capacities up to that cycle (trailing windows; a test perturbs
the future and checks nothing earlier changes). Monitoring starts at cycle 101. Parameters are tuned
on the 41 training cells and evaluated on the other 83 cells.
Ground truth comes from the whole trajectory and is used only for scoring: knee onset and knee point
from two-segment fits (median at 53% and 77% of life) and end of life
(EOL, first cycle below 0.88 Ah). A first alarm is a *true warning* if it comes no earlier than 50
cycles before knee onset and before EOL, a *false alarm* if earlier, *missed* otherwise.

| detector (test cells) | true warning % | false alarm % | missed | median lead to EOL | 25th pct lead | median delay after knee onset |
|---|---|---|---|---|---|---|
| capacity < 95% of early level (naive baseline) | 97.6 | 2.4 | 0 | 186 | 144 | 189 |
| rolling slope (trailing OLS fade rate) | 100.0 | 0.0 | 0 | 142 | 120 | 235 |
| CUSUM on one-cycle capacity drops | 89.2 | 8.4 | 2 | 328 | 247 | 48 |
| model-based: quadratic forecast of cycles to 0.88 Ah | 100.0 | 0.0 | 0 | 114 | 97 | 260 |

Reading: CUSUM warns earliest (median 328 cycles before EOL) but has the most false
alarms on the test cells (8.4%, versus 0.0% on the training cells it was
tuned on) and the only misses. In `results/warning/alarm_examples.png` some CUSUM alarms line up with
small capacity discontinuities caused by test interruptions rather than with the knee, the usual
weakness of a change detector on lab data with pauses. The rolling-slope and forecast detectors have
no false alarms or misses on the test cells but warn later. The naive capacity threshold gives more
lead time than both of them at the cost of 2.4% false alarms; in this dataset the fancier
detectors do not clearly beat it. Lead times are in cycles
of this lab protocol, not field time. Files: `results/warning/`.

## 3. Failure-case library

25 of 124 cells carry at least one signature
(`results/failure_cases/failure_cases.csv` and `.md`, one evidence plot per cell):
early knee 6, life far below the early-life prediction 1,
far above it 1, internal-resistance step 3,
temperature step 11, noisy capacity 6,
physically impossible capacity values 4. Each row gives the measured evidence and a
root-cause **hypothesis** with a reference (e.g. lithium plating under fast charge for the early knees;
thermocouple contact loss or chamber drift for temperature steps, both documented by the data provider).
None of them is verified: there is no teardown or diagnostic data.

## 4. Impedance vs capacity (NASA PCoE, 4 lab cells)

The 887 impedance operations of B0005/6/7/18 carry fitted Re and Rct. Within every cell Re
rises as capacity falls (Spearman ρ of Re vs capacity -0.96 to -0.76); Rct does so
clearly in three cells but hardly in B0018 (ρ from -0.93 to -0.10). But once the number of discharges so far is controlled for, the
partial correlation is at most 0.25 in absolute value, and in leave-one-cell-out regression
Re+Rct predict capacity worse (RMSE 0.186 Ah; 0.236 Ah using growth from each cell's
own start) than discharge count alone (0.091 Ah). So in these four cells resistance growth
and capacity fade move together with time on test; the data do not show resistance carrying extra
information. Four cells cannot settle this. Files: `results/nasa_eis/`.

## 5. Analyst layer: DuckDB and ClickHouse

`sql/duckdb/` and `sql/clickhouse/` hold the schema/DDL and 8 templates: cohort
fade curves, knee candidates (trailing-window regression in a window function), IR vs capacity,
early-life feature ranking (training cells only), survival by cycle, data quality, charging-policy
leaderboard, and an early-failure screen that fits the variance model in pure SQL (it matches the
Python model to 0.000 cycles). The SQL knee screen flags 124 cells before
EOL with median lead 146 cycles.
`scripts/verify_clickhouse.py` loads the same tables into ClickHouse 24.8.14.39 (Docker, 1 GB
memory cap), runs every template and compares with DuckDB: 8 of 8 templates give
identical results at the precision the templates round to (up to 6 decimals; max absolute
difference 0.0e+00); the container is removed afterwards.

## Tests

`python -m pytest`: loaders (on synthetic MATLAB-v7.3-style files), features with a future-leakage
check, splits (no cell on both sides), lead-time arithmetic, knee labels, SQL templates, NASA loader
and README numbers. Mutation check: `scripts/mutation_test.py` applied 30 deliberate bugs
(leakage, wrong split, off-by-one lead time, wrong units, ...); the 62 tests caught
30 of 30. Earlier runs were weaker and are kept in
`results/mutation_report.json`: the first caught 27 of 28 (missed:
"leakage: centred (non-causal) smoothing online"), the second 28 of 30 (missed: "leakage: CUSUM reference statistics computed over the whole life"
and "leakage: threshold baseline reference level from the whole life"). Both times the causality tests only looked at thresholded alarm flags,
which a leak can leave unchanged; tests on the continuous online statistics were added after each.

## Layout

```
bew/            library code (loader, features, models, knee, warning, failure cases, NASA, SQL, report)
run_all.py      one command for every result + README
scripts/        data fetch, ClickHouse check, mutation test
sql/            DuckDB schema + templates, ClickHouse DDL + templates
tests/          pytest suite; synthetic fixtures only
results/        generated outputs (JSON, CSV, PNG)
```

## Citation

Data: K. A. Severson, P. M. Attia et al., "Data-driven prediction of battery cycle life before capacity
degradation", *Nature Energy* 4, 383–391 (2019); data.matr.io, CC BY 4.0.
B. Saha and K. Goebel, "Battery Data Set", NASA Prognostics Data Repository, NASA Ames Research Center (2007).

## License

Code: MIT (see `LICENSE`). Severson et al. 2019 data: CC BY 4.0 (data.matr.io), not redistributed here. NASA PCoE battery data: cite Saha & Goebel 2007; only aggregate results are included.
