# battery-early-warning

Early-life cycle-life prediction, online knee warning, a failure-case library and an SQL analyst
layer, built on the public Severson et al. (2019) fast-charging dataset ({{n_cells}} commercial
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
{{n_cells}} 颗磷酸铁锂快充电芯 + NASA PCoE 4 颗实验室电芯的阻抗数据。内容包括：用前 100 圈预测寿命并
识别早衰电芯（按论文原划分复现，方差模型的误差与论文表 1 基本一致）；只用历史数据的在线拐点预警，比较
四种方法的提前量与误报；{{fc_cases}} 个异常电芯案例（根因一律写成"假设"并附文献）；DuckDB +
ClickHouse 两套 SQL 模板并核对结果一致。不涉及 BMS 部署、专有数据或现场数据。

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # or: uv venv && uv pip install -r requirements.txt
python run_all.py --download        # fetches {{raw_gb}} GB from data.matr.io into $BEW_DATA_ROOT (default D:/data);
                                     # the 4 NASA files go in $BEW_DATA_ROOT/nasa_pcoe (see DATA_SOURCES.md)
python -m pytest                     # unit tests (CI runs these on a synthetic fixture)
python scripts/verify_clickhouse.py  # optional: needs Docker; runs the ClickHouse templates and compares with DuckDB
python scripts/mutation_test.py      # optional: breaks the code on purpose and checks the tests notice
```

Raw data never enters the repo. Every downloaded file is SHA-256 hashed into
`results/provenance.json`; licences and citations are in [DATA_SOURCES.md](DATA_SOURCES.md).

## 1. Early prediction from the first 100 cycles

Protocol follows the paper and the official loader: same cell exclusions and merge, same split
({{n_train}} train / {{n_primary}} primary test / {{n_secondary}} secondary test, by cell),
target log10(cycle life), elastic net with repeated 4-fold CV on the training cells only. Features
are my reconstruction from the paper's main text (the supplementary feature table was not used),
so the discharge and full models use {{dis_ncand}} and {{full_ncand}} candidate features instead
of 13 and 20.

RMSE in cycles; in brackets: primary test without the anomalous cell {{outlier_id}} (life
{{outlier_life}}), as in the paper.

| model | train | primary test | secondary test | paper (train / primary / secondary) |
|---|---|---|---|---|
| variance (1 feature, OLS) | {{var_rmse_tr}} | {{var_rmse_pt}} ({{var_rmse_px}}) | {{var_rmse_st}} | {{paper_var_rmse_tr}} / {{paper_var_rmse_pt}} ({{paper_var_rmse_px}}) / {{paper_var_rmse_st}} |
| discharge (elastic net, {{dis_nsel}} of {{dis_ncand}} selected) | {{dis_rmse_tr}} | {{dis_rmse_pt}} ({{dis_rmse_px}}) | {{dis_rmse_st}} | {{paper_dis_rmse_tr}} / {{paper_dis_rmse_pt}} ({{paper_dis_rmse_px}}) / {{paper_dis_rmse_st}} |
| full (elastic net, {{full_nsel}} of {{full_ncand}} selected) | {{full_rmse_tr}} | {{full_rmse_pt}} ({{full_rmse_px}}) | {{full_rmse_st}} | {{paper_full_rmse_tr}} / {{paper_full_rmse_pt}} ({{paper_full_rmse_px}}) / {{paper_full_rmse_st}} |
| random forest (full features) | {{rf_rmse_tr}} | {{rf_rmse_pt}} ({{rf_rmse_px}}) | {{rf_rmse_st}} | not in paper |

Mean absolute percentage error (%):

| model | train | primary test | secondary test | paper |
|---|---|---|---|---|
| variance | {{var_mpe_tr}} | {{var_mpe_pt}} ({{var_mpe_px}}) | {{var_mpe_st}} | {{paper_var_mpe_tr}} / {{paper_var_mpe_pt}} ({{paper_var_mpe_px}}) / {{paper_var_mpe_st}} |
| discharge | {{dis_mpe_tr}} | {{dis_mpe_pt}} ({{dis_mpe_px}}) | {{dis_mpe_st}} | {{paper_dis_mpe_tr}} / {{paper_dis_mpe_pt}} ({{paper_dis_mpe_px}}) / {{paper_dis_mpe_st}} |
| full | {{full_mpe_tr}} | {{full_mpe_pt}} ({{full_mpe_px}}) | {{full_mpe_st}} | {{paper_full_mpe_tr}} / {{paper_full_mpe_pt}} ({{paper_full_mpe_px}}) / {{paper_full_mpe_st}} |
| random forest | {{rf_mpe_tr}} | {{rf_mpe_pt}} ({{rf_mpe_px}}) | {{rf_mpe_st}} | — |

Where I agree and differ with the paper:

- **Variance model reproduces.** Correlation of log10 var ΔQ100-10(V) with log10 cycle life is
  {{corr_logvar}}; the single-feature model matches Table 1 to within one cycle of RMSE.
- **Discharge model is close**, the **full model is not**: my full model is worse on the primary test
  set. Most likely cause: my 19 features are a reconstruction, not the paper's exact 20
  (e.g. I use a sum of average temperature instead of the paper's temperature integral).
- **The "stronger" model is not stronger.** A random forest on the same features is worse than the
  linear models on both test sets: with {{n_train}} training cells, trees cannot extrapolate to the
  shortest- and longest-lived cells.
- **One fixed split is one draw.** Over {{rs_n}} random 50/50 cell splits of batches 1–2, test RMSE
  median (10th–90th percentile) is {{rs_var}} ({{rs_var_p10}}–{{rs_var_p90}}) for the variance
  model and {{rs_full}} ({{rs_full_p10}}–{{rs_full_p90}}) for the full model, so on average the full
  model does beat the variance model; the ranking on the single paper split is not stable.
- **Post-hoc cleaning rule, stated openly.** The first discharge-model run predicted ~0 cycles for
  two cells because of one impossible capacity reading in each (far above the 1.1 Ah rating). I then
  added one physical-range rule (capacity must be in (0, 1.32] Ah) and applied it everywhere. The
  variance model is unaffected by it.

**Early-failure classification** (life < 550 cycles, the paper's Table 2 threshold;
{{ef_rate}}% of all cells). Accuracy in %, early-failure recall in brackets:

| classifier | train | primary test | secondary test | paper (train / primary / secondary) |
|---|---|---|---|---|
| variance, cycles 4–5 only | {{cls_v5_acc_tr}} | {{cls_v5_acc_pt}} ({{cls_v5_rec_pt}}) | {{cls_v5_acc_st}} ({{cls_v5_rec_st}}) | {{paper_cls_var_tr}} / {{paper_cls_var_pt}} / {{paper_cls_var_st}} |
| variance, cycles 10–100 | {{cls_v100_acc_tr}} | {{cls_v100_acc_pt}} ({{cls_v100_rec_pt}}) | {{cls_v100_acc_st}} ({{cls_v100_rec_st}}) | — |
| L1 logistic, all 100-cycle features | {{cls_f100_acc_tr}} | {{cls_f100_acc_pt}} ({{cls_f100_rec_pt}}) | {{cls_f100_acc_st}} ({{cls_f100_rec_st}}) | full classifier uses 5 cycles: {{paper_cls_full_tr}} / {{paper_cls_full_pt}} / {{paper_cls_full_st}} |

The secondary test set contains only {{cls_v5_nef_st}} early-failure cell, so its accuracy says
little about catching early failures (predicting "long-lived" for every cell already scores high).
The primary test set ({{cls_v5_nef_pt}} early failures out of {{n_primary}}) is the informative one.

Files: `results/early_prediction/metrics.json`, `predictions.csv`, `pred_vs_obs.png`.

## 2. Online knee / accelerated-fade warning

At every cycle a detector sees only capacities up to that cycle (trailing windows; a test perturbs
the future and checks nothing earlier changes). Monitoring starts at cycle 101. Parameters are tuned
on the {{n_train}} training cells and evaluated on the other {{w_slope_n}} cells.
Ground truth comes from the whole trajectory and is used only for scoring: knee onset and knee point
from two-segment fits (median at {{kn_onset_frac}}% and {{kn_point_frac}}% of life) and end of life
(EOL, first cycle below 0.88 Ah). A first alarm is a *true warning* if it comes no earlier than 50
cycles before knee onset and before EOL, a *false alarm* if earlier, *missed* otherwise.

| detector (test cells) | true warning % | false alarm % | missed | median lead to EOL | 25th pct lead | median delay after knee onset |
|---|---|---|---|---|---|---|
| capacity < 95% of early level (naive baseline) | {{w_thr_tw}} | {{w_thr_fa}} | {{w_thr_miss}} | {{w_thr_lead}} | {{w_thr_lead25}} | {{w_thr_delay}} |
| rolling slope (trailing OLS fade rate) | {{w_slope_tw}} | {{w_slope_fa}} | {{w_slope_miss}} | {{w_slope_lead}} | {{w_slope_lead25}} | {{w_slope_delay}} |
| CUSUM on one-cycle capacity drops | {{w_cusum_tw}} | {{w_cusum_fa}} | {{w_cusum_miss}} | {{w_cusum_lead}} | {{w_cusum_lead25}} | {{w_cusum_delay}} |
| model-based: quadratic forecast of cycles to 0.88 Ah | {{w_fcst_tw}} | {{w_fcst_fa}} | {{w_fcst_miss}} | {{w_fcst_lead}} | {{w_fcst_lead25}} | {{w_fcst_delay}} |

Reading: CUSUM warns earliest (median {{w_cusum_lead}} cycles before EOL) but has the most false
alarms on the test cells ({{w_cusum_fa}}%, versus {{w_cusum_fa_train}}% on the training cells it was
tuned on) and the only misses. In `results/warning/alarm_examples.png` some CUSUM alarms line up with
small capacity discontinuities caused by test interruptions rather than with the knee, the usual
weakness of a change detector on lab data with pauses. The rolling-slope and forecast detectors have
no false alarms or misses on the test cells but warn later. The naive capacity threshold gives more
lead time than both of them at the cost of {{w_thr_fa}}% false alarms; in this dataset the fancier
detectors do not clearly beat it. Lead times are in cycles
of this lab protocol, not field time. Files: `results/warning/`.

## 3. Failure-case library

{{fc_cases}} of {{fc_screened}} cells carry at least one signature
(`results/failure_cases/failure_cases.csv` and `.md`, one evidence plot per cell):
early knee {{fc_early_knee}}, life far below the early-life prediction {{fc_life_overpredicted}},
far above it {{fc_life_underpredicted}}, internal-resistance step {{fc_ir_jump}},
temperature step {{fc_temperature_step}}, noisy capacity {{fc_noisy_capacity}},
physically impossible capacity values {{fc_invalid_values}}. Each row gives the measured evidence and a
root-cause **hypothesis** with a reference (e.g. lithium plating under fast charge for the early knees;
thermocouple contact loss or chamber drift for temperature steps, both documented by the data provider).
None of them is verified: there is no teardown or diagnostic data.

## 4. Impedance vs capacity (NASA PCoE, 4 lab cells)

The {{na_ops}} impedance operations of B0005/6/7/18 carry fitted Re and Rct. Within every cell Re
rises as capacity falls (Spearman ρ of Re vs capacity {{na_re_min}} to {{na_re_max}}); Rct does so
clearly in three cells but hardly in B0018 (ρ from {{na_rct_min}} to {{na_rct_max}}). But once the number of discharges so far is controlled for, the
partial correlation is at most {{na_partial}} in absolute value, and in leave-one-cell-out regression
Re+Rct predict capacity worse (RMSE {{na_loco_rr}} Ah; {{na_loco_drr}} Ah using growth from each cell's
own start) than discharge count alone ({{na_loco_dc}} Ah). So in these four cells resistance growth
and capacity fade move together with time on test; the data do not show resistance carrying extra
information. Four cells cannot settle this. Files: `results/nasa_eis/`.

## 5. Analyst layer: DuckDB and ClickHouse

`sql/duckdb/` and `sql/clickhouse/` hold the schema/DDL and {{sql_n_templates}} templates: cohort
fade curves, knee candidates (trailing-window regression in a window function), IR vs capacity,
early-life feature ranking (training cells only), survival by cycle, data quality, charging-policy
leaderboard, and an early-failure screen that fits the variance model in pure SQL (it matches the
Python model to {{sql_py_diff}} cycles). The SQL knee screen flags {{sql_knee_flagged}} cells before
EOL with median lead {{sql_knee_lead}} cycles.
`scripts/verify_clickhouse.py` loads the same tables into ClickHouse {{ch_version}} (Docker, 1 GB
memory cap), runs every template and compares with DuckDB: {{ch_equal}} of {{ch_total}} templates give
identical results at the precision the templates round to (up to 6 decimals; max absolute
difference {{ch_maxdiff}}); the container is removed afterwards.

## Tests

`python -m pytest`: loaders (on synthetic MATLAB-v7.3-style files), features with a future-leakage
check, splits (no cell on both sides), lead-time arithmetic, knee labels, SQL templates, NASA loader
and README numbers. Mutation check: `scripts/mutation_test.py` applied {{mu_total}} deliberate bugs
(leakage, wrong split, off-by-one lead time, wrong units, ...); the {{mu_tests}} tests caught
{{mu_caught}} of {{mu_total}}. Earlier runs were weaker and are kept in
`results/mutation_report.json`: the first caught {{mu_prev_caught}} of {{mu_prev_total}} (missed:
"{{mu_prev_missed}}"), the second {{mu_prev2_caught}} of {{mu_prev2_total}} (missed: "{{mu_prev2_missed_a}}"
and "{{mu_prev2_missed_b}}"). Both times the causality tests only looked at thresholded alarm flags,
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
