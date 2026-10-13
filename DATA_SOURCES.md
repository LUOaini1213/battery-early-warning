# Data sources, licences and provenance

Raw data is never stored in this repository. It lives under `BEW_DATA_ROOT` (default `D:/data`).
SHA-256 hashes of every raw file are written to `results/provenance.json` by `run_all.py`.
The only data committed here is (a) a small **synthetic** fixture under `tests/fixtures/` and
(b) derived results (aggregate metrics, per-cell summary tables and plots) under `results/`.

## 1. Severson et al. 2019 fast-charging dataset (primary)

- **Project page:** https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
  ("Data-driven prediction of battery cycle life before capacity degradation"), Toyota Research
  Institute experimental data platform. Read on 2026-10-13 via the platform API
  (`/1/api/v1/edp/projects/5c48dd2bc625d700019f3204`, `/1/api/v1/edp/configuration`).
- **Licence:** the platform states, site-wide, "All data is released under CC BY 4"; its
  configuration links the licence to https://creativecommons.org/licenses/by/4.0/ (Creative
  Commons Attribution 4.0 International). The licence allows sharing and adaptation, including
  derived results, provided attribution is given and changes are indicated. Nothing on the project
  page restricts the analysis done here.
- **Citation requested on the project page:** "If using this dataset in a publication, please cite:
  Severson et al. Data-driven prediction of battery cycle life before capacity degradation.
  Nature Energy volume 4, pages 383–391 (2019)." (doi:10.1038/s41560-019-0356-8)
- **What was downloaded:** the three per-batch MATLAB structs (the smallest official form that has
  both per-cycle summaries and the interpolated discharge curves; the alternative "BEEP structured
  data" zip is 27.9 GB). Downloaded from `https://data.matr.io/1/api/v1/file/<file_id>/download`
  on 2026-10-13 to `D:\data\severson2019\raw\`:

  | file | file_id | bytes | SHA-256 |
  |---|---|---|---|
  | 2017-05-12_batchdata_updated_struct_errorcorrect.mat | 5c86c0b5fa2ede00015ddf66 | 3,025,320,241 | 9d928ab978f0e3c70b31cb833a749fedd35094d01af76475d69b40aa3497f5ba |
  | 2017-06-30_batchdata_updated_struct_errorcorrect.mat | 5c86bf13fa2ede00015ddd82 | 2,007,331,155 | 63ab200d09ecb237fee5ef3a5c5db76e3212e3206a0bd92f769e1427fed338b8 |
  | 2018-04-12_batchdata_updated_struct_errorcorrect.mat | 5c86bd64fa2ede00015ddbb2 | 3,236,690,412 | 62c30e413b63e6144720e016deed3661fac8468641794a5807b123fe84717998 |

- **Changes made (as CC BY requires us to indicate):** cells excluded and the five batch-1 cells
  continued in batch 2 merged exactly as in the official starter code
  (github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation,
  `Load Data.ipynb`); discharge capacities outside (0, 1.32] Ah and non-positive IR / temperature /
  charge-time placeholders are masked (`bew/cleaning.py`). Only per-cycle summaries and the
  interpolated discharge curves of cycles 1–100 are extracted.
- **Derived tables** (`cells.parquet`, `cycles.parquet`, `qdlin_first100.npz`) are written next to
  the raw files in `D:\data\severson2019\derived\` and are not committed.

## 2. NASA PCoE Battery Data Set (secondary, mechanism illustration only)

- **Repository page:** https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
  (entry 5, "Battery"); download link given there:
  https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip (archive member
  `1. BatteryAgingARC-FY08Q4.zip`).
- **Terms:** no dataset-specific licence is stated. The repository asks users to acknowledge the
  repository and the data donors, and states that users employ the data at their own risk.
  The raw files are therefore not redistributed here; only aggregate results are committed.
- **Citation:** B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics Data Repository,
  NASA Ames Research Center, Moffett Field, CA.
- **Files used:** B0005.mat, B0006.mat, B0007.mat, B0018.mat, copied (read-only source) from an
  earlier local download of the same archive to `D:\data\nasa_pcoe\`. Hashes (identical to the
  hashes recorded at that download):

  | file | bytes | SHA-256 |
  |---|---|---|
  | B0005.mat | 15,956,874 | 0eae4585baf3f200c09fe24c5ab884f1889679fc75206ca1aa19da704104f0b0 |
  | B0006.mat | 16,028,159 | fa818ab4db5db8ab21e910b6dd6c3e20d3761bb9672089e1a4de8f96074616c5 |
  | B0007.mat | 16,049,460 | d022afa086efaf54ab5b63f05220f5be178c8027e2fa8d68589db0e85a441a3b |
  | B0018.mat | 8,502,218 | d1e6c923a43ea1c9666b3a90bbb521757a067fd60b4d17dbfaa49c50b179da69 |

## 3. Literature used for hypotheses and methods

All DOIs checked against Crossref on 2026-10-13.

- Severson et al., Nature Energy 4, 383–391 (2019), doi:10.1038/s41560-019-0356-8 — protocol, Tables 1–2.
- Attia et al., J. Electrochem. Soc. 169, 060517 (2022), doi:10.1149/1945-7111/ac6d13 — knees review.
- Waldmann, Hogg, Wohlfahrt-Mehrens, J. Power Sources 384, 107–124 (2018), doi:10.1016/j.jpowsour.2018.02.063 — Li plating review.
- Birkl et al., J. Power Sources 341, 373–386 (2017), doi:10.1016/j.jpowsour.2016.12.011 — degradation diagnostics.
- Vetter et al., J. Power Sources 147, 269–281 (2005), doi:10.1016/j.jpowsour.2005.01.006 — ageing mechanisms.
- Epding et al., J. Energy Storage 22, 249–256 (2019), doi:10.1016/j.est.2019.02.015 — capacity recovery after rests.
- Fermín-Cueto et al., Energy and AI 1, 100006 (2020), doi:10.1016/j.egyai.2020.100006 — knee point / knee onset.
- Page, Biometrika 41, 100–115 (1954), doi:10.1093/biomet/41.1-2.100 — CUSUM.
- Zou & Hastie, J. R. Stat. Soc. B 67, 301–320 (2005), doi:10.1111/j.1467-9868.2005.00503.x — elastic net.
