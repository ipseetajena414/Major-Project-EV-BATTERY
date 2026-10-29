# Member 2 — RUL validation and generalization audit (NASA NMC)

Date: 2026-10-29
Branch: `member2` (no branch change, no stage, no commit, no push)
Scope: isolated evaluation only. No production artifact was modified, retrained, or promoted.

---

## 0. Bottom line

The shipped NMC RUL model is **not validated** and **must not be treated as a production
RUL predictor**. The existing production RUL test split is degenerate: all 97 test rows are
post-threshold with a zero-variance target, so the previously reported test MAE of 0.0588
measures "predict 0 after the cell is already below 80% SOH", not remaining-life forecasting.

A leakage-safe leave-one-cell-out (LOCO) re-evaluation does show real signal. A LightGBM
retrained per fold on pre-threshold cycles reaches pooled pre-threshold MAE **6.78 cycles**
(RMSE 9.12, R² 0.776) versus **13.75 cycles** for the best simple SOH-extrapolation baseline
and **17.06 cycles** for a training-only constant. That is a substantial gain, and it exactly
reproduces the earlier diagnostic number (6.782), so the prior result was real.

However, the same evaluation shows that the advantage does **not** hold where it matters most.
Within the last 10 cycles before the threshold, a simple causal median-fade extrapolation
beats LightGBM (MAE 1.33 vs 2.54), and LightGBM systematically over-predicts (bias +1.31).
Retrained models also keep predicting positive remaining life after the threshold (up to
23 cycles on held-out B0018). Changing only the crossing convention from "first dip" to
"sustained 3 cycles" raises pooled MAE from 6.78 to 9.65 (+42%). Three cells cannot support a
generalization claim, and the 80% threshold is a project convention, not a physical EOL
measurement.

**Decision: not credible enough for promotion or candidate export. Continued RUL research is
justified only after the data and split deficiencies below are fixed; no candidate model should
be developed for deployment from the current three-cell dataset.**

---

## 1. What was inspected (read-only)

| Artifact | Role | SHA-256 (first 16) |
|---|---|---|
| `master_table.parquet` | source telemetry / capacity labels | `6fa7475d18100078` |
| `member2_modeling/build_and_train.py` | target construction, split, training | `b56ce13f7eab8a42` |
| `member2_modeling/feature_schema.json` | 37-feature RUL schema, `model_status` | `989cf8eb9a199f4b` |
| `member2_modeling/model_manifest.json` | status/caveat text, prior test metrics | `bfdf2d4a8a8ec552` |
| `member2_modeling/nmc_rul_lgbm_model.joblib` | shipped model (500 trees, best_iter 180) | `206009c8b8929bbf` |
| `member2_modeling/predictor.py` | `estimate_rul` API, clips to ≥0 | `ef12025e1c9daf01` |

The evaluation script re-hashes all six files before and after every run and aborts if any
byte changes; the run reports `protected_artifacts_unchanged: true`.

Prior claims in `model_manifest.json` / `experiments/results/rul_pre_eol_results.csv`
(pooled pre-threshold LOCO MAE 6.782, n=180) are reproduced exactly by this audit
(`lgbm_pre_eol_fixed200`). The manifest itself already flags the production RUL test as
"All 97 test targets are post-EOL (RUL=0)"; this audit confirms and quantifies that.

---

## 2. Target definition and audit

**Operational threshold:** SOH < 80.0% (strict). This is a **project-defined maintenance
threshold**, not a validated manufacturer or experimental end-of-life definition.

**Primary label:** `crossing_pos` = first observed cycle position with SOH < 80.0%.
`RUL(pos) = crossing_pos − pos` for `pos < crossing_pos`, and 0 at/after the crossing.
Units are **observed cycles**. For these three cells cycle IDs are contiguous (verified:
B0005/B0006 1–168, B0018 1–132), so observed cycles equal physical cycle steps; this is a
property of this dataset, not a general guarantee.

**Sensitivity label:** `sustained3` = first cycle starting a run of 3 consecutive
sub-threshold cycles, falling back to the first dip if no such run exists.

| cell | cycles | SOH first | SOH min | first cross (cycle/pos/SOH) | sustained-3 cross | pre rows | post rows | cycles above 80% after first cross | below runs | last below |
|---|---|---|---|---|---|---|---|---|---|---|
| B0005 | 168 (1–168) | 92.82 | 64.37 | 75 / 74 / 79.52 | 75 | 74 | 94 | 1 | 2 | 168 |
| B0006 | 168 (1–168) | 101.77 | 57.69 | 63 / 62 / 79.95 | 63 | 62 | 106 | 0 | 1 | 168 |
| B0018 | 132 (1–132) | 92.75 | 67.05 | 45 / 44 / 79.77 | 59 | 44 | 88 | 13 | 2 | 132 |

Observations:

- **No cell starts below threshold** and **no crossing is unestablishable**; all three cross.
- **SOH is non-monotonic.** B0018 dips below 80% at cycle 45, returns above 80% for 13 later
  cycles, and only stays below from cycle 59. Under the primary (first-dip) convention those
  13 recovered cycles are labeled RUL=0 while their measured SOH is still ≥80% — a known
  label/target inconsistency, quantified by the sensitivity run. B0005 has one rebound cycle
  after its first dip; B0006 has none.
- **B0006 exceeds nominal capacity** (SOH max 101.77, i.e., 2.035 Ah vs 2.0 Ah nominal),
  indicating capacity-measurement/calibration noise in early cycles. B0005 starts already aged
  at 92.8%. "Beginning of life" is therefore not consistent across cells.
- All trajectories extend far below the threshold (88–106 post rows), but there is **no
  independent EOL ground truth**: the cells were not cycled to a manufacturer-defined failure
  criterion. "EOL" here means only "crossed 80% SOH".

Primary pre-threshold population: **n=180 rows** (74/62/44), RUL mean 31.8, median 30.5,
range 1–74. These are 180 rows but only **3 independent trajectories**; the effective sample
size for generalization is 3.

---

## 3. Production split audit

Production uses a per-cell chronological 60/20/20 split and trains RUL on **all** rows
(pre- and post-threshold), with median imputation from train, 500 trees, early stopping (30).
Because each crossing occurs in the first 36–45% of the trajectory, the last 40% of every cell
is entirely post-threshold:

| cell | train (n / pre / post) | val (n / pre / post) | test (n / pre / post) | test cycle range |
|---|---|---|---|---|
| B0005 | 100 / 74 / 26 | 33 / 0 / 33 | 35 / 0 / 35 | 134–168 |
| B0006 | 100 / 62 / 38 | 33 / 0 / 33 | 35 / 0 / 35 | 134–168 |
| B0018 | 79 / 44 / 35 | 26 / 0 / 26 | 27 / 0 / 27 | 106–132 |

**Test = 97 rows, 0 pre-threshold, 97 post-threshold, target min=max=0.** The reported
production test MAE (0.0588) is therefore an evaluation of zero-target recognition on rows the
model has effectively already seen the end of, not of RUL forecasting. Validation is degenerate
for the same reason. Only the training split contains any pre-threshold rows (180), and those
are exactly the rows the shipped model was fitted on.

Consequences:

1. The production RUL test cannot detect a broken RUL model.
2. Any model trained with those zero rows receives 40% of its training labels at RUL=0.
3. The schema/manifest `model_status: "unvalidated_pre_eol"` is accurate and should remain.

---

## 4. Validation protocol

- **Leave-one-cell-out.** Each fold holds out one entire cell; the held-out cell contributes
  no fitting rows, no imputation statistics, no early-stopping rows, and no feature selection.
  Folds are disjoint; each row is predicted exactly once. No random splits, no shuffling.
- **Primary metric population:** pre-threshold rows only (RUL > 0). Post-threshold rows are
  scored and reported separately and are never pooled into the primary score.
- **No tuning on test.** All model configurations are fixed and documented:
  - `lgbm_pre_eol_fixed200`: exact prior diagnostic configuration (200 trees, lr 0.05,
    15 leaves, depth 5, min_child 10, subsample/colsample 0.8, L1/L2 0.1, seed 42), trained
    only on pre-threshold rows of the two training cells.
  - `lgbm_production_protocol`: production configuration (500 trees, lr 0.05, 31 leaves,
    depth 6, min_child 5, subsample/colsample 0.8, L1/L2 0.1, seed 42) with per-cell 60%
    fitting and 20% early stopping on the training cells only; median imputer fit on those
    fitting rows only.
  - A third reference point is the **shipped artifact**, evaluated only in-sample and labeled
    as such.
- **Baselines** (all use training cells only, except the causal ones which use the held-out
  cell's own past SOH history):
  - `train_median_constant`: median pre-threshold RUL of the training cells (no features).
  - `soh_extrapolation_global`: `(SOH_now − 80) / (−slope)`, slope = median per-cell OLS SOH
    slope over training pre-threshold rows.
  - `soh_extrapolation_causal`: per-row OLS slope over that cell's SOH history up to now.
  - `soh_extrapolation_causal_median`: per-row median one-cycle SOH delta over its history.
  - `last_fade_persistence`: one-cycle fade persistence (SOH last delta extrapolated).
- **Metrics:** MAE/RMSE in cycles, R², bias, median AE, max AE, MAPE, near-EOL subset (RUL≤10),
  RUL buckets (1–5, 6–10, 11–30, 31+), post-threshold metrics, and counts of negative raw
  predictions. Predictions are clipped at 0 for scoring; raw negatives are counted separately.
- **Determinism:** fits are seeded and reproduce exactly; tests assert bit-identical repeated
  predictions.

---

## 5. Results — primary target (first crossing, pre-threshold only)

### 5.1 Pooled over the three held-out cells (n = 180)

| model | MAE | RMSE | R² | bias | median AE | max AE |
|---|---|---|---|---|---|---|
| **lgbm_pre_eol_fixed200** | **6.78** | **9.12** | **0.776** | −3.44 | 5.27 | 27.65 |
| lgbm_production_protocol | 7.98 | 10.23 | 0.718 | −5.24 | 6.40 | 27.83 |
| soh_extrapolation_global | 13.75 | 18.12 | 0.117 | +5.10 | 9.91 | 43.59 |
| train_median_constant | 17.06 | 20.46 | −0.126 | −1.90 | 16.00 | 47.00 |
| last_fade_persistence | 38.63 | 137.00 | −49.5 | +20.40 | 11.16 | 1333.72 |
| soh_extrapolation_causal | 42.26 | 85.52 | −18.7 | +37.75 | 13.18 | 379.31 |
| soh_extrapolation_causal_median | 78.83 | 160.66 | −68.4 | +69.92 | 8.35 | 595.86 |

All models produced **0 negative raw predictions** on pre-threshold rows. The unstable causal
baselines are dominated by tiny/mixed one-cycle deltas producing very large RUL estimates; they
are included because they are the closest analogue to "persistence", and their instability is
itself informative. The stable simple comparison is the global training-slope extrapolation
(13.75).

### 5.2 Per-cell MAE (cycles)

| held-out cell | LGBM prior cfg | LGBM production cfg | global SOH extrapolation | median constant | best simple baseline |
|---|---|---|---|---|---|
| B0005 (n=74) | 9.94 | 10.58 | 10.73 | 19.99 | 10.73 (global) |
| B0006 (n=62) | 5.11 | 7.70 | 22.68 | 15.53 | 11.78 (causal median) |
| B0018 (n=44) | 3.83 | 3.99 | 6.25 | 14.27 | 4.74 (causal median) |

LightGBM beats every simple baseline on every cell, but the margin on B0005 is only 7%.
B0005 is the slowest-fading cell (crossing at 75 of 168), so training on B0006+B0018
under-estimates its remaining life: mean prediction 27.9 cycles vs mean true RUL 37.5.
Prediction range is 3.8–52.6 vs a true range of 1–74 (compression toward mid-life).

### 5.3 Near threshold (RUL ≤ 10; 30 rows, 10 per cell)

| model | MAE | RMSE | bias |
|---|---|---|---|
| soh_extrapolation_causal_median | **1.33** | **2.05** | −0.58 |
| last_fade_persistence | 2.12 | 3.19 | −0.87 |
| lgbm_pre_eol_fixed200 | 2.54 | 3.26 | **+1.31** |
| lgbm_production_protocol | 3.20 | 4.05 | −2.30 |
| soh_extrapolation_global | 3.68 | 5.63 | +2.74 |
| train_median_constant | 25.00 | 25.35 | +25.00 |

The LightGBM advantage disappears in the final 10 cycles, where accurate RUL matters most: the
causal median-fade baseline is roughly twice as accurate and unbiased, while LightGBM
over-predicts remaining life by ~1.3 cycles on average (worst per-cell error 7.1 cycles).

### 5.4 Error by true-RUL bucket (MAE, cycles)

| bucket | LGBM prior cfg | LGBM production cfg | global extrapolation | causal median | constant |
|---|---|---|---|---|---|
| 1–5 (n=15) | 3.25 | 1.36 | 2.92 | 0.89 | 27.50 |
| 6–10 (n=15) | 1.84 | 5.04 | 4.43 | 1.78 | 22.50 |
| 11–30 (n=60) | 5.10 | 8.24 | 8.82 | 12.64 | 10.20 |
| 31+ (n=90) | 9.32 | 9.40 | 20.39 | 148.79 | 18.98 |

The far-from-threshold improvement (31+ bucket: 9.3 vs 20.4) is genuine, but it is achieved
with under-prediction (pooled bias −3.4) and the near-threshold buckets are not better than a
simple extrapolation. On long-horizon rows (31+, mean true RUL ≈ 48), 9.3 cycles MAE is ~19%
relative error.

### 5.5 Post-threshold rows, reported separately (RUL = 0)

| held-out cell | LGBM prior cfg MAE / max pred | LGBM production cfg MAE / max pred | global extrapolation MAE / max pred |
|---|---|---|---|
| B0005 (94) | 3.71 / 5.46 | 0.32 / 2.74 | 0.01 / 0.97 |
| B0006 (106) | 4.05 / 4.94 | 3.23 / 3.52 | 0.00 / 0.00 |
| B0018 (88) | 9.86 / 21.40 | 1.62 / 23.08 | 1.85 / 27.15 |

A model that has crossed the threshold should report 0. The pre-threshold-only LightGBM still
assigns positive life to many post-threshold rows (B0018 mean 9.9, max 21.4), because its
features — including current SOH — still look like early-life cycles, especially after a
rebound. This is an invalid prediction shape, not a negative-value error; it is a practical
failure mode for a maintenance alarm.

### 5.6 Target-definition sensitivity (`sustained3`, n = 194)

| model | MAE | RMSE | R² | bias |
|---|---|---|---|---|
| lgbm_pre_eol_fixed200 | 9.65 | 11.22 | 0.662 | +0.27 |
| lgbm_production_protocol | 12.03 | 13.81 | 0.488 | +0.71 |
| soh_extrapolation_global | 16.76 | 23.87 | −0.531 | +7.63 |
| train_median_constant | 16.75 | 19.69 | −0.041 | −0.54 |

Adding the 14 recovered B0018 cycles raises LGBM MAE by 42% and changes per-cell bias
dramatically (B0005 +0.37, B0006 +9.09, B0018 −9.27). The ranking of models is preserved, but
the absolute error is highly sensitive to a defensible change in the EOL convention. No single
"RUL accuracy" number should be quoted without the convention.

### 5.7 Shipped artifact (in-sample only, not validation)

| cell | pre n | pre MAE | post mean prediction |
|---|---|---|---|
| B0005 | 74 | 0.082 | 0.010 |
| B0006 | 62 | 0.086 | 0.033 |
| B0018 | 44 | 0.077 | 0.059 |

Overall in-sample pre-threshold MAE 0.082 cycles. The shipped model was trained on all three
cells, so this near-perfect fit is **memorization, not evidence**. Note also that its post-
threshold outputs are ≈0 while the LOCO-retrained models' are not — the shipped model learned
the 0-label from these same cells.

---

## 6. Leakage and limitation audit

| Check | Finding |
|---|---|
| Cell boundaries | LOCO folds are cell-disjoint; held-out cell excluded from fitting, imputation, early stopping. Verified by tests. |
| Target timing | RUL is the distance to the first crossing position computed separately from features; `rul_cycles`, `cycle_id`, `cycle_position`, `crossing_position` are not features. Test-verified target shuffling leaves predictions unchanged. |
| Repeated cycle-level labels | 180 pre rows are highly autocorrelated within 3 trajectories. Treating them as 180 independent samples would overstate evidence; metrics are descriptive for these cells only. |
| Feature provenance | 37 features are within-cycle discharge aggregates (voltage/current/energy/Ah/dvdt/time/temperature) plus **current-cycle** SOH. No cycle index, age, cumulative throughput, future SOH, or target-derived quantity. All 37 populated for NMC; no fully-null feature. |
| Prediction-time availability | Every feature is measurable from the current discharge cycle; SOH requires the current cycle's measured capacity. Temperature features are populated (~23–42 °C). No future data required. |
| Preprocessing isolation | Median imputation (production protocol) fit on training-cell fitting rows only; no scaling. Batch-independence test-verified. |
| Test-driven tuning | None. Configurations were fixed before evaluation and not altered after seeing results. No model selection on held-out metrics. |
| Threshold vs physical EOL | 80% SOH is a project convention. No manufacturer limit, no measured failure, no validated EOL event. Results must not be described as predicting physical end of life. |
| Post-threshold treatment | Scored separately; excluded from primary metrics. Positive post-threshold predictions are reported as a failure mode, not hidden by clipping. |
| Negative predictions | Zero raw negatives on pre-threshold rows for all models; clipping at 0 does not affect the primary scores. |
| Determinism | Fixed seeds; repeated fits produce identical predictions (test). |
| Artifact integrity | Protected files hashed before/after each run; unchanged. |

### Does three cells / available crossings suffice?

No. Three LOCO folds with two training cells each cannot separate cell-to-cell variability
from model error, cannot estimate uncertainty, and cannot support a claim of generalization to
new cells, protocols, temperatures, or chemistries. The per-cell MAE spread (3.8–9.9 cycles)
is as large as the pooled gain over baselines. All cells come from the same NASA NMC 18650
family; fade rates differ by ~2× (B0005 crosses at cycle 75, B0018 at 45). The 80% threshold
is inherited, not validated for this chemistry.

---

## 7. Decision

**RUL prediction is not credible enough for promotion, export, or production use.**

What is supported:

- There is real cross-cell predictive signal in the 37-feature LightGBM: pooled pre-threshold
  MAE 6.78 cycles, R² 0.776, beating the best stable simple baseline by ~51% and the
  training-only constant by ~60%, consistently across all three held-out cells.
- The earlier 6.782 diagnostic is reproducible; it was not a reporting error.

What blocks a candidate:

1. The production RUL test is 100% post-threshold and cannot measure forecasting.
2. Near threshold (RUL≤10) a simple causal median-fade extrapolation is better (1.33 vs 2.54
   MAE) and LightGBM is biased high (+1.31); the model is weakest exactly where RUL matters.
3. Retrained models report positive remaining life after the threshold (up to 21–23 cycles).
4. Errors are convention-sensitive: +42% MAE when the crossing rule is strengthened.
5. Only three trajectories, one chemistry, no physical EOL ground truth, no uncertainty
   estimates, no calibration.

**Further development is justified only as research on an expanded dataset, not as candidate
development for deployment.** The next candidate should not be built until the split and data
deficiencies are corrected.

### Remains unvalidated

- Generalization to new cells, protocols, temperatures, and chemistries.
- Any relationship to physical/experimental EOL (only the 80% SOH convention is tested).
- Uncertainty/calibration of RUL predictions.
- Behavior between threshold crossing and true failure.
- Sensitivity to SOH measurement noise (B0006 SOH >100%, B0005 starting at 92.8%).
- Robustness of near-threshold predictions; current evidence favors simple extrapolation there.
- Whether features add information beyond current SOH (no ablation was run here).

### Required data / changes before re-evaluation

1. **More trajectories:** at least 8–10 independent cells spanning several fade rates and
   protocols (ideally temperatures), so LOCO has ≥7 training cells per fold and per-cell error
   spread can be estimated.
2. **Validated EOL measurement:** a documented manufacturer/experimental failure criterion, or
   cycling continued past the threshold to a measured EOL, instead of an assumed 80% SOH.
3. **A pre-threshold test protocol:** hold out cells or chronological windows that contain
   pre-threshold cycles (e.g., leave-cell-out, or test up to the crossing), replacing the
   current all-post-threshold test split.
4. **Pre-registered target convention** (first dip vs sustained crossing) and threshold, with
   sensitivity reported.
5. **Uncertainty quantification** (per-prediction intervals or at least fold-level spread) and
   near-threshold-focused metrics as the primary decision metric.
6. **Feature ablation** against a current-SOH-only baseline to confirm the telemetry features
   add value beyond SOH.

---

## 8. Deliverables and reproduction

Evaluation code: `member2_modeling/experiments/rul_validation.py`
Tests: `member2_modeling/experiments/test_rul_validation.py`
Results directory: `member2_modeling/experiments/results/rul/`

| File | Contents |
|---|---|
| `nmc_rul_target_audit.csv` | per-cell target/crossing audit |
| `nmc_rul_production_split_audit.json` | reproduced 60/20/20 split, test composition |
| `nmc_rul_primary_pooled.csv` | pooled pre-threshold metrics, first-crossing target |
| `nmc_rul_primary_by_cell.csv` | per-held-out-cell metrics |
| `nmc_rul_primary_near_eol.csv` | RUL≤10 metrics (pooled and per cell) |
| `nmc_rul_primary_error_buckets.csv` | error by true-RUL bucket |
| `nmc_rul_primary_predictions.csv` | row-level predictions/errors (all 468 rows × 7 models) |
| `nmc_rul_post_threshold_metrics.csv` | post-threshold scores, pooled per fold |
| `nmc_rul_post_threshold_by_cell.csv` | post-threshold scores per cell |
| `nmc_rul_sensitivity_sustained3_*.csv` | sustained-3 convention (pooled, by cell, near-EOL, buckets, predictions) |
| `nmc_rul_model_metadata.json` | fold training sizes, early-stopping iterations |
| `nmc_rul_shipped_model_diagnostic.csv` | in-sample diagnostic of the shipped artifact |
| `nmc_rul_summary.json` | full summary, hashes, caveats |
| `MEMBER2_RUL_VALIDATION_REPORT.md` | this report |

Reproduce:

```bash
.venv/Scripts/python.exe member2_modeling/experiments/rul_validation.py
.venv/Scripts/python.exe -m unittest member2_modeling.experiments.test_rul_validation -v
```

Actual test result: **28 tests, OK** (target construction, transient rebounds, gapped cycle
ids, LOCO separation, feature exclusions, target-shuffle invariance, causal prefix invariance,
train-only preprocessing, batch independence, determinism, negative-prediction accounting,
real-data audit, no-write run).

No file outside `member2_modeling/experiments/` was created or modified. No git operation was
performed. The six protected artifacts were verified byte-identical before and after the run.
