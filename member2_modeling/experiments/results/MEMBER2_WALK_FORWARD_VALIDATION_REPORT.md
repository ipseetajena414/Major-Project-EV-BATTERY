# Member 2 — Walk-Forward Validation of LFP SOH and NMC Next-Observed-Record Baselines

**Date:** 2026-10-29  
**Branch:** `member2`  
**Scope:** Experimental validation only. No production model, scaler, schema, manifest, parquet, pipeline, or inference API was changed.

## 1. Purpose and guardrails

This experiment tests whether the earlier single-split findings survive chronological walk-forward validation:

- the LFP unit-safe Ridge model beats persistence using only the existing 22 unit-safe features;
- the NMC mean-drift baseline is stable across folds and cells.

All models and baselines were fixed before execution. Test windows were never used for imputation, scaling, feature selection, alpha selection, early stopping, or model choice. The experiment writes only the files listed in Section 8.

## 2. Data and target definition

### NMC

- NASA Ames PCoE cells: `B0005`, `B0006`, `B0018`.
- 468 labelled cycles, 465 next-observed-record pairs, 3 cells.
- No missing cycle IDs and no null target SOH in the constructed pairs.
- Production feature set: 37 features.
- Current is verified in amperes and temperature is present.

### LFP

- One cell only: `B5` / `BATTERY_5` (`LFP_MultiStat_B5`).
- 250 observed cycle IDs in blocks `1-100`, `151-200`, and `401-500`.
- Cycle 40 has a null SOH and is removed. This leaves 249 labelled cycles and 248 supervised pairs.
- Missing cycle IDs: `101-150` and `201-400`. No missing cycle was fabricated or interpolated.
- Unit-safe feature set: 22 features. The seven current-derived features are removed: `ah_throughput`, `energy_wh`, `avg_power_w`, `i_mean`, `i_std`, `i_min`, and `i_max`. `current_soh_pct` and the voltage/time features remain.

### Target construction

For each cell, cycles are sorted by `cycle_id`. The target for pair `t` is the SOH at the next **observed labelled record**, `t+1`, not necessarily the next physical cycle ID:

```text
pair t:  features(cycle t)  ->  target_soh_pct(cycle t+1)
```

This distinction matters for LFP because three pairs cross gaps:

| Pair | Gap in cycle IDs | Current SOH | Target SOH | SOH change | Evaluation status |
|---|---:|---:|---:|---:|---|
| 39 -> 41 | 2 | 94.55091 | 94.25463 | -0.29628 | training only |
| 100 -> 151 | 51 | 93.69711 | 92.76357 | -0.93354 | training only |
| 200 -> 401 | 201 | 92.11158 | 92.21367 | +0.10209 | tested in fold 2 |

The `39 -> 41` pair is a gap in *labelled records* because cycle 40 exists but has null SOH. The other two gaps are absent cycle IDs. The `200 -> 401` target is slightly higher than its current SOH; this is a long-horizon regeneration observation, not adjacent-cycle drift.

## 3. Validation design

### 3.1 LFP: single-cell expanding-window walk-forward

- 248 LFP pairs, no shuffling.
- Initial training fraction: 40% = 99 pairs.
- Five expanding folds with cuts `[99, 129, 159, 188, 218, 248]`.
- Test windows: 30, 30, 29, 30, 30 pairs; pooled test n = 149.
- Each training window is all earlier pairs from the same cell; each test window is the next contiguous chronological block.
- The first 99 pairs are training-only, so early-life cycles are not evaluated out of sample.
- Ridge pipeline: median imputation and standardization fit inside each training window; alpha chosen from `[0.001, 0.01, 0.1, 1, 10, 100, 1000]` on a chronological inner validation split (last 20% of each cell's training window), then refit on the full outer training window.
- Baselines use only outer training data (except `last_delta`, which uses the current and previous observed SOH, both available at prediction time):
  - persistence: predict `current_soh_pct`;
  - pooled mean drift: add the training mean of `target_soh_pct - current_soh_pct`;
  - last delta: add the observed previous SOH change;
  - linear trend: per-cell OLS of target SOH on observed record position, fit on training;
  - train mean: predict the training target mean.

| Fold | Train pairs | Test pairs | Train current cycles | Test current cycles | Ridge alpha | Ridge MAE | Persistence MAE | Ridge RMSE | Ridge R² |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 99 | 30 | 1-100 | 151-180 | 1.0 | 0.0614 | 0.0814 | 0.0843 | 0.7882 |
| 2 | 129 | 30 | 1-180 | 181-200; 401-410 | 0.01 | 0.0801 | 0.1157 | 0.1017 | 0.9280 |
| 3 | 159 | 29 | 1-410 | 411-439 | 0.01 | 0.0928 | 0.1291 | 0.1100 | 0.7314 |
| 4 | 188 | 30 | 1-439 | 440-469 | 0.001 | 0.0945 | 0.1400 | 0.1102 | 0.8062 |
| 5 | 218 | 30 | 1-469 | 470-499 | 0.001 | 0.0831 | 0.1087 | 0.0967 | 0.8084 |

Fold 2 is the only fold whose test window contains a gap pair (`200 -> 401`). The reported `test current cycles` are current-cycle IDs; the fold's target cycle IDs extend one observed record further.

### 3.2 NMC: pooled per-cell expanding-window walk-forward

- 465 pairs across three cells, no shuffling.
- Initial training fraction: 50% of each cell's pairs.
- Five folds; each cell's training window expands within that cell and its test window is the next chronological block.
- Pooled test n = 234 of 465 pairs. Per-cell test counts are 84 (`B0005`), 84 (`B0006`), and 66 (`B0018`). The first half of each cell's trajectory is never evaluated out of sample.
- Test windows are disjoint, but later training windows contain earlier test records, so folds are not statistically independent.

| Fold | Train pairs | Test pairs | Persistence MAE | Pooled mean drift MAE | Cell mean drift MAE | Delta LGBM + EFC MAE | Ridge (37 features) MAE | LightGBM absolute MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 231 | 47 | 0.7872 | 0.6018 | 0.6204 | 0.6289 | 0.6908 | 1.3657 |
| 2 | 278 | 47 | 0.4906 | 0.3850 | 0.3967 | 0.4374 | 0.7322 | 2.6125 |
| 3 | 325 | 46 | 0.5523 | 0.4656 | 0.4741 | 0.4658 | 0.4810 | 1.5393 |
| 4 | 371 | 47 | 0.3809 | 0.3039 | 0.3056 | 0.3132 | 0.2455 | 1.8596 |
| 5 | 418 | 47 | 0.5275 | 0.4417 | 0.4352 | 0.4317 | 0.6027 | 1.7603 |
| Pooled | - | 234 | 0.5477 | 0.4395 | 0.4463 | 0.4554 | 0.5507 | 1.8287 |

`nmc_walk_forward_fold_metrics.csv` contains the full fold-level table: n, MAE, RMSE, R², MAPE, persistence MAE on the same rows, and the MAE delta for all five folds and eight models. `nmc_walk_forward_cell_metrics.csv` and `nmc_walk_forward_pooled_metrics.csv` contain the per-cell versions.

Fold-level RMSE for the key models:

| Fold | Persistence | Pooled mean drift | Cell mean drift | Delta LGBM + EFC |
|---:|---:|---:|---:|---:|
| 1 | 1.4777 | 1.4662 | 1.4695 | 1.4698 |
| 2 | 0.6793 | 0.6601 | 0.6621 | 0.6585 |
| 3 | 0.9529 | 0.9809 | 0.9824 | 0.9823 |
| 4 | 0.5461 | 0.5095 | 0.5113 | 0.5012 |
| 5 | 0.8054 | 0.7889 | 0.7859 | 0.7682 |
| Pooled | 0.9487 | 0.9411 | 0.9424 | 0.9379 |

Models:

- persistence;
- pooled mean drift: one training-fold drift, shared across cells;
- cell mean drift: per-cell training drift with a pooled fallback;
- linear trend;
- Ridge with the 37 production features;
- existing LightGBM configuration with an absolute target;
- experimental delta-target LightGBM (`target - current_soh_pct`, then add current SOH back);
- experimental delta-target LightGBM plus `efc_cumulative` (38 features).

Ridge preprocessing was fit only inside each training window. LightGBM early stopping used the last 20% of each cell's outer training window; the final model was refit on the full outer training window. LightGBM used fixed seed 42, `deterministic=True`, `force_row_wise=True`, and `n_jobs=1`.

Target ranges confirm why absolute tree models are at a disadvantage: in every NMC fold, the test target minimum is below the training target minimum (for example, fold 1: training 73.38-101.26, test 71.30-80.29; fold 5: training 62.40-101.26, test 57.69-71.34). A tree model cannot extrapolate below the lowest target it has seen.

## 4. LFP results

### 4.1 Pooled metrics

| Model | n | MAE | RMSE | R² | MAE - persistence |
|---|---:|---:|---:|---:|---:|
| persistence | 149 | 0.114881 | 0.133010 | 0.947319 | 0 |
| pooled mean drift | 149 | 0.110527 | 0.133231 | 0.947143 | -0.004354 |
| last delta | 149 | 0.096426 | 0.148979 | 0.933909 | -0.018456 |
| linear trend | 149 | 0.553252 | 0.681650 | -0.383604 | +0.438371 |
| train mean | 149 | 1.743333 | 1.778084 | -8.414425 | +1.628452 |
| **unit-safe Ridge** | **149** | **0.082309** | **0.100977** | **0.969638** | **-0.032572** |

Unit-safe Ridge improves MAE by 0.0326 (28.3% relative to persistence) and RMSE by 0.0320 (24.1%). It beats persistence in **5 of 5 folds**. Fold-level MAE improvement ranges from 0.0200 to 0.0455. Fold 3 is the one fold where `last_delta` has slightly lower MAE than Ridge (0.0907 vs 0.0928), but Ridge remains better than both persistence and pooled mean drift.

### 4.2 Gap sensitivity

The single tested gap pair is unusually favorable to Ridge: absolute error 0.0049, versus 0.1021 for persistence. Removing that pair changes the pooled test set to n = 148 but does not change the conclusion:

- Ridge adjacent-only MAE: 0.082832; RMSE: 0.101317.
- Persistence adjacent-only MAE: 0.114967; RMSE: 0.133194.
- Ridge beats persistence in all five folds without the gap pair.

The two largest test blocks also both favor Ridge:

| Test current-cycle block | Ridge MAE | Persistence MAE |
|---|---:|---:|
| 151-200 (n = 50) | 0.0683 | 0.0929 |
| 401-500 (n = 99) | 0.0894 | 0.1260 |

These block metrics are derived from `lfp_walk_forward_predictions.csv` by grouping on the current `cycle_id`; the canonical fold and pooled metrics are in `lfp_walk_forward_metrics.csv`.

### 4.3 Interpretation

Within B5, the unit-safe Ridge result is consistent across folds, across both evaluated cycle blocks, and after removing the gap pair. It is not a single-fold or single-record artifact.

It is still **one cell**. The validation establishes temporal robustness on B5, not generalization to another LFP cell, another chemistry, another current-unit convention, or another dataset. The model also never predicts the first 99 pairs out of sample. These limitations are material.

## 5. NMC results

### 5.1 Pooled metrics

All models use the same 234 evaluation rows.

| Model | MAE | RMSE | R² | MAE - persistence |
|---|---:|---:|---:|---:|
| persistence | 0.547671 | 0.948671 | 0.953586 | 0 |
| **pooled mean drift** | **0.439482** | **0.941056** | **0.954328** | **-0.108189** |
| cell mean drift | 0.446265 | 0.942394 | 0.954198 | -0.101406 |
| delta LightGBM + EFC | 0.455372 | 0.937915 | 0.954632 | -0.092299 |
| delta LightGBM (production features) | 0.471433 | 0.941709 | 0.954265 | -0.076238 |
| Ridge (37 production features) | 0.550714 | 0.934076 | 0.955003 | +0.003043 |
| LightGBM absolute (production configuration) | 1.828697 | 2.287053 | 0.730244 | +1.281026 |
| linear trend | 2.330639 | 2.809502 | 0.592922 | +1.782968 |

Pooled mean drift reduces MAE by 0.1082 (19.8% relative) but reduces RMSE by only 0.0076 (0.8%); its R² gain is 0.0007. The MAE improvement is therefore clearer than the squared-error improvement.

### 5.2 Consistency across folds and cells

**MAE improvement is consistent:**

- Pooled mean drift beats persistence in **5 of 5 folds**; fold gains are 0.1853, 0.1056, 0.0866, 0.0771, and 0.0858.
- It also beats persistence in **15 of 15 fold-cell units**; the smallest gain is still positive (0.0384, fold 4 / `B0005`).
- Cell mean drift also beats persistence in **15 of 15 fold-cell units**.
- Delta LightGBM + EFC beats persistence in **15 of 15 fold-cell units**; delta LightGBM with production features wins 14 of 15.

**RMSE is mixed:**

- Pooled mean drift has worse RMSE than persistence in **8 of 15 fold-cell units**, although its pooled RMSE is slightly better (0.9411 vs 0.9487).
- At the fold level, RMSE improves in folds 1, 2, 4, and 5 and worsens in fold 3 (0.9809 vs 0.9529).
- A constant negative drift helps absolute error when the next observed SOH is lower, but it can increase squared error where the local drift is near zero or positive. MAE and RMSE are therefore telling slightly different stories.

**Per-cell pooled results (MAE / RMSE):**

| Cell | n | Persistence | Pooled mean drift | Cell mean drift | Delta LGBM + EFC |
|---|---:|---:|---:|---:|---:|
| `B0005` | 84 | 0.4235 / 0.7107 | 0.3363 / 0.7041 | 0.3565 / 0.6995 | 0.3316 / 0.6914 |
| `B0006` | 84 | 0.5866 / 1.0666 | 0.4691 / 1.0565 | 0.4656 / 1.0622 | 0.4880 / 1.0520 |
| `B0018` | 66 | 0.6561 / 1.0489 | 0.5331 / 1.0432 | 0.5358 / 1.0440 | 0.5714 / 1.0498 |

Pooled mean drift improves both MAE and RMSE for every cell when results are pooled across folds. The per-cell result is consistent; the per-fold-cell RMSE caveat remains.

### 5.3 Model comparisons

- **Pooled vs cell mean drift:** pooled drift has lower pooled MAE (0.4395 vs 0.4463) and lower pooled RMSE (0.9411 vs 0.9424). With only three cells, the shared drift is the simpler and slightly better baseline. Cell drift does not reveal a clear cell-specific advantage.
- **Delta LightGBM variants:** both beat persistence in MAE, but neither consistently beats pooled mean drift. Overall, delta + EFC has MAE 0.4554 versus 0.4395 for drift; it is better in fold 5 and for `B0005`, roughly tied in fold 3, and worse in the other folds/cells. Its RMSE is slightly better (0.9379 vs 0.9411). This is not a demonstrated improvement over the simple baseline.
- **Absolute production LightGBM:** MAE 1.8287, far worse than persistence. The model cannot follow SOH below its training target range. Delta targeting removes much of that failure, but early stopping in folds 2-5 selected only 1-3 iterations for the delta models, indicating a weak or unstable signal at this data size.
- **Ridge:** pooled MAE 0.5507 is essentially the same as persistence (0.5477), despite the lowest RMSE (0.9341). It wins only 3 of 5 folds and 8 of 15 fold-cell units against persistence. It is not a stable MAE improvement.
- **Linear trend:** poor; extrapolating a per-cell linear trend over the test horizon is not supported by this data.

### 5.4 Interpretation

The mean-drift MAE improvement is **stable across the available folds and cells**, not driven by one trajectory. Every one of the 15 fold-cell MAE comparisons is positive, and the improvement appears in all three cells.

The result is not uniform in squared-error terms, and the baseline is extremely simple. It is evidence that a constant one-step drift is a strong NMC baseline, not evidence that the feature-based models have learned useful degradation dynamics. The production LightGBM and Ridge results do not support replacing the baseline with a learned model on this evidence.

## 6. Leakage, reproducibility, and evaluation-design checks

### 6.1 Checks performed

- **Target timing:** the target is `shift(-1)` within each cell after sorting by `cycle_id`; it is explicitly labeled as the next observed labelled record, not the next physical cycle. Tests cover a synthetic gap (`[1, 2, 4, 5]`) and a `39 -> 41` case.
- **Cell boundaries:** pairs are constructed independently per cell; no pair crosses cells. NMC has no missing IDs; LFP has the three documented gaps.
- **Fold boundaries:** expanding folds are chronological and use only earlier pairs for training; tests assert `train.pair_index.max() < test.pair_index.min()` for both single-cell and pooled schemes. Test windows are contiguous and disjoint.
- **No test-target leakage:** altering all test targets by +50 leaves Ridge and LightGBM predictions unchanged. The models are fit only on training-window targets.
- **Preprocessing isolation:** imputation, scaling, alpha selection, and LightGBM early stopping are confined to the training window. The inner validation split is chronological within each cell.
- **Determinism:** repeated Ridge and LightGBM fits produce identical predictions and selected alpha/best iteration; LightGBM uses deterministic flags and one thread.
- **Feature timing:** features are computed from the current cycle's telemetry and current SOH. No feature name contains `target`, `rul`, `next`, `future`, or `eol`. The LFP feature list is programmatically checked to contain no current-derived features.
- **Gap handling:** `cycle_gap` and `is_gap_pair` are explicit; gap pairs are reported separately, and adjacent-only LFP metrics are reported alongside all-pair metrics.

### 6.2 Design choices that can bias the results

- **Walk-forward dependence:** later training windows contain earlier test records. Test windows are disjoint, but the folds are not independent experiments. Fold-level counts should not be treated as independent replicates.
- **Pair overlap:** pair `t` predicts the SOH that is the current SOH of pair `t+1`. Errors are serially related, so effective sample size is below the row count; no naive row-level significance test is reported.
- **Single LFP cell:** all LFP conclusions are within-cell temporal results. There is no independent-cell holdout.
- **LFP horizon mix:** 245 of 248 LFP pairs are adjacent observed records; three cross gaps of 2, 51, and 201 cycle IDs. The tested gap pair has a much longer horizon than the adjacent pairs. The adjacent-only results address this, but the primary pooled LFP MAE mixes horizons.
- **LFP initial training block:** the first 99 pairs (cycles 1-100) are never test records. Early-life performance is not evaluated out of sample.
- **NMC first half:** the first 50% of each cell's trajectory is training-only. Early-life behavior is not evaluated out of sample.
- **NMC small cell count:** three cells, 234 pooled test rows, and 15 fold-cell units. This is enough to see consistency, not enough to characterize cell-to-cell variability.
- **NMC MAE vs RMSE:** mean drift optimizes a constant offset and shows a clearer MAE than RMSE gain. Reporting only MAE would overstate the improvement; reporting only RMSE would hide it.
- **R² instability:** R² on small fold-cell subsets is unstable, especially near the end of life and when the variance of the target is small. It is reported for completeness, not used for selection.
- **Multiple comparisons:** eight NMC model variants were evaluated. No test result was used to select a variant, but an apparent winner among many variants still warrants independent confirmation.
- **LightGBM early stopping:** inner validation is small (47-84 rows pooled). The delta variants selected as few as one iteration, so their performance is sensitive to the stopping window even though the final refit is deterministic.
- **No production inference:** the experiment did not load production model weights or scalers. It does not validate the deployed inference path.

## 7. Tests and reproducibility

Commands actually executed:

```text
./.venv/Scripts/python.exe -m unittest member2_modeling.experiments.test_walk_forward -v
./.venv/Scripts/python.exe -m unittest member2_modeling.experiments.test_experimental_features -v
./.venv/Scripts/python.exe member2_modeling/experiments/walk_forward_validation.py
```

Results:

- `test_walk_forward`: **9 tests, OK**.
- `test_experimental_features`: **10 tests, OK**.
- Walk-forward script: completed and wrote the files listed below.

The first run of the new fold-boundary test expected cuts `[99, 129, 159, 189, 219, 248]` and failed. The actual deterministic cuts are `[99, 129, 159, 188, 218, 248]`; the test expectation was corrected and the final suite passes. This was a test-expectation error, not a model or data failure.

Environment: Python 3.12.10, NumPy 2.5.4, pandas 3.1.0, LightGBM 4.7.0, scikit-learn 1.9.1.

Input hashes at run time:

| Input | SHA-256 prefix |
|---|---|
| `master_table.parquet` | `6fa7475d18100078` |
| `member2_modeling/feature_schema.json` | `989cf8eb9a199f4b` |
| `member2_modeling/experiments/walk_forward_validation.py` | `6fdc9048126f9118` |

The run summary records the full hashes in `walk_forward_summary.json`.

## 8. Files created or changed

New experiment code:

- `member2_modeling/experiments/walk_forward_validation.py`
- `member2_modeling/experiments/test_walk_forward.py`

New results under `member2_modeling/experiments/results/`:

- `lfp_walk_forward_folds.csv`
- `lfp_walk_forward_metrics.csv`
- `lfp_walk_forward_predictions.csv`
- `lfp_walk_forward_gap_pairs.csv`
- `lfp_walk_forward_gap_pair_errors.csv`
- `nmc_walk_forward_folds.csv`
- `nmc_walk_forward_fold_metrics.csv`
- `nmc_walk_forward_cell_metrics.csv`
- `nmc_walk_forward_pooled_metrics.csv`
- `nmc_walk_forward_predictions.csv`
- `nmc_walk_forward_model_meta.csv`
- `walk_forward_summary.json`
- `MEMBER2_WALK_FORWARD_VALIDATION_REPORT.md` (this report)

No production artifact was written. The current production hashes are unchanged from the prior review:

| Artifact | SHA-256 prefix |
|---|---|
| `master_table.parquet` | `6fa7475d18100078` |
| `master_table_build.py` | `3adb5776439c3bac` |
| `member2_modeling/build_and_train.py` | `b56ce13f7eab8a42` |
| `member2_modeling/predictor.py` | `ef12025e1c9daf01` |
| `member2_modeling/feature_schema.json` | `989cf8eb9a199f4b` |
| `member2_modeling/model_manifest.json` | `bfdf2d4a8a8ec552` |
| `member2_modeling/lfp_lgbm_model.joblib` | `12546c6a432d2096` |
| `member2_modeling/nmc_lgbm_model.joblib` | `b3774599f39db1a9` |
| `member2_modeling/nmc_rul_lgbm_model.joblib` | `206009c8b8929bbf` |
| `member2_modeling/nmc_bilstm_model.pt` | `7dcaf0f8b5d22919` |
| `member2_modeling/scaler_lfp.joblib` | `7111527c8e2a58f8` |
| `member2_modeling/scaler_nmc.joblib` | `2d3a0917ece60b49` |
| `member2_modeling/scaler_nmc_seq.joblib` | `2d3a0917ece60b49` |

No model was exported or promoted. Nothing was staged, committed, or pushed.

## 9. Decision

**LFP unit-safe Ridge:** robust enough for **further candidate development**, not for production promotion. It beats persistence in all five chronological folds, in both evaluated cycle blocks, and after removing the single tested gap pair. The evidence is limited to one cell, one chemistry, one dataset, and 149 evaluated pairs; independent-cell validation is still required.

**NMC mean drift:** the MAE improvement is stable across all five folds and all 15 fold-cell units, and it is not driven by one trajectory. Its RMSE improvement is weaker and mixed at the fold-cell level. Keep mean drift as the NMC baseline; this experiment does not show that the feature-based models add consistent value. The absolute production LightGBM should not be treated as reliable under this walk-forward design.

**Delta models:** delta LightGBM improves over absolute LightGBM and generally beats persistence, but does not consistently beat pooled mean drift. It is not a demonstrated production improvement at this data size.

**Still unvalidated:** independent LFP cells and chemistries; LFP current-derived features (blocked by unit uncertainty); LFP thermal/EFC features (temperature unavailable); NMC behavior in the first half of each trajectory; longer horizons than the next observed record; RUL; and the deployed inference path.

**Recommended next step:** if an independent LFP cell becomes available, pre-register the same 22-feature Ridge pipeline and evaluate it without using that cell for alpha selection or early stopping. For NMC, keep the mean-drift baseline and require any candidate feature model to beat it across all cells and folds before promotion. Do not export a production model from this experiment.
