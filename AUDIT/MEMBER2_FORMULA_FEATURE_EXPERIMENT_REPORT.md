# Member 2 — Formula Feature Experiment Report

**Project:** Autonomous Multi-Agent EV Battery Life & Range Predictor
**Module:** Member 2 — feature engineering, SOH prediction, RUL evaluation, Member 4 integration
**Date:** 2026-10-29
**Status:** research-stage experiment. No production model, scaler, schema, manifest or integration contract was replaced.

---

## 1. Scope and protected assets

This pass tested whether the registered research formulas and scientifically defensible derived
features improve next-record SOH prediction, and separately reassessed RUL. Work was isolated in
`member2_modeling/experiments/`.

Verified before and after the run:

| Asset | SHA-256 (first 16) | Status |
|---|---|---|
| `master_table.parquet` | `6fa7475d18100078` | unchanged; 515,270 rows × 13 columns |
| `master_table_build.py` | `3adb5776439c3bac` | unchanged |
| `member2_modeling/build_and_train.py` | `b56ce13f7eab8a42` | unchanged |
| `member2_modeling/predictor.py` | `ef12025e1c9daf01` | unchanged |
| `member2_modeling/feature_schema.json` | `989cf8eb9a199f4b` | unchanged |
| `member2_modeling/model_manifest.json` | `bfdf2d4a8a8ec552` | unchanged |
| `member2_modeling/INTEGRATION_CONTRACT.md` | `0de851584a933375` | unchanged |
| `nmc_lgbm_model.joblib` | `b3774599f39db1a9` | unchanged |
| `lfp_lgbm_model.joblib` | `12546c6a432d2096` | unchanged |
| `nmc_rul_lgbm_model.joblib` | `206009c8b8929bbf` | unchanged |
| `nmc_bilstm_model.pt` | `7dcaf0f8b5d22919` | unchanged |
| scalers (`scaler_*.joblib`) | — | mtime 13:46, untouched |

No raw archive, parquet, builder script or production weight was written by this pass. No commit
or push was performed.

---

## 2. Source files and access status

| Source | Status |
|---|---|
| `FORMULA/paper_formula_register (1).csv` | Present in the repository. 16 formula entries, 16 columns, 15,051 bytes. Parsed in full. |
| `paper_access_status (1).csv` | **Not present in the repository.** Supplied as an uploaded attachment (19,073 bytes). Parsed: 49 `Papers` rows + 10 duplicate `Top10` rows. |
| Paffumi & Martini (2021), `10.3390/wevj12010048` | No local PDF. Register states full text was read; independent MDPI re-fetch returned HTTP 403. Equations below are register-captured, not independently re-verified in this pass. |
| Cristea et al. (2026), `10.3390/app16157756` | No local PDF. Register explicitly says Tables 4–5 coefficients were **not captured**; MDPI re-fetch returned HTTP 403. Coefficients remain unavailable. |
| Original sources credited by both papers | Not present and not checked. |

Access is recorded honestly in the matrix. A printed equation is not treated as validated, and no
coefficient was copied from a similar paper or inferred.

---

## 3. Formula feasibility outcome

`AUDIT/MEMBER2_FORMULA_IMPLEMENTATION_MATRIX.csv` contains 16 rows × 21 columns covering equation,
paper metadata, access status, credited original source, parameters, units, chemistry, inputs,
availability, status, output type, validation/leakage notes and decision.

| Status | Count | Formulas |
|---|---:|---|
| IMPLEMENTABLE | 0 | — |
| ADAPTABLE | 2 | `P1-EoL` (80% EoL as a project adaptation), `C-Eq10-12` (cell-level EFC, NMC only) |
| CONDITIONAL | 0 | — |
| SCENARIO_ONLY | 6 | `P1-Eq5`, `P1-HVAC`, `P1-Eff`, `P1-Cons`, `C-Eq3`, `C-Eq16-17` |
| BLOCKED | 8 | `P1-Eq1`, `P1-Eq3`, `C-Eq1`, `C-Eq2`, `C-Eq4`, `C-Eq5-9`, `C-Eq13-14`, `C-Eq15` |

Critical blockers:

- `P1-Eq1` is parameterized for NCM-LMO, not NASA NMC, LFP or Na-ion. Calendar age is unavailable,
  the loss unit is not stated, and credited originals were not read.
- `P1-Eq3` has an activation-energy exponent sign that conflicts with `P1-Eq1`, requires an
  unavailable `SOC_min`, and uses NCM-LMO parameters. Remains blocked.
- Cristea Tables 4–5 coefficients were never captured, so no calendar/cycle capacity-loss curve can
  be parameterized. SoC, DoD, calendar age and physical cycle counts are absent from the data.
- `C-Eq15` is a model output on a 0–1 scale. It is not a measured SOH label and was not merged with
  `soh_pct`.
- Vehicle/pack formulas (`P1-Eq5`, HVAC, efficiency, consumption, `C-Eq3`, `C-Eq16-17`) require pack
  energy/consumption assumptions. They were not applied to cell data and remain Member 3 scenarios.

No formula-generated SOH label was created. Measured `soh_pct` is the only target used below.

---

## 4. Input audit relevant to feature construction

| Input | NMC (NASA) | LFP B5 | Na-ion |
|---|---|---|---|
| Current | Amperes; discharge about −2 A | Constant −0.625, `CURRENT_UNIT_UNDERDETERMINED` | Mixed phases; excluded from models |
| Temperature | Present (22.4–42.0 °C), no nulls | Missing for all rows | Missing for all rows |
| Time | `time_sec` resets per cycle; raw order unsorted | Present; discharge-only | Present |
| Capacity/SOH | Broadcast per cycle, 2.0 Ah rated | 1.25 Ah conditional rating; cycle 40 null SOH | Per-cell nominal; excluded |
| SOC/DoD/SOC_min | Absent | Absent | Absent |
| Calendar age | Absent | Absent | Absent |
| Cycle coverage | 1–168 / 1–168 / 1–132 | 250 observed IDs in 1–100, 151–200, 401–500 | 25 per cell |

Verified supporting fact: NMC discharge integration of `|I|·dt` matches the broadcast capacity label
with mean ratio 1.0079 (range 0.99996–1.0287). This supports the ampere unit and the trapezoidal
method for NMC only. LFP current-derived features remain blocked.

---

## 5. Implemented features and numerical tests

File: `member2_modeling/experiments/experimental_features.py`.

| Feature | Chemistry | Type | Definition / provenance |
|---|---|---|---|
| `ah_discharge_trapz` | NMC | telemetry-derived | Σ 0.5·(|I_k|+|I_{k+1}|)·Δt/3600 over `current_a < 0`, sorted, duplicate timestamps dropped, non-negative Δt |
| `efc_cumulative` | NMC | project adaptation of `C-Eq10-12` | cumulative observed discharge Ah ÷ 2.0 Ah rated; observed records only |
| `thermal_hours_outside_20_30` | NMC | telemetry-derived | Σ Δt/3600 where midpoint T is outside [20, 30] °C |
| `thermal_degree_hours_above_30` | NMC | telemetry-derived | Σ max(T_mid − 30, 0)·Δt/3600 |
| `thermal_degree_hours_below_20` | NMC | telemetry-derived | Σ max(20 − T_mid, 0)·Δt/3600 |
| `v_at_q20`, `v_at_q50`, `v_at_q80` | NMC | telemetry-derived | voltage interpolated at 20/50/80 % of that cycle's cumulative discharge Ah |
| dV/dQ median and IQR | NMC | diagnostic only | 20–80 % Q range, uniform Q grid, Savitzky–Golay smoothing; not used as a model feature |

LFP current throughput, LFP EFC and LFP/Na-ion thermal exposure are not computed. Their columns are
left missing with a blocking status; no temperature or unit was imputed.

`test_experimental_features.py` contains 10 hand-calculated tests and all pass:

- trapezoid Ah = 3/3600 Ah on a hand-built interval;
- time-weighted thermal hours = 2/3600 h and degree-hours = 5/3600 °C·h;
- interpolation at 20/50/80 % Q returns 3.94/3.85/3.76 V on a linear fixture;
- unsorted rows with a duplicate timestamp produce the same integral;
- LFP throughput and thermal features are NaN/BLOCKED;
- missing NMC temperature produces missing thermal features, not an imputed value;
- cumulative EFC on three observed records = [0.5, 0.75, 1.75];
- constant-slope dV/dQ diagnostic returns −36 V/Ah;
- unit-safe LFP feature list removes all seven current-derived features.

---

## 6. Experiment protocol

- Reused production `build_cycle_feature_table` and per-cell chronological 60/20/20 split.
- Targets are `shift(-1)` SOH: the **next observed record**, not necessarily the next physical cycle.
- NMC: 465 pairs (train 278, validation 92, test 95).
- LFP: 248 pairs (train 148, validation 49, test 51).
- Every model for a chemistry is scored on exactly the same target rows.
- Ridge `alpha` and the final model/feature-set choice use validation MAE only. The selected
  configuration's test metric is reported once.
- Models: persistence, mean one-step drift (feature-free), per-cell linear trend, ridge, existing
  LightGBM, and delta-target LightGBM.
- `mean_delta` predicts `current_soh + mean(train target − current)` and was added because a
  systematic one-step decay means "persistence" is not a sufficient baseline.

Metric caveat: pooled R² is inflated by between-cell SOH level differences. MAE on identical rows is
the primary comparison.

---

## 7. NMC results

### 7.1 Model comparison on production features

| Model | Val MAE | Val RMSE | Val R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0.4704 | 0.6304 | 0.9045 | 0.5040 | 0.8669 | 0.9395 |
| **Mean one-step drift** | **0.3700** | 0.6206 | 0.9074 | **0.4138** | 0.8666 | 0.9395 |
| Delta LightGBM | 0.4214 | 0.6092 | 0.9108 | 0.4534 | 0.8631 | 0.9400 |
| Ridge (α = 10) | 0.6493 | 0.7714 | 0.8570 | 1.4501 | 1.9084 | 0.7066 |
| Existing LightGBM | 3.5827 | 4.0338 | −2.9103 | 7.9678 | 8.6932 | −5.0872 |
| Linear trend | 2.2577 | 3.0783 | −1.2773 | 4.7174 | 5.8064 | −1.7156 |

The training mean one-step drift is −0.2486 SOH points per observed record. Adding that constant
improves MAE over persistence by 21.3 % on validation and 17.9 % on test. This is the selected NMC
configuration, and it uses **no features at all**.

The delta LightGBM beats persistence, but loses to the constant-drift baseline on MAE for both
validation and test. On RMSE it is only marginally lower than the drift baseline (0.6092 vs 0.6206
on validation; 0.8631 vs 0.8666 on test), which does not offset the higher MAE. Its early stopping
selects 2 trees, so it is effectively a noisy estimate of the same drift. The existing absolute-target
LightGBM is unusable in this formulation (R² below −2.9).

### 7.2 Feature ablation (NMC)

Values are MAE improvement versus the same model with `prod` features. Positive is better.

| Added / changed feature | Delta val | Delta test | Ridge val | Ridge test |
|---|---:|---:|---:|---:|
| `efc_cumulative` | +2.8 % | +1.2 % | +15.1 % | +32.5 % |
| `ah_discharge_trapz` added | +2.8 % | +1.2 % | +1.6 % | +0.6 % |
| `ah_throughput` replaced by `ah_discharge_trapz` | +2.6 % | −0.8 % | −0.2 % | +0.4 % |
| Thermal exposure (3 features) | +1.5 % | −0.0 % | +4.5 % | +5.4 % |
| `v_at_q20/50/80` | 0.0 % | 0.0 % | +0.4 % | +1.1 % |
| All cleared features | +4.2 % | −3.2 % | +17.7 % | +33.6 % |

`ah_discharge_trapz` and `efc_cumulative` produce numerically identical delta-model metrics: EFC is
the cumulative sum of the same Ah values divided by a constant, so a tree split on one carries the
same information as the other in this setup. They are not independent evidence.

Interpretation:

- `efc_cumulative` is the only derived feature with a consistent gain across both model families and
  both splits. The gain is large for ridge, but ridge remains worse than persistence in absolute
  terms; the gain is only 1.2 % on test for the delta model.
- Thermal exposure helps ridge modestly and is neutral for the delta model; not robust.
- Voltage-at-capacity features add nothing measurable.
- The all-features set is best on validation but **worse than `prod` on test** for the delta model.
  That is a validation-overfit signal, not a demonstrated improvement.
- No feature-based NMC model beats the feature-free mean-drift baseline (val 0.3700, test 0.4138).

**NMC conclusion: no formula-derived feature provided a robust improvement in next-record SOH
prediction.** The only defensible NMC gain is the constant-drift correction to persistence, which is
not a formula feature and is not a learned model.

---

## 8. LFP results

### 8.1 Model comparison

| Model | Val MAE | Val RMSE | Val R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0.1199 | 0.1377 | 0.7449 | 0.1313 | 0.1473 | 0.6300 |
| **Ridge, `prod` (α = 0.01)** | **0.0824** | 0.0976 | 0.8718 | **0.0863** | 0.1082 | 0.8002 |
| **Ridge, `unit_safe` (α = 0.01)** | 0.0846 | 0.0990 | 0.8682 | 0.0896 | 0.1129 | 0.7825 |
| Delta LightGBM, `prod` | 0.1055 | 0.1262 | 0.7858 | 0.1194 | 0.1442 | 0.6454 |
| Delta LightGBM, `unit_safe` | 0.1049 | 0.1236 | 0.7944 | 0.1192 | 0.1410 | 0.6610 |
| Mean one-step drift | 0.1173 | 0.1366 | 0.7488 | 0.1288 | 0.1486 | 0.6231 |
| Existing LightGBM | 0.9033 | 0.9363 | −10.80 | 0.9256 | 0.9533 | −14.50 |
| Linear trend | 0.2872 | 0.3423 | −0.577 | 1.3332 | 1.3765 | −31.32 |

Training mean one-step drift is −0.0236 SOH points per record.

- Ridge on `prod` is the validation-selected LFP model and beats persistence by 31.3 % (val) and
  34.3 % (test).
- `unit_safe` removes the seven current-derived features: `ah_throughput`, `energy_wh`,
  `avg_power_w`, `i_mean`, `i_std`, `i_min`, `i_max`. Ridge on this 22-feature subset still beats
  persistence by 29.5 % (val) and 31.7 % (test), at a small cost versus `prod` (−2.6 % val,
  −3.9 % test).
- Because LFP current units are underdetermined, the `unit_safe` result is the physically defensible
  one. The `prod` result shows that the unverified-unit features carry additional signal, but that
  signal cannot be interpreted physically.
- The mean-drift and delta baselines give only small gains; the LFP gain comes from the regularized
  model, not from a research formula.

### 8.2 Adjacent versus gap-crossing transitions

| Transition class | Count | Detail |
|---|---:|---|
| Train gap-crossing | 2 | 39→41 (ΔSOH −0.2963), 100→151 (ΔSOH −0.9335) |
| Validation adjacent | 48 | — |
| Validation gap-crossing | 1 | 200→401; current 92.1116, target 92.2137, ΔSOH +0.1021; persistence absolute error 0.1021 |
| Test adjacent | 51 | all test transitions |
| Test gap-crossing | 0 | — |

Validation metrics above include the single 200→401 pair (n = 49). Test metrics are entirely
adjacent-record performance. A model-level gap error is not reported because n = 1 would not be a
meaningful estimate; the observed SOH change and the persistence error are reported instead. The
200→401 pair is never presented as test performance.

---

## 9. RUL reassessment

### 9.1 Production test composition (confirmed)

The existing chronological RUL test contains **97 rows, all 97 post-EOL** (RUL = 0). The reported
production RUL error therefore measures post-EOL behaviour only and cannot demonstrate pre-EOL
forecasting. This confirms the earlier audit finding.

### 9.2 Pre-EOL leave-one-cell-out diagnostic

RUL labels use the first observed crossing below 80 % of the 2.0 Ah rating. This is an explicit
project adaptation; it is not the paper's vehicle usable-energy definition. First crossings are at
observed cycles 75 (B0005), 63 (B0006) and 45 (B0018).

| Held-out cell | n test rows | Constant median MAE | SOH extrapolation MAE | LightGBM MAE | LightGBM R² | LightGBM MAPE |
|---|---:|---:|---:|---:|---:|---:|
| B0005 | 74 | 19.99 | 10.73 | 9.94 | 0.668 | 35.4 % |
| B0006 | 62 | 15.53 | 22.68 | 5.11 | 0.864 | 28.0 % |
| B0018 | 44 | 14.27 | 6.25 | 3.83 | 0.846 | 43.0 % |
| **Pooled** | **180** | **17.06** | **13.75** | **6.78** | **0.776** | **34.7 %** |

The pooled LightGBM MAE is 6.78 cycles versus 13.75 for SOH-threshold extrapolation and 17.06 for
the constant median. It wins in all three folds. However:

- there are only **three cells**, so the effective sample size is three trajectories, not 180
  independent units;
- the SOH-extrapolation baseline is unstable across cells (it is worse than the constant median on
  B0006), which shows how strongly the folds depend on the held-out trajectory;
- MAPE is 28–43 % because near-EOL RUL values are small;
- the diagnostic is one-step/row-wise, not a recursive multi-horizon forecast.

**RUL conclusion: the pre-EOL diagnostic is encouraging but remains small-sample and unvalidated.
The production RUL model is unchanged and must not be described as pre-EOL validated.** A meaningful
validation requires additional cells or an external dataset and a multi-horizon protocol.

---

## 10. Missing-data and leakage compliance

- Temperature was never imputed. LFP and Na-ion thermal features are absent, not fabricated.
- SOC, DoD, SOC_min, calendar age, pack energy and pack configuration were not invented.
- B5 cycle 40's null SOH was dropped; it was not replaced by a formula estimate.
- B5 missing cycle-ID ranges were not interpolated. The 39→41, 100→151 and 200→401 transitions are
  reported as observed-record transitions.
- No formula-generated SOH was used as a target. Every target is measured `soh_pct`.
- Features use only current/past telemetry. Cumulative features are computed over observed records,
  grouped by cell.
- Splits are chronological per cell; no shuffling.
- Ridge `alpha` and model/feature selection use validation MAE. Test rows are reported once for the
  selected configuration and were not used to tune features or hyperparameters.

---

## 11. Integration and candidate decision

No change was made to `feature_schema.json`, `model_manifest.json` or `INTEGRATION_CONTRACT.md`,
because the evidence does not yet justify a production replacement:

- NMC: no feature-based model beat the feature-free drift baseline; adding formula features to the
  delta model did not produce a consistent test gain.
- LFP: unit-safe ridge is a credible experimental candidate, but it is trained on one cell
  (BATTERY_5) with 148 training pairs and has not been validated on another LFP cell or dataset.
  It also predicts only the next observed record.
- RUL: the pre-EOL evidence is a three-cell diagnostic, not validation.

No candidate model file was exported. The LFP ridge pipeline is reproducible from the script and the
recorded feature list/alpha; exporting a partially validated weight into the model directory would
risk it being mistaken for a production artifact. If a later, broader validation supports adoption,
the candidate contract would be: `unit_safe` 22-feature LFP input, ridge α = 0.01, target = next
observed-record `soh_pct`, chemistry = LFP B5 only, and an explicit "research-stage" provenance tag.

The eight NMC experimental features remain documented with provenance in the results JSON but are
not part of the production schema.

---

## 12. Limitations

- Only one LFP cell and three NMC cells; cell-level generalisation is untested.
- The LFP target is the next observed record, and the test split contains no gap-crossing pair. The
  single validation gap pair cannot support a gap-performance estimate.
- NMC current is verified in A, but the LFP current unit is still underdetermined; all new LFP
  current and EFC features are blocked.
- Thermal features exist only for NMC. Na-ion is excluded from the models.
- All targets and features are cell-level; no pack-level energy or range claim is made.
- Formula parameters for the registered calendar/cycle-loss equations remain unavailable or
  chemistry-inapplicable; none of those equations was tested.
- Repeated pipeline runs during debugging were deterministic. No test-based tuning was performed,
  but test metrics were inspected before the final selection rule was corrected to include all model
  families. The final choice is validation-based and reported without further change.

---

## 13. Deliverables, changed files and reproducibility

### 13.1 New files (untracked; the "diff" for each is its full new content)

| File | Bytes | Lines | Purpose |
|---|---:|---:|---|
| `AUDIT/MEMBER2_FORMULA_IMPLEMENTATION_MATRIX.csv` | 15,094 | 17 | 16-formula evidence matrix |
| `AUDIT/MEMBER2_FORMULA_DERIVATION_AND_ML_PLAN.md` | 15,338 | 221 | feasibility/plan document |
| `AUDIT/MEMBER2_FORMULA_FEATURE_EXPERIMENT_REPORT.md` | this file | — | results and conclusions |
| `member2_modeling/experiments/experimental_features.py` | 11,593 | 337 | cleared feature code and provenance flags |
| `member2_modeling/experiments/test_experimental_features.py` | 5,971 | 157 | 10 hand-calculated tests |
| `member2_modeling/experiments/run_experiments.py` | 24,053 | 631 | isolated baselines, ablations, LFP gap reporting, RUL LOO |
| `member2_modeling/experiments/results/feature_ablation.csv` | 21,131 | 109 | all model/split/feature-set metrics |
| `member2_modeling/experiments/results/experiment_results.json` | 61,288 | 1,898 | provenance, splits, selection, gap summary, metrics |
| `member2_modeling/experiments/results/rul_pre_eol_results.csv` | 1,632 | 10 | per-fold RUL diagnostic |
| `member2_modeling/experiments/results/cycle_features_experimental.parquet` | 338,652 | — | isolated experimental cycle table |

### 13.2 Pre-existing changes, preserved untouched

```
M  .gitignore
AM AUDIT/MEMBER2_CYCLE_LEVEL_DATA_QUALITY_REPORT.md
?? member2_modeling/
```

`git diff --cached --stat` shows only `.gitignore` (the pre-existing `.venv/` ignore) and the audit
report; the working-tree diff on the audit report is its pre-existing staged-vs-worktree correction.
Neither was edited in this pass. The rest of `member2_modeling/` was already untracked before this
task; its production files and weights were not modified.

### 13.3 Reproduction

```
./.venv/Scripts/python.exe member2_modeling/experiments/run_experiments.py
./.venv/Scripts/python.exe -m unittest member2_modeling.experiments.test_experimental_features -v
```

Environment recorded in `experiment_results.json`: Python 3.12.10, numpy 2.5.4, pandas 3.1.0,
LightGBM 4.7.0, scikit-learn 1.9.1. Input parquet hash is recorded and matches `6fa7475d…`.

For untracked new files, `git diff` is empty by definition. The exact new-file diff can be produced
without staging with:

```
git diff --no-index -- /dev/null AUDIT/MEMBER2_FORMULA_IMPLEMENTATION_MATRIX.csv
```

and likewise for each file in 13.1.

---

## 14. Conclusions

1. **No registered research formula is implementable as a capacity-loss model** with the available
   sources and inputs. Fourteen of sixteen are blocked or scenario-only; the remaining two are
   adaptations, and both were tested only as limited NMC features.
2. **The NMC experiment produced no feature-based improvement.** The only gain over persistence is a
   constant one-step drift correction (−0.2486 SOH points per record), which reduces MAE by 21.3 %
   on validation and 17.9 % on test without using features. EFC is the only derived feature with a
   consistent (but small in the winning family) gain.
3. **LFP ridge regression beats persistence on validation and test**, including a unit-safe variant
   that avoids the underdetermined current unit. This is a research-stage candidate, not a
   production replacement; it is based on one cell and a next-observed-record target.
4. **Production RUL remains unvalidated.** Its test set is entirely post-EOL. A three-cell pre-EOL
   leave-one-cell-out diagnostic beats naive baselines, but the effective sample is three
   trajectories and the result is not sufficient for production claims.
5. **No production artifact, schema, manifest, contract, raw data or builder was changed.** No model
   files were exported, and nothing was committed or pushed.
