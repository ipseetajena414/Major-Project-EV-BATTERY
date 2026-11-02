# Member 2 Advanced Battery Intelligence Report

Date: 2026-11-02
Package: `member2_modeling/artifacts/analytics_v1/` (schema `analytics_v1`)
Rebuild: `.venv/Scripts/python.exe -m member2_modeling.analytics.run_analytics --rebuild-rul`

## 1. Scope and evidence base

This report covers the ten requested capabilities. All outputs derive from the active
Member 1 datasets, the legacy cycle-level `master_table.parquet`, and frozen Member 2
model artifacts. No Member 3 output was created, inferred, or back-filled; Member 3
fields are nullable in the combined schema.

The package contains 37 hashed files plus `ANALYTICS_MANIFEST.json`. Every required
Member 5 file is present (Section 9).

## 2. Anomaly detection

### Method

- **Sequential cycles:** robust z-score (median/MAD, threshold 4.0) on seven telemetry
  features, plus one IsolationForest per cell for cells with at least 20 cycles.
- **Snapshot rows:** cross-sectional robust z-score (threshold 6.0) on sensor-like
  numeric columns, top 300 rows retained.
- **Output:** one row per scored cycle with `is_statistical_anomaly`, the triggering
  feature and score, a JSON detail, `interpretation`, and `confirmed_fault=false`.

### Results

| Scope | Rows scored | Flagged | Fraction |
|---|---:|---:|---:|
| Sequential (8 cells) | 817 | 242 | 29.6% |
| `EV_FAILURE_PREDICTION` | 20,000 | 144 | 0.72% |
| `SOLID_STATE_TELEMETRY_2026` | 10,000 | 1,639 | 16.39% |

The sequential rate is high because the IsolationForest contamination is automatic on
short per-cell histories. It is a screening flag, not a diagnosis.

### Honest limitation

The repository has **no verified fault labels**. Precision, recall, and fault-detection
accuracy cannot be computed. No row is called a failure, a defect, or a safety event.
The 29.6% flag rate should be read as "unusual for this cell's own history", not as a
fault prevalence estimate.

## 3. Prediction uncertainty

| Task | Method | Nominal | Empirical coverage | Mean width | Status |
|---|---|---:|---:|---:|---|
| `ev_capacity` | validation residual interval | 0.90 | 0.882 | 12.73 kWh | measured |
| `ev_soh` | validation residual interval | 0.90 | 0.902 | 11.63 pp | measured |
| `solid_soh` | validation residual interval | 0.90 | 0.887 | 2.54 pp | measured |
| `synthetic_capacity` | validation residual interval | 0.90 | 0.900 | 17.92 | measured |
| `ev_failure` | class probabilities | — | — | — | **uncalibrated** |
| `solid_tier` | class probabilities | — | — | — | **uncalibrated** |
| `synthetic_class` | class probabilities | — | — | — | **uncalibrated** |
| NMC RUL | split-conformal absolute residuals | 0.90 | 0.833 | — | measured, below target |
| LFP RUL | curve-family spread | — | — | — | **not calibrated** |
| Range estimate | optimistic/conservative scenarios | — | — | — | **scenario spread, not CI** |

Class probabilities are explicitly labelled `uncalibrated_class_probabilities` and are
never described as calibrated probabilities. Residual intervals describe the frozen test
split; they are not guarantees for new operating conditions.

## 4. Degradation analytics

- `degradation_trajectories.csv` contains 1,819 rows: 819 observed cycle-level SOH rows
  for all nine cells with at least one cycle, plus 1,000 forecast rows for BATTERY_5
  only. Every row carries explicit `record_type`, `is_measured`, `is_forecast`, and
  `validated` flags.
- `degradation_summary.csv` reports observed cycles, first/last/first-cycle SOH, total
  fade, fade rate per 100 cycles over all history and the last 50 cycles, crossing
  position, `rul_support_status`, `eol_projection_available`, and
  `history_is_observed=true`.
- Historical observations are never mixed into forecast rows.
  `rul_degradation_trajectories.csv` mirrors the observed/forecast separation of
  `degradation_trajectories.csv` (819 observed and 1,000 forecast rows). NMC end-of-life
  forecasts are not in that file: they are the 549 origin-level rows for B0005, B0006 and
  B0018 in `rul_eol_trajectories.csv`, which contains forecasts only.

Cells with no EOL projection have `eol_projection_available=false` and no forecast row.
No forecast is drawn for a snapshot dataset.

## 5. Explainability

- Method: permutation importance on transformed features, 300 test rows per task, top 15
  raw features per task. Total 85 rows across the seven tasks.
- **SHAP is unavailable**: the package is not installed and dependency pins are frozen.
  `shap_available=false` and `dependency_pins_changed=false` are recorded in the summary.
- Feature importance is model reliance on this sample. It is **not** a causal effect and
  is unstable when features are correlated.

Notable result: `battery_capacity_kwh` dominates `ev_capacity` because the target
`remaining_capacity` is algebraically derived from it. The feature importance correctly
exposes this target-construction dependency; it is not evidence that pack capacity
causes remaining capacity in a physical sense.

## 6. Battery health status

Configurable bands, validated for unique labels, unique thresholds, and a lowest band at
0%:

| Band | Minimum SOH |
|---|---:|
| GOOD | 90% |
| FAIR | 80% |
| POOR | 70% |
| CRITICAL | 0% |

Latest-observation results: GOOD 3, FAIR 2, POOR 1, CRITICAL 3 (9 cells). Every row
carries `informational_only=true`, `certified_safety_diagnosis=false`, and a disclaimer.
Bands are project-defined, not manufacturer limits. A single latest SOH value ignores
rate of change and operating context.

## 7. Inference interface

`BatteryIntelligenceAPI` remains the single entry point. RUL is added through
`predict_rul(...)`; existing prediction, sequence, range, requirements, and metadata
routes are unchanged.

Validation performed:

- Unknown dataset or target raises `KeyError`.
- Empty input, unknown chemistry, duplicate sequence keys, and invalid numeric values
  raise `CustomInputError`.
- Feature records are checked against the task's required schema; missing required
  features are reported, not silently imputed.
- Chemistry is explicit. Cross-chemistry RUL custom histories are flagged
  `NOT_VALIDATED_FOR_CHEMISTRY`.
- Out-of-distribution inputs are not claimed to be in-distribution: the response keeps
  its EXPERIMENTAL status and provenance.
- RUL histories require actual, strictly increasing cycle ids and finite SOH in (0, 110];
  histories shorter than three cycles are rejected.

## 8. Model comparison

`model_comparison.csv` has 22 rows: each selected snapshot model against its best simple
baseline on the same frozen test rows, plus RUL candidates on the same LOCO origins.

| Task | Selected | Selected metric | Best baseline | Baseline metric | Delta |
|---|---|---:|---|---:|---:|
| `ev_capacity` (MAE) | HGB | 3.209 | train median | 26.115 | -22.906 |
| `ev_failure` (balanced acc.) | logistic regression | 0.914 | majority class | 0.500 | +0.414 |
| `ev_soh` (MAE) | HGB | 2.747 | train median | 7.378 | -4.631 |
| `solid_soh` (MAE) | LightGBM | 0.647 | train mean | 5.674 | -5.027 |
| `solid_tier` (balanced acc.) | LightGBM | 0.901 | chemistry majority | 0.305 | +0.596 |
| `synthetic_capacity` (MAE) | Ridge | 3.818 | chemistry median | 8.149 | -4.331 |
| `synthetic_class` (balanced acc.) | logistic regression | 0.793 | majority class | 0.333 | +0.460 |

The frozen test set was not re-run or re-selected. RUL comparison is in
`MEMBER2_RUL_IMPLEMENTATION_REPORT.md`; the best single model (HGB, MAE 12.07) was not
selected because a simpler model was within 0.60 cycles.

## 9. Reproducibility and Member 5 integration

### Provenance

`prediction_provenance.json` records schema and registry versions, Python/platform,
per-task dataset provenance and feature schemas, the RUL provenance object, and the
traceability rule: snapshot predictions carry `snapshot_row_id`, RUL predictions carry
`battery_id` and the source cycle id. `ANALYTICS_MANIFEST.json` SHA-256 hashes every
delivered file.

### Required Member 5 files

| File | Status |
|---|---|
| `model_metrics.csv` | 7 snapshot tasks, wide |
| `model_metrics.json` | nested, includes uncertainty and RUL |
| `prediction_results.csv` | 17,111 rows (16,486 snapshot + 625 RUL) |
| `regression_residuals.csv` | 10,635 rows |
| `classification_results.csv` | 9 class-level rows |
| `confusion_matrices.json` | all classification tasks |
| `chemistry_metrics.csv` | 32 chemistry/task rows |
| `dataset_summary.json` | snapshot inventory, sequential source, RUL support |
| `model_comparison.csv` | 22 rows |
| `prediction_provenance.json` | complete |
| `member3_combined_analytics_schema.json` | draft-07, 24 fields |
| `member3_results_template.csv` | header-only, no fabricated rows |
| `MEMBER5_ANALYTICS_HANDOVER.md` | integration guidance |
| `ANALYTICS_MANIFEST.json` | hashes and policy flags |

Additional outputs: anomaly flags and summaries, health status, degradation summary and
trajectories, uncertainty summary, feature importance, `member3_soh_sequences.csv`, and
all `rul_v1` artifacts.

### Member 3 boundary

Member 3 owns `range_km`, `usable_energy_kwh`, `consumption_kwh_per_km`,
`vehicle_config_id`, and `member3_model_id`. These columns are nullable in the combined
schema and are empty in the template. No Member 3 output was fabricated. The schema
description states this explicitly.

## 10. Defect corrections

| ID | Defect | Correction | Evidence |
|---|---|---|---|
| F-01 | `ev_failure_predictions.csv` had null `snapshot_row_id` | Failure and range rows now read `snapshot_row_id` from the prediction artifact; bundle validation rejects null ids | `member3_exports_v2_fixed/`: 4,000 failure rows, 0 null ids, 4,000 unique; 3,687 range rows, 0 null ids; validation `ok=true` |
| F-02 | `remaining_capacity` described as measured | Description now states it is derived as `battery_capacity_kwh × battery_health_percent / 100` (rounded to 2 decimals), not an independent measurement, and verified only where both inputs are present | Max deviation from the formula is 0.005 on the 17,883 rows where both inputs are present, consistent with rounding; rows with a null input are not verified |

The existing `member3_exports/` and `member3_exports_v2/` directories were **not**
modified. The corrected bundle is additive at `member3_exports_v2_fixed/`.

## 11. Verification

- Member 2 suite: **109 passed**.
- Root integration suite: **60 passed**.
- Analytics package rebuilt end to end in 183 s.
- Required files and manifest hashes verified.
- No protected member directory, dependency pin, or protected evaluation was modified.

## 12. Unsupported or limited capabilities

| Capability | Status | Reason |
|---|---|---|
| Snapshot RUL | UNAVAILABLE | No chronological EOL trajectory |
| Na-ion RUL | UNAVAILABLE | 25 cycles, no genuine crossing |
| LFP RUL validation | unavailable | No LFP cell reaches 80% |
| Fault diagnosis | unsupported | No verified fault labels |
| Calibrated class probabilities | unsupported | Not calibrated |
| SHAP | unsupported | Package absent, pins frozen |
| Causal claims | unsupported | Observational data only |
| Member 3 range values | not produced | Owned by Member 3 |
