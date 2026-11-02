# Member 2 — Data, Split and Leakage Audit

**Audit date:** 2026-11-02
**Repository state:** branch `member2-upgrade` (upstream `origin/member5/integration`). The working tree already carried the Member 2 work as staged additions; this audit added no source changes and regenerated no artifact.
**Audit type:** read-only code inspection + read-only data inspection + executed non-destructive checks.
**Scope:** the seven registered snapshot tasks in `member2_modeling/snapshot_ml/` and the Member 1 processed datasets they consume.

Finding labels used throughout: **VERIFIED**, **EXPERIMENTAL**, **UNVERIFIED**, **NOT SUPPORTED**, **BLOCKED**.

---

## 1. Data provenance and utilization

Source of truth: `data/processed/dataset_registry.json`, `data/processed/data_quality_report.json`, `data/processed/data_dictionary.csv`, `member1_pipeline/loader.py`, `member1_pipeline/cleaning.py`.

| Dataset | Source file | `data_origin` | Grain | Rows × Cols | Sequence | Used by |
|---|---|---|---|---:|---|---|
| `EV_FAILURE_PREDICTION` | `ev battery_failure prediction Dataset.csv` | `source_dataset` | vehicle pack snapshot | 20,000 × 82 | No | `ev_soh`, `ev_capacity`, `ev_failure` |
| `SOLID_STATE_TELEMETRY_2026` | `solid_state_ev_battery_telemetry_enriched_2026.csv` | `source_dataset` | cell snapshot | 10,000 × 41 | No | `solid_soh`, `solid_tier` |
| `SYNTHETIC_BATTERY_LIFECYCLE` | `synthetic_battery_lifecycle.csv` | `synthetic` | synthetic battery snapshot | 2,000 × 20 | No | `synthetic_capacity`, `synthetic_class` |

Processed files and SHA-256 values are recorded in the registry and re-checked by `loader`:

| Dataset | Processed path | Processed SHA-256 |
|---|---|---|
| EV | `DATA/processed/ev_battery_failure_clean.parquet` | `6b4f9035…5f57dc9d` |
| Solid-state | `DATA/processed/solid_state_telemetry_clean.parquet` | `b49529cc…43cc87513` |
| Synthetic | `DATA/processed/synthetic_battery_lifecycle_clean.parquet` | `9f7e8ff5…49bf6238` |

- **VERIFIED** — all seven tasks load their dataset through `member1_pipeline.loader.load_dataset`, never through the legacy `master_table.parquet`. The loader verifies the processed SHA-256 against the registry before returning a frame.
- **VERIFIED** — no Member 1 file is modified by Member 2. `member1_pipeline.build --check` reports `VALIDATION OK` after the audit.
- **VERIFIED** — `sequence_available = false` for all three active datasets in the registry; no Member 2 snapshot task fits a sequence model (`leakage_audit.json` → `sequence_model_not_used = true`).

### 1.1 Data completeness

| Dataset | Duplicate full rows | Missing cells | Missing rows retained | Cleared raw rows |
|---|---:|---:|---|---:|
| EV | 0 | 53,344 | Yes | 0 |
| Solid-state | 0 | 0 | Yes | 0 |
| Synthetic | 0 | 0 | Yes | 0 |

EV missingness is spread across many columns (largest: `fast_charge_ratio` 988, `average_speed` 984, `manufacturing_year` 982, `home_charging_ratio` 980, `sensor_fault_count` 968). It is retained as null and imputed only inside the training-fitted pipeline. The registry and quality report agree: `raw_data_modified = false`, `rows_dropped = 0`.

Chemistry distributions:

| Dataset | Chemistry column | Distribution |
|---|---|---|
| EV | `battery_chemistry` | NMC 7,576 · LFP 5,867 · LMO 2,998 · NCA 1,484 · LTO 1,353 · missing 722 |
| Solid-state | `electrolyte_chemistry` | Sulfide 4,441 · Oxide 2,385 · Polymer 1,979 · Halide 1,195 |
| Synthetic | `electrolyte_type` | Solid_State 685 · Lithium_Sulfur 667 · Sodium_Ion 648 |

---

## 2. Entity audit — are the identifiers physical?

This is the decisive question for whether the grouped split proves anything.

| Dataset | Entity column | Rows | Unique IDs | Rows per ID | Observed pattern |
|---|---|---:|---:|---|---|
| EV | `vehicle_id` | 20,000 | 20,000 | 1 | `EV100000`, `EV100001`, … sequential |
| EV | `battery_serial` | 20,000 | 20,000 | 1 | `BAT-10000000`, `BAT-10000001`, … sequential |
| Solid-state | `cell_id` | 10,000 | 10,000 | 1 | `BAT-BYD-008797A7`, `BAT-BYD-017F5910`, … manufacturer code + 8 hex characters |
| Synthetic | `battery_id` | 2,000 | 1,978 | 1 (×1,956) or 2 (×22) | `BAT_54170`, `BAT_67194`, … random 5-digit |

Findings:

- **VERIFIED** — EV identifiers are perfectly sequential integers rendered as strings (`EV100000…EV101999`; `BAT-10000000…BAT-10019999`). This is an identifier-assignment pattern, not a record of real vehicle serials. The dataset also mixes 21 unrelated vehicle brands and 8 battery manufacturers into a single synthetic-looking fleet.
- **VERIFIED** — solid-state `cell_id` values are also algorithmically generated (manufacturer prefix + hexadecimal suffix), one per row.
- **VERIFIED** — for EV and solid-state, entity count equals row count in every split. The grouped splitter therefore cannot separate repeated observations of one physical unit, because no unit repeats.
- **VERIFIED** — the synthetic dataset is the only one with repeated entity IDs (22 batteries, each twice); 16 of those 22 repeated IDs carry conflicting metadata (chemistry or class). The grouped splitter keeps each `battery_id` inside a single split, so the conflicting duplicates do not cross train/val/test.
- **UNVERIFIED** — whether `EV_FAILURE_PREDICTION` and `SOLID_STATE_TELEMETRY_2026` are real measured campaigns or simulated/synthesized data. The registry labels them `data_origin = source_dataset`, but nothing in the repository documents the origin beyond the file name. **Genuine physical entity identity cannot be established from the data or metadata available.**
- **NOT SUPPORTED** — any claim of "vehicle-level holdout" or "cell-level generalization" for EV/solid-state. The grouping key is unique per row, so the split is mathematically equivalent to a random row split. The code is correct; the claim it is used to support is not.

Consequence: `entity_grouped_stratified_60_20_20_stratified_group_kfold` is accurate as a *description of the code path*, but for EV and solid-state it does **not** provide entity-level independence. It must be described as an **i.i.d.-equivalent random holdout with row-unique identifiers**. `synthetic_capacity` / `synthetic_class` do obtain real duplicate protection for their 22 repeated IDs.

---

## 3. Split construction

Implementation: `member2_modeling/snapshot_ml/splits.py`.

- **VERIFIED** — 60/20/20 split. Outer `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`; `folds[0]` gives pool + test. Inner `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=43)` splits the pool into train/val. Fallback on `ValueError` is `GroupShuffleSplit` (0.2 test, 0.25 of the remainder for validation) with the warning recorded in `config.json`.
- **VERIFIED** — regression strata are quantile bins of the target (`_regression_strata`, q=5→2, minimum 5 rows per bin). Classification strata are the class labels.
- **VERIFIED** — `_verify_disjoint` raises `SplitError` if any entity appears in more than one split, and rejects empty splits. This check did not fire for any task.
- **VERIFIED** — splits are recovered deterministically from the frozen `random_state` and are fingerprinted with a SHA-256 over the sorted entity lists; the digests in `config.json` and `leakage_audit.json` match.

Observed split composition (from each task's `config.json`):

| Task | Train | Val | Test | Train entities | Val entities | Test entities | Overlap |
|---|---:|---:|---:|---:|---:|---:|---|
| `ev_soh` | 11,426 | 3,809 | 3,809 | 11,426 | 3,809 | 3,809 | 0/0/0 |
| `ev_capacity` | 11,631 | 3,878 | 3,877 | 11,631 | 3,878 | 3,877 | 0/0/0 |
| `ev_failure` | 12,000 | 4,000 | 4,000 | 12,000 | 4,000 | 4,000 | 0/0/0 |
| `solid_soh` | 6,000 | 2,000 | 2,000 | 6,000 | 2,000 | 2,000 | 0/0/0 |
| `solid_tier` | 5,999 | 2,001 | 2,000 | 5,999 | 2,001 | 2,000 | 0/0/0 |
| `synthetic_capacity` | 1,200 | 400 | 400 | 1,187 | 395 | 396 | 0/0/0 |
| `synthetic_class` | 1,200 | 400 | 400 | 1,185 | 396 | 397 | 0/0/0 |

`ev_soh` and `ev_capacity` have fewer rows than 20,000 because rows with a null target are dropped before splitting (956 and 614 respectively). `ev_failure`, `solid_*` and `synthetic_*` drop none.

### 3.1 Chronology

- **VERIFIED** — none of the three datasets contains a record timestamp. Time-adjacent columns are `manufacturing_year`, `vehicle_age_years` (EV), and `charge_time_mins` (synthetic); `cycle_number` / `cycle_count` are per-record state counters, not an ordered history.
- **VERIFIED** — because there is no time axis, no chronological split can be constructed, and the current split is not chronological.
- **NOT SUPPORTED** — temporal generalization ("predicts a future record") is not established by this protocol. The split proves out-of-sample performance on *unseen rows*, not on *future time*.

### 3.2 Stratification

- **VERIFIED** — stratification is target-based and does not consume entity identity, so it cannot move an entity across splits (the grouping constraint is enforced separately by `StratifiedGroupKFold`).
- **VERIFIED** — stratified counts are stable for classification (largest deviation: `solid_tier` 5,999/2,001/2,000).
- **EXPERIMENTAL** — target-stratified splitting changes the test-set target distribution relative to the natural distribution. Reported metrics are therefore conditional on the stratified test mix, not on a random deployment mix. This is disclosed but not corrected in the reported metrics.

---

## 4. Leakage controls

### 4.1 Column exclusion

`member2_modeling/data_integration/profiles.py` and each task's `config.json` record the forbidden columns. The audit compared them with the actual feature matrices:

| Task | Forbidden columns | Forbidden used as feature |
|---|---|---:|
| `ev_soh` | `state_of_health` (target), `battery_health_percent`, `capacity_loss_percent`, `remaining_capacity`, `predicted_remaining_life_cycles` | 0 |
| `ev_capacity` | `remaining_capacity` (target), `state_of_health`, `battery_health_percent`, `capacity_loss_percent`, `predicted_remaining_life_cycles` | 0 |
| `ev_failure` | `battery_failure` (target), `state_of_health`, `battery_health_percent`, `capacity_loss_percent`, `remaining_capacity`, `predicted_remaining_life_cycles` | 0 |
| `solid_soh` | `state_of_health_pct` (target), `degradation_risk_tier`, `projected_cycle_life_80pct_soh`, `cot_electrochemical_risk_rationale`, `confidence_score` | 0 |
| `solid_tier` | `degradation_risk_tier` (target), `state_of_health_pct`, `projected_cycle_life_80pct_soh`, `cot_electrochemical_risk_rationale`, `confidence_score` | 0 |
| `synthetic_capacity` | `discharge_capacity_ah` (target), `state_of_health` | 0 |
| `synthetic_class` | `state_of_health` (target), `discharge_capacity_ah` | 0 |

- **VERIFIED** — `feature_columns_for_target` admits only columns whose data-dictionary role is exactly `feature`. Derived score columns (`battery_stress_index`, `aging_score`, `thermal_health_score`, `charging_quality_score`, `driving_stress_score`), unit-ambiguous columns (`internal_resistance`, `remaining_capacity` on EV), enriched risk scores, projections and rationale/confidence text on solid-state are all excluded by role, not by heuristic.
- **VERIFIED** — `TabularFeaturePipeline` raises if a forbidden or target column reaches `fit`; `leakage_audit.json` confirms `forbidden_used_as_feature = 0` for all seven tasks.
- **VERIFIED** — inference drops target/leakage columns even if a caller supplies them (`inference.py`, `forbidden_present` branch), and warns instead of silently using them.

### 4.2 Preprocessing isolation

- **VERIFIED** — `TabularFeaturePipeline` is fitted on `data.iloc[split.train]` only. Medians, category vocabularies and `__MISSING__` handling are computed from train. Validation and test are transformed with the frozen pipeline.
- **VERIFIED** — model selection uses validation only; each task's `config.json` records `selection_rule` and the candidate scores; the test split is scored once after selection.
- **VERIFIED** — no feature selection, resampling, or hyperparameter search touches the test split. Candidate hyperparameters are fixed in `candidates.py`; there is no grid search.
- **VERIFIED** — validation data was additionally used for LightGBM early stopping (`eval_X`/`eval_y` on the validation split). This is selection on validation, not on test — acceptable, but it means the validation split is not a clean unbiased estimate for the LightGBM tasks.
- **VERIFIED** — classification models use `predict`/`predict_proba` with balanced class weights; no threshold was tuned on test.

### 4.3 Residual leakage risks that cannot be removed by column exclusion

These are not implementation bugs, but they limit what the metrics mean:

1. **`ev_capacity` target is algebraically derived.** `remaining_capacity = battery_capacity_kwh × battery_health_percent / 100` to within 0.005 kWh (6 of 20,000 rows exceed 0.005). `capacity_loss_percent = 100 − battery_health_percent` exactly. `battery_capacity_kwh` remains a permitted feature. The model therefore partly reconstructs a formula whose output is a product of an input feature and a hidden label.
2. **`ev_soh` is highly reconstructable from non-label telemetry.** A 5-fold linear probe on permitted numeric features reaches R² ≈ 0.90 for `state_of_health`, and the two SOH labels agree at r = 0.9898. The target is either strongly determined by the telemetry or generated from it. This is "label reconstruction" risk, not column leakage.
3. **`solid_tier` is a rule-generated tier.** Tier ranges overlap heavily in SOH (Tier 4: 70.00–100.00; Tier 3: 78.00–99.21; Tier 2: 85.02–92.00; Tier 1: 92.01–100.00), so SOH alone does not determine the tier (train balanced accuracy 0.847). Adding cycle/thermal/resistance signals raises it to 0.917; an unlimited tree on all permitted features reaches 1.000. The label is regenerable from the inputs even though SOH is excluded.
4. **`synthetic_class` is an exact capacity threshold per chemistry.** Capacity is excluded as a feature, but `cycle_count` and other synthetic signals reconstruct the class (train balanced accuracy 0.769 at depth 3 → 1.000 at unlimited depth). This is label reconstruction from a synthetic generator, not health prediction.
5. **`synthetic_capacity` shares its generator with the class label.** Both targets come from the same synthetic campaign; the two tasks must not be treated as independent evidence.
6. **No external validation set exists.** All reported numbers are internal holdouts. No second dataset from a different source, lab, or fleet has been scored.

---

## 5. Verdict

| Criterion | Verdict | Basis |
|---|---|---|
| Code enforces entity-disjoint splits | **VERIFIED** | `_verify_disjoint`, overlap 0 in all tasks |
| Splits are meaningfully entity-level for EV/solid | **NOT SUPPORTED** | IDs are unique per row; grouping is equivalent to a random row split |
| Splits are meaningfully entity-level for synthetic | **VERIFIED (limited)** | 22 repeated IDs, kept within split; 16 have conflicting metadata |
| No transductive preprocessing leakage | **VERIFIED** | pipeline fitted on train only; forbidden lists enforced |
| No direct target/leakage columns used | **VERIFIED** | 0 forbidden features; data-dictionary roles enforced |
| Test split untouched by selection | **VERIFIED** | selection on validation; single frozen test scoring |
| Chronological validity | **NOT SUPPORTED** | no timestamps; no chronological split possible |
| Derived-target / label-reconstruction independence | **NOT SUPPORTED** for `ev_capacity`, `synthetic_class`; **EXPERIMENTAL/at risk** for `ev_soh`, `solid_tier` | identities and recovery probes above |
| External validation | **NOT SUPPORTED** | no external dataset was scored |

**Overall:** the split and preprocessing *mechanisms* are correctly implemented and audited. The scientific claim they support is weaker than the label "entity-grouped" suggests, because the EV and solid-state identifiers are row-unique synthetic keys. Entity-level generalization on those two datasets is **UNVERIFIED**, and the derived/near-derived targets limit what the high R² values mean.
