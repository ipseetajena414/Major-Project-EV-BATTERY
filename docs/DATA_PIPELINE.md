# Member 1 data pipeline

Status: active. This document describes the datasets that replaced the retired
NASA/BatteryArchive-derived master table, the exact outputs in `DATA/processed/`,
the cleaning and leakage rules, and how downstream members should load the data.

## 1. Active datasets

| Dataset ID | Source file (repo root) | Raw rows | Raw cols | Processed cols | Origin | Grain |
|---|---|---:|---:|---:|---|---|
| `EV_FAILURE_PREDICTION` | `ev battery_failure prediction Dataset.csv` | 20,000 | 70 | 82 | source dataset | one snapshot per vehicle/pack |
| `SOLID_STATE_TELEMETRY_2026` | `solid_state_ev_battery_telemetry_enriched_2026.csv` | 10,000 | 29 | 41 | source dataset | one snapshot per cell |
| `SYNTHETIC_BATTERY_LIFECYCLE` | `synthetic_battery_lifecycle.csv` | 2,000 | 8 | 20 | synthetic | one snapshot per synthetic battery |

All three raw files are read-only inputs. Their SHA-256 hashes are recorded in
`DATA/processed/dataset_inventory.json` and re-checked after every build.

Every processed row carries 12 added columns: four provenance columns
(`source_row_id`, `dataset_id`, `data_origin`, `source_file`), `chemistry_family`,
and seven quality columns (`missing_field_count`, `missing_fields`,
`outlier_field_count`, `outlier_fields`, `invalid_field_count`, `invalid_fields`,
`quality_flag`). The full per-column schema, dtype, role, unit and missing count
is in `DATA/processed/data_dictionary.csv` (143 rows).

## 2. Processed outputs

| File | Purpose |
|---|---|
| `DATA/processed/ev_battery_failure_clean.parquet` | EV/pack snapshot dataset |
| `DATA/processed/solid_state_telemetry_clean.parquet` | Solid-state cell snapshot dataset |
| `DATA/processed/synthetic_battery_lifecycle_clean.parquet` | Synthetic lifecycle dataset |
| `DATA/processed/dataset_inventory.json` | Rows, columns, dtypes, missing, duplicates, IDs, chemistry, ranges, sequence analysis |
| `DATA/processed/data_quality_report.json` | Cleaning summary, coercions, flag counts, outliers, decisions |
| `DATA/processed/data_dictionary.csv` | Column contract with roles and units |
| `DATA/processed/leakage_audit.json` | Targets, identifiers, allowed features, excluded fields with evidence |
| `DATA/processed/dataset_registry.json` | Active registry, hashes, grain, sequence flags, legacy index |

There is no unified master table. `dataset_registry.json` sets
`active_master_table` to `null`.

## 3. Cleaning rules

The pipeline never deletes records and never imputes values.

- Raw files are opened read-only; no row is dropped.
- Numeric fields are coerced with `errors="coerce"`; any coercion is counted in
  `type_coercions` (currently zero for all three datasets).
- Declared integer fields use pandas nullable integer dtypes when every observed
  value is integral. If a field contains a fractional value, the whole column is
  kept as a nullable float, every fractional row is flagged `NON_INTEGER_VALUE`,
  and the field is listed in `invalid_fields`. Values are never rounded.
- Text fields use the pandas `string` dtype.
- Leading/trailing whitespace is stripped; empty strings become null.
- Missing values stay null and are counted in `missing_field_count` /
  `missing_fields`.
- Outliers are flagged with the 1.5 IQR rule (`OUTLIER_DETECTED`) and retained.
- Values outside configured physical ranges are flagged (`INVALID_VALUE`).
- Categorical values outside the declared source vocabulary are flagged
  (`CATEGORY_OUT_OF_VOCABULARY`).
- Chemistry labels are normalized into `chemistry_family`; the source label is
  preserved in its original column. Null and the literal strings `nan`, `none`,
  `null`, `<na>` map to null, not to a fake chemistry.
- `quality_flag` is `OK` when no anomaly was found, otherwise a semicolon-separated
  list of flags. Flag definitions are in `data_quality_report.json`.

Rows with `quality_flag == "OK"`: 623 of 20,000 (EV), 7,333 of 10,000
(solid-state), 1,951 of 2,000 (synthetic). The remaining rows are retained and
flagged.

Dataset-specific rules:

- **EV:** the two SOH labels are audited independently. Rows where only one is
  present are flagged `MISSING_ONE_SOH_LABEL`; rows where both differ by more
  than 1 percentage point are flagged `SOH_LABEL_DISAGREEMENT`; rows missing
  `cycle_count`, `pack_voltage` or `remaining_capacity` are flagged
  `MISSING_CORE_MEASUREMENT`.
- **Solid-state:** rows whose `projected_cycle_life_80pct_soh` is below the
  current `cycle_number` are flagged `PROJECTED_RUL_BEFORE_CURRENT_CYCLE`; rows
  with SOH at or below 80% are flagged `SOH_AT_OR_BELOW_80`.
- **Synthetic:** repeated `battery_id` values are flagged `ID_REPEATED`, and
  repeated IDs with conflicting electrolyte labels are flagged
  `ID_METADATA_CONFLICT`.

## 4. Targets and leakage

### Declared targets

| Dataset | Target | Task |
|---|---|---|
| EV | `battery_failure` | binary classification |
| Solid-state | `degradation_risk_tier` | 4-class classification; never a feature |
| Synthetic | `state_of_health` | 3-class classification (`Healthy`, `Degraded`, `End_of_Life`) |

### Candidate targets (audited, not auto-promoted)

| Dataset | Field | Evidence |
|---|---|---|
| EV | `state_of_health` | 956 missing; independent of `battery_health_percent` |
| EV | `battery_health_percent` | 898 missing; independent label |
| Solid-state | `state_of_health_pct` | 0 missing, range 70-100; candidate SOH regression target |

### Leakage exclusions

| Dataset | Excluded field | Reason |
|---|---|---|
| EV | `predicted_remaining_life_cycles` | Model prediction, not measured RUL; R² = 0.821 from cycle + SOH alone |
| EV | `capacity_loss_percent` | Exactly `100 - battery_health_percent` on all 18,192 paired rows |
| EV | `state_of_health` | SOH label; cannot be a feature when the other SOH label is the target |
| EV | `battery_health_percent` | SOH label; cannot be a feature when the other SOH label is the target |
| Solid-state | `degradation_risk_tier` | Target; strongly SOH-aligned |
| Solid-state | `cot_electrochemical_risk_rationale` | All 10,000 rows restate SOH, cycle and tier |
| Solid-state | `projected_cycle_life_80pct_soh` | Projected, not observed; below the current cycle for 1,904 rows (19.0%) |
| Solid-state | `confidence_score` | Undocumented enriched score; correlation with SOH 0.303 |
| Solid-state | `thermal_stability_margin_c` | Exactly `thermal_runaway_onset_c - operating_temperature_c` on all 10,000 rows |
| Solid-state | `dendrite_penetration_risk`, `fast_charge_acceptance_score`, `dendrite_mitigation_efficiency` | Enriched risk scores with undocumented derivation |

The two SOH labels are near-duplicates (correlation 0.990) but not identical:
only 63 of 18,200 paired rows are exactly equal, 8,008 differ by more than 1
point, and the maximum difference is 6.22. They must not be used together as
feature and target.

Solid-state `degradation_risk_tier` is strongly SOH-banded but not a pure
function of the band: Tier 1 is entirely 92-100 and Tier 2 entirely 85-92, while
Tier 3 and Tier 4 overlap (169 Tier-3 and 110 Tier-4 rows are at 92-100).

Allowed feature counts after exclusions: EV 56, solid-state 19, synthetic 6.
EV also has 7 `feature_candidate_*` fields that need a documented modelling
decision: `internal_resistance` and `remaining_capacity` (units/grain
ambiguous), plus `aging_score`, `battery_stress_index`, `thermal_health_score`,
`charging_quality_score`, `driving_stress_score` (undocumented composites).
`remaining_capacity` averages 0.256 of `battery_capacity_kwh * 1000 / pack_voltage`
and is not pack Ah.

## 5. Sequence availability

None of the three datasets supports sequence modelling:

- EV: 20,000 unique `vehicle_id` and 20,000 unique `battery_serial` values, one
  row each; `cycle_count` is a scalar snapshot.
- Solid-state: 10,000 unique `cell_id` values, one row each; `cycle_number` is a
  snapshot, not a history.
- Synthetic: 1,978 unique `battery_id` values across 2,000 rows; at most two
  snapshots per battery, and 11 repeated IDs carry conflicting electrolyte
  labels.

`loader.build_sequences()` raises `NoSequenceDataError` for every dataset. Do not
train a BiLSTM or any sequence model on these files without new time-series data.

## 6. Synthetic data limitations

`synthetic_battery_lifecycle.csv` is generated, not observed. Its
`state_of_health` column is categorical only. There is no numeric SOH and no
measured RUL, and the pipeline does not create either. Labels are strongly
associated with discharge capacity, internal resistance and cycle count, which
is consistent with generated thresholds. Keep these rows separated from source
datasets; use `data_origin == "synthetic"` or
`loader.load_synthetic()`.

## 7. Regeneration

```bash
.venv/Scripts/python.exe -m member1_pipeline.build            # build DATA/processed
.venv/Scripts/python.exe -m member1_pipeline.build --check    # validate only
.venv/Scripts/python.exe -m pytest tests -q                   # contract tests
```

`python build_master_table.py` is now a deprecation shim that delegates to the
pipeline above. It no longer writes `master_table.parquet`.

The build is deterministic: two builds produce byte-identical parquet files and
reports. It verifies that raw SHA-256 hashes are unchanged before and after the
build.

`--check` is not a row-count-only check. For each dataset it verifies:

- the current raw file hash equals the `source_sha256` recorded in both
  `dataset_inventory.json` and `dataset_registry.json`;
- the current processed file hash equals `processed_sha256` in the registry;
- the registry `processed_file` name matches the configured dataset identity;
- the processed `dataset_id` is constant and equals the expected dataset,
  `data_origin` matches the spec, and `source_row_id` is 1..N;
- every raw source column, provenance column, quality column, identifier,
  chemistry column and declared target is present;
- row/column counts, column order and column dtypes match the inventory.

## 8. Downstream loading

```python
from member1_pipeline import loader

ev = loader.load_ev_failure()
solid = loader.load_solid_state()
synthetic = loader.load_synthetic()

features = loader.feature_columns("EV_FAILURE_PREDICTION")
targets = loader.target_columns("EV_FAILURE_PREDICTION")
excluded = loader.excluded_columns("SOLID_STATE_TELEMETRY_2026")
```

Rules for Members 2-5:

- Load each dataset separately; do not merge vehicle, cell and synthetic
  snapshots into one table.
- Use `loader.feature_columns()` rather than hand-picking columns; it excludes
  targets, identifiers, provenance, quality and leakage fields.
- Use `loader.target_columns()` and decide explicitly which candidate SOH label
  to model.
- Never convert the synthetic categorical `state_of_health` into a numeric
  percentage or RUL.
- Do not train thermal-stress features across datasets: only the EV dataset has
  measured temperatures; solid-state and synthetic temperatures are not
  cell-level cycle histories.
- Read `dataset_registry.json` for row/column counts, hashes, grain and sequence
  availability.

Member 2 currently reads root `master_table.parquet` in
`member2_modeling/build_and_train.py`, and Member 5's `m5integration/paths.py`
points `MASTER_TABLE` at the same path. That file was the retired legacy table.
It is now archived; downstream code must switch to `member1_pipeline.loader` or
explicitly opt in with `loader.load_legacy_master_table(allow_legacy=True)` for
an audit. Their model, range, agent and dashboard code was not modified.

## 9. Legacy archive

Retired artifacts are in `legacy/member1/` with hashes and restore instructions.
The archived 8-column parquet had 23,768 rows; Na-ion voltage and current were
hardcoded constants and Na-ion SOH was derived from resistance. The archived
4,789-row CSV was a partial LFP B1-only build. Neither is an active input.

## 10. Verification results

- `python -m member1_pipeline.build`: 20,000 x 82, 10,000 x 41, 2,000 x 20; raw
  hashes unchanged.
- `python -m member1_pipeline.build --check`: `VALIDATION OK`.
- `pytest tests -q`: 28 passed.
- Processed row counts equal raw row counts; zero rows dropped; zero imputation.
