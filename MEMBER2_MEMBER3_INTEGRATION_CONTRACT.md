# Member 2 → Member 3 Integration Contract (Snapshot Models)

**Contract version:** `2.0.0-snapshot-draft`
**Audit date:** 2026-11-02
**Status:** EXPERIMENTAL / RESEARCH_ONLY only. `production_ready = false` everywhere.
**Bundles on disk:**

| Bundle | Path | Contents | Consumer status |
|---|---|---|---|
| Legacy cycle bundle | `member2_modeling/artifacts/member3_exports/` | `observed_soh.csv`, `predicted_soh.csv`, `baseline_soh.csv`, `range_from_*.csv`, `PROVENANCE.json` (95 rows each, `LEGACY_NMC_SEQUENTIAL`) | Compatible with `member3/integration/member2_range_adapter.py` |
| Snapshot bundle v2 | `member2_modeling/artifacts/member3_exports_v2/` | `ev_soh_predictions.csv`, `solid_state_soh_predictions.csv`, `ev_failure_predictions.csv`, `ev_range_estimate.csv`, `SNAPSHOT_PREDICTION_CONTRACT.json`, `PROVENANCE.json`, `VALIDATION.json`, `COMPATIBILITY.md` | **Not compatible** with the existing cycle adapter; requires a new snapshot consumer |

This document describes the snapshot bundle. It does not modify `member3/`, and no Member 3 file was touched.

---

## 1. What Member 3 can consume today

Four CSV files, all rows from the frozen test split of the respective models:

| File | Rows | Task | Model ID | Status |
|---|---:|---|---|---|
| `ev_soh_predictions.csv` | 3,809 | `ev_soh` | `member2::ev_soh::2.0.0` | EXPERIMENTAL |
| `solid_state_soh_predictions.csv` | 2,000 | `solid_soh` | `member2::solid_soh::2.0.0` | EXPERIMENTAL |
| `ev_failure_predictions.csv` | 4,000 | `ev_failure` | `member2::ev_failure::2.0.0` | EXPERIMENTAL |
| `ev_range_estimate.csv` | 3,687 | formula estimate from `ev_soh` | `member2::ev_soh::2.0.0` | Calculated estimate |

Bundle validation (`validate_bundle`) result:

```json
{"ok": true, "problems": [],
 "rows": {"ev_soh_predictions.csv": 3809,
          "solid_state_soh_predictions.csv": 2000,
          "ev_failure_predictions.csv": 4000,
          "ev_range_estimate.csv": 3687}}
```

Registry verification immediately before this audit: **42 artifact files checked, 0 problems**.

---

## 2. Output schemas

### 2.1 SOH files (`ev_soh_predictions.csv`, `solid_state_soh_predictions.csv`)

| Column | dtype | Unit / values | Meaning |
|---|---|---|---|
| `snapshot_row_id` | int64 | 1-based source row | Verified to join back to `source_row_id` in the processed dataset (100% entity match) |
| `entity_id` | string | `vehicle_id` / `cell_id` | Row-unique synthetic identifier; **not a confirmed physical unit** |
| `dataset_id` | string | `EV_FAILURE_PREDICTION` / `SOLID_STATE_TELEMETRY_2026` | Source dataset |
| `chemistry` | string | see §5 | Source chemistry label |
| `observed_soh_pct` | float64 | percent | Source label (`state_of_health` / `state_of_health_pct`) |
| `predicted_soh_pct` | float64 | percent | Model point prediction |
| `interval_90_low` | float64 | percent | Point ± validation-residual q90 (EV ±5.8129; solid ±1.2708) |
| `interval_90_high` | float64 | percent | Same |
| `unit` | string | `%` | Fixed |
| `model_id` | string | `member2::<task>::2.0.0` | Model identity |
| `model_status` | string | `EXPERIMENTAL` | Registry status |
| `validation_status` | string | `grouped_holdout_evaluated_not_production_validated` | Honest label |
| `split` | string | `test` | Frozen test split |
| `quality_flag` | string | e.g. `OK`, `MISSING_FIELDS;SOH_LABEL_DISAGREEMENT` | Source quality flags |

### 2.2 Failure file (`ev_failure_predictions.csv`)

| Column | dtype | Unit / values | Meaning |
|---|---|---|---|
| `snapshot_row_id` | float64 | **all null (defect F-01)** | Intended source row id; see §9 |
| `entity_id` | string | `vehicle_id` | Row-unique identifier; 4,000 unique values, so row alignment is still possible via `entity_id` |
| `dataset_id` | string | `EV_FAILURE_PREDICTION` | Source |
| `chemistry` | string | see §5 | Source chemistry |
| `observed_failure` | int64 | 0/1 | Source label |
| `predicted_failure_probability` | float64 | 0–1, uncalibrated | Positive-class probability |
| `predicted_class` | int64 | 0/1 | Argmax at 0.5 with balanced class weights; positive rate 13.25% vs observed 6.93% |
| `model_id` | string | `member2::ev_failure::2.0.0` | Model identity |
| `model_status` | string | `EXPERIMENTAL` | Status |
| `validation_status` | string | `grouped_holdout_evaluated_not_production_validated` | Honest label |
| `split` | string | `test` | Frozen test split |
| `quality_flag` | string | source flags | Quality flags |

**Consumer guidance:** use `predicted_failure_probability`, not `predicted_class`. The default 0.5 threshold over-selects the positive class (13.25% vs 6.93% observed). Any threshold used for a decision must be chosen by the consumer and validated on its own population.

### 2.3 Range file (`ev_range_estimate.csv`)

| Column | dtype | Unit / values | Meaning |
|---|---|---|---|
| `snapshot_row_id` | int64 | source row | Joins to the EV source record |
| `entity_id` | string | `vehicle_id` | Row-unique |
| `chemistry` | string | see §5 | Source chemistry |
| `battery_capacity_kwh` | float64 | kWh | Rated pack energy from the source record |
| `soh_pct` | float64 | percent | **Model-predicted** SOH (`soh_source = ev_soh_model_prediction`) |
| `soh_source` | string | `ev_soh_model_prediction` | Provenance of the SOH input |
| `range_km` | float64 | km | **Calculated** central estimate |
| `range_km_optimistic` | float64 | km | 0.15 kWh/km scenario |
| `range_km_conservative` | float64 | km | 0.22 kWh/km scenario |
| `unit` | string | `km` | Fixed |
| `kind` | string | `calculated_estimate` | Never measured |
| `is_measured` | bool | `False` | Fixed |
| `validated_accuracy` | bool | `False` | Fixed |
| `validation_status` | string | `not_validated_no_range_target` | Fixed |
| `consumption_kwh_per_km` | float64 | kWh/km | `0.18` central assumption |

The formula is:

```
usable_energy_kwh = rated_capacity_kwh × (soh_pct / 100) × (discharge_efficiency / 100)
range_km          = usable_energy_kwh / consumption_kwh_per_km
```

`discharge_efficiency` comes from the source record when finite, otherwise 100%. The default `discharge_efficiency = 100%` is an assumption, not a measurement. 122 of 3,809 EV test rows are absent from the range file because `battery_capacity_kwh` is null.

**Range is a calculated estimate, not a measured or model-predicted range.** No measured range or consumption target exists in any active dataset, so accuracy cannot be validated.

---

## 3. Input schemas

Exact schemas are frozen in each task's `schema.json` and in `model_registry.json` → `required_features`.

| Task | Required features | Chemistry column | Unit of output |
|---|---:|---|---|
| `ev_soh` | 56 | `battery_chemistry` | percent |
| `ev_capacity` | 56 | `battery_chemistry` | kWh pack energy |
| `ev_failure` | 56 | `battery_chemistry` | failure probability |
| `solid_soh` | 19 | `electrolyte_chemistry` | percent |
| `solid_tier` | 19 | `electrolyte_chemistry` | risk tier class |
| `synthetic_capacity` | 5 | `electrolyte_type` | Ah (synthetic) |
| `synthetic_class` | 5 | `electrolyte_type` | Healthy / Degraded / End_of_Life |

### 3.1 EV feature set (56, shared by `ev_soh`, `ev_capacity`, `ev_failure`)

Numeric (48): `manufacturing_year`, `battery_capacity_kwh`, `odometer_km`, `vehicle_age_years`, `cycle_count`, `state_of_charge`, `depth_of_discharge`, `cell_voltage_avg`, `cell_voltage_std`, `pack_voltage`, `cell_temperature_avg`, `cell_temperature_max`, `charge_efficiency`, `discharge_efficiency`, `charging_cycles_last_month`, `fast_charge_ratio`, `slow_charge_ratio`, `average_charge_power_kw`, `average_charging_time`, `overnight_charging_ratio`, `home_charging_ratio`, `charging_interruptions`, `overcharge_events`, `average_speed`, `average_trip_distance`, `aggressive_acceleration_score`, `hard_braking_score`, `regenerative_braking_usage`, `highway_driving_ratio`, `city_driving_ratio`, `daily_distance`, `average_ambient_temperature`, `maximum_temperature`, `minimum_temperature`, `humidity`, `altitude`, `dust_exposure`, `last_service_days`, `cooling_system_health`, `firmware_updates`, `previous_faults`, `maintenance_score`, `thermal_runaway_risk`, `voltage_imbalance`, `temperature_variance`, `sensor_fault_count`, `BMS_warning_count`, `abnormal_voltage_events`.

Categorical (8): `vehicle_brand`, `vehicle_model`, `vehicle_type`, `battery_manufacturer`, `battery_chemistry`, `drive_type`, `fleet_or_private`, `terrain_type`.

Documented units for the important numerics: `battery_capacity_kwh` kWh; voltages V; temperatures degC; efficiencies percent; `thermal_runaway_risk` score 0–100; `humidity` percent; `altitude` m; `average_speed` km/h; distances km.

### 3.2 Solid-state feature set (19, shared by `solid_soh`, `solid_tier`)

Numeric (13): `nominal_capacity_ah` (Ah), `gravimetric_energy_density_wh_kg` (Wh/kg), `volumetric_energy_density_wh_l` (Wh/L), `cycle_number` (cycles, snapshot value), `charging_c_rate` (C), `discharging_c_rate` (C), `operating_temperature_c` (degC), `ambient_temperature_c` (degC), `internal_resistance_mohm` (mOhm), `coulombic_efficiency_pct` (percent), `stack_pressure_mpa` (MPa), `thermal_runaway_onset_c` (degC), `nominal_voltage_plateau_v` (V).

Categorical (6): `manufacturer`, `electrolyte_chemistry`, `anode_composition`, `cathode_chemistry`, `target_ev_application`, `commercial_readiness_level`.

### 3.3 Synthetic feature set (5, shared by both synthetic tasks)

Numeric (4): `cycle_count` (cycles), `internal_resistance_mohms` (mOhm), `temperature_celsius` (degC), `charge_time_mins` (minutes).
Categorical (1): `electrolyte_type`.

---

## 4. Model identity and provenance

| Field | Value |
|---|---|
| Registry | `member2_modeling/artifacts/snapshot_ml/model_registry.json` |
| Registry SHA-256 (at bundle build) | `94e9f9bf24b067a527e2612d24bf186d906ea1e467ae7abacc9e018692176f05` |
| Model version | `2.0.0` |
| Production ready | `false` (all tasks) |
| Artifacts per task | `config.json`, `schema.json`, `metrics.json`, `model.joblib`, `feature_pipeline.joblib`, `test_predictions.parquet` (42 files, all checksum-verified) |
| Dataset provenance | `PROVENANCE.json` records processed paths and processed SHA-256 for EV and solid-state |
| Registry verification | `registry_verification.json` → `ok = true`, 42 checked, 0 problems |
| Member 3 modified | `false` |

The bundle was built from the frozen registry; no model was retrained for this audit.

---

## 5. Supported chemistries

| Bundle file | Training vocabulary | Unseen chemistry behaviour |
|---|---|---|
| `ev_soh_predictions.csv` | `LFP`, `LMO`, `LTO`, `NCA`, `NMC`, `__MISSING__` | Inference raises `InferenceInputError` unless `allow_unseen_chemistry=True`, and warns when opted in |
| `solid_state_soh_predictions.csv` | `Sulfide_Based_Li10GeP2S12`, `Oxide_Based_LLZO`, `Polymer_Composite`, `Halide_Based_Li3YCl6`, `__MISSING__` | Same |
| `ev_failure_predictions.csv` | Same as EV SOH | Same |
| `ev_range_estimate.csv` | Same as EV SOH | Same |
| `synthetic_capacity.csv` (not exported) | `Solid_State`, `Lithium_Sulfur`, `Sodium_Ion` | Fails cross-chemistry (R² −7.60 / −7.16) |

The existing Member 3 cycle adapter accepts only `NMC` and `LFP`. The snapshot EV bundle contains `LMO`, `NCA`, `LTO` and missing chemistry as well; the solid-state bundle contains electrolyte chemistries the adapter does not accept at all.

---

## 6. Prediction semantics

| Output | Measured? | Predicted? | Derived? | What it is |
|---|---|---|---|---|
| `observed_soh_pct` / `observed_failure` | Yes (source label) | No | No | Supplied label, origin undocumented |
| `predicted_soh_pct` | No | Yes | No | Model output |
| `predicted_failure_probability` | No | Yes | No | Uncalibrated model probability |
| `ev_capacity` (not exported) | No | Yes | **Yes** | Algebraically derived target |
| `solid_tier` (not exported) | No | Yes | **Yes** | Rule-generated tier |
| `synthetic_*` (not exported) | No | Yes | **Yes** | Synthetic generation |
| `range_km` | No | No | **Yes** | Formula estimate under assumed consumption |

---

## 7. Why the snapshot bundle is not compatible with the existing adapter

`member3/integration/member2_range_adapter.py` requires exactly:

- `cycle_index` — integer cycle position
- `cell_id` — cell identifier
- `chemistry` — must be `NMC` or `LFP`
- `soh_pct` — 1 < SOH ≤ 100

The snapshot bundle:

1. has **no `cycle_index`**. All active datasets are snapshots; `sequence_available = false`. The only cycle-like columns (`cycle_count`, `cycle_number`) are per-record state counters, not an ordered history. Emitting an index would fabricate temporal structure.
2. names its identifier `entity_id` (and it is row-unique for EV/solid), not `cell_id`.
3. contains `LMO`, `NCA`, `LTO` and solid-state electrolyte chemistries that the adapter rejects.
4. has one row per record, not a trajectory per cell.

For these reasons `SNAPSHOT_PREDICTION_CONTRACT.json` sets `member3_adapter.compatible = false`. The existing legacy cycle bundle remains the only adapter-compatible export and is unchanged.

---

## 8. Safe consumption paths

### Option A — new snapshot consumer (recommended)

Member 3 adds a new file-based reader for the snapshot contract:

- validate the columns in `SNAPSHOT_PREDICTION_CONTRACT.json`;
- join on `snapshot_row_id` (SOH/range) or `entity_id` (failure until F-01 is fixed);
- treat `predicted_*` as experimental point estimates;
- treat `interval_90_*` as empirical residual intervals, not calibrated predictions;
- never interpret `entity_id` as a physical cell/vehicle unless the provenance is independently confirmed;
- never feed snapshot rows into the cycle adapter.

### Option B — use the range file directly

`ev_range_estimate.csv` can be consumed as a pre-computed range estimate. The consumer must preserve `kind = calculated_estimate`, `is_measured = False`, `validated_accuracy = False`. It must not claim measured range. Member 3's `RangeCalculator` additionally applies a vehicle profile; the snapshot range file does not include Member 3's vehicle configuration, so mixing the two would double-apply assumptions.

### Option C — call Member 2 inference live

`SnapshotInferenceAPI(registry_root).predict(task_id, records)` provides:

- strict required-feature checking;
- unseen-chemistry rejection with explicit opt-in;
- missing-value blocking or median/missing-category filling;
- out-of-training-range warnings;
- physical-bound flags for SOH (0–100) and capacity (≥0);
- `production_ready: false` in every response.

This is the safest path for new records, but it requires the caller to supply the exact 56- or 19-feature schema.

---

## 9. Defects and required authorizations

### F-01 (resolved, Member 2-owned): failure bundle loses `snapshot_row_id`

`member2_modeling/snapshot_ml/member3_bundle.py` now reads
`record.get("snapshot_row_id", record.get("source_row_id"))`, so the failure export keeps
the source row id. In `member3_exports_v2_fixed`, all 4,000 failure rows have a non-null,
unique `snapshot_row_id`. The historical pre-fix bundle remains at
`member3_exports_v2` and is not the current handover artifact. Regression tests assert
non-nullness and uniqueness, and `validate_bundle` rejects duplicate ids.

### F-02 (resolved, documentation): `remaining_capacity` definition

`remaining_capacity` is described in the source, registry, model config and generated
reports as derived algebraically from `battery_capacity_kwh × battery_health_percent /
100` (rounded to 2 decimals), not as a measurement. The description also states that
the derivation is verified only on rows where both inputs are present.

---

## 10. Missing capabilities

| Capability | Status | Reason |
|---|---|---|
| RUL prediction | **NOT AVAILABLE** | No observed EOL trajectory in active data |
| Next-cycle SOH | **NOT AVAILABLE** | No chronological sequences |
| Temporal forecasting | **NOT AVAILABLE** | Snapshot rows only |
| Measured-range validation | **NOT AVAILABLE** | No range/consumption target |
| External validation | **NOT AVAILABLE** | No second-source dataset |
| Calibrated probabilities | **NOT AVAILABLE** | No calibration layer fitted |
| Cross-chemistry guarantees | **NOT AVAILABLE** | Only in-distribution evaluation |
| Synthetic tasks in bundle | **INTENTIONALLY EXCLUDED** | RESEARCH_ONLY |
| Physical unit identity | **UNVERIFIED** | Synthetic row-unique IDs |
| Production use | **NOT APPROVED** | `production_ready = false` |

---

## 11. What Member 3 must not assume

1. That `entity_id` identifies a real vehicle or cell.
2. That two records for the same `entity_id` are a time series.
3. That `predicted_soh_pct` is a measured value.
4. That `range_km` is a measured or accuracy-validated range.
5. That `predicted_failure_probability` is calibrated, or that `predicted_class` is a validated decision threshold.
6. That performance transfers to new chemistries, new fleets, or a future time period.
7. That the bundle is adapter-compatible with `member2_range_adapter.py`.
8. That any output is production-ready.

---

## 12. Data required before real-world validation

To move any snapshot task beyond EXPERIMENTAL, the project needs:

1. **Confirmed physical identifiers** — real VIN/pack serial or cell serial with repeat observations per unit, so an entity-level holdout is meaningful.
2. **Record timestamps and repeated observations** — for chronological splits and any next-cycle/SOH-trajectory task.
3. **Independently measured targets** — measured capacity (not `battery_capacity_kwh × health%`), measured SOH from a reference test, verified failure events with an event date.
4. **Documented label-generation process** — for `battery_failure`, `degradation_risk_tier` and the synthetic labels.
5. **Measured range/consumption data** — for any range-accuracy claim.
6. **An external validation set** — different source, fleet, or lab, scored once after freezing.
7. **Calibration data** — a held-out calibration split for failure probabilities.
8. **Chemistry-specific sample sizes** — enough independent units per chemistry for per-chemistry models.

---

## 13. Change control

- Do not overwrite `member2_modeling/artifacts/member3_exports/` (legacy cycle bundle).
- Do not regenerate `member3_exports_v2/` without explicit authorization; F-01 is the only known defect and its fix is specified above.
- Do not modify anything under `member3/`.
- Any model re-freeze must regenerate `config.json`, `schema.json`, `metrics.json`, `model_registry.json`, `registry_verification.json` and both downstream bundles as one atomic change, and must not reuse the test split for selection.

**Contract verdict:** the snapshot bundle is internally consistent, checksum-verified and honestly labelled, but it is **not** a drop-in replacement for the legacy cycle bundle, and it carries one traceability defect (F-01). Member 3 can consume it only through a new snapshot-aware path and only as experimental data.
