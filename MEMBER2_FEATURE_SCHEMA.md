# Member 2 Feature Schema

Schema version: `2.0.0` (health models, tabular pipeline, sequence builder).

## 1. Principles

- Features are fitted on the training split only. Medians and categorical
  vocabularies come from training data; a holdout transform never consults holdout
  statistics.
- The target column and every declared leakage column are forbidden as inputs.
- Categorical values are one-hot encoded against a training vocabulary. Unseen or
  missing values become `__MISSING__`, an explicit category present in every fitted
  vocabulary.
- Sequence windows are causal: a window ends at the target record and contains no
  future measurements.
- No sequence index is invented. Only the legacy NASA NMC table has real
  per-cycle ordering.

## 2. Tabular pipeline

`member2_modeling/features/tabular.py` — `TabularFeaturePipeline`

| Aspect | Behavior |
|---|---|
| Numeric imputation | training median; non-numeric values coerced to NaN then imputed |
| Categorical encoding | one-hot over sorted training vocabulary + `__MISSING__` |
| Forbidden inputs | target column plus declared leakage columns raise `FeatureSchemaError` |
| Output | `float64` matrix; `feature_names_out()` returns encoded names |
| Persistence | `to_schema()` stores medians, vocabularies, encoded names |

## 3. Health-model feature sets

Artifacts: `member2_modeling/artifacts/health/<DATASET_ID>/<TARGET>/model_schema.json`.

### 3.1 EV_FAILURE_PREDICTION (`state_of_health`, `remaining_capacity`)

56 encoded features (48 numeric + 8 categorical).

Numeric: `manufacturing_year`, `battery_capacity_kwh`, `odometer_km`,
`vehicle_age_years`, `cycle_count`, `state_of_charge`, `depth_of_discharge`,
`cell_voltage_avg`, `cell_voltage_std`, `pack_voltage`, `cell_temperature_avg`,
`cell_temperature_max`, `charge_efficiency`, `discharge_efficiency`,
`charging_cycles_last_month`, `fast_charge_ratio`, `slow_charge_ratio`,
`average_charge_power_kw`, `average_charging_time`, `overnight_charging_ratio`,
`home_charging_ratio`, `charging_interruptions`, `overcharge_events`,
`average_speed`, `average_trip_distance`, `aggressive_acceleration_score`,
`hard_braking_score`, `regenerative_braking_usage`, `highway_driving_ratio`,
`city_driving_ratio`, `daily_distance`, `average_ambient_temperature`,
`maximum_temperature`, `minimum_temperature`, `humidity`, `altitude`,
`dust_exposure`, `last_service_days`, `cooling_system_health`, `firmware_updates`,
`previous_faults`, `maintenance_score`, `thermal_runaway_risk`, `voltage_imbalance`,
`temperature_variance`, `sensor_fault_count`, `BMS_warning_count`,
`abnormal_voltage_events`.

Categorical: `vehicle_brand`, `vehicle_model`, `vehicle_type`, `battery_manufacturer`,
`battery_chemistry`, `drive_type`, `fleet_or_private`, `terrain_type`.

Excluded: `state_of_health`, `battery_health_percent`, `capacity_loss_percent`,
`remaining_capacity`, `predicted_remaining_life_cycles`.

Chemistry validation column: registry `chemistry_family` (values `LFP`, `LMO`, `LTO`,
`NCA`, `NMC`).

### 3.2 SOLID_STATE_TELEMETRY_2026 (`state_of_health_pct`)

19 encoded features (13 numeric + 6 categorical).

Numeric: `nominal_capacity_ah`, `gravimetric_energy_density_wh_kg`,
`volumetric_energy_density_wh_l`, `cycle_number`, `charging_c_rate`,
`discharging_c_rate`, `operating_temperature_c`, `ambient_temperature_c`,
`internal_resistance_mohm`, `coulombic_efficiency_pct`, `stack_pressure_mpa`,
`thermal_runaway_onset_c`, `nominal_voltage_plateau_v`.

Categorical: `manufacturer`, `electrolyte_chemistry`, `anode_composition`,
`cathode_chemistry`, `target_ev_application`, `commercial_readiness_level`.

Excluded: `state_of_health_pct`, `degradation_risk_tier`,
`projected_cycle_life_80pct_soh`, `cot_electrochemical_risk_rationale`,
`confidence_score`.

### 3.3 SYNTHETIC_BATTERY_LIFECYCLE (`discharge_capacity_ah`)

5 encoded features (4 numeric + 1 categorical).

Numeric: `cycle_count`, `internal_resistance_mohms`, `temperature_celsius`,
`charge_time_mins`.
Categorical: `electrolyte_type`.

Excluded: `discharge_capacity_ah`, `state_of_health`.

## 4. Hybrid sequence features (legacy NMC)

Artifact: `member2_modeling/artifacts/hybrid_nmc_legacy/hybrid_model/hybrid_schema.json`.

11 per-cycle features, in order:

1. `current_soh_pct`
2. `v_mean`
3. `v_min`
4. `v_end`
5. `v_slope`
6. `i_mean`
7. `ah_throughput`
8. `energy_wh`
9. `discharge_time_s`
10. `temp_mean`
11. `temp_range`

Entity: `cell_id`. Order: `cycle_id`. Target: `target_soh_pct`. Sequence length: 5.
Windows are causal and built per cell after sorting by `cycle_id`. The scaler is fitted
on training rows only; `target_soh_pct` never appears inside an input window.

## 5. Inference-time behavior

`BatteryIntelligenceAPI` follows the fitted schema:

- All required feature columns must be supplied or explicitly reported as missing and
  imputed. Partial rows return status `partial` with the missing-feature list.
- Target and leakage columns are rejected with `CustomInputError`.
- Numeric values must be finite numbers or numeric strings; booleans and nulls are
  rejected.
- The chemistry value must belong to the dataset's label vocabulary.
- SOH outputs are clipped to `[0, 100]`; capacity outputs are clipped to `>= 0`.
- Hybrid sequence inference defaults to `chemistry="NMC"` and a caller-provided
  `cell_id`. Rows with fewer than 5 history records receive the learned tabular-only
  fusion fallback and are flagged `sequence_available=false`.
