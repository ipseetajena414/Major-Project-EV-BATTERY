# Member 2 Health Model Evaluation — Snapshot Datasets

Split protocol: deterministic stratified 60/20/20 snapshot split. These
datasets contain independent snapshots, so this estimates i.i.d.
generalization, not a temporal or vehicle-level forecast.

| Dataset | Target | Kind | Readiness | Test n | Model MAE | Median MAE | Chem-median MAE |
|---|---|---|---|---:|---:|---:|---:|
| `SOLID_STATE_TELEMETRY_2026` | `state_of_health_pct` | source_label | EXPERIMENTAL | 2000 | 0.6296 | 5.6391 | 5.6391 |
| `SYNTHETIC_BATTERY_LIFECYCLE` | `discharge_capacity_ah` | synthetic_label | RESEARCH_ONLY | 400 | 4.1541 | 15.9074 | 8.2823 |
| `EV_FAILURE_PREDICTION` | `state_of_health` | source_label | EXPERIMENTAL | 3809 | 2.7843 | 7.3387 | 7.3409 |
| `EV_FAILURE_PREDICTION` | `remaining_capacity` | derived_from_health_percent | EXPERIMENTAL | 3878 | 3.4397 | 26.1527 | 26.1726 |

## Per-chemistry test MAE

| Dataset | Target | Chemistry | n | MAE |
|---|---|---|---:|---:|
| `EV_FAILURE_PREDICTION` | `state_of_health` | LFP | 1134 | 2.7161 |
| `EV_FAILURE_PREDICTION` | `state_of_health` | LMO | 560 | 2.7037 |
| `EV_FAILURE_PREDICTION` | `state_of_health` | LTO | 245 | 2.8592 |
| `EV_FAILURE_PREDICTION` | `state_of_health` | NCA | 264 | 2.7699 |
| `EV_FAILURE_PREDICTION` | `state_of_health` | NMC | 1458 | 2.8302 |
| `EV_FAILURE_PREDICTION` | `remaining_capacity` | LFP | 1139 | 3.2761 |
| `EV_FAILURE_PREDICTION` | `remaining_capacity` | LMO | 573 | 3.4679 |
| `EV_FAILURE_PREDICTION` | `remaining_capacity` | LTO | 261 | 3.6312 |
| `EV_FAILURE_PREDICTION` | `remaining_capacity` | NCA | 275 | 3.9314 |
| `EV_FAILURE_PREDICTION` | `remaining_capacity` | NMC | 1500 | 3.4462 |
| `SOLID_STATE_TELEMETRY_2026` | `state_of_health_pct` | Solid-State | 2000 | 0.6296 |
| `SYNTHETIC_BATTERY_LIFECYCLE` | `discharge_capacity_ah` | Lithium-Sulfur | 138 | 4.4677 |
| `SYNTHETIC_BATTERY_LIFECYCLE` | `discharge_capacity_ah` | Sodium-ion | 134 | 3.7759 |
| `SYNTHETIC_BATTERY_LIFECYCLE` | `discharge_capacity_ah` | Solid-State | 128 | 4.2119 |

## Capacity semantics

### `EV_FAILURE_PREDICTION`

- Capacity column: remaining_capacity (kWh; algebraically derived, not measured)
- Predicted capacity available: True
- Algebraic derivation: SOH x `battery_capacity_kwh` (kWh)
- Note: remaining_capacity is derived pack energy in kWh (algebraically from battery_capacity_kwh and battery_health_percent), not cell Ah.
- Note: A capacity model trained on remaining_capacity predicts pack energy.
- Note: SOH x battery_capacity_kwh is an algebraic estimate, not a measurement.

### `SOLID_STATE_TELEMETRY_2026`

- Measured column: none (n/a)
- Predicted capacity available: False
- Algebraic derivation: SOH x `nominal_capacity_ah` (Ah)
- Note: No measured capacity column exists; only nominal_capacity_ah and SOH.
- Note: Capacity can only be derived algebraically from nominal capacity and SOH.

### `SYNTHETIC_BATTERY_LIFECYCLE`

- Measured column: discharge_capacity_ah (Ah)
- Predicted capacity available: True
- Algebraic derivation: none
- Note: discharge_capacity_ah is a synthetic measured label, not experimental truth.
- Note: No rated nominal capacity column exists, so no algebraic derivation is exposed.


## Not trained in this phase

- `EV_FAILURE_PREDICTION` / `battery_health_percent`: Duplicate SOH label; state_of_health is the primary audited target.
- `EV_FAILURE_PREDICTION` / `predicted_remaining_life_cycles`: Supplied projection, not an observed RUL label.
- `SOLID_STATE_TELEMETRY_2026` / `degradation_risk_tier`: Classification target; outside the Phase 4 SOH/capacity regression scope.
- `SOLID_STATE_TELEMETRY_2026` / `projected_cycle_life_80pct_soh`: Projected cycle life is not observed RUL.
- `SYNTHETIC_BATTERY_LIFECYCLE` / `state_of_health`: Categorical synthetic label; regression modules do not consume it.
