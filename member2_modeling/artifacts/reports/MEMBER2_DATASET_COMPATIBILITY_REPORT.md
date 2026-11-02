# Member 2 Dataset Compatibility Report

Generated from `DATA/processed/` registry, inventory, data dictionary and
the actual Parquet schemas. No row order is interpreted as battery history.

**Active unified master table:** none (`active_master_table = null`).

## Observed RUL availability

Observed RUL is unavailable for all three datasets. The solid-state
`projected_cycle_life_80pct_soh` column and the EV
`predicted_remaining_life_cycles` column are projections, not observed
end-of-life trajectories, and are never used as RUL labels.

## Dataset summary

| Dataset | Rows | Grain | Sequence data | Permitted targets |
|---|---:|---|---|---|
| `EV_FAILURE_PREDICTION` | 20000 | vehicle_pack_snapshot | no (snapshot-only) | battery_failure, state_of_health, battery_health_percent, remaining_capacity |
| `SOLID_STATE_TELEMETRY_2026` | 10000 | cell_snapshot | no (snapshot-only) | state_of_health_pct, degradation_risk_tier |
| `SYNTHETIC_BATTERY_LIFECYCLE` | 2000 | synthetic_battery_snapshot | no (snapshot-only) | state_of_health, discharge_capacity_ah |

## Target details

### `EV_FAILURE_PREDICTION`

Source: `ev battery_failure prediction Dataset.csv`
SHA-256: `3b971b47e6dba73eef7edd60c7841332c5c34be17b649b9b5f036001315b28ea`
Chemistry labels: LFP, LMO, LTO, NCA, NMC

| Target | Task | Training | Readiness | RUL |
|---|---|---|---|---|
| `battery_failure` | classification | yes | EXPERIMENTAL | UNAVAILABLE |
| `state_of_health` | regression | yes | EXPERIMENTAL | UNAVAILABLE |
| `battery_health_percent` | regression | yes | EXPERIMENTAL | UNAVAILABLE |
| `remaining_capacity` | regression | yes | EXPERIMENTAL | UNAVAILABLE |
| `predicted_remaining_life_cycles` | projection_not_rul | no | UNAVAILABLE | UNAVAILABLE |

Leakage-excluded columns: `battery_health_percent`, `capacity_loss_percent`, `predicted_remaining_life_cycles`, `remaining_capacity`, `state_of_health`

### `SOLID_STATE_TELEMETRY_2026`

Source: `solid_state_ev_battery_telemetry_enriched_2026.csv`
SHA-256: `0fc38a2d4fe6289d712d85112f62d928c78419e562f2ebd9a2f94583489cd690`
Chemistry labels: Solid-State

| Target | Task | Training | Readiness | RUL |
|---|---|---|---|---|
| `state_of_health_pct` | regression | yes | EXPERIMENTAL | UNAVAILABLE |
| `degradation_risk_tier` | classification | yes | EXPERIMENTAL | UNAVAILABLE |
| `projected_cycle_life_80pct_soh` | projection_not_rul | no | UNAVAILABLE | UNAVAILABLE |

Leakage-excluded columns: `confidence_score`, `cot_electrochemical_risk_rationale`, `degradation_risk_tier`, `projected_cycle_life_80pct_soh`, `state_of_health_pct`

### `SYNTHETIC_BATTERY_LIFECYCLE`

Source: `synthetic_battery_lifecycle.csv`
SHA-256: `4a754a7937f564375611a080dac9b6b3feca97e1f8264de5b30b7f81aed7d224`
Chemistry labels: Lithium-Sulfur, Sodium-ion, Solid-State

| Target | Task | Training | Readiness | RUL |
|---|---|---|---|---|
| `state_of_health` | classification | yes | RESEARCH_ONLY | UNAVAILABLE |
| `discharge_capacity_ah` | regression | yes | RESEARCH_ONLY | UNAVAILABLE |

Leakage-excluded columns: `discharge_capacity_ah`, `state_of_health`

Synthetic labels are not measured experimental ground truth. Solid-state records are mixed electrolyte/cathode metadata and are not assumed to be genuine solid-state cells.
