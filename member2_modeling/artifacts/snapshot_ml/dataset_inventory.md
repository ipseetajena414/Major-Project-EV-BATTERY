# Member 2 dataset inventory

Generated from `DATA/processed/` and `member1_pipeline`; no legacy table is used.

| Dataset | Rows | Entity | Unique | Chemistry values | Features | Member 2 targets |
|---|---:|---|---:|---:|---:|---|
| `EV_FAILURE_PREDICTION` | 20,000 | `vehicle_id` | 20,000 | 5 | 56 | `state_of_health`, `remaining_capacity`, `battery_failure` |
| `SOLID_STATE_TELEMETRY_2026` | 10,000 | `cell_id` | 10,000 | 4 | 19 | `state_of_health_pct`, `degradation_risk_tier` |
| `SYNTHETIC_BATTERY_LIFECYCLE` | 2,000 | `battery_id` | 1,978 | 3 | 6 | `discharge_capacity_ah`, `state_of_health` |

## Sequence and RUL availability

- `EV_FAILURE_PREDICTION`: sequential=False; temporal_validation allowed=False; RUL allowed=False.
- `SOLID_STATE_TELEMETRY_2026`: sequential=False; temporal_validation allowed=False; RUL allowed=False.
- `SYNTHETIC_BATTERY_LIFECYCLE`: sequential=False; temporal_validation allowed=False; RUL allowed=False.

## Missing labels

- `EV_FAILURE_PREDICTION`: `state_of_health`=956, `remaining_capacity`=614, `battery_failure`=0
- `SOLID_STATE_TELEMETRY_2026`: `state_of_health_pct`=0, `degradation_risk_tier`=0
- `SYNTHETIC_BATTERY_LIFECYCLE`: `discharge_capacity_ah`=0, `state_of_health`=0
