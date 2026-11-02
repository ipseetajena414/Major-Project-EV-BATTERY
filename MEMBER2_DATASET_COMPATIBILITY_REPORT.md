# Member 2 Dataset Compatibility Report

Machine-readable source: `member2_modeling/artifacts/reports/member2_dataset_compatibility.json`
Human-readable mirror: `member2_modeling/artifacts/reports/MEMBER2_DATASET_COMPATIBILITY_REPORT.md`
Generated from the Member 1 processed parquet files; raw data is never modified.

## 1. Summary

| Dataset | Rows | Columns | Origin | Record grain | Sequence | Training | Readiness |
|---|---:|---:|---|---|---|---|---|
| `EV_FAILURE_PREDICTION` | 20,000 | 82 | source_dataset | vehicle pack snapshot | No | Permitted | EXPERIMENTAL |
| `SOLID_STATE_TELEMETRY_2026` | 10,000 | 41 | source_dataset | cell snapshot | No | Permitted | EXPERIMENTAL |
| `SYNTHETIC_BATTERY_LIFECYCLE` | 2,000 | 20 | synthetic | synthetic battery snapshot | No | Permitted | RESEARCH_ONLY |

All three files have **zero duplicate full rows**. No artificial unified master table
exists; `active_master_table` is `null` by design.

## 2. EV_FAILURE_PREDICTION

Source: `ev battery_failure prediction Dataset.csv`
Processed SHA-256: `6b4f9035f50d54d72c6e653a5f1f6676b69896ba4b883e48ad1c6f1e5f57dc9d`
Source SHA-256: `3b971b47e6dba73eef7edd60c7841332c5c34be17b649b9b5f036001315b28ea`

| Target | Task | Label origin | Unit | Training | RUL status | Readiness |
|---|---|---|---|---|---|---|
| `battery_failure` | classification | source_dataset | — | Permitted | UNAVAILABLE | EXPERIMENTAL |
| `state_of_health` | regression | source_dataset | % | Permitted | UNAVAILABLE | EXPERIMENTAL |
| `battery_health_percent` | regression | source_dataset | % | Permitted | UNAVAILABLE | EXPERIMENTAL (not trained: duplicate SOH label) |
| `remaining_capacity` | regression | derived_from_health_percent | kWh (pack energy, not cell Ah) | Permitted | UNAVAILABLE | EXPERIMENTAL |
| `predicted_remaining_life_cycles` | projection, not RUL | source_dataset | cycles (projected) | **Not permitted** | UNAVAILABLE | UNAVAILABLE |

- Chemistry labels: `LFP`, `LMO`, `LTO`, `NCA`, `NMC`.
- `remaining_capacity` is derived algebraically as
  `battery_capacity_kwh × battery_health_percent / 100` (rounded to 2 decimals); it is
  not an independent measurement. The derivation is verified only on rows where both
  inputs are present, not on rows with a null input.
- Identifiers: `vehicle_id` and `battery_serial` are unique for all 20,000 rows.
- `cycle_count` is a snapshot attribute, **not** a chronological sequence index.
- Leakage columns: `state_of_health`, `battery_health_percent`,
  `capacity_loss_percent`, `remaining_capacity`, `predicted_remaining_life_cycles`.
- Missingness: every column has roughly 3–5% nulls; no column is fully populated
  except identifiers and provenance fields.
- Suitability: experimentation ✅, validation ✅, augmentation ❌, inference ❌
  (snapshot data cannot represent a vehicle's future trajectory).

## 3. SOLID_STATE_TELEMETRY_2026

Source: `solid_state_ev_battery_telemetry_enriched_2026.csv`
Processed SHA-256: `b49529ccffebe735be022e306339596d490ad717d61960a1ab1b71c43cc87513`
Source SHA-256: `0fc38a2d4fe6289d712d85112f62d928c78419e562f2ebd9a2f94583489cd690`

| Target | Task | Label origin | Unit | Training | RUL status | Readiness |
|---|---|---|---|---|---|---|
| `state_of_health_pct` | regression | source_dataset | % | Permitted | UNAVAILABLE | EXPERIMENTAL |
| `degradation_risk_tier` | classification | source_dataset | — | Permitted | UNAVAILABLE | EXPERIMENTAL (not trained in Phase 4) |
| `projected_cycle_life_80pct_soh` | projection, not RUL | source_dataset | cycles (projected) | **Not permitted** | UNAVAILABLE | UNAVAILABLE |

- Chemistry label: `Solid-State`.
- Identifier `cell_id` is unique for all 10,000 rows.
- `cycle_number` is a snapshot attribute, not a chronological sequence index.
- Leakage columns: `state_of_health_pct`, `degradation_risk_tier`,
  `projected_cycle_life_80pct_soh`, `cot_electrochemical_risk_rationale`,
  `confidence_score`.
- Capacity: no measured capacity column exists. Only the algebraic derivation
  `nominal_capacity_ah × SOH` is exposed.
- Suitability: experimentation ✅, validation ✅, augmentation ❌, inference ❌.

## 4. SYNTHETIC_BATTERY_LIFECYCLE

Source: `synthetic_battery_lifecycle.csv`
Processed SHA-256: `9f7e8ff595499bdc7a70bf29e7cc3d975db4fa8ca57b41236123350496bf6238`
Source SHA-256: `4a754a7937f564375611a080dac9b6b3feca97e1f8264de5b30b7f81aed7d224`

| Target | Task | Label origin | Unit | Training | RUL status | Readiness |
|---|---|---|---|---|---|---|
| `discharge_capacity_ah` | regression | synthetic | Ah (synthetic) | Permitted | UNAVAILABLE | RESEARCH_ONLY |
| `state_of_health` | classification | synthetic | — | Permitted | UNAVAILABLE | RESEARCH_ONLY (not trained as regression) |

- Chemistry labels: `Lithium-Sulfur`, `Sodium-ion`, `Solid-State`.
- Identifier `battery_id` has 1,978 unique values across 2,000 rows (22 repeated rows);
  duplicate `(battery_id)` rows are not a valid chronological history.
- `cycle_count` is a snapshot attribute, not a chronological sequence index.
- Leakage columns: `state_of_health`, `discharge_capacity_ah`.
- Suitability: experimentation ✅, augmentation ✅, validation ❌, inference ❌.
- Synthetic labels are **not measured experimental truth**.

## 5. Legacy sequential source

`LEGACY_NMC_SEQUENTIAL` is not a Member 1 processed dataset. It is derived from the
pre-existing `master_table.parquet` (SHA-256
`6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb`) and contains
NASA NMC cells `B0005`, `B0006`, `B0018` with per-cycle measurements.

- 278 model-ready next-record SOH pairs.
- Per-cell chronological 60/20/20 split: 91 train / 92 validation / 95 test rows.
- Target: SOH of the next observed cycle record (positional shift by 1); not
  necessarily the next physical cycle.
- Used only for the hybrid sequence experiment and Member 3 export.

## 6. Cross-dataset rules enforced

1. Snapshot row order is never treated as a chronological trajectory.
2. A target column is never used as an input feature.
3. Leakage columns are excluded from the feature set for the target being modeled.
4. Chemistry labels are validated against the dataset's own vocabulary.
5. `projected_cycle_life_80pct_soh` and `predicted_remaining_life_cycles` are
   projections, never RUL labels, and are never trained on.
6. Synthetic labels are reported as synthetic in every prediction response and report.
