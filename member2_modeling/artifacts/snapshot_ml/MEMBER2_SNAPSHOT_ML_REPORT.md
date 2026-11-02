# Member 2 snapshot ML report

Registry version: `2.0.0`

Protocol: entity-grouped, target-stratified 60/20/20 split. Preprocessing is
fitted on the training split only, models are selected on validation only, and
the test split is scored once after the configuration is frozen.

## Dataset utilization

| Dataset | Rows | Unique entities | Features | Targets modelled | Sequence |
|---|---:|---:|---:|---|---|
| `EV_FAILURE_PREDICTION` | 20,000 | 20,000 | 56 | `state_of_health`, `remaining_capacity`, `battery_failure` | no |
| `SOLID_STATE_TELEMETRY_2026` | 10,000 | 10,000 | 19 | `state_of_health_pct`, `degradation_risk_tier` | no |
| `SYNTHETIC_BATTERY_LIFECYCLE` | 2,000 | 1,978 | 6 | `discharge_capacity_ah`, `state_of_health` | no |

## Regression tasks

| Task | Selected model | Test n | MAE | RMSE | R2 | Bias | Train-median MAE | Chemistry-median MAE | Beats median |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ev_capacity` | hist_gradient_boosting | 3877 | 3.2091 | 5.3059 | 0.9699 | 0.0702 | 26.1151 | 26.1217 | yes |
| `ev_soh` | hist_gradient_boosting | 3809 | 2.7470 | 3.4766 | 0.8541 | -0.0890 | 7.3785 | 7.3873 | yes |
| `solid_soh` | lightgbm | 2000 | 0.6472 | 0.8126 | 0.9853 | -0.0216 | 5.6745 | 5.6744 | yes |
| `synthetic_capacity` | ridge | 400 | 3.8176 | 4.9945 | 0.9331 | 0.3630 | 16.2499 | 8.1490 | yes |

## Classification tasks

| Task | Selected model | Test n | Accuracy | Balanced acc | Macro F1 | Log loss | Majority bal. acc | Chemistry-majority bal. acc | Beats majority |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ev_failure` | logistic_regression | 4000 | 0.9233 | 0.9137 | 0.7884 | 0.1965 | 0.5000 | 0.5000 | yes |
| `solid_tier` | lightgbm | 2000 | 0.8995 | 0.9010 | 0.8994 | 0.2692 | 0.2500 | 0.3052 | yes |
| `synthetic_class` | logistic_regression | 400 | 0.7750 | 0.7932 | 0.6824 | 0.4988 | 0.3333 | 0.3333 | yes |

## Selected configuration per task

| Task | Status | Model | Key parameters | Split hash (first 16) |
|---|---|---|---|---|
| `ev_capacity` | EXPERIMENTAL | hist_gradient_boosting | learning_rate=0.06, max_iter=300, max_leaf_nodes=31 | `71eb79fbba34c741` |
| `ev_failure` | EXPERIMENTAL | logistic_regression | C=1.0, class_weight=balanced | `d05a6bfab31a60e0` |
| `ev_soh` | EXPERIMENTAL | hist_gradient_boosting | learning_rate=0.06, max_iter=300, max_leaf_nodes=31 | `9975be708285adab` |
| `solid_soh` | EXPERIMENTAL | lightgbm | learning_rate=0.05, min_child_samples=20, n_estimators=600, num_leaves=31 | `6556459fa29764d5` |
| `solid_tier` | EXPERIMENTAL | lightgbm | class_weight=balanced, learning_rate=0.05, n_estimators=600, num_leaves=31 | `e9a673b64debc339` |
| `synthetic_capacity` | RESEARCH_ONLY | ridge | alpha=1.0 | `be70bd4dc26b1e45` |
| `synthetic_class` | RESEARCH_ONLY | logistic_regression | C=1.0, class_weight=balanced | `09101e7cf3da85d9` |

## Per-chemistry test performance

### `ev_capacity`

| Chemistry | n | Primary metric |
|---|---:|---:|
| LFP | 1117 | 3.1616 |
| LMO | 579 | 3.1458 |
| LTO | 241 | 3.1497 |
| NCA | 264 | 3.5265 |
| NMC | 1538 | 3.2020 |
| __MISSING__ | 138 | 3.4341 |

### `ev_failure`

| Chemistry | n | Primary metric |
|---|---:|---:|
| LFP | 1161 | 0.9111 |
| LMO | 578 | 0.9519 |
| LTO | 273 | 0.9730 |
| NCA | 284 | 0.9611 |
| NMC | 1561 | 0.8876 |
| __MISSING__ | 143 | 0.9457 |

### `ev_soh`

| Chemistry | n | Primary metric |
|---|---:|---:|
| LFP | 1091 | 2.6540 |
| LMO | 538 | 2.7611 |
| LTO | 251 | 2.8948 |
| NCA | 288 | 2.6981 |
| NMC | 1500 | 2.7962 |
| __MISSING__ | 141 | 2.7272 |

### `solid_soh`

| Chemistry | n | Primary metric |
|---|---:|---:|
| Halide_Based_Li3YCl6 | 236 | 0.6205 |
| Oxide_Based_LLZO | 494 | 0.6243 |
| Polymer_Composite | 386 | 0.6386 |
| Sulfide_Based_Li10GeP2S12 | 884 | 0.6709 |

### `solid_tier`

| Chemistry | n | Primary metric |
|---|---:|---:|
| Halide_Based_Li3YCl6 | 250 | 0.9302 |
| Oxide_Based_LLZO | 460 | 0.9026 |
| Polymer_Composite | 391 | 0.9023 |
| Sulfide_Based_Li10GeP2S12 | 899 | 0.8813 |

### `synthetic_capacity`

| Chemistry | n | Primary metric |
|---|---:|---:|
| Lithium_Sulfur | 138 | 3.9893 |
| Sodium_Ion | 131 | 3.6191 |
| Solid_State | 131 | 3.8352 |

### `synthetic_class`

| Chemistry | n | Primary metric |
|---|---:|---:|
| Lithium_Sulfur | 134 | 0.8112 |
| Sodium_Ion | 132 | 0.7524 |
| Solid_State | 134 | 0.8309 |


## Tasks not supported by the active data

- **next_cycle_soh** (UNAVAILABLE): All three active Member 1 datasets are snapshot tables. EV has one row per vehicle, solid-state one row per cell, and synthetic at most two conflicting snapshots per battery. No chronological cycle history exists.
- **rul** (UNAVAILABLE_ON_ACTIVE_DATA): No active dataset contains an observed degradation trajectory to end of life. projected_cycle_life_80pct_soh and predicted_remaining_life_cycles are supplied projections, not observed RUL labels.
- **ev_range** (CALCULATED_ESTIMATE_ONLY): No measured driving-range target exists in the EV dataset. Range can only be calculated from rated pack energy, SOH and an assumed consumption rate; it cannot be validated for accuracy against ground truth.
- **seq_chemistry_transfer** (UNAVAILABLE): Cross-chemistry transfer learning needs compatible feature schemas and enough entities per chemistry. The three snapshots use different grains and columns, so a single shared sequence model would be fabricated.

## Leakage controls

- Machine audit status: OK.
- Features come from the Member 1 data dictionary role `feature` only.
- Target and every audited leakage column are rejected by the feature pipeline.
- Splits are grouped by vehicle, cell or synthetic battery; overlap is recorded as zero.
- Imputation medians and categorical vocabularies are fitted on the training split.
- Test rows are never used for selection, early stopping, scaling or encoding.

## Artifact verification

- Tasks registered: 7
- Files hashed and checked: 42
- Checksum status: OK
