# Member 2 RUL Target Audit

RUL is **never approved** for operational use in this project. Snapshot
datasets are unavailable for RUL, and the legacy NMC evidence is
pre-EOL, experimental, and limited to three cells.

End-of-life threshold: **80.0% SOH**, strict (`SOH < 80.0`).

## Availability

| Dataset | Status | Approved | Unit |
|---|---|---|---|
| `EV_FAILURE_PREDICTION` | UNAVAILABLE | no | n/a |
| `SOLID_STATE_TELEMETRY_2026` | UNAVAILABLE | no | n/a |
| `SYNTHETIC_BATTERY_LIFECYCLE` | UNAVAILABLE | no | n/a |
| `LEGACY_NMC_SEQUENTIAL` | EXPERIMENTAL_PRE_EOL_ONLY | no | observed_cycles |

## Legacy NMC crossing audit

Positions are 0-based observed-record indices, not arithmetic cycle-ID
differences.

| Convention | Cell | Observed cycles | Crossing position | Crossing cycle | Pre | Post | Above EOL after crossing |
|---|---|---:|---:|---:|---:|---:|---:|
| first | B0005 | 168 | 74 | 75 | 74 | 94 | 1 |
| first | B0006 | 168 | 62 | 63 | 62 | 106 | 0 |
| first | B0018 | 132 | 44 | 45 | 44 | 88 | 13 |
| sustained3 | B0005 | 168 | 74 | 75 | 74 | 94 | 1 |
| sustained3 | B0006 | 168 | 62 | 63 | 62 | 106 | 0 |
| sustained3 | B0018 | 132 | 58 | 59 | 58 | 74 | 0 |

## Existing validation evidence (read-only reference)

Source: `member2_modeling/experiments/results/rul/nmc_rul_summary.json`
SHA-256: `2efeddba737ea9d1d72bed9a7744319b60b8422f2d50be3e8b6ebe817f76f974`
Protocol: leave-one-cell-out; pre-threshold rows primary

| Metric | Value |
|---|---:|
| Pre-EOL MAE (first crossing, n=180) | 6.7822 |
| Pre-EOL RMSE | 9.1232 |
| Pre-EOL R² | 0.7761 |
| Sustained-3 pre-EOL MAE (n=194) | 9.6523 |
| Near-EOL MAE, LightGBM | 2.5445 |
| Near-EOL MAE, last-fade persistence | 2.1162 |
| Production test pre-threshold rows | 0 |
| Shipped model in-sample pre-EOL MAE | 0.0822 |

Pre-threshold and post-threshold rows are reported separately and are
never pooled. The production test split contains only post-threshold
rows, so it cannot measure pre-EOL forecasting.

## Caveats

- 80% is a project-defined threshold, not a manufacturer EOL definition.
- SOH is non-monotonic: B0018 first crosses at cycle 45 but returns above 80% for 13 later cycles.
- The production test split is 100% post-threshold, so its RUL target has zero variance.
- The shipped model was trained on all three cells; its pre-EOL accuracy is in-sample only.
- Only three cells are available, so LOCO trains on two cells and tests on one.
