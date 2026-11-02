# Member 2 RUL Validation Report

> **Status note (2026-11-02):** this report records the pre-implementation audit.
> RUL estimation has since been implemented for genuine sequential data; the API now
> returns EXPERIMENTAL NMC predictions and one censored LFP projection instead of
> empty lists. The no-approval decision and all dataset-support findings remain in
> force. See `MEMBER2_RUL_IMPLEMENTATION_REPORT.md` for current behavior and metrics.

Artifacts:
`member2_modeling/artifacts/rul/rul_availability.json`,
`member2_modeling/artifacts/rul/MEMBER2_RUL_TARGET_AUDIT.md`.
Source modules: `member2_modeling/rul/`.

## 1. Decision

**RUL is never approved for operational use.** `rul_approved = false`.
`require_approved_rul(dataset_id)` raises `RULNotApprovedError` for every dataset.

The inference API returns `status = "unavailable"`, `approved_for_use = false`, and an
empty prediction list for every RUL request.

## 2. Dataset availability

| Dataset | Sequence | Status | Unit | Approved |
|---|---|---|---|---|
| `EV_FAILURE_PREDICTION` | No | UNAVAILABLE | — | No |
| `SOLID_STATE_TELEMETRY_2026` | No | UNAVAILABLE | — | No |
| `SYNTHETIC_BATTERY_LIFECYCLE` | No | UNAVAILABLE | — | No |
| `LEGACY_NMC_SEQUENTIAL` | Yes | EXPERIMENTAL_PRE_EOL_ONLY | observed_cycles | No |

Snapshot datasets have no observed chronological end-of-life trajectory. Row order and
cycle identifiers do not establish RUL. `projected_cycle_life_80pct_soh` and
`predicted_remaining_life_cycles` are supplied projections, not observed RUL, and are
never used as labels.

## 3. Label conventions

Threshold rule: **strict `SOH < 80.0`** (a value of exactly 80.0 is not a crossing).
RUL is measured in observed record positions, not calendar time.

| Convention | Definition |
|---|---|
| `first` | first record below 80% SOH |
| `sustained3` | first record of 3 consecutive records below 80% SOH |

`add_rul_labels()` returns `rul_cycles` (`Float64`), `is_post_threshold` (`boolean`),
and `crossing_position` (`Int64`). Cells that never cross receive NA for all three —
never zero. Pre-threshold (`RUL > 0`) and post-threshold (`RUL == 0`) rows are never
pooled.

## 4. Legacy target audit

Source: `master_table.parquet` (SHA-256
`6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb`),
NASA Ames PCoE, cells `B0005`, `B0006`, `B0018`.

### First-crossing convention

| Cell | Crossing cycle ID | Crossing position | Observed cycles | Pre | Post | Cycles above 80% after crossing |
|---|---:|---:|---:|---:|---:|---:|
| B0005 | 75 | 74 | 168 | 74 | 94 | 1 |
| B0006 | 63 | 62 | 168 | 62 | 106 | 0 |
| B0018 | 45 | 44 | 132 | 44 | 88 | 13 |
| **Total** | | | 468 | **180** | **288** | |

No censored cells.

### Sustained-3 convention

| Cell | Crossing cycle ID | Crossing position | Observed cycles | Pre | Post | Cycles above 80% after crossing |
|---|---:|---:|---:|---:|---:|---:|
| B0005 | 75 | 74 | 168 | 74 | 94 | 1 |
| B0006 | 63 | 62 | 168 | 62 | 106 | 0 |
| B0018 | 59 | 58 | 132 | 58 | 74 | 0 |
| **Total** | | | 468 | **194** | **274** | |

No censored cells.

B0018 is the key ambiguity: it first crosses at position 44, returns above 80% for
13 later cycles, and only sustains a crossing at position 58 (cycle 59).

## 5. Existing validation evidence (read-only)

Path: `member2_modeling/experiments/results/rul/nmc_rul_summary.json`
SHA-256: `2efeddba737ea9d1d72bed9a7744319b60b8422f2d50be3e8b6ebe817f76f974`

| Metric | Value |
|---|---:|
| Validation protocol | leave-one-cell-out; pre-threshold rows primary |
| Primary pre-EOL MAE | 6.7822 |
| Primary pre-EOL RMSE | 9.1232 |
| Primary pre-EOL R² | 0.7761 |
| Primary pre-EOL rows | 180 |
| Sustained-3 pre-EOL MAE | 9.6523 |
| Sustained-3 pre-EOL rows | 194 |
| Near-EOL MAE (LightGBM, fixed 200) | 2.5445 |
| Near-EOL MAE (last-fade persistence) | 2.1162 |
| Production test pre-threshold rows | 0 |
| Shipped model pre-EOL MAE (in-sample) | 0.0822 |

## 6. Caveats recorded with the evidence

1. 80% is a project-defined threshold, not a manufacturer EOL definition.
2. SOH is non-monotonic: B0018 returns above 80% for 13 cycles after its first crossing.
3. The production test split is 100% post-threshold, so its RUL target has zero variance.
4. The shipped model was trained on all three cells; its pre-EOL accuracy is in-sample.
5. Only three cells exist, so LOCO trains on two cells and tests on one.

## 7. Evaluation helper semantics

`evaluate_rul(truth, predictions, near_eol_threshold=5.0)` reports:

- counts: pre-threshold, post-threshold, near-EOL;
- separate `overall`, `pre_threshold`, `post_threshold`, and `near_eol` metrics;
- buckets `1-5`, `6-10`, `11-30`, `31+` assigned by **predicted** RUL among
  pre-threshold rows only;
- an explicit `pooling_rule` stating that pre- and post-threshold rows are never pooled.

Post-threshold rows are excluded from buckets because their true RUL is zero and cannot
be ranked.
