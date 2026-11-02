# Member 2 Model Evaluation Report

Sources:
`member2_modeling/artifacts/health/health_model_evaluation.json`,
`member2_modeling/artifacts/hybrid_nmc_legacy/hybrid_evaluation.json`,
`member2_modeling/artifacts/rul/rul_availability.json`.

**No model in this report is production-validated. `production_ready = false`.**

## 1. Health models (tabular LightGBM)

All health models use a deterministic stratified **snapshot** split (60/20/20).
This estimates i.i.d. generalization, not a temporal or vehicle-level forecast.

| Dataset | Target | Unit | Readiness | Test n | MAE | RMSE | R² | MAPE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| EV_FAILURE_PREDICTION | `remaining_capacity` | kWh (pack energy) | EXPERIMENTAL | 3,878 | 3.4397 | 5.5495 | 0.9671 | 5.46% |
| EV_FAILURE_PREDICTION | `state_of_health` | % | EXPERIMENTAL | 3,809 | 2.7843 | 3.5189 | 0.8484 | 3.49% |
| SOLID_STATE_TELEMETRY_2026 | `state_of_health_pct` | % | EXPERIMENTAL | 2,000 | 0.6296 | 0.7940 | 0.9857 | 0.73% |
| SYNTHETIC_BATTERY_LIFECYCLE | `discharge_capacity_ah` | Ah (synthetic) | RESEARCH_ONLY | 400 | 4.1541 | 5.4402 | 0.9180 | 5.71% |

### Baseline comparison (test split)

| Dataset | Target | Model MAE | Train-median MAE | Chemistry-median MAE | Beats both |
|---|---|---:|---:|---:|---|
| EV_FAILURE_PREDICTION | `remaining_capacity` | 3.4397 | 26.1527 | 26.1726 | Yes |
| EV_FAILURE_PREDICTION | `state_of_health` | 2.7843 | 7.3387 | 7.3409 | Yes |
| SOLID_STATE_TELEMETRY_2026 | `state_of_health_pct` | 0.6296 | 5.6391 | 5.6391 | Yes |
| SYNTHETIC_BATTERY_LIFECYCLE | `discharge_capacity_ah` | 4.1541 | 15.9074 | 8.2823 | Yes |

### Dropped null targets and rows

| Dataset / target | Rows used | Null targets dropped | Test rows |
|---|---:|---:|---:|
| EV `remaining_capacity` | 19,386 | 614 | 3,878 |
| EV `state_of_health` | 19,044 | 956 | 3,809 |
| Solid-state SOH | 10,000 | 0 | 2,000 |
| Synthetic capacity | 2,000 | 0 | 400 |

Dropped rows only have a null label; no non-null target is deleted.

### Not trained

| Dataset | Column | Reason |
|---|---|---|
| EV | `battery_health_percent` | duplicate SOH label; `state_of_health` is primary |
| EV | `predicted_remaining_life_cycles` | supplied projection, not observed RUL |
| Solid-state | `degradation_risk_tier` | classification, outside Phase 4 regression scope |
| Solid-state | `projected_cycle_life_80pct_soh` | projected cycle life, not observed RUL |
| Synthetic | `state_of_health` | categorical synthetic label, not consumed by regression |

## 2. Hybrid model (legacy NASA NMC)

Chronological per-cell holdout. Test split: 95 rows, 95 with sequence history.

| Predictor | MAE | RMSE | R² |
|---|---:|---:|---:|
| Fused hybrid | 6.1105 | 6.5007 | -2.4039 |
| Tabular branch | 8.8222 | 9.2874 | -5.9477 |
| Sequence branch | 7.1294 | — | — |
| Persistence | **0.5040** | 0.8669 | 0.9395 |
| Training-mean fade | **0.4138** | 0.8666 | 0.9395 |

Validation split (92 rows) shows the same ordering: fused 3.3822, tabular 4.4916,
persistence 0.4704, mean-fade 0.3700.

**Benchmark warning (verbatim):** "The persistence and training-mean-fade baselines
outperform the fused hybrid on the chronological test split. The fusion improves on
both learned branches but is not production-validated."

## 3. RUL

No RUL model is approved. Existing evidence is read-only and historical:

| Metric | Value |
|---|---:|
| Primary pre-EOL MAE (LOCO) | 6.7822 |
| Primary pre-EOL RMSE | 9.1232 |
| Primary pre-EOL R² | 0.7761 |
| Primary pre-EOL rows | 180 |
| Sustained-3 pre-EOL MAE | 9.6523 |
| Near-EOL MAE (LightGBM) | 2.5445 |
| Near-EOL MAE (persistence) | 2.1162 |
| Shipped model pre-EOL MAE (in-sample) | 0.0822 |
| Production test pre-threshold rows | 0 |

See `MEMBER2_RUL_VALIDATION_REPORT.md` for the full audit.

## 4. Chemistry-level evaluation

Per-chemistry metrics exist for every dataset and target; the best and worst chemistries
are summarized in `MEMBER2_CHEMISTRY_COMPARISON.md`.

## 5. Readiness statements

- Health models: EXPERIMENTAL (synthetic capacity: RESEARCH_ONLY).
- Hybrid: EXPERIMENTAL; worse than simple baselines.
- RUL: UNAVAILABLE for all snapshot datasets; EXPERIMENTAL_PRE_EOL_ONLY and never
  approved for the legacy table.
- All models: `production_ready = false`.
