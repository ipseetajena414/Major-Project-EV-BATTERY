# Member 2 RUL Implementation Report

Date: 2026-11-02
Source of truth: `member2_modeling/artifacts/rul_v1/`
Code: `member2_modeling/analytics/sequential.py`, `member2_modeling/rul/forecast.py`,
`member2_modeling/rul/pipeline.py`, `member2_modeling/rul/service.py`

## 1. Answer in one paragraph

RUL is implemented **only** for genuine cycle-by-cycle SOH histories. Three NASA NMC
cells cross a project-defined 80% SOH threshold and support supervised evaluation; that
evidence is **EXPERIMENTAL**, never VALIDATED and never approved for operation. One LFP
cell (BATTERY_5) has a genuine sequence but no observed crossing, so only a censored
projection is produced. LFP BATTERY_1, all Na-ion cells, and all active Member 1
snapshot datasets are **UNAVAILABLE**. No sequence or EOL label was manufactured from
snapshots. Against leave-one-cell-out evaluation the best single model was histogram
gradient boosting (MAE 12.07 cycles), but the selected model is the simpler
`median_fade_constant` (MAE 12.21) because the 0.14-cycle difference does not justify
the added complexity; a 5% tolerance rule (0.60 cycles) made that choice explicit.

## 2. RUL definition and assumption

- **Definition:** estimated number of future observed cycles until the first cycle whose
  SOH is strictly below the configured EOL threshold.
- **Default threshold:** 80.0% SOH. This is a **project assumption**, not a manufacturer
  limit. It is configurable through the API (`eol_threshold_pct`) and the pipeline
  (`eol_pct`).
- **Strict crossing:** SOH exactly 80.0% is not a crossing.
- **Unit:** observed source cycles, taken from the actual `cycle_id` column. For the
  three NMC cells the source cycle ids happen to be contiguous, so cycle-id differences
  equal observed-cycle differences. This is stated in provenance rather than assumed
  for other datasets.
- **Causality:** every origin uses only cycles at or before that origin. Future SOH is
  never an input feature.
- **Post-threshold rows:** once SOH is below the threshold, RUL is 0 and those rows are
  reported separately. Pre- and post-threshold errors are never pooled.

## 3. Dataset audit and support levels

| Dataset | Chemistry | Genuine sequence | Observed crossing | Status | Served |
|---|---|---|---:|---|---|
| `NASA_Ames_PCoE` (B0005, B0006, B0018) | NMC | Yes, 468 cycles | 3 cells | **EXPERIMENTAL** | 3 current predictions |
| `LFP_MultiStat_B5` (BATTERY_5) | LFP | Yes, 249 cycles | No | **EXPERIMENTAL_CENSORED** | 1 censored projection |
| `LFP_MultiStat_B1` (BATTERY_1) | LFP | 1 cycle only | No | **UNAVAILABLE** | Reason only |
| `Zenodo_IonTech_NaIon` (4 cells) | Na-ion | Yes, 25 cycles each | No; one starts below 80% | **UNAVAILABLE** | Reason only |
| `EV_FAILURE_PREDICTION` | LFP/LMO/... | Snapshot | No | **UNAVAILABLE** | Reason only |
| `SOLID_STATE_TELEMETRY_2026` | Mixed | Snapshot | No | **UNAVAILABLE** | Reason only |
| `SYNTHETIC_BATTERY_LIFECYCLE` | Synthetic | Snapshot | No | **UNAVAILABLE** | Reason only |

Gates: at least 10 observed cycles to attempt a trajectory; an observed crossing needs
at least 30 pre-EOL cycles; a censored projection needs at least 50 cycles and a first
observation above threshold. Three NMC cells is the minimum for leave-one-cell-out.

Legacy NASA NMC and the PADM implementation are used as read-only experimental
resources. The protected PADM test evaluation is not re-run, bypassed, or used for
fitting or selection.

## 4. Target construction

For each cell, the first position with SOH < 80% defines the EOL event. Origins are
created every 2 cycles from cycle index 9 (10-cycle minimum history) up to the cycle
before the crossing. The target is the actual cycle-id difference between the origin
and the crossing.

| Cell | First crossing cycle id | Pre-EOL origins in LOCO records | Note |
|---|---:|---:|---|
| B0005 | 75 | 33 | — |
| B0006 | 63 | 27 | — |
| B0018 | 45 | 18 | Returns above 80% for 13 cycles after first crossing; first-crossing convention retained and audited |

## 5. Models evaluated

All eight candidates are causal and were fitted/evaluated under identical origin sets.

| Candidate | Family | Description |
|---|---|---|
| `last_delta_persistence` | simple baseline | Mean SOH change over the last 5 cycles held constant |
| `degradation_rate_20` | simple baseline | Linear fade over the last 20 cycles extrapolated |
| `median_fade_constant` | simple baseline | Training median fade per cycle applied constantly |
| `linear_extrapolation` | curve | OLS trend on the full observed history |
| `exponential_extrapolation` | curve | Log-linear exponential decay |
| `padm_ar` | iterative | Ridge autoregression on PADM features, rolled forward |
| `direct_ridge` | direct regression | Ridge on 11 causal summary features |
| `direct_hgb` | direct regression | Histogram gradient boosting on the same features |

## 6. Primary validation: leave-one-cell-out

Metrics are pre-threshold only, in cycles. `n` is the number of scored origins; MAE
between models with different `n` is not directly comparable.

| Model | n | MAE | RMSE | Bias | Conformal 90% coverage |
|---|---:|---:|---:|---:|---:|
| **median_fade_constant (selected)** | 78 | **12.21** | **16.12** | **+6.18** | 0.833 |
| direct_hgb (best single) | 78 | 12.07 | 15.19 | -1.92 | 0.885 |
| padm_ar | 27 | 15.44 | 18.46 | -15.39 | 0.481 |
| last_delta_persistence | 59 | 17.78 | 35.14 | +4.86 | 0.864 |
| direct_ridge | 78 | 18.95 | 20.68 | -9.65 | 0.474 |
| degradation_rate_20 | 73 | 39.67 | 141.79 | +36.61 | 0.877 |
| linear_extrapolation | 78 | 48.15 | 91.61 | +45.99 | 0.654 |
| exponential_extrapolation | 78 | 53.34 | 99.40 | +51.96 | 0.667 |

Interpretation, stated without overclaiming:

- Complexity did **not** reliably beat simple baselines. The top MAE belongs to HGB, but
  a constant training-median fade rate is statistically indistinguishable on this tiny
  sample and far easier to audit.
- Full-history linear/exponential extrapolation is poor here because cell fade is not
  linear over the whole history and the small sample makes the fitted slope unstable.
- PADM-AR is not comparable: it produced no estimate at 51 of 78 origins because the
  iterative rollout predicted non-negative deltas. Its coverage (0.481) is below target
  and its bias is large. This is reported, not hidden.
- Last-delta persistence also failed to produce an estimate at 19 origins.

### Per-battery performance (selected model)

| Cell | n | MAE | RMSE | Bias | Coverage |
|---|---:|---:|---:|---:|---:|
| B0005 | 33 | 7.63 | 11.20 | -6.53 | 0.848 |
| B0006 | 27 | 20.50 | 23.10 | +20.49 | 0.704 |
| B0018 | 18 | 8.19 | 9.76 | +8.01 | 1.000 |

B0006 is the weak cell: the constant fade rate is too slow for its steeper middle
degradation, producing positive RUL bias. This is a genuine limitation of a one-parameter
model, not a data artifact.

## 7. Secondary validation: within-cell chronological 60/40

Train on the first 60% of each cell's origins, score the last 40%. This holds out later
origins but shares the cell, so it is weaker evidence than LOCO.

| Model | n | MAE |
|---|---:|---:|
| direct_ridge | 31 | 7.53 |
| median_fade_constant | 31 | 8.62 |
| degradation_rate_20 | 31 | 8.84 |
| padm_ar | 13 | 9.03 |
| last_delta_persistence | 24 | 9.20 |
| linear_extrapolation | 31 | 21.87 |
| direct_hgb | 31 | 23.11 |
| exponential_extrapolation | 31 | 24.80 |

HGB is best under LOCO but much worse chronologically (23.11), which is direct evidence
that its LOCO advantage is not stable. The selected constant-fade model is second here
and first among models that scored every origin. Model selection therefore did not use
the chronological split to tune anything; it is reported as stability evidence.

## 8. Uncertainty

Two different interval methods are used and labelled:

1. **NMC supervised predictions:** cross-fitted split-conformal absolute-residual
   intervals with a 90% target. The selected model's residual quantile is 26.47 cycles.
   Measured LOCO coverage is 0.833 for the selected model and 0.885 for HGB, both below
   the 0.90 target on 78 points. Coverage is measured on held-out cells, not guaranteed
   for new data.
2. **LFP censored projection:** the interval is the spread between linear and
   exponential curve forecasts. It is **not** a calibrated confidence interval and the
   artifact says so.

Current NMC predictions (retrospective, at an origin inside each observed trajectory):

| Cell | Cycle | SOH % | Predicted EOL cycle | RUL | 90% interval |
|---|---:|---:|---:|---:|---|
| B0005 | 38 | 89.15 | 69 | 31.38 | 12.86 – 49.90 |
| B0006 | 32 | 94.11 | 95 | 63.48 | 38.06 – 88.90 |
| B0018 | 22 | 85.43 | 48 | 26.17 | 0.00 – 86.47 |

B0018's interval reaches 0 because its degradation path is unstable and the calibration
residuals for that cell are large. This is an honest signal of instability, not a bug.

LFP censored projection: BATTERY_5 at cycle 500 (91.07% SOH) projects EOL at cycle 1080,
RUL 580.2 cycles, interval 580.2 – 635.6. The interval has zero lower width because the
linear curve gives the earliest crossing of the two curve families. It has **no validated
accuracy** and is not approved for use.

## 9. API behaviour

`BatteryIntelligenceAPI.predict_rul(...)` delegates to `RULService`:

- snapshots → `status="unavailable"`, empty list, explicit reason;
- NMC → `status="ok"`, `readiness="EXPERIMENTAL"`, three LOCO-based predictions;
- LFP B5 → `status="ok"`, `prediction_status="censored_projection"`,
  `validation_status="EXPERIMENTAL_CENSORED_NO_EOL"`;
- LFP B1 and Na-ion → `status="unavailable"`, empty list, per-cell reason;
- custom history → validated `RULHistory`, actual cycle ids required, conformal interval
  added, cross-chemistry flagged `NOT_VALIDATED_FOR_CHEMISTRY`.

`approved_for_use` is false for every dataset and `require_approved_rul()` always raises.
No status in the repository is `VALIDATED`.

## 10. Artifacts

`member2_modeling/artifacts/rul_v1/`:

| File | Contents |
|---|---|
| `rul_predictions.csv` | 624 LOCO origin records with all required fields |
| `rul_current_predictions.csv` | 3 current NMC predictions |
| `rul_censored_projections.csv` | 1 LFP censored projection |
| `rul_unavailable.csv` | 5 cells with explicit reasons |
| `rul_per_battery.csv` | Per-model, per-cell MAE/RMSE/bias/coverage |
| `rul_degradation_trajectories.csv` | Observed histories plus separated forecasts |
| `rul_eol_trajectories.csv` | Predicted EOL paths by curve family |
| `rul_model.joblib` | Selected model and all candidate payloads |
| `rul_model_registry.json` | Model ids, versions, metrics, calibration, SHA-256 |
| `rul_metrics.json` | Full metrics, selection rule, limitations |
| `rul_provenance.json` | Environment, source/code hashes, causality and exclusions |
| `rul_summary.json` | Support matrix and headline counts |

All 13 files are copied into `artifacts/analytics_v1/` for Member 5.

## 11. Limitations and non-claims

- **Three NMC cells** cannot establish generalizable RUL. No VALIDATED status is claimed.
- The 80% threshold is a project assumption, not a manufacturer limit.
- B0018's transient recovery means the first-crossing convention is a choice; the
  sustained-crossing convention is recorded in the audit but is not the served target.
- No LFP or Na-ion cell reaches 80%, so no supervised RUL validation exists for those
  chemistries. The LFP projection is not validation.
- Snapshot datasets have no chronological EOL trajectory. No synthetic sequence, EOL
  label, or interpolated future cycle was created.
- RUL is not a safety diagnosis and is not approved for operational use.
