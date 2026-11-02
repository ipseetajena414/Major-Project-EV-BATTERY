# Member 5 Analytics Handover

Generated from frozen Member 2 artifacts. No Member 3 or Member 5 directory was
modified. All RUL values are experimental and not approved for operational use.

## 1. What is in this package

| File | Purpose |
|---|---|
| `prediction_results.csv` | Frozen test predictions plus RUL leave-one-cell-out predictions |
| `model_metrics.csv` / `model_metrics.json` | Selected models and baselines with frozen metrics |
| `regression_residuals.csv` | Per-record residuals for snapshot regression and RUL |
| `classification_results.csv` | Per-class precision/recall and overall classification metrics |
| `confusion_matrices.json` | Confusion matrices, including per-chemistry blocks |
| `chemistry_metrics.csv` | Metrics split by chemistry |
| `dataset_summary.json` | Dataset inventory and RUL support matrix |
| `model_comparison.csv` | Selected model versus best baseline on the same rows |
| `prediction_provenance.json` | Source hashes, artifact hashes, execution metadata |
| `feature_importance.csv` | Permutation importance (no SHAP; dependency pins frozen) |
| `uncertainty_summary.csv/json` | Interval coverage and calibration status |
| `anomaly_flags.csv` | Statistical telemetry outliers; no fault labels |
| `health_status.csv` | Informational SOH bands |
| `degradation_trajectories.csv` | Observed SOH rows plus BATTERY_5-only forecasts, clearly separated; NMC EOL forecasts live in `rul_eol_trajectories.csv` |
| `degradation_summary.csv/json` | Per-cell fade rates and support status |
| `rul_*.csv/json` | Full RUL artifacts copied from `rul_v1`, including per-battery metrics and EOL trajectories |
| `member3_combined_analytics_schema.json` | JSON Schema for joined Member 2 + Member 3 rows |
| `member3_results_template.csv` | Header-only template; no fabricated Member 3 rows |
| `member3_soh_sequences.csv` | Observed NMC/LFP cycles in Member 3 adapter columns |
| `ANALYTICS_MANIFEST.json` | File hashes and sizes for this package |

## 2. Recommended Member 5 workflow

```python
import pandas as pd

base = pd.read_csv("member2_modeling/artifacts/analytics_v1/prediction_results.csv")
rul = base[(base.domain == "rul") & (base.record_scope == "leave_one_cell_out")]
selected = rul[rul.model_id == "median_fade_constant"]

# Actual versus predicted RUL for Member 5 error plots.
error = selected.dropna(subset=["observed", "prediction"]).copy()
error["residual"] = error["prediction"] - error["observed"]

# Historical SOH only; forecast rows are flagged separately.
traj = pd.read_csv("member2_modeling/artifacts/analytics_v1/degradation_trajectories.csv")
observed = traj[traj.record_type == "observed"]
forecast = traj[traj.record_type == "forecast"]
```

## 3. Confidence and status rules

- Snapshot regression: 90% split-residual intervals. Empirical coverage on the
  frozen test split is reported in `uncertainty_summary.csv`.
- Snapshot classification: probabilities are **uncalibrated**. Do not label them
  calibrated.
- RUL: prefer the simplest candidate whose pre-threshold MAE is within 0.60 cycles of the best candidate; selected model
  `median_fade_constant` with pre-threshold MAE
  12.215 cycles. Only three NMC cells exist.
- LFP: censored projection only, no observed EOL.
- Snapshot datasets and Na-ion: RUL UNAVAILABLE.
- Anomaly flags are statistical outliers, not confirmed faults.
- Health bands are informational, not a safety diagnosis.

## 4. Joining Member 3 results

Member 3 owns `range_km`, `usable_energy_kwh`, `consumption_kwh_per_km`,
`vehicle_config_id` and `member3_model_id`. Join on `(cell_id, cycle_index)` or
on `(battery_id, cycle_index)` when identifiers agree. Never impute a missing
range; leave it null. The schema forbids additional properties so that
downstream columns stay stable.

## 5. Reproducibility

Rebuild this package with:

```bash
.venv/Scripts/python.exe -m member2_modeling.analytics.run_analytics --rebuild-rul
```

The RUL pipeline reads the legacy `master_table.parquet` read-only and never
re-runs or bypasses the protected PADM test evaluation sentinel.
