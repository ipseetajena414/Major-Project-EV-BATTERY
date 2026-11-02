# Member 2 College Project Demonstration

A reproducible, self-contained walkthrough of the battery intelligence system for a
project review, viva, or lab demonstration.

Date: 2026-11-02
Environment: Python 3.12.10 in `.venv`; Windows or Linux
Runtime: about 2 minutes for the examples; about 4 minutes to rebuild everything

## 1. What the demonstration proves

1. A snapshot battery dataset can be scored for SOH, capacity, and failure probability
   with a validated API and traceable row ids.
2. Sequential telemetry can be screened for statistical anomalies without inventing
   fault labels.
3. RUL is produced only from genuine cycle histories and is explicitly labelled
   EXPERIMENTAL; snapshots and short sequences return UNAVAILABLE with reasons.
4. Every result can be exported as graph-ready CSV/JSON for Member 5.

## 2. Setup

From the repository root:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-member2.txt
.venv/Scripts/python.exe -m pytest member2_modeling/tests -q
```

Expected: `109 passed`.

## 3. Architecture in one diagram

```text
Member 1 datasets (snapshots)          Legacy master_table.parquet (cycles)
        |                                        |
        v                                        v
data_integration/profiles.py           analytics/sequential.py
        |                                        |
        v                                        v
snapshot_ml (7 frozen tasks)           rul/forecast.py -> rul/pipeline.py
        |                                        |
        +----------------+-----------------------+
                         v
              inference/service.py  (BatteryIntelligenceAPI)
                         v
              analytics_v1 exports (CSV / JSON / schema)
                         v
                 Member 5 dashboards, Member 3 join
```

Ownership boundary: Member 2 never writes to `member3/`, `member4/`, `member5/`, or
`data/processed/`, and never fabricates Member 3 range values.

## 4. Example 1 — score one snapshot vehicle

```python
import pandas as pd
from member1_pipeline import loader
from member2_modeling.inference import BatteryIntelligenceAPI

api = BatteryIntelligenceAPI()
frame = loader.load_dataset("EV_FAILURE_PREDICTION")
requirements = api.requirements("EV_FAILURE_PREDICTION", "state_of_health")
row = frame.iloc[2]
record = {
    column: row[column].item() if hasattr(row[column], "item") else row[column]
    for column in requirements["feature_columns"]
    if pd.notna(row[column])
}
result = api.predict(
    "EV_FAILURE_PREDICTION", "state_of_health", [record], chemistry="NMC"
)
print(result["status"], result["predictions"][0]["prediction"])
```

Actual response (trimmed):

```json
{
  "status": "ok",
  "task": "regression",
  "dataset_id": "EV_FAILURE_PREDICTION",
  "target": "state_of_health",
  "unit": "%",
  "readiness": "EXPERIMENTAL",
  "completeness": 0.9464,
  "missing_features": ["average_charging_time", "cell_voltage_std", "cycle_count"],
  "predictions": [{"row": 0, "status": "ok", "prediction": 59.2492, "clipped": false}],
  "provenance": {
    "model_artifact": "member2_modeling/artifacts/health/EV_FAILURE_PREDICTION/state_of_health",
    "model_type": "tabular_lightgbm_regressor",
    "training_rows": 19044,
    "test_mae": 2.7843,
    "test_n": 3809
  }
}
```

Teaching point: the warning states that three required features were imputed and that
snapshot predictions are not vehicle-level forecasts. The missing-feature report is not
hidden.

## 5. Example 2 — current RUL for the three NMC cells

```python
import json
from member2_modeling.inference import BatteryIntelligenceAPI

api = BatteryIntelligenceAPI()
print(json.dumps(api.rul_support()["datasets"]["NASA_Ames_PCoE"], indent=2))
rul = api.predict_rul("LEGACY_NMC_SEQUENTIAL")
for record in rul["predictions"]:
    print(
        record["battery_id"],
        "cycle", record["current_cycle"],
        "SOH", round(record["current_soh_pct"], 2),
        "RUL", round(record["estimated_rul_cycles"], 2),
        "90% interval", (
            round(record["lower_uncertainty_bound"], 2),
            round(record["upper_uncertainty_bound"], 2),
        ),
    )
```

Actual output:

```text
B0005 cycle 38 SOH 89.15 RUL 31.38 90% interval (12.86, 49.90)
B0006 cycle 32 SOH 94.11 RUL 63.48 90% interval (38.06, 88.90)
B0018 cycle 22 SOH 85.43 RUL 26.17 90% interval (0.00, 86.47)
```

Teaching point: `readiness="EXPERIMENTAL"`, `approved_for_use=false`, and the B0018
interval is wide because that cell is unstable. Nothing is presented as validated.

## 6. Example 3 — custom history, including a rejection

```python
from member2_modeling.inference import BatteryIntelligenceAPI

api = BatteryIntelligenceAPI()
history = [
    {"cycle_id": cycle, "soh_pct": 100.0 - 0.35 * cycle}
    for cycle in range(1, 41)
]
result = api.predict_rul("LEGACY_NMC_SEQUENTIAL", history=history)
print(result["predictions"][0]["estimated_rul_cycles"])

try:
    api.predict_rul(
        "LEGACY_NMC_SEQUENTIAL",
        history=[{"cycle_id": 3, "soh_pct": 96.0}, {"cycle_id": 1, "soh_pct": 99.0},
                 {"cycle_id": 2, "soh_pct": 97.0}],
    )
except ValueError as error:
    print("rejected:", error)
```

Actual RUL: `23.75` cycles at cycle 40 (86.0% SOH). The second call is rejected because
cycle ids are not strictly increasing. No future SOH is accepted as input.

## 7. Example 4 — anomalies, health, and degradation for graphs

```python
import pandas as pd

base = "member2_modeling/artifacts/analytics_v1/"
anomalies = pd.read_csv(base + "anomaly_flags.csv")
health = pd.read_csv(base + "health_status.csv")
degradation = pd.read_csv(base + "degradation_trajectories.csv")

flagged = anomalies[anomalies["is_statistical_anomaly"]]
print(len(flagged), "of", len(anomalies), "cycles flagged")
print(health[["battery_id", "current_soh_pct", "health_band"]])
print(degradation.groupby(["dataset_id", "record_type"]).size())
```

Actual headline results:

```text
242 of 817 cycles flagged
B0005 / B0006 / B0018 -> CRITICAL band by latest observed SOH
GOOD 3, FAIR 2, POOR 1, CRITICAL 3 across 9 cells
observed and forecast rows are separate in degradation_trajectories.csv
```

Teaching point: `confirmed_fault` is always false. These are statistical flags, not
diagnoses.

## 8. Example 5 — compare models honestly

```python
import pandas as pd

comparison = pd.read_csv(
    "member2_modeling/artifacts/analytics_v1/model_comparison.csv"
)
print(
    comparison[
        ["domain", "task_id", "model_id", "model_role", "primary_metric",
         "primary_value", "selected"]
    ].to_string(index=False)
)
```

Key rows:

```text
ev_capacity  hist_gradient_boosting  selected       mae  3.209
ev_capacity  train_median            best_baseline  mae 26.115
ev_failure   logistic_regression     selected       balanced_accuracy 0.914
ev_failure   majority_class          best_baseline  balanced_accuracy 0.500
RUL          median_fade_constant    selected       mae 12.215
RUL          direct_hgb              best_single    mae 12.072
```

Teaching point: HGB has the best RUL MAE, but the simpler constant-fade model is
selected because it is within 0.60 cycles. Complexity is not automatically rewarded.

## 9. Graph-ready files

| Question | File |
|---|---|
| Predicted vs observed, all tasks | `prediction_results.csv` |
| Residual plots | `regression_residuals.csv` |
| Confusion matrices | `confusion_matrices.json` |
| Metrics by task | `model_metrics.csv`, `model_metrics.json` |
| Metrics by chemistry | `chemistry_metrics.csv` |
| Model vs baseline | `model_comparison.csv` |
| SOH trajectories | `degradation_trajectories.csv` |
| SOH fade summary | `degradation_summary.csv` |
| RUL vs actual, per origin | `rul_predictions.csv` |
| RUL current estimates | `rul_current_predictions.csv` |
| RUL per battery | `rul_per_battery.csv` |
| Predicted EOL paths | `rul_eol_trajectories.csv` |
| Anomaly timeline | `anomaly_flags.csv` |
| Health bands | `health_status.csv` |
| Uncertainty coverage | `uncertainty_summary.csv` |
| Provenance | `prediction_provenance.json` |

`member3_soh_sequences.csv` provides the adapter-shaped SOH series. It contains observed
sequence data only; snapshot datasets are not inserted.

## 10. Build everything from scratch

```bash
.venv/Scripts/python.exe -m member2_modeling.analytics.run_analytics --rebuild-rul
```

This writes `member2_modeling/artifacts/rul_v1/` and
`member2_modeling/artifacts/analytics_v1/`, then hashes every delivered file in
`ANALYTICS_MANIFEST.json`. The legacy `master_table.parquet` is read-only.

## 11. Expected viva questions

**Why only three batteries for RUL?**
Only three NMC cells have both a genuine chronological sequence and an observed 80%
crossing. LFP and Na-ion cells never cross in the available history.

**Why is RUL not validated?**
Three independent cells are too few for a validated claim. Leave-one-cell-out is used,
but the error bars remain wide and no VALIDATED status is claimed.

**Why not generate sequences from snapshots?**
Snapshot rows are independent vehicles, not successive cycles of one battery. Ordering
them would create a trajectory that was never measured.

**Why is the anomaly rate 29.6%?**
Per-cell IsolationForest contamination is automatic and the histories are short. The
flag means "unusual for this cell", not "faulty".

**Why is SHAP missing?**
It is not installed and dependency pins are frozen. Permutation importance is the
documented substitute and is not causal.

**Why does capacity dominate `ev_capacity` importance?**
The target `remaining_capacity` is derived from `battery_capacity_kwh`; the importance
exposes that dependency. It is not a physical causal claim.

## 12. What is intentionally not shown

- No Member 3 range result is generated.
- No fault label or safety diagnosis is assigned.
- No snapshot RUL is produced.
- No calibrated probability is claimed for classifiers.
- No future cycle is invented for extrapolation.
