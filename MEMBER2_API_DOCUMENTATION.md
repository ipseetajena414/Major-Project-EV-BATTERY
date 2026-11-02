# Member 2 API Documentation

Module: `member2_modeling/inference/`
Entry point: `BatteryIntelligenceAPI`

The API loads **real fitted artifacts** from
`member2_modeling/artifacts/health/` and
`member2_modeling/artifacts/hybrid_nmc_legacy/`. It does not retrain or modify models.

## 1. Quick start

```python
from member2_modeling.inference import BatteryIntelligenceAPI

api = BatteryIntelligenceAPI()
result = api.predict(
    "EV_FAILURE_PREDICTION",
    "state_of_health",
    {
        "vehicle_brand": "Tata",
        "vehicle_model": "Nexon EV",
        "battery_chemistry": "LFP",
        "chemistry_family": "LFP",
        "cycle_count": 420,
        "odometer_km": 48000,
        # other features may be omitted; they are median-imputed with a warning
    },
)
print(result["status"], result["predictions"])
```

## 2. Available routes

| Dataset | Target | Unit | Readiness | Features |
|---|---|---|---:|---:|
| `EV_FAILURE_PREDICTION` | `remaining_capacity` | kWh | EXPERIMENTAL | 56 |
| `EV_FAILURE_PREDICTION` | `state_of_health` | % | EXPERIMENTAL | 56 |
| `SOLID_STATE_TELEMETRY_2026` | `state_of_health_pct` | % | EXPERIMENTAL | 19 |
| `SYNTHETIC_BATTERY_LIFECYCLE` | `discharge_capacity_ah` | Ah | RESEARCH_ONLY | 5 |

`api.available_routes()` returns this list programmatically.
`api.requirements(dataset_id, target)` returns the exact feature list, chemistry column,
forbidden columns, unit, readiness, and target kind.
`api.hybrid_requirements()` returns the 11 hybrid features, sequence length 5, entity
and order columns, readiness, and benchmark warning.

## 3. `predict(dataset_id, target, records, *, chemistry=None)`

`records` accepts a mapping, a pandas DataFrame, or an iterable of mappings.

Return shape:

```json
{
  "status": "ok | partial | error",
  "task": "regression",
  "dataset_id": "...",
  "target": "...",
  "unit": "%",
  "prediction_kind": "model_prediction",
  "capacity_kind": "predicted | null",
  "readiness": "EXPERIMENTAL | RESEARCH_ONLY",
  "label_origin": "source_dataset | synthetic",
  "target_kind": "...",
  "production_ready": false,
  "completeness": 0.87,
  "missing_features": ["..."],
  "warnings": ["..."],
  "predictions": [
    {"row": 0, "status": "ok", "prediction": 82.4, "clipped": false,
     "missing_feature_count": 7}
  ],
  "provenance": {
    "artifact_path": "...",
    "feature_pipeline_path": "...",
    "source_sha256": "...",
    "processed_sha256": "...",
    "training_rows": 19044,
    "test_mae": 2.7843,
    "test_rows": 3809,
    "split_protocol": "deterministic_stratified_snapshot_split_60_20_20",
    "read_only": true
  }
}
```

Behavior:

- Missing features are median-imputed; the response lists them and lowers
  `completeness`. Status becomes `partial` only when some rows fail validation.
- Target and leakage columns are rejected per response row with status `error`.
- Non-finite numeric values, booleans, and nulls are rejected.
- Unknown chemistry values are rejected against the dataset vocabulary.
- Unrecognized non-leakage fields are ignored and listed in `warnings`.
- SOH outputs are clipped to `[0, 100]`; capacity outputs to `>= 0`; `clipped` reports it.
- Synthetic and solid-state routes always carry their readiness warning.

## 4. Domain-specific warnings

The response includes route-specific warnings, for example:

- EV: `remaining_capacity` is pack energy in kWh, not cell Ah.
- Solid-state: snapshot data, not a temporal forecast.
- Synthetic: labels are synthetic, not measured experimental truth.
- Any route: no production validation; snapshot split is i.i.d. only.

## 5. `predict_capacity_from_soh(dataset_id, records, *, chemistry=None)`

Derives capacity algebraically from a predicted SOH and a rated capacity. This is
**not** a model prediction and **not** a measurement.

| Dataset | Source column | Unit | Scope | Supported |
|---|---|---|---|---|
| `EV_FAILURE_PREDICTION` | `battery_capacity_kwh` | kWh | vehicle pack energy | Yes |
| `SOLID_STATE_TELEMETRY_2026` | `nominal_capacity_ah` | Ah | cell nominal capacity | Yes |
| `SYNTHETIC_BATTERY_LIFECYCLE` | — | — | — | No (`status="unsupported"`) |

Each successful row returns `kind="algebraic"`, `is_measured=false`,
`is_predicted=false`, `basis="soh_pct_times_rated_capacity"`.

## 6. `predict_rul(dataset_id="LEGACY_NMC_SEQUENTIAL", *, chemistry=None, cell_id=None, history=None, model="auto", eol_threshold_pct=80.0)`

RUL is never approved for operational use (`approved_for_use=false`), but genuine
sequential evidence is served where it exists.

| Dataset | Status | Behaviour |
|---|---|---|
| Snapshot datasets | `unavailable` | No sequence is manufactured from independent rows. |
| `LEGACY_NMC_SEQUENTIAL` | `ok`, `readiness="EXPERIMENTAL"` | Leave-one-cell-out NMC predictions for B0005, B0006, B0018. |
| `LFP_MultiStat_B5` | `ok`, `prediction_status="censored_projection"` | No observed crossing; extrapolated with a curve-family spread, not a calibrated interval. |
| `LFP_MultiStat_B1`, `Zenodo_IonTech_NaIon` | `unavailable` | Too few cycles or already below threshold at the first observation. |

Each prediction carries `battery_id`, `current_cycle`, `current_soh_pct`,
`eol_threshold_pct`, `predicted_eol_cycle`, `estimated_rul_cycles`,
`lower_uncertainty_bound`, `upper_uncertainty_bound`, `model_id`, `model_version`,
`chemistry`, `prediction_status`, `validation_status`, and `provenance`.

Pass `history=rows` (each row needs an actual `cycle_id`/`cycle_index` and `soh_pct`)
to score a custom causal history. Cross-chemistry custom histories are flagged
`NOT_VALIDATED_FOR_CHEMISTRY`. See `MEMBER2_RUL_IMPLEMENTATION_REPORT.md`.

## 6a. RUL and analytics introspection

- `rul_metadata()`, `rul_support()`, `rul_models()` expose artifacts, the support
  matrix, and candidate metrics without re-fitting anything.
- `python -m member2_modeling.analytics.run_analytics` builds the Member 5 package at
  `member2_modeling/artifacts/analytics_v1/`.

## 7. `predict_sequence(records, *, chemistry="NMC", cell_id="CUSTOM")`

Runs the fused hybrid on an ordered per-cell history.

- `chemistry` must be `NMC`; other values return `status="unsupported"` with a reason.
- Each record supplies the 11 hybrid features. `cell_id` and `cycle_id` are optional;
  defaults are the `cell_id` argument and the record index.
- Duplicate `(cell_id, cycle_id)` pairs raise `CustomInputError`.
- Rows with fewer than 5 prior records receive the learned tabular-only fallback
  (`sequence_available=false`).
- Each row returns `prediction` (fused SOH), `tabular_prediction`,
  `sequence_prediction` (null when unavailable), and `sequence_available`.
- The response repeats the EXPERIMENTAL readiness, the persistence-baseline benchmark
  warning, and the fallback warning.

## 8. Errors

| Error | Raised when |
|---|---|
| `CustomInputError` | empty input, invalid keys, invalid numeric value, unknown chemistry, duplicate sequence key |
| `KeyError` | unknown dataset/target route |
| `FileNotFoundError` | required model artifact is absent |

## 9. Command-line entry points

```bash
# Train health models (writes only to member2_modeling/artifacts/health)
python -m member2_modeling.health.train_health_models

# Train the hybrid (writes only to member2_modeling/artifacts/hybrid_nmc_legacy)
python -m member2_modeling.hybrid.train_hybrid

# Build compatibility reports
python -c "from member2_modeling.data_integration.profiles import build_compatibility_report; build_compatibility_report()"

# Build Member 3 exports
python -c "from member2_modeling.member3_export import build_member3_exports; build_member3_exports()"

# RUL audit
python -c "from member2_modeling.rul import build_rul_report; build_rul_report()"
```
