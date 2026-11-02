# Member 2 Provenance Guide

## 1. What provenance is recorded

Every Member 2 artifact records:

- the input dataset identity and SHA-256;
- the exact feature columns and model parameters;
- the split protocol and training/test row counts;
- readiness status and `production_ready`;
- benchmark warnings and known caveats;
- the list of protected artifacts that were **not** modified;
- a UTC generation timestamp.

No provenance record is inferred after the fact; values are written by the training or
export code at generation time.

## 2. Input dataset hashes

| Dataset | Source file | Source SHA-256 | Processed SHA-256 |
|---|---|---|---|
| `EV_FAILURE_PREDICTION` | `ev battery_failure prediction Dataset.csv` | `3b971b47e6dba73eef7edd60c7841332c5c34be17b649b9b5f036001315b28ea` | `6b4f9035f50d54d72c6e653a5f1f6676b69896ba4b883e48ad1c6f1e5f57dc9d` |
| `SOLID_STATE_TELEMETRY_2026` | `solid_state_ev_battery_telemetry_enriched_2026.csv` | `0fc38a2d4fe6289d712d85112f62d928c78419e562f2ebd9a2f94583489cd690` | `b49529ccffebe735be022e306339596d490ad717d61960a1ab1b71c43cc87513` |
| `SYNTHETIC_BATTERY_LIFECYCLE` | `synthetic_battery_lifecycle.csv` | `4a754a7937f564375611a080dac9b6b3feca97e1f8264de5b30b7f81aed7d224` | `9f7e8ff595499bdc7a70bf29e7cc3d975db4fa8ca57b41236123350496bf6238` |
| `LEGACY_NMC_SEQUENTIAL` | `master_table.parquet` | `6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb` | n/a (read-only legacy input) |

Source hashes are read from the Member 1 registry and inventory. Processed hashes are
read from the same registry, not recomputed from raw files.

## 3. Artifact-level provenance

### Health models

`member2_modeling/artifacts/health/health_model_evaluation.json` contains, per dataset
and target:

```json
{
  "dataset_id": "EV_FAILURE_PREDICTION",
  "target": "state_of_health",
  "target_kind": "source_label",
  "readiness": "EXPERIMENTAL",
  "label_origin": "source_dataset",
  "unit": "%",
  "feature_count": 56,
  "artifact_dir": "member2_modeling/artifacts/health/EV_FAILURE_PREDICTION/state_of_health",
  "model": {
    "model_type": "tabular_lightgbm_regressor",
    "schema_version": "2.0.0",
    "production_ready": false
  },
  "evaluation": {
    "split_protocol": "deterministic_stratified_snapshot_split",
    "counts": {"rows": 19044, "test": 3809, "dropped_null_target": 956},
    "model": {"mae": 2.7843, "rmse": 3.5189, "r2": 0.8484, "mape_pct": 3.49}
  }
}
```

Each model directory also stores `model_config.json` (parameters, readiness, training
counts) and `model_schema.json` (features, medians, vocabularies, evaluation).

### Hybrid model

`member2_modeling/artifacts/hybrid_nmc_legacy/hybrid_manifest.json` records:

- `training_dataset`: source file, SHA-256, cells, cycles per cell, model-ready pairs;
- `dataset_eligibility`: sequence eligibility for all four dataset IDs;
- `feature_columns`, `sequence_length`, `n_folds`, `parameters`;
- `readiness_status = "EXPERIMENTAL"`, `production_ready = false`;
- `benchmark_warning`;
- `artifacts`: model directory, evaluation JSON/Markdown;
- `protected_artifacts_untouched`.

The fitted model directory contains `hybrid_config.json`, `hybrid_schema.json`,
`tabular_branch.joblib`, `sequence_builder.joblib`, `sequence_branch.pt`, and
`fusion.joblib`. `hybrid_schema.json` records the fitted fusion coefficients and branch
descriptions.

### Member 3 exports

`member2_modeling/artifacts/member3_exports/PROVENANCE.json` records:

- dataset, split name and protocol (per-cell chronological 60/20/20, `test`);
- target definition and cycle-index semantics;
- role files, row counts, and SHA-256 for `observed_soh.csv`, `predicted_soh.csv`,
  `baseline_soh.csv`;
- source file, SHA-256, and cells;
- model artifact path, manifest SHA-256, readiness, `production_ready`, benchmark
  warning;
- predicted and baseline MAE/RMSE/bias;
- `excluded_rows` with reasons (empty for the current run);
- warnings;
- adapter compatibility and `member3_files_modified: false`.

## 4. RUL provenance

`member2_modeling/artifacts/rul/rul_availability.json` records:

- availability and reason per dataset;
- `rul_approved = false`;
- threshold rule (`strict SOH < 80.0`) and supported conventions;
- legacy target audit per convention (per-cell crossing position, cycle ID, pre/post
  counts, censored cells, cycles above threshold after crossing);
- a read-only reference to the existing RUL evidence, including its SHA-256
  (`2efeddba737ea9d1d72bed9a7744319b60b8422f2d50be3e8b6ebe817f76f974`).

The historical RUL experiment was not rerun or overwritten.

## 5. Model registry

There is no new central registry file, and the legacy
`member2_modeling/model_manifest.json` was **not modified**. Discovery works as follows:

| Artifact | Registry role |
|---|---|
| `artifacts/health/health_model_evaluation.json` | health model catalogue + metrics |
| `artifacts/health/<dataset>/<target>/model_config.json` | per-model parameters/readiness |
| `artifacts/health/<dataset>/<target>/model_schema.json` | per-model schema/evaluation |
| `artifacts/hybrid_nmc_legacy/hybrid_manifest.json` | hybrid model catalogue |
| `artifacts/rul/rul_availability.json` | RUL approval registry |
| `artifacts/member3_exports/PROVENANCE.json` | export registry |

`BatteryIntelligenceAPI.available_routes()` is the programmatic catalogue for health
routes.

## 6. Reproduction

All generation commands are deterministic (`random_state = 42` where applicable) but
depend on the pinned environment described in the project memory
(`.venv` Python 3.12.10 with real dependencies).

```bash
# 1. Compatibility audit
.venv/Scripts/python.exe -c "from member2_modeling.data_integration.profiles import build_compatibility_report; build_compatibility_report()"

# 2. Health models
.venv/Scripts/python.exe -m member2_modeling.health.train_health_models

# 3. Hybrid model
.venv/Scripts/python.exe -m member2_modeling.hybrid.train_hybrid

# 4. RUL audit
.venv/Scripts/python.exe -c "from member2_modeling.rul import build_rul_report; build_rul_report()"

# 5. Member 3 exports
.venv/Scripts/python.exe -c "from member2_modeling.member3_export import build_member3_exports; build_member3_exports()"
```

Each command writes only inside `member2_modeling/artifacts/`. Re-running replaces its
own new artifacts; it never writes to Member 1, `member3/`, `member4/`, `member5/`, the
legacy Member 2 models, or `master_table.parquet`.

## 7. Protected artifacts

Recorded in `health_model_evaluation.json` and `hybrid_manifest.json`:

```
DATA/processed/
master_table.parquet
member3/
member2_modeling/*_model.joblib
member2_modeling/*.pt
member2_modeling/scaler_*.joblib
member2_modeling/model_manifest.json
member2_modeling/feature_schema.json
```

None of these were modified during Member 2 implementation.
