# Member 2 Implementation Status

**Scope:** battery degradation modeling layer (SOH, capacity, RUL gating, chemistry-aware
features, hybrid LightGBM + BiLSTM, custom inference API, Member 3 exports).

**Readiness:** every model remains **EXPERIMENTAL** or **RESEARCH_ONLY**.
`production_ready` is `false` in every manifest and evaluation artifact.

> **Latest closure (2026-11-02): snapshot ML suite v2.0.0.**
> Seven tasks (4 regression, 3 classification) are trained on the three active
> Member 1 datasets with entity-grouped splits, train-only preprocessing and a
> checksummed registry. See `MEMBER2_FINAL_ML_SYSTEM_REPORT.md` and
> `member2_modeling/artifacts/snapshot_ml/`. Reproduce with
> `.venv/Scripts/python.exe -m member2_modeling.snapshot_ml.run`.
> Full test suite: **376 passed, 10 subtests passed**.

## 1. What was delivered

| Area | Module | Status |
|---|---|---|
| Member 1 integration | `member2_modeling/data_integration/profiles.py` | Complete |
| Compatibility audit | `member2_modeling/artifacts/reports/` | Complete |
| Chemistry-aware features | `member2_modeling/features/tabular.py`, `features/sequence.py` | Complete |
| SOH / capacity health models | `member2_modeling/health/` | Complete (4 models) |
| Hybrid LightGBM + BiLSTM | `member2_modeling/hybrid/` | Complete, EXPERIMENTAL |
| RUL conventions and gating | `member2_modeling/rul/` | Complete, never approved |
| Custom inference API | `member2_modeling/inference/` | Complete |
| Member 3 exports | `member2_modeling/member3_export.py` | Complete, adapter-validated |

## 1a. Snapshot ML suite (v2.0.0, 2026-11-02)

| Area | Module | Status |
|---|---|---|
| Task registry (7 tasks) | `member2_modeling/snapshot_ml/specs.py` | Complete |
| Entity-grouped stratified splits | `snapshot_ml/splits.py` | Complete |
| Multi-candidate training/selection | `snapshot_ml/training.py`, `candidates.py` | Complete |
| Registry + SHA-256 verification | `snapshot_ml/registry.py` | 42 files OK |
| Inventory + leakage audit | `snapshot_ml/inventory.py`, `audit.py` | OK |
| Inference API + CLI | `snapshot_ml/inference.py`, `cli.py` | Complete |
| Formula range estimate | `snapshot_ml/range_estimate.py` | Calculated only |
| Cross-chemistry diagnostic | `snapshot_ml/chemistry_holdout.py` | Complete |
| Proposed Member 3 bundle v2 | `snapshot_ml/member3_bundle.py` | Validated, adapter-incompatible by design |

Artifacts: `member2_modeling/artifacts/snapshot_ml/` (7 model dirs + registry,
inventory, audit, cross-chemistry results, report) and
`member2_modeling/artifacts/member3_exports_v2/`. New tests:
`member2_modeling/tests/test_snapshot_ml.py` (10 tests).

Reproduction commands:

```bash
.venv/Scripts/python.exe -m member2_modeling.snapshot_ml.run
.venv/Scripts/python.exe -m member2_modeling.snapshot_ml.chemistry_holdout
.venv/Scripts/python.exe -m member2_modeling.snapshot_ml.member3_bundle
.venv/Scripts/python.exe -m member2_modeling.snapshot_ml.cli info --task ev_soh
```
| Tests | `member2_modeling/tests/` + experiment tests | 178 collected |

## 2. Built artifacts

```
member2_modeling/artifacts/
├── health/
│   ├── health_model_evaluation.json / .md
│   ├── EV_FAILURE_PREDICTION/{remaining_capacity,state_of_health}/
│   ├── SOLID_STATE_TELEMETRY_2026/state_of_health_pct/
│   └── SYNTHETIC_BATTERY_LIFECYCLE/discharge_capacity_ah/
├── hybrid_nmc_legacy/
│   ├── hybrid_model/ (config, schema, tabular, sequence, fusion, builder)
│   ├── hybrid_manifest.json
│   └── hybrid_evaluation.json / .md
├── reports/
│   ├── member2_dataset_compatibility.json
│   └── MEMBER2_DATASET_COMPATIBILITY_REPORT.md
├── rul/
│   ├── rul_availability.json
│   └── MEMBER2_RUL_TARGET_AUDIT.md
└── member3_exports/
    ├── observed_soh.csv
    ├── predicted_soh.csv
    ├── baseline_soh.csv
    ├── PROVENANCE.json
    └── range_from_{observed,predicted,baseline}.csv (adapter verification)
```

Every health model directory contains `model.joblib`, `feature_pipeline.joblib`,
`model_config.json`, and `model_schema.json`.

## 3. Dataset posture

| Dataset | Rows | Grain | Sequence | Training | Readiness |
|---|---:|---|---|---|---|
| `EV_FAILURE_PREDICTION` | 20,000 | vehicle pack snapshot | No | Permitted | EXPERIMENTAL |
| `SOLID_STATE_TELEMETRY_2026` | 10,000 | cell snapshot | No | Permitted | EXPERIMENTAL |
| `SYNTHETIC_BATTERY_LIFECYCLE` | 2,000 | synthetic snapshot | No | Permitted | RESEARCH_ONLY |
| `LEGACY_NMC_SEQUENTIAL` | 278 model-ready pairs | NASA NMC per-cycle | Yes | Legacy research input | EXPERIMENTAL |

The three Member 1 processed datasets are snapshot-only. Their row order is never
treated as a degradation trajectory, so no sequence branch or hybrid fusion is fitted
on them. The only sequential source is the legacy NASA NMC table (`master_table.parquet`),
used explicitly as labelled legacy research input.

## 4. Verification (run for this closure)

Commands executed from the repository root with the pinned `.venv`:

```bash
.venv/Scripts/python.exe -m pytest -q --import-mode=importlib
.venv/Scripts/python.exe -m member1_pipeline.build --check
```

Results (current snapshot-suite closure, 2026-11-02):

- Full suite: **376 passed, 10 subtests passed** (52.75 s).
- Member 2 test directory: **94 passed** (21.01 s).
- Snapshot suite tests: **10 passed** (56.56 s).
- Member 1 build check: **VALIDATION OK** (earlier closure; unchanged data).
- Snapshot registry verification: **42 files, all checksums OK**.
- Leakage audit: **OK**.
- Member 3 bundle validation: **OK**.
- 15 warnings are pre-existing `LGBMDeprecationWarning: eval_set` messages from
  `member2_modeling/experiments/test_rul_validation.py` and
  `test_walk_forward.py`; no new code uses `eval_set`.

No test was weakened, deleted, or skipped to obtain a pass.

## 5. Known limitations

- Health models are validated with **i.i.d. snapshot splits**, not temporal or
  vehicle-level holdouts.
- The hybrid model is trained on 278 legacy NMC pairs from three cells. The
  persistence and mean-fade baselines outperform it on the chronological test split
  (fused MAE 6.1105 vs persistence 0.5040).
- RUL is never approved. The shipped RUL model is in-sample, the production test split
  is 100% post-threshold, and only three cells exist.
- Synthetic labels are not measured experimental truth.
- `projected_cycle_life_80pct_soh` is a supplied projection, not observed RUL.
- Member 3 combines SOH with a partially sourced vehicle profile, so range outputs
  remain estimates requiring validation.

## 6. Preservation

No Member 1 processed dataset, Member 1 report, legacy Member 2 model (`*.joblib`,
`*.pt`), scaler, `feature_schema.json`, `model_manifest.json`, `member3/`, `member4/`,
`member5/`, root historical artifact, or `master_table.parquet` was modified.
All new work is additive under `member2_modeling/` plus new root documentation.
