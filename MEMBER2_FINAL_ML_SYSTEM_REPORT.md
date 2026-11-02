# Member 2 — Final Data-Driven ML System Report

**Scope:** battery-health, capacity, failure and degradation-tier models built on the
active Member 1 processed datasets, plus inference, range estimation, testing and the
Member 3 hand-off package.

**Version:** model suite `2.0.0` · registry version `2.0.0`
**Date:** 2026-11-02
**Environment:** Python 3.12.10 · pandas 3.1.0 · numpy 2.5.4 · lightgbm 4.7.0 ·
scikit-learn 1.9.1 · joblib 1.6.0 · pyarrow 26.0.0 · torch 2.15.0+cpu
**Branch:** `member2-upgrade` (no commit, merge, rebase or push was performed)

Every model in this report is **EXPERIMENTAL** or **RESEARCH_ONLY**.
`production_ready` is `false` in every registry entry, contract and bundle.

---

## 1. Executive summary

Member 2 now trains and registers **seven tasks** on the three active Member 1
datasets: four regressions and three classifications. Previously only four
regression targets were modelled; the classification labels
(`battery_failure`, `degradation_risk_tier`, synthetic SOH class) had no trained
model.

The suite replaces the old row-wise snapshot split with an **entity-grouped,
target-stratified 60/20/20 split**, fits all preprocessing on the training split
only, selects among four to five candidates on validation only, and scores the
test split exactly once. Every selected model beats its median/majority baseline
on the task's primary metric, but several results carry strong caveats that are
documented rather than hidden:

- `ev_capacity` learns an **algebraically derived** target (capacity =
  rated capacity × health / 100), so its R² is not measured-capacity accuracy.
- `ev_failure` has high balanced accuracy but **lower raw accuracy than the
  majority class**; balanced accuracy is the honest metric for a 6.9 % positive
  class.
- `synthetic_*` tasks use generated labels and are **RESEARCH_ONLY**.
- Cross-chemistry holdout **fails badly** for synthetic capacity
  (R² ≈ −7.2 to −7.6 when Lithium-Sulfur or Sodium-Ion is unseen), which
  confirms that the within-chemistry test score cannot be generalised across
  chemistries.
- No active dataset contains cycle sequences, observed RUL, or measured range.
  Those outputs remain unavailable or calculated-only.

---

## 2. Dataset inventory and feasibility

| Dataset ID | Rows | Entities | Columns | Features used | Target(s) modelled | Grain | Sequence | Chemistry values |
|---|---:|---:|---:|---:|---|---|---|---|
| `EV_FAILURE_PREDICTION` | 20,000 | 20,000 (`vehicle_id`) | 82 | 56 | `state_of_health`, `remaining_capacity`, `battery_failure` | vehicle pack snapshot | no | LFP, LMO, LTO, NCA, NMC |
| `SOLID_STATE_TELEMETRY_2026` | 10,000 | 10,000 (`cell_id`) | 41 | 19 | `state_of_health_pct`, `degradation_risk_tier` | cell snapshot | no | Halide_Based_Li3YCl6, Oxide_Based_LLZO, Polymer_Composite, Sulfide_Based_Li10GeP2S12 |
| `SYNTHETIC_BATTERY_LIFECYCLE` | 2,000 | 1,978 (`battery_id`) | 20 | 6 | `discharge_capacity_ah`, `state_of_health` | synthetic snapshot | no | Lithium-Sulfur, Sodium-Ion, Solid-State |

All three are consumed read-only through `member1_pipeline.loader`. The registry
reports `sequence_available=false` for every dataset and `active_master_table=null`;
no legacy `master_table.parquet` was substituted.

### Feasibility matrix

| Candidate task | Label exists? | Supported? | Decision |
|---|---|---|---|
| Next-cycle SOH | no | no | `UNAVAILABLE` — snapshot tables only |
| RUL (cycles to EOL) | no | no | `UNAVAILABLE_ON_ACTIVE_DATA` — projected labels are not observed RUL |
| Measured EV range | no | no | `CALCULATED_ESTIMATE_ONLY` — no range target |
| Sequence / cross-chemistry transfer | no | no | `UNAVAILABLE` — incompatible grains and schemas |
| EV SOH | yes (`state_of_health`) | yes | `EXPERIMENTAL` |
| EV remaining capacity | yes, derived | yes | `EXPERIMENTAL`, derived target |
| EV failure | yes (`battery_failure`) | yes | `EXPERIMENTAL` |
| Solid-state SOH | yes | yes | `EXPERIMENTAL` |
| Degradation tier | yes | yes | `EXPERIMENTAL` |
| Synthetic capacity | yes, generated | yes | `RESEARCH_ONLY` |
| Synthetic SOH class | yes, generated | yes | `RESEARCH_ONLY` |

---

## 3. Dataset-to-model mapping

| Task ID | Dataset | Target | Type | Status | Artifact directory |
|---|---|---|---|---|---|
| `ev_soh` | `EV_FAILURE_PREDICTION` | `state_of_health` | regression | EXPERIMENTAL | `member2_modeling/artifacts/snapshot_ml/ev_soh/` |
| `ev_capacity` | `EV_FAILURE_PREDICTION` | `remaining_capacity` | regression | EXPERIMENTAL | `.../ev_capacity/` |
| `ev_failure` | `EV_FAILURE_PREDICTION` | `battery_failure` | binary | EXPERIMENTAL | `.../ev_failure/` |
| `solid_soh` | `SOLID_STATE_TELEMETRY_2026` | `state_of_health_pct` | regression | EXPERIMENTAL | `.../solid_soh/` |
| `solid_tier` | `SOLID_STATE_TELEMETRY_2026` | `degradation_risk_tier` | multiclass (4) | EXPERIMENTAL | `.../solid_tier/` |
| `synthetic_capacity` | `SYNTHETIC_BATTERY_LIFECYCLE` | `discharge_capacity_ah` | regression | RESEARCH_ONLY | `.../synthetic_capacity/` |
| `synthetic_class` | `SYNTHETIC_BATTERY_LIFECYCLE` | `state_of_health` | multiclass (3) | RESEARCH_ONLY | `.../synthetic_class/` |

Each directory contains `model.joblib`, `feature_pipeline.joblib`, `config.json`,
`schema.json`, `metrics.json` and `test_predictions.parquet`.

---

## 4. Architecture

```
Member 1 processed parquet (read-only)
        │
        ▼
specs.py ── task registry (target, type, status, units, notes)
        │
        ▼
splits.py ── StratifiedGroupKFold 60/20/20, grouped by vehicle/cell/battery,
             target-quantile or class strata, disjointness asserted
        │
        ▼
training.py ── TabularFeaturePipeline fitted on train only
             ├─ numeric: median imputation
             ├─ categorical: missing category + vocabulary from train only
             ├─ candidates: Ridge, ElasticNet, RandomForest,
             │              HistGradientBoosting, LightGBM
             │              (+ LogisticRegression for classification)
             ├─ selection on validation metric only
             ├─ single test evaluation + bootstrap CI
             └─ per-chemistry diagnostics + residual-q90 interval
        │
        ▼
registry.py ── SHA-256 checksums, environment, provenance, limitations
        │
        ├── inference.py ── SnapshotInferenceAPI + CLI
        ├── range_estimate.py ── formula estimate, never measured
        ├── chemistry_holdout.py ── leave-one-chemistry-out diagnostic
        └── member3_bundle.py ── proposed snapshot export v2
```

`run.py` orchestrates inventory → training → registry → leakage audit →
verification → Markdown report.

---

## 5. Leakage prevention

- Feature columns come only from the Member 1 data-dictionary role `feature`.
- Target and audited leakage columns are rejected before fitting:
  - EV: `state_of_health`, `battery_health_percent`, `capacity_loss_percent`,
    `remaining_capacity`, `predicted_remaining_life_cycles`.
  - Solid-state: `state_of_health_pct`, `degradation_risk_tier`, projected cycle
    life, risk rationale and enriched scores.
  - Synthetic: the modelled target, and `discharge_capacity_ah` when predicting
    the categorical SOH class.
- Splits are grouped by `vehicle_id` / `cell_id` / `battery_id`; measured
  entity overlap is zero for every pair of splits in every task.
- Duplicate synthetic `battery_id` rows stay in one split.
- Imputation medians and categorical vocabularies are fitted on train only.
- Validation is used for candidate selection and LightGBM early stopping; the
  test split is untouched until the configuration is frozen.
- The machine leakage audit (`leakage_audit.json`) returns **OK**.

---

## 6. Models and selection protocol

Candidates per regression task: Ridge, ElasticNet, RandomForest,
HistGradientBoosting, LightGBM. Classification replaces the regressors with
LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier and
LGBMClassifier.

Selection rule: minimum validation MAE (regression) or maximum validation
balanced accuracy (classification). The test split is scored once.

| Task | Selected model | Key parameters | Validation selection score |
|---|---|---|---|
| `ev_soh` | HistGradientBoosting | lr 0.06, 300 iters, 31 leaves | MAE 2.8096 |
| `ev_capacity` | HistGradientBoosting | lr 0.06, 300 iters, 31 leaves | MAE 3.0195 |
| `ev_failure` | LogisticRegression | C=1.0, class_weight=balanced | balanced acc 0.9240 |
| `solid_soh` | LightGBM | lr 0.05, 600 trees, 31 leaves | MAE 0.5949 |
| `solid_tier` | LightGBM | lr 0.05, 600 trees, class_weight=balanced | balanced acc 0.9016 |
| `synthetic_capacity` | Ridge | alpha 1.0 | MAE 3.9349 |
| `synthetic_class` | LogisticRegression | C=1.0, class_weight=balanced | balanced acc 0.8098 |

Full candidate scores are stored per task in `config.json`
(`candidate_selection`). Representative spreads:

- `ev_soh`: Ridge 3.1193, ElasticNet 3.1148, RF 2.8473, HGB 2.8096, LGBM 2.8111.
- `ev_capacity`: Ridge 4.2172, ElasticNet 4.2302, RF 3.1019, HGB 3.0195, LGBM 3.0371.
- `ev_failure` balanced accuracy: LogisticRegression 0.9240, RF 0.6517,
  HGB 0.8715, LGBM 0.8334.
- `solid_soh`: Ridge 0.7969, ElasticNet 0.7962, RF 0.6295, HGB 0.6016, LGBM 0.5949.
- `solid_tier` balanced accuracy: LogisticRegression 0.7768, RF 0.8848,
  HGB 0.8965, LGBM 0.9016.
- `synthetic_capacity`: Ridge 3.9349, ElasticNet 3.9418, RF 4.1479,
  HGB 4.1503, LGBM 4.1654.
- `synthetic_class` balanced accuracy: LogisticRegression 0.8098, RF 0.6314,
  HGB 0.7828, LGBM 0.7060.

No single algorithm dominates. Linear models win on the small synthetic set;
boosted trees win on the larger EV and solid-state sets. This is evidence that
candidate comparison was real rather than a single fixed model.

---

## 7. Results

### 7.1 Regression (single grouped-holdout test evaluation)

| Task | n | MAE | RMSE | R² | Bias | MAPE | 95 % MAE CI | Train-median MAE | Chemistry-median MAE |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `ev_soh` | 3,809 | **2.7470** | 3.4766 | 0.8541 | −0.0890 | 3.45 % | 2.677–2.808 | 7.3785 | 7.3873 |
| `ev_capacity` | 3,877 | **3.2091** | 5.3059 | 0.9699 | +0.0702 | 4.91 % | 3.082–3.363 | 26.1151 | 26.1217 |
| `solid_soh` | 2,000 | **0.6472** | 0.8126 | 0.9853 | −0.0216 | 0.75 % | 0.627–0.669 | 5.6745 | 5.6744 |
| `synthetic_capacity` | 400 | **3.8176** | 4.9945 | 0.9331 | +0.3630 | 5.24 % | 3.503–4.147 | 16.2499 | 8.1490 |

### 7.2 Classification

| Task | n | Accuracy | Balanced accuracy | Macro-F1 | Log loss | Brier | Majority balanced acc | Chemistry-majority balanced acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ev_failure` | 4,000 | 0.9233 | **0.9137** | 0.7884 | 0.1965 | 0.0544 | 0.5000 | 0.5000 |
| `solid_tier` | 2,000 | 0.8995 | **0.9010** | 0.8994 | 0.2692 | — | 0.2500 | 0.3052 |
| `synthetic_class` | 400 | 0.7750 | **0.7932** | 0.6824 | 0.4988 | — | 0.3333 | 0.3333 |

`ev_failure` confusion matrix (test): 3,443 true negatives, 280 false positives,
27 false negatives, 250 true positives. Raw accuracy (0.9233) is **below** the
majority-class accuracy (0.9308); the model is useful because balanced accuracy
and positive-class recall are far above the trivial baseline. Calling it
"93 % accurate" without that context would be misleading.

### 7.3 Comparison with the previous Member 2 suite (v1.0.0)

The v1 models used a row-wise i.i.d. 60/20/20 split. The v2 numbers below use the
harder entity-grouped split, so they are **not an apples-to-apples improvement
claim**; they show that performance survives group isolation.

| Task | v1 MAE (i.i.d.) | v2 MAE (grouped) | Change | v1 R² | v2 R² |
|---|---:|---:|---:|---:|---:|
| `ev_soh` | 2.7843 | 2.7470 | −0.037 | 0.8484 | 0.8541 |
| `ev_capacity` | 3.4397 | 3.2091 | −0.231 | 0.9671 | 0.9699 |
| `solid_soh` | 0.6296 | 0.6472 | +0.018 | 0.9857 | 0.9853 |
| `synthetic_capacity` | 4.1541 | 3.8176 | −0.337 | 0.9180 | 0.9331 |

The solid-state change is small and slightly worse; it is consistent with the
harder grouped split but is not presented as an improvement. Three of the four
v1 targets improve under the stricter protocol, and three previously unmodelled
classification tasks now exist.

### 7.4 Cross-chemistry holdout (diagnostic only)

Train on all other chemistries, score the excluded one:

| Task | Held-out chemistry | Primary metric | Verdict |
|---|---|---|---|
| `ev_soh` | each of LFP/LMO/LTO/NCA/NMC | MAE 2.74–2.86 | stable |
| `ev_capacity` | each of LFP/LMO/LTO/NCA/NMC | MAE 3.05–3.14 | stable (derived target) |
| `ev_failure` | each of LFP/LMO/LTO/NCA/NMC | balanced acc 0.920–0.947 | stable |
| `solid_soh` | each of 4 electrolytes | MAE 0.625–0.648 | stable |
| `solid_tier` | Halide / Oxide / Polymer / Sulfide | balanced acc 0.924 / 0.886 / 0.774 / 0.843 | moderate spread |
| `synthetic_capacity` | Solid-State | MAE 4.04, R² 0.715 | acceptable |
| `synthetic_capacity` | Lithium-Sulfur | MAE 29.82, **R² −7.60** | **fails** |
| `synthetic_capacity` | Sodium-Ion | MAE 28.75, **R² −7.16** | **fails** |
| `synthetic_class` | Lithium-Sulfur / Sodium-Ion / Solid-State | balanced acc 0.63 / 0.59 / 0.63 | weak |

This is the strongest generalisation warning in the suite. The synthetic capacity
model memorises chemistry-specific capacity scales; it must not be used on an
unseen chemistry. The `RESEARCH_ONLY` status is therefore load-bearing, not
decorative.

---

## 8. Artifacts and registry

- `member2_modeling/artifacts/snapshot_ml/model_registry.json` — 7 task entries,
  selected model, metrics, baselines, split hash, entity overlap, provenance,
  required features, limitations, SHA-256 for every artifact file.
- `registry_verification.json` — 42 files hashed; all checksums match.
- `dataset_inventory.json` / `.md` — row counts, entity counts, missingness,
  chemistry distribution, intended tasks.
- `leakage_audit.json` — machine audit, status OK.
- `cross_chemistry.json` — leave-one-chemistry-out diagnostics.
- `MEMBER2_SNAPSHOT_ML_REPORT.md` — generated per-run report.
- `member2_modeling/artifacts/member3_exports_v2/` — proposed downstream bundle.

No existing artifact was deleted or overwritten. The old `health/`,
`hybrid_nmc_legacy/`, `delta/`, `rul/` and `member3_exports/` directories are
untouched.

---

## 9. Inference and range estimation

`SnapshotInferenceAPI` loads the registry and returns prediction, unit, model ID,
status, intervals or class probabilities, input validation and provenance.

Verified on real dataset rows:

| Call | Result |
|---|---|
| `ev_soh` on EV row 0 | 81.559 % (90 % interval 75.746–87.372) |
| `ev_capacity` on EV row 0 | 35.658 kWh (90 % interval 29.290–42.025) |
| `ev_failure` on EV row 0 | class 0, P(failure) = 0.00043 |
| `solid_soh` on solid row 0 | 81.118 % (90 % interval 79.848–82.389) |
| `solid_tier` on solid row 0 | Accelerated_Degradation_Tier_3, P = 0.9767 |
| unseen chemistry `UNOBTAINIUM` | rejected with `InferenceInputError` |
| range from a real EV row | 120.89 km at 0.18 kWh/km |

Range is a **formula estimate**: usable kWh = rated kWh × SOH × discharge
efficiency, divided by an assumed consumption. Every output carries
`kind="calculated_estimate"`, `is_measured=false`, `validated_accuracy=false`.
There is no measured range ground truth, so no accuracy claim is made.

---

## 10. Tests and verification actually executed

All commands were run from the repository root with the pinned `.venv`.

```bash
# Full suite
.venv/Scripts/python.exe -m pytest -q --import-mode=importlib
# -> 376 passed, 15 warnings, 10 subtests passed in 52.75s

# Snapshot ML tests only
.venv/Scripts/python.exe -m pytest member2_modeling/tests/test_snapshot_ml.py -q
# -> 10 passed in 56.56s

# Member 2 test directory
.venv/Scripts/python.exe -m pytest member2_modeling/tests -q
# -> 94 passed in 21.01s
```

The 15 warnings are the pre-existing `LGBMDeprecationWarning: eval_set` messages
from `member2_modeling/experiments/test_rul_validation.py` and
`test_walk_forward.py`; the new suite uses `eval_X` / `eval_y`.

Additional executed checks:

- Registry verification: 42 files, all checksums OK.
- Leakage audit: OK.
- Member 3 bundle validation: OK
  (`ev_soh` 3,809 rows, `solid_state_soh` 2,000, `ev_failure` 4,000,
  `ev_range` 3,687).
- Inference smoke test across all five production-shaped tasks: passed.
- Unseen-chemistry rejection: passed.

No test was skipped, weakened or deleted to obtain these results.

---

## 11. Reproduction

```bash
# 1. Train all seven tasks, build registry, inventory, audit, report
.venv/Scripts/python.exe -m member2_modeling.snapshot_ml.run

# 2. Re-run a subset
.venv/Scripts/python.exe -m member2_modeling.snapshot_ml.run --tasks ev_soh solid_soh

# 3. Cross-chemistry diagnostic
.venv/Scripts/python.exe -m member2_modeling.snapshot_ml.chemistry_holdout

# 4. Proposed Member 3 bundle
.venv/Scripts/python.exe -m member2_modeling.snapshot_ml.member3_bundle

# 5. Inference
.venv/Scripts/python.exe -m member2_modeling.snapshot_ml.cli info --task ev_soh
.venv/Scripts/python.exe -m member2_modeling.snapshot_ml.cli predict \
    --task ev_soh --input records.json

# 6. Verification
.venv/Scripts/python.exe -c "from member2_modeling.snapshot_ml.registry import load_registry, verify_registry; from pathlib import Path; r=Path('member2_modeling/artifacts/snapshot_ml'); print(verify_registry(r))"
```

Determinism: split hash, random seed (42), environment versions and artifact
checksums are recorded. Re-running on the same data and environment reproduces
the same split and artifacts.

---

## 12. Changes made

New, under Member 2 ownership only:

- `member2_modeling/snapshot_ml/` — `specs.py`, `splits.py`, `metrics.py`,
  `candidates.py`, `training.py`, `registry.py`, `inventory.py`, `audit.py`,
  `run.py`, `inference.py`, `range_estimate.py`, `member3_bundle.py`,
  `chemistry_holdout.py`, `cli.py`, `__init__.py`.
- `member2_modeling/tests/test_snapshot_ml.py` — 10 tests.
- `member2_modeling/artifacts/snapshot_ml/` — 7 model directories + registry,
  inventory, audit, cross-chemistry diagnostic and reports.
- `member2_modeling/artifacts/member3_exports_v2/` — proposed bundle (new
  directory; the existing `member3_exports/` was not touched).
- `MEMBER2_FINAL_ML_SYSTEM_REPORT.md` (this file) and an update to
  `MEMBER2_IMPLEMENTATION_STATUS.md`.

No file under `member1_pipeline/`, `member3/`, `member4/`, `member5/`,
`DATA/processed/`, or any existing Member 2 checkpoint was modified.

---

## A. Dataset utilization verdict

**PASS.** All three active Member 1 datasets are consumed from their registered
processed paths. Seven targets are modelled across them. The three datasets that
cannot support a task (sequence/RUL/range/cross-chemistry transfer) are documented
with evidence instead of being faked.

## B. Implemented models verdict

**PASS.** Four regressions and three classifications are trained, selected from
multiple candidates, registered and loadable. The previous gap — no classifier
for `battery_failure`, `degradation_risk_tier` or synthetic SOH class — is closed.

## C. Performance verdict

**CONDITIONAL PASS.**

- Every selected model beats its median/majority baseline on the primary metric.
- Grouped-holdout scores survive entity isolation and are comparable to or better
  than the old i.i.d. scores (solid-state is marginally worse, as expected).
- Caveats that prevent an unqualified pass:
  - `ev_capacity` predicts an algebraically derived target.
  - `ev_failure` raw accuracy is below the majority class.
  - `synthetic_*` labels are generated, and synthetic capacity fails
    cross-chemistry holdout.
  - `solid_tier` does not generalise uniformly across electrolytes.

## D. Verification verdict

**PASS.** 376 tests pass, 10 new snapshot tests pass, 42 artifact checksums match,
the leakage audit is OK, the Member 3 bundle validates, and inference was executed
on real records.

## E. Changes verdict

**PASS.** All changes are additive and confined to Member 2-owned paths plus new
root documentation. Protected directories and legacy artifacts are untouched.

## F. Remaining work

1. Obtain a sequence dataset with observed per-cycle SOH to support next-cycle
   prediction and real RUL; then retrain with chronological validation.
2. Obtain measured range/consumption data before any range-accuracy claim.
3. Audit whether `battery_failure` and `degradation_risk_tier` are rule-generated
   from the supplied features; if so, treat their metrics as label-recovery, not
   physical prediction.
4. Revisit synthetic capacity with per-chemistry models or a chemistry-invariant
   target definition; do not deploy the current global model across chemistries.
5. Align Member 3 with the snapshot contract (no fabricated `cycle_index`) or
   provide Member 3 a real sequential source.
6. Add prediction-interval calibration testing once a temporal holdout exists.

## G. Readiness verdicts

| Area | Verdict | Evidence |
|---|---|---|
| Member 2 internal delivery | **PASS (experimental)** | 7 tasks trained, registered, checksummed, tested |
| Production readiness | **NOT READY** | `production_ready=false`; snapshot-only; derived/synthetic labels |
| SOH prediction | **EXPERIMENTAL — usable with caveats** | grouped MAE 2.75 pts (EV), 0.65 pts (solid) |
| Capacity prediction | **EXPERIMENTAL — derived target** | MAE 3.21 kWh, but target is algebraic |
| Failure classification | **EXPERIMENTAL — label audit required** | balanced acc 0.914; raw acc below majority |
| Degradation tier | **EXPERIMENTAL — per-chemistry limits** | balanced acc 0.901; cross-chemistry 0.774–0.924 |
| Synthetic tasks | **RESEARCH ONLY** | generated labels; cross-chemistry failure |
| RUL | **UNAVAILABLE** | no observed EOL trajectory in active data |
| Range | **CALCULATED ESTIMATE ONLY** | formula output, `validated_accuracy=false` |
| Member 3 integration | **PARTIAL** | new snapshot bundle validated; legacy cycle adapter incompatible |

**Overall:** Member 2 has a complete, reproducible, leakage-controlled experimental
ML system on the active Member 1 data. It is not production-ready, and the
remaining gaps are data-availability gaps (sequences, RUL, measured range) plus a
label-provenance audit, not missing modelling code.
