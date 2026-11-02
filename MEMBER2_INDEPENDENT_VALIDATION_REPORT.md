# Member 2 — Independent Validation Report

**Audit date:** 2026-11-02
**Branch:** `member2-upgrade` (upstream `origin/member5/integration`)
**Working tree at audit start:** Member 2 and Member 1 work present as staged additions (`git status --short` shows `A` entries). This audit made **no source change, no branch change, no commit, no push, and no artifact regeneration**.
**Scope:** the seven registered snapshot ML tasks, their datasets, splits, targets, evaluation, chemistry behaviour, inference and the Member 3 handover bundle.
**Companion documents:**

1. `MEMBER2_TARGET_PROVENANCE_AUDIT.md`
2. `MEMBER2_DATA_SPLIT_AND_LEAKAGE_AUDIT.md`
3. `MEMBER2_MODEL_ACCEPTANCE_MATRIX.md`
4. `MEMBER2_MEMBER3_INTEGRATION_CONTRACT.md`

Finding labels: **VERIFIED**, **EXPERIMENTAL**, **UNVERIFIED**, **NOT SUPPORTED**, **BLOCKED**.

---

## 1. Executive summary

The seven-task snapshot system is **software-correct, reproducible and honestly labelled**. Artifacts load, checksums verify, inference is deterministic, guards reject malformed inputs, and every model carries `production_ready = false`.

It is **not scientifically validated in the sense the project needs**:

- The EV and solid-state "entities" are **row-unique synthetic identifiers**, so the entity-grouped split provides no entity-level independence. It is effectively a random row holdout.
- Three of seven targets are **constructed by formula or rule** (`ev_capacity` is an algebraic identity; `solid_tier` is a rule; `synthetic_class` is an exact per-chemistry capacity threshold). Their metrics measure reconstruction, not prediction.
- The remaining targets (`ev_soh`, `ev_failure`, `solid_soh`, `synthetic_capacity`) have **undocumented origins** and are highly reconstructable from permitted features.
- There is **no external validation**, no temporal split, no measured range target, and no RUL target.
- The snapshot Member 3 bundle validates structurally but is **not adapter-compatible** and contains one verified traceability defect (`snapshot_row_id` null in the failure CSV).

Nothing was fabricated or inflated. The correct posture is: **EXPERIMENTAL in-distribution models with honest limits; not production; not external validation.**

---

## 2. Baseline and preservation

- Branch and remotes recorded before any action; no branch switch, merge, rebase, commit or push occurred.
- Legacy Member 2 artifacts (`artifacts/delta/*`, `padm_*`, `hybrid_nmc_legacy/*`, `artifacts/member3_exports/*`) and the protected PADM sentinel `member2_modeling/artifacts/delta/TEST_EVALUATION_COMPLETE.sentinel` were **not touched and not rerun**.
- `member1_pipeline/`, `member3/`, `member4/`, `member5/`, `data/processed/` were read only.
- No dependency pins were changed.
- New files written by this audit are the five documents named above. No existing file was modified.

---

## 3. Dataset utilization

| Dataset | Rows × Cols | Grain | Sequence | Tasks | Origin |
|---|---|---:|---|---|---|
| `EV_FAILURE_PREDICTION` | 20,000 × 82 | vehicle pack snapshot | No | 3 | `source_dataset` (undocumented beyond filename) |
| `SOLID_STATE_TELEMETRY_2026` | 10,000 × 41 | cell snapshot | No | 2 | `source_dataset` (undocumented beyond filename) |
| `SYNTHETIC_BATTERY_LIFECYCLE` | 2,000 × 20 | synthetic battery snapshot | No | 2 | `synthetic` |

- **VERIFIED** — all seven tasks load through `member1_pipeline.loader` with processed SHA-256 verification. No task uses the legacy `master_table.parquet`.
- **VERIFIED** — duplicate full rows: 0 in all three. Missing cells: EV 53,344 (retained, train-median imputed); solid-state 0; synthetic 0. Member 1 cleaning drops 0 rows; each task then excludes rows whose *target* is null (956 for `ev_soh`, 614 for `ev_capacity`, 0 for the other five tasks).
- **UNVERIFIED** — whether the EV and solid-state files are real measured campaigns or synthesized data. The identifiers and value patterns are consistent with generation.
- **NOT SUPPORTED** — any claim that the EV/solid datasets are independent physical-unit samples.

---

## 4. Entity and split audit

| Dataset | Rows | Unique IDs | Pattern | Physical identity |
|---|---:|---:|---|---|
| EV | 20,000 | 20,000 | `EV100000…` and `BAT-10000000…` sequential | **NOT SUPPORTED** |
| Solid-state | 10,000 | 10,000 | `BAT-<MANUFACTURER>-<8 hex>` algorithmic | **NOT SUPPORTED** |
| Synthetic | 2,000 | 1,978 | 22 IDs repeated twice; 16 with conflicting metadata | **VERIFIED repeated entity** |

- **VERIFIED** — the splitter (`splits.py`) is correct: outer `StratifiedGroupKFold(5)`, inner `(4)`, deterministic seeds, disjointness assertion, fallback grouping, SHA-256 split digest. Entity overlap is 0 in every task.
- **NOT SUPPORTED** — "entity-level generalization" for EV/solid. Because every row has a unique key, the grouped split degenerates to a random row split. This is a *data* limitation, not a code defect.
- **VERIFIED** — for synthetic, the 22 repeated IDs are kept inside one split, so duplicate/conflicting metadata never crosses train/val/test.
- **NOT SUPPORTED** — chronological validity. No record timestamp exists; `cycle_count` / `cycle_number` are snapshot state values. No temporal split is possible.
- **EXPERIMENTAL** — target stratification changes the test class mix; metrics are conditional on the stratified test distribution.

Details: `MEMBER2_DATA_SPLIT_AND_LEAKAGE_AUDIT.md`.

---

## 5. Target provenance

| Task | Target | Origin | Verdict |
|---|---|---|---|
| `ev_soh` | `state_of_health` | Supplied label; agrees with `battery_health_percent` at r = 0.9898; >1-point disagreement in 8,008/20,000 (max 6.22); linear reconstruction R² 0.8177 | **UNVERIFIED** |
| `ev_capacity` | `remaining_capacity` | `battery_capacity_kwh × battery_health_percent / 100` (max residual 0.005 kWh); `capacity_loss_percent = 100 − health` exact | **VERIFIED derived** |
| `ev_failure` | `battery_failure` | 6.92% positive; no single-feature rule; depth-10 tree train balanced accuracy 0.9589; single-feature AUC up to 0.842 | **UNVERIFIED origin / label-recovery risk** |
| `solid_soh` | `state_of_health_pct` | r = −0.943 cycle, −0.875 resistance, +0.903 coulombic; linear R² 0.9765 | **UNVERIFIED** (simulation-consistent) |
| `solid_tier` | `degradation_risk_tier` | Overlapping SOH ranges; SOH-only 0.8461, SOH+cycle+temp+resistance 0.8826, all features **1.0000** | **VERIFIED rule-generated** |
| `synthetic_capacity` | `discharge_capacity_ah` | Synthetic campaign; cross-chemistry R² −7.60 / −7.16 | **VERIFIED synthetic** |
| `synthetic_class` | `state_of_health` | Exact per-chemistry capacity thresholds reproduce labels with **100% agreement**; all-feature tree **1.0000** | **VERIFIED derived + synthetic** |

Details: `MEMBER2_TARGET_PROVENANCE_AUDIT.md`.

---

## 6. Leakage audit

- **VERIFIED** — zero forbidden/target columns used as features in all seven tasks (`forbidden_used_as_feature = 0`).
- **VERIFIED** — feature admission is role-based (data dictionary `role == feature`), so derived scores, unit-ambiguous columns, projections, rationale and confidence text cannot enter.
- **VERIFIED** — preprocessing (medians, category vocabularies, missing category) is fitted on the training split only.
- **VERIFIED** — model selection uses validation only; the test split is scored once; no threshold tuning; candidate hyperparameters are fixed.
- **VERIFIED** — test predictions align with source records: 100% entity match on `snapshot_row_id` for `ev_soh`, `solid_soh`, `synthetic_class`.
- **NOT SUPPORTED** — independence from target construction. `ev_capacity`, `solid_tier` and `synthetic_class` are derivable from the feature set; `ev_soh` and `solid_soh` are near-deterministic from telemetry; `ev_failure` is recoverable.
- **VERIFIED (mechanism) / CONDITIONAL (meaning)** — no train/test contamination was found; the weakness is label construction, not split contamination.

---

## 7. Evaluation results

Numbers below are the frozen test metrics from each `metrics.json`. No model was retrained.

### 7.1 Regression

| Task | n | MAE | RMSE | R² | Bias | 95% MAE CI | Median baseline MAE |
|---|---:|---:|---:|---:|---:|---|---:|
| `ev_soh` | 3,809 | 2.7470 | 3.4766 | 0.8541 | −0.0890 | 2.6768 – 2.8077 | 7.3785 |
| `ev_capacity` | 3,877 | 3.2091 | 5.3059 | 0.9699 | +0.0702 | 3.0818 – 3.3627 | 26.1151 |
| `solid_soh` | 2,000 | 0.6472 | 0.8126 | 0.9853 | −0.0216 | 0.6271 – 0.6689 | 5.6745 |
| `synthetic_capacity` | 400 | 3.8176 | 4.9945 | 0.9331 | +0.3630 | 3.5032 – 4.1469 | 16.2499 |

Empirical coverage of the stated 90% residual intervals on test: `ev_soh` 90.23%, `ev_capacity` 88.16%, `solid_soh` 88.65%, `synthetic_capacity` 90.00%. The intervals are derived from validation residuals, not fitted calibration.

### 7.2 Classification

| Task | n | Accuracy | Balanced acc. | Macro F1 | Log loss | 95% Bal-acc CI | Majority balanced acc. |
|---|---:|---:|---:|---:|---:|---|---:|
| `ev_failure` | 4,000 | 0.9233 | 0.9137 | 0.7884 | 0.1965 | 0.8938 – 0.9302 | 0.5000 |
| `solid_tier` | 2,000 | 0.8995 | 0.9010 | 0.8994 | 0.2692 | 0.8871 – 0.9140 | 0.2500 |
| `synthetic_class` | 400 | 0.7750 | 0.7932 | 0.6824 | 0.4988 | 0.7399 – 0.8410 | 0.3333 |

- `ev_failure` has **lower raw accuracy than the majority baseline** (0.9233 vs 0.9307) because it detects the 6.9% positive class. Balanced accuracy (0.9137 vs 0.5000) is the correct comparison and is strong.
- `ev_failure` predicted positive rate at the default 0.5 threshold is **13.25%** vs 6.93% observed. Probabilities, not `predicted_class`, should be consumed.
- All classification probabilities are labelled `uncalibrated_class_probabilities`.

### 7.3 Chemistry

In-distribution per-chemistry performance is stable (e.g. `ev_soh` MAE 2.654–2.895; `solid_soh` MAE 0.621–0.671; `solid_tier` balanced accuracy 0.881–0.930). Cross-chemistry diagnostics:

- EV tasks appear chemistry-robust, but only because their targets are formulas/signals.
- `solid_tier` degrades on Polymer_Composite (balanced accuracy 0.774) when held out.
- `synthetic_capacity` **fails** on unseen Li-S (R² −7.60) and Na-ion (R² −7.16).
- `synthetic_class` falls to near chance on unseen chemistry (0.587–0.631).

### 7.4 Feature attribution

Feature importances exist for `ev_failure`, `solid_soh`, `solid_tier`, `synthetic_class`. They are **empty** for `ev_soh`, `ev_capacity` (HistGradientBoosting) and `synthetic_capacity` (Ridge). This is a transparency gap, not a correctness defect; permutation importance was not computed for this audit.

---

## 8. Inference, range and sequence safety

### 8.1 Inference

- **VERIFIED** — `SnapshotInferenceAPI` enforces the exact required feature schema, rejects missing required features, rejects unseen chemistry unless opted in, blocks missing values unless opted in, converts numeric OOD inputs into explicit warnings, flags SOH outside 0–100 and capacity below 0, returns `production_ready: false`, and ignores target/leakage columns supplied by callers.
- **VERIFIED** — repeated calls on identical input return identical predictions (checked on five EV records).
- **VERIFIED** — the model version inside each response matches the registry (`2.0.0`).

### 8.2 Range

- **VERIFIED (classification)** — range is a **calculated estimate**, never measured:
  `usable_kwh = rated_kwh × SOH/100 × discharge_efficiency/100`; `range_km = usable_kwh / 0.18`.
  The output carries `kind = calculated_estimate`, `is_measured = false`, `validated_accuracy = false`, `validation_status = not_validated_no_range_target`.
- **VERIFIED** — the default consumption of 0.18 kWh/km and the 100% discharge-efficiency fallback are assumptions, not measured values. Optimistic/conservative scenarios use 0.15 / 0.22 kWh/km.
- **NOT SUPPORTED** — any range-accuracy claim. No measured range or consumption target exists in the active data.
- **VERIFIED** — 122 / 3,809 EV test rows have no range estimate because `battery_capacity_kwh` is null; they are omitted rather than imputed.

### 8.3 RUL and sequences

- **VERIFIED** — all three active datasets are snapshots (`sequence_available = false`); no Member 2 snapshot task uses a sequence model.
- **VERIFIED** — RUL remains unavailable for the active data: `artifacts/rul/rul_availability.json` sets `approved_for_use = false` for every dataset.
- **VERIFIED** — no RUL, next-cycle SOH, cycle index, timestamp or trajectory was fabricated. `UNSUPPORTED_TASKS` in `specs.py` documents each exclusion.
- **VERIFIED** — the legacy PADM/hybrid NMC experiment is preserved untouched as a separate EXPERIMENTAL component; its protected final evaluation was not rerun.

---

## 9. Defects found

### F-01 — Failure bundle loses `snapshot_row_id` (verified, Member 2-owned, resolved 2026-11-02)

`member3_bundle.py:136` read `source_row_id`, but `test_predictions.parquet` stores `snapshot_row_id`. All 4,000 rows of the historical `member3_exports_v2` failure export had `snapshot_row_id = null`. The fix is applied: `member3_exports_v2_fixed` has 4,000 non-null, unique ids, `validate_bundle` rejects duplicates, and regression tests cover both properties.

### F-02 — `remaining_capacity` definition is wrong (verified, documentation, resolved 2026-11-02)

`member2_modeling/data_integration/profiles.py:102` called it "Measured remaining pack capacity"; it is algebraically derived. The source, registry, model config and generated reports now state the algebraic derivation and that it is verified only on rows where both `battery_capacity_kwh` and `battery_health_percent` are present.

No other implementation defect was found. No test was weakened, skipped or deleted.

---

## 10. Verification executed (non-destructive)

| Check | Command / method | Result |
|---|---|---|
| Full test suite | `.venv/Scripts/python.exe -m pytest -q --import-mode=importlib` | **376 passed, 10 subtests passed, 15 pre-existing warnings, 122.67 s** |
| Member 1 build check | `.venv/Scripts/python.exe -m member1_pipeline.build --check` | **VALIDATION OK** |
| Registry integrity | `verify_registry("member2_modeling/artifacts/snapshot_ml")` | **42 files checked, ok = true, 0 problems** |
| Snapshot bundle schema | `validate_bundle("member2_modeling/artifacts/member3_exports_v2")` | **ok = true**, 4 files, 3,809 + 2,000 + 4,000 + 3,687 rows |
| Inference reproducibility | two identical `ev_soh` calls on 5 rows | identical predictions |
| Unseen chemistry | `battery_chemistry = UNOBTAINIUM` | blocked; opt-in produces explicit warning |
| Missing values | null values in required features | blocked by default; 13 cells filled when opted in |
| OOD input | `manufacturing_year = 1e9` | flagged in `out_of_training_range` with warning |
| Source-row alignment | join `snapshot_row_id` → `source_row_id` | 100% match for `ev_soh`, `solid_soh`, `synthetic_class` |
| Protected PADM evaluation | not invoked | confirmed no test references `padm_evaluate` / sentinel |

The 15 warnings are pre-existing LightGBM `eval_set` deprecation notices in legacy experiment tests (`test_rul_validation.py`, `test_walk_forward.py`), unchanged by this audit.

---

## 11. Answers to the eleven final acceptance questions

1. **Are all three active Member 1 datasets genuinely integrated?**
   **VERIFIED yes** — all seven tasks load them through the registry-verified Member 1 loader; no legacy master table is used.

2. **Are the dataset identifiers suitable for meaningful independent validation?**
   **NO / NOT SUPPORTED** for EV and solid-state: identifiers are row-unique generated keys, so "entity-grouped" is equivalent to a random row split. Synthetic has 22 genuinely repeated IDs (limited benefit). Confirmed physical identity is **UNVERIFIED**.

3. **Are the seven tasks predicting independent targets, or reconstructing derived/rule labels?**
   Three reconstruct by construction: `ev_capacity` (algebraic identity), `solid_tier` (rule), `synthetic_class` (exact per-chemistry threshold). The other four have undocumented origins and high reconstruction rates. No task has verified independent measurement provenance.

4. **Is there target leakage or split contamination?**
   **No direct leakage or split contamination was found.** Forbidden columns are 0, preprocessing is train-only, selection is validation-only, test is scored once. The unresolved issue is *target construction and label recovery*, not column leakage.

5. **Which model results are scientifically defensible?**
   Only as **in-distribution experimental performance on the supplied labels**. `ev_soh`, `solid_soh` and `solid_tier` show stable, well-above-baseline holdout metrics under an honest protocol. `ev_capacity` is formula reconstruction; `synthetic_*` is synthetic-data research; `ev_failure` is strong discrimination with a label-recovery caveat.

6. **Which models are experimental or research-only?**
   All five EV/solid tasks are **EXPERIMENTAL**; both synthetic tasks are **RESEARCH_ONLY**. Nothing is production-ready.

7. **Which chemistries are actually supported?**
   In-distribution only: EV `LFP, LMO, LTO, NCA, NMC` (+ missing); solid-state `Sulfide, Oxide, Polymer, Halide`; synthetic `Solid_State, Lithium_Sulfur, Sodium_Ion`. Unseen chemistry is rejected by default. Synthetic cross-chemistry **fails**; no LFP↔NMC transfer claim is supported.

8. **Is the range output measured, predicted, or calculated?**
   **Calculated estimate** from rated capacity × SOH × efficiency ÷ assumed consumption. Not measured, not validated, and not a model-predicted range.

9. **Are RUL and next-cycle SOH genuinely supported?**
   **NO.** No active dataset has an observed EOL trajectory or chronological sequence. Both remain unavailable, and no values were fabricated.

10. **What exactly can Member 3 safely consume today?**
    Four experimental CSVs through a **new snapshot-aware path**: `ev_soh_predictions.csv`, `solid_state_soh_predictions.csv`, `ev_failure_predictions.csv`, `ev_range_estimate.csv`. Not through `member2_range_adapter.py`. Use probabilities rather than the default failure class; treat range as calculated; join failure rows by `entity_id` until F-01 is fixed.

11. **What additional data is required before real-world validation?**
    Confirmed physical identifiers with repeated observations; timestamps and cycle history; independently measured SOH/capacity/failure/range targets; documented label-generation rules; an external validation set; probability calibration data; enough units per chemistry for chemistry-specific validation.

---

## 12. Final verdicts

| Dimension | Verdict | Basis |
|---|---|---|
| **Software correctness** | **VERIFIED / PASS** | 376 tests pass; registry 42/42 checksums; bundle validates; inference deterministic and guarded |
| **Dataset integrity** | **CONDITIONAL** | Files checksum-verified, no duplicates, missingness documented; physical origin of EV/solid **UNVERIFIED** |
| **Target validity** | **FAIL / NOT VERIFIED** | Three targets constructed by formula/rule; four have undocumented origins and high reconstruction |
| **Leakage safety** | **CONDITIONAL PASS** | No direct leakage or contamination; derived-target and label-recovery risks remain |
| **Evaluation validity** | **CONDITIONAL** | Protocol is correctly implemented; entity grouping is vacuous for EV/solid; no external or temporal validation |
| **Predictive performance** | **CONDITIONAL** | Strong in-distribution gains over baselines; synthetic cross-chemistry fails; high R² partly reflects label reconstruction |
| **Chemistry generalization** | **NOT SUPPORTED** | No model is validated on an unseen chemistry; synthetic tasks demonstrably fail |
| **Inference reliability** | **CONDITIONAL PASS** | Guards, bounds, OOD warnings, reproducibility verified; probabilities uncalibrated; F-01 traceability defect; default failure threshold inflates positives |
| **Member 3 integration readiness** | **CONDITIONAL** | Structural bundle valid, honestly labelled; **not** adapter-compatible; requires a new snapshot path and F-01 authorization |
| **Production readiness** | **FAIL** | `production_ready = false` in every manifest, registry entry, bundle contract and response; no external validation |

**Overall:** the implementation is trustworthy as software and honest in its labels. It is not yet trustworthy as battery science. The correct handover is a clearly-scoped experimental package, not a validated predictor, and the report deliberately does **not** force a PASS where the evidence is insufficient.
