# Member 2 — Model Acceptance Matrix

**Audit date:** 2026-11-02
**Source of numbers:** each task's frozen `metrics.json`, `config.json`, `schema.json`, `model_registry.json`, plus audit probes documented in `MEMBER2_TARGET_PROVENANCE_AUDIT.md` and `MEMBER2_DATA_SPLIT_AND_LEAKAGE_AUDIT.md`.
**No model was retrained and no protected evaluation was rerun.**

Acceptance codes:

| Code | Meaning |
|---|---|
| **PASS** | Evidence meets the criterion as stated |
| **CONDITIONAL** | Meets the criterion only under an explicit scope restriction |
| **FAIL** | Evidence contradicts the criterion |
| **NOT VERIFIED** | Insufficient evidence to decide |
| **BLOCKED** | Cannot be established without data or authorization outside this audit |

---

## 1. Held-out performance vs baselines

### 1.1 Regression

| Task | Model | n | MAE | RMSE | R² | Bias | MAPE % | Median AE | 95% MAE CI | Train-median MAE | Chemistry-median MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `ev_soh` | HistGradientBoosting | 3,809 | 2.7470 | 3.4766 | 0.8541 | −0.0890 | 3.45 | 2.2569 | 2.6768 – 2.8077 | 7.3785 | 7.3873 |
| `ev_capacity` | HistGradientBoosting | 3,877 | 3.2091 | 5.3059 | 0.9699 | +0.0702 | 4.91 | 2.0182 | 3.0818 – 3.3627 | 26.1151 | 26.1217 |
| `solid_soh` | LightGBM | 2,000 | 0.6472 | 0.8126 | 0.9853 | −0.0216 | 0.75 | 0.5529 | 0.6271 – 0.6689 | 5.6745 | 5.6744 |
| `synthetic_capacity` | Ridge | 400 | 3.8176 | 4.9945 | 0.9331 | +0.3630 | 5.24 | 2.8639 | 3.5032 – 4.1469 | 16.2499 | 8.1490 |

Every task beats both naive baselines on MAE. Bias is small relative to the target spread. The 95% intervals are bootstrap resamples over test rows; they describe sampling variability of the metric, not model error coverage.

Caveat on meaning:

- `ev_capacity` is formula reconstruction (algebraic identity), not capacity measurement.
- `ev_soh` and `solid_soh` targets are strongly reconstructable from permitted telemetry (linear R² 0.8177 / 0.9765); the high R² is partly label reconstruction.
- `synthetic_capacity` is synthetic data and fails cross-chemistry (§3).

### 1.2 Classification

| Task | Model | n | Accuracy | Balanced acc. | Macro F1 | Weighted F1 | Log loss | Brier | ROC AUC | 95% Bal-acc CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ev_failure` | LogisticRegression (balanced) | 4,000 | 0.9233 | 0.9137 | 0.7884 | 0.9339 | 0.1965 | 0.0544 | 0.9749 | 0.8938 – 0.9302 |
| `solid_tier` | LightGBM (balanced) | 2,000 | 0.8995 | 0.9010 | 0.8994 | 0.8994 | 0.2692 | — | — | 0.8871 – 0.9140 |
| `synthetic_class` | LogisticRegression (balanced) | 400 | 0.7750 | 0.7932 | 0.6824 | 0.7864 | 0.4988 | — | — | 0.7399 – 0.8410 |

Naive baselines on the same test rows:

| Task | Majority accuracy | Majority balanced acc. | Majority macro F1 | Chemistry-majority balanced acc. |
|---|---:|---:|---:|---:|
| `ev_failure` | 0.9307 | 0.5000 | 0.4821 | 0.5000 |
| `solid_tier` | 0.3755 | 0.2500 | 0.1365 | 0.3052 |
| `synthetic_class` | 0.5975 | 0.3333 | 0.2493 | 0.3333 |

Interpretation:

- `ev_failure` has **lower raw accuracy than the majority baseline** (0.9233 vs 0.9307) because it predicts the rare class. Its balanced accuracy (0.9137 vs 0.5000) and macro F1 (0.7884 vs 0.4821) are the meaningful comparison and are strongly better. This trade-off must be stated wherever accuracy is quoted.
- `solid_tier` and `synthetic_class` beat all baselines on balanced accuracy and macro F1.

### 1.3 Per-class detail

`ev_failure` confusion matrix (rows = observed, columns = predicted, labels `[0, 1]`):

```
[[3443,  280],
 [  27,  250]]
```

| Class | Support | Precision | Recall |
|---|---:|---:|---:|
| 0 (no failure) | 3,723 | 0.9922 | 0.9248 |
| 1 (failure) | 277 | 0.4717 | 0.9025 |

`solid_tier` confusion matrix (rows = observed, columns = predicted; order `Tier_3, Tier_4, Tier_1, Tier_2`):

```
[[661,  30,  26,  34],
 [ 38, 294,   0,   4],
 [  6,   2, 428,  20],
 [ 23,   1,  17, 416]]
```

| Class | Support | Precision | Recall |
|---|---:|---:|---:|
| Accelerated_Degradation_Tier_3 | 751 | 0.9080 | 0.8802 |
| Critical_Thermal_Risk_Tier_4 | 336 | 0.8991 | 0.8750 |
| Minimal_Degradation_Tier_1 | 456 | 0.9087 | 0.9386 |
| Moderate_Degradation_Tier_2 | 457 | 0.8776 | 0.9103 |

`synthetic_class` confusion matrix (rows = observed, columns = predicted; order `Degraded, End_of_Life, Healthy`):

```
[[ 87, 34, 19],
 [  2, 19,  0],
 [ 33,  2, 204]]
```

| Class | Support | Precision | Recall |
|---|---:|---:|---:|
| Degraded | 140 | 0.7131 | 0.6214 |
| End_of_Life | 21 | 0.3455 | 0.9048 |
| Healthy | 239 | 0.9148 | 0.8536 |

---

## 2. Per-chemistry performance (main model, in-distribution)

| Task | Chemistries in test | Range of primary metric | Weakest chemistry |
|---|---|---|---|
| `ev_soh` | LFP, LMO, LTO, NCA, NMC, missing | MAE 2.654 – 2.895 | LTO (2.895) |
| `ev_capacity` | LFP, LMO, LTO, NCA, NMC, missing | MAE 3.146 – 3.526 | NCA (3.526) |
| `ev_failure` | LFP, LMO, LTO, NCA, NMC, missing | Balanced acc. 0.888 – 0.973 | NMC (0.888) |
| `solid_soh` | Sulfide, Oxide, Polymer, Halide | MAE 0.621 – 0.671 | Sulfide (0.671) |
| `solid_tier` | Sulfide, Oxide, Polymer, Halide | Balanced acc. 0.881 – 0.930 | Sulfide (0.881) |
| `synthetic_capacity` | Solid_State, Li-S, Na-ion | R² 0.726 – 0.781 | Li-S (0.726) |
| `synthetic_class` | Solid_State, Li-S, Na-ion | Balanced acc. 0.752 – 0.831 | Na-ion (0.752) |

All chemistries present in the training data also appear in test. No task was evaluated on a chemistry absent from its training vocabulary.

---

## 3. Cross-chemistry generalization (diagnostic only)

Protocol: train on the other chemistries, score the held-out chemistry. Diagnostic; not used for model selection.

| Task | Held-out chemistries | Outcome | Verdict |
|---|---|---|---|
| `ev_capacity` | LFP 0.9725 · LMO 0.9723 · LTO 0.9709 · NCA 0.9701 · NMC 0.9704 · missing 0.9742 (R²) | Appears to transfer | **Misleading** — target is an algebraic identity, so transfer is guaranteed by construction |
| `ev_soh` | LFP 0.8460 · LMO 0.8531 · LTO 0.8501 · NCA 0.8471 · NMC 0.8444 · missing 0.8300 (R²) | Within 0.02 of in-distribution | **CONDITIONAL** — but target is near-deterministic from telemetry |
| `ev_failure` | Balanced acc. 0.9202 – 0.9470 | Stable | **CONDITIONAL** — label-recovery risk remains |
| `solid_soh` | R² 0.9852 – 0.9858 | Stable | **CONDITIONAL** — simulated telemetry |
| `solid_tier` | Balanced acc. 0.7743 – 0.9241 | Degrades on Polymer (0.774) | **CONDITIONAL** — rule recovery, chemistry-sensitive |
| `synthetic_capacity` | Li-S **−7.5966**, Na-ion **−7.1569**, Solid_State +0.7146 (R²) | Fails | **FAIL** for unseen chemistry |
| `synthetic_class` | Balanced acc. 0.5866 – 0.6307 | Near chance | **FAIL** for unseen chemistry |

No model should be described as "cross-chemistry validated". The EV tasks only look chemistry-robust because their targets are derivable formulas/signals.

---

## 4. Criterion-by-task acceptance matrix

Codes: **P** = PASS, **C** = CONDITIONAL, **F** = FAIL, **NV** = NOT VERIFIED, **B** = BLOCKED.

| Criterion | ev_soh | ev_capacity | ev_failure | solid_soh | solid_tier | synth_capacity | synth_class |
|---|---|---|---|---|---|---|---|
| Artifacts load and checksums verify | P | P | P | P | P | P | P |
| Deterministic / reproducible inference | P | P | P | P | P | P | P |
| Dataset provenance documented | P | P | P | P | P | P | P |
| Target provenance documented | NV | F | NV | NV | F | C (synthetic, labelled) | F |
| Target is independent of features | NV | F | NV | NV | F | C | F |
| Entity-level split meaningful | NV | NV | NV | NV | NV | C | C |
| No direct leakage columns | P | P | P | P | P | P | P |
| Train-only preprocessing | P | P | P | P | P | P | P |
| Validation-only selection | P | P | P | P | P | P | P |
| Beats naive baseline | P | P | P (balanced) | P | P | P (in-dist.) | P |
| Chemistry coverage in train = test | P | P | P | P | P | P | P |
| Cross-chemistry generalization | C | F (meaningless) | C | C | C | F | F |
| Calibrated uncertainty | C | C | C | C | C | C | C |
| Inference OOD / missing / unseen-chem safety | P | P | P | P | P | P | P |
| Range/sequence claims excluded | P | P | P | P | P | P | P |
| Production ready | F | F | F | F | F | F | F |

Notes on the non-obvious codes:

- **Target provenance `F` for `ev_capacity`, `solid_tier`, `synthetic_class`**: the target is a formula/rule by construction, so it is not an independent outcome.
- **Target provenance `NV` for `ev_soh`, `ev_failure`, `solid_soh`**: origin is undocumented and reconstruction is high.
- **Entity-level `NV` for all EV/solid tasks**: identifiers are row-unique synthetic keys; grouping is equivalent to a random row split.
- **Entity-level `C` for synthetic tasks**: 22 repeated IDs, 16 with conflicting metadata, correctly kept within a split.
- **Calibrated uncertainty `C` everywhere**:
  - regression uses an empirical validation-residual q90 interval (±5.8129 for `ev_soh`, ±6.3672 for `ev_capacity`, ±1.2708 for `solid_soh`, ±8.9592 for `synthetic_capacity`), not a calibrated predictive distribution. A post-hoc audit check found 90.2% / 88.2% / 88.7% / 90.0% empirical coverage of the stated 90% intervals on test, which is reasonable but not a calibration guarantee;
  - classification probabilities are explicitly `uncalibrated_class_probabilities`; probabilities are used directly and no calibration layer (isotonic/Platt) was fitted. The `ev_failure` default threshold yields a 13.25% positive rate against a 6.93% observed rate.
- **Production ready `F` everywhere**: `production_ready = false` in every registry entry, manifest, bundle contract and inference response.

Export traceability note: `ev_failure_predictions.csv` in the snapshot bundle has `snapshot_row_id = null` in all 4,000 rows (defect F-01, `member3_bundle.py:136`). Row alignment for that file currently works only through the unique `entity_id`. The model predictions themselves are unaffected. Exact patch and authorization requirement are in `MEMBER2_MEMBER3_INTEGRATION_CONTRACT.md` §9.

---

## 5. Task-level acceptance verdict

| Task | Status | Acceptable use today | Not acceptable use |
|---|---|---|---|
| `ev_soh` | **EXPERIMENTAL** | In-distribution SOH estimate on EV snapshot records from the same source, with ±5.81 q90 interval and explicit uncertainty | Measurement-grade SOH, temporal prediction, vehicle-level validation, RUL |
| `ev_capacity` | **EXPERIMENTAL** | Sanity-check reconstruction of the supplied capacity formula | Claiming measured-capacity accuracy |
| `ev_failure` | **EXPERIMENTAL** | Rare-class risk ranking (balanced accuracy 0.914); probability is uncalibrated | Quoting raw accuracy without the majority-baseline comparison; certified failure prediction |
| `solid_soh` | **EXPERIMENTAL** | In-distribution SOH estimate on solid-state snapshots (±1.27 q90) | Real cell-health measurement; chemistry transfer beyond the four trained electrolytes |
| `solid_tier` | **EXPERIMENTAL** | Recovering the supplied tier rule for triage-style ranking | Independent thermal-risk discovery |
| `synthetic_capacity` | **RESEARCH_ONLY** | Method development on synthetic data | Any real-world capacity claim, any unseen chemistry |
| `synthetic_class` | **RESEARCH_ONLY** | Method development on synthetic data | Health-state prediction, cross-chemistry use |

---

## 6. Verdict

**Software acceptance:** PASS — artifacts, registry, inference, tests all verify.
**Scientific acceptance:** CONDITIONAL at best; FAIL for `ev_capacity`, `synthetic_class` as independent targets and for synthetic cross-chemistry.
**Production acceptance:** FAIL for all seven tasks; every artifact states `production_ready = false`.

No task in this matrix should be promoted to production on the evidence available. The defensible claim is: *seven reproducible in-distribution snapshot models exist, each with honest scope limits, and none is validated outside its source dataset.*
