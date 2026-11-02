# Member 2 — Target Provenance and Label-Leakage Audit

**Audit date:** 2026-11-02
**Scope:** every target modelled by the seven registered snapshot tasks.
**Method:** read-only inspection of `data/processed/data_dictionary.csv`, `member2_modeling/data_integration/profiles.py`, each task's `config.json` / `metrics.json`, plus read-only reconstruction probes executed against the processed datasets.
**Important:** the probes below are *audit diagnostics*. They were run on the full dataset (or training data) to test whether a label is reconstructable. **None of them replaces or reruns the frozen test evaluation, and none of them is reported as a model metric.**

Labels: **VERIFIED**, **EXPERIMENTAL**, **UNVERIFIED**, **NOT SUPPORTED**, **BLOCKED**.

---

## 1. Target provenance table

| Task | Dataset | Target | Origin classification | Evidence | Verdict |
|---|---|---|---|---|---|
| `ev_soh` | EV | `state_of_health` | Supplied label; origin undocumented | Agrees with `battery_health_percent` at r = 0.9898; differs by >1 point in 8,008/20,000 rows (max 6.22); reconstructable from permitted telemetry at R² = 0.8177 (5-fold linear) | **UNVERIFIED** as an independent measurement |
| `ev_capacity` | EV | `remaining_capacity` | **Algebraically derived** | `= battery_capacity_kwh × battery_health_percent / 100`; max abs residual 0.005 kWh; only 6/20,000 rows exceed 0.005 | **VERIFIED derived** |
| `ev_failure` | EV | `battery_failure` | Supplied binary label; generating rule undocumented | 1,384/20,000 positive (6.92%); not recoverable by any single column, but depth-10 tree reaches train balanced accuracy 0.9589 | **UNVERIFIED origin; label-recovery risk** |
| `solid_soh` | Solid-state | `state_of_health_pct` | Supplied label; origin undocumented | r = −0.943 with `cycle_number`, −0.875 with `internal_resistance_mohm`, +0.903 with `coulombic_efficiency_pct`; 5-fold linear R² = 0.9765 | **UNVERIFIED**; strongly consistent with simulated telemetry |
| `solid_tier` | Solid-state | `degradation_risk_tier` | **Rule-derived from SOH plus other signals** | Tier SOH ranges overlap (Tier 4 70.00–100.00, Tier 3 78.00–99.21, Tier 2 85.02–92.00, Tier 1 92.01–100.00); SOH-only tree 0.8461; SOH+cycle+temperature+resistance 0.8826; all permitted features unlimited depth **1.0000** | **VERIFIED rule-generated** |
| `synthetic_capacity` | Synthetic | `discharge_capacity_ah` | **Synthetic generated** | `data_origin = synthetic`; generator not shipped; values within per-chemistry bands | **VERIFIED synthetic** |
| `synthetic_class` | Synthetic | `state_of_health` | **Rule-generated per chemistry from capacity** | Per-chemistry capacity thresholds reproduce the class with **100% agreement** (see §5); capacity excluded as a feature | **VERIFIED derived + synthetic** |

---

## 2. EV SOH and remaining capacity (deep dive)

### 2.1 The capacity identity

For all 20,000 EV rows:

- `remaining_capacity − battery_capacity_kwh × battery_health_percent / 100` has maximum absolute value **0.005000**; only **6 rows** exceed 0.005.
- `capacity_loss_percent = 100 − battery_health_percent` holds to floating-point precision (max residual 7.1e-15).
- `battery_capacity_kwh` is a permitted feature for `ev_capacity` with 9,771 distinct values.

Therefore `remaining_capacity` is not an independent pack-energy measurement. It is the product of a feature and a hidden label. Because `battery_health_percent` is excluded but its linear factor is present, the reported MAE 3.209 / R² 0.970 should be read as **formula reconstruction accuracy**, not capacity-estimation accuracy. This matches the warning already in `specs.py`, but the data-dictionary string does not yet say so.

### 2.2 The two SOH labels

- `state_of_health` vs `battery_health_percent`: r = **0.9898**, mean absolute difference small, but **8,008 / 20,000 rows differ by more than 1 point**, maximum difference **6.22**.
- Capacity is tied to `battery_health_percent`, not to the modelled `state_of_health`.
- A 5-fold linear model on the 56 permitted features reconstructs `state_of_health` with **R² = 0.8177 (sd 0.0038)**.

Interpretation: `state_of_health` is strongly determined by the telemetry available as features. Either it was generated from those signals, or the signals are so informative that the task is close to deterministic. **Independent measurement is UNVERIFIED.**

### 2.3 EV failure label

- Positive class 1,384 / 20,000 (6.92%), so accuracy alone is misleading; balanced accuracy is the right selection metric.
- Single-feature AUC (as-is sign):

| Feature | AUC |
|---|---:|
| `thermal_runaway_risk` | 0.842 |
| `capacity_loss_percent` (excluded label) | 0.819 |
| `internal_resistance` (excluded, unit-ambiguous) | 0.800 |
| `state_of_health` (excluded) | 0.184 (i.e. 0.816 inverted) |

- No single permitted column reaches the final model's balanced accuracy.
- A depth-10 tree fitted on all permitted features reaches **train balanced accuracy 0.9589** (train-fit, not validation).
- Conclusion: the label is **not** a simple one-line rule over one column, but it is largely recoverable from the permitted feature set. This is **label-recovery risk**, not proof of independent failure prediction. The origin of `battery_failure` is not documented; it may be a probabilistic or multi-signal generator.

---

## 3. Solid-state SOH and risk tier (deep dive)

### 3.1 SOH

`state_of_health_pct` is almost linear in other telemetry: `cycle_number` r = −0.943, `internal_resistance_mohm` r = −0.875, `coulombic_efficiency_pct` r = +0.903, and a 5-fold linear probe on permitted features reaches **R² = 0.9765**. The dataset is a single snapshot per cell with no timestamps. The pattern is fully consistent with a simulator, but the generator is not shipped. **Origin UNVERIFIED.**

### 3.2 Risk tier

The tier is not a pure SOH threshold — the SOH ranges overlap:

| Tier | n | SOH min | SOH max | SOH mean |
|---|---:|---:|---:|---:|
| `Accelerated_Degradation_Tier_3` | 3,758 | 78.00 | 99.21 | 83.46 |
| `Critical_Thermal_Risk_Tier_4` | 1,682 | 70.00 | 100.00 | 79.33 |
| `Minimal_Degradation_Tier_1` | 2,278 | 92.01 | 100.00 | 95.26 |
| `Moderate_Degradation_Tier_2` | 2,282 | 85.02 | 92.00 | 88.71 |

Reconstruction probes (train data, balanced accuracy):

| Inputs | Balanced accuracy |
|---|---:|
| SOH only (depth 6) | 0.8461 |
| SOH + cycle + operating temperature + internal resistance (depth 6) | 0.8826 |
| All permitted features (depth 10) | 0.9324 |
| All permitted features (unlimited depth) | **1.0000** |

The label is rule-generated from SOH and additional signals. SOH is correctly excluded, but the remaining signals determine the tier exactly. The model is therefore reconstructing the tiering rule, not discovering an independent risk process. **VERIFIED rule-generated; predictive value conditional on the rule.**

---

## 4. Synthetic capacity

`SYNTHETIC_BATTERY_LIFECYCLE` is registered with `data_origin = synthetic`. `discharge_capacity_ah` is generated inside the same campaign that generates the class label and the telemetry. There is no measured cell. Reported MAE 3.8176 / R² 0.9331 is **synthetic-data fit**, not battery validation.

A cross-chemistry diagnostic (train on the other two chemistries, score the held-out one) fails:

| Held-out chemistry | R² | MAE (Ah) | Train-median MAE |
|---|---:|---:|---:|
| Lithium_Sulfur | **−7.5966** | 29.82 | 29.24 |
| Sodium_Ion | **−7.1569** | 28.75 | 29.87 |
| Solid_State | +0.7146 | 4.04 | 9.37 |

The pooled test metric (R² 0.9331) is carried by Solid_State and by chemistry being available as a feature; the model does **not** transfer across synthetic chemistries. **RESEARCH_ONLY / NOT SUPPORTED for unseen chemistry.**

---

## 5. Synthetic class

The class is an **exact per-chemistry capacity threshold**. Applying these rules reproduces `state_of_health` with **100% agreement (2,000/2,000)**:

| Chemistry | End_of_Life | Degraded | Healthy |
|---|---|---|---|
| Lithium_Sulfur | — | capacity ≤ 95.95 | capacity ≥ 96.17 |
| Sodium_Ion | capacity ≤ 47.80 | 48.01 – 63.99 | ≥ 64.01 |
| Solid_State | capacity ≤ 59.97 | 60.04 – 79.94 | ≥ 80.01 |

`discharge_capacity_ah` is correctly excluded as a feature. Nevertheless:

- A depth-10 tree on permitted features reaches train balanced accuracy **0.9061**.
- An unlimited-depth tree reaches **1.0000**.
- A cross-chemistry holdout gives balanced accuracy 0.5866 (Sodium_Ion) / 0.6292 (Lithium_Sulfur) / 0.6307 (Solid_State), i.e. near chance once the chemistry-specific capacity distribution is unseen.

Conclusion: `synthetic_class` reconstructs a threshold rule from correlated synthetic signals. It is **label recovery**, not independent health-state prediction, and it does not generalize across chemistries.

---

## 6. Feature-to-target leakage report

### 6.1 Direct leakage controls (all VERIFIED)

| Control | Effective? | Evidence |
|---|---|---|
| Target removed from features | Yes | `feature_columns_for_target`, pipeline raises on target |
| Related labels removed | Yes | Forbidden lists in each `config.json`; `forbidden_used_as_feature = 0` |
| Derived-score columns excluded | Yes | Data-dictionary role `feature_candidate_derived` |
| Unit-ambiguous columns excluded | Yes | `feature_candidate_unit_ambiguous` |
| Projections / rationale / confidence excluded | Yes | `excluded_*` roles |
| Identifier / provenance columns excluded | Yes | Role filter + EV identifier exclusion |
| Preprocessing fitted on train only | Yes | `TabularFeaturePipeline.fit(data.iloc[split.train])` |
| Inference ignores supplied target columns | Yes | Warning + drop in `inference.py` |

### 6.2 Indirect leakage / reconstruction risk (NOT removable by exclusion)

| Task | Mechanism | Strength | Consequence |
|---|---|---|---|
| `ev_capacity` | Target = feature × hidden label | Exact (0.005 kWh) | Performance is formula reconstruction |
| `ev_soh` | Target determined by permitted telemetry | Linear R² 0.8177 | High R² is partly label reconstruction |
| `ev_failure` | Label recoverable from permitted features | Tree train bal. acc. 0.9589 | Cannot separate "predicts failure" from "recovers generator" |
| `solid_soh` | Nearly linear in permitted signals | Linear R² 0.9765 | Simulated-data fit, not measurement |
| `solid_tier` | Rule over permitted signals + SOH | Exact at unlimited depth | Rule reconstruction |
| `synthetic_capacity` | Synthetic generator | R² 0.9331 in-distribution, negative cross-chemistry | Not real-world evidence |
| `synthetic_class` | Exact capacity threshold | 100% rule agreement | Label recovery |

**No direct column leakage was found in any task.** The residual risk is *target construction*, not feature handling.

---

## 7. Documentation defect found (resolved 2026-11-02)

`member2_modeling/data_integration/profiles.py:102` previously defined `remaining_capacity` as:

> `"Measured remaining pack capacity in the snapshot."`

Status: resolved. The source, registry, model config and generated reports now describe the algebraic derivation and state that it is verified only on rows where both inputs are present.

This contradicts the verified algebraic identity. It is a provenance/documentation defect because the string is copied into each `config.json` (`target.definition`) and surfaced to Member 3.

**Proposed minimal change (not applied, because frozen artifacts would then disagree with source):**

```python
# member2_modeling/data_integration/profiles.py:102
"Pack capacity reconstructed as battery_capacity_kwh * battery_health_percent / 100 "
"(verified to within 0.005 kWh); algebraically derived, not an independent measurement.",
```

Applying this without regenerating `config.json` / `model_registry.json` would desynchronise code and artifacts. It should be applied only as part of an authorized re-freeze, together with the matching artifact regeneration. The audit reports it rather than silently editing or regenerating.

---

## 8. Verdict

| Question | Verdict |
|---|---|
| Is any target directly measured and independently verified? | **NOT SUPPORTED** — no target has documented measurement provenance |
| Are labels used as features? | **VERIFIED no** |
| Are derived targets identified? | **VERIFIED yes** — `ev_capacity`, `solid_tier`, `synthetic_class` |
| Is `synthetic_class` more than a threshold rule? | **NOT SUPPORTED** — exact rule, 100% agreement |
| Is `ev_capacity` more than formula reconstruction? | **NOT SUPPORTED** — algebraic identity |
| Is `ev_failure` independent failure prediction? | **UNVERIFIED** — recoverable, origin undocumented |
| Are `ev_soh` / `solid_soh` independent measurements? | **UNVERIFIED** — near-deterministic from telemetry |
| Is `synthetic_capacity` real-world evidence? | **NOT SUPPORTED** — synthetic, fails cross-chemistry |

**Overall:** the leakage *plumbing* is sound. The scientific meaning of the targets is not. Three of seven tasks reconstruct a formula/rule by construction; the remaining four have undocumented origins and high reconstruction rates. This must be stated explicitly in any Member 3 handover.
