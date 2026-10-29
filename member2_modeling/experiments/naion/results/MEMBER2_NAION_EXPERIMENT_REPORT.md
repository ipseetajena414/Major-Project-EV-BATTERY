# Member 2 — Dedicated Na-ion SOH Experiment (LOCO)

**Scope:** Na-ion (`NaIon_Standard_Cycling`, 4 cells × 25 cycles) only.
No NMC or LFP number appears in this report and no Na-ion metric is combined
with them. No production artifact was read for inference, modified, exported or
registered. No branch, staging, commit or push action was taken.

**Code:** `member2_modeling/experiments/naion/`
**Results:** `member2_modeling/experiments/naion/results/`
**Tests:** 27/27 passing.

---

## 1. Decision summary

| Question | Answer |
|---|---|
| What can be predicted for Na-ion today? | One-step (next observed record, ≈1 cycle) SOH, at roughly **0.033 percentage points MAE**, using a training-only constant drift or a causal local trend. |
| Which baselines are credible? | **Pooled training-cell mean drift** and **causal per-cell local trend**. Both beat persistence in 4/4 held-out cells. |
| Does ML improve on them? | **No, not credibly.** Delta-target Ridge reaches 0.0402 pp (12-feature) / 0.0311 pp (15-feature sensitivity), statistically and practically indistinguishable from the drift baselines, and unstable across alpha selection. Absolute-target Ridge fails (0.3600 pp). |
| Is RUL supported? | **No.** Fade is ≤1.05 pp over 25 cycles and no cell crosses end-of-life in-record. |
| Is generalizable SOH prediction supported? | **No.** One protocol, four cells, no temperature, no hold-out beyond the same four cells. |
| Production candidate? | **None.** Na-ion remains unsupported in the production schema/manifest, unchanged. |

The honest headline is a negative ML result: a zero-feature drift baseline is as
good as the best learned model on this dataset, and the dataset is too small and
too shallow to justify anything stronger.

---

## 2. Data audit (measured, not assumed)

Input: `master_table.parquet`, read-only, SHA-256
`6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb`
(recorded in `results/naion_audit.json`).

| Check | Result |
|---|---|
| Na-ion rows | 41,701 |
| Dataset / protocol | `Zenodo_IonTech_NaIon` / `NaIon_Standard_Cycling` (single protocol) |
| Cells | 4: `NA-ion_270040-1-1-64`, `-1-2-63`, `-1-3-62`, `-1-4-61` |
| Cycles | 25 per cell, ids 1–25 contiguous, 100 total |
| Rows per cycle | 282–716 (cell-dependent sampling) |
| `temp_c` | **null in all 41,701 rows**; every row flagged `TEMP_UNMEASURED` |
| Cycle labels | constant within each cycle; `capacity_ah == soh_pct / 100` exactly (max deviation ~1e-16) |
| Duplicate `(cell, cycle, time_sec)` | 0 |
| Shared `(cell, time_sec)` across cycles | 34 timestamps / 68 rows — all cycle boundaries (end of cycle N equals start of cycle N+1); no duplicates inside a cycle |
| Time monotonic within cycle | no violations |
| Phase structure | every cycle has exactly one current-sign change; charge (`current_a > 0`) precedes discharge (`current_a < 0`) |
| Current ranges | ±2.0, ±5.5, ±4.2, ±5.8 A for cells 1–4 |
| Unit check | integrating `|current|·dt` per phase reproduces `capacity_ah`: discharge max abs error **0.00451 Ah** (mean 0.00148), charge max abs error **0.02406 Ah** (mean 0.00156) |

SOH trajectories and fade:

| Cell | First → last (%) | Δ (pp) | Decreasing / increasing steps |
|---|---|---|---|
| `-1-64` | 79.0241 → 78.6288 | −0.3953 | 21 / 3 |
| `-2-63` | 89.6524 → 88.5981 | −1.0543 | 23 / 1 |
| `-3-62` | 88.0663 → 87.5767 | −0.4896 | 23 / 1 |
| `-4-61` | 90.6596 → 90.3530 | −0.3066 | 22 / 2 |

Cell `-1-64` starts **below 80% SOH**, so it never crosses end-of-life during
the record; the other three stay 8–10 pp above it. No cell provides a
degradation-to-EOL trajectory.

These findings agree with `AUDIT/MEMBER2_CYCLE_LEVEL_DATA_QUALITY_REPORT.md`
(100 cycles, temperature unmeasured, mixed phases, shallow fade, RUL infeasible).
That report was cited, not modified.

---

## 3. Sample and target definition

- **Sample unit:** one `(cell_id, cycle_id)` row. Cycle-level SOH/capacity labels
  are broadcast across telemetry rows in the source table; repeated telemetry
  rows are never samples.
- **Target:** SOH (%) at the **next observed record** in that cell
  (`shift(-1)` per cell), not `cycle_id + 1` arithmetic.
- **Here the two coincide:** all 96 pairs have `cycle_gap == 1`, so next observed
  record = next physical cycle. The code does not assume this; gaps are detected
  and would be reported.
- **Samples:** 96 pairs (24 per cell). Every cycle except the last per cell is an
  input; every cycle except the first is a target.

---

## 4. Features (units justified, nothing manufactured)

Twelve predeclared, non-redundant inputs. Charge = `current_a > 0`; discharge =
`current_a < 0`, verified per cycle above.

| Feature | Unit | Source |
|---|---|---|
| `current_soh_pct` | % | cycle-level SOH label at time t (the only label-derived input) |
| `cycle_position` | cycles | 0-based observed-cycle index |
| `chg_duration_frac` | fraction | charge-phase time span / total cycle span |
| `chg_v_mean`, `dis_v_mean` | V | mean measured voltage per phase |
| `chg_v_q50`, `dis_v_q50` | V | median measured voltage per phase |
| `chg_v_slope_norm`, `dis_v_slope_norm` | V | OLS voltage-vs-time slope inside phase × phase span |
| `chg_v_range`, `dis_v_range` | V | peak-to-peak measured voltage per phase |
| `transition_voltage` | V | last measured voltage of the charge phase |

Full provenance, including trust level, is in
`results/naion_feature_provenance.csv`. **Not used:** `capacity_ah`, raw
`soh_pct` as a feature, temperature (unavailable), integrated Ah, current
magnitude, C-rate, EFC, SOC, calendar age, paper-formula coefficients. The unit
integration is a validation check only and never enters a model.

---

## 5. Validation protocol

- **Leave-one-cell-out (LOCO), 4 folds.** The held-out cell is never used for
  imputation, scaling, alpha selection or fitting. Imputer (median) and scaler
  are inside a scikit-learn `Pipeline` fitted per training fold.
- **Nested inner LOCO** over the three training cells selects Ridge alpha from
  `{0.001 … 1000}`; the selected alpha is refit on all three training cells.
- **Baselines:**
  - `persistence` — predict current SOH.
  - `pooled_mean_drift` — training cells' mean one-step delta added to current SOH.
  - `causal_cell_mean_drift` — expanding mean of *that cell's already observed*
    deltas; pair 0 falls back to the training-cell drift.
  - `causal_cell_trend` — anchored linear trend on that cell's observed SOH
    history (≥6 points), else the training drift.
  - `last_delta` — repeat the cell's previous observed delta.
  - `train_mean` — training-cell mean target (sanity reference).
- **Ridge targets:** absolute SOH and delta (`target − current`), both with the
  same 12 features.
- **Metrics:** MAE and RMSE in percentage points, n, R². R² is reported but is
  dominated by between-cell SOH level differences (79/89.7/88.1/90.7) and is not
  used to choose a model.

---

## 6. Results (n = 96 pairs, 4 LOCO folds)

Pooled across all held-out folds:

| Model | MAE (pp) | RMSE (pp) | R² | ΔMAE vs persistence | Beats persistence (cells) |
|---|---|---|---|---|---|
| persistence | 0.050043 | 0.079644 | 0.99970 | — | 0/4 |
| **pooled_mean_drift** | **0.032846** | 0.076882 | 0.99972 | −0.017197 | **4/4** |
| causal_cell_mean_drift | 0.070093 | 0.120260 | 0.99932 | +0.020050 | 1/4 |
| **causal_cell_trend** | **0.032003** | **0.076631** | 0.99972 | −0.018040 | **4/4** |
| last_delta | 0.043506 | 0.102364 | 0.99951 | −0.006537 | 3/4 |
| train_mean | 5.196548 | 6.127130 | −0.77492 | +5.146505 | 0/4 |
| ridge_absolute | 0.359994 | 0.499340 | 0.98821 | +0.309951 | 0/4 |
| ridge_delta | 0.040248 | 0.073231 | 0.99975 | −0.009795 | 3/4 |

Per-cell MAE (pp) for the primary models:

| Held-out cell | persistence | pooled_mean_drift | causal_cell_trend | ridge_delta |
|---|---|---|---|---|
| `-1-64` | 0.03448 | **0.01755** | 0.02342 | 0.05401 |
| `-2-63` | 0.06436 | 0.04919 | **0.03132** | 0.04809 |
| `-3-62` | 0.06033 | 0.03802 | 0.04317 | **0.03390** |
| `-4-61` | 0.04100 | 0.02663 | **0.03010** | 0.02500 |

Sensitivity (15-feature set adds three exact linear combinations of retained
voltages; `results/naion_sensitivity_redundant_features.csv`):

| Model | Pooled MAE (pp) | Per-cell MAE (pp) |
|---|---|---|
| ridge_absolute_15feat | 0.373013 | 0.2309 / 0.9864 / 0.0771 / 0.1976 |
| ridge_delta_15feat | 0.031124 | 0.0311 / 0.0346 / 0.0338 / 0.0249 |

One-step deltas across all pairs: mean −0.02339 pp, median −0.03385, min
−0.10621, max **+0.47914**, std 0.07653 pp. The single large positive jump (and
the 3–1 upward steps per cell) is why per-cell drift estimates are noisy.

Selected Ridge alphas (`results/naion_model_metadata.json`): absolute-target
Ridge chose 0.001–0.01 (effectively unregularized) and delta-target Ridge chose
100–1000. Moving from 12 to 15 features moved delta-Ridge pooled MAE from
0.0402 to 0.0311 pp, with different alphas per fold — model-selection variance
on four cells is larger than the gap between candidates.

---

## 7. Interpretation

1. **A constant drift is the credible signal.** All four cells fade by
  0.3–1.05 pp over 24 steps. Adding the mean one-step delta of the *other three
   cells* cuts MAE by 34% versus persistence and wins in every held-out cell.
   This is a training-only baseline with no features.
2. **A causal local trend is equally credible and needs no cross-cell transfer.**
   It wins in 4/4 cells and ties the pooled drift pooled (0.03200 vs 0.03285 pp).
   Its first six pairs use the training drift as a warm-up.
3. **Per-cell expanding drift is not credible** (0.0701 pp, worse than
   persistence in 3/4 cells): 1–23 noisy, occasionally positive deltas per cell
   are a poor drift estimator, while the cross-cell pooled mean is stable.
4. **Absolute-target Ridge fails** (0.360 pp): the held-out cell's SOH level is
   outside the three training levels, and the model extrapolates badly. This
   reproduces the known cross-cell level problem and is not a Na-ion-specific
   insight.
5. **Delta-target Ridge does not earn its complexity.** At 12 features it is
   worse than both credible baselines (0.0402 vs 0.0320–0.0328 pp) and better
   than persistence in only 3/4 cells. At 15 features it ties the baselines
   (0.0311 pp), but that configuration adds exactly collinear features, was
   reported only as a sensitivity check, and its alpha choices are unstable.
   There is no evidence that the voltage/time features add predictive value
   beyond a constant drift at this sample size.
6. **The metrics are sane.** `train_mean` degrades to 5.20 pp, confirming the
   evaluation is not accidentally trivial; `ridge_absolute` fails for a
   diagnosable reason; the pooled drift improves exactly where the trajectories
   fade.

R² values near 0.9997 are an artifact of comparing cells at different SOH
levels. Per-cell R² is 0.84–0.98 and simply reflects shallow monotone fade.
MAE in percentage points is the decision metric.

---

## 8. What cannot be validated

- **RUL.** No EOL crossing in-record (cell `-1-64` starts below 80%, the others
  end 8–10 pp above). Shallow fade and 24 steps per cell cannot identify a
  knee or an end-of-life time.
- **Multi-step / long-horizon SOH.** Only the one-step (next observed record)
  target is evaluated. Error compounds and is unmeasured beyond one step.
- **Generalization.** LOCO tests transfer among four cells of one protocol. It
  says nothing about other Na-ion chemistries, formats, C-rates or packs.
- **Temperature and thermal effects.** `temp_c` is null for every row; no
  thermal feature exists or was imputed.
- **Any ML advantage.** With four cells the confidence interval on a 0.008 pp
  MAE difference is far wider than the difference itself. The experiment cannot
  establish that learned voltage features help.
- **Production use.** The production schema and manifest still declare Na-ion
  SOH/RUL unsupported; this experiment does not change that contract.

---

## 9. Additional data that would change the answer

1. **More cells** (≥10–20), ideally from multiple packs, to make LOCO a real
   generalization test and to stabilize alpha/model selection.
2. **Deeper aging to EOL** (or at least through a knee) so RUL and
   degradation-phase models can be validated rather than assumed.
3. **Measured temperature** across cycles, plus controlled temperature
   variation, to test the thermal features that are currently impossible.
4. **Protocol/C-rate variation** so the model is not tied to
   `NaIon_Standard_Cycling`.
5. **Intermediate checkpoints / more cycles per cell** to test horizons longer
   than one step.

With those, current-derived features (Ah throughput, C-rate) would also become
defensible; on this dataset they were deliberately excluded even though the
unit check passed, because they add no validation power at 24 steps per cell.

---

## 10. Reproducibility and file inventory

Run from the repository root:

```bash
.venv/Scripts/python.exe member2_modeling/experiments/naion/naion_evaluation.py
.venv/Scripts/python.exe -m unittest discover -s member2_modeling/experiments/naion -p "test_*.py" -v
```

The evaluation is deterministic: fixed folds, fixed alpha grid, no shuffling,
no randomness. Re-running overwrites `results/` with identical values.

Files created (all under `member2_modeling/experiments/naion/`):

| File | Purpose |
|---|---|
| `naion_features.py` | table loading, audit, unit check, phase split, cycle features, next-observed pairs, provenance |
| `naion_evaluation.py` | LOCO folds, baselines, train-only Ridge, metrics, result writing, CLI |
| `test_naion_experiment.py` | 27 focused tests |
| `results/naion_audit.json` | full data audit incl. hash, duplicates, phase structure, units |
| `results/naion_feature_provenance.csv` | feature units, derivation, trust level |
| `results/naion_cycle_features.csv` | 100 cycle-level rows with features and labels |
| `results/naion_pairs.csv` | 96 next-observed pairs |
| `results/naion_metrics_pooled.csv` | pooled metrics per model |
| `results/naion_metrics_by_cell.csv` | per-held-out-cell metrics per model |
| `results/naion_predictions.csv` | every prediction with error |
| `results/naion_model_metadata.json` | alphas, inner-LOCO scores, features per fold |
| `results/naion_sensitivity_redundant_features.csv` | 15-feature pooled sensitivity |
| `results/naion_sensitivity_by_cell.csv` | 15-feature per-cell sensitivity |
| `results/naion_summary.json` | machine-readable decision summary |

Not modified: `master_table.parquet`, raw data, `member2_modeling/feature_schema.json`,
`model_manifest.json`, `predictor.py`, `build_and_train.py`, model/scaler weights,
the dataset-building pipeline, and all LFP/NMC experiment results.

---

## 11. Final answer

For Na-ion, on this dataset, the defensible product is a **one-cycle-ahead SOH
drift estimate**, not a learned degradation model: apply the mean one-step drift
of other cells (MAE 0.0328 pp) or a short causal local trend (MAE 0.0320 pp) to
the current SOH. Persistence is worse (0.0500 pp) and a learned Ridge model is
not better than the drift baselines. RUL, long-horizon SOH and cross-condition
generalization remain unsupported until more cells, deeper aging, and measured
temperature are available.
