# Member 2 — Formula Derivation and ML Plan

**Project:** Autonomous Multi-Agent EV Battery Life & Range Predictor  
**Module:** Member 2 — feature engineering, SOH/capacity prediction, RUL evaluation, Member 4 integration  
**Date:** 2026-10-29  
**Status:** derivation/feasibility plan before implementation; no production model artifact is replaced by this plan.

---

## 1. Source files and access status

| Source | Repository status | Content status |
|---|---|---|
| `FORMULA/paper_formula_register (1).csv` | Present, 15,051 bytes, 16 formula rows, 16 columns | Parsed in full. Filename carries a `(1)` suffix; content is the 16-entry register described in the task. |
| `paper_access_status (1).csv` | **Not present in the repository**; supplied as an uploaded attachment (19,073 bytes) | Parsed: 49 `Papers` rows plus 10 `Top10` duplicate rows. Most paper rows are `METADATA ONLY - full text not accessed`. |
| Paffumi & Martini (2021), DOI `10.3390/wevj12010048` | No local PDF | Register states `FULL TEXT READ (MDPI open access HTML)`. Independent re-fetch in this pass returned HTTP 403; equations/parameters below are therefore **register-captured, not independently re-verified here**. |
| Cristea et al. (2026), DOI `10.3390/app16157756` | No local PDF | Register states `FULL TEXT READ (MDPI open access HTML); coefficient Tables 4-5 NOT captured`. Independent re-fetch returned HTTP 403. Coefficients are unavailable. |
| Original sources credited by both papers | Not present | **Not read or checked.** The register says the papers attribute most equations to earlier references (Wang 2014/2016, Purewal 2014, Cordoba-Arenas 2015, Liu 2026, Najera 2023, etc.). |

**Rule applied:** a formula is not “validated” because it is printed in a paper. A formula is implementable only when its source, parameters, units, chemistry applicability and all required inputs are verified.

---

## 2. Data and input audit (actual `master_table.parquet`)

Verified file identity: SHA-256 `6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb`, 515,270 rows × 13 columns.

| Input | NMC (NASA) | LFP B5 | Na-ion |
|---|---|---|---|
| Current | Amperes; discharge about −2 A, small near-zero positive samples at the end; register/builder says A | Constant −0.625; quality flag `CURRENT_UNIT_UNDERDETERMINED`; **do not assume A** | Mixed charge/discharge; excluded from Member 2 models |
| Temperature | Present, 22.4–42.0 °C, no nulls | Missing for all rows | Missing for all rows |
| Time | `time_sec` resets per cycle; raw row order is not time-sorted | `time_sec` present; discharge-only | `time_sec` present; mixed phases |
| Capacity / SOH | `capacity_ah` and `soh_pct` broadcast per cycle; SOH relative to 2.0 Ah | `capacity_ah` from B5 summary; SOH conditional, 1.25 Ah rated; cycle 40 null | Per-cell nominal from source; excluded |
| SOC / DoD / SOC_min | Not present | Not present | Not present |
| Calendar age / absolute dates | Not present | Not present | Not present |
| Cycle count | Observed discharge-cycle position 1–168 per cell | 250 observed IDs in blocks 1–100, 151–200, 401–500; 250 IDs absent | 25 observed per cell |

Additional verified facts:
- B5 has 250 observed cycle IDs; 101–150 (50) and 201–400 (200) are absent.
- B5 `cycle_id=40` has a null SOH label; the pipeline drops it, leaving 249 labelled cycles and 248 model-ready input→target pairs.
- B5 gap-crossing model-ready pairs are 39→41 (train), 100→151 (train), 200→401 (val); the test split has none.
- The LFP target is the **next observed record**, not necessarily the next physical cycle.
- For NMC, discharge-only trapezoidal integration of |I|·dt gives Ah values within a mean ratio of 1.0079 of the broadcast `capacity_ah` label (range 0.99996–1.0287), supporting the A unit and integration method for NMC only.

---

## 3. Formula disposition summary

Full row-level evidence, inputs and leakage notes are in `AUDIT/MEMBER2_FORMULA_IMPLEMENTATION_MATRIX.csv`.

| Status | Count | Formulas |
|---|---:|---|
| **IMPLEMENTABLE** | 0 | None. No registered formula passes every source/parameter/input/chemistry check. |
| **ADAPTABLE** | 2 | `P1-EoL` (80% EoL as an explicit project adaptation), `C-Eq10-12` (cell-level EFC adaptation for NMC only) |
| **CONDITIONAL** | 0 | — |
| **SCENARIO_ONLY** | 6 | `P1-Eq5`, `P1-HVAC`, `P1-Eff`, `P1-Cons`, `C-Eq3`, `C-Eq16-17` |
| **BLOCKED** | 8 | `P1-Eq1`, `P1-Eq3`, `C-Eq1`, `C-Eq2`, `C-Eq4`, `C-Eq5-9`, `C-Eq13-14`, `C-Eq15` |

Why the key candidates are not implementable:
- `P1-Eq1`: NCM-LMO parameterization, output loss unit not stated, calendar age unavailable, original refs 14–15 not read. Applying it to NASA NMC/LFP/Na-ion because the chemistry names sound similar is prohibited.
- `P1-Eq3`: printed activation-energy exponent has the opposite sign to `P1-Eq1`; `SOC_min` is unavailable in every dataset; parameters are NCM-LMO only; original ref 16 not read. Keep blocked.
- `P1-Eq5`: vehicle-level reserve assumption (25% PHEV / 15% BEV), not a measured cell quantity.
- `C-Eq1`/`C-Eq2`/`C-Eq4`/`C-Eq5-9`/`C-Eq13-14`: Tables 4–5 coefficients were not captured and could not be re-fetched; SoC, DoD, calendar time and/or physical cycle counts are unavailable. Do not invent coefficients.
- `C-Eq15`: the paper’s “SoH = 1 − Q_loss_tot” is a **model output** on a 0–1 scale; it must never replace or be merged with measured `soh_pct`.
- `C-Eq3`/`C-Eq16-17`: pack-level kWh. Cell Ah is not interchangeable with pack kWh; Member 3 owns the pack/range scenario.

---

## 4. Cleared derived features

Only features that pass the unit, provenance and missing-data checks are implemented. All are written to the isolated experimental pipeline, never to the production artifacts.

### 4.1 NMC discharge Ah throughput (telemetry-derived)

Definition:

```
Ah_discharge = Σ 0.5·(|I_k| + |I_{k+1}|)·Δt_k / 3600
```

over the discharge phase (`current_a < 0`), after sorting by `time_sec`, dropping duplicate timestamps, and clipping `Δt` to non-negative values. Units: Ah. `current_a` is verified in A for NMC. This is **not** a paper formula; it is a deterministic telemetry integral with an independent check against `capacity_ah` (mean ratio 1.0079).

### 4.2 NMC equivalent full cycles (project adaptation of C-Eq10-12)

The paper’s `EFC = W_batt / C_nom` uses vehicle throughput energy. The cell-level adaptation is:

```
EFC_cumulative = Σ observed discharge Ah / 2.0 Ah
```

where 2.0 Ah is the NASA NMC nominal rated capacity from `master_table_build.py`. Summation is over **observed records only**; no missing B5-style records exist for NMC, and no interpolation is performed. This is explicitly a **project adaptation**, not the paper’s vehicle-level equation. LFP EFC is blocked because `current_a` units are underdetermined.

### 4.3 NMC thermal-stress exposure (telemetry-derived)

Using timestamps and the trapezoidal midpoint temperature over each interval:

```
hours_outside_20_30 = Σ Δt/3600 for midpoint T outside [20, 30] °C
degree_hours_above_30 = Σ max(T_mid − 30, 0)·Δt/3600
degree_hours_below_20 = Σ max(20 − T_mid, 0)·Δt/3600
```

Units: h and °C·h. This is **time-weighted**, not sample-count-weighted. NMC only. LFP/Na-ion are left missing, never imputed.

### 4.4 NMC voltage at discharge-capacity fractions (telemetry-derived)

After constructing cumulative discharge capacity Q, interpolate voltage at 20%, 50% and 80% of that cycle’s total discharge Ah (`v_at_q20`, `v_at_q50`, `v_at_q80`). Units: V. This is a deterministic curve-shape descriptor, not a registered paper formula. It avoids dQ/dV peak selection and division by near-zero voltage increments.

### 4.5 Incremental-capacity diagnostic (not a model feature)

`dV/dQ` is computed over the 20–80% Q range on a uniform Q grid with Savitzky–Golay smoothing, and summarized as median and IQR. It is reported as a **feasibility diagnostic only**. No peak detection, no dQ/dV model feature, and no LFP application. Rationale: the register contains no verified ICA/DVA method, and the corresponding paper in the access list is metadata-only.

### 4.6 Explicitly not implemented

- Any calendar/cycle capacity-loss curve (`P1-Eq1/3/5`, `C-Eq1/2/4/13-14`).
- Any SOC/DoD/SOC_min-derived feature.
- Any LFP current integral or LFP EFC.
- Any formula-generated SOH label.
- Any imputation of B5 cycle 40, missing cycle IDs, or missing temperatures.
- Any range/HVAC/charging-efficiency model (scenario-only; Member 3).

---

## 5. ML experiment plan (isolated)

Location: `member2_modeling/experiments/`. No production weight, scaler, manifest or schema is overwritten.

### 5.1 Data and splits

- Reuse the production feature extractor `build_cycle_feature_table` and the production chronological split (`60/20/20` per cell, positional, no shuffle) for comparability.
- Build future-SOH pairs with positional `shift(-1)`; target = next observed record.
- NMC: 465 pairs (278/92/95). LFP: 248 pairs (148/49/51).
- Evaluate every model on exactly the same test target rows for a given chemistry.
- LFP results are reported separately for adjacent observed transitions and gap-crossing transitions. The test split has no gap-crossing pair; the 200→401 pair is in validation, so validation gap error is reported separately and never presented as test performance.

### 5.2 Baselines and models

For each chemistry and feature set:

1. **Persistence** — predict `target_soh_pct = current_soh_pct`.
2. **Mean-drift** — predict `current_soh_pct + mean(train target − current)`; added during execution because a one-step decay is systematic and a feature-free drift baseline is needed to test whether learned models add value.
3. **Linear trend** — per-cell ordinary least squares of target SOH on input cycle position, fit on train, evaluated on the same test rows. If a cell has too few train pairs, fall back to the pooled train trend and report the fallback.
4. **Ridge regression** — median-impute train statistics, standardize on train, select `alpha` on validation only, evaluate once on test.
5. **Existing LightGBM** — production hyperparameters and early stopping on validation.
6. **Delta LightGBM** — target is `target_soh_pct − current_soh_pct`; prediction adds `current_soh_pct` back. Treated as an experiment, not a guaranteed improvement.

### 5.3 Feature sets and ablation

NMC:
- `prod`: existing 37 production features.
- `prod + ah_discharge`: corrected NMC discharge Ah.
- `prod + efc_cumulative`: cumulative EFC adaptation.
- `prod + thermal`: time-weighted thermal exposure features.
- `prod + v_at_q`: voltage-at-capacity fractions.
- `prod + all_cleared`: all of the above.
- Ablation decision uses **validation MAE only**. The selected set is evaluated once on test. No repeated test tuning.

LFP:
- `prod`: existing 29 features (includes current-derived features that are not unit-verified).
- `unit_safe`: voltage/time/shape features only; all current-derived features removed.
- No new formula-derived LFP features exist because current units and temperature are unavailable.

### 5.4 Metrics and decision rule

Report MAE, RMSE, R², n, feature set, split, and persistence comparison for every model. A candidate is called an improvement only if it beats persistence on the **same test rows** with a lower MAE. Otherwise the report states that no improvement was achieved. Test metrics are not used for tuning.

---

## 6. RUL reassessment plan

- Preserve the existing conclusion: pre-EOL RUL forecasting is unvalidated.
- Confirm the production test problem: chronological test targets are 97/97 post-EOL (RUL = 0), so the reported MAE cannot demonstrate pre-EOL capability.
- Build RUL labels only from observed NMC trajectories and the documented 80% cell-SOH project adaptation of `P1-EoL`.
- Pre-EOL rows are those before the first observed 80% crossing (`RUL > 0`); post-EOL rows are excluded from pre-EOL evaluation.
- Use **leave-one-cell-out** validation across the three NASA cells: train on two cells’ pre-EOL rows, test on the held-out cell’s pre-EOL rows. Report each fold and pooled metrics.
- Baselines: (a) constant median training RUL; (b) linear SOH-threshold extrapolation using the training median SOH slope per cycle. Compare a small LightGBM with the same pre-EOL splits.
- With only three cells, this is a **small-sample diagnostic**, not a validated production RUL model. If the folds are unstable or the baseline is not beaten, document the blocker.

---

## 7. Missing-data and leakage rules

1. Never impute temperature, SOC, DoD, calendar age, capacity or SOH labels.
2. Never interpolate B5 cycle IDs or treat missing cycles as observed.
3. Never replace B5 cycle 40’s null SOH with a formula estimate.
4. Never train a measured-SOH target on formula-generated SOH.
5. Keep measured SOH, deterministic telemetry features, formula estimates and ML predictions in separate fields with provenance.
6. Compute cumulative features from current/past observed records only; no future information.
7. Split chronologically per cell before fitting models; no shuffling.
8. Do not use the test set for feature selection or hyperparameter tuning.

---

## 8. Deliverables and stop conditions

Planned outputs:
- `AUDIT/MEMBER2_FORMULA_IMPLEMENTATION_MATRIX.csv` — completed.
- `AUDIT/MEMBER2_FORMULA_DERIVATION_AND_ML_PLAN.md` — this file.
- `member2_modeling/experiments/experimental_features.py` — cleared feature code and provenance flags.
- `member2_modeling/experiments/test_experimental_features.py` — hand-calculated unit tests.
- `member2_modeling/experiments/run_experiments.py` — isolated baselines, models, ablations, LFP adjacent/gap reporting and RUL pre-EOL diagnostic.
- `member2_modeling/experiments/results/` — metrics JSON/CSV and candidate artifacts, clearly marked experimental.
- `AUDIT/MEMBER2_FORMULA_FEATURE_EXPERIMENT_REPORT.md` — results, blockers, conclusions.

Stop conditions:
- If a formula lacks a verified parameter table, unit, chemistry match or input, it remains blocked.
- If a feature cannot be computed for a chemistry without assuming an unverified unit, it is not computed for that chemistry.
- If no candidate beats persistence on equivalent test rows, report “no improvement achieved.”
- Do not overwrite production weights/scalers/schema/manifest, do not modify `master_table.parquet` or `master_table_build.py`, and do not commit or push.

---

## 9. Execution outcome

Executed 2026-10-29. Results are in `member2_modeling/experiments/results/` and are analysed in `AUDIT/MEMBER2_FORMULA_FEATURE_EXPERIMENT_REPORT.md`.

- No registered formula reached IMPLEMENTABLE; two adaptations were tested as NMC features (discharge Ah, cumulative EFC) alongside deterministic thermal and voltage-shape features.
- NMC: no feature-based model beat the feature-free mean-drift baseline. The mean-drift baseline itself beat persistence on validation and test.
- LFP: ridge regression on the existing production features (and on a unit-safe subset) beat persistence; this remains an experimental candidate, not a production replacement.
- RUL: the production test set is confirmed to be 97/97 post-EOL; the pre-EOL leave-one-cell-out diagnostic is small-sample only and does not validate production RUL.

