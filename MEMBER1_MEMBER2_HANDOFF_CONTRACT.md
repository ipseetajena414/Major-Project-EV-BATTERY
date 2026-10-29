# MEMBER 1 → MEMBER 2 HANDOFF CONTRACT

**Project:** Autonomous Multi-Agent EV Battery Life & Range Predictor
**From:** Member 1 (data engineering and data quality)
**To:** Member 2 (SOH / capacity modelling and validation)
**Date:** 2026-10-29
**Status:** Conditional handoff. No LFP target is fully verified ground truth.

---

## 1. What is delivered

All paths are relative to the repository root `C:\Users\KIIT0001\Desktop\MAJOR PROJ 2`.

| Artefact | Path | Grain | Rows | Use |
|---|---|---|---|---|
| B2 cycle features | `DATA/B2_LFP_Member1_audit.zip :: LFP_Member1_audit/exports/cycle_features.parquet` | cycle | 1,506 × 255 | Primary modelling table |
| B2 Q–V curves | same zip `:: exports/qv_curves.parquet` | cycle | 1,506 × 3 (101-point curves) | Curve features / validation |
| B2 candidate labels | `AUDIT/candidates/B2_candidate_cycle_labels.csv` | cycle | 1,506 × 27 | Re-derived labels + provenance |
| B2 channel mask | `AUDIT/candidates/B2_channel_transition_mask.csv` | cycle | 1,506 | Exclusion/covariate treatment |
| B2 treatment summary | `AUDIT/candidates/B2_channel_transition_summary.json` | summary | — | Split counts |
| B1 cycle labels | `AUDIT/candidates/B1_candidate_cycle_labels.csv` | cycle | 1,460 × 19 | Corrected labels (canonical bug fixed) |
| B3 cycle labels | `AUDIT/candidates/B3_candidate_cycle_labels.csv` | cycle | 1,979 × 18 | Conditional labels |
| B4 cycle inventory | `AUDIT/candidates/B4_candidate_cycle_labels.csv` | cycle | 1,577 × 14 | Telemetry only, no target |
| B5 cycle labels | `AUDIT/candidates/B5_candidate_cycle_labels.csv` | cycle | 1,498 × 21 | Conditional labels |
| B6 cycle chronology | `AUDIT/candidates/B6_candidate_cycle_chronology.csv` | cycle | 1,702 × 30 | Validated chronology + labels |
| Sampled telemetry | `AUDIT/candidates/telemetry_B1..B6.parquet` | sample | see summaries | 50 charge + 50 discharge points/cycle |
| Candidate master | `master_table_candidate.parquet` / `.csv` | sample | see validation JSON | Unified sample table |
| Feature/target manifest | `AUDIT/MEMBER1_FEATURE_TARGET_MANIFEST.csv` | column | 255 | Leakage control |
| Target definitions | `AUDIT/MEMBER1_TARGET_DEFINITIONS.csv` | dataset | 8 | Denominator/units |

The canonical `master_table.parquet` / `.csv` were **not** modified. The candidate is separate and must not be promoted without review.

---

## 2. Dataset ranking for modelling

Use in this order, with the stated caveats:

1. **B2** — 1,496 analysis-eligible cycles; 10 invalid; cycle-level features and 101-point Q–V curves; SOH is a **0–1 fraction**. Handle the channel transition.
2. **B1** — 1,456 computed labels of 1,460 cycles; 110 cycles have underdetermined offsets; 4 partial-discharge cycles blocked.
3. **B6** — 1,697 conditional labels of 1,702 cycles; chronology reconstructed from sheet header Date/Time and File List timestamps (no global cycle id in source).
4. **B3** — 1,973 labels of 1,979 cycles; 8 quality-X and 3 quality-B cycles; four instrument channels without documented physical cell identity.
5. **B5** — 1,498 cycles; `capacity_ground_truth = NO` for every cycle; Qdis > Qchg in 1,488/1,498; 6 partial discharges including cycle 40.
6. **B4** — 1,577 cycles of telemetry; **no audited capacity or SOH**. Use only for unsupervised/auxiliary work, never as a labelled target.

NASA NMC and Na-ion are available in the candidate master (direct capacity, NASA has measured temperature; Na-ion has none).

---

## 3. Targets

| Dataset | Target column | Formula | Unit | Status |
|---|---|---|---|---|
| B2 | `soh_rated` | `q_discharge_ah / 1.25 Ah` | fraction 0–1 | CONDITIONAL |
| B1 | `soh_pct_nominal_1.25Ah` | `capacity_ah / 1.25 Ah × 100` | % | CONDITIONAL |
| B6 | `soh_nominal_pct_conditional` | `capacity / 1.25 Ah × 100` | % | CONDITIONAL |
| B3 | `SOH_vs_nominal_1.25Ah_pct` | `Qdis_Ah / 1.25 Ah × 100` | % | CONDITIONAL |
| B5 | `soh_nominal_pct_CONDITIONAL_1.25Ah_rated` | `capacity / 1.25 Ah × 100` | % | CONDITIONAL, not ground truth |
| B4 | — | not derived | — | BLOCKED |
| NASA | `soh_pct` | `capacity / 2.0 Ah × 100` | % | VERIFIED |
| Na-ion | `soh_pct` | `capacity / nominal_capacity_in_Ah × 100` | % | VERIFIED |

**Critical:** `soh_rated` in B2 is a fraction. `0.90` means `90.0 %`. Do not compare it directly with percentage columns without converting.

Nominal 1.25 Ah is a rated value, **not** measured beginning-of-life capacity. B1 measured BOL reference is 1.220588 Ah; B2 provides `soh_initial_ref`; B3 provides `SOH_vs_initial_2C_pct`. Do not treat nominal SOH as capacity retention against measured BOL.

---

## 4. Leakage controls (mandatory)

For any model whose target is same-cycle capacity or SOH, the following B2 columns are **forbidden as inputs**:

- `q_discharge_ah` — the target itself.
- `soh_rated` — target.
- `soh_initial_ref` — normalised by the first valid cycle; leaks trajectory-wide information.
- `soh_slope_per_100cyc` — fitted over a cycle window; contains future cycles.
- `q_fade_rate_ah_per_100cyc` — fitted fade rate; contains future cycles.
- `coulombic_efficiency` — ratio that embeds `q_discharge_ah`.

Additional rules:

- Do not broadcast a cycle-level label to telemetry rows and then use it as a feature.
- Do not create rolling windows, normalisation statistics, or target transforms that use future cycles.
- Split chronologically; do not shuffle cycles from the same cell across train/test.
- `MEMBER1_FEATURE_TARGET_MANIFEST.csv` classifies every column. Columns marked `CONDITIONAL_INPUT / REVIEW` require an ablation check that they cannot reconstruct the target.

---

## 5. B2 channel transition (Member 2 decision)

Source channel changes from `Un1Ch8` to `Un1Ch2` at cycle **999**. Cycle 999 is already invalid; cycles **1000–1015** are eligible but on the new channel.

| Treatment | Cycles | Chronological 70/15/15 | Notes |
|---|---|---|---|
| Exclusion (recommended primary) | 1,480 | 1,036 / 222 / 222 | Remove cycles 999–1015 |
| Channel covariate (sensitivity) | 1,496 | 1,047 / 224 / 225 | Keep all valid cycles, pass channel as covariate |

These are data-treatment options for Member 2 to review, not a physical correction. Report both if feasible; state which one produced the headline metrics.

---

## 6. Quality flags and exclusions

`quality_flag` is a semicolon-separated string. Common tokens:

- `TEMP_UNMEASURED` — LFP sources have no temperature channel.
- `CURRENT_UNIT_UNDERDETERMINED` — header says `I(A/cm2)` but surface area is 0 and values match protocol amperes; do not compute current-density features.
- `OFFSET_UNDERDETERMINED` — B1 cycle id uncertain (110 cycles).
- `PARTIAL_DISCHARGE` — incomplete discharge; capacity not a full-discharge ground truth.
- `CAPACITY_GROUND_TRUTH_NO` — B5, all cycles.
- `CHANNEL_TRANSITION` — B2 cycle at/after channel switch.
- `INVALID_CYCLE` — B2 10 cycles.
- `TARGET_BLOCKED_NO_AUDITED_TARGET` — B4.
- `CYCLE_ID_RECONSTRUCTED` — cycle id inferred from sheet order (B4, parts of B1/B6).

Filtering rule: a cycle is model-eligible only if `label_status` is `CONDITIONAL` or `VERIFIED`, `capacity_ah` is non-null, and the dataset-specific quality tier is acceptable.

---

## 7. What Member 2 may and may not claim

**May:**
- Train and validate SOH regression on B2 (primary), B1/B6/B3 (secondary), with conditional labels.
- Report cross-cell generalisation using cell-level holdout.
- Use NASA NMC and Na-ion for cross-chemistry sanity checks, clearly separated.

**May not:**
- Claim LFP capacity/SOH ground truth for B4.
- Claim B5 capacity as ground truth.
- Use LFP temperature as measured (it is absent).
- Compute EFC or current-density features from the `A/cm2` header without resolving units.
- Compare B2 fractional SOH to percentage SOH without conversion.
- Promote the candidate master or overwrite the canonical master.

---

## 8. Required return from Member 2

For each prediction, return:

```json
{
  "dataset_id": "LFP_MultiStat_B2",
  "cell_id": "BATTERY_2",
  "cycle_id": 1000,
  "target": "soh_pct",
  "value": 91.4,
  "unit": "%",
  "uncertainty": 1.8,
  "model_version": "member2-soh-v1",
  "training_data": "B2_1,496_conditional",
  "quality_flag": "CHANNEL_TRANSITION",
  "supported": true
}
```

Also deliver: train/validation/test split definition, feature list, leakage audit, per-dataset metrics, calibration/uncertainty method, and a list of unsupported cases. Do not modify Member 1's data artefacts.
