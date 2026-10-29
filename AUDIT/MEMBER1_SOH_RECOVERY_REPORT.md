# Member 1 — Missing & Questionable SOH Label Recovery Audit

**Project:** Autonomous Multi-Agent EV Battery Life & Range Predictor
**Module:** Member 1 — Telemetry Ingestion, Preprocessing Verification & Master-Table Integration
**Date:** 2026-10-29
**Status:** `INVESTIGATION_COMPLETE` — 1 value recovered with conditions, 1 value confirmed unrecoverable
**Canonical dataset:** **not modified** (hashes re-verified before and after this audit)

---

## 1. Scope

This audit answers one question for every SOH label in the canonical master table: *is a missing or questionable value recoverable from an authoritative source without inventing anything?*

It covers all **819 (dataset, cell, cycle) label groups** in [master_table.parquet](file:///c:/Users/KIIT0001/Desktop/MAJOR%20PROJ%202/master_table.parquet):

| Label state | Cycles | What it means |
|---|---|---|
| `VALID_MEASURED` | 461 | NASA NMC cycles without a rated-baseline effect |
| `VALID_DEFINITIONAL` | 107 | Value present; the apparent anomaly is the SOH definition (NASA `>100%`, Na-ion `<100%` at BOL) |
| `CONDITIONAL_SOURCE_CAVEAT` | 249 | B5 values present but conditional (charge-balance discrepancy); **not missing** |
| `MISSING_NOT_RECOVERABLE` | **1** | **B5 cycle 40** — partial discharge; no full capacity exists anywhere |
| `QUESTIONABLE_RECOVERED_WITH_CONDITIONS` | **1** | **B1** — label belonged to a different cycle; repaired from the audited per-cycle value |
| **Total** | **819** | |

Only **two** labels required investigation as genuinely missing or mis-assigned. Both are documented below with source, method and evidence. A complete machine-readable row exists for every cycle in [`soh_recovery_log.csv`](file:///c:/Users/KIIT0001/Desktop/MAJOR%20PROJ%202/AUDIT/member1_audit_outputs/soh_recovery_log.csv).

**Recovery rules applied:** a value was adopted only when an authoritative artifact contained a value for the *exact* cycle, with a documented calculation and a known definition. No interpolation, no forward/backward fill, no scaling, no borrowing from neighbouring cycles. Where no such value exists, the label stays missing.

---

## 2. Guardrails

| Check | Result |
|---|---|
| `master_table.parquet` SHA-256 before and after | `6fa7475d…73b2bb` — **unchanged** |
| `master_table.csv` SHA-256 before and after | `9e2379…9895b` — **unchanged** |
| Rows / columns | 515,270 × 13 — unchanged |
| Files written | only inside `AUDIT/member1_audit_outputs/` |
| Interpolation / forward-fill / fabrication | none |

The canonical table therefore still carries the *old* B1 label (1.202223 Ah) and the partial B5 throughput (0.5428382 Ah). The recovery log is an overlay for Member 2; applying it to the dataset requires a deliberate rebuild (see §8).

---

## 3. Investigation 1 — LFP B5, cycle 40 (missing SOH)

### 3.1 Question

The master has 625 telemetry rows for cycle 40 with `capacity_ah = 0.5428382` and `soh_pct = NULL`. Is a defensible full-discharge capacity — and therefore SOH — recoverable?

### 3.2 Sources consulted

| Source | Locator | What it contains |
|---|---|---|
| Raw workbook | `DATA/B5_Discharge0.5C_Charge1C (1).zip` → `B5_Discharge0.5C_Charge1C/Cycle_0050.xlsx`, sheet `Run01_Un1Ch1_Step2_Cycle40` | Raw `T(Seconds)`, `E(V)`, `I(A/cm2)`; protocol header |
| Raw workbook digest | SHA-256 `3d4d2e3237c08a9c74d99039a0197d03dda8972e65e8d552bfddae9de2516a3a` | Integrity pin for the workbook used |
| Audit summary | `AUDIT/B5.zip` → `B5_Cycle_Summary.csv`, row `global_cycle_id_nominal_checkpoint = 40` (content SHA-1 `064e53c3db59779fe1868c8807020c0a1a4837b8`) | Per-cycle status and calculated capacities |
| Exclusion log | `AUDIT/B5.zip` → `B5_Exclusion_Transformation_Log.csv` | Cycle 40 listed among flagged cycle segments |
| Audit report | `AUDIT/B5.zip` → `B5_Audit_Report.md` | Documents interrupted discharges and warns that throughput must not be chained across restarts |

### 3.3 Evidence

Protocol recorded in the workbook header: `Charge CC: 1.25 A, CV: 3.65 V … Discharge CC: 0.625 A, XV: 2.5 V`. The discharge cutoff is **2.5 V**.

| Quantity | Value | Meaning |
|---|---|---|
| Discharge rows | 3,129 | Segment length |
| Time span | 419,354 s → 422,482 s (3,128 s) | ~52 min of discharge instead of the several hours a full 0.625 A discharge needs |
| Voltage at start → end | 3.5935 V → **3.2128 V** | Ends far above the 2.5 V cutoff |
| Samples at or below 2.5 V | **0** | Cutoff never reached |
| Mean \|I\| | 0.62475 A | Consistent with the 0.625 A CC protocol |
| Partial throughput (trapezoid) | **0.5428382 Ah** | Matches the master value and the audit summary to 7 decimals |
| Partial throughput (sample-and-hold sensitivity) | 0.5430117 Ah | Method-independent within 0.03% |
| Legacy proxy | 0.5432292 Ah | Also partial |
| Neighbouring complete-cycle median (checkpoint 50) | 1.1909 Ah | Full capacity for this stage |
| Partial ÷ full | **0.456** | Less than half a cycle was delivered |
| Workbook record ends at last discharge row | yes | No continuation exists in the file |

Audit summary for the cycle: `cycle_status = DISCHARGE_INTERRUPTED`, `dis_complete = False`, `capacity_ground_truth = NO`, `capacity_status = NOT_CALCULATED_INCOMPLETE_DISCHARGE`, `modelling_use_capacity_soh = EXCLUDE`, `discharge_capacity_ah_CALCULATED_CONDITIONAL_not_ground_truth = NaN`, `soh_nominal_pct_CONDITIONAL_1.25Ah_rated = NaN`. Exactly **one** summary row exists for cycle 40, i.e. there is no repeat measurement.

### 3.4 Verdict: **NOT RECOVERABLE**

The cell was discharged for 3,128 s at ~0.625 A and the record stops at 3.2128 V. There is no measurement of the remaining ~55% of the discharge anywhere in the workspace, and the audit deliberately withholds both the conditional capacity and SOH. Any SOH derived from 0.5428 Ah would either treat partial throughput as full capacity (wrong by ~2.2×) or assume a full-capacity value — both are fabrication.

**Disposition:** SOH stays `NULL`; capacity stays flagged as partial and is **not** offered as a target (`adopted_*` fields are null in the recovery log). This matches the audit's `EXCLUDE` decision.

**What would make it recoverable:** a repeat complete 0.625 A CC discharge to 2.5 V for this cycle, or the original laboratory record of that discharge, or a documented full-discharge capacity for this exact cycle. None exists in the workspace.

*Note:* the B5 source audit excludes 7 interrupted/truncated cycles in total (40, 583, 749, 890, 1099, 1108, 1290); only cycle 40 falls inside the ingested blocks (1–100, 151–200, 401–500), so it is the only partial label in the master.

---

## 4. Investigation 2 — LFP B1 (questionable label, wrong cycle)

### 4.1 Defect

`master_table_build.py` builds `cap_map = dict(zip(cycle_checkpoint, capacity_ah))` from a table where one checkpoint (`Cycle_0050`) spans 50 cycles, then looks up by checkpoint. The lookup silently returns the **last** cycle of the checkpoint.

- Telemetry rows actually come from **global_cycle_est = 2** (verified in `LFP_staging_sample_raw_rows.csv`, checkpoint 50).
- The master labelled them with **est = 50**'s capacity: 1.202223 Ah / 96.178%.
- The audited value for the telemetry's own cycle is **1.220155 Ah / 97.612%**.

### 4.2 Sources consulted and evidence

| Source | Locator |
|---|---|
| Cycle-level audit table | `AUDIT/B1_LFP_Audit_Output.zip` → `B1_LFP_Audit_Output/LFP_staging_cycle_level.csv` (checkpoint 50, est 2): `1.220155 Ah`, `97.612%`, `quality_flag = ok`, `value_origin = calculated`, `formula_id = F1;F2;F3` |
| Feasibility table | `…/LFP_capacity_soh_feasibility.csv` (checkpoint_label 50, est 2): `cycle_offset_confidence = VERIFIED`, `discharge_segment_complete = True`, `capacity_status = COMPUTED_VERIFIED_WITH_CONDITIONS`, `soh_status = COMPUTED`, `feasibility_overall = FEASIBLE` |
| Row provenance | `…/LFP_staging_sample_raw_rows.csv`: every master telemetry row is from `global_cycle_est = 2` |

### 4.3 Verdict: **RECOVERED WITH CONDITIONS**

- **Adopted value:** `capacity_ah = 1.220155`, `soh_pct = 97.612`.
- **Source:** the B1 audit's own per-cycle table for the exact cycle the telemetry belongs to.
- **Method:** direct lookup of the audited computed value — no interpolation, no scaling, no neighbour borrowing. This is a *pairing* repair: the master's value belongs to a different cycle of the same checkpoint.
- **Why CONDITIONAL, not HIGH:** the B1 workbook header is `I(A/cm2)` with `Surface Area = 0`, so total amperes remain `UNDERDETERMINED_TOTAL_AMPERES_PLAUSIBLE`. The existing and recovered values share that caveat, so neither can be ground truth.

---

## 5. Definitional cases — investigated, no change

These look anomalous but are correct under the documented SOH definition (`capacity_ah ÷ manufacturer rated nominal × 100`, **not** retention versus measured BOL):

| Case | Cycles | Explanation | Action |
|---|---|---|---|
| NASA `B0006`, SOH 100.03–101.77% | 7 | Capacity referenced to the 2.0 Ah rating; a fresh cell can exceed its rating | Retained unchanged |
| Na-ion, SOH 78.6–91.0% | 100 | Referenced to 1.0 Ah rated nominal; measured BOL is below rating, and the trajectory is nearly flat over 25 cycles | Retained unchanged |

Neither is a missing value, so no recovery applies.

---

## 6. Absent cycles are not missing labels (B5 101–150, 201–400)

The master intentionally ingests B5 blocks 1–100, 151–200 and 401–500. The 250 cycles in 101–150 and 201–400 are **absent by ingestion design**, not missing labels. `B5_Cycle_Summary.csv` does contain computed conditional values for them, so they *could* be ingested later — but that is a coverage decision requiring a rebuild, it is outside this recovery audit, and it is **not** counted in the usable-cycle numbers below.

---

## 7. Impact on usable cycle count and Member 2 eligibility

| Tier | Before | After |
|---|---|---|
| HIGH confidence | 568 | 568 |
| CONDITIONAL | 249 | **250** |
| INVALID | 2 | **1** |
| **Usable as target (HIGH + CONDITIONAL)** | **817** | **818** |

- **B5 cycle 40:** remains `INVALID` / not usable. No change.
- **B1:** moves `INVALID → CONDITIONAL`. This is the **only** change, and it is the entire +1. No value was promoted to HIGH.
- Per dataset, usable targets are: NASA 468, Na-ion 100, B5 249, **B1 1 (new)**.

**Member 2 guidance**

- Treat the B1 cycle as **conditional / auxiliary only**, with the `I(A/cm2)` unit caveat; never as ground truth and never for thermal features (temperature is unmeasured).
- Keep B5 cycle 40 excluded from capacity/SOH training.
- Do not read the B1 recovery as a licence to interpolate other gaps; it is a disjoint-evidence repair, not an imputation.

---

## 8. Reproduce

```bash
python AUDIT/soh_recovery.py
python -m unittest AUDIT/test_soh_recovery.py -v
```

Expected: `recovered: 1`, `not recoverable: 1`, `usable before -> after: 817 -> 818`, master hash unchanged, 19 tests passing. The script aborts if the canonical hashes change during the run.

---

## 9. Files

| File | Purpose |
|---|---|
| [AUDIT/soh_recovery.py](file:///c:/Users/KIIT0001/Desktop/MAJOR%20PROJ%202/AUDIT/soh_recovery.py) | Reproducible read-only recovery audit |
| [AUDIT/test_soh_recovery.py](file:///c:/Users/KIIT0001/Desktop/MAJOR%20PROJ%202/AUDIT/test_soh_recovery.py) | 19 tests pinning every decision above |
| [soh_recovery_log.csv](file:///c:/Users/KIIT0001/Desktop/MAJOR%20PROJ%202/AUDIT/member1_audit_outputs/soh_recovery_log.csv) | All 819 cycles: state, adopted values, usable before/after, source, method, evidence |
| [soh_recovery_report.json](file:///c:/Users/KIIT0001/Desktop/MAJOR%20PROJ%202/AUDIT/member1_audit_outputs/soh_recovery_report.json) | Machine-readable summary, evidence and impact |
| This report | Narrative record of both investigations |

No canonical data, raw archive or pre-existing user file was modified.
