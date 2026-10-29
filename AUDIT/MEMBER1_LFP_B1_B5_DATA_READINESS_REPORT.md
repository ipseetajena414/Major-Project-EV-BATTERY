# Member 1 — LFP B1–B5 Data Investigation & Validation Readiness

**Project:** Autonomous Multi-Agent EV Battery Life & Range Predictor
**Author:** Member 1 (data ingestion, validation, provenance)
**Date:** 2026-10-29
**Status:** Read-only investigation completed. No canonical dataset was modified, no model code was touched, nothing was staged or committed.

Related artifacts:

| Artifact | Purpose |
|---|---|
| `AUDIT/MEMBER1_LFP_B1_B5_COVERAGE_MATRIX.csv` | One row per LFP cell: source status, coverage, suitability, blocker, action |
| `AUDIT/MEMBER1_LFP_FEATURE_READINESS_MATRIX.csv` | 14 feature families × 5 cells = 70 evidence rows |
| `AUDIT/lfp_b1_b5_inventory.py` | Read-only utility that regenerates the matrix content and evidence JSON |
| `AUDIT/test_lfp_b1_b5_inventory.py` | 32 tests over archives, samples, master facts and deliverables |
| `AUDIT/member1_audit_outputs/lfp_b1_b5_evidence.json` | Machine-readable evidence (hashes, samples, facts) |
| `AUDIT/MEMBER1_SOH_RECOVERY_REPORT.md` | Prior SOH recovery investigation (B5 cycle 40, B1 label) |
| `AUDIT/MEMBER1_DATA_READINESS_HANDOFF.md` | Prior master-table handoff (preserved, unchanged) |

---

## 0. Phase 0 — Repository protection record

Recorded before any new work, on the existing checkout (no branch switch, no merge):

```
git branch --show-current
main

git log -5 --oneline
e36eb73 Add Member 1 ingestion pipeline, master table parquet, audit validation reports and project blueprint
18fb971 first commit

git status --short
?? AUDIT/MEMBER1_DATA_READINESS_HANDOFF.md
?? AUDIT/MEMBER1_FEATURE_READINESS_MATRIX.csv
?? AUDIT/MEMBER1_FORMULA_VALIDATION_MATRIX.csv
?? AUDIT/MEMBER1_LFP_B1_B5_COVERAGE_MATRIX.csv
?? AUDIT/MEMBER1_LFP_FEATURE_READINESS_MATRIX.csv
?? AUDIT/MEMBER1_SOH_RECOVERY_REPORT.md
?? AUDIT/lfp_b1_b5_inventory.py
?? AUDIT/make_matrices.py
?? AUDIT/member1_audit.py
?? AUDIT/member1_audit_outputs/
?? AUDIT/soh_recovery.py
?? AUDIT/test_lfp_b1_b5_inventory.py
?? AUDIT/test_member1_audit.py
?? AUDIT/test_soh_recovery.py

git diff --stat
(empty)

git diff --cached --stat
(empty)
```

All pre-existing Member 1 work is untracked in the working tree and was preserved. Nothing was reset, cleaned, stashed or discarded. `master_table.parquet`, `master_table.csv`, `DATA/*` and `AUDIT/*.zip` were only read. All new writes are confined to `AUDIT/` (matrices, report, scripts) and `AUDIT/member1_audit_outputs/`.

---

## 1. Scope and method

The attached archives (`B1_LFP_Audit_Output.zip`, `B3_Discharge2C_Charge0.5C.zip`, `B4_Discharge3C_Charge0.5C.zip`, `B5_Discharge0.5C_Charge1C.zip`, `LFP_B6_Audit_Package.zip`) were hashed and compared with the workspace copies. **All five are byte-identical to files already in the repository**; no new raw data was introduced by this request. B1's raw archive and B2's derived export are not in the attachment set but are present in `DATA/`.

Every number in this report was produced by one of:

1. Recounting worksheets directly from the raw `.xlsx` files inside the archives (`xl/workbook.xml` `<sheet>` entries).
2. Re-integrating one complete cycle per raw cell from the workbook itself (trapezoid of |I| dt) and comparing with the audit tables.
3. Read-only queries on `master_table.parquet`.
4. Reading the audit bundles supplied by the campaign owner / previous audits.

No label, cycle or temperature value was invented. Nothing derived here has been merged into the canonical master table.

---

## 2. Verified source inventory

| Cell | Raw source | Size (bytes) | SHA-256 (first 16) | Audit / derived source | Size (bytes) |
|---|---|---|---|---|---|
| B1 | `DATA/B1_Discharge0.5C_Charge0.5C(1).zip` | 482,913,032 | `fe04aec9b706aa82` | `AUDIT/B1_LFP_Audit_Output.zip` | 412,476 |
| B2 | **absent** (no XLSX) | — | — | `DATA/B2_LFP_Member1_audit.zip` (derived) | 9,520,288 |
| B3 | `DATA/B3_Discharge2C_Charge0.5C (1).zip` | 392,547,570 | `aff29ebec7befc8a` | `AUDIT/Existing_Audits_Available_B3_B4 (1).zip` | 1,139,252 |
| B4 | `DATA/B4_Discharge3C_Charge0.5C (1).zip` | 290,588,332 | `a17779c495f2fdd6` | same B3/B4 bundle (partial) | — |
| B5 | `DATA/B5_Discharge0.5C_Charge1C (1).zip` | 355,306,217 | `f3d6b1ef56992f57` | `AUDIT/B5.zip` | 112,296,675 |
| B6 | **absent** (raw) | — | — | `AUDIT/LFP_B6_Audit_Package (1).zip` (out of scope) | 494,690 |

Canonical master (unchanged):

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `master_table.parquet` | 9,247,868 | `6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb` |
| `master_table.csv` | 97,661,860 | `9e23792615b2f08692944680d041957f776428b7563a90b214b77ee74629895b` |

### Raw workbook / cycle-sheet inventory (recounted, not inferred)

| Cell | Workbooks | Cycle worksheets (`*Step2_Cycle*`) | Notes |
|---|---|---|---|
| B1 | 19 | 1,460 | checkpoints 50–1500 |
| B3 | 25 | 1,979 | checkpoints 50–2000 |
| B4 | 21 | 1,577 | checkpoints 50–1600 |
| B5 | 17 | 1,491 | checkpoints 50–1500, no 300 block |

### Provenance / licensing

The repository documents these as "LFP MultiStat 18650" campaign exports (Biologic EC-Lab text exports embedded in Excel). **No dataset DOI, landing page, licence or campaign metadata file exists in this workspace.** A targeted web search did not conclusively identify the originating publication or repository; no external source is claimed here. See §9.

---

## 3. Coverage summary (canonical master table)

| dataset_id | Rows | Cells | Observed cycle IDs | Labelled (cell, cycle) pairs | In master? |
|---|---|---|---|---|---|
| `LFP_MultiStat_B1` | 4,789 | 1 | 1 (cycle_id = 2) | 1 | Partial (1 of 1,460) |
| `LFP_MultiStat_B2` | 0 | 0 | 0 | 0 | No |
| `LFP_MultiStat_B3` | 0 | 0 | 0 | 0 | No |
| `LFP_MultiStat_B4` | 0 | 0 | 0 | 0 | No |
| `LFP_MultiStat_B5` | 333,344 | 1 | 250 | 249 | Partial (250 of 1,499 audit records) |

The brief's initial facts are **confirmed exactly**: B1 = 4,789 rows / 1 observed cycle; B2–B4 = no master coverage; B5 = 333,344 rows / 250 observed cycles.

---

## 4. Per-cell findings

### 4.1 B1 — `LFP_MultiStat_B1` (0.5C discharge / 0.5C charge)

**Source status:** `SOURCE_VERIFIED_MASTER_PARTIAL`.

- Raw archive verified: 19 workbooks, 1,460 cycle worksheets, 482.9 MB.
- Audit table (`LFP_staging_cycle_level.csv`): 1,460 rows, 19 checkpoints, **1,456 capacity/SOH labels**, 4 cycles without a label (3 `partial_discharge`, 1 `polarity_anomaly;partial_discharge`).
- Quality flags: 1,339 `ok`; 108 `ok;underdetermined_cycle_offset`; plus 2 more underdetermined combined with other flags (**110 total**); remaining flags include `cross_sheet_time_reset`, `ce_gt_120`, `out_of_protocol_voltage`, `current_excursion`.
- **Independent re-derivation:** `Cycle_1500.xlsx` sheet `Run01_Un1Ch7_Step2_Cycle001` → 13,993 rows, 6,366 discharge samples, **Q = 1.102908 Ah**, V_end = 2.4944 V, I ≈ 0.624 A. Audit at checkpoint 1500 / global cycle 1401: **1.102908 Ah, SOH 88.233 %** — matches to <1e-6 Ah.
- Master coverage is an **ingestion omission plus a label-mapping bug**, not a source limitation:
  - The builder reads `LFP_staging_sample_raw_rows.csv` only (checkpoint 50, telemetry `global_cycle_est = 2`) and samples every 3rd row → 4,789 rows.
  - It builds `dict(cycle_checkpoint → capacity_ah)` from a table where each checkpoint spans up to 100 cycles, so the lookup collapses to the **last cycle of the checkpoint** (cycle 50, 1.202223 Ah = 96.178 %) while the telemetry belongs to cycle 2 (audit value **1.220155 Ah = 97.612 %**). The current master label is therefore the wrong cycle's label.
- Temperature: absent (`Aux Count: 0`; no thermal channel).
- BOL reference is the first observed cycle, not a measured beginning-of-life capacity; nominal 1.25 Ah is the manufacturer rating.
- **Suitability:** `POTENTIALLY_SUITABLE_AFTER_VERIFICATION` — 1,456 labels exist but the master contains 1, and 110 labels carry an underdetermined cycle offset.

### 4.2 B2 — `LFP_MultiStat_B2` (1C discharge / 0.5C charge)

**Source status:** `ONLY_DERIVED_DATA_AVAILABLE`.

- Raw workbooks **absent**; the 20 referenced XLSX files (381,209,935 bytes total) are listed with per-file hashes in the derived export but are not in this workspace.
- Derived export verified internally: `cycle_features.parquet` **1,506 × 255**, of which **1,496 in analysis** (10 invalid); `qv_curves.parquet` 1,506 × 3 (101-point curves); contiguous chronological `splits.json` (train 1,047 / val 224 / test 225).
- SOH = `q_discharge_ah / 1.250 Ah`; **`soh_rated` is a 0–1 fraction** (range 0.8554–0.9668) and must be ×100 for percent.
- Instrument channel changes **Un1Ch8 → Un1Ch2 at cycle 999** (998 vs 508 cycles) with a ≈0.9 % capacity step; provenance is 1,346 `file_list_indexed`, 91 `inferred_from_sheet_names`, 69 `continuation_session` (**160 non-direct**).
- Current units are explicitly amperes (BioLogic `I / A`).
- Temperature: not measured.
- Verification report in the export: 14 checks, all passed.
- **Suitability:** `SUITABLE_FOR_INDEPENDENT_VALIDATION` at **cycle level only** — no raw voltage/current/time telemetry is available for bespoke feature extraction.

### 4.3 B3 — `LFP_MultiStat_B3` (2C discharge / 0.5C charge)

**Source status:** `SOURCE_VERIFIED_NOT_INGESTED`.

- Raw archive verified: 25 workbooks, **1,979 cycle worksheets**, 392.5 MB.
- Audit table: 1,979 rows; **1,973 Qdis values, 1,971 SOH labels**; tiers A = 1,968, B = 3, X = 8. Tier X = 6 `no_discharge_in_sheet` + 2 `discharge_interrupted_before_2.5V` (0.2506 Ah and 0.2381 Ah, excluded from SOH).
- Cycle IDs 1–2000 with **201–221 absent** (21 IDs; the `Cycle_0250.xlsx` workbook holds only 29 cycle sheets). The gap is in the source, not an ingestion artifact.
- Four instrument channels (`Un1Ch2` 879, `Un1Ch8` 500, `Un1Ch1` 500, `Un1Ch6` 100) — physical cell identity across channels is not documented.
- Coulombic efficiency median **0.9828** (physically plausible, unlike B5's 1.0278); 396 cycles carry timestamp-gap flags.
- **Independent re-derivation:** `Cycle_0050.xlsx` sheet `Run01_Un1Ch2_Step2_Cycle10` → 8,773 rows, discharge 1,680 samples, **Q = 1.165491 Ah**, V_end = 2.4968 V, I ≈ 2.499 A. Audit: **1.165491 Ah, SOH 93.239 %, tier A** — exact match.
- Temperature: absent (`Aux Count: 0`).
- **Suitability:** `POTENTIALLY_SUITABLE_AFTER_VERIFICATION` — best candidate for a new independent cell, pending extraction, unit attestation and cell-identity review.

### 4.4 B4 — `LFP_MultiStat_B4` (3C discharge / 0.5C charge)

**Source status:** `SOURCE_VERIFIED_AUDIT_INCOMPLETE`.

- Raw archive verified: 21 workbooks, **1,577 cycle worksheets**, 290.6 MB.
- The supplied B4 audit covers only **16 workbooks / 1,350 cycle sheets**; `Cycle_0150, 0200, 0250, 0300, 0350` (5 workbooks) are unaudited.
- Every audited row has `metrics_status = NOT_CALCULATED: current label conflicts with protocol setpoints; phase segmentation not validated`; the `discharge_capacity_ah` and `soh_nominal_pct` columns exist but are **100 % empty**. This is a blanket block, not a per-cycle measurement failure.
- Within the audited range: 1,347 of 1,350 sheets reach ≤2.55 V; 3 have no discharge; 1 cycle (`Cycle_1000`, cycle 86) reaches 3.887 V during charge.
- Raw current confirms the protocol despite the header: negative median **−3.7495 A** (3C), positive median **+0.6259 A** (0.5C).
- **Independent re-derivation:** `Cycle_0050.xlsx` sheet `Run01_Un1Ch6_Step2_Cycle10` → 8,358 rows, discharge 1,140 samples, **Q = 1.186418 Ah**, V_end = 2.4976 V, I ≈ 3.750 A. The discharge is complete and physically plausible; the audit's blanket block is stricter than the data.
- Temperature: absent (`Aux Count: 0`).
- **Suitability:** `POTENTIALLY_SUITABLE_AFTER_VERIFICATION` — needs a full 21-workbook re-audit with sign-based phase segmentation and unit attestation before labels exist.

### 4.5 B5 — `LFP_MultiStat_B5` (0.5C discharge / 1C charge)

**Source status:** `SOURCE_VERIFIED_MASTER_PARTIAL_CONDITIONAL`.

- Raw archive verified: 17 workbooks, **1,491 `Step2_Cycle` worksheets**, 355.3 MB.
- Source audit: **1,499 records** (1,491 `CYCLE` + 8 `CYCLE_RENAMED_SHEET`), **1,492 SOH labels**, 7 excluded cycles; `capacity_ground_truth = NO` for **every** cycle.
- Charge balance: Qdis > Qchg in **1,488 / 1,498** paired cycles; CE median **1.0278** (range 0.99957–1.04546). An excess discharge capacity of ~2.8 % is not physically explainable from the recorded channels.
- Master coverage: 333,344 rows, 250 observed cycle IDs in blocks **1–100, 151–200, 401–500**, **249 labelled**; the master ingests only 4 of 17 workbooks (checkpoints 50/100/200/500). Cycle 40 is present with a null SOH on all 625 rows; 43 discharge rows dip below 2.5 V (min 2.4947 V).
- Definition used: `capacity_ah / 1.25 Ah × 100`, conditional, computed from the raw trapezoidal column (this is why cycle 40 keeps a partial capacity while its SOH is null).
- Temperature: absent (`Aux Count: 0`).
- **Suitability:** `BLOCKED_BY_LABEL_QUALITY` — absolute capacity is not ground truth, charge balance is inconsistent, and B5 is already the cell Member 2 trained on, so it cannot serve as independent validation.

### 4.6 B6 — `LFP_MultiStat_B6` (0.5C discharge / 2C charge) — out of scope

The attached `LFP_B6_Audit_Package` was recorded only because it was supplied. Its metadata references `B6_Discharge0.5C_Charge2C.zip` (368,836,614 bytes), **19 workbooks, 1,702 cycle-like worksheets, 1,697 complete-discharge labels**; the raw archive is **absent** from the workspace and from Downloads. It cannot be used as verified data. If the raw archive were supplied, B6 would be a natural fourth independent LFP cell.

---

## 5. B5 missing label and cycle-gap analysis

### 5.1 Cycle 40 (rechecked; conclusions unchanged from the prior recovery report)

| Item | Value |
|---|---|
| Rows in master | 625 |
| Capacity label | 0.5428382 Ah (partial discharge throughput) |
| SOH label | NULL (all rows) |
| Raw run | `Cycle_0050.xlsx` sheet `Run01_Un1Ch1_Step2_Cycle40`, 3,129 rows |
| Voltage span | 3.5935 V → 3.2128 V (never reaches 2.5 V) |
| Discharge throughput | 0.5428382 Ah = **45.6 %** of the neighbouring-cycle median (1.1909 Ah) |
| Source status | `DISCHARGE_INTERRUPTED`, audit `EXCLUDE` |
| Recoverable? | **No** — no measured full-discharge capacity exists for this cycle |

The audit and the raw workbook agree. There is no authoritative capacity to recover; the value stays missing and excluded. The trajectory is **not** interpolated.

### 5.2 Cycle gaps

- **Absent cycle IDs:** 101–150 and 201–400 (250 IDs) are not present in the master at all. They exist in the source (the master only ingested 4 of 17 workbooks). They are *not* observed cycles with missing labels.
- **Observed cycle with a missing label:** only cycle 40 (250 observed, 249 labelled).
- **Pair semantics:** the labelled sequence has 249 cycles and therefore **248 next-observed-record pairs**. Those pairs are not 248 physical cycles: jumps such as 100 → 151 and 200 → 401 cross 50- and 200-cycle physical gaps.
- **Consequence for modeling:** any lag/ΔSOH feature must be computed on observed records and must carry the elapsed cycle gap (or be restricted to contiguous blocks). Nothing is fabricated to fill gaps.

### 5.3 Charge-balance documentation

The B5 audit names the charge capacity `chg_q_ah_UNSUPPORTED_diagnostic_only` and sets `capacity_ground_truth = NO`. No document in the workspace explains the physical origin of Qdis > Qchg. Therefore the labels remain conditional: usable as a consistent proxy for within-cell relative ageing, not as absolute ground-truth capacity. No smoothing or correction was applied.

---

## 6. Feature readiness (summary of the 70-row matrix)

| Feature family | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|
| Discharge capacity label | CONDITIONAL (unit; master label bug) | READY (cycle table) | CONDITIONAL (unit) | CONDITIONAL (not computed) | CONDITIONAL (not ground truth) |
| SOH % label | CONDITIONAL | READY (×100) | CONDITIONAL | CONDITIONAL (not computed) | CONDITIONAL (cycle 40 missing) |
| Voltage series | READY_WITH_CAVEAT | NOT_AVAILABLE | READY | READY_WITH_CAVEAT | READY_WITH_CAVEAT |
| dQ/dV, dV/dQ, ICA peaks | CONDITIONAL | CONDITIONAL (101-pt) | CONDITIONAL | CONDITIONAL | CONDITIONAL |
| Ah / energy throughput, C-rate, CC-CV fraction | CONDITIONAL | READY (cycle level) | CONDITIONAL | CONDITIONAL | CONDITIONAL |
| Temperature | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| Resistance proxy | CONDITIONAL | READY (provided) | CONDITIONAL | CONDITIONAL | CONDITIONAL |
| Cycle index / EFC | READY_WITH_CAVEAT | READY | READY_WITH_CAVEAT | CONDITIONAL | READY_WITH_CAVEAT |
| Lagged SOH | BLOCKED (1 master cycle) | READY_WITH_CAVEAT | READY_WITH_CAVEAT | BLOCKED | READY_WITH_CAVEAT |

Key points:

- **Current units.** All LFP workbooks label current `I(A/cm2)` while `Surface Area = 0`; the numeric values match the protocol setpoints in amperes (0.625 / 1.25 / 2.5 / 3.75 A). The header is a label error, but no original metadata attests it. Unit-dependent features (Ah throughput, C-rate, energy) stay **CONDITIONAL** until attestation.
- **Time.** 1 Hz source. Master is decimated (B1 every 3rd row, B5 every 5th discharge row); incremental-capacity features therefore require the raw/clean files, not the master.
- **Phase.** B1 master mixes charge and discharge rows; B5 master is discharge-only; B3/B4 raw have no explicit phase column and need sign-based segmentation. B4's audit used "unknown subphase".
- **Temperature.** No workbook has a temperature channel (`Aux Count: 0`). No measured temperature exists for B1–B6; scenario temperatures must never be presented as measurements.
- **Leakage.** Capacity/SOH are cycle-level scalars broadcast across telemetry rows. In B2 the columns `q_rated_ah`, `q_discharge_ah`, `q_charge_ah`, `soh_rated`, `soh_initial_ref`, `soh_slope_per_100cyc` are label-derived and must never be model inputs.
- **BOL.** No cell has a measured beginning-of-life capacity; nominal ratings are manufacturer values (1.25 Ah for LFP). SOH is therefore capacity relative to the nominal rating (B1/B3/B5) or to the first observed cycle (B2/B3 secondary reference).

---

## 7. Independent-validation suitability

| Cell | Status | Evidence |
|---|---|---|
| B1 | `POTENTIALLY_SUITABLE_AFTER_VERIFICATION` | 1,456 labels available in source; 1 in master; 110 offset-uncertain; unit attestation pending |
| B2 | `SUITABLE_FOR_INDEPENDENT_VALIDATION` | 1,496 analysis cycles, audited exports, 14/14 checks; cycle-level only; channel change at 999; raw absent |
| B3 | `POTENTIALLY_SUITABLE_AFTER_VERIFICATION` | 1,971 labels, exact re-derivation, plausible CE (0.983); not ingested; 4 channels; gap 201–221 |
| B4 | `POTENTIALLY_SUITABLE_AFTER_VERIFICATION` | Complete raw discharges; labels not computed; 5 workbooks unaudited; unit/phase attestation pending |
| B5 | `BLOCKED_BY_LABEL_QUALITY` | Capacity not ground truth; CE 1.0278; already the training cell |

**Recommendation.**

1. **Immediate (no new extraction):** use **B2** for cycle-level independent validation of Member 2's Ridge SOH predictor (SOH ×100; exclude/model cycles 999–1015; never feed label-derived columns). This tests generalization to a different LFP cell today.
2. **Priority new independent cell:** extract **B3** (1,971 labels, raw 1 Hz data, 2C protocol). It is stronger than B1 for validation because it has more labels, a plausible charge balance, and raw telemetry for feature parity with B5.
3. **Secondary:** extract **B1** (0.5C/0.5C) once the 110 offset-uncertain labels are resolved, and **B4** after a full re-audit. Either would add a protocol-diverse validation cell.
4. **B5** stays the conditional training/evaluation cell. It cannot validate itself.

---

## 8. Prioritized action plan

### P0 — Essential verification

| # | Task | Owner | Deliverable | Acceptance criteria | Current blocker |
|---|---|---|---|---|---|
| P0-1 | Current-unit attestation for LFP `I(A/cm2)` header | Member 1 + campaign owner | Signed provenance note mapping header to amperes using protocol setpoints | Same rule applied to B1–B5; documented in `AUDIT/`; no numeric change to master | No original metadata/documentation in workspace |
| P0-2 | B5 label closure | Member 1 | Re-confirm 250 observed / 249 labelled / cycle 40 excluded; 1,488/1,498 imbalance recorded | Reproducible via `lfp_b1_b5_inventory.py`; no fabricated values | None (complete) |
| P0-3 | Independent validation run on B2 | Member 2 (review), Member 1 (data slice) | Cycle-level validation report of the Ridge SOH model on B2 | Uses B2 SOH×100, contiguous splits, no label-derived inputs, excludes/model channel change | Member 2 sign-off |
| P0-4 | B3 candidate extraction | Member 1 | Separate candidate dataset (not in `master_table.parquet`) with 1,979 cycles / 1,971 labels | ≥3 sampled cycles reproduce audit to 1e-6 Ah; gaps preserved; contiguous indexing | Unit attestation; cell-identity review |
| P0-5 | B1 candidate re-ingestion | Member 1 | Candidate dataset covering 1,460 cycles / 1,456 labels | Per-cycle capacity reproduces audit; label–cycle mapping fixed; no overwrite | 110 offset-uncertain labels; unit attestation |
| P0-6 | Temperature declaration | Member 1 | Statement in readiness report and schema notes | No temperature feature or imputation in any candidate dataset | None (complete) |

### P1 — Improve available coverage

| # | Task | Owner | Deliverable | Acceptance criteria | Current blocker |
|---|---|---|---|---|---|
| P1-1 | B4 full re-audit (21 workbooks) | Member 1 | Capacity/SOH table for 1,577 cycle sheets | Sign-based segmentation; 3 no-discharge cycles excluded; 3.887 V excursion flagged; ≥3 sampled cycles match raw | Unit attestation |
| P1-2 | Resolve B1 underdetermined offsets (110 labels) | Member 1 | Offset mapping per checkpoint | Each label re-anchored to a documented cycle or explicitly excluded | Workbook global counters; no external log |
| P1-3 | B2 channel-change assessment | Member 1 + Member 2 | Analysis of 999–1015 step | Effect quantified; exclusion or covariate decision documented | None |
| P1-4 | B5 imbalance documentation | Member 1 | Short note anchored to audit fields | No claim beyond `charge balance not ground truth`; no smoothing | Source documentation absent |
| P1-5 | Candidate dataset acceptance review | Member 2 + team | Review of B1/B3/B4 candidates | Label definitions, splits and gaps approved before any promotion | Depends on P0-4/P0-5/P1-1 |

### P2 — Future acquisition

| # | Task | Owner | Deliverable | Acceptance criteria | Current blocker |
|---|---|---|---|---|---|
| P2-1 | Obtain raw B2 workbooks (20 XLSX, 381 MB) | Team / campaign owner | Raw archive in `DATA/` | Hashes match the derived export's manifest | Not in workspace; no download attempt (large archive, no need yet) |
| P2-2 | Obtain B6 raw archive (`B6_Discharge0.5C_Charge2C.zip`, ~369 MB) | Team | Raw archive + audit reconciliation | 1,697 conditional labels reproducible from raw | Raw absent; out of scope |
| P2-3 | Obtain dataset documentation (DOI, licence, nominal-capacity attestation, BOL measurements) | Team | Provenance record | Source/author/licence and measurement definitions documented | Not in repo; web search inconclusive |
| P2-4 | Measured BOL capacity per cell | Campaign owner | BOL reference values | Independent measurement, not first-observed cycle | Not measured/available |

No archive was downloaded and no dependency was changed.

---

## 9. Remaining unknowns and source-acquisition needs

1. **Dataset identity/licence:** the workspace documents "LFP MultiStat 18650" but no DOI, landing page, author, or licence. External verification was attempted and was inconclusive. This is recorded rather than guessed.
2. **Current-unit attestation:** numeric values are amperes; the header says `A/cm2` with zero area. Attestation requires the original campaign metadata or the export software settings.
3. **Cell identity for B3/B4:** multiple EC-Lab channels (`Un1Ch1/2/6/8`) are pooled into one "cell". Whether all channels are the same physical cell, and how channel switches map to cycle order, is undocumented.
4. **B2 raw telemetry:** unavailable. Cycle-level validation is possible; voltage/current/time feature parity with B5 is not.
5. **B6:** only an external audit package; no raw data.
6. **BOL:** no measured beginning-of-life capacity for any LFP cell; the 1.25 Ah rating is nominal/manufacturer.
7. **Temperature:** not measured anywhere in B1–B6; no acquisition path is known.

---

## 10. Final decisions

1. **Which of B1–B5 are suitable for modeling or independent validation?**
   - **B2** — `SUITABLE_FOR_INDEPENDENT_VALIDATION` at cycle level today.
   - **B1, B3, B4** — `POTENTIALLY_SUITABLE_AFTER_VERIFICATION`; B3 is the priority extraction target.
   - **B5** — `BLOCKED_BY_LABEL_QUALITY` as an independent cell; remains a conditional training/evaluation cell.

2. **Is B1's single-cycle coverage an ingestion problem or genuinely limited source coverage?**
   Ingestion. The source has 19 workbooks / 1,460 cycle sheets / 1,456 labels. The master reads one staging sample and additionally mislabels the telemetry with the checkpoint's last cycle instead of the telemetry's own cycle.

3. **Can B2, B3 or B4 be verified and safely incorporated?**
   - B2: already verified at cycle level from audited derived exports; safely usable now with the stated caveats (SOH ×100, channel change, label columns excluded as inputs).
   - B3: source verified locally; requires candidate extraction and unit/identity attestation, then review.
   - B4: source verified locally, but no labels exist; requires a full 21-workbook re-audit first.

4. **Can B5 cycle 40's missing SOH be authoritatively recovered?**
   No. The raw discharge is interrupted at 3.2128 V with 0.5428 Ah (45.6 % of neighbouring cycles). No authoritative full-discharge capacity exists. Keep the value missing and the cycle excluded; do not interpolate.

5. **What data is required to validate Member 2's LFP Ridge model on an independent cell?**
   One independent LFP cell with per-cycle measured discharge capacity, a documented nominal capacity and current units, chronological cycle ordering, and either contiguous coverage or explicit gap handling. Today that is **B2 at cycle level**; after extraction, **B3** (preferred) or **B1/B4**. Temperature is not required because none exists, but it cannot be substituted with scenario values.

6. **What can Member 1 complete now, and what requires Member 2 or the team?**
   - **Member 1 now:** unit-attestation note, B2 validation slice, B3/B1 candidate extraction (separate from canonical data), B4 re-audit, offset/gap analysis, matrices and tests.
   - **Member 2:** review candidate label definitions, run the independent-cell validation, decide channel-covariate handling.
   - **Team / campaign owner:** raw B2 workbooks, B6 raw archive, original dataset documentation (DOI/licence), unit attestation, measured BOL capacities.

---

## 11. Quality gates and deliverables

| Gate | Result |
|---|---|
| `python AUDIT/lfp_b1_b5_inventory.py` | Passes; writes 2 matrices + evidence JSON |
| `python -m unittest AUDIT/test_lfp_b1_b5_inventory.py` | 32 tests, all pass |
| `python -m unittest AUDIT/test_member1_audit.py` | Existing canonical-master audit tests, all pass |
| `python -m unittest AUDIT/test_soh_recovery.py` | 19 existing recovery tests, all pass |
| CSV parse check | Both matrices load with the exact 20 / 10 required columns (5 / 70 rows) |
| `master_table.parquet` / `master_table.csv` | SHA-256 unchanged before/after (`6fa7475d…b2bb` / `9e237926…895b`) |
| Raw archives / audit bundles | Unchanged; hashes recorded |
| Member 2 artifacts | Not touched (no model, scaler, schema, manifest or inference file modified) |
| Git | Nothing staged, committed or pushed; working tree contains only uncommitted Member 1 files |

### Created / modified files (this phase)

Created:

- `AUDIT/MEMBER1_LFP_B1_B5_DATA_READINESS_REPORT.md` (this file)
- `AUDIT/MEMBER1_LFP_B1_B5_COVERAGE_MATRIX.csv`
- `AUDIT/MEMBER1_LFP_FEATURE_READINESS_MATRIX.csv`
- `AUDIT/lfp_b1_b5_inventory.py`
- `AUDIT/test_lfp_b1_b5_inventory.py`
- `AUDIT/member1_audit_outputs/lfp_b1_b5_evidence.json`

Modified: none of the previously existing files.

### Independently verified vs assumed vs unknown

**Verified by direct inspection in this session:** archive byte-identity and hashes; workbook/cycle-sheet counts for B1/B3/B4/B5; one full cycle re-integration per raw cell matching the audit to ≤1e-6 Ah (B1, B3) and a plausible complete 3C discharge for B4; master coverage counts and B5 blocks/gaps/cycle 40; B5 audit counts, exclusions, and charge-balance ratio; B2 derived counts, splits, channels, label scale and provenance mix; B4 audit coverage gap and blanket `NOT_CALCULATED`; B6 absences and reference counts.

**Assumed (stated, not proven):** that the EC-Lab numeric current column is in amperes (supported by protocol setpoints but not attested); that pooled channels represent one physical cell; that the audit tables were computed correctly for cycles not sampled here.

**Unknown:** dataset DOI/licence/authorship; measured BOL capacities; origin of the B5 charge-balance discrepancy; availability of B2/B6 raw data outside this workspace; whether external archives would hash-match the referenced manifests.

---

## Closure addendum — 2026-10-29

This report is a dated B1–B5 snapshot. Two statements in it are superseded by the later B2/B6 follow-up and closure pass:

* **B2 raw is now available** at `DATA/B2_Discharge1C_Charge0.5C.zip` (SHA-256 `b75c36d9…c10279`), not "NOT_IN_WORKSPACE".
* **B6 raw is now available** at `DATA/B6_Discharge0.5C_Charge2C.zip` (SHA-256 `25fe5a24…84b97a`) and has been ingested: 1,702 cycles / 170,101 telemetry rows, 1,697 conditional labels.
* The B1 `Cycle_1300` investigation is resolved as **retained gap**: 20 unlabelled ASCII exports cover cycles 1201–1220 but cannot be mapped to global IDs with certainty. See `AUDIT/MEMBER1_B1_CYCLE1300_GAP_EVIDENCE.json`.

The historical matrices and evidence JSON are left unchanged; current coverage lives in `AUDIT/MEMBER1_DATASET_COVERAGE.csv` and `MEMBER1_STATUS.json`.
