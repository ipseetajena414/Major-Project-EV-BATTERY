# Member 1 — B2 and B6 Raw Archive Follow-Up

**Project:** Autonomous Multi-Agent EV Battery Life & Range Predictor
**Role:** Member 1 — ingestion, data-quality validation, provenance, cleaned-data contract
**Date:** 2026-10-29
**Scope:** the two newly supplied archives `B2_Discharge1C_Charge0.5C.zip` and `B6_Discharge0.5C_Charge2C.zip`
**Status:** read-only follow-up complete. Canonical `master_table.parquet` / `master_table.csv` unchanged. Nothing staged, committed or pushed. No model trained.

---

## 1. Final decision (answer first)

**Yes — the new raw data materially improves the options for independent LFP validation, but it does not make either cell usable without further checks.**

Before this follow-up the workspace had:

* **B2** — derived tables only; raw workbooks absent; `source_status = ONLY_DERIVED_DATA_AVAILABLE`.
* **B6** — an audit package only, explicitly out of scope; no raw data and no global cycle id.

After this follow-up:

* **B2** is now **fully raw-verified**. All 20 workbooks in the supplied archive are byte-identical to the 20 workbooks referenced by the derived export, and two cycle capacities re-derived from raw match the export to <1e-8 Ah. B2 has a contiguous global cycle index 1…1506, 1,496 in-analysis cycles, two label definitions (nominal SOH and retention vs first valid cycle), and a verified ampere current convention. It becomes the **strongest independent validation cell in the project**.
* **B6** is now **raw-verified and chronologically reconstructable**. All 19 workbooks match the audit manifest, 1,697 of 1,702 cycle-like sheets carry conditional capacity/SOH labels, and the audit's deliberate omission of `global_cycle_id` is resolved by a **candidate** chronology built from the instrument's own Date/Time fields. That chronology validates perfectly (0 duplicates, 0 reversals, contiguous checkpoint blocks, 0 header/File-List disagreements). B6 is a credible **second** independent cell *after* current-unit attestation and an independent review of the reconstructed order.

The practically important change is the ranking: the project moves from roughly one conditionally usable independent cell (B3/B1) to **two strictly better candidates (B2 first, B6 second)**. B5 remains excluded because it is the training cell.

Nothing was filled, imputed or overwritten. The 10 flagged B2 cycles and the 5 screened-out B6 cycles remain unlabelled.

---

## 2. Repository state and safety (Phase 0)

Commands run before and after the work:

| Check | Result |
|---|---|
| `git branch --show-current` | `main` |
| `git diff --stat` | empty (no tracked modifications) |
| `git diff --cached --stat` | empty (nothing staged) |
| `git status --short` | only pre-existing untracked `AUDIT/*` artifacts plus the new files listed in §11 |
| `master_table.parquet` SHA-256 | `6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb` (unchanged) |
| `master_table.csv` SHA-256 | `9e23792615b2f08692944680d041957f776428b7563a90b214b77ee74629895b` (unchanged) |

Both supplied archives remain **in `Downloads`, untouched**; neither was copied, extracted in place, renamed or modified. All scripts open them read-only and write only under `AUDIT/candidates/` or `AUDIT/*.csv`.

No Member 2 production file, builder, model or canonical dataset was modified. The earlier `MEMBER1_LFP_B1_B5_*` deliverables were **not** overwritten; the extended B1–B6 matrices are new files that explicitly supersede them (§9).

---

## 3. Archive verification and provenance

### 3.1 Supplied files

| Cell | Path | Bytes | SHA-256 |
|---|---|---|---|
| B2 | `C:\Users\KIIT0001\Downloads\B2_Discharge1C_Charge0.5C.zip` | 372,458,134 | `b75c36d96abd2ae17858eda570dd6627ef06433bd88ac55faf73d688cbc10279` |
| B6 | `C:\Users\KIIT0001\Downloads\B6_Discharge0.5C_Charge2C.zip` | 368,836,614 | `25fe5a2417d68334eed59667c162db6851e1970a2109e523caf91dd11484b97a` |

Both open cleanly with `zipfile`; no CRC or parse errors were encountered.

### 3.2 Contents and byte identity

| | B2 | B6 |
|---|---|---|
| Workbooks (`.xlsx`) | 20, under `B2_Discharge1C_Charge0.5C/` | 19, under `B6_Discharge0.5C_Charge2C/` |
| Worksheets | 1,601 | 1,760 |
| Cycle telemetry sheets | 1,506 named `…Step2_CycleNN` | 1,686 named `…Step2_CycleNN` + 16 generic `Sheet1…Sheet16` = **1,702** |
| Workbook-level match | **20/20** SHA-256 match `DATA/B2_LFP_Member1_audit.zip::raw_integrity.csv` | **19/19** SHA-256 match `AUDIT/LFP_B6_Audit_Package (1).zip::LFP_workbook_summary.csv` |
| Extra/missing workbooks | none | none |

This resolves the prior **"B2 derived-only"** limitation directly: the raw files that produced the derived export are present and byte-identical. For B6 it confirms that the audit package describes exactly the supplied archive and not a different copy.

### 3.3 Source metadata and protocol (read from the files)

Common format: BioLogic/MultiStat `MultiStat_DCData_1.0`, MultiStat version 1.7f, columns `T(Seconds) | E(V) | I(A/cm2)`, `Surface Area: 0`, `Aux Count: 0`, 1 Hz.

| | B2 | B6 |
|---|---|---|
| Internal cell folder | `LFP 18650 (1250mAh)\BATTERY 2` | `…\BATTERY 6` |
| Charge | CC 0.625 A (0.50C) → 3.65 V, CV → 0.025 A | CC 2.4 A (2C) → 3.65 V, CV → 0.025 A |
| Discharge | CC 1.25 A (1.00C) → 2.50 V | CC 0.625 A (0.50C) → 2.50 V |
| Step time limit | 18,000 s | 18,000 s |
| Temperature channel | none | none |
| Wall-clock span | 2024-12-15 → 2025-11-17 (337.1 days) | 2025-02-26 → 2026-01-24 (332 days) |
| Instrument channels | Un1Ch8 (998 cycles) → Un1Ch2 (508) at cycle 999 | Un1Ch1 (201 cycles) → Un1Ch6 (1,485) at candidate index 202 |

### 3.4 Licensing and provenance

Both cells belong to the same official dataset, confirmed this session against the Zenodo API record:

* **Record:** “Experimental Dataset for Cycle Ageing of LFP 18650 Lithium-Ion Batteries under Different Charge and Discharge C-Rates”
* **Creators:** Noor Raihan Japar, Yousif Al-Sagheer, Robert Steinberger-Wilckens
* **Published:** 2026-09-03
* **DOI (version):** `10.5281/zenodo.22286450`; **concept DOI:** `10.5281/zenodo.22286449`
* **Licence:** **CC BY 4.0**
* **Official file:** `LFP_18650_Battery_Ageing_Raw_Data.zip` (2,262,650,872 bytes)
* **Institution record:** `https://research.birmingham.ac.uk/en/datasets/experimental-dataset-for-cycle-ageing-of-lfp-18650-lithium-ion-ba/`

The supplied archives are therefore legal to use with attribution. Provenance is no longer “unknown” as it was in the B1–B5 report. The archive split by cell is an external packaging choice; the instrument files themselves are the source of record.

---

## 4. B2 — independent recount and label audit

### 4.1 Cycle records

| Quantity | Value |
|---|---|
| Workbooks / cycle sheets | 20 / 1,506 |
| `cycle_index` | contiguous **1…1506**, no gaps, no duplicates |
| Valid / in-analysis cycles | 1,496 / 1,496 |
| Capacity/SOH values present | 1,502 (4 null) |
| Flagged excluded | **10** |
| Provenance mix | 1,346 File-List indexed, 91 inferred from sheet names, 69 continuation sessions |
| Sessions | 24 |

Per-workbook record counts are irregular (`Cycle_0050…0450` = 50 each; `Cycle_0600` = 150; `Cycle_0700…0900` = 100 each; `Cycle_1000` = 98; `Cycle_1100` = 17; `Cycle_1200…1500` = 100 each; `Cycle_1600` = 91), which is why the derived export keeps a derived global index rather than trusting filenames. The index is contiguous and was not rebuilt here.

### 4.2 Excluded cycles (preserved, not filled)

| cycle_index | Workbook / sheet | Reason |
|---|---|---|
| 41 | Cycle_0050 / Cycle41 | no discharge capacity (charge only, 0.0007 Ah) |
| 101 | Cycle_0150 / Cycle01 | incomplete (charge 0.004 Ah) |
| 151 | Cycle_0200 / Cycle01 | incomplete (charge 0.0004 Ah) |
| 972 | Cycle_1000 / Cycle072 | no discharge capacity |
| 998 | Cycle_1000 / Cycle098 | no capacity at all |
| 999 | Cycle_1100 / Cycle01 | no charge capacity; first cycle after channel switch |
| 1185 | Cycle_1300 / Cycle070 | no discharge capacity |
| 1186 | Cycle_1300 / Cycle071 | no charge capacity |
| 1411 | Cycle_1500 / Cycle096 | incomplete discharge → SOH 7.02 % (do not use) |
| 1412 | Cycle_1500 / Cycle097 | incomplete charge (0.09 Ah) |

All ten remain in the candidate table with `label_status = excluded_flagged`; none was deleted or imputed.

### 4.3 Labels and units

* `soh_rated` = `q_discharge_ah / 1.250 Ah × 100` → **nominal-based**, identical definition to B5/B6. In-analysis range **85.54 – 96.68 %**.
* `soh_initial_ref` = `q_discharge / q_discharge(first valid cycle)` → **within-cell retention proxy**. In-analysis range **88.63 – 100.17 %** (slightly above 100 % because the first valid cycle is not necessarily the maximum).
* Neither is an independently measured beginning-of-life reference. Both are **integrated throughput**, not ground truth from a reference performance test.
* Current is **amperes**; the `I(A/cm2)` header with `Surface Area = 0` is a label error. Numeric levels (≈1.25 A discharge / 0.625 A charge) match the protocol setpoints.
* Coulombic efficiency median 0.9757 (1,491 reliable pairs); the apparent < 1 artefact is the CV-charge vs 2.5 V-cutoff protocol, not a measurement of true CE.

### 4.4 Raw re-derivation (independent check)

| cycle_index | Raw Qdis (Ah) | Table Qdis (Ah) | \|diff\| |
|---|---|---|---|
| 50 | 1.194911 | 1.194911 | 5e-9 |
| 1506 | 1.070049 | 1.070049 | 6e-9 |

Both cycles were integrated directly from the raw `T/E/I` sheets with the same trapezoid rule, independently of the export code.

### 4.5 B2 caveats for Member 2

* **Channel switch at cycle 999** (Un1Ch8 → Un1Ch2, 2025-08-22): +0.9 % capacity step and −17.5 mV mean-discharge-voltage shift in contradictory directions, i.e. a contact/measurement artefact. Exclude or model cycles 999–1015.
* No temperature; no SOC/DoD; no measured BOL.
* Labels are cycle-level. The 255-column export contains label-derived columns (`q_discharge_ah`, `q_charge_ah`, `soh_*`, `coulombic_efficiency`) that must **never** be model inputs.

---

## 5. B6 — independent recount and label audit

### 5.1 Cycle records

| Quantity | Value |
|---|---|
| Workbooks | 19 |
| Named cycle sheets | 1,686 |
| Generic cycle-like sheets | 16 (`Sheet1…Sheet16` in `Cycle_0700.xlsx` only) |
| Total cycle-like records | **1,702** |
| Conditional capacity/SOH labels | **1,697** |
| Screened out (unlabelled) | **5** |
| Audit `global_cycle_id` | **absent for every row** (all NaN) |
| Chronological coverage | 2025-02-26 → 2026-01-24 (332 days) |

Per-workbook counts: `Cycle_0050` = **51**, `Cycle_0100/0150/0200` = 50 each, `Cycle_0700` = **101**, the remaining 14 workbooks = 100 each. Filename checkpoints are therefore **approximate session breakpoints, not exact global cycle numbers**; the cumulative offset is +1 from the 50-block and +2 from the 700-block onward.

### 5.2 Why the audit has no global cycle id

Each workbook is a separate logger session, the `File List` sheet only indexes the first 20 cycles of a workbook (and misses the 16 generic sheets entirely), and sheet names restart at `Cycle01` in each file with inconsistent zero-padding (`Cycle01` vs `Cycle001`). The audit correctly refused to invent an index. It records `checkpoint_from_filename` plus per-workbook `source_cycle_tab_number`, and nothing else.

### 5.3 The five screened-out cycles (preserved, not filled)

| Workbook | Sheet | Reason |
|---|---|---|
| `Cycle_0050` | `Run01_Un1Ch1_Step2_Cycle21` | short / incomplete discharge |
| `Cycle_0700` | `Sheet1` | terminal voltage did not approach 2.50 V |
| `Cycle_0900` | `Run01_Un1Ch6_Step2_Cycle071` | terminal voltage did not approach 2.50 V |
| `Cycle_1500` | `Run01_Un1Ch6_Step2_Cycle093` | no negative-current samples |
| `Cycle_1700` | `Run01_Un1Ch6_Step2_Cycle100` | terminal voltage did not approach 2.50 V |

### 5.4 Labels, units and quality

* 1,697 labels; `soh_nominal_pct_conditional` = capacity / 1.25 Ah × 100. Range **81.70 – 99.65 %**, median **87.99 %**. First labelled cycle 99.65 %, last labelled 81.70 % — a smooth monotone fade.
* Qdis range **1.0212 – 1.2457 Ah**.
* CE median **0.9886** (1,696 pairs, zero-charge cycle excluded) — physically plausible unlike B5 (1.0278), which strengthens confidence in the integration.
* Current: numeric values match the 0.625 A discharge / 2.4 A charge setpoints, but the header says `A/cm2` with `Surface Area: 0`. Audit status: **1,701 VERIFIED_WITH_CONDITIONS, 1 UNDERDETERMINED**. This is a *conditional* ampere interpretation, not an attested unit.
* Quality flags: 1,300 cycles carry only `out_of_protocol_voltage_strict_bound` (a strict ±band on any sample); only **2** have a material excursion > 20 mV; **378** carry a time-gap flag; 4 carry cross-sheet timestamp overlap/reset.
* Two instrument channels (Un1Ch1 201 cycles, Un1Ch6 1,485) and 19 sessions with capacity steps of 0.5–0.8 pp at file boundaries (e.g. 601→602 +0.54 pp, 617→618 +0.83 pp after an 11-hour gap). These are session/contact offsets, not ageing.
* No temperature channel (`Aux Count 0`); no SOC/DoD; no measured BOL; `physical_cell_count_verified = NO` (identity is source-label based).

### 5.5 Raw re-derivation (independent check)

`Cycle_0050 / Cycle10`: raw 1.236828 Ah discharge / 1.217425 Ah charge = audit values. `Cycle_0700 / Sheet10`: raw 1.121569 / 1.141055 Ah = audit values. Both match exactly at the printed precision.

---

## 6. B6 candidate chronology (the new capability)

### 6.1 Method

`AUDIT/lfp_b6_chronology.py` reads, for every cycle-like sheet:

1. the `Date:` and `Time:` fields inside the MultiStat header block, and
2. the `Time` column of the workbook's `File List` sheet, where the sheet is indexed.

Where both exist, the File List value is used (sub-second precision); otherwise the header value (1-second precision). Rows are sorted by timestamp, then by checkpoint and tab number. Capacity/SOH values are **copied unchanged** from the audit; the script adds no label.

### 6.2 Validation results

| Check | Result |
|---|---|
| Cycle-like sheets extracted | 1,702 (matches audit exactly) |
| Orderable | 1,702 / 1,702 |
| Missing timestamps | 0 |
| Duplicate timestamps | **0** |
| Backwards timestamps in candidate order | **0** |
| Checkpoint regressions | **0** |
| Non-contiguous checkpoint blocks | **0** |
| Within-workbook tab order runs backwards | **0** |
| Header vs File List mismatches | **0** (1,620 sheets have both; 82 header-only) |
| Generic `Sheet1…16` placement | candidate indices **602–617**, 2025-07-17 10:04 → 2025-07-18 21:42, i.e. *before* the named cycles of the same workbook (2025-07-19 10:27) |
| Physical plausibility | 1,697 labels fall monotonically from 99.65 % to 81.70 %; the only > 2 pp step follows the interrupted cycle 093 (cross-sheet reset) |

This is a strong reconstruction, but it is a **candidate**, not an audited `global_cycle_id`. The audit itself declined to assign one, and this script preserves that distinction in the column name and in the summary JSON.

### 6.3 What the chronology reveals

* `Cycle_0050` contains 51 real cycles, not 50.
* `Cycle_0700` contains 101 real cycles: 16 previously unindexed sheets (17–18 July) followed by 85 named cycles (19–27 July).
* Filename checkpoints therefore drift from the true record count. Any model that uses the filename number as a cycle index inherits a 1–2 cycle offset after the 50th and 700th blocks.

---

## 7. Candidate data delivered (isolated)

Nothing was written to `DATA/`, `master_table.*`, the production builder, or any Member 2 file.

| File | Rows | Purpose |
|---|---|---|
| `AUDIT/candidates/B2_candidate_cycle_labels.csv` | 1,506 × 27 | B2 cycle labels, raw-verified, flags preserved |
| `AUDIT/candidates/B2_candidate_summary.json` | — | archive hashes, 20/20 workbook match, spot checks |
| `AUDIT/candidates/B6_candidate_cycle_chronology.csv` | 1,702 × 30 | B6 candidate order + unchanged audit labels/flags |
| `AUDIT/candidates/B6_candidate_chronology_summary.json` | — | timestamp reconciliation and order validation |

### 7.1 B2 candidate contract

Key columns: `cycle_index`, `source_file`, `worksheet`, `timestamp`, `channel`, `provenance`, `valid_cycle`, `in_analysis`, `q_discharge_ah`, `q_charge_ah`, `soh_rated_pct`, `soh_initial_ref_pct`, `coulombic_efficiency`, `ce_reliable`, `source_sha256`, `raw_manifest_verified`, `label_status`, `notes`.

`label_status` is `conditional_nominal_soh` (1,496) or `excluded_flagged` (10). No row is filled.

### 7.2 B6 candidate contract

Key columns: `candidate_global_index`, `datetime_local`, `datetime_source` (`file_list`/`sheet_header`), `source_file`, `workbook_checkpoint`, `worksheet`, `sheet_type`, `source_cycle_tab_number`, `channel`, `protocol_repeat`, charge/discharge setpoints, `sample_count`, `discharge_capacity_ah_conditional`, `charge_capacity_ah_conditional`, `soh_nominal_pct_conditional`, `soh_provenance`, `complete_discharge_screen`, `current_unit_assessment`, `quality_flag`, `label_status`, `notes`.

`label_status` is `conditional_nominal_soh` (1,697) or `unlabelled_screen_failed` (5). `candidate_global_index` is explicitly **not** `global_cycle_id`.

### 7.3 Integration method and checks

1. Verify archive SHA-256 (§3.1) before and after the run; abort on change.
2. Confirm every workbook hash against the existing manifest (20/20, 19/19).
3. Re-derive sample cycles from raw and compare to the label source (1e-8 Ah B2; exact B6).
4. Copy labels and flags unchanged; derive only `*_pct` scaling and `label_status`.
5. Validate B6 order against three independent invariants: no timestamp reversal, contiguous checkpoint blocks, agreement with the File List.
6. Write only inside `AUDIT/candidates/`.
7. Run the test suite (96 tests total) and re-hash canonical data.

---

## 8. Reconciliation with the master table and earlier findings

| Question | Answer |
|---|---|
| Does B2 raw resolve missing source data? | **Yes.** The prior report's “raw not available” is now false; raw telemetry exists and matches the export. |
| Does B2 resolve a mapping/ingestion issue? | No new ingestion issue. The derived global index 1…1506 is contiguous and re-derivable; the irregular per-file counts are real, not a mapping bug. |
| Can existing code parse the new archives? | Yes — the existing helpers (`read_cycle_sheet`, `segment_cycle`) parse both; B2 uses the same MultiStat layout. |
| Are labels/units compatible with the B5 contract? | **Labels yes, units conditionally.** All three use capacity/1.25 Ah; B2 and B6 are amperes under a wrong `A/cm2` header, exactly like B5. B2 additionally exposes a first-cycle retention target. |
| Does B6 resolve missing source data? | **Yes for raw telemetry; no for chronology.** Raw is verified, but the audit deliberately supplies no global cycle id. The candidate chronology fills that gap with a validated reconstruction. |
| What still needs manual review? | B6 current-unit attestation, B6 chronology sign-off, B2 channel-switch handling, and the choice of B2 target definition. |

### 8.1 Changed conclusions versus `MEMBER1_LFP_B1_B5_DATA_READINESS_REPORT.md`

| Prior conclusion | Current conclusion | Evidence |
|---|---|---|
| B2 “ONLY_DERIVED_DATA_AVAILABLE”; raw absent | **SOURCE_VERIFIED_RAW_AND_DERIVED** | 20/20 workbook SHA-256 match; two raw re-derivations |
| B2 `voltage_series` NOT_AVAILABLE | **READY_WITH_CAVEAT** (raw 1 Hz) | raw archive |
| B2 has one SOH target (`soh_rated`) | B2 has **two**: nominal and first-cycle retention | dataset card + candidate table |
| B6 “out of scope / audit-only” | **added as a candidate cell** with verified raw and candidate chronology | 19/19 match; chronology validation |
| Provenance/licence unknown | **CC BY 4.0**, Zenodo `10.5281/zenodo.22286450` | Zenodo API |
| B2 suitability already SUITABLE | unchanged, but now **evidenced by raw**, not only by the derived export | — |

The three earlier B1–B5 deliverables are unchanged on disk; the new B1–B6 matrices supersede them and mark every changed row.

---

## 9. Extended matrices

Two new files preserve the earlier B1–B5 snapshot and extend it:

* `AUDIT/MEMBER1_LFP_B1_B6_COVERAGE_MATRIX.csv` — 6 cells × 23 columns. New columns `source_cycle_records`, `source_labelled_records`, `change_vs_b1_b5_report`; master-side counts renamed `master_*` to remove the earlier ambiguity (the old `observed_cycle_count` meant “cycles present in the master”, not cycles in the source).
* `AUDIT/MEMBER1_LFP_B1_B6_FEATURE_READINESS_MATRIX.csv` — 85 rows (70 prior + 15 B6). B2 rows updated from “derived-only” to “source and derived”; B6 rows added for all 14 standard features plus `cycle_start_timestamp`.
* `AUDIT/MEMBER1_LFP_B1_B6_MATRIX_CHANGELOG.json` — machine-readable record of what changed and why.

B6 headline values in the coverage matrix: 1,702 source records, 1,697 conditional labels, SOH 81.70–99.65 %, CE 0.9886, two channels, no temperature, candidate chronology, suitability `POTENTIALLY_SUITABLE_AFTER_VERIFICATION`.

---

## 10. B1–B6 ranking for independent validation

Criteria, in the order requested: (1) verified raw availability, (2) trustworthy cycle labels, (3) chronological coverage, (4) verified units/compatible features, (5) suitability as an independent cell for Member 2's B5 Ridge model. B5 is the training cell and is not an independent holdout by definition.

| Rank | Cell | 1 Raw | 2 Labels | 3 Chronology | 4 Units/features | 5 Independent? | Overall |
|---|---|---|---|---|---|---|---|
| **1** | **B2** | **20/20 verified** | 1,496 in-analysis; two targets; 10 flagged | contiguous 1…1506; 337 d | amperes; rich features; channel switch | **Yes** | **Best independent cell** |
| **2** | **B6** | **19/19 verified** | 1,697 conditional; CE 0.9886 | candidate order validated; 332 d | amperes *conditional*; 2 channels | **Yes, after checks** | Strong second cell |
| 3 | B3 | verified | 1,971 labels; 29-cycle gap | not ingested; 4 channels | amperes | Yes, conditional | Good but more caveats |
| 4 | B1 | verified | 1,456 labels; 110 offset-uncertain | master only 1 cycle | amperes | Yes, conditional | Needs full re-ingestion |
| 5 | B4 | verified | **0 computed labels**; 5 workbooks unaudited | 3C; partial audit | amperes | Yes, but weak | Re-audit needed |
| 6 | B5 | verified | label quality blocked (CE 1.028) | 250 master cycles | amperes | **No — training cell** | Not a holdout |

Rationale for the top two:

* **B2** wins on label breadth (1,496), a truly contiguous chronology, a clean ampere convention, 255 engineered features, and an explicit first-cycle retention target. Its only material confound is the cycle-999 channel switch.
* **B6** wins on label consistency (CE 0.9886, no B5-style charge-balance anomaly) and now has a validated chronology, but its current unit is only *conditional*, its chronology is a reconstruction, and 19 sessions introduce 0.5–0.8 pp offsets. It is a genuine second option, not a replacement for B2.

B6 is **not** assumed suitable merely because its raw archive exists: it clears raw availability and label quality, but criteria 3 and 4 are met only with the candidate chronology and a unit attestation.

---

## 11. What Member 2 must verify before training or evaluating

### B2

1. **Choose the target.** `soh_rated_pct` (nominal) is definition-compatible with B5; `soh_initial_ref_pct` is the within-cell retention target. Do not mix them.
2. **Handle the channel switch.** Exclude or covary cycles 999–1015; check that the Ridge model's feature set does not include channel-correlated artefacts.
3. **Never use label columns as inputs** (`q_discharge_ah`, `q_charge_ah`, `soh_*`, `coulombic_efficiency`, `ce_reliable`).
4. **Do not reuse B2's internal split** (`train 1–1053 / val 1054–1279 / test 1280–1506`) when B2 is the external holdout; that split is for B2-internal modelling.
5. **Group labels by `cycle_index`.** The 255-column export is cycle-level; the raw telemetry is row-level. Broadcasting a label over telemetry rows leaks the target.
6. **Confirm feature-schema compatibility** with the B5 Ridge pipeline (feature names, units, availability); retrain or re-select features rather than silently zero-filling missing ones.

### B6

1. **Attest the current unit.** Confirm that `I` is total amperes, not A/cm², by checking the 0.625 A discharge and 2.4 A charge plateaus against the protocol setpoints. Until then treat capacities as conditional.
2. **Review the candidate chronology.** Spot-check it against the File List dates and workbook order. Confirm the 16 generic sheets belong to the same physical cell and precede the named 85-cycle block. Do not rename `candidate_global_index` to `global_cycle_id` without sign-off.
3. **Handle session offsets.** The 19 workbooks show 0.5–0.8 pp capacity steps at boundaries; model session as a covariate or de-trend. Do not interpret these steps as ageing.
4. **Use only labelled cycles.** 1,697 are usable; keep the 5 screened-out cycles unlabelled. Do not fill them.
5. **Accept the missing inputs:** no temperature, no SOC/DoD, no measured BOL.
6. **Do not use filename checkpoints as cycle numbers** after the 50th and 700th blocks (offset +1/+2).

### Both

* Treat integrated capacity/SOH as **conditional labels**, never as independently verified ground truth; no reference performance test exists in either archive.
* Do not use B5 as a holdout and do not retrain on B2/B6 until the above checks pass.
* If the raw archives are needed repo-local for reproducibility, copy them into `DATA/` under a new name and record their SHA-256; do not rename or re-zip the originals.

---

## 12. Reproducible scripts and tests

| File | Role |
|---|---|
| `AUDIT/lfp_b6_chronology.py` | reads B6 raw + audit, writes candidate chronology CSV/JSON |
| `AUDIT/lfp_b2_candidate.py` | verifies B2 raw against the export manifest, writes candidate labels CSV/JSON |
| `AUDIT/lfp_b2_b6_matrices.py` | builds the B1–B6 coverage and feature matrices + changelog |
| `AUDIT/test_lfp_b2_b6_followup.py` | 27 tests covering archives, candidates, chronology, matrices, canonical immutability |

The B2 script and the tests reuse `AUDIT/lfp_b1_b5_inventory.py` helpers for sheet parsing and trapezoid integration; the B6 script only orders timestamps and reads headers, so it does not need them.

---

## 13. Quality gates (actual results)

| Gate | Result |
|---|---|
| `python AUDIT/lfp_b6_chronology.py` | 1,702 sheets; 0 duplicate/reversed timestamps; 0 checkpoint regressions; 0 header/File-List mismatches |
| `python AUDIT/lfp_b2_candidate.py` | 20/20 workbook hashes match; cycles 50/1506 re-derived to <1e-8 Ah; 1,496 labels + 10 flagged |
| `python AUDIT/lfp_b2_b6_matrices.py` | 6 coverage rows; 85 feature rows |
| New tests | **27 passed** |
| Existing suites (`test_lfp_b1_b5_inventory`, `test_member1_audit`, `test_soh_recovery`) | **69 passed** |
| Total | **96 passed, 0 failed** |
| New CSVs parse | yes (row counts 1,506 / 1,702 / 6 / 85) |
| B2/B6 raw archives unchanged | yes (hash re-checked before/after) |
| `master_table.parquet` / `.csv` unchanged | yes (hashes above) |
| Member 2 files/models modified | none |
| Git | branch `main`; no tracked diff; nothing staged; nothing committed or pushed |

---

## 14. Exact files created or modified

**Created (all untracked; nothing staged):**

* `AUDIT/MEMBER1_LFP_B2_B6_RAW_ARCHIVE_FOLLOWUP.md` (this report)
* `AUDIT/MEMBER1_LFP_B1_B6_COVERAGE_MATRIX.csv`
* `AUDIT/MEMBER1_LFP_B1_B6_FEATURE_READINESS_MATRIX.csv`
* `AUDIT/MEMBER1_LFP_B1_B6_MATRIX_CHANGELOG.json`
* `AUDIT/candidates/B2_candidate_cycle_labels.csv`
* `AUDIT/candidates/B2_candidate_summary.json`
* `AUDIT/candidates/B6_candidate_cycle_chronology.csv`
* `AUDIT/candidates/B6_candidate_chronology_summary.json`
* `AUDIT/lfp_b6_chronology.py`
* `AUDIT/lfp_b2_candidate.py`
* `AUDIT/lfp_b2_b6_matrices.py`
* `AUDIT/test_lfp_b2_b6_followup.py`

**Modified:** none. The prior `MEMBER1_LFP_B1_B5_*` files, all `DATA/*` archives, `master_table.*`, and every Member 2 artifact are unchanged.

**External inputs (not modified, not copied):**

* `C:\Users\KIIT0001\Downloads\B2_Discharge1C_Charge0.5C.zip`
* `C:\Users\KIIT0001\Downloads\B6_Discharge0.5C_Charge2C.zip`

---

## 15. Closure addendum — 2026-10-29 (supersedes §14 external-input status)

Both archives have since been copied into the repository working tree and every B2/B6 script path now points there. Byte identity was re-verified against the hashes recorded during the follow-up.

| Archive | Final location | Size (bytes) | SHA-256 | Verified |
|---|---|---:|---|---|
| `B2_Discharge1C_Charge0.5C.zip` | `DATA/B2_Discharge1C_Charge0.5C.zip` | 372,458,134 | `b75c36d96abd2ae17858eda570dd6627ef06433bd88ac55faf73d688cbc10279` | byte-identical to the `Downloads` copy |
| `B6_Discharge0.5C_Charge2C.zip` | `DATA/B6_Discharge0.5C_Charge2C.zip` | 368,836,614 | `25fe5a2417d68334eed59667c162db6851e1970a2109e523caf91dd11484b97a` | byte-identical to the `Downloads` copy |

Consequences for the historical statements above:

* "External inputs (not modified, not copied)" is superseded: the originals remain in `Downloads` untouched, and copies now live in `DATA/`.
* `DATA/` is `.gitignore`d, so the archives are **not** Git-tracked. The sizes (355 MB + 352 MB) also exceed GitHub's 100 MB file ceiling; Git LFS or approved external storage remains required for archive distribution.
* `AUDIT/member1_audit_outputs/lfp_b1_b5_evidence.json` is a dated B1–B5 snapshot. Its `b6_out_of_scope_note.raw_present_in_workspace = false` reflects the earlier brief, not current state; the live `b6_note()` now reports presence and the corresponding test asserts `True`.
* Candidate master rebuilt from the `DATA/` paths: 1,148,003 rows, 13 columns, parquet SHA-256 `4b1ec1348d6b179eb103616fe30dc6de94fceec3e9273f6e33990fbb060feb21`.
