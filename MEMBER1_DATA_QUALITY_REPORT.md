# Member 1 — Data Quality Report

All figures below were produced by executing the scripts in `AUDIT/` during this session (2026-10-29) against the raw archives and existing audit packages. Every number is traceable to a machine-readable artefact listed in `MEMBER1_SOURCE_INVENTORY.csv`, `AUDIT/candidates/*_summary.json`, `AUDIT/MEMBER1_DATASET_COVERAGE.csv` and `master_table_candidate_validation.json`. Historical audit claims are marked **HISTORICAL** and were re-verified against raw files where the raw archive was accessible.

---

## 1. Scope and method

| Item | Value |
|---|---|
| Raw archives read | `B1_LFP_Audit_Output.zip`, `B5.zip`, `Existing_Audits_Available_B3_B4 (1).zip`, `LFP_B6_Audit_Package (1).zip`, B2/B6 raw `.zip` (now stored in `DATA/`), NASA `.mat` directory, Na-ion `.pkl` directory |
| Extraction method | Direct XLSX XML parsing (`xl/workbook.xml` + `xl/_rels/workbook.xml.rels` + `<row>/<c>` regex); openpyxl only used for inventory classification of generic tabs |
| Telemetry sampling | Deterministic 50 charge + 50 discharge points per cycle (`np.linspace` over row indices), preserving measured values only |
| Label sources | Historical audit tables per dataset (listed per section) |
| Canonical baseline | `master_table.parquet` SHA-256 `6fa7475d…3b2bb`, `master_table.csv` SHA-256 `9e237926…29895b`, 515,270 rows — unchanged |
| Candidate master | `master_table_candidate.parquet` / `.csv`, 1,148,003 rows × 13 columns |

No raw file was modified. No label was imputed, interpolated or synthesised.

---

## 2. Dataset findings

### 2.1 B1 — LFP, `LFP_MultiStat_B1` (CONDITIONAL / CONDITIONAL)

**Verified now**
- 19 workbooks, 1,480 worksheet tabs of which **1,460 are `Step2_Cycle` cycle tabs** and 20 are raw EC‑Lab export tabs (`SheetN` with `Begin Header`, >500 rows).
- 1,456 capacity/SOH labels reproduced from `B1_LFP_Audit_Output/LFP_capacity_soh_feasibility.csv`.
- Telemetry: 145,901 rows, 1,460 cycles, balanced 72,951 charge / 72,950 discharge samples.

**Defect — canonical span/join bug (reproduced, not merely inspected)**
- Canonical `LFP_MultiStat_B1` contains **only `cycle_id = 2`**, 4,789 rows, with a **constant** `capacity_ah = 1.202223` and `soh_pct = 96.178`.
- Those values equal the **cycle‑50** row of `Cycle_0050.xlsx / Run01_Un1Ch*_Step2_Cycle50`, not cycle 2.
- Root cause in `master_table_build.py`: `cycle_cap_map = dict(zip(df_cycle['cycle_checkpoint'], df_cycle['capacity_ah']))`. Every per-cycle label row shares the workbook checkpoint key (e.g. 50), so `zip` collapses 50 rows to one and the last wins. The single label was then broadcast to every row of the only ingested sheet.
- True cycle‑2 label from the feasibility table is **1.220155 Ah / 97.612 %**.
- Resolution in candidate: labels joined on `(source_file, worksheet)` instead of checkpoint; B1 now spans 1,460 cycles / 145,901 rows.

**Residual limitations**
- 110 cycle offsets underdetermined (`Cycle_0200` contributes 30 tabs, `Cycle_1300` 80 tabs) → `OFFSET_UNDERDETERMINED`, `cycle_confidence = underdetermined`.
- Cycle IDs 151–170 and 1201–1220 do not exist in the label table. The 20 ASCII cycle exports in `Cycle_1300.xlsx` (`Sheet21`…`Sheet2`, 23–26 Sep 2025) match export-index cycles 001–020 by timestamp and fill the 1,200→1,300 block together with the 80 Step2 tabs, but they carry no audited capacity/SOH labels and the export index lists 21 cycles while only 20 data tabs exist. They are therefore **retained as a documented gap, not integrated**; see `AUDIT/MEMBER1_B1_CYCLE1300_GAP_EVIDENCE.json`. No cycle ID, capacity or SOH was invented.
- 4 partial-discharge cycles blocked (`BLOCKED_PARTIAL_DISCHARGE`).
- `I(A/cm2)` header with `Surface Area = 0`; numeric values are amperes (confirmed against 0.625 A = 0.5C protocol). Unit is conditional.
- Spike scan: 1 current row / 1 voltage row outside protocol bounds, flagged and retained.

### 2.2 B2 — LFP, `LFP_MultiStat_B2` (CONDITIONAL / CONDITIONAL, highest priority)

**Verified now**
- 20 workbooks verified **byte-identical** to the historical manifest (all SHA-256 and sizes match; no missing, no extras).
- `cycle_features.parquet` = 1,506 × 255; `qv_curves.parquet` = 1,506 × 3 (`cycle_index, discharge_qv, charge_qv`).
- 1,496 analysis-eligible cycles; **10 invalid** (`41, 101, 151, 972, 998, 999, 1185, 1186, 1411, 1412`).
- `soh_rated` is a **0–1 fraction** (0.0702–0.9668); in-analysis range 85.54–96.68 %. `soh_initial_ref` 0.0728–1.0017.
- Channel transition verified: **first `Un1Ch2` sheet is cycle 999**; counts 998 × Ch8, 508 × Ch2.
- Spot checks reproduce raw: cycle 50 `1.194911 Ah` (Δ 5e‑9), cycle 1506 `1.070049 Ah` (Δ 6e‑9).
- Telemetry: 150,172 rows, 1,505 cycles. Cycle 998 produces **no charge/discharge samples** (all-zero current, 3,600 rest rows) and is excluded; 2 raw workbooks carry 1 current + 1 voltage spike row, flagged.

**Leakage control (feature/target manifest, 255 columns audited)**
- Direct targets: `q_discharge_ah`, `soh_rated`.
- Derived-target leakage: `soh_initial_ref`, `soh_slope_per_100cyc`, `q_fade_rate_ah_per_100cyc`, `coulombic_efficiency` — must not appear as features for a same-cycle target.
- `MEMBER1_FEATURE_TARGET_MANIFEST.csv` marks every column `FEATURE_SAFE / TARGET / TARGET_DERIVED_LEAKAGE / IDENTIFIER / METADATA`; `MEMBER1_TARGET_DEFINITIONS.csv` records the 8 dataset target definitions.

**Channel treatments (explicit, Member 2 decides)**
| Treatment | Cycles | 70/15/15 chronological | Note |
|---|---|---|---|
| EXCLUDE window 999–1015 | 1,480 | 1,036 / 222 / 222 | 999 already invalid; 1000–1015 eligible but removed |
| Channel covariate | 1,496 | 1,047 / 224 / 225 | retain all valid cycles, pass channel |
- Mask: `AUDIT/candidates/B2_channel_transition_mask.csv` (1,506 rows).
- Recommendation: EXCLUDE primary, covariate sensitivity.

**Residual limitations**
- No temperature channel.
- 160 cycles have provenance-inferred filenames (HISTORICAL, re-checked against raw file hashes).
- Labels are cycle-level; must be joined once per `(cell, cycle)`, never per row.

### 2.3 B3 — LFP, `LFP_MultiStat_B3` (CONDITIONAL / CONDITIONAL)

**Verified now**
- 25 workbooks, 1,979 cycle tabs, 1,979 audited, 1,973 Qdis labels.
- Missing cycle IDs **201–221 confirmed exactly** (21 IDs, contiguous).
- Four instrument channels confirmed: `Un1Ch2` 879, `Un1Ch8` 500, `Un1Ch1` 500, `Un1Ch6` 100.
- Sample `1.165491 Ah` reproduced from raw.
- Quality tiers: A 1,968 / B 3 / X 8.
- Capacity 0.238124–1.165723 Ah; SOH 78.139–93.258 %.
- Telemetry: 197,441 rows, 1,979 cycles; no spike rows.

**Residual limitations**
- Physical cell identity per channel undocumented — channel is a covariate only, not a cell label.
- Current unit conditional (`I(A/cm2)` header, area 0).
- 8 quality-X and 3 quality-B cycles are `BLOCKED`.

### 2.4 B4 — LFP, `LFP_MultiStat_B4` (CONDITIONAL / **BLOCKED** targets)

**Verified now**
- 21 workbooks, 1,577 cycle tabs, all audited for coverage. Historical audit covered **16 workbooks / 1,350 sheets**; the 5 previously unaudited checkpoints (150, 200, 250, 300, 350 → 228 tabs) are now read.
- Telemetry: 157,501 rows, 1,577 cycles, **0 capacity labels**.
- Missing cycle IDs 201–222 and 1101 (reconstructed from tab order).
- Capacity and SOH are **null by design** — no ground truth exists in the audit or raw data.

**Decision** — B4 is included in the candidate master for *telemetry only*. No synthetic capacity/SOH was created. `target_status = BLOCKED`.

**Residual limitations**
- Current unit and charge/discharge segmentation unresolved → all cycles `UNLABELED`, `CYCLE_ID_RECONSTRUCTED`.
- Old-audit "1,382 rows" reflects 16 checkpoints × weighted cycles; not directly comparable.

### 2.5 B5 — LFP, `LFP_MultiStat_B5` (CONDITIONAL / CONDITIONAL)

**Verified now**
- 17 workbooks, **1,498 real cycles** = 1,491 `Step2_Cycle` + 7 renamed/generic tabs.
- Canonical master contained only 4 workbooks (50, 100, 200, 500) / 250 cycles; candidate covers all 17 / 1,498.
- Cycle‑40 interruption reproduced: `Run01_Un1Ch1_Step2_Cycle40` = **0.5428382 Ah**, `dis_complete = False`, `DISCHARGE_INTERRUPTED`.
- Qdis > Qchg in **1,488 / 1,497** cycles that have both values (1,498 total; one cycle lacks a discharge capacity); CE ≈ **1.0278** — systematic, not random.
- `capacity_ground_truth = NO` for all cycles. Capacity 0.2884506–1.196897 Ah.
- Telemetry: 149,750 rows, 1,498 cycles; no spike rows.

**Defect — historical audit overcount**
- `B5_Cycle_Summary.csv` has 1,499 rows because the **`File List` tab of `Cycle_1200.xlsx` was misclassified as `CYCLE_RENAMED_SHEET`** (nominal checkpoint 1108, capacity 0.164741, SOH null). It is a file inventory, not a cycle.
- Corrected count = 1,498; the row is dropped and documented in `AUDIT/candidates/member1_cycle_candidates_summary.json`.

**Residual limitations**
- 6 partial-discharge cycles (`PARTIAL_DISCHARGE`, capacity conditional).
- SOH uses nominal 1.25 Ah denominator, not measured BOL.

### 2.6 B6 — LFP, `LFP_MultiStat_B6` (CONDITIONAL / CONDITIONAL)

**Verified now**
- 19 workbooks, 1,702 cycle-like tabs, **1,702 cycles** and 1,697 labels. Includes 16 generic `SheetN` raw-export tabs in `Cycle_0700.xlsx` that are genuine cycles (ordered by timestamp), not file lists.
- Chronology reconstructed from header `Date/Time` and `File List` fields: 1,620 sheets have both sources, **0 mismatches**, 82 header-only, 0 without any timestamp, 0 duplicate timestamps, 0 backwards steps.
- Labels: 1,697 conditional nominal SOH (81.699–99.653 %, median 87.988), 5 unlabelled screen failures.
- Telemetry: 170,101 rows, 1,702 cycles; no spike rows.

**Residual limitations**
- Chronology is candidate order only (no audited global cycle ID).
- `master_table_build.py` hard-codes `C:\Users\KIIT0001\...` paths — replaced in the candidate pipeline by repo-relative/arg-driven paths; canonical script left untouched.
- Current unit and no-temperature caveats as B1/B3.

### 2.7 NASA Ames PCoE — NMC (VERIFIED / VERIFIED)

- 48 `.mat` cells in the archive; **3 target cells** (B0005, B0006, B0018) with **468** discharge cycles.
- 135,436 telemetry rows preserved unchanged from canonical.
- Measured voltage / current / temperature and direct capacity. Rated 2.0 Ah.
- Limitation: only 3 of 48 cells used; protocol differs from LFP cells.

### 2.8 Zenodo IonTech Na-ion (VERIFIED / VERIFIED)

- 65 cell pickles verified; canonical build uses **4 cells × 25 cycles = 100 cycles**, 41,701 rows preserved unchanged.
- Direct discharge capacity and per-cell nominal capacity.
- Limitation: temperature not measured; only 4/65 cells and 25 cycles/cell in canonical; chemistry differs from LFP.

---

## 3. Cross-dataset conformity (Phase 8)

| Dimension | Finding |
|---|---|
| Time unit | `T(Seconds)` for all six LFP MultiStat datasets; NASA uses `cycle` + relative time; Na-ion uses cycle index. Candidate `time_sec` populated only where the source is seconds. |
| Current unit | Raw values are amperes. The `I(A/cm2)` header is wrong: `Surface Area = 0` makes the stated unit undefined. Setpoint cross-check (B1 0.625 A / B2 1.25 A / B3 2.5 A / B4 3.75 A / B5 1.25 A charge / B6 2.4 A charge) confirms amperes. All LFP rows carry `CURRENT_UNIT_UNDERDETERMINED`. |
| Sign convention | Positive = charge, negative = discharge (EC-Lab convention). Verified per cycle by sign of `current_a` against `phase`. |
| Temperature | **1,012,567 nulls** in candidate. LFP workbooks contain no temperature column; NASA has measured temperature; Na-ion does not. Historical claim of 379,834 nulls refers to old column layout and is superseded. |
| SOH definition | `soh_pct = 100 × Q_discharge / 1.25 Ah` (nominal) for LFP. Not retention vs measured BOL. B2's `soh_rated` is the same definition expressed as a 0–1 fraction. NASA/Na-ion SOH is not redefined. |
| Identity/order | `(dataset_id, cell_id, cycle_id)` is unique in the candidate (0 duplicate keys). LFP `cycle_id` is reconstructed from workbook/tab order and is `CYCLE_ID_RECONSTRUCTED` for B4/B5/B6 and the offset-underdetermined B1 cycles. |
| Phase labels | `charge` / `discharge` derived from current sign; 50+50 samples retained per cycle. |

---

## 4. Quality-flag taxonomy and observed counts

| Flag | Meaning | Rows |
|---|---|---|
| `TEMP_UNMEASURED;CURRENT_UNIT_UNDERDETERMINED` | LFP B1/B2/B3 rows | 478,402 |
| `…;CYCLE_ID_RECONSTRUCTED` | LFP B6 | 172,551 |
| `…;NO_AUDITED_LABEL;UNLABELED;CYCLE_ID_RECONSTRUCTED` | B4 | 157,501 |
| `…;CAPACITY_GROUND_TRUTH_NO;CYCLE_ID_RECONSTRUCTED` | B5 | 149,200 |
| `VALID` | NASA | 135,436 |
| `TEMP_UNMEASURED` | Na-ion | 41,701 |
| `…;OFFSET_UNDERDETERMINED;CYCLE_ID_RECONSTRUCTED` | B1 underdetermined | 11,000 |
| `…;PARTIAL_DISCHARGE;…` | B1/B5/B6 partial | 550 + 450 + 350 |
| `…;INVALID_CYCLE` | B2 | 519 |
| `…;NO_DISCHARGE_CAPACITY` | B2 cycle 998 | 288 |
| `…;INVALID_CYCLE;CHANNEL_TRANSITION` | B2 window | 50 |
| `…;CURRENT_OUT_OF_PROTOCOL_BOUNDS` / `VOLTAGE_OUT_OF_PROTOCOL_BOUNDS` | spike rows | 1 / 1 |

Full breakdown: `master_table_candidate_validation.json → quality_flag_top`.

---

## 5. Verified defects and resolutions

| # | Defect | Dataset | Evidence | Resolution |
|---|---|---|---|---|
| 1 | Canonical B1 collapsed to cycle 2 with cycle‑50 label | B1 | canonical constant 1.202223 / 96.178 | `(source_file, worksheet)` join; full 1,460-cycle candidate |
| 2 | Canonical B5 only 4/17 workbooks, 250 cycles | B5 | canonical 4 cells/250 cycles | all 17 workbooks / 1,498 cycles in candidate |
| 3 | `File List` tab counted as a cycle | B5 | 1,499 vs 1,498 audit rows | row dropped, count corrected |
| 4 | B6 generic cycle tabs not counted (1,686 vs 1,702) | B6 | inventory undercount | classify `SheetN` by `Begin Header` + >500 rows; 16 genuine cycles recovered |
| 5 | B2 cycle 998 all-zero current | B2 | 3,600 rest rows | excluded from telemetry; counted invalid |
| 6 | Candidate schema drifted with helper column | candidate | `schema_conformity` false | helper computed separately; 13 columns restored |
| 7 | NASA/Na-ion cycle counts collapsed by per-cell reset | candidate | 168/25 vs 468/100 | composite `dataset|cell|cycle` counting |
| 8 | B4 five unaudited checkpoints | B4 | old 16 wb vs new 21 | all 21 read; capacity stays null |
| 9 | Hard-coded Windows paths in build script | B6 | `master_table_build.py` | candidate pipeline path-arg driven; canonical untouched |

---

## 6. Residual risks

1. **B4 has no target.** Any downstream SOH for B4 is blocked. The candidate includes its telemetry with explicitly null capacity/SOH.
2. **All LFP capacity/SOH values are conditional** on the nominal 1.25 Ah denominator and on amperes-as-current despite the `A/cm2` header.
3. **B1 offset underdetermination** (110 cycles) and **B5/B6 reconstructed order** mean cycle-level alignment is best-effort; calibration boundaries must be respected.
4. **B2 channel transition** is a measurement artefact; exclusion vs covariate is a modelling decision left to Member 2.
5. **Reproducibility is raw-dependent.** The LFP rebuild requires the raw MultiStat archives now stored in `DATA/` (`B2_Discharge1C_Charge0.5C.zip`, `B6_Discharge0.5C_Charge2C.zip`, plus B1/B3/B4/B5), which are gitignored because of size. They must be restored from approved storage before rerunning. NASA/Na-ion are reproducible from the repository.
6. **B5 Qdis > Qchg (1,488/1,498)** and **B3 missing 201–221** are source properties, not processing errors; both are surfaced rather than corrected.
