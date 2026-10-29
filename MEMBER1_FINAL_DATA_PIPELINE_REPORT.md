# Member 1 — Final Data Pipeline Report

**Scope:** complete data-engineering and data-quality work for all eight sources (LFP B1–B6, NASA NMC, Na-ion), a versioned candidate master table, and the Member 2 → Member 3 data contract.
**Date:** 2026-10-29
**Decision:** the candidate master is **delivered for review and not promoted**. The canonical master is byte-identical to its baseline.

---

## 0. Repository safety and state

| Check | Result |
|---|---|
| Branch / HEAD at build time | `main` / `e36eb73` (read-only baseline) |
| Delivery branch | `member1-final-audit`, branched from `e36eb73` |
| Tracked files modified | `.gitignore` only (excludes the 215 MB candidate CSV) |
| Canonical `master_table.parquet` | SHA `6fa7475d…3b2bb`, 515,270 rows — unchanged |
| Canonical `master_table.csv` | SHA `9e237926…29895b`, 515,270 rows — unchanged |
| Production models / scalers / manifests / inference APIs | untouched |
| Member 2 prediction API, Member 3 range calculator | untouched |
| `main` rewritten or force-pushed | no |

All work is additive and confined to `AUDIT/`, the two `master_table_candidate.*` files, and new root-level markdown/JSON documents.

---

## 1. Source inventory (Phase 1)

Eight sources were opened and hashed; availability was verified by reading each archive, not inferred from filenames.

| Dataset | ID | Chemistry | Raw source | SHA-256 (short) | Opened |
|---|---|---|---|---|---|
| B1 | LFP_MultiStat_B1 | LFP | `DATA/B1_Discharge0.5C_Charge0.5C(1).zip` | `fe04aec9…5a5f` | yes |
| B2 | LFP_MultiStat_B2 | LFP | `DATA/B2_Discharge1C_Charge0.5C.zip` | `b75c36d9…0279` | yes |
| B3 | LFP_MultiStat_B3 | LFP | `DATA/B3_Discharge2C_Charge0.5C (1).zip` | `aff29ebe…9bb2` | yes |
| B4 | LFP_MultiStat_B4 | LFP | `DATA/B4_Discharge3C_Charge0.5C (1).zip` | `a17779c4…98c6` | yes |
| B5 | LFP_MultiStat_B5 | LFP | `DATA/B5_Discharge0.5C_Charge1C (1).zip` | `f3d6b1ef…7b3b` | yes |
| B6 | LFP_MultiStat_B6 | LFP | `DATA/B6_Discharge0.5C_Charge2C.zip` | `25fe5a24…b97a` | yes |
| NASA | NASA_Ames_PCoE | NMC | `DATA/archive (4).zip` | `95df7163…4eff` | yes |
| Na-ion | Zenodo_IonTech_NaIon | Na-ion | `DATA/NA-ion.zip` | `fbb213c3…83f8` | yes |

Full detail, including derived audit packages and workbook/sheet/cycle counts: `AUDIT/MEMBER1_SOURCE_INVENTORY.csv` and `AUDIT/MEMBER1_SOURCE_TRACEABILITY.csv`.

**Archive locations:** the B2 and B6 raw archives were verified in `Downloads` (sizes 372,458,134 / 368,836,614 bytes, SHA-256 matching the inventory) and copied byte-identically into `DATA/`, which is gitignored under the team's size policy. Every build now reads from `DATA/`. The candidate is reproducible in this checkout; a fresh clone must restore the raw archives (and the audit zips) from approved storage first.

---

## 2. B1 — label collapse repaired (Phase 2)

- 19 workbooks, **1,460 `Step2_Cycle` tabs**, 1,456 capacity labels.
- Canonical contained **one cycle (cycle 2)**, 4,789 rows, with constant `1.202223 Ah / 96.178 %`.
- Provenance: those values belong to **cycle 50** of `Cycle_0050.xlsx`. `master_table_build.py` mapped labels with `dict(zip(df_cycle['cycle_checkpoint'], df_cycle['capacity_ah']))`, which collapses all cycles in a workbook to the checkpoint key; the surviving value was broadcast to the only ingested sheet.
- True cycle‑2 label: **1.220155 Ah / 97.612 %**.
- Candidate joins on `(source_file, worksheet)`; B1 now covers 1,460 cycles / **145,901 rows** with 1,456 labelled cycles.
- 110 offsets underdetermined, 4 partial discharges blocked. Cycle IDs 151–170 and 1201–1220 are absent from the label table. The **20 ASCII cycle exports** in `Cycle_1300.xlsx` (`Sheet21`…`Sheet2`, 23–26 Sep 2025) match export-index cycles 001–020 by timestamp and, together with the 80 Step2 tabs, fill the 1,200→1,300 block; however, the export index lists 21 cycles while only 20 data tabs exist, and none has an audited capacity/SOH label. They are **retained as a documented gap, not integrated** — see `AUDIT/MEMBER1_B1_CYCLE1300_GAP_EVIDENCE.json`.
- Regression test: `test_23_b1_mapping_bug_fixed`, `test_24_b1_flags`.

## 3. B2 — highest priority (Phase 3)

| Check | Result |
|---|---|
| Feature matrix | 1,506 × 255, verified from `DATA/B2_LFP_Member1_audit.zip` |
| Q–V curves | 1,506 × 3 (`cycle_index, discharge_qv, charge_qv`) |
| Raw archive | 20 workbooks, all hashes and sizes match the manifest |
| Analysis-eligible | 1,496; invalid 10 (`41,101,151,972,998,999,1185,1186,1411,1412`) |
| SOH scale | `soh_rated` is a 0–1 fraction (0.0702–0.9668); in-analysis 85.54–96.68 % |
| Channel | 998 × Ch8 then 508 × Ch2; first Ch2 at **cycle 999**; window 999–1015 |
| Provenance-inferred cycles | 160 (historical, re-validated by hash) |
| Temperature | not measured |

**Leakage controls.** `MEMBER1_FEATURE_TARGET_MANIFEST.csv` classifies all 255 columns. Direct targets: `q_discharge_ah`, `soh_rated`. Derived leakage: `soh_initial_ref`, `soh_slope_per_100cyc`, `q_fade_rate_ah_per_100cyc`, `coulombic_efficiency`.

**Channel treatments.** `B2_channel_transition_mask.csv` supports two explicit options:

| Option | Cycles | 70/15/15 chronological |
|---|---:|---|
| EXCLUDE 999–1015 (primary recommendation) | 1,480 | 1,036 / 222 / 222 |
| Channel covariate (sensitivity) | 1,496 | 1,047 / 224 / 225 |

Cycle 998 has 3,600 all-zero rest rows and contributes no charge/discharge samples; telemetry has 1,505 cycles, labels 1,506.

## 4. B3 (Phase 4)

25 workbooks / 1,979 cycle tabs / 1,979 audited / 1,973 Qdis labels. Missing IDs **201–221 confirmed exactly**. Four channels: `Un1Ch2` 879, `Un1Ch8` 500, `Un1Ch1` 500, `Un1Ch6` 100. Sample `1.165491 Ah` reproduced. Quality tiers A 1,968 / B 3 / X 8. Candidate: 1,979 cycles / **197,441 rows**. Physical cell identity per channel remains undocumented.

## 5. B4 (Phase 5)

21 workbooks / 1,577 cycle tabs, all read. The historical audit covered 16 workbooks / 1,350 sheets; five checkpoints (150, 200, 250, 300, 350 → 228 tabs) were previously unaudited and are now covered. **No audited capacity or SOH exists and none was synthesised**: candidate capacity/SOH are null and `target_status = BLOCKED`. 1,577 cycles / **157,501 rows** of telemetry are included for completeness, flagged `NO_AUDITED_LABEL;UNLABELED;CYCLE_ID_RECONSTRUCTED`.

## 6. B5 (Phase 6)

17 workbooks, **1,498 real cycles** (1,491 Step2 + 7 renamed). Canonical had 4 workbooks / 250 cycles. Historical audit's 1,499 rows included a **`File List` tab misclassified as a cycle** — excluded. Cycle 40 interruption reproduced: **0.5428382 Ah**, `dis_complete = False`. Qdis > Qchg in **1,488 / 1,497** comparable cycles (CE ≈ 1.0278); `capacity_ground_truth = NO` for all. Candidate: 1,498 cycles / **149,750 rows**, 1,497 with conditional capacity.

## 7. B6 (Phase 7)

19 workbooks / **1,702 cycles** / 1,697 labels; 16 generic cycle tabs recovered in `Cycle_0700.xlsx`. Chronology reconstructed from header `Date/Time` and `File List`: 1,620 sheets with both, **0 mismatches**, 0 unorderable, 0 duplicate timestamps, 0 backwards steps. Candidate: **170,101 rows**. The hard-coded Windows path in `master_table_build.py` was not reused; the candidate pipeline is path-arg driven.

## 8. Cross-dataset consistency (Phase 8)

- **Time:** `T(Seconds)` for LFP; NASA/Na-ion use cycle-relative fields. `time_sec` populated only where seconds are the source unit.
- **Current:** raw numbers are **amperes**; the `I(A/cm2)` header is invalid because `Surface Area = 0`. Setpoint checks (0.625/1.25/2.5/3.75/1.25/2.4 A) confirm. All LFP rows carry `CURRENT_UNIT_UNDERDETERMINED`.
- **Temperature:** 1,012,567 nulls in the candidate. LFP has no temperature channel; NASA is measured; Na-ion is not.
- **SOH:** nominal `100 × Qdis / 1.25 Ah` for LFP, not retention vs measured BOL. B2's `soh_rated` is the same quantity as a 0–1 fraction.
- **Identity:** `(dataset_id, cell_id, cycle_id, time_sec)` unique (0 duplicates). LFP cycle IDs are reconstructed where flagged.

## 9. Candidate master table (Phase 9)

| Property | Value |
|---|---|
| Rows | 1,148,003 |
| Columns | 13, exact canonical schema |
| Cycle identity | 819 → 10,289 composite `(dataset_id, cell_id, cycle_id)` |
| Duplicate keys | 0 |
| LFP rows | 970,866 rebuilt from raw |
| NASA + Na-ion rows | 177,137 preserved unchanged |
| Candidate parquet SHA-256 | `4b1ec1348d6b179eb103616fe30dc6de94fceec3e9273f6e33990fbb060feb21` |
| Candidate CSV SHA-256 | `0ffe404bc931c08903f3c0ddf915ff19793f70f28be5054b5d200c30f8b50657` |
| Impossible values | voltage 0, temperature 0, SOH 0, negative capacity 0 |

Per-dataset manifest, validation JSON and the canonical-vs-candidate comparison are in `master_table_candidate_manifest.csv`, `master_table_candidate_validation.json`, `MEMBER1_MASTER_TABLE_COMPARISON.md`.

**The candidate is not promoted, not renamed and not a replacement for the canonical table.** Promotion criteria are listed in the comparison document.

## 10. Readiness and handoff (Phases 10–11)

- Consolidated per-dataset feature readiness: `AUDIT/MEMBER1_LFP_B1_B6_FEATURE_READINESS_MATRIX.csv` (plus the earlier B1–B5 matrices).
- `MEMBER1_MEMBER2_HANDOFF_CONTRACT.md` — safe immediate / conditional exploratory / derived-only / blocked categories, leakage rules, quality-flag glossary, required return JSON.
- `MEMBER2_MEMBER3_INTERFACE_DATA_CONTRACT.md` — SOH 0–100 %, capacity only with a verified reference, chemistry/cell context, cycle/time, model version, quality flags, refusal cases, and the cell-Ah ≠ pack-kWh constraint.

## 11. Test evidence (Phase 12)

| Suite | Tests | Result |
|---|---:|---|
| `AUDIT/test_member1_pipeline.py` | 36 | PASS |
| `AUDIT/test_member1_audit.py` | 18 | PASS |
| `AUDIT/test_soh_recovery.py` | 19 | PASS |
| `AUDIT/test_lfp_b1_b5_inventory.py` | 32 | PASS |
| `AUDIT/test_lfp_b2_b6_followup.py` | 27 | PASS |
| **Total** | **132** | **132 passed / 0 failed / 0 errors / 0 skipped** |

Exact commands and timings: `MEMBER1_REPRODUCIBLE_COMMANDS.md`. Tests were executed this session; no result is inferred from code inspection.

## 12. Deliverable index

| Deliverable | Path |
|---|---|
| Final pipeline report | `MEMBER1_FINAL_DATA_PIPELINE_REPORT.md` |
| Source inventory | `AUDIT/MEMBER1_SOURCE_INVENTORY.csv` |
| Source traceability | `AUDIT/MEMBER1_SOURCE_TRACEABILITY.csv` |
| B1 1201–1220 gap evidence | `AUDIT/MEMBER1_B1_CYCLE1300_GAP_EVIDENCE.json` |
| Dataset coverage/status | `AUDIT/MEMBER1_DATASET_COVERAGE.csv` |
| Data quality report | `MEMBER1_DATA_QUALITY_REPORT.md` |
| Feature/target manifest | `AUDIT/MEMBER1_FEATURE_TARGET_MANIFEST.csv` |
| Target definitions | `AUDIT/MEMBER1_TARGET_DEFINITIONS.csv` |
| B1–B6 candidate exports | `AUDIT/candidates/` |
| B2 cycle-level dataset + mask | `AUDIT/candidates/B2_candidate_cycle_labels.csv`, `B2_channel_transition_mask.csv` |
| Candidate master | `master_table_candidate.parquet` / `.csv` |
| Candidate validation | `master_table_candidate_validation.json` |
| Candidate manifest | `master_table_candidate_manifest.csv` |
| Master comparison | `MEMBER1_MASTER_TABLE_COMPARISON.md` |
| Member 1→2 contract | `MEMBER1_MEMBER2_HANDOFF_CONTRACT.md` |
| Member 2→3 contract | `MEMBER2_MEMBER3_INTERFACE_DATA_CONTRACT.md` |
| Commands and tests | `MEMBER1_REPRODUCIBLE_COMMANDS.md` |
| Machine-readable status | `MEMBER1_STATUS.json` |
| Change log | `MEMBER1_CHANGELOG.md` |

## 13. Remaining actions

**Member 1 (close-out)**

1. Raw archives are now in `DATA/`; restore them from approved storage on any fresh clone (Git excludes `DATA/`).
2. Reviewer sign-off on the candidate (schema, row counts, flags, B4/B5 caveats) — currently **pending**, not claimed.
3. Decide whether the 20 `Cycle_1300` ASCII exports should be added in a future version as *unlabelled telemetry*; global IDs 1201–1220 cannot be proven because the export index lists 21 cycles and only 20 tabs exist.

**Member 2**

1. Accept or reject the nominal-1.25-Ah SOH definition and the B2 channel-treatment choice.
2. Build features only from `FEATURE_SAFE` columns; never use the leakage columns for same-cycle targets.
3. Use `soh_rated × 100` for B2 and never treat B2 0–1 values as percentages.
4. Keep B4 target-blocked and B5 `capacity_ground_truth = NO`; do not impute capacity.
5. Return predictions with the contract JSON, including quality flags and unsupported cases.
