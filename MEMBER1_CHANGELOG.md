# Member 1 — Change Log

Session date: **2026-10-29**. Repository: `C:\Users\KIIT0001\Desktop\MAJOR PROJ 2`, branch `main`, HEAD `e36eb73` at session start; closure work lands on the Member 1 branch `member1-final-audit`.

## Repository state

- **No tracked file was modified or overwritten in place.** All Member 1 work is new files; the only pre-existing tracked file touched is `.gitignore` (candidate CSV exclusion).
- `master_table.parquet` and `master_table.csv` were read but never written. Their SHA-256 values were verified before and after every candidate build and match the baseline.
- No model artifact, scaler, manifest, inference interface, Member 2 API or Member 3 range calculator was touched.
- No Member 2 commit was amended, reverted or force-pushed.

## Added — pipeline scripts (`AUDIT/`)

| File | Purpose | Status |
|---|---|---|
| `member1_inventory.py` | full source inventory + canonical baseline (modified in-session to classify generic cycle tabs) | executed |
| `member1_lfp_telemetry.py` | direct-XLSX sampled telemetry for B1–B6 | executed for all six |
| `member1_postprocess_telemetry.py` | protocol-bound spike flags | executed |
| `member1_cycle_candidates.py` | B1/B2/B3/B4/B5 cycle-label candidates, B6 chronology, B2 channel mask and feature/target manifest | executed |
| `member1_master_candidate.py` | versioned candidate master assembly | executed |
| `member1_coverage.py` | 8-dataset coverage/status table | executed |
| `test_member1_pipeline.py` | 36 new end-to-end tests | executed, 36/36 pass |

Pre-existing audit scripts (`member1_audit.py`, `soh_recovery.py`, `make_matrices.py`, `lfp_b1_b5_inventory.py`, `lfp_b2_b6_matrices.py`, `lfp_b2_candidate.py`, `lfp_b6_chronology.py` and their tests) were left unchanged and re-run.

## Added — data artefacts

| File | Rows | Note |
|---|---:|---|
| `AUDIT/MEMBER1_SOURCE_INVENTORY.csv` | 8 | raw + derived source inventory with SHA-256 |
| `AUDIT/MEMBER1_BASELINE.json` | — | canonical hash/row baseline |
| `AUDIT/MEMBER1_SOURCE_TRACEABILITY.csv` | 8 | raw → audit → label → candidate chain |
| `AUDIT/MEMBER1_DATASET_COVERAGE.csv` | 8 | status/target status/counts per dataset |
| `AUDIT/MEMBER1_FEATURE_TARGET_MANIFEST.csv` | 255 | column role + leakage classification |
| `AUDIT/MEMBER1_TARGET_DEFINITIONS.csv` | 8 | target definition per dataset |
| `AUDIT/candidates/telemetry_B{1..6}.parquet` | 970,866 total | sampled telemetry, 13-col schema + provenance |
| `AUDIT/candidates/telemetry_B*_summary.json` | — | per-sheet match and label-match counts |
| `AUDIT/candidates/B1_candidate_cycle_labels.csv` | 1,460 | includes corrected cycle joins |
| `AUDIT/candidates/B2_candidate_cycle_labels.csv` | 1,506 | channel/provenance/leakage annotations |
| `AUDIT/candidates/B2_channel_transition_mask.csv` | 1,506 | per-cycle channel + treatment eligibility |
| `AUDIT/candidates/B3_candidate_cycle_labels.csv` | 1,979 | channels, quality tiers, missing IDs |
| `AUDIT/candidates/B4_candidate_cycle_labels.csv` | 1,577 | capacity/SOH null |
| `AUDIT/candidates/B5_candidate_cycle_labels.csv` | 1,498 | File List row excluded |
| `AUDIT/candidates/B6_candidate_cycle_chronology.csv` | 1,702 | reconstructed chronology |
| `AUDIT/candidates/member1_cycle_candidates_summary.json` | — | counts and defect evidence |
| `master_table_candidate.parquet` | 1,148,003 | SHA `4b1ec1348d6b179eb103616fe30dc6de94fceec3e9273f6e33990fbb060feb21` |
| `master_table_candidate.csv` | 1,148,003 | SHA `0ffe404bc931c08903f3c0ddf915ff19793f70f28be5054b5d200c30f8b50657` (not Git-tracked: 224 MB) |
| `master_table_candidate_validation.json` | — | schema/null/range/impossible-value checks |
| `master_table_candidate_manifest.csv` | 8 | per-dataset rows/cycles/source |
| `AUDIT/MEMBER1_B1_CYCLE1300_GAP_EVIDENCE.json` | — | raw-workbook evidence for the retained 1201–1220 gap |

## Added — documentation

| File | Contents |
|---|---|
| `MEMBER1_FINAL_DATA_PIPELINE_REPORT.md` | phase-by-phase report and decisions |
| `MEMBER1_DATA_QUALITY_REPORT.md` | per-dataset defects, limits, flag taxonomy |
| `MEMBER1_MASTER_TABLE_COMPARISON.md` | canonical vs candidate, promotion criteria |
| `MEMBER1_REPRODUCIBLE_COMMANDS.md` | exact commands and test evidence |
| `MEMBER1_STATUS.json` | machine-readable status of all deliverables |
| `MEMBER1_CHANGELOG.md` | this file |

Pre-existing documents `MEMBER1_MEMBER2_HANDOFF_CONTRACT.md` and `MEMBER2_MEMBER3_INTERFACE_DATA_CONTRACT.md` were written earlier in the session and are included in the deliverable set.

## Defects fixed in-session

1. B1 canonical cycle-label collapse (checkpoint `zip`) — repaired by `(source_file, worksheet)` joins.
2. B5 `File List` tab counted as a cycle — count corrected 1,499 → 1,498.
3. B6 generic cycle tabs undercounted (1,686 → 1,702) — inventory classifier fixed.
4. Candidate master helper column leaked into output schema — schema restored to 13 columns.
5. NASA/Na-ion cycle counts collapsed by per-cell `cycle_id` reset — composite key counting.
6. B2 cycle 998 (all-zero current) produced no phase samples — explicitly excluded.
7. B5/B6 renamed tabs were skipped by the Step2-only reader — reader extended; 0 unmatched sheets.

## Closure pass (2026-10-29)

1. **B2/B6 raw archives verified and relocated into the repo working tree.** `B2_Discharge1C_Charge0.5C.zip` (SHA `b75c36d9…c10279`) and `B6_Discharge0.5C_Charge2C.zip` (SHA `25fe5a24…84b97a`) were copied from `C:\Users\KIIT0001\Downloads` into `DATA/`, verified byte-identical, and every script path was switched to the repo-relative `DATA/` location. `DATA/` is `.gitignore`d, so the archives are not committed.
2. **B1 missing cycles 1201–1220 investigated.** `Cycle_1300.xlsx` contains 80 `Step2_Cycle*` tabs, 20 unlabelled ASCII cycle exports (`Sheet21`→`Sheet2`, 23–26 Sep 2025) and an export index listing 21 cycles. The exports evidently cover the gap but carry no cycle identifiers, capacity or SOH labels, and the index count disagrees with the tab count. Decision recorded in `AUDIT/MEMBER1_B1_CYCLE1300_GAP_EVIDENCE.json`: `RETAIN_GAP_NOT_INTEGRATED`. No cycles, timestamps, capacity or SOH values were invented; candidate master is unchanged. Regression test 36 asserts the gap stays documented and absent.
3. **Candidate master rebuilt from `DATA/` paths.** 1,148,003 rows, 13 columns, 0 duplicate composite keys. New hashes recorded above; two consecutive rebuilds produced identical bytes. Canonical master hashes unchanged.
4. **Coverage/quality/status/comparison/report documents updated** for the new hashes, `DATA/` source locations, the B1 gap finding and `review_signoff: PENDING_NO_FORMAL_REVIEWER`.

## Explicitly not done

- Candidate not promoted or renamed; sign-off is pending, not granted.
- B4 capacity/SOH not synthesised.
- B5 `capacity_ground_truth` not upgraded.
- No imputation of temperature or current units.
- Unlabelled B1 ASCII exports not integrated (identifier mapping unproven).
- Raw ZIP archives not committed to Git (gitignored `DATA/`, plus size policy).
- `master_table_candidate.csv` not committed (224 MB exceeds the 100 MB Git file limit).
