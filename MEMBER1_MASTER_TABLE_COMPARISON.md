# Member 1 — Canonical vs Candidate Master Table Comparison

**Canonical master is untouched.** Hashes were re-verified after the candidate build:

| File | SHA-256 | Rows | Status |
|---|---|---|---|
| `master_table.parquet` | `6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb` | 515,270 | unchanged |
| `master_table.csv` | `9e23792615b2f08692944680d041957f776428b7563a90b214b77ee74629895b` | 515,270 | unchanged |
| `master_table_candidate.parquet` | `4b1ec1348d6b179eb103616fe30dc6de94fceec3e9273f6e33990fbb060feb21` | 1,148,003 | new, review-only |
| `master_table_candidate.csv` | `0ffe404bc931c08903f3c0ddf915ff19793f70f28be5054b5d200c30f8b50657` | 1,148,003 | new, review-only |
| `master_table_candidate_validation.json` | — | — | machine-readable validation |
| `master_table_candidate_manifest.csv` | — | — | per-dataset provenance |

The candidate is **not** promoted, renamed or substituted for the canonical table.

---

## 1. Per-dataset comparison

Canonical "cycles" are counted as unique `(dataset_id, cell_id, cycle_id)` composites because `cycle_id` resets per cell for NASA/Na-ion.

| Dataset | Canonical rows | Canonical cell-cycles | Candidate rows | Candidate cell-cycles | Δ cycles | Δ rows | Primary change |
|---|---:|---:|---:|---:|---:|---:|---|
| LFP_MultiStat_B1 | 4,789 | 1 | 145,901 | 1,460 | +1,459 | +141,112 | canonical had one sheet / mis-joined cycle‑50 label; candidate rebuilds 1,460 cycles from raw |
| LFP_MultiStat_B2 | 0 | 0 | 150,172 | 1,505 | +1,505 | +150,172 | previously absent; rebuilt from raw archive |
| LFP_MultiStat_B3 | 0 | 0 | 197,441 | 1,979 | +1,979 | +197,441 | previously absent; rebuilt from raw archive |
| LFP_MultiStat_B4 | 0 | 0 | 157,501 | 1,577 | +1,577 | +157,501 | telemetry only; capacity/SOH null (targets blocked) |
| LFP_MultiStat_B5 | 333,344 | 250 | 149,750 | 1,498 | +1,248 | −183,594 | 4→17 workbooks; canonical was dense sampling of 4 books, candidate is 100 pts/cycle across all 1,498 |
| LFP_MultiStat_B6 | 0 | 0 | 170,101 | 1,702 | +1,702 | +170,101 | previously absent; rebuilt from raw archive |
| NASA_Ames_PCoE | 135,436 | 468 | 135,436 | 468 | 0 | 0 | preserved byte-for-byte from canonical |
| Zenodo_IonTech_NaIon | 41,701 | 100 | 41,701 | 100 | 0 | 0 | preserved byte-for-byte from canonical |
| **Total** | **515,270** | **819** | **1,148,003** | **10,289** | **+9,470** | **+632,733** | |

Row-count change is not a coverage metric: the candidate uses a fixed 50 charge + 50 discharge samples per cycle, whereas the canonical table sampled `iloc[::3]` (B1) and `iloc[::5]` (B5) and therefore had uneven density.

## 2. Target availability comparison

| Dataset | Canonical capacity non-null | Candidate capacity non-null | Canonical SOH non-null | Candidate SOH non-null | Notes |
|---|---:|---:|---:|---:|---|
| B1 | 4,789 (100 %, wrong cycle) | 145,551 rows / 1,456 cycles | 4,789 | 145,551 | corrected labels |
| B2 | 0 | 150,100 rows / 1,502 cycles | 0 | 150,100 | new |
| B3 | 0 | 197,153 rows / 1,973 cycles | 0 | 197,153 | new |
| B4 | 0 | 0 | 0 | 0 | intentionally null |
| B5 | 333,344 | 149,750 rows / 1,497 cycles | 332,719 | 149,750 | conditional; not ground truth |
| B6 | 0 | 169,651 rows / 1,697 cycles | 0 | 169,651 | new |
| NASA | 135,436 | 135,436 | 135,436 | 135,436 | unchanged |
| Na-ion | 41,701 | 41,701 | 41,701 | 41,701 | unchanged |

## 3. Schema and integrity

| Check | Canonical | Candidate |
|---|---|---|
| Columns | 13 | 13 (identical names and order) |
| Duplicate `(dataset_id, cell_id, cycle_id, time_sec)` keys | — | 0 |
| `time_sec` nulls | — | 0 |
| Impossible voltage (<0 or >6 V) | — | 0 |
| Impossible temperature (<−50 or >100 °C) | — | 0 |
| SOH outside 0–150 % | — | 0 |
| Negative capacity | — | 0 |
| Candidate parquet re-read parity | — | true |

Observed candidate ranges: voltage 1.9991–4.347412 V; current −5.800386…5.800516 A; temperature 22.3503–42.0075 °C (NASA/Na-ion only); capacity 0.08777–2.03534 Ah; SOH 7.0216–101.7669 % (the 7.02 % minimum is B2 cycle 1411, an incomplete discharge already flagged invalid and excluded from modelling ranges).

## 4. What changed and why (source-level)

| Change | Justification |
|---|---|
| B1 telemetry rebuilt | Canonical ingested 1 of 1,460 sheets and applied a collapsed checkpoint label |
| B2/B3/B6 added | Raw archives accessible and audited; previously absent from canonical |
| B4 added as unlabelled telemetry | Raw accessible, 21 workbooks audited; no ground-truth target exists — not synthesised |
| B5 rebuilt across all workbooks | Canonical used 4 of 17 workbooks; candidate is complete |
| Sampling unified | 50 charge + 50 discharge per cycle for all LFP, deterministic and documented |
| Quality flags extended | New flags: `TEMP_UNMEASURED`, `CURRENT_UNIT_UNDERDETERMINED`, `CYCLE_ID_RECONSTRUCTED`, `OFFSET_UNDERDETERMINED`, `NO_AUDITED_LABEL`, `CAPACITY_GROUND_TRUTH_NO`, `CHANNEL_TRANSITION`, `INVALID_CYCLE`, `PARTIAL_DISCHARGE`, `NO_DISCHARGE_CAPACITY`, spike flags |
| Non-LFP rows preserved | NASA and Na-ion copied unchanged; no re-derivation |

## 5. Reproducibility statement

- The **NASA and Na-ion** portions are reproducible from data inside the repository.
- The **six LFP** portions require the raw MultiStat archives. All of them — including `B2_Discharge1C_Charge0.5C.zip` and `B6_Discharge0.5C_Charge2C.zip` — are now present in the gitignored `DATA/` folder, and the build scripts read from there. The candidate is therefore reproducible **in this checkout**.
- A **fresh clone** must restore `DATA/` from approved storage first, because Git excludes it by size policy. Full reproducibility from a bare clone is **not** claimed.
- Exact commands are in `MEMBER1_REPRODUCIBLE_COMMANDS.md`.

## 6. Promotion criteria (for the reviewer)

Formal reviewer sign-off is **pending**; no approval is claimed.

The candidate should only be promoted after:

1. `AUDIT/test_member1_pipeline.py` and the existing suite pass (see `MEMBER1_STATUS.json`).
2. Member 2 accepts the target definitions and the B2 channel treatment choice.
3. A reviewer signs off that B4 remains target-blocked and B5 remains `capacity_ground_truth = NO`.
4. The canonical hashes above are re-verified immediately before and after any promotion step.
