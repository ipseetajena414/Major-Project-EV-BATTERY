# Member 1 — Reproducible Commands and Test Evidence

Interpreter used for every command: `/c/Users/KIIT0001/anaconda3/python.exe` (Python 3.13, pandas 2.2.x, pyarrow, numpy, scipy) on Windows 10, run from the repository root `C:\Users\KIIT0001\Desktop\MAJOR PROJ 2`.

Nothing here stages, commits or pushes. No command writes to `master_table.parquet`, `master_table.csv`, or any Member 2/3 artefact.

---

## 1. Build pipeline (execution order)

| # | Command | Output | Last run |
|---|---|---|---|
| 1 | `python AUDIT/member1_inventory.py` | `AUDIT/MEMBER1_SOURCE_INVENTORY.csv`, `AUDIT/MEMBER1_BASELINE.json` | PASS |
| 2 | `python AUDIT/member1_lfp_telemetry.py B1` | `AUDIT/candidates/telemetry_B1.parquet`, `telemetry_B1_summary.json` | PASS |
| 3 | `python AUDIT/member1_lfp_telemetry.py B2` | B2 artefacts | PASS |
| 4 | `python AUDIT/member1_lfp_telemetry.py B3` | B3 artefacts | PASS |
| 5 | `python AUDIT/member1_lfp_telemetry.py B4` | B4 artefacts | PASS |
| 6 | `python AUDIT/member1_lfp_telemetry.py B5` | B5 artefacts | PASS |
| 7 | `python AUDIT/member1_lfp_telemetry.py B6` | B6 artefacts | PASS |
| 8 | `python AUDIT/member1_postprocess_telemetry.py` | spike flags in the telemetry parquets; `telemetry_postprocess_summary.json` | PASS |
| 9 | `python AUDIT/member1_cycle_candidates.py` | `B{1,3,4,5}_candidate_cycle_labels.csv`, `B6_candidate_cycle_chronology.csv`, `B2_candidate_cycle_labels.csv`, `B2_channel_transition_mask.csv`, `MEMBER1_FEATURE_TARGET_MANIFEST.csv`, `MEMBER1_TARGET_DEFINITIONS.csv` | PASS |
| 10 | `python AUDIT/member1_master_candidate.py` | `master_table_candidate.parquet/.csv`, `_validation.json`, `_manifest.csv` | PASS |
| 11 | `python AUDIT/member1_coverage.py` | `AUDIT/MEMBER1_DATASET_COVERAGE.csv` | PASS |

Estimated wall time: inventory ~2 min, each telemetry tag 4–6 min (direct XML parsing), postprocess <1 min, candidates <1 min, candidate master <1 min, coverage <5 s.

B2 and B6 telemetry read the raw archives `DATA\B2_Discharge1C_Charge0.5C.zip` and `DATA\B6_Discharge0.5C_Charge2C.zip`. If those are absent the corresponding steps fail loudly with `missing`/`FileNotFoundError`; they do not fall back to derived data. `DATA/` is `.gitignore`d, so a fresh clone must restore the archives (see §5).

## 2. Test execution and results

Commands and exact results from this session:

```
python AUDIT/test_member1_pipeline.py
  Ran 36 tests in 3.796s    OK

python AUDIT/test_member1_audit.py
  Ran 18 tests in 5.378s    OK

python AUDIT/test_soh_recovery.py
  Ran 19 tests in 6.243s    OK

python AUDIT/test_lfp_b1_b5_inventory.py
  Ran 32 tests in 17.752s   OK

python AUDIT/test_lfp_b2_b6_followup.py
  Ran 27 tests in 1.646s    OK
```

**Total: 132 tests, 132 passed, 0 failed, 0 errors, 0 skipped.**

The new 36-test suite covers: canonical hash/row integrity (4), candidate schema/rows/duplicates/datasets/bounds/keys (10), B2 shape/invalid/channel/treatment/SOH/leakage (7), B1 coverage/bug/flags plus the 1201–1220 gap evidence (4), B3 missing IDs/channels (1), B4 blocked targets (1), B5 counts/interruption/charge-balance (3), B6 chronology (1), coverage table/telemetry summaries/target definitions/handoff docs/inventory (5).

## 3. Verification queries (read-only, executed)

```python
# canonical integrity
sha256sum master_table.parquet master_table.csv
# 6fa7475d...3b2bb   master_table.parquet
# 9e237926...29895b  master_table.csv
```

```python
import pandas as pd
c = pd.read_parquet("master_table_candidate.parquet")
len(c)                                   # 1_148_003
list(c.columns) == MASTER_SCHEMA         # True
c.duplicated(["dataset_id","cell_id","cycle_id","time_sec"]).sum()  # 0
```

```python
# B1 canonical defect evidence
canon = pd.read_parquet("master_table.parquet")
b1 = canon[canon.dataset_id == "LFP_MultiStat_B1"]
b1.cycle_id.unique(), b1.capacity_ah.unique(), b1.soh_pct.unique()
# array([2]), array([1.202223]), array([96.178])
```

```python
# B2 raw manifest parity and spot check
import json
json.load(open("AUDIT/candidates/B2_candidate_summary.json"))["raw_manifest"]
# {"workbooks": 20, "all_hashes_match": true, "all_sizes_match": true, ...}
```

## 4. Determinism notes

- Telemetry sampling is index-based (`np.linspace` over each sheet's charge/discharge rows), not random — reruns produce identical parquet content and hashes.
- `member1_master_candidate.py` sorts by `dataset_id, cell_id, cycle_id, time_sec` with `mergesort` before writing, so ordering is stable.
- The candidate CSV hash depends on pandas float formatting; the parquet hash is the authoritative artefact hash.

## 5. Pre-conditions for a clean rerun

1. Raw LFP archives present in `DATA/`: B1, B2, B3, B4, B5, B6 plus NASA/Na-ion. B2 must be `DATA/B2_Discharge1C_Charge0.5C.zip` (SHA-256 `b75c36d96abd2ae17858eda570dd6627ef06433bd88ac55faf73d688cbc10279`); B6 must be `DATA/B6_Discharge0.5C_Charge2C.zip` (SHA-256 `25fe5a2417d68334eed59667c162db6851e1970a2109e523caf91dd11484b97a`).
2. Audit zips present: `AUDIT/B1_LFP_Audit_Output.zip`, `AUDIT/B5.zip`, `AUDIT/Existing_Audits_Available_B3_B4 (1).zip`, `AUDIT/LFP_B6_Audit_Package (1).zip`, `DATA/B2_LFP_Member1_audit.zip`.
3. Canonical master present and matching the baseline hashes above.
4. `DATA/` is gitignored, so it is absent from a fresh clone. Restore the archives at their recorded SHA-256 values before rebuilding; the pipeline does not download or substitute data.
5. Do not run `member1_master_candidate.py` with the canonical files open in Excel/pandas — it only reads them, but a writer lock would abort the hash verification.
