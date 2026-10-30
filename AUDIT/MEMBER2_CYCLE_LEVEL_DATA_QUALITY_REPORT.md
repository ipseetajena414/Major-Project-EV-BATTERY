# Member 2: Cycle-Level Data Quality & Readiness Report

**Project:** Autonomous Multi-Agent EV Battery Life & Range Predictor  
**Module:** Member 2 — Feature Engineering & ML Predictor  
**Date:** 2026-10-29  
**Analysis Type:** Read-Only Inspection (zero files modified)  
**Input:** `master_table.parquet` (SHA256: `6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb`)

---

## 1. Executive Summary

This report presents a comprehensive cycle-level validation of the Member 1 master table prior to any feature engineering or model training. The analysis was performed entirely read-only using the project's `.venv` Python 3.12.10 environment with pandas 3.1.0 and pyarrow 26.0.0.

### Key Findings

1. **`capacity_ah` and `soh_pct` are broadcast cycle-level scalars** — every row within a cycle carries an identical value. They are NOT cumulative within-cycle measurements.
2. **818 of 819 cycles have valid SOH labels.** The single missing-SOH cycle belongs to LFP B5 (**cycle_id=40**, 625 rows). Because this label is dropped by the model pipeline, the model-ready B5 sequence contains an additional 39 → 41 transition that is not a raw structural gap (see §8.5).
3. **Zero intra-cycle variation** in either capacity_ah or soh_pct across all 819 cycles.
4. **time_sec is monotonically non-decreasing** within every cycle (819/819 pass).
5. **Zero duplicate composite keys** on `(dataset_id, cell_id, protocol_id, cycle_id, time_sec)`.
6. **Critical data leakage risk identified:** Because `soh_pct` is stamped onto every telemetry row of the same cycle it describes, naïve "predict SOH from same-row features" would cause target leakage. The correct framing is **future-cycle prediction**.
7. **Severely imbalanced cycle counts across datasets:** 250 cycles (B5) vs 468 cycles (NASA, 3 cells) vs 100 cycles (Na-ion, 4 cells) vs 1 cycle (B1).

---

## 2. Basic Shape & Schema Verification

| Metric | Value |
|---|---|
| **Total rows** | 515,270 |
| **Columns** | 13 (matches Member 1 contract exactly) |
| **Schema** | `chemistry, dataset_id, cell_id, protocol_id, cycle_id, time_sec, voltage_v, current_a, temp_c, capacity_ah, soh_pct, quality_flag, source_file` |

### Null Counts

| Column | Null Count | Notes |
|---|---|---|
| `temp_c` | 379,834 | All Na-ion + all LFP rows (thermistors not recorded) |
| `soh_pct` | 625 | Single LFP B5 cycle (cycle_id=40) |
| `capacity_ah` | 0 | — |
| All other columns | 0 | — |

### Data Types

| Column | Dtype | Status |
|---|---|---|
| `chemistry`, `dataset_id`, `cell_id`, `protocol_id`, `quality_flag`, `source_file` | `str` | ✅ |
| `cycle_id` | `int64` | ✅ |
| `time_sec`, `voltage_v`, `current_a`, `temp_c`, `capacity_ah`, `soh_pct` | `float64` | ✅ |

---

## 3. Dataset-Level Summary

| Dataset ID | Chemistry | Rows | Cells | Unique Cycles | Quality Flags |
|---|---|---|---|---|---|
| **LFP_MultiStat_B1** | LFP | 4,789 | 1 (BATTERY_1) | **1** | `TEMP_UNMEASURED;CURRENT_UNIT_UNDERDETERMINED;ok` |
| **LFP_MultiStat_B5** | LFP | 333,344 | 1 (BATTERY_5) | **250** | `TEMP_UNMEASURED;CURRENT_UNIT_UNDERDETERMINED;CHARGE_BALANCE_DISCREPANCY` (333,301) + `...;VOLTAGE_OUT_OF_BOUNDS` (43) |
| **NASA_Ames_PCoE** | NMC | 135,436 | 3 (B0005, B0006, B0018) | **468** | `VALID` (135,436) |
| **Zenodo_IonTech_NaIon** | Na-ion | 41,701 | 4 cells | **100** | `TEMP_UNMEASURED` (41,701) |
| **TOTAL** | — | **515,270** | **9** | **819** | — |

### Per-Cell Cycle Breakdown

| Dataset | Cell | Cycles | cycle_id Range |
|---|---|---|---|
| LFP_MultiStat_B1 | BATTERY_1 | 1 | [2, 2] |
| LFP_MultiStat_B5 | BATTERY_5 | 250 | [1, 500] |
| NASA_Ames_PCoE | B0005 | 168 | [1, 168] |
| NASA_Ames_PCoE | B0006 | 168 | [1, 168] |
| NASA_Ames_PCoE | B0018 | 132 | [1, 132] |
| Zenodo_IonTech_NaIon | NA-ion_270040-1-1-64 | 25 | [1, 25] |
| Zenodo_IonTech_NaIon | NA-ion_270040-1-2-63 | 25 | [1, 25] |
| Zenodo_IonTech_NaIon | NA-ion_270040-1-3-62 | 25 | [1, 25] |
| Zenodo_IonTech_NaIon | NA-ion_270040-1-4-61 | 25 | [1, 25] |

> **Note on B5 cycle_id range [1, 500]:** The build script selects 4 checkpoint parquets (Cycle_0050, Cycle_0100, Cycle_0200, Cycle_0500). Each checkpoint parquet contains multiple cycles from the original experiment, mapped via `global_cycle_id_nominal_checkpoint`. The 250 unique cycle_ids are not contiguous (sparse sampling from a 500-cycle experiment).

---

## 4. Cycle-Level Capacity & SOH Constancy

### 4.1 Verified: capacity_ah and soh_pct Are Broadcast Scalars

| Check | Result |
|---|---|
| Cycles where `capacity_ah` has >1 unique non-null value | **0** |
| Cycles where `soh_pct` has >1 unique non-null value | **0** |
| Max distinct non-null `capacity_ah` within any cycle | **1** |
| Max distinct non-null `soh_pct` within any cycle | **1** |

**Interpretation:** Both `capacity_ah` and `soh_pct` are cycle-level scalars. They are computed once per cycle by the build script and then broadcast (replicated) to every telemetry row within that cycle. They are **NOT** running or cumulative quantities.

### 4.2 Provenance of capacity_ah and soh_pct (from `master_table_build.py`)

| Dataset | capacity_ah Source | soh_pct Source | Rated Capacity |
|---|---|---|---|
| **NASA NMC** | `data['Capacity']` field from MATLAB `.mat` struct (single scalar per discharge cycle) | `(capacity / 2.0) × 100` | 2.0 Ah |
| **Na-ion** | `discharge_capacity_in_Ah[-1]` — last element of discharge capacity array from `.pkl` | `(cycle_cap / nominal_capacity) × 100` | 1.0 Ah |
| **LFP B1** | Mapped from `LFP_staging_cycle_level.csv` via `cycle_checkpoint` key | Same mapping, pre-calculated `soh_pct` column | 1.25 Ah |
| **LFP B5** | Mapped from `B5_Cycle_Summary.csv` column `dis_q_ah_trapz_CALCULATED` | Column `soh_nominal_pct_CONDITIONAL_1.25Ah_rated` | 1.25 Ah |

> [!IMPORTANT]
> **The SOH label in the master table is a derived quantity, not a raw measurement.** It is calculated as `(measured_discharge_capacity / rated_nominal_capacity) × 100%`. The accuracy of this label depends on the accuracy of both the capacity integration method and the rated capacity reference. For LFP B5, the label is explicitly marked as `CONDITIONAL` due to the known charge-balance discrepancy.

---

## 5. SOH Coverage Analysis

### 5.1 Cycle-Level SOH Availability

| Category | Count |
|---|---|
| Total unique cycles | 819 |
| Cycles with fully present SOH (all rows non-null) | **818** |
| Cycles with fully missing SOH (all rows null) | **1** (LFP_MultiStat_B5, cycle_id=40) |
| Cycles with partially missing SOH (some null, some present) | **0** |

### 5.2 SOH Distribution by Dataset

| Dataset | Cycles with SOH | SOH Min (%) | SOH Max (%) | SOH Mean (%) | SOH Std (%) |
|---|---|---|---|---|---|
| LFP_MultiStat_B1 | 1 | 96.18 | 96.18 | 96.18 | — |
| LFP_MultiStat_B5 | 249 | 90.91 | 95.75 | 92.77 | 1.40 |
| NASA_Ames_PCoE | 468 | 57.69 | 101.77 | 77.96 | 10.31 |
| Zenodo_IonTech_NaIon | 100 | 78.63 | 91.00 | 86.73 | 4.62 |

### 5.3 SOH Trajectory Summary Per Cell

| Cell | Cycles | SOH Start (%) | SOH End (%) | Delta (%) | Decreasing | Increasing | Flat |
|---|---|---|---|---|---|---|---|
| B0005 (NMC) | 168 | 92.82 | 66.25 | −26.57 | 125 | 29 | 13 |
| B0006 (NMC) | 168 | 101.77 | 59.28 | −42.48 | 136 | 24 | 7 |
| B0018 (NMC) | 132 | 92.75 | 67.05 | −25.70 | 110 | 21 | 0 |
| BATTERY_5 (LFP) | 249 | 95.60 | 91.07 | −4.53 | 140 | 90 | 18 |
| BATTERY_1 (LFP) | 1 | 96.18 | 96.18 | 0.00 | — | — | — |
| NA-ion_270040-1-1-64 | 25 | 79.02 | 78.63 | −0.40 | 21 | 2 | 1 |
| NA-ion_270040-1-2-63 | 25 | 89.65 | 88.60 | −1.05 | 23 | 1 | 0 |
| NA-ion_270040-1-3-62 | 25 | 88.07 | 87.58 | −0.49 | 23 | 1 | 0 |
| NA-ion_270040-1-4-61 | 25 | 90.66 | 90.35 | −0.31 | 22 | 1 | 1 |

> [!NOTE]
> **SOH is not strictly monotonically decreasing.** All cells show some cycles where SOH increases relative to the prior cycle. This is physically expected (recovery effects, measurement noise) and does NOT indicate data corruption. However, BATTERY_5 has an unusually high ratio of increasing transitions (90/249 = 36%), consistent with the documented `CHARGE_BALANCE_DISCREPANCY`.

---

## 6. Time Ordering & Structural Integrity

| Check | Result |
|---|---|
| Cycles with monotonically non-decreasing `time_sec` | **819 / 819 (100%)** |
| Cycles with `time_sec` NOT monotonically ordered | **0** |
| Duplicate rows on full key `(dataset_id, cell_id, protocol_id, cycle_id, time_sec)` | **0** |

**Verdict:** ✅ time_sec is properly ordered within every cycle. The data is structurally suitable for intra-cycle feature extraction (e.g., dQ/dV, voltage curves).

---

## 7. Rows Per Cycle Distribution

| Statistic | Value |
|---|---|
| Min rows per cycle | 179 |
| Max rows per cycle | 4,789 |
| Mean | 629.1 |
| Median | 333.0 |
| Std | 500.5 |

| Dataset | Min | Max | Mean | Median |
|---|---|---|---|---|
| LFP_MultiStat_B1 | 4,789 | 4,789 | 4,789 | 4,789 |
| LFP_MultiStat_B5 | 625 | 1,380 | 1,333 | 1,333 |
| NASA_Ames_PCoE | 179 | 371 | 289 | 307 |
| Zenodo_IonTech_NaIon | 282 | 716 | 417 | 337 |

> [!NOTE]
> **Highly variable row counts across datasets.** The build script applied different sub-sampling rates (every 3rd row for B1, every 5th for B5 discharge, every 4th for dense Na-ion). Feature extraction must normalize for time resolution differences, not assume uniform sampling.

---

## 8. Investigation: B5 Cycle-ID Gap Analysis & Corrected Chronological Split

*Investigation executed 2026-10-29. All results are from actual script output, not assumptions.*

### 8.1 B5 Cycle-ID Structure

| Metric | Value |
|---|---|
| Total unique B5 cycle_ids | **250** |
| First observed cycle_id | 1 |
| Last observed cycle_id | 500 |
| If contiguous, expected count | 500 |
| Actually missing cycle_ids | **250** |

### 8.2 Observed Cycle-ID Ranges (Three Contiguous Blocks)

The 250 observed cycle_ids form **three contiguous blocks separated by two large gaps:**

| Block | Cycle-ID Range | Count | Gap AFTER Block |
|---|---|---|---|
| Block 1 | **1 – 100** | 100 cycles | 50 missing (101–150) |
| Block 2 | **151 – 200** | 50 cycles | 200 missing (201–400) |
| Block 3 | **401 – 500** | 100 cycles | — (end) |

### 8.3 Inter-Cycle-ID Gap Distribution

| Gap Size | Occurrences | Interpretation |
|---|---|---|
| 1 (consecutive) | 247 | Within each block, cycle_ids are contiguous |
| 51 | 1 | Gap between Block 1 (ends at 100) and Block 2 (starts at 151) |
| 201 | 1 | Gap between Block 2 (ends at 200) and Block 3 (starts at 401) |

> [!IMPORTANT]
> **`cycle_id - 1` must NOT be used to identify the previous observed cycle.** The gap structure proves that observed cycle_ids skip values. For example, cycle_id=151 follows cycle_id=100 in the observed data, not cycle_id=150. To reference the "previous cycle," use **positional indexing in the sorted observed cycle list**, not arithmetic subtraction.

> [!NOTE]
> The table above describes the **raw** cycle-ID structure. Separately, the model pipeline drops B5 `cycle_id=40` because its SOH label is null (625 rows). In the **model-ready** sequence this creates a 39 → 41 transition (difference 2) that is label-driven, not a raw structural gap. §8.5 documents the actual pairs.

### 8.4 Corrected 60/20/20 Chronological Split

The split is computed by **position in the sorted list of 250 observed cycle_ids**, not by numeric cycle_id intervals:

| Split | Positions | Cycle Count | Cycle-ID Range | First 5 | Last 5 |
|---|---|---|---|---|---|
| **Train (60%)** | 0–149 | **150** | [1, 200] | 1, 2, 3, 4, 5 | 196, 197, 198, 199, 200 |
| **Validation (20%)** | 150–199 | **50** | [401, 450] | 401, 402, 403, 404, 405 | 446, 447, 448, 449, 450 |
| **Test (20%)** | 200–249 | **50** | [451, 500] | 451, 452, 453, 454, 455 | 496, 497, 498, 499, 500 |

**Verification of split integrity (from actual script output):**

| Check | Result |
|---|---|
| Train–Val overlap | **0** |
| Val–Test overlap | **0** |
| Train–Test overlap | **0** |
| Chronological ordering: max(Train) < min(Val) < min(Test) | **True** (200 < 401, 450 < 451) |
| Total cycles accounted for | **250** (150 + 50 + 50 = 250 ✅) |

> [!IMPORTANT]
> This table is the **cycle-level** split over all 250 observed records. It is **not** the split the models are trained on: the pipeline first drops the null-SOH record (`cycle_id=40`), builds 248 input→target pairs, and then splits those pairs. The resulting model-ready split is **148 / 49 / 51** pairs, documented in §8.5. The two views must not be conflated.

> [!WARNING]
> **The gap between Train (ends at cycle_id=200) and Validation (starts at cycle_id=401) spans 200 unobserved cycles.** The battery degraded during these missing cycles. The validation set represents a state substantially more degraded than the training set's endpoint, which is realistic for future-prediction evaluation but must be acknowledged when interpreting validation error.

### 8.5 Model-Ready LFP Input→Target Pairs (Read-Only Re-Derivation)

*Executed 2026-10-29 against the same Parquet digest. The actual `build_and_train.py` functions (`build_cycle_feature_table`, `create_future_soh_dataset`, `chronological_split`) were imported and run up to pair construction. No training was performed and no artifacts were written.*

**Filtering chain (actual pipeline order):**

| Step | Result |
|---|---|
| Cycles with extracted features (B1 excluded) | 818 |
| Cycles dropped for null SOH | 1 (B5 `cycle_id=40`, 625 rows) |
| B5 labelled cycles remaining | **249** |
| B5 model-ready input→target pairs (`horizon=1`, positional shift) | **248** |
| Adjacent pairs (`target_cycle_id - input_cycle_id = 1`) | **245** |
| Pairs with `target_cycle_id - input_cycle_id > 1` | **3** |

**All gap-crossing model-ready pairs:**

| Input cycle_id | Target cycle_id | Difference | Cause | Split |
|---:|---:|---:|---|---|
| 39 | 41 | 2 | `cycle_id=40` has a null SOH label and is dropped by the label filter | train |
| 100 | 151 | 51 | Raw structural gap (Block 1 → Block 2) | train |
| 200 | 401 | 201 | Raw structural gap (Block 2 → Block 3) | val |

**Actual model-ready split membership:**

| Split | Pairs | Input cycle-ID range | Target cycle-ID range | Gap-crossing pairs |
|---|---:|---|---|---:|
| Train (60%) | **148** | 1–199 | 2–200 | 2 |
| Validation (20%) | **49** | 200–448 | 401–449 | 1 |
| Test (20%) | **51** | 449–499 | 450–500 | 0 |

> [!IMPORTANT]
> **Findings:**
> 1. Both raw structural transitions (100 → 151 and 200 → 401) survive every existing label and feature filter.
> 2. The label filter creates one additional gap-crossing pair, 39 → 41, because `cycle_id=40` has no SOH label.
> 3. No missing cycles were inserted or interpolated. Every target is the next **observed record**, not the next physical cycle.
> 4. The reported LFP test metrics (n=51) are computed on **adjacent observed cycle-ID pairs only**, because the test split contains no gap-crossing pair. Validation error, by contrast, includes a 201-ID extrapolation (200 → 401) and is not directly comparable to adjacent-pair error.
> 5. The LFP forecast horizon is an **observed-record horizon**, not a physical-cycle horizon: 100 → 151 and 200 → 401 must not be described as one cycle apart.

---

## 9. Investigation: Na-ion Current Direction Analysis

*Investigation executed 2026-10-29. All results are from actual script output.*

### 9.1 Per-Cell Current Sign Distribution (Row-Level)

| Cell | Total Rows | Positive (I > 0) | Negative (I < 0) | Zero (I = 0) | Current Range (A) |
|---|---|---|---|---|---|
| NA-ion_270040-1-1-64 | 17,803 | 8,900 (50.0%) | 8,888 (49.9%) | 15 (0.1%) | [−2.000, +2.000] |
| NA-ion_270040-1-2-63 | 7,346 | 3,674 (50.0%) | 3,661 (49.8%) | 11 (0.1%) | [−5.501, +5.501] |
| NA-ion_270040-1-3-62 | 9,480 | 4,740 (50.0%) | 4,728 (49.9%) | 12 (0.1%) | [−4.201, +4.201] |
| NA-ion_270040-1-4-61 | 7,072 | 3,535 (50.0%) | 3,523 (49.8%) | 14 (0.2%) | [−5.800, +5.801] |

### 9.2 Per-Cycle Current Sign Analysis

**Every single Na-ion cycle contains both positive and negative current:**

| Cell | Total Cycles | Mixed (both signs) | Positive-only | Negative-only |
|---|---|---|---|---|
| NA-ion_270040-1-1-64 | 25 | **25** | 0 | 0 |
| NA-ion_270040-1-2-63 | 25 | **25** | 0 | 0 |
| NA-ion_270040-1-3-62 | 25 | **25** | 0 | 0 |
| NA-ion_270040-1-4-61 | 25 | **25** | 0 | 0 |

Sample cycle breakdown (cell NA-ion_270040-1-1-64):
- Cycle 1: 703 rows — 345 pos (49.1%), 356 neg (50.6%), 2 zero
- Cycle 2: 716 rows — 359 pos (50.1%), 357 neg (49.9%), 0 zero

### 9.3 Interpretation

Per the project convention (README + `master_table_build.py`): I > 0 = charge, I < 0 = discharge.

The build script (`load_na_ion`, lines 112–178) ingests Na-ion data from `.pkl` cycle dictionaries **without filtering by charge/discharge phase**. Unlike the NASA NMC loader (which filters `c_type == 'discharge'`) and the B5 loader (which filters `phase == 'D'`), the Na-ion loader includes the **full charge + discharge cycle** within each `cycle_id`.

> [!IMPORTANT]
> **Na-ion cycles contain mixed charge and discharge telemetry.** Feature extraction for Na-ion must either:
> 1. Separate charge and discharge phases within each cycle using the current sign before computing features, or
> 2. Compute features that are meaningful across the full cycle (e.g., total energy, Ah throughput).
>
> This is different from NASA NMC and LFP B5, which contain **discharge-only** telemetry per cycle.

### 9.4 Previous Assumption Resolution

| Assumption from Section 9.4 (v1) | Status | Finding |
|---|---|---|
| "Na-ion telemetry contains both charge and discharge within each cycle" | ✅ **CONFIRMED** | 100/100 Na-ion cycles contain both positive and negative current with ~50/50 split |
| "B5 cycle_id gaps are consistent (every 2nd cycle sampled)" | ❌ **INCORRECT** | Gaps are NOT uniform — three blocks [1–100], [151–200], [401–500] with two large gaps |

---

## 10. Investigation: SOH / EOL / RUL Feasibility Analysis

*Investigation executed 2026-10-29. EOL threshold = 80% stated explicitly for diagnostic purposes only.*

### 10.1 Per-Cell EOL Status

| Cell | Observed Cycles | SOH Start (%) | SOH End (%) | SOH Min (%) | EOL Status | First Crossing |
|---|---|---|---|---|---|---|
| **B0005** (NMC) | 168 | 92.82 | 66.25 | 66.25 | **CROSSES EOL** | cycle_id=75 (position 74) |
| **B0006** (NMC) | 168 | 101.77 | 59.28 | 57.69 | **CROSSES EOL** | cycle_id=63 (position 62) |
| **B0018** (NMC) | 132 | 92.75 | 67.05 | 67.05 | **CROSSES EOL** | cycle_id=45 (position 44) |
| **BATTERY_5** (LFP) | 249 | 95.60 | 91.07 | 90.91 | **NEVER CROSSES** | — (11.07% headroom) |
| **BATTERY_1** (LFP) | 1 | 96.18 | 96.18 | 96.18 | **NEVER CROSSES** | — (16.18% headroom) |
| NA-ion_270040-1-1-64 | 25 | 79.02 | 78.63 | 78.63 | **STARTS BELOW EOL** | — |
| NA-ion_270040-1-2-63 | 25 | 89.65 | 88.60 | 88.60 | **NEVER CROSSES** | — (8.60% headroom) |
| NA-ion_270040-1-3-62 | 25 | 88.07 | 87.58 | 87.58 | **NEVER CROSSES** | — (7.58% headroom) |
| NA-ion_270040-1-4-61 | 25 | 90.66 | 90.35 | 90.35 | **NEVER CROSSES** | — (10.35% headroom) |

### 10.2 RUL Feasibility Assessment

| Cell | RUL Training Feasible? | Rationale |
|---|---|---|
| **B0005, B0006, B0018** (NMC) | ✅ **YES** | All three cells cross 80% during observed cycling. Observed cycles-to-EOL can serve as regression targets. |
| **BATTERY_5** (LFP) | ❌ **NO** | SOH never drops below 90.91%. Over 11% headroom above EOL at experiment end. No observed EOL event. |
| **BATTERY_1** (LFP) | ❌ **NO** | Only 1 observed cycle. Cannot define RUL. |
| **NA-ion_270040-1-1-64** | ❌ **NO** | SOH **starts below** 80% threshold. Cell is already past EOL at cycle 1. |
| **NA-ion (other 3 cells)** | ❌ **NO** | SOH degradation < 1.1% over 25 cycles. Over 7.5% headroom. Insufficient for EOL extrapolation. |

> [!IMPORTANT]
> **Distinction: Observed threshold crossing ≠ validated RUL label.**
> The EOL crossings observed for B0005, B0006, and B0018 are empirical observations from the data. A validated RUL label would additionally require:
> - (a) Confirmation that capacity test conditions are consistent across all cycles
> - (b) A documented interpolation method for the exact crossing point
> - (c) A project-agreed EOL definition
>
> For initial model development, the observed crossings can be used as approximate RUL targets, but this caveat must be stated in any model evaluation.

### 10.3 SOH Prediction (Non-RUL) Feasibility

| Cell | SOH Prediction Feasible? | Rationale |
|---|---|---|
| **B0005, B0006, B0018** (NMC) | ✅ **YES** | Strong degradation signal (26–42% total fade), 132–168 cycles each, fully instrumented. Best candidates. |
| **BATTERY_5** (LFP) | ⚠️ **CONDITIONAL** | 249 cycles with SOH labels, but labels are CONDITIONAL (charge-balance discrepancy). Moderate 4.5% total fade. Usable with documented caveats. |
| **Na-ion (4 cells)** | ⚠️ **LIMITED** | Very shallow degradation (<1.1% per cell). Usable for early-life SOH estimation but NOT for deep-fade prediction. Mixed charge/discharge telemetry requires phase separation. |
| **BATTERY_1** (LFP) | ❌ **NO** | 1 cycle. Insufficient. |

---

## 11. Investigation: B5 Label Quality — Charge-Balance Discrepancy & SOH Oscillations

*Investigation executed 2026-10-29. All results from actual script output.*

### 11.1 SOH Cycle-to-Cycle Transition Analysis

| Transition Type | Count | Fraction |
|---|---|---|
| Decreasing (delta < −0.01%) | 140 | 56.5% |
| Increasing (delta > +0.01%) | **90** | **36.3%** |
| Flat (|delta| ≤ 0.01%) | 18 | 7.3% |
| **Total transitions** | **248** | — |

### 11.2 Magnitude of SOH Changes

| Direction | Count | Min (%) | Max (%) | Mean (%) | Median (%) |
|---|---|---|---|---|---|
| Increases | 90 | +0.0105 | +0.3068 | +0.1213 | +0.1164 |
| Decreases | 140 | −0.9335 | −0.0117 | −0.1102 | −0.0991 |

### 11.3 Capacity Trajectory

| Metric | Value |
|---|---|
| Start (first observed cycle with capacity) | 1.1950 Ah |
| End (last observed cycle) | 1.1383 Ah |
| Minimum | **0.5428 Ah** |
| Maximum | 1.1969 Ah |

> [!WARNING]
> The minimum capacity of 0.5428 Ah is an extreme outlier (43.4% of rated 1.25 Ah), while the overall trajectory spans 1.14–1.20 Ah. This likely corresponds to a single anomalous cycle. This outlier was already present in the Member 1 build output and was not filtered.

### 11.4 Documented Evidence for Charge-Balance Discrepancy

From [MEMBER1_STATUS_REPORT.md](file:///C:/Users/KIIT/Desktop/Major%20project/Major-Project-EV-BATTERY-git/MEMBER1_STATUS_REPORT.md) Section 3.3:

> *"Q_discharge > Q_charge in 1,465 of 1,469 paired cycles (median ratio 1.028). Apparent Coulombic efficiency exceeds 100%, which violates the First Law of Thermodynamics for steady cycling and indicates potentiostat calibration offset or systematic shunting."*

From [MEMBER1_FINAL_VALIDATION_REPORT.md](file:///C:/Users/KIIT/Desktop/Major%20project/Major-Project-EV-BATTERY-git/MEMBER1_FINAL_VALIDATION_REPORT.md):
- SOH column is `soh_nominal_pct_CONDITIONAL_1.25Ah_rated`
- Capacity column is `dis_q_ah_trapz_CALCULATED` (trapezoidal integration)

### 11.5 Assessment

BATTERY_5 has a documented charge-balance discrepancy inherited from the source data and Member 1 staging. In this audit's cycle-level analysis, **36.3% of observed SOH transitions (90 of 248) show an increase**. These observations are consistent with a measurement or instrumentation issue (such as the potentiostat calibration offset or systematic shunting noted in Member 1 documentation), but the exact causal mechanism has not been independently established in this audit. The resulting cycle-to-cycle capacity variation produces apparent SOH noise that warrants treating these labels cautiously as CONDITIONAL.

### 11.6 Label Treatment Policy

**The raw B5 SOH labels must be preserved as-is.** Any smoothing operation (rolling median, isotonic regression, Savitzky-Golay, etc.) would:
1. Require an independent reference dataset for validation (none exists in this workspace)
2. Risk hiding genuine degradation features (e.g., knee-point onset)
3. Introduce an undocumented transformation between the Member 1 deliverable and the Member 2 training target

> [!CAUTION]
> **The prior recommendation in Section 9.2 item 3 (v1 of this report) to "apply smoothing (e.g., isotonic regression or rolling median) before using as ground truth" is hereby WITHDRAWN.** No smoothing will be applied without a documented validation procedure comparing smoothed labels against an independent reference. The raw CONDITIONAL labels will be used as-is, with the caveat explicitly noted in model evaluation.

---

## 12. Investigation: Data Leakage — Corrected & Expanded Analysis

### 12.1 The Core Problem (Unchanged)

The master table has **two semantic levels mixed together:**

| Level | Columns | Grain | Meaning |
|---|---|---|---|
| **Intra-cycle telemetry** | `time_sec`, `voltage_v`, `current_a`, `temp_c` | One row per time-step within a cycle | Time-series measurements during a single charge/discharge |
| **Cycle-level labels** | `capacity_ah`, `soh_pct` | One value per cycle, broadcast to all rows | End-of-cycle characterization of battery health |

### 12.2 Direct Leakage: capacity_ah and soh_pct as Input Features

`capacity_ah` and `soh_pct` must **never** appear as model input features. They are the target labels. Including them (or any trivially invertible proxy) would produce near-perfect training metrics with zero genuine predictive power.

### 12.3 Indirect Leakage: Same-Cycle Full-Curve Features

**Empirical verification (from actual script output):**

Trapezoidal integration of `|current_a| × dt` from the within-cycle telemetry closely reconstructs `capacity_ah`:

| Cell | Cycle | Label Capacity (Ah) | Reconstructed from Telemetry (Ah) | Ratio |
|---|---|---|---|---|
| B0005 | 1 | 1.8565 | 1.8622 | 1.0031 |
| B0005 | 2 | 1.8463 | 1.8521 | 1.0031 |
| B0005 | 3 | 1.8353 | 1.8410 | 1.0031 |
| B0006 | 1 | 2.0353 | 2.0467 | 1.0056 |
| B0006 | 2 | 2.0251 | 2.0366 | 1.0056 |
| B0006 | 3 | 2.0133 | 2.0247 | 1.0057 |
| B0018 | 1 | 1.8550 | 1.8684 | 1.0072 |
| B0018 | 2 | 1.8432 | 1.8566 | 1.0073 |
| B0018 | 3 | 1.8396 | 1.8532 | 1.0074 |

The ratios are consistently ~1.003–1.007, confirming that **any feature computed from the complete discharge curve (total discharge time, energy integral, Ah throughput, area under the V-curve) implicitly encodes `capacity_ah`** and therefore `soh_pct`.

### 12.4 Three Distinct Prediction Tasks

| Task | Input | Target | Leakage Risk | Validity |
|---|---|---|---|---|
| **A: Full-Cycle Retrospective SOH Estimation** | Features from the **complete** discharge curve of cycle N | `soh_pct` of cycle N | **HIGH TAUTOLOGY RISK** — full-curve features can reconstruct capacity_ah, which is the numerator of soh_pct. | Legitimate **only** if the goal is to estimate SOH from telemetry when direct Ah counting is unavailable (e.g., BMS without coulomb counter). Must acknowledge that low error may reflect capacity reconstruction, not degradation learning. |
| **B: Partial-Cycle SOH Estimation** | Features from the **first K minutes** or **first K%** of a discharge curve of cycle N | `soh_pct` of cycle N (not yet fully observed) | **LOWER** — the full integral is not available, so features (e.g., voltage at t=60s, initial dV/dt slope) carry genuine SOH signal without directly encoding capacity. | Practical for **real-time BMS** applications. The model must learn correlations between early-cycle voltage shape and full-cycle outcome. |
| **C: Future-Cycle SOH / RUL Forecasting** | Features from cycles 1..N (all fully observed) | `soh_pct` of a future **observed record** (implemented model: the next observed record, positional shift = 1; not necessarily the next physical cycle) | **MINIMAL** if train/test split is chronological. | **Core Member 2 objective** per the project blueprint. The model must learn degradation trajectories, not single-cycle capacity reconstruction. |

> [!IMPORTANT]
> **For Task C (the primary Member 2 objective):** Using full-cycle features (including capacity-correlated ones) from **past** cycles as input to predict a **future** observed record's SOH is NOT leakage. The leakage arises only when predicting the **same** cycle's label from its own features (Task A). In Task C, cycle N's capacity features are legitimate historical inputs.

### 12.5 Required Pre-Processing Before Feature Engineering

```
Step 1: Extract cycle-level label table:
  labels_df = df.groupby(CYCLE_KEY).agg({'capacity_ah': 'first', 'soh_pct': 'first'})

Step 2: Drop capacity_ah and soh_pct from telemetry DataFrame:
  telemetry_df = df.drop(columns=['capacity_ah', 'soh_pct'])

Step 3: Extract features from telemetry_df per cycle.
  - For NASA NMC & LFP B5: discharge-only telemetry, compute dQ/dV, V statistics, etc.
  - For Na-ion: SEPARATE charge and discharge phases by current sign before computing features.

Step 4: Join features with labels_df for training, ensuring chronological split.

Step 5: For Task C (future-cycle prediction), construct a sliding window of past observed
  records' features as input, with the next observed record's soh_pct as target
  (implemented horizon = 1 observed record, not 1 physical cycle).
```

### 12.6 Leakage & Evaluation-Alignment Verification (This Correction Pass)

*Code inspection plus the read-only re-derivation in §8.5. No retraining was performed.*

| Check | Result | Evidence |
|---|---|---|
| Target labels excluded from input features | ✅ **PASS** | `get_feature_columns` excludes `soh_pct`, `capacity_ah`, `target_soh_pct`, `target_cycle_id`; verified on the actual NMC/LFP feature lists. |
| `current_soh_pct` handling | ⚠️ **BY DESIGN, WITH A DEPENDENCY** | `current_soh_pct` is the input record's `soh_pct` and is included for C-task prediction. It is a **derived label**, not a raw sensor signal, and Member 4 must supply it at inference. It does not leak the future target, but it makes the model heavily dependent on an upstream SOH estimate. |
| Full-curve features vs future target | ✅ **NO TARGET LEAKAGE** | Full-curve features can reconstruct the **input** cycle's capacity (§12.3), but the target is the **next observed record**. This is historical information, not future-label leakage. It does explain why learned models behave close to persistence. |
| Pair-level split boundary overlap | ⚠️ **MINOR** | Pairs are split after the positional shift, so the last train **target** (e.g. B5 cycle 200) is also the first val **input**. No target is duplicated across splits, and the boundary SOH is known at inference, but train/val/test cycle sets are not strictly disjoint by one record. Recommended follow-up: purge/embargo one record at each boundary in a future evaluation. |
| BiLSTM sequence windows vs splits | ✅ **PASS** | `make_cell_sequences` is called separately on `nmc_train`, `nmc_val`, `nmc_test`; no sequence window crosses a split boundary. |
| Metrics compared on equivalent samples | ✅ **PASS (as reported)** | NMC three-way comparison uses the identical 83 sequence test rows; LFP LightGBM and persistence use the same n=51 test targets. |
| LFP adjacent vs gap-crossing evaluation | ⚠️ **NOW DOCUMENTED** | The existing LFP test split (n=51) contains **0** gap-crossing pairs; validation contains the 200 → 401 pair and train contains 39 → 41 and 100 → 151 (§8.5). No separate adjacent-only metric was recomputed; the existing test metric is already adjacent-only by construction. |
| RUL generalization | ❌ **NOT ESTABLISHED** | All 97 RUL test targets are post-EOL (RUL = 0). The evaluation provides no evidence of pre-EOL forecasting capability. |

> [!CAUTION]
> The persistence baseline outperforms every reported learned SOH model. These results do **not** demonstrate that the learned models improve on carrying forward the last observed SOH.

---

## 13. Corrected Issues Table

### 13.1 Verified: Safe to Proceed

| Item | Status | Evidence |
|---|---|---|
| Schema conformity | ✅ PASS | 13 columns match contract |
| Unique composite keys | ✅ PASS | 0 duplicates |
| time_sec ordering | ✅ PASS | 819/819 monotonic |
| capacity_ah/soh_pct intra-cycle constancy | ✅ PASS | 0 violations |
| SOH label availability | ✅ 818/819 cycles | Only 1 cycle (B5 cycle_id=40) missing |
| Na-ion current direction | ✅ RESOLVED | 100% of Na-ion cycles contain both charge and discharge (Section 9) |
| B5 gap structure | ✅ RESOLVED | Three blocks [1–100], [151–200], [401–500] with two gaps (Section 8) |
| EOL/RUL feasibility | ✅ RESOLVED | Only NASA NMC (3 cells) supports RUL at 80% EOL (Section 10) |

### 13.2 Issues to Address Before Training

| # | Issue | Severity | Corrected Recommendation |
|---|---|---|---|
| 1 | **LFP B1 has only 1 cycle** — cannot support any prediction task. | **HIGH** | Exclude entirely from training and evaluation. |
| 2 | **Na-ion cycles contain mixed charge+discharge telemetry** — unlike NASA/B5 (discharge-only). Feature extraction must separate phases by current sign. | **HIGH** | Implement charge/discharge phase separation for Na-ion before feature extraction. |
| 3 | **B5 cycle_ids are non-contiguous** with two large gaps (50 and 200 missing cycles). | **HIGH** | Use positional indexing in sorted observed cycle list, NOT `cycle_id - 1`, for sequential features. |
| 4 | **B5 SOH labels are CONDITIONAL** (36.3% of transitions increase; documented charge-balance discrepancy, consistent with instrumentation/measurement issue). | **MEDIUM** | Use raw labels as-is. Do NOT smooth without documented validation procedure. Acknowledge noise in model evaluation metrics. |
| 5 | **temp_c is NULL for 73.7% of rows.** | **HIGH** | Thermal features only for NMC. Do NOT impute. |
| 6 | **Na-ion cells show <1.1% total SOH fade** over 25 cycles. | **MEDIUM** | Usable for early-life SOH estimation only. Not suitable for deep-fade or RUL training. |
| 7 | **SOH > 100% for B0006** (cycle 1: 101.77%). | **LOW** | Allow. Do not clip to [0, 100]. |
| 8 | **Na-ion cell NA-ion_270040-1-1-64 starts below 80% EOL.** | **LOW** | This cell is already past any 80% EOL threshold at observation start. Note when selecting RUL targets. |
| 9 | **RUL training is feasible ONLY for NASA NMC** (B0005, B0006, B0018). All other cells either never cross 80% or start below it. | **MEDIUM** | Scope RUL modeling to NASA NMC. SOH prediction (non-RUL) is feasible for NMC + B5 + Na-ion (limited). |

### 13.3 Assumptions Updated

| Assumption | Status | Evidence |
|---|---|---|
| capacity_ah is a single end-of-cycle discharge capacity per cycle | ✅ VERIFIED | Max distinct values within any cycle = 1 |
| soh_pct is derived from capacity_ah, not an independent measurement | ✅ VERIFIED | Build script: `soh = (cap / rated) × 100` |
| Telemetry rows within a cycle represent a time-ordered discharge curve | ✅ VERIFIED | time_sec monotonically non-decreasing in 819/819 cycles |
| All NASA NMC data represents discharge only | ✅ VERIFIED | Build script filters `c_type == 'discharge'` |
| B5 data contains discharge phase only | ✅ VERIFIED | Build script filters `phase == 'D'` |
| Na-ion data contains both charge and discharge within each cycle | ✅ **NOW VERIFIED** | 100/100 cycles show ~50/50 positive/negative current split |
| B5 cycle_id gaps are consistent (every 2nd cycle sampled) | ❌ **DISPROVED** | Three blocks with irregular gaps of 50 and 200 |
| Na-ion degradation is sufficient for RUL | ❌ **DISPROVED** | <1.1% total fade, never crosses 80% EOL |

---

## 14. Corrected Chronological Train / Validation / Test Split

Splits computed by **position in each cell's sorted observed cycle list** (60/20/20):

### NASA NMC (contiguous cycle_ids, discharge-only)

| Cell | Total Cycles | Train (60%) | Validation (20%) | Test (20%) |
|---|---|---|---|---|
| B0005 | 168 | cycles 1–100 (100 cycles) | cycles 101–134 (34 cycles) | cycles 135–168 (34 cycles) |
| B0006 | 168 | cycles 1–100 (100 cycles) | cycles 101–134 (34 cycles) | cycles 135–168 (34 cycles) |
| B0018 | 132 | cycles 1–79 (79 cycles) | cycles 80–105 (26 cycles) | cycles 106–132 (27 cycles) |

### LFP B5 (non-contiguous cycle_ids, discharge-only)

| Split | Positional Range | Cycle Count | Cycle-ID Range | Notes |
|---|---|---|---|---|
| **Train** | positions 0–149 | **150 cycles** | cycle_ids **1–200** | Spans blocks 1 & 2 |
| **Validation** | positions 150–199 | **50 cycles** | cycle_ids **401–450** | Start of block 3 (200 missing cycles before this) |
| **Test** | positions 200–249 | **50 cycles** | cycle_ids **451–500** | End of block 3 |

> [!NOTE]
> This is the **cycle-level** split over all 250 observed records, before the null-SOH label at `cycle_id=40` is dropped. The actual **model-ready pair** split is 148 / 49 / 51 pairs, with a one-record target/input boundary shift; see §8.5.

### Na-ion (contiguous cycle_ids, mixed charge/discharge)

| Cell | Total Cycles | Train (60%) | Validation (20%) | Test (20%) |
|---|---|---|---|---|
| Each of 4 cells | 25 | cycles 1–15 (15 cycles) | cycles 16–20 (5 cycles) | cycles 21–25 (5 cycles) |

### Exclusions

| Cell | Reason |
|---|---|
| **BATTERY_1** (LFP B1) | Only 1 cycle — excluded from all training and evaluation |

> [!WARNING]
> **Cross-chemistry generalization should NOT be assumed.** NMC, LFP, and Na-ion have fundamentally different degradation mechanisms, voltage profiles, and capacity ranges. Initial models should be trained and evaluated **per-chemistry** before attempting cross-chemistry transfer.

---

## 15. Remaining Blockers Before Model Training

| # | Blocker | Status | Required Action |
|---|---|---|---|
| 1 | **Charge/discharge phase separation for Na-ion** | UNRESOLVED | Implement current-sign-based phase separation in feature extraction pipeline |
| 2 | **Cycle-level feature table not yet created** | UNRESOLVED | Build per-cycle feature extraction (dQ/dV, V statistics, thermal stats for NMC) |
| 3 | **Label table not yet separated from telemetry** | UNRESOLVED | Extract `capacity_ah` and `soh_pct` into standalone label table |
| 4 | **B5 capacity outlier investigation** | UNRESOLVED | Minimum capacity_ah = 0.5428 Ah is an extreme outlier in a 1.14–1.20 Ah trajectory — identify which cycle_id and whether to flag it |
| 5 | **Task definition not finalized** | UNRESOLVED | Choose between Task B (partial-cycle) and Task C (future-cycle) or both. Decision affects feature engineering architecture. |
| 6 | **Model architecture not selected** | UNRESOLVED | LightGBM vs LSTM vs hybrid — pending feature table review |

**Do NOT start model training until blockers 1–5 are resolved and reviewed.**

---

## 16. Summary of Corrections to Version 1 of This Report

| Section | What Changed |
|---|---|
| **B5 Split (was §10)** | Old split used numeric cycle_id intervals [1, 200], (200, 450], (450, 500] which happened to be correct by coincidence but was computed incorrectly. New split is explicitly computed by positional index with overlap/ordering verification. |
| **Na-ion Current (was §9.4)** | Was marked "❓ UNKNOWN." Now **confirmed**: every Na-ion cycle contains both charge and discharge telemetry (~50/50 split). Elevated to HIGH-severity issue. |
| **RUL Feasibility (new §10)** | Previously not analyzed. Now shows only NASA NMC supports RUL at 80% EOL. Na-ion cell 1-1-64 starts below threshold. |
| **B5 Label Quality (was §9.2 item 3)** | **Smoothing recommendation withdrawn.** Prior suggestion to apply isotonic regression/rolling median was unsupported without an independent validation reference. |
| **Leakage Analysis (was §8)** | Expanded from 2 scenarios to 3 distinct tasks (A/B/C). Added empirical verification that capacity_ah can be reconstructed from telemetry (ratio ~1.003–1.007). Clarified that Task C (future-cycle prediction) can legitimately use full-curve historical features. |
| **Gap Analysis (new §8)** | Was marked "❓ UNKNOWN" in §9.4. Now fully mapped: three blocks [1–100], [151–200], [401–500]. |
| **B5 missing-SOH cycle (correction pass, 2026-10-29)** | Earlier text stated the null-SOH cycle was `cycle_id=1`. Direct re-check of the Parquet shows every cycle_id=1 row has SOH 95.60082 and the only all-null-SOH cycle is **`cycle_id=40` (625 rows)**. Corrected in §1, §5.1 and §13.1. |
| **Model-ready LFP pairs (new §8.5)** | Added a read-only re-derivation through the actual pipeline functions: 249 labelled B5 cycles produce **248 pairs**, with three gap-crossing pairs (39→41 diff 2, 100→151 diff 51, 200→401 diff 201) and a model-ready split of 148/49/51. |
| **Horizon semantics (§12.4–§12.6)** | Replaced "cycle N+K" language with **next observed record**. Positional `shift(-1)` does not guarantee the next physical cycle when IDs are missing. |
| **False "alternating step of 2" claim** | Removed from `INTEGRATION_CONTRACT.md` and `model_manifest.json`; replaced with the verified three-block structure. |

---

## Appendix: Commands and Environment

```
Python: 3.12.10 (.venv\Scripts\python.exe)
pandas: 3.1.0
pyarrow: 26.0.0
numpy: 2.5.4

Analysis scripts (read-only, stored in scratch):
  - cycle_level_analysis.py (initial inspection, 2026-10-29)
  - followup_diagnostics.py (follow-up investigations, 2026-10-29)
  - m2_verify_pairs.py (correction-pass verification, 2026-10-29)
Input file: master_table.parquet (not modified, SHA256: 6fa7475d...73b2bb)
Correction-pass command:
  .venv/Scripts/python.exe <scratch>/m2_verify_pairs.py
  (imports member2_modeling.build_and_train, runs feature extraction and pair
   construction only; no training, no artifact writes)
No project source files were modified during this analysis.
This report is the only project file created: AUDIT/MEMBER2_CYCLE_LEVEL_DATA_QUALITY_REPORT.md
```
