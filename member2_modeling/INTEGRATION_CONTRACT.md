# Member 2 → Member 4 Model Integration Contract
**Autonomous Multi-Agent EV Battery Life & Range Predictor**  
**Document Version:** 1.2.0 (Cycle-ID, Horizon & Readiness Corrections)  
**Date:** October 2026  
**Module:** Member 2 (Battery SOH Prediction & RUL Estimation)  
**Consumer:** Member 4 (Multi-Agent System Integration / Orchestrator)  
**Schema Version:** `1.0.0`  
**Model Artifact Set:** `schema 1.0.0`, manifest created `2026-10-29T13:46:30`  
**Model Status:** Research-stage / experimental. **No Member 2 model is production-ready.**  
**Dataset Digest:** `SHA256: 6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb` (master_table.parquet, 515,270 rows)

---

## 1. Executive Summary & Delivery Checklist

| Requirement | Delivery / File Location | Status |
| :--- | :--- | :--- |
| **Exported SOH Model (NMC)** | `member2_modeling/nmc_lgbm_model.joblib` | Delivered — **research-stage** |
| **Exported SOH Model (LFP)** | `member2_modeling/lfp_lgbm_model.joblib` | Delivered — **research-stage, CONDITIONAL labels** |
| **Exported BiLSTM Sequence Model (NMC)** | `member2_modeling/nmc_bilstm_model.pt` | Delivered — **research-stage; not served by the inference API** |
| **Exported RUL Model (NMC)** | `member2_modeling/nmc_rul_lgbm_model.joblib` | Delivered — **UNVALIDATED for pre-EOL forecasting** |
| **Feature Scalers** | `scaler_nmc.joblib`, `scaler_lfp.joblib`, `scaler_nmc_seq.joblib` | Delivered |
| **Machine-Readable Schema** | `member2_modeling/feature_schema.json` | Delivered |
| **Model Manifest & Metrics** | `member2_modeling/model_manifest.json` | Delivered |
| **Python Inference API** | `member2_modeling/predictor.py` (`Member2Predictor`) | Delivered & smoke-tested |

> [!WARNING]
> The learned SOH models **underperform a persistence baseline** (`SOH_pred = current SOH`) on the reported chronological holdout. The RUL model has **no validated pre-EOL test evidence**. See §5.4 before integrating either output into range or charging decisions.

---

## 2. Disposition of Candidate Features Requested by Member 4

Member 4 requested 7 candidate features. Master telemetry does not contain SOC or universal temperature. Each candidate is addressed below with technical justification:

| Candidate Feature | Status | Master Table Telemetry Reality | Member 2 Solution / Substitute |
| :--- | :---: | :--- | :--- |
| `mean_temp_c` | **RESTRICTED** | `temp_c` is missing in 379,834 rows (73.7%). Available ONLY in NASA Ames NMC (135,436 rows). All LFP and Na-ion rows have `temp_c = NULL`. | Supported in `nmc_lgbm` and `nmc_bilstm` as `temp_mean`. **Excluded** from LFP models to prevent synthetic data fabrication. |
| `max_c_rate` | **BLOCKED** | LFP has documented `CURRENT_UNIT_UNDERDETERMINED` flag. NASA current is in Amperes. Without confirmed rated capacity calibration and verified physical units, computing C-rate creates systematic distortion. | Replaced by direct physical telemetry: `i_max`, `i_mean`, `i_min`, `i_std`, and `ah_throughput`. |
| `target_soc` | **BLOCKED** | No SOC column exists in `master_table.parquet`. SOC requires continuous integration across charge-discharge cycles or OCV lookup tables. | Telemetry is discharge-only for NASA and LFP. Target SOC cannot be derived without charge cycle telemetry. |
| `min_soc` | **BLOCKED** | Same reason as `target_soc`. | Replaced by `v_end`, `v_min`, and `v_delta` which capture discharge cutoff depth physically. |
| `hours_above_80soc_per_cycle` | **BLOCKED** | Requires SOC time-series tracking. Telemetry consists solely of discharge segments for LFP and NASA NMC. | Physics-based proxy: `discharge_time_s` and `energy_wh`. |
| `fast_charge_fraction` | **BLOCKED** | NASA NMC and LFP B5 datasets contain **zero charge-phase telemetry** (discharge-only). Na-ion contains mixed phases but lacks fast-charge labeling. | Cannot fabricate charge profiles. Blocked until charge telemetry is integrated. |
| `n_cycles` | **SUPPORTED** | Master table contains non-consecutive `cycle_id` values. For LFP B5 there are **250 observed IDs in three contiguous blocks: 1–100, 151–200 and 401–500**. IDs 101–150 (50 IDs) and 201–400 (200 IDs) are absent between the observed minimum and maximum. Numeric `cycle_id` must NOT be used directly. | Supported as `cycle_position` (0-indexed position in the sorted observed cycle sequence for the cell). |

---

## 3. Supported Chemistries & Evidence

### 3.1 Chemistry Matrix

| Chemistry | SOH Prediction | RUL Estimation | Thermal Features | Status | Evidence / Caveats |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NMC** | **YES** | **UNVALIDATED** | **YES** | **EXPERIMENTAL (SOH)** | 468 cycles across 3 NASA cells (`B0005`, `B0006`, `B0018`). Overall degradation from ~100% to ~57% SOH, with non-monotonic cycle-to-cycle variation. Crosses 80% EOL threshold. Full thermal telemetry available. **The learned SOH model performs worse than persistence on the reported holdout** (MAE 7.97 vs 0.50). RUL model is trained but **unvalidated for pre-EOL forecasting** because all 97 holdout test cycles are post-EOL. |
| **LFP** | **YES** | **NO** | **NO** | **CONDITIONAL / EXPERIMENTAL** | **250 observed cycles** in cell `B5` (333,344 rows). Cycle IDs form three contiguous blocks — **1–100, 151–200, 401–500** — with 50 IDs absent in 101–150 and 200 IDs absent in 201–400. One observed cycle (`cycle_id=40`, 625 rows) has a **null SOH label**, leaving **249 labelled cycles** and **248 model-ready input→target pairs**. 36.3% of consecutive SOH transitions increase (documented charge-balance discrepancy). Minimum observed SOH is 90.91% (never reaches 80% EOL, so RUL is infeasible). No temperature telemetry. **The learned SOH model performs worse than persistence** (MAE 0.93 vs 0.13). |
| **Na-ion** | **NO** | **NO** | **NO** | **BLOCKED** | 100 cycles across 4 cells (25 cycles each). SOH ranges per cell: `NA-ion_...-64` (78.63%–79.22%), `...-63` (88.60%–89.90%), `...-62` (87.58%–88.55%), `...-61` (90.35%–91.00%). Intra-cell degradation is minimal ($\Delta \le 1.30\%$). Telemetry contains mixed charge/discharge with negative and positive currents requiring phase separation. Excluded from model training. |
| **LFP B1** | **NO** | **NO** | **NO** | **EXCLUDED** | Only 1 cycle observed (4,789 rows). Chronological splitting and degradation learning are mathematically impossible. |

> [!IMPORTANT]
> The dataset digest identifies the exact Parquet bytes. It does **not**, by itself, establish that the missing cycle IDs never existed in the original experiment; the Parquet only proves which IDs are present in this file.

---

## 4. Model Architectures & Feature Specifications

Feature **units and definitions** are recorded in `feature_schema.json` under `feature_definitions`. The feature orders below are the exact inference orders used by `Member2Predictor` (the caller may pass a dict in any order; the predictor applies the schema order).

### 4.1 Model 1: NMC SOH Regressor (`nmc_lgbm`)
- **Type:** LightGBM Regressor (`nmc_lgbm_model.joblib`)
- **Task:** Predict the SOH of the **next observed cycle record** from the current record's features ($\widehat{SOH}_{N+1} = f(X_N)$). For NASA NMC, cycle IDs are contiguous per cell, so "next observed record" equals the next physical cycle.
- **Prediction horizon:** **1 observed record** (positional shift), not a configurable physical-cycle horizon.
- **Features (37 ordered features):**
  `ah_throughput`, `avg_power_w`, `current_soh_pct`, `discharge_time_s`, `dvdt_max`, `dvdt_mean`, `dvdt_min`, `dvdt_std`, `energy_wh`, `i_max`, `i_mean`, `i_min`, `i_std`, `n_samples`, `temp_end`, `temp_max`, `temp_mean`, `temp_min`, `temp_range`, `temp_rise`, `temp_start`, `temp_std`, `v_delta`, `v_end`, `v_max`, `v_mean`, `v_min`, `v_q10`, `v_q25`, `v_q50`, `v_q75`, `v_q90`, `v_range`, `v_slope`, `v_start`, `v_std`, `v_time_integral`.

### 4.2 Model 2: LFP SOH Regressor (`lfp_lgbm`)
- **Type:** LightGBM Regressor (`lfp_lgbm_model.joblib`)
- **Task:** Predict the SOH of the **next observed cycle record**. This is an **observed-record horizon model**, not a physical-cycle horizon model.
- **Horizon limitation:** The B5 target is built with positional `shift(-1)` over sorted observed records. The transitions **100 → 151** (51 ID units) and **200 → 401** (201 ID units), plus **39 → 41** (2 ID units, caused by the null label at `cycle_id=40`), are **not one physical cycle apart**. Forecast errors across these gaps are not comparable to adjacent-cycle errors. The existing test split (`n=51`, IDs 449–499 → 450–500) contains **no gap-crossing pair**; validation contains the 200 → 401 pair.
- **Features (29 ordered features):** Same as NMC, **excluding the 8 `temp_*` features**.

### 4.3 Model 3: NMC BiLSTM Sequence Regressor (`nmc_bilstm`)
- **Type:** PyTorch Bidirectional LSTM (`nmc_bilstm_model.pt`)
- **Input Dimension:** `(batch_size, 5, 37)` — a window of **5 consecutive observed cycle records**.
- **Architecture:** 2 BiLSTM layers (hidden_dim=64), fully connected regression head.
- **Scaler:** `scaler_nmc_seq.joblib` (StandardScaler; in the current export it is the same fitted object as `scaler_nmc.joblib`).
- **Split handling:** Sequence windows are built **separately inside train, validation and test**, so no window crosses a split boundary. Test sequences: 83.
- **Integration limitation:** This artifact is delivered for reproducibility but is **not currently served by `Member2Predictor`**; the `.pt` state dict must be loaded into a reconstructed `BiLSTMPredictor` and the NMC scaler applied. Member 4 should not assume an API method exists for it.

### 4.4 Model 4: NMC RUL Regressor (`nmc_rul_lgbm`) — *UNVALIDATED PRE-EOL*
- **Type:** LightGBM Regressor (`nmc_rul_lgbm_model.joblib`)
- **Task:** Predict remaining observed cycles until first crossing below 80.0% SOH.
- **Status:** **UNVALIDATED FOR PRE-EOL FORECASTING.** In chronological testing, **all 97 test targets were post-EOL (ground truth RUL = 0.0)**. The test MAE of 0.0588 cycles reflects zero-variance test targets, not verified degradation-trajectory forecasting. Do not use this output to estimate time-to-EOL for a battery that has not yet crossed 80% SOH.

---

## 5. Model Evaluation & Benchmark Comparison

### 5.1 Protocol: Strict Chronological Per-Cell Split
Splits were constructed strictly per-cell on sorted observed records (60% Train, 20% Val, 20% Test) with **no shuffling and no lookahead of future labels into inputs**. Actual model-ready pair counts:

| Chemistry | Train pairs | Val pairs | Test pairs |
| :--- | ---: | ---: | ---: |
| NMC | 278 | 92 | 95 (83 BiLSTM test sequences) |
| LFP B5 | 148 | 49 | 51 |

> [!NOTE]
> Pairs are split **after** the positional target shift. Consequently the last train target and the first validation input are the same cycle record (e.g. B5 `cycle_id=200`), and likewise at the val/test boundary. No target is duplicated across splits, and that record's SOH is known at inference, but the cycle sets are not strictly disjoint by one record. See §5.5.

### 5.2 NMC Identical-Target Sequence Comparison ($N = 83$ test sequence cycles)
To ensure a strictly fair comparison, all three models were evaluated on the **exact same 83 sequence test rows**:

| Evaluator / Model | Test MAE | Test RMSE | $R^2$ Score | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Persistence Baseline** ($SOH_{N+1} \approx SOH_N$) | **0.4474%** | **0.7006%** | **+0.9578** | Benchmark baseline: captures gradual cycle-to-cycle degradation. |
| **PyTorch BiLSTM** ($W=5$) | **4.9173%** | **5.7347%** | **-1.8291** | Recurrent representation reduces the tree-bounded error, but remains ~11× worse than persistence. |
| **LightGBM Regressor** | **8.3767%** | **9.0229%** | **-6.0034** | Tree-based leaf bounds fail under chronological extrapolation. |

> [!IMPORTANT]
> **On the identical 83 test rows, the persistence baseline has the lowest MAE (0.4474%) and the only positive $R^2$ (+0.9578). Neither learned model outperforms simply carrying forward the current SOH.** The table is a benchmark against which the learned models fail, not evidence of predictive superiority.

#### Per-Cell Breakdown on Identical 83 Test Rows:
- **Cell B0005 ($n=30$):** Persistence MAE = 0.32% | BiLSTM MAE = 4.16% | LightGBM MAE = 8.29%
- **Cell B0006 ($n=30$):** Persistence MAE = 0.46% | BiLSTM MAE = 8.10% | LightGBM MAE = 11.62%
- **Cell B0018 ($n=23$):** Persistence MAE = 0.61% | BiLSTM MAE = 1.76% | LightGBM MAE = 4.25%

### 5.3 Technical Root Cause: Why LightGBM MAE is 7.97%–8.38% vs Persistence 0.45%–0.50%
1. **Chronological Domain Shift:** In training, NMC SOH spans 71.30% to 101.26% (mean: 84.57%). In the chronological holdout test set (the final 20% of life), actual SOH degrades into the 57.69% to 73.02% range (mean: 65.63%).
2. **Decision Tree Extrapolation Bounds:** Gradient boosted trees (LightGBM) partition feature space into piecewise constant leaves. A decision tree **cannot predict values lower than the minimum target seen in its training leaves**. LightGBM's predictions bottom out at ~72.5%–73.6%. For an actual test SOH of 65.6%, the minimum tree prediction yields an automatic ~8.0% error.
3. **LFP B5 Analog:** On LFP B5 (test $n=51$), LightGBM MAE is 0.93% ($R^2 = -14.50$) while persistence MAE is 0.13% ($R^2 = +0.63$) due to the identical bounded tree effect.
4. **Integration Directive for Member 4:** **Never deploy open-loop ML outputs without residual anchoring.** Member 4's orchestrator must predict degradation increments ($\Delta SOH$) anchored to the observed current SOH ($SOH_N$), or use the persistence baseline as a primary gating signal.

### 5.4 Model Readiness Summary

| Artifact | Readiness | Evidence / Limitation |
| :--- | :--- | :--- |
| `nmc_lgbm` | **Experimental / research-stage** | Test MAE 7.97 vs persistence 0.504 ($n=95$); identical-subset MAE 8.3767 vs 0.4474 ($n=83$). |
| `lfp_lgbm` | **Experimental / research-stage, CONDITIONAL labels** | Test MAE 0.9256 vs persistence 0.1313 ($n=51$); test is adjacent-pair-only; labels have a documented charge-balance discrepancy. |
| `nmc_bilstm` | **Experimental / research-stage** | Test MAE 4.9173 vs persistence 0.4474 ($n=83$); not served by `Member2Predictor`. |
| `nmc_rul_lgbm` | **Unvalidated for pre-EOL** | 97/97 chronological test targets are post-EOL (RUL = 0); test MAE 0.0588 reflects zero-variance targets. |

**Member 4 must not describe any of these artifacts as production-ready.** There is no calibration, conformal interval, or validated uncertainty quantification. The only benchmark that currently generalises acceptably is the persistence baseline.

### 5.5 Evaluation & Leakage Caveats (Verified by Inspection)

1. **Boundary overlap:** Because target pairs are split after the shift, the last train target is the first validation input (and the last validation target is the first test input). No split target is duplicated, but a one-record purge/embargo is recommended for any future formal evaluation.
2. **`current_soh_pct` dependency:** This input is the current record's `soh_pct` label. It does not leak the future target, but Member 4 must supply it; the model is dependent on an upstream SOH estimate rather than a raw sensor signal.
3. **Full-curve feature correlation:** Input features can reconstruct the **current** cycle's capacity (audit §12.3). This is not future-label leakage, but it explains why the learned models track persistence instead of learning stronger degradation dynamics.
4. **LFP gap composition:** Train contains the 39 → 41 and 100 → 151 pairs; validation contains 200 → 401; **test contains none**. The reported LFP test metrics are therefore adjacent observed-cycle-ID pairs only.
5. **BiLSTM split handling:** Sequence windows are constructed independently within each split; no window crosses a split boundary.
6. **RUL scope:** The 97 test targets are all post-EOL; no pre-EOL capability is established.
7. **Metrics provenance:** The metrics above are the previously reported values. They were **not recomputed** during this documentation correction.

---

## 6. Python Inference API (`Member2Predictor`)

Member 4 interacts with Member 2 models through the hardened `Member2Predictor` class:

```python
from member2_modeling import Member2Predictor

# Initialize predictor
predictor = Member2Predictor("member2_modeling/")

# 1. Predict SOH of the NEXT OBSERVED RECORD (NMC)
#    horizon_cycles is positional: 1 = next observed record, not 1 physical cycle.
res_soh = predictor.predict_future_soh(features_dict, chemistry="NMC", horizon_cycles=1)
print(f"Predicted SOH: {res_soh.prediction}% [Status: {res_soh.status}]")

# 2. Predict Capacity (NMC, rated=2.0 Ah)
res_cap = predictor.predict_capacity(features_dict, chemistry="NMC", horizon_cycles=1)
print(f"Predicted Capacity: {res_cap.prediction} Ah")
# Defined algebraically as: C_pred = SOH_pred / 100 * C_rated
# (2.0 Ah for NMC, 1.25 Ah for LFP). This is not an independently validated
# capacity measurement.

# 3. Estimate Remaining Useful Life (RUL) — UNVALIDATED pre-EOL
res_rul = predictor.estimate_rul(features_dict, chemistry="NMC")
print(f"RUL: {res_rul.prediction} cycles [Caveat: {res_rul.caveat}]")
```

### 6.1 Input / Output Contract

| Item | NMC SOH | LFP SOH | NMC RUL |
| :--- | :--- | :--- | :--- |
| Input type | `dict[str, float]`, 37 named features | `dict[str, float]`, 29 named features | `dict[str, float]`, 37 named features |
| Feature order | Any dict order; the predictor applies the schema order | Same | Same |
| Missing feature | `status="error"`, `Missing features: [...]` | Same | Same |
| Extra feature | Ignored with a Python warning | Same | Same |
| NaN / non-numeric | `status="error"`, no imputation | Same | Same |
| Preprocessing | `scaler_required=false`; LightGBM consumes raw feature values | Same | Same |
| Output | `PredictionResult` with `status`, `prediction`, `unit="%"`, `horizon_cycles=1` | Same | `unit="cycles"` |
| Unsupported chemistry/horizon | `status="unsupported"` with error text | Same | Non-NMC → `unsupported` |
| BiLSTM | Input shape `(batch, 5, 37)`; **no API method**; scaler `scaler_nmc_seq.joblib` required | N/A | N/A |

`PredictionResult.to_dict()` **omits fields whose value is `None`** (there are no `caveat: null` keys in the actual response). Example actual NMC response:

```json
{
  "status": "ok",
  "prediction": 74.9645,
  "unit": "%",
  "horizon_cycles": 1,
  "chemistry": "NMC",
  "model_name": "nmc_lgbm",
  "schema_version": "1.0.0",
  "model_status": "experimental_research_stage",
  "caveat": "RESEARCH-STAGE MODEL: on the chronological holdout (n=95), the persistence baseline MAE was 0.504% versus 7.968% for this model. Do not treat this output as production-ready."
}
```

Unsupported requests return the actual predictor messages (not the older illustrative text):

```json
{
  "status": "unsupported",
  "chemistry": "Na-ion",
  "schema_version": "1.0.0",
  "error": "Chemistry 'Na-ion' not supported. Supported: NMC, LFP."
}
```

```json
{
  "status": "unsupported",
  "horizon_cycles": 5,
  "chemistry": "NMC",
  "schema_version": "1.0.0",
  "error": "Only horizon_cycles=1 is currently supported."
}
```

---

## 7. Software Dependencies & Runtime Environment

Reconciled with the exact installed virtual environment packages and re-verified on 2026-10-29:
- **Python:** `3.12.10`
- **NumPy:** `2.5.4`
- **Pandas:** `3.1.0`
- **PyArrow:** `26.0.0`
- **LightGBM:** `4.7.0`
- **PyTorch:** `2.15.0+cpu`
- **Scikit-Learn:** `1.9.1`
- **Joblib:** `1.6.0`

---

## 8. Known Limitations & Unresolved Items

| # | Item | Status / Required follow-up |
|---|---|---|
| 1 | **B5 null SOH label at `cycle_id=40`** | Verified: 625 rows, all-null SOH. The pipeline drops this cycle, which creates the 39 → 41 pair. Do not impute or interpolate. |
| 2 | **B5 structural gaps** | Verified present IDs form 1–100, 151–200, 401–500; 50 IDs absent in 101–150 and 200 absent in 201–400. The Parquet does not establish whether those cycles existed upstream. |
| 3 | **LFP CONDITIONAL labels** | 36.3% of consecutive SOH transitions increase. Raw labels are used as-is; no smoothing is applied. |
| 4 | **Learned SOH models vs persistence** | Both NMC and LFP learned models have higher MAE than persistence on the reported holdout. Do not describe them as improvements. |
| 5 | **RUL pre-EOL validation** | Not established. All 97 test targets are post-EOL. |
| 6 | **BiLSTM serving** | No inference method exists in `Member2Predictor`; the `.pt` file requires architecture reconstruction. |
| 7 | **Uncertainty / calibration** | None provided. No prediction intervals, conformal sets, or calibration study. |
| 8 | **Split-boundary overlap** | One record overlaps each boundary after the target shift; purge/embargo is recommended for any future formal evaluation. |
| 9 | **Adjacent-cycle-only LFP metric** | The existing test split is adjacent-only by construction (0 gap-crossing pairs). No separate adjacent-cycle metric was recomputed in this correction; the reported test metric already satisfies that condition. |
| 10 | **Out-of-distribution inputs** | Unknown/unverified. No operating-envelope validation exists beyond the source cells. Member 4 should reject or flag inputs outside the observed feature ranges. |

**Verification provenance:** The cycle-ID structure, null-label cycle, model-ready pair counts and split membership were re-derived read-only from `master_table.parquet` on 2026-10-29. Model files were loaded and the inference API smoke test passed. **Reported metrics were not recomputed and were left unchanged.**
