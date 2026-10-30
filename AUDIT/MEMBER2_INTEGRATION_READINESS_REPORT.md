# Member 2 — Integration Readiness Audit

**Project:** Autonomous Multi-Agent EV Battery Life & Range Predictor
**Audited module:** Member 2 (SOH prediction, RUL estimation)
**Branch:** `member2`
**Date:** 2026-10-29
**Contract draft:** `member2_modeling/integration_readiness/battery_health_contract_v1_1_0_draft.json` (`contract_version = "1.1.0-draft"`; supersedes `v1_0_0`, retained for traceability)
**Handoff document:** `member2_modeling/integration_readiness/HANDOFF_TO_MEMBER3.md`
**Audit type:** Read-only inspection plus additive files under `member2_modeling/integration_readiness/` and `AUDIT/`

> **No production change.** Model weights, scalers, `feature_schema.json`, `model_manifest.json`, `predictor.py`, `build_and_train.py`, `master_table.parquet` and `INTEGRATION_CONTRACT.md` were not modified. No branch switch, stage, commit or push. Hashes in §1.3 prove this.
>
> **No Member 3 implementation.** Member 3 owns the range calculator. This audit stops at the Member 2 boundary: it documents the payload Member 3 would receive and the inputs Member 3 must supply. No range, energy or consumption figure is computed anywhere in this work.

---

## 0. Bottom line

| Question | Answer |
| :--- | :--- |
| Can NMC SOH be wired to a downstream consumer today? | **Yes, as an experimental signal only.** API is functional and validated at the interface level (tests pass). Predictive quality is *worse than carrying the current SOH forward*. |
| Can LFP SOH be wired the same way? | **Yes, experimental, with conditional labels.** Same reliability caveat, plus non-contiguous cycle IDs and a documented charge-balance label discrepancy. |
| Can Na-ion be wired? | **No.** Explicit `UNSUPPORTED_CHEMISTRY`; the master table contains no trained Na-ion artifact and the observed degradation is too small to learn. |
| Can RUL be wired? | **No.** `nmc_rul_lgbm` has zero pre-EOL test evidence (97/97 holdout targets post-80 %). The handoff contract suppresses the numeric value and reports `unavailable / UNVALIDATED_PRE_EOL`. |
| Can Member 3 be integrated *now*? | **No.** `RangeCalculator` source does not exist in this repository or any git ref, so its real signature is unverified; and Member 3's vehicle inputs (pack kWh, mass, Cd, area, Crr, drivetrain efficiency, drive cycle, ambient, HVAC) are not present as sourced data anywhere in the repo. |
| Is the data Member 2 hands over sufficient for Member 3? | **Sufficient as the battery-state input, not sufficient for a validated range and not sufficient on its own to start.** See §4.6 and `HANDOFF_TO_MEMBER3.md`. |
| Is anything here production-ready? | **No.** No calibration, no uncertainty quantification, no out-of-distribution check, no vehicle-level validation. |

---

## 1. Repository state inspected

### 1.1 Git state (recorded, not modified)

```
branch: member2
M   .gitignore
AM  AUDIT/MEMBER2_CYCLE_LEVEL_DATA_QUALITY_REPORT.md
??  AUDIT/MEMBER2_FORMULA_DERIVATION_AND_ML_PLAN.md
??  AUDIT/MEMBER2_FORMULA_FEATURE_EXPERIMENT_REPORT.md
??  AUDIT/MEMBER2_FORMULA_IMPLEMENTATION_MATRIX.csv
??  member2_modeling/
```

Observations:

- `member2_modeling/` is **untracked** on this branch, so `git diff` cannot prove model files are unchanged. Hash comparison in §1.3 is used instead.
- The pre-existing modifications (`.gitignore`, the cycle-level data-quality report) predate this audit and were left untouched.
- Nothing was staged, committed or pushed during this audit.

### 1.2 Files read for this audit

| File | Role | State |
| :--- | :--- | :--- |
| `member2_modeling/predictor.py` | Inference API (`Member2Predictor`, `PredictionResult`) | Read-only, unchanged |
| `member2_modeling/feature_schema.json` | Feature names, order, units, blocked features, `schema_version 1.0.0` | Read-only, unchanged |
| `member2_modeling/model_manifest.json` | Statuses, metrics, chemistries, `sample_io` | Read-only, unchanged |
| `member2_modeling/INTEGRATION_CONTRACT.md` | Existing v1.2.0 contract to **Member 4** | Read-only, **not overwritten** |
| `member2_modeling/build_and_train.py` | Training/splitting pipeline | Read-only, unchanged |
| `member2_modeling/experiments/test_*.py` | Existing experiment test suites | Read-only, executed |
| `master_table.parquet` | Canonical dataset | Read-only, hash unchanged |

### 1.3 Artifact integrity (recomputed at audit time)

| Artifact | SHA-256 | Compared against |
| :--- | :--- | :--- |
| `master_table.parquet` | `6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb` | Manifest `parquet_sha256` — match |
| `member2_modeling/feature_schema.json` | `989cf8eb9a199f4b7384f9ef28299577d2a589cec93f8da75f49aafccb971a98` | RUL experiment baseline — match |
| `member2_modeling/model_manifest.json` | `bfdf2d4a8a8ec552774752e833d98a7b0d64f44e2fc2d482a95c1223f2a0b353` | RUL experiment baseline — match |
| `member2_modeling/nmc_rul_lgbm_model.joblib` | `206009c8b8929bbf7a4ffcd93b5ed3178f2d38118172a3997ec4ec16f5434abb` | RUL experiment baseline — match |
| `member2_modeling/build_and_train.py` | `b56ce13f7eab8a427ff319ef867404371265eac16a959ef2570dd68bffeda8a6` | RUL experiment baseline — match |
| `member2_modeling/predictor.py` | `ef12025e1c9daf01b8e12a28d45b3268fea5d3cf73d0f03e471ee3a34642b327` | RUL experiment baseline — match |
| `member2_modeling/nmc_lgbm_model.joblib` | `b3774599f39db1a983263dca38ff3c8f80e178a811e292e9d37b6cc13f172f45` | — (unchanged this session) |
| `member2_modeling/lfp_lgbm_model.joblib` | `12546c6a432d209619be77afe2d9a0740bb8fac4b258b9d333b3fca23b1f7e15` | — |
| `member2_modeling/nmc_bilstm_model.pt` | `7dcaf0f8b5d22919795024c0c2feeaf4bffa6155758b952fd1dafe5c3d87cf34` | — |
| `member2_modeling/scaler_nmc.joblib` | `2d3a0917ece60b493dbd472ad36adf3a5ce8338a39f803dcc05773b206e6e39d` | — |
| `member2_modeling/scaler_lfp.joblib` | `7111527c8e2a58f8ced8f0709baef7fdf7acac9c59c000f31c310f01e38b46a0` | — |
| `member2_modeling/INTEGRATION_CONTRACT.md` | `0de851584a933375fb8ce164030c2a198480f6007ef3042474a8241dbde146a7` | — |

`scaler_nmc.joblib` and `scaler_nmc_seq.joblib` are byte-identical (same hash), consistent with the manifest statement that the sequence scaler is the same fitted object.

### 1.4 Member 3 availability (verified negative result)

Searched the working tree, all git refs, and the sibling project directory:

- No `range_calculator.py`, no `RangeCalculator` class, no file matching `*range*calc*`.
- `git log --all -- "*range*"` returns nothing.
- No Member 3 package, module or stub exists anywhere reachable.

**Consequence:** the widely-quoted signature `estimate(soh_pct, cycle, ambient_c, hvac_power_kw)` is **provisional and unverified**. It is treated in this audit as a hypothesis about Member 3's interface, not as a code fact. Any integration built on it is a design proposal pending Member 3's actual source.

### 1.5 Verified code facts vs. historical report claims

| Claim | Classification |
| :--- | :--- |
| `predict_future_soh` supports only NMC/LFP and only `horizon_cycles=1` | **Verified in code and by execution** |
| Na-ion returns `status="unsupported"` | **Verified by execution** |
| Extra feature keys are ignored with a `UserWarning` and prediction still succeeds | **Verified by execution** (see §2.4) |
| RUL returns `status="ok", prediction=0.0` for an NMC sample | **Verified by execution**; this is why the adapter suppresses it |
| NMC SOH MAE 7.968 vs persistence 0.504 (n=95); LFP 0.926 vs 0.131 (n=51) | **Historical reported metrics — not recomputed in this audit** |
| RUL 97/97 holdout targets post-EOL | **Historical, independently re-derived earlier in the RUL audit** (`experiments/results/rul/`) |
| `master_table.parquet` contains 515,270 rows | **As reported in manifest; not re-counted here** |

---

## 2. Current supported behavior, per chemistry

The only inference entry point is `Member2Predictor` in `member2_modeling/predictor.py`. `PredictionResult.to_dict()` omits any field whose value is `None`.

### 2.1 Capability matrix

| Chemistry | SOH inference implemented | Required features | Unit | Horizon | Model status | Validation status | RUL |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- | :--- |
| **NMC** | Yes (`nmc_lgbm`) | 37 (incl. 8 `temp_*`) | % | 1 observed record | `experimental_research_stage` | Underperforms persistence (reported) | **Unavailable** (`unvalidated_pre_eol`) |
| **LFP** | Yes (`lfp_lgbm`) | 29 (no `temp_*`) | % | 1 observed record | `experimental_research_stage` (manifest: `..._conditional_labels`) | Conditional labels; underperforms persistence (reported) | **Unsupported** (never crosses 80 % EOL in observed data) |
| **Na-ion** | **No** | — | — | — | No artifact | Not trained (ΔSOH ≤ 1.3 % per cell) | No |
| **LFP B1** | **No** | — | — | — | Excluded (1 cycle) | — | No |
| Unknown string | **No** | — | — | — | — | — | — |

### 2.2 Exact API surface (as implemented)

| Function | Arguments | Success output | Failure behavior |
| :--- | :--- | :--- | :--- |
| `predict_future_soh(features, chemistry, horizon_cycles=1)` | `features: dict[str,float]`; `chemistry: "NMC"\|"LFP"`; `horizon_cycles: int` | `status="ok"`, `prediction` (float, %), `unit="%"`, `horizon_cycles`, `chemistry`, `model_name`, `schema_version="1.0.0"`, `model_status="experimental_research_stage"`, `caveat` | Unknown chemistry → `unsupported` + `"Chemistry '<X>' not supported. Supported: NMC, LFP."`; horizon ≠ 1 → `unsupported` + `"Only horizon_cycles=1 is currently supported."`; missing/NaN features → `error` (`"Missing features: [...]"`, `"NaN values in features: [...]"`) |
| `predict_capacity(features, chemistry, horizon_cycles=1)` | same | `prediction` (Ah), `unit="Ah"` | Passes through the SOH error; **derived algebraically**, not measured |
| `estimate_rul(features, chemistry)` | same, NMC only | `status="ok"`, `prediction` (cycles, clipped at 0), `model_status="unvalidated_pre_eol"` | Non-NMC → `unsupported`; still returns a number for NMC — **the contract layer blocks it** |

Rated reference capacities used by `predict_capacity`: `{"NMC": 2.0, "LFP": 1.25}` Ah (cell-level).

### 2.3 What each chemistry's output actually means

- **NMC:** SOH of the *next observed cycle record* from the current record's features. NASA NMC cycle IDs are contiguous per cell, so for this source "next observed record" equals the next physical cycle. Includes an upstream dependency: `current_soh_pct` is the current record's measured SOH label, not a raw sensor signal.
- **LFP (B5):** SOH of the *next observed record*. Observed IDs form blocks 1–100, 151–200, 401–500. Transitions 100→151 and 200→401 are **not one physical cycle apart**; `cycle_id=40` has a null label and is dropped, creating a 39→41 pair. 36.3 % of consecutive SOH transitions *increase* (documented charge-balance discrepancy), so labels are conditional.
- **Na-ion:** no model, no output.
- **Capacity (all):** `SOH/100 × rated capacity`. It is an algebraic restatement of the SOH prediction and must never be read as a measured capacity or as pack energy.

### 2.4 Interface-level hazards found in the base API

1. **Cross-chemistry scoring is silent.** Passing the 37-feature NMC sample to `predict_future_soh(..., chemistry="LFP")` returns `status="ok"`, `prediction=94.3044` with only:
   `UserWarning: Ignoring unexpected features: ['temp_end', 'temp_max', 'temp_mean', 'temp_min', 'temp_range', 'temp_rise', 'temp_start', 'temp_std']`
   The adapter rejects this with `FEATURE_SET_MISMATCH` in strict mode (default).
2. **RUL returns a number even when unvalidated.** `estimate_rul(sample, "NMC")` → `status="ok", prediction=0.0, model_status="unvalidated_pre_eol"`. A naive consumer that only checks `status` would publish "0 cycles remaining" as fact. The adapter replaces this with `rul.status="unavailable"`, `rul_cycles=null`.
3. **`horizon_cycles` is positional, not physical.** A consumer reading `horizon_cycles=1` as "one physical cycle" is wrong for LFP.
4. **No out-of-distribution, calibration or uncertainty output exists.** These are recorded as `checks_not_performed` in the contract, not as passed checks.
5. **The chain cannot start from telemetry alone.** Verified against the canonical parquet: `extract_cycle_features()` emits 36 of 37 NMC features and 28 of 29 LFP features. The missing one is `current_soh_pct`, which `build_and_train.py` injects from the measured SOH label (`inputs["current_soh_pct"] = inputs["soh_pct"]`, line 273). A raw discharge record therefore cannot produce a prediction; an upstream current-SOH measurement (BMS capacity estimate, reference test or label) is mandatory. Without it the adapter returns `error / MISSING_FEATURES` naming `current_soh_pct`, with no default or imputation.

---

## 3. Battery-health output contract (v1.1.0-draft)

Files:

- `member2_modeling/integration_readiness/battery_health_contract_v1_1_0_draft.json` — current JSON Schema draft 2020-12, `additionalProperties: false` at every level.
- `member2_modeling/integration_readiness/battery_health_contract_v1_0_0_draft.json` — superseded draft, kept for traceability.
- `member2_modeling/integration_readiness/adapter.py` — produces and validates that shape from the unchanged API.
- `member2_modeling/integration_readiness/HANDOFF_TO_MEMBER3.md` — field-by-field sufficiency audit for Member 3.
- `member2_modeling/integration_readiness/__init__.py` — exports `CONTRACT_VERSION`, `RUL_APPROVED_FOR_USE`, `build_health_result`, `build_range_arguments`, `validate_soh_output`.

The contract is an **additive wrapper**. It does not alter `Member2Predictor`; Member 4's existing v1.2.0 contract still describes the raw API and was not overwritten.

### 3.1 Top-level fields

| Field | Type | Required | Meaning / provenance |
| :--- | :--- | :---: | :--- |
| `contract_version` | string | yes | `"1.1.0-draft"` |
| `status` | enum | yes | `ok` \| `ok_experimental` \| `unsupported` \| `error` |
| `production_ready` | boolean | yes | Always `false`; derived from model status, never caller-supplied |
| `subject` | object | yes | Chemistry + identifiers supplied by the caller |
| `soh` | object | yes | Always present; `predicted_soh_pct` is `null` unless a model genuinely scored |
| `rul` | object | yes | Never numeric while `RUL_APPROVED_FOR_USE = False` |
| `capacity` | object \| null | yes | `null` unless explicitly requested; always Ah, never kWh |
| `quality` | object | yes | Flags, warnings, checks passed, checks *not* performed, label provenance |
| `errors` | array | yes | Machine-readable `{code, message, field?}`; empty on success |
| `source` | object | yes | Module, function, schema version, model file — actual code path |
| `request` | object | yes | Echo of `horizon_cycles` |

`subject` (all required, caller-supplied, never inferred; invalid values are replaced by `null` so error responses stay schema-valid): `chemistry`, `cell_id`, `dataset_id`, `cycle_id` (integer), `cycle_position` (0-based contiguous observed index, integer or null), `prediction_timestamp` (ISO-8601 string).

`soh` (all required): `status`, `predicted_soh_pct` (number 0–100 or null), `current_observed_soh_pct` (number 0–100 or null — the measured current-cycle SOH, i.e. the persistence baseline), `unit` (`"%"`), `horizon` (`{value, unit:"observed_cycle_records", semantics}`), `model_id`, `model_status`, `validation_status`, `uncertainty` (`{status:"not_provided"}` — no interval is invented).

`rul` (all required): `status`, `rul_cycles` (null unless approved), `unit` (`"cycles"`), `reason`, `approved_for_use`, `model_id`, `model_status`. Schema rule: `status="available"` requires `approved_for_use=true` **and** a numeric `rul_cycles`; with the current constant no such object can be produced.

`capacity` (when present): `capacity_ah`, `unit:"Ah"`, `reference_capacity_ah`, `basis:"derived_soh_pct_times_rated_capacity"`, `validation_status:"not_independently_validated"`, `is_pack_energy:false`.

### 3.2 State machine

| Situation | `status` | `soh.predicted_soh_pct` | `rul.rul_cycles` | Error code |
| :--- | :--- | :---: | :---: | :--- |
| NMC successful inference | `ok_experimental` | number | `null` | — |
| LFP successful inference | `ok_experimental` | number | `null` | — |
| Na-ion / unknown chemistry | `unsupported` | `null` | `null` | `UNSUPPORTED_CHEMISTRY` |
| Horizon ≠ 1 | `unsupported` | `null` | `null` | `UNSUPPORTED_HORIZON` |
| Missing/invalid provenance | `error` | `null` | `null` | `MISSING_PROVENANCE` / `INVALID_PROVENANCE` |
| Missing / NaN / non-finite feature | `error` | `null` | `null` | `FEATURE_MISSING` / `FEATURE_NOT_FINITE` |
| Feature keys ≠ declared chemistry schema (strict mode) | `error` | `null` | `null` | `FEATURE_SET_MISMATCH` |
| Model predicts non-finite or outside 0–100 | `error` | `null` | `null` | `INVALID_SOH_OUTPUT` |
| RUL requested while unapproved | (unchanged) | — | `null`, `reason="UNVALIDATED_PRE_EOL"`, `approved_for_use=false` | — |

Rules enforced in code: no interpolation, no default fill, no unit conversion, no uncertainty number, no RUL number.

### 3.3 Actual outputs (reproduced from the real predictor)

NMC with the manifest `sample_io` features, `cell_id="SAMPLE_CELL"`, `dataset_id="SAMPLE_DATASET"`, `cycle_id=1`:

```json
{
  "contract_version": "1.1.0-draft",
  "status": "ok_experimental",
  "production_ready": false,
  "subject": {"chemistry": "NMC", "cell_id": "SAMPLE_CELL", "dataset_id": "SAMPLE_DATASET",
              "cycle_id": 1, "cycle_position": 0, "prediction_timestamp": "2026-10-29T00:00:00Z"},
  "soh": {
    "status": "ok",
    "predicted_soh_pct": 74.9645,
    "current_observed_soh_pct": 69.30557575767696,
    "unit": "%",
    "horizon": {"value": 1, "unit": "observed_cycle_records",
                "semantics": "SOH of the next observed cycle record (positional shift by 1). This is not necessarily the next physical cycle; LFP B5 contains observed-ID transitions 100->151 and 200->401."},
    "model_id": "nmc_lgbm",
    "model_status": "experimental_research_stage",
    "validation_status": "research_stage_not_production_ready",
    "uncertainty": {"status": "not_provided"}
  },
  "rul": {"status": "unavailable", "rul_cycles": null, "unit": "cycles",
          "reason": "UNVALIDATED_PRE_EOL", "approved_for_use": false,
          "model_id": "nmc_rul_lgbm", "model_status": "unvalidated_pre_eol"},
  "capacity": null,
  "quality": {
    "quality_flags": ["CHEMISTRY_DECLARED_BY_CALLER", "CURRENT_SOH_IS_UPSTREAM_LABEL",
                      "MODEL_EXPERIMENTAL_RESEARCH_STAGE", "MODEL_UNDERPERFORMS_PERSISTENCE",
                      "NO_OUT_OF_DISTRIBUTION_CHECK", "NO_UNCERTAINTY_QUANTIFICATION",
                      "RUL_NOT_APPROVED", "SOH_HORIZON_OBSERVED_RECORD_NOT_PHYSICAL_CYCLE"],
    "warnings": ["RESEARCH-STAGE MODEL: on the chronological holdout (n=95), the persistence baseline MAE was 0.504% versus 7.968% for this model. Do not treat this output as production-ready.",
                 "RUL is not exposed: no approved RUL model exists; the shipped NMC RUL artifact is unvalidated for pre-EOL forecasting."],
    "checks_passed": ["provenance_supplied", "feature_set_matches_schema", "feature_values_finite",
                      "soh_output_within_0_100", "model_loaded_and_scored"],
    "checks_not_performed": ["out_of_distribution", "sensor_calibration", "label_freshness",
                             "uncertainty_calibration", "vehicle_level_range_validation"],
    "label_provenance": {"soh_labels": "measured_capacity_ratio", "status": "observed"}
  },
  "errors": [],
  "source": {"module": "member2_modeling/predictor.py", "function": "Member2Predictor.predict_future_soh",
             "schema_version": "1.0.0", "model_file": "nmc_lgbm_model.joblib"},
  "request": {"horizon_cycles": 1}
}
```

Na-ion returns `status="unsupported"`, `soh.predicted_soh_pct=null`, `rul.reason="CHEMISTRY_UNSUPPORTED"`, flag `NAION_INFERENCE_UNSUPPORTED`, and:

```json
"errors": [{"code": "UNSUPPORTED_CHEMISTRY",
            "message": "Chemistry 'Na-ion' not supported. Supported SOH chemistries: NMC, LFP.",
            "field": "chemistry"}]
```

LFP differs by `model_id="lfp_lgbm"`, `validation_status="research_stage_conditional_labels_not_production_ready"`, extra flag `CONDITIONAL_LABELS`, `label_provenance.status="conditional"`, and `rul.reason="CHEMISTRY_UNSUPPORTED"`.

### 3.4 Capacity object (opt-in, still Ah)

```json
{"capacity_ah": 1.4993, "unit": "Ah", "reference_capacity_ah": 2.0,
 "basis": "derived_soh_pct_times_rated_capacity",
 "validation_status": "not_independently_validated", "is_pack_energy": false}
```

This is a **cell-level** quantity for a 2.0 Ah NMC cell. It is not a pack rating and must never be multiplied, scaled or renamed into kWh by either member.

---

## 4. Member 2 → Member 3 handoff

### 4.1 What Member 2 supplies (field mapping)

| Member 3 needs | Member 2 field | Type | Unit | Provenance | Status |
| :--- | :--- | :--- | :---: | :--- | :--- |
| State of health | `soh.predicted_soh_pct` | float | % (0–100) | `nmc_lgbm` / `lfp_lgbm`, research-stage | **Available (experimental)** — only when `status ∈ {ok_experimental}` |
| Current measured SOH (anchor) | `soh.current_observed_soh_pct` | float | % (0–100) | Feature `current_soh_pct` (measured capacity ratio) | **Available** — persistence baseline value; null if unavailable |
| Chemistry context | `subject.chemistry` | string | — | Caller-declared, echoed | Available |
| Cell identity | `subject.cell_id`, `subject.dataset_id` | string | — | Caller-supplied | Available (required) |
| Cycle identity | `subject.cycle_id` | int | observed cycle ID | Caller-supplied | Available — **non-contiguous** for LFP B5 |
| Cycle position | `subject.cycle_position` | int or null | 0-based index | Caller-supplied (optional) | Available when supplied |
| Timestamp | `subject.prediction_timestamp` | ISO-8601 string | — | Caller-supplied | Available (required) |
| Model identity | `soh.model_id`, `source.model_file` | string | — | Code constants | Available |
| Validation status | `soh.validation_status`, `soh.model_status` | string | — | Manifest/code | Available |
| Quality flags / warnings | `quality.quality_flags`, `quality.warnings` | array[string] | — | Derived | Available |
| Label provenance | `quality.label_provenance` | object | — | Manifest | Available |
| Cell capacity | `capacity.capacity_ah` | float | Ah | Derived from SOH × rated | Optional, **not measured** |
| Reference capacity | `capacity.reference_capacity_ah` | float | Ah | `RATED_CAPACITY` constant | Optional |
| RUL | `rul.rul_cycles` | null | cycles | — | **Unavailable by design** |
| Uncertainty | `soh.uncertainty` | object | — | — | **Not provided** (`status:"not_provided"`) |

### 4.2 What Member 3 must obtain separately (not present in this repository)

Tracked as requirements only in `member2_modeling/integration_readiness/vehicle_inputs_required.json`. Every entry has `"status": "not_supplied"` and **no value**; nothing in this audit asserts these exist.

| Input | Required unit | Source / provenance that must be cited |
| :--- | :--- | :--- |
| Vehicle identity | — | Make/model/variant or test mule ID |
| Pack nominal energy | kWh | OEM spec sheet or teardown measurement |
| Pack usable energy **or** usable fraction | kWh / ratio (one of) | OEM buffer spec or measured usable capacity |
| Mass | kg | Kerb mass or test weight |
| Drag coefficient `Cd` | dimensionless | OEM/wind-tunnel publication |
| Frontal area | m² | OEM spec or measurement |
| Rolling resistance coefficient | dimensionless | Tyre/vehicle dataset |
| Drivetrain efficiency | ratio 0–1 | Test bench or OEM |
| Regenerative braking assumption | — | Documented model (e.g. recovery fraction) |
| Drive cycle time & speed | s, m/s | Source-attributed cycle (WLTP/EPA/custom), time series |
| Road grade | ratio or % | If absent, must be documented as flat-road |
| Ambient temperature | °C | Measured or documented scenario |
| HVAC / auxiliary load | kW | Measured or documented scenario, with duty assumption |
| Validation consumption / range | kWh/km, km | Published or measured vehicle data — required before any range claim |

**Rule:** a placeholder in a config file is not evidence. Until each row has a source, the range output is unvalidated regardless of how good the SOH input is.

### 4.3 Exact interface mapping (provisional)

Member 2 provides `build_range_arguments(contract, *, ambient_c, hvac_power_kw)`. Its actual output for the NMC example:

```json
{
  "status": "ok_experimental",
  "arguments": {"soh_pct": 74.9645, "cycle": 1, "ambient_c": 25.0, "hvac_power_kw": 0.5},
  "missing": [],
  "signature_verified": false,
  "signature_note": "Provisional four-argument form soh_pct, cycle, ambient_c, hvac_power_kw. Member 3 range_calculator.py is not present in the repository or git history, so the real signature is unverified.",
  "cycle_note": "cycle is the observed cycle_id passed through unchanged. It is not a contiguous counter: LFP B5 observed IDs are 1-100, 151-200 and 401-500. If Member 3 needs a contiguous cycle count, read contract.subject.cycle_position (null when the caller did not supply it).",
  "cycle_semantics": "observed_cycle_id",
  "soh_semantics": "soh_pct is the predicted SOH of the next observed record. contract.soh.current_observed_soh_pct is the measured current SOH (the persistence baseline) if an anchored or gated use is preferred; persistence beat this model on every reported holdout.",
  "assumption": "SOH linearly scales usable pack energy. This is an approximation; the resulting range is unvalidated until checked against vehicle-specific evidence.",
  "production_ready": false
}
```

Answering the verification question directly: **the existing function cannot be verified** because no existing function exists. The four-argument form is a placeholder hypothesis only; `signature_verified` is `false` in every response. Member 2 did not create, modify or stub `RangeCalculator`.

Boundary rules encoded in the payload:

1. `soh_pct` is the only Member 2 value; `cycle` is passed through as the observed ID.
2. `ambient_c` and `hvac_power_kw` must be supplied by Member 3; if absent the payload is `blocked / MISSING_INPUTS` with `missing` listing them.
3. Invalid values (non-numeric, non-finite, negative HVAC) → `blocked / INVALID_SCENARIO_INPUT`.
4. An unsupported or failed SOH contract → `blocked / SOH_UNAVAILABLE` or `INVALID_SOH`; no arguments are produced.
5. The payload contains **no** pack kWh, cell Ah, energy or range field — verified by test.
6. Cell Ah and pack kWh are never converted into each other. `capacity.is_pack_energy` is `false`.

### 4.4 What Member 3 would do (description only, not implemented here)

```
inputs: contract (Member 2) + vehicle scenario (Member 3, sourced)
1. reject unless contract.status is ok/ok_experimental and soh in 0..100
2. handoff = build_range_arguments(contract, ambient_c, hvac_power_kw)
3. reject unless handoff.status == "ok_experimental"
4. usable_kwh = pack_usable_kwh (sourced); energy_available = usable_kwh * soh_pct/100
5. range = f(energy_available, mass, Cd, area, Crr, efficiency, regen, cycle, ambient, HVAC)
6. label the result experimental; compare against validation consumption/range
```

Step 4's linear scaling is an **approximation**: real usable energy is not guaranteed proportional to capacity SOH near end of life, and pack-level effects (cell imbalance, thermal derating, BMS limits) are absent from cell-level SOH.

### 4.5 Blocked states Member 3 will observe

| Condition | Payload | Member 3 action |
| :--- | :--- | :--- |
| Na-ion | `SOH_UNAVAILABLE`, `arguments: null` | Do not compute range; report chemistry unsupported |
| RUL requested | `rul.status="unavailable"`, `rul_cycles: null` | Range must not depend on RUL |
| Missing ambient/HVAC | `MISSING_INPUTS`, `missing: [...]` | Request scenario inputs; do not default silently |
| SOH null or out of 0–100 | `INVALID_SOH` | Treat as failure, not 0 % |
| Experimental success | `ok_experimental`, `production_ready:false` | May compute a labelled experimental range only |

### 4.6 Sufficiency verdict for Member 3

Full field-by-field audit: `member2_modeling/integration_readiness/HANDOFF_TO_MEMBER3.md`.

| Member 3 need | Supplied by Member 2 | Verdict |
| :--- | :--- | :--- |
| SOH point estimate | `soh.predicted_soh_pct` | Supplied, **worse than persistence** on every reported holdout |
| Current SOH anchor | `soh.current_observed_soh_pct` (new in 1.1.0) | Supplied; the value Member 3 should prefer or gate on |
| SOH uncertainty | `soh.uncertainty` | **Not available** — no calibration exists |
| Current-cycle SOH **input** to the model | Feature `current_soh_pct` (echoed as `soh.current_observed_soh_pct`) | Supplied only if an upstream measurement exists — **not derivable from telemetry** |
| Chemistry, cell, dataset, cycle, timestamp | `subject.*` | Supplied; `cycle_position` optional, `cycle_id` may be non-contiguous |
| Model identity and readiness | `soh.model_id`, `soh.validation_status`, `production_ready` | Supplied |
| Quality flags, warnings, label provenance | `quality.*` | Supplied; must be propagated verbatim |
| Explicit failure states | `status`, `errors`, `rul.reason` | Supplied and enforced by tests |
| RUL | `rul.rul_cycles` | **Withheld** — unvalidated pre-EOL |
| Cell capacity | `capacity.capacity_ah` | Optional, **derived**, never pack kWh |
| Pack energy, vehicle, drive cycle, ambient/HVAC scenario, validation range | — | **Not in the repository** — Member 3 / wider team |
| Real `estimate()` signature | — | **Unverified** — no Member 3 source exists |

**Verdict:** the handoff is **sufficient in shape** as the battery-state input — a bounded SOH, an observed anchor, provenance, quality flags and hard blocking, with nothing left to infer from prose. It is **not sufficient** for a validated range (the SOH signal underperforms persistence, has no uncertainty and no vehicle-level validation), and it is **not sufficient on its own** to start integration because Member 3's pack/vehicle/scenario inputs and its actual interface are absent. Operationally, the chain also requires an **upstream current-SOH measurement**, because `current_soh_pct` is a measured quantity rather than a telemetry-derived feature.

Real-record check: features rebuilt from the canonical parquet for four actual cycles (NMC B0005 100→101, B0018 90→91, LFP B5 499→500 and the gap-crossing 100→151) all produced `ok_experimental` contracts with complete provenance and four-argument handoffs. Those numbers demonstrate plumbing, not accuracy — two beat the anchor, two did not — and are asserted in `RealDataHandoffTests`.

---

## 5. Tests — focused additions and actual results

New file: `member2_modeling/integration_readiness/test_integration_readiness.py` (36 tests). It exercises the **real** loaded models read-only, retrains nothing, writes nothing, and contains no Member 3 implementation (the handoff tests assert only that the payload stops at the four arguments and carries no energy, capacity or range value).

Coverage:

- **Contract shape:** top-level and nested key parity against the 1.1.0 schema, contract-version consistency between adapter and schema (and that the 1.0.0 draft is retained but superseded), subject echoing, `cycle_position` null-when-absent, SOH fields and horizon semantics, RUL never numeric, capacity omitted by default, quality flags/checks, source matches the real function, JSON-serializable with no NaN.
- **Sufficiency:** every promised Member 3 field present with its documented type, including `current_observed_soh_pct` and an explicitly null `rul_cycles`.
- **Real records:** features rebuilt from the canonical parquet for NMC B0005 and LFP BATTERY_5; complete handoff produced for both; missing `current_soh_pct` blocks with `MISSING_FEATURES` naming it.
- **Capacity:** 1.4993 Ah from 74.9645 % on a 2.0 Ah cell; no kWh field.
- **Unsupported:** Na-ion (both SOH values null), LFP conditional path, horizon 5, unknown chemistry.
- **Guards:** missing/invalid provenance, non-integer cycle, invalid `cycle_position` (`-1`, `"3"`, `2.5`, `True`), missing feature, NaN feature, NMC keys rejected for LFP under strict mode, non-strict mode flags extras, SOH range validator, capacity not SOH-validated.
- **Handoff:** exactly four arguments, `cycle_semantics`/`soh_semantics`/`cycle_note` present, missing/invalid scenario blocking, unsupported contract cannot reach a range call, no energy/range leakage.

Executed results (Python 3.12.10, `.venv`):

| Suite | Command | Result |
| :--- | :--- | :--- |
| Integration readiness (new) | `python -m unittest member2_modeling.integration_readiness.test_integration_readiness` | **36 passed**, 4.5 s |
| Experiments (RUL + walk-forward + features) | `python -m unittest member2_modeling.experiments.test_rul_validation member2_modeling.experiments.test_walk_forward member2_modeling.experiments.test_experimental_features` | **47 passed**, 4.8 s (only LightGBM `eval_set` deprecation warnings) |
| Na-ion | `python -m unittest member2_modeling.experiments.naion.test_naion_experiment` | **27 passed**, 2.6 s |
| Predictor smoke test | `python -m member2_modeling.predictor` | **PASSED** — SOH 74.9645 %, capacity 1.4993 Ah, RUL 0.0 (`unvalidated_pre_eol`), Na-ion `unsupported`, horizon 5 `unsupported`, missing features `error` |

Total: **110 tests passed, 0 failed**, plus the smoke test. Note that `unittest discover` from `member2_modeling/` only reaches the integration-readiness package; the experiment suites are run by explicit module name as above.

The RUL audit (`member2_modeling/experiments/results/rul/MEMBER2_RUL_VALIDATION_REPORT.md`, 28 tests) remains valid and was not re-run for this report.

---

## 6. Files created / changed in this audit

**Created (all additive):**

| Path | Purpose |
| :--- | :--- |
| `AUDIT/MEMBER2_INTEGRATION_READINESS_REPORT.md` | This report |
| `member2_modeling/integration_readiness/__init__.py` | Package exports |
| `member2_modeling/integration_readiness/adapter.py` | Contract builder + handoff mapping + validators |
| `member2_modeling/integration_readiness/battery_health_contract_v1_1_0_draft.json` | Current versioned JSON Schema (draft 2020-12) |
| `member2_modeling/integration_readiness/battery_health_contract_v1_0_0_draft.json` | Superseded draft, retained for traceability |
| `member2_modeling/integration_readiness/HANDOFF_TO_MEMBER3.md` | Field-by-field Member 2 → Member 3 sufficiency audit |
| `member2_modeling/integration_readiness/vehicle_inputs_required.json` | Member 3 input requirements, values deliberately absent |
| `member2_modeling/integration_readiness/test_integration_readiness.py` | 36 focused tests |

**Changed:** none. No production artifact was edited; no existing contract was overwritten; `.gitignore` and the pre-existing AUDIT changes were left as found.

**Not done:** no branch switch, no `git add`, no commit, no push, no Member 3 code, no range/energy computation, no retraining.

---

## 7. Integration readiness verdict

### Can connect now (experimental path only)

- **Member 2 SOH (NMC)** → any consumer that understands `status`, reads `predicted_soh_pct` only when `ok_experimental`, and surfaces `quality.warnings` verbatim.
- **Member 2 SOH (LFP)** → same, with `CONDITIONAL_LABELS` and the gap-crossing horizon caveat propagated.
- **Member 2 contract adapter** → stable typed payload for Member 4's orchestrator; it replaces the two silent hazards in §2.4 with hard errors.

### Remains experimental (do not promote)

- `nmc_lgbm`, `lfp_lgbm`: worse than persistence on their own holdouts.
- `nmc_bilstm`: worse than persistence, and not served by the API at all.
- `nmc_rul_lgbm`: no pre-EOL evidence; suppressed by the contract.

### Smallest remaining blockers to true end-to-end integration

1. **Member 3 source does not exist.** The `RangeCalculator` signature, its energy model and its input provenance cannot be verified. Nothing in §4.3 is confirmed against real code.
2. **Member 3 vehicle inputs have no sourced values** — pack usable kWh, mass, Cd, area, Crr, efficiency, regen, drive cycle, ambient, HVAC, validation range.
3. **SOH quality:** the SOH input to any range model is currently less accurate than `SOH_pred = SOH_current`; using it as an anchor adds noise rather than information.
4. **Upstream SOH measurement:** `current_soh_pct` is a measured label, not a telemetry feature — no BMS capacity estimate or reference test exists in this project, so real-time inference has no starting input.
5. **RUL absent:** any range or charging policy that depends on remaining life has no approved input.
6. **Member 1 dependency:** validated B2/B6 (and wider LFP coverage) would give more than one LFP trajectory, allowing a real adjacent-cycle test split instead of the current single-cell, adjacent-pairs-only LFP evidence. Until then, LFP results stay conditional.
7. **No uncertainty, no calibration, no OOD detection** — required before any user-facing range number.

---

## 8. Ownership

**Member 1 (data):** canonical dataset provenance; B2/B6 validation; cycle-ID gap documentation; label-quality statements (charge balance, null labels); the guarantee that the Parquet digest is the one trained against.

**Member 2 (this audit):** feature sets and units; model artifacts and their status; `Member2Predictor` behavior; battery-health contract draft; adapter guards; per-chemistry support matrix; SOH/RUL caveats; the handoff payload and its blocked states. **Not** responsible for range, energy conversion, vehicle parameters or Member 3's signature.

**Member 3 (range):** `RangeCalculator` implementation and its real signature; pack energy (nominal and usable) with provenance; vehicle parameters and their units; drive cycle, ambient and HVAC scenarios; linear-SOH-scaling assumption validation; vehicle-level range/consumption validation; any error propagation back to Member 2.

**Member 4 / wider team (orchestration):** wiring, schema-version pinning, surfacing `production_ready:false` and warnings in any UI, deciding whether a persistence baseline is the primary gating signal, and the one-record purge/embargo at split boundaries if formal evaluation is re-run.

---

## 9. Outstanding dependency

Member 1 is still investigating B2/B6. **Nothing in this audit waits on it**, and no model was promoted or retrained pending new data. When validated B2/B6 trajectories land:

- LFP evidence could move from one cell to several, enabling a test split that actually contains gap-crossing pairs.
- The conditional-label caveat should be re-examined rather than carried forward by default.
- Any resulting model change is a separate, approved piece of work — not part of this readiness audit.
