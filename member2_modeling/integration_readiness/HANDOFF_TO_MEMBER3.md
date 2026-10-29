# Member 2 → Member 3 Handoff: Battery-Health Data Sufficiency

**From:** Member 2 (SOH prediction / RUL estimation)
**To:** Member 3 (range estimation)
**Contract:** `battery_health_contract_v1_1_0_draft.json` → `contract_version = "1.1.0-draft"`
**Date:** 2026-10-29
**Status of every field below:** experimental. `production_ready` is hard-coded `false`.

This document answers one question: **is what Member 2 hands over enough for Member 3 to do its work?**

Short answer: **enough for the battery-state input, not enough for a validated range result, and not enough on its own for Member 3 to start at all** — because Member 3's own vehicle and pack inputs do not exist in this repository and the `RangeCalculator` signature is unverified.

---

## 1. Scope

Member 2 provides battery health. Member 3 provides the vehicle, the pack energy and the drive scenario, and owns every range calculation. This handoff deliberately contains **no pack energy, no kWh, no consumption and no range value**; those are Member 3 outputs.

`RangeCalculator` source does not exist in this repository, in any git ref, or in the sibling project directory. The four-argument form used in §4 is **provisional** and every response carries `"signature_verified": false`.

---

## 2. What Member 2 supplies (complete field list)

Every field is always present. `null` means "not available", never zero and never a guess.

### 2.1 `subject` — identity

| Field | Type | Unit | Example | Source | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `chemistry` | string \| null | — | `"NMC"` | Caller-declared | Echoed, not detected from data. Flag `CHEMISTRY_DECLARED_BY_CALLER` always set. |
| `cell_id` | string \| null | — | `"B0005"` | Caller | Required for a valid result; non-empty string. |
| `dataset_id` | string \| null | — | `"NASA_NMC"` | Caller | Required for a valid result; non-empty string. |
| `cycle_id` | integer \| null | observed cycle record | `42` | Caller | **May be non-contiguous** (LFP B5: 1–100, 151–200, 401–500). Do not use arithmetic on it. |
| `cycle_position` | integer \| null | 0-based index | `41` | Caller (optional) | Contiguous position in the cell's sorted observed-cycle list. Supply it if Member 3 needs a monotone cycle counter. `null` if not supplied. |
| `prediction_timestamp` | string \| null | ISO-8601 | `"2026-10-29T00:00:00Z"` | Caller | Echoed verbatim. |

### 2.2 `soh` — state of health

| Field | Type | Unit | Example | Provenance | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `status` | enum | — | `"ok"` | — | `ok` \| `unavailable`. |
| `predicted_soh_pct` | number \| null | % (0–100) | `74.9645` | `nmc_lgbm` / `lfp_lgbm` | SOH of the **next observed record**. Null unless a model actually scored. |
| `current_observed_soh_pct` | number \| null | % (0–100) | `69.3056` | Feature `current_soh_pct` (measured capacity ratio) | The measured SOH of the **current** record. Carrying it forward *is* the persistence baseline. Null only if unavailable. |
| `unit` | const | — | `"%"` | — | — |
| `horizon.value` | integer \| null | observed records | `1` | — | Only 1 is supported. |
| `horizon.unit` | const | — | `"observed_cycle_records"` | — | Explicitly **not** physical cycles. |
| `horizon.semantics` | string | — | see example | — | States the positional shift and the LFP gaps. |
| `model_id` | string \| null | — | `"nmc_lgbm"` | Code | — |
| `model_status` | string \| null | — | `"experimental_research_stage"` | Manifest | — |
| `validation_status` | string | — | `"research_stage_not_production_ready"` | Manifest | LFP: `"research_stage_conditional_labels_not_production_ready"`. |
| `uncertainty` | object | — | `{"status":"not_provided"}` | — | **No interval exists.** Do not invent one. |

What `74.9645` vs `69.3056` means: the model predicts the next cycle is *higher* than the current measured SOH, while the reported holdout says this model has **worse** error than simply carrying `69.3056` forward (NMC MAE 7.968 vs persistence 0.504, n=95; LFP 0.926 vs 0.131, n=51). A range consumer should prefer the anchor or a gated blend, not the raw prediction.

#### Operational prerequisite: `current_soh_pct` is not derivable from telemetry

Verified against the canonical parquet on 2026-10-29: `extract_cycle_features()` emits **36 of the 37** NMC features and **28 of the 29** LFP features. The missing one is `current_soh_pct`, which `build_and_train.py` injects from the measured SOH label (`inputs["current_soh_pct"] = inputs["soh_pct"]`). It is a **measured quantity**, not a signal computed from the discharge curve.

Consequence for the whole chain: a raw discharge record alone **cannot** start Member 2. Somebody must supply the current SOH of that cycle — from a BMS capacity estimate, a reference charge/discharge test, or an upstream label. When it is absent the contract returns `status="error"`, `code="MISSING_FEATURES"`, message naming `current_soh_pct`; there is no default and no imputation. This is the real operational boundary between Member 3's telemetry and Member 2's model.

### 2.3 `rul` — remaining useful life

Always `{"status":"unavailable","rul_cycles":null,"approved_for_use":false,...}`.

- NMC: `reason="UNVALIDATED_PRE_EOL"`, `model_id="nmc_rul_lgbm"` — 97/97 chronological holdout targets were already post-80 % EOL.
- LFP: `reason="CHEMISTRY_UNSUPPORTED"` — observed minimum SOH is 90.91 %, EOL is never reached.
- Na-ion: `reason="CHEMISTRY_UNSUPPORTED"`.

Member 3 must not build any range or charging logic that depends on RUL.

### 2.4 `capacity` — opt-in, cell-level

Present only when `include_capacity=True`:

```json
{"capacity_ah": 1.4993, "unit": "Ah", "reference_capacity_ah": 2.0,
 "basis": "derived_soh_pct_times_rated_capacity",
 "validation_status": "not_independently_validated", "is_pack_energy": false}
```

This is a **single-cell** quantity (2.0 Ah NMC cell, 1.25 Ah LFP cell) derived algebraically from SOH. It is not measured and it is not pack energy. **Never** multiply it by a guessed cell count, never convert it to kWh, never treat it as a pack rating.

### 2.5 `quality` — trust signals

| Field | Meaning |
| :--- | :--- |
| `quality_flags` | Always includes `MODEL_EXPERIMENTAL_RESEARCH_STAGE`, `MODEL_UNDERPERFORMS_PERSISTENCE`, `NO_UNCERTAINTY_QUANTIFICATION`, `NO_OUT_OF_DISTRIBUTION_CHECK`, `SOH_HORIZON_OBSERVED_RECORD_NOT_PHYSICAL_CYCLE`, `CURRENT_SOH_IS_UPSTREAM_LABEL`, `RUL_NOT_APPROVED`, `CHEMISTRY_DECLARED_BY_CALLER`. LFP adds `CONDITIONAL_LABELS`. Na-ion adds `CHEMISTRY_UNSUPPORTED` and `NAION_INFERENCE_UNSUPPORTED`. |
| `warnings` | Human-readable sentences, including the exact holdout comparison. Propagate verbatim; do not summarise away the caveat. |
| `checks_passed` | `provenance_supplied`, `feature_set_matches_schema`, `feature_values_finite`, `soh_output_within_0_100`, `model_loaded_and_scored`. |
| `checks_not_performed` | `out_of_distribution`, `sensor_calibration`, `label_freshness`, `uncertainty_calibration`, `vehicle_level_range_validation`. |
| `label_provenance` | `{"soh_labels":"measured_capacity_ratio","status":"observed"\|"conditional"\|"not_applicable"}`. |

### 2.6 `source`, `errors`, `request`

- `source`: `module`, `function` (`Member2Predictor.predict_future_soh`), `schema_version` (`1.0.0`), `model_file`.
- `errors`: empty on success; otherwise `{code, message, field?}` with codes `UNSUPPORTED_CHEMISTRY`, `UNSUPPORTED_HORIZON`, `INVALID_PROVENANCE`, `MISSING_PROVENANCE`, `MISSING_FEATURES`, `INVALID_FEATURE_VALUES`, `FEATURE_SET_MISMATCH`, `INVALID_SOH_OUTPUT`, `PREDICTION_EXCEPTION`, `MODEL_SCHEMA_UNAVAILABLE`.
- `request`: `{"horizon_cycles": 1}`.

---

## 3. Sufficiency audit against Member 3's needs

Legend: **OK** supplied · **CAVEAT** supplied but not fit for a validated result · **MISSING** not in the repository at all · **N/A** intentionally withheld.

| Member 3 requirement | Supplied? | Where / who | Verdict |
| :--- | :---: | :--- | :--- |
| SOH of the battery (%) | Yes | `soh.predicted_soh_pct` | **CAVEAT** — point estimate; worse than persistence on every reported holdout |
| Current-cycle SOH **input** required by the model | Yes, but from upstream | Feature `current_soh_pct`, echoed as `soh.current_observed_soh_pct` | **CAVEAT** — measured quantity; not derivable from telemetry, so an upstream SOH source is mandatory |
| Current measured SOH anchor (%) | Yes (new in 1.1.0) | `soh.current_observed_soh_pct` | **OK** as an anchor; still a label, not a live sensor |
| SOH uncertainty / confidence interval | No | `soh.uncertainty.status = "not_provided"` | **MISSING** — no calibration exists; must be treated as a single point |
| Chemistry | Yes | `subject.chemistry` | **OK** — but caller-declared, not inferred |
| Cell / dataset identity | Yes | `subject.cell_id`, `subject.dataset_id` | **OK** |
| Cycle identity, contiguous counter | Yes if caller supplies `cycle_position` | `subject.cycle_id`, `subject.cycle_position` | **OK** with caller discipline; `cycle_id` alone is **not** a counter |
| Timestamp | Yes | `subject.prediction_timestamp` | **OK** — caller-supplied |
| Model identity and version | Yes | `soh.model_id`, `source.model_file`, `source.schema_version` | **OK** |
| Validation / readiness status | Yes | `soh.validation_status`, `soh.model_status`, `production_ready` | **OK** |
| Quality flags, warnings, label provenance | Yes | `quality.*` | **OK** |
| Explicit unsupported / error state | Yes | `status`, `errors`, `rul.reason` | **OK** |
| RUL (cycles to 80 %) | No, by design | `rul.status="unavailable"` | **N/A** — cannot be used |
| Cell capacity (Ah) | Opt-in | `capacity.capacity_ah` | **CAVEAT** — derived, not measured |
| Pack nominal / usable energy (kWh) | No | — | **MISSING** — Member 3 / wider team |
| Vehicle mass, Cd, frontal area, Crr, drivetrain efficiency, regen assumption | No | — | **MISSING** — Member 3 |
| Drive cycle speed/time profile, road grade | No | — | **MISSING** — Member 3 |
| Ambient temperature, HVAC/auxiliary load | No | — | **MISSING** — Member 3 scenario; the two arguments Member 2 forwards, never invents |
| Published/measured consumption or range for validation | No | — | **MISSING** — wider team |
| Real `RangeCalculator` signature | No | — | **MISSING** — source absent; provisional form only |

Where the missing items live as *requirements only* (no values): `vehicle_inputs_required.json`. Every entry has `"status":"not_supplied"` and no `"value"`. A placeholder in a config file is not evidence and must not be cited as one.

### 3.1 Verdict

1. **As a battery-state input: sufficient in shape.** Member 3 can receive a bounded, provenance-carrying SOH with an explicit anchor, full quality flags and hard blocking on every unsupported case. Nothing has to be guessed or parsed from prose.
2. **As a validated range input: not sufficient.** The SOH signal is less accurate than persistence, carries no uncertainty, and has no vehicle-level validation. The range result will inherit that.
3. **As a complete start signal: not sufficient.** Member 3's own inputs (pack energy, vehicle, scenario, validation data) are absent, and the interface it will expose is unverified.
4. **Operationally: an upstream current-SOH measurement is mandatory.** The chain starts from `current_soh_pct`, which telemetry alone does not provide.
5. **Chemistries:** NMC and LFP only. Na-ion is explicitly blocked and must stay blocked in any Member 3 path.

---

## 4. Exact call mapping (provisional)

```python
from member2_modeling.predictor import Member2Predictor
from member2_modeling.integration_readiness import build_health_result, build_range_arguments

predictor = Member2Predictor("member2_modeling/")

contract = build_health_result(
    predictor,
    features,                 # 37 NMC / 29 LFP keys, exactly the schema set
    chemistry="NMC",          # "NMC" | "LFP"; anything else is blocked
    provenance={              # all strings non-empty, cycle_id integer
        "cell_id": "B0005",
        "dataset_id": "NASA_NMC",
        "cycle_id": 42,
        "cycle_position": 41,          # optional but recommended
        "prediction_timestamp": "2026-10-29T00:00:00Z",
    },
    include_capacity=False,   # True only if Member 3 explicitly wants cell Ah
)

handoff = build_range_arguments(
    contract,
    ambient_c=25.0,           # Member 3 scenario input, °C
    hvac_power_kw=0.5,        # Member 3 scenario input, kW, >= 0
)

if handoff["status"] != "ok_experimental":
    handle_blocked(handoff)   # reason: SOH_UNAVAILABLE | INVALID_SOH | MISSING_INPUTS | INVALID_SCENARIO_INPUT
else:
    arguments = handoff["arguments"]   # exactly four keys
    # Member 3 owns everything after this line.
```

Actual `handoff` for the example above:

```json
{
  "status": "ok_experimental",
  "arguments": {"soh_pct": 74.9645, "cycle": 42, "ambient_c": 25.0, "hvac_power_kw": 0.5},
  "missing": [],
  "signature_verified": false,
  "signature_note": "Provisional four-argument form ... Member 3 range_calculator.py is not present in the repository or git history, so the real signature is unverified.",
  "cycle_note": "cycle is the observed cycle_id passed through unchanged. It is not a contiguous counter ... read contract.subject.cycle_position ...",
  "cycle_semantics": "observed_cycle_id",
  "soh_semantics": "soh_pct is the predicted SOH of the next observed record. contract.soh.current_observed_soh_pct is the measured current SOH (the persistence baseline) ...",
  "assumption": "SOH linearly scales usable pack energy. This is an approximation; the resulting range is unvalidated until checked against vehicle-specific evidence.",
  "production_ready": false
}
```

#### Verified on real cycle records (not just the manifest sample)

Feature extraction was re-run on the canonical parquet for four real records, with `current_soh_pct` supplied from the measured label, and pushed through the same adapter:

| Cell | Chemistry | Cycle (observed → next) | `current_observed_soh_pct` | Predicted next SOH | Actual next SOH | Contract status |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| B0005 | NMC | 100 → 101 | 74.2934 | 74.6422 | 74.0207 | `ok_experimental` |
| B0018 | NMC | 90 → 91 | 70.7727 | 73.7128 | 72.7288 | `ok_experimental` |
| BATTERY_5 | LFP | 499 → 500 | 91.0017 | 92.4233 | 91.0670 | `ok_experimental` |
| BATTERY_5 | LFP | 100 → 151 | 93.6971 | 93.0258 | 92.7636 | `ok_experimental` |

These values illustrate scale and prove plumbing, **not accuracy**: two predictions beat the anchor, two are worse, which is exactly the instability the reported holdout summarises. The last row is the gap-crossing 100 → 151 transition and must not be read as a one-cycle forecast. The same four cases are asserted in `test_integration_readiness.py::RealDataHandoffTests`.

Open interface questions Member 3 must answer before the mapping is final:1. Does `estimate` really take `soh_pct, cycle, ambient_c, hvac_power_kw`? (Unverified — no source exists.)
2. Does `cycle` mean an observed cycle ID, a contiguous cycle count, or an age in cycles? If contiguous, Member 3 must read `subject.cycle_position`.
3. Does the calculator expect SOH fraction (0–1) or percent (0–100)? Member 2 supplies percent.
4. Is pack energy already inside the calculator (derived from a vehicle config), or must it be passed? Member 2 does not supply it either way.

---

## 5. Blocked states Member 3 will see

| Situation | `handoff` | Correct behaviour |
| :--- | :--- | :--- |
| Na-ion / unknown chemistry | `blocked`, `SOH_UNAVAILABLE` | No range. Report unsupported. |
| Invalid or absent SOH | `blocked`, `INVALID_SOH` | Fail loudly; never treat null as 0 %. |
| Missing `ambient_c` / `hvac_power_kw` | `blocked`, `MISSING_INPUTS`, `missing` lists them | Ask for the scenario input; do not silently default. |
| Non-numeric / negative HVAC | `blocked`, `INVALID_SCENARIO_INPUT` | Reject the scenario. |
| RUL needed by the design | `rul.status="unavailable"` | The design must not depend on RUL. |
| Successful experimental call | `ok_experimental`, `production_ready:false` | May compute a labelled experimental range only. |

---

## 6. What Member 3 must supply before this handoff is "sufficient"

Checklist, all required, each with a cited source:

- [ ] Real `RangeCalculator` implementation (file path + signature) checked into the repo.
- [ ] An upstream source of `current_soh_pct` for the current cycle (BMS capacity estimate, reference test, or label), with its accuracy stated.
- [ ] Pack nominal energy (kWh) and usable energy (kWh) or usable fraction, with source.
- [ ] Vehicle identity, mass (kg), Cd, frontal area (m²), Crr, drivetrain efficiency, regen assumption, each with source/units.
- [ ] Drive cycle time (s) and speed (m/s) time series with attribution; road grade or a documented flat-road assumption.
- [ ] Ambient temperature (°C) and HVAC/auxiliary load (kW) scenarios, measured or documented.
- [ ] Consumption and range validation data (kWh/km, km) for at least one real vehicle/cycle pair.
- [ ] A documented decision on anchoring: raw predicted SOH, current observed SOH, or a gated blend — with the persistence baseline reported alongside.
- [ ] Confirmation that no path converts cell Ah to pack kWh.
- [ ] Any emitted range labelled experimental, with `quality.warnings` propagated verbatim.

Until every box is ticked, Member 2's data is a **sufficient input** but the **integration is not sufficient** to produce a defensible range.

---

## 7. Ownership

| Item | Owner |
| :--- | :--- |
| SOH model, feature schema, caveats, quality flags | Member 2 |
| Contract shape, validation, blocked states, cycle/capacity semantics | Member 2 |
| Pack energy, vehicle parameters, drive cycle, ambient/HVAC scenario | Member 3 |
| Range computation and its validation | Member 3 |
| Real `estimate()` signature and its documentation | Member 3 |
| Canonical dataset provenance, LFP B5 label caveats, future B2/B6 data | Member 1 |
| Orchestration, schema pinning, surfacing warnings | Member 4 / wider team |

---

## 8. Change log

**1.1.0-draft (current)** — added `subject.cycle_position` and `soh.current_observed_soh_pct`; subject values are now sanitised so error responses remain schema-valid; handoff carries `cycle_semantics` and `soh_semantics`.

**1.0.0-draft** — first draft: identity, SOH, RUL, capacity, quality, errors, source, request; hard-blocked RUL; strict feature-set guard.
