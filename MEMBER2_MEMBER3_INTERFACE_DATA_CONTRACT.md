# MEMBER 2 → MEMBER 3 INTERFACE DATA CONTRACT

**Project:** Autonomous Multi-Agent EV Battery Life & Range Predictor
**From:** Member 2 (SOH / capacity model)
**To:** Member 3 (EV range calculator)
**Version:** 1.0 (2026-10-29)
**Owner of this contract:** Member 1 (data definitions), implemented by Member 2.

---

## 1. Purpose

Define exactly what Member 2 may send to Member 3, in what units, with what provenance, and which values Member 3 must refuse. The contract prevents a cell-level amp-hour estimate from being silently treated as pack energy or as a range guarantee.

## 2. Required prediction record

One record per prediction:

```json
{
  "schema_version": "1.0",
  "dataset_id": "LFP_MultiStat_B2",
  "cell_id": "BATTERY_2",
  "chemistry": "LFP",
  "cycle_id": 1000,
  "cycle_time_s": 1735000000,
  "soh_pct": 91.4,
  "capacity_ah": 1.1425,
  "capacity_reference_ah": 1.25,
  "capacity_reference_type": "NOMINAL_RATED",
  "capacity_status": "CONDITIONAL",
  "model_version": "member2-soh-v1",
  "training_scope": "LFP B2; 1,496 conditional cycles",
  "uncertainty_pct": 1.8,
  "quality_flag": "CHANNEL_TRANSITION",
  "supported": true
}
```

### Field rules

| Field | Type | Rule |
|---|---|---|
| `schema_version` | string | `"1.0"` |
| `dataset_id` | enum | B1–B6, NASA, Na-ion |
| `cell_id` | string | Source cell id, never a pack id |
| `chemistry` | enum | `LFP`, `NMC`, `Na-ion` |
| `cycle_id` | integer | Source cycle identity, not row index |
| `cycle_time_s` | integer / null | Unix seconds if known, else null |
| `soh_pct` | float | **0–100**, not 0–1 |
| `capacity_ah` | float / null | Cell amp-hours; null if no verified reference |
| `capacity_reference_ah` | float / null | Denominator used |
| `capacity_reference_type` | enum | `NOMINAL_RATED`, `MEASURED_BOL`, `FIRST_OBSERVED`, `NONE` |
| `capacity_status` | enum | `VERIFIED`, `CONDITIONAL`, `BLOCKED` |
| `model_version` | string | Immutable tag |
| `training_scope` | string | Datasets/cycles used |
| `uncertainty_pct` | float / null | Prediction interval half-width in SOH points |
| `quality_flag` | string | Inherited or model-added flags |
| `supported` | bool | False when the model has no valid basis |

## 3. Hard constraints

1. **SOH is always 0–100 %.** B2's internal `soh_rated` fraction must be multiplied by 100 before leaving Member 2.
2. **Capacity in Ah is a cell-level quantity.** It is not pack energy and not range.
3. **No capacity without a reference.** If `capacity_reference_type = NONE`, `capacity_ah` must be null.
4. **B4 capacity/SOH is BLOCKED.** Member 2 must not emit a B4 target prediction as validated. If an exploratory estimate is emitted, `supported=false`, `capacity_status=BLOCKED`, and Member 3 must not use it in range.
5. **B5 capacity is not ground truth.** `capacity_status=CONDITIONAL`; never label it verified.
6. **LFP temperature is unmeasured.** Do not pass imputed temperature as measured. Any thermal correction must be labelled `model_assumption`.
7. **Do not extrapolate silently.** Predictions outside the training cycle range or beyond the last observed cycle must carry `extrapolation=true` and are not valid for range guarantees.
8. **Do not mix chemistries in one trajectory.** Each record carries chemistry; Member 3 must not average SOH across chemistries.

## 4. Cell Ah → pack energy (Member 3 responsibility)

Member 3 must apply an explicit pack model. Minimum required inputs:

- `cells_series` (S)
- `cells_parallel` (P)
- `nominal_cell_voltage_v`
- `pack_overhead_factor` (cooling, BMS, inverter), documented
- `usable_soc_window` (e.g., 0.10–0.90)

```text
pack_energy_kwh = soh_fraction
                × capacity_ah
                × cells_parallel
                × nominal_cell_voltage_v
                × cells_series
                × usable_soc_window
                × pack_overhead_factor
                / 1000
```

`soh_fraction = soh_pct / 100`.

Example: a 1.25 Ah LFP cell at 91.4 % SOH in a 4S2P pack at 3.2 V nominal, 80 % usable window, 0.90 overhead:
`0.914 × 1.25 × 2 × 3.2 × 4 × 0.8 × 0.9 / 1000 ≈ 0.0211 kWh`.

This is cell-chemistry energy, not measured vehicle consumption. Range additionally needs Wh/km, temperature, aux load, and terrain — Member 3's model.

## 5. Consumption and range

```text
range_km = usable_pack_energy_kwh / consumption_kwh_per_km
```

Rules:
- `consumption_kwh_per_km` must come from a documented source and carry its own uncertainty.
- Do not use a single constant across seasons/chemistries without a stated assumption.
- Report range as a distribution or interval, never a single guaranteed number.
- If `supported=false` or `capacity_status=BLOCKED`, return `range_status="unsupported"` and no number.

## 6. Traceability

Every Member 3 output must include:

- `model_version` (Member 2) and `range_model_version` (Member 3)
- `dataset_id`, `cell_id`, `cycle_id`
- `capacity_reference_type`
- `pack_config` (S, P, nominal voltage, usable window)
- `consumption_source`
- `quality_flag`
- `assumptions` list

## 7. Refusal cases

Member 3 must refuse to produce a range when:

- `supported=false`
- `soh_pct` is null or outside 0–100
- `capacity_ah` is null with `capacity_reference_type=NONE`
- `capacity_status=BLOCKED`
- chemistry is unknown
- pack configuration is missing
- prediction is flagged `extrapolation=true` for a guarantee-grade use

Refusal output:

```json
{"range_status": "unsupported", "reason": "capacity_status=BLOCKED (B4)", "quality_flag": "TARGET_BLOCKED_NO_AUDITED_TARGET"}
```

## 8. Versioning and change control

- Bump `schema_version` for breaking changes; keep backward-compatible fields additive.
- Never rename `soh_pct` or change its scale.
- Member 2 and Member 3 must both reject records whose `schema_version` is unknown.
- This contract does not authorise modifying Member 1 data artefacts or the canonical master table.
