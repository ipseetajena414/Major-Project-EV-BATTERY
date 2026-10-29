# Member 3 — Data Collection Checklist

Collect these items before making vehicle-specific claims. Keep source URL/document, retrieval date, units, and whether each value is measured, manufacturer-specified, or assumed.

## A. Vehicle configuration (required for a meaningful range estimate)
- [ ] Vehicle make/model/year OR a clearly labelled representative vehicle
- [ ] Nominal battery pack energy (kWh)
- [ ] Usable battery energy (kWh), preferably manufacturer-specified
- [ ] Rated range (km) and test standard (EPA/WLTP/other)
- [ ] Vehicle mass including representative payload (kg)
- [ ] Drag coefficient Cd (dimensionless)
- [ ] Frontal area A (m²)
- [ ] Rolling resistance coefficient Crr (dimensionless), with source/assumption
- [ ] Drivetrain efficiency or motor/inverter efficiency map
- [ ] Maximum regenerative braking power / regen efficiency (if modelled)
- [ ] Auxiliary load and HVAC power assumptions (kW)

## B. Drive cycle (required)
- [ ] Speed vs time (seconds, m/s or km/h)
- [ ] Distance or enough speed/time points to integrate distance
- [ ] Road grade/slope vs time if available; otherwise state flat-road assumption
- [ ] Cycle source, licence, and attribution
- [ ] Use UDDS/HWFET or WLTP data only after obtaining the official file and recording source/version. Do not rename synthetic data as a standard cycle.
- [ ] Note: the shipped `udds_city.csv`/`hwfet_highway.csv` match the EPA trace format but have **no recorded attribution or version — provenance pending verification**.

## C. Battery-health interface
- [ ] Member 1 `master_table.parquet` and its schema/quality flags (SHA256 `6fa7475d…b2bb` for the audited copy)
- [ ] Member 2 prediction CSV with required columns `cycle_index, cell_id, chemistry, soh_pct`; `unit` must be `%` when present; chemistry must be `NMC` or `LFP`
- [ ] Confirm SOH scale (adapter accepts percentage points only, `1.0 < soh <= 100.0`, and rejects 0–1 fractions)
- [ ] Confirm whether capacity is cell-level or pack-level (current artifacts are **cell-level**)
- [ ] Record `model_name`, `model_status`, `split`, `horizon_cycles` and `window_end_cycle_id` from the artifact
- [ ] Pack series/parallel configuration and usable-energy mapping, if converting cell data to vehicle pack energy (currently a linear approximation, not a pack model)
- [ ] Preserve dataset_id, cell_id, protocol_id, source_file and quality flags (adapter preserves cell_id and quality_flag verbatim)
- [ ] Exclude blocked/unverified capacity and SOH from ground-truth evaluation
- [ ] Do not label synthetic or hand-written trajectories `BOL_MEASURED`/`LSTM_PREDICTED`; use `SYNTHETIC_TEST_ONLY` unless supported by measurements or actual inference output

## D. Weather / HVAC scenarios
- [ ] Ambient temperature (°C)
- [ ] HVAC power (kW) measured or documented assumption for each scenario
- [ ] Any cabin preconditioning or heat-pump assumptions
- [ ] Duration of scenario / drive cycle

## E. Validation data (strongly recommended)
- [ ] Published vehicle energy consumption for the same vehicle and drive cycle
- [ ] Measured trip energy and distance, if available
- [ ] Baseline range and test standard
- [ ] Error metric: MAE, percentage error, or range interval

## File formats suggested
- `vehicle_config.json`: vehicle parameters and provenance notes
- `drive_cycles/<cycle_name>.csv`: `time_s,speed_mps,grade_pct`
- `battery_health.csv`: `timestamp_or_cycle,soh_pct,capacity_ah,quality_flag,source_file`
- `validation_trips.csv`: `trip_id,distance_km,energy_kwh,ambient_c,hvac_kw,source`

## F. Current repository status (audited 2026-10-29)

Profiles in `member3/vehicle_configs/`. Status is **verified from file contents**, not from successful loading. All eight reference profiles carry `PARTIALLY_SOURCED_UNCALIBRATED`.

| Profile | Pack kWh | Rated range (cycle) | Sourced from | Still provisional |
| :--- | :--- | :--- | :--- | :--- |
| `tata_curvv_ev_55kwh_partial_reference.json` | 55 | 502 km (cycle not stated) | ev.tata.cars OEM page | All dynamics, efficiencies, usable fraction, HVAC/aux |
| `tata_nexon_ev_45kwh_partial_reference.json` | 45 | 489 km (ARAI/MIDC) | ev.tata.cars OEM page | Same |
| `tata_punch_ev_40kwh_partial_reference.json` | 40 | 468 km (ARAI) | ev.tata.cars OEM page | Same |
| `byd_atto3_gl_49_92kwh_partial_reference.json` | 49.92 | 400 km (NEDC, secondary source) | BYD India OEM pack spec; Wikipedia for range | Same |
| `mahindra_xev9e_59kwh_partial_reference.json` | 59 | 542 km (MIDC/ARAI) | Mahindra OEM brochure | Same |
| `mg4_ev_51kwh_partial_reference.json` | 51 | 350 km claimed (cycle unstated, secondary source) | MG global spec sheet; zecar.com for range | Same; placeholder 2100 kg mass is oversized for this class |
| `hyundai_ioniq5_84kwh_partial_reference.json` | 84 | 690 km (ARAI/MIDC) | hyundai.com/in OEM page | Same |
| `tesla_model_y_lr_78kwh_partial_reference.json` | ≈78 nominal / ≈75 usable | 600 km (WLTP; 586 km in one secondary source) | Wikipedia (Cd 0.23, range, pack) | Same; usable fraction ≈0.96 is derived, not manufacturer-confirmed |
| `illustrative_placeholder.json` | 75 | 650 km (illustrative only) | — | Everything |

Drive cycles ingested so far: `example_cycle.csv` (synthetic), `udds_city.csv` and `hwfet_highway.csv` (converted from `drive_cycles/source/UDDS.txt` and `HWFET.txt`; format consistent with EPA UDDS/HWFET but **provenance pending verification**). WLTP Class 3 is still outstanding.

Member 2 prediction artifacts present: `integration/predictions/member2_nmc_B0005_bilstm_predictions.csv` and `..._lgbm_predictions.csv` (30 rows each, cell B0005, NASA PCoE, NMC, cycles 139–168, Member 2 commit `b2a2b51`). Provenance, artifact SHA256s, reproduced metrics and caveats are in `integration/predictions/PROVENANCE.json`. Both models underperform a persistence baseline on Member 2's holdout; LFP and Na-ion trajectories have not been produced.

Fill the checkboxes below per vehicle **only when the value is genuinely sourced**; otherwise leave it unchecked and leave the JSON value labelled as an assumption.

## Rules
1. Never invent manufacturer specifications or claim assumptions are measured.
2. Keep missing temperatures as null when they were not measured in the battery dataset.
3. Cell-level LFP/NMC/Na-ion data do not automatically determine a real vehicle pack's usable kWh; pack mapping is a separate assumption/model.
4. Treat the 650 km baseline as a project reference unless tied to a named vehicle and standard.
5. Report uncertainty and limitations, not just a single range number.
6. A configuration that loads and produces a range estimate is **not** validated. Validation requires sourced dynamics, calibration against a certified cycle, and a documented error metric.
7. Do not force simulated range to match an expected figure (e.g. 650 → 550 km). Report the model's output with its quality flags and disagreement against the baseline.
8. Prediction quality flags must be propagated from Member 2 to Member 3 output; never upgrade a flag without evidence.
