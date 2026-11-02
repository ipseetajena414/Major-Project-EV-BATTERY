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
- [ ] Since Phase 4 that pending status is enforced by code, not only by this note: `scenario_engine.inspect_cycle()` audits each trace (rows, duration, integrated distance, sampling interval, speed span, grade span, SHA-256 of the CSV and of its `drive_cycles/source/` file) and every experiment manifest carries the label. Current recorded state:

| File | Duration | Distance | Sampling | Max speed | Label recorded by the code |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `drive_cycles/udds_city.csv` | 1369 s | 11.990 km | uniform 1 Hz | 25.347 m/s | `UNVERIFIED_TRANSCRIBED_CYCLE` |
| `drive_cycles/hwfet_highway.csv` | 765 s | 16.507 km | uniform 1 Hz | 26.778 m/s | `UNVERIFIED_TRANSCRIBED_CYCLE` |
| `drive_cycles/example_cycle.csv` | 60 s | 0.270 km | uniform 10 s spacing (0.1 Hz) | 8.000 m/s | `ILLUSTRATIVE_SYNTHETIC_CYCLE` (smoke test; a trace supplied by path and not registered is labelled `USER_SUPPLIED_CYCLE`) |

- [ ] Closing condition for the two unverified labels: obtain the official file, record publisher, version and licence, and match a publisher checksum. Until then the audits state that reproducing the published duration and distance is **corroboration, not certification**, and the HWFET file's internal header reading `HWYCOL.TXT` is reported as an unresolved naming discrepancy.
- [ ] WLTP remains absent. A missing official cycle is recorded as missing; no synthetic trace is substituted for it, and the generated mountain-pass and aggressive profiles are labelled `SYNTHETIC_SCENARIO` in every row, figure and manifest.
- [ ] The full record is generated, not typed: `python tools/audit_drive_cycles.py --out DRIVE_CYCLE_PROVENANCE_REPORT.md` rebuilds [DRIVE_CYCLE_PROVENANCE_REPORT.md](DRIVE_CYCLE_PROVENANCE_REPORT.md) with measured-versus-claimed dimensions, CSV and source hashes, the audit warnings, the scenario-to-cycle table and the promotion conditions.

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

Phase 3 can now consume a cabin thermal model instead of a single HVAC power figure. Every field below is optional and unset in the shipped profiles, so collecting it is what promotes a profile from the illustrative defaults; the values currently in `examples/hvac_cabin_illustrative_config.json` are placeholders, not measurements.
- [ ] `cabin_thermal_capacitance_j_per_k` (J/K) and `cabin_ua_w_per_k` (W/K) — both are recoverable from one recorded cabin soak/cool-down trace (initial slope gives UA·ΔT, the decay time constant gives C).
- [ ] `cabin_solar_load_w` (W) and `cabin_occupancy_load_w` (W) for a stated sun position and occupant count; the model holds them constant over a cycle.
- [ ] `heat_pump_cop_*` / `ac_cooling_cop_*` from a manufacturer or bench curve at named rating points (the shipped anchors sit on the EN 14511 **A7** 7 °C-air heating and **A35** 35 °C-air cooling *test conditions*; the standard fixes those conditions but no COP, so the anchor values and slopes here are illustrative assumptions, not vehicle data) and `ptc_heating_efficiency` (fraction, 0–1] if the vehicle uses resistive heat.
- [ ] `max_heating_electrical_kw` / `max_cooling_electrical_kw` from the fuse or inverter budget for the HVAC circuit.
- [ ] A measured cabin temperature trace and total HVAC energy for one run, to validate the model rather than only exercise it.
- [ ] Evidence for choosing a mode at all: which actuator this vehicle actually has, since `HEAT_PUMP`, `PTC` and `AC_COOLING` are mutually exclusive here.

## E. Validation data (strongly recommended)
- [ ] Published vehicle energy consumption for the same vehicle and drive cycle
- [ ] Measured trip energy and distance, if available
- [ ] Baseline range and test standard
- [ ] Error metric: MAE, percentage error, or range interval

Phase 4 automates the sweep but does not supply the missing validation data. `run_experiments.py` exercises the −10 °C / 20 °C / 38 °C ambient cases, the six scenarios, the four HVAC modes and the SOH cases (100 %, 82 %) against **one disclosed reference configuration**, and each experiment directory records the cycle and profile hashes, the matrix, the assumptions and the boundary statement. Two things it deliberately does not do: it does not sweep a chemistry axis (the engine has no chemistry-specific physics — SOH is a linear multiplier on usable energy, so a chemistry comparison would be a label over identical numbers), and it does not tune toward the 650 km figure. `rated_range_km_reference` is inert metadata; `reference_range_comparison.json` reports the residual gap (183.481 km to the best simulated scenario at the reference condition) as an `ILLUSTRATIVE OBJECTIVE, NOT A GUARANTEED RESULT`. Closing that gap needs the items above.

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

**Phase 3 cabin thermal status (2026-11-02):** none of the ten shipped profiles — the nine in `vehicle_configs/` plus the root `vehicle_config.json` — carries a single cabin parameter, so all of them run `hvac_model_mode = "EMPIRICAL"` and their results are exactly what they were before Phase 3. The thermal model's own inputs — capacitance, UA, solar and occupancy gains, COP curves, PTC efficiency, actuator power caps — are **entirely uncollected for every vehicle** and are supplied only as documented illustrative defaults or via `examples/hvac_cabin_illustrative_config.json`. No cabin number in this repository has come from a soak test, a climate chamber or a logged thermistor trace.

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
