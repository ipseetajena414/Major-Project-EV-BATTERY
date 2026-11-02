# Member 3 — RangeCalculator (Vehicle Dynamics & Range Derivation)

This module provides a **transparent, physics-based lumped-parameter vehicle road load and EV driving range calculation engine**. It computes aerodynamic drag, rolling resistance, inertial acceleration, and road-grade forces, integrated across drive cycles to estimate electrical consumption and driving range.

## Included Modules
- `range_calculator.py`: Interval-based road load integration, ideal-gas air density from temperature, station pressure or ISA altitude, steady along-track wind, empirical HVAC comfort model, single-node cabin thermal model with heat-pump/PTC/AC actuator modes, regenerative braking dynamics, hardened vehicle configuration validation, and trajectory evaluation.
- `vehicle_config.json`: Legacy illustrative baseline retained for backward compatibility (not a real vehicle).
- `vehicle_configs/`: One JSON profile per reference vehicle, each with explicit provenance. See [Vehicle Configuration Library](#vehicle-configuration-library).
- `examples/hvac_cabin_illustrative_config.json`: Separated illustrative Phase 3 cabin/HVAC configuration. It is deliberately **not** in `vehicle_configs/`, because the shipped reference profiles stay free of invented cabin parameters; every value in it is labelled an assumption.
- `examples/matrix_example.json`: Copyable Phase 4 experiment-matrix definition for `run_experiments.py --matrix`.
- `scenario_engine.py`: Phase 4 scenario library — drive-cycle inspection and provenance labelling, six deterministic scenarios (A–F), configurable experiment matrices, run execution through the existing `RangeCalculator.estimate()` API, attribution probes and the documented 56-column output schema. It contains **no second simulator**.
- `run_experiments.py`: Phase 4 experiment runner CLI. Writes one never-overwritten directory per experiment under `experiments/results/` with `scenario_results.csv`, `experiment_manifest.json`, `scenario_summary.json`, `scenario_phases.json`, optional `reference_range_comparison.json`, optional `figures/` and `SHA256SUMS.txt`. `experiment_manifest.json` is built for reproducibility: Python/implementation/platform, a sha256 for every source file that can move a number (`range_calculator.py`, `scenario_engine.py`, `run_experiments.py`, `tools/experiment_plots.py`, `tools/audit_drive_cycles.py`), hashes of the cycles, vehicle profiles and any supplied cabin configuration, the matrix recorded by value, and a git record that states whether the working tree was dirty instead of implying that HEAD covers uncommitted source.
- `tools/experiment_plots.py`: Phase 4 figure generator. Seven report figures, rendered with matplotlib when installed and with a standard-library SVG backend otherwise.
- `experiments/results/`: Generated experiment directories. Regenerate with the runner; never hand-edit.
- `tools/audit_vehicle_configs.py`: Regenerates `CONFIGURATION_VALIDATION_REPORT.md` from the shipped profiles (load/validate, provenance classification, plausibility bands, cross-profile placeholders, simulated vs certified-range consumption).
- `CONFIGURATION_VALIDATION_REPORT.md`: Phase 2 configuration audit evidence. Generated, not hand-written — rerun the tool above to refresh it.
- `tools/audit_drive_cycles.py`: Phase 4 cycle audit. Runs the same validation gates the runner uses, compares each trace against the totals its own file claims, re-hashes the CSV on disk, and records which scenario consumes it. Read-only.
- `DRIVE_CYCLE_PROVENANCE_REPORT.md`: Phase 4 drive-cycle provenance evidence. Generated, not hand-written — the test suite regenerates it and asserts it is byte-identical apart from the timestamp.
- `drive_cycles/example_cycle.csv`: Synthetic cycle for baseline validation and smoke testing.
- `drive_cycles/udds_city.csv`, `drive_cycles/hwfet_highway.csv`: supplied time-speed traces converted from `drive_cycles/source/` by `convert_drive_cycles.py`. Their content matches the EPA UDDS/HWFET format, but attribution and version are unrecorded — provenance pending verification. Phase 4 labels both as `UNVERIFIED_TRANSCRIBED_CYCLE`; `scenario_engine.inspect_cycle()` measures and hashes each one and reports how far it is from the published nominal totals.
- `integration/`: File-based Member 2 → Member 3 adapter, genuine Bi-LSTM/LightGBM prediction artifacts, provenance record and range-trajectory outputs. See `integration/README.md`.
- `tests/test_range_calculator.py`: 195 physics, API, environmental, thermal, numerical and regression tests (45 in `RangeCalculatorTests`, 32 in `RegenerativeBrakingIntegrationTests`, 25 in `VehicleConfigValidationTests`, 15 in `EnvironmentalModelTests`, 13 in `RelativeWindTests`, and 65 Phase 3 HVAC tests in `HvacModeSelectionTests`, `CabinThermalModelTests`, `HvacEnergyAccountingTests`, `CabinThermalNumericalStabilityTests`, `HvacTrajectoryAndCliTests`). `tests/test_member2_integration.py`: 18 adapter tests.
- `tests/test_scenario_engine.py`: 78 Phase 4 tests — cycle validation gates, provenance labels, deterministic synthetic traces, scenario labelling, matrix expansion, execution accounting, failure reporting, output schema, the generated drive-cycle provenance report, and the source-hash/version/dirty-tree reproducibility records. `tests/test_run_experiments.py`: 39 Phase 4 tests — output files, manifest provenance, checksum verification, per-source-file hashes in the manifest, never-overwrite semantics, exit codes, the 650 km reference comparison and the figures.
- `DATA_COLLECTION_CHECKLIST.md`: Formal checklist of required vehicle parameters, cycle sources, and validation data.
- `RUN_INSTRUCTIONS.md`: Execution instructions, CLI reference, and Member 1/2 integration workflows.
- `requirements.txt`: Standard library only; zero external third-party dependencies required. Matplotlib is an **optional** extra for PNG figures — without it the Phase 4 plots fall back to SVG.

## Quick Start
From this directory:

```bash
# Run baseline range calculation on example cycle
python range_calculator.py --config vehicle_config.json --cycle drive_cycles/example_cycle.csv --soh-pct 100 --ambient-c 25

# Run a reference vehicle profile on a UDDS trace
python range_calculator.py --config vehicle_configs/tata_curvv_ev_55kwh_partial_reference.json --cycle drive_cycles/udds_city.csv --soh-pct 92 --ambient-c 35

# Add environment detail: altitude (ISA pressure), station pressure, along-track wind
python range_calculator.py --config vehicle_configs/tata_nexon_ev_45kwh_partial_reference.json --cycle drive_cycles/hwfet_highway.csv --ambient-c 15 --altitude-m 1650 --headwind-mps 6

# Regenerate the configuration validation report
python tools/audit_vehicle_configs.py

# Cabin thermal model: heat pump at -5 C starting from an ambient-soaked cabin
python range_calculator.py --config examples/hvac_cabin_illustrative_config.json --cycle drive_cycles/example_cycle.csv --ambient-c -5

# Same profile, explicit constant COP and a cooling scenario at 40 C with solar gain
python range_calculator.py --config examples/hvac_cabin_illustrative_config.json --cycle drive_cycles/udds_city.csv --hvac-mode AC_COOLING --ambient-c 40 --cabin-initial-c 55 --solar-load-w 600 --cooling-cop 2.8

# The electrical override still wins over the thermal model
python range_calculator.py --config examples/hvac_cabin_illustrative_config.json --cycle drive_cycles/example_cycle.csv --hvac-mode HEAT_PUMP --hvac-kw 2.5

# Phase 4: list the cycles, scenarios, presets and the output schema
python run_experiments.py --list

# Run the main scenario matrix on the one disclosed reference configuration
python run_experiments.py --preset main --plots

# The 650 km reference comparison (reported as a gap, never tuned to)
python run_experiments.py --preset main --plots --reference-experiment

# One ad-hoc run: mountain pass, heat pump, 5 m/s headwind, 800 m altitude
python run_experiments.py --single --scenario E_MOUNTAIN_PASS --hvac-mode HEAT_PUMP \
  --headwind 5 --altitude 800 --cabin-config examples/hvac_cabin_illustrative_config.json

# Regenerate the figures for an experiment that already ran
python tools/experiment_plots.py --experiment experiments/results/main_<stamp>

# Regenerate the drive-cycle provenance and validation audit
python tools/audit_drive_cycles.py --out DRIVE_CYCLE_PROVENANCE_REPORT.md

# Run the complete test suite (330 tests: 195 range-engine + 18 Member 2 adapter
# + 117 Phase 4 scenario-engine and runner tests)
python -m unittest discover -s tests -v
```

> [!WARNING]
> The reference profiles are **partially sourced**: only pack kWh and rated range are tied to a manufacturer or documented source. Mass, drag, frontal area, rolling resistance, efficiencies, usable-energy fraction and HVAC/aux loads are placeholders inherited from the illustrative baseline. Computed ranges are not vehicle predictions. The example cycle is **synthetic**, and the UDDS/HWFET traces have provenance pending verification — none is an officially attributed certification cycle.

## Member 2 Integration

`integration/member2_range_adapter.py` consumes a Member 2 SOH prediction CSV, validates it against a documented contract (`cycle_index, cell_id, chemistry, soh_pct`; `1.0 < soh <= 100.0`; NMC/LFP only; unique, strictly increasing cycle indices), and feeds the trajectory to `estimate_trajectory()`. No deep-learning library is imported. Published predictions are genuine inference output from Member 2's own Bi-LSTM and served LightGBM on cell B0005 (NASA PCoE, NMC), with provenance in `integration/predictions/PROVENANCE.json`.

```bash
python -m integration.member2_range_adapter \
  --predictions integration/predictions/member2_nmc_B0005_bilstm_predictions.csv \
  --vehicle-config vehicle_configs/tata_curvv_ev_55kwh_partial_reference.json \
  --cycle drive_cycles/udds_city.csv \
  --out integration/outputs/nmc_B0005_bilstm_range_curvv_udds.csv \
  --ambient-c 30
```

Published results on the Curvv profile at 30 °C: Bi-LSTM-based 186.65–189.85 km, LightGBM-based 196.19–204.05 km, observed-anchor baseline 172.85–182.61 km. These are **estimates only** — both Member 2 models underperform a persistence baseline, and the vehicle profile is uncalibrated.

The `integration/outputs/*.csv` files above were produced before the Phase 1 regenerative-braking revision and have deliberately not been overwritten. Re-running the same command with the current engine shifts these ranges by +0.019 % (1 Hz UDDS is already finely resolved, so interval-average limiting was nearly exact here); regenerate them explicitly when the evidence trail is refreshed.

## Vehicle Configuration Library

Profiles in `vehicle_configs/`:

| Profile | Vehicle | Pack kWh | Rated range reference |
| :--- | :--- | :--- | :--- |
| `tata_curvv_ev_55kwh_partial_reference.json` | Tata Curvv.ev | 55 | 502 km (cycle not stated) |
| `tata_nexon_ev_45kwh_partial_reference.json` | Tata Nexon.ev | 45 | 489 km (ARAI/MIDC) |
| `tata_punch_ev_40kwh_partial_reference.json` | Tata Punch.ev | 40 | 468 km (ARAI) |
| `byd_atto3_gl_49_92kwh_partial_reference.json` | BYD Atto 3 GL | 49.92 | 400 km (NEDC) |
| `mahindra_xev9e_59kwh_partial_reference.json` | Mahindra XEV 9e | 59 | 542 km (MIDC/ARAI) |
| `mg4_ev_51kwh_partial_reference.json` | MG4 EV | 51 | 350 km claimed (cycle unstated) |
| `hyundai_ioniq5_84kwh_partial_reference.json` | Hyundai IONIQ 5 | 84 | 690 km (ARAI/MIDC) |
| `tesla_model_y_lr_78kwh_partial_reference.json` | Tesla Model Y LR AWD | ≈78 | 600 km (WLTP) |
| `illustrative_placeholder.json` | Not a real vehicle | 75 | 650 km (illustrative) |

Select a profile with `--config <path>`; the resolved `vehicle_name` and `quality_status` are printed in the JSON output. Every reference profile reports `quality_status = REQUIRES_VALIDATION`; only the illustrative profiles report `ILLUSTRATIVE_UNVALIDATED`. Source URLs, retrieval date, sourced fields and assumption lists live in each file's `provenance` block — read them before using a number. See `MEMBER3_COMPLETE_OVERVIEW.md` Section 2C for the full provenance audit.

## Physics & Modeling Capabilities
1. **Interval-Based Mechanical Work Integration:**
   - Evaluates interval acceleration $a_{i+1/2} = \frac{v_{i+1} - v_i}{\Delta t_i}$.
   - Exact analytical kinetic energy change $\Delta W_{\text{inertia}} = \frac{1}{2} m_{\text{eff}} (v_{i+1}^2 - v_i^2)$ ensuring exact conservation of energy on closed cycles.
2. **Rotational Inertia Equivalent Mass:**
   - Includes rotational drivetrain inertia factor ($\delta = 1.05$ default, $m_{\text{eff}} = m \cdot (1 + \delta)$) accounting for wheels, brake rotors, and motor rotor inertia.
3. **Air Density from Temperature, Station Pressure or Altitude:**
   - Density always comes from the ideal gas law:
     $$\rho=\frac{p}{R_{\text{spec}}\,T_{\text{K}}},\qquad T_{\text{K}}=T_{\text{amb}}+273.15$$
     capturing the $+9.5\%$ aerodynamic drag penalty in cold weather ($-10^\circ\text{C}$).
   - With only `ambient_c`, pressure is standard sea level (101325 Pa) — the historical behaviour. With `altitude_m`, pressure is the ISA troposphere relation
     $$p(h)=p_0\Big(1-\frac{L h}{T_0}\Big)^{g_0/(R_{\text{spec}}L)},\qquad L=0.0065\ \text{K/m},\ T_0=288.15\ \text{K}$$
     documented as valid from $-500$ m to $11\,000$ m (the tropopause); outside that the input is rejected. `pressure_pa` is accepted for a measured station pressure.
   - Precedence, so altitude is never applied twice: `pressure_pa` beats `altitude_m` for pressure; a supplied `ambient_c` beats the ISA temperature and receives **no** lapse correction, because a measured ambient already is the ground temperature. ISA temperature is used only when `ambient_c` is omitted.
   - When all three are omitted, the configured `air_density_kg_m3` is used unchanged.
4. **Steady Along-Track Wind:**
   - `headwind_mps` is the longitudinal wind component with an explicit convention: **positive = headwind** (air moving toward the vehicle nose), negative = tailwind. The aerodynamic term uses relative air speed $u=v+w$ while power remains force times **ground** speed:
     $$F_{\text{aero}}=\tfrac{1}{2}\rho C_d A\,u|u|,\qquad P_{\text{aero}}=F_{\text{aero}}\,v$$
     so a headwind raises drag and a tailwind lowers it; when a tailwind exceeds the vehicle speed, $u<0$ and the air pushes the car (aero becomes thrust), which the integrator handles by splitting the interval at $u=0$.
   - A crosswind is **rejected**, not approximated: this model is one-dimensional, and $C_d A$ for a yawed vehicle is a different (unmodelled) problem.
   - Wind is constant for the whole cycle — no gusts, shear or altitude-dependent wind profile.
5. **Empirical HVAC Thermal Comfort Model (default, unchanged by Phase 3):**
   - Automatically estimates heating and cooling loads from ambient temperature if explicit HVAC power is omitted:
     - Deadband ($18^\circ\text{C}\text{--}22^\circ\text{C}$): Base ventilation ($0.25\text{ kW}$ default).
     - Heating ($T_{\text{amb}} < 18^\circ\text{C}$): Linear heating ramp up to $6.0\text{ kW}$ nominal cap (or default if higher).
     - Cooling ($T_{\text{amb}} > 22^\circ\text{C}$): Linear cooling ramp up to $3.5\text{ kW}$ nominal cap (or default if higher).
     - Monotonically non-decreasing away from deadband, bounded by $15.0\text{ kW}$ automotive capacity, and overridable via `--hvac-kw`.
6. **Cabin Thermal Model and HVAC Actuator Modes (Phase 3):**
   - `hvac_model_mode` selects the HVAC physics. Its default is `EMPIRICAL` — the whole-cycle comfort curve in item 5 — and the other three modes must be asked for explicitly, in a profile, as `estimate(..., hvac_model_mode=...)`, or with `--hvac-mode`. Choosing the default changes nothing: no existing call, profile or output is reinterpreted.

   | Mode | Actuator | Electrical → thermal | Labels |
   | :--- | :--- | :--- | :--- |
   | `EMPIRICAL` (default) | none; empirical comfort curve | not applicable | `CONFIG_DEFAULT`, `COMFORT_DEADBAND_VENTILATION`, `EMPIRICAL_HEATING_DEMAND`, `EMPIRICAL_COOLING_DEMAND` |
   | `HEAT_PUMP` | heat pump, **heating only** | $P_{\text{elec}}=Q_{\text{heat}}/COP_{\text{heat}}$ | `CABIN_HEAT_PUMP` |
   | `PTC` | resistive heater, **heating only** | $P_{\text{elec}}=Q_{\text{heat}}/\eta_{\text{PTC}}$ | `CABIN_PTC` |
   | `AC_COOLING` | evaporator, **cooling only** | $P_{\text{elec}}=\lvert Q_{\text{cool}}\rvert/COP_{\text{cool}}$ | `CABIN_AC_COOLING` |

   - One actuator per mode, deliberately: `HEAT_PUMP` and `PTC` never cool and `AC_COOLING` never heats, so the two heating paths cannot both claim the same heat (`HYBRID` is not implemented) and a demand the mode cannot serve is reported instead of silently met (`AUTO` is not implemented — its choice would be an unstated tuning decision).
   - **Plant.** A single lumped cabin node, integrated per drive-cycle interval:
     $$C_{\text{cabin}}\frac{\mathrm{d}T_{\text{cab}}}{\mathrm{d}t}=\underbrace{UA\,(T_{\text{amb}}-T_{\text{cab}})+Q_{\text{solar}}+Q_{\text{occ}}}_{Q_{\text{environment}}}+Q_{\text{HVAC}}$$
     Sign convention: $Q_{\text{HVAC}}>0$ heats the cabin and $Q_{\text{HVAC}}<0$ removes heat, while the drawn electrical power $P_{\text{elec}}$ is always non-negative. Units: $C_{\text{cabin}}$ in J/K, $UA$ in W/K, $Q$ in W, temperatures in °C (only differences enter).
   - **Initial condition.** `cabin_initial_temp_c = null` means the cabin has soaked to ambient, so the run starts at $T_{\text{amb}}$ (reported as `cabin_temp_start_c`, with the source recorded as `AMBIENT_SOAK`); an explicit value models a pre-conditioned or sun-soaked start.
   - **Control.** The thermal demand is the steady-state load to the setpoint plus a proportional correction, and it is held constant across the interval (zero-order hold, matching an actuator that does not chase the temperature within a second):
     $$Q_{\text{demand}}=UA\,(T_{\text{set}}-T_{\text{amb}})-Q_{\text{solar}}-Q_{\text{occ}}+K_p\,(T_{\text{set}}-T_{\text{cab}}),\qquad K_p=\frac{C_{\text{cabin}}}{\tau_{\text{recovery}}}$$
     `cabin_recovery_time_constant_s = null` sets $K_p=UA$ (the fastest stable setting). With no cap engaged the equilibrium of the closed loop is exactly $T_{\text{set}}$ and it settles with time constant $C_{\text{cabin}}/(UA+K_p)$.
   - **Integration method.** Analytic, not Euler: over an interval the node relaxes to $T_\infty=T_{\text{amb}}+(Q_{\text{HVAC}}+Q_{\text{solar}}+Q_{\text{occ}})/UA$ as
     $$T_{n+1}=T_\infty+(T_n-T_\infty)\,e^{-UA\,\Delta t/C_{\text{cabin}}}$$
     so the drift part is exact for any step size and $\Delta t$ only enters through the held $Q_{\text{HVAC}}$. `cabin_ua_w_per_k = 0` is rejected for these modes: with no conductive leak there is no steady state and held heat would march the cabin linearly past the setpoint.
   - **Stability, as derived and enforced.** The error obeys $e_{n+1}=e_n\,[(1+k)\rho-k]$ with $k=K_p/UA$ and $\rho=e^{-UA\Delta t/C_{\text{cabin}}}$, which contracts for every allowed interval whenever $k\le1$, i.e. $\tau_{\text{recovery}}\ge C_{\text{cabin}}/UA$. `VehicleConfig.validate()` refuses a shorter recovery constant rather than letting a scenario oscillate. Intervals much longer than $C_{\text{cabin}}/UA$ stay bounded but ring about the setpoint instead of approaching it monotonically.
   - **Power limits.** `max_heating_electrical_kw` and `max_cooling_electrical_kw` cap the **electrical** side of the actuator; blower base plus actuator additionally never exceeds the engine's `MAX_HVAC_ELECTRICAL_KW = 15.0` kW ceiling, and the actuator is throttled to the remaining headroom with the thermal output recomputed from the throttled input — never truncated behind the model's back. When a cap binds, `hvac_power_limited` is true and the cabin approaches but does not hold the setpoint; when the mode has no compatible actuator, `cabin_setpoint_unmet` is true and delivery is zero.
   - **COP and efficiency provenance.** The default heating curve is $COP_{\text{heat}}=3.0-0.05\,(7-T_{\text{amb}})$ clipped to $[1.2,5.0]$ and the cooling curve $COP_{\text{cool}}=3.5-0.05\,(T_{\text{amb}}-35)$ clipped to $[1.5,6.0]$. The reference points 7 °C and 35 °C name the EN 14511 **A7** heating and **A35** cooling nominal test duties (7 °C and 35 °C outdoor dry-bulb air). A7 is the air side of the air-to-water duty written **A7/W35**, where W35 fixes a 35 °C water *supply* temperature; a vehicle cabin heat pump delivers air rather than water, so the W35 sink qualifier does not apply here. **EN 14511 prescribes those test conditions and no COP value at all: the duty COPs (3.0 and 3.5), the 0.05/K slopes and the clip bands are illustrative engineering assumptions — not measured vehicle-specific data, and not COPs prescribed by the standard.** Supplying `heat_pump_cop` / `ac_cooling_cop` (`--heat-pump-cop`, `--cooling-cop`) replaces the curve with an explicitly configured **constant**, which the `limitations` text then labels a simplified approximation. `ptc_heating_efficiency` must lie in $(0,1]$: a resistive heater converts electrical to thermal at $\eta$, so its COP is $\eta<1$ by construction.
   - **Configuration fields** (all optional, all validated against `CONFIG_FIELD_LIMITS`; the defaults are illustrative compact-EV placeholders, not vehicle data):

   | Field | Unit | Default | Meaning |
   | :--- | :--- | :--- | :--- |
   | `hvac_model_mode` | enum | `EMPIRICAL` | which HVAC physics runs |
   | `cabin_target_temp_c` | °C | 22.0 | setpoint $T_{\text{set}}$ |
   | `cabin_initial_temp_c` | °C | `null` = soak to ambient | $T_{\text{cab}}(0)$ |
   | `cabin_thermal_capacitance_j_per_k` | J/K | 1 000 000 | $C_{\text{cabin}}$ |
   | `cabin_ua_w_per_k` | W/K | 150 | lumped envelope conductance $UA$; must be > 0 for a cabin mode |
   | `cabin_solar_load_w` | W | 0.0 | $Q_{\text{solar}}$ gain, steady for the cycle |
   | `cabin_occupancy_load_w` | W | 0.0 | $Q_{\text{occ}}$ occupant/internal gain |
   | `cabin_recovery_time_constant_s` | s | `null` = $C/UA$ | $\tau_{\text{recovery}}$, sets $K_p=C/\tau$; validated as $\ge C/UA$ |
   | `heat_pump_cop_reference` | – | 3.0 | COP at the reference temperature |
   | `heat_pump_cop_reference_temp_c` | °C | 7.0 | EN 14511 A7 air-side heating duty (a test condition, not a vehicle value) |
   | `heat_pump_cop_slope_per_k` | 1/K | 0.05 | COP lost per kelvin below the reference point |
   | `heat_pump_cop_min` / `heat_pump_cop_max` | – | 1.2 / 5.0 | clipping band for the heating curve |
   | `ptc_heating_efficiency` | fraction | 0.95 | $\eta_{\text{PTC}}\in(0,1]$ |
   | `ac_cooling_cop_reference` | – | 3.5 | COP at the reference temperature |
   | `ac_cooling_cop_reference_temp_c` | °C | 35.0 | EN 14511 A35 cooling duty (a test condition, not a vehicle value) |
   | `ac_cooling_cop_slope_per_k` | 1/K | 0.05 | COP lost per kelvin above the reference point |
   | `ac_cooling_cop_min` / `ac_cooling_cop_max` | – | 1.5 / 6.0 | clipping band for the cooling curve |
   | `max_heating_electrical_kw` | kW | 8.0 | electrical cap on the heating actuator |
   | `max_cooling_electrical_kw` | kW | 6.0 | electrical cap on the cooling actuator |

   - **Reported keys** (additive; `null` outside cabin modes): `hvac_model_mode`, `auxiliary_energy_kwh`, `hvac_energy_kwh`, `hvac_heating_thermal_energy_kwh`, `hvac_cooling_thermal_energy_kwh`, `cabin_target_temp_c`, `cabin_temp_start_c`, `cabin_temp_end_c`, `cabin_time_constant_s`, `hvac_heating_cop`, `hvac_heating_efficiency`, `hvac_cooling_cop`, `cabin_setpoint_unmet`, `hvac_power_limited`.
   - **Energy accounting.** `auxiliary_energy_kwh = ∫P_aux dt` and `hvac_energy_kwh = ∫P_HVAC dt` accumulate in separate buckets, `aux_hvac_energy_kwh` keeps its historical meaning and is exactly their sum, and
     $$E_{\text{cycle}}=E_{\text{traction}}-E_{\text{regen}}+E_{\text{aux}}+E_{\text{HVAC}}$$
     HVAC is never also inside the auxiliary bucket. The reported `hvac_power_kw` in a cabin mode is the energy-weighted mean of the interval powers. Within the HVAC bucket the blower base `hvac_power_kw_default` is counted exactly once: a cabin mode bills it alongside the actuator (`base_kw + Q_actuator/COP`), `EMPIRICAL` takes its whole HVAC number from it (it is the floor of the comfort curve), and a manual `hvac_power_kw` **replaces** it rather than adding to it, which is the pre-Phase-3 meaning of that argument. `auxiliary_energy_kwh` is `auxiliary_power_kw × duration` in every mode, with no blower in it.
   - **Precedence.** `hvac_power_kw` (`--hvac-kw`) is an **electrical** override and wins over every mode, including the thermal ones: it is never reinterpreted as a thermal load, its label stays `MANUAL_OVERRIDE`, and the cabin diagnostics are `null`. `hvac_model_mode` in the output then records only what was *requested*, and a `limitations` entry states that the override short-circuited that model, so the two can never be confused.
   - The ten shipped profiles carry none of these fields, so they load and compute exactly as before Phase 3. `examples/hvac_cabin_illustrative_config.json` is the separated, fully labelled illustrative configuration; use it instead of editing a reference profile.
7. **Regenerative Braking from the Instantaneous Wheel-Power Profile:**
   - Each interval is scored against the wheel-power curve
     $$P_{\text{wheel}}(v) = \big(m_{\text{eff}}\,a + m g (C_{rr}\cos\theta + \sin\theta)\big)v + \tfrac{1}{2}\rho C_d A\,v^3$$
     with $v(t)$ piecewise linear and $\theta$ piecewise constant between cycle points (the grade of an interval is its endpoint average). $P_{\text{wheel}}>0$ is traction, $P_{\text{brake}}=\max(0,-P_{\text{wheel}})$ is braking, so grade, aerodynamic and rolling resistance all contribute braking even at constant speed.
   - The interval is split at physical events — the 0.5 and 2.0 m/s blend thresholds, the wheel-power sign change $v^\ast=\sqrt{-(m_{\text{eff}}a+F_{\text{resist}})/(\tfrac{1}{2}\rho C_d A)}$, and every crossing of the charge-power limit (located from the exact critical points of the power quartic, then bisection) — and each sub-piece is integrated in closed form from the velocity moments $\int v^{n}\,dt$. With wind, the split also includes $u=0$, and each side is a cubic $L v+Q v^{2}+C v^{3}$ in ground speed, so the same moments stay exact.
   - Recovered electrical power and its limit:
     $$P_{\text{regen}}(t)=\min\!\big(\eta_{\text{regen}}\,\beta(v(t))\,P_{\text{brake}}(t),\;P_{\max}\big),\qquad \beta(v)=\begin{cases}0,&v\le 0.5\ \text{m/s}\\(v-0.5)/1.5,&0.5<v<2.0\ \text{m/s}\\1,&v\ge 2.0\ \text{m/s}\end{cases}$$
     $P_{\max}$ (`max_regen_power_kw`) is a **battery electrical charge-power limit** applied to instantaneous power; inside a capped sub-piece the recovered energy is $P_{\max}\Delta t$, never an interval-average clamp.
   - Full braking ledger, so nothing is counted twice: `mechanical_braking_kwh` $=\int P_{\text{brake}}dt$, `regen_mechanical_absorbed_kwh` = the mechanical share routed through the motor, `regen_recovered_kwh` $=\eta_{\text{regen}}\times$ that, `friction_braking_kwh` = `mechanical_braking_kwh` − `regen_mechanical_absorbed_kwh`, i.e. the mechanical energy the friction brakes must dissipate when blending or the charge-power limit blocks the motor. The $\eta_{\text{regen}}$ loss is electrical conversion loss and is deliberately **not** booked as friction braking. Recovery can never exceed available braking energy.
   - `effective_regen_blend_factor()` is retained only as a legacy diagnostic. Its kinetic-energy weighting is **not** applied to interval work any more: it mis-weights aero/roll/grade braking power and cannot represent instantaneous charge-power limiting.
   - Convergence, stated as measured: refining the *same* trajectory (linearly interpolating 2×, 4×, 8× into every interval of the 1 Hz UDDS trace) leaves every ledger term unchanged to $\sim10^{-13}$ relative — the integrator has no step-size error on a given polyline. Sampling the trajectory more coarsely is a different input: the same trace decimated to 2 Hz and 5 Hz shifts net cycle energy by −0.97 % and −3.48 %, and those figures are input-representation error, not solver error.
8. **Vehicle Configuration Validation and Provenance Audit:**
   - `VehicleConfig` validates every field on construction: type, finiteness and a documented physical envelope (`CONFIG_FIELD_LIMITS`) chosen to catch unit and transcription errors — mass in grams, Cd as a percentage, regen power in watts — without refusing any plausible car.
   - Booleans are rejected as numeric values (Python's `bool` is an `int` subclass, so `True` would otherwise pass as 1.0 kg or 1.0 kW); so are NaN, ±inf, strings, `None` for a required parameter, unknown JSON keys, non-dict JSON documents and missing required fields (the error names them). Nothing invalid is silently replaced by a default.
   - `provenance` structure is validated: it must be a mapping, `status` a non-empty string, `note`/`manufacturer_source`/`retrieved_date` strings, `sourced_fields` a field-name → source-string mapping whose keys really exist on the config, and `assumptions` a list of strings.
   - `tools/audit_vehicle_configs.py` audits all ten shipped profiles (nine plus the root config) and writes `CONFIGURATION_VALIDATION_REPORT.md`: per-profile load/validate status, a per-parameter SOURCED / assumed / placeholder / unlabelled matrix, simulated vs certified-range implied consumption, cross-profile placeholder detection and self-contradiction findings.
9. **Member 2 SOH Trajectory API:**
   - `estimate_trajectory()` accepts a sequence of SOH percentages, or a sequence of per-point dictionaries (`soh_pct` with optional `cycle_index`, `quality_flag`, `cell_id`), and preserves Member 1/2 quality flags. A mapping of column name to array is not a supported input shape and is rejected with a message naming the offending entry. It forwards the optional `pressure_pa`, `altitude_m`, `headwind_mps` and `crosswind_mps` scenario arguments unchanged to every point, and likewise the Phase 3 HVAC arguments (`hvac_model_mode`, `cabin_*`, `heat_pump_cop`, `ac_cooling_cop`, `max_heating_electrical_kw`, `max_cooling_electrical_kw`), which are appended after the existing ones so positional calls made before Phase 3 keep working. Each point is an independent evaluation: a cabin-thermal mode restarts from its soak temperature at every SOH point instead of carrying heat across the ageing series, so the cabin diagnostics repeat identically along the trajectory while usable energy keeps shrinking.

10. **Scenario Experiment Framework (Phase 4):**
    - `scenario_engine.py` reuses `RangeCalculator.estimate()`; it adds no road-load, regen or HVAC physics of its own. Six scenarios are defined once, as data: A urban (UDDS trace), B highway (HWFET trace), C winter highway (HWFET at −10 °C, heat pump), D summer traffic (UDDS at 38 °C, AC cooling with 600 W solar), E mountain pass and F aggressive driving.
    - Drive cycles are **validated, not trusted**: `inspect_cycle()` refuses non-increasing time, negative or non-finite values, missing or non-numeric cells, a missing header, durations outside 10 s–7200 s, sampling gaps over 600 s, zero total distance and a speed column above 50 m/s (the gate that catches a km/h or mph trace supplied as `speed_mps`). It then hands the trace to the engine's own loader, so nothing the framework accepts is rejected by the physics layer.
    - Provenance is recorded per cycle: `UNVERIFIED_TRANSCRIBED_CYCLE` for UDDS/HWFET (their measured duration and distance corroborate the published totals; corroboration is not certification), `ILLUSTRATIVE_SYNTHETIC_CYCLE` for the smoke-test trace and `SYNTHETIC_SCENARIO` for E and F. No generated trace is presented as an EPA or WLTP cycle, and no synthetic cycle is substituted for a shipped one.
    - E and F are deterministic piecewise-linear profiles on a uniform grid — no RNG, shared segment boundaries, and a physical-sanity gate on acceleration (≤ 6 m/s²), speed (≤ 50 m/s) and grade (±30 %). Mountain grades are documented design values (+6 % climb, +1 % crest, −6 % descent) and the measured net elevation residual is reported instead of being forced to zero.
    - `run_experiments.py` expands a configurable matrix (preset, JSON file or one explicit run) over scenario × vehicle × ambient × SOH × HVAC mode × headwind × altitude, executes it through the existing API and writes a never-overwritten experiment directory with `scenario_results.csv` (56 documented columns, each with its source and unit), `experiment_manifest.json`, `scenario_summary.json`, `scenario_phases.json`, optional `reference_range_comparison.json`, optional `figures/` and `SHA256SUMS.txt`.
    - A failing run is a `FAILED` CSV row carrying its inputs, `error_type` and `error_message`, plus a non-zero exit status — never a silently smaller experiment.
    - All figures compare like with like: one disclosed reference configuration, one SOH or one ambient sweep at a time, provenance in the axis labels and an illustrative-data footer on every chart. Matplotlib is used when importable; otherwise the same chart specifications render as SVG with the standard library only.

## Important Limitations & Scientific Boundaries
- `vehicle_config.json` is an illustrative placeholder and `vehicle_configs/*_partial_reference.json` profiles are only partially sourced: pack kWh and rated range are sourced, but dynamics, efficiencies and usable energy remain provisional. No profile is manufacturer-validated.
- Certified ranges (ARAI/MIDC, NEDC, WLTP) come from test cycles different from the synthetic and supplied UDDS/HWFET-format traces shipped here; simulated ranges will not match them.
- The 650 km reference in project documentation is an illustrative baseline, not an empirical test result of a specific production car.
- SOH is approximated as a linear multiplier on usable pack energy; future work will incorporate cell-to-pack aggregation and internal resistance rise.
- The regen integrator is exact **for the model it integrates**, not for the road: speed is piecewise linear and grade piecewise constant between cycle points, so a coarse cycle still under-resolves real braking events, and coasting is represented as a forced deceleration rather than a drivetrain loss.
- `max_regen_power_kw` is a single constant electrical charge-power limit. Battery charge acceptance that falls off with high SOC and low temperature, motor/inverter torque-speed limits and friction-blend brake-by-wire behaviour are not modelled, so recovery near full charge or in cold conditions is optimistic.
- A cycle that begins fast and ends slow can report `NET_GENERATING_DESCENT` and an unbounded range. Part of that recovery is the vehicle's own initial kinetic energy, harvested **once**; it is not a sustainable repeated-cycle consumption rate and the unbounded value must never be presented as a driving range. The engine attaches this caveat to `limitations` whenever the status is returned.
- The atmospheric model is **dry air**. No humidity or virtual-temperature correction is applied, so humid monsoon-season air (lower density, less drag) is slightly over-penalised. ISA pressure is defined only from −500 m to 11 km and station pressure from 20 to 120 kPa; outside those ranges the input is rejected rather than extrapolated.
- Wind support is a single steady along-track value for the whole cycle: no gusts, no boundary-layer shear with height, no wind rose, and no yaw-dependent $C_d A$. A crosswind is rejected outright because a 1-D longitudinal model cannot represent it.
- The configuration envelopes in `CONFIG_FIELD_LIMITS` are wide physical plausibility limits intended to catch unit and transcription errors. Passing validation means a profile is *self-consistent*, not that any number in it is correct for that car.
- The cabin thermal model is a **single-node, uncalibrated** representation: one capacitance, one lumped UA, steady solar and occupancy gains, and a proportional setpoint controller held constant over each cycle interval. It is not a fully calibrated vehicle cabin simulation — there is no glazing/body/ventilation breakdown, no humidity or latent load, no defrost cycle, no seat heaters, no battery thermal management, no pre-conditioning schedule, and no sun-angle or window-position model. `C`, `UA`, the gains and every COP/efficiency figure are configurable illustrative assumptions unless a profile supplies sourced values, and the shipped reference profiles supply none.
- The COP curves are clipped linear approximations anchored at the EN 14511 **A7** (7 °C outdoor air) heating and **A35** (35 °C outdoor air) cooling test duties — the air-to-water spelling A7/W35 adds a 35 °C water supply temperature that does not apply to an air-delivered cabin. EN 14511 defines those *conditions* and no COP, so the duty COPs, their 0.05/K slopes and their clip bands are illustrative engineering assumptions: not measured vehicle COP maps, and not values taken from the standard. No measured vehicle-specific degradation curve was available for these placeholders, so none is claimed. `HEAT_PUMP` and `PTC` are alternative actuators, never summed: a `HEAT_PUMP` scenario delivers no cooling and an `AC_COOLING` scenario delivers no heating, and an incompatible demand is reported through `cabin_setpoint_unmet` rather than served.
- Cabin results depend on the cycle's time step through the zero-order-hold control, and that dependence is measured, not assumed: refining the grid converges at first order, and the recovery time constant is validated to stay at or above `C/UA` so the discrete error map keeps contracting. Intervals much longer than `C/UA` remain bounded but oscillate about the setpoint.
- `CONFIGURATION_VALIDATION_REPORT.md` records what the audit found: nine parameters are identical across all ten profiles (template placeholders), `air_density_kg_m3` carries no provenance statement at all, the MG4 profile ships 2100 kg while its own note says the verified kerb mass is 1600–1700 kg, and certified ranges imply 7.69–13.11 kWh/100 km against ~15.5 kWh/100 km simulated on UDDS (1.18x–2.01x). These profiles are pipeline fixtures, not calibrated vehicles.
- Scenario ranges are simulated on an **illustrative, uncalibrated** configuration. They may not be presented as equal to real-world measured range, and the 650 km figure stays an illustrative objective: the framework reports the gap between it and the best simulated scenario rather than tuning any parameter to close it.
- The Phase 4 framework deliberately compares scenarios, temperatures, HVAC modes and SOH assumptions on **one** reference configuration. The ten shipped profiles share mass, Cd-area, Crr, drivetrain and auxiliary template values and differ mainly in pack capacity, so a vehicle-vs-vehicle comparison with this engine measures the template, not the car.
- SOH values in a scenario matrix are standalone assumptions handed to the existing `soh_pct` argument. They are not Member 2 predictions and no chemistry-specific or cell-level ageing model backs them.
- Per-phase energies in `scenario_phases.json` and the mountain figure are independent estimates over re-zeroed slices of the trace, so they sum approximately, not exactly; a zero-distance phase is reported with a `skipped_reason` instead of a fabricated number.
- Attribution figures (aero, rolling, grade, HVAC) are differential sensitivity probes on the reference configuration — for example the same run with `drag_coefficient_cd` driven to a numerically negligible value. They are not claims about a car that exists.
- Do not use unresolved or unverified cell capacities as ground truth; always preserve quality flags from Member 1.
- Member 2 predictions are research-stage: on cell B0005's chronological holdout both the Bi-LSTM (MAE 4.16) and LightGBM (8.29) trail a persistence baseline (0.32). The Bi-LSTM is not exposed by the served `Member2Predictor` API. Integrated range outputs are therefore estimates requiring validation, not predictions.
