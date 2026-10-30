# Member 3 — RangeCalculator (Vehicle Dynamics & Range Derivation)

This module provides a **transparent, physics-based lumped-parameter vehicle road load and EV driving range calculation engine**. It computes aerodynamic drag, rolling resistance, inertial acceleration, and road-grade forces, integrated across drive cycles to estimate electrical consumption and driving range.

## Included Modules
- `range_calculator.py`: Interval-based road load integration, temperature-dependent air density, empirical HVAC comfort model, regenerative braking dynamics, and trajectory evaluation.
- `vehicle_config.json`: Legacy illustrative baseline retained for backward compatibility (not a real vehicle).
- `vehicle_configs/`: One JSON profile per reference vehicle, each with explicit provenance. See [Vehicle Configuration Library](#vehicle-configuration-library).
- `drive_cycles/example_cycle.csv`: Synthetic cycle for baseline validation and smoke testing.
- `drive_cycles/udds_city.csv`, `drive_cycles/hwfet_highway.csv`: supplied time-speed traces converted from `drive_cycles/source/` by `convert_drive_cycles.py`. Their content matches the EPA UDDS/HWFET format, but attribution and version are unrecorded — provenance pending verification.
- `integration/`: File-based Member 2 → Member 3 adapter, genuine Bi-LSTM/LightGBM prediction artifacts, provenance record and range-trajectory outputs. See `integration/README.md`.
- `tests/test_range_calculator.py`: 34 physics/API tests. `tests/test_member2_integration.py`: 18 adapter tests.
- `DATA_COLLECTION_CHECKLIST.md`: Formal checklist of required vehicle parameters, cycle sources, and validation data.
- `RUN_INSTRUCTIONS.md`: Execution instructions, CLI reference, and Member 1/2 integration workflows.
- `requirements.txt`: Standard library only; zero external third-party dependencies required.

## Quick Start
From this directory:

```bash
# Run baseline range calculation on example cycle
python range_calculator.py --config vehicle_config.json --cycle drive_cycles/example_cycle.csv --soh-pct 100 --ambient-c 25

# Run a reference vehicle profile on a UDDS trace
python range_calculator.py --config vehicle_configs/tata_curvv_ev_55kwh_partial_reference.json --cycle drive_cycles/udds_city.csv --soh-pct 92 --ambient-c 35

# Run the complete test suite (52 tests: 34 physics/API + 18 integration)
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
3. **Temperature-Dependent Air Density:**
   - Computes dynamic air density via the ideal gas law:
     $$\rho(T_{\text{amb}}) = \frac{p_0}{R_{\text{spec}} \cdot (T_{\text{amb}} + 273.15)}$$
     capturing the $+9.5\%$ aerodynamic drag penalty in cold weather ($-10^\circ\text{C}$).
4. **Empirical HVAC Thermal Comfort Model:**
   - Automatically estimates heating and cooling loads from ambient temperature if explicit HVAC power is omitted:
     - Deadband ($18^\circ\text{C}\text{--}22^\circ\text{C}$): Base ventilation ($0.25\text{ kW}$).
     - Heating ($T_{\text{amb}} < 18^\circ\text{C}$): Linear heating ramp up to $6.0\text{ kW}$ cap.
     - Cooling ($T_{\text{amb}} > 22^\circ\text{C}$): Linear cooling ramp up to $3.5\text{ kW}$ cap.
     - Fully overridable via `--hvac-kw`.
5. **Regenerative Braking with Blending Cutoff:**
   - Blends mechanical to friction braking below $2.0\text{ m/s}$ ($7.2\text{ km/h}$) with full cutoff below $0.5\text{ m/s}$ ($1.8\text{ km/h}$) reflecting real electric motor back-EMF collapse.
   - Clamped by `max_regen_power_kw`.
6. **Member 2 SOH Trajectory API:**
   - `estimate_trajectory()` accepts a sequence or dictionary of SOH values over cell lifetime and preserves Member 1/2 quality flags.

## Important Limitations & Scientific Boundaries
- `vehicle_config.json` is an illustrative placeholder and `vehicle_configs/*_partial_reference.json` profiles are only partially sourced: pack kWh and rated range are sourced, but dynamics, efficiencies and usable energy remain provisional. No profile is manufacturer-validated.
- Certified ranges (ARAI/MIDC, NEDC, WLTP) come from test cycles different from the synthetic and supplied UDDS/HWFET-format traces shipped here; simulated ranges will not match them.
- The 650 km reference in project documentation is an illustrative baseline, not an empirical test result of a specific production car.
- SOH is approximated as a linear multiplier on usable pack energy; future work will incorporate cell-to-pack aggregation and internal resistance rise.
- Do not use unresolved or unverified cell capacities as ground truth; always preserve quality flags from Member 1.
- Member 2 predictions are research-stage: on cell B0005's chronological holdout both the Bi-LSTM (MAE 4.16) and LightGBM (8.29) trail a persistence baseline (0.32). The Bi-LSTM is not exposed by the served `Member2Predictor` API. Integrated range outputs are therefore estimates requiring validation, not predictions.
