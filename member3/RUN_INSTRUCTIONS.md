# Run and Integration Instructions

## 1. Run Baseline Range Calculation
```bash
# Run baseline estimate with ambient temperature and automatic empirical HVAC demand
python range_calculator.py --config vehicle_config.json --cycle drive_cycles/example_cycle.csv --soh-pct 100 --ambient-c 25

# Run cold-weather scenario with explicit HVAC override
python range_calculator.py --config vehicle_config.json --cycle drive_cycles/example_cycle.csv --soh-pct 85 --ambient-c -10 --hvac-kw 3.5

# Pass data quality pedigree flags from Member 1 or Member 2
python range_calculator.py --config vehicle_config.json --cycle drive_cycles/example_cycle.csv --soh-pct 92.5 --quality-flag MEMBER1_VALIDATED
```

## 1b. Environmental Scenarios: Altitude, Station Pressure and Wind
All four arguments are optional; omitting them reproduces the historical behaviour exactly.
```bash
# Lehck (3,500 m) at 5 C: ISA pressure at altitude, supplied ambient temperature kept as-is
python range_calculator.py --config vehicle_configs/tata_nexon_ev_45kwh_partial_reference.json \
  --cycle drive_cycles/hwfet_highway.csv --soh-pct 95 --ambient-c 5 --altitude-m 3500

# Measured station pressure wins over altitude for pressure (altitude is not applied twice)
python range_calculator.py --config vehicle_configs/tata_nexon_ev_45kwh_partial_reference.json \
  --cycle drive_cycles/udds_city.csv --ambient-c 15 --altitude-m 1650 --pressure-pa 83000

# 6 m/s headwind (positive = headwind; negative = tailwind)
python range_calculator.py --config vehicle_configs/tata_nexon_ev_45kwh_partial_reference.json \
  --cycle drive_cycles/hwfet_highway.csv --ambient-c 15 --headwind-mps 6

# Crosswind is refused rather than approximated as a headwind
python range_calculator.py --config vehicle_configs/tata_nexon_ev_45kwh_partial_reference.json \
  --cycle drive_cycles/udds_city.csv --crosswind-mps 4   # ValueError: not modelled
```
Bounds enforced: `--altitude-m` in [-500, 11000], `--pressure-pa` in [20000, 120000],
`--headwind-mps` in [-60, 60]. Result JSON reports `atmospheric_pressure_pa`,
`altitude_m`, `air_density_source` and `headwind_mps` next to `air_density_kg_m3`.

## 1c. Audit the Vehicle Configurations
```bash
python tools/audit_vehicle_configs.py            # rewrites CONFIGURATION_VALIDATION_REPORT.md
```
The report is generated output, not a hand-written claim: it validates every shipped
profile, classifies each parameter as SOURCED / assumed / placeholder / unlabelled, and
compares simulated consumption with the consumption implied by each certified range.

## 1d. HVAC Modes and Cabin Thermal Scenarios (Phase 3)
`--hvac-mode` selects the HVAC physics and **must be asked for**: the default stays `EMPIRICAL`, the pre-Phase-3 whole-cycle comfort curve, so a run that does not name a thermal model produces exactly the numbers it produced before.

| Mode | Actuator | Electrical draw | Reported `hvac_mode` |
| :--- | :--- | :--- | :--- |
| `EMPIRICAL` (default) | none | comfort curve from ambient | `CONFIG_DEFAULT` / `COMFORT_DEADBAND_VENTILATION` / `EMPIRICAL_HEATING_DEMAND` / `EMPIRICAL_COOLING_DEMAND` |
| `HEAT_PUMP` | heating only | `Q_heat / COP_heat` | `CABIN_HEAT_PUMP` |
| `PTC` | heating only | `Q_heat / efficiency_PTC` | `CABIN_PTC` |
| `AC_COOLING` | cooling only | `|Q_cool| / COP_cool` | `CABIN_AC_COOLING` |

```bash
# Heat pump on the example cycle; the cabin soaks to the -5 C ambient at t=0
python range_calculator.py --config examples/hvac_cabin_illustrative_config.json \
  --cycle drive_cycles/example_cycle.csv --hvac-mode HEAT_PUMP --ambient-c -5

# PTC resistive heating on UDDS, cabin pre-conditioned to 20 C, 6 kW electrical cap
python range_calculator.py --config examples/hvac_cabin_illustrative_config.json \
  --cycle drive_cycles/udds_city.csv --hvac-mode PTC --ambient-c -10 \
  --cabin-initial-c 20 --cabin-target-c 21 --max-heating-kw 6

# AC cooling after a sun soak, explicit constant cooling COP, with solar gain
python range_calculator.py --config examples/hvac_cabin_illustrative_config.json \
  --cycle drive_cycles/udds_city.csv --hvac-mode AC_COOLING --ambient-c 40 \
  --cabin-initial-c 55 --solar-load-w 600 --cooling-cop 2.8

# The electrical override still wins over the thermal model: no cabin state is evolved
python range_calculator.py --config examples/hvac_cabin_illustrative_config.json \
  --cycle drive_cycles/example_cycle.csv --hvac-mode HEAT_PUMP --hvac-kw 2.5

# A mode that does not exist is refused by argparse before anything runs (exit code 2)
python range_calculator.py --config vehicle_config.json --hvac-mode HYBRID
```

New optional arguments — `--hvac-kw` keeps its old meaning and precedence, and every cabin argument is ignored by `EMPIRICAL`:

| Argument | Unit | Meaning |
| :--- | :--- | :--- |
| `--hvac-mode` | enum | `EMPIRICAL`, `HEAT_PUMP`, `PTC`, `AC_COOLING` |
| `--cabin-target-c` | °C | setpoint (config default `cabin_target_temp_c`) |
| `--cabin-initial-c` | °C | cabin temperature at t=0; omit it to soak to ambient |
| `--solar-load-w` | W | steady solar gain into the cabin |
| `--occupancy-load-w` | W | steady occupant/internal gain |
| `--heat-pump-cop` | – | constant heating COP, replacing the configured curve |
| `--cooling-cop` | – | constant cooling COP, replacing the configured curve |
| `--max-heating-kw` | kW | electrical cap on the heating actuator |
| `--max-cooling-kw` | kW | electrical cap on the cooling actuator |

A cabin run adds `hvac_model_mode`, `auxiliary_energy_kwh`, `hvac_energy_kwh`, `hvac_heating_thermal_energy_kwh`, `hvac_cooling_thermal_energy_kwh`, `cabin_target_temp_c`, `cabin_temp_start_c`, `cabin_temp_end_c`, `cabin_time_constant_s`, `hvac_heating_cop`, `hvac_heating_efficiency`, `hvac_cooling_cop`, `cabin_setpoint_unmet` and `hvac_power_limited`; all are `null` for `EMPIRICAL`. `hvac_energy_kwh` is the **electrical** draw (that is what the pack pays for) and the `*_thermal_energy_kwh` keys are the heat moved, linked by the COP or the PTC efficiency. `aux_hvac_energy_kwh` is unchanged and still equals `auxiliary_energy_kwh + hvac_energy_kwh` exactly.

`examples/hvac_cabin_illustrative_config.json` carries the cabin parameters. The ten shipped `vehicle_configs/` profiles deliberately carry **none** of them, so adding the model could not disturb an existing result; copy the example instead of editing a reference profile. Every value in it is an uncalibrated engineering placeholder — `C = 250 kJ/K`, `UA = 250 W/K` give a 1000 s cabin that drifts visibly over a cycle, the COP anchors sit on the EN 14511 **A7** (7 °C outdoor air) heating and **A35** (35 °C outdoor air) cooling test duties — A7/W35 is the air-to-water spelling of the same heating point, and the W35 water-supply qualifier does not apply to an air-delivered cabin — and since EN 14511 fixes only the conditions, the anchor COPs (3.0 and 3.5), their 0.05/K slopes and their clip bands are illustrative assumptions, not vehicle measurements and not values prescribed by the standard, and the `limitations` output repeats that this is not a calibrated cabin.

Numerical behaviour, for interpretation of the output: each interval holds its HVAC heat output constant (zero-order hold) and integrates the cabin exactly, `T_next = T_inf + (T - T_inf) * exp(-UA*dt/C)`. `--hvac-mode` therefore needs `cabin_ua_w_per_k > 0` and a `cabin_recovery_time_constant_s` of at least `C/UA`; a shorter value is rejected at load time, because the discrete error map `e_next = e * ((1+k)*exp(-UA*dt/C) - k)` would stop contracting. Coarser cycles settle differently from finer ones at first order, so compare cabin results only at the same time resolution.

## 1e. Scenario Experiments, Provenance and Figures (Phase 4)
```bash
# What is available: cycles with their provenance, scenarios, presets, output schema
python run_experiments.py --list

# The main matrix: 28 runs — the 4 unrestricted scenarios over 3 ambients x 2 SOH
# cases, plus winter and summer, which are defined BY their temperature and so
# collapse to their single pinned ambient
python run_experiments.py --preset main --plots

# The 650 km reference comparison, written next to the main matrix results
python run_experiments.py --preset main --plots --reference-experiment

# The attribution preset: the same probes on their own, 3 runs
python run_experiments.py --preset attribution

# Other axes, one at a time: HVAC modes, SOH sweep, the two thermal scenarios
python run_experiments.py --preset hvac
python run_experiments.py --preset soh --soh-values 100,90,82,75
python run_experiments.py --preset thermal

# A user-defined matrix instead of a preset
python run_experiments.py --matrix examples/matrix_example.json

# One explicit run, with the cabin parameters taken from a supplied configuration file
python run_experiments.py --single --scenario E_MOUNTAIN_PASS --hvac-mode HEAT_PUMP \
  --ambient -10 --soh 82 --headwind 5 --altitude 800 \
  --cabin-config examples/hvac_cabin_illustrative_config.json

# Figures for an experiment that already ran (matplotlib if installed, else SVG)
python tools/experiment_plots.py --experiment experiments/results/main_<stamp>
python tools/experiment_plots.py --backends

# Regenerate the drive-cycle provenance and validation audit
python tools/audit_drive_cycles.py --out DRIVE_CYCLE_PROVENANCE_REPORT.md
python tools/audit_drive_cycles.py --stdout
```

Every run writes into a **new** directory under `experiments/results/<experiment_id>/`; an existing directory is never reused, so an explicit `--experiment-id` that already exists exits with status 2 and leaves the old files untouched. A generated id that collides gets a `_02` suffix instead.

| File | Contents |
| :--- | :--- |
| `scenario_results.csv` | one row per run, 56 columns, every value with a documented source and unit |
| `experiment_manifest.json` | date, git commit plus a dirty-working-tree disclosure, the Python/implementation/platform record and a sha256 for every result-affecting source file, matrix definition, scenario parameters, cycle/vehicle/cabin-config hashes, HVAC settings, model assumptions, units, results, boundary statement, per-file checksums |
| `scenario_summary.json` | per-scenario means and the failure list |
| `scenario_phases.json` | segment tables of the synthetic scenarios |
| `reference_range_comparison.json` | the 650 km comparison: per-scenario consumption, range, difference, attribution probes, per-phase mountain accounting |
| `figures/` | the seven report figures plus `figure_manifest.json` |
| `SHA256SUMS.txt` | checksum of every file above, figures included |

Exit status is 0 when every run succeeded, 1 when any run reported `FAILED` (the row is still written, with its inputs, `error_type` and `error_message`) and 2 for a usage error or a refused overwrite.

Reproducibility is recorded rather than assumed. `software_version` carries `python`, `implementation`, `platform`, whether matplotlib was importable, and a `<name>_sha256` / `<name>_path` pair for each of `range_calculator.py`, `scenario_engine.py`, `run_experiments.py`, `tools/experiment_plots.py` and `tools/audit_drive_cycles.py` — every file that can move a published number, not only the two obvious ones. `code_version` reports the commit, branch and a `working_tree_dirty` boolean, and its `note` states plainly that the commit is the last *committed* state and does not represent uncommitted source while the tree is dirty; the per-file hashes are the authoritative record in that case. `hvac_settings.cabin_config_files` hashes any file passed to `--cabin-config`, because those bytes change the thermal result; the built-in illustrative cabin set is instead recorded by value in `hvac_settings.illustrative_cabin_set`. Two runs of the same matrix on the same files therefore agree column-for-column; only the timestamps, the experiment id and the self-referential manifest checksum differ.

Charts compare like with like on the one disclosed reference configuration: figures are restricted to a single SOH or a single ambient sweep, scenarios C and D carry their defining −10 °C / 38 °C and are labelled as such, and every figure footer states that the configuration is an `ILLUSTRATIVE_ASSUMPTIONS` placeholder, that UDDS/HWFET rows are unverified transcriptions, that E/F rows are `SYNTHETIC_SCENARIO` profiles and that SOH is a standalone assumption. Matplotlib is optional; without it the identical chart specifications are rendered as SVG using only the standard library, and `experiment_manifest.json` records which backend produced the figures.

## 2. Run Comprehensive Unit & Physics Tests
```bash
python -m unittest discover -s tests -v
```
All 330 tests should pass (195 Member 3 range-engine tests: 45 `RangeCalculatorTests` + 32 regenerative-braking + 25 configuration-validation + 15 environmental + 13 wind + 65 Phase 3 HVAC: 17 mode-selection + 13 cabin-thermal + 12 energy-accounting + 11 numerical-stability + 12 trajectory/CLI; 18 Member 2 integration tests; and 117 Phase 4 tests: 78 `test_scenario_engine.py` — 22 drive-cycle validation, 8 synthetic trace, 5 scenario labelling, 8 matrix, 16 execution, 4 failure reporting, 7 output schema and reproducibility, 5 generated audit report, 3 source-hash/version/dirty-tree records — + 39 `test_run_experiments.py` — 12 output files, 8 usage, 6 plots, 6 reference experiment, 3 never-overwrite, 2 listing, 2 exit codes) in a few seconds.

## 3. Python API Integration with Member 2 (SOH Trajectories)
Member 3 provides a decoupled interface requiring zero deep learning dependencies. The example below is **synthetic test data only** — the SOH values are hand-written to exercise the API and are labelled `SYNTHETIC_TEST_ONLY`; they are not measurements and not model output. For real Member 2 predictions use the file-based adapter in Section 5.

```python
from range_calculator import RangeCalculator, VehicleConfig

# 1. Load vehicle parameters and drive cycle
config = VehicleConfig.from_json("vehicle_config.json")
calc = RangeCalculator(config)
cycle = calc.load_cycle("drive_cycles/example_cycle.csv")

# 2. Single-point evaluation
result = calc.estimate(
    cycle=cycle,
    soh_pct=88.5,
    ambient_c=0.0,
    quality_flag="SYNTHETIC_TEST_ONLY",
    cell_id="SYNTHETIC_CELL_01"
)
print(f"Range: {result['estimated_range_km']:.1f} km (Consumption: {result['consumption_kwh_per_100km']:.2f} kWh/100km)")

# 2b. Braking ledger: mechanical braking energy is split between what the motor
# absorbs and what the friction brakes dissipate; regen_recovered_kwh is the
# electrical energy that actually reaches the battery.
print(
    f"Braking (kWh): mechanical={result['mechanical_braking_kwh']:.4f} "
    f"regen_mechanical_absorbed={result['regen_mechanical_absorbed_kwh']:.4f} "
    f"regen_electrical={result['regen_recovered_kwh']:.4f} "
    f"friction={result['friction_braking_kwh']:.4f}"
)

# 2c. Phase 3 cabin thermal HVAC: select a mode explicitly and read both ledgers.
#     Omitting every hvac_* argument below reproduces the pre-Phase-3 result exactly.
cabin = calc.estimate(
    cycle=cycle, soh_pct=100.0, ambient_c=-5.0,
    hvac_model_mode="HEAT_PUMP", cabin_target_temp_c=21.0,
    cabin_initial_temp_c=None,          # None = cabin soaked to ambient
    cabin_solar_load_w=200.0, heat_pump_cop=2.5, max_heating_electrical_kw=8.0,
)
print(
    f"HVAC electrical={cabin['hvac_energy_kwh']:.4f} kWh "
    f"heat delivered={cabin['hvac_heating_thermal_energy_kwh']:.4f} kWh "
    f"cabin {cabin['cabin_temp_start_c']:.1f} -> {cabin['cabin_temp_end_c']:.1f} C "
    f"unmet={cabin['cabin_setpoint_unmet']} capped={cabin['hvac_power_limited']}"
)

# 3. Trajectory evaluation with SYNTHETIC values (not measurements, not inference)
soh_trajectory = [
    {"cycle_index": 0, "soh_pct": 100.0, "quality_flag": "SYNTHETIC_TEST_ONLY"},
    {"cycle_index": 250, "soh_pct": 95.2, "quality_flag": "SYNTHETIC_TEST_ONLY"},
    {"cycle_index": 500, "soh_pct": 89.8, "quality_flag": "SYNTHETIC_TEST_ONLY"},
    {"cycle_index": 750, "soh_pct": 84.1, "quality_flag": "SYNTHETIC_TEST_ONLY"},
    {"cycle_index": 1000, "soh_pct": 79.5, "quality_flag": "SYNTHETIC_TEST_ONLY"},
]
trajectory_results = calc.estimate_trajectory(
    cycle=cycle,
    soh_trajectory=soh_trajectory,
    ambient_c=20.0
    # Any Phase 3 HVAC argument may be added here (hvac_model_mode, cabin_*, *_cop,
    # max_*_electrical_kw) and is forwarded unchanged to every point. Each point is an
    # independent evaluation: a cabin mode restarts from its soak temperature instead
    # of carrying heat across the ageing series, so the cabin diagnostics repeat
    # identically while usable energy keeps shrinking with SOH.
)
for pt in trajectory_results:
    print(f"Cycle {pt['cycle_index']}: SOH={pt['soh_pct']:.1f}% -> Range={pt['estimated_range_km']:.1f} km [{pt['quality_flag']}]")
```

## 4. Replacing Example Inputs with Validated Sourced Data
- Select a vehicle profile with `--config`, e.g. `python range_calculator.py --config vehicle_configs/tata_nexon_ev_45kwh_partial_reference.json --cycle drive_cycles/udds_city.csv --soh-pct 90 --ambient-c 30`.
- Profiles live in `vehicle_configs/`; one JSON per vehicle. Each carries `provenance.status = PARTIALLY_SOURCED_UNCALIBRATED` until every required value is sourced and calibrated. `vehicle_config.json` is the legacy illustrative placeholder.
- Supply drive cycles with columns `time_s,speed_mps,grade_pct`. `drive_cycles/udds_city.csv` and `drive_cycles/hwfet_highway.csv` are supplied traces converted from `drive_cycles/source/`; their content matches the EPA UDDS/HWFET time-speed format, but no attribution, version or licence is recorded, so **their provenance is pending verification**. WLTP is pending.
- Document provenance, temperature scenarios, and payload assumptions.
- Run the suite with `python -m unittest discover -s tests -v` before and after editing any configuration.

## 5. File-Based Member 2 → Member 3 Integration
Member 3 consumes Member 2 SOH predictions as a CSV artifact; no deep-learning dependency is installed in the range calculator.

```bash
cd member3
python -m integration.member2_range_adapter \
  --predictions integration/predictions/member2_nmc_B0005_bilstm_predictions.csv \
  --vehicle-config vehicle_configs/tata_curvv_ev_55kwh_partial_reference.json \
  --cycle drive_cycles/udds_city.csv \
  --out integration/outputs/nmc_B0005_bilstm_range_curvv_udds.csv \
  --ambient-c 30
```

Required prediction columns: `cycle_index, cell_id, chemistry, soh_pct`. Optional: `unit` (must be `%`), `source` (`predicted`/`observed`), `current_observed_soh_pct`, `observed_next_soh_pct`, `quality_flag`, `split`, `model_name`, `model_status`. Rows must be strictly increasing in `cycle_index` per cell and unique on `(cell_id, cycle_index)`. See `integration/README.md` and `integration/predictions/PROVENANCE.json`.

Check the published artifacts:
```bash
python -c "import csv;print(sum(1 for _ in open('integration/outputs/nmc_B0005_bilstm_range_curvv_udds.csv'))-1, 'rows')"
```