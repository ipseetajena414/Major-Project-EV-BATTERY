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

## 2. Run Comprehensive Unit & Physics Tests
```bash
python -m unittest discover -s tests -v
```
All 52 tests should pass (34 Member 3 physics/API tests + 18 Member 2 integration tests) in well under a second.

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