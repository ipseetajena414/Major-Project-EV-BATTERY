# Member 4 — Multi-Agent Simulation & BMS Advisory

A Driver Persona agent (always wants a fast charge to 100 %) and a BMS Safety Supervisor agent (thermal, C-rate and SOC limits)
run through a LangGraph loop for up to 20 simulated years. Five charging strategies are compared across four chemistries
and three climates, and the results are exported for Member 5's dashboard.

```
driver_proposes -> bms_evaluates -> apply_action -> update_battery_state -> check_end_of_life -> (loop)
```

Numbers are decided by Python (`bms.py`, `placeholders.py`). The LLM only writes the explanation text.

## Run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v                 # 13 tests

# Run simulations
python run_simulation.py hot                            # or: temperate | cold | all
python run_simulation.py hot -v tata_nexon_ev_45kwh_partial_reference.json
python run_simulation.py --list-vehicles                # list all 9 vehicle configurations
python run_simulation.py --list-cycles                  # list available drive cycles (UDDS, HWFET, etc.)
```

Outputs go to `results/<climate>/` (`results.json`, `summary.csv`, `soh_curves.png`). `sample_results/` holds a committed copy for all three climates.
Gemini text is switched on automatically when `GOOGLE_API_KEY` is in `.env` (copy `.env.example`). Without a key the run uses templated text.
Set `GEMINI_MODEL` in `.env` or `config.py` (default: `gemini-2.5-flash`).

### For Member 5 (Dashboard Lead)

From Python:
```python
from run_simulation import run_simulation, run_all, set_active_vehicle, set_active_drive_cycle

# Run with defaults (Tata Curvv.ev 55 kWh, UDDS cycle)
res = run_simulation("LFP", "full_protection", "hot")

# Or specify vehicle and cycle dynamically:
res = run_simulation("NMC", "bms_supervised", "temperate",
                     vehicle="tata_nexon_ev_45kwh_partial_reference.json",
                     cycle="udds_city.csv")
```
`run_simulation` returns a dictionary containing the summary and per-block history.

## Integration status — read before quoting any number

| Part | Source | Status |
|---|---|---|
| Range | Member 3 `RangeCalculator` (vendored in `vendor/member3/`) | **Fully integrated.** Powered by Member 3's lumped-parameter road load engine, empirical HVAC comfort model, and regenerative braking. Default vehicle is the sourced **Tata Curvv.ev 55 kWh** (502 km certified range reference) on the standard **EPA UDDS city cycle**. The complete vehicle library (8 OEM references + 1 placeholder) is vendored in `vendor/member3/vehicle_configs/`. |
| Degradation | Assumed coefficients in `config.CHEMISTRIES` | **Placeholder.** Rankings are plausible, absolute years are not predictions. |
| Member 2 SOH model | `member2_modeling/` (not vendored) | **Not plugged in, by design.** It takes 29 discharge-curve statistics and predicts next-record SOH. Its own schema marks temperature, C-rate, SOC and fast-charge features as BLOCKED, which are exactly what this simulation varies. The LFP model is also worse than a persistence baseline (MAE 0.93 vs 0.13 pp). |

`results.json` carries an explicit `model_provenance` block recording the exact vehicle profile, pack energy, rated reference range, drive cycle, engine limitations, and data quality pedigree.

## Vehicle Configuration Library

Profiles in `vendor/member3/vehicle_configs/`:
- `tata_curvv_ev_55kwh_partial_reference.json` (Default, 55 kWh, 502 km ARAI reference)
- `tata_nexon_ev_45kwh_partial_reference.json` (45 kWh, 489 km ARAI reference)
- `tata_punch_ev_40kwh_partial_reference.json` (40 kWh, 468 km ARAI reference)
- `mahindra_xev9e_59kwh_partial_reference.json` (59 kWh, 542 km ARAI reference)
- `byd_atto3_gl_49_92kwh_partial_reference.json` (49.92 kWh, 400 km NEDC reference)
- `mg4_ev_51kwh_partial_reference.json` (51 kWh, 350 km claimed reference)
- `hyundai_ioniq5_84kwh_partial_reference.json` (84 kWh, 690 km ARAI reference)
- `tesla_model_y_lr_78kwh_partial_reference.json` (78 kWh, 600 km WLTP reference)
- `illustrative_placeholder.json` (75 kWh, 650 km illustrative baseline)

Standard drive cycles in `vendor/member3/drive_cycles/`:
- `udds_city.csv` (Default, EPA Urban Dynamometer Driving Schedule)
- `hwfet_highway.csv` (EPA Highway Fuel Economy Test)
- `example_cycle.csv` / `synthetic_example_cycle.csv` (software smoke-test baseline)

## Files
`config.py` thresholds, vehicle/cycle resolvers, chemistries, scenarios, climates · `state.py` shared state · `bms.py` rules · `agents.py` Driver/BMS text (optional Gemini) · `graph.py` LangGraph loop · `placeholders.py` model layer · `run_simulation.py` runner, CLI, export, guidelines · `tests/` · `vendor/member3/` · `sample_results/` · `results/`
