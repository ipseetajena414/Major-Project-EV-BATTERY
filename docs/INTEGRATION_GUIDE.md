# Integration guide

Integration branch: `member5/integration` (local; **not pushed**). Created from `origin/main` and merged with `--no-ff`, original branches untouched.

| Member | Remote branch | Commit | Merged | Entry points |
|---|---|---|---|---|
| 1 | `origin/main` (base); `member1-final-audit` @ bedf607950 **not merged** | e36eb7312a | base only | `master_table.parquet` |
| 2 | `origin/member2` | b2a2b51a1d | yes, no conflicts | `member2_modeling/predictor.py`, `integration_readiness/adapter.py` |
| 3 | `origin/member3/range-calculator` | 64accfe43c | yes, no conflicts | `member3/range_calculator.py` |
| 4 | `origin/member4-simulation` | e9b8bf2f4c | yes, no conflicts | `ev_member4/run_simulation.py`, `bms.py` |

`member1-final-audit` holds a candidate master table and audit scripts. It was left out on purpose: the instructions treat Member 1 as a read-only upstream dependency and its candidate table has not been adopted as canonical.

## Findings to know before the demo
1. Member 4 ships a vendored copy of Member 3 (`ev_member4/vendor/member3`). `range_calculator.py` and all tests are byte-identical to `member3/`. Differences: the vendored `vehicle_config.json` default is the Tata Curvv.ev 55 kWh partially sourced reference (502 km), whereas `member3/vehicle_config.json` is the 75 kWh / 650 km illustrative placeholder; the vendored copy also has an extra `drive_cycles/synthetic_example_cycle.csv`. The canonical implementation is `member3/` (Member 3's branch). The dashboard range page uses `member3/` directly; the Member 4 simulation passes explicit vehicle paths.
2. Member 4's degradation coefficients are placeholders, and Member 4 does not call Member 2 (by its own documentation). Simulated years-to-EOL are not predictions.
3. Member 4's README says 13 tests; 8 ran with `unittest discover`.
4. Member 2's `build_and_train.py` imports `torch`, so real-cycle feature extraction (Battery Health page) needs torch. Member 2's `build_range_arguments` targets a provisional calculator signature and always forwards predicted SOH, so the dashboard uses its own `soh_for_range` instead.
5. All vehicle configs are partially sourced/illustrative; the dashboard always shows that warning.
6. Member 3's `estimate()` has no driving-style or auxiliary-power input, so those controls are not offered.
7. The `GEMINI_MODEL` default in Member 4 was not verified by me; text generation was not tested.

## Run
`pip install -r requirements.txt` (= `requirements-member2.txt`, pinned to `model_manifest.json`, plus `requirements-app.txt`) then `streamlit run app.py` from the repo root. Tests: `python -m pytest tests -q`, `cd member3 && python -m pytest -q tests`, `cd ev_member4 && python -m unittest discover -s tests`.

## Data selection and provenance
Battery-health selection is by dataset, then cell, then cycle; `health_for_cycle` applies `dataset_id` as a parquet filter and returns an error (never a silent fallback) if it does not match or if a selection spans several datasets. Member 1's original `quality_flag` is kept outside the Member 2 contract (`member1_quality_flag`; several distinct per-row flags in one cycle are kept verbatim and joined with ` | `) and is passed to `RangeCalculator.estimate(quality_flag=...)` only when the SOH came from the Battery Health selection.
