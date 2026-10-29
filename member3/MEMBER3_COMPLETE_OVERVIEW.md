# Member 3: Vehicle Dynamics & Range Derivation — Complete Project Dossier

**Autonomous Multi-Agent EV Battery Life & Range Predictor**

**Role:** Member 3 — Vehicle Dynamics Engineer & Range Calculation Specialist

**Repository Branch:** `member3/range-calculator`

**Current Status:** Physics engine stable and test suite passing (52/52: 34 core + 18 integration); file-based Member 2 → Member 3 integration verified end-to-end on cell B0005. Vehicle parameters remain provisional — see [Section 2C](#c-vehicle-configuration-library-partially-sourced-references)

---

## 1. Role & Scope of Member 3 in the Multi-Agent System

In this major project, three autonomous agents / members collaborate to predict EV battery degradation and real-world driving range:
- **Member 1 (Data Engineering & Curation):** Ingests raw cycling datasets (NMC, LFP, Sodium-ion, Solid-State), cleans anomalies, calculates cell health metrics, and builds the validated `master_table.parquet`.
- **Member 2 (Machine Learning & Degradation Forecasting):** Trains LSTM/Bi-LSTM models on `master_table.parquet` for SOH, capacity and RUL. Its served inference API (`Member2Predictor`) exposes LightGBM models; the Bi-LSTM checkpoint is delivered but not served.
- **Member 3 (Vehicle Dynamics & Range Calculator — THIS ROLE):**
  - Implements a lumped-parameter vehicle dynamics physics engine.
  - Ingests drive cycles (time, speed, road grade) and vehicle specifications (mass, drag coefficient, frontal area, rolling resistance, battery capacity).
  - Integrates electrical traction and regenerative braking power.
  - Models temperature-dependent air density and HVAC thermal loads.
  - Takes Member 2's predicted SOH trajectories and maps cell degradation to driving-range decay. Project documents use `650 km → 550 km` as an **illustrative narrative**, not a validated result.

---

## 2. What Member 3 Has Done (Completed Work)

### A. Core Architecture & Starter Implementation
1. **Standard-Library Physics Engine (`range_calculator.py`):**
   - Built with pure Python standard library (`math`, `dataclasses`, `csv`, `json`, `pathlib`, `argparse`). Zero external third-party dependencies required.
2. **Vehicle Configuration System (`vehicle_config.json`, `vehicle_configs/`):**
   - Implemented `VehicleConfig` dataclass with runtime validation (`__post_init__`), loading vehicle mass, frontal area, aerodynamic drag $C_d$, rolling resistance $C_{rr}$, nominal pack kWh, usable energy fraction, and drivetrain efficiencies.
   - `vehicle_configs/` holds one JSON profile per reference vehicle; `--config` selects the profile at the command line.
3. **Drive Cycle Ingestion (`drive_cycles/`):**
   - Built `load_cycle()` parser for CSV drive cycles with automatic validation of `time_s`, `speed_mps`, and `grade_pct`.
4. **Documentation & Process Framework:**
   - Authored `DATA_COLLECTION_CHECKLIST.md`, `RUN_INSTRUCTIONS.md`, and `README.md`.

---

### B. Major Technical Audit & Engineering Overhaul (Recent Milestone)
A rigorous technical audit was conducted, resolving critical physical and numerical deficiencies:

1. **Interval-Based Mechanical Work Integration (Fixing Decoupled Kinetic Energy):**
   - *Previous Defect:* The starter code computed nodal acceleration using 3-point central differences ($a_i = \frac{v_{i+1}-v_{i-1}}{t_{i+1}-t_{i-1}}$) and multiplied by node velocity. On triangular speed profiles ($0 \to 10 \to 0\text{ m/s}$), peak velocity had $a=0$ while endpoints had $v=0$, causing kinetic energy of acceleration to be completely missed ($0.0\text{ kWh}$ integrated).
   - *Resolution:* Re-architected to **interval-based energy integration**:
     $$\Delta t = t_{i+1} - t_i, \quad \bar{v} = \frac{v_i + v_{i+1}}{2}, \quad \Delta s = \bar{v} \Delta t$$
     $$W_{\text{inertia}} = \frac{1}{2} m_{\text{eff}} (v_{i+1}^2 - v_i^2) \quad \text{(Exact analytical kinetic energy integral)}$$
     $$\bar{F}_{\text{aero}} = \frac{1}{2} \rho C_d A \left(\frac{v_i^2 + v_{i+1}^2}{2}\right), \quad W_{\text{aero}} = \bar{F}_{\text{aero}} \Delta s$$
     $$W_{\text{roll}} = m g C_{rr} \cos\theta \Delta s \quad (\bar{v} > 0), \quad W_{\text{grade}} = m g \sin\theta \Delta s$$
     $$W_{\text{wheel}} = W_{\text{inertia}} + W_{\text{aero}} + W_{\text{roll}} + W_{\text{grade}}$$
   - Guarantees exact conservation of energy ($\oint m a v dt = 0$) on closed drive cycles.

2. **Rotational Inertia Equivalent Mass ($\delta = 1.05$):**
   - Replaced point-mass assumption ($F = m \cdot a$) with effective rotational inertia:
     $$m_{\text{eff}} = m_{\text{vehicle}} \cdot (1 + \delta), \quad \delta = 0.05 \text{ (5% rotational inertia of wheels, rotors, motor)}$$

3. **Dynamic Air Density via Ideal Gas Law ($\rho(T)$):**
   - Eliminated static sea-level air density ($1.225\text{ kg/m}^3$). Air density is now dynamically derived from ambient temperature:
     $$\rho(T_{\text{amb}}) = \frac{p_0}{R_{\text{spec}} \cdot (T_{\text{amb}} + 273.15)} = \frac{101,325}{287.058 \cdot (T_{\text{amb}} + 273.15)}$$
   - Captures $+9.5\%$ higher aerodynamic drag in winter ($-10^\circ\text{C}$, $\rho = 1.341\text{ kg/m}^3$) versus mild weather ($+15^\circ\text{C}$, $\rho = 1.225\text{ kg/m}^3$).

4. **Empirical HVAC Thermal Comfort Model:**
   - Converted `ambient_c` from an unused pass-through argument into an active thermal demand model:
     - **Comfort Deadband ($18^\circ\text{C} \le T_{\text{amb}} \le 22^\circ\text{C}$):** Base ventilation only ($0.25\text{ kW}$).
     - **Heating ($T_{\text{amb}} < 18^\circ\text{C}$):** $P_{\text{heat}} = \min(6.0, P_{\text{default}} + 0.18 \times (18.0 - T_{\text{amb}}))\text{ kW}$.
     - **Cooling ($T_{\text{amb}} > 22^\circ\text{C}$):** $P_{\text{cool}} = \min(3.5, P_{\text{default}} + 0.12 \times (T_{\text{amb}} - 22.0))\text{ kW}$.
     - Overridable via `--hvac-kw`.

5. **Regenerative Braking Blending & Low-Speed Cutoff:**
   - Real electric motors collapse back-EMF near zero speed. Implemented a blending ramp:
     - $\bar{v} \ge 2.0\text{ m/s}$ ($7.2\text{ km/h}$): 100% available regen.
     - $\bar{v} \le 0.5\text{ m/s}$ ($1.8\text{ km/h}$): 0% regen (friction brakes only).
     - Clamped by `max_regen_power_kw`.
   - Handled sustained downhill trips (`NET_GENERATING_DESCENT`, infinite range) without crashing.

6. **Input Guardrails & Fractional SOH Ambiguity Trap:**
   - Added explicit trap for decimal SOH: $0 < SOH \le 1.0$ (e.g. $0.85$ intended as $85\%$) raises a helpful `ValueError`.
   - Upper bound checked ($1.0 < SOH \le 120.0\%$, with note for $>100\%$ manufacturing margin).
   - Validated bounds for grade ($\pm 30\%$), temperature ($-50^\circ\text{C}$ to $+60^\circ\text{C}$), HVAC ($\le 15\text{ kW}$), and vehicle speed ($\le 120\text{ m/s} = 432\text{ km/h}$).

7. **Member 2 Batch Trajectory API:**
   - Implemented `estimate_trajectory()` to accept a list/series of predicted SOH values (e.g. from an LSTM degradation curve) and propagate Member 1/2 quality flags (`quality_flag`, `cell_id`).

8. **Test Suite Expansion (8 $\rightarrow$ 52 Passing Tests):**
   - `tests/test_range_calculator.py`: 34 unit, analytical physics, conservation, boundary, and trajectory tests. `tests/test_member2_integration.py`: 18 adapter/integration tests (2 additionally exercise the published artifacts when present). Full suite: 52 tests, 0 failures.

### C. Vehicle Configuration Library (Partially Sourced References)

`member3/vehicle_configs/` contains one JSON profile per reference vehicle. **Only the fields inside `provenance.sourced_fields` are sourced**; every profile carries `status = PARTIALLY_SOURCED_UNCALIBRATED`, and all mass, aerodynamic, rolling-resistance, efficiency, usable-fraction, and auxiliary/HVAC values are provisional placeholders inherited from the original illustrative baseline. Loading successfully is **not** validation.

| File | Vehicle / variant | Pack kWh | Rated range reference (cycle) | Source | Source type | Genuinely sourced | Provisional |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tata_curvv_ev_55kwh_partial_reference.json` | Tata Curvv.ev 55 kWh | 55 | 502 km (cycle not stated on page) | https://ev.tata.cars/curvv/ev.html | OEM website | Pack kWh, rated range | Mass, Cd, A, Crr, efficiencies, HVAC/aux, usable fraction |
| `tata_nexon_ev_45kwh_partial_reference.json` | Tata Nexon.ev 45 kWh | 45 | 489 km (ARAI/MIDC) | https://ev.tata.cars/nexon/ev/specifications.html | OEM website | Pack kWh, rated range | Same as above |
| `tata_punch_ev_40kwh_partial_reference.json` | Tata Punch.ev 40 kWh (facelift long range) | 40 | 468 km (ARAI) | https://ev.tata.cars/punch/ev/specifications.html | OEM website | Pack kWh, rated range | Same as above |
| `byd_atto3_gl_49_92kwh_partial_reference.json` | BYD Atto 3 GL | 49.92 | 400 km (NEDC) | https://bydautoindia.com/bydatto3 (pack); Wikipedia BYD Atto 3 (range) | OEM spec sheet + secondary encyclopedia | Pack kWh; range is secondary-source | Same as above |
| `mahindra_xev9e_59kwh_partial_reference.json` | Mahindra XEV 9e 59 kWh | 59 | 542 km (MIDC/ARAI) | https://www.mahindraelectricsuv.com (official brochure PDF) | OEM brochure | Pack kWh, rated range | Same as above |
| `mg4_ev_51kwh_partial_reference.json` | MG4 EV 51 kWh Standard Range | 51 | 350 km claimed (cycle not stated; India ARAI pending) | https://www.mgmotors.com.np/product/mg4-ev (pack); zecar.com (range) | OEM global spec sheet + secondary listing | Pack kWh; range is secondary-source | Same as above; placeholder mass (2100 kg) is ~400-500 kg above this class |
| `hyundai_ioniq5_84kwh_partial_reference.json` | Hyundai IONIQ 5 84 kWh (India facelift) | 84 | 690 km (ARAI/MIDC) | https://www.hyundai.com/in/en/find-a-car/ioniq-5/highlights | OEM website | Pack kWh, rated range | Same as above |
| `tesla_model_y_lr_78kwh_partial_reference.json` | Tesla Model Y Long Range AWD (Juniper) | ≈78 nominal / ≈75 usable | 600 km (WLTP; a secondary source lists 586 km) | https://en.wikipedia.org/wiki/Tesla_Model_Y | Secondary encyclopedia | Cd 0.23; pack and range are secondary-source | Usable fraction ≈0.96, mass, frontal area, Crr, efficiencies, HVAC/aux |
| `illustrative_placeholder.json` | Not a real vehicle | 75 | 650 km (illustrative baseline) | — | Illustrative only | Nothing | Everything |

Notes:
- Certified ranges (ARAI/MIDC, NEDC, WLTP) are laboratory figures; none of them is the synthetic example cycle or the supplied UDDS/HWFET traces used by this model, so computed ranges will not reproduce them.
- "MG4 INNOCCENT" was not a verifiable MG product/variant name; the profile and file were corrected to **MG4 EV**. India-market launch and ARAI certification remain unconfirmed.
- Ordering of magnitude: with shared placeholder dynamics, computed estimates on `example_cycle.csv` rank by pack size only. That is an artefact of unsourced parameters, not a vehicle comparison.

Select a profile from the command line:

```bash
python range_calculator.py --config vehicle_configs/tata_curvv_ev_55kwh_partial_reference.json \
  --cycle drive_cycles/udds_city.csv --soh-pct 92 --ambient-c 35
```

### D. Member 2 → Member 3 Integration (Implemented & Verified)

Member 2's served API is `Member2Predictor` (LightGBM). Its Bi-LSTM (`nmc_bilstm_model.pt`) is delivered but **not served**; it was executed offline through Member 2's own `BiLSTMPredictor` class and `scaler_nmc_seq.joblib`. Member 2's model, preprocessing, checkpoint and prediction code were not modified. The two sides are joined by a CSV boundary so the range calculator imports no deep-learning library.

| Component | Path | Notes |
| :--- | :--- | :--- |
| Adapter | `integration/member2_range_adapter.py` | Validates `cycle_index, cell_id, chemistry, soh_pct`; enforces `1.0 < soh <= 100.0`, NMC/LFP only, unique and strictly increasing cycle indices; calls `estimate_trajectory()`; exports CSV. |
| Prediction artifacts | `integration/predictions/` | 30 genuine prediction rows each for Bi-LSTM and LightGBM, cell B0005 (NASA PCoE, NMC), cycles 139–168, Member 2 commit `b2a2b51`; SHA256s and environment in `PROVENANCE.json`. |
| Range outputs | `integration/outputs/` | Three trajectories on the Curvv profile + UDDS trace at 30 °C, all labelled `REQUIRES_VALIDATION`. |
| Contract doc | `integration/README.md` | Column contract, quality-flag rules, generation commands, limitations. |
| Tests | `tests/test_member2_integration.py` | 18 tests: schema, units, identity, ordering, invalid/missing input, output schema, flag propagation, row counts, missing-artifact errors. |

Published results (same vehicle profile, same cycle, only the SOH source differs):

| SOH source | Quality flag | SOH range | Estimated range |
| :--- | :--- | :--- | :--- |
| Bi-LSTM (`nmc_bilstm`) | `MEMBER2_BILSTM_EXPERIMENTAL_RESEARCH_STAGE` | 69.51–70.70 % | 186.65–189.85 km |
| LightGBM (`nmc_lgbm`, served) | `MEMBER2_LGBM_EXPERIMENTAL_RESEARCH_STAGE` | 73.06–75.99 % | 196.19–204.05 km |
| Observed SOH (persistence baseline) | `PERSISTENCE_BASELINE_FROM_OBSERVED_SOH` | 64.37–68.01 % | 172.85–182.61 km |

The 30.5 km spread between the two models is a model-disagreement signal, not a validation result. Consumption is SOH-independent in the current linear model (18.435 kWh/100 km), so range scales linearly with SOH.

Member 2's reported per-cell test MAE on B0005 is reproduced exactly by re-execution (Bi-LSTM 4.159, LightGBM 8.2945, persistence 0.3172); **both models trail persistence**. There is no uncertainty quantification. The `650 km → 550 km` framing remains an illustrative project narrative; no number here was forced to match it.

---

## 3. What Needs to Be Done (Pending Work & Roadmap)

To complete Member 3's deliverable for the major project, the following remaining tasks must be executed:

### Task 1: Ingest Real-World Sourced Vehicle Parameters (Partially Complete)
- **Current State:** Eight reference profiles exist in `vehicle_configs/` (see Section 2C). Pack energy and rated range are sourced for each; **mass, $C_d$, $A$, $C_{rr}$, efficiencies, usable-energy fraction and auxiliary/HVAC loads remain provisional** for every vehicle. `vehicle_config.json` and `vehicle_configs/illustrative_placeholder.json` remain illustrative baselines.
- **Action Required:**
  - Source kerb mass, drag coefficient, frontal area, rolling resistance and drivetrain efficiency per vehicle from manufacturer brochures, homologation documents (`typeapproval`/ARAI filings), or independent tests (e.g. coastdown data). Record URL, source type, retrieval date and cycle for each value in `provenance.sourced_fields`.
  - Calibrate usable-energy fraction and drivetrain/regen efficiencies so that simulated consumption on a standard cycle reproduces the certified figure within a stated tolerance.
  - Promote a profile to a validated status only after calibration and an explicit validation check; do not re-label profiles merely because they load and run.

### Task 2: Ingest Official Standard Drive Cycles (Partially Complete)
- **Current State:** `drive_cycles/` contains the synthetic 60-second `example_cycle.csv` plus `udds_city.csv` and `hwfet_highway.csv`, converted from supplied traces in `drive_cycles/source/` by `convert_drive_cycles.py`. The trace format (1 Hz target speed in mph, UDDS/HWFET labels) matches EPA data, but **no attribution, version or licence is recorded — provenance is pending verification**. WLTP Class 3 is still missing.
- **Action Required:**
  - Download and convert **WLTP Class 3** (1800 s, Low/Medium/High/Extra-High phases) to `drive_cycles/wltp_class3.csv`.
  - Record cycle version, licence and attribution for every file; do not rename synthetic data as a standard cycle.
  - Benchmark the reference profiles against their certified figures on the matching cycle before any calibration claim.

### Task 3: End-to-End Pipeline with Member 2 (SOH Trajectory → Range) — Implemented (Partially Verified)
- **Current State:** File-based adapter built and tested (Section 2D). Genuine Bi-LSTM and LightGBM predictions for cell B0005 were run end-to-end through the Curvv profile on the UDDS trace. The pipeline is verified for NMC at cell level; LFP and Na-ion trajectories have not been run.
- **Verified scope:** contract validation, unit handling, chronological ordering, identity preservation, quality-flag propagation, output schema, row counts, missing-artifact errors (18 tests).
- **Remaining risks:** both Member 2 models underperform a persistence baseline; Bi-LSTM is not served by the API; no uncertainty bounds; vehicle dynamics uncalibrated; prediction horizon is positional (one observed record), not a physical cycle horizon; cell-to-pack mapping is linear.
- **Action Required:**
  - Obtain LFP predictions (Member 2 LFP models use a 29-feature schema without `temp_*`) and run the adapter; Na-ion has no supported model.
  - Extend the adapter only after Member 2 provides pack-level or multi-cell output; do not average cell SOH into pack energy without a documented aggregation model.
  - Re-run the pipeline when a vehicle profile is validated, and compare against certified cycles once those are attributed.

### Task 4: Cell-to-Pack Aggregation & Internal Resistance Expansion
- **Current State:** Usable energy scales linearly ($E = E_{\text{nom}} \times \text{fraction} \times SOH/100$).
- **Action Required:**
  - Account for series-string weakest-cell limitation in high-voltage EV packs.
  - Implement internal resistance ($R_0$) rise: As the battery ages, $R_0$ increases ($I^2 R$ heat losses increase), causing terminal voltage to hit cutoff earlier under aggressive acceleration.

### Task 5: Automated Scenario Runner & Visualization Scripts
- **Action Required:**
  - Create a scenario execution script (`run_experiments.py`) that sweeps:
    - **Ambient temperatures:** $-10^\circ\text{C}$ (Winter), $+20^\circ\text{C}$ (Mild), $+38^\circ\text{C}$ (Summer).
    - **Drive styles:** Urban (UDDS), Highway (HWFET), Mixed (WLTP).
    - **Chemistries:** NMC vs LFP vs Sodium-ion.
  - Export CSV tables and PNG/PDF range degradation graphs for the final project report.

---

## 4. Key Files in `member3/`

| File | Purpose |
| :--- | :--- |
| [`member3/range_calculator.py`](file:///c:/Users/KIIT0001/OneDrive/Desktop/MAIN%20PROJECT/Major-Project-EV-BATTERY/member3/range_calculator.py) | **Main vehicle dynamics & range calculation engine.** Contains interval-based physics, ideal gas law air density, empirical HVAC comfort model, regen blending, input validation, and trajectory API. |
| [`member3/vehicle_config.json`](file:///c:/Users/KIIT0001/OneDrive/Desktop/MAIN%20PROJECT/Major-Project-EV-BATTERY/member3/vehicle_config.json) | **Legacy illustrative configuration.** Fully synthetic baseline retained for backward compatibility; not a real vehicle. |
| [`member3/vehicle_configs/`](file:///c:/Users/KIIT0001/OneDrive/Desktop/MAIN%20PROJECT/Major-Project-EV-BATTERY/member3/vehicle_configs) | **Vehicle profile library.** Eight partially sourced reference profiles (Tata Curvv.ev, Tata Nexon.ev, Tata Punch.ev, BYD Atto 3, Mahindra XEV 9e, MG4 EV, Hyundai IONIQ 5, Tesla Model Y LR AWD) plus the illustrative placeholder. Only `provenance.sourced_fields` are sourced; see Section 2C. |
| [`member3/tests/test_range_calculator.py`](file:///c:/Users/KIIT0001/OneDrive/Desktop/MAIN%20PROJECT/Major-Project-EV-BATTERY/member3/tests/test_range_calculator.py) | **Core test suite.** 34 tests covering backward compatibility, kinetic energy conservation, analytical aero/roll work, regen limits, temperature scaling, input validation, and trajectories. |
| [`member3/tests/test_member2_integration.py`](file:///c:/Users/KIIT0001/OneDrive/Desktop/MAIN%20PROJECT/Major-Project-EV-BATTERY/member3/tests/test_member2_integration.py) | **Integration test suite.** 18 tests for the prediction contract, unit/SOH validation, ordering, identity, quality flags, output schema, and missing-artifact errors. |
| [`member3/integration/member2_range_adapter.py`](file:///c:/Users/KIIT0001/OneDrive/Desktop/MAIN%20PROJECT/Major-Project-EV-BATTERY/member3/integration/member2_range_adapter.py) | **File-based Member 2 → Member 3 adapter.** Validates prediction CSVs, calls `estimate_trajectory()`, exports range CSVs. Standard library only. |
| [`member3/integration/predictions/`](file:///c:/Users/KIIT0001/OneDrive/Desktop/MAIN%20PROJECT/Major-Project-EV-BATTERY/member3/integration/predictions) | **Genuine prediction artifacts.** Bi-LSTM and LightGBM outputs for cell B0005 plus `PROVENANCE.json` (commit, SHA256, environment, scope, caveats). |
| [`member3/integration/outputs/`](file:///c:/Users/KIIT0001/OneDrive/Desktop/MAIN%20PROJECT/Major-Project-EV-BATTERY/member3/integration/outputs) | **Range trajectories.** Three Curvv + UDDS runs at 30 °C: Bi-LSTM, LightGBM, observed-SOH baseline. |
| [`member3/integration/README.md`](file:///c:/Users/KIIT0001/OneDrive/Desktop/MAIN%20PROJECT/Major-Project-EV-BATTERY/member3/integration/README.md) | **Integration contract documentation.** Column contract, quality flags, generation commands, limitations. |
| [`member3/drive_cycles/example_cycle.csv`](file:///c:/Users/KIIT0001/OneDrive/Desktop/MAIN%20PROJECT/Major-Project-EV-BATTERY/member3/drive_cycles/example_cycle.csv) | **Example synthetic drive cycle.** 60-second smoke-test trace with `time_s,speed_mps,grade_pct`. |
| [`member3/README.md`](file:///c:/Users/KIIT0001/OneDrive/Desktop/MAIN%20PROJECT/Major-Project-EV-BATTERY/member3/README.md) | **Module documentation.** Explains architecture, physics formulas, limitations, and quick start. |
| [`member3/RUN_INSTRUCTIONS.md`](file:///c:/Users/KIIT0001/OneDrive/Desktop/MAIN%20PROJECT/Major-Project-EV-BATTERY/member3/RUN_INSTRUCTIONS.md) | **Step-by-step instructions.** CLI commands, Python API integration with Member 2, and troubleshooting. |
| [`member3/DATA_COLLECTION_CHECKLIST.md`](file:///c:/Users/KIIT0001/OneDrive/Desktop/MAIN%20PROJECT/Major-Project-EV-BATTERY/member3/DATA_COLLECTION_CHECKLIST.md) | **Formal data checklist.** Required vehicle specifications, drive cycle attribution rules, and validation guidelines. |
| [`member3/requirements.txt`](file:///c:/Users/KIIT0001/OneDrive/Desktop/MAIN%20PROJECT/Major-Project-EV-BATTERY/member3/requirements.txt) | **Dependency specification.** Standard Python library only; zero pip installations needed. |
