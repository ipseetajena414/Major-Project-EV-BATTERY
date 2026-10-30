# Autonomous Multi-Agent EV Battery Life & Range Predictor


## Run the integrated app (Member 5)

```bash
pip install -r requirements.txt
streamlit run app.py          # http://localhost:8501
python -m pytest tests -q     # Member 5 integration tests
```

Docker (not built or tested in the authoring environment): `docker build -t ev-battery . && docker run -p 8501:8501 ev-battery`.
See `docs/ARCHITECTURE.md`, `docs/INTEGRATION_GUIDE.md` and `docs/RAG_SOURCES.md`. Range figures come from the Member 3 engine with partially sourced/illustrative vehicle inputs, so they are illustrative estimates; the headline "650 km to 550 km" below is a project hypothesis, not a computed result. Learned SOH models are experimental and RUL is not approved.

> **Analyzing Real-World Range Decay ($650\text{ km} \rightarrow 550\text{ km}$) Across NMC, LFP, Sodium-ion & Solid-State Batteries via Open-Source AI Agents & Machine Learning**

---

## 🚀 Project Overview

Modern electric vehicles (EVs) often experience noticeable degradation in real-world driving range compared to their factory ratings (e.g., $650\text{ km} \rightarrow 550\text{ km}$ or lower) due to chemical degradation, temperature extremes, aggressive DC fast charging (DCFC), and dynamic HVAC thermal loads.

This project implements an end-to-end, multi-agent AI framework coupled with physics-based battery models and supervised machine learning regressors to forecast multi-cycle capacity fade and real-world range drops across multiple battery chemistries:
- **Lithium-ion (NMC):** NASA Ames Prognostics Center of Excellence (PCoE)
- **Lithium Iron Phosphate (LFP):** MultiStat 18650 C-rate aging campaigns (B1–B6)
- **Sodium-ion (Na-ion):** Zenodo / IonTech low-temperature & rate capability cycling
- **Solid-State & Advanced Chemistries:** Literature-derived RAG & PyBaMM synthesis

---

## 🏛️ End-to-End Multi-Agent Architecture (5-Team Workload)

```
┌────────────────────────────────────────────────────────────────────────┐
│ DATA SOURCES: NASA PCoE (NMC) │ MultiStat/Zenodo (LFP) │ IonTech (Na) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ MEMBER 1: Ingestion & Preprocessing Node [STATUS: COMPLETED]           │
│ - Sensor noise clipping (2.0V - 4.5V) │ CC/CV slicing │ SOH Extraction │
│ - Common Master Table Integration: 515,270 validated telemetry rows   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Structured Parquet Streams
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ MEMBER 2: Feature Engineering & ML Predictor                           │
│ - dQ/dV Peak Tracking │ Thermal Stress (T > 35°C) │ High C-rate (1.5C) │
│ - Models: LightGBM Tabular Regressors & LSTM Sequence Multi-Cycle Fade │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ SOH & Capacity Trajectory
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ MEMBER 3: Real-World Vehicle Dynamics & Range Derivation               │
│ - Lumped-Parameter Road Loads (Aerodynamic Drag, Rolling Resistance)   │
│ - HVAC Thermal Penalties (Hot/Cold) │ Real Range: 650 km → 550 km     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Pack Telemetry & Degradation State
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ MEMBER 4: Multi-Agent Simulation Sandbox & Advisory                    │
│ - Driver Persona Agent (Aggressive DCFC) ↔ BMS Supervisor Agent       │
│ - Prescriptive 20%–80% SOC rules & Dynamic Pre-conditioning Guidelines │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ RAG Queries & Visualization Payload
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ MEMBER 5: System Integrator, Literature Synthesis (RAG) & UI Lead      │
│ - ArXiv/ScienceDirect RAG Agent: Solid-State & Sodium-ion Trade-offs  │
│ - Streamlit Dashboard: Real-time Chemistry, Climate, and Range Sliders │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Member 1: Master Table & Telemetry Ingestion Contract

Member 1 delivers the standardized, noise-filtered, unit-verified telemetry stream uniting multi-chemistry laboratory cycling data.

### Unified Master Table Schema (13 Columns)

| Field Name | Type | Description | Unit / Values |
|---|---|---|---|
| `chemistry` | String | Battery chemical family | `'NMC'`, `'LFP'`, `'Na-ion'` |
| `dataset_id` | String | Testing campaign & source rig identifier | `'NASA_Ames_PCoE'`, `'LFP_MultiStat_B1'`, `'..._B5'`, `'Zenodo_IonTech_NaIon'` |
| `cell_id` | String | Physical / experimental cell label | `'B0005'`, `'B0006'`, `'B0018'`, `'BATTERY_1'`, `'BATTERY_5'`, etc. |
| `protocol_id` | String | Operating C-rates & thermal setpoints | e.g. `'NASA_CC_2A_DISCHARGE_24C'`, `'LFP_0.5C_DISCHARGE_1C_CHARGE'` |
| `cycle_id` | Integer | 1-indexed sequential cycle index | $1, 2, 3, \dots, N$ |
| `time_sec` | Float | Elapsed time within cycle step | Seconds ($s$) |
| `voltage_v` | Float | Cell terminal voltage | Volts ($V$) |
| `current_a` | Float | Applied current (Standard: $<0$ discharge, $>0$ charge) | Amperes ($A$) |
| `temp_c` | Float | Cell thermistor temperature (Strictly `NULL` if unmeasured) | Degrees Celsius ($^\circ\text{C}$) |
| `capacity_ah`| Float | Cycle discharge capacity | Ampere-hours ($Ah$) |
| `soh_pct` | Float | State of Health relative to nominal rated capacity | Percent ($\%$) |
| `quality_flag`| String | Machine-readable provenance & audit flags | `'VALID'`, `'TEMP_UNMEASURED'`, `'CURRENT_UNIT_UNDERDETERMINED'`, etc. |
| `source_file`| String | Original raw filename for full audit traceability | `.mat`, `.xlsx`, `.parquet`, `.pkl` |

### Integrated Master Table Metrics

- **Total Telemetry Rows:** `515,270`
- **Duplicate Composite Keys `(dataset_id, cell_id, cycle_id, time_sec)`:** `0`
- **Schema Conformity:** `100% True`
- **Breakdown by Chemistry:**
  - **LFP:** 338,133 rows
  - **NMC:** 135,436 rows
  - **Na-ion:** 41,701 rows

---

## 🔍 Dataset Governance, Inclusion & Exclusion Register

| Dataset ID | Chemistry | Raw Status | Audit Status | Master Status | Governance Findings & Audit Policies |
|---|---|---|---|---|---|
| **`NASA_Ames_PCoE`** | NMC | Present | Verified | **INCLUDED** | Cells B0005, B0006, B0018. Fully instrumented thermistors (22.4°C–42.0°C), voltage (2.0–4.2V), discharge capacity (1.15–2.04 Ah) and SOH (57.7%–101.8%). |
| **`Zenodo_IonTech_NaIon`** | Na-ion | Present | Verified | **INCLUDED** | 4 benchmark cells. **Zero Data Invention:** Raw pkl explicitly contains `temperature_in_C: None`; temperatures are strictly preserved as `NULL` with `TEMP_UNMEASURED` flag. |
| **`LFP_MultiStat_B1`** | LFP | Present | Verified | **INCLUDED** | 0.5C dis / 0.5C chg. Current header is `I(A/cm2)` while protocol is 0.625 A; preserved conditionally under `CURRENT_UNIT_UNDERDETERMINED`. Temp unmeasured. |
| **`LFP_MultiStat_B2`** | LFP | Missing | Verified | **DERIVED EXPORT** | Raw XLSX missing from workspace. Cleaned feature exports (`cycle_features.parquet`, `qv_curves.parquet`) available in `DATA/B2_LFP_Member1_audit.zip`. |
| **`LFP_MultiStat_B3`** | LFP | Present | Audited | **EXCLUDED** | 2C dis / 0.5C chg. Excluded from master telemetry stream to prevent data redundancy and memory inflation. |
| **`LFP_MultiStat_B4`** | LFP | Present | Audited | **BLOCKED** | 3C dis / 0.5C chg. Audit marked metrics as `NOT_CALCULATED` due to unresolvable conflict between `A/cm2` header and 3C (3.75A) protocol. **Excluded from ground truth.** |
| **`LFP_MultiStat_B5`** | LFP | Present | Verified | **INCLUDED** | 0.5C dis / 1C chg. Cleaned telemetry parquets included with `CHARGE_BALANCE_DISCREPANCY` flag ($Q_{\text{dis}} > Q_{\text{chg}}$ in 99.7% cycles). |
| **`LFP_MultiStat_B6`** | LFP | **MISSING**| Audit only | **BLOCKED** | Physical archive `B6_Discharge0.5C_Charge2C.zip` is completely absent from workspace. External audit summary is not conflated with raw data verification. |

---

## 🛠️ Repository Structure

```
├── README.md                              # Main project documentation & blueprint overview
├── master_table_build.py                  # Standalone, reproducible master table ingestion pipeline
├── master_table.parquet                   # Primary 13-column ML telemetry stream (8.8 MB, 515,270 rows)
├── master_table_final_validation.json     # Automated machine-readable validation metrics & SHA256 hashes
├── master_table_validation_report.json    # Initial verification summary report
├── MEMBER1_FINAL_VALIDATION_REPORT.md     # Member 1 comprehensive independent audit & handoff report
├── MEMBER1_STATUS_REPORT.md               # Member 1 status report
├── BLUEPRINT OF ENTIRE PROJ/              # System architecture specification & roadmap PDF
├── FORMULA/                               # Literature formula register & governing equations
└── AUDIT/                                 # Condition-level audit packages & provenance registers
```

---

## ⚙️ How to Reproduce Master Table Integration

Ensure dependencies are installed:
```bash
pip install pandas pyarrow scipy openpyxl fastparquet
```

Execute the deterministic ingestion pipeline:
```bash
python master_table_build.py
```

The script will ingest raw MAT, PKL, and verified XLSX/Parquet sources, enforce physical sanity bounds, check for duplicates, generate `master_table.parquet`, `master_table.csv`, and output verification statistics.

---

## 🤝 Downstream Handoff Contracts

- **Member 2 (Feature Engineering & ML Predictor):** Ingest directly from `master_table.parquet`. When extracting thermal stress features ($T > 35^\circ\text{C}$), filter by `chemistry == 'NMC'` or `temp_c IS NOT NULL`. Do NOT train thermal regressors on Na-ion or LFP datasets.
- **Member 3 (Vehicle Dynamics & Range Derivation):** Use validated `soh_pct` trajectories to drive the lumped-parameter vehicle road load model ($650\text{ km} \rightarrow 550\text{ km}$).
- **Member 4 (BMS Advisory):** Restrict charging bounds using the verified protocol C-rates ($0.5\text{C}$ to $2\text{C}$).
