# Autonomous Multi-Agent EV Battery Life & Range Predictor

> **Analyzing real-world range decay (650 km -> 550 km) across NMC, LFP,
> sodium-ion and solid-state batteries with open-source AI agents and machine
> learning.**

---

## Project overview

Modern EVs often lose real-world range relative to their factory rating due to
chemical degradation, temperature extremes, aggressive DC fast charging and HVAC
loads. This project combines physics-based battery models, supervised ML and a
multi-agent advisory layer.

Member 1 owns ingestion, preprocessing and data governance. The active Member 1
datasets are the three snapshot datasets listed below; the earlier
NASA/BatteryArchive-derived cell-cycle master table is retired and archived.

---

## Architecture (5-team workload)

```
DATA SOURCES
  EV failure/pack snapshots | Solid-state cell snapshots | Synthetic lifecycle
        |
        v
MEMBER 1 - Ingestion, cleaning, quality flags, leakage audit, dataset registry
        |  DATA/processed/*.parquet + *.json
        v
MEMBER 2 - Feature engineering and ML models (tabular snapshots)
        |  SOH / failure / degradation signals
        v
MEMBER 3 - Vehicle dynamics, road loads, HVAC penalties, range derivation
        |  Pack state and range estimates
        v
MEMBER 4 - Multi-agent driver/BMS simulation and prescriptive advice
        |  Simulation results and advisory payloads
        v
MEMBER 5 - RAG literature synthesis and Streamlit dashboard
```

---

## Member 1: active datasets

| Dataset ID | Source file | Rows | Processed columns | Grain | Origin |
|---|---|---:|---:|---|---|
| `EV_FAILURE_PREDICTION` | `ev battery_failure prediction Dataset.csv` | 20,000 | 82 | one snapshot per vehicle/pack | source dataset |
| `SOLID_STATE_TELEMETRY_2026` | `solid_state_ev_battery_telemetry_enriched_2026.csv` | 10,000 | 41 | one snapshot per cell | source dataset |
| `SYNTHETIC_BATTERY_LIFECYCLE` | `synthetic_battery_lifecycle.csv` | 2,000 | 20 | one snapshot per synthetic battery | synthetic |

The three datasets are processed separately. They are **not** merged into a
cell-cycle master table, because none of them contains a real cell-level
voltage/current/temperature time series.

### Processed outputs (`DATA/processed/`)

| File | Contents |
|---|---|
| `ev_battery_failure_clean.parquet` | EV/pack snapshot dataset |
| `solid_state_telemetry_clean.parquet` | Solid-state cell snapshot dataset |
| `synthetic_battery_lifecycle_clean.parquet` | Synthetic lifecycle dataset |
| `dataset_inventory.json` | Counts, dtypes, missing, duplicates, IDs, chemistry, ranges, sequence checks |
| `data_quality_report.json` | Cleaning summary, coercions, quality-flag counts, outliers, decisions |
| `data_dictionary.csv` | 143-column contract with role, unit and missing count |
| `leakage_audit.json` | Targets, identifiers, allowed features, excluded fields, evidence |
| `dataset_registry.json` | Active registry, hashes, grain, sequence flags, legacy index |

Each processed row has 12 added columns: `source_row_id`, `dataset_id`,
`data_origin`, `source_file`, `chemistry_family`, `missing_field_count`,
`missing_fields`, `outlier_field_count`, `outlier_fields`, `invalid_field_count`,
`invalid_fields` and `quality_flag`.

### Targets and leakage

| Dataset | Target | Candidate target |
|---|---|---|
| EV | `battery_failure` (binary classification) | `state_of_health`, `battery_health_percent` |
| Solid-state | `degradation_risk_tier` (4-class, never a feature) | `state_of_health_pct` |
| Synthetic | `state_of_health` (3-class categorical) | none |

Excluded as leakage or non-measured values:

- EV: `predicted_remaining_life_cycles` (prediction, not RUL),
  `capacity_loss_percent` (exactly 100 minus `battery_health_percent`), and both
  SOH labels when the other is the target.
- Solid-state: `cot_electrochemical_risk_rationale` (restates SOH),
  `projected_cycle_life_80pct_soh` (projected; below the current cycle for 19.0%
  of rows), `confidence_score`, `thermal_stability_margin_c` (exact deterministic
  derivative) and three enriched risk scores.
- Synthetic: no numeric SOH or RUL exists; none is invented.

Allowed features after exclusions: 56 (EV), 19 (solid-state), 6 (synthetic).

### Sequence availability

All three datasets are snapshot-only:

- EV: one row per `vehicle_id` and per `battery_serial`.
- Solid-state: one row per `cell_id`.
- Synthetic: 1,978 `battery_id` values across 2,000 rows, with conflicting
  electrolyte labels on 11 repeated IDs.

`loader.build_sequences()` raises `NoSequenceDataError` for every dataset. Do not
train a BiLSTM or other sequence model on these files without new time-series
data.

### Synthetic data

`synthetic_battery_lifecycle.csv` is generated. Its SOH label is categorical
only; the pipeline never converts it to a percentage or invents RUL. Keep it
separate from source datasets (`data_origin == "synthetic"`).

---

## Repository structure

```
├── README.md
├── docs/DATA_PIPELINE.md                 # Full data contract and cleaning rules
├── member1_pipeline/                     # Active ingestion pipeline and loader
│   ├── config.py                         # Dataset specs, roles, ranges, vocabularies
│   ├── cleaning.py                       # Type coercion, flags, outlier and range checks
│   ├── audit.py                          # Inventory, quality, dictionary, leakage, registry
│   ├── build.py                          # Build and validation CLI
│   └── loader.py                         # Downstream loading interface
├── DATA/processed/                       # Tracked processed outputs (raw archives stay local)
├── tests/                                # Pipeline contract tests
├── legacy/member1/                       # Retired master table artifacts (audit only)
├── build_master_table.py                 # Deprecated shim -> member1_pipeline.build
├── AUDIT/                                # Historical audit packages
├── BLUEPRINT OF ENTIRE PROJ/             # Architecture specification
└── FORMULA/                              # Literature formula register
```

---

## Reproduce

```bash
.venv/Scripts/python.exe -m member1_pipeline.build
.venv/Scripts/python.exe -m member1_pipeline.build --check
.venv/Scripts/python.exe -m pytest tests -q
```

Requires pandas, pyarrow and pytest. The build is deterministic, reads raw files
read-only, drops zero rows, performs no imputation, and re-checks raw SHA-256
hashes after writing. `--check` verifies the current raw hashes, processed
hashes, dataset identity, required schema, row/column counts and dtypes against
`dataset_inventory.json` and `dataset_registry.json`, not just row counts.

Declared integer fields containing fractional values are not rounded; the values
are preserved, flagged `NON_INTEGER_VALUE`, and listed in `invalid_fields`.

`python build_master_table.py` now prints a deprecation notice and delegates to
the pipeline. It no longer writes a master table.

---

## Downstream loading

```python
from member1_pipeline import loader

ev = loader.load_ev_failure()
solid = loader.load_solid_state()
synthetic = loader.load_synthetic()

features = loader.feature_columns("EV_FAILURE_PREDICTION")
targets = loader.target_columns("EV_FAILURE_PREDICTION")
excluded = loader.excluded_columns("SOLID_STATE_TELEMETRY_2026")
```

- Load datasets separately; do not fuse different grains.
- Use the loader's feature/target/excluded helpers instead of hand-picked
  columns.
- Solid-state and synthetic temperatures are not cell-level cycle histories;
  do not train thermal-stress features across datasets.
- Read `DATA/processed/dataset_registry.json` for counts, hashes, grain and
  sequence availability.

Member 2's training script and Member 5's `MASTER_TABLE` path still point at the
retired root `master_table.parquet`. That file is archived. Downstream code must
switch to `member1_pipeline.loader`, or opt in explicitly with
`loader.load_legacy_master_table(allow_legacy=True)` for an audit. Member 2-5
model, range, agent and dashboard code was not modified.

---

## Legacy data

Retired artifacts are preserved in `legacy/member1/` with SHA-256 hashes and
restore instructions. The archived root parquet had 23,768 rows and 8 columns,
with hardcoded Na-ion voltage/current and a resistance-derived Na-ion SOH. The
archived 4,789-row CSV was a partial LFP B1-only build. Neither is an active
input, and `master_table_build.py` must not be run.

See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for the full contract.
