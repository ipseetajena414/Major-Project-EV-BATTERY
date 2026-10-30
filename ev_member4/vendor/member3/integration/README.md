# Member 2 → Member 3 Integration (File-Based Boundary)

This package connects Member 2's SOH predictions to Member 3's range calculator using a CSV artifact boundary. The range calculator has **no deep-learning dependency**; Member 2 artifacts are never imported at runtime.

## Files

| Path | Role |
| :--- | :--- |
| `member2_range_adapter.py` | Validates prediction CSVs, builds the SOH trajectory, calls `RangeCalculator.estimate_trajectory()`, exports results CSV. |
| `predictions/PROVENANCE.json` | Exact provenance of the prediction artifacts (commit, SHA256, environment, scope, reproduced metrics, caveats). |
| `predictions/member2_nmc_B0005_bilstm_predictions.csv` | Genuine Bi-LSTM inference output, 30 rows, cycles 139–168 (see caveats). |
| `predictions/member2_nmc_B0005_lgbm_predictions.csv` | Served `Member2Predictor` (LightGBM) output for the same windows. |
| `outputs/*.csv` | Range trajectories produced by the adapter. |

## Prediction CSV contract

Required columns:

| Column | Type | Meaning |
| :--- | :--- | :--- |
| `cycle_index` | integer, strictly increasing per cell, unique per `(cell_id, cycle_index)` | Index of the predicted record in the source dataset's cycle sequence. |
| `cell_id` | non-empty string | Cell identifier, preserved verbatim into the output. |
| `chemistry` | `NMC` or `LFP` | Only chemistries covered by Member 2 models. Na-ion is rejected. |
| `soh_pct` | float, `1.0 < soh <= 100.0` | State of health in **percentage points** on the 0–100 scale. |

Optional columns honored: `unit` (must be `%`), `source` (`predicted` / `observed`, default `predicted`), `current_observed_soh_pct`, `observed_next_soh_pct`, `quality_flag`, `split`, `model_name`, `model_status`, `horizon_cycles`, `window_end_cycle_id`.

Rejected inputs: missing file, missing columns, non-`%` units, Na-ion or unknown chemistry, duplicate `(cell_id, cycle_index)`, non-strictly-increasing `cycle_index`, SOH expressed as a fraction (0–1), SOH ≤ 0, SOH > 100, and non-finite values.

## Quality flags

`source=predicted` rows default to `MEMBER2_PREDICTION_UNVERIFIED`; `source=observed` rows default to `PERSISTENCE_BASELINE_FROM_OBSERVED_SOH`. The flag is passed through `estimate_trajectory()` unchanged so it survives into the output CSV. Never label synthetic trajectories `BOL_MEASURED` or `LSTM_PREDICTED`.

## Usage

```bash
cd member3
python -m integration.member2_range_adapter \
  --predictions integration/predictions/member2_nmc_B0005_bilstm_predictions.csv \
  --vehicle-config vehicle_configs/tata_curvv_ev_55kwh_partial_reference.json \
  --cycle drive_cycles/udds_city.csv \
  --out integration/outputs/nmc_B0005_bilstm_range_curvv_udds.csv \
  --ambient-c 30
```

`--soh-field current_observed_soh_pct` evaluates the observed-anchor baseline instead of the prediction. `--quality-flag` overrides the default flag (use only when a documented pedigree exists).

Output columns: `cycle_index, cell_id, chemistry, soh_pct, soh_source, quality_flag, estimated_range_km, consumption_kwh_per_100km, current_observed_soh_pct, observed_next_soh_pct, split, model_name, model_status, vehicle_name, estimate_quality_status, ambient_c, hvac_power_kw, vehicle_config, drive_cycle, generated_at_utc, notes`.

## How the prediction artifacts were generated

Both CSVs come from re-executing Member 2's own code against `master_table.parquet` (SHA256 `6fa7475d…b2bb`) in an isolated venv outside the repository, using the exact dependency versions from `member2_modeling/model_manifest.json`. Member 2's model, preprocessing, checkpoint and prediction code were **not modified**.

- **Bi-LSTM:** `BiLSTMPredictor(input_dim=37, hidden_dim=64, num_layers=2, dropout=0.2)` loaded from `nmc_bilstm_model.pt`, with `scaler_nmc_seq.joblib` and 5-record windows.
- **LightGBM:** the served `Member2Predictor.predict_future_soh()` path (`nmc_lgbm`, 37 features).

Correctness cross-check: re-executing Member 2's evaluation on cell B0005's test split reproduces the manifest metrics (Bi-LSTM MAE 4.159 vs 4.16 reported, LightGBM 8.2945 vs 8.29, persistence 0.3172 vs 0.32). See `predictions/PROVENANCE.json`.

## Limitations

- Both models **underperform a persistence baseline** on Member 2's chronological holdout (Bi-LSTM 4.16, LightGBM 8.29, persistence 0.32 %MAE on B0005).
- `nmc_bilstm` is **not served** by `Member2Predictor`; the API serves LightGBM. The Bi-LSTM artifact was executed through Member 2's own predictor class for research comparison only.
- `cycle_index` is the source dataset's record index and the horizon is one *observed record* (positional shift), not one physical cycle.
- SOH is **cell-level**; Member 3 maps it to pack energy with a linear approximation, not a validated pack model.
- No uncertainty quantification, calibration, or out-of-distribution detection exists.
- RUL from Member 2 is unvalidated and is not consumed here.
- Range outputs combine an experimental SOH model with `PARTIALLY_SOURCED_UNCALIBRATED` vehicle dynamics; they are **not validated real-world range predictions**.
