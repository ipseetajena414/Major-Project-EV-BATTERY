# Member 2 Chemistry Comparison

Source: `member2_modeling/artifacts/health/health_model_evaluation.json`
(`chemistry_comparison` and `datasets.*.targets.*.evaluation.per_chemistry`).

Chemistry is used as an input category (one-hot) and as an evaluation stratifier.
Metrics below are test-split values from deterministic stratified **snapshot** splits,
not temporal validation.

## 1. EV_FAILURE_PREDICTION — `remaining_capacity` (kWh pack energy)

| Chemistry | n | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| LFP | 1,139 | **3.2761** | 5.2622 | 0.9707 | 5.33% |
| LMO | 573 | 3.4679 | 5.4495 | 0.9672 | 5.14% |
| LTO | 261 | 3.6312 | 5.6396 | 0.9679 | 5.96% |
| NCA | 275 | **3.9314** | 6.7969 | 0.9507 | 6.16% |
| NMC | 1,500 | 3.4462 | 5.5827 | 0.9663 | 5.45% |

Best MAE: LFP. Worst MAE and RMSE: NCA.

## 2. EV_FAILURE_PREDICTION — `state_of_health` (%)

| Chemistry | n | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| LFP | 1,134 | 2.7161 | 3.4773 | 0.8503 | 3.43% |
| LMO | 560 | **2.7037** | 3.4739 | 0.8672 | 3.41% |
| LTO | 245 | **2.8592** | 3.5003 | 0.8439 | 3.55% |
| NCA | 264 | 2.7699 | 3.5252 | 0.8484 | 3.44% |
| NMC | 1,458 | 2.8302 | 3.5471 | 0.8436 | 3.55% |

Differences are small (0.16 percentage points MAE spread). LMO has the best MAE and R²;
LTO the worst MAE, NMC the worst R².

## 3. SOLID_STATE_TELEMETRY_2026 — `state_of_health_pct` (%)

| Chemistry | n | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| Solid-State | 2,000 | 0.6296 | 0.7940 | 0.9857 | 0.73% |

Only one chemistry label exists, so this is not a comparison across chemistries.

## 4. SYNTHETIC_BATTERY_LIFECYCLE — `discharge_capacity_ah` (Ah, synthetic)

| Chemistry | n | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| Lithium-Sulfur | 138 | 4.4677 | 5.6592 | 0.7313 | **4.71%** |
| Sodium-ion | 134 | **3.7759** | **5.0989** | 0.7312 | 6.92% |
| Solid-State | 128 | 4.2119 | 5.5443 | 0.6723 | 5.51% |

Sodium-ion has the best MAE/RMSE; Lithium-Sulfur the best MAPE and R²; Solid-State the
worst R². All labels are synthetic, so no chemistry conclusion is experimentally
established.

## 5. Hybrid NMC (legacy NASA)

The hybrid model is trained and evaluated only on NMC. It has no cross-chemistry
counterpart in this repository, so it provides no comparison evidence.

## 6. Interpretation rules

- Per-chemistry metrics are descriptive only. No statistical significance test was run.
- Chemistry groups are unbalanced (e.g., NCA n=275 vs NMC n=1,500), so small-group
  differences are noisy.
- Snapshot splits do not test degradation over time.
- Synthetic chemistry rankings must not be presented as real-world battery rankings.
- No chemistry-specific production model exists.
