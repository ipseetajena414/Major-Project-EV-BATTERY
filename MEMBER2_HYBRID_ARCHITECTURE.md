# Member 2 Hybrid Architecture — LightGBM + BiLSTM with Real Fusion

Artifacts: `member2_modeling/artifacts/hybrid_nmc_legacy/`
Source: `member2_modeling/hybrid/`
Dataset: `LEGACY_NMC_SEQUENTIAL` (NASA NMC cells `B0005`, `B0006`, `B0018`)
Readiness: **EXPERIMENTAL**, `production_ready = false`.

## 1. Why the three Member 1 datasets cannot train this model

The three Member 1 processed datasets are snapshot tables. Their registry entries
declare `sequence_available = false`; row order and cycle identifiers are not a
chronological degradation trajectory. Training a BiLSTM on them would invent history
that does not exist. They remain tabular-only inputs for the health models and for
experimentation.

The only sequential source is the pre-existing legacy NASA NMC table. It is used
explicitly as labelled legacy research input, never presented as a Member 1 processed
dataset.

## 2. Architecture

```
                 ┌──────────────────────────────┐
 per-cycle rows  │  causal sequence windows (5) │
 (cell_id,       └──────────────┬───────────────┘
  cycle_id)                     │
        │                       ▼
        │            ┌────────────────────┐
        ├───────────▶│  BiLSTM branch     │──┐ sequence_pred
        │            │  32 hidden, 1 layer│  │ (NaN when history < 5)
        │            └────────────────────┘  │
        │                                    ▼
        │            ┌────────────────────┐  ┌─────────────────────────┐
        └───────────▶│ LightGBM branch    │─▶│ out-of-fold Ridge fusion│──▶ fused SOH
                     │ tabular features   │  │ fit on OOF predictions  │
                     └────────────────────┘  └─────────────────────────┘
                              tabular_pred             ▲
                                                       │
                                    learned tabular-only fallback
                                    when sequence_pred is unavailable
```

Both branches predict the **same** target: SOH of the next observed cycle record
(positional shift by 1, not necessarily the next physical cycle).

## 3. Fusion is real, not two disconnected models

`member2_modeling/hybrid/fusion.py` — `OutOfFoldFusion`

1. Rows are assigned to folds chronologically **within each cell**.
2. Each branch predicts held-out fold rows. Those predictions are out-of-fold (OOF).
3. A Ridge regressor (`alpha = 1.0`) is fitted on `[tabular_pred, sequence_pred]`
   against the true target using only rows where both OOF predictions are finite.
4. A second Ridge fallback is fitted on `tabular_pred` alone.
5. At inference, rows with a sequence window use the fused regressor; rows without
   enough history use the learned fallback. Rows are never silently dropped.
6. Fused outputs are clipped to `[0, 100]`.

Training and inference therefore see the same distribution of branch predictions. The
fusion layer cannot learn to trust an in-sample prediction that is not reproducible at
inference time.

## 4. Training protocol

- Per-cell chronological 60/20/20 split.
- Model-ready pairs: 278 (91 train / 92 validation / 95 test).
- OOF folds: 3 configured, 3 effective.
- Fusion rows: 278; rows with sequence: 242.
- Sequence training windows: 226.
- Tabular LightGBM: 200 estimators, learning rate 0.05, 15 leaves, max depth 5,
  min child samples 10, `random_state = 42`; best iteration 115.
- BiLSTM: hidden 32, 1 layer, dropout 0.1, 60 epochs, batch 32, learning rate 1e-3,
  patience 10, `random_state = 42`.
- Sequence scaler fitted on training rows only.

## 5. Fitted fusion coefficients

| Mode | Intercept | `tabular_pred` | `sequence_pred` |
|---|---:|---:|---:|
| Fused (sequence available) | 84.1360 | 2.6769 | 3.8485 |
| Fallback (tabular only) | -59.0153 | 1.6905 | — |

## 6. Evaluation summary

Chronological test split, 95 rows, all with sequence history:

| Predictor | MAE | RMSE | R² |
|---|---:|---:|---:|
| Fused hybrid | 6.1105 | 6.5007 | -2.4039 |
| Tabular branch | 8.8222 | 9.2874 | -5.9477 |
| Sequence branch | 7.1294 | — | — |
| Persistence baseline | **0.5040** | 0.8669 | 0.9395 |
| Mean-fade baseline | **0.4138** | 0.8666 | 0.9395 |

The fusion improves on both learned branches but **loses to the simple baselines**.
This is recorded in `hybrid_evaluation.json` and repeated in every delivery document.

## 7. Artifact layout

| File | Contents |
|---|---|
| `hybrid_model/hybrid_config.json` | features, sequence length, folds, params |
| `hybrid_model/hybrid_schema.json` | full description, branch details, fusion coefficients |
| `hybrid_model/tabular_branch.joblib` | fitted LightGBM estimator |
| `hybrid_model/sequence_builder.joblib` | fitted sequence scaler/window builder |
| `hybrid_model/sequence_branch.pt` | fitted BiLSTM weights |
| `hybrid_model/fusion.joblib` | fused Ridge + fallback Ridge + scaler |
| `hybrid_manifest.json` | provenance, readiness, protected artifacts |
| `hybrid_evaluation.json` / `.md` | metrics and baseline comparisons |

## 8. Eligibility rules

`member2_modeling/hybrid/eligibility.py`:

| Dataset | Sequence | Hybrid status | Eligible components |
|---|---|---|---|
| `EV_FAILURE_PREDICTION` | No | UNAVAILABLE | tabular only |
| `SOLID_STATE_TELEMETRY_2026` | No | UNAVAILABLE | tabular only |
| `SYNTHETIC_BATTERY_LIFECYCLE` | No | UNAVAILABLE | tabular only |
| `LEGACY_NMC_SEQUENTIAL` | Yes | EXPERIMENTAL | tabular, sequence, fusion |

An unknown dataset ID raises `KeyError`.
