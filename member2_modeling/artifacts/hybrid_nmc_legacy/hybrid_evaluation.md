# Member 2 Hybrid Training Report — Legacy NMC

Source: `master_table.parquet`
SHA-256: `6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb`
Cells: B0005, B0006, B0018
Model-ready pairs: 465
Target: SOH of the next observed cycle record (positional shift by 1); not necessarily the next physical cycle.

## Evaluation

| Split | Rows | Seq avail. | Fusion MAE | Tabular MAE | Persistence MAE | Mean-fade MAE |
|---|---:|---:|---:|---:|---:|---:|
| val | 92 | 92 | 3.3822 | 4.4916 | 0.4704 | 0.3700 |
| test | 95 | 95 | 6.1105 | 8.8222 | 0.5040 | 0.4138 |

## Fusion usage

- Fusion inputs are chronological out-of-fold branch predictions.
- Rows without enough history use the learned tabular-only fallback.
- The model is EXPERIMENTAL and not production-validated.
