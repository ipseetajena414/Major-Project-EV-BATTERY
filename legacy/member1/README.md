# Legacy Member 1 artifacts (retired)

These files are preserved for traceability only. None of them is an active
input to the current pipeline or to downstream Members 2–5.

## Why they were retired

The earlier workflow mapped three snapshot datasets into a single synthetic
cell-cycle table:

- Na-ion `voltage_v` was hardcoded to `3.1` and `current_a` to `15.0`.
- Na-ion `soh_pct` was computed as `100 - internal_resistance_mohms / 100`,
  which is not a measured state of health.
- Solid-state `capacity_ah` was set to `nominal_capacity_ah`, which is a rating,
  not a measured discharge capacity.
- `time_sec` and `current_a` were algebraically reconstructed from other
  columns instead of being observed.
- The resulting table had no provenance columns and no per-row quality flags.
- Rows were dropped (`dropna`) without an audit trail.

The current pipeline in `member1_pipeline/` never creates those fabricated
columns. It keeps the three datasets separate, adds provenance and quality
flags, retains every row, and never imputes values.

## Inventory

| File | Rows | Columns | SHA-256 |
|---|---:|---:|---|
| `master_table_8col_fabricated.parquet` | 23,768 | 8 | `e5f699e19626559d50495e4d9e564086682f9b8eb2e47a4cc42da9162f1bd18c` |
| `master_table_4789_LFP_B1_regression.csv` | 4,789 | 13 | `b700a1b8c23211d8e3bfbbec9c81087be1dcd929ec0a91ac8f41bbe4b606ebc7` |
| `master_table_build.py` | — | — | `3adb5776439c3bac6318d9ce73be62e41328ff69a73e54b254f4bce6f6e732da` |
| `master_table_final_validation.json` | — | — | `e4bb771cd010ed0eadb358cb877eba9a492bc30bdc09e3bde0a2e5197510112a` |
| `master_table_validation_report.json` | — | — | `b2c23fa9b8746db80f75d0712f785e466d9f982008aefdb2fb92a09bc4b2bd05` |
| `MEMBER1_STATUS_REPORT.md` | — | — | `49f5086f72a16c97bf0db2d7b7b6ba74eee2b40888f0cc7cd639d10e72700c02` |
| `MEMBER1_FINAL_VALIDATION_REPORT.md` | — | — | `21b16b31117949e1444ee50aaaa175fe9937f688c71a06d277ce456b58663c52` |

The archived 8-column parquet was the root `master_table.parquet`. Its chemistry
row counts were Solid-State 10,000, NMC 6,623, LFP 5,145 and Na-ion 2,000.

`master_table_build.py` must not be run in this checkout. It requires
`DATA/archive (4).zip`, `DATA/NA-ion.zip`, `AUDIT/B1_LFP_Audit_Output.zip` and
`AUDIT/B5.zip`, none of which is present. A previous run produced only the
4,789-row LFP B1 file above.

## Restore (only if you deliberately need the retired table)

```bash
cp legacy/member1/master_table_8col_fabricated.parquet master_table.parquet
```

Restoring it makes it the accidental primary input again. Prefer loading the
active datasets through `member1_pipeline.loader`.
