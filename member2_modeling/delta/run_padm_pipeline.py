"""
PADM Master Runner
==================

Executes the full 11-step PADM pipeline in sequence:

  Step 1-2  : Build delta pairs and chronological splits
  Step 3-4  : Fit scaler; compute train_mean_delta
  Step 5    : Feature set is locked in delta_features.PADM_FEATURE_COLS
  Step 6    : Hyperparameter selection on validation SOH MAE
  Step 7    : LOCO diagnostic
  Step 8    : Final model fitting on train + val
  Step 9    : Test evaluation (once only — guarded by sentinel file)
  Step 10   : Quality flag assignment (done inside evaluate)
  Step 11   : Member 3 CSV export

Usage
-----
  # From repo root, with .venv activated:
  python -m member2_modeling.delta.run_padm_pipeline

  # Or with explicit python path:
  .venv\\Scripts\\python.exe -m member2_modeling.delta.run_padm_pipeline

Flags
-----
  --skip-export   : Stop after Step 9 (skip Member 3 CSV generation)
  --force-eval    : Bypass the test-evaluation sentinel (USE WITH CAUTION)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Add workspace root to sys.path so member2_modeling is importable
WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

PARQUET_PATH = WORKSPACE / "master_table.parquet"

from member2_modeling.build_and_train import (
    build_cycle_feature_table,
    CYCLE_KEY,
)
from member2_modeling.delta.delta_features import (
    build_and_save_delta_dataset,
    PADM_FEATURE_COLS,
    ARTIFACT_DIR as DELTA_ARTIFACT_DIR,
)
from member2_modeling.delta.padm_train import train_padm
from member2_modeling.delta.padm_evaluate import evaluate_padm_on_test
from member2_modeling.delta.padm_export import build_padm_member3_exports


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main(skip_export: bool = False, force_eval: bool = False) -> None:
    t0 = time.time()

    print_section("MEMBER 2 — PADM PIPELINE")
    print(f"  Workspace:    {WORKSPACE}")
    print(f"  Parquet:      {PARQUET_PATH}")
    print(f"  Artifact dir: {DELTA_ARTIFACT_DIR}")

    # ── Verify parquet ──────────────────────────────────────────────────────
    if not PARQUET_PATH.exists():
        print(f"\nERROR: master_table.parquet not found at {PARQUET_PATH}")
        sys.exit(1)

    with open(PARQUET_PATH, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f"\n  master_table.parquet SHA256: {sha}")

    # ── Load telemetry ──────────────────────────────────────────────────────
    print_section("Step 1-2 | Loading telemetry and building delta pairs")
    import warnings
    warnings.filterwarnings("ignore")

    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Loaded {len(df):,} telemetry rows, "
          f"{df['chemistry'].nunique()} chemistries")

    # Exclude LFP B1 (1 cycle — cannot train)
    df = df[df["dataset_id"] != "LFP_MultiStat_B1"].copy()

    # Build cycle-level feature table (NMC, LFP, Na-ion all extracted)
    print("\n  Extracting cycle-level features...")
    cycle_df = build_cycle_feature_table(df)
    cycle_df = cycle_df.dropna(subset=["soh_pct"]).copy()
    cycle_df = cycle_df.sort_values(CYCLE_KEY)
    cycle_df["cycle_position"] = cycle_df.groupby("cell_id").cumcount()

    nmc_cycles = cycle_df[cycle_df["chemistry"] == "NMC"]
    print(f"  NMC cycles: {len(nmc_cycles)} across "
          f"{nmc_cycles['cell_id'].nunique()} cells")

    # Build delta pairs, split, impute, save parquet
    dataset_result = build_and_save_delta_dataset(
        cycle_df,
        artifact_dir=DELTA_ARTIFACT_DIR,
    )
    print(f"\n  Rows per split: {dataset_result['rows_per_split']}")
    print(f"  Locked feature columns ({len(dataset_result['feature_cols'])}):")
    for fc in dataset_result['feature_cols']:
        print(f"    - {fc}")

    # ── Load processed delta dataset ────────────────────────────────────────
    delta_df = pd.read_parquet(DELTA_ARTIFACT_DIR / "delta_pairs_nmc.parquet")

    # ── Steps 3-8: Train PADM ───────────────────────────────────────────────
    print_section("Steps 3-8 | Training PADM (Ridge + ElasticNet)")
    train_result = train_padm(
        delta_df,
        feature_cols=PADM_FEATURE_COLS,
        artifact_dir=DELTA_ARTIFACT_DIR,
    )

    print(f"\n  Hyperparameters selected:")
    print(f"    Ridge alpha:           {train_result['hyperparameters']['ridge_alpha']}")
    print(f"    ElasticNet alpha:      {train_result['hyperparameters']['enet_alpha']}")
    print(f"    ElasticNet l1_ratio:   {train_result['hyperparameters']['enet_l1_ratio']}")
    print(f"\n  Validation SOH MAE:")
    print(f"    Ridge:      {train_result['validation_mae']['ridge']:.4f}%")
    print(f"    ElasticNet: {train_result['validation_mae']['elasticnet']:.4f}%")
    print(f"\n  LOCO pooled SOH MAE: "
          f"{train_result['loco_diagnostic']['pooled_soh_mae']:.4f}%")

    # ── Step 9-10: Test evaluation ──────────────────────────────────────────
    print_section("Steps 9-10 | Test Evaluation (single run)")
    eval_result = evaluate_padm_on_test(
        delta_df,
        feature_cols=PADM_FEATURE_COLS,
        artifact_dir=DELTA_ARTIFACT_DIR,
        force=force_eval,
    )

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │           FINAL TEST EVALUATION RESULTS             │")
    print("  ├─────────────────────────────┬───────────────────────┤")
    for model_key in ["ridge", "elasticnet", "persistence", "mean_fade"]:
        m = eval_result["pooled_metrics"][model_key]
        label_map = {
            "ridge": "PADM-Ridge        ",
            "elasticnet": "PADM-ElasticNet   ",
            "persistence": "Persistence       ",
            "mean_fade": "Mean-Fade         ",
        }
        mae_str = f"{m['mae']:.4f}%" if m['mae'] is not None else "N/A"
        print(f"  │ {label_map[model_key]} │ MAE = {mae_str:>8}           │")
    print("  └─────────────────────────────┴───────────────────────┘")

    print(f"\n  Best PADM model:    {eval_result['best_padm_model']}")
    print(f"  Ridge flag:         {eval_result['quality_flags']['ridge']}")
    print(f"  ElasticNet flag:    {eval_result['quality_flags']['elasticnet']}")
    print(f"  Export model:       {eval_result['export_model']}")
    print(f"  Gate tests passed:  {eval_result['all_gates_passed']}")
    if not eval_result["all_gates_passed"]:
        failed = [k for k, v in eval_result["gate_tests"].items() if not v]
        print(f"  FAILED GATES:       {failed}")

    # ── Step 11: Member 3 export ────────────────────────────────────────────
    if skip_export:
        print("\n  [--skip-export] Member 3 CSV export skipped.")
    else:
        print_section("Step 11 | Member 3 CSV Export")
        if not eval_result["all_gates_passed"]:
            print("  WARNING: Not all gate tests passed.")
            print("  Member 3 export will be attempted but may raise an error.")

        export_result = build_padm_member3_exports(
            artifact_dir=DELTA_ARTIFACT_DIR,
        )
        print(f"\n  Exported {export_result['rows']} rows to {export_result['out_dir']}")
        print(f"  Excluded rows: {export_result['excluded_rows']}")
        print(f"  Export model:  {export_result['export_model']}")
        print(f"  Quality flag:  {export_result['quality_flag']}")
        print(f"\n  Predicted MAE (Member 3 export): "
              f"{export_result['metrics']['predicted']['mae']:.4f}%")
        print(f"  Baseline MAE  (Member 3 export): "
              f"{export_result['metrics']['baseline']['mae']:.4f}%")

    elapsed = time.time() - t0
    print_section(f"PADM PIPELINE COMPLETE ({elapsed:.1f}s)")
    print("  All artifacts saved to:")
    print(f"    {DELTA_ARTIFACT_DIR}")
    print("\n  Files generated:")
    for path in sorted(DELTA_ARTIFACT_DIR.iterdir()):
        print(f"    {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PADM pipeline")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Stop after test evaluation; skip Member 3 CSV generation",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Bypass test-evaluation sentinel (use with caution)",
    )
    args = parser.parse_args()
    main(skip_export=args.skip_export, force_eval=args.force_eval)
