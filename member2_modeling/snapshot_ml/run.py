"""Reproduce the full Member 2 snapshot model suite.

Usage:
    python -m member2_modeling.snapshot_ml.run
    python -m member2_modeling.snapshot_ml.run --output <dir> --tasks ev_soh solid_soh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .audit import build_leakage_audit
from .inventory import build_inventory, inventory_markdown
from .registry import (
    REGISTRY_VERSION,
    build_registry,
    verify_registry,
    write_registry,
)
from .specs import TASKS, UNSUPPORTED_TASKS, task_ids
from .training import train_task

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "member2_modeling" / "artifacts" / "snapshot_ml"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _regression_row(task_id: str, metrics: dict[str, Any]) -> str:
    test = metrics["test_metrics"]
    median = metrics["baselines"]["train_median"]["mae"]
    chemistry = metrics["baselines"]["train_chemistry_median"]["mae"]
    improved = None if test["mae"] is None or median is None else test["mae"] < median
    return (
        f"| `{task_id}` | {metrics['selected_model']['name']} | {test['n']} | "
        f"{_fmt(test['mae'])} | {_fmt(test['rmse'])} | {_fmt(test['r2'])} | "
        f"{_fmt(test['bias'])} | {_fmt(median)} | {_fmt(chemistry)} | "
        f"{'yes' if improved else 'no'} |"
    )


def _classification_row(task_id: str, metrics: dict[str, Any]) -> str:
    test = metrics["test_metrics"]
    majority = metrics["baselines"]["majority_class"]
    chemistry = metrics["baselines"]["train_chemistry_majority"]
    improved = (
        None
        if test["balanced_accuracy"] is None or majority["balanced_accuracy"] is None
        else test["balanced_accuracy"] > majority["balanced_accuracy"]
    )
    return (
        f"| `{task_id}` | {metrics['selected_model']['name']} | {test['n']} | "
        f"{_fmt(test['accuracy'])} | {_fmt(test['balanced_accuracy'])} | "
        f"{_fmt(test['macro_f1'])} | {_fmt(test['log_loss'])} | "
        f"{_fmt(majority['balanced_accuracy'])} | {_fmt(chemistry['balanced_accuracy'])} | "
        f"{'yes' if improved else 'no'} |"
    )


def report_markdown(
    registry: dict[str, Any],
    inventory: dict[str, Any],
    verification: dict[str, Any],
    leakage: dict[str, Any],
) -> str:
    lines = [
        "# Member 2 snapshot ML report",
        "",
        f"Registry version: `{registry['registry_version']}`",
        "",
        "Protocol: entity-grouped, target-stratified 60/20/20 split. Preprocessing is",
        "fitted on the training split only, models are selected on validation only, and",
        "the test split is scored once after the configuration is frozen.",
        "",
        "## Dataset utilization",
        "",
        "| Dataset | Rows | Unique entities | Features | Targets modelled | Sequence |",
        "|---|---:|---:|---:|---|---|",
    ]
    for dataset_id, entry in inventory.items():
        targets = ", ".join(f"`{value}`" for value in entry["targets_used_by_member2"])
        lines.append(
            f"| `{dataset_id}` | {entry['records']:,} | {entry['unique_entities']:,} | "
            f"{entry['feature_count']} | {targets} | no |"
        )

    lines.extend(["", "## Regression tasks", ""])
    lines.append(
        "| Task | Selected model | Test n | MAE | RMSE | R2 | Bias | Train-median MAE | Chemistry-median MAE | Beats median |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for task_id, entry in registry["tasks"].items():
        if entry["task_type"] == "regression":
            lines.append(_regression_row(task_id, entry))

    lines.extend(["", "## Classification tasks", ""])
    lines.append(
        "| Task | Selected model | Test n | Accuracy | Balanced acc | Macro F1 | Log loss | Majority bal. acc | Chemistry-majority bal. acc | Beats majority |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for task_id, entry in registry["tasks"].items():
        if entry["task_type"] != "regression":
            lines.append(_classification_row(task_id, entry))

    lines.extend(["", "## Selected configuration per task", ""])
    lines.append("| Task | Status | Model | Key parameters | Split hash (first 16) |")
    lines.append("|---|---|---|---|---|")
    for task_id, entry in registry["tasks"].items():
        params = ", ".join(
            f"{key}={value}" for key, value in entry["selected_model"]["params"].items()
        )
        lines.append(
            f"| `{task_id}` | {entry['status']} | {entry['selected_model']['name']} | "
            f"{params} | `{entry['split_sha256'][:16]}` |"
        )

    lines.extend(["", "## Per-chemistry test performance", ""])
    for task_id, entry in registry["tasks"].items():
        per_chemistry = entry.get("test_metrics", {}).get("per_chemistry")
        if not per_chemistry:
            continue
        lines.append(f"### `{task_id}`")
        lines.append("")
        lines.append("| Chemistry | n | Primary metric |")
        lines.append("|---|---:|---:|")
        for chemistry, values in per_chemistry.items():
            metric = (
                values.get("mae")
                if entry["task_type"] == "regression"
                else values.get("balanced_accuracy")
            )
            lines.append(f"| {chemistry} | {values.get('n')} | {_fmt(metric)} |")
        lines.append("")

    lines.extend(["", "## Tasks not supported by the active data", ""])
    for name, entry in UNSUPPORTED_TASKS.items():
        lines.append(f"- **{name}** ({entry['status']}): {entry['reason']}")

    lines.extend(
        [
            "",
            "## Leakage controls",
            "",
            f"- Machine audit status: {'OK' if leakage['ok'] else 'FAILED'}.",
            "- Features come from the Member 1 data dictionary role `feature` only.",
            "- Target and every audited leakage column are rejected by the feature pipeline.",
            "- Splits are grouped by vehicle, cell or synthetic battery; overlap is recorded as zero.",
            "- Imputation medians and categorical vocabularies are fitted on the training split.",
            "- Test rows are never used for selection, early stopping, scaling or encoding.",
            "",
            "## Artifact verification",
            "",
            f"- Tasks registered: {verification['tasks']}",
            f"- Files hashed and checked: {verification['checked_files']}",
            f"- Checksum status: {'OK' if verification['ok'] else 'FAILED'}",
        ]
    )
    if verification["problems"]:
        lines.extend(["", "Problems:"])
        lines.extend(f"- {problem}" for problem in verification["problems"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=task_ids(),
        choices=task_ids(),
        help="Task ids to train (default: all).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    results = []
    for task_id in args.tasks:
        spec = TASKS[task_id]
        print(f"Training {task_id} ({spec.dataset_id} / {spec.target})...")
        result = train_task(spec, output_root=output, random_state=args.random_state)
        results.append(result)
        test = result["metrics"]["test"]
        primary = "mae" if spec.task_type == "regression" else "balanced_accuracy"
        print(
            f"  model={result['metrics']['selected_model']['name']} "
            f"test_{primary}={test[primary]:.4f} n={test['n']}"
        )

    registry = build_registry(results, output)
    write_registry(registry, output)

    inventory = build_inventory()
    (output / "dataset_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (output / "dataset_inventory.md").write_text(
        inventory_markdown(inventory), encoding="utf-8"
    )

    leakage = build_leakage_audit(registry)
    (output / "leakage_audit.json").write_text(
        json.dumps(leakage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    verification = verify_registry(output)
    (output / "registry_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "MEMBER2_SNAPSHOT_ML_REPORT.md").write_text(
        report_markdown(registry, inventory, verification, leakage), encoding="utf-8"
    )
    print(f"Registry: {output / 'model_registry.json'}")
    print(f"Report:   {output / 'MEMBER2_SNAPSHOT_ML_REPORT.md'}")
    print(
        f"Verification: {'OK' if verification['ok'] else 'FAILED'} "
        f"({verification['checked_files']} files)"
    )
    return 0 if verification["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
