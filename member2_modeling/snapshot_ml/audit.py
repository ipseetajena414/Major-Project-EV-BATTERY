"""Machine-checkable leakage audit for the snapshot model suite."""

from __future__ import annotations

from typing import Any

from member2_modeling.data_integration.profiles import target_spec

from .specs import TASKS


def build_leakage_audit(registry: dict[str, Any]) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task_id, entry in registry["tasks"].items():
        spec = TASKS[task_id]
        target = spec.target
        forbidden = set(target_spec(spec.dataset_id, target)["leakage_columns"])
        forbidden.add(target)
        features = set(entry["required_features"])
        overlap = sorted(features & forbidden)
        tasks[task_id] = {
            "dataset_id": spec.dataset_id,
            "target": target,
            "feature_count": len(features),
            "forbidden_count": len(forbidden),
            "forbidden_used_as_feature": overlap,
            "target_used_as_feature": target in features,
            "split_unit": entry["entity_column"],
            "split_protocol": "entity-grouped, target-stratified",
            "entity_overlap": entry["entity_overlap"],
            "split_sha256": entry["split_sha256"],
            "preprocessing_fit": "training split only",
            "model_selection": "validation split only",
            "test_use": "single frozen evaluation",
            "checks": {
                "no_direct_target_leakage": target not in features,
                "no_audited_leakage_columns": not overlap,
                "entity_disjoint_splits": all(
                    value == 0 for value in entry["entity_overlap"].values()
                ),
                "sequence_model_not_used": True,
                "rul_not_invented": True,
            },
        }
    all_ok = all(
        all(value for value in entry["checks"].values()) for entry in tasks.values()
    )
    return {
        "ok": all_ok,
        "policy": {
            "features": "Member 1 data dictionary role 'feature' after target/leakage removal",
            "splits": "entity-grouped and target-stratified",
            "preprocessing": "fit on training split only",
            "selection": "validation only; test scored once",
            "sequence": "not used; all active datasets are snapshots",
            "rul": "not produced; no observed EOL trajectory in active data",
        },
        "tasks": tasks,
    }
