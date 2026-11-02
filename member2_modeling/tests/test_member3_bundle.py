"""Regression tests for the snapshot Member 3 bundle record identifiers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from member2_modeling.snapshot_ml.member3_bundle import build_bundle, validate_bundle

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "snapshot_ml"
BUNDLE_FILES = (
    "ev_soh_predictions.csv",
    "solid_state_soh_predictions.csv",
    "ev_failure_predictions.csv",
    "ev_range_estimate.csv",
)


def test_bundle_snapshot_row_ids_are_non_null_and_unique(tmp_path: Path) -> None:
    destination = tmp_path / "bundle"
    result = build_bundle(source_root=ARTIFACT_ROOT, output=destination)

    assert result["validation"]["ok"], result["validation"]["problems"]
    for name in BUNDLE_FILES:
        frame = pd.read_csv(destination / name)
        assert not frame.empty, name
        assert frame["snapshot_row_id"].notna().all(), name
        assert frame["snapshot_row_id"].is_unique, name


def test_validate_bundle_rejects_duplicate_snapshot_row_ids(tmp_path: Path) -> None:
    destination = tmp_path / "bundle"
    build_bundle(source_root=ARTIFACT_ROOT, output=destination)

    path = destination / "ev_failure_predictions.csv"
    frame = pd.read_csv(path)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(path, index=False)

    validation = validate_bundle(destination)
    assert validation["ok"] is False
    assert any("duplicate" in problem for problem in validation["problems"])
