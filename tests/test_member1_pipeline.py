"""Contract tests for the active Member 1 data pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from member1_pipeline import build, cleaning, loader
from member1_pipeline.config import (
    DATASETS,
    LEGACY_DIR,
    REPO_ROOT,
    dataset_spec,
    normalize_chemistry,
)
from member1_pipeline.cleaning import PROVENANCE_COLUMNS, QUALITY_COLUMNS, sha256_file

DATASET_IDS = list(DATASETS)
EXPECTED_FILES = {
    "EV_FAILURE_PREDICTION": "ev_battery_failure_clean.parquet",
    "SOLID_STATE_TELEMETRY_2026": "solid_state_telemetry_clean.parquet",
    "SYNTHETIC_BATTERY_LIFECYCLE": "synthetic_battery_lifecycle_clean.parquet",
}
EXPECTED_ROWS = {
    "EV_FAILURE_PREDICTION": 20_000,
    "SOLID_STATE_TELEMETRY_2026": 10_000,
    "SYNTHETIC_BATTERY_LIFECYCLE": 2_000,
}
REQUIRED_PROVENANCE = ["source_row_id", "dataset_id", "data_origin", "source_file", "quality_flag"]


@pytest.fixture(scope="module")
def processed(processed_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        dataset_id: loader.load_dataset(dataset_id, processed_dir=processed_dir)
        for dataset_id in DATASET_IDS
    }


@pytest.fixture(scope="module")
def registry(processed_dir: Path) -> dict:
    return loader.load_registry(processed_dir)


@pytest.fixture(scope="module")
def leakage(processed_dir: Path) -> dict:
    return loader.load_leakage_audit(processed_dir)


@pytest.fixture(scope="module")
def inventory(processed_dir: Path) -> dict:
    return loader.load_inventory(processed_dir)


@pytest.fixture(scope="module")
def dictionary(processed_dir: Path) -> pd.DataFrame:
    return loader.load_data_dictionary(processed_dir)


@pytest.fixture(scope="module")
def quality(processed_dir: Path) -> dict:
    return json.loads((processed_dir / "data_quality_report.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw() -> dict[str, pd.DataFrame]:
    return {
        dataset_id: pd.read_csv(
            REPO_ROOT / dataset_spec(dataset_id)["source_file"], low_memory=False
        )
        for dataset_id in DATASET_IDS
    }


# ---------------------------------------------------------------- outputs exist


def test_processed_outputs_exist(processed_dir: Path) -> None:
    for filename in EXPECTED_FILES.values():
        assert (processed_dir / filename).exists(), filename
    for report in [
        "dataset_inventory.json",
        "data_quality_report.json",
        "data_dictionary.csv",
        "leakage_audit.json",
        "dataset_registry.json",
    ]:
        assert (processed_dir / report).exists(), report


def test_output_check_passes(processed_dir: Path) -> None:
    result = build.check_artifacts(processed_dir)
    assert result["ok"], result["errors"]


def _copy_processed(source: Path, destination: Path) -> Path:
    target = destination / "processed"
    shutil.copytree(source, target)
    return target


def _edit_json(path: Path, mutate) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_check_detects_raw_source_hash_mismatch(
    processed_dir: Path, tmp_path: Path
) -> None:
    copy = _copy_processed(processed_dir, tmp_path)

    def mutate(data: dict) -> None:
        data["active_datasets"]["EV_FAILURE_PREDICTION"]["source_sha256"] = "0" * 64

    _edit_json(copy / "dataset_registry.json", mutate)
    result = build.check_artifacts(copy)
    assert not result["ok"]
    assert any("raw source hash" in error.lower() for error in result["errors"])


def test_check_detects_inventory_source_hash_mismatch(
    processed_dir: Path, tmp_path: Path
) -> None:
    copy = _copy_processed(processed_dir, tmp_path)

    def mutate(data: dict) -> None:
        data["SOLID_STATE_TELEMETRY_2026"]["source_sha256"] = "0" * 64

    _edit_json(copy / "dataset_inventory.json", mutate)
    result = build.check_artifacts(copy)
    assert not result["ok"]
    assert any("raw source hash" in error.lower() for error in result["errors"])


def test_check_detects_processed_hash_mismatch(
    processed_dir: Path, tmp_path: Path
) -> None:
    copy = _copy_processed(processed_dir, tmp_path)
    path = copy / "ev_battery_failure_clean.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "odometer_km"] = float(frame.loc[0, "odometer_km"]) + 1.0
    frame.to_parquet(path, index=False)

    result = build.check_artifacts(copy)
    assert not result["ok"]
    assert any("processed hash" in error.lower() for error in result["errors"])


def test_check_detects_missing_required_schema(
    processed_dir: Path, tmp_path: Path
) -> None:
    copy = _copy_processed(processed_dir, tmp_path)
    path = copy / "ev_battery_failure_clean.parquet"
    frame = pd.read_parquet(path).drop(columns=["battery_failure"])
    frame.to_parquet(path, index=False)

    result = build.check_artifacts(copy)
    assert not result["ok"]
    assert any("battery_failure" in error for error in result["errors"])


def test_check_detects_dataset_identity_mismatch(
    processed_dir: Path, tmp_path: Path
) -> None:
    copy = _copy_processed(processed_dir, tmp_path)
    path = copy / "solid_state_telemetry_clean.parquet"
    frame = pd.read_parquet(path)
    frame["dataset_id"] = "WRONG_DATASET"
    frame.to_parquet(path, index=False)

    result = build.check_artifacts(copy)
    assert not result["ok"]
    assert any("dataset_id" in error for error in result["errors"])


def test_check_detects_null_dataset_identity(
    processed_dir: Path, tmp_path: Path
) -> None:
    copy = _copy_processed(processed_dir, tmp_path)
    path = copy / "ev_battery_failure_clean.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "dataset_id"] = None
    frame.to_parquet(path, index=False)

    result = build.check_artifacts(copy)
    assert not result["ok"]
    assert any("dataset_id" in error for error in result["errors"])


def test_check_detects_dtype_drift(processed_dir: Path, tmp_path: Path) -> None:
    copy = _copy_processed(processed_dir, tmp_path)
    path = copy / "synthetic_battery_lifecycle_clean.parquet"
    frame = pd.read_parquet(path)
    frame["cycle_count"] = frame["cycle_count"].astype("string")
    frame.to_parquet(path, index=False)

    result = build.check_artifacts(copy)
    assert not result["ok"]
    assert any("dtype" in error.lower() for error in result["errors"])


def test_row_counts_match_raw(processed: dict, raw: dict) -> None:
    for dataset_id in DATASET_IDS:
        assert len(processed[dataset_id]) == EXPECTED_ROWS[dataset_id]
        assert len(processed[dataset_id]) == len(raw[dataset_id])


# ------------------------------------------------------- provenance and quality


def test_provenance_and_quality_columns(processed: dict) -> None:
    for dataset_id, frame in processed.items():
        for column in REQUIRED_PROVENANCE:
            assert column in frame.columns, (dataset_id, column)
        for column in QUALITY_COLUMNS:
            assert column in frame.columns, (dataset_id, column)
        assert frame["dataset_id"].nunique() == 1
        assert frame["dataset_id"].iloc[0] == dataset_id
        assert frame["source_row_id"].tolist() == list(range(1, len(frame) + 1))
        assert frame["quality_flag"].notna().all()


def test_quality_flag_tokens_are_declared(processed: dict, quality: dict) -> None:
    declared = set(quality["flag_definitions"])
    for frame in processed.values():
        tokens = {
            token
            for value in frame["quality_flag"].unique()
            for token in str(value).split(";")
        }
        assert tokens <= declared, tokens - declared


def test_no_rows_dropped_and_no_imputation(quality: dict) -> None:
    assert quality["raw_data_modified"] is False
    assert quality["imputation_applied"] is False
    assert quality["rows_dropped"] == 0
    for dataset_id, entry in quality["datasets"].items():
        assert entry["rows_dropped"] == 0, dataset_id
        assert entry["rows_after_cleaning"] == entry["rows_in"], dataset_id
        assert entry["missing_values_retained"] is True, dataset_id


def test_missing_counts_preserved(processed: dict, raw: dict) -> None:
    for dataset_id, raw_frame in raw.items():
        clean = processed[dataset_id]
        for column in raw_frame.columns:
            assert int(clean[column].isna().sum()) == int(raw_frame[column].isna().sum()), (
                dataset_id,
                column,
            )


def test_source_duplicates_are_retained(processed: dict, raw: dict) -> None:
    for dataset_id, raw_frame in raw.items():
        source_columns = list(raw_frame.columns)
        raw_dupes = int(raw_frame.duplicated(subset=source_columns).sum())
        clean_dupes = int(processed[dataset_id].duplicated(subset=source_columns).sum())
        assert clean_dupes == raw_dupes, dataset_id


def test_raw_files_are_unchanged(inventory: dict) -> None:
    for dataset_id, entry in inventory.items():
        path = REPO_ROOT / entry["source_file"]
        assert sha256_file(path) == entry["source_sha256"], dataset_id


# ------------------------------------------------------------------- integrity


def test_identifier_columns_are_populated(processed: dict) -> None:
    for dataset_id in DATASET_IDS:
        frame = processed[dataset_id]
        for column in dataset_spec(dataset_id)["id_columns"]:
            assert column in frame.columns
            assert frame[column].notna().all(), (dataset_id, column)


def test_targets_are_present_and_valid(processed: dict) -> None:
    ev = processed["EV_FAILURE_PREDICTION"]
    assert set(ev["battery_failure"].dropna().unique()) <= {0, 1}
    assert ev["battery_failure"].notna().all()

    solid = processed["SOLID_STATE_TELEMETRY_2026"]
    allowed = DATASETS["SOLID_STATE_TELEMETRY_2026"]["allowed_categories"][
        "degradation_risk_tier"
    ]
    assert set(solid["degradation_risk_tier"].dropna().unique()) <= allowed
    assert solid["degradation_risk_tier"].notna().all()

    synthetic = processed["SYNTHETIC_BATTERY_LIFECYCLE"]
    allowed = DATASETS["SYNTHETIC_BATTERY_LIFECYCLE"]["allowed_categories"]["state_of_health"]
    assert set(synthetic["state_of_health"].dropna().unique()) <= allowed
    assert synthetic["state_of_health"].notna().all()


def test_chemistry_labels_are_consistent(processed: dict, raw: dict) -> None:
    # The solid-state source records electrolyte subtypes (e.g. LLZO), all of
    # which belong to the Solid-State family; other datasets map one-to-one.
    canonical = {"NMC", "LFP", "LMO", "NCA", "LTO", "Solid-State", "Sodium-ion", "Lithium-Sulfur"}
    for dataset_id, frame in processed.items():
        chemistry_column = dataset_spec(dataset_id)["chemistry_column"]
        assert not (frame["chemistry_family"] == "<NA>").any()
        assert set(frame["chemistry_family"].dropna()) <= canonical, dataset_id
        for source_value, family in zip(
            raw[dataset_id][chemistry_column], frame["chemistry_family"]
        ):
            if pd.isna(source_value):
                assert pd.isna(family), dataset_id
            elif dataset_id == "SOLID_STATE_TELEMETRY_2026":
                assert family == "Solid-State", dataset_id
            else:
                assert family == normalize_chemistry(source_value), dataset_id


def test_category_vocabularies(processed: dict) -> None:
    for dataset_id, frame in processed.items():
        for column, allowed in dataset_spec(dataset_id)["allowed_categories"].items():
            assert set(frame[column].dropna().unique()) <= allowed, (dataset_id, column)


def test_numeric_ranges(processed: dict) -> None:
    for dataset_id, frame in processed.items():
        for column, (minimum, maximum) in dataset_spec(dataset_id)["valid_ranges"].items():
            values = frame[column].dropna().astype(float)
            assert np.isfinite(values).all(), (dataset_id, column)
            assert values.between(minimum, maximum).all(), (dataset_id, column)


# --------------------------------------------------- integer-column integrity


def _ev_frame(battery_failure: list[float], manufacturing_year: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vehicle_id": ["V1", "V2", "V3"],
            "battery_serial": ["B1", "B2", "B3"],
            "battery_chemistry": ["NMC", "LFP", "NMC"],
            "manufacturing_year": manufacturing_year,
            "cycle_count": [100, 200, 300],
            "state_of_health": [95.0, 90.0, 85.0],
            "battery_health_percent": [95.0, 90.0, 85.0],
            "pack_voltage": [400.0, 400.0, 400.0],
            "remaining_capacity": [50.0, 50.0, 50.0],
            "battery_failure": battery_failure,
        }
    )


def test_fractional_integer_values_are_preserved_and_flagged(monkeypatch) -> None:
    frame = _ev_frame(
        battery_failure=[0.0, 1.0, 0.5],
        manufacturing_year=[2020.0, 2021.0, 2020.5],
    )
    monkeypatch.setattr(cleaning, "read_raw", lambda dataset_id: frame.copy())

    clean, summary = cleaning.clean_dataset("EV_FAILURE_PREDICTION")

    assert clean.loc[2, "battery_failure"] == 0.5
    assert clean.loc[2, "manufacturing_year"] == 2020.5
    assert clean["battery_failure"].astype("float64").tolist() == [0.0, 1.0, 0.5]
    assert clean["manufacturing_year"].astype("float64").tolist() == [
        2020.0,
        2021.0,
        2020.5,
    ]
    assert pd.api.types.is_float_dtype(clean["battery_failure"])
    assert pd.api.types.is_float_dtype(clean["manufacturing_year"])

    flagged = set(str(clean.loc[2, "quality_flag"]).split(";"))
    assert "NON_INTEGER_VALUE" in flagged
    invalid_fields = set(str(clean.loc[2, "invalid_fields"]).split(";"))
    assert {"battery_failure", "manufacturing_year"} <= invalid_fields

    clean_flags = set(str(clean.loc[0, "quality_flag"]).split(";"))
    assert "NON_INTEGER_VALUE" not in clean_flags

    assert summary["type_coercions"]["battery_failure:fractional"] == 1
    assert summary["type_coercions"]["manufacturing_year:fractional"] == 1


def test_integral_declared_integer_columns_stay_integer(monkeypatch) -> None:
    frame = _ev_frame(
        battery_failure=[0.0, 1.0, 0.0],
        manufacturing_year=[2020.0, 2021.0, 2022.0],
    )
    monkeypatch.setattr(cleaning, "read_raw", lambda dataset_id: frame.copy())

    clean, summary = cleaning.clean_dataset("EV_FAILURE_PREDICTION")

    assert str(clean["battery_failure"].dtype) == "Int8"
    assert str(clean["manufacturing_year"].dtype) == "Int16"
    assert not clean["quality_flag"].str.contains("NON_INTEGER_VALUE").any()
    assert "battery_failure:fractional" not in summary["type_coercions"]
    assert "manufacturing_year:fractional" not in summary["type_coercions"]


# ----------------------------------------------------------------- grain rules


def test_snapshot_datasets_have_no_sequences(registry: dict, inventory: dict) -> None:
    for dataset_id in DATASET_IDS:
        assert registry["active_datasets"][dataset_id]["sequence_available"] is False
        assert inventory[dataset_id]["sequence_analysis"]["chronological_history_possible"] is False
        assert inventory[dataset_id]["sequence_analysis"]["reason"]
        with pytest.raises(loader.NoSequenceDataError):
            loader.build_sequences(dataset_id)


def test_solid_state_has_no_invented_time_series(processed: dict) -> None:
    frame = processed["SOLID_STATE_TELEMETRY_2026"]
    for column in ["time_sec", "voltage_v", "current_a", "capacity_ah"]:
        assert column not in frame.columns


def test_synthetic_data_is_separated(processed: dict) -> None:
    frame = processed["SYNTHETIC_BATTERY_LIFECYCLE"]
    assert set(frame["data_origin"].unique()) == {"synthetic"}
    for column in ["soh_pct", "rul_cycles", "remaining_life_cycles"]:
        assert column not in frame.columns
    assert not any("rul" in column.lower() for column in frame.columns)


def test_active_datasets_have_no_fabricated_master_columns(dictionary: pd.DataFrame) -> None:
    solid_columns = set(dictionary.loc[
        dictionary["dataset_id"] == "SOLID_STATE_TELEMETRY_2026", "column_name"
    ])
    assert not solid_columns & {"time_sec", "voltage_v", "current_a", "capacity_ah"}


# --------------------------------------------------------------- leakage rules


def test_leakage_exclusions_are_enforced(leakage: dict, processed: dict) -> None:
    for dataset_id, entry in leakage["datasets"].items():
        features = set(entry["features_allowed"])
        assert features, dataset_id
        assert not features & set(entry["targets"]), dataset_id
        assert not features & set(entry["candidate_targets"]), dataset_id
        assert not features & set(entry["identifiers"]), dataset_id
        excluded = {item["column"] for item in entry["excluded_fields"]}
        assert not features & excluded, dataset_id
        assert not features & set(PROVENANCE_COLUMNS), dataset_id
        assert not features & set(QUALITY_COLUMNS), dataset_id
        for column in excluded:
            assert column in processed[dataset_id].columns, (dataset_id, column)


def test_known_leakage_fields_are_not_features(leakage: dict) -> None:
    ev = set(leakage["datasets"]["EV_FAILURE_PREDICTION"]["features_allowed"])
    assert "predicted_remaining_life_cycles" not in ev
    assert "capacity_loss_percent" not in ev
    assert "state_of_health" not in ev
    assert "battery_health_percent" not in ev

    solid = set(leakage["datasets"]["SOLID_STATE_TELEMETRY_2026"]["features_allowed"])
    for column in [
        "state_of_health_pct",
        "degradation_risk_tier",
        "projected_cycle_life_80pct_soh",
        "cot_electrochemical_risk_rationale",
        "confidence_score",
        "thermal_stability_margin_c",
    ]:
        assert column not in solid

    synthetic = set(leakage["datasets"]["SYNTHETIC_BATTERY_LIFECYCLE"]["features_allowed"])
    assert "state_of_health" not in synthetic


# --------------------------------------------------------------------- loaders


def test_loader_feature_and_target_contract(processed_dir: Path, leakage: dict) -> None:
    for dataset_id in DATASET_IDS:
        features = loader.feature_columns(dataset_id, processed_dir=processed_dir)
        assert features == leakage["datasets"][dataset_id]["features_allowed"]
        review = loader.feature_columns(
            dataset_id, include_review=True, processed_dir=processed_dir
        )
        assert set(features) <= set(review)
        excluded = loader.excluded_columns(dataset_id, processed_dir=processed_dir)
        assert not set(features) & set(excluded)
        targets = loader.target_columns(dataset_id, processed_dir=processed_dir)
        assert targets
        assert not set(features) & set(targets)
        assert loader.identifier_columns(dataset_id)


def test_registry_paths_resolve(registry: dict, processed_dir: Path) -> None:
    for dataset_id, entry in registry["active_datasets"].items():
        assert entry["status"] == "active"
        assert (processed_dir / entry["processed_file"]).exists()
        assert entry["rows"] == EXPECTED_ROWS[dataset_id]
    assert registry["active_master_table"] is None


def test_no_root_master_table_is_active(repo_root: Path) -> None:
    assert not (repo_root / "master_table.parquet").exists()
    source = (repo_root / "build_master_table.py").read_text(encoding="utf-8")
    assert "DEPRECATED" in source
    assert "to_parquet" not in source


# ---------------------------------------------------------------------- legacy


def test_legacy_master_table_is_blocked_by_default() -> None:
    with pytest.raises(loader.LegacyDatasetError):
        loader.load_legacy_master_table()


def test_legacy_master_table_loads_only_with_opt_in() -> None:
    frame = loader.load_legacy_master_table(allow_legacy=True)
    assert frame.shape == (23_768, 8)
    assert "legacy_warning" in frame.attrs
    assert (LEGACY_DIR / "master_table_4789_LFP_B1_regression.csv").exists()


def test_legacy_artifacts_are_registered(registry: dict) -> None:
    entries = {entry["id"]: entry for entry in registry["legacy_datasets"]}
    assert entries["LEGACY_MASTER_TABLE_8COL"]["sha256"] == (
        "e5f699e19626559d50495e4d9e564086682f9b8eb2e47a4cc42da9162f1bd18c"
    )
    for entry in entries.values():
        if entry.get("path", "").startswith("legacy/"):
            assert (REPO_ROOT / entry["path"]).exists(), entry["id"]


# ------------------------------------------------------------- reproducibility


def test_reproducible_preprocessing(
    rebuilt_outputs: Path, tmp_path: Path
) -> None:
    second = tmp_path / "second_build"
    second.mkdir()
    build.build_all(second)

    names = [
        *EXPECTED_FILES.values(),
        "dataset_inventory.json",
        "data_quality_report.json",
        "data_dictionary.csv",
        "leakage_audit.json",
        "dataset_registry.json",
    ]
    for name in names:
        assert sha256_file(rebuilt_outputs / name) == sha256_file(second / name), name


# ------------------------------------------------------------------- reporting


def test_reports_reconcile_with_outputs(
    registry: dict, inventory: dict, quality: dict, dictionary: pd.DataFrame, processed: dict
) -> None:
    for dataset_id, frame in processed.items():
        assert inventory[dataset_id]["processed_rows"] == len(frame)
        assert inventory[dataset_id]["processed_columns"] == frame.shape[1]
        assert inventory[dataset_id]["rows"] == len(frame)
        assert quality["datasets"][dataset_id]["rows_after_cleaning"] == len(frame)
        assert registry["active_datasets"][dataset_id]["rows"] == len(frame)
        subset = dictionary[dictionary["dataset_id"] == dataset_id]
        assert set(subset["column_name"]) == set(frame.columns)
        dtype_map = dict(zip(subset["column_name"], subset["dtype"]))
        for column in frame.columns:
            assert dtype_map[column] == str(frame[column].dtype), (dataset_id, column)
