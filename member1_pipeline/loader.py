"""Downstream loader interface for the active Member 1 datasets.

Members 2-5 should read data through this module instead of reading
master_table.parquet directly. The legacy master table is archived and is
blocked unless a caller explicitly opts in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import LEGACY_DIR, PROCESSED_DIR, dataset_ids, dataset_spec


class LegacyDatasetError(RuntimeError):
    """Raised when code tries to load a retired master table by default."""


class NoSequenceDataError(RuntimeError):
    """Raised when sequence construction is requested for snapshot-only data."""


REQUIRED_PROVENANCE = ["source_row_id", "dataset_id", "data_origin", "source_file", "quality_flag"]


def _processed_dir(processed_dir: Path | str | None) -> Path:
    return Path(processed_dir) if processed_dir is not None else PROCESSED_DIR


def load_registry(processed_dir: Path | str | None = None) -> dict:
    path = _processed_dir(processed_dir) / "dataset_registry.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_inventory(processed_dir: Path | str | None = None) -> dict:
    path = _processed_dir(processed_dir) / "dataset_inventory.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_leakage_audit(processed_dir: Path | str | None = None) -> dict:
    path = _processed_dir(processed_dir) / "leakage_audit.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_data_dictionary(processed_dir: Path | str | None = None) -> pd.DataFrame:
    return pd.read_csv(_processed_dir(processed_dir) / "data_dictionary.csv")


def load_dataset(
    dataset_id: str,
    columns: list[str] | None = None,
    processed_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load one active processed dataset, optionally selecting columns."""
    if dataset_id not in dataset_ids():
        raise KeyError(
            f"Unknown dataset_id {dataset_id!r}; use one of {', '.join(dataset_ids())}"
        )
    spec = dataset_spec(dataset_id)
    path = _processed_dir(processed_dir) / spec["processed_filename"]
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run python -m member1_pipeline.build"
        )
    frame = pd.read_parquet(path)
    missing = [column for column in REQUIRED_PROVENANCE if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing provenance columns: {missing}")
    if columns is not None:
        unknown = [column for column in columns if column not in frame.columns]
        if unknown:
            raise KeyError(f"Unknown columns for {dataset_id}: {unknown}")
        frame = frame.loc[:, columns]
    frame.attrs["dataset_id"] = dataset_id
    frame.attrs["data_origin"] = spec["data_origin"]
    frame.attrs["record_grain"] = spec["record_grain"]
    return frame


def load_ev_failure(
    columns: list[str] | None = None, processed_dir: Path | str | None = None
) -> pd.DataFrame:
    return load_dataset("EV_FAILURE_PREDICTION", columns, processed_dir)


def load_solid_state(
    columns: list[str] | None = None, processed_dir: Path | str | None = None
) -> pd.DataFrame:
    return load_dataset("SOLID_STATE_TELEMETRY_2026", columns, processed_dir)


def load_synthetic(
    columns: list[str] | None = None, processed_dir: Path | str | None = None
) -> pd.DataFrame:
    return load_dataset("SYNTHETIC_BATTERY_LIFECYCLE", columns, processed_dir)


def _roles(dataset_id: str, processed_dir: Path | str | None) -> dict[str, str]:
    dictionary = load_data_dictionary(processed_dir)
    subset = dictionary[dictionary["dataset_id"] == dataset_id]
    return dict(zip(subset["column_name"], subset["role"]))


def feature_columns(
    dataset_id: str,
    include_review: bool = False,
    processed_dir: Path | str | None = None,
) -> list[str]:
    """Return leakage-safe feature columns.

    include_review=True adds derived/unit-ambiguous candidates that require a
    documented modelling decision; it never adds excluded fields.
    """
    roles = _roles(dataset_id, processed_dir)
    allowed = {"feature"}
    if include_review:
        allowed |= {"feature_candidate_derived", "feature_candidate_unit_ambiguous"}
    return [column for column, role in roles.items() if role in allowed]


def target_columns(
    dataset_id: str, include_candidates: bool = True, processed_dir: Path | str | None = None
) -> list[str]:
    roles = _roles(dataset_id, processed_dir)
    allowed = {"target", "target_excluded_as_feature"}
    if include_candidates:
        allowed.add("target_candidate")
    return [column for column, role in roles.items() if role in allowed]


def excluded_columns(
    dataset_id: str, processed_dir: Path | str | None = None
) -> list[str]:
    roles = _roles(dataset_id, processed_dir)
    return [
        column
        for column, role in roles.items()
        if role.startswith("excluded") or role == "target_excluded_as_feature"
    ]


def identifier_columns(
    dataset_id: str, processed_dir: Path | str | None = None
) -> list[str]:
    return list(dataset_spec(dataset_id)["id_columns"])


def sequence_available(dataset_id: str, processed_dir: Path | str | None = None) -> bool:
    registry = load_registry(processed_dir)
    return bool(registry["active_datasets"][dataset_id]["sequence_available"])


def build_sequences(dataset_id: str, processed_dir: Path | str | None = None) -> None:
    if not sequence_available(dataset_id, processed_dir):
        raise NoSequenceDataError(
            f"{dataset_id} has no chronological histories; it contains snapshot rows only. "
            "Do not train a BiLSTM sequence model on it."
        )
    raise NoSequenceDataError(f"No sequence builder is defined for {dataset_id}.")


def dataset_summary(processed_dir: Path | str | None = None) -> pd.DataFrame:
    registry = load_registry(processed_dir)
    rows = []
    for dataset_id, entry in registry["active_datasets"].items():
        rows.append(
            {
                "dataset_id": dataset_id,
                "rows": entry["rows"],
                "columns": entry["columns"],
                "data_origin": entry["data_origin"],
                "record_grain": entry["record_grain"],
                "sequence_available": entry["sequence_available"],
                "targets": ", ".join(entry["targets"]),
            }
        )
    return pd.DataFrame(rows)


def load_legacy_master_table(
    path: Path | str | None = None,
    allow_legacy: bool = False,
) -> pd.DataFrame:
    """Load the archived legacy master table only after an explicit opt-in."""
    if not allow_legacy:
        raise LegacyDatasetError(
            "The legacy master_table.parquet is archived and is not an active input. "
            "Use load_dataset() for the three active datasets. If you truly need the "
            "archived table for an audit, pass allow_legacy=True and state why."
        )
    target = Path(path) if path is not None else LEGACY_DIR / "master_table_8col_fabricated.parquet"
    if not target.exists():
        raise FileNotFoundError(f"Legacy master table not found: {target}")
    frame = pd.read_parquet(target) if target.suffix == ".parquet" else pd.read_csv(target)
    frame.attrs["legacy_warning"] = (
        "LEGACY DATA: archived for audit only. It has no provenance columns; "
        "Na-ion voltage and current were constants and Na-ion SOH was derived from resistance."
    )
    return frame


def legacy_artifacts(processed_dir: Path | str | None = None) -> list[dict]:
    return load_registry(processed_dir).get("legacy_datasets", [])
