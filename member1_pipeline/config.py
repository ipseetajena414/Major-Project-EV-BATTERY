"""Configuration for the Member 1 data pipeline.

All paths are resolved from the repository root. Configuration here is
declarative: dataset identity, provenance, target roles, units, valid ranges,
and allowed category vocabularies. Cleaning and auditing code read this file.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "DATA" / "processed"
LEGACY_DIR = REPO_ROOT / "legacy" / "member1"

PIPELINE_VERSION = "1.0.0"

DATASETS: dict[str, dict] = {
    "EV_FAILURE_PREDICTION": {
        "source_file": "ev battery_failure prediction Dataset.csv",
        "processed_filename": "ev_battery_failure_clean.parquet",
        "data_origin": "source_dataset",
        "record_grain": "vehicle_pack_snapshot",
        "entity_columns": ["vehicle_id", "battery_serial"],
        "cycle_column": "cycle_count",
        "cycle_is_sequence_index": False,
        "sequence_available": False,
        "sequence_reason": (
            "Exactly one row per vehicle_id and one row per battery_serial; "
            "cycle_count is a single scalar per vehicle, not a repeated cycle history."
        ),
        "chemistry_column": "battery_chemistry",
        "chemistry_family_column": "chemistry_family",
        "id_columns": ["vehicle_id", "battery_serial"],
        "critical_columns": ["vehicle_id", "battery_serial", "battery_failure"],
        "targets": {
            "battery_failure": {
                "task": "classification",
                "description": "Binary battery-failure label provided by the source dataset.",
            }
        },
        "candidate_targets": {
            "state_of_health": "SOH percentage; audited independently of battery_health_percent.",
            "battery_health_percent": (
                "SOH percentage; audited independently of state_of_health. "
                "capacity_loss_percent is exactly 100 minus this field."
            ),
        },
        "outlier_columns": [
            "battery_capacity_kwh",
            "odometer_km",
            "cycle_count",
            "cell_voltage_avg",
            "pack_voltage",
            "cell_temperature_avg",
            "cell_temperature_max",
            "internal_resistance",
            "charge_efficiency",
            "discharge_efficiency",
            "remaining_capacity",
            "average_charge_power_kw",
            "average_charging_time",
            "average_speed",
            "daily_distance",
        ],
        "valid_ranges": {
            "manufacturing_year": (1990, 2030),
            "battery_capacity_kwh": (0, 1000),
            "odometer_km": (0, 2_000_000),
            "vehicle_age_years": (0, 50),
            "cycle_count": (0, 20_000),
            "battery_health_percent": (0, 100),
            "state_of_charge": (0, 100),
            "depth_of_discharge": (0, 100),
            "state_of_health": (0, 100),
            "cell_voltage_avg": (0, 6),
            "pack_voltage": (0, 1000),
            "cell_temperature_avg": (-60, 100),
            "cell_temperature_max": (-60, 120),
            "internal_resistance": (0, 10),
            "charge_efficiency": (0, 100),
            "discharge_efficiency": (0, 100),
            "remaining_capacity": (0, 1000),
            "capacity_loss_percent": (0, 100),
            "average_ambient_temperature": (-60, 60),
            "maximum_temperature": (-60, 80),
            "minimum_temperature": (-60, 60),
            "battery_failure": (0, 1),
        },
        "allowed_categories": {
            "battery_chemistry": {"NMC", "LFP", "LMO", "NCA", "LTO"},
            "vehicle_type": {"Hatchback", "Crossover", "Van", "Truck", "SUV", "Sedan"},
            "drive_type": {"AWD", "RWD", "FWD"},
            "fleet_or_private": {"Private", "Fleet"},
            "terrain_type": {"Coastal", "Mountainous", "Flat", "Hilly", "Desert"},
        },
    },
    "SOLID_STATE_TELEMETRY_2026": {
        "source_file": "solid_state_ev_battery_telemetry_enriched_2026.csv",
        "processed_filename": "solid_state_telemetry_clean.parquet",
        "data_origin": "source_dataset",
        "record_grain": "cell_snapshot",
        "entity_columns": ["cell_id"],
        "cycle_column": "cycle_number",
        "cycle_is_sequence_index": False,
        "sequence_available": False,
        "sequence_reason": (
            "Exactly one row per cell_id; cycle_number is a single snapshot value per cell, "
            "so no chronological history can be reconstructed."
        ),
        "chemistry_column": "electrolyte_chemistry",
        "chemistry_family_column": "chemistry_family",
        "id_columns": ["cell_id"],
        "critical_columns": ["cell_id", "state_of_health_pct", "cycle_number"],
        "targets": {
            "degradation_risk_tier": {
                "task": "classification",
                "description": (
                    "Provided risk tier. It is derived from SOH and other signals, "
                    "so it is a target for risk classification but never an input feature."
                ),
            }
        },
        "candidate_targets": {
            "state_of_health_pct": "Candidate SOH regression target.",
        },
        "outlier_columns": [
            "nominal_capacity_ah",
            "cycle_number",
            "internal_resistance_mohm",
            "state_of_health_pct",
            "coulombic_efficiency_pct",
            "operating_temperature_c",
            "ambient_temperature_c",
            "stack_pressure_mpa",
            "thermal_runaway_onset_c",
            "projected_cycle_life_80pct_soh",
        ],
        "valid_ranges": {
            "nominal_capacity_ah": (0, 1000),
            "gravimetric_energy_density_wh_kg": (0, 2000),
            "volumetric_energy_density_wh_l": (0, 3000),
            "cycle_number": (0, 100_000),
            "charging_c_rate": (0, 20),
            "discharging_c_rate": (0, 20),
            "operating_temperature_c": (-60, 150),
            "ambient_temperature_c": (-60, 60),
            "internal_resistance_mohm": (0, 1000),
            "state_of_health_pct": (0, 100),
            "coulombic_efficiency_pct": (0, 100),
            "stack_pressure_mpa": (0, 100),
            "dendrite_penetration_risk": (0, 1),
            "thermal_runaway_onset_c": (0, 1000),
            "nominal_voltage_plateau_v": (0, 6),
            "thermal_stability_margin_c": (-100, 1000),
            "fast_charge_acceptance_score": (0, 100),
            "dendrite_mitigation_efficiency": (0, 100),
            "projected_cycle_life_80pct_soh": (0, 100_000),
            "confidence_score": (0, 1),
        },
        "allowed_categories": {
            "electrolyte_chemistry": {
                "Sulfide_Based_Li10GeP2S12",
                "Oxide_Based_LLZO",
                "Polymer_Composite",
                "Halide_Based_Li3YCl6",
            },
            "anode_composition": {
                "Silicon_Dominant_Nanowire",
                "Lithium_Metal_Foil",
                "Anode_Free_InSitu_Plating",
                "Graphite_Silicon_Composite",
            },
            "cathode_chemistry": {
                "High_Nickel_NMC811",
                "Ultra_Nickel_NMC955",
                "LFP_Advanced",
                "Solid_State_Sulfur",
            },
            "degradation_risk_tier": {
                "Minimal_Degradation_Tier_1",
                "Moderate_Degradation_Tier_2",
                "Accelerated_Degradation_Tier_3",
                "Critical_Thermal_Risk_Tier_4",
            },
            "commercial_readiness_level": {
                "TRL_6_Laboratory_Prototype",
                "TRL_7_Automotive_Testing",
                "TRL_8_Pilot_Production",
                "TRL_9_Commercial_Fleet",
            },
            "target_ev_application": {
                "Long_Range_Luxury_Sedan",
                "Urban_Compact_SUV",
                "Commercial_Class8_Hauler",
                "Hypercar_Track",
                "eVTOL_Aerospace",
            },
        },
    },
    "SYNTHETIC_BATTERY_LIFECYCLE": {
        "source_file": "synthetic_battery_lifecycle.csv",
        "processed_filename": "synthetic_battery_lifecycle_clean.parquet",
        "data_origin": "synthetic",
        "record_grain": "synthetic_battery_snapshot",
        "entity_columns": ["battery_id"],
        "cycle_column": "cycle_count",
        "cycle_is_sequence_index": False,
        "sequence_available": False,
        "sequence_reason": (
            "1,978 battery IDs across 2,000 rows; at most two snapshots per battery, "
            "and some repeated IDs carry conflicting electrolyte labels, so no reliable "
            "chronological sequence exists."
        ),
        "chemistry_column": "electrolyte_type",
        "chemistry_family_column": "chemistry_family",
        "id_columns": ["battery_id"],
        "critical_columns": ["battery_id", "state_of_health", "cycle_count"],
        "targets": {
            "state_of_health": {
                "task": "classification",
                "description": (
                    "Categorical SOH label (Healthy, Degraded, End_of_Life). "
                    "No numeric SOH or measured RUL is provided or invented."
                )
            }
        },
        "candidate_targets": {},
        "outlier_columns": [
            "cycle_count",
            "internal_resistance_mohms",
            "temperature_celsius",
            "discharge_capacity_ah",
            "charge_time_mins",
        ],
        "valid_ranges": {
            "cycle_count": (0, 100_000),
            "internal_resistance_mohms": (0, 1000),
            "temperature_celsius": (-60, 150),
            "discharge_capacity_ah": (0, 1000),
            "charge_time_mins": (0, 10_000),
        },
        "allowed_categories": {
            "electrolyte_type": {"Solid_State", "Lithium_Sulfur", "Sodium_Ion"},
            "state_of_health": {"Healthy", "Degraded", "End_of_Life"},
        },
    },
}

# Roles: target, target_candidate, identifier, feature, metadata,
# feature_candidate_derived, feature_candidate_unit_ambiguous, and the
# excluded_* roles. Anything not listed defaults by column type.
ROLE_OVERRIDES: dict[str, dict[str, str]] = {
    "EV_FAILURE_PREDICTION": {
        "vehicle_id": "identifier",
        "battery_serial": "identifier",
        "battery_failure": "target",
        "state_of_health": "target_candidate",
        "battery_health_percent": "target_candidate",
        "predicted_remaining_life_cycles": "excluded_predicted_rul",
        "capacity_loss_percent": "excluded_derived",
        "aging_score": "feature_candidate_derived",
        "battery_stress_index": "feature_candidate_derived",
        "thermal_health_score": "feature_candidate_derived",
        "charging_quality_score": "feature_candidate_derived",
        "driving_stress_score": "feature_candidate_derived",
        "remaining_capacity": "feature_candidate_unit_ambiguous",
        "internal_resistance": "feature_candidate_unit_ambiguous",
    },
    "SOLID_STATE_TELEMETRY_2026": {
        "cell_id": "identifier",
        "state_of_health_pct": "target_candidate",
        "degradation_risk_tier": "target_excluded_as_feature",
        "cot_electrochemical_risk_rationale": "excluded_leakage_text",
        "projected_cycle_life_80pct_soh": "excluded_projected_rul",
        "confidence_score": "excluded_enriched_confidence",
        "thermal_stability_margin_c": "excluded_derived",
        "dendrite_penetration_risk": "excluded_enriched_risk_score",
        "fast_charge_acceptance_score": "excluded_enriched_risk_score",
        "dendrite_mitigation_efficiency": "excluded_enriched_risk_score",
    },
    "SYNTHETIC_BATTERY_LIFECYCLE": {
        "battery_id": "identifier",
        "state_of_health": "target",
    },
}

UNITS: dict[str, dict[str, str]] = {
    "EV_FAILURE_PREDICTION": {
        "manufacturing_year": "year",
        "battery_capacity_kwh": "kWh",
        "odometer_km": "km",
        "vehicle_age_years": "years",
        "cycle_count": "cycles",
        "battery_health_percent": "percent",
        "state_of_charge": "percent",
        "depth_of_discharge": "percent",
        "state_of_health": "percent",
        "cell_voltage_avg": "V",
        "cell_voltage_std": "V",
        "pack_voltage": "V",
        "cell_temperature_avg": "degC",
        "cell_temperature_max": "degC",
        "internal_resistance": "ohm (unit not declared by source; range 0.03-1.12)",
        "charge_efficiency": "percent",
        "discharge_efficiency": "percent",
        "remaining_capacity": "Ah (grain ambiguous: not equal to pack kWh / pack voltage)",
        "capacity_loss_percent": "percent",
        "charging_cycles_last_month": "cycles/month",
        "average_charge_power_kw": "kW",
        "average_charging_time": "minutes",
        "average_speed": "km/h",
        "average_trip_distance": "km",
        "daily_distance": "km",
        "average_ambient_temperature": "degC",
        "maximum_temperature": "degC",
        "minimum_temperature": "degC",
        "humidity": "percent",
        "altitude": "m",
        "last_service_days": "days",
        "thermal_runaway_risk": "score 0-100",
        "voltage_imbalance": "score (unit not declared)",
        "temperature_variance": "score (unit not declared)",
        "battery_stress_index": "score 0-100",
        "aging_score": "score 0-100",
        "thermal_health_score": "score 0-100",
        "charging_quality_score": "score 0-100",
        "driving_stress_score": "score 0-100",
        "predicted_remaining_life_cycles": "cycles (predicted, not measured)",
        "battery_failure": "0/1 label",
    },
    "SOLID_STATE_TELEMETRY_2026": {
        "nominal_capacity_ah": "Ah",
        "gravimetric_energy_density_wh_kg": "Wh/kg",
        "volumetric_energy_density_wh_l": "Wh/L",
        "cycle_number": "cycles (snapshot value)",
        "charging_c_rate": "C",
        "discharging_c_rate": "C",
        "operating_temperature_c": "degC",
        "ambient_temperature_c": "degC",
        "internal_resistance_mohm": "mOhm",
        "state_of_health_pct": "percent",
        "coulombic_efficiency_pct": "percent",
        "stack_pressure_mpa": "MPa",
        "dendrite_penetration_risk": "risk index 0-1",
        "thermal_runaway_onset_c": "degC",
        "nominal_voltage_plateau_v": "V",
        "thermal_stability_margin_c": "degC (onset minus operating temperature)",
        "fast_charge_acceptance_score": "score 0-100",
        "dendrite_mitigation_efficiency": "score 0-100",
        "projected_cycle_life_80pct_soh": "cycles (projected, not observed RUL)",
        "confidence_score": "confidence 0-1",
    },
    "SYNTHETIC_BATTERY_LIFECYCLE": {
        "cycle_count": "cycles",
        "internal_resistance_mohms": "mOhm",
        "temperature_celsius": "degC",
        "discharge_capacity_ah": "Ah",
        "charge_time_mins": "minutes",
        "state_of_health": "category (Healthy/Degraded/End_of_Life)",
    },
}

DESCRIPTIONS: dict[str, dict[str, str]] = {
    "EV_FAILURE_PREDICTION": {
        "vehicle_id": "Synthetic vehicle identifier; one row per vehicle.",
        "battery_serial": "Battery serial identifier; one row per battery.",
        "battery_chemistry": "Source chemistry label (NMC, LFP, LMO, NCA, LTO).",
        "manufacturing_year": "Vehicle manufacturing year.",
        "battery_capacity_kwh": "Rated pack energy capacity.",
        "odometer_km": "Vehicle odometer reading.",
        "cycle_count": "Scalar lifetime cycle count snapshot for the vehicle.",
        "battery_health_percent": "SOH percentage; independent source field.",
        "state_of_health": "SOH percentage; independent source field distinct from battery_health_percent.",
        "state_of_charge": "Snapshot state of charge.",
        "depth_of_discharge": "Snapshot depth of discharge.",
        "cell_voltage_avg": "Average cell voltage snapshot.",
        "cell_voltage_std": "Standard deviation of cell voltage snapshot.",
        "pack_voltage": "Pack voltage snapshot.",
        "cell_temperature_avg": "Average cell temperature snapshot.",
        "cell_temperature_max": "Maximum cell temperature snapshot.",
        "internal_resistance": "Internal resistance value; source does not declare units.",
        "charge_efficiency": "Charge efficiency percentage.",
        "discharge_efficiency": "Discharge efficiency percentage.",
        "remaining_capacity": "Remaining capacity value; source grain is ambiguous.",
        "capacity_loss_percent": "Derived: exactly 100 minus battery_health_percent when both are present.",
        "average_charge_power_kw": "Average charging power.",
        "average_charging_time": "Average charging time.",
        "predicted_remaining_life_cycles": "Model-provided predicted remaining life; not measured RUL.",
        "battery_failure": "Binary failure classification target.",
        "aging_score": "Composite aging score; derivation not documented in the source.",
        "battery_stress_index": "Composite battery stress score; derivation not documented.",
        "thermal_health_score": "Composite thermal health score; derivation not documented.",
        "charging_quality_score": "Composite charging quality score; derivation not documented.",
        "driving_stress_score": "Composite driving stress score; derivation not documented.",
    },
    "SOLID_STATE_TELEMETRY_2026": {
        "cell_id": "Cell identifier; one row per cell.",
        "manufacturer": "Cell manufacturer.",
        "electrolyte_chemistry": "Solid electrolyte family.",
        "anode_composition": "Anode material family.",
        "cathode_chemistry": "Cathode material family.",
        "nominal_capacity_ah": "Nominal cell capacity; repeated as a snapshot value.",
        "cycle_number": "Cycle count snapshot for the cell at the observation time.",
        "internal_resistance_mohm": "Internal resistance.",
        "state_of_health_pct": "SOH percentage; candidate regression target.",
        "coulombic_efficiency_pct": "Coulombic efficiency.",
        "stack_pressure_mpa": "Applied stack pressure.",
        "dendrite_penetration_risk": "Enriched dendrite risk index; derivation not documented.",
        "thermal_runaway_onset_c": "Thermal runaway onset temperature.",
        "nominal_voltage_plateau_v": "Nominal voltage plateau.",
        "thermal_stability_margin_c": "Derived: thermal runaway onset minus operating temperature.",
        "fast_charge_acceptance_score": "Enriched fast-charge score; derivation not documented.",
        "dendrite_mitigation_efficiency": "Enriched dendrite mitigation score; derivation not documented.",
        "projected_cycle_life_80pct_soh": "Projected cycle life to 80% SOH; not observed RUL.",
        "degradation_risk_tier": "Provided risk tier; derived from SOH and other signals.",
        "cot_electrochemical_risk_rationale": "Generated narrative that restates features, SOH and risk tier.",
        "confidence_score": "Enriched confidence score; derivation and calibration not documented.",
    },
    "SYNTHETIC_BATTERY_LIFECYCLE": {
        "battery_id": "Synthetic battery identifier.",
        "electrolyte_type": "Synthetic electrolyte family label.",
        "cycle_count": "Synthetic cycle count snapshot.",
        "internal_resistance_mohms": "Synthetic internal resistance.",
        "temperature_celsius": "Synthetic operating temperature.",
        "discharge_capacity_ah": "Synthetic discharge capacity.",
        "charge_time_mins": "Synthetic charge time.",
        "state_of_health": "Categorical SOH label; no numeric SOH is provided.",
    },
}

CHEMISTRY_CANONICAL: dict[str, str] = {
    "NMC": "NMC",
    "LFP": "LFP",
    "LMO": "LMO",
    "NCA": "NCA",
    "LTO": "LTO",
    "SOLID-STATE": "Solid-State",
    "SODIUM-ION": "Sodium-ion",
    "NA-ION": "Sodium-ion",
    "NAION": "Sodium-ion",
    "LITHIUM-SULFUR": "Lithium-Sulfur",
    "LI-S": "Lithium-Sulfur",
}


def normalize_chemistry(value: object) -> str | None:
    """Map inconsistent source chemistry labels to one canonical vocabulary."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    key = re.sub(r"[\s_]+", "-", text).upper()
    return CHEMISTRY_CANONICAL.get(key, text)


def dataset_ids() -> list[str]:
    return list(DATASETS)


def dataset_spec(dataset_id: str) -> dict:
    try:
        return DATASETS[dataset_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset_id {dataset_id!r}; known IDs: {', '.join(DATASETS)}"
        ) from exc


def role_for(dataset_id: str, column: str, dtype_is_numeric: bool) -> str:
    overrides = ROLE_OVERRIDES.get(dataset_id, {})
    if column in overrides:
        return overrides[column]
    spec = dataset_spec(dataset_id)
    if column in spec["id_columns"]:
        return "identifier"
    if column in spec["targets"]:
        return "target"
    if column in spec["candidate_targets"]:
        return "target_candidate"
    return "feature"


def unit_for(dataset_id: str, column: str) -> str:
    return UNITS.get(dataset_id, {}).get(column, "")


def description_for(dataset_id: str, column: str) -> str:
    explicit = DESCRIPTIONS.get(dataset_id, {}).get(column)
    if explicit:
        return explicit
    return "Source field: " + column.replace("_", " ")
