"""Member 2 battery-health integration contract — v1.1.0-draft.

This module is an *isolated adapter*. It does not modify Member2Predictor,
the model artifacts, the schema, the manifest or any canonical dataset.

It converts the existing ``PredictionResult`` objects into a stable,
explicitly-versioned result contract that a downstream consumer (Member 3 /
Member 4) can validate before using a prediction. It also exposes the
handoff helper that prepares the provisional RangeCalculator arguments.

Governance rules encoded here:

* A successful inference is reported as ``ok_experimental``, never ``ok``,
  because no Member 2 predictor is production-validated.
* ``rul.rul_cycles`` is always null. There is no approved RUL model; the
  shipped RUL artifact is unvalidated for pre-EOL forecasting.
* A prediction is rejected if the supplied feature keys do not exactly match
  the declared chemistry's schema. The underlying predictor silently accepts
  extra keys, so an NMC feature dict can otherwise be scored by the LFP model.
* Predicted SOH is rejected unless it is finite and within 0–100 %.
* No uncertainty estimate is invented.
* ``soh.current_observed_soh_pct`` carries the measured SOH anchor so a
  consumer can use the persistence baseline instead of an extrapolated value.
"""
from __future__ import annotations

import math
import warnings
from numbers import Integral, Real
from typing import Any, Mapping

from member2_modeling.predictor import Member2Predictor

CONTRACT_VERSION = "1.1.0-draft"

SUPPORTED_SOH_CHEMISTRIES = ("NMC", "LFP")
MODEL_KEYS = {"NMC": "nmc_lgbm", "LFP": "lfp_lgbm"}

# There is no approved RUL model. This flag is deliberately hard-coded false:
# exposing a numeric RUL requires a separate governance decision backed by
# pre-threshold validation evidence, not a code toggle.
RUL_APPROVED_FOR_USE = False
RUL_MODEL_ID = "nmc_rul_lgbm"

SOH_HORIZON_SEMANTICS = (
    "SOH of the next observed cycle record (positional shift by 1). "
    "This is not necessarily the next physical cycle; LFP B5 contains "
    "observed-ID transitions 100->151 and 200->401."
)

BASE_FLAGS = (
    "MODEL_EXPERIMENTAL_RESEARCH_STAGE",
    "MODEL_UNDERPERFORMS_PERSISTENCE",
    "SOH_HORIZON_OBSERVED_RECORD_NOT_PHYSICAL_CYCLE",
    "CURRENT_SOH_IS_UPSTREAM_LABEL",
    "NO_UNCERTAINTY_QUANTIFICATION",
    "NO_OUT_OF_DISTRIBUTION_CHECK",
    "CHEMISTRY_DECLARED_BY_CALLER",
    "RUL_NOT_APPROVED",
)

CHECKS_NOT_PERFORMED = (
    "out_of_distribution",
    "sensor_calibration",
    "label_freshness",
    "uncertainty_calibration",
    "vehicle_level_range_validation",
)


def _error(code: str, message: str, field: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "message": message}
    if field is not None:
        item["field"] = field
    return item


def _subject(chemistry: Any, provenance: Mapping | None) -> dict[str, Any]:
    """Echo only schema-valid provenance values, so error responses stay valid."""
    prov = dict(provenance or {}) if isinstance(provenance, Mapping) else {}

    def as_str(value: Any) -> str | None:
        return value if isinstance(value, str) else None

    def as_int(value: Any, *, minimum: int | None = None) -> int | None:
        if isinstance(value, bool) or not isinstance(value, Integral):
            return None
        if minimum is not None and value < minimum:
            return None
        return int(value)

    return {
        "chemistry": chemistry if isinstance(chemistry, str) else None,
        "cell_id": as_str(prov.get("cell_id")),
        "dataset_id": as_str(prov.get("dataset_id")),
        "cycle_id": as_int(prov.get("cycle_id")),
        "cycle_position": as_int(prov.get("cycle_position"), minimum=0),
        "prediction_timestamp": as_str(prov.get("prediction_timestamp")),
    }


def _horizon(horizon_cycles: int | None) -> dict[str, Any]:
    return {
        "value": horizon_cycles,
        "unit": "observed_cycle_records",
        "semantics": SOH_HORIZON_SEMANTICS,
    }


def _empty_soh(horizon_cycles: int | None) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "predicted_soh_pct": None,
        "current_observed_soh_pct": None,
        "unit": "%",
        "horizon": _horizon(horizon_cycles),
        "model_id": None,
        "model_status": None,
        "validation_status": "not_available",
        "uncertainty": {"status": "not_provided"},
    }


def _rul_state(chem: str | None) -> dict[str, Any]:
    if chem == "NMC":
        return {
            "status": "unavailable",
            "rul_cycles": None,
            "unit": "cycles",
            "reason": "UNVALIDATED_PRE_EOL",
            "approved_for_use": False,
            "model_id": RUL_MODEL_ID,
            "model_status": "unvalidated_pre_eol",
        }
    return {
        "status": "unavailable",
        "rul_cycles": None,
        "unit": "cycles",
        "reason": "CHEMISTRY_UNSUPPORTED" if chem else "CHEMISTRY_NOT_DECLARED",
        "approved_for_use": False,
        "model_id": None,
        "model_status": None,
    }


def validate_soh_output(value: Any) -> bool:
    """Accept only finite numeric SOH values in the closed interval [0, 100]."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    value = float(value)
    return math.isfinite(value) and 0.0 <= value <= 100.0


def _finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _validate_provenance(provenance: Mapping | None) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(provenance, Mapping):
        return [_error("MISSING_PROVENANCE", "provenance mapping is required", "provenance")]
    for field in ("cell_id", "dataset_id"):
        value = provenance.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                _error("INVALID_PROVENANCE", f"{field} must be a non-empty string", field)
            )
    cycle_id = provenance.get("cycle_id")
    if isinstance(cycle_id, bool) or not isinstance(cycle_id, Integral):
        errors.append(
            _error("INVALID_PROVENANCE", "cycle_id must be an integer", "cycle_id")
        )
    cycle_position = provenance.get("cycle_position")
    if cycle_position is not None and (
        isinstance(cycle_position, bool)
        or not isinstance(cycle_position, Integral)
        or cycle_position < 0
    ):
        errors.append(
            _error(
                "INVALID_PROVENANCE",
                "cycle_position must be a non-negative integer when supplied",
                "cycle_position",
            )
        )
    return errors


def _feature_problems(
    features: Any, expected: list[str], strict: bool
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Return (errors, warnings, ignored_extra_keys)."""
    errors: list[dict[str, Any]] = []
    warning_text: list[str] = []
    if not isinstance(features, Mapping):
        return [_error("INVALID_FEATURES", "features must be a mapping", "features")], [], []
    expected_set = set(expected)
    provided = set(features)
    missing = sorted(expected_set - provided)
    extra = sorted(provided - expected_set)
    if missing:
        errors.append(
            _error(
                "MISSING_FEATURES",
                f"Missing features for declared chemistry: {missing}",
                "features",
            )
        )
    if extra:
        if strict:
            errors.append(
                _error(
                    "FEATURE_SET_MISMATCH",
                    "Feature keys do not match the declared chemistry schema; "
                    f"unexpected keys: {extra}. This guard prevents cross-chemistry "
                    "scoring (the base predictor ignores extras with only a warning).",
                    "features",
                )
            )
        else:
            warning_text.append(
                f"Unexpected feature keys ignored by the model: {extra}"
            )
    bad_values = []
    for name in expected:
        if name not in features:
            continue
        try:
            value = float(features[name])
        except (TypeError, ValueError):
            bad_values.append(name)
            continue
        if not math.isfinite(value):
            bad_values.append(name)
    if bad_values:
        errors.append(
            _error(
                "INVALID_FEATURE_VALUES",
                f"Non-finite or non-numeric feature values: {sorted(bad_values)}",
                "features",
            )
        )
    return errors, warning_text, extra if not strict else []


def _finalize(
    status: str,
    subject: dict[str, Any],
    soh: dict[str, Any],
    errors: list[dict[str, Any]],
    flags: set[str],
    warning_text: list[str],
    checks_passed: list[str],
    capacity: dict[str, Any] | None,
    horizon_cycles: int | None,
    source_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chem = subject.get("chemistry")
    chem = chem.upper() if isinstance(chem, str) else None
    label_status = {
        "NMC": "observed",
        "LFP": "conditional",
    }.get(chem, "not_applicable")
    source = {
        "module": "member2_modeling/predictor.py",
        "function": "Member2Predictor.predict_future_soh",
        "schema_version": None,
    }
    if source_extra:
        source.update(source_extra)
    return {
        "contract_version": CONTRACT_VERSION,
        "status": status,
        "production_ready": False,
        "subject": subject,
        "soh": soh,
        "rul": _rul_state(chem),
        "capacity": capacity,
        "quality": {
            "quality_flags": sorted(flags),
            "warnings": warning_text,
            "checks_passed": checks_passed,
            "checks_not_performed": list(CHECKS_NOT_PERFORMED),
            "label_provenance": {
                "soh_labels": "measured_capacity_ratio" if chem in SUPPORTED_SOH_CHEMISTRIES else None,
                "status": label_status,
            },
        },
        "errors": errors,
        "source": source,
        "request": {"horizon_cycles": horizon_cycles},
    }


def build_health_result(
    predictor: Member2Predictor,
    features: Mapping[str, Any],
    chemistry: str,
    provenance: Mapping[str, Any],
    *,
    horizon_cycles: int = 1,
    strict_feature_set: bool = True,
    include_capacity: bool = False,
) -> dict[str, Any]:
    """Build the v1.1.0-draft battery-health result from the real predictor.

    Args:
        predictor: an initialised ``Member2Predictor``.
        features: feature-name → value mapping for the current cycle.
        chemistry: caller-declared chemistry ("NMC", "LFP", "Na-ion", ...).
        provenance: must contain ``cell_id``, ``dataset_id`` and integer
            ``cycle_id``; optional ``cycle_position`` (0-based contiguous
            observed index) and ``prediction_timestamp``.
        horizon_cycles: passed through to the predictor (only 1 is supported).
        strict_feature_set: reject extra feature keys instead of ignoring them.
        include_capacity: also return the derived cell-level capacity in Ah.
    """
    chem = chemistry.upper() if isinstance(chemistry, str) else None
    subject = _subject(chemistry, provenance)
    flags = set(BASE_FLAGS)
    warnings_out: list[str] = []
    source_extra = {"schema_version": getattr(predictor, "schema_version", None)}

    prov_errors = _validate_provenance(provenance)
    if prov_errors:
        flags.add("PROVENANCE_MISSING_OR_INVALID")
        return _finalize(
            "error", subject, _empty_soh(horizon_cycles), prov_errors,
            flags, warnings_out, [], None, horizon_cycles, source_extra,
        )

    if isinstance(horizon_cycles, bool) or not isinstance(horizon_cycles, Integral) or horizon_cycles != 1:
        flags.add("HORIZON_UNSUPPORTED")
        return _finalize(
            "unsupported",
            subject,
            _empty_soh(horizon_cycles),
            [_error(
                "UNSUPPORTED_HORIZON",
                "Only horizon_cycles=1 (next observed record) is supported.",
                "horizon_cycles",
            )],
            flags,
            warnings_out,
            ["provenance_supplied"],
            None,
            horizon_cycles,
            source_extra,
        )

    if chem not in SUPPORTED_SOH_CHEMISTRIES:
        if chem == "NA-ION":
            flags.add("NAION_INFERENCE_UNSUPPORTED")
        flags.add("CHEMISTRY_UNSUPPORTED")
        return _finalize(
            "unsupported",
            subject,
            _empty_soh(horizon_cycles),
            [_error(
                "UNSUPPORTED_CHEMISTRY",
                f"Chemistry '{chemistry}' not supported. Supported SOH chemistries: NMC, LFP.",
                "chemistry",
            )],
            flags,
            warnings_out,
            ["provenance_supplied"],
            None,
            horizon_cycles,
            source_extra,
        )

    schema_info = predictor.get_feature_schema(chem)
    if not schema_info:
        flags.add("MODEL_SCHEMA_UNAVAILABLE")
        return _finalize(
            "error", subject, _empty_soh(horizon_cycles),
            [_error("MODEL_SCHEMA_UNAVAILABLE", f"No schema entries for {chem}.", "chemistry")],
            flags, warnings_out, ["provenance_supplied"], None, horizon_cycles, source_extra,
        )

    expected = list(schema_info["feature_names"])
    feature_errors, feature_warnings, ignored_extra = _feature_problems(
        features, expected, strict_feature_set
    )
    warnings_out.extend(feature_warnings)
    if ignored_extra:
        flags.add("EXTRA_FEATURE_KEYS_IGNORED")
    if feature_errors:
        flags.add("FEATURE_VALIDATION_FAILED")
        return _finalize(
            "error", subject, _empty_soh(horizon_cycles), feature_errors,
            flags, warnings_out, ["provenance_supplied"], None, horizon_cycles, source_extra,
        )

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = predictor.predict_future_soh(features, chem, horizon_cycles)
        warnings_out.extend(str(item.message) for item in caught)
    except Exception as exc:  # noqa: BLE001 - surface as contract error
        flags.add("PREDICTION_EXCEPTION")
        return _finalize(
            "error", subject, _empty_soh(horizon_cycles),
            [_error("PREDICTION_EXCEPTION", f"{type(exc).__name__}: {exc}")],
            flags, warnings_out, ["provenance_supplied"], None, horizon_cycles, source_extra,
        )

    if result.status != "ok":
        code = "UNSUPPORTED_CHEMISTRY" if result.status == "unsupported" else "PREDICTION_ERROR"
        flags.add("INFERENCE_NOT_AVAILABLE")
        return _finalize(
            result.status,
            subject,
            _empty_soh(result.horizon_cycles),
            [_error(code, result.error or f"Predictor returned status={result.status}.")],
            flags,
            warnings_out,
            ["provenance_supplied"],
            None,
            result.horizon_cycles,
            source_extra,
        )

    if not validate_soh_output(result.prediction):
        flags.add("INVALID_SOH_OUTPUT")
        warnings_out.append(
            "The model returned a non-finite or out-of-range SOH value; "
            "the value was rejected and is not exposed."
        )
        return _finalize(
            "error",
            subject,
            _empty_soh(result.horizon_cycles),
            [_error(
                "INVALID_SOH_OUTPUT",
                "Predicted SOH must be finite and within 0–100 %; value rejected.",
                "soh.predicted_soh_pct",
            )],
            flags,
            warnings_out,
            ["provenance_supplied", "feature_set_matches_schema", "feature_values_finite"],
            None,
            result.horizon_cycles,
            source_extra,
        )

    current_soh = features.get("current_soh_pct") if isinstance(features, Mapping) else None
    current_soh_pct = float(current_soh) if validate_soh_output(current_soh) else None
    if current_soh_pct is None:
        warnings_out.append(
            "Current observed SOH was not available from the feature set; "
            "soh.current_observed_soh_pct is null and no persistence anchor is supplied."
        )
    soh = {
        "status": "ok",
        "predicted_soh_pct": float(result.prediction),
        "current_observed_soh_pct": current_soh_pct,
        "unit": "%",
        "horizon": _horizon(result.horizon_cycles),
        "model_id": result.model_name,
        "model_status": result.model_status,
        "validation_status": (
            "research_stage_not_production_ready"
            if chem == "NMC"
            else "research_stage_conditional_labels_not_production_ready"
        ),
        "uncertainty": {"status": "not_provided"},
    }
    if chem == "LFP":
        flags.add("CONDITIONAL_LABELS")
    if result.caveat:
        warnings_out.append(result.caveat)
    warnings_out.append(
        "RUL is not exposed: no approved RUL model exists; "
        "the shipped NMC RUL artifact is unvalidated for pre-EOL forecasting."
    )

    capacity = None
    if include_capacity:
        try:
            cap = predictor.predict_capacity(features, chem, horizon_cycles)
        except Exception as exc:  # noqa: BLE001
            warnings_out.append(f"Capacity derivation failed: {type(exc).__name__}: {exc}")
        else:
            if cap.status == "ok" and _finite_nonnegative(cap.prediction):
                capacity = {
                    "capacity_ah": float(cap.prediction),
                    "unit": "Ah",
                    "reference_capacity_ah": float(predictor.RATED_CAPACITY[chem]),
                    "basis": "derived_soh_pct_times_rated_capacity",
                    "validation_status": "not_independently_validated",
                    "is_pack_energy": False,
                }
                flags.add("CAPACITY_DERIVED_FROM_SOH")
            else:
                warnings_out.append("Capacity derivation was unavailable or invalid; omitted.")

    checks_passed = [
        "provenance_supplied",
        "feature_set_matches_schema",
        "feature_values_finite",
        "soh_output_within_0_100",
        "model_loaded_and_scored",
    ]
    source_extra["model_file"] = (
        predictor.schema.get("models", {}).get(result.model_name, {}).get("model_file")
    )
    return _finalize(
        "ok_experimental",
        subject,
        soh,
        [],
        flags,
        warnings_out,
        checks_passed,
        capacity,
        result.horizon_cycles,
        source_extra,
    )


def build_range_arguments(
    contract: Mapping[str, Any],
    *,
    ambient_c: float | None = None,
    hvac_power_kw: float | None = None,
) -> dict[str, Any]:
    """Prepare the provisional ``RangeCalculator.estimate`` argument mapping.

    The four-argument signature (``soh_pct``, ``cycle``, ``ambient_c``,
    ``hvac_power_kw``) is *unverified*: no Member 3 source file exists in this
    repository or its git history. The returned ``arguments`` object contains
    exactly those four keys and nothing else.

    Member 2 owns only ``soh_pct`` and the observed ``cycle`` identifier.
    ``ambient_c`` and ``hvac_power_kw`` must come from the Member 3 vehicle
    scenario; they are never invented here.
    """
    notes = {
        "signature_verified": False,
        "signature_note": (
            "Provisional four-argument form soh_pct, cycle, ambient_c, "
            "hvac_power_kw. Member 3 range_calculator.py is not present in the "
            "repository or git history, so the real signature is unverified."
        ),
        "cycle_note": (
            "cycle is the observed cycle_id passed through unchanged. It is not "
            "a contiguous counter: LFP B5 observed IDs are 1-100, 151-200 and "
            "401-500. If Member 3 needs a contiguous cycle count, read "
            "contract.subject.cycle_position (null when the caller did not "
            "supply it)."
        ),
        "cycle_semantics": "observed_cycle_id",
        "soh_semantics": (
            "soh_pct is the predicted SOH of the next observed record. "
            "contract.soh.current_observed_soh_pct is the measured current SOH "
            "(the persistence baseline) if an anchored or gated use is preferred; "
            "persistence beat this model on every reported holdout."
        ),
        "assumption": (
            "SOH linearly scales usable pack energy. This is an approximation; "
            "the resulting range is unvalidated until checked against "
            "vehicle-specific evidence."
        ),
        "production_ready": bool(contract.get("production_ready", False)),
    }

    if contract.get("status") not in ("ok", "ok_experimental"):
        return {
            "status": "blocked",
            "reason": "SOH_UNAVAILABLE",
            "message": "A valid experimental SOH result is required before calling the range calculator.",
            "arguments": None,
            "missing": [],
            **notes,
        }

    soh = contract.get("soh") or {}
    soh_value = soh.get("predicted_soh_pct")
    if not validate_soh_output(soh_value):
        return {
            "status": "blocked",
            "reason": "INVALID_SOH",
            "message": "The contract does not contain a finite SOH in 0–100 %.",
            "arguments": None,
            "missing": [],
            **notes,
        }

    missing: list[str] = []
    cycle = (contract.get("subject") or {}).get("cycle_id")
    if isinstance(cycle, bool) or not isinstance(cycle, Integral):
        missing.append("cycle")
    if ambient_c is None:
        missing.append("ambient_c")
    elif isinstance(ambient_c, bool) or not isinstance(ambient_c, Real) or not math.isfinite(float(ambient_c)):
        return {
            "status": "blocked",
            "reason": "INVALID_SCENARIO_INPUT",
            "message": "ambient_c must be a finite number in °C.",
            "arguments": None,
            "missing": [],
            **notes,
        }
    if hvac_power_kw is None:
        missing.append("hvac_power_kw")
    elif (
        isinstance(hvac_power_kw, bool)
        or not isinstance(hvac_power_kw, Real)
        or not math.isfinite(float(hvac_power_kw))
        or float(hvac_power_kw) < 0.0
    ):
        return {
            "status": "blocked",
            "reason": "INVALID_SCENARIO_INPUT",
            "message": "hvac_power_kw must be a finite non-negative number in kW.",
            "arguments": None,
            "missing": [],
            **notes,
        }

    if missing:
        return {
            "status": "blocked",
            "reason": "MISSING_INPUTS",
            "message": "Member 3 scenario inputs are required: " + ", ".join(missing),
            "arguments": None,
            "missing": missing,
            **notes,
        }

    return {
        "status": "ok_experimental",
        "arguments": {
            "soh_pct": float(soh_value),
            "cycle": int(cycle),
            "ambient_c": float(ambient_c),
            "hvac_power_kw": float(hvac_power_kw),
        },
        "missing": [],
        **notes,
    }
