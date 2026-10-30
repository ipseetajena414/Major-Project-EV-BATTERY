"""Isolated integration-readiness contract for Member 2.

This package does not change the production predictor API. It wraps
``Member2Predictor`` and adds explicit contract/quality/error handling.
"""

from .adapter import (
    CONTRACT_VERSION,
    RUL_APPROVED_FOR_USE,
    build_health_result,
    build_range_arguments,
    validate_soh_output,
)

__all__ = [
    "CONTRACT_VERSION",
    "RUL_APPROVED_FOR_USE",
    "build_health_result",
    "build_range_arguments",
    "validate_soh_output",
]
