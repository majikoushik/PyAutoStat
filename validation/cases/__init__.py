"""PyAutoStat reference validation cases registry."""

from __future__ import annotations

from typing import Any

from validation.cases.association_categorical import get_association_categorical_cases
from validation.cases.foundational import get_foundational_cases
from validation.cases.multigroup import get_multigroup_cases
from validation.cases.regression import get_regression_cases
from validation.cases.reliability import get_reliability_cases
from validation.cases.repeated_factorial import get_repeated_factorial_cases


def get_all_reference_cases() -> list[dict[str, Any]]:
    """Return the complete inventory of reference validation cases across all 24 methods."""
    cases: list[dict[str, Any]] = []
    cases.extend(get_foundational_cases())
    cases.extend(get_multigroup_cases())
    cases.extend(get_repeated_factorial_cases())
    cases.extend(get_regression_cases())
    cases.extend(get_association_categorical_cases())
    cases.extend(get_reliability_cases())
    return cases


# Alias for backward compatibility
get_reference_cases = get_all_reference_cases

__all__ = [
    "get_all_reference_cases",
    "get_reference_cases",
    "get_foundational_cases",
    "get_multigroup_cases",
    "get_repeated_factorial_cases",
    "get_regression_cases",
    "get_association_categorical_cases",
    "get_reliability_cases",
]
