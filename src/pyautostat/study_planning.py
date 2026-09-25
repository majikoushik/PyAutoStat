"""Prospective mean-comparison sample-size and precision planning."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from scipy import stats

from .exceptions import InvalidDataError
from .specifications import _json_value

STUDY_PLANNING_CAPABILITIES = (
    "independent_mean_power",
    "independent_mean_precision",
    "paired_mean_power",
    "paired_mean_precision",
)


@dataclass(frozen=True)
class StudyPlanningResult:
    status: str
    planning_type: str
    method_family: str
    target_quantity: str
    assumptions: dict[str, Any]
    alpha: float | None = None
    confidence_level: float | None = None
    target_power: float | None = None
    achieved_power: float | None = None
    target_half_width: float | None = None
    achieved_half_width: float | None = None
    required_n1: int | None = None
    required_n2: int | None = None
    required_pairs: int | None = None
    total_required_n: int | None = None
    allocation_ratio: float | None = None
    achieved_allocation_ratio: float | None = None
    calculation_method: str = ""
    approximation_notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"available", "unavailable"}:
            raise InvalidDataError("Study planning status is invalid.")
        if self.planning_type not in {"power", "precision"}:
            raise InvalidDataError("planning_type must be power or precision.")
        for name in ("method_family", "target_quantity", "calculation_method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise InvalidDataError(f"{name} must be non-empty text.")
        if not isinstance(self.assumptions, dict):
            raise InvalidDataError("planning assumptions must be a mapping.")
        for name in ("required_n1", "required_n2", "required_pairs", "total_required_n"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 2
            ):
                raise InvalidDataError(f"{name} must be an integer of at least 2 or None.")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in (*self.approximation_notes, *self.warnings)
        ):
            raise InvalidDataError("Planning notes and warnings must contain non-empty text.")
        self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return _json_value({"schema_version": 1, **vars(self)})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)


def _finite_positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidDataError(f"{name} must be a finite positive number.")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise InvalidDataError(f"{name} must be a finite positive number.")
    return number


def _probability(value: Any, name: str) -> float:
    number = _finite_positive(value, name)
    if number >= 1:
        raise InvalidDataError(f"{name} must be strictly between 0 and 1.")
    return number


def _nonzero_magnitude(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidDataError(f"{name} must be a finite nonzero number.")
    number = float(value)
    if not math.isfinite(number) or number == 0:
        raise InvalidDataError(f"{name} must be a finite nonzero number.")
    return abs(number)


def _bound(maximum: Any, name: str) -> int:
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 2:
        raise InvalidDataError(f"{name} must be an integer of at least 2.")
    return maximum


def _welch_values(n1: int, n2: int, sd1: float, sd2: float) -> tuple[float, float]:
    first = sd1 * sd1 / n1
    second = sd2 * sd2 / n2
    standard_error = math.sqrt(first + second)
    degrees = (first + second) ** 2 / (first * first / (n1 - 1) + second * second / (n2 - 1))
    return standard_error, degrees


def _two_sided_power(ncp: float, degrees: float, alpha: float) -> float:
    critical = float(stats.t.ppf(1 - alpha / 2, degrees))
    # Some supported SciPy builds emit an internal divide-by-zero warning for
    # low-df noncentral-t evaluations while still returning finite probabilities.
    # Scope warning handling to these calls and reject any unusable output below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        power = float(stats.nct.cdf(-critical, degrees, ncp) + stats.nct.sf(critical, degrees, ncp))
    if not math.isfinite(power):
        raise InvalidDataError("The noncentral-t power calculation was nonfinite.")
    return min(1.0, max(0.0, power))


class StudyPlanner:
    """Bounded prospective calculations that never consume an observed result."""

    def __init__(self, *, on_result: Callable[[StudyPlanningResult], None] | None = None) -> None:
        self._on_result = on_result

    def _finish(self, result: StudyPlanningResult) -> StudyPlanningResult:
        if self._on_result is not None:
            self._on_result(result)
        return result

    def independent_mean_power(
        self,
        *,
        target_difference: float,
        sd_group1: float,
        sd_group2: float,
        alpha: float = 0.05,
        target_power: float = 0.80,
        allocation_ratio: float = 1.0,
        max_n: int = 100_000,
    ) -> StudyPlanningResult:
        difference = _nonzero_magnitude(target_difference, "target_difference")
        sd1 = _finite_positive(sd_group1, "sd_group1")
        sd2 = _finite_positive(sd_group2, "sd_group2")
        alpha = _probability(alpha, "alpha")
        target_power = _probability(target_power, "target_power")
        ratio = _finite_positive(allocation_ratio, "allocation_ratio")
        maximum = _bound(max_n, "max_n")
        for n1 in range(2, maximum + 1):
            n2 = max(2, math.ceil(ratio * n1))
            if n2 > maximum:
                break
            standard_error, degrees = _welch_values(n1, n2, sd1, sd2)
            achieved = _two_sided_power(difference / standard_error, degrees, alpha)
            if achieved >= target_power:
                return self._finish(
                    StudyPlanningResult(
                        "available",
                        "power",
                        "welch_independent_means",
                        "mean_difference",
                        {
                            "target_difference": difference,
                            "sd_group1": sd1,
                            "sd_group2": sd2,
                            "two_sided": True,
                        },
                        alpha=alpha,
                        confidence_level=1 - alpha,
                        target_power=target_power,
                        achieved_power=achieved,
                        required_n1=n1,
                        required_n2=n2,
                        total_required_n=n1 + n2,
                        allocation_ratio=ratio,
                        achieved_allocation_ratio=n2 / n1,
                        calculation_method="bounded integer search with noncentral t approximation",
                        approximation_notes=(
                            "Prospective Welch-test power approximation under supplied "
                            "SD assumptions.",
                        ),
                    )
                )
        return self._finish(
            StudyPlanningResult(
                "unavailable",
                "power",
                "welch_independent_means",
                "mean_difference",
                {
                    "target_difference": difference,
                    "sd_group1": sd1,
                    "sd_group2": sd2,
                },
                alpha=alpha,
                target_power=target_power,
                allocation_ratio=ratio,
                calculation_method="bounded integer search with noncentral t approximation",
                warnings=(f"No solution was found with each group bounded by max_n={maximum}.",),
            )
        )

    def independent_mean_precision(
        self,
        *,
        sd_group1: float,
        sd_group2: float,
        confidence_level: float = 0.95,
        target_half_width: float,
        allocation_ratio: float = 1.0,
        max_n: int = 100_000,
    ) -> StudyPlanningResult:
        sd1 = _finite_positive(sd_group1, "sd_group1")
        sd2 = _finite_positive(sd_group2, "sd_group2")
        confidence = _probability(confidence_level, "confidence_level")
        half_width = _finite_positive(target_half_width, "target_half_width")
        ratio = _finite_positive(allocation_ratio, "allocation_ratio")
        maximum = _bound(max_n, "max_n")
        for n1 in range(2, maximum + 1):
            n2 = max(2, math.ceil(ratio * n1))
            if n2 > maximum:
                break
            standard_error, degrees = _welch_values(n1, n2, sd1, sd2)
            achieved = float(stats.t.ppf((1 + confidence) / 2, degrees)) * standard_error
            if achieved <= half_width:
                return self._finish(
                    StudyPlanningResult(
                        "available",
                        "precision",
                        "welch_independent_means",
                        "mean_difference",
                        {"sd_group1": sd1, "sd_group2": sd2, "two_sided": True},
                        alpha=1 - confidence,
                        confidence_level=confidence,
                        target_half_width=half_width,
                        achieved_half_width=achieved,
                        required_n1=n1,
                        required_n2=n2,
                        total_required_n=n1 + n2,
                        allocation_ratio=ratio,
                        achieved_allocation_ratio=n2 / n1,
                        calculation_method=(
                            "bounded integer search using Welch-Satterthwaite t interval"
                        ),
                        approximation_notes=(
                            "Precision is conditional on the supplied population SD assumptions.",
                        ),
                    )
                )
        return self._finish(
            StudyPlanningResult(
                "unavailable",
                "precision",
                "welch_independent_means",
                "mean_difference",
                {"sd_group1": sd1, "sd_group2": sd2},
                confidence_level=confidence,
                target_half_width=half_width,
                allocation_ratio=ratio,
                calculation_method="bounded integer search using Welch-Satterthwaite t interval",
                warnings=(f"No solution was found with each group bounded by max_n={maximum}.",),
            )
        )

    def paired_mean_power(
        self,
        *,
        target_mean_difference: float,
        sd_difference: float,
        alpha: float = 0.05,
        target_power: float = 0.80,
        max_pairs: int = 100_000,
    ) -> StudyPlanningResult:
        difference = _nonzero_magnitude(target_mean_difference, "target_mean_difference")
        sd = _finite_positive(sd_difference, "sd_difference")
        alpha = _probability(alpha, "alpha")
        target_power = _probability(target_power, "target_power")
        maximum = _bound(max_pairs, "max_pairs")
        for pairs in range(2, maximum + 1):
            achieved = _two_sided_power(difference * math.sqrt(pairs) / sd, pairs - 1, alpha)
            if achieved >= target_power:
                return self._finish(
                    StudyPlanningResult(
                        "available",
                        "power",
                        "paired_means",
                        "mean_paired_difference",
                        {
                            "target_mean_difference": difference,
                            "sd_difference": sd,
                            "two_sided": True,
                        },
                        alpha=alpha,
                        confidence_level=1 - alpha,
                        target_power=target_power,
                        achieved_power=achieved,
                        required_pairs=pairs,
                        calculation_method="bounded integer search with paired noncentral t",
                        approximation_notes=("Required N counts complete pairs, not raw rows.",),
                    )
                )
        return self._finish(
            StudyPlanningResult(
                "unavailable",
                "power",
                "paired_means",
                "mean_paired_difference",
                {"target_mean_difference": difference, "sd_difference": sd},
                alpha=alpha,
                target_power=target_power,
                calculation_method="bounded integer search with paired noncentral t",
                warnings=(f"No solution was found below max_pairs={maximum}.",),
            )
        )

    def paired_mean_precision(
        self,
        *,
        sd_difference: float,
        confidence_level: float = 0.95,
        target_half_width: float,
        max_pairs: int = 100_000,
    ) -> StudyPlanningResult:
        sd = _finite_positive(sd_difference, "sd_difference")
        confidence = _probability(confidence_level, "confidence_level")
        half_width = _finite_positive(target_half_width, "target_half_width")
        maximum = _bound(max_pairs, "max_pairs")
        for pairs in range(2, maximum + 1):
            achieved = float(stats.t.ppf((1 + confidence) / 2, pairs - 1)) * sd / math.sqrt(pairs)
            if achieved <= half_width:
                return self._finish(
                    StudyPlanningResult(
                        "available",
                        "precision",
                        "paired_means",
                        "mean_paired_difference",
                        {"sd_difference": sd, "two_sided": True},
                        alpha=1 - confidence,
                        confidence_level=confidence,
                        target_half_width=half_width,
                        achieved_half_width=achieved,
                        required_pairs=pairs,
                        calculation_method="bounded integer search using paired t interval",
                        approximation_notes=("Required N counts complete pairs, not raw rows.",),
                    )
                )
        return self._finish(
            StudyPlanningResult(
                "unavailable",
                "precision",
                "paired_means",
                "mean_paired_difference",
                {"sd_difference": sd},
                confidence_level=confidence,
                target_half_width=half_width,
                calculation_method="bounded integer search using paired t interval",
                warnings=(f"No solution was found below max_pairs={maximum}.",),
            )
        )
