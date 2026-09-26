"""Explicit, estimand-aware sensitivity analysis over existing methods."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .exceptions import InvalidDataError
from .narrate import sensitivity_verdict
from .results import AnalysisResult, AnalysisStatus
from .specifications import AnalysisSpecification, _json_value

SENSITIVITY_SCHEMA_VERSION = 1


class SensitivityStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class ScenarioStatus(str, Enum):
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"
    FAILED = "failed"


class Comparability(str, Enum):
    SAME_ESTIMAND = "same_estimand"
    DIFFERENT_ESTIMAND = "different_estimand"
    INCOMPATIBLE = "incompatible"
    UNAVAILABLE = "unavailable"


_PLANNING = {"planned", "exploratory", "unknown"}
_QUANTITIES = {
    "welch_t": "mean_difference",
    "student_t": "mean_difference",
    "paired_t": "mean_difference",
    "mann_whitney_u": "rank_biserial",
    "one_way_anova": "eta_squared",
    "kruskal_wallis": "epsilon_squared",
    "pearson_correlation": "pearson_r",
    "pearson_chi_square": "cramers_v",
}


def _text(value: Any, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        raise InvalidDataError(
            f"{name} must be non-empty text" + (" when supplied." if optional else ".")
        )


@dataclass(frozen=True)
class SensitivitySpecification:
    """One researcher-declared analytical variation; it never contains data."""

    name: str
    specification: AnalysisSpecification
    method_id: str | None = None
    rationale: str | None = None
    planning_status: str = "unknown"
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.name, "scenario name")
        if not isinstance(self.specification, AnalysisSpecification):
            raise InvalidDataError("scenario specification must be an AnalysisSpecification.")
        _text(self.method_id, "method_id", optional=True)
        _text(self.rationale, "scenario rationale", optional=True)
        if self.planning_status not in _PLANNING:
            raise InvalidDataError("planning_status must be planned, exploratory, or unknown.")
        if any(not isinstance(item, str) or not item.strip() for item in self.assumptions):
            raise InvalidDataError("scenario assumptions must contain non-empty text.")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": SENSITIVITY_SCHEMA_VERSION,
                "name": self.name,
                "specification": self.specification.to_dict(),
                "method_id": self.method_id,
                "rationale": self.rationale,
                "planning_status": self.planning_status,
                "assumptions": self.assumptions,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SensitivitySpecification:
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise InvalidDataError("Sensitivity specification schema_version must be 1.")
        try:
            return cls(
                name=payload["name"],
                specification=AnalysisSpecification.from_dict(payload["specification"]),
                method_id=payload.get("method_id"),
                rationale=payload.get("rationale"),
                planning_status=payload.get("planning_status", "unknown"),
                assumptions=tuple(payload.get("assumptions", ())),
            )
        except (KeyError, TypeError) as exc:
            raise InvalidDataError(
                "Sensitivity specification fields are missing or invalid."
            ) from exc


@dataclass(frozen=True)
class SensitivityScenario(SensitivitySpecification):
    """Named alias for callers who prefer scenario terminology."""


@dataclass(frozen=True)
class SensitivityScenarioResult:
    name: str
    specification: SensitivitySpecification
    requested_method_id: str | None
    method_id: str | None
    status: ScenarioStatus
    comparability: Comparability
    estimate_quantity: str | None = None
    primary_estimate: float | None = None
    effect_size_quantity: str | None = None
    effect_size: float | None = None
    confidence_interval: dict[str, Any] | None = None
    p_value: float | None = None
    sample_size: int | None = None
    excluded_rows: int | None = None
    group_contrast: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None
    analysis: AnalysisResult | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _text(self.name, "scenario result name")
        if not isinstance(self.specification, SensitivitySpecification):
            raise InvalidDataError(
                "scenario result specification must be a SensitivitySpecification."
            )
        _text(self.requested_method_id, "requested_method_id", optional=True)
        _text(self.method_id, "method_id", optional=True)
        try:
            object.__setattr__(self, "status", ScenarioStatus(self.status))
            object.__setattr__(self, "comparability", Comparability(self.comparability))
        except (TypeError, ValueError) as exc:
            raise InvalidDataError("Scenario status or comparability is invalid.") from exc
        for name in ("primary_estimate", "effect_size", "p_value"):
            value = getattr(self, name)
            if value is not None and _finite(value) is None:
                raise InvalidDataError(f"scenario result {name} must be finite or None.")
        for name in ("sample_size", "excluded_rows"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise InvalidDataError(
                    f"scenario result {name} must be a nonnegative integer or None."
                )
        if any(not isinstance(item, str) or not item.strip() for item in self.warnings):
            raise InvalidDataError("scenario result warnings must contain non-empty text.")
        _text(self.error, "scenario result error", optional=True)
        if self.analysis is not None and not isinstance(self.analysis, AnalysisResult):
            raise InvalidDataError("scenario result analysis must be an AnalysisResult or None.")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": SENSITIVITY_SCHEMA_VERSION,
                "name": self.name,
                "specification": self.specification.to_dict(),
                "requested_method_id": self.requested_method_id,
                "method_id": self.method_id,
                "status": self.status.value,
                "comparability": self.comparability.value,
                "estimate_quantity": self.estimate_quantity,
                "primary_estimate": self.primary_estimate,
                "effect_size_quantity": self.effect_size_quantity,
                "effect_size": self.effect_size,
                "confidence_interval": self.confidence_interval,
                "p_value": self.p_value,
                "sample_size": self.sample_size,
                "excluded_rows": self.excluded_rows,
                "group_contrast": self.group_contrast,
                "comparison": self.comparison,
                "warnings": self.warnings,
                "error": self.error,
            }
        )


@dataclass(frozen=True)
class SensitivityResult:
    base_result: AnalysisResult
    scenario_results: tuple[SensitivityScenarioResult, ...]
    status: SensitivityStatus
    comparison_summary: dict[str, Any]
    warnings: tuple[str, ...]
    provenance: dict[str, Any]
    reproducibility: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.base_result, AnalysisResult):
            raise InvalidDataError("base_result must be an AnalysisResult.")
        try:
            object.__setattr__(self, "status", SensitivityStatus(self.status))
        except (TypeError, ValueError) as exc:
            raise InvalidDataError("Sensitivity result status is invalid.") from exc
        if any(not isinstance(item, SensitivityScenarioResult) for item in self.scenario_results):
            raise InvalidDataError("scenario_results contains an invalid record.")
        if not self.scenario_results:
            raise InvalidDataError("scenario_results must retain at least one declared scenario.")
        if any(not isinstance(item, str) or not item.strip() for item in self.warnings):
            raise InvalidDataError("sensitivity warnings must contain non-empty text.")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": SENSITIVITY_SCHEMA_VERSION,
                "status": self.status.value,
                "base_result": self.base_result.to_dict(),
                "scenario_results": [item.to_dict() for item in self.scenario_results],
                "comparison_summary": self.comparison_summary,
                "warnings": self.warnings,
                "provenance": self.provenance,
                "reproducibility": self.reproducibility,
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    @property
    def verdict(self) -> str:
        """Return a deterministic decision-consistency narrative from stored scenarios."""
        specification = self.base_result.specification
        alpha = specification.options.alpha if specification is not None else 0.05
        base_p = _finite(self.base_result.values.get("p_value"))
        completed = [
            item for item in self.scenario_results if item.status is ScenarioStatus.COMPLETED
        ]
        same_estimand = [
            item for item in completed if item.comparability is Comparability.SAME_ESTIMAND
        ]
        mixed_estimands = any(
            item.comparability is Comparability.DIFFERENT_ESTIMAND for item in completed
        )
        return sensitivity_verdict(
            _hypothesis_decision(base_p, alpha),
            tuple(_hypothesis_decision(item.p_value, alpha) for item in completed),
            same_estimand=bool(same_estimand),
            completed_scenarios=len(completed),
            mixed_estimands=mixed_estimands,
            alpha=alpha,
        )

    def compare(self) -> str:
        """Return a descriptive text comparison without inferring robustness.

        Values are read from the stored results; no statistics are recalculated.
        Hypothesis-decision consistency is assessed only for completed scenarios
        with the same estimand and scientific comparison identity as the primary
        analysis.
        """
        from .results import _method_label

        lines: list[str] = []
        sep = "-" * 78
        base = self.base_result

        base_p = _finite(base.values.get("p_value"))
        base_estimate = _finite(base.values.get("primary_estimate"))
        base_effect_info = base.values.get("effect_size", {})
        base_effect = (
            _finite(base_effect_info.get("value")) if isinstance(base_effect_info, dict) else None
        )
        base_effect_name = (
            base_effect_info.get("name", "effect")
            if isinstance(base_effect_info, dict)
            else "effect"
        )
        base_p_text = f"p = {base_p:.4g}" if base_p is not None else "p = unavailable"
        base_estimate_text = (
            f"est = {base_estimate:.4g}" if base_estimate is not None else "est = unavailable"
        )
        base_effect_text = (
            f"{base_effect_name} = {base_effect:.4g}" if base_effect is not None else ""
        )

        lines.extend((sep, " SENSITIVITY ANALYSIS - descriptive result comparison", sep))
        primary_label = f" Primary ({base.method_label}):"
        lines.append(
            f"{primary_label:<40} {base_p_text:<18} {base_estimate_text:<20} {base_effect_text}"
        )

        completed_same_estimand = 0
        completed_different_estimand = 0
        for scenario in self.scenario_results:
            method_label = _method_label(scenario.method_id)
            p_text = (
                f"p = {scenario.p_value:.4g}" if scenario.p_value is not None else "p = unavailable"
            )
            estimate_text = (
                f"est = {scenario.primary_estimate:.4g}"
                if scenario.primary_estimate is not None
                else "est = unavailable"
            )
            effect_name = scenario.effect_size_quantity or "effect"
            effect_text = (
                f"{effect_name} = {scenario.effect_size:.4g}"
                if scenario.effect_size is not None
                else ""
            )
            notes = []
            if scenario.status is not ScenarioStatus.COMPLETED:
                notes.append(scenario.status.value)
            if scenario.comparability is Comparability.DIFFERENT_ESTIMAND:
                notes.append("different estimand")
            elif scenario.comparability is Comparability.INCOMPATIBLE:
                notes.append("incompatible comparison")
            elif scenario.comparability is Comparability.UNAVAILABLE:
                notes.append("comparison unavailable")
            note_text = f" [{'; '.join(notes)}]" if notes else ""
            scenario_label = f" Scenario ({method_label}):"
            lines.append(
                f"{scenario_label:<40} {p_text:<18} {estimate_text:<20} {effect_text}{note_text}"
            )
            if (
                scenario.status is ScenarioStatus.COMPLETED
                and scenario.comparability is Comparability.SAME_ESTIMAND
            ):
                completed_same_estimand += 1
            elif (
                scenario.status is ScenarioStatus.COMPLETED
                and scenario.comparability is Comparability.DIFFERENT_ESTIMAND
            ):
                completed_different_estimand += 1

        lines.append(sep)
        lines.append(
            f" {completed_same_estimand} completed same-estimand scenario(s); "
            f"{completed_different_estimand} completed scenario(s) use a different estimand."
        )
        lines.append(" " + self.verdict)

        lines.append(sep)
        return "\n".join(lines)


def estimate_quantity(method_id: str) -> str | None:
    """Return the stable primary quantity identity for a supported method."""
    return _QUANTITIES.get(method_id)


def _identity(result: AnalysisResult) -> dict[str, Any]:
    specification = result.specification
    if specification is None:
        return {}
    question = specification.question
    return {
        "objective": question.objective.value if question.objective is not None else None,
        "outcome": question.outcome,
        "predictor": question.predictor,
        "design": specification.design.value,
        "estimand": question.estimand,
        "unit_id": specification.unit_id,
        "estimate_quantity": estimate_quantity(result.method_id),
        "contrast": result.metadata.get("contrast"),
    }


def classify_comparability(base: AnalysisResult, scenario: AnalysisResult) -> Comparability:
    """Classify scientific comparability without consulting p-values."""
    if (
        base.status is not AnalysisStatus.AVAILABLE
        or scenario.status is not AnalysisStatus.AVAILABLE
    ):
        return Comparability.UNAVAILABLE
    left, right = _identity(base), _identity(scenario)
    if not left or not right:
        return Comparability.UNAVAILABLE
    for field_name in ("objective", "outcome", "predictor", "design"):
        if left[field_name] != right[field_name]:
            return Comparability.INCOMPATIBLE
    if left["design"] == "paired" and left["unit_id"] != right["unit_id"]:
        return Comparability.INCOMPATIBLE
    if left["estimand"] != right["estimand"]:
        return Comparability.DIFFERENT_ESTIMAND
    if left["design"] == "paired":
        if left["contrast"] != right["contrast"]:
            return Comparability.INCOMPATIBLE
    if left["estimate_quantity"] != right["estimate_quantity"]:
        return Comparability.DIFFERENT_ESTIMAND
    return Comparability.SAME_ESTIMAND


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _hypothesis_decision(p_value: float | None, alpha: float) -> str | None:
    if p_value is None:
        return None
    return "reject null" if p_value < alpha else "fail to reject null"


def _interval(result: AnalysisResult) -> dict[str, Any] | None:
    interval = result.values.get("confidence_interval")
    if not isinstance(interval, dict):
        return None
    lower, upper, level = map(
        _finite, (interval.get("lower"), interval.get("upper"), interval.get("level"))
    )
    if lower is None or upper is None or level is None or lower > upper or not 0 < level < 1:
        return None
    return _json_value(interval)


def _contrast_orientation(base: AnalysisResult, scenario: AnalysisResult) -> str:
    left = base.metadata.get("contrast")
    right = scenario.metadata.get("contrast")
    if left == right:
        return "same"
    if isinstance(left, dict) and isinstance(right, dict):
        if left.get("first") == right.get("second") and left.get("second") == right.get("first"):
            return "reversed"
    return "different"


def _direction(value: float) -> str:
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def compare_same_estimand(base: AnalysisResult, scenario: AnalysisResult) -> dict[str, Any]:
    """Compare compatible estimates descriptively, normalizing a reversed contrast."""
    base_estimate = _finite(base.values.get("primary_estimate"))
    scenario_estimate = _finite(scenario.values.get("primary_estimate"))
    if base_estimate is None or scenario_estimate is None:
        return {"available": False, "reason": "A finite primary estimate is unavailable."}
    orientation = _contrast_orientation(base, scenario)
    if orientation == "different":
        return {"available": False, "reason": "Group contrasts do not identify the same groups."}
    normalized = -scenario_estimate if orientation == "reversed" else scenario_estimate
    difference = normalized - base_estimate
    scale = max(abs(base_estimate), abs(normalized), 1.0)
    relative = (
        None if abs(base_estimate) <= math.ulp(scale) * 32 else difference / abs(base_estimate)
    )
    base_interval = _interval(base)
    scenario_interval = _interval(scenario)
    normalized_interval = scenario_interval
    if scenario_interval is not None and orientation == "reversed":
        normalized_interval = {
            **scenario_interval,
            "lower": -float(scenario_interval["upper"]),
            "upper": -float(scenario_interval["lower"]),
        }
    overlap = None
    same_level = None
    if base_interval is not None and normalized_interval is not None:
        overlap = max(float(base_interval["lower"]), float(normalized_interval["lower"])) <= min(
            float(base_interval["upper"]), float(normalized_interval["upper"])
        )
        same_level = math.isclose(
            float(base_interval["level"]), float(normalized_interval["level"]), abs_tol=1e-12
        )
    return _json_value(
        {
            "available": True,
            "contrast_orientation": orientation,
            "orientation_transformation": (
                "Scenario estimate and interval signs were reversed for comparison with the "
                "base first-minus-second contrast."
                if orientation == "reversed"
                else None
            ),
            "base_estimate": base_estimate,
            "scenario_estimate_raw": scenario_estimate,
            "scenario_estimate_normalized": normalized,
            "estimate_difference_from_base": difference,
            "relative_estimate_change": relative,
            "direction_consistent": _direction(base_estimate) == _direction(normalized),
            "base_confidence_interval": base_interval,
            "scenario_confidence_interval_normalized": normalized_interval,
            "interval_overlap": overlap,
            "same_confidence_level": same_level,
            "interval_overlap_is_descriptive": overlap is not None,
            "sample_size_change": (
                scenario.sample_size - base.sample_size
                if scenario.sample_size is not None and base.sample_size is not None
                else None
            ),
        }
    )


def scenario_values(result: AnalysisResult) -> dict[str, Any]:
    """Extract validated, aggregate display values from one analysis result."""
    effect = result.values.get("effect_size")
    effect_name = effect.get("name") if isinstance(effect, dict) else None
    effect_value = _finite(effect.get("value")) if isinstance(effect, dict) else None
    effect_quantities = {
        "Cohen's d": "cohens_d",
        "Cohen's dz": "cohens_dz",
        "rank-biserial correlation": "rank_biserial",
        "eta-squared": "eta_squared",
        "epsilon-squared": "epsilon_squared",
        "Cramer's V": "cramers_v",
        "Pearson r": "pearson_r",
    }
    return {
        "estimate_quantity": estimate_quantity(result.method_id),
        "primary_estimate": _finite(result.values.get("primary_estimate")),
        "effect_size_quantity": effect_quantities.get(str(effect_name)),
        "effect_size": effect_value,
        "confidence_interval": _interval(result),
        "p_value": _finite(result.values.get("p_value")),
        "sample_size": result.sample_size,
        "excluded_rows": result.excluded_rows,
        "group_contrast": _json_value(result.metadata.get("contrast")),
    }
