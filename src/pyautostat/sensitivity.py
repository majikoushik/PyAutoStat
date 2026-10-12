"""Explicit, estimand-aware sensitivity analysis over existing methods."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .exceptions import InvalidDataError
from .method_registry import METHOD_REGISTRY, estimate_quantity_for_method
from .narrate import sensitivity_verdict
from .results import AnalysisResult, AnalysisStatus
from .specifications import AnalysisSpecification, _json_value

SENSITIVITY_SCHEMA_VERSION = 2
SENSITIVITY_SPECIFICATION_SCHEMA_VERSION = 1


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

# Canonical mapping derived from method_registry
_QUANTITIES = {
    reg.method_id: reg.sensitivity_quantity
    for reg in METHOD_REGISTRY.values()
    if reg.sensitivity_quantity is not None
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
                "schema_version": SENSITIVITY_SPECIFICATION_SCHEMA_VERSION,
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
class SensitivityEstimate:
    """One term- or coefficient-level estimate within a structured sensitivity scenario."""

    key: str
    quantity: str
    estimate: float | None
    standard_error: float | None = None
    confidence_interval: dict[str, Any] | None = None
    p_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.key, "estimate key")
        _text(self.quantity, "estimate quantity")
        if self.estimate is not None and _finite(self.estimate) is None:
            raise InvalidDataError(
                f"SensitivityEstimate {self.key!r} estimate must be finite or None."
            )
        if self.standard_error is not None:
            se = _finite(self.standard_error)
            if se is None or se < 0:
                raise InvalidDataError(
                    f"SensitivityEstimate {self.key!r} standard_error "
                    "must be a finite nonnegative number or None."
                )
        if self.p_value is not None:
            pv = _finite(self.p_value)
            if pv is None or not (0.0 <= pv <= 1.0):
                raise InvalidDataError(
                    f"SensitivityEstimate {self.key!r} p_value must be in [0, 1] or None."
                )
        if self.confidence_interval is not None and not isinstance(self.confidence_interval, dict):
            raise InvalidDataError(
                f"SensitivityEstimate {self.key!r} confidence_interval must be a dict or None."
            )
        if not isinstance(self.metadata, dict):
            raise InvalidDataError(
                f"SensitivityEstimate {self.key!r} metadata must be a dictionary."
            )

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "key": self.key,
                "quantity": self.quantity,
                "estimate": self.estimate,
                "standard_error": self.standard_error,
                "confidence_interval": self.confidence_interval,
                "p_value": self.p_value,
                "metadata": self.metadata,
            }
        )


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
    estimates: tuple[SensitivityEstimate, ...] = ()

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

        if not isinstance(self.estimates, tuple):
            try:
                object.__setattr__(self, "estimates", tuple(self.estimates))
            except TypeError as exc:
                raise InvalidDataError(
                    "scenario result estimates must be an iterable of SensitivityEstimate."
                ) from exc
        for item in self.estimates:
            if not isinstance(item, SensitivityEstimate):
                raise InvalidDataError(
                    "Every element in scenario result estimates must be a SensitivityEstimate."
                )

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
                "estimates": [item.to_dict() for item in self.estimates],
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
        completed = [
            item for item in self.scenario_results if item.status is ScenarioStatus.COMPLETED
        ]
        same_estimand = [
            item for item in completed if item.comparability is Comparability.SAME_ESTIMAND
        ]
        mixed_estimands = any(
            item.comparability is Comparability.DIFFERENT_ESTIMAND for item in completed
        )

        if self.base_result.method_id == "cronbach_alpha":
            if not completed:
                return "UNAVAILABLE: No sensitivity scenario completed."
            if not same_estimand:
                return (
                    "DESCRIPTIVE: No completed alternative shares the primary item set and scale "
                    "definition. Alpha estimates reflect different scale definitions."
                )
            if mixed_estimands:
                return (
                    "CONSISTENT across methods (note: not all test the same estimand). "
                    "Descriptive Cronbach's alpha scale consistency evaluated across scenarios "
                    "with mixed scale definitions. Scenarios with changed item sets define "
                    "different scales."
                )
            return (
                "CONSISTENT scale reliability across declared scenario(s). Descriptive Cronbach's "
                "alpha estimates reflect the same declared item set."
            )

        if self.base_result.method_id in {"linear_regression", "logistic_regression"}:
            if not completed:
                return "UNAVAILABLE: No sensitivity scenario completed."
            if not same_estimand:
                return (
                    "DESCRIPTIVE: No completed alternative shares the primary regression model "
                    "specification. Conditional coefficients cannot be directly compared "
                    "across different specifications."
                )
            all_decisions_consistent = True
            for sc in same_estimand:
                comp = sc.comparison or {}
                if comp.get("all_matched_terms_decision_consistent") is False:
                    all_decisions_consistent = False
                    break
            if not all_decisions_consistent:
                return (
                    "INCONSISTENT: Coefficient hypothesis decisions differ across covariance "
                    f"specifications for one or more terms at alpha = {alpha:.4g}."
                )
            agreement = (
                f"All comparable regression coefficient hypothesis decisions agree at alpha = "
                f"{alpha:.4g}."
            )
            if mixed_estimands:
                return (
                    "CONSISTENT across methods (note: not all test the same estimand). "
                    f"{agreement} Direct numerical comparison across different estimands is not "
                    "appropriate. This descriptive agreement does not by itself establish "
                    "robustness."
                )
            return (
                "CONSISTENT: All comparable analyses yield consistent hypothesis decisions "
                f"across model terms at alpha = {alpha:.4g}. This descriptive agreement does not "
                "by itself establish robustness."
            )

        base_p = _finite(self.base_result.values.get("p_value"))
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

        lines.extend((sep, " SENSITIVITY ANALYSIS - descriptive result comparison", sep))

        if base.method_id in {"linear_regression", "logistic_regression"}:
            return self._compare_regression(lines, sep)
        if base.method_id == "cronbach_alpha":
            return self._compare_cronbach(lines, sep)

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

    def _compare_regression(self, lines: list[str], sep: str) -> str:
        base = self.base_result
        primary_label = f" Primary ({base.method_label}):"
        lines.append(f"{primary_label}")
        base_coeffs = base.values.get("coefficients", [])
        if isinstance(base_coeffs, list):
            for c in base_coeffs:
                term = str(c.get("term", ""))
                est = _finite(c.get("estimate"))
                se = _finite(c.get("standard_error"))
                pv = _finite(c.get("p_value"))
                est_txt = f"est = {est:.4g}" if est is not None else "est = unavailable"
                se_txt = f"SE = {se:.4g}" if se is not None else "SE = unavailable"
                p_txt = f"p = {pv:.4g}" if pv is not None else "p = unavailable"
                lines.append(f"   {term:<25} {est_txt:<18} {se_txt:<18} {p_txt}")

        completed_same_estimand = 0
        completed_different_estimand = 0
        for scenario in self.scenario_results:
            from .results import _method_label

            method_label = _method_label(scenario.method_id)
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
            scenario_label = f" Scenario '{scenario.name}' ({method_label}):{note_text}"
            lines.append(f"{scenario_label}")
            for est_item in scenario.estimates:
                term = est_item.key
                est = est_item.estimate
                se = est_item.standard_error
                pv = est_item.p_value
                est_txt = f"est = {est:.4g}" if est is not None else "est = unavailable"
                se_txt = f"SE = {se:.4g}" if se is not None else "SE = unavailable"
                p_txt = f"p = {pv:.4g}" if pv is not None else "p = unavailable"
                lines.append(f"   {term:<25} {est_txt:<18} {se_txt:<18} {p_txt}")

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

    def _compare_cronbach(self, lines: list[str], sep: str) -> str:
        base = self.base_result
        primary_label = f" Primary ({base.method_label}):"
        base_alpha = _finite(base.values.get("cronbach_alpha", base.values.get("primary_estimate")))
        base_items = ", ".join(base.values.get("items", []))
        alpha_txt = f"alpha = {base_alpha:.4g}" if base_alpha is not None else "alpha = unavailable"
        lines.append(f"{primary_label:<40} {alpha_txt:<20} items = [{base_items}]")

        completed_same_estimand = 0
        completed_different_estimand = 0
        for scenario in self.scenario_results:
            from .results import _method_label

            method_label = _method_label(scenario.method_id)
            alpha = scenario.primary_estimate
            alpha_txt = f"alpha = {alpha:.4g}" if alpha is not None else "alpha = unavailable"
            spec_items = scenario.specification.specification.question.items or ()
            items_txt = f"items = [{', '.join(spec_items)}]"
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
            lines.append(f"{scenario_label:<40} {alpha_txt:<20} {items_txt}{note_text}")

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
    return estimate_quantity_for_method(method_id)


def _identity(result: AnalysisResult) -> dict[str, Any]:
    specification = result.specification
    if specification is None:
        return {}
    question = specification.question
    return {
        "objective": question.objective.value if question.objective is not None else None,
        "outcome": question.outcome,
        "predictor": question.predictor,
        "items": list(question.items) if question.items is not None else None,
        "design": specification.design.value,
        "estimand": question.estimand,
        "reference_value": question.reference_value,
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
    for field_name in (
        "objective",
        "outcome",
        "predictor",
        "design",
        "reference_value",
    ):
        if left[field_name] != right[field_name]:
            return Comparability.INCOMPATIBLE
    if left["design"] == "paired" and left["unit_id"] != right["unit_id"]:
        return Comparability.INCOMPATIBLE
    if left["estimand"] != right["estimand"]:
        return Comparability.DIFFERENT_ESTIMAND
    if left["design"] == "paired":
        if left["contrast"] != right["contrast"]:
            return Comparability.INCOMPATIBLE

    # Regression models: ordinary least squares or binary logistic regression
    if base.method_id in {"linear_regression", "logistic_regression"} or scenario.method_id in {
        "linear_regression",
        "logistic_regression",
    }:
        if base.method_id != scenario.method_id:
            return Comparability.DIFFERENT_ESTIMAND
        base_spec = base.specification
        scenario_spec = scenario.specification
        if base_spec is None or scenario_spec is None:
            return Comparability.UNAVAILABLE
        base_preds = tuple(base_spec.question.predictors or ())
        scenario_preds = tuple(scenario_spec.question.predictors or ())
        if base_preds != scenario_preds:
            return Comparability.DIFFERENT_ESTIMAND
        base_refs = base_spec.options.reference_levels or {}
        scenario_refs = scenario_spec.options.reference_levels or {}
        if base_refs != scenario_refs:
            return Comparability.DIFFERENT_ESTIMAND
        if base.method_id == "logistic_regression":
            if base_spec.question.event_level != scenario_spec.question.event_level:
                return Comparability.DIFFERENT_ESTIMAND
        return Comparability.SAME_ESTIMAND

    # Multi-item scale reliability
    if base.method_id == "cronbach_alpha" or scenario.method_id == "cronbach_alpha":
        if base.method_id != scenario.method_id:
            return Comparability.DIFFERENT_ESTIMAND
        base_spec = base.specification
        scenario_spec = scenario.specification
        if base_spec is None or scenario_spec is None:
            return Comparability.UNAVAILABLE
        base_items = tuple(base_spec.question.items or ())
        scenario_items = tuple(scenario_spec.question.items or ())
        if base_items != scenario_items:
            return Comparability.DIFFERENT_ESTIMAND
        return Comparability.SAME_ESTIMAND

    mean_omnibus_pair = {base.method_id, scenario.method_id} == {
        "welch_anova",
        "one_way_anova",
    }
    if left["estimate_quantity"] != right["estimate_quantity"] and not mean_omnibus_pair:
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
    return _interval_from_dict(interval)


def _interval_from_dict(interval: Any) -> dict[str, Any] | None:
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
    if {base.method_id, scenario.method_id} == {"welch_anova", "one_way_anova"}:
        base_summaries = base.values.get("group_summaries")
        scenario_summaries = scenario.values.get("group_summaries")
        if not isinstance(base_summaries, list) or not isinstance(scenario_summaries, list):
            return {"available": False, "reason": "Recorded group means are unavailable."}
        base_means = {
            str(item.get("group")): _finite(item.get("mean"))
            for item in base_summaries
            if isinstance(item, dict)
        }
        scenario_means = {
            str(item.get("group")): _finite(item.get("mean"))
            for item in scenario_summaries
            if isinstance(item, dict)
        }
        if base_means.keys() != scenario_means.keys() or any(
            value is None for value in (*base_means.values(), *scenario_means.values())
        ):
            return {"available": False, "reason": "Group mean identities do not match."}
        differences: dict[str, float] = {}
        for group in base_means:
            base_mean = base_means[group]
            scenario_mean = scenario_means[group]
            assert base_mean is not None and scenario_mean is not None
            differences[group] = scenario_mean - base_mean
        return _json_value(
            {
                "available": True,
                "comparison_quantity": "group mean vector",
                "base_group_means": base_means,
                "scenario_group_means": scenario_means,
                "group_mean_differences": differences,
                "all_group_means_equal_within_tolerance": all(
                    math.isclose(value, 0.0, rel_tol=1e-10, abs_tol=1e-12)
                    for value in differences.values()
                ),
                "sample_size_change": (
                    scenario.sample_size - base.sample_size
                    if scenario.sample_size is not None and base.sample_size is not None
                    else None
                ),
            }
        )

    if base.method_id in {"linear_regression", "logistic_regression"}:
        return _compare_same_estimand_regression(base, scenario)

    if base.method_id == "cronbach_alpha":
        return _compare_same_estimand_cronbach(base, scenario)

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


def _compare_same_estimand_regression(
    base: AnalysisResult, scenario: AnalysisResult
) -> dict[str, Any]:
    base_coeffs = base.values.get("coefficients", [])
    scenario_coeffs = scenario.values.get("coefficients", [])
    if not isinstance(base_coeffs, list) or not isinstance(scenario_coeffs, list):
        return {"available": False, "reason": "Recorded regression coefficients are unavailable."}

    base_by_term: dict[str, dict[str, Any]] = {
        str(c["term"]): c
        for c in base_coeffs
        if isinstance(c, dict) and isinstance(c.get("term"), str)
    }
    scenario_by_term: dict[str, dict[str, Any]] = {
        str(c["term"]): c
        for c in scenario_coeffs
        if isinstance(c, dict) and isinstance(c.get("term"), str)
    }

    alpha = base.specification.options.alpha if base.specification is not None else 0.05

    all_term_keys: list[str] = list(base_by_term.keys())
    for k in scenario_by_term:
        if k not in all_term_keys:
            all_term_keys.append(k)

    matched_keys: list[str] = []
    unmatched_base: list[str] = []
    unmatched_scenario: list[str] = []
    terms_comparison: dict[str, Any] = {}

    for term_key in all_term_keys:
        b_term = base_by_term.get(term_key)
        s_term = scenario_by_term.get(term_key)
        if b_term is None:
            unmatched_scenario.append(term_key)
            s_est = _finite(s_term.get("estimate")) if s_term is not None else None
            terms_comparison[term_key] = {
                "term": term_key,
                "comparison_available": False,
                "status": "unmatched_in_base",
                "reason": "Term is present in scenario model but absent from base model.",
                "scenario_estimate": s_est,
            }
        elif s_term is None:
            unmatched_base.append(term_key)
            b_est = _finite(b_term.get("estimate")) if b_term is not None else None
            terms_comparison[term_key] = {
                "term": term_key,
                "comparison_available": False,
                "status": "unmatched_in_scenario",
                "reason": "Term is present in base model but absent from scenario model.",
                "base_estimate": b_est,
            }
        else:
            matched_keys.append(term_key)
            b_est = _finite(b_term.get("estimate"))
            s_est = _finite(s_term.get("estimate"))
            b_se = _finite(b_term.get("standard_error"))
            s_se = _finite(s_term.get("standard_error"))
            b_p = _finite(b_term.get("p_value"))
            s_p = _finite(s_term.get("p_value"))
            b_ci = _interval_from_dict(b_term.get("confidence_interval"))
            s_ci = _interval_from_dict(s_term.get("confidence_interval"))

            est_diff = (s_est - b_est) if (s_est is not None and b_est is not None) else None
            se_diff = (s_se - b_se) if (s_se is not None and b_se is not None) else None
            sign_consistent = (
                (_direction(b_est) == _direction(s_est))
                if (b_est is not None and s_est is not None)
                else None
            )

            ci_overlap = None
            same_ci_level = None
            if b_ci is not None and s_ci is not None:
                ci_overlap = max(float(b_ci["lower"]), float(s_ci["lower"])) <= min(
                    float(b_ci["upper"]), float(s_ci["upper"])
                )
                same_ci_level = math.isclose(
                    float(b_ci["level"]), float(s_ci["level"]), abs_tol=1e-12
                )

            b_dec = _hypothesis_decision(b_p, alpha)
            s_dec = _hypothesis_decision(s_p, alpha)
            dec_consistent = (b_dec == s_dec) if (b_dec is not None and s_dec is not None) else None

            terms_comparison[term_key] = {
                "term": term_key,
                "comparison_available": True,
                "base_estimate": b_est,
                "scenario_estimate": s_est,
                "estimate_difference": est_diff,
                "sign_consistent": sign_consistent,
                "base_standard_error": b_se,
                "scenario_standard_error": s_se,
                "standard_error_change": se_diff,
                "base_confidence_interval": b_ci,
                "scenario_confidence_interval": s_ci,
                "interval_overlap": ci_overlap,
                "same_confidence_level": same_ci_level,
                "base_p_value": b_p,
                "scenario_p_value": s_p,
                "base_decision": b_dec,
                "scenario_decision": s_dec,
                "decision_consistent": dec_consistent,
            }

    sample_size_change = (
        scenario.sample_size - base.sample_size
        if scenario.sample_size is not None and base.sample_size is not None
        else None
    )

    all_sign_consistent = (
        all(bool(terms_comparison[k].get("sign_consistent")) for k in matched_keys)
        if matched_keys
        else None
    )
    all_dec_consistent = (
        all(bool(terms_comparison[k].get("decision_consistent")) for k in matched_keys)
        if matched_keys
        else None
    )
    intervals_present = [
        terms_comparison[k].get("interval_overlap")
        for k in matched_keys
        if terms_comparison[k].get("interval_overlap") is not None
    ]
    overlap_summary = all(bool(x) for x in intervals_present) if intervals_present else None

    return _json_value(
        {
            "available": True,
            "comparison_quantity": "coefficient_vector",
            "terms": terms_comparison,
            "matched_terms": matched_keys,
            "unmatched_base_terms": unmatched_base,
            "unmatched_scenario_terms": unmatched_scenario,
            "direction_consistent": all_sign_consistent,
            "all_matched_terms_sign_consistent": all_sign_consistent,
            "all_matched_terms_decision_consistent": all_dec_consistent,
            "interval_overlap": overlap_summary,
            "sample_size_change": sample_size_change,
        }
    )


def _compare_same_estimand_cronbach(
    base: AnalysisResult, scenario: AnalysisResult
) -> dict[str, Any]:
    base_est = _finite(base.values.get("cronbach_alpha", base.values.get("primary_estimate")))
    scenario_est = _finite(
        scenario.values.get("cronbach_alpha", scenario.values.get("primary_estimate"))
    )
    if base_est is None or scenario_est is None:
        return {"available": False, "reason": "A finite Cronbach alpha estimate is unavailable."}

    diff = scenario_est - base_est
    base_interval = _interval(base)
    scenario_interval = _interval(scenario)
    overlap = None
    same_level = None
    if base_interval is not None and scenario_interval is not None:
        overlap = max(float(base_interval["lower"]), float(scenario_interval["lower"])) <= min(
            float(base_interval["upper"]), float(scenario_interval["upper"])
        )
        same_level = math.isclose(
            float(base_interval["level"]), float(scenario_interval["level"]), abs_tol=1e-12
        )

    base_scoring = base.values.get("scoring", {})
    scenario_scoring = scenario.values.get("scoring", {})
    scoring_changed = base_scoring != scenario_scoring

    return _json_value(
        {
            "available": True,
            "comparison_quantity": "cronbach_alpha",
            "base_estimate": base_est,
            "scenario_estimate_raw": scenario_est,
            "scenario_estimate_normalized": scenario_est,
            "estimate_difference_from_base": diff,
            "direction_consistent": _direction(base_est) == _direction(scenario_est),
            "base_confidence_interval": base_interval,
            "scenario_confidence_interval_normalized": scenario_interval,
            "interval_overlap": overlap,
            "same_confidence_level": same_level,
            "interval_overlap_is_descriptive": overlap is not None,
            "reverse_scoring_change": scoring_changed,
            "scoring_provenance": {
                "base_scoring": base_scoring,
                "scenario_scoring": scenario_scoring,
            },
            "sample_size_change": (
                scenario.sample_size - base.sample_size
                if scenario.sample_size is not None and base.sample_size is not None
                else None
            ),
        }
    )


def scenario_values(result: AnalysisResult) -> dict[str, Any]:
    """Extract validated, aggregate display values from one analysis result."""
    if result.method_id in {"linear_regression", "logistic_regression"}:
        return _scenario_values_regression(result)
    if result.method_id == "cronbach_alpha":
        return _scenario_values_cronbach(result)

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
        "estimates": (),
    }


def _scenario_values_regression(result: AnalysisResult) -> dict[str, Any]:
    coeffs = result.values.get("coefficients", [])
    estimates: list[SensitivityEstimate] = []
    qty = (
        "regression_coefficient"
        if result.method_id == "linear_regression"
        else "log_odds_coefficient"
    )
    if isinstance(coeffs, list):
        for term in coeffs:
            if isinstance(term, dict) and "term" in term:
                key = str(term["term"])
                est = _finite(term.get("estimate"))
                se = _finite(term.get("standard_error"))
                pv = _finite(term.get("p_value"))
                ci = _interval_from_dict(term.get("confidence_interval"))
                meta = {
                    k: v
                    for k, v in term.items()
                    if k
                    not in {"term", "estimate", "standard_error", "p_value", "confidence_interval"}
                }
                estimates.append(
                    SensitivityEstimate(
                        key=key,
                        quantity=qty,
                        estimate=est,
                        standard_error=se,
                        confidence_interval=ci,
                        p_value=pv,
                        metadata=meta,
                    )
                )
    return {
        "estimate_quantity": "coefficient_vector",
        "primary_estimate": None,
        "effect_size_quantity": None,
        "effect_size": None,
        "confidence_interval": None,
        "p_value": None,
        "sample_size": result.sample_size,
        "excluded_rows": result.excluded_rows,
        "group_contrast": None,
        "estimates": tuple(estimates),
    }


def _scenario_values_cronbach(result: AnalysisResult) -> dict[str, Any]:
    alpha = _finite(result.values.get("cronbach_alpha", result.values.get("primary_estimate")))
    return {
        "estimate_quantity": "cronbach_alpha",
        "primary_estimate": alpha,
        "effect_size_quantity": None,
        "effect_size": None,
        "confidence_interval": _interval(result),
        "p_value": None,
        "sample_size": result.sample_size,
        "excluded_rows": result.excluded_rows,
        "group_contrast": None,
        "estimates": (),
    }
