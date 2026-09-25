"""Pre-analysis plans and transparent comparison with later execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .exceptions import InvalidDataError
from .practical_significance import MeaningfulEffectThreshold
from .results import AnalysisResult
from .sensitivity import SensitivitySpecification, estimate_quantity
from .specifications import AnalysisSpecification, _json_value

ANALYSIS_PLAN_SCHEMA_VERSION = 1


class AnalysisPlanStatus(str, Enum):
    READY = "ready"
    NEEDS_INPUT = "needs_input"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class StatisticalAnalysisPlan:
    specification: AnalysisSpecification
    status: AnalysisPlanStatus
    primary_method_id: str | None
    method_rationale: str | None
    effect_quantity: str | None
    confidence_interval_quantity: str | None
    missing_data_policy: str
    exclusion_rule: str
    outlier_rule: str
    sensitivity_scenarios: tuple[SensitivitySpecification, ...] = ()
    meaningful_threshold: MeaningfulEffectThreshold | None = None
    multiplicity_policy: str = "not_applicable"
    multiplicity_method: str | None = None
    report_style: str = "general"
    audit_policy: str = "enabled"
    reproducibility_settings: dict[str, Any] | None = None
    planning_status: str = "unknown"
    created_after_analysis: bool = False
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.specification, AnalysisSpecification):
            raise InvalidDataError("plan specification must be an AnalysisSpecification.")
        try:
            object.__setattr__(self, "status", AnalysisPlanStatus(self.status))
        except (TypeError, ValueError) as exc:
            raise InvalidDataError("analysis plan status is invalid.") from exc
        if self.primary_method_id is not None and (
            not isinstance(self.primary_method_id, str) or not self.primary_method_id.strip()
        ):
            raise InvalidDataError("primary_method_id must be non-empty text or None.")
        if self.status is AnalysisPlanStatus.READY and self.primary_method_id is None:
            raise InvalidDataError("A ready analysis plan requires a primary_method_id.")
        for name in (
            "method_rationale",
            "effect_quantity",
            "confidence_interval_quantity",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise InvalidDataError(f"{name} must be non-empty text or None.")
        for name in ("missing_data_policy", "exclusion_rule", "outlier_rule"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise InvalidDataError(f"{name} must be non-empty text.")
        if self.multiplicity_policy not in {
            "not_applicable",
            "none_planned",
            "planned_method",
            "unknown",
        }:
            raise InvalidDataError("multiplicity_policy is invalid.")
        if self.multiplicity_method is not None and (
            not isinstance(self.multiplicity_method, str) or not self.multiplicity_method.strip()
        ):
            raise InvalidDataError("multiplicity_method must be non-empty text or None.")
        if self.multiplicity_policy == "planned_method" and self.multiplicity_method is None:
            raise InvalidDataError(
                "multiplicity_policy='planned_method' requires multiplicity_method."
            )
        if self.report_style not in {"general", "apa", "ieee"}:
            raise InvalidDataError("report_style must be general, apa, or ieee.")
        if self.audit_policy not in {"enabled", "disabled"}:
            raise InvalidDataError("audit_policy must be enabled or disabled.")
        if self.planning_status not in {"planned", "exploratory", "unknown"}:
            raise InvalidDataError("planning_status must be planned, exploratory, or unknown.")
        if any(
            not isinstance(item, SensitivitySpecification) for item in self.sensitivity_scenarios
        ):
            raise InvalidDataError("sensitivity_scenarios contains an invalid specification.")
        if self.meaningful_threshold is not None and not isinstance(
            self.meaningful_threshold, MeaningfulEffectThreshold
        ):
            raise InvalidDataError("meaningful_threshold has an invalid type.")
        if not isinstance(self.created_after_analysis, bool):
            raise InvalidDataError("created_after_analysis must be a Boolean.")
        if self.reproducibility_settings is not None and not isinstance(
            self.reproducibility_settings, dict
        ):
            raise InvalidDataError("reproducibility_settings must be a mapping or None.")
        if any(not isinstance(item, str) or not item.strip() for item in self.warnings):
            raise InvalidDataError("plan warnings must contain non-empty text.")
        if any(not isinstance(item, str) or not item.strip() for item in self.limitations):
            raise InvalidDataError("plan limitations must contain non-empty text.")
        self.to_dict()

    def to_specification(self) -> AnalysisSpecification:
        return self.specification

    def to_dict(self) -> dict[str, Any]:
        question = self.specification.question
        return _json_value(
            {
                "schema_version": ANALYSIS_PLAN_SCHEMA_VERSION,
                "status": self.status.value,
                "objective": question.objective.value if question.objective else None,
                "research_description": question.description,
                "outcome": question.outcome,
                "predictor": question.predictor,
                "unit_id": self.specification.unit_id,
                "estimand": question.estimand,
                "study_design": self.specification.design.value,
                "primary_method_id": self.primary_method_id,
                "method_rationale": self.method_rationale,
                "alpha": self.specification.options.alpha,
                "confidence_level": self.specification.options.confidence_level,
                "alternative_hypothesis": _alternative_hypothesis(self.primary_method_id),
                "effect_quantity": self.effect_quantity,
                "confidence_interval_quantity": self.confidence_interval_quantity,
                "missing_data_policy": self.missing_data_policy,
                "exclusion_rule": self.exclusion_rule,
                "outlier_rule": self.outlier_rule,
                "variable_declarations": self.specification.data_dictionary,
                "planned_sensitivity_scenarios": [
                    item.to_dict() for item in self.sensitivity_scenarios
                ],
                "meaningful_threshold": self.meaningful_threshold.to_dict()
                if self.meaningful_threshold
                else None,
                "multiplicity_policy": self.multiplicity_policy,
                "multiplicity_method": self.multiplicity_method,
                "planned_report_style": self.report_style,
                "audit_policy": self.audit_policy,
                "reproducibility_settings": self.reproducibility_settings
                or {"fingerprint": True, "automatic_replay": False},
                "researcher_planning_declaration": self.planning_status,
                "created_after_analysis": self.created_after_analysis,
                "warnings": self.warnings,
                "limitations": self.limitations,
                "provenance": self.provenance or {},
                "specification": self.specification.to_dict(),
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StatisticalAnalysisPlan:
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise InvalidDataError("StatisticalAnalysisPlan schema_version must be 1.")
        try:
            scenarios = tuple(
                SensitivitySpecification.from_dict(item)
                for item in payload.get("planned_sensitivity_scenarios", ())
            )
            threshold_payload = payload.get("meaningful_threshold")
            threshold = (
                MeaningfulEffectThreshold.from_dict(threshold_payload)
                if threshold_payload is not None
                else None
            )
            return cls(
                specification=AnalysisSpecification.from_dict(payload["specification"]),
                status=payload["status"],
                primary_method_id=payload.get("primary_method_id"),
                method_rationale=payload.get("method_rationale"),
                effect_quantity=payload.get("effect_quantity"),
                confidence_interval_quantity=payload.get("confidence_interval_quantity"),
                missing_data_policy=payload["missing_data_policy"],
                exclusion_rule=payload["exclusion_rule"],
                outlier_rule=payload["outlier_rule"],
                sensitivity_scenarios=scenarios,
                meaningful_threshold=threshold,
                multiplicity_policy=payload.get("multiplicity_policy", "not_applicable"),
                multiplicity_method=payload.get("multiplicity_method"),
                report_style=payload.get("planned_report_style", "general"),
                audit_policy=payload.get("audit_policy", "enabled"),
                reproducibility_settings=payload.get("reproducibility_settings"),
                planning_status=payload.get("researcher_planning_declaration", "unknown"),
                created_after_analysis=payload.get("created_after_analysis", False),
                warnings=tuple(payload.get("warnings", ())),
                limitations=tuple(payload.get("limitations", ())),
                provenance=payload.get("provenance"),
            )
        except (KeyError, TypeError) as exc:
            raise InvalidDataError("Analysis plan fields are missing or invalid.") from exc


@dataclass(frozen=True)
class PlanAdherenceResult:
    status: str
    comparisons: tuple[dict[str, Any], ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"matched", "changed"}:
            raise InvalidDataError("plan-adherence status must be matched or changed.")
        if self.reason is not None and (
            not isinstance(self.reason, str) or not self.reason.strip()
        ):
            raise InvalidDataError("A supplied adherence reason must be non-empty text.")
        self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": 1,
                "status": self.status,
                "comparisons": self.comparisons,
                "researcher_reason": self.reason,
                "misconduct_inference": False,
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)


def compare_plan_to_result(
    plan: StatisticalAnalysisPlan, result: AnalysisResult, *, reason: str | None = None
) -> PlanAdherenceResult:
    if not isinstance(plan, StatisticalAnalysisPlan) or not isinstance(result, AnalysisResult):
        raise InvalidDataError("plan adherence requires a plan and AnalysisResult.")
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise InvalidDataError("A supplied adherence reason must be non-empty text.")
    executed = result.specification
    fields = {
        "objective": (
            plan.specification.question.objective.value
            if plan.specification.question.objective
            else None,
            executed.question.objective.value
            if executed is not None and executed.question.objective
            else None,
        ),
        "outcome": (
            plan.specification.question.outcome,
            executed.question.outcome if executed else None,
        ),
        "predictor": (
            plan.specification.question.predictor,
            executed.question.predictor if executed else None,
        ),
        "estimand": (
            plan.specification.question.estimand,
            executed.question.estimand if executed else None,
        ),
        "design": (
            plan.specification.design.value,
            executed.design.value if executed else None,
        ),
        "unit_id": (plan.specification.unit_id, executed.unit_id if executed else None),
        "condition_order": (
            list(plan.specification.condition_order)
            if plan.specification.condition_order is not None
            else None,
            list(executed.condition_order)
            if executed is not None and executed.condition_order is not None
            else None,
        ),
        "primary_method_id": (plan.primary_method_id, result.method_id),
        "alpha": (
            plan.specification.options.alpha,
            executed.options.alpha if executed else None,
        ),
        "confidence_level": (
            plan.specification.options.confidence_level,
            executed.options.confidence_level if executed else None,
        ),
    }
    comparisons = tuple(
        {
            "field": field,
            "status": "not_recorded"
            if planned is None
            else "matched"
            if planned == actual
            else "changed",
            "planned": planned,
            "executed": actual,
        }
        for field, (planned, actual) in fields.items()
    )
    status = "changed" if any(item["status"] == "changed" for item in comparisons) else "matched"
    return PlanAdherenceResult(status, comparisons, reason)


def planned_quantity(method_id: str | None) -> str | None:
    return estimate_quantity(method_id) if method_id is not None else None


def planned_interval_quantity(method_id: str | None) -> str | None:
    if method_id in {"welch_t", "student_t", "paired_t"}:
        return "mean_difference"
    return None


def _alternative_hypothesis(method_id: str | None) -> str | None:
    if method_id in {"welch_t", "student_t", "paired_t", "pearson_correlation"}:
        return "two-sided"
    if method_id in {"one_way_anova", "kruskal_wallis"}:
        return "at least one group differs"
    if method_id == "pearson_chi_square":
        return "variables are associated"
    return None
