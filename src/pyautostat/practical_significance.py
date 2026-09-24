"""Researcher-defined effect thresholds, kept separate from null-hypothesis tests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .exceptions import InvalidDataError
from .results import AnalysisResult, AnalysisStatus
from .sensitivity import estimate_quantity
from .specifications import _json_value

PRACTICAL_SIGNIFICANCE_SCHEMA_VERSION = 1
SIGNED_QUANTITIES = {"mean_difference", "cohens_d", "pearson_r", "rank_biserial"}
NONNEGATIVE_QUANTITIES = {"cramers_v", "eta_squared", "epsilon_squared"}
SUPPORTED_QUANTITIES = SIGNED_QUANTITIES | NONNEGATIVE_QUANTITIES
_DIRECTIONS = {"two_sided", "positive", "negative", "nonnegative"}
_FORMAL_REQUESTS = {"equivalence", "noninferiority"}


@dataclass(frozen=True)
class MeaningfulEffectThreshold:
    """A threshold supplied by the researcher, never inferred from a p-value."""

    quantity: str
    minimum_magnitude: float
    direction: str = "two_sided"
    unit: str | None = None
    rationale: str | None = None
    planning_status: str = "unknown"

    def __post_init__(self) -> None:
        if self.quantity not in SUPPORTED_QUANTITIES:
            raise InvalidDataError(
                "threshold quantity must be one of: " + ", ".join(sorted(SUPPORTED_QUANTITIES))
            )
        if (
            isinstance(self.minimum_magnitude, bool)
            or not isinstance(self.minimum_magnitude, (int, float))
            or not math.isfinite(float(self.minimum_magnitude))
            or self.minimum_magnitude < 0
        ):
            raise InvalidDataError("minimum_magnitude must be a finite nonnegative number.")
        object.__setattr__(self, "minimum_magnitude", float(self.minimum_magnitude))
        if self.direction not in _DIRECTIONS | _FORMAL_REQUESTS:
            raise InvalidDataError(
                "direction must be two_sided, positive, negative, nonnegative, "
                "equivalence, or noninferiority."
            )
        if self.quantity in NONNEGATIVE_QUANTITIES and self.direction not in {
            "nonnegative",
            *_FORMAL_REQUESTS,
        }:
            raise InvalidDataError(
                "A nonnegative effect quantity requires direction='nonnegative'."
            )
        if self.quantity in SIGNED_QUANTITIES and self.direction == "nonnegative":
            raise InvalidDataError("A signed effect quantity cannot use direction='nonnegative'.")
        for name in ("unit", "rationale"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise InvalidDataError(f"{name} must be non-empty text when supplied.")
        if self.quantity != "mean_difference" and self.unit is not None:
            raise InvalidDataError(
                "Standardized and correlation effect thresholds must not declare raw units."
            )
        if self.planning_status not in {"planned", "exploratory", "unknown"}:
            raise InvalidDataError("planning_status must be planned, exploratory, or unknown.")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": PRACTICAL_SIGNIFICANCE_SCHEMA_VERSION,
                "quantity": self.quantity,
                "minimum_magnitude": self.minimum_magnitude,
                "direction": self.direction,
                "unit": self.unit,
                "rationale": self.rationale,
                "planning_status": self.planning_status,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MeaningfulEffectThreshold:
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise InvalidDataError("Meaningful-effect threshold schema_version must be 1.")
        try:
            return cls(
                quantity=payload["quantity"],
                minimum_magnitude=payload["minimum_magnitude"],
                direction=payload.get("direction", "two_sided"),
                unit=payload.get("unit"),
                rationale=payload.get("rationale"),
                planning_status=payload.get("planning_status", "unknown"),
            )
        except (KeyError, TypeError) as exc:
            raise InvalidDataError(
                "Meaningful-effect threshold fields are missing or invalid."
            ) from exc


@dataclass(frozen=True)
class PracticalSignificanceResult:
    status: str
    quantity: str
    estimate: float | None
    threshold: MeaningfulEffectThreshold
    confidence_interval: dict[str, Any] | None
    point_estimate_relation: str
    confidence_interval_relation: str
    uncertainty_status: str
    statistical_significance: str
    conclusion: str
    warnings: tuple[str, ...]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.status not in {"complete", "partial", "unsupported", "unavailable"}:
            raise InvalidDataError("Practical-significance status is invalid.")
        if not isinstance(self.threshold, MeaningfulEffectThreshold):
            raise InvalidDataError("threshold must be a MeaningfulEffectThreshold.")
        if self.quantity != self.threshold.quantity:
            raise InvalidDataError("result quantity must match the threshold quantity.")
        if self.estimate is not None and (
            isinstance(self.estimate, bool)
            or not isinstance(self.estimate, (int, float))
            or not math.isfinite(float(self.estimate))
        ):
            raise InvalidDataError("Practical-significance estimate must be finite or None.")
        if any(not isinstance(item, str) or not item.strip() for item in self.warnings):
            raise InvalidDataError("Practical-significance warnings must contain non-empty text.")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": PRACTICAL_SIGNIFICANCE_SCHEMA_VERSION,
                "status": self.status,
                "quantity": self.quantity,
                "estimate": self.estimate,
                "threshold": self.threshold.to_dict(),
                "confidence_interval": self.confidence_interval,
                "point_estimate_relation": self.point_estimate_relation,
                "confidence_interval_relation": self.confidence_interval_relation,
                "uncertainty_status": self.uncertainty_status,
                "statistical_significance": self.statistical_significance,
                "conclusion": self.conclusion,
                "warnings": self.warnings,
                "provenance": self.provenance,
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidDataError(f"The analysis has no finite {name}.")
    output = float(value)
    if not math.isfinite(output):
        raise InvalidDataError(f"The analysis has no finite {name}.")
    return output


def _metric(
    result: AnalysisResult, quantity: str
) -> tuple[float, dict[str, Any] | None, str | None]:
    primary_quantity = estimate_quantity(result.method_id)
    if quantity == primary_quantity:
        estimate = _finite(result.values.get("primary_estimate"), quantity)
        interval = result.values.get("confidence_interval")
        unit = result.values.get("estimate_unit")
    else:
        effect = result.values.get("effect_size")
        if not isinstance(effect, dict):
            raise InvalidDataError(f"The analysis does not provide quantity {quantity!r}.")
        aliases = {
            "Cohen's d": "cohens_d",
            "rank-biserial correlation": "rank_biserial",
            "eta-squared": "eta_squared",
            "epsilon-squared": "epsilon_squared",
            "Cramer's V": "cramers_v",
            "Pearson r": "pearson_r",
        }
        observed = aliases.get(str(effect.get("name")))
        if observed != quantity:
            raise InvalidDataError(
                f"Threshold quantity {quantity!r} does not match an available result quantity."
            )
        estimate = _finite(effect.get("value"), quantity)
        interval = effect.get("confidence_interval")
        unit = None
    checked_interval = None
    if interval is not None:
        if not isinstance(interval, dict):
            raise InvalidDataError("The selected quantity has an invalid confidence interval.")
        lower = _finite(interval.get("lower"), "confidence interval lower bound")
        upper = _finite(interval.get("upper"), "confidence interval upper bound")
        level = _finite(interval.get("level"), "confidence level")
        if lower > upper or not 0 < level < 1:
            raise InvalidDataError("The selected quantity has an invalid confidence interval.")
        interval_aliases = {
            "mean difference": "mean_difference",
            "Cohen's d": "cohens_d",
            "rank-biserial correlation": "rank_biserial",
            "eta-squared": "eta_squared",
            "epsilon-squared": "epsilon_squared",
            "Cramer's V": "cramers_v",
            "Pearson r": "pearson_r",
        }
        if interval_aliases.get(str(interval.get("quantity"))) != quantity:
            raise InvalidDataError(
                "The selected quantity and confidence-interval quantity do not match."
            )
        if not isinstance(interval.get("method"), str) or not interval["method"].strip():
            raise InvalidDataError("The selected quantity has an invalid interval method.")
        checked_interval = _json_value(interval)
    if quantity in NONNEGATIVE_QUANTITIES and estimate < 0:
        raise InvalidDataError("A nonnegative effect quantity has an invalid negative estimate.")
    if (
        checked_interval is not None
        and quantity in NONNEGATIVE_QUANTITIES
        and float(checked_interval["lower"]) < 0
    ):
        raise InvalidDataError("A nonnegative effect quantity has an invalid negative interval.")
    if quantity in {"pearson_r", "rank_biserial"} and abs(estimate) > 1:
        raise InvalidDataError("A correlation quantity is outside its valid range.")
    if checked_interval is not None and quantity in {"pearson_r", "rank_biserial"}:
        if float(checked_interval["lower"]) < -1 or float(checked_interval["upper"]) > 1:
            raise InvalidDataError("A correlation interval is outside its valid range.")
    return estimate, checked_interval, unit if isinstance(unit, str) else None


def _point_relation(value: float, threshold: MeaningfulEffectThreshold) -> str:
    delta = threshold.minimum_magnitude
    if threshold.direction == "two_sided":
        if value >= delta:
            return "positive_meaningful_region"
        if value <= -delta:
            return "negative_meaningful_region"
        return "below_meaningful_magnitude"
    if threshold.direction == "positive":
        return "meets_positive_threshold" if value >= delta else "below_positive_threshold"
    if threshold.direction == "negative":
        return "meets_negative_threshold" if value <= -delta else "above_negative_threshold"
    return "meets_nonnegative_threshold" if value >= delta else "below_nonnegative_threshold"


def _interval_relation(lower: float, upper: float, threshold: MeaningfulEffectThreshold) -> str:
    delta = threshold.minimum_magnitude
    if threshold.direction == "two_sided":
        if lower >= delta:
            return "entirely_positive_meaningful"
        if upper <= -delta:
            return "entirely_negative_meaningful"
        if lower > -delta and upper < delta:
            return "entirely_within_negligible_region"
        if lower <= -delta and upper >= delta:
            return "spans_both_directions"
        return "crosses_meaningful_boundary"
    if threshold.direction == "positive":
        if lower >= delta:
            return "entirely_above_positive_threshold"
        if upper < delta:
            return "entirely_below_positive_threshold"
        return "crosses_positive_threshold"
    if threshold.direction == "negative":
        if upper <= -delta:
            return "entirely_below_negative_threshold"
        if lower > -delta:
            return "entirely_above_negative_threshold"
        return "crosses_negative_threshold"
    if lower >= delta:
        return "entirely_at_or_above_nonnegative_threshold"
    if upper < delta:
        return "entirely_below_nonnegative_threshold"
    return "crosses_nonnegative_threshold"


def assess_practical_significance(
    result: AnalysisResult,
    threshold: MeaningfulEffectThreshold,
    *,
    planning_status: str,
) -> PracticalSignificanceResult:
    """Compare one recorded estimate and its existing interval with a supplied threshold."""
    if not isinstance(result, AnalysisResult):
        raise InvalidDataError("practical_significance requires an AnalysisResult.")
    if not isinstance(threshold, MeaningfulEffectThreshold):
        raise InvalidDataError("threshold must be a MeaningfulEffectThreshold.")
    if result.status is not AnalysisStatus.AVAILABLE:
        return PracticalSignificanceResult(
            "unavailable",
            threshold.quantity,
            None,
            threshold,
            None,
            "unavailable",
            "unavailable",
            "unavailable",
            "unavailable",
            "The base analysis is unavailable, so the threshold cannot be assessed.",
            tuple(result.warnings),
            {"planning_status": planning_status, "local_record_only": True},
        )
    if threshold.direction in _FORMAL_REQUESTS:
        label = "equivalence" if threshold.direction == "equivalence" else "noninferiority"
        return PracticalSignificanceResult(
            "unsupported",
            threshold.quantity,
            None,
            threshold,
            None,
            "unavailable",
            "unavailable",
            "unsupported",
            "unavailable",
            f"Formal {label} inference is not currently implemented.",
            ("An ordinary two-sided analysis is not a validated formal " + label + " test.",),
            {"planning_status": planning_status, "local_record_only": True},
        )
    estimate, interval, result_unit = _metric(result, threshold.quantity)
    if threshold.unit is not None and result_unit is not None and threshold.unit != result_unit:
        raise InvalidDataError(
            f"Threshold unit {threshold.unit!r} does not match result unit {result_unit!r}."
        )
    warnings: list[str] = []
    if threshold.unit is not None and result_unit is None:
        warnings.append("The supplied threshold unit could not be independently verified.")
    point = _point_relation(estimate, threshold)
    if interval is None:
        interval_relation = "unavailable"
        status = "partial"
        uncertainty = "confidence_interval_unavailable"
        conclusion = (
            "The point estimate was compared with the researcher-defined threshold, but "
            "uncertainty relative to that threshold cannot be assessed because a confidence "
            "interval is unavailable."
        )
        warnings.append("No confidence interval is available for the selected quantity.")
    else:
        interval_relation = _interval_relation(
            float(interval["lower"]), float(interval["upper"]), threshold
        )
        status = "complete"
        uncertainty = "confidence_interval_assessed"
        conclusion = (
            "The point estimate and reported confidence interval were compared descriptively "
            "with the researcher-defined threshold. This is not a formal equivalence or "
            "noninferiority test."
        )
    p_value = result.values.get("p_value")
    alpha = result.specification.options.alpha if result.specification is not None else 0.05
    statistical = (
        "evidence_against_null"
        if isinstance(p_value, (int, float)) and math.isfinite(float(p_value)) and p_value < alpha
        else "no_evidence_against_null"
        if isinstance(p_value, (int, float)) and math.isfinite(float(p_value))
        else "unavailable"
    )
    return PracticalSignificanceResult(
        status,
        threshold.quantity,
        estimate,
        threshold,
        interval,
        point,
        interval_relation,
        uncertainty,
        statistical,
        conclusion,
        tuple(warnings),
        {
            "planning_status": planning_status,
            "researcher_supplied_rationale": threshold.rationale,
            "local_record_only": True,
            "external_preregistration_verified": False,
        },
    )
