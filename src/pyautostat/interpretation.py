"""Deterministic explanations of completed Phase 6 analysis records.

This module reads recorded results. It never calls a statistical backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .exceptions import InvalidDataError
from .results import AnalysisResult, AnalysisStatus
from .specifications import SCHEMA_VERSION, _json_value


class InterpretationStatus(str, Enum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class InterpretationFinding:
    """A stable decision code with the source fields behind its explanation."""

    code: str
    message: str
    supporting_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise InvalidDataError("Interpretation finding code and message are required.")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "code": self.code,
                "message": self.message,
                "supporting_fields": self.supporting_fields,
            }
        )


@dataclass(frozen=True)
class InterpretationResult:
    """Serializable narrative and coded findings tied to one execution result."""

    status: InterpretationStatus
    method_id: str
    execution_status: AnalysisStatus
    summary: str
    method_explanation: str | None = None
    hypothesis_interpretation: str | None = None
    effect_interpretation: str | None = None
    uncertainty_interpretation: str | None = None
    assumption_notes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    conclusion: str | None = None
    findings: tuple[InterpretationFinding, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "status", InterpretationStatus(self.status))
            object.__setattr__(self, "execution_status", AnalysisStatus(self.execution_status))
        except ValueError as exc:
            raise InvalidDataError("Invalid interpretation or execution status.") from exc
        if not self.method_id or not self.summary:
            raise InvalidDataError("Interpretation method_id and summary are required.")
        if any(not isinstance(item, InterpretationFinding) for item in self.findings):
            raise InvalidDataError("findings must contain InterpretationFinding records.")
        self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": SCHEMA_VERSION,
                "status": self.status.value,
                "method_id": self.method_id,
                "execution_status": self.execution_status.value,
                "summary": self.summary,
                "method_explanation": self.method_explanation,
                "hypothesis_interpretation": self.hypothesis_interpretation,
                "effect_interpretation": self.effect_interpretation,
                "uncertainty_interpretation": self.uncertainty_interpretation,
                "assumption_notes": self.assumption_notes,
                "limitations": self.limitations,
                "conclusion": self.conclusion,
                "findings": [item.to_dict() for item in self.findings],
                "warnings": self.warnings,
                "metadata": self.metadata,
            }
        )


_METHODS = {
    "dataset_profile": ("Dataset profile", None),
    "welch_t": ("Welch's independent-samples t-test", "Cohen's d"),
    "student_t": ("Student's pooled-variance t-test", "Cohen's d"),
    "paired_t": ("Paired-samples t-test", "Cohen's dz"),
    "mann_whitney_u": ("Mann-Whitney U test", "rank-biserial correlation"),
    "one_way_anova": ("One-way ANOVA", "eta-squared"),
    "kruskal_wallis": ("Kruskal-Wallis test", "epsilon-squared (rank)"),
    "pearson_correlation": ("Pearson correlation", "Pearson r"),
    "pearson_chi_square": ("Pearson chi-square independence test", "Cramer's V"),
}
_SIGNED_GROUP = {"welch_t", "student_t", "paired_t", "mann_whitney_u"}
_GROUP = _SIGNED_GROUP | {"one_way_anova", "kruskal_wallis", "pearson_chi_square"}
_NONNEGATIVE = {"one_way_anova", "kruskal_wallis", "pearson_chi_square"}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _p_display(value: float) -> str:
    if value == 0:
        return "p < 0.001 (computational zero)"
    return f"p = {_fmt(value)}"


def _finding(findings: list[InterpretationFinding], code: str, message: str, *fields: str) -> None:
    findings.append(InterpretationFinding(code, message, fields))


def _unavailable(
    result: AnalysisResult, reason: str, warnings: list[str] | None = None
) -> InterpretationResult:
    warning_list = list(result.warnings)
    warning_list.extend(warnings or ())
    return InterpretationResult(
        status=InterpretationStatus.UNAVAILABLE,
        method_id=result.method_id,
        execution_status=result.status,
        summary=f"Interpretation unavailable: {reason}",
        findings=(InterpretationFinding("interpretation_unavailable", reason),),
        warnings=tuple(dict.fromkeys(warning_list)),
        limitations=(reason,),
        metadata={"sample_size": result.sample_size, "excluded_rows": result.excluded_rows},
    )


def _context(result: AnalysisResult) -> tuple[str, str | None]:
    """Return the method description and the verified group contrast, if any."""
    method = result.method_id
    assert result.specification is not None
    question = result.specification.question
    if method in _GROUP:
        if not question.outcome or not question.predictor:
            raise InvalidDataError("The group comparison lacks outcome or group variables.")
        order = result.metadata.get("group_order")
        required = (
            2
            if method in _SIGNED_GROUP
            else 3
            if method
            in (
                "one_way_anova",
                "kruskal_wallis",
            )
            else 2
        )
        if (
            not isinstance(order, list)
            or len(order) < required
            or len(set(map(str, order))) != len(order)
        ):
            raise InvalidDataError("Group ordering is missing or inconsistent.")
        if method in _SIGNED_GROUP:
            if len(order) != 2:
                raise InvalidDataError("A two-group result must name exactly two groups.")
            contrast = result.metadata.get("contrast")
            expected_definition = (
                "first condition minus second condition"
                if method == "paired_t"
                else "first group minus second group"
            )
            if (
                not isinstance(contrast, dict)
                or contrast.get("definition") != expected_definition
                or contrast.get("first") != order[0]
                or contrast.get("second") != order[1]
            ):
                raise InvalidDataError("The recorded first-minus-second contrast is inconsistent.")
            if method == "paired_t":
                pairs = result.metadata.get("sample", {}).get("complete_pairs")
                return (
                    f"{_METHODS[method][0]} compared paired {question.outcome} values across "
                    f"{question.predictor} conditions {order[0]!r} and {order[1]!r} "
                    f"using {pairs} complete pairs.",
                    f"{order[0]!r} minus {order[1]!r}",
                )
            return (
                f"{_METHODS[method][0]} compared {question.outcome} across "
                f"{question.predictor} categories {order[0]!r} and {order[1]!r}.",
                f"{order[0]!r} minus {order[1]!r}",
            )
        if method == "pearson_chi_square":
            outcomes = result.metadata.get("outcome_order")
            if not isinstance(outcomes, list) or len(outcomes) < 2:
                raise InvalidDataError("The categorical outcome ordering is missing.")
            return (
                f"{_METHODS[method][0]} assessed association between "
                f"{question.predictor} and {question.outcome}.",
                None,
            )
        return (
            f"{_METHODS[method][0]} compared {question.outcome} across "
            f"{len(order)} {question.predictor} groups.",
            None,
        )
    if method == "pearson_correlation":
        order = result.metadata.get("variable_order")
        if (
            not question.outcome
            or not question.predictor
            or order != [question.outcome, question.predictor]
        ):
            raise InvalidDataError("Pearson variable ordering is missing or inconsistent.")
        return (
            f"Pearson correlation assessed linear association between {order[0]} and {order[1]}.",
            None,
        )
    raise InvalidDataError("The method has no inferential interpretation.")


def _assumption_notes(result: AnalysisResult) -> tuple[str, ...]:
    notes: list[str] = []
    if result.method_id in {"student_t", "one_way_anova"}:
        notes.append(
            "This pooled method assumes equal population variances; "
            "Levene's test cannot prove this."
        )
    elif result.method_id == "welch_t":
        notes.append(
            "Welch's test does not require equal population variances; independence "
            "and appropriate mean-inference conditions still matter."
        )
    elif result.method_id == "paired_t":
        notes.append(
            "The paired t-test models within-unit differences; complete pairs must be "
            "independent across units and suitable for mean inference."
        )
    if result.assumptions:
        notes.append(
            "Required conditions: "
            + "; ".join(result.assumptions)
            + ". Calculation alone does not verify them."
        )
    diagnostics = result.metadata.get("diagnostics")
    if isinstance(diagnostics, dict):
        normality = diagnostics.get("normality")
        if isinstance(normality, list):
            for item in normality:
                if not isinstance(item, dict):
                    continue
                status = item.get("status")
                group = item.get("group")
                if status == "not_rejected":
                    notes.append(
                        f"Normality was not rejected for group {group!r}; "
                        "this does not establish normality."
                    )
                elif status == "rejected":
                    notes.append(
                        f"The normality diagnostic rejected normality for group {group!r}."
                    )
                elif status:
                    notes.append(f"Normality diagnostic for group {group!r}: {status}.")
        variance = diagnostics.get("equal_variance_status")
        if variance == "not_rejected":
            notes.append(
                "The equal-variance diagnostic did not reject equal variances; "
                "it does not prove them."
            )
        elif variance == "rejected":
            notes.append("The equal-variance diagnostic rejected equal variances.")
        elif variance and variance != "not_applicable":
            notes.append(f"Equal-variance diagnostic: {variance}.")
    recommendation = result.recommendation
    if recommendation is not None:
        checks = recommendation.context.get("assumption_checks")
        if isinstance(checks, list):
            for check in checks:
                if (
                    isinstance(check, dict)
                    and check.get("assumption") == "independent_observations"
                    and check.get("status") == "confirmed"
                ):
                    notes.append(
                        "Independence was declared in the study design; "
                        "numerical values do not verify it."
                    )
                    break
    return tuple(dict.fromkeys(notes))


def _profile(result: AnalysisResult) -> InterpretationResult:
    profile = result.values.get("profile")
    if not isinstance(profile, dict) or result.sample_size is None:
        return _unavailable(result, "The descriptive profile or row count is missing.")
    descriptive = profile.get("descriptive", {})
    categorical = profile.get("categorical_summary", {})
    if not isinstance(descriptive, dict) or not isinstance(categorical, dict):
        return _unavailable(result, "The descriptive profile has an invalid structure.")
    numeric_details: list[dict[str, Any]] = []
    for name, values in descriptive.items():
        if isinstance(name, str) and isinstance(values, dict):
            numeric_details.append(
                {
                    "column": name,
                    "count": values.get("count"),
                    "mean": values.get("mean"),
                    "median": values.get("median"),
                    "std": values.get("std"),
                }
            )
    try:
        _json_value(numeric_details)
    except InvalidDataError:
        return _unavailable(result, "The descriptive profile contains invalid numerical details.")
    summary = (
        f"Descriptive profile of {result.sample_size} rows: "
        f"{len(descriptive)} numeric and {len(categorical)} categorical summaries."
    )
    return InterpretationResult(
        status=InterpretationStatus.AVAILABLE,
        method_id=result.method_id,
        execution_status=result.status,
        summary=summary,
        method_explanation=(
            "This profile describes observed data; it does not test a population hypothesis."
        ),
        conclusion=summary,
        findings=(InterpretationFinding("descriptive_only", summary, ("values.profile",)),),
        warnings=result.warnings,
        limitations=(
            "No population inference or causal conclusion follows from a descriptive profile.",
        ),
        metadata={
            "sample_size": result.sample_size,
            "excluded_rows": result.excluded_rows,
            "numeric_summaries": numeric_details,
            "categorical_summaries": len(categorical),
        },
    )


class InterpretationEngine:
    """Apply method-specific, deterministic rules to a Phase 6 result."""

    def interpret(self, result: AnalysisResult) -> InterpretationResult:
        if not isinstance(result, AnalysisResult):
            raise InvalidDataError("interpret() requires an AnalysisResult.")
        if result.status is AnalysisStatus.UNAVAILABLE:
            reason = result.metadata.get("reason")
            return _unavailable(
                result,
                reason if isinstance(reason, str) and reason else "The analysis was unavailable.",
            )
        if result.method_id not in _METHODS:
            return _unavailable(result, f"Method {result.method_id!r} is not supported.")
        if result.method_id == "dataset_profile":
            return _profile(result)
        if result.specification is None:
            return _unavailable(result, "The validated analysis specification is missing.")
        if result.sample_size is None or result.sample_size < 1:
            return _unavailable(result, "The analyzed sample size is missing or invalid.")
        sample = result.metadata.get("sample")
        if isinstance(sample, dict) and (
            sample.get("analyzed_rows") != result.sample_size
            or sample.get("excluded_rows") != result.excluded_rows
        ):
            return _unavailable(
                result, "The recorded sample counts contradict the result envelope."
            )
        try:
            method_text, contrast = _context(result)
        except InvalidDataError as exc:
            return _unavailable(result, str(exc))

        method = result.method_id
        values = result.values
        findings: list[InterpretationFinding] = []
        warnings = list(result.warnings)
        limitations: list[str] = []
        partial = False
        alpha = _finite(result.specification.options.alpha)
        if alpha is None or not 0 < alpha < 1:
            alpha = None
            partial = True
            warnings.append("The recorded significance threshold is invalid or missing.")
        p = _finite(values.get("p_value"))
        if p is None or not 0 <= p <= 1:
            p = None
            partial = True
            warnings.append(
                "The p-value is unavailable or outside [0, 1]; no significance decision was made."
            )
        statistic = _finite(values.get("test_statistic"))
        if statistic is None:
            partial = True
            warnings.append("The test statistic is unavailable or nonfinite.")
        estimate = _finite(values.get("primary_estimate"))
        expected_effect = _METHODS[method][1]
        effect = values.get("effect_size")
        effect_value = _finite(effect.get("value")) if isinstance(effect, dict) else None
        is_mean_test = method in {"welch_t", "student_t", "paired_t"}
        if estimate is None:
            partial = True
            warnings.append("The primary estimate is unavailable or nonfinite.")
        if (
            not isinstance(effect, dict)
            or effect.get("name") != expected_effect
            or effect_value is None
        ):
            effect_value = None
            partial = True
            warnings.append("The method-specific effect measure is unavailable or invalid.")
        if (
            is_mean_test
            and estimate is not None
            and effect_value is not None
            and ((estimate > 0) != (effect_value > 0) or (estimate < 0) != (effect_value < 0))
        ):
            effect_value = None
            partial = True
            warnings.append(
                "Cohen's d has a direction inconsistent with the recorded mean difference; "
                "its interpretation is unavailable."
            )
        if not is_mean_test and effect_value is None:
            estimate = None
        if estimate is not None and (
            (method in _NONNEGATIVE and estimate < 0)
            or (method in {"pearson_correlation", "mann_whitney_u"} and abs(estimate) > 1)
            or (method in {"one_way_anova", "pearson_chi_square"} and estimate > 1)
        ):
            partial = True
            warnings.append("The effect estimate is outside the range of its named measure.")
            estimate = None
        if (
            estimate is not None
            and effect_value is not None
            and not is_mean_test
            and not math.isclose(estimate, effect_value, rel_tol=1e-10, abs_tol=1e-12)
        ):
            partial = True
            warnings.append("The primary estimate disagrees with the recorded effect measure.")
            estimate = None

        hypothesis: str | None = None
        conclusion: str | None = None
        null = result.metadata.get("null_hypothesis")
        null_quantity = result.metadata.get("null_quantity")
        alternative = result.metadata.get("alternative_hypothesis")
        hypothesis_metadata_valid = (
            isinstance(null, str)
            and bool(null.strip())
            and isinstance(null_quantity, str)
            and bool(null_quantity.strip())
            and null_quantity == values.get("estimate_name")
            and isinstance(alternative, str)
            and bool(alternative.strip())
            and (
                alternative == "two-sided"
                if method in _SIGNED_GROUP | {"pearson_correlation"}
                else alternative == "association"
                if method == "pearson_chi_square"
                else alternative == "at least one group differs"
            )
        )
        if not hypothesis_metadata_valid:
            partial = True
            warnings.append("The null or alternative hypothesis metadata is incomplete.")
        if p is not None:
            p_text = _p_display(p)
            if p == 0:
                warnings.append(
                    "The numerical p-value is zero at machine precision; "
                    "its exact magnitude is unknown."
                )
            null_text = (
                f" The recorded null hypothesis is: {null}"
                if isinstance(null, str) and null
                else ""
            )
            if alpha is not None and statistic is not None and hypothesis_metadata_valid:
                if p < alpha:
                    conclusion = (
                        "The result provides evidence against the null hypothesis "
                        f"at alpha = {_fmt(alpha)}."
                    )
                    code = "evidence_against_null"
                else:
                    conclusion = (
                        "The result does not provide sufficient evidence to reject the null "
                        f"hypothesis at alpha = {_fmt(alpha)}."
                    )
                    code = "insufficient_evidence_against_null"
                _finding(
                    findings, code, conclusion, "values.p_value", "specification.options.alpha"
                )
                hypothesis = f"{p_text}.{null_text} {conclusion}"
            else:
                hypothesis = f"{p_text}.{null_text} No threshold decision is available."
                _finding(findings, "p_value_without_decision", hypothesis, "values.p_value")
        else:
            hypothesis = (
                "A valid p-value is unavailable; statistical significance cannot be classified."
            )
            _finding(findings, "p_value_unavailable", hypothesis, "values.p_value")

        effect_text: str | None = None
        if estimate is not None or (is_mean_test and effect_value is not None):
            name = values.get("estimate_name")
            if not isinstance(name, str) or not name:
                partial = True
                warnings.append("The primary estimate has no named quantity.")
            else:
                unit = values.get("estimate_unit")
                unit_text = f" {unit}" if isinstance(unit, str) and unit.strip() else ""
                if is_mean_test:
                    if estimate is not None:
                        direction = (
                            "positive" if estimate > 0 else "negative" if estimate < 0 else "zero"
                        )
                        direction_note = (
                            "The first group's observed mean was higher."
                            if estimate > 0
                            else "The first group's observed mean was lower."
                            if estimate < 0
                            else "The observed group means were equal."
                        )
                        effect_text = (
                            f"The estimated {name} ({contrast}) was {_fmt(estimate)}{unit_text}. "
                            f"{direction_note}"
                        )
                        _finding(
                            findings,
                            f"estimate_{direction}",
                            effect_text,
                            "values.primary_estimate",
                            "metadata.contrast",
                        )
                    if effect_value is not None:
                        d_text = (
                            f"Cohen's dz was {_fmt(effect_value)} "
                            "(mean paired difference divided by the SD of paired differences)."
                            if method == "paired_t"
                            else f"Cohen's d was {_fmt(effect_value)} "
                            "(first minus second, divided by the pooled sample SD)."
                        )
                        effect_text = f"{effect_text} {d_text}" if effect_text else d_text
                        _finding(
                            findings,
                            "standardized_effect_reported",
                            d_text,
                            "values.effect_size",
                            "metadata.contrast",
                        )
                    else:
                        _finding(
                            findings,
                            "effect_unavailable",
                            "Cohen's d is unavailable or inconsistent.",
                            "values.effect_size",
                        )
                elif (
                    estimate is not None and effect_value is not None and method == "mann_whitney_u"
                ):
                    direction = (
                        "positive" if estimate > 0 else "negative" if estimate < 0 else "zero"
                    )
                    effect_text = (
                        f"Rank-biserial correlation ({contrast}) was {_fmt(estimate)}. "
                        "Its sign describes the observed rank ordering; "
                        "it is not a median difference."
                    )
                elif (
                    estimate is not None and effect_value is not None and method == "one_way_anova"
                ):
                    direction = "positive" if estimate > 0 else "zero"
                    effect_text = (
                        f"Eta-squared was {_fmt(estimate)}, the reported sample proportion of "
                        "variance associated with group membership."
                    )
                elif (
                    estimate is not None and effect_value is not None and method == "kruskal_wallis"
                ):
                    direction = "positive" if estimate > 0 else "zero"
                    effect_text = (
                        f"Epsilon-squared was {_fmt(estimate)}, "
                        "the reported truncated rank effect measure."
                    )
                elif (
                    estimate is not None
                    and effect_value is not None
                    and method == "pearson_correlation"
                ):
                    direction = (
                        "positive" if estimate > 0 else "negative" if estimate < 0 else "zero"
                    )
                    effect_text = (
                        f"Pearson r was {_fmt(estimate)}, indicating "
                        f"{direction} linear association in the analyzed observations."
                        if estimate != 0
                        else "The observed Pearson r was zero; this does not establish "
                        "population independence."
                    )
                elif estimate is not None and effect_value is not None:
                    direction = "positive" if estimate > 0 else "zero"
                    effect_text = (
                        f"Cramer's V was {_fmt(estimate)}, a nonnegative measure of categorical "
                        "association without a direction."
                    )
                if not is_mean_test and effect_text is not None:
                    _finding(
                        findings,
                        f"estimate_{direction}"
                        if method not in _NONNEGATIVE
                        else "estimate_nonnegative",
                        effect_text,
                        "values.primary_estimate",
                        "values.effect_size",
                    )
        else:
            _finding(
                findings,
                "effect_unavailable",
                "A trustworthy effect estimate is unavailable.",
                "values.primary_estimate",
                "values.effect_size",
            )
        if method in {"welch_t", "student_t"} and isinstance(effect, dict):
            effect_interval = effect.get("confidence_interval")
            if effect_interval is not None:
                if effect_value is None:
                    partial = True
                    warnings.append(
                        "The standardized-effect interval cannot be interpreted "
                        "without a valid Cohen's d."
                    )
                elif not isinstance(effect_interval, dict):
                    partial = True
                    warnings.append("The standardized-effect interval has an invalid structure.")
                else:
                    effect_low = _finite(effect_interval.get("lower"))
                    effect_high = _finite(effect_interval.get("upper"))
                    effect_level = _finite(effect_interval.get("level"))
                    if (
                        effect_low is None
                        or effect_high is None
                        or effect_level is None
                        or not 0 < effect_level < 1
                        or not math.isclose(
                            effect_level,
                            result.specification.options.confidence_level,
                            abs_tol=1e-12,
                        )
                        or effect_low > effect_high
                        or effect_interval.get("quantity") != expected_effect
                        or effect_interval.get("method")
                        != "independent within-group percentile bootstrap"
                    ):
                        partial = True
                        warnings.append("The standardized-effect interval needs review.")
                    elif effect_text is not None:
                        effect_text += (
                            f" Its {_fmt(100 * effect_level)}% bootstrap interval was "
                            f"{_fmt(effect_low)} to {_fmt(effect_high)}."
                        )
                        _finding(
                            findings,
                            "effect_interval_reported",
                            "A separate standardized-effect interval is available.",
                            "values.effect_size.confidence_interval",
                        )

        interval_text: str | None = None
        interval = values.get("confidence_interval")
        valid_interval = False
        if interval is None:
            partial = True
            limitations.append("A confidence interval for the primary estimate is unavailable.")
            interval_text = (
                "Uncertainty could not be quantified from a confidence interval for this estimate."
            )
            _finding(findings, "interval_unavailable", interval_text, "values.confidence_interval")
        elif not isinstance(interval, dict):
            partial = True
            warnings.append("The recorded confidence interval has an invalid structure.")
        else:
            low = _finite(interval.get("lower"))
            high = _finite(interval.get("upper"))
            level = _finite(interval.get("level"))
            quantity = interval.get("quantity")
            expected_method = (
                "analytical paired t interval"
                if method == "paired_t"
                else "analytical t interval"
                if is_mean_test
                else "observation-row percentile bootstrap"
                if method == "pearson_chi_square"
                else "independent within-group percentile bootstrap"
                if method in {"mann_whitney_u", "one_way_anova", "kruskal_wallis"}
                else None
            )
            if (
                low is None
                or high is None
                or level is None
                or not 0 < level < 1
                or not math.isclose(
                    level, result.specification.options.confidence_level, abs_tol=1e-12
                )
                or low > high
                or quantity != values.get("estimate_name")
                or expected_method is None
                or interval.get("method") != expected_method
                or (is_mean_test and (estimate is None or not low <= estimate <= high))
            ):
                partial = True
                warnings.append(
                    "The reported interval has invalid bounds, level, quantity, method, "
                    "or analytical estimate coverage; review it."
                )
                _finding(
                    findings,
                    "interval_requires_review",
                    warnings[-1],
                    "values.confidence_interval",
                    "values.primary_estimate",
                )
            else:
                valid_interval = True
                unit = values.get("estimate_unit")
                unit_text = f" {unit}" if isinstance(unit, str) and unit.strip() else ""
                interval_text = (
                    f"The {_fmt(100 * level)}% confidence interval for {quantity} was "
                    f"{_fmt(low)} to {_fmt(high)}{unit_text}."
                )
                null_value = _finite(result.metadata.get("null_value"))
                if (
                    method in _SIGNED_GROUP | {"pearson_correlation"}
                    and null_value is not None
                    and result.metadata.get("null_quantity") == quantity
                ):
                    if low <= null_value <= high:
                        interval_text += " The interval includes the recorded null value."
                        code = "interval_includes_null"
                    else:
                        interval_text += " The interval excludes the recorded null value."
                        code = "interval_excludes_null"
                    _finding(
                        findings,
                        code,
                        interval_text,
                        "values.confidence_interval",
                        "metadata.null_value",
                    )
                else:
                    _finding(
                        findings, "interval_reported", interval_text, "values.confidence_interval"
                    )
                if (
                    method in {"welch_t", "student_t", "paired_t"}
                    and p is not None
                    and alpha is not None
                    and null_value is not None
                    and interval.get("method")
                    in {"analytical t interval", "analytical paired t interval"}
                    and result.metadata.get("alternative_hypothesis") == "two-sided"
                    and math.isclose(level, 1 - alpha, abs_tol=1e-12)
                    and ((p < alpha) == (low <= null_value <= high))
                ):
                    partial = True
                    warnings.append(
                        "The analytical interval and two-sided p-value imply "
                        "different threshold decisions."
                    )
        if valid_interval and method in {"mann_whitney_u", "pearson_chi_square"}:
            limitations.append(
                "The bootstrap effect interval and hypothesis test use different constructions; "
                "their threshold decisions need not agree."
            )

        if method in {"one_way_anova", "kruskal_wallis"}:
            limitations.append("The omnibus test does not identify specific group differences.")
        if method in {"pearson_correlation", "pearson_chi_square"}:
            limitations.append("Observed association alone does not establish causation.")
        if result.excluded_rows:
            limitations.append(f"{result.excluded_rows} row(s) were excluded from this analysis.")
        if conclusion is not None:
            limitations.append(
                "Statistical significance alone does not establish practical importance."
            )
        summary = (
            f"{method_text} {conclusion}"
            if conclusion is not None
            else f"{method_text} A complete hypothesis conclusion is unavailable."
        )
        return InterpretationResult(
            status=InterpretationStatus.PARTIAL if partial else InterpretationStatus.AVAILABLE,
            method_id=method,
            execution_status=result.status,
            summary=summary,
            method_explanation=method_text,
            hypothesis_interpretation=hypothesis,
            effect_interpretation=effect_text,
            uncertainty_interpretation=interval_text,
            assumption_notes=_assumption_notes(result),
            limitations=tuple(limitations),
            conclusion=conclusion,
            findings=tuple(findings),
            warnings=tuple(dict.fromkeys(warnings)),
            metadata={
                "sample_size": result.sample_size,
                "excluded_rows": result.excluded_rows,
                "alpha": alpha,
                "p_value": p,
                "test_statistic": statistic,
                "primary_estimate": estimate,
                "contrast": contrast,
                "confidence_interval": interval if valid_interval else None,
            },
        )
