"""Deterministic explanations of completed analysis records.

This module reads recorded results. It never calls a statistical backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .exceptions import InvalidDataError
from .narrate import (
    _confidence_interval_width,
    assumption_grade,
    effect_narrative,
    hypothesis_verdict,
    reliability_diagnostics_narrative,
    reliability_item_narrative,
    reliability_narrative,
)
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

    @property
    def findings_plain(self) -> str:
        """Return all finding messages as a numbered, human-readable string.

        Each line is taken directly from the already-generated finding message so
        that the text is deterministic and consistent with the full result.

        Example usage::

            print(interpretation.findings_plain)
            # 1. The result provides evidence against the null hypothesis at alpha = 0.05.
            # 2. The estimated mean difference ('new' minus 'standard') was 7.33 points.
            # 3. Cohen's d was 1.253 (first minus second, divided by the pooled sample SD).
        """
        if not self.findings:
            return "No findings recorded."
        lines = [f"{i}. {f.message}" for i, f in enumerate(self.findings, start=1)]
        return "\n".join(lines)

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
    "one_sample_t": ("One-sample t-test", "one-sample Cohen's d"),
    "wilcoxon_signed_rank": (
        "Paired Wilcoxon signed-rank test",
        "matched-pairs rank-biserial correlation",
    ),
    "mann_whitney_u": ("Mann-Whitney U test", "rank-biserial correlation"),
    "welch_anova": ("Welch one-way ANOVA", "global standardized effect"),
    "one_way_anova": ("One-way ANOVA", "eta-squared"),
    "kruskal_wallis": ("Kruskal-Wallis test", "epsilon-squared (rank)"),
    "pearson_correlation": ("Pearson correlation", "Pearson r"),
    "spearman_correlation": ("Spearman rank correlation", "Spearman rho"),
    "pearson_chi_square": ("Pearson chi-square independence test", "Cramer's V"),
    "fisher_exact": ("Fisher's exact test", "sample odds ratio"),
    "linear_regression": ("Ordinary least-squares linear regression", "R-squared"),
    "mcnemar": ("Exact two-sided McNemar test", "paired proportion difference"),
    "point_biserial_correlation": ("Point-biserial correlation", "point-biserial r"),
    "kendall_tau_b": ("Kendall's tau-b", "Kendall's tau-b"),
    "partial_pearson_correlation": ("Partial Pearson correlation", "partial Pearson r"),
}
_SIGNED_GROUP = {
    "welch_t",
    "student_t",
    "paired_t",
    "wilcoxon_signed_rank",
    "mann_whitney_u",
    "mcnemar",
}
_GROUP = _SIGNED_GROUP | {
    "welch_anova",
    "one_way_anova",
    "kruskal_wallis",
    "pearson_chi_square",
}
_NONNEGATIVE = {"one_way_anova", "kruskal_wallis", "pearson_chi_square"}
_CORRELATIONS = {
    "pearson_correlation",
    "spearman_correlation",
    "point_biserial_correlation",
    "kendall_tau_b",
    "partial_pearson_correlation",
}
_TWO_SIDED = _SIGNED_GROUP | _CORRELATIONS | {"one_sample_t"}


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


def _regression_interpretation(result: AnalysisResult) -> InterpretationResult:
    if result.specification is None or result.sample_size is None:
        return _unavailable(result, "The regression specification or sample accounting is missing.")
    values = result.values
    fit = values.get("model_fit")
    coefficients = values.get("coefficients")
    diagnostics = values.get("diagnostics")
    if (
        not isinstance(fit, dict)
        or not isinstance(coefficients, list)
        or not isinstance(diagnostics, dict)
    ):
        return _unavailable(result, "The structured regression result is incomplete.")
    r_squared = _finite(fit.get("r_squared"))
    adjusted = _finite(fit.get("adjusted_r_squared"))
    model_f = _finite(fit.get("model_f_statistic"))
    model_p = _finite(fit.get("model_f_p_value"))
    residual_df = _finite(fit.get("residual_degrees_of_freedom"))
    model_df = _finite(fit.get("model_degrees_of_freedom"))
    if r_squared is None or adjusted is None:
        return _unavailable(result, "R-squared or adjusted R-squared is unavailable.")
    covariance = values.get("covariance_type")
    summary = (
        f"The fitted OLS model used {result.sample_size} complete cases and accounted for "
        f"approximately {100 * r_squared:.1f}% of observed outcome variance in this sample "
        f"(R-squared = {_fmt(r_squared)}, adjusted R-squared = {_fmt(adjusted)})."
    )
    hypothesis = None
    findings: list[InterpretationFinding] = []
    _finding(findings, "regression_model_fit", summary, "values.model_fit")
    if (
        model_f is not None
        and model_p is not None
        and model_df is not None
        and residual_df is not None
    ):
        hypothesis = (
            f"The model-level test was F({_fmt(model_df)}, {_fmt(residual_df)}) = "
            f"{_fmt(model_f)}, {_p_display(model_p)}. It tests whether all non-intercept "
            "population slopes are zero, or equivalently whether the included slopes jointly "
            "improve fit over an intercept-only model under the selected covariance inference."
        )
        _finding(
            findings,
            "regression_model_test",
            hypothesis,
            "values.model_fit.model_f_statistic",
            "values.model_fit.model_f_p_value",
        )
    coefficient_messages = []
    for item in coefficients:
        if not isinstance(item, dict) or item.get("kind") == "intercept":
            continue
        estimate = _finite(item.get("estimate"))
        p_value = _finite(item.get("p_value"))
        interval = item.get("confidence_interval")
        if estimate is None or p_value is None or not isinstance(interval, dict):
            continue
        predictor = item.get("predictor")
        if item.get("kind") == "categorical":
            meaning = (
                f"{predictor} level {item.get('level')!r} versus reference "
                f"{item.get('reference_level')!r}"
            )
        else:
            meaning = f"a one-unit increase in {predictor}"
        message = (
            f"Holding the other included predictors constant, {meaning} was associated with "
            f"an estimated {_fmt(estimate)}-unit difference in the outcome "
            f"(CI {_fmt(float(interval['lower']))} to {_fmt(float(interval['upper']))}; "
            f"{_p_display(p_value)})."
        )
        coefficient_messages.append(message)
        if len(coefficient_messages) <= 5:
            _finding(findings, "regression_coefficient", message, "values.coefficients")
    uncertainty = " ".join(coefficient_messages[:5])
    if len(coefficient_messages) > 5:
        uncertainty += (
            f" {len(coefficient_messages) - 5} additional coefficient records are preserved "
            "in the structured result and report table."
        )
    bp = diagnostics.get("breusch_pagan")
    normality = diagnostics.get("residual_normality")
    influence = diagnostics.get("influence")
    vif = diagnostics.get("vif")
    condition_number = _finite(diagnostics.get("condition_number"))
    assumption_notes = [
        "Rows must be independent observational units; this is researcher-declared and not "
        "verified from numerical values.",
        "The conditional mean should be adequately linear in the numerical predictors at the "
        "specified additive scale; review residual-versus-fitted behavior for curvature or "
        "other systematic structure.",
        "Standardized betas are reported only for continuous predictors. They put those slopes "
        "on a common sample-standard-deviation scale, but do not establish causal or absolute "
        "predictor importance.",
    ]
    if isinstance(bp, dict) and _finite(bp.get("lm_p_value")) is not None:
        bp_status = "rejected" if bp.get("status") == "rejected" else "not rejected"
        assumption_notes.append(
            f"Breusch-Pagan LM {_p_display(float(bp['lm_p_value']))}; the constant-variance null "
            f"was {bp_status} at the recorded alpha. This diagnostic neither proves a variance "
            f"structure nor changed the selected {covariance} covariance estimator."
        )
    if isinstance(normality, dict) and _finite(normality.get("p_value")) is not None:
        normality_status = "rejected" if normality.get("status") == "rejected" else "not rejected"
        assumption_notes.append(
            f"Jarque-Bera residual diagnostic {_p_display(float(normality['p_value']))}; the "
            f"normal-residual null was {normality_status}. OLS point estimation does not require "
            "normal residuals, but exact small-sample inference can be sensitive; larger samples "
            "can reduce sensitivity to moderate departures while severe outliers or heavy tails "
            "may still matter. The diagnostic alone neither proves normality nor invalidates OLS."
        )
    if isinstance(vif, dict):
        maximum = _finite(vif.get("maximum"))
        if maximum is not None:
            assumption_notes.append(
                f"Maximum non-intercept VIF was {_fmt(maximum)}; conventional cutoffs are review "
                "heuristics rather than automatic variable-selection rules."
            )
    if isinstance(influence, dict):
        assumption_notes.append(
            f"Influence heuristics flagged {influence.get('flagged_count', 0)} observations; "
            "no observations were deleted and row identities are not exposed."
        )
    if condition_number is not None:
        assumption_notes.append(
            f"The design condition number was {_fmt(condition_number)}. It is scale-sensitive "
            "and is a review aid rather than an automatic validity rule."
        )
    limitations = (
        "Coefficient estimates describe conditional associations under this specified model; "
        "regression does not establish causation.",
        "R-squared summarizes in-sample fit and is not out-of-sample predictive accuracy.",
        "Residual and influence diagnostics are screening evidence, not automatic model-selection "
        "or deletion rules.",
    )
    return InterpretationResult(
        status=InterpretationStatus.AVAILABLE,
        method_id=result.method_id,
        execution_status=result.status,
        summary=summary,
        method_explanation=(
            f"OLS estimated the conditional mean with an intercept and {covariance} covariance "
            "inference using one complete-case sample."
            + (
                " HC3 changes the estimated coefficient uncertainty, not the OLS point "
                "coefficients or analyzed data."
                if covariance == "HC3"
                else ""
            )
        ),
        hypothesis_interpretation=hypothesis,
        effect_interpretation=summary,
        uncertainty_interpretation=uncertainty or "No non-intercept coefficient was available.",
        assumption_notes=tuple(assumption_notes),
        limitations=limitations,
        conclusion=(
            summary
            + " Conditional associations should be interpreted with the recorded diagnostics."
        ),
        findings=tuple(findings),
        warnings=result.warnings,
        metadata={
            "sample_size": result.sample_size,
            "excluded_rows": result.excluded_rows,
            "covariance_type": covariance,
            "coefficient_count": len(coefficients),
            "p_value": model_p,
            "test_statistic": model_f,
            "primary_estimate": r_squared,
            "confidence_interval": None,
        },
    )


def _logistic_interpretation(result: AnalysisResult) -> InterpretationResult:
    """Interpret model-level logistic inference and preserve coefficient records."""
    values = result.values
    fit = values.get("model_fit")
    coefficients = values.get("coefficients")
    diagnostics = values.get("diagnostics")
    if (
        not isinstance(fit, dict)
        or not isinstance(coefficients, list)
        or not isinstance(diagnostics, dict)
    ):
        return _unavailable(result, "The structured logistic model result is incomplete.")
    event = values.get("event_level")
    p_value = _finite(fit.get("lr_p_value"))
    statistic = _finite(fit.get("lr_statistic"))
    pseudo_r2 = _finite(fit.get("mcfadden_r2"))
    if event is None or p_value is None or statistic is None:
        return _unavailable(result, "The modeled event or likelihood-ratio result is missing.")
    alpha = result.specification.options.alpha if result.specification is not None else 0.05
    evidence = (
        "provided evidence that at least one non-intercept coefficient differs from zero"
        if p_value < alpha
        else "did not provide evidence that a non-intercept coefficient differs from zero"
    )
    summary = (
        f"Binary logistic regression modeled event {event!r} in {result.sample_size} complete "
        f"observations and {evidence} (likelihood-ratio p = {_fmt(p_value)})."
    )
    findings = [
        InterpretationFinding(
            "model_likelihood_ratio",
            summary,
            ("values.model_fit.lr_statistic", "values.model_fit.lr_p_value"),
        )
    ]
    for coefficient in coefficients:
        if not isinstance(coefficient, dict) or coefficient.get("term_type") == "intercept":
            continue
        odds_ratio = _finite(coefficient.get("odds_ratio"))
        if odds_ratio is not None:
            if coefficient.get("term_type") == "categorical":
                message = (
                    f"Holding the other included predictors constant, the estimated odds of "
                    f"event {event!r} for {coefficient.get('level')!r} were "
                    f"{_fmt(odds_ratio)} times those for reference category "
                    f"{coefficient.get('reference_level')!r}."
                )
            else:
                direction = "lower" if odds_ratio < 1 else "higher" if odds_ratio > 1 else "equal"
                message = (
                    f"Holding the other included predictors constant, a one-unit increase in "
                    f"{coefficient.get('term_label')} was associated with {direction} estimated "
                    f"event odds ({_fmt(odds_ratio)} times as large)."
                )
            findings.append(
                InterpretationFinding(
                    "logistic_odds_ratio",
                    message,
                    ("values.coefficients",),
                )
            )
    limitations = (
        "Odds ratios describe odds, not constant probability differences.",
        "Holding other included predictors constant does not establish causation.",
        "Model fit is in-sample and is not validated predictive performance.",
        "McFadden pseudo-R-squared is likelihood based and is not ordinary R-squared.",
    )
    return InterpretationResult(
        status=InterpretationStatus.AVAILABLE,
        method_id=result.method_id,
        execution_status=result.status,
        summary=summary,
        method_explanation=(
            "The model estimates conditional log odds of the declared event. Exponentiated "
            "coefficients are adjusted odds ratios using the recorded predictor coding."
        ),
        hypothesis_interpretation=summary,
        effect_interpretation=(
            f"McFadden pseudo-R-squared was {_fmt(pseudo_r2)}."
            if pseudo_r2 is not None
            else "McFadden pseudo-R-squared was unavailable."
        ),
        assumption_notes=(
            "Verify independent observational units, the event definition, predictor units "
            "and reference levels, model form, and estimation stability.",
        ),
        limitations=limitations,
        conclusion=summary,
        findings=tuple(findings),
        warnings=result.warnings,
        metadata={
            "sample_size": result.sample_size,
            "excluded_rows": result.excluded_rows,
            "p_value": p_value,
            "test_statistic": statistic,
            "primary_estimate": pseudo_r2,
        },
    )


def _context(result: AnalysisResult) -> tuple[str, str | None]:
    """Return the method description and the verified group contrast, if any."""
    method = result.method_id
    assert result.specification is not None
    question = result.specification.question
    if method == "one_sample_t":
        reference = question.reference_value
        sample_mean = _finite(result.values.get("sample_mean"))
        if not question.outcome or reference is None or sample_mean is None:
            raise InvalidDataError("The one-sample outcome, mean, or reference value is missing.")
        return (
            f"One-sample t-test compared the population mean of {question.outcome} with the "
            f"declared reference value {_fmt(reference)}; the observed sample mean was "
            f"{_fmt(sample_mean)}.",
            f"observed mean minus reference value {_fmt(reference)}",
        )
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
                "welch_anova",
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
                if method in {"paired_t", "wilcoxon_signed_rank", "mcnemar"}
                else "first group minus second group"
            )
            if (
                not isinstance(contrast, dict)
                or contrast.get("definition") != expected_definition
                or contrast.get("first") != order[0]
                or contrast.get("second") != order[1]
            ):
                raise InvalidDataError("The recorded first-minus-second contrast is inconsistent.")
            if method in {"paired_t", "wilcoxon_signed_rank", "mcnemar"}:
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
    if method in _CORRELATIONS:
        order = result.metadata.get("variable_order")
        if (
            not question.outcome
            or not question.predictor
            or order != [question.outcome, question.predictor]
        ):
            raise InvalidDataError("Correlation variable ordering is missing or inconsistent.")
        target = {
            "pearson_correlation": "linear",
            "spearman_correlation": "monotonic rank",
            "point_biserial_correlation": "binary-continuous",
            "kendall_tau_b": "ordinal concordance",
            "partial_pearson_correlation": "linearly adjusted partial",
        }[method]
        return (
            f"{_METHODS[method][0]} assessed {target} association between "
            f"{order[0]} and {order[1]}.",
            None,
        )
    if method == "fisher_exact":
        order = result.metadata.get("variable_order")
        rows = result.metadata.get("row_order")
        columns = result.metadata.get("column_order")
        observed = result.metadata.get("observed_counts")
        if (
            not question.outcome
            or not question.predictor
            or order != [question.outcome, question.predictor]
            or not isinstance(rows, list)
            or len(rows) != 2
            or not isinstance(columns, list)
            or len(columns) != 2
            or not isinstance(observed, list)
            or len(observed) != 2
            or any(not isinstance(row, list) or len(row) != 2 for row in observed)
        ):
            raise InvalidDataError("The Fisher 2x2 table context is missing or inconsistent.")
        return (
            f"Fisher's exact test assessed association between {order[0]} and {order[1]} "
            "using the recorded ordered 2x2 table.",
            None,
        )
    raise InvalidDataError("The method has no inferential interpretation.")


def _assumption_notes(result: AnalysisResult) -> tuple[str, ...]:
    notes: list[str] = []

    def graded(*args: Any, **kwargs: Any) -> str:
        severity, message = assumption_grade(*args, **kwargs)
        return f"[{severity}] {message}"

    if result.method_id in {"student_t", "one_way_anova"}:
        notes.append(graded("equal_variance", "required", method_id=result.method_id))
    elif result.method_id in {"welch_t", "welch_anova"}:
        notes.append(graded("equal_variance", "not_required", method_id=result.method_id))
    elif result.method_id in {"paired_t", "wilcoxon_signed_rank"}:
        notes.append(graded("paired_structure", "required", method_id=result.method_id))
    if result.assumptions:
        notes.append(
            "[CAUTION] Required conditions: "
            + "; ".join(result.assumptions)
            + ". Calculation alone does not verify them. Worth noting; low risk given context."
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
                if status:
                    notes.append(
                        graded(
                            "normality",
                            str(status),
                            n=item.get("sample_size"),
                            p_value=item.get("p_value"),
                            group=group,
                            method_id=result.method_id,
                        )
                    )
        variance = diagnostics.get("equal_variance_status")
        if variance and variance != "not_applicable":
            notes.append(
                graded(
                    "equal_variance",
                    str(variance),
                    p_value=diagnostics.get("levene_p_value"),
                    method_id=result.method_id,
                )
            )
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
                    notes.append(graded("independence", "confirmed"))
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


def _reliability_interpretation(result: AnalysisResult) -> InterpretationResult:
    """Interpret internal consistency without turning alpha into a pass/fail rule."""
    values = result.values
    alpha = _finite(values.get("cronbach_alpha"))
    items = values.get("items")
    item_statistics = values.get("item_statistics")
    interval = values.get("confidence_interval")
    missingness = values.get("missingness")
    negative = values.get("negative_inter_item_correlations")
    scoring = values.get("scoring")
    mean_inter_item = _finite(values.get("mean_inter_item_correlation"))
    if (
        alpha is None
        or result.sample_size is None
        or not isinstance(items, list)
        or len(items) < 2
        or not isinstance(item_statistics, list)
        or len(item_statistics) != len(items)
        or not isinstance(missingness, list)
        or not isinstance(negative, dict)
        or not isinstance(scoring, dict)
    ):
        return _unavailable(result, "The reliability estimate or diagnostics are incomplete.")
    interval_available = (
        isinstance(interval, dict)
        and interval.get("status") == "available"
        and _finite(interval.get("lower")) is not None
        and _finite(interval.get("upper")) is not None
    )
    partial = not interval_available or any(
        item.get("corrected_item_total_status") != "available" for item in item_statistics
    )
    summary = reliability_narrative(
        alpha, len(items), result.sample_size, interval if isinstance(interval, dict) else None
    )
    item_text = reliability_item_narrative(item_statistics)
    pair_count = negative.get("count")
    if not isinstance(pair_count, int) or pair_count < 0:
        pair_count = 0
        partial = True
    diagnostics_text = reliability_diagnostics_narrative(mean_inter_item, pair_count)
    original = result.sample_size + (result.excluded_rows or 0)
    excluded = result.excluded_rows or 0
    excluded_percentage = 100 * excluded / original if original else 0.0
    missing_counts: list[tuple[str, int]] = []
    for item in missingness:
        if not isinstance(item, dict):
            continue
        missing_count = item.get("missing_count")
        if (
            isinstance(missing_count, int)
            and not isinstance(missing_count, bool)
            and missing_count >= 0
        ):
            missing_counts.append((str(item.get("item")), missing_count))
    if len(missing_counts) != len(items):
        partial = True
    maximum_missing = max((count for _, count in missing_counts), default=0)
    most_missing = [
        name for name, count in missing_counts if count == maximum_missing and maximum_missing > 0
    ]
    missing_text = (
        f"Complete-case analysis retained {result.sample_size} of {original} respondents "
        f"and excluded {excluded} ({excluded_percentage:.1f}%). "
        + (
            f"The largest per-item missing count was {maximum_missing} for "
            + ", ".join(repr(item) for item in most_missing)
            + "."
            if most_missing
            else "No selected item had a missing value."
        )
    )
    reversed_items = scoring.get("reversed_items")
    if isinstance(reversed_items, list) and reversed_items:
        reversed_names = [
            str(item.get("item")) for item in reversed_items if isinstance(item, dict)
        ]
        scoring_text = (
            "Explicit researcher-supplied reverse scoring was applied on an internal copy to "
            + ", ".join(repr(item) for item in reversed_names)
            + "; no automatic reversal was performed."
        )
    else:
        scoring_text = "No reverse scoring was applied or inferred automatically."
    findings = (
        InterpretationFinding(
            "reliability_estimate", summary, ("values.cronbach_alpha", "values.confidence_interval")
        ),
        InterpretationFinding("item_diagnostics", item_text, ("values.item_statistics",)),
        InterpretationFinding(
            "inter_item_diagnostics",
            diagnostics_text,
            (
                "values.mean_inter_item_correlation",
                "values.negative_inter_item_correlations",
            ),
        ),
        InterpretationFinding("complete_case_sample", missing_text, ("metadata.sample",)),
        InterpretationFinding("scoring_record", scoring_text, ("values.scoring",)),
    )
    limitations = (
        "Cronbach's alpha summarizes internal consistency; it does not establish "
        "unidimensionality or construct validity.",
        "This result does not establish content, criterion, convergent, or discriminant "
        "validity, measurement invariance, test-retest reliability, or inter-rater reliability.",
        "Alpha depends on item count and covariance and can be increased by redundant items.",
        "Numeric Likert-style scores are treated with ordinary variances and correlations; "
        "ordinal or polychoric reliability is not estimated.",
        "Complete-case analysis may differ from the target population when missingness is "
        "systematic.",
        "Item deletion or reverse scoring should be theory-driven and declared, not selected "
        "solely to improve alpha.",
    )
    uncertainty = (
        summary.split(" complete respondents.", 1)[1].strip()
        if interval_available and " complete respondents." in summary
        else (
            "The requested bootstrap interval is unavailable; the point estimate and other "
            "diagnostics remain usable."
        )
    )
    return InterpretationResult(
        status=InterpretationStatus.PARTIAL if partial else InterpretationStatus.AVAILABLE,
        method_id=result.method_id,
        execution_status=result.status,
        summary=summary,
        method_explanation=(
            "Cronbach's alpha uses sample item variances and the variance of their total score. "
            "No hypothesis test or universal adequacy cutoff was applied."
        ),
        uncertainty_interpretation=uncertainty,
        assumption_notes=(item_text, diagnostics_text, missing_text, scoring_text),
        limitations=limitations,
        conclusion=(
            "The estimate describes the internal consistency of this declared item set in the "
            "analyzed complete-case sample; substantive adequacy requires context and theory."
        ),
        findings=findings,
        warnings=result.warnings,
        metadata={
            "alpha": alpha,
            "item_count": len(items),
            "sample_size": result.sample_size,
            "excluded_rows": result.excluded_rows,
            "interval_status": interval.get("status") if isinstance(interval, dict) else None,
            "non_inferential": True,
        },
    )


class InterpretationEngine:
    """Apply method-specific, deterministic rules to an analysis result."""

    def interpret(self, result: AnalysisResult) -> InterpretationResult:
        if not isinstance(result, AnalysisResult):
            raise InvalidDataError("interpret() requires an AnalysisResult.")
        if result.status is AnalysisStatus.UNAVAILABLE:
            reason = result.metadata.get("reason")
            return _unavailable(
                result,
                reason if isinstance(reason, str) and reason else "The analysis was unavailable.",
            )
        if result.method_id == "linear_regression":
            return _regression_interpretation(result)
        if result.method_id == "logistic_regression":
            return _logistic_interpretation(result)
        if result.method_id == "cronbach_alpha":
            return _reliability_interpretation(result)
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
        is_mean_test = method in {"welch_t", "student_t", "paired_t", "one_sample_t"}
        global_effect_not_applicable = (
            method == "welch_anova"
            and isinstance(effect, dict)
            and effect.get("status") == "not_applicable"
            and effect.get("name") == expected_effect
        )
        if estimate is None and method != "welch_anova":
            partial = True
            warnings.append("The primary estimate is unavailable or nonfinite.")
        if not global_effect_not_applicable and (
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
        if not is_mean_test and effect_value is None and method != "welch_anova":
            estimate = None
        if estimate is not None and (
            (method in _NONNEGATIVE and estimate < 0)
            or (
                method
                in {
                    "pearson_correlation",
                    "spearman_correlation",
                    "point_biserial_correlation",
                    "kendall_tau_b",
                    "partial_pearson_correlation",
                    "mann_whitney_u",
                    "wilcoxon_signed_rank",
                }
                and abs(estimate) > 1
            )
            or (method in {"one_way_anova", "pearson_chi_square"} and estimate > 1)
            or (method == "fisher_exact" and estimate < 0)
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
                if method in _TWO_SIDED
                else alternative == "association"
                if method in {"pearson_chi_square", "fisher_exact"}
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
                f"The recorded null hypothesis is: {null} "
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
                measure = effect.get("name") if isinstance(effect, dict) else ""
                hypothesis = null_text + hypothesis_verdict(
                    p,
                    alpha,
                    effect_value,
                    str(measure),
                    n=result.sample_size,
                )
            else:
                hypothesis = f"{p_text}. {null_text}No threshold decision is available."
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
                            "The observed sample mean was above the reference."
                            if method == "one_sample_t" and estimate > 0
                            else "The observed sample mean was below the reference."
                            if method == "one_sample_t" and estimate < 0
                            else "The observed sample mean equaled the reference."
                            if method == "one_sample_t"
                            else "The first group's observed mean was higher."
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
                        assert isinstance(effect, dict)
                        definition = effect.get("definition")
                        d_text = effect_narrative(
                            str(effect.get("name")),
                            effect_value,
                            orientation=contrast,
                            definition=definition if isinstance(definition, str) else None,
                        )
                        d_text = d_text.replace(
                            f"{effect.get('name')} = ", f"{effect.get('name')} was ", 1
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
                    assert isinstance(effect, dict)
                    definition = effect.get("definition")
                    effect_text = effect_narrative(
                        str(effect.get("name")),
                        estimate,
                        orientation=contrast,
                        definition=definition if isinstance(definition, str) else None,
                    )
                elif (
                    estimate is not None and effect_value is not None and method == "one_way_anova"
                ):
                    direction = "positive" if estimate > 0 else "zero"
                    assert isinstance(effect, dict)
                    definition = effect.get("definition")
                    effect_text = effect_narrative(
                        str(effect.get("name")),
                        estimate,
                        definition=definition if isinstance(definition, str) else None,
                    )
                elif (
                    estimate is not None and effect_value is not None and method == "kruskal_wallis"
                ):
                    direction = "positive" if estimate > 0 else "zero"
                    assert isinstance(effect, dict)
                    definition = effect.get("definition")
                    effect_text = effect_narrative(
                        str(effect.get("name")),
                        estimate,
                        definition=definition if isinstance(definition, str) else None,
                    )
                elif estimate is not None and effect_value is not None and method in _CORRELATIONS:
                    direction = (
                        "positive" if estimate > 0 else "negative" if estimate < 0 else "zero"
                    )
                    assert isinstance(effect, dict)
                    definition = effect.get("definition")
                    effect_text = effect_narrative(
                        str(effect.get("name")),
                        estimate,
                        definition=definition if isinstance(definition, str) else None,
                    )
                elif estimate is not None and effect_value is not None:
                    direction = "positive" if estimate > 0 else "zero"
                    assert isinstance(effect, dict)
                    definition = effect.get("definition")
                    effect_text = effect_narrative(
                        str(effect.get("name")),
                        estimate,
                        definition=definition if isinstance(definition, str) else None,
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
        elif method == "welch_anova" and global_effect_not_applicable:
            effect_text = (
                "No global standardized effect is reported for Welch ANOVA; the recorded group "
                "means and Games-Howell mean differences carry the magnitude information."
            )
            _finding(
                findings,
                "global_effect_not_applicable",
                effect_text,
                "values.group_summaries",
                "values.pairwise_comparisons",
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
                        width = _confidence_interval_width(effect_value, effect_low, effect_high)
                        if width == "moderate width":
                            effect_text += f" Relative to Cohen's d, it has {width}."
                        elif width is not None:
                            effect_text += f" Relative to Cohen's d, it is {width}."
                        _finding(
                            findings,
                            "effect_interval_reported",
                            "A separate standardized-effect interval is available.",
                            "values.effect_size.confidence_interval",
                        )

        interval_text: str | None = None
        interval = values.get("confidence_interval")
        valid_interval = False
        if interval is None and method == "welch_anova":
            interval_text = (
                "An omnibus confidence interval is not applicable; simultaneous Games-Howell "
                "intervals are recorded for every pairwise mean difference."
            )
            _finding(
                findings,
                "pairwise_intervals_reported",
                interval_text,
                "values.pairwise_comparisons",
            )
        elif interval is None:
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
            expected_methods = (
                {
                    "analytical one-sample t interval",
                    "analytical one-sample t interval (degenerate zero-variance sample)",
                }
                if method == "one_sample_t"
                else {"analytical paired t interval"}
                if method == "paired_t"
                else {"analytical t interval"}
                if is_mean_test
                else {"paired-observation percentile bootstrap"}
                if method
                in {
                    "spearman_correlation",
                    "point_biserial_correlation",
                    "kendall_tau_b",
                }
                else {"complete-row percentile bootstrap with model refitting"}
                if method == "partial_pearson_correlation"
                else {"paired-unit percentile bootstrap"}
                if method == "mcnemar"
                else {"observation-row percentile bootstrap"}
                if method == "pearson_chi_square"
                else {"independent within-group percentile bootstrap"}
                if method in {"mann_whitney_u", "one_way_anova", "kruskal_wallis"}
                else set()
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
                or not expected_methods
                or interval.get("method") not in expected_methods
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
                if (
                    method
                    in {
                        "mann_whitney_u",
                        "one_way_anova",
                        "kruskal_wallis",
                        "pearson_chi_square",
                    }
                    and estimate is not None
                ):
                    width = _confidence_interval_width(estimate, low, high)
                    if width == "moderate width":
                        interval_text += f" Relative to the effect estimate, it has {width}."
                    elif width is not None:
                        interval_text += f" Relative to the effect estimate, it is {width}."
                null_value = _finite(result.metadata.get("null_value"))
                if (
                    method in _TWO_SIDED
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
                    method in {"welch_t", "student_t", "paired_t", "one_sample_t"}
                    and p is not None
                    and alpha is not None
                    and null_value is not None
                    and interval.get("method")
                    in {
                        "analytical t interval",
                        "analytical paired t interval",
                        "analytical one-sample t interval",
                        "analytical one-sample t interval (degenerate zero-variance sample)",
                    }
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

        if method in {"welch_anova", "one_way_anova", "kruskal_wallis"}:
            pairwise = values.get("pairwise_comparisons")
            if isinstance(pairwise, list) and pairwise:
                rejected = sum(
                    item.get("decision") == "reject" for item in pairwise if isinstance(item, dict)
                )
                procedure = result.metadata.get("pairwise_method")
                pairwise_text = (
                    f"The complete {procedure} family contained {len(pairwise)} comparisons; "
                    f"{rejected} rejected its pairwise null after the recorded multiplicity "
                    "control. Pairwise results were calculated regardless of the omnibus p-value."
                )
                _finding(
                    findings,
                    "pairwise_family_reported",
                    pairwise_text,
                    "values.pairwise_comparisons",
                    "metadata.multiplicity_control",
                )
            else:
                partial = True
                warnings.append("The complete pairwise follow-up family is unavailable.")
            limitations.append(
                "The omnibus result alone does not identify specific group differences; "
                "interpret the separately multiplicity-controlled pairwise family."
            )
        if method in {
            "pearson_correlation",
            "spearman_correlation",
            "point_biserial_correlation",
            "kendall_tau_b",
            "partial_pearson_correlation",
            "mcnemar",
            "pearson_chi_square",
            "fisher_exact",
        }:
            limitations.append("Observed association alone does not establish causation.")
        if method == "spearman_correlation":
            limitations.append(
                "Spearman correlation describes monotonic rank association, not necessarily "
                "a linear relationship."
            )
        if method == "kendall_tau_b":
            limitations.append(
                "Kendall tau-b describes pairwise ordinal concordance, not linear association "
                "or percent variance explained."
            )
        if method == "partial_pearson_correlation":
            limitations.append(
                "Adjustment describes association conditional on the included controls; it "
                "does not establish that confounding has been removed."
            )
        if method == "mcnemar":
            limitations.append(
                "McNemar inference concerns paired marginal event probabilities and does not "
                "by itself establish a causal condition effect."
            )
        if method == "wilcoxon_signed_rank":
            limitations.append(
                "A location-shift interpretation requires a suitably symmetric paired-"
                "difference distribution; this is not universally a median-difference test."
            )
        if method == "fisher_exact":
            limitations.append(
                "The primary sample odds ratio has no supported confidence interval in this "
                "release, and its direction depends on the recorded level order."
            )
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
