"""Dispatch validated recommendations to existing numerical backends."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy import stats

from .analyzer import StatisticalAnalyzer
from .exceptions import InsufficientDataError, InvalidTestError, PyAutoStatError
from .inference import paired_values
from .question_builder import prepare_question
from .recommendation import METHOD_CAPABILITIES, recommend_from_draft
from .report import _json_safe
from .results import AnalysisResult, AnalysisStatus, Recommendation, RecommendationStatus
from .specifications import AnalysisSpecification

_GROUP_BACKENDS: dict[str, tuple[str, str, bool]] = {
    "welch_t": ("ttest", "mean", False),
    "student_t": ("ttest", "mean", True),
    "mann_whitney_u": ("mannwhitney", "distribution", False),
    "welch_anova": ("welch", "mean", False),
    "one_way_anova": ("anova", "mean", False),
    "kruskal_wallis": ("kruskal", "distribution", False),
}

_EFFECT_DEFINITIONS = {
    "welch_t": "First minus second group mean, divided by the pooled sample SD.",
    "student_t": "First minus second group mean, divided by the pooled sample SD.",
    "mann_whitney_u": "2 times first-group U divided by n1*n2, minus 1; ties count half.",
    "welch_anova": "Not applicable; group means and pairwise mean differences are reported.",
    "one_way_anova": "Between-group sum of squares divided by total sum of squares.",
    "kruskal_wallis": "Truncated rank epsilon-squared from H, group count, and sample size.",
}

_NULL_HYPOTHESES = {
    "welch_t": "The two population means are equal.",
    "student_t": "The two population means are equal.",
    "mann_whitney_u": "The two underlying rank distributions are equal.",
    "welch_anova": "All population means are equal.",
    "one_way_anova": "All group population means are equal.",
    "kruskal_wallis": "All group rank distributions are equal.",
    "pearson_correlation": "The population Pearson linear correlation is zero.",
    "pearson_chi_square": "The two categorical variables are independent.",
    "paired_t": "The population mean paired difference is zero.",
    "one_sample_t": "The population mean equals the declared reference value.",
    "wilcoxon_signed_rank": (
        "The paired-difference distribution is centered at zero under the signed-rank model."
    ),
    "spearman_correlation": "The population Spearman monotonic correlation is zero.",
    "fisher_exact": "The two binary categorical variables are independent.",
    "linear_regression": "All non-intercept population slope coefficients are zero.",
}


def _number(value: Any, name: str, *, probability: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise InsufficientDataError(f"The backend returned an invalid {name}.")
    result = float(value)
    if probability and not 0 <= result <= 1:
        raise InsufficientDataError(f"The backend returned a {name} outside [0, 1].")
    return result


def _label(value: Any) -> str | int | float | bool:
    """Convert a group label to a JSON scalar without changing its order."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise InsufficientDataError("A backend group label is nonfinite.")
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _degrees(value: Any, method_id: str) -> float | int | list[float | int] | None:
    if method_id == "mann_whitney_u":
        return None
    if method_id in ("one_way_anova", "welch_anova"):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise InsufficientDataError("The backend returned invalid ANOVA degrees of freedom.")
        checked = [_number(item, "ANOVA degrees of freedom") for item in value]
        if any(item <= 0 for item in checked):
            raise InsufficientDataError("ANOVA degrees of freedom must be positive.")
        return checked
    checked_value = _number(value, "degrees of freedom")
    if checked_value <= 0:
        raise InsufficientDataError("Degrees of freedom must be positive.")
    return checked_value


def _interval(
    raw: Any, quantity: str, method: str, confidence_level: float
) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise InsufficientDataError("The backend returned an invalid confidence interval.")
    lower = _number(raw.get("lower"), "confidence interval lower bound")
    upper = _number(raw.get("upper"), "confidence interval upper bound")
    level = _number(raw.get("level"), "confidence level", probability=True)
    if lower > upper or not math.isclose(level, confidence_level, abs_tol=1e-12):
        raise InsufficientDataError("The backend returned an inconsistent confidence interval.")
    interval = {key: _json_safe(value) for key, value in raw.items()}
    interval.update({"quantity": quantity, "method": method, "level": level})
    return interval


def _warnings(recommendation: Recommendation, backend: Iterable[str] = ()) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*recommendation.warnings, *backend)))


def _unavailable(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
    reason: str,
) -> AnalysisResult:
    return AnalysisResult(
        method_id=recommendation.method_id or "unselected",
        status=AnalysisStatus.UNAVAILABLE,
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, [reason]),
        metadata={
            "method_name": recommendation.method_name,
            "reason": reason,
            "sample": {
                "original_rows": len(analyzer.df),
                "analyzed_rows": None,
                "excluded_rows": None,
            },
        },
        specification=specification,
        recommendation=recommendation,
    )


def _regression_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    question = specification.question
    assert question.outcome is not None and question.predictors is not None
    variable_types = recommendation.context.get("variable_types")
    if not isinstance(variable_types, dict):
        raise InsufficientDataError("Regression analytical variable types are unavailable.")
    raw = analyzer.linear_regression(
        question.outcome,
        question.predictors,
        variable_types=variable_types,
        covariance_type=specification.options.covariance_type,
        reference_levels=specification.options.reference_levels,
        data_dictionary=specification.data_dictionary,
        confidence_level=specification.options.confidence_level,
        alpha=specification.options.alpha,
    )
    sample = raw["sample"]
    fit = raw["model_fit"]
    diagnostics = raw["diagnostics"]
    values = {
        "test_statistic": fit["model_f_statistic"],
        "degrees_of_freedom": [
            fit["model_degrees_of_freedom"],
            fit["residual_degrees_of_freedom"],
        ],
        "p_value": fit["model_f_p_value"],
        "primary_estimate": fit["r_squared"],
        "estimate_name": "R-squared",
        "confidence_interval": None,
        "effect_size": {
            "name": "R-squared",
            "value": fit["r_squared"],
            "definition": "Observed outcome variance accounted for by the fitted in-sample model.",
            "confidence_interval": None,
            "status": "available",
        },
        "outcome": raw["outcome"],
        "predictors": raw["predictors"],
        "target": raw["target"],
        "covariance_type": raw["covariance_type"],
        "intercept": raw["intercept"],
        "model_fit": fit,
        "coefficients": raw["coefficients"],
        "design_matrix": raw["design_matrix"],
        "diagnostics": diagnostics,
    }
    return AnalysisResult(
        method_id="linear_regression",
        status=AnalysisStatus.AVAILABLE,
        sample_size=int(sample["analyzed_rows"]),
        excluded_rows=int(sample["excluded_rows"]),
        values=_json_safe(values),
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, raw.get("warnings", [])),
        metadata={
            "method_name": recommendation.method_name,
            "backend_test": raw["method"],
            "numerical_source": "statsmodels.api.OLS",
            "sample": _json_safe(sample),
            "regression_specification": {
                "outcome": raw["outcome"],
                "predictors": raw["predictors"],
                "target": raw["target"],
                "covariance_type": raw["covariance_type"],
                "intercept": True,
                "reference_levels": {
                    item["predictor"]: item["reference_level"]
                    for item in raw["design_matrix"]["coding"]
                    if item.get("reference_level") is not None
                },
                "variable_types": variable_types,
                "complete_case_policy": sample["missing_data_policy"],
            },
            "diagnostics": _json_safe(diagnostics),
            "null_hypothesis": _NULL_HYPOTHESES["linear_regression"],
            "inference": True,
        },
        specification=specification,
        recommendation=recommendation,
    )


def _group_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    method_id = recommendation.method_id
    assert method_id is not None
    group_col = specification.question.predictor
    outcome_col = specification.question.outcome
    assert group_col is not None and outcome_col is not None
    backend_type, target, equal_var = _GROUP_BACKENDS[method_id]
    seed = specification.options.random_seed
    effective_seed = 0 if seed is None else seed
    if method_id == "welch_anova":
        raw = analyzer.welch_anova(
            group_col,
            outcome_col,
            confidence_level=specification.options.confidence_level,
            alpha=specification.options.alpha,
        )
    else:
        raw = analyzer.hypothesis_tests(
            group_col,
            outcome_col,
            test_type=backend_type,
            estimand=target,
            equal_var=equal_var,
            confidence_level=specification.options.confidence_level,
            bootstrap_samples=499,
            random_state=effective_seed,
            alpha=specification.options.alpha,
        )
    statistic = _number(raw.get("statistic"), "test statistic")
    p_value = _number(raw.get("p_value"), "p-value", probability=True)
    effect = raw.get("effect_size")
    if not isinstance(effect, dict):
        raise InsufficientDataError("The backend did not provide a valid effect-size record.")
    effect_raw = effect.get("value")
    effect_value = _number(effect_raw, "effect size") if effect_raw is not None else None
    effect_name = effect.get("name")
    if not isinstance(effect_name, str) or not effect_name:
        raise InsufficientDataError("The backend did not name its effect size.")
    effect_interval = _interval(
        effect.get("confidence_interval"),
        effect_name,
        "independent within-group percentile bootstrap",
        specification.options.confidence_level,
    )
    groups = [_label(item) for item in raw["groups"]]
    group_sizes = [
        {"group": _label(item["group"]), "size": int(item["size"])} for item in raw["group_sizes"]
    ]
    analyzed = int(raw["sample_size"])
    excluded = int(raw["excluded_rows"])
    if (
        analyzed + excluded != len(analyzer.df)
        or sum(int(x["size"]) for x in group_sizes) != analyzed
    ):
        raise InsufficientDataError("Backend sample counts disagree with the input dataset.")
    if [item["group"] for item in group_sizes] != groups:
        raise InsufficientDataError("Backend group sizes disagree with its group order.")
    contrast = (
        {"definition": "first group minus second group", "first": groups[0], "second": groups[1]}
        if len(groups) == 2
        else None
    )
    if method_id in ("welch_t", "student_t"):
        primary = _number(raw.get("mean_difference"), "mean difference")
        estimate_name = "mean difference"
        confidence_interval = _interval(
            raw.get("confidence_interval"),
            estimate_name,
            "analytical t interval",
            specification.options.confidence_level,
        )
        if confidence_interval is None:
            raise InsufficientDataError("The t-test backend did not provide its mean interval.")
    elif method_id == "welch_anova":
        primary = None
        estimate_name = "group means"
        confidence_interval = None
    else:
        assert effect_value is not None
        primary = effect_value
        estimate_name = effect_name
        confidence_interval = effect_interval
    unit = (specification.data_dictionary or {}).get(outcome_col, {}).get("unit")
    if method_id not in ("welch_t", "student_t"):
        unit = None
    assumptions = raw.get("assumptions", {})
    if not isinstance(assumptions, dict):
        raise InsufficientDataError("The backend returned invalid assumption diagnostics.")
    diagnostics = _json_safe(assumptions)
    values: dict[str, Any] = {
        "test_statistic": statistic,
        "degrees_of_freedom": _degrees(raw.get("degrees_of_freedom"), method_id),
        "p_value": p_value,
        "primary_estimate": primary,
        "estimate_name": estimate_name,
        "estimate_unit": unit,
        "effect_size": {
            "name": effect_name,
            "value": effect_value,
            "definition": _EFFECT_DEFINITIONS[method_id],
            "confidence_interval": effect_interval,
            "status": effect.get("status", "available"),
            "reason": effect.get("reason"),
        },
        "confidence_interval": confidence_interval,
    }
    pairwise = raw.get("pairwise_comparisons")
    if len(groups) >= 3:
        expected_pairs = len(groups) * (len(groups) - 1) // 2
        if not isinstance(pairwise, list) or len(pairwise) != expected_pairs:
            raise InsufficientDataError("The backend returned an incomplete pairwise family.")
        values["group_summaries"] = _json_safe(raw.get("group_summaries"))
        values["pairwise_comparisons"] = _json_safe(pairwise)
    return AnalysisResult(
        method_id=method_id,
        status=AnalysisStatus.AVAILABLE,
        sample_size=analyzed,
        excluded_rows=excluded,
        values=values,
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, assumptions.get("warnings", [])),
        metadata={
            "method_name": recommendation.method_name,
            "backend_test": raw["test"],
            "numerical_source": "StatisticalAnalyzer.hypothesis_tests",
            "sample": {
                "original_rows": len(analyzer.df),
                "analyzed_rows": analyzed,
                "excluded_rows": excluded,
                "group_sizes": group_sizes,
            },
            "group_order": groups,
            "contrast": contrast,
            "null_hypothesis": _NULL_HYPOTHESES[method_id],
            "null_value": 0.0,
            "null_quantity": estimate_name,
            "alternative_hypothesis": "two-sided"
            if len(groups) == 2
            else "at least one group differs",
            "diagnostics": diagnostics,
            "pairwise_method": raw.get("pairwise_method"),
            "multiplicity_control": raw.get("multiplicity_control"),
            "pairwise_comparison_count": len(pairwise) if isinstance(pairwise, list) else 0,
            "bootstrap_default_resamples": 499,
            "effective_random_seed": effective_seed,
        },
        specification=specification,
        recommendation=recommendation,
    )


def _pearson_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    first = specification.question.outcome
    second = specification.question.predictor
    assert first is not None and second is not None
    subset = analyzer.df[[first, second]]
    declarations = {
        name: value
        for name, value in (specification.data_dictionary or {}).items()
        if name in (first, second)
    }
    profile = StatisticalAnalyzer(subset).analyze_all(
        data_dictionary=declarations if declarations else None
    )
    correlation = profile.get("correlation", {})
    pearson = correlation.get("pearson", {})
    coefficient = pearson.get("matrix", {}).get(first, {}).get(second)
    p_value = correlation.get("p_values", {}).get(first, {}).get(second)
    if coefficient is None or p_value is None:
        raise InsufficientDataError(
            "The Pearson backend did not provide a reliable coefficient and p-value."
        )
    r = _number(coefficient, "Pearson coefficient")
    p = _number(p_value, "Pearson p-value", probability=True)
    count = int(pearson["sample_sizes"][first][second])
    expected = int(subset.dropna().shape[0])
    if count != expected or count < 3 or not -1 <= r <= 1:
        raise InsufficientDataError("Pearson pair counts or coefficient are invalid.")
    relevant_warnings = [
        item["message"]
        for item in profile.get("analysis_warnings", [])
        if item["section"] == "correlation"
    ]
    return AnalysisResult(
        method_id="pearson_correlation",
        status=AnalysisStatus.AVAILABLE,
        sample_size=count,
        excluded_rows=len(analyzer.df) - count,
        values={
            "test_statistic": r,
            "degrees_of_freedom": None,
            "p_value": p,
            "primary_estimate": r,
            "estimate_name": "Pearson r",
            "estimate_unit": None,
            "effect_size": {
                "name": "Pearson r",
                "value": r,
                "definition": "Signed linear correlation between the two quantitative variables.",
                "confidence_interval": None,
            },
            "confidence_interval": None,
        },
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, relevant_warnings),
        metadata={
            "method_name": recommendation.method_name,
            "numerical_source": "StatisticalAnalyzer.analyze_all correlation profile",
            "sample": {
                "original_rows": len(analyzer.df),
                "analyzed_rows": count,
                "excluded_rows": len(analyzer.df) - count,
                "effective_pair_count": count,
            },
            "variable_order": [first, second],
            "null_hypothesis": _NULL_HYPOTHESES["pearson_correlation"],
            "null_value": 0.0,
            "null_quantity": "Pearson r",
            "alternative_hypothesis": "two-sided",
            "diagnostics": {
                "independent_observational_pairs": "Declared design; not verified from values."
            },
        },
        specification=specification,
        recommendation=recommendation,
    )


def _one_sample_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    outcome = specification.question.outcome
    reference = specification.question.reference_value
    assert outcome is not None and reference is not None
    raw = analyzer.one_sample_t_test(
        outcome,
        reference,
        confidence_level=specification.options.confidence_level,
    )
    statistic = (
        _number(raw["statistic"], "one-sample t statistic")
        if raw.get("statistic") is not None
        else None
    )
    p_raw = raw.get("p_value")
    p_value = (
        _number(p_raw, "one-sample t p-value", probability=True) if p_raw is not None else None
    )
    difference = _number(raw["mean_difference"], "mean difference")
    sample_mean = _number(raw["sample_mean"], "sample mean")
    standard_error = _number(raw["standard_error"], "standard error")
    interval_raw = raw.get("confidence_interval")
    if not isinstance(interval_raw, dict):
        raise InsufficientDataError("The one-sample backend did not provide its mean interval.")
    interval = {
        **_json_safe(interval_raw),
        "quantity": "mean difference from reference",
    }
    effect_value = raw.get("effect_size", {}).get("value")
    effect = _number(effect_value, "one-sample Cohen's d") if effect_value is not None else None
    analyzed = int(raw["sample_size"])
    excluded = int(raw["excluded_rows"])
    unit = (specification.data_dictionary or {}).get(outcome, {}).get("unit")
    return AnalysisResult(
        method_id="one_sample_t",
        status=AnalysisStatus.AVAILABLE,
        sample_size=analyzed,
        excluded_rows=excluded,
        values={
            "test_statistic": statistic,
            "degrees_of_freedom": int(raw["degrees_of_freedom"]),
            "p_value": p_value,
            "primary_estimate": difference,
            "estimate_name": "mean difference from reference",
            "estimate_unit": unit,
            "sample_mean": sample_mean,
            "reference_value": float(reference),
            "standard_error": standard_error,
            "effect_size": {
                "name": "one-sample Cohen's d",
                "value": effect,
                "definition": (
                    "Observed sample mean minus reference value, divided by the sample SD."
                ),
                "confidence_interval": None,
                "status": "available" if effect is not None else "unavailable_zero_variance",
            },
            "confidence_interval": interval,
        },
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, raw.get("warnings", [])),
        metadata={
            "method_name": recommendation.method_name,
            "numerical_source": "scipy.stats.ttest_1samp",
            "sample": {
                "original_rows": len(analyzer.df),
                "analyzed_rows": analyzed,
                "excluded_rows": excluded,
            },
            "reference_value": float(reference),
            "contrast": {
                "definition": "observed sample mean minus reference value",
                "reference_value": float(reference),
            },
            "null_hypothesis": _NULL_HYPOTHESES["one_sample_t"],
            "null_value": 0.0,
            "null_quantity": "mean difference from reference",
            "alternative_hypothesis": "two-sided",
            "diagnostics": {
                "sample_mean": sample_mean,
                "sample_standard_deviation": raw["sample_standard_deviation"],
                "standard_error": standard_error,
                "independent_observations": "Declared design; not verified from values.",
            },
        },
        specification=specification,
        recommendation=recommendation,
    )


def _paired_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    outcome = specification.question.outcome
    condition = specification.question.predictor
    unit_id = specification.unit_id
    assert outcome is not None and condition is not None and unit_id is not None
    pairs = paired_values(analyzer.df, unit_id, condition, outcome, specification.condition_order)
    first = pairs["first"]
    second = pairs["second"]
    differences = pairs["differences"]
    order = pairs["condition_order"]
    mean_difference = float(np.mean(differences))
    sd_difference = float(np.std(differences, ddof=1))
    if not math.isfinite(sd_difference) or sd_difference <= 0:
        raise InsufficientDataError("Paired differences need finite nonzero variation.")
    test = stats.ttest_rel(first, second, nan_policy="raise")
    statistic = _number(test.statistic, "paired t statistic")
    p_value = _number(test.pvalue, "paired t p-value", probability=True)
    complete_pairs = int(pairs["complete_pairs"])
    degrees = complete_pairs - 1
    standard_error = sd_difference / math.sqrt(complete_pairs)
    critical = float(stats.t.ppf((1 + specification.options.confidence_level) / 2, degrees))
    if not math.isfinite(critical):
        raise InsufficientDataError("The paired confidence interval critical value is invalid.")
    margin = critical * standard_error
    interval = {
        "quantity": "mean paired difference",
        "lower": mean_difference - margin,
        "upper": mean_difference + margin,
        "level": specification.options.confidence_level,
        "method": "analytical paired t interval",
    }
    effect = mean_difference / sd_difference
    analyzed_rows = int(pairs["analyzed_rows"])
    excluded_rows = int(pairs["excluded_rows"])
    total_units = int(pairs["total_units"])
    incomplete_units = int(pairs["incomplete_units"])
    missing_unit_rows = int(pairs["missing_unit_rows"])
    labels = [_label(item) for item in order]
    contrast = {
        "definition": "first condition minus second condition",
        "first": labels[0],
        "second": labels[1],
    }
    unit = (specification.data_dictionary or {}).get(outcome, {}).get("unit")
    return AnalysisResult(
        method_id="paired_t",
        status=AnalysisStatus.AVAILABLE,
        sample_size=analyzed_rows,
        excluded_rows=excluded_rows,
        values={
            "test_statistic": statistic,
            "degrees_of_freedom": degrees,
            "p_value": p_value,
            "primary_estimate": mean_difference,
            "estimate_name": "mean paired difference",
            "estimate_unit": unit,
            "effect_size": {
                "name": "Cohen's dz",
                "value": effect,
                "definition": (
                    "Mean paired difference divided by the sample SD of paired differences."
                ),
                "confidence_interval": None,
            },
            "confidence_interval": interval,
        },
        assumptions=recommendation.required_assumptions,
        warnings=tuple(recommendation.warnings),
        metadata={
            "method_name": recommendation.method_name,
            "numerical_source": "scipy.stats.ttest_rel",
            "sample": {
                "original_rows": len(analyzer.df),
                "analyzed_rows": analyzed_rows,
                "excluded_rows": excluded_rows,
                "total_units": total_units,
                "complete_pairs": complete_pairs,
                "incomplete_units": incomplete_units,
                "excluded_units": incomplete_units,
                "missing_unit_rows": missing_unit_rows,
                "complete_pair_rule": "one usable observation in each declared condition",
            },
            "unit_id": unit_id,
            "group_order": labels,
            "condition_order": labels,
            "contrast": contrast,
            "null_hypothesis": _NULL_HYPOTHESES["paired_t"],
            "null_value": 0.0,
            "null_quantity": "mean paired difference",
            "alternative_hypothesis": "two-sided",
            "diagnostics": {
                "paired_difference_sd": sd_difference,
                "independent_pairs": "Declared design; not verified from values.",
            },
        },
        specification=specification,
        recommendation=recommendation,
    )


def _wilcoxon_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    outcome = specification.question.outcome
    condition = specification.question.predictor
    unit_id = specification.unit_id
    assert outcome is not None and condition is not None and unit_id is not None
    raw = analyzer.paired_wilcoxon(
        unit_id,
        condition,
        outcome,
        condition_order=specification.condition_order,
    )
    statistic = _number(raw["statistic"], "Wilcoxon statistic")
    p_value = _number(raw["p_value"], "Wilcoxon p-value", probability=True)
    effect = _number(raw["effect_size"]["value"], "matched-pairs rank-biserial correlation")
    labels = [_label(item) for item in raw["condition_order"]]
    contrast = {
        "definition": "first condition minus second condition",
        "first": labels[0],
        "second": labels[1],
    }
    analyzed = int(raw["analyzed_rows"])
    excluded = int(raw["excluded_rows"])
    return AnalysisResult(
        method_id="wilcoxon_signed_rank",
        status=AnalysisStatus.AVAILABLE,
        sample_size=analyzed,
        excluded_rows=excluded,
        values={
            "test_statistic": statistic,
            "degrees_of_freedom": None,
            "p_value": p_value,
            "primary_estimate": effect,
            "estimate_name": "matched-pairs rank-biserial correlation",
            "estimate_unit": None,
            "effect_size": {
                "name": "matched-pairs rank-biserial correlation",
                "value": effect,
                "definition": (
                    "Positive minus negative signed-rank sums divided by their total; positive "
                    "values favor the first declared condition."
                ),
                "confidence_interval": None,
            },
            "confidence_interval": None,
        },
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, raw.get("warnings", [])),
        metadata={
            "method_name": recommendation.method_name,
            "numerical_source": "scipy.stats.wilcoxon",
            "sample": {
                "original_rows": len(analyzer.df),
                "analyzed_rows": analyzed,
                "excluded_rows": excluded,
                "total_units": int(raw["total_units"]),
                "complete_pairs": int(raw["complete_pairs"]),
                "incomplete_units": int(raw["incomplete_units"]),
                "excluded_units": int(raw["incomplete_units"]),
                "missing_unit_rows": int(raw["missing_unit_rows"]),
                "complete_pair_rule": "one usable observation in each declared condition",
                "nonzero_differences": int(raw["nonzero_differences"]),
                "zero_differences": int(raw["zero_differences"]),
            },
            "unit_id": unit_id,
            "group_order": labels,
            "condition_order": labels,
            "contrast": contrast,
            "zero_method": raw["zero_method"],
            "p_value_method": raw["method"],
            "p_value_method_parameter": raw["method_parameter"],
            "p_value_method_detail": raw["method_detail"],
            "rank_sums": {
                "positive": raw["positive_rank_sum"],
                "negative": raw["negative_rank_sum"],
            },
            "null_hypothesis": _NULL_HYPOTHESES["wilcoxon_signed_rank"],
            "null_value": 0.0,
            "null_quantity": "matched-pairs rank-biserial correlation",
            "alternative_hypothesis": "two-sided",
            "diagnostics": {
                "zero_method": "wilcox (zero differences omitted from ranks)",
                "independent_pairs": "Declared design; not verified from values.",
            },
        },
        specification=specification,
        recommendation=recommendation,
    )


def _spearman_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    first = specification.question.outcome
    second = specification.question.predictor
    assert first is not None and second is not None
    seed = specification.options.random_seed
    effective_seed = 0 if seed is None else seed
    raw = analyzer.spearman_correlation(
        first,
        second,
        confidence_level=specification.options.confidence_level,
        bootstrap_samples=499,
        random_state=effective_seed,
    )
    rho = _number(raw["statistic"], "Spearman rho")
    p_value = _number(raw["p_value"], "Spearman p-value", probability=True)
    interval = _interval(
        raw.get("confidence_interval"),
        "Spearman rho",
        "paired-observation percentile bootstrap",
        specification.options.confidence_level,
    )
    analyzed = int(raw["sample_size"])
    excluded = int(raw["excluded_rows"])
    return AnalysisResult(
        method_id="spearman_correlation",
        status=AnalysisStatus.AVAILABLE,
        sample_size=analyzed,
        excluded_rows=excluded,
        values={
            "test_statistic": rho,
            "degrees_of_freedom": None,
            "p_value": p_value,
            "primary_estimate": rho,
            "estimate_name": "Spearman rho",
            "estimate_unit": None,
            "effect_size": {
                "name": "Spearman rho",
                "value": rho,
                "definition": "Signed monotonic rank correlation between paired observations.",
                "confidence_interval": interval,
            },
            "confidence_interval": interval,
        },
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, raw.get("warnings", [])),
        metadata={
            "method_name": recommendation.method_name,
            "numerical_source": "scipy.stats.spearmanr",
            "sample": {
                "original_rows": len(analyzer.df),
                "analyzed_rows": analyzed,
                "excluded_rows": excluded,
                "effective_pair_count": analyzed,
            },
            "variable_order": [first, second],
            "ties": _json_safe(raw["ties"]),
            "null_hypothesis": _NULL_HYPOTHESES["spearman_correlation"],
            "null_value": 0.0,
            "null_quantity": "Spearman rho",
            "alternative_hypothesis": "two-sided",
            "diagnostics": {
                "ties": _json_safe(raw["ties"]),
                "independent_observational_pairs": "Declared design; not verified from values.",
            },
            "bootstrap_default_resamples": 499,
            "effective_random_seed": effective_seed,
        },
        specification=specification,
        recommendation=recommendation,
    )


def _categorical_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    outcome = specification.question.outcome
    predictor = specification.question.predictor
    assert outcome is not None and predictor is not None
    seed = specification.options.random_seed
    effective_seed = 0 if seed is None else seed
    raw = analyzer.categorical_association(
        predictor,
        outcome,
        confidence_level=specification.options.confidence_level,
        bootstrap_samples=499,
        random_state=effective_seed,
    )
    statistic = _number(raw.get("statistic"), "chi-square statistic")
    p_value = _number(raw.get("p_value"), "chi-square p-value", probability=True)
    effect = raw.get("effect_size", {})
    magnitude = _number(effect.get("value"), "Cramer's V")
    interval = _interval(
        effect.get("confidence_interval"),
        "Cramer's V",
        "observation-row percentile bootstrap",
        specification.options.confidence_level,
    )
    analyzed = int(raw["sample_size"])
    excluded = int(raw["excluded_rows"])
    observed = raw["observed_counts"]
    if analyzed + excluded != len(analyzer.df) or sum(map(sum, observed)) != analyzed:
        raise InsufficientDataError("Backend contingency counts disagree with the dataset.")
    groups = [_label(item) for item in raw["groups"]]
    outcomes = [_label(item) for item in raw["outcomes"]]
    group_sizes = [
        {"group": group, "size": int(sum(row))} for group, row in zip(groups, observed, strict=True)
    ]
    assumptions = raw.get("assumptions", {})
    return AnalysisResult(
        method_id="pearson_chi_square",
        status=AnalysisStatus.AVAILABLE,
        sample_size=analyzed,
        excluded_rows=excluded,
        values={
            "test_statistic": statistic,
            "degrees_of_freedom": int(raw["degrees_of_freedom"]),
            "p_value": p_value,
            "primary_estimate": magnitude,
            "estimate_name": "Cramer's V",
            "estimate_unit": None,
            "effect_size": {
                "name": "Cramer's V",
                "value": magnitude,
                "definition": "Square root of chi-square divided by N times the smaller axis df.",
                "confidence_interval": interval,
            },
            "confidence_interval": interval,
        },
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, assumptions.get("warnings", [])),
        metadata={
            "method_name": recommendation.method_name,
            "numerical_source": "StatisticalAnalyzer.categorical_association",
            "sample": {
                "original_rows": len(analyzer.df),
                "analyzed_rows": analyzed,
                "excluded_rows": excluded,
                "group_sizes": group_sizes,
            },
            "group_order": groups,
            "outcome_order": outcomes,
            "observed_counts": _json_safe(observed),
            "expected_counts": _json_safe(raw["expected_counts"]),
            "null_hypothesis": _NULL_HYPOTHESES["pearson_chi_square"],
            "null_value": 0.0,
            "null_quantity": "Cramer's V",
            "alternative_hypothesis": "association",
            "diagnostics": _json_safe(assumptions),
            "bootstrap_default_resamples": 499,
            "effective_random_seed": effective_seed,
        },
        specification=specification,
        recommendation=recommendation,
    )


def _fisher_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    row_variable = specification.question.outcome
    column_variable = specification.question.predictor
    assert row_variable is not None and column_variable is not None
    raw = analyzer.fisher_exact(row_variable, column_variable)
    odds_raw = raw.get("odds_ratio")
    odds_ratio = _number(odds_raw, "sample odds ratio") if odds_raw is not None else None
    p_value = _number(raw["p_value"], "Fisher p-value", probability=True)
    observed = raw["observed_counts"]
    analyzed = int(raw["sample_size"])
    excluded = int(raw["excluded_rows"])
    if (
        len(observed) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in observed)
        or sum(map(sum, observed)) != analyzed
        or analyzed + excluded != len(analyzer.df)
    ):
        raise InsufficientDataError("Fisher contingency counts disagree with the dataset.")
    row_levels = [_label(item) for item in raw["row_levels"]]
    column_levels = [_label(item) for item in raw["column_levels"]]
    return AnalysisResult(
        method_id="fisher_exact",
        status=AnalysisStatus.AVAILABLE,
        sample_size=analyzed,
        excluded_rows=excluded,
        values={
            "test_statistic": odds_ratio,
            "degrees_of_freedom": None,
            "p_value": p_value,
            "primary_estimate": odds_ratio,
            "estimate_name": "sample odds ratio",
            "estimate_unit": None,
            "effect_size": {
                "name": "sample odds ratio",
                "value": odds_ratio,
                "definition": raw["odds_ratio_definition"],
                "confidence_interval": None,
                "status": raw["odds_ratio_status"],
            },
            "confidence_interval": None,
        },
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, raw.get("warnings", [])),
        metadata={
            "method_name": recommendation.method_name,
            "numerical_source": "scipy.stats.fisher_exact",
            "sample": {
                "original_rows": len(analyzer.df),
                "analyzed_rows": analyzed,
                "excluded_rows": excluded,
                "row_totals": [int(sum(row)) for row in observed],
            },
            "variable_order": [row_variable, column_variable],
            "row_variable": row_variable,
            "column_variable": column_variable,
            "row_order": row_levels,
            "column_order": column_levels,
            "row_ordering": raw["row_ordering"],
            "column_ordering": raw["column_ordering"],
            "observed_counts": _json_safe(observed),
            "odds_ratio_status": raw["odds_ratio_status"],
            "null_hypothesis": _NULL_HYPOTHESES["fisher_exact"],
            "null_value": 1.0,
            "null_quantity": "sample odds ratio",
            "alternative_hypothesis": "association",
            "diagnostics": {
                "table_shape": [2, 2],
                "confidence_interval_status": "not_supported",
                "independent_observations": "Declared design; not verified from values.",
            },
        },
        specification=specification,
        recommendation=recommendation,
    )


def execute_specification(
    analyzer: StatisticalAnalyzer, specification: AnalysisSpecification
) -> AnalysisResult:
    """Revalidate the request and execute only its freshly selected capability."""
    draft = prepare_question(analyzer.df, specification=specification)
    recommendation = recommend_from_draft(analyzer.df, draft)
    specification = draft.specification
    if (
        recommendation.status != RecommendationStatus.READY
        or recommendation.method_availability != "runnable"
    ):
        reason = (
            "; ".join(recommendation.blockers)
            or recommendation.rationale
            or ("Essential research information is unresolved.")
        )
        return _unavailable(analyzer, specification, recommendation, reason)
    method_id = recommendation.method_id
    if (
        method_id not in METHOD_CAPABILITIES
        or method_id is None
        or METHOD_CAPABILITIES[method_id].availability != "runnable"
    ):
        return _unavailable(
            analyzer, specification, recommendation, "Unknown selected method; no analysis ran."
        )
    try:
        if method_id == "dataset_profile":
            profile = analyzer.analyze_all(data_dictionary=specification.data_dictionary)
            return AnalysisResult(
                method_id=method_id,
                status=AnalysisStatus.AVAILABLE,
                sample_size=len(analyzer.df),
                excluded_rows=0,
                values={
                    "profile": _json_safe(profile),
                    "test_statistic": None,
                    "p_value": None,
                    "primary_estimate": None,
                    "confidence_interval": None,
                },
                warnings=_warnings(
                    recommendation,
                    [item["message"] for item in profile.get("analysis_warnings", [])],
                ),
                metadata={
                    "method_name": recommendation.method_name,
                    "numerical_source": "StatisticalAnalyzer.analyze_all",
                    "inference": False,
                    "sample": {
                        "original_rows": len(analyzer.df),
                        "analyzed_rows": len(analyzer.df),
                        "excluded_rows": 0,
                    },
                },
                specification=specification,
                recommendation=recommendation,
            )
        if method_id == "linear_regression":
            return _regression_result(analyzer, specification, recommendation)
        if method_id in _GROUP_BACKENDS:
            return _group_result(analyzer, specification, recommendation)
        if method_id == "one_sample_t":
            return _one_sample_result(analyzer, specification, recommendation)
        if method_id == "paired_t":
            return _paired_result(analyzer, specification, recommendation)
        if method_id == "wilcoxon_signed_rank":
            return _wilcoxon_result(analyzer, specification, recommendation)
        if method_id == "pearson_correlation":
            return _pearson_result(analyzer, specification, recommendation)
        if method_id == "spearman_correlation":
            return _spearman_result(analyzer, specification, recommendation)
        if method_id == "pearson_chi_square":
            return _categorical_result(analyzer, specification, recommendation)
        if method_id == "fisher_exact":
            return _fisher_result(analyzer, specification, recommendation)
        raise InvalidTestError(f"No execution adapter exists for {method_id!r}.")
    except PyAutoStatError as exc:
        return _unavailable(analyzer, specification, recommendation, str(exc))


def execute_selected_method(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    method_id: str,
) -> AnalysisResult:
    """Execute one explicitly selected, compatible method.

    This entry point exists for declared sensitivity specifications. It does not
    participate in automatic recommendation and it never substitutes another
    method when the requested method is unavailable.
    """
    if not isinstance(method_id, str) or not method_id.strip():
        raise InvalidTestError("A sensitivity method_id must be non-empty text.")
    draft = prepare_question(analyzer.df, specification=specification)
    specification = draft.specification
    ordinary = recommend_from_draft(analyzer.df, draft)
    capability = METHOD_CAPABILITIES.get(method_id)
    if capability is None or capability.availability != "runnable":
        return _unavailable(
            analyzer,
            specification,
            ordinary,
            f"Requested sensitivity method {method_id!r} is not a runnable capability.",
        )
    question = specification.question
    objective = question.objective.value if question.objective is not None else None
    if (
        draft.status.value != "ready"
        or objective != capability.objective
        or question.estimand != capability.target
        or specification.design.value not in capability.designs
    ):
        reason = "; ".join(draft.blockers) or (
            f"Method {method_id!r} is incompatible with the declared objective, "
            "estimand, or study design."
        )
        return _unavailable(analyzer, specification, ordinary, reason)
    explicit = Recommendation(
        status=RecommendationStatus.READY,
        method_id=method_id,
        method_name=capability.name,
        rationale=(
            "Explicitly requested sensitivity method; this selection was not made "
            "from its p-value or diagnostics."
        ),
        required_assumptions=capability.assumptions,
        warnings=ordinary.warnings,
        method_availability="runnable",
        decision_trace=(
            {
                "key": "sensitivity_method",
                "value": method_id,
                "reason": "Researcher-declared analytical variation.",
            },
        ),
        context={
            "objective": objective,
            "estimand": question.estimand,
            "design": specification.design.value,
            "selection": "explicit_sensitivity_scenario",
        },
    )
    try:
        if method_id == "linear_regression":
            return _regression_result(analyzer, specification, explicit)
        if method_id in _GROUP_BACKENDS:
            return _group_result(analyzer, specification, explicit)
        if method_id == "one_sample_t":
            return _one_sample_result(analyzer, specification, explicit)
        if method_id == "paired_t":
            return _paired_result(analyzer, specification, explicit)
        if method_id == "wilcoxon_signed_rank":
            return _wilcoxon_result(analyzer, specification, explicit)
        if method_id == "pearson_correlation":
            return _pearson_result(analyzer, specification, explicit)
        if method_id == "spearman_correlation":
            return _spearman_result(analyzer, specification, explicit)
        if method_id == "pearson_chi_square":
            return _categorical_result(analyzer, specification, explicit)
        if method_id == "fisher_exact":
            return _fisher_result(analyzer, specification, explicit)
        raise InvalidTestError(f"No execution adapter exists for {method_id!r}.")
    except PyAutoStatError as exc:
        return _unavailable(analyzer, specification, explicit, str(exc))
