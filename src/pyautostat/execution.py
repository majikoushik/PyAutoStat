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
from .inference_extended import (
    kendall_tau_b as _kendall_tau_b_backend,
)
from .inference_extended import (
    mcnemar_test as _mcnemar_backend,
)
from .inference_extended import (
    point_biserial_correlation as _point_biserial_backend,
)
from .logistic_regression import fit_logit
from .logistic_regression import partial_pearson_correlation as _partial_pearson_backend
from .question_builder import prepare_question
from .recommendation import METHOD_CAPABILITIES, recommend_from_draft
from .report import _json_safe
from .results import AnalysisResult, AnalysisStatus, Recommendation, RecommendationStatus
from .specifications import AnalysisSpecification
from .uncertainty import (
    fisher_z_correlation_ci,
    paired_cohen_dz_ci,
)

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
    "repeated_measures_anova": (
        "Condition sum of squares divided by condition sum of squares plus error sum of "
        "squares (partial eta-squared)."
    ),
    "friedman_test": "Friedman Q divided by n*(k - 1) (Kendall's W rank concordance).",
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
    # Phase 6
    "logistic_regression": "All non-intercept population log-odds coefficients are zero.",
    "mcnemar": "The paired binary outcome proportions are equal (marginal homogeneity).",
    "point_biserial_correlation": "The population point-biserial correlation is zero.",
    "kendall_tau_b": "The population Kendall\u2019s tau-b is zero.",
    "partial_pearson_correlation": (
        "The partial population Pearson correlation is zero, controlling for the declared "
        "covariates."
    ),
    # Phase 7
    "repeated_measures_anova": "All repeated-condition population means are equal.",
    "friedman_test": "The repeated-condition within-unit rank distributions are equal.",
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
    if method_id in ("one_way_anova", "welch_anova", "repeated_measures_anova"):
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
        bootstrap_samples=specification.options.bootstrap_samples,
        random_state=specification.options.random_seed,
    )
    sample = raw["sample"]
    fit = raw["model_fit"]
    diagnostics = raw["diagnostics"]
    r2_ci = fit.get("r_squared_confidence_interval")
    values = {
        "test_statistic": fit["model_f_statistic"],
        "degrees_of_freedom": [
            fit["model_degrees_of_freedom"],
            fit["residual_degrees_of_freedom"],
        ],
        "p_value": fit["model_f_p_value"],
        "primary_estimate": fit["r_squared"],
        "estimate_name": "R-squared",
        "confidence_interval": r2_ci,
        "effect_size": {
            "name": "R-squared",
            "value": fit["r_squared"],
            "definition": "Observed outcome variance accounted for by the fitted in-sample model.",
            "confidence_interval": r2_ci,
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
            "bootstrap": _json_safe(r2_ci) if r2_ci is not None else None,
            "bootstrap_default_resamples": specification.options.bootstrap_samples,
            "effective_random_seed": (
                r2_ci.get("random_seed")
                if isinstance(r2_ci, dict) and r2_ci.get("random_seed") is not None
                else (
                    0
                    if specification.options.random_seed is None
                    else specification.options.random_seed
                )
            ),
        },
        specification=specification,
        recommendation=recommendation,
    )


def _reliability_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    items = specification.question.items
    assert items is not None
    raw = analyzer.scale_reliability(
        items,
        confidence_level=specification.options.confidence_level,
        bootstrap_samples=specification.options.bootstrap_samples,
        random_state=specification.options.random_seed,
        reverse_scoring=specification.options.reverse_scoring,
    )
    frequencies: dict[str, Any] = {}
    dictionary = specification.data_dictionary or {}
    for item in items:
        if dictionary.get(item, {}).get("type") != "ordinal":
            continue
        try:
            frequencies[item] = analyzer.frequency_table(item, data_dictionary=dictionary)
        except PyAutoStatError:
            continue
    values = {
        "primary_estimate": raw["cronbach_alpha"],
        "estimate_name": "Cronbach's alpha",
        "confidence_interval": raw["confidence_interval"],
        "target": raw["target"],
        "items": raw["items"],
        "item_count": raw["item_count"],
        "cronbach_alpha": raw["cronbach_alpha"],
        "item_statistics": raw["item_statistics"],
        "inter_item_correlations": raw["inter_item_correlations"],
        "mean_inter_item_correlation": raw["mean_inter_item_correlation"],
        "negative_inter_item_correlations": raw["negative_inter_item_correlations"],
        "missingness": raw["missingness"],
        "scoring": raw["scoring"],
        "formula": raw["formula"],
        "item_frequencies": frequencies,
    }
    sample = raw["sample"]
    return AnalysisResult(
        method_id="cronbach_alpha",
        status=AnalysisStatus.AVAILABLE,
        sample_size=int(sample["analyzed_rows"]),
        excluded_rows=int(sample["excluded_rows"]),
        values=_json_safe(values),
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, raw["warnings"]),
        metadata={
            "method_name": recommendation.method_name,
            "numerical_source": "NumPy sample covariance and correlation",
            "sample": _json_safe(sample),
            "formula": _json_safe(raw["formula"]),
            "bootstrap": _json_safe(raw["confidence_interval"]),
            "inference": False,
            "hypothesis_test": False,
            "raw_data_included": False,
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
    ci = fisher_z_correlation_ci(
        r,
        count,
        confidence_level=specification.options.confidence_level,
    )
    ci_available = ci if (isinstance(ci, dict) and ci.get("status") == "available") else None
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
                "confidence_interval": ci,
            },
            "confidence_interval": ci_available,
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
    effect_ci = raw.get("effect_size", {}).get("confidence_interval")
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
                "confidence_interval": effect_ci,
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
    effect_ci = paired_cohen_dz_ci(
        mean_difference,
        sd_difference,
        complete_pairs,
        confidence_level=specification.options.confidence_level,
    )
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
                "confidence_interval": effect_ci,
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
        confidence_level=specification.options.confidence_level,
        bootstrap_samples=specification.options.bootstrap_samples,
        random_state=specification.options.random_seed,
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
    interval = raw.get("confidence_interval")
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
                "confidence_interval": interval,
            },
            "confidence_interval": interval,
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
            "bootstrap": raw.get("bootstrap"),
            "bootstrap_default_resamples": specification.options.bootstrap_samples,
            "effective_random_seed": (
                raw["bootstrap"]["random_seed"]
                if isinstance(raw.get("bootstrap"), dict)
                else (
                    0
                    if specification.options.random_seed is None
                    else specification.options.random_seed
                )
            ),
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


def _logistic_regression_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    question = specification.question
    assert question.outcome is not None and question.predictors is not None
    variable_types = recommendation.context.get("variable_types")
    if not isinstance(variable_types, dict):
        raise InsufficientDataError(
            "Logistic regression analytical variable types are unavailable."
        )
    raw = fit_logit(
        analyzer.df,
        question.outcome,
        question.predictors,
        variable_types=variable_types,
        event_level=question.event_level,
        covariance_type=specification.options.covariance_type,
        reference_levels=specification.options.reference_levels,
        data_dictionary=specification.data_dictionary,
        confidence_level=specification.options.confidence_level,
        alpha=specification.options.alpha,
    )
    fit = raw["model_fit"]
    sample = raw["sample"]
    n = sample["analyzed_rows"]
    values = {
        "test_statistic": fit["lr_statistic"],
        "degrees_of_freedom": fit["lr_degrees_of_freedom"],
        "p_value": fit["lr_p_value"],
        "primary_estimate": fit["mcfadden_r2"],
        "estimate_name": "McFadden pseudo-R\u00b2",
        "estimate_unit": None,
        "effect_size": {
            "name": "McFadden pseudo-R\u00b2",
            "value": fit["mcfadden_r2"],
            "definition": (
                "1 minus log-likelihood of fitted model divided by log-likelihood of null model; "
                "a likelihood-based fit index, not comparable with OLS R\u00b2."
            ),
            "confidence_interval": None,
            "status": "available" if fit["mcfadden_r2"] is not None else "unavailable",
        },
        "confidence_interval": None,
        "outcome": raw["outcome"],
        "predictors": raw["predictors"],
        "target": raw["target"],
        "event_level": raw["event_level"],
        "non_event_level": raw["non_event_level"],
        "event_count": raw["event_count"],
        "non_event_count": raw["non_event_count"],
        "event_rate": raw["event_rate"],
        "intercept": True,
        "covariance_type": specification.options.covariance_type,
        "model_fit": fit,
        "coefficients": raw["coefficients"],
        "design_matrix": raw["design_matrix"],
        "diagnostics": raw["diagnostics"],
    }
    return AnalysisResult(
        method_id="logistic_regression",
        status=AnalysisStatus.AVAILABLE,
        sample_size=n,
        excluded_rows=int(sample["excluded_rows"]),
        values=_json_safe(values),
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, raw.get("warnings", [])),
        metadata={
            "method_name": recommendation.method_name,
            "backend_test": raw["method"],
            "numerical_source": "statsmodels.api.Logit",
            "sample": _json_safe(sample),
            "regression_specification": {
                "outcome": raw["outcome"],
                "predictors": raw["predictors"],
                "target": raw["target"],
                "event_level": raw["event_level"],
                "non_event_level": raw["non_event_level"],
                "intercept": True,
                "reference_levels": {
                    item["predictor"]: item["reference_level"]
                    for item in raw["design_matrix"]["coding"]
                    if item.get("reference_level") is not None
                },
                "variable_types": variable_types,
                "complete_case_policy": sample["missing_data_policy"],
            },
            "diagnostics": _json_safe(raw["diagnostics"]),
            "null_hypothesis": _NULL_HYPOTHESES["logistic_regression"],
            "inference": True,
        },
        specification=specification,
        recommendation=recommendation,
    )


def _mcnemar_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    outcome = specification.question.outcome
    condition = specification.question.predictor
    unit_id = specification.unit_id
    assert outcome is not None and condition is not None and unit_id is not None
    seed = specification.options.random_seed
    raw = _mcnemar_backend(
        analyzer.df,
        unit_id,
        condition,
        outcome,
        specification.condition_order,
        event_level=specification.question.event_level,
        confidence_level=specification.options.confidence_level,
        bootstrap_samples=specification.options.bootstrap_samples,
        random_state=0 if seed is None else seed,
    )
    table = raw["transition_table"]
    sample = raw["sample"]
    analyzed = int(sample["analyzed_rows"])
    excluded = int(sample["excluded_rows"])
    statistic = _number(raw["statistic"], "McNemar exact statistic")
    p_value = _number(raw["p_value"], "McNemar p-value", probability=True)
    difference = _number(raw["paired_proportion_difference"], "paired proportion difference")
    interval = _interval(
        raw.get("confidence_interval"),
        "paired proportion difference",
        "paired-unit percentile bootstrap",
        specification.options.confidence_level,
    )
    return AnalysisResult(
        method_id="mcnemar",
        status=AnalysisStatus.AVAILABLE,
        sample_size=analyzed,
        excluded_rows=excluded,
        values=_json_safe(
            {
                "test_statistic": statistic,
                "degrees_of_freedom": None,
                "p_value": p_value,
                "primary_estimate": difference,
                "estimate_name": "paired proportion difference",
                "estimate_unit": "proportion",
                "effect_size": {
                    "name": "paired proportion difference",
                    "value": difference,
                    "definition": (
                        "first-condition event proportion minus second-condition event proportion"
                    ),
                    "confidence_interval": interval,
                },
                "confidence_interval": interval,
                "transition_table": table,
                "event_level": raw["event_level"],
                "non_event_level": raw["non_event_level"],
                "condition_order": raw["condition_order"],
                "first_event_proportion": raw["first_event_proportion"],
                "second_event_proportion": raw["second_event_proportion"],
                "matched_odds_ratio": raw["matched_odds_ratio"],
            }
        ),
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, raw.get("warnings", [])),
        metadata={
            "method_name": recommendation.method_name,
            "numerical_source": "scipy.stats.binomtest",
            "method": raw["method"],
            "statistic_type": raw.get("statistic_type"),
            "sample": {
                **sample,
            },
            "unit_id": unit_id,
            "condition_variable": condition,
            "outcome": outcome,
            "condition_order": raw["condition_order"],
            "group_order": raw["condition_order"],
            "contrast": {
                "first": raw["condition_order"][0],
                "second": raw["condition_order"][1],
                "definition": "first condition minus second condition",
            },
            "event_level": raw["event_level"],
            "null_hypothesis": _NULL_HYPOTHESES["mcnemar"],
            "null_value": 0.0,
            "null_quantity": "paired proportion difference",
            "alternative_hypothesis": "two-sided",
            "diagnostics": {
                "discordance_cell_b": table["discordant_b"],
                "discordance_cell_c": table["discordant_c"],
                "total_discordant": table["total_discordant"],
                "paired_binary_design": "Declared design; not verified from values.",
            },
            "bootstrap": raw["bootstrap"],
            "bootstrap_default_resamples": specification.options.bootstrap_samples,
            "effective_random_seed": raw["bootstrap"]["random_seed"],
        },
        specification=specification,
        recommendation=recommendation,
    )


def _point_biserial_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    first = specification.question.outcome
    second = specification.question.predictor
    assert first is not None and second is not None
    types = recommendation.context.get("variable_types", {})
    if types.get(first) in {"nominal_categorical", "boolean"}:
        binary_col, continuous_col = first, second
    else:
        continuous_col, binary_col = first, second
    seed = specification.options.random_seed
    raw = _point_biserial_backend(
        analyzer.df,
        continuous_col,
        binary_col,
        positive_level=specification.question.event_level,
        confidence_level=specification.options.confidence_level,
        bootstrap_samples=specification.options.bootstrap_samples,
        random_state=0 if seed is None else seed,
    )
    r_pb = _number(raw["point_biserial_r"], "point-biserial r")
    p_value = _number(raw["p_value"], "point-biserial p-value", probability=True)
    analyzed = int(raw["sample_size"])
    excluded = int(raw["excluded_rows"])
    interval = _interval(
        raw.get("confidence_interval"),
        "point-biserial r",
        "paired-observation percentile bootstrap",
        specification.options.confidence_level,
    )
    return AnalysisResult(
        method_id="point_biserial_correlation",
        status=AnalysisStatus.AVAILABLE,
        sample_size=analyzed,
        excluded_rows=excluded,
        values=_json_safe(
            {
                "test_statistic": r_pb,
                "degrees_of_freedom": int(raw["degrees_of_freedom"]),
                "p_value": p_value,
                "primary_estimate": r_pb,
                "estimate_name": "point-biserial r",
                "estimate_unit": None,
                "effect_size": raw["effect_size"],
                "confidence_interval": interval,
                "binary_encoding": raw["binary_encoding"],
                "group_sizes": raw["group_sizes"],
                "group_means": raw["group_means"],
            }
        ),
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, raw.get("warnings", [])),
        metadata={
            "method_name": recommendation.method_name,
            "numerical_source": "scipy.stats.pointbiserialr",
            "sample": {
                "original_rows": len(analyzer.df),
                "analyzed_rows": analyzed,
                "excluded_rows": excluded,
            },
            "variable_order": [first, second],
            "binary_variable": binary_col,
            "continuous_variable": continuous_col,
            "positive_level": raw["binary_encoding"]["positive_level"],
            "null_hypothesis": _NULL_HYPOTHESES["point_biserial_correlation"],
            "null_value": 0.0,
            "null_quantity": "point-biserial r",
            "alternative_hypothesis": "two-sided",
            "diagnostics": {
                "binary_encoding": raw["binary_encoding"],
                "independent_observational_pairs": "Declared design; not verified from values.",
            },
            "bootstrap": raw["bootstrap"],
            "bootstrap_default_resamples": specification.options.bootstrap_samples,
            "effective_random_seed": raw["bootstrap"]["random_seed"],
        },
        specification=specification,
        recommendation=recommendation,
    )


def _kendall_tau_b_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    first = specification.question.outcome
    second = specification.question.predictor
    assert first is not None and second is not None
    seed = specification.options.random_seed
    effective_seed = 0 if seed is None else seed
    raw = _kendall_tau_b_backend(
        analyzer.df,
        first,
        second,
        confidence_level=specification.options.confidence_level,
        bootstrap_samples=specification.options.bootstrap_samples,
        random_state=effective_seed,
    )
    tau = _number(raw["tau_b"], "Kendall's tau-b")
    p_value = _number(raw["p_value"], "Kendall tau-b p-value", probability=True)
    analyzed = int(raw["sample_size"])
    excluded = int(raw["excluded_rows"])
    interval = _interval(
        raw.get("confidence_interval"),
        "Kendall's tau-b",
        "paired-observation percentile bootstrap",
        specification.options.confidence_level,
    )
    return AnalysisResult(
        method_id="kendall_tau_b",
        status=AnalysisStatus.AVAILABLE,
        sample_size=analyzed,
        excluded_rows=excluded,
        values=_json_safe(
            {
                "test_statistic": tau,
                "degrees_of_freedom": None,
                "p_value": p_value,
                "primary_estimate": tau,
                "estimate_name": "Kendall's tau-b",
                "estimate_unit": None,
                "effect_size": raw["effect_size"],
                "confidence_interval": interval,
                "ties": raw["ties"],
            }
        ),
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, raw.get("warnings", [])),
        metadata={
            "method_name": recommendation.method_name,
            "numerical_source": "scipy.stats.kendalltau",
            "sample": {
                "original_rows": len(analyzer.df),
                "analyzed_rows": analyzed,
                "excluded_rows": excluded,
                "effective_pair_count": analyzed,
            },
            "variable_order": [first, second],
            "ties": _json_safe(raw["ties"]),
            "null_hypothesis": _NULL_HYPOTHESES["kendall_tau_b"],
            "null_value": 0.0,
            "null_quantity": "Kendall's tau-b",
            "alternative_hypothesis": "two-sided",
            "diagnostics": {
                "ties": _json_safe(raw["ties"]),
                "independent_observational_pairs": "Declared design; not verified from values.",
            },
            "bootstrap": raw["bootstrap"],
            "bootstrap_default_resamples": specification.options.bootstrap_samples,
            "effective_random_seed": effective_seed,
        },
        specification=specification,
        recommendation=recommendation,
    )


def _partial_pearson_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    first = specification.question.outcome
    second = specification.question.predictor
    assert first is not None and second is not None
    controls = specification.question.controls
    if not controls:
        raise InsufficientDataError(
            "Partial Pearson correlation requires at least one control variable."
        )
    raw = _partial_pearson_backend(
        analyzer.df,
        first,
        second,
        controls,
        confidence_level=specification.options.confidence_level,
        bootstrap_samples=specification.options.bootstrap_samples,
        random_state=(
            0 if specification.options.random_seed is None else specification.options.random_seed
        ),
    )
    pr = _number(raw["partial_r"], "partial Pearson r")
    p_value = _number(raw["p_value"], "partial Pearson p-value", probability=True)
    analyzed = int(raw["sample_size"])
    excluded = int(raw["excluded_rows"])
    interval = _interval(
        raw.get("confidence_interval"),
        "partial Pearson r",
        "complete-row percentile bootstrap with model refitting",
        specification.options.confidence_level,
    )
    return AnalysisResult(
        method_id="partial_pearson_correlation",
        status=AnalysisStatus.AVAILABLE,
        sample_size=analyzed,
        excluded_rows=excluded,
        values=_json_safe(
            {
                "test_statistic": (
                    _number(raw["statistic"], "partial Pearson t statistic")
                    if raw.get("statistic") is not None
                    else None
                ),
                "degrees_of_freedom": int(raw["degrees_of_freedom"]),
                "p_value": p_value,
                "primary_estimate": pr,
                "estimate_name": "partial Pearson r",
                "estimate_unit": None,
                "effect_size": raw["effect_size"],
                "confidence_interval": interval,
                "controls": list(controls),
                "control_design": raw["control_design"],
            }
        ),
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, raw.get("warnings", [])),
        metadata={
            "method_name": recommendation.method_name,
            "numerical_source": "OLS residualisation (statsmodels.api.OLS) + numpy corrcoef",
            "sample": {
                "original_rows": len(analyzer.df),
                "analyzed_rows": analyzed,
                "excluded_rows": excluded,
            },
            "variable_order": [first, second],
            "controls": list(controls),
            "null_hypothesis": _NULL_HYPOTHESES["partial_pearson_correlation"],
            "null_value": 0.0,
            "null_quantity": "partial Pearson r",
            "alternative_hypothesis": "two-sided",
            "diagnostics": {
                "control_count": len(controls),
                "residualisation": "OLS; intercept included",
                "causal_control_note": raw["limitation"],
            },
            "bootstrap": raw["bootstrap"],
            "bootstrap_default_resamples": specification.options.bootstrap_samples,
            "effective_random_seed": raw["bootstrap"]["random_seed"],
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
    raw = analyzer.fisher_exact(
        row_variable,
        column_variable,
        confidence_level=specification.options.confidence_level,
    )
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
    ci = raw.get("confidence_interval")
    ci_available = ci if (isinstance(ci, dict) and ci.get("status") == "available") else None
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
                "confidence_interval": ci,
                "status": raw["odds_ratio_status"],
            },
            "confidence_interval": ci_available,
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
            "null_quantity": "odds ratio",
            "alternative_hypothesis": "association",
            "diagnostics": {
                "table_shape": [2, 2],
                "confidence_interval_status": ("available" if ci is not None else "unavailable"),
                "independent_observations": "Declared design; not verified from values.",
            },
        },
        specification=specification,
        recommendation=recommendation,
    )


def _repeated_measures_anova_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    outcome = specification.question.outcome
    condition = specification.question.predictor
    unit_id = specification.unit_id
    assert outcome is not None and condition is not None and unit_id is not None

    raw = analyzer.repeated_measures_anova(
        unit_id,
        condition,
        outcome,
        condition_order=specification.condition_order,
        alpha=specification.options.alpha,
        confidence_level=specification.options.confidence_level,
    )
    statistic = _number(raw.get("statistic"), "F statistic")
    primary_p = _number(raw.get("primary_p_value"), "primary p-value", probability=True)
    uncorrected_p = _number(raw.get("uncorrected_p_value"), "uncorrected p-value", probability=True)
    corrected_p = _number(raw.get("corrected_p_value"), "corrected p-value", probability=True)

    prim_dfs = raw.get("primary_degrees_of_freedom", {})
    df_list = [prim_dfs.get("numerator"), prim_dfs.get("denominator")]
    degrees_of_freedom = _degrees(df_list, "repeated_measures_anova")

    effect = raw.get("effect_size", {})
    effect_val = _number(effect.get("value"), "partial eta-squared")
    effect_ci = effect.get("confidence_interval")
    effect_size_record = {
        "name": "partial_eta_squared",
        "value": effect_val,
        "definition": _EFFECT_DEFINITIONS["repeated_measures_anova"],
        "confidence_interval": effect_ci,
        "status": "available",
        "reason": None,
    }

    sample_info = raw.get("sample", {})
    analyzed_rows = int(sample_info.get("analyzed_rows", len(analyzer.df)))
    excluded_rows = int(sample_info.get("excluded_rows", 0))

    sphericity = _json_safe(raw.get("sphericity"))
    greenhouse_geisser = _json_safe(raw.get("greenhouse_geisser"))
    condition_summaries = _json_safe(raw.get("condition_summaries"))
    pairwise_comparisons = _json_safe(raw.get("pairwise_comparisons"))
    multiplicity = _json_safe(raw.get("multiplicity"))

    values: dict[str, Any] = {
        "statistic": statistic,
        "test_statistic": statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": primary_p,
        "uncorrected_p_value": uncorrected_p,
        "corrected_p_value": corrected_p,
        "primary_estimate": None,
        "estimate_name": "condition means",
        "estimate_unit": (specification.data_dictionary or {}).get(outcome, {}).get("unit"),
        "effect_size": effect_size_record,
        "confidence_interval": None,
        "sphericity": sphericity,
        "greenhouse_geisser": greenhouse_geisser,
        "sums_of_squares": _json_safe(raw.get("sums_of_squares")),
        "mean_squares": _json_safe(raw.get("mean_squares")),
        "condition_summaries": condition_summaries,
        "pairwise_comparisons": pairwise_comparisons,
        "multiplicity": multiplicity,
        "primary_inference": raw.get("correction_applied"),
        "primary_inference_rule": raw.get("primary_inference_rule"),
    }

    order_labels = [_label(item) for item in raw.get("condition_order", ())]
    return AnalysisResult(
        method_id="repeated_measures_anova",
        status=AnalysisStatus.AVAILABLE,
        sample_size=analyzed_rows,
        excluded_rows=excluded_rows,
        values=values,
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, raw.get("warnings", ())),
        metadata={
            "method_name": recommendation.method_name,
            "backend_test": raw.get("method"),
            "numerical_source": "pyautostat.repeated_measures.repeated_measures_anova",
            "sample": sample_info,
            "unit_id": unit_id,
            "condition_order": order_labels,
            "group_order": order_labels,
            "null_hypothesis": _NULL_HYPOTHESES["repeated_measures_anova"],
            "null_value": 0.0,
            "null_quantity": "condition means",
            "alternative_hypothesis": "at least one condition mean differs",
            "sphericity": sphericity,
            "greenhouse_geisser": greenhouse_geisser,
            "pairwise_method": "paired_t",
            "multiplicity_control": "holm",
            "pairwise_comparison_count": len(pairwise_comparisons)
            if isinstance(pairwise_comparisons, list)
            else 0,
            "primary_inference": raw.get("correction_applied"),
        },
        specification=specification,
        recommendation=recommendation,
    )


def _friedman_result(
    analyzer: StatisticalAnalyzer,
    specification: AnalysisSpecification,
    recommendation: Recommendation,
) -> AnalysisResult:
    outcome = specification.question.outcome
    condition = specification.question.predictor
    unit_id = specification.unit_id
    assert outcome is not None and condition is not None and unit_id is not None

    raw = analyzer.friedman_test(
        unit_id,
        condition,
        outcome,
        condition_order=specification.condition_order,
        alpha=specification.options.alpha,
        confidence_level=specification.options.confidence_level,
        bootstrap_samples=specification.options.bootstrap_samples,
        random_state=specification.options.random_seed,
    )
    statistic = _number(raw.get("statistic"), "Friedman Q statistic")
    p_value = _number(raw.get("p_value"), "Friedman p-value", probability=True)
    df_val = _degrees(raw.get("degrees_of_freedom"), "friedman_test")

    effect = raw.get("effect_size", {})
    effect_val = _number(effect.get("value"), "Kendall's W")
    effect_ci = effect.get("confidence_interval")
    effect_size_record = {
        "name": "Kendall's W",
        "value": effect_val,
        "definition": _EFFECT_DEFINITIONS["friedman_test"],
        "confidence_interval": effect_ci,
        "status": "available",
        "reason": None,
    }

    sample_info = raw.get("sample", {})
    analyzed_rows = int(sample_info.get("analyzed_rows", len(analyzer.df)))
    excluded_rows = int(sample_info.get("excluded_rows", 0))

    condition_summaries = _json_safe(raw.get("condition_summaries"))
    pairwise_comparisons = _json_safe(raw.get("pairwise_comparisons"))
    multiplicity = _json_safe(raw.get("multiplicity"))

    values: dict[str, Any] = {
        "statistic": statistic,
        "test_statistic": statistic,
        "degrees_of_freedom": df_val,
        "p_value": p_value,
        "primary_estimate": None,
        "estimate_name": "within-unit rank distributions",
        "estimate_unit": None,
        "effect_size": effect_size_record,
        "confidence_interval": None,
        "condition_summaries": condition_summaries,
        "pairwise_comparisons": pairwise_comparisons,
        "multiplicity": multiplicity,
    }

    order_labels = [_label(item) for item in raw.get("condition_order", ())]
    return AnalysisResult(
        method_id="friedman_test",
        status=AnalysisStatus.AVAILABLE,
        sample_size=analyzed_rows,
        excluded_rows=excluded_rows,
        values=values,
        assumptions=recommendation.required_assumptions,
        warnings=_warnings(recommendation, raw.get("warnings", ())),
        metadata={
            "method_name": recommendation.method_name,
            "backend_test": raw.get("method"),
            "numerical_source": "pyautostat.repeated_measures.friedman_test",
            "sample": sample_info,
            "unit_id": unit_id,
            "condition_order": order_labels,
            "group_order": order_labels,
            "null_hypothesis": _NULL_HYPOTHESES["friedman_test"],
            "null_value": 0.0,
            "null_quantity": "within-unit rank distributions",
            "alternative_hypothesis": "at least one condition rank distribution differs",
            "pairwise_method": "wilcoxon_signed_rank",
            "multiplicity_control": "holm",
            "pairwise_comparison_count": len(pairwise_comparisons)
            if isinstance(pairwise_comparisons, list)
            else 0,
            "bootstrap": raw.get("bootstrap"),
            "bootstrap_default_resamples": specification.options.bootstrap_samples,
            "effective_random_seed": (
                raw["bootstrap"]["random_seed"]
                if isinstance(raw.get("bootstrap"), dict)
                else (
                    0
                    if specification.options.random_seed is None
                    else specification.options.random_seed
                )
            ),
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
        if method_id == "cronbach_alpha":
            return _reliability_result(analyzer, specification, recommendation)
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
        if method_id == "logistic_regression":
            return _logistic_regression_result(analyzer, specification, recommendation)
        if method_id == "mcnemar":
            return _mcnemar_result(analyzer, specification, recommendation)
        if method_id == "point_biserial_correlation":
            return _point_biserial_result(analyzer, specification, recommendation)
        if method_id == "kendall_tau_b":
            return _kendall_tau_b_result(analyzer, specification, recommendation)
        if method_id == "partial_pearson_correlation":
            return _partial_pearson_result(analyzer, specification, recommendation)
        if method_id == "repeated_measures_anova":
            return _repeated_measures_anova_result(analyzer, specification, recommendation)
        if method_id == "friedman_test":
            return _friedman_result(analyzer, specification, recommendation)
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
        if method_id == "cronbach_alpha":
            return _reliability_result(analyzer, specification, explicit)
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
        if method_id == "logistic_regression":
            return _logistic_regression_result(analyzer, specification, explicit)
        if method_id == "mcnemar":
            return _mcnemar_result(analyzer, specification, explicit)
        if method_id == "point_biserial_correlation":
            return _point_biserial_result(analyzer, specification, explicit)
        if method_id == "kendall_tau_b":
            return _kendall_tau_b_result(analyzer, specification, explicit)
        if method_id == "partial_pearson_correlation":
            return _partial_pearson_result(analyzer, specification, explicit)
        if method_id == "repeated_measures_anova":
            return _repeated_measures_anova_result(analyzer, specification, explicit)
        if method_id == "friedman_test":
            return _friedman_result(analyzer, specification, explicit)
        raise InvalidTestError(f"No execution adapter exists for {method_id!r}.")
    except PyAutoStatError as exc:
        return _unavailable(analyzer, specification, explicit, str(exc))
