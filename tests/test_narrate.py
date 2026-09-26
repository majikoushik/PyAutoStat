"""Deterministic narration consumes recorded effects without calculating them."""

import math

import pandas as pd
import pytest

from pyautostat import (
    ResearchAssistant,
    assumption_grade,
    column_story,
    dataset_opening,
    effect_narrative,
    hypothesis_verdict,
    insight_narrative,
    interval_verdict,
    sensitivity_verdict,
)
from pyautostat.narrate import (
    ASSUMPTION_SEVERITY_DESCRIPTIONS,
    _confidence_interval_narrative,
    _confidence_interval_width,
    _data_quality_story,
    _distribution_story,
    _prioritised_actions,
    _top_correlation_pair,
)


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (0.0, "negligible"),
        (0.1999, "negligible"),
        (0.2, "small"),
        (0.4999, "small"),
        (0.5, "medium"),
        (0.7999, "medium"),
        (0.8, "large"),
        (1.9999, "large"),
        (2.0, "very large"),
        (2.5, "very large"),
    ],
)
def test_cohens_d_boundaries(value, label):
    narrative = effect_narrative("cohens_d", value)
    assert f"labelled '{label}'" in narrative
    assert f"Cohen's d = {value:.4g}" in narrative


@pytest.mark.parametrize(
    ("value", "label"),
    [(-0.2, "small"), (-0.5, "medium"), (-0.8, "large"), (-2.5, "very large")],
)
def test_cohens_d_uses_absolute_magnitude_but_preserves_sign(value, label):
    narrative = effect_narrative("Cohen's d", value)
    assert f"Cohen's d = {value:.4g}" in narrative
    assert f"labelled '{label}'" in narrative


def test_enhancement_plan_example_includes_effect_sample_and_ci_commentary():
    narrative = effect_narrative("cohens_d", 2.5, n=100, ci=(2.1, 2.9))
    assert "very large" in narrative
    assert "2.5" in narrative
    assert "n = 100" in narrative
    assert "confidence interval [2.1, 2.9]" in narrative
    assert "narrow (precise estimate)" in narrative


@pytest.mark.parametrize(
    ("measure", "value", "expected"),
    [
        ("eta_squared", 0.1, "labelled 'medium'"),
        ("cramers_v", 0.4, "labelled 'medium'"),
        ("pearson_r", -0.4, "labelled 'medium'"),
        ("rank_biserial", -0.4, "does not assign a conventional magnitude label"),
        ("epsilon_squared", 0.4, "does not assign a conventional magnitude label"),
        ("Cohen's dz", -0.8, "labelled 'large'"),
    ],
)
def test_supported_effects_reuse_existing_magnitude_policy(measure, value, expected):
    assert expected in effect_narrative(measure, value)


def test_measure_specific_explanations_preserve_direction_and_scope():
    rank = effect_narrative("rank_biserial", -0.4, orientation="'A' minus 'B'")
    pearson = effect_narrative("pearson_r", 0.0)
    cramer = effect_narrative("cramers_v", 0.3)
    epsilon = effect_narrative("epsilon_squared", 0.2)
    assert "'A' minus 'B'" in rank
    assert "not a median difference" in rank
    assert "zero" in pearson
    assert "does not establish population independence" in pearson
    assert "nonnegative" in cramer and "no direction" in cramer
    assert "Kruskal-Wallis" in epsilon


@pytest.mark.parametrize("value", [None, math.nan, math.inf, -math.inf])
def test_unavailable_or_nonfinite_effect_does_not_become_zero(value):
    narrative = effect_narrative("cohens_d", value)
    assert "unavailable" in narrative
    assert "Cohen's d = 0" not in narrative
    assert "negligible" not in narrative


def test_unsupported_and_out_of_range_effects_are_safe():
    assert "unsupported" in effect_narrative("odds_ratio", 1.2)
    assert "outside its valid range" in effect_narrative("pearson_r", 1.1)
    assert "outside its valid range" in effect_narrative("eta_squared", -0.1)


@pytest.mark.parametrize(
    ("estimate", "interval", "expected"),
    [
        (1.0, (0.9, 1.1), "narrow (precise estimate)"),
        (1.0, (0.25, 1.25), "moderate width"),
        (1.0, (-1.0, 2.0), "wide — the true effect is poorly constrained"),
        (-1.0, (-1.1, -0.9), "narrow (precise estimate)"),
        (0.0, (0.0, 0.0), "narrow (precise estimate)"),
        (0.0, (-0.1, 0.1), "wide — the true effect is poorly constrained"),
    ],
)
def test_confidence_interval_width_classification(estimate, interval, expected):
    lower, upper = interval
    assert _confidence_interval_width(estimate, lower, upper) == expected
    assert expected in effect_narrative("cohens_d", estimate, ci=interval)


def test_confidence_interval_uses_recorded_level_and_method():
    ci = {
        "lower": 0.8,
        "upper": 1.2,
        "level": 0.9,
        "method": "independent within-group percentile bootstrap",
    }
    narrative = effect_narrative("cohens_d", 1.0, ci=ci)
    assert "90% bootstrap confidence interval" in narrative
    assert "95%" not in narrative


@pytest.mark.parametrize(
    "ci",
    [
        (None, 1.0),
        (0.0, None),
        (math.nan, 1.0),
        (0.0, math.inf),
        (2.0, 1.0),
        {"lower": 0.0},
        "not an interval",
    ],
)
def test_invalid_or_incomplete_ci_is_omitted_safely(ci):
    narrative = effect_narrative("cohens_d", 1.0, ci=ci)
    assert "confidence interval" not in narrative


def test_ci_helper_does_not_invent_confidence_level():
    narrative = _confidence_interval_narrative(1.0, (0.8, 1.2))
    assert narrative is not None
    assert "confidence interval" in narrative
    assert "%" not in narrative


def test_identical_inputs_produce_identical_text():
    arguments = {
        "measure": "cohens_d",
        "value": -0.8,
        "n": 25,
        "ci": {"lower": -1.0, "upper": -0.6, "level": 0.95},
        "orientation": "first minus second",
    }
    assert effect_narrative(**arguments) == effect_narrative(**arguments)


def test_interpretation_integration_adds_rich_effect_narrative():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 8 + ["B"] * 8,
            "score": [10.0, 11, 12, 13, 14, 15, 16, 17, 1.0, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    assistant = ResearchAssistant(frame)
    result = assistant.analyze(
        assistant.prepare_question(
            objective="compare_groups",
            outcome="score",
            predictor="group",
            estimand="mean",
            design="independent",
            variable_types={"score": "continuous", "group": "nominal"},
        )
    )
    interpretation = assistant.interpret(result)
    assert "statistically significant" in interpretation.hypothesis_interpretation
    assert "conventionally labelled" in interpretation.effect_interpretation
    assert "pooled sample SD" in interpretation.effect_interpretation
    assert "'A' minus 'B'" in interpretation.effect_interpretation
    assert "Relative to Cohen's d" in interpretation.effect_interpretation
    assert all(note.startswith("[") for note in interpretation.assumption_notes)


@pytest.mark.parametrize(
    ("p", "label", "expected"),
    [
        (0.049999, "large", "statistically significant"),
        (0.049999, "small", "even a trivial effect"),
        (0.05, "large", "may be under-powered"),
        (0.050001, "small", "little evidence"),
    ],
)
def test_hypothesis_verdict_covers_four_quadrants_and_strict_alpha(p, label, expected):
    narrative = hypothesis_verdict(
        p, 0.05, 0.8 if label == "large" else 0.1, "cohens_d", label, 5000
    )
    assert expected in narrative
    assert "Cohen's d" in narrative


def test_hypothesis_verdict_formats_small_p_and_large_n_without_rounding_to_zero():
    narrative = hypothesis_verdict(0.000006, 0.05, 0.04, "cohens_d", "small", 5000)
    assert "6e-06" in narrative
    assert "n = 5,000" in narrative
    assert "p = 0.000" not in narrative
    assert narrative == hypothesis_verdict(0.000006, 0.05, 0.04, "cohens_d", "small", 5000)


def test_hypothesis_verdict_does_not_classify_unavailable_or_unknown_effect_as_small():
    assert "four-quadrant verdict cannot be assigned" in hypothesis_verdict(
        0.01, 0.05, None, "cohens_d"
    )
    unknown = hypothesis_verdict(0.01, 0.05, 0.3, "rank_biserial")
    assert "no supported magnitude label" in unknown
    assert "effect is small" not in unknown
    assert "unavailable" in hypothesis_verdict(None, 0.05, 0.8, "cohens_d")


@pytest.mark.parametrize(
    ("n", "severity"),
    [(14, "WARNING"), (15, "CAUTION"), (29, "CAUTION"), (30, "INFO")],
)
def test_assumption_grade_normality_boundaries(n, severity):
    observed, message = assumption_grade(
        "normality", "rejected", n=n, p_value=0.001, group="A", method_id="welch_t"
    )
    assert observed == severity
    assert f"n = {n}" in message
    assert "p = 0.001" in message
    assert ASSUMPTION_SEVERITY_DESCRIPTIONS[severity] in message


def test_assumption_grade_handles_nonrejection_missing_n_welch_and_independence():
    assert assumption_grade("normality", "not_rejected", n=20)[0] == "INFO"
    missing = assumption_grade("normality", "rejected")
    assert missing[0] == "WARNING" and "sample size is unavailable" in missing[1]
    welch = assumption_grade("equal_variance", "rejected", p_value=0.003, method_id="welch_t")
    assert welch[0] == "INFO" and "Welch's correction" in welch[1]
    assert "p = 0.003" in welch[1]
    assert (
        "Levene's test reported"
        not in assumption_grade("equal_variance", "rejected", method_id="welch_t")[1]
    )
    independence = assumption_grade("independence", "confirmed")
    assert independence[0] == "CAUTION" and "study design" in independence[1]
    assert assumption_grade("pairing", "critical")[0] == "CRITICAL"


_POINT_RELATIONS = [
    "positive_meaningful_region",
    "negative_meaningful_region",
    "below_meaningful_magnitude",
    "meets_positive_threshold",
    "below_positive_threshold",
    "meets_negative_threshold",
    "above_negative_threshold",
    "meets_nonnegative_threshold",
    "below_nonnegative_threshold",
]
_INTERVAL_RELATIONS = [
    "entirely_positive_meaningful",
    "entirely_negative_meaningful",
    "entirely_within_negligible_region",
    "spans_both_directions",
    "crosses_meaningful_boundary",
    "entirely_above_positive_threshold",
    "entirely_below_positive_threshold",
    "crosses_positive_threshold",
    "entirely_below_negative_threshold",
    "entirely_above_negative_threshold",
    "crosses_negative_threshold",
    "entirely_at_or_above_nonnegative_threshold",
    "entirely_below_nonnegative_threshold",
    "crosses_nonnegative_threshold",
    "unavailable",
]


@pytest.mark.parametrize("point", _POINT_RELATIONS)
@pytest.mark.parametrize("interval", _INTERVAL_RELATIONS)
def test_interval_verdict_covers_all_current_relation_combinations(point, interval):
    narrative = interval_verdict(point, interval, 6.0, 5.0, quantity="mean_difference")
    assert narrative.startswith("VERDICT:")
    assert "relations are unsupported" not in narrative


@pytest.mark.parametrize(
    ("point", "interval", "label"),
    [
        ("positive_meaningful_region", "entirely_positive_meaningful", "CONFIRMED"),
        ("positive_meaningful_region", "crosses_meaningful_boundary", "LIKELY"),
        ("below_meaningful_magnitude", "entirely_within_negligible_region", "BELOW"),
        ("below_meaningful_magnitude", "crosses_meaningful_boundary", "INCONCLUSIVE"),
    ],
)
def test_interval_verdict_plan_examples(point, interval, label):
    assert label in interval_verdict(point, interval, 6.0, 5.0, quantity="mean_difference")


@pytest.mark.parametrize(
    ("estimate", "phrase"),
    [
        (0.99, "0.99x"),
        (1.0, "ratio = 1.00"),
        (1.99, "ratio = 1.99"),
        (2.0, "2.0 times"),
        (4.99, "5.0 times"),
        (5.0, "5.0x"),
        (-2.0, "2.0 times"),
    ],
)
def test_interval_verdict_ratio_boundaries_and_negative_magnitudes(estimate, phrase):
    narrative = interval_verdict(
        "positive_meaningful_region",
        "entirely_positive_meaningful",
        estimate,
        1.0,
        quantity="mean_difference",
    )
    assert phrase in narrative


def test_interval_verdict_zero_threshold_missing_interval_direction_and_fallback():
    zero = interval_verdict(
        "meets_positive_threshold",
        "unavailable",
        1.0,
        0.0,
        quantity="mean_difference",
        direction="positive",
        orientation="'before' minus 'after'",
    )
    assert "finite magnitude ratio is not reported" in zero
    assert "POINT ESTIMATE ONLY" in zero
    assert "'before' minus 'after'" in zero
    assert "relations are unsupported" in interval_verdict(
        "future_point", "future_interval", 1.0, 0.5, quantity="mean_difference"
    )


@pytest.mark.parametrize(
    ("base", "scenarios", "same", "completed", "mixed", "expected"),
    [
        ("reject null", ["reject null"], True, 1, False, "ROBUST"),
        ("reject null", ["reject null"], True, 2, True, "CONSISTENT across methods"),
        ("reject null", ["reject null", "fail to reject null"], True, 2, True, "INCONSISTENT"),
        ("reject null", ["fail to reject null"], True, 1, False, "INCONSISTENT"),
        ("reject null", [], False, 1, True, "DESCRIPTIVE"),
        ("reject null", [], False, 0, False, "UNAVAILABLE"),
        (None, ["reject null"], True, 1, False, "UNAVAILABLE"),
    ],
)
def test_sensitivity_verdict_states(base, scenarios, same, completed, mixed, expected):
    narrative = sensitivity_verdict(
        base,
        scenarios,
        same_estimand=same,
        completed_scenarios=completed,
        mixed_estimands=mixed,
        alpha=0.05,
    )
    assert expected in narrative
    assert narrative == sensitivity_verdict(
        base,
        scenarios,
        same_estimand=same,
        completed_scenarios=completed,
        mixed_estimands=mixed,
        alpha=0.05,
    )


@pytest.mark.parametrize(
    ("rows", "size_label"),
    [
        (0, "very small"),
        (29, "very small"),
        (30, "small sample"),
        (99, "small sample"),
        (100, "moderate sample"),
        (499, "moderate sample"),
        (500, "medium-sized"),
        (4999, "medium-sized"),
        (5000, "large dataset"),
    ],
)
def test_dataset_opening_size_boundaries(rows, size_label):
    profile = {"overview": {"total_rows": rows, "total_columns": 4}}
    text = dataset_opening(profile)
    assert f"{rows:,}" in text
    assert "4 columns" in text
    assert size_label in text
    assert text == dataset_opening(profile)


def test_dataset_opening_handles_unavailable_dimensions():
    assert "unavailable" in dataset_opening({}).lower()


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (
            {"a": {"a": 1.0, "b": 0.8}, "b": {"a": 0.8, "b": 1.0}},
            ("a", "b", 0.8),
        ),
        (
            {"a": {"b": -0.9}, "b": {"a": -0.9}},
            ("a", "b", -0.9),
        ),
        (
            {"z": {"a": 0.8}, "a": {"z": 0.8, "b": -0.8}, "b": {"a": -0.8}},
            ("a", "b", -0.8),
        ),
        ({"a": {"a": 1.0, "b": 0.4}, "b": {"a": 0.4, "b": 1.0}}, None),
        ({"a": {"a": 1.0, "b": math.nan}, "b": {"a": math.nan}}, None),
    ],
)
def test_top_correlation_pair_is_signed_finite_and_deterministic(matrix, expected):
    profile = {"correlation": {"pearson": {"matrix": matrix}}}
    assert _top_correlation_pair(profile) == expected
    assert _top_correlation_pair(profile) == expected


def test_data_quality_story_covers_missingness_duplicates_and_unavailable_metadata():
    clean = {
        "missing_data": {"total_missing_cells": 0, "by_column": {}},
        "data_quality": {"completeness": 1.0, "duplicate_rows": 0},
    }
    assert "No missing values" in _data_quality_story(clean)
    one = {
        "missing_data": {"by_column": {"x": {"count": 4, "percentage": 20.0}}},
        "data_quality": {"completeness": 0.8, "duplicate_rows": 0},
    }
    assert "concentrated in 'x'" in _data_quality_story(one)
    multiple = {
        "missing_data": {
            "by_column": {
                "x": {"count": 4, "percentage": 20.0},
                "y": {"count": 2, "percentage": 10.0},
            }
        },
        "data_quality": {
            "completeness": 0.7,
            "duplicate_rows": 3,
            "duplicate_rows_percentage": 15.0,
            "issues": [
                {
                    "code": "declared_range_violation",
                    "severity": "review",
                    "section": "data_dictionary",
                    "message": "Two values fall outside the declared range.",
                    "recommendation": "Review values and the declared range.",
                }
            ],
        },
    }
    text = _data_quality_story(multiple)
    assert "Multiple columns" in text
    assert "3 duplicate row" in text
    assert "outside the declared range" in text
    assert "unavailable" in _data_quality_story({}).lower()


@pytest.mark.parametrize(
    ("normality", "phrase"),
    [
        (
            {"a": {"shapiro_wilk": {"status": "not_rejected"}}},
            "did not reject normality",
        ),
        (
            {
                "a": {"shapiro_wilk": {"status": "rejected"}},
                "b": {"shapiro_wilk": {"status": "not_rejected"}},
                "c": {"shapiro_wilk": {"status": "not_rejected"}},
            },
            "some evaluated",
        ),
        (
            {
                "a": {"shapiro_wilk": {"status": "rejected"}},
                "b": {"shapiro_wilk": {"status": "rejected"}},
                "c": {"shapiro_wilk": {"status": "not_rejected"}},
            },
            "most evaluated",
        ),
        (
            {
                "a": {"shapiro_wilk": {"status": "rejected"}},
                "b": {"shapiro_wilk": {"status": "rejected"}},
                "c": {"shapiro_wilk": {"status": "rejected"}},
            },
            "all 3 evaluated",
        ),
        ({}, "unavailable for all"),
    ],
)
def test_distribution_story_covers_rejection_states(normality, phrase):
    columns = list(normality) or ["a", "b"]
    profile = {"descriptive": {column: {} for column in columns}, "normality": normality}
    assert phrase in _distribution_story(profile)


def test_profile_actions_are_prioritized_deduplicated_and_stable():
    profile = {
        "descriptive": {"x": {}, "y": {}},
        "normality": {
            "x": {"shapiro_wilk": {"status": "rejected"}},
            "y": {"shapiro_wilk": {"status": "not_rejected"}},
        },
        "missing_data": {
            "by_column": {
                "x": {"count": 4, "percentage": 40.0},
                "y": {"count": 2, "percentage": 20.0},
            }
        },
        "data_quality": {"duplicate_rows": 2},
        "correlation": {"pearson": {"matrix": {"x": {"y": 0.9}, "y": {"x": 0.9}}}},
    }
    actions = _prioritised_actions(profile)
    assert len(actions) == len(set(actions)) == 3
    assert "missing-data" in actions[0]
    assert "duplicate" in actions[1]
    assert actions == _prioritised_actions(profile)


def test_profile_actions_prioritize_recorded_high_quality_issue_and_detected_outliers():
    profile = {
        "data_quality": {
            "issues": [
                {
                    "code": "custom_recorded_issue",
                    "severity": "high",
                    "section": "data_quality",
                    "recommendation": "Review the recorded high-priority quality issue.",
                }
            ]
        },
        "outliers": {"x": {"iqr": {"percentage": 12.0}}},
    }
    actions = _prioritised_actions(profile)
    assert actions[0] == "Review the recorded high-priority quality issue."
    assert "IQR-flagged" in actions[1]


@pytest.mark.parametrize(
    ("skewness", "shape"),
    [
        (0.1, "approximately symmetric"),
        (0.5, "mild right skew"),
        (1.0, "moderate right skew"),
        (2.0, "strong right skew"),
        (-0.5, "mild left skew"),
        (-1.0, "moderate left skew"),
        (-2.0, "strong left skew"),
    ],
)
def test_column_story_skewness_states(skewness, shape):
    stats = {
        "count": 10,
        "mean": 5.0,
        "median": 5.0,
        "std": 2.0,
        "min": 1.0,
        "max": 9.0,
        "skewness": skewness,
    }
    text = column_story("score", stats, unit="points")
    assert shape in text
    assert "points" in text
    assert text == column_story("score", stats, unit="points")


def test_column_story_mean_median_constant_and_unavailable_states():
    equal = {"count": 3, "mean": 2, "median": 2, "std": 1, "min": 1, "max": 3}
    assert "effectively equal" in column_story("equal", equal)
    small = {"count": 10, "mean": 5.2, "median": 5, "std": 2, "min": 1, "max": 9}
    assert "differ slightly" in column_story("small", small)
    large = {"count": 10, "mean": 8, "median": 5, "std": 2, "min": 1, "max": 10}
    assert "differ materially" in column_story("large", large)
    constant = {"count": 4, "mean": 7, "median": 7, "std": 0, "min": 7, "max": 7}
    assert "constant" in column_story("constant", constant)
    missing_sd = {"count": 4, "mean": 2, "median": 2, "min": 1, "max": 3}
    assert "SD is unavailable" in column_story("missing_sd", missing_sd)
    assert "Skewness is unavailable" in column_story("missing_sd", missing_sd)
    assert "no finite" in column_story("empty", {"count": 0, "mean": None}).lower()
    assert "no finite" in column_story("bad", {"count": 3, "mean": math.inf}).lower()


def test_insight_narrative_uses_lead_other_findings_context_and_severity():
    findings = [
        {"pair": ["b", "c"], "correlation": 0.8},
        {"pair": ["a", "b"], "correlation": -0.95},
        {"pair": ["c", "d"], "correlation": 0.75},
        {"pair": ["d", "e"], "correlation": 0.72},
    ]
    context = {"severity": "medium", "objective": "regression"}
    text = insight_narrative("Multicollinearity", findings, context)
    assert text.startswith("[MEDIUM]")
    assert "'a' and 'b'" in text
    assert "Other findings (3 more)" in text
    assert "standard errors" in text
    assert "'d' and 'e'" not in text
    assert text == insight_narrative("Multicollinearity", findings, context)
