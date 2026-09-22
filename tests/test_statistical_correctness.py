"""Independent arithmetic references and statistical boundary cases for Phase 2."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, f, norm, t

from pyautostat import (
    InsufficientDataError,
    InvalidTestError,
    ResearchAssistant,
    StatisticalAnalyzer,
)


def _groups(*values):
    return pd.DataFrame(
        {
            "group": [f"G{index}" for index, group in enumerate(values) for _ in group],
            "value": [item for group in values for item in group],
        }
    )


def test_welch_t_and_student_t_against_hand_formula():
    frame = _groups([1, 2, 3, 4], [3, 5, 7, 9])
    analyzer = StatisticalAnalyzer(frame)
    n1 = n2 = 4
    v1, v2 = 5 / 3, 20 / 3
    difference = -3.5
    pooled_variance = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    expected_d = difference / math.sqrt(pooled_variance)
    expected_se = math.sqrt(v1 / n1 + v2 / n2)
    expected_t = difference / expected_se
    expected_df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))

    for equal_var, df in ((False, expected_df), (True, 6)):
        result = analyzer.hypothesis_tests(
            "group",
            "value",
            "ttest",
            equal_var=equal_var,
            bootstrap_samples=0,
            confidence_level=0.9,
        )
        margin = t.ppf(0.95, df) * expected_se
        assert result["statistic"] == pytest.approx(expected_t)
        assert result["p_value"] == pytest.approx(2 * t.sf(abs(expected_t), df))
        assert result["degrees_of_freedom"] == pytest.approx(df)
        assert result["mean_difference"] == pytest.approx(difference)
        assert result["effect_size"]["value"] == pytest.approx(expected_d)
        assert result["confidence_interval"]["lower"] == pytest.approx(difference - margin)
        assert result["confidence_interval"]["upper"] == pytest.approx(difference + margin)
        assert result["sample_size"] == 8
        assert result["excluded_rows"] == 0
        assert result["equal_variance"] is equal_var


def test_welch_unbalanced_groups_against_hand_standard_error():
    first, second = [1, 2, 3], [2, 4, 6, 8, 10]
    result = StatisticalAnalyzer(_groups(first, second)).hypothesis_tests(
        "group", "value", "ttest", bootstrap_samples=0
    )
    v1, v2 = 1, 10
    difference = 2 - 6
    se = math.sqrt(v1 / 3 + v2 / 5)
    df = (v1 / 3 + v2 / 5) ** 2 / ((v1 / 3) ** 2 / 2 + (v2 / 5) ** 2 / 4)
    assert result["statistic"] == pytest.approx(difference / se)
    assert result["degrees_of_freedom"] == pytest.approx(df)
    assert result["confidence_interval"]["upper"] == pytest.approx(
        difference + t.ppf(0.975, df) * se
    )


def test_t_direction_reverses_and_one_constant_group_is_valid():
    frame = _groups([1, 2, 3, 4], [3, 5, 7, 9])
    forward = StatisticalAnalyzer(frame).hypothesis_tests(
        "group", "value", "ttest", bootstrap_samples=0
    )
    reversed_frame = pd.concat([frame.iloc[4:], frame.iloc[:4]], ignore_index=True)
    reverse = StatisticalAnalyzer(reversed_frame).hypothesis_tests(
        "group", "value", "ttest", bootstrap_samples=0
    )
    assert reverse["statistic"] == pytest.approx(-forward["statistic"])
    assert reverse["mean_difference"] == pytest.approx(-forward["mean_difference"])
    assert reverse["effect_size"]["value"] == pytest.approx(-forward["effect_size"]["value"])
    assert reverse["confidence_interval"]["lower"] == pytest.approx(
        -forward["confidence_interval"]["upper"]
    )
    constant = StatisticalAnalyzer(_groups([1, 1, 1], [1, 2, 3])).hypothesis_tests(
        "group", "value", "ttest", bootstrap_samples=0
    )
    assert math.isfinite(constant["statistic"])


def test_auto_requires_target_and_does_not_follow_normality_screen():
    frame = _groups([1, 1, 1, 1, 1, 20], [2, 2, 2, 2, 2, 30])
    analyzer = StatisticalAnalyzer(frame)
    with pytest.raises(InvalidTestError, match="estimand"):
        analyzer.hypothesis_tests("group", "value", bootstrap_samples=0)
    mean = analyzer.hypothesis_tests("group", "value", estimand="mean", bootstrap_samples=0)
    rank = analyzer.hypothesis_tests("group", "value", estimand="distribution", bootstrap_samples=0)
    assert mean["test"] == "t-test"
    assert mean["equal_variance"] is False
    assert rank["test"] == "Mann-Whitney U"
    assert mean["assumptions"]["estimand"] == "mean"
    assert rank["assumptions"]["estimand"] == "distribution"
    assert "cannot be verified" in mean["assumptions"]["independent_observations"]
    with pytest.raises(InvalidTestError, match="Welch ANOVA"):
        StatisticalAnalyzer(_groups([1, 2, 3], [2, 3, 4], [3, 4, 5])).hypothesis_tests(
            "group", "value", estimand="mean", bootstrap_samples=0
        )
    with pytest.raises(InvalidTestError, match="does not target"):
        analyzer.hypothesis_tests("group", "value", "mannwhitney", estimand="mean")


def test_mann_whitney_ties_u_and_rank_direction():
    frame = _groups([1, 2, 2, 5, 7], [2, 3, 4, 6, 8])
    result = StatisticalAnalyzer(frame).hypothesis_tests(
        "group", "value", "mannwhitney", bootstrap_samples=0
    )
    # Count all A>B pairs plus half the three A=B pairs: U=8.
    u = 8
    variance = 25 / 12 * (11 - 24 / 90)  # One tie of size three.
    expected_p = 2 * norm.sf((12.5 - u - 0.5) / math.sqrt(variance))
    assert result["statistic"] == pytest.approx(u)
    assert result["p_value"] == pytest.approx(expected_p)
    assert result["effect_size"]["value"] == pytest.approx(-0.36)
    assert result["effect_size"]["interpretation"] is None
    assert any("Small tied" in item for item in result["assumptions"]["warnings"])
    reversed_frame = pd.concat([frame.iloc[5:], frame.iloc[:5]], ignore_index=True)
    reverse = StatisticalAnalyzer(reversed_frame).hypothesis_tests(
        "group", "value", "mannwhitney", bootstrap_samples=0
    )
    assert reverse["effect_size"]["value"] == pytest.approx(0.36)
    assert reverse["p_value"] == pytest.approx(result["p_value"])


def test_anova_f_eta_and_kruskal_h_epsilon_from_rank_sums():
    anova = StatisticalAnalyzer(_groups([1, 2, 3], [2, 4, 6], [5, 7, 9])).hypothesis_tests(
        "group", "value", "anova", bootstrap_samples=0
    )
    assert anova["statistic"] == pytest.approx(19 / 3)
    assert anova["p_value"] == pytest.approx(f.sf(19 / 3, 2, 6))
    assert anova["degrees_of_freedom"] == [2, 6]
    assert anova["effect_size"]["value"] == pytest.approx(38 / 56)

    rank = StatisticalAnalyzer(
        _groups([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15])
    ).hypothesis_tests("group", "value", "kruskal", bootstrap_samples=0)
    assert rank["statistic"] == pytest.approx(12.5)
    assert rank["p_value"] == pytest.approx(chi2.sf(12.5, 2))
    assert rank["degrees_of_freedom"] == 2
    assert rank["effect_size"]["value"] == pytest.approx(10.5 / 12)
    assert rank["effect_size"]["interpretation"] is None


def test_chi_square_and_categorical_effects_from_hand_table():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 40 + ["B"] * 40,
            "outcome": ["yes"] * 30 + ["no"] * 10 + ["yes"] * 10 + ["no"] * 30,
        }
    )
    result = StatisticalAnalyzer(frame).categorical_association(
        "group", "outcome", success_value="yes", bootstrap_samples=0
    )
    assert result["observed_counts"] == [[30, 10], [10, 30]]
    assert result["expected_counts"] == [[20, 20], [20, 20]]
    assert result["statistic"] == pytest.approx(20)
    assert result["p_value"] == pytest.approx(chi2.sf(20, 1))
    assert result["degrees_of_freedom"] == 1
    assert result["effect_size"]["value"] == pytest.approx(0.5)
    assert result["cohens_h"]["value"] == pytest.approx(math.pi / 3)
    assert result["excluded_rows"] == 0
    reversed_frame = pd.concat([frame.iloc[40:], frame.iloc[:40]], ignore_index=True)
    reverse = StatisticalAnalyzer(reversed_frame).categorical_association(
        "group", "outcome", success_value="yes", bootstrap_samples=0
    )
    assert reverse["cohens_h"]["value"] == pytest.approx(-math.pi / 3)
    assert reverse["effect_size"]["value"] == pytest.approx(0.5)

    with_missing = pd.concat(
        [frame, pd.DataFrame({"group": ["A"], "outcome": [None]})], ignore_index=True
    )
    excluded = StatisticalAnalyzer(with_missing).categorical_association(
        "group", "outcome", success_value="yes", bootstrap_samples=0
    )
    assert excluded["sample_size"] == 80
    assert excluded["excluded_rows"] == 1


def test_categorical_bootstrap_is_reproducible_and_counts_draws():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 40 + ["B"] * 40,
            "outcome": ["yes"] * 30 + ["no"] * 10 + ["yes"] * 10 + ["no"] * 30,
        }
    )
    analyzer = StatisticalAnalyzer(frame)
    first = analyzer.categorical_association(
        "group", "outcome", success_value="yes", bootstrap_samples=100, random_state=13
    )
    second = analyzer.categorical_association(
        "group", "outcome", success_value="yes", bootstrap_samples=100, random_state=13
    )
    assert first == second
    interval = first["effect_size"]["confidence_interval"]
    assert interval["method"] == "observation-row percentile bootstrap"
    assert interval["requested_resamples"] == 100
    assert interval["valid_resamples"] <= 100
    assert interval["random_seed"] == 13
    assert math.isfinite(interval["lower"]) and math.isfinite(interval["upper"])
    assert first["cohens_h"]["confidence_interval"]["valid_resamples"] <= 100

    original = np.random.get_state()
    try:
        np.random.seed(321)
        before = np.random.get_state()
        analyzer.categorical_association(
            "group",
            "outcome",
            success_value="yes",
            bootstrap_samples=100,
            random_state=13,
        )
        after = np.random.get_state()
        assert np.array_equal(before[1], after[1])
        assert before[2:] == after[2:]
    finally:
        np.random.set_state(original)


def test_correlation_reference_and_pairwise_missing():
    frame = pd.DataFrame(
        {"x": [1, 2, 3, 4, np.nan], "y": [1, 3, 2, 4, 99], "negative": [4, 3, 2, 1, 0]}
    )
    correlations = StatisticalAnalyzer(frame).analyze_all()["correlation"]
    assert correlations["pearson"]["matrix"]["x"]["y"] == pytest.approx(0.8)
    assert correlations["spearman"]["matrix"]["x"]["y"] == pytest.approx(0.8)
    assert correlations["kendall"]["matrix"]["x"]["y"] == pytest.approx(2 / 3)
    assert correlations["p_values"]["x"]["y"] == pytest.approx(0.2)
    assert correlations["pearson"]["matrix"]["x"]["negative"] == pytest.approx(-1)


def test_nearly_constant_pearson_is_unavailable_with_warning():
    frame = pd.DataFrame({"x": [1e12 + index * 1e-3 for index in range(5)], "y": [1, 2, 3, 4, 5]})
    profile = StatisticalAnalyzer(frame).analyze_all()
    assert profile["correlation"]["pearson"]["matrix"]["x"]["y"] is None
    assert profile["correlation"]["p_values"]["x"]["y"] is None
    assert any(
        item["section"] == "correlation" and item["code"] == "numerical_warning"
        for item in profile["analysis_warnings"]
    )
    assert profile["outliers"]["x"]["z_score"]["count"] is None


def test_profile_normality_and_outlier_reference_values():
    frame = pd.DataFrame({"x": [1, 2, 3, 4, 20], "short": [1, 2, np.nan, np.nan, np.nan]})
    profile = ResearchAssistant(frame).profile()
    desc = profile["descriptive"]["x"]
    assert desc["mean"] == pytest.approx(6)
    assert desc["median"] == 3
    assert desc["variance"] == pytest.approx(62.5)
    assert desc["q1"] == 2
    assert desc["q3"] == 4
    outliers = profile["outliers"]["x"]
    assert outliers["iqr"]["lower_bound"] == -1
    assert outliers["iqr"]["upper_bound"] == 7
    assert outliers["iqr"]["count"] == 1
    assert profile["normality"]["x"]["shapiro_wilk"]["status"] in ("not_rejected", "rejected")
    assert "short" not in profile["normality"]


def test_shapiro_three_point_statistic_and_anderson_critical_grid():
    profile = StatisticalAnalyzer(pd.DataFrame({"x": [1.0, 2.0, 4.0]})).analyze_all()
    normality = profile["normality"]["x"]
    # For n=3 the Shapiro numerator is (max-min)^2/2; denominator is 14/3.
    assert normality["shapiro_wilk"]["statistic"] == pytest.approx(27 / 28)
    assert normality["shapiro_wilk"]["status"] == "not_rejected"
    # SciPy releases differ here: some return negative critical values for n=3.
    if "anderson_darling" in normality:
        anderson = normality["anderson_darling"]
        assert min(anderson["critical_values"]) > 0
        assert "is_normal" not in anderson
    else:
        assert any(
            "critical-value grid" in warning["message"] for warning in profile["analysis_warnings"]
        )


def test_zero_mad_with_varying_values_is_unavailable():
    profile = StatisticalAnalyzer(pd.DataFrame({"x": [1, 1, 1, 1, 20]})).analyze_all()
    assert profile["outliers"]["x"]["iqr"]["count"] == 1
    assert profile["outliers"]["x"]["mad"]["count"] is None
    assert any(
        item["section"] == "outliers" and item["code"] == "undefined_result"
        for item in profile["analysis_warnings"]
    )


def test_iqr_z_score_and_modified_z_score_flag_reference_outlier():
    values = list(range(10)) + [100]
    outliers = StatisticalAnalyzer(pd.DataFrame({"x": values})).analyze_all()["outliers"]["x"]
    assert outliers["iqr"]["count"] == 1
    assert outliers["z_score"]["count"] == 1
    assert outliers["mad"]["count"] == 1
    # Median=5, MAD=3; the large value has modified Z > 3.5.
    assert 0.6745 * (100 - 5) / 3 > outliers["mad"]["threshold"]


def test_degenerate_and_bootstrap_metadata():
    with pytest.raises(InsufficientDataError, match="variation"):
        StatisticalAnalyzer(_groups([1, 1, 1], [2, 2, 2])).hypothesis_tests(
            "group", "value", "ttest", bootstrap_samples=0
        )
    analyzer = StatisticalAnalyzer(_groups([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]))
    first = analyzer.hypothesis_tests(
        "group", "value", "ttest", bootstrap_samples=100, random_state=7
    )
    second = analyzer.hypothesis_tests(
        "group", "value", "ttest", bootstrap_samples=100, random_state=7
    )
    assert (
        first["effect_size"]["confidence_interval"] == second["effect_size"]["confidence_interval"]
    )
    interval = first["effect_size"]["confidence_interval"]
    assert interval["requested_resamples"] == 100
    assert interval["valid_resamples"] <= 100
    assert interval["random_seed"] == 7
    assert math.isfinite(interval["lower"])
    assert math.isfinite(interval["upper"])
    assert first["assumptions"]["bootstrap"]["valid_resamples"] == interval["valid_resamples"]


def test_unavailable_bootstrap_interval_is_counted_and_warned():
    analyzer = StatisticalAnalyzer(_groups([1, 1], [1, 2]))
    result = analyzer.hypothesis_tests(
        "group", "value", "ttest", bootstrap_samples=100, random_state=0
    )
    assert result["effect_size"]["confidence_interval"] is None
    assert result["assumptions"]["bootstrap"]["valid_resamples"] == 44
    assert any("interval unavailable" in item for item in result["assumptions"]["warnings"])


def test_bootstrap_does_not_change_global_numpy_random_state():
    original = np.random.get_state()
    try:
        np.random.seed(123)
        before = np.random.get_state()
        StatisticalAnalyzer(_groups([1, 2, 3], [4, 5, 6])).hypothesis_tests(
            "group", "value", "ttest", bootstrap_samples=100, random_state=9
        )
        after = np.random.get_state()
        assert before[0] == after[0]
        assert np.array_equal(before[1], after[1])
        assert before[2:] == after[2:]
    finally:
        np.random.set_state(original)
