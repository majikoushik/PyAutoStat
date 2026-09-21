import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyautostat import InsufficientDataError, InvalidTestError, StatisticalAnalyzer


def test_auto_selects_welch_for_stated_mean_target(two_group_normal_df):
    result = StatisticalAnalyzer(two_group_normal_df).hypothesis_tests(
        "group", "value", estimand="mean"
    )
    assert result["test"] == "t-test"
    assert result["equal_variance"] is False
    assert result["groups"] == ["A", "B"]


def test_ttest_reports_cohens_d_and_confidence_interval(two_group_normal_df):
    result = StatisticalAnalyzer(two_group_normal_df).hypothesis_tests(
        "group", "value", estimand="mean"
    )

    effect_size = result["effect_size"]
    assert effect_size["name"] == "Cohen's d"
    assert effect_size["interpretation"] in ("negligible", "small", "medium", "large")
    # Group B (mean 55) > Group A (mean 50), and 'value' is group1 - group2 = A - B
    assert effect_size["value"] < 0

    ci = result["confidence_interval"]
    assert ci["level"] == pytest.approx(0.95)
    assert ci["lower"] < ci["upper"]
    # Real difference of ~5 with n=40/group and std=5 should be a clearly non-zero effect
    assert ci["upper"] < 0


def test_auto_selects_mannwhitney_for_stated_distribution_target(two_group_skewed_df):
    result = StatisticalAnalyzer(two_group_skewed_df).hypothesis_tests(
        "group", "value", estimand="distribution"
    )
    assert result["test"] == "Mann-Whitney U"
    effect_size = result["effect_size"]
    assert effect_size["name"] == "rank-biserial correlation"
    assert -1.0 <= effect_size["value"] <= 1.0
    assert effect_size["confidence_interval"]["valid_resamples"] >= 250


def test_explicit_test_type_overrides_auto_selection(two_group_normal_df):
    result = StatisticalAnalyzer(two_group_normal_df).hypothesis_tests(
        "group", "value", test_type="mannwhitney"
    )
    assert result["test"] == "Mann-Whitney U"


def test_three_groups_run_anova_when_requested(three_group_df):
    result = StatisticalAnalyzer(three_group_df).hypothesis_tests("group", "value", "anova")
    assert result["test"] == "One-way ANOVA"
    assert set(result["groups"]) == {"A", "B", "C"}
    effect_size = result["effect_size"]
    assert effect_size["name"] == "eta-squared"
    assert 0.0 <= effect_size["value"] <= 1.0


def test_three_groups_kruskal_when_requested(three_group_df):
    result = StatisticalAnalyzer(three_group_df).hypothesis_tests(
        "group", "value", test_type="kruskal"
    )
    assert result["test"] == "Kruskal-Wallis"
    effect_size = result["effect_size"]
    assert effect_size["name"] == "epsilon-squared (rank)"
    assert effect_size["confidence_interval"]["valid_resamples"] >= 250


def test_auto_selects_kruskal_when_multigroup_normality_fails():
    rng = np.random.default_rng(17)
    values = np.concatenate([rng.exponential(scale=s, size=40) for s in (1, 2, 3)])
    frame = pd.DataFrame({"group": ["A"] * 40 + ["B"] * 40 + ["C"] * 40, "value": values})
    result = StatisticalAnalyzer(frame).hypothesis_tests(
        "group", "value", bootstrap_samples=100, estimand="distribution"
    )

    assert result["test"] == "Kruskal-Wallis"
    assert result["assumptions"]["selected_test_type"] == "kruskal"
    assert any(entry["status"] == "rejected" for entry in result["assumptions"]["normality"])
    assert "stated distribution" in result["assumptions"]["selection_reason"]


def test_assumption_details_and_reproducible_effect_size_interval(three_group_df):
    analyzer = StatisticalAnalyzer(three_group_df)
    first = analyzer.hypothesis_tests("group", "value", "anova", bootstrap_samples=100)
    second = analyzer.hypothesis_tests("group", "value", "anova", bootstrap_samples=100)

    assert first["assumptions"]["requested_test_type"] == "anova"
    assert [item["sample_size"] for item in first["assumptions"]["normality"]] == [30] * 3
    ci = first["effect_size"]["confidence_interval"]
    assert ci == second["effect_size"]["confidence_interval"]
    assert np.isfinite(ci["lower"]) and np.isfinite(ci["upper"])
    assert ci["lower"] <= ci["upper"]
    assert ci["valid_resamples"] == 100


def test_confidence_options_and_small_kruskal_groups(two_group_normal_df):
    analyzer = StatisticalAnalyzer(two_group_normal_df)
    result = analyzer.hypothesis_tests(
        "group", "value", "ttest", confidence_level=0.9, bootstrap_samples=0
    )
    assert result["confidence_interval"]["level"] == 0.9
    assert result["effect_size"]["confidence_interval"] is None

    for kwargs in (
        {"confidence_level": 1},
        {"confidence_level": float("nan")},
        {"bootstrap_samples": 99},
        {"bootstrap_samples": True},
        {"random_state": -1},
    ):
        with pytest.raises(InvalidTestError):
            analyzer.hypothesis_tests("group", "value", **kwargs)

    frame = pd.DataFrame({"group": ["A"] * 4 + ["B"] * 4 + ["C"] * 4, "value": list(range(12))})
    with pytest.raises(InsufficientDataError, match="at least 5"):
        StatisticalAnalyzer(frame).hypothesis_tests("group", "value", "kruskal")


def test_hypothesis_statistics_match_scipy_and_effect_direction(three_group_df):
    analyzer = StatisticalAnalyzer(three_group_df)
    group_values = [
        three_group_df.loc[three_group_df["group"] == group, "value"] for group in ("A", "B", "C")
    ]
    anova = analyzer.hypothesis_tests("group", "value", "anova", bootstrap_samples=0)
    expected_anova = stats.f_oneway(*group_values)
    assert anova["statistic"] == pytest.approx(expected_anova.statistic)
    assert anova["p_value"] == pytest.approx(expected_anova.pvalue)

    kruskal = analyzer.hypothesis_tests("group", "value", "kruskal", bootstrap_samples=0)
    expected_kruskal = stats.kruskal(*group_values)
    assert kruskal["statistic"] == pytest.approx(expected_kruskal.statistic)
    assert kruskal["p_value"] == pytest.approx(expected_kruskal.pvalue)

    two = pd.DataFrame(
        {"group": ["larger"] * 5 + ["smaller"] * 5, "value": [6, 7, 8, 9, 10, 1, 2, 3, 4, 5]}
    )
    mann = StatisticalAnalyzer(two).hypothesis_tests(
        "group", "value", "mannwhitney", bootstrap_samples=0
    )
    assert mann["effect_size"]["value"] == pytest.approx(1.0)
    assert mann["statistic"] == pytest.approx(
        stats.mannwhitneyu([6, 7, 8, 9, 10], [1, 2, 3, 4, 5]).statistic
    )
