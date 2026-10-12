"""Tests for PR-5 Part B design-aware missing data hardening and non-mutation."""

import numpy as np
import pandas as pd
import pytest

from pyautostat.analyzer import StatisticalAnalyzer
from pyautostat.exceptions import InvalidDataError
from pyautostat.reliability import calculate_reliability
from pyautostat.research_assistant import ResearchAssistant


def test_independent_two_group_with_nan_outcome():
    original_df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "score": [10.0, np.nan, 12.0, 9.0, 11.0, 10.0, 12.0, 11.0, 9.0, 10.0]
            + [15.0, 14.0, 16.0, 15.0, 17.0, 14.0, 16.0, 15.0, 17.0, 15.0],
        }
    )
    copy_df = original_df.copy(deep=True)
    ra = ResearchAssistant(original_df)
    wf = ra.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
    )
    assert wf.analysis is not None
    assert wf.analysis.sample_size == 19
    assert wf.analysis.excluded_rows == 1
    # Check no mutation of input df
    pd.testing.assert_frame_equal(original_df, copy_df)


def test_independent_two_group_with_missing_group():
    original_df = pd.DataFrame(
        {
            "group": ["A"] * 9 + [None] + ["B"] * 10,
            "score": [10.0, 11.0, 12.0, 9.0, 11.0, 10.0, 12.0, 11.0, 9.0, 10.0]
            + [15.0, 14.0, 16.0, 15.0, 17.0, 14.0, 16.0, 15.0, 17.0, 15.0],
        }
    )
    copy_df = original_df.copy(deep=True)
    ra = ResearchAssistant(original_df)
    wf = ra.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
    )
    assert wf.analysis is not None
    assert wf.analysis.sample_size == 19
    assert wf.analysis.excluded_rows == 1
    pd.testing.assert_frame_equal(original_df, copy_df)


def test_one_sample_comparison_with_nan():
    original_df = pd.DataFrame(
        {
            "score": [10.0, 12.0, np.nan, 14.0, 16.0, 11.0, 13.0, 15.0],
        }
    )
    copy_df = original_df.copy(deep=True)
    analyzer = StatisticalAnalyzer(original_df)
    res = analyzer.one_sample_t_test("score", reference_value=10.0)
    assert res["sample_size"] == 7
    assert res["excluded_rows"] == 1
    pd.testing.assert_frame_equal(original_df, copy_df)


def test_paired_comparison_with_incomplete_pair():
    # Unit 3 only has Pre, missing Post
    original_df = pd.DataFrame(
        {
            "id": ["U1", "U1", "U2", "U2", "U3"],
            "cond": ["Pre", "Post", "Pre", "Post", "Pre"],
            "score": [10.0, 15.0, 12.0, 17.0, 11.0],
        }
    )
    copy_df = original_df.copy(deep=True)
    analyzer = StatisticalAnalyzer(original_df)
    res = analyzer.paired_wilcoxon("id", "cond", "score", condition_order=("Pre", "Post"))
    assert res["total_units"] == 3
    assert res["complete_pairs"] == 2
    assert res["incomplete_units"] == 1
    assert res["excluded_rows"] == 1
    pd.testing.assert_frame_equal(original_df, copy_df)


def test_repeated_measures_with_incomplete_units():
    # 3 complete units (U1, U2, U3) and 1 incomplete unit (U4 missing condition C3)
    original_df = pd.DataFrame(
        {
            "id": ["U1", "U1", "U1", "U2", "U2", "U2", "U3", "U3", "U3", "U4", "U4"],
            "cond": ["C1", "C2", "C3", "C1", "C2", "C3", "C1", "C2", "C3", "C1", "C2"],
            "score": [10.0, 12.0, 15.0, 11.0, 13.0, 16.0, 10.5, 12.5, 14.5, 9.0, 11.0],
        }
    )
    copy_df = original_df.copy(deep=True)
    analyzer = StatisticalAnalyzer(original_df)
    res = analyzer.repeated_measures_anova(
        "id", "cond", "score", condition_order=("C1", "C2", "C3")
    )
    sample_info = res["sample"]
    assert sample_info["total_units"] == 4
    assert sample_info["complete_units"] == 3
    assert sample_info["incomplete_units"] == 1
    pd.testing.assert_frame_equal(original_df, copy_df)


def test_regression_with_missing_outcome():
    original_df = pd.DataFrame(
        {
            "y": [1.1, 2.3, np.nan, 3.8, 5.2, 6.1, 7.3, 7.9, 9.1, 10.2],
            "x": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        }
    )
    copy_df = original_df.copy(deep=True)
    ra = ResearchAssistant(original_df)
    wf = ra.run(
        objective="regression",
        outcome="y",
        predictors=["x"],
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "continuous", "x": "continuous"},
    )
    assert wf.analysis is not None
    assert wf.analysis.sample_size == 9
    assert wf.analysis.excluded_rows == 1
    pd.testing.assert_frame_equal(original_df, copy_df)


def test_regression_with_missing_predictor():
    original_df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x1": [2.0, 4.0, np.nan, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "x2": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
        }
    )
    copy_df = original_df.copy(deep=True)
    ra = ResearchAssistant(original_df)
    wf = ra.run(
        objective="regression",
        outcome="y",
        predictors=["x1", "x2"],
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "continuous", "x1": "continuous", "x2": "continuous"},
    )
    assert wf.analysis is not None
    assert wf.analysis.sample_size == 9
    assert wf.analysis.excluded_rows == 1
    pd.testing.assert_frame_equal(original_df, copy_df)


def test_correlation_with_incomplete_pair():
    original_df = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }
    )
    copy_df = original_df.copy(deep=True)
    ra = ResearchAssistant(original_df)
    wf = ra.run(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="linear",
        design="independent",
        variable_types={"x": "continuous", "y": "continuous"},
    )
    assert wf.analysis is not None
    assert wf.analysis.sample_size == 5
    assert wf.analysis.excluded_rows == 1
    pd.testing.assert_frame_equal(original_df, copy_df)


def test_categorical_association_missing_rows():
    original_df = pd.DataFrame(
        {
            "cat1": ["A", "A", None, "B", "B", "B", "A", "B", "A", "B"],
            "cat2": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
        }
    )
    copy_df = original_df.copy(deep=True)
    ra = ResearchAssistant(original_df)
    wf = ra.run(
        objective="association",
        outcome="cat1",
        predictor="cat2",
        estimand="categorical_independence",
        design="independent",
        variable_types={"cat1": "nominal", "cat2": "nominal"},
    )
    assert wf.analysis is not None
    assert wf.analysis.sample_size == 9
    assert wf.analysis.excluded_rows == 1
    pd.testing.assert_frame_equal(original_df, copy_df)


def test_cronbach_reliability_with_incomplete_respondents():
    original_df = pd.DataFrame(
        {
            "item1": [1.0, 2.0, 3.0, 4.0, np.nan, 5.0, 4.0, 3.0],
            "item2": [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0],
            "item3": [1.0, 2.0, 2.0, 4.0, 5.0, 5.0, 4.0, 3.0],
        }
    )
    copy_df = original_df.copy(deep=True)
    res = calculate_reliability(original_df, ["item1", "item2", "item3"])
    assert res["sample"]["analyzed_rows"] == 7
    assert res["sample"]["excluded_rows"] == 1
    pd.testing.assert_frame_equal(original_df, copy_df)


def test_infinities_are_rejected_not_silently_dropped():
    df_inf = pd.DataFrame(
        {
            "score": [1.0, 2.0, np.inf, 4.0],
        }
    )
    with pytest.raises(InvalidDataError, match="infinity"):
        StatisticalAnalyzer(df_inf)
