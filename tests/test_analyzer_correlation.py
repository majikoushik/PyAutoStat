import pandas as pd
import pytest

from pyautostat import StatisticalAnalyzer


def test_correlation_matches_pandas(mixed_df):
    corr = StatisticalAnalyzer(mixed_df).analyze_all()["correlation"]

    expected_pearson = mixed_df[["age", "income"]].corr(method="pearson")
    assert corr["pearson"]["matrix"]["age"]["income"] == pytest.approx(
        expected_pearson.loc["age", "income"]
    )
    assert set(corr) == {"pearson", "spearman", "kendall", "p_values"}
    assert "income" in corr["p_values"]["age"]


def test_correlation_empty_with_fewer_than_two_numeric_columns():
    df = pd.DataFrame({"only_numeric": [1, 2, 3], "cat": ["a", "b", "c"]})
    corr = StatisticalAnalyzer(df).analyze_all()["correlation"]
    assert corr == {}
