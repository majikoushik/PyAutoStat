import pandas as pd
import pytest

from pyautostat import StatisticalAnalyzer


def test_descriptive_stats_match_pandas(numeric_df):
    desc = StatisticalAnalyzer(numeric_df).analyze_all()["descriptive"]
    col = numeric_df["linear"]
    stats = desc["linear"]

    assert stats["count"] == len(col)
    assert stats["mean"] == pytest.approx(col.mean())
    assert stats["median"] == pytest.approx(col.median())
    assert stats["std"] == pytest.approx(col.std())
    assert stats["variance"] == pytest.approx(col.var())
    assert stats["min"] == pytest.approx(col.min())
    assert stats["max"] == pytest.approx(col.max())
    assert stats["q1"] == pytest.approx(col.quantile(0.25))
    assert stats["q3"] == pytest.approx(col.quantile(0.75))
    assert stats["iqr"] == pytest.approx(col.quantile(0.75) - col.quantile(0.25))
    assert stats["skewness"] == pytest.approx(col.skew())
    assert stats["kurtosis"] == pytest.approx(col.kurtosis())


def test_coefficient_of_variation_is_none_when_mean_is_zero():
    df = pd.DataFrame({"centered": [-2, -1, 0, 1, 2]})
    desc = StatisticalAnalyzer(df).analyze_all()["descriptive"]
    assert desc["centered"]["coefficient_of_variation"] is None


def test_descriptive_stats_only_covers_numeric_columns(mixed_df):
    desc = StatisticalAnalyzer(mixed_df).analyze_all()["descriptive"]
    assert set(desc.keys()) == {"age", "income"}
