import pytest

from pyautostat import StatisticalAnalyzer


def test_iqr_outlier_detected_for_extreme_value(numeric_df):
    outliers = StatisticalAnalyzer(numeric_df).analyze_all()["outliers"]

    assert outliers["linear"]["iqr"]["count"] == 0
    assert outliers["with_outlier"]["iqr"]["count"] == 1
    assert outliers["with_outlier"]["iqr"]["percentage"] == pytest.approx(10.0)


def test_zscore_and_mad_methods_present_for_every_numeric_column(numeric_df):
    outliers = StatisticalAnalyzer(numeric_df).analyze_all()["outliers"]

    for col in ("linear", "with_outlier"):
        assert set(outliers[col]) == {"iqr", "z_score", "mad"}
        assert outliers[col]["z_score"]["threshold"] == 3.0
        assert outliers[col]["mad"]["threshold"] == 3.5
