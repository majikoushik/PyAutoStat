import pandas as pd
import pytest

from pyautostat import StatisticalAnalyzer


def test_missing_data_counts(missing_df):
    missing = StatisticalAnalyzer(missing_df).analyze_all()["missing_data"]

    assert missing["by_column"]["a"]["count"] == 1
    assert missing["by_column"]["a"]["percentage"] == pytest.approx(20.0)
    assert missing["by_column"]["b"]["count"] == 0
    assert missing["total_missing_cells"] == 1
    assert missing["overall_missing_percentage"] == pytest.approx(1 / 10 * 100)


def test_data_quality_completeness_and_duplicates(duplicate_df):
    quality = StatisticalAnalyzer(duplicate_df).analyze_all()["data_quality"]

    assert quality["completeness"] == pytest.approx(1.0)
    assert quality["duplicate_rows"] == 1
    assert quality["duplicate_rows_percentage"] == pytest.approx(25.0)


def test_data_quality_uniqueness_ratio():
    df = pd.DataFrame({"col": [1, 1, 2, 3]})
    quality = StatisticalAnalyzer(df).analyze_all()["data_quality"]
    assert quality["uniqueness"]["col"] == pytest.approx(3 / 4)
