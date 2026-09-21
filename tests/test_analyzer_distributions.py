import pandas as pd

from pyautostat import StatisticalAnalyzer


def test_symmetric_data_is_approximately_symmetric():
    df = pd.DataFrame({"symmetric": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    dist = StatisticalAnalyzer(df).analyze_all()["distributions"]
    assert dist["symmetric"]["skewness_interpretation"] == "Approximately symmetric"


def test_right_skewed_data_is_detected():
    df = pd.DataFrame({"skewed": [1, 1, 1, 1, 2, 2, 3, 100]})
    dist = StatisticalAnalyzer(df).analyze_all()["distributions"]
    assert "Positively skewed" in dist["skewed"]["skewness_interpretation"]


def test_range_is_max_minus_min(numeric_df):
    dist = StatisticalAnalyzer(numeric_df).analyze_all()["distributions"]
    assert dist["linear"]["range"] == 9.0
