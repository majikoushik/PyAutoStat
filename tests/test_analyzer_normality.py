import pandas as pd

from pyautostat import StatisticalAnalyzer


def test_normality_reports_all_three_tests(numeric_df):
    normality = StatisticalAnalyzer(numeric_df).analyze_all()["normality"]

    for col in ("linear", "with_outlier"):
        tests = normality[col]
        assert set(tests) == {"shapiro_wilk", "d_agostino_pearson", "anderson_darling"}
        for name in ("shapiro_wilk", "d_agostino_pearson"):
            assert 0.0 <= tests[name]["p_value"] <= 1.0
            assert tests[name]["is_normal"] in (True, False)
        assert len(tests["anderson_darling"]["critical_values"]) == len(
            tests["anderson_darling"]["significance_levels"]
        )


def test_normality_skipped_for_columns_with_fewer_than_three_values():
    df = pd.DataFrame({"tiny": [1, 2]})
    normality = StatisticalAnalyzer(df).analyze_all()["normality"]
    assert normality == {}
