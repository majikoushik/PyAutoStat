from pyautostat import StatisticalAnalyzer


def test_overview_shape_and_columns(mixed_df):
    overview = StatisticalAnalyzer(mixed_df).analyze_all()["overview"]

    assert overview["shape"] == mixed_df.shape
    assert overview["total_rows"] == len(mixed_df)
    assert overview["total_columns"] == len(mixed_df.columns)
    assert overview["numeric_columns"] == 2  # age, income
    assert overview["categorical_columns"] == 1  # department
    assert set(overview["columns"]) == set(mixed_df.columns)
