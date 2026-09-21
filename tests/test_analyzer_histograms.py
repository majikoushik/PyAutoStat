from pyautostat import StatisticalAnalyzer


def test_histogram_bins_cover_all_non_null_values(numeric_df):
    histograms = StatisticalAnalyzer(numeric_df).analyze_all()["histograms"]

    for col in ("linear", "with_outlier"):
        hist = histograms[col]
        assert sum(hist["counts"]) == numeric_df[col].dropna().shape[0]
        assert len(hist["bin_edges"]) == len(hist["counts"]) + 1


def test_histograms_only_cover_numeric_columns(mixed_df):
    histograms = StatisticalAnalyzer(mixed_df).analyze_all()["histograms"]
    assert set(histograms.keys()) == {"age", "income"}
