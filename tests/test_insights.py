from pyautostat.insights import InsightEngine


def test_no_insights_for_clean_data():
    results = {
        "normality": {
            "a": {
                "shapiro_wilk": {"is_normal": True},
                "d_agostino_pearson": {"is_normal": True},
            }
        },
        "outliers": {"a": {"iqr": {"percentage": 0.0, "count": 0}}},
        "missing_data": {"overall_missing_percentage": 0.0, "by_column": {}},
        "correlation": {"pearson": {"matrix": {}}},
        "distributions": {"a": {"skewness_interpretation": "Approximately symmetric"}},
        "data_quality": {"completeness": 1.0, "duplicate_rows_percentage": 0.0},
    }
    summary = InsightEngine(results).get_summary()
    assert summary["total_insights"] == 0
    assert summary["insights"] == []


def test_flags_non_normal_columns_when_multiple_tests_fail():
    results = {
        "normality": {
            "a": {
                "shapiro_wilk": {"is_normal": False},
                "d_agostino_pearson": {"is_normal": False},
            }
        }
    }
    insights = InsightEngine(results).generate_insights()
    assert any(i["category"] == "Normality" for i in insights)


def test_flags_high_outlier_percentage():
    results = {"outliers": {"a": {"iqr": {"percentage": 12.0, "count": 12}}}}
    insights = InsightEngine(results).generate_insights()
    assert insights[0]["category"] == "Outliers"
    assert insights[0]["details"][0]["column"] == "a"


def test_flags_high_missing_data_as_high_severity():
    results = {
        "missing_data": {
            "overall_missing_percentage": 25.0,
            "by_column": {"a": {"percentage": 30.0, "count": 3}},
        }
    }
    insights = InsightEngine(results).generate_insights()
    missing_insight = next(i for i in insights if i["category"] == "Missing Data")
    assert missing_insight["severity"] == "high"


def test_flags_multicollinearity_for_high_correlation_pairs():
    results = {"correlation": {"pearson": {"matrix": {"a": {"b": 0.95}, "b": {"a": 0.95}}}}}
    insights = InsightEngine(results).generate_insights()
    assert insights[0]["category"] == "Multicollinearity"
    assert insights[0]["details"][0]["pair"] == ["a", "b"]


def test_get_summary_counts_severities():
    results = {"data_quality": {"completeness": 0.5, "duplicate_rows_percentage": 0.0}}
    summary = InsightEngine(results).get_summary()
    assert summary["total_insights"] == 1
    assert summary["high_severity"] == 1
