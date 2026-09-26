import copy
import json

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
    assert "No profile-based insight" in summary["narrative"]
    assert summary["recommended_actions"] == []


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


def test_connected_multicollinearity_narrative_uses_objective_and_lead_finding():
    results = {
        "correlation": {
            "pearson": {
                "matrix": {
                    "a": {"b": 0.8, "c": -0.95, "d": 0.75, "e": 0.72},
                    "b": {"a": 0.8},
                    "c": {"a": -0.95},
                    "d": {"a": 0.75},
                    "e": {"a": 0.72},
                }
            }
        }
    }
    engine = InsightEngine(results, objective="regression")
    structured = engine.generate_insights()
    before = copy.deepcopy(structured)
    narrative = engine.get_narrative()

    assert "[MEDIUM]" in narrative
    assert "'a' and 'c'" in narrative
    assert "Other findings (3 more)" in narrative
    assert "standard errors" in narrative
    assert "'a' and 'e'" not in narrative
    assert structured == before
    assert narrative == engine.get_narrative()


def test_summary_ranks_actions_without_changing_structured_severity_or_values():
    results = {
        "missing_data": {
            "overall_missing_percentage": 25.0,
            "by_column": {"x": {"percentage": 30.0, "count": 3}},
        },
        "outliers": {
            "z": {
                "iqr": {
                    "percentage": 12.0,
                    "count": 2,
                    "lower_bound": 0.0,
                    "upper_bound": 10.0,
                }
            }
        },
        "correlation": {"pearson": {"matrix": {"x": {"z": 0.9}, "z": {"x": 0.9}}}},
    }
    engine = InsightEngine(results)
    generated = engine.generate_insights()
    before = copy.deepcopy(generated)
    summary = engine.get_summary()

    assert summary["insights"] == before
    assert summary["insights"][1]["severity"] == "high"
    assert "missing" in summary["recommended_actions"][0].lower()
    assert len(summary["recommended_actions"]) == len(set(summary["recommended_actions"]))
    assert "MISSING DATA" in summary["narrative"]
    json.dumps(summary, allow_nan=False)


def test_explicit_exploration_context_uses_noncausal_consequence():
    results = {"correlation": {"pearson": {"matrix": {"a": {"b": 0.8}, "b": {"a": 0.8}}}}}
    narrative = InsightEngine(results, objective="exploration").get_narrative()
    assert "overlapping linear information" in narrative
    assert "causes" not in narrative.lower()
