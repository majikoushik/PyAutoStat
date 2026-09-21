"""Bad-data and export boundaries that must fail predictably or remain usable."""

import json

import numpy as np
import pandas as pd
import pytest

from pyautostat import (
    InsightEngine,
    InsufficientDataError,
    InsufficientGroupsError,
    InvalidDataError,
    InvalidTestError,
    ReportError,
    ReportGenerator,
    StatisticalAnalyzer,
    detect_column_types,
    suggest_column_roles,
)


@pytest.mark.parametrize(
    "frame,message",
    [
        (pd.DataFrame(index=[0, 1]), "0 columns"),
        (pd.DataFrame([[1, 2]], columns=["x", "x"]), "unique"),
        (pd.DataFrame({0: [1, 2]}), "non-empty strings"),
        (pd.DataFrame({"x": [1, np.inf]}), "infinity"),
        (pd.DataFrame({"x": [1 + 2j, 3 + 4j]}), "complex"),
        (pd.DataFrame({"x": [[1], [2]]}), "nested"),
        (pd.DataFrame({"x": [{"a": 1}, {"a": 2}]}), "nested"),
    ],
)
def test_invalid_frames_raise_clear_package_errors(frame, message):
    with pytest.raises(InvalidDataError, match=message):
        StatisticalAnalyzer(frame)


def test_all_missing_numeric_column_is_explicit_and_exportable():
    df = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "group": ["a", "b", "c"]})
    result = StatisticalAnalyzer(df).analyze_all()

    assert result["descriptive"]["x"]["count"] == 0
    assert result["descriptive"]["x"]["mean"] is None
    assert result["data_quality"]["uniqueness"]["x"] is None
    assert result["outliers"]["x"]["iqr"]["percentage"] is None
    assert result["outliers"]["x"]["iqr"]["count"] is None
    assert result["normality"] == {}
    assert any(w["code"] == "all_missing" for w in result["analysis_warnings"])

    report = ReportGenerator(result, InsightEngine(result).get_summary())
    assert "N/A" in report.to_html()
    assert "NaN" not in report.to_json()
    assert json.loads(report.to_json())["analysis"]["descriptive"]["x"]["mean"] is None


def test_small_and_constant_samples_do_not_emit_undefined_normality_results():
    df = pd.DataFrame({"short": [1.0, 2.0, 3.0, np.nan, np.nan], "constant": [7] * 5})
    result = StatisticalAnalyzer(df).analyze_all()

    assert set(result["normality"]["short"]) == {"shapiro_wilk", "anderson_darling"}
    assert "constant" not in result["normality"]
    assert result["outliers"]["constant"]["z_score"]["count"] is None
    assert result["outliers"]["constant"]["mad"]["count"] is None
    assert result["distributions"]["constant"]["skewness_interpretation"] == "Unavailable"
    assert any("at least 8" in w["message"] for w in result["analysis_warnings"])


def test_nullable_integer_column_remains_analyzable():
    df = pd.DataFrame({"value": pd.Series([1, 2, None, 4, 5, 6, 7, 8, 9], dtype="Int64")})
    result = StatisticalAnalyzer(df).analyze_all()
    assert result["descriptive"]["value"]["count"] == 8
    assert result["histograms"]["value"]["counts"]


def test_extreme_finite_values_do_not_abort_analysis_or_return_invalid_intervals():
    values = [1e308, -1e308] * 4
    frame = pd.DataFrame({"value": values, "group": ["a"] * 4 + ["b"] * 4})
    result = StatisticalAnalyzer(frame).analyze_all()
    assert result["distributions"]["value"]["is_bimodal"] is None
    assert any(w["section"] == "distributions" for w in result["analysis_warnings"])

    with pytest.raises(InsufficientDataError, match="confidence interval"):
        StatisticalAnalyzer(frame).hypothesis_tests("group", "value", "ttest")


def test_subnormal_variation_is_not_reported_as_zero_standard_deviation():
    result = StatisticalAnalyzer(pd.DataFrame({"value": [1e-300, 2e-300, 3e-300]})).analyze_all()
    assert result["descriptive"]["value"]["std"] is None
    assert any(w["code"] == "numeric_underflow" for w in result["analysis_warnings"])


@pytest.mark.parametrize(
    "test_type,groups",
    [("anova", 2), ("kruskal", 2), ("ttest", 3), ("mannwhitney", 3)],
)
def test_incompatible_test_type_is_rejected(test_type, groups):
    df = pd.DataFrame(
        {
            "group": [f"g{i}" for i in range(groups) for _ in range(4)],
            "value": list(range(groups * 4)),
        }
    )
    with pytest.raises(InvalidTestError, match="groups"):
        StatisticalAnalyzer(df).hypothesis_tests("group", "value", test_type)


def test_hypothesis_rejects_invalid_values_and_too_few_observations():
    nonnumeric = pd.DataFrame({"group": ["a", "a", "b", "b"], "value": ["x", "y", "z", "w"]})
    with pytest.raises(InvalidDataError, match="real numeric"):
        StatisticalAnalyzer(nonnumeric).hypothesis_tests("group", "value")

    one_value = pd.DataFrame({"group": ["a", "a", "b", "b"], "value": [1, 2, 3, np.nan]})
    with pytest.raises(InsufficientDataError, match="at least 2"):
        StatisticalAnalyzer(one_value).hypothesis_tests("group", "value")

    missing_group = pd.DataFrame({"group": ["a", "a", "b", "b"], "value": [1, 2, np.nan, np.nan]})
    with pytest.raises(InsufficientGroupsError, match="usable"):
        StatisticalAnalyzer(missing_group).hypothesis_tests("group", "value")


def test_hypothesis_rejects_invalid_argument_types():
    analyzer = StatisticalAnalyzer(
        pd.DataFrame({"group": ["a", "a", "b", "b"], "value": [1, 2, 3, 4]})
    )
    for args in (([], "value", "auto"), ("group", None, "auto"), ("group", "value", [])):
        with pytest.raises(InvalidTestError):
            analyzer.hypothesis_tests(*args)


def test_constant_groups_do_not_return_undefined_test_results():
    df = pd.DataFrame({"group": ["a"] * 4 + ["b"] * 4, "value": [1] * 4 + [2] * 4})
    with pytest.raises(InsufficientDataError, match="variation"):
        StatisticalAnalyzer(df).hypothesis_tests("group", "value", "ttest")

    three = pd.DataFrame(
        {"group": ["a"] * 3 + ["b"] * 3 + ["c"] * 3, "value": [1] * 3 + [2] * 3 + [3] * 3}
    )
    with pytest.raises(InsufficientDataError, match="variation"):
        StatisticalAnalyzer(three).hypothesis_tests("group", "value", "anova")


def test_html_escapes_untrusted_text():
    label = "<script>alert(1)</script>"
    df = pd.DataFrame({label: list(range(8)), "other": list(range(8, 16))})
    result = StatisticalAnalyzer(df).analyze_all()
    insights = {
        "total_insights": 1,
        "insights": [
            {"category": label, "severity": "high", "finding": label, "recommendation": [label]}
        ],
    }
    html = ReportGenerator(result, insights).to_html(title=label)

    assert label not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_json_converts_numpy_scalars_and_nonfinite_values():
    report = ReportGenerator({"value": np.float64(np.nan), "count": np.int64(2)})
    assert json.loads(report.to_json()) == {
        "timestamp": report.timestamp,
        "analysis": {"value": None, "count": 2},
        "insights": {},
    }


def test_csv_creates_directory_and_guards_formula_text(tmp_path):
    report = ReportGenerator(
        {"descriptive": {"=SUM(1,1)": {"mean": 2.0}}},
        {"insights": [{"category": "Test", "finding": "=1+1", "recommendation": []}]},
    )
    target = tmp_path / "nested" / "csv"
    report.to_csv(target)
    assert target.is_dir()
    assert "'=SUM(1,1)" in (target / "descriptive_stats.csv").read_text()
    assert "'=1+1" in (target / "insights.csv").read_text()


def test_report_write_errors_are_package_errors(tmp_path):
    report = ReportGenerator({})
    with pytest.raises(ReportError, match="Could not write report"):
        report.to_json(tmp_path)
    occupied = tmp_path / "occupied"
    occupied.write_text("file")
    with pytest.raises(ReportError, match="Could not write CSV"):
        report.to_csv(occupied)


def test_public_helpers_validate_input_types():
    with pytest.raises(InvalidDataError, match="DataFrame"):
        suggest_column_roles([])
    with pytest.raises(InvalidDataError, match="DataFrame"):
        detect_column_types([])
    with pytest.raises(InvalidDataError, match="mapping"):
        InsightEngine([])
    with pytest.raises(InvalidDataError, match="mapping"):
        ReportGenerator([])
