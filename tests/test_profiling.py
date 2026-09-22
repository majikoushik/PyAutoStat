"""Phase 3 dataset-only profiling behavior and independent reference counts."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from pyautostat import InsightEngine, ReportGenerator, ResearchAssistant, StatisticalAnalyzer


def test_dataframe_only_profile_keeps_existing_api_and_reference_statistics():
    frame = pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0, 4.0, 20.0],
            "group": ["A", "A", "B", "B", "B"],
            "active": [True, False, True, False, True],
            "day": pd.date_range("2024-01-01", periods=5),
        }
    )
    before = frame.copy(deep=True)
    profile = ResearchAssistant(frame).profile()
    pd.testing.assert_frame_equal(frame, before)

    assert profile["overview"]["total_rows"] == 5
    assert profile["overview"]["total_columns"] == 4
    assert profile["overview"]["datetime_columns"] == 1
    assert profile["overview"]["boolean_columns"] == 1
    assert profile["overview"]["memory_usage_bytes"] > 0
    assert profile["descriptive"]["score"]["mean"] == pytest.approx(6)
    assert profile["descriptive"]["score"]["median"] == 3
    assert profile["descriptive"]["score"]["variance"] == pytest.approx(62.5)
    assert profile["descriptive"]["score"]["q1"] == 2
    assert profile["descriptive"]["score"]["q3"] == 4
    assert profile["categorical_summary"]["group"]["observed_categories"] == 2
    assert profile["categorical_summary"]["group"]["frequencies"][0] == {
        "value": "B",
        "count": 3,
        "percentage": 60.0,
    }
    assert profile["variable_intelligence"]["active"]["suggested_type"] == "boolean"
    assert profile["variable_intelligence"]["day"]["suggested_type"] == "datetime"
    assert profile["profile_metadata"]["correlation_missing_policy"] == "pairwise_complete"
    assert StatisticalAnalyzer(frame).analyze_all()["descriptive"]["score"]["mean"] == 6
    assert (
        json.loads(ReportGenerator(profile, InsightEngine(profile).get_summary()).to_json())[
            "analysis"
        ]["overview"]["total_rows"]
        == 5
    )


def test_categorical_modes_ties_and_high_cardinality_limit():
    frame = pd.DataFrame({"category": ["A", "B", "A", "B", None]})
    summary = ResearchAssistant(frame).profile()["categorical_summary"]["category"]
    assert summary["mode_values"] == ["A", "B"]
    assert summary["mode_tie_count"] == 2
    assert summary["missing_count"] == 1
    assert summary["frequencies"][0]["percentage"] == 50

    many = ResearchAssistant(pd.DataFrame({"category": [f"c{i}" for i in range(25)]})).profile()
    summary = many["categorical_summary"]["category"]
    assert len(summary["frequencies"]) == 20
    assert summary["other_category_count"] == 5
    assert summary["other_observation_count"] == 5
    assert summary["mode_tie_count"] == 25
    assert summary["mode_truncated"] is True


def test_all_missing_categorical_and_numeric_columns_are_explicit():
    frame = pd.DataFrame(
        {
            "empty_text": pd.Series([None, None, None], dtype="object"),
            "empty_number": [np.nan, np.nan, np.nan],
        }
    )
    profile = ResearchAssistant(frame).profile()
    assert profile["categorical_summary"]["empty_text"]["observed_categories"] == 0
    assert profile["categorical_summary"]["empty_text"]["mode_values"] == []
    assert profile["descriptive"]["empty_number"]["count"] == 0
    assert profile["missing_data"]["completely_missing_rows"] == 3
    assert profile["missing_data"]["complete_rows"] == 0


def test_missing_cells_rows_patterns_and_duplicate_overlap_are_distinct():
    frame = pd.DataFrame(
        {
            "a": [1.0, 1.0, np.nan, np.nan],
            "b": [None, None, 2.0, 2.0],
            "label": ["x", "x", "y", "y"],
        }
    )
    profile = ResearchAssistant(frame).profile()
    missing = profile["missing_data"]
    quality = profile["data_quality"]
    assert missing["total_missing_cells"] == 4
    assert missing["overall_missing_percentage"] == pytest.approx(100 / 3)
    assert missing["rows_with_missing"] == 4
    assert missing["completely_missing_rows"] == 0
    assert missing["complete_rows"] == 0
    assert {tuple(p["missing_columns"]): p["row_count"] for p in missing["common_patterns"]} == {
        ("a",): 2,
        ("b",): 2,
    }
    assert missing["by_column"]["a"]["available_count"] == 2
    assert quality["duplicate_rows"] == 2  # Two rows repeat their predecessors.
    assert quality["duplicate_group_rows"] == 4
    assert quality["missing_duplicate_overlap_rows"] == 4
    assert ResearchAssistant(frame).complete_case_count(["a", "label"]) == {
        "columns": ["a", "label"],
        "available_rows": 2,
        "excluded_rows": 2,
        "total_rows": 4,
    }


def test_identifier_suggestion_excludes_numeric_id_from_correlations():
    frame = pd.DataFrame(
        {
            "student_id": [101, 102, 103, 104, 105],
            "x": [1, 2, 3, 4, 5],
            "y": [5, 4, 3, 2, 1],
        }
    )
    analyzer = StatisticalAnalyzer(frame)
    profile = analyzer.analyze_all()
    assert analyzer.numeric_cols == ["student_id", "x", "y"]
    assert profile["variable_intelligence"]["student_id"]["suggested_type"] == "identifier"
    assert "student_id" not in profile["descriptive"]
    assert "student_id" not in profile["correlation"]["pearson"]["matrix"]
    assert profile["correlation"]["pearson"]["matrix"]["x"]["y"] == pytest.approx(-1)
    assert any(
        issue["code"] == "numeric_identifier_excluded"
        for issue in profile["data_quality"]["issues"]
    )
    # Name alone is weak evidence: repeated numeric values may be measurements.
    repeated = ResearchAssistant(pd.DataFrame({"student_id": [1, 1, 2, 2]})).profile()
    assert "student_id" in repeated["descriptive"]


def test_pairwise_correlation_counts_match_included_rows_and_perfect_coefficients():
    frame = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1, 2, np.nan, 4, np.nan, 6],
            "z": [6, np.nan, 4, 3, 2, 1],
        }
    )
    correlation = ResearchAssistant(frame).profile()["correlation"]
    expected = {"x": {"y": 4, "z": 5}, "y": {"x": 4, "z": 3}, "z": {"x": 5, "y": 3}}
    for method in ("pearson", "spearman", "kendall"):
        sizes = correlation[method]["sample_sizes"]
        for left, pairs in expected.items():
            for right, count in pairs.items():
                assert sizes[left][right] == count
    assert correlation["pearson"]["matrix"]["x"]["y"] == pytest.approx(1)
    assert correlation["pearson"]["matrix"]["x"]["z"] == pytest.approx(-1)
    assert correlation["spearman"]["matrix"]["x"]["z"] == pytest.approx(-1)
    assert correlation["kendall"]["matrix"]["x"]["z"] == pytest.approx(-1)


def test_constant_column_has_unavailable_correlation_and_structured_issue():
    frame = pd.DataFrame({"constant": [7] * 5, "x": [1, 2, 3, 4, 5]})
    profile = ResearchAssistant(frame).profile()
    corr = profile["correlation"]
    assert corr["pearson"]["matrix"]["constant"]["x"] is None
    assert corr["p_values"]["constant"]["x"] is None
    assert corr["pearson"]["sample_sizes"]["constant"]["x"] == 5
    assert corr["pearson"]["undefined_pairs"][0]["columns"] == ["constant", "x"]
    assert any(issue["code"] == "constant_column" for issue in profile["data_quality"]["issues"])


def test_one_pairwise_observation_is_unavailable_without_backend_small_sample_warning():
    frame = pd.DataFrame({"x": [1.0, 2.0, np.nan], "y": [1.0, np.nan, 3.0]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        profile = ResearchAssistant(frame).profile()
    assert not any(item.category.__name__ == "SmallSampleWarning" for item in caught)
    for method in ("pearson", "spearman", "kendall"):
        assert profile["correlation"][method]["sample_sizes"]["x"]["y"] == 1
        assert profile["correlation"][method]["matrix"]["x"]["y"] is None
        assert profile["correlation"][method]["undefined_pairs"][0]["reason"].startswith(
            "Fewer than two"
        )
    assert profile["correlation"]["p_values"]["x"]["y"] is None


def test_outlier_denominator_and_histogram_metadata_are_explicit():
    frame = pd.DataFrame({"value": [1.0, 1.0, 1.0, 1.0, 20.0, np.nan]})
    profile = ResearchAssistant(frame).profile(histogram_bins=5)
    iqr = profile["outliers"]["value"]["iqr"]
    assert iqr["count"] == 1
    assert iqr["percentage"] == 20
    assert iqr["usable_count"] == 5
    assert iqr["threshold"] == 1.5
    assert profile["outliers"]["value"]["mad"]["status"] == "unavailable"
    assert profile["histograms"]["value"]["requested_bins"] == 5
    assert sum(profile["histograms"]["value"]["counts"]) == 5
    assert len(profile["histograms"]["value"]["bin_edges"]) == 6
    assert profile["distributions"]["value"]["peak_heuristic"]["formal_test"] is False
    assert any(issue["code"] == "potential_outliers" for issue in profile["data_quality"]["issues"])


def test_optional_row_positions_are_integer_offsets_and_never_remove_records():
    frame = pd.DataFrame(
        {
            "value": [1.0, 1.0, 1.0, 1.0, 20.0],
            "group": ["A", "A", "B", "B", "C"],
        },
        index=[9, 9, 4, 4, 4],
    )
    before = frame.copy(deep=True)
    profile = ResearchAssistant(frame).profile(include_row_positions=True)
    pd.testing.assert_frame_equal(frame, before)
    assert profile["outliers"]["value"]["iqr"]["flagged_positions"] == [4]
    assert profile["outliers"]["value"]["mad"]["flagged_positions"] is None
    assert profile["data_quality"]["duplicate_group_positions"] == [0, 1, 2, 3]
    assert profile["data_quality"]["repeated_row_positions"] == [1, 3]
    assert profile["profile_metadata"]["row_positions_included"] is True
