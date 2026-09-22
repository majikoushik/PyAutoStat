"""Declared data dictionary validation and non-mutating overrides."""

import json

import pandas as pd
import pytest

from pyautostat import InvalidDataError, ReportGenerator, ResearchAssistant


def test_metadata_overrides_type_and_role_without_mutating_values():
    frame = pd.DataFrame(
        {
            "student_id": [101, 102, 103, 104],
            "rating": [1, 2, 3, 4],
            "group": ["A", "A", "B", "C"],
        }
    )
    before = frame.copy(deep=True)
    profile = ResearchAssistant(frame).profile(
        data_dictionary={
            "student_id": {"type": "continuous", "role": "measurement", "unit": "points"},
            "rating": {"type": "ordinal", "ordinal_order": [1, 2, 3, 4], "label": "Rating"},
            "group": {"type": "nominal", "allowed_values": ["A", "B"]},
        }
    )
    pd.testing.assert_frame_equal(frame, before)
    assert (
        profile["variable_intelligence"]["student_id"]["suggested_type"] == "continuous_numerical"
    )
    assert profile["variable_intelligence"]["student_id"]["suggested_role"] == "measurement"
    assert profile["variable_intelligence"]["student_id"]["type_source"] == "declared"
    assert "student_id" in profile["descriptive"]
    assert profile["variable_intelligence"]["rating"]["suggested_type"] == "ordinal_categorical"
    assert "rating" not in profile["descriptive"]
    assert profile["data_dictionary"]["rating"]["ordinal_order"] == [1, 2, 3, 4]
    assert any(
        issue["code"] == "undeclared_category" and issue["evidence"]["count"] == 1
        for issue in profile["data_quality"]["issues"]
    )
    assert (
        json.loads(ReportGenerator(profile).to_json())["analysis"]["data_dictionary"]["student_id"][
            "unit"
        ]
        == "points"
    )


def test_valid_range_and_declared_missing_codes_are_reported_but_not_applied():
    frame = pd.DataFrame({"exam_score": [0, 80, 101, 999]})
    before = frame.copy(deep=True)
    ordinary = ResearchAssistant(frame).profile()
    declared = ResearchAssistant(frame).profile(
        data_dictionary={
            "exam_score": {
                "type": "continuous",
                "label": "Examination score",
                "unit": "points",
                "valid_range": [0, 100],
                "missing_codes": [999],
            }
        }
    )
    pd.testing.assert_frame_equal(frame, before)
    assert ordinary["missing_data"]["total_missing_cells"] == 0
    assert declared["missing_data"]["total_missing_cells"] == 0
    assert declared["missing_data"]["by_column"]["exam_score"]["declared_missing_code_count"] == 1
    assert declared["profile_metadata"]["missing_codes_applied"] is False
    assert declared["descriptive"]["exam_score"]["count"] == 4
    assert any(
        issue["code"] == "declared_range_violation" and issue["evidence"]["count"] == 2
        for issue in declared["data_quality"]["issues"]
    )
    assert any(
        issue["code"] == "missing_codes_not_applied" for issue in declared["data_quality"]["issues"]
    )


def test_declared_identifier_repeats_are_not_called_exact_duplicates():
    frame = pd.DataFrame({"participant": ["A", "A", "B"], "visit": [1, 2, 1]})
    profile = ResearchAssistant(frame).profile(
        data_dictionary={"participant": {"type": "identifier", "role": "identifier"}}
    )
    assert profile["data_quality"]["duplicate_rows"] == 0
    assert profile["data_quality"]["repeated_identifiers"]["participant"] == 1
    assert any(
        issue["code"] == "repeated_identifier" for issue in profile["data_quality"]["issues"]
    )


@pytest.mark.parametrize(
    "metadata,fragment",
    [
        ({"absent": {"type": "continuous"}}, "unknown column"),
        ({"x": []}, "metadata mapping"),
        ({"x": {"type": "magic"}}, "type must be"),
        ({"x": {"type": ["continuous"]}}, "type must be"),
        ({"x": {"role": "magic"}}, "role must be"),
        ({"x": {"role": ["identifier"]}}, "role must be"),
        ({"x": {"valid_range": [5, 1]}}, "valid_range"),
        ({"x": {"valid_range": [0, float("inf")]}}, "valid_range"),
        ({"x": {"missing_codes": [99, 99]}}, "missing_codes"),
        ({"x": {"type": "continuous", "ordinal_order": [1, 2]}}, "conflicts"),
        ({"x": {"unsupported": True}}, "unsupported fields"),
    ],
)
def test_invalid_data_dictionary_is_actionable(metadata, fragment):
    with pytest.raises(InvalidDataError, match=fragment):
        ResearchAssistant(pd.DataFrame({"x": [1, 2, 3]})).profile(data_dictionary=metadata)


@pytest.mark.parametrize("bins", [0, -1, 1.5, True, 1001])
def test_invalid_histogram_bins_are_rejected(bins):
    with pytest.raises(InvalidDataError, match="histogram_bins"):
        ResearchAssistant(pd.DataFrame({"x": [1, 2, 3]})).profile(histogram_bins=bins)


def test_row_position_option_requires_boolean():
    with pytest.raises(InvalidDataError, match="include_row_positions"):
        ResearchAssistant(pd.DataFrame({"x": [1, 2, 3]})).profile(include_row_positions="yes")


@pytest.mark.parametrize("columns", [[], ["absent"], ["x", "x"], "x"])
def test_complete_case_column_validation(columns):
    with pytest.raises(InvalidDataError):
        ResearchAssistant(pd.DataFrame({"x": [1, 2, 3]})).complete_case_count(columns)
