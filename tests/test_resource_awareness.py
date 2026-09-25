"""Advisory profiling resource metadata must never change data or analysis."""

import json

import numpy as np
import pandas as pd
import pytest

from pyautostat import ReportGenerator, ResearchAssistant


def test_small_profile_reports_deep_memory_and_preserves_dataframe():
    frame = pd.DataFrame(
        {
            "score": [1.0, np.nan, 3.0, 4.0],
            "label": ["short", "a much longer value", None, "text"],
        }
    )
    before = frame.copy(deep=True)
    expected_bytes = int(frame.memory_usage(index=True, deep=True).sum())

    profile = ResearchAssistant(frame).profile()
    resource = profile["resource_info"]

    assert resource["row_count"] == 4
    assert resource["column_count"] == 2
    assert resource["estimated_memory_bytes"] == expected_bytes
    assert resource["estimated_memory_mib"] == pytest.approx(expected_bytes / 1024**2)
    assert resource["estimated_memory_bytes"] > 0
    assert np.isfinite(resource["estimated_memory_mib"])
    assert resource["resource_level"] == "normal"
    assert resource["resource_warnings"] == []
    assert resource["sampling_applied"] is False
    assert resource["truncation_applied"] is False
    assert resource["source_data_modified"] is False
    json.dumps(resource, allow_nan=False)
    exported = json.loads(ReportGenerator(profile).to_json())
    assert exported["analysis"]["resource_info"] == resource
    pd.testing.assert_frame_equal(frame, before)
    assert frame.dtypes.equals(before.dtypes)
    assert frame.isna().equals(before.isna())


@pytest.mark.parametrize(
    ("large_threshold", "very_large_threshold", "level", "code", "severity"),
    [
        (1, 10**9, "large", "large_dataframe_profile", "advisory"),
        (1, 1, "very_large", "very_large_dataframe_profile", "strong_advisory"),
    ],
)
def test_memory_policy_emits_contextual_advisories(
    monkeypatch, large_threshold, very_large_threshold, level, code, severity
):
    import pyautostat.profiling as profiling

    monkeypatch.setattr(profiling, "_LARGE_DATAFRAME_BYTES", large_threshold)
    monkeypatch.setattr(profiling, "_VERY_LARGE_DATAFRAME_BYTES", very_large_threshold)
    frame = pd.DataFrame({"score": [1.0, 2.0, 3.0], "group": ["A", "A", "B"]})

    resource = ResearchAssistant(frame).profile()["resource_info"]

    warning = next(item for item in resource["resource_warnings"] if item["code"] == code)
    assert resource["resource_level"] == level
    assert warning["severity"] == severity
    assert str(resource["estimated_memory_bytes"]) in warning["message"]
    assert warning["context"]["estimated_memory_bytes"] == resource["estimated_memory_bytes"]
    assert resource["policy"]["large_dataframe_bytes"] == large_threshold


def test_wide_numeric_profile_warns_without_skipping_correlations(monkeypatch):
    import pyautostat.profiling as profiling

    monkeypatch.setattr(profiling, "_WIDE_CORRELATION_COLUMN_COUNT", 4)
    frame = pd.DataFrame(
        {
            "score_a": [1, 2, 3, 4, 5, 6],
            "score_b": [2, 1, 4, 3, 6, 5],
            "score_c": [6, 5, 4, 3, 2, 1],
            "score_d": [1, 3, 2, 6, 4, 5],
        }
    )

    profile = ResearchAssistant(frame).profile()
    resource = profile["resource_info"]
    warning = next(
        item for item in resource["resource_warnings"] if item["code"] == "wide_correlation_profile"
    )

    assert resource["numeric_column_count"] == 4
    assert resource["correlation_matrix_dimension"] == [4, 4]
    assert resource["correlation_distinct_pair_count"] == 6
    assert "4 x 4" in warning["message"]
    assert warning["context"]["distinct_pair_count"] == 6
    assert set(profile["correlation"]["pearson"]["matrix"]) == set(frame.columns)


def test_resource_estimation_failure_is_advisory(monkeypatch):
    frame = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]})
    assistant = ResearchAssistant(frame)

    def unavailable(*args, **kwargs):
        raise RuntimeError("deliberate estimate failure")

    monkeypatch.setattr(pd.DataFrame, "memory_usage", unavailable)
    profile = assistant.profile()
    resource = profile["resource_info"]

    assert resource["resource_level"] == "unknown"
    assert resource["estimated_memory_bytes"] is None
    assert resource["estimated_memory_mib"] is None
    assert resource["resource_warnings"][0]["code"] == "memory_estimate_unavailable"
    assert profile["descriptive"]["score"]["mean"] == pytest.approx(2.5)
    json.dumps(resource, allow_nan=False)


def test_resource_awareness_never_samples_or_reduces_rows(monkeypatch):
    frame = pd.DataFrame({"score": list(range(20)), "group": ["A"] * 10 + ["B"] * 10})
    assistant = ResearchAssistant(frame)

    def forbidden(*args, **kwargs):
        raise AssertionError("profiling must not sample")

    monkeypatch.setattr(pd.DataFrame, "sample", forbidden)
    profile = assistant.profile()

    assert profile["overview"]["total_rows"] == len(frame)
    assert profile["descriptive"]["score"]["count"] == len(frame)
    assert sum(profile["histograms"]["score"]["counts"]) == len(frame)


def test_resource_warning_does_not_change_guided_analysis(monkeypatch):
    import pyautostat.profiling as profiling

    frame = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "score": [2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0],
        }
    )
    assistant = ResearchAssistant(frame)
    arguments = {
        "objective": "compare_groups",
        "outcome": "score",
        "predictor": "group",
        "estimand": "mean",
        "design": "independent",
        "variable_types": {"score": "continuous"},
    }
    before = assistant.run(**arguments)
    assert before.analysis is not None

    monkeypatch.setattr(profiling, "_LARGE_DATAFRAME_BYTES", 1)
    monkeypatch.setattr(profiling, "_VERY_LARGE_DATAFRAME_BYTES", 10**9)
    profile = assistant.profile()
    after = assistant.run(**arguments)
    assert after.analysis is not None

    assert profile["resource_info"]["resource_level"] == "large"
    assert after.analysis.to_dict() == before.analysis.to_dict()
    assert after.analysis.method_id == "welch_t"
    assert after.analysis.specification.question.estimand == "mean"
    assert after.analysis.specification.options.alpha == 0.05
    assert after.analysis.metadata["group_order"] == before.analysis.metadata["group_order"]
