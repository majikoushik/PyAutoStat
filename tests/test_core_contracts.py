"""Behavior tests for the core facade and serializable contracts."""

import json
import math

import pandas as pd
import pytest

import pyautostat
from pyautostat import (
    AnalysisOptions,
    AnalysisSpecification,
    ColumnNotFoundError,
    InsightEngine,
    InvalidDataError,
    InvalidTestError,
    Objective,
    ReportGenerator,
    ResearchAssistant,
    ResearchQuestion,
    StatisticalAnalyzer,
    StudyDesign,
    detect_column_types,
    suggest_column_roles,
)
from pyautostat.results import (
    AnalysisResult,
    Diagnostic,
    MissingInformation,
    Recommendation,
)


def test_assistant_profile_matches_analyzer_and_preserves_source():
    frame = pd.DataFrame({"group": ["A", "A", "B", "B"], "score": [10, 12, 15, 17]})
    before = frame.copy(deep=True)
    assistant = ResearchAssistant(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert assistant.profile() == StatisticalAnalyzer(frame).analyze_all()
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("bad", [None, [], pd.DataFrame(), pd.DataFrame(index=[0])])
def test_assistant_reuses_analyzer_validation(bad):
    with pytest.raises(InvalidDataError) as assistant_error:
        ResearchAssistant(bad)
    with pytest.raises(InvalidDataError) as analyzer_error:
        StatisticalAnalyzer(bad)
    assert str(assistant_error.value) == str(analyzer_error.value)


def test_legacy_exports_are_available():
    assert len(pyautostat.__all__) == 97
    assert len(set(pyautostat.__all__)) == 97
    assert {"InsightEngine", "ReportGenerator"} <= set(pyautostat.__all__)
    for name in pyautostat.__all__:
        assert hasattr(pyautostat, name), f"Missing public export: {name}"
        assert getattr(pyautostat, name) is not None
    for exported in (
        StatisticalAnalyzer,
        InsightEngine,
        ReportGenerator,
        detect_column_types,
        suggest_column_roles,
        InvalidDataError,
        InvalidTestError,
        ColumnNotFoundError,
    ):
        assert exported is not None
    for exc_name in (
        "InvalidDataError",
        "ColumnNotFoundError",
        "InsufficientGroupsError",
        "InsufficientDataError",
        "InvalidTestError",
        "ReportError",
        "UnsupportedPresentationError",
    ):
        exc_cls = getattr(pyautostat, exc_name)
        assert issubclass(exc_cls, pyautostat.PyAutoStatError)
        assert issubclass(exc_cls, Exception)
    assert len(pyautostat.METHOD_CONTRACTS) == 24
    assert pyautostat.BUNDLE_SCHEMA_VERSION == 1


def test_specification_round_trip_preserves_unknown_and_nulls():
    spec = AnalysisSpecification(
        question=ResearchQuestion(
            objective=Objective.COMPARE_GROUPS,
            outcome="score",
            predictor="group",
            estimand="difference in means",
        ),
        variable_metadata={"score": "points"},
    )
    payload = spec.to_dict()
    assert payload["design"] == "unknown"
    assert spec.design is StudyDesign.UNKNOWN
    assert payload["question"]["description"] is None
    assert payload["options"]["alpha"] == 0.05
    assert AnalysisSpecification.from_dict(json.loads(json.dumps(payload, allow_nan=False))) == spec
    assert AnalysisSpecification().question.objective is None


@pytest.mark.parametrize("bad", [0, 1, -0.1, float("nan"), float("inf"), True, "0.05"])
def test_invalid_probability_is_rejected(bad):
    for field in ("alpha", "confidence_level"):
        with pytest.raises(InvalidDataError, match=field):
            AnalysisOptions(**{field: bad})


@pytest.mark.parametrize("bad", ["unknown_method", True, 42])
def test_invalid_objective_is_rejected(bad):
    with pytest.raises(InvalidDataError, match="objective"):
        ResearchQuestion(objective=bad)


@pytest.mark.parametrize("bad", ["randomized", True, 5])
def test_invalid_design_is_rejected(bad):
    with pytest.raises(InvalidDataError, match="design"):
        AnalysisSpecification(design=bad)


def test_invalid_names_seed_and_schema_are_rejected():
    with pytest.raises(InvalidDataError, match="outcome"):
        ResearchQuestion(outcome=" ")
    with pytest.raises(InvalidDataError, match="random_seed"):
        AnalysisOptions(random_seed=True)
    with pytest.raises(InvalidDataError, match="schema_version"):
        AnalysisSpecification.from_dict({"schema_version": 3})
    with pytest.raises(InvalidDataError, match="variable_metadata"):
        AnalysisSpecification(variable_metadata={"score": ""})


def test_missing_information_and_result_contracts_serialize():
    missing = MissingInformation("design", "Confirm whether observations are independent.")
    recommendation = Recommendation(status="needs_input", missing_information=(missing,))
    payload = recommendation.to_dict()
    assert payload["method_id"] is None
    assert payload["missing_information"] == [missing.to_dict()]
    assert payload["status"] == "needs_input"
    diagnostic = Diagnostic("variance", "unknown", "No diagnostic run.", {"statistic": None})
    result = AnalysisResult("welch_t", "unavailable", values={"p_value": None})
    for record in (recommendation, diagnostic, result):
        assert json.loads(json.dumps(record.to_dict(), allow_nan=False)) == record.to_dict()

    # H3: NumPy scalar, pd.NA, datetime, and negative zero serialization invariants
    import numpy as np

    from pyautostat.presentation.formatting import format_number
    from pyautostat.result_access import format_apa_number, get_p_value, get_statistic
    from pyautostat.specifications import _json_value

    assert _json_value(np.int64(42)) == 42
    assert _json_value(np.float64(3.14)) == 3.14
    assert _json_value(np.bool_(True)) is True
    assert _json_value(pd.NA) is None
    assert _json_value(pd.NaT) is None
    assert _json_value(pd.Timestamp("2026-01-01")) == "2026-01-01T00:00:00"

    # Negative zero display normalization
    assert format_number(-0.0) == "0.00"
    assert format_number(-0.0001, decimals=2) == "0.00"
    assert format_number(-0.0001, decimals=2, unit="$") == "$0.00"
    assert format_number(-5.0, decimals=2, unit="$") == "-$5.00"
    assert format_apa_number(-0.0) == "0.00"
    assert format_apa_number(-0.0001, decimals=2) == "0.00"

    # result_access handles None in f_test gracefully
    dummy_res = AnalysisResult("intraclass_correlation", "available", values={"f_test": None})
    assert get_statistic(dummy_res) is None
    assert get_p_value(dummy_res) is None


def test_nonfinite_and_unsupported_result_values_fail_serialization():
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(InvalidDataError, match="finite"):
            AnalysisResult("method", "available", values={"p_value": value}).to_dict()
    with pytest.raises(InvalidDataError, match="JSON-compatible"):
        Diagnostic("id", "unknown", "message", {"frame": pd.DataFrame()}).to_dict()


def test_invalid_recommendation_states_are_rejected():
    with pytest.raises(InvalidDataError, match="missing_information"):
        Recommendation(status="needs_input")
    with pytest.raises(InvalidDataError, match="blockers"):
        Recommendation(status="unsupported")
    with pytest.raises(InvalidDataError, match="method_id"):
        Recommendation(status="ready")
