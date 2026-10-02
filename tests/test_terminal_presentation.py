"""Comprehensive unit and integration tests for the PyAutoStat Rich terminal presentation layer."""

from __future__ import annotations

import io
from copy import deepcopy

import pandas as pd
import pytest
from rich.console import Console

import pyautostat
from pyautostat import (
    AnalysisOptions,
    AnalysisResult,
    AnalysisStatus,
    ResearchAssistant,
    UnsupportedPresentationError,
    show,
)


@pytest.fixture
def sample_numeric_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        }
    )


@pytest.fixture
def welch_workflow(sample_numeric_df: pd.DataFrame):
    return ResearchAssistant(sample_numeric_df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=42),
    )


@pytest.fixture
def pearson_workflow():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [2.1, 3.9, 6.2, 8.1, 9.8, 12.3, 13.9, 16.1, 18.0, 20.2],
        }
    )
    return ResearchAssistant(df).run(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="linear",
        design="independent",
        variable_types={"x": "continuous", "y": "continuous"},
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=42),
    )


@pytest.fixture
def profile_result(sample_numeric_df: pd.DataFrame):
    return ResearchAssistant(sample_numeric_df).profile()


def _capture_show(target, detail="standard", width=80, no_color=False) -> str:
    """Helper to capture show() output string cleanly."""
    console = Console(
        record=True,
        width=width,
        no_color=no_color,
        force_terminal=True,
    )
    show(target, detail=detail, console=console)
    return console.export_text()


# ── Test Suite ───────────────────────────────────────────────────────────────


def test_rich_dependency_import():
    """Verify that Rich is available as a core dependency."""
    import rich

    assert rich is not None


def test_public_show_export():
    """Verify that show is exported from top-level pyautostat package."""
    assert hasattr(pyautostat, "show")
    assert callable(pyautostat.show)
    assert "show" in pyautostat.__all__


def test_unsupported_presentation_error_export():
    """Verify UnsupportedPresentationError is exported from top-level."""
    assert hasattr(pyautostat, "UnsupportedPresentationError")
    assert issubclass(pyautostat.UnsupportedPresentationError, Exception)


def test_welch_standard_mode_renders_all_required_elements(welch_workflow):
    """Verify standard mode for Welch t-test includes required statistical components."""
    output = _capture_show(welch_workflow, detail="standard")

    assert "Independent Group Comparison" in output
    assert "score by group" in output
    assert "Welch's independent-samples t-test" in output
    assert "Population mean difference" in output
    assert "Contrast" in output
    assert "'A' - 'B'" in output or "A - B" in output
    assert "10 rows" in output
    assert "GROUP SUMMARY" in output
    assert "Mean difference" in output
    assert "-10.00" in output  # Preserves signed estimate
    assert "95% CI" in output
    assert "Cohen's d" in output
    assert "-6.32" in output
    assert "p-value" in output
    assert "<0.001" in output
    assert "INTERPRETATION" in output
    assert "DIAGNOSTIC CONTEXT" in output
    assert "Equal variance" in output
    assert "IMPORTANT LIMITATIONS" in output
    assert "Statistical significance alone does not establish practical importance." in output

    # Must NOT dump raw dict repr
    assert "{'primary_estimate':" not in output
    assert "{'status':" not in output


def test_welch_compact_mode(welch_workflow):
    """Verify compact mode renders concise single-line representation."""
    output = _capture_show(welch_workflow, detail="compact")

    assert "Welch t-test" in output
    assert "N=10" in output
    assert "diff=-10.00" in output
    assert "d=-6.32" in output
    assert "p=<0.001" in output
    # Compact should not include multi-paragraph narrative
    assert "INTERPRETATION" not in output


def test_welch_full_mode(welch_workflow):
    """Verify full mode includes audit, reproducibility, and rationale sections."""
    output = _capture_show(welch_workflow, detail="full")

    assert "Independent Group Comparison" in output
    assert "RECOMMENDATION RATIONALE" in output
    assert "AUDIT VERIFICATION" in output
    assert "PASSED" in output
    assert "REPRODUCIBILITY METADATA" in output
    assert "Random seed" in output


def test_pearson_standard_mode(pearson_workflow):
    """Verify Pearson correlation standard mode rendering."""
    output = _capture_show(pearson_workflow, detail="standard")

    assert "Association Analysis" in output
    assert "x <-> y" in output
    assert "Pearson correlation" in output
    assert "Linear association" in output
    assert "Correlation" in output
    assert "r = 1.000" in output or "r = 0.99" in output
    assert "95% CI" in output
    assert "p-value" in output
    assert "INTERPRETATION" in output
    assert "DIAGNOSTIC / CONTEXT" in output
    assert "Pairwise complete N" in output
    assert "IMPORTANT LIMITATIONS" in output
    assert "Observed association alone does not establish causation." in output


def test_pearson_compact_mode(pearson_workflow):
    """Verify Pearson compact mode rendering."""
    output = _capture_show(pearson_workflow, detail="compact")

    assert "Pearson r" in output
    assert "N=10" in output
    assert "p=" in output
    assert "95% CI" in output


def test_profile_standard_mode(profile_result):
    """Verify dataset profile standard mode rendering."""
    output = _capture_show(profile_result, detail="standard")

    assert "Dataset Profile" in output
    assert "10 rows x 2 variables" in output
    assert "DATA QUALITY" in output
    assert "Rows" in output
    assert "Columns" in output
    assert "Missing cells" in output
    assert "Duplicate rows" in output
    assert "NUMERIC VARIABLES" in output
    assert "CATEGORICAL VARIABLES" in output
    assert "score" in output
    assert "group" in output
    assert "REVIEW CUES" in output


def test_profile_compact_mode(profile_result):
    """Verify dataset profile compact mode rendering."""
    output = _capture_show(profile_result, detail="compact")

    assert "Dataset Profile" in output
    assert "10 rows x 2 vars" in output
    assert "missing=" in output
    assert "duplicates=" in output


def test_needs_input_workflow_rendering(sample_numeric_df):
    """Verify needs_input workflow renders actionable prompts without implying failure."""
    wf_needs = ResearchAssistant(sample_numeric_df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
    )
    assert wf_needs.status.value == "needs_input"

    output = _capture_show(wf_needs)
    assert "Additional Information Required" in output
    assert "Analysis has not been run." in output
    assert "REQUIRED INFORMATION" in output
    assert "ALREADY KNOWN" in output
    assert "score" in output
    assert "FAILED" not in output


def test_unsupported_workflow_rendering(sample_numeric_df):
    """Verify unsupported workflow renders clear blocker explanation."""
    wf_unsupported = ResearchAssistant(sample_numeric_df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="clustered",
        variable_types={"score": "continuous"},
    )
    assert wf_unsupported.status.value == "unsupported"

    output = _capture_show(wf_unsupported)
    assert "Unsupported Specification" in output
    assert "BLOCKERS" in output
    assert "No substitute statistical method was run." in output


def test_data_limited_workflow_rendering():
    """Verify data_limited workflow renders numerical limitation explanation."""
    too_small = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 3.0]})
    wf_limited = ResearchAssistant(too_small).run(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="linear",
        design="independent",
        variable_types={"x": "continuous", "y": "continuous"},
    )
    assert wf_limited.status.value == "data_limited"

    output = _capture_show(wf_limited)
    assert "Analysis Data-Limited" in output
    assert "BLOCKERS" in output
    assert "at least three" in output
    assert "complete pairs" in output


def test_failed_workflow_rendering(sample_numeric_df, monkeypatch):
    """Verify failed workflow state renders failure reason without raw traceback."""
    assistant = ResearchAssistant(sample_numeric_df)

    def unavailable_analysis(draft, **kwargs):
        return AnalysisResult(
            method_id="welch_t",
            status=AnalysisStatus.UNAVAILABLE,
            warnings=("Computational singular matrix in test execution.",),
            specification=draft.specification,
            recommendation=assistant.recommend_test(draft),
        )

    monkeypatch.setattr(assistant, "analyze", unavailable_analysis)
    wf_failed = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )
    assert wf_failed.status.value == "failed"

    output = _capture_show(wf_failed)
    assert "Analysis Failed" in output
    assert "FAILURE DETAILS" in output
    assert "Computational singular matrix" in output
    assert "Traceback" not in output


@pytest.mark.parametrize("width", [70, 90, 120])
def test_console_widths_render_without_error(welch_workflow, width):
    """Verify layout adapts gracefully across 70, 90, and 120 character widths."""
    output = _capture_show(welch_workflow, width=width)
    assert "Independent Group Comparison" in output
    assert "Mean difference" in output
    assert len(output) > 100


def test_no_color_console_behavior(welch_workflow):
    """Verify NO_COLOR / no-color console disables color system and color codes."""
    string_io = io.StringIO()
    console = Console(file=string_io, no_color=True, force_terminal=True, width=80)
    show(welch_workflow, console=console)
    output = string_io.getvalue()
    assert console.no_color is True
    # No RGB truecolor or 256-color escape sequences
    assert "38;2;" not in output
    assert "48;2;" not in output
    assert "Independent Group Comparison" in output


def test_non_tty_captured_output_has_no_malformed_escapes(welch_workflow):
    """Verify non-TTY file-captured output has no malformed terminal artifacts."""
    string_io = io.StringIO()
    console = Console(file=string_io, force_terminal=False, width=80)
    show(welch_workflow, console=console)
    text = string_io.getvalue()

    assert "Independent Group Comparison" in text
    assert "\x1b[" not in text


def test_serialization_safety_show_does_not_mutate_input(welch_workflow, profile_result):
    """Verify that show() is strictly read-only and never mutates result objects."""
    wf_dict_before = deepcopy(welch_workflow.to_dict())
    prof_dict_before = deepcopy(profile_result)

    _capture_show(welch_workflow)
    _capture_show(profile_result)

    assert welch_workflow.to_dict() == wf_dict_before
    assert profile_result == prof_dict_before


def test_no_statistical_recalculation_during_presentation(welch_workflow, monkeypatch):
    """Verify that the presenter consumes only stored numbers and never calls stat engines."""
    import scipy.stats

    # Monkeypatching inferential functions to fail if invoked
    monkeypatch.setattr(
        scipy.stats,
        "ttest_ind",
        lambda *args, **kwargs: pytest.fail("Presenter called scipy.stats.ttest_ind!"),
    )
    monkeypatch.setattr(
        scipy.stats,
        "pearsonr",
        lambda *args, **kwargs: pytest.fail("Presenter called scipy.stats.pearsonr!"),
    )

    # Calling show() must succeed without triggering any statistical backend
    output = _capture_show(welch_workflow)
    assert "-10.00" in output


def test_invalid_detail_mode_raises_value_error(welch_workflow):
    """Verify invalid detail mode raises ValueError with helpful message."""
    with pytest.raises(ValueError, match="Invalid detail mode"):
        show(welch_workflow, detail="invalid_mode")


def test_unsupported_object_type_raises_type_error():
    """Verify arbitrary object types raise clear TypeError instead of raw repr."""
    with pytest.raises(TypeError, match="Unsupported target type for show"):
        show([1, 2, 3])

    with pytest.raises(TypeError, match="Unsupported target type for show"):
        show("string_argument")


def test_unsupported_method_raises_presentation_error(sample_numeric_df):
    """Verify non-pilot statistical methods raise clear UnsupportedPresentationError."""
    # Mann-Whitney is not in the pilot (only welch_t and pearson_correlation)
    wf_mann = ResearchAssistant(sample_numeric_df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="distribution",
        design="independent",
        variable_types={"score": "continuous"},
    )
    assert wf_mann.analysis.method_id == "mann_whitney_u"

    with pytest.raises(
        UnsupportedPresentationError,
        match="not supported in the initial presentation layer pilot",
    ):
        show(wf_mann)
