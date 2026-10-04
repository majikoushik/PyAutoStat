"""Unit tests for the PyAutoStat public PDF export API."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from pyautostat import (
    AnalysisOptions,
    ReportError,
    ResearchAssistant,
    save_pdf,
    to_pdf,
)
from pyautostat.presentation import DisplayMetric, PresentationView


@pytest.fixture
def sample_workflow():
    df = pd.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            "score": [10.0, 11.0, 10.5, 12.0, 11.5] * 4 + [14.0, 15.0, 14.5, 16.0, 15.5] * 4,
        }
    )
    assistant = ResearchAssistant(df)
    return assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=42),
    )


@pytest.fixture
def sample_report(sample_workflow):
    return sample_workflow.report


@pytest.fixture
def sample_view():
    return PresentationView(
        title="Custom Presentation View",
        subtitle="Testing PDF export with PresentationView",
        family="comparison",
        design_metrics=[DisplayMetric("Design", "Two-sample independent")],
        key_metrics=[DisplayMetric("Difference", "3.50")],
        tables=[],
        diagnostics=[],
        interpretation="Statistically distinguishable difference.",
        limitations=["Observational data."],
        warnings=[],
        metadata={"detail": "standard"},
        compact_text="Difference = 3.50",
    )


def test_public_imports():
    from pyautostat import save_pdf as top_save_pdf
    from pyautostat import to_pdf as top_to_pdf
    from pyautostat.presentation import save_pdf as pres_save_pdf
    from pyautostat.presentation import to_pdf as pres_to_pdf
    from pyautostat.presentation.pdf import (
        check_playwright_available,
        html_to_pdf_bytes,
    )
    from pyautostat.presentation.pdf import (
        save_pdf as pdf_save_pdf,
    )
    from pyautostat.presentation.pdf import (
        to_pdf as pdf_to_pdf,
    )

    assert top_to_pdf is pdf_to_pdf
    assert top_save_pdf is pdf_save_pdf
    assert pres_to_pdf is pdf_to_pdf
    assert pres_save_pdf is pdf_save_pdf
    assert callable(html_to_pdf_bytes)
    assert callable(check_playwright_available)


def test_to_pdf_parameter_validation(sample_workflow):
    with pytest.raises(ValueError, match="Invalid detail mode"):
        to_pdf(sample_workflow, detail="ultra")

    with pytest.raises(ReportError, match="style must be general, apa, or ieee"):
        to_pdf(sample_workflow, style="fancy")

    with pytest.raises(ReportError, match="title must be a non-empty string"):
        to_pdf(sample_workflow, title="")

    with pytest.raises(ReportError, match="title must be a non-empty string"):
        to_pdf(sample_workflow, title=123)

    with pytest.raises(ReportError, match="include_figures must be a Boolean"):
        to_pdf(sample_workflow, include_figures="yes")

    with pytest.raises(ReportError, match="landscape must be a Boolean"):
        to_pdf(sample_workflow, landscape="horizontal")

    with pytest.raises(ReportError, match="page_numbers must be a Boolean"):
        to_pdf(sample_workflow, page_numbers=1)

    with pytest.raises(ReportError, match="Unsupported page size"):
        to_pdf(sample_workflow, page_size="A3")

    with pytest.raises(ReportError, match="Invalid page size"):
        to_pdf(sample_workflow, page_size="")


@patch("pyautostat.presentation.pdf.api.html_to_pdf_bytes")
def test_to_pdf_static_delegates_to_html(mock_backend, sample_workflow):
    mock_backend.return_value = b"%PDF-1.4 static test"

    pdf = to_pdf(
        sample_workflow,
        detail="full",
        title="Custom Title",
        style="apa",
        page_size="letter",
        landscape=True,
        page_numbers=False,
    )

    assert pdf == b"%PDF-1.4 static test"
    mock_backend.assert_called_once()
    html_arg = mock_backend.call_args[0][0]
    assert "<!doctype html>" in html_arg.lower()
    assert "Custom Title" in html_arg
    kwargs = mock_backend.call_args[1]
    assert kwargs["page_size"] == "Letter"
    assert kwargs["landscape"] is True
    assert kwargs["page_numbers"] is False
    assert kwargs["wait_for_figures"] is False


@patch("pyautostat.presentation.pdf.api.html_to_pdf_bytes")
def test_to_pdf_figures_delegates_to_interactive_html(mock_backend, sample_workflow):
    mock_backend.return_value = b"%PDF-1.4 figures test"

    pdf = to_pdf(
        sample_workflow,
        include_figures=True,
        page_size="a4",
    )

    assert pdf == b"%PDF-1.4 figures test"
    mock_backend.assert_called_once()
    html_arg = mock_backend.call_args[0][0]
    assert "window.__pyautostatFiguresExpected" in html_arg
    kwargs = mock_backend.call_args[1]
    assert kwargs["page_size"] == "A4"
    assert kwargs["wait_for_figures"] is True


@patch("pyautostat.presentation.pdf.api.html_to_pdf_bytes")
def test_to_pdf_figures_fallback_when_no_figures_exist(mock_backend, sample_view):
    mock_backend.return_value = b"%PDF-1.4 view test"

    # PresentationView produces no figure specs; should render static without requiring Plotly
    pdf = to_pdf(sample_view, include_figures=True)

    assert pdf == b"%PDF-1.4 view test"
    mock_backend.assert_called_once()
    kwargs = mock_backend.call_args[1]
    assert kwargs["wait_for_figures"] is False


def test_save_pdf_requires_pdf_extension(sample_workflow, tmp_path):
    with pytest.raises(ReportError, match="must have a .pdf extension"):
        save_pdf(sample_workflow, tmp_path / "report.html")

    with pytest.raises(ReportError, match="must have a .pdf extension"):
        save_pdf(sample_workflow, tmp_path / "report.txt")


@patch("pyautostat.presentation.pdf.api.to_pdf")
def test_save_pdf_overwrite_protection(mock_to_pdf, sample_workflow, tmp_path):
    mock_to_pdf.return_value = b"%PDF-1.4 mock content"
    dest = tmp_path / "output.pdf"
    dest.write_bytes(b"existing content")

    with pytest.raises(ReportError, match="Report destination already exists"):
        save_pdf(sample_workflow, dest, overwrite=False)

    saved_path = save_pdf(sample_workflow, dest, overwrite=True)
    assert saved_path == dest
    assert dest.read_bytes() == b"%PDF-1.4 mock content"


@patch("pyautostat.presentation.pdf.api.to_pdf")
def test_research_report_pdf_methods(mock_to_pdf, sample_report, tmp_path):
    mock_to_pdf.return_value = b"%PDF-1.4 report bytes"

    # to_pdf on ResearchReport
    data = sample_report.to_pdf(detail="compact", page_size="Letter")
    assert data == b"%PDF-1.4 report bytes"

    # save_pdf on ResearchReport with audit ledger recording
    saved_formats = []
    sample_report._on_save = lambda fmt: saved_formats.append(fmt)

    out_file = tmp_path / "saved_report.pdf"
    result_path = sample_report.save_pdf(out_file)
    assert result_path == out_file
    assert out_file.read_bytes() == b"%PDF-1.4 report bytes"
    assert saved_formats == ["pdf"]

    # save_pdf via top-level function delegates cleanly to target.save_pdf without duplicate audit
    out_file2 = tmp_path / "saved_report2.pdf"
    save_pdf(sample_report, out_file2)
    assert out_file2.read_bytes() == b"%PDF-1.4 report bytes"
    assert saved_formats == ["pdf", "pdf"]
