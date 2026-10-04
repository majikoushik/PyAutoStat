"""Real-browser integration tests for PyAutoStat publication-ready PDF export.

Tests headless Chromium printing, layout fidelity, page formats, offline security,
zero statistical recalculation, raw-data privacy, and dynamic confidence interval preservation.
"""

from __future__ import annotations

import io
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pypdf
import pytest

from pyautostat import (
    AnalysisOptions,
    ResearchAssistant,
    to_pdf,
)
from pyautostat.presentation.pdf import check_playwright_available, html_to_pdf_bytes

# Skip all integration tests if Playwright or Chromium is not available
try:
    check_playwright_available()
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        browser.close()
    HAS_CHROMIUM = True
except Exception:
    HAS_CHROMIUM = False

pytestmark = pytest.mark.skipif(
    not HAS_CHROMIUM,
    reason="Playwright and Chromium browser binary are required for PDF integration tests.",
)


@pytest.fixture
def sample_numeric_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    group_a = rng.normal(loc=50.0, scale=5.0, size=25)
    group_b = rng.normal(loc=55.0, scale=5.0, size=25)
    return pd.DataFrame(
        {
            "group": ["Control"] * 25 + ["Treatment"] * 25,
            "score": np.concatenate([group_a, group_b]),
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
        variable_types={"score": "continuous", "group": "nominal"},
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=42),
    )


@pytest.fixture
def ols_workflow():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.1, 2.9, 4.2, 5.0, 5.9, 7.1, 8.0, 9.2, 10.1],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x2": [0.5, 0.8, 1.2, 1.9, 2.4, 3.1, 3.5, 4.2, 4.8, 5.5],
        }
    )
    return ResearchAssistant(df).run(
        objective="regression",
        outcome="y",
        predictors=["x1", "x2"],
        design="independent",
        estimand="conditional_mean",
        variable_types={"y": "continuous", "x1": "continuous", "x2": "continuous"},
    )


@pytest.fixture
def logistic_workflow():
    df = pd.DataFrame(
        {
            "event": ["no", "yes"] * 10,
            "x": list(range(20)),
        }
    )
    return ResearchAssistant(df).run(
        objective="regression",
        outcome="event",
        predictors=["x"],
        design="independent",
        estimand="event_probability",
        event_level="yes",
        variable_types={"event": "nominal", "x": "continuous"},
    )


@pytest.fixture
def chisq_workflow():
    df = pd.DataFrame(
        {
            "treatment": ["A"] * 25 + ["B"] * 25,
            "outcome": ["pass"] * 20 + ["fail"] * 5 + ["pass"] * 10 + ["fail"] * 15,
        }
    )
    return ResearchAssistant(df).run(
        objective="association",
        outcome="outcome",
        predictor="treatment",
        design="independent",
        estimand="categorical_independence",
        variable_types={"outcome": "nominal", "treatment": "nominal"},
    )


def test_welch_static_pdf(welch_workflow):
    pdf = to_pdf(welch_workflow, detail="standard", include_figures=False)
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 2000

    reader = pypdf.PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) >= 1
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Welch" in text or "Two-sample" in text or "Difference" in text
    assert "95% CI" in text


def test_ols_full_pdf(ols_workflow):
    pdf = to_pdf(ols_workflow, detail="full", include_figures=False)
    assert pdf.startswith(b"%PDF")

    reader = pypdf.PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) >= 1
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Regression" in text or "Coefficients" in text or "R-squared" in text


def test_research_report_pdf(welch_workflow, tmp_path):
    report = welch_workflow.report
    assert report is not None

    pdf = report.to_pdf(detail="full", page_numbers=True)
    assert pdf.startswith(b"%PDF")

    reader = pypdf.PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) >= 1
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Executive Summary" in text or "Status" in text
    assert "Sample" in text or "Design" in text
    assert "Page 1 of" in text or "1 of" in text

    dest = tmp_path / "report_integration.pdf"
    report.save_pdf(dest)
    assert dest.exists()
    assert dest.read_bytes().startswith(b"%PDF")


def test_page_size_and_orientation():
    html = "<html><body><h1>Page Size Test</h1></body></html>"

    # A4 Portrait
    a4_pdf = html_to_pdf_bytes(html, page_size="A4", landscape=False)
    a4_reader = pypdf.PdfReader(io.BytesIO(a4_pdf))
    a4_page = a4_reader.pages[0]
    # A4: 595.28 x 841.89 pt
    assert 590 <= float(a4_page.mediabox.width) <= 600
    assert 835 <= float(a4_page.mediabox.height) <= 845

    # Letter Portrait
    letter_pdf = html_to_pdf_bytes(html, page_size="Letter", landscape=False)
    letter_reader = pypdf.PdfReader(io.BytesIO(letter_pdf))
    letter_page = letter_reader.pages[0]
    # Letter: 612 x 792 pt
    assert 608 <= float(letter_page.mediabox.width) <= 616
    assert 788 <= float(letter_page.mediabox.height) <= 796

    # Landscape: width > height
    landscape_pdf = html_to_pdf_bytes(html, page_size="A4", landscape=True)
    land_reader = pypdf.PdfReader(io.BytesIO(landscape_pdf))
    land_page = land_reader.pages[0]
    assert float(land_page.mediabox.width) > float(land_page.mediabox.height)


def test_offline_network_blocked():
    # Attempting to load an external resource must be blocked by route interception
    html = (
        "<!doctype html><html><body>"
        "<h1>Offline Report</h1>"
        '<img src="https://example.com/blocked_image.png" alt="external image">'
        '<script src="https://example.com/blocked_script.js"></script>'
        "</body></html>"
    )
    pdf = html_to_pdf_bytes(html)
    assert pdf.startswith(b"%PDF")
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    assert "Offline Report" in reader.pages[0].extract_text()


def test_figure_enabled_pdf_generation(
    welch_workflow, ols_workflow, logistic_workflow, chisq_workflow
):
    # Welch with figure
    welch_pdf = to_pdf(welch_workflow, include_figures=True)
    assert welch_pdf.startswith(b"%PDF")
    welch_reader = pypdf.PdfReader(io.BytesIO(welch_pdf))
    assert len(welch_reader.pages) >= 1

    # OLS with forest plot
    ols_pdf = to_pdf(ols_workflow, include_figures=True)
    assert ols_pdf.startswith(b"%PDF")
    ols_reader = pypdf.PdfReader(io.BytesIO(ols_pdf))
    assert len(ols_reader.pages) >= 1

    # Logistic with forest plot
    logistic_pdf = to_pdf(logistic_workflow, include_figures=True)
    assert logistic_pdf.startswith(b"%PDF")
    log_reader = pypdf.PdfReader(io.BytesIO(logistic_pdf))
    assert len(log_reader.pages) >= 1

    # Chi-square with heatmap
    chisq_pdf = to_pdf(chisq_workflow, include_figures=True)
    assert chisq_pdf.startswith(b"%PDF")
    chisq_reader = pypdf.PdfReader(io.BytesIO(chisq_pdf))
    assert len(chisq_reader.pages) >= 1


def test_zero_statistical_recalculation(welch_workflow):
    # Monkeypatch statistical engines to raise if called during PDF export
    with patch(
        "scipy.stats.ttest_ind",
        side_effect=RuntimeError("Statistical recalculation during PDF export!"),
    ):
        with patch(
            "scipy.stats.norm",
            side_effect=RuntimeError("Statistical recalculation during PDF export!"),
        ):
            # 1. Static PDF
            pdf_static = to_pdf(welch_workflow)
            assert pdf_static.startswith(b"%PDF")

            # 2. Report PDF
            pdf_report = welch_workflow.report.to_pdf()
            assert pdf_report.startswith(b"%PDF")

            # 3. Figure-enabled PDF
            pdf_figures = to_pdf(welch_workflow, include_figures=True)
            assert pdf_figures.startswith(b"%PDF")


def test_raw_data_privacy_guarantee():
    sentinel_values = [
        913.123456789,
        827.987654321,
        765.432109876,
        654.321098765,
    ]
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "score": sentinel_values,
        }
    )
    workflow = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
        options=AnalysisOptions(confidence_level=0.95),
    )

    pdf_static = to_pdf(workflow, include_figures=False)
    pdf_figures = to_pdf(workflow, include_figures=True)

    for pdf in (pdf_static, pdf_figures):
        reader = pypdf.PdfReader(io.BytesIO(pdf))
        full_text = "\n".join(page.extract_text() for page in reader.pages)
        for s in sentinel_values:
            assert str(s) not in full_text


def test_dynamic_confidence_interval_labels():
    df = pd.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            "val": [10.0, 11.0, 10.5, 12.0, 11.5] * 4 + [14.0, 15.0, 14.5, 16.0, 15.5] * 4,
        }
    )
    asst = ResearchAssistant(df)

    # 90% CI
    wf90 = asst.run(
        objective="compare_groups",
        outcome="val",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"val": "continuous", "group": "nominal"},
        options=AnalysisOptions(confidence_level=0.90),
    )
    pdf90 = to_pdf(wf90)
    reader90 = pypdf.PdfReader(io.BytesIO(pdf90))
    text90 = "\n".join(p.extract_text() for p in reader90.pages)
    assert "90% CI" in text90

    # 99% CI
    wf99 = asst.run(
        objective="compare_groups",
        outcome="val",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"val": "continuous", "group": "nominal"},
        options=AnalysisOptions(confidence_level=0.99),
    )
    pdf99 = to_pdf(wf99)
    reader99 = pypdf.PdfReader(io.BytesIO(pdf99))
    text99 = "\n".join(p.extract_text() for p in reader99.pages)
    assert "99% CI" in text99


def test_partial_result_pdf_export(welch_workflow):
    # Simulate partial result: effect_size is None
    vals = deepcopy(welch_workflow.analysis.values)
    vals["effect_size"] = None
    partial_analysis = replace(welch_workflow.analysis, values=vals)

    pdf = to_pdf(partial_analysis)
    assert pdf.startswith(b"%PDF")
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) >= 1
    text = "\n".join(p.extract_text() for p in reader.pages)
    assert "Mean difference" in text or "Difference" in text
