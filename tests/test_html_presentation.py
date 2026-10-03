"""Comprehensive unit and integration tests for the PyAutoStat HTML presentation layer.

Validates:
- Public to_html and save_html APIs, detail modes, style modes, overwrite safety.
- Cross-renderer fidelity with the normalized presentation view model.
- Complete coverage for Phase 1 representative methods:
  * Welch independent-samples t-test
  * Pearson correlation
  * OLS linear regression
  * Kruskal-Wallis rank test
- Zero-recalculation guarantee under inferential engine monkeypatching.
- Accessibility, print stylesheets, responsive containers, semantic tables.
- Security against XSS and text escaping.
- Dynamic confidence interval formatting (90%, 95%, 99%).
- ResearchReport integration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pyautostat import (
    AnalysisOptions,
    ReportError,
    ResearchAssistant,
    ResearchWorkflowResult,
    save_html,
    to_html,
)
from pyautostat.presentation import PresentationView, TerminalView, adapt

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def two_group_numeric_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        }
    )


@pytest.fixture
def welch_workflow(two_group_numeric_df: pd.DataFrame) -> ResearchWorkflowResult:
    return ResearchAssistant(two_group_numeric_df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=42),
    )


@pytest.fixture
def pearson_workflow() -> ResearchWorkflowResult:
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
def ols_workflow() -> ResearchWorkflowResult:
    df = pd.DataFrame(
        {
            "y": [1.5, 2.8, 3.2, 4.9, 5.1, 6.4, 7.2, 8.5, 9.1, 10.3],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x2": [0.5, 1.2, 1.8, 2.5, 3.1, 3.9, 4.8, 5.2, 6.1, 7.0],
        }
    )
    return ResearchAssistant(df).run(
        objective="regression",
        outcome="y",
        predictors=["x1", "x2"],
        design="independent",
        estimand="conditional_mean",
        variable_types={"y": "continuous", "x1": "continuous", "x2": "continuous"},
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=42),
    )


@pytest.fixture
def kruskal_workflow() -> ResearchWorkflowResult:
    df = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "score": [1, 2, 3, 4, 5, 3, 4, 6, 7, 8, 8, 9, 10, 11, 13],
        }
    )
    return ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="distribution",
        variable_types={"score": "continuous", "group": "nominal"},
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=42),
    )


# ── Public API Tests ─────────────────────────────────────────────────────────


def test_to_html_returns_valid_html_string(welch_workflow):
    html = to_html(welch_workflow)
    assert isinstance(html, str)
    assert "<!doctype html>" in html
    assert '<html lang="en">' in html
    assert '<meta charset="utf-8">' in html
    assert "Independent Group Comparison" in html
    assert "</html>" in html


def test_save_html_writes_file_and_protects_overwrite(welch_workflow, tmp_path: Path):
    dest = tmp_path / "report.html"
    assert not dest.exists()

    saved = save_html(welch_workflow, dest)
    assert saved == dest
    assert dest.exists()
    content = dest.read_text(encoding="utf-8")
    assert "<!doctype html>" in content
    assert "Welch's independent-samples t-test" in content

    # Overwrite protection: should fail by default
    with pytest.raises(ReportError, match="Report destination already exists"):
        save_html(welch_workflow, dest, overwrite=False)

    # Overwrite allowed when overwrite=True
    save_html(welch_workflow, dest, overwrite=True)
    assert dest.exists()


def test_invalid_detail_mode_raises(welch_workflow):
    with pytest.raises(ValueError, match="Invalid detail mode 'invalid'"):
        to_html(welch_workflow, detail="invalid")


def test_invalid_style_mode_raises(welch_workflow):
    with pytest.raises(ReportError, match="style must be general, apa, or ieee"):
        to_html(welch_workflow, style="invalid")


def test_invalid_title_raises(welch_workflow):
    with pytest.raises(ReportError, match="title must be a non-empty string"):
        to_html(welch_workflow, title="   ")


def test_title_override_honored(welch_workflow):
    html = to_html(welch_workflow, title="Quarterly Sales Group Comparison")
    assert "Quarterly Sales Group Comparison" in html
    assert "<title>Quarterly Sales Group Comparison</title>" in html


# ── Target Types Supported ───────────────────────────────────────────────────


def test_direct_analysis_result_renders_safely(welch_workflow):
    assert welch_workflow.analysis is not None
    html = to_html(welch_workflow.analysis)
    assert "<!doctype html>" in html
    assert "Welch's independent-samples t-test" in html
    assert "Mean difference" in html
    assert "-10.00" in html


def test_research_report_renders_via_to_html(welch_workflow):
    assert welch_workflow.report is not None
    html = to_html(welch_workflow.report)
    assert "<!doctype html>" in html
    assert "Executive Summary" in html
    assert "Research question and design" in html


def test_presentation_view_renders_directly(welch_workflow):
    view = adapt(welch_workflow)
    assert isinstance(view, PresentationView)
    assert isinstance(view, TerminalView)
    html = to_html(view)
    assert "<!doctype html>" in html
    assert "Welch's independent-samples t-test" in html


# ── Detail Modes ─────────────────────────────────────────────────────────────


def test_detail_compact_renders_concise_summary(welch_workflow):
    html = to_html(welch_workflow, detail="compact")
    assert "KEY RESULTS" in html
    assert "Mean difference" in html
    assert "-10.00" in html
    # Should not render full design grid or summary tables in compact mode
    assert "ANALYSIS DESIGN" not in html
    assert "GROUP SUMMARY" not in html
    assert "DIAGNOSTIC CONTEXT" not in html


def test_detail_standard_renders_complete_sections(welch_workflow):
    html = to_html(welch_workflow, detail="standard")
    assert "ANALYSIS DESIGN" in html
    assert "KEY RESULTS" in html
    assert "GROUP SUMMARY" in html
    assert "INTERPRETATION" in html
    assert "DIAGNOSTIC CONTEXT" in html
    assert "IMPORTANT LIMITATIONS" in html


def test_detail_full_renders_governance_and_records(welch_workflow):
    html = to_html(welch_workflow, detail="full")
    assert "ANALYSIS DESIGN" in html
    assert "KEY RESULTS" in html
    assert "GROUP SUMMARY" in html
    assert "DIAGNOSTIC CONTEXT" in html
    assert "ANALYSIS RECORD" in html


# ── Reporting Styles (General, APA, IEEE) ─────────────────────────────────────


def test_style_ieee_numbers_sections(welch_workflow):
    html = to_html(welch_workflow, style="ieee")
    assert "1. ANALYSIS DESIGN" in html
    assert "2. KEY RESULTS" in html
    assert "3. GROUP SUMMARY" in html


def test_style_general_uses_clean_headings(welch_workflow):
    html = to_html(welch_workflow, style="general")
    assert "1. ANALYSIS DESIGN" not in html
    assert "ANALYSIS DESIGN" in html


def test_style_apa_preserves_wording(welch_workflow):
    html = to_html(welch_workflow, style="apa")
    assert "<!doctype html>" in html
    assert "ANALYSIS DESIGN" in html


# ── Security & Escaping Tests ────────────────────────────────────────────────


def test_escaping_of_malicious_inputs():
    malicious_outcome = "<script>alert('outcome')</script>"
    malicious_group_a = "<b>GroupA</b>"
    malicious_group_b = "<img src=x onerror=alert('group')>"

    df = pd.DataFrame(
        {
            malicious_outcome: [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            "grp": [
                malicious_group_a,
                malicious_group_a,
                malicious_group_a,
                malicious_group_b,
                malicious_group_b,
                malicious_group_b,
            ],
        }
    )
    workflow = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome=malicious_outcome,
        predictor="grp",
        estimand="mean",
        design="independent",
        variable_types={malicious_outcome: "continuous", "grp": "nominal"},
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=42),
    )
    html = to_html(workflow, title="<script>alert('title')</script>")

    # Executable scripts must NEVER be present
    assert "<script>" not in html
    assert "<img src=x" not in html
    assert "<b>GroupA</b>" not in html

    # Escaped safe representations MUST be present
    assert (
        "&lt;script&gt;alert(&#x27;outcome&#x27;)&lt;/script&gt;" in html
        or "&lt;script&gt;" in html
    )
    assert "&lt;img src=x" in html or "&lt;img" in html
    assert "&lt;b&gt;GroupA&lt;/b&gt;" in html
    assert "&lt;script&gt;alert(&#x27;title&#x27;)&lt;/script&gt;" in html


# ── Method 1: Welch t-test Fidelity & Structure ──────────────────────────────


def test_welch_t_html_fidelity(welch_workflow):
    view = adapt(welch_workflow)
    html = to_html(welch_workflow)

    # Design
    assert "Welch's independent-samples t-test" in html
    assert "Population mean difference" in html
    assert "'A' - 'B'" in html

    # Key Results
    assert "Mean difference" in html
    assert "-10.00" in html  # Preserves signed estimate
    assert "95% CI" in html
    assert "Cohen's d" in html
    assert "p-value" in html

    # Cross-renderer value consistency with view
    diff_metric = next(m for m in view.key_metrics if m.label == "Mean difference")
    ci_metric = next(m for m in view.key_metrics if "95% CI" in m.label)
    assert diff_metric.value in html
    assert ci_metric.value in html

    # Group summary table
    assert "GROUP SUMMARY" in html
    assert '<th scope="col" class="align-left">Group</th>' in html
    assert '<th scope="col" class="align-right">N</th>' in html

    # Diagnostics
    assert "[NOT ASSUMED]" in html
    assert "Welch test does not assume equal population variances." in html


# ── Method 2: Pearson Correlation Fidelity & Structure ───────────────────────


def test_pearson_correlation_html_fidelity(pearson_workflow):
    view = adapt(pearson_workflow)
    html = to_html(pearson_workflow)

    # Design
    assert "Pearson correlation" in html
    assert "Linear association" in html
    assert "10 paired observations" in html

    # Key Results
    assert "Correlation" in html
    assert "95% CI" in html
    assert "p-value" in html

    # Cross-renderer check
    r_metric = next(m for m in view.key_metrics if m.label == "Correlation")
    p_metric = next(m for m in view.key_metrics if m.label == "p-value")
    assert r_metric.value in html
    assert p_metric.value in html or "&lt;0.001" in html

    # Limitations
    assert "causation" in html.lower()


# ── Method 3: Linear Regression (OLS) Fidelity & Structure ───────────────────


def test_linear_regression_html_fidelity(ols_workflow):
    view = adapt(ols_workflow)
    html = to_html(ols_workflow, detail="full")

    # Design / Specification
    assert "MODEL SPECIFICATION" in html
    assert "Ordinary Least Squares (OLS) Regression" in html
    assert "Outcome" in html
    assert "Predictors" in html

    # Model Fit
    assert "MODEL FIT" in html
    assert "R-squared" in html
    assert "Adjusted R-squared" in html
    assert "Model F-test" in html
    assert "Model p-value" in html

    # Coefficients Table
    assert "MODEL COEFFICIENTS" in html
    assert '<th scope="col" class="align-left">Term</th>' in html
    assert '<th scope="col" class="align-right">Estimate</th>' in html
    assert '<th scope="col" class="align-right">Std Error</th>' in html
    assert '<th scope="col" class="align-right">t</th>' in html
    assert '<th scope="col" class="align-right">p-value</th>' in html

    # Stable source order (Intercept first, then x1, x2 in table cells)
    pos_intercept = html.find('<td class="align-left">Intercept</td>')
    pos_x1 = html.find('<td class="align-left">x1</td>')
    pos_x2 = html.find('<td class="align-left">x2</td>')
    assert -1 < pos_intercept < pos_x1 < pos_x2

    # Diagnostics
    assert "Breusch-Pagan" in html
    assert "[DOCUMENTED]" in html or "[REVIEW]" in html

    # Cross-renderer check
    r2_metric = next(m for m in view.key_metrics if m.label == "R-squared")
    assert r2_metric.value in html


# ── Method 4: Kruskal-Wallis Fidelity & Structure ────────────────────────────


def test_kruskal_wallis_html_fidelity(kruskal_workflow):
    view = adapt(kruskal_workflow)
    html = to_html(kruskal_workflow, detail="full")

    # Design
    assert "Kruskal-Wallis rank sum test" in html
    assert "Equality of population rank distributions" in html

    # Omnibus
    assert "OMNIBUS KEY RESULT" in html
    assert "Omnibus Test" in html
    assert "H" in html
    assert "p-value" in html

    # Group Summary schema: Group, N, Median (No fabricated IQR)
    assert "GROUP SUMMARY" in html
    assert '<th scope="col" class="align-left">Group</th>' in html
    assert '<th scope="col" class="align-right">N</th>' in html
    assert '<th scope="col" class="align-right">Median</th>' in html
    assert '<th scope="col" class="align-right">IQR</th>' not in html

    # Dunn-Holm Pairwise Comparisons
    assert "DUNN-HOLM PAIRWISE COMPARISONS" in html
    assert "Contrast" in html
    assert "Mean-Rank Diff" in html
    assert "Dunn z" in html
    assert "Rank-biserial r" in html
    assert "Adjusted p" in html
    assert "Decision" in html

    # Diagnostics: Equal variance NOT APPLICABLE
    assert "[NOT APPLICABLE]" in html
    assert (
        "equal-variance" in html.lower()
        or "not invoked" in html.lower()
        or "not assumed" in html.lower()
    )

    # Cross-renderer check
    h_metric = next(m for m in view.key_metrics if m.label == "Omnibus Test")
    assert h_metric.value in html


def test_kruskal_wallis_preserves_group_label_zero():
    df = pd.DataFrame(
        {
            "group": [0] * 5 + [1] * 5 + [2] * 5,
            "score": [1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 7, 8, 9, 10, 11],
        }
    )
    wf = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="distribution",
        variable_types={"score": "continuous", "group": "nominal"},
    )
    html = to_html(wf)
    # The numeric 0 group label must be preserved in HTML, not empty or None
    assert '<td class="align-left">0</td>' in html


# ── Dynamic Confidence Levels ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "conf_level,expected_label",
    [
        (0.90, "90% CI"),
        (0.95, "95% CI"),
        (0.99, "99% CI"),
    ],
)
def test_dynamic_confidence_interval_labels(two_group_numeric_df, conf_level, expected_label):
    wf = ResearchAssistant(two_group_numeric_df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
        options=AnalysisOptions(
            alpha=1.0 - conf_level,
            confidence_level=conf_level,
            random_seed=42,
        ),
    )
    html = to_html(wf)
    assert expected_label in html
    assert f"Effect {expected_label}" in html


# ── Zero-Recalculation Test ──────────────────────────────────────────────────


def test_zero_recalculation_on_html_rendering(welch_workflow):
    """Ensure that to_html never recalculates or re-runs any statistical inference."""
    result = welch_workflow.analysis
    assert result is not None

    # Patch scipy and statsmodels inference functions to raise if called
    with (
        patch("scipy.stats.ttest_ind", side_effect=RuntimeError("ttest_ind must not be called")),
        patch("scipy.stats.pearsonr", side_effect=RuntimeError("pearsonr must not be called")),
        patch("scipy.stats.kruskal", side_effect=RuntimeError("kruskal must not be called")),
    ):
        html_wf = to_html(welch_workflow)
        html_res = to_html(result)

    assert "<!doctype html>" in html_wf
    assert "<!doctype html>" in html_res
    assert "Mean difference" in html_wf
    assert "Mean difference" in html_res


# ── Accessibility & Print Styles ─────────────────────────────────────────────


def test_accessibility_and_print_structure(welch_workflow):
    html = to_html(welch_workflow)

    # HTML5 standards
    assert '<html lang="en">' in html
    assert '<meta charset="utf-8">' in html
    assert '<meta name="viewport"' in html

    # Semantic headings
    assert "<h1" in html
    assert "<h2" in html
    assert '<th scope="col"' in html

    # Responsive table container
    assert '<div class="table-container">' in html

    # Print stylesheet
    assert "@media print" in html
    assert "break-inside: avoid" in html
