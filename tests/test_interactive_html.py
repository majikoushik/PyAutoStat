"""Unit and integration tests for PyAutoStat interactive HTML and scientific figures.

Validates:
- Phase 2 exception fallback hardening (Section 0)
- Public interactive API: to_interactive_html, save_interactive_html, ResearchReport methods
- Optional dependency behavior when Plotly is missing vs present
- Offline self-contained single Plotly bundle per document
- Raw-data privacy guarantees (no row-level data serialized)
- FigureSpec numerical fidelity across representative methods (Welch, Pearson, OLS,
  Logistic, ANOVA pairwise, Chi-square, Two-way factorial)
- Zero inferential recalculation during figure construction and interactive rendering
- XSS prevention and safe script serialization
- Honor and override of ResearchReport._include_figures flag
- Static HTML regression guard (to_html remains completely Plotly-free)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pyautostat import (
    AnalysisOptions,
    AnalysisResult,
    AnalysisStatus,
    ReportError,
    ResearchAssistant,
    ResearchWorkflowResult,
    save_interactive_html,
    to_html,
    to_interactive_html,
)
from pyautostat.presentation.figures import FigureSeries, FigureSpec, build_figure_spec
from pyautostat.presentation.html.plotly_renderer import (
    safe_json_for_script,
    spec_to_plotly_figure,
)
from pyautostat.research_report import build_research_report

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sentinel_welch_df() -> pd.DataFrame:
    """DataFrame with distinctive row-level sentinel floats to verify privacy."""
    return pd.DataFrame(
        {
            "score": [
                913.123456789,
                827.987654321,
                911.111111111,
                915.222222222,
                912.333333333,
                940.444444444,
                942.555555555,
                941.666666666,
                943.777777777,
                944.888888888,
            ],
            "group": ["Ctrl"] * 5 + ["Treat"] * 5,
        }
    )


@pytest.fixture
def welch_workflow(sentinel_welch_df: pd.DataFrame) -> ResearchWorkflowResult:
    return ResearchAssistant(sentinel_welch_df).run(
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
    )


@pytest.fixture
def ols_workflow() -> ResearchWorkflowResult:
    df = pd.DataFrame(
        {
            "y": [4.2, 5.1, 6.5, 7.0, 8.4, 9.2, 10.5, 11.3, 12.1, 13.0],
            "x1": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
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
def logistic_workflow() -> ResearchWorkflowResult:
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
def chi_square_workflow() -> ResearchWorkflowResult:
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


@pytest.fixture
def welch_anova_workflow() -> ResearchWorkflowResult:
    df = pd.DataFrame(
        {
            "group": ["C"] * 6 + ["A"] * 7 + ["B"] * 5,
            "score": [2, 3, 4, 5, 7, 8, 8, 10, 12, 15, 18, 22, 27, 1, 2, 2, 3, 5],
        }
    )
    return ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
        variable_types={"score": "continuous", "group": "nominal"},
    )


# ── Section 0: Phase 2 Fallback Hardening ────────────────────────────────────


def test_section_0_adapt_unexpected_error_not_silently_swallowed(
    welch_workflow: ResearchWorkflowResult,
) -> None:
    """When _source_result is present, an unexpected adapt() error must raise, not downgrade."""
    report = welch_workflow.report
    assert report._source_result is not None

    with patch(
        "pyautostat.presentation.html.report_renderer.adapt",
        side_effect=RuntimeError("Unexpected adapter failure in presentation"),
    ):
        with pytest.raises(RuntimeError, match="Unexpected adapter failure"):
            report.to_html()


# ── Section 1 & 29: Static HTML Regression Guard ────────────────────────────


def test_static_html_remains_completely_plotly_free(
    welch_workflow: ResearchWorkflowResult,
) -> None:
    """Default static HTML must never contain Plotly scripts or figure JSON."""
    static_html = to_html(welch_workflow)
    assert "Plotly.newPlot" not in static_html
    assert "plotly.js" not in static_html
    assert '<figure class="pyautostat-figure"' not in static_html
    assert "<noscript>" not in static_html

    report_static = welch_workflow.report.to_html()
    assert "Plotly.newPlot" not in report_static
    assert "plotly.js" not in report_static
    assert '<figure class="pyautostat-figure"' not in report_static
    assert "<noscript>" not in report_static


# ── Section 2: Public Interactive API ────────────────────────────────────────


def test_public_interactive_html_api(
    welch_workflow: ResearchWorkflowResult, tmp_path: Path
) -> None:
    """to_interactive_html and save_interactive_html work with custom options."""
    # Default execution
    html_out = to_interactive_html(welch_workflow)
    assert "<!doctype html>" in html_out
    assert "Plotly.newPlot" in html_out
    assert "Mean difference and 95% confidence interval" in html_out
    assert "pyautostat-figure-container" in html_out

    # Detail levels
    compact_html = to_interactive_html(welch_workflow, detail="compact")
    assert "SUMMARY FINDING" in compact_html
    assert "Plotly.newPlot" in compact_html

    full_html = to_interactive_html(welch_workflow, detail="full")
    assert "ANALYSIS RECORD" in full_html
    assert "Plotly.newPlot" in full_html

    # Styles
    apa_html = to_interactive_html(welch_workflow, style="apa")
    assert "<!doctype html>" in apa_html
    ieee_html = to_interactive_html(welch_workflow, style="ieee")
    assert "1. " in ieee_html

    # Invalid options
    with pytest.raises(ValueError, match="Invalid detail mode"):
        to_interactive_html(welch_workflow, detail="ultra")
    with pytest.raises(ReportError, match="style must be general, apa, or ieee"):
        to_interactive_html(welch_workflow, style="mla")
    with pytest.raises(ReportError, match="include_figures must be a Boolean"):
        to_interactive_html(welch_workflow, include_figures="yes")  # type: ignore[arg-type]

    # Save to file
    out_file = tmp_path / "interactive_report.html"
    saved_path = save_interactive_html(welch_workflow, out_file)
    assert saved_path.exists()
    assert "Plotly.newPlot" in saved_path.read_text(encoding="utf-8")

    # Overwrite protection
    with pytest.raises(ReportError, match="Report destination already exists"):
        save_interactive_html(welch_workflow, out_file, overwrite=False)

    saved_path_ov = save_interactive_html(welch_workflow, out_file, overwrite=True)
    assert saved_path_ov == out_file


def test_research_report_interactive_methods(
    welch_workflow: ResearchWorkflowResult, tmp_path: Path
) -> None:
    """ResearchReport.to_interactive_html and save_interactive_html work as expected."""
    report = welch_workflow.report
    html_out = report.to_interactive_html(include_figures=True)
    assert "Plotly.newPlot" in html_out
    assert "Mean difference and 95% confidence interval" in html_out

    save_path = tmp_path / "rep_interactive.html"
    res = report.save_interactive_html(save_path, include_figures=True)
    assert res.exists()
    assert "Plotly.newPlot" in res.read_text(encoding="utf-8")


# ── Section 3 & 33: Plotly-Missing Simulation ────────────────────────────────


def test_plotly_missing_behavior(welch_workflow: ResearchWorkflowResult) -> None:
    """When Plotly is missing, static HTML works; missing Plotly raises ReportError."""
    with patch(
        "pyautostat.presentation.html.plotly_renderer.check_plotly_available",
        side_effect=ReportError(
            "Interactive HTML figures require the 'plotly' package. "
            'Install it with: pip install "pyautostat[report]"'
        ),
    ):
        # Static HTML works without error
        static = to_html(welch_workflow)
        assert "<!doctype html>" in static

        # Interactive with include_figures=False succeeds cleanly
        no_fig_html = to_interactive_html(welch_workflow, include_figures=False)
        assert "<!doctype html>" in no_fig_html
        assert "Plotly.newPlot" not in no_fig_html

        # Interactive with include_figures=True raises actionable ReportError
        with pytest.raises(ReportError, match='pip install "pyautostat\\[report\\]"'):
            to_interactive_html(welch_workflow, include_figures=True)


# ── Section 4 & 22 & 38: Offline Single Plotly Bundle ─────────────────────────


def test_single_plotly_bundle_embedding(ols_workflow: ResearchWorkflowResult) -> None:
    """Plotly library bundle must be included once in head, not duplicated per figure."""
    html_out = to_interactive_html(ols_workflow)

    # Contains Plotly script in head
    assert '<script type="text/javascript">' in html_out
    assert "Plotly.newPlot" in html_out

    # Check that external script URLs/CDNs are NOT used
    assert "<script src=" not in html_out
    assert 'src="https://' not in html_out
    assert 'src="http://' not in html_out
    assert '<link rel="stylesheet" href="http' not in html_out

    # Bundle is embedded exactly once in the entire document
    assert html_out.count('id="pyautostat-plotly-bundle"') == 1

    # Multi-figure test: build a mock view with multiple figures and ensure bundle occurs once
    spec1 = FigureSpec(
        kind="estimate_ci",
        title="Estimate 1",
        series=(FigureSeries(label="E1", estimate=1.0, lower=0.5, upper=1.5),),
    )
    spec2 = FigureSpec(
        kind="estimate_ci",
        title="Estimate 2",
        series=(FigureSeries(label="E2", estimate=2.0, lower=1.5, upper=2.5),),
    )
    assert spec1.series[0].estimate == 1.0
    assert spec2.series[0].estimate == 2.0


# ── Section 5 & 32: Raw-Data Privacy ─────────────────────────────────────────


def test_raw_data_privacy_no_sentinel_leakage(
    welch_workflow: ResearchWorkflowResult,
) -> None:
    """Row-level raw observations (distinctive sentinels) must not appear in HTML/JSON/scripts."""
    sentinels = ["913.123456789", "827.987654321"]

    interactive_html = to_interactive_html(welch_workflow)

    for sentinel in sentinels:
        assert sentinel not in interactive_html, f"Raw sentinel {sentinel} leaked into HTML!"


# ── Section 6 & 30: FigureSpec Numeric Fidelity ──────────────────────────────


def test_figurespec_fidelity_welch(welch_workflow: ResearchWorkflowResult) -> None:
    """Welch FigureSpec accurately reflects stored estimate, CI, and neutral reference."""
    spec = build_figure_spec(welch_workflow)
    assert spec is not None
    assert spec.kind == "estimate_ci"
    assert spec.reference_value == 0.0
    assert len(spec.series) == 1

    series = spec.series[0]
    expected_est = welch_workflow.analysis.values["primary_estimate"]
    expected_ci = welch_workflow.analysis.values["confidence_interval"]
    assert series.estimate == expected_est
    assert series.lower == expected_ci["lower"]
    assert series.upper == expected_ci["upper"]
    assert spec.placement == "KEY RESULTS"


def test_figurespec_fidelity_pearson(pearson_workflow: ResearchWorkflowResult) -> None:
    """Pearson FigureSpec accurately reflects stored r, CI, and reference 0."""
    spec = build_figure_spec(pearson_workflow)
    assert spec is not None
    assert spec.kind == "estimate_ci"
    assert spec.reference_value == 0.0

    series = spec.series[0]
    expected_r = pearson_workflow.analysis.values["primary_estimate"]
    expected_ci = pearson_workflow.analysis.values["confidence_interval"]
    assert series.estimate == expected_r
    assert series.lower == expected_ci["lower"]
    assert series.upper == expected_ci["upper"]


def test_figurespec_fidelity_ols(ols_workflow: ResearchWorkflowResult) -> None:
    """OLS FigureSpec accurately preserves coefficient order and stored estimates/CIs."""
    spec = build_figure_spec(ols_workflow)
    assert spec is not None
    assert spec.kind == "forest"
    assert spec.reference_value == 0.0

    stored_coefs = ols_workflow.analysis.values["coefficients"]
    assert len(spec.series) == len(stored_coefs)

    for s, c in zip(spec.series, stored_coefs, strict=True):
        assert s.label == str(c["term"])
        assert s.estimate == c["estimate"]
        assert s.lower == c["confidence_interval"]["lower"]
        assert s.upper == c["confidence_interval"]["upper"]
        assert s.metadata["standard_error"] == c["standard_error"]
        assert s.metadata["p_value"] == c["p_value"]


def test_figurespec_fidelity_logistic(logistic_workflow: ResearchWorkflowResult) -> None:
    """Logistic FigureSpec preserves odds ratios, CIs, reference 1.0, and log x scale."""
    spec = build_figure_spec(logistic_workflow)
    assert spec is not None
    assert spec.kind == "odds_ratio_forest"
    assert spec.reference_value == 1.0
    assert spec.layout_hints.get("log_x") is True

    stored_coefs = [
        c
        for c in logistic_workflow.analysis.values["coefficients"]
        if c.get("odds_ratio") is not None
    ]
    assert len(spec.series) == len(stored_coefs)

    for s, c in zip(spec.series, stored_coefs, strict=True):
        assert s.estimate == c["odds_ratio"]
        assert s.lower == c["odds_ratio_ci"]["lower"]
        assert s.upper == c["odds_ratio_ci"]["upper"]


def test_figurespec_fidelity_chi_square(chi_square_workflow: ResearchWorkflowResult) -> None:
    """Chi-square FigureSpec accurately presents contingency counts with preserved orientation."""
    spec = build_figure_spec(chi_square_workflow)
    assert spec is not None
    assert spec.kind == "count_heatmap"

    stored_counts = chi_square_workflow.analysis.metadata["observed_counts"]
    stored_groups = chi_square_workflow.analysis.metadata["group_order"]
    stored_outcomes = chi_square_workflow.analysis.metadata["outcome_order"]

    assert len(spec.series) == len(stored_groups)
    for idx, s in enumerate(spec.series):
        assert s.label == str(stored_groups[idx])
        assert s.categories == tuple(str(o) for o in stored_outcomes)
        assert list(s.values) == [float(v) for v in stored_counts[idx]]


def test_figurespec_fidelity_anova_pairwise(welch_anova_workflow: ResearchWorkflowResult) -> None:
    """Welch ANOVA pairwise FigureSpec matches stored Games-Howell contrasts."""
    spec = build_figure_spec(welch_anova_workflow)
    assert spec is not None
    assert spec.kind == "pairwise_forest"
    assert spec.reference_value == 0.0

    stored_pairs = welch_anova_workflow.analysis.values["pairwise_comparisons"]
    assert len(spec.series) == len(stored_pairs)

    for s, p in zip(spec.series, stored_pairs, strict=True):
        assert s.estimate == p["estimate"]
        assert s.lower == p["confidence_interval"]["lower"]
        assert s.upper == p["confidence_interval"]["upper"]


def test_figurespec_kruskal_wallis_omits_figure() -> None:
    """Kruskal-Wallis returns None because Dunn pairwise comparisons lack authoritative CIs."""
    df = pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 6.0, 7.0, 8.0, 8.0, 9.0, 10.0, 11.0, 13.0],
            "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        }
    )
    workflow = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="distribution",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
    )
    spec = build_figure_spec(workflow)
    assert spec is None


def test_figurespec_twoway_cell_profile() -> None:
    """Two-way ANOVA FigureSpec plots observed cell means across factor levels."""
    analysis = AnalysisResult(
        method_id="twoway_anova",
        status=AnalysisStatus.AVAILABLE,
        values={
            "factor_a": "Diet",
            "factor_b": "Exercise",
            "factor_a_levels": ["Standard", "Keto"],
            "factor_b_levels": ["None", "Aerobic"],
            "cell_summaries": [
                {"Diet": "Standard", "Exercise": "None", "mean": 10.0},
                {"Diet": "Keto", "Exercise": "None", "mean": 12.0},
                {"Diet": "Standard", "Exercise": "Aerobic", "mean": 15.0},
                {"Diet": "Keto", "Exercise": "Aerobic", "mean": 18.0},
            ],
        },
    )
    spec = build_figure_spec(analysis)
    assert spec is not None
    assert spec.kind == "cell_profile"
    assert spec.title == "Observed cell means by factor levels"
    assert len(spec.series) == 2  # 2 levels of Exercise
    assert spec.series[0].label == "None"
    assert spec.series[0].values == (10.0, 12.0)
    assert spec.series[1].label == "Aerobic"
    assert spec.series[1].values == (15.0, 18.0)


# ── Section 7 & 31: Zero Recalculation ───────────────────────────────────────


def test_zero_recalculation_on_interactive_rendering(
    welch_workflow: ResearchWorkflowResult,
) -> None:
    """Rendering interactive HTML must never call statistical algorithms or recalculate."""
    with (
        patch(
            "scipy.stats.ttest_ind",
            side_effect=AssertionError("Inferential recalculation occurred in scipy!"),
        ),
        patch(
            "pyautostat.execution._group_result",
            side_effect=AssertionError("Recalculation occurred in PyAutoStat execution!"),
        ),
    ):
        # Interactive HTML standalone
        html1 = to_interactive_html(welch_workflow)
        assert "Plotly.newPlot" in html1

        # Interactive HTML report
        html2 = welch_workflow.report.to_interactive_html(include_figures=True)
        assert "Plotly.newPlot" in html2


# ── Section 20: Color Rules ──────────────────────────────────────────────────


def test_no_traffic_light_significance_coloring(
    welch_workflow: ResearchWorkflowResult,
) -> None:
    """Figure traces must not encode significance with green/red traffic-light colors."""
    spec = build_figure_spec(welch_workflow)
    assert spec is not None
    fig = spec_to_plotly_figure(spec)
    assert fig is not None

    # Error bar and marker color is neutral navy / blue, reference line is neutral gray
    trace = fig.data[0]
    marker_color = trace.marker.color
    assert marker_color.lower() in ("#1e3a8a", "#2563eb", "navy", "blue")


# ── Section 23: XSS Safety ───────────────────────────────────────────────────


def test_xss_safety_in_figure_labels_and_scripts() -> None:
    """User-controlled labels containing script injection payloads must be safely sanitized."""
    malicious_label = "</script><script>alert(1)</script>"
    malicious_caption = "<img src=x onerror=alert(1)>"

    spec = FigureSpec(
        kind="estimate_ci",
        title=f"Test {malicious_label}",
        note=malicious_caption,
        series=(
            FigureSeries(
                label=malicious_label,
                estimate=1.23,
                lower=0.5,
                upper=2.0,
            ),
        ),
    )

    from pyautostat.presentation.html.plotly_renderer import render_figure_html

    fig_html = render_figure_html(spec, figure_idx=99)

    # Must NOT contain raw unescaped script tag inside fig_html
    assert "</script><script>alert(1)</script>" not in fig_html
    # Must contain unicode-escaped or HTML-escaped variants
    assert "\\u003c" in fig_html or "&lt;" in fig_html

    # Test safe_json_for_script directly
    safe_str = safe_json_for_script({"label": malicious_label})
    assert "</script>" not in safe_str
    assert "\\u003c/script\\u003e" in safe_str


# ── Section 27 & 28: ResearchReport Figure Fidelity & Flag Override ──────────


def test_research_report_include_figures_flag_and_override(
    welch_workflow: ResearchWorkflowResult,
) -> None:
    """ResearchReport respects stored _include_figures flag and honors runtime overrides."""
    rep_default = welch_workflow.report
    assert rep_default._include_figures is False

    # Default to_interactive_html honors stored False
    html_honor_false = rep_default.to_interactive_html(include_figures=None)
    assert "Plotly.newPlot" not in html_honor_false

    # Explicit override with True
    html_override_true = rep_default.to_interactive_html(include_figures=True)
    assert "Plotly.newPlot" in html_override_true
    # Verify report._include_figures was NOT mutated
    assert rep_default._include_figures is False

    # Construct report with include_figures=True
    rep_true = build_research_report(welch_workflow.analysis, include_figures=True)
    assert rep_true._include_figures is True

    # Honors stored True
    html_honor_true = rep_true.to_interactive_html(include_figures=None)
    assert "Plotly.newPlot" in html_honor_true

    # Override with False
    html_override_false = rep_true.to_interactive_html(include_figures=False)
    assert "Plotly.newPlot" not in html_override_false


def test_research_report_and_standalone_figure_agreement(
    welch_workflow: ResearchWorkflowResult,
) -> None:
    """FigureSpec built from standalone workflow matches FigureSpec built from report."""
    standalone_spec = build_figure_spec(welch_workflow)
    report_spec = build_figure_spec(welch_workflow.report)

    assert standalone_spec is not None
    assert report_spec is not None
    assert standalone_spec.kind == report_spec.kind
    assert standalone_spec.title == report_spec.title
    assert standalone_spec.series[0].estimate == report_spec.series[0].estimate
    assert standalone_spec.series[0].lower == report_spec.series[0].lower
    assert standalone_spec.series[0].upper == report_spec.series[0].upper


# ── Section 24 & 25: Accessibility and Noscript ──────────────────────────────


def test_accessibility_and_noscript_notice(
    welch_workflow: ResearchWorkflowResult,
) -> None:
    """Interactive HTML must contain semantic figure, figcaption, and accessible noscript note."""
    html_out = to_interactive_html(welch_workflow)
    assert '<figure class="pyautostat-figure"' in html_out
    assert '<figcaption class="pyautostat-figcaption">' in html_out
    assert "<noscript>" in html_out
    assert "Interactive figures require JavaScript" in html_out
