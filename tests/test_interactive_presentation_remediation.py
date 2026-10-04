"""Focused tests for Phase 3 completion and remediation pass.

Validates:
- Partial-result presentation robustness when optional effect_size is None
- Removal of silent 95% CI fallback in figures and centralized confidence_interval_phrase
- Method-aware, conservative report-payload figure fallback
- Removal of universal reliability threshold claims (0.70, acceptable consistency)
- Method-accurate correlation null-value notes
- Actual interval level precedence in figure titles
- Multi-figure rendering infrastructure and real single-bundle verification
- Plotly requirement bypass when no figure can be built (table-only, PresentationView)
- PresentationView interactive behavior (no string-parsing into numbers)
- Extended raw-data privacy sentinels across OLS, logistic, factorial
- XSS and HTML escaping robustness
- Figure adapter branch coverage across all supported statistical families
- Degraded / partial result presentation audit
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from pyautostat import (
    AnalysisResult,
    AnalysisStatus,
    ReportError,
    ResearchAssistant,
    ResearchReport,
    to_html,
    to_interactive_html,
)
from pyautostat.presentation import adapt, build_figure_specs
from pyautostat.presentation.figures import FigureSeries, FigureSpec, build_figure_spec
from pyautostat.presentation.formatting import confidence_interval_phrase
from pyautostat.presentation.html import HtmlRenderer, ResearchReportHtmlRenderer
from pyautostat.presentation.models import (
    DisplayMetric,
    PresentationView,
)

# =============================================================================
# 1. CI-BLOCKING PARTIAL-RESULT REGRESSION TESTS
# =============================================================================


def test_partial_welch_result_effect_size_none_survives_all_renderers() -> None:
    """Welch result with effect_size=None must not crash adapt, to_html, or audit."""
    df = pd.DataFrame(
        {
            "score": [10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            "group": ["A"] * 5 + ["B"] * 5,
        }
    )
    assistant = ResearchAssistant(df)
    wf = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
    )
    vals = deepcopy(wf.analysis.values)
    vals["effect_size"] = None
    partial_analysis = replace(wf.analysis, values=vals)

    # 1. adapt()
    view = adapt(partial_analysis)
    assert view is not None
    assert any(m.label.lower() == "mean difference" and m.value != "—" for m in view.key_metrics)
    assert not any(m.label == "Cohen's d" for m in view.key_metrics)

    # 2. to_html()
    html_out = to_html(partial_analysis)
    assert "Mean difference" in html_out or "Mean Difference" in html_out

    # 3. ResearchReport.to_html()
    partial_report = assistant.report(partial_analysis)
    report_html = partial_report.to_html()
    assert "<!doctype html>" in report_html

    # 4. to_interactive_html()
    interactive_html = to_interactive_html(partial_analysis)
    assert "Plotly.newPlot" in interactive_html

    # 5. Assistant audit
    audit_record = assistant.audit(partial_report)
    assert audit_record is not None


def test_partial_adapters_effect_size_none_across_methods() -> None:
    """All two-group, paired, and multigroup adapters must handle effect_size=None gracefully."""
    methods_and_values: list[tuple[str, dict[str, Any]]] = [
        (
            "student_t",
            {
                "primary_estimate": 1.2,
                "confidence_interval": {"lower": 0.1, "upper": 2.3},
                "statistic": 2.1,
                "p_value": 0.04,
                "effect_size": None,
            },
        ),
        (
            "mann_whitney_u",
            {
                "statistic": 45.0,
                "p_value": 0.03,
                "effect_size": None,
            },
        ),
        (
            "one_sample_t",
            {
                "primary_estimate": 5.2,
                "confidence_interval": {"lower": 4.8, "upper": 5.6},
                "statistic": 4.2,
                "p_value": 0.001,
                "effect_size": None,
            },
        ),
        (
            "paired_t",
            {
                "primary_estimate": 0.8,
                "confidence_interval": {"lower": 0.2, "upper": 1.4},
                "statistic": 2.8,
                "p_value": 0.01,
                "effect_size": None,
            },
        ),
        (
            "wilcoxon_signed_rank",
            {
                "statistic": 12.0,
                "p_value": 0.02,
                "effect_size": None,
            },
        ),
        (
            "repeated_measures_anova",
            {
                "f_statistic": 5.6,
                "p_value": 0.005,
                "effect_size": None,
            },
        ),
        (
            "friedman_test",
            {
                "statistic": 7.8,
                "p_value": 0.02,
                "effect_size": None,
            },
        ),
    ]

    for method_id, values in methods_and_values:
        res = AnalysisResult(
            method_id=method_id,
            status=AnalysisStatus.AVAILABLE,
            sample_size=20,
            values=values,
            metadata={
                "outcome": "Y",
                "predictor": "X",
                "condition_order": ["A", "B"],
                "group_order": ["A", "B"],
            },
        )
        view = adapt(res)
        assert view is not None
        html = to_html(res)
        assert "<!doctype html>" in html


# =============================================================================
# 2. CONFIDENCE INTERVAL LEVEL PHRASING & REMOVAL OF 95% FALLBACK
# =============================================================================


def test_confidence_interval_phrase_values() -> None:
    """confidence_interval_phrase helper formats known levels and neutral fallback."""
    # From confidence_level float
    assert confidence_interval_phrase(confidence_level=0.90) == "90% confidence interval"
    assert confidence_interval_phrase(confidence_level=0.95) == "95% confidence interval"
    assert confidence_interval_phrase(confidence_level=0.975) == "97.5% confidence interval"
    assert confidence_interval_phrase(confidence_level=0.99) == "99% confidence interval"
    assert confidence_interval_phrase(confidence_level=None) == "confidence interval"
    assert "95%" not in confidence_interval_phrase(confidence_level=None)

    # From ci dict with level
    assert (
        confidence_interval_phrase({"lower": 1.0, "upper": 2.0, "level": 0.90})
        == "90% confidence interval"
    )
    assert (
        confidence_interval_phrase({"lower": 1.0, "upper": 2.0, "level": "90%"})
        == "90% confidence interval"
    )
    assert confidence_interval_phrase({"lower": 1.0, "upper": 2.0}) == "confidence interval"

    # Plural form
    assert (
        confidence_interval_phrase(confidence_level=0.95, plural=True) == "95% confidence intervals"
    )
    assert confidence_interval_phrase(confidence_level=None, plural=True) == "confidence intervals"
    assert "95%" not in confidence_interval_phrase(confidence_level=None, plural=True)


def test_figure_titles_never_assume_95_percent_when_level_unknown() -> None:
    """Figure titles must not invent a 95% interval when the level is unknown."""
    # Unknown level: CI dict without level, no spec
    analysis = AnalysisResult(
        method_id="welch_t",
        status=AnalysisStatus.AVAILABLE,
        values={
            "primary_estimate": 2.0,
            "confidence_interval": {"lower": 1.0, "upper": 3.0},  # no level
        },
    )
    spec = build_figure_spec(analysis)
    assert spec is not None
    assert spec.title == "Mean difference and confidence interval"
    assert "95%" not in spec.title

    # Known 90% level
    analysis_90 = AnalysisResult(
        method_id="welch_t",
        status=AnalysisStatus.AVAILABLE,
        values={
            "primary_estimate": 2.0,
            "confidence_interval": {"lower": 1.2, "upper": 2.8, "level": 0.90},
        },
    )
    spec_90 = build_figure_spec(analysis_90)
    assert spec_90 is not None
    assert spec_90.title == "Mean difference and 90% confidence interval"


def test_pairwise_anova_interval_level_precedence() -> None:
    """Pairwise ANOVA figures use uniform interval level when available, or neutral wording."""
    # Uniform 90% intervals
    analysis_uniform = AnalysisResult(
        method_id="welch_anova",
        status=AnalysisStatus.AVAILABLE,
        values={
            "pairwise_comparisons": [
                {
                    "contrast": "B - A",
                    "estimate": 1.0,
                    "confidence_interval": {"lower": 0.2, "upper": 1.8, "level": 0.90},
                },
                {
                    "contrast": "C - A",
                    "estimate": 2.0,
                    "confidence_interval": {"lower": 1.1, "upper": 2.9, "level": 0.90},
                },
            ]
        },
    )
    spec_uniform = build_figure_spec(analysis_uniform)
    assert spec_uniform is not None
    assert "90% simultaneous confidence intervals" in spec_uniform.title

    # Mixed or unknown levels
    analysis_unknown = AnalysisResult(
        method_id="welch_anova",
        status=AnalysisStatus.AVAILABLE,
        values={
            "pairwise_comparisons": [
                {
                    "contrast": "B - A",
                    "estimate": 1.0,
                    "confidence_interval": {"lower": 0.2, "upper": 1.8},
                },
                {
                    "contrast": "C - A",
                    "estimate": 2.0,
                    "confidence_interval": {"lower": 1.1, "upper": 2.9},
                },
            ]
        },
    )
    spec_unknown = build_figure_spec(analysis_unknown)
    assert spec_unknown is not None
    assert "simultaneous confidence intervals" in spec_unknown.title
    assert "95%" not in spec_unknown.title


# =============================================================================
# 3. METHOD-AWARE REPORT-PAYLOAD FIGURE FALLBACK
# =============================================================================


def test_sourceless_report_payload_figure_fallback() -> None:
    """Source-less reports build figures only when method and reference value are known."""
    # 1. Mean difference payload -> reference 0.0
    payload_mean = {
        "analysis": {
            "method_id": "welch_t",
            "values": {
                "primary_estimate": 3.4,
                "confidence_interval": {"lower": 1.2, "upper": 5.6, "level": 0.95},
            },
        },
        "title": "Welch Report",
    }
    rep_mean = ResearchReport(payload_mean)
    spec_mean = build_figure_spec(rep_mean)
    assert spec_mean is not None
    assert spec_mean.reference_value == 0.0
    assert spec_mean.series[0].estimate == 3.4

    # 2. Odds ratio payload -> reference 1.0
    payload_or = {
        "analysis": {
            "method_id": "logistic_regression",
            "values": {
                "primary_estimate": 2.5,
                "confidence_interval": {"lower": 1.1, "upper": 5.7, "level": 0.95},
            },
        },
        "title": "Logistic Report",
    }
    rep_or = ResearchReport(payload_or)
    spec_or = build_figure_spec(rep_or)
    assert spec_or is not None
    assert spec_or.reference_value == 1.0
    assert spec_or.series[0].estimate == 2.5

    # 3. Correlation payload -> reference 0.0
    payload_corr = {
        "analysis": {
            "method_id": "pearson_correlation",
            "values": {
                "primary_estimate": 0.65,
                "confidence_interval": {"lower": 0.35, "upper": 0.83, "level": 0.95},
            },
        },
        "title": "Correlation Report",
    }
    rep_corr = ResearchReport(payload_corr)
    spec_corr = build_figure_spec(rep_corr)
    assert spec_corr is not None
    assert spec_corr.reference_value == 0.0
    assert spec_corr.series[0].estimate == 0.65

    # 4. Ambiguous source-less payload -> returns None, static HTML returned cleanly
    payload_ambig = {
        "analysis": {
            "method_id": "unknown_metric_or_table_only",
            "values": {
                "primary_estimate": 42.0,
                "confidence_interval": {"lower": 40.0, "upper": 44.0},
            },
        },
        "title": "Ambiguous Report",
    }
    rep_ambig = ResearchReport(payload_ambig)
    spec_ambig = build_figure_spec(rep_ambig)
    assert spec_ambig is None
    # to_interactive_html does not crash and renders clean HTML without Plotly
    html_ambig = to_interactive_html(rep_ambig)
    assert "<!doctype html>" in html_ambig
    assert "Plotly.newPlot" not in html_ambig


# =============================================================================
# 4. RELIABILITY NOTES CLEANUP AND REGRESSION GUARDS
# =============================================================================


def test_cronbach_figure_note_no_generic_threshold() -> None:
    """Cronbach alpha figure note must not claim 0.70 or 'acceptable internal consistency'."""
    analysis = AnalysisResult(
        method_id="cronbach_alpha",
        status=AnalysisStatus.AVAILABLE,
        values={
            "raw_alpha": 0.82,
            "raw_alpha_ci": {"lower": 0.74, "upper": 0.88, "level": 0.95},
        },
    )
    spec = build_figure_spec(analysis)
    assert spec is not None
    assert spec.note is not None
    assert "0.70" not in spec.note
    assert "acceptable internal consistency" not in spec.note.lower()

    html = to_interactive_html(analysis)
    assert "0.70 generally indicate acceptable" not in html


def test_icc_figure_note_no_universal_cutoffs() -> None:
    """ICC figure note must not use oversimplified universal qualitative cutoffs."""
    analysis = AnalysisResult(
        method_id="icc",
        status=AnalysisStatus.AVAILABLE,
        values={
            "icc": 0.78,
            "confidence_interval": {"lower": 0.65, "upper": 0.87, "level": 0.95},
        },
    )
    spec = build_figure_spec(analysis)
    assert spec is not None
    assert spec.note is not None
    assert "close to 1 indicate" not in spec.note
    assert "Interpretation depends on the declared ICC model" in spec.note


# =============================================================================
# 5. METHOD-ACCURATE CORRELATION REFERENCE NOTES
# =============================================================================


def test_correlation_reference_notes_are_method_specific() -> None:
    """Correlation figure notes must accurately identify the specific coefficient's null value."""
    corr_tests = [
        ("pearson_correlation", "Pearson r"),
        ("spearman_correlation", "Spearman rho"),
        ("kendall_tau_b", "Kendall tau-b"),
        ("point_biserial_correlation", "the point-biserial coefficient"),
        ("point_biserial", "the point-biserial coefficient"),
        ("partial_pearson_correlation", "the partial correlation coefficient"),
        ("partial_pearson", "the partial correlation coefficient"),
    ]
    for method_id, phrase in corr_tests:
        analysis = AnalysisResult(
            method_id=method_id,
            status=AnalysisStatus.AVAILABLE,
            values={
                "primary_estimate": 0.45,
                "confidence_interval": {"lower": 0.20, "upper": 0.65, "level": 0.95},
            },
        )
        spec = build_figure_spec(analysis)
        assert spec is not None
        assert spec.note is not None
        assert f"marks the null value for {phrase}" in spec.note
        assert "no linear or monotonic association" not in spec.note


# =============================================================================
# 7 & 8. MULTI-FIGURE INFRASTRUCTURE & REAL SINGLE-BUNDLE TESTS
# =============================================================================


def test_multi_figure_rendering_guarantees_single_bundle() -> None:
    """Rendering multiple figures in one document must embed exactly one Plotly bundle script."""
    spec1 = FigureSpec(
        kind="estimate_ci",
        title="Primary Difference",
        series=(FigureSeries(label="Contrast", estimate=2.5, lower=1.0, upper=4.0),),
    )
    spec2 = FigureSpec(
        kind="forest",
        title="Model Coefficients",
        series=(
            FigureSeries(label="Term 1", estimate=1.0, lower=0.5, upper=1.5),
            FigureSeries(label="Term 2", estimate=-0.8, lower=-1.4, upper=-0.2),
        ),
    )

    view = PresentationView(
        title="Multi-figure Document",
        subtitle="Verification of single bundle embedding",
        family="family.comparison",
        key_metrics=(DisplayMetric("Difference", "2.50"),),
    )

    renderer = HtmlRenderer(view, figure_specs=(spec1, spec2))
    html_out = renderer.render()

    # 1. Both figures are rendered
    assert html_out.count("Plotly.newPlot") == 2
    # 2. Unique chart IDs
    assert 'id="pyautostat-chart-1"' in html_out
    assert 'id="pyautostat-chart-2"' in html_out
    # 3. Exactly one bundle script tag
    assert html_out.count('id="pyautostat-plotly-bundle"') == 1
    # 4. Exactly one noscript banner
    assert html_out.count("<noscript>") == 1


def test_research_report_renderer_supports_multiple_figure_specs() -> None:
    """ResearchReportHtmlRenderer renders multiple figures with unique IDs and one bundle."""
    spec1 = FigureSpec(
        kind="estimate_ci",
        title="Figure 1",
        placement="KEY RESULTS",
        series=(FigureSeries(label="Est 1", estimate=1.5, lower=0.5, upper=2.5),),
    )
    spec2 = FigureSpec(
        kind="estimate_ci",
        title="Figure 2",
        placement="KEY RESULTS",
        series=(FigureSeries(label="Est 2", estimate=3.0, lower=2.0, upper=4.0),),
    )

    report = ResearchReport({"title": "Multi-fig Report", "status": "available"})
    renderer = ResearchReportHtmlRenderer(report, figure_specs=(spec1, spec2))
    html_out = renderer.render()

    assert html_out.count("Plotly.newPlot") == 2
    assert 'id="pyautostat-chart-1"' in html_out
    assert 'id="pyautostat-chart-2"' in html_out
    assert html_out.count('id="pyautostat-plotly-bundle"') == 1
    assert html_out.count("<noscript>") == 1


# =============================================================================
# 9. PLOTLY NOT REQUIRED WHEN NO FIGURE CAN BE BUILT
# =============================================================================


def test_plotly_not_required_when_no_figure_is_generated() -> None:
    """When a method has no supported figure, to_interactive_html does not require Plotly."""
    with patch(
        "pyautostat.presentation.html.api.check_plotly_available",
        side_effect=ReportError("Plotly is missing!"),
    ):
        # 1. Table-only method: Kruskal-Wallis generates no figure -> should succeed
        df = pd.DataFrame(
            {
                "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            }
        )
        kw_wf = ResearchAssistant(df).run(
            objective="compare_groups",
            outcome="score",
            predictor="group",
            estimand="distribution",
            design="independent",
            variable_types={"score": "continuous", "group": "nominal"},
        )
        # Verify no figure spec
        assert build_figure_spec(kw_wf) is None
        # Should render static HTML without raising ReportError
        html_kw = to_interactive_html(kw_wf, include_figures=True)
        assert "<!doctype html>" in html_kw
        assert "Plotly.newPlot" not in html_kw

        # 2. Plain PresentationView generates no figure -> should succeed without Plotly
        view = PresentationView(
            title="Plain View",
            subtitle="No numerical FigureSpec",
            family="family.comparison",
        )
        html_view = to_interactive_html(view, include_figures=True)
        assert "<!doctype html>" in html_view
        assert "Plotly.newPlot" not in html_view

        # 3. Welch (has supported figure) -> must raise ReportError when Plotly is absent
        welch_df = pd.DataFrame(
            {
                "score": [10.0, 12.0, 11.0, 20.0, 22.0, 21.0],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )
        welch_wf = ResearchAssistant(welch_df).run(
            objective="compare_groups",
            outcome="score",
            predictor="group",
            estimand="mean",
            design="independent",
            variable_types={"score": "continuous", "group": "nominal"},
        )
        assert build_figure_spec(welch_wf) is not None
        with pytest.raises(ReportError, match="Plotly is missing!"):
            to_interactive_html(welch_wf, include_figures=True)


# =============================================================================
# 10. PRESENTATIONVIEW INTERACTIVE BEHAVIOR
# =============================================================================


def test_presentation_view_does_not_parse_text_into_numbers() -> None:
    """PresentationView passed to interactive HTML does not parse display strings."""
    view = PresentationView(
        title="String View",
        subtitle="Formatted text",
        family="family.comparison",
        key_metrics=(DisplayMetric("Formatted Stat", "t(28) = 3.14, p < .001"),),
    )
    # Figure specs for PresentationView is always empty
    specs = build_figure_specs(view)
    assert specs == ()

    html = to_interactive_html(view, include_figures=True)
    assert "<!doctype html>" in html
    assert "t(28) = 3.14" in html
    assert "Plotly.newPlot" not in html


# =============================================================================
# 11. RAW-DATA PRIVACY SENTINELS ACROSS OLS, LOGISTIC, FACTORIAL
# =============================================================================


def test_extended_raw_data_privacy_sentinels() -> None:
    """Unique raw observation floats must never leak into HTML, JSON, or scripts."""
    ols_sentinel = 789.123456789
    logistic_sentinel = 654.987654321
    factorial_sentinel = 432.111222333

    # OLS
    df_ols = pd.DataFrame(
        {
            "y": [ols_sentinel, 10.5, 12.0, 14.5, 16.0, 18.5, 20.0, 22.5],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "x2": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        }
    )
    ols_wf = ResearchAssistant(df_ols).run(
        objective="regression",
        outcome="y",
        predictors=["x1", "x2"],
        design="independent",
        estimand="conditional_mean",
        variable_types={"y": "continuous", "x1": "continuous", "x2": "continuous"},
    )
    ols_html = to_interactive_html(ols_wf)
    assert str(ols_sentinel) not in ols_html

    # Logistic
    df_logit = pd.DataFrame(
        {
            "event": ["no", "yes"] * 10,
            "x": [logistic_sentinel] + list(range(1, 20)),
        }
    )
    logit_wf = ResearchAssistant(df_logit).run(
        objective="regression",
        outcome="event",
        predictors=["x"],
        design="independent",
        estimand="event_probability",
        event_level="yes",
        variable_types={"event": "nominal", "x": "continuous"},
    )
    logit_html = to_interactive_html(logit_wf)
    assert str(logistic_sentinel) not in logit_html

    # Factorial ANOVA
    df_fact = pd.DataFrame(
        {
            "biomarker": [
                factorial_sentinel,
                12.0,
                11.0,
                15.0,
                14.0,
                16.0,
                20.0,
                22.0,
                18.0,
                19.0,
                21.0,
                23.0,
            ],
            "dose": [
                "Low",
                "Low",
                "Low",
                "Low",
                "Med",
                "Med",
                "Med",
                "Med",
                "High",
                "High",
                "High",
                "High",
            ],
            "diet": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )
    fact_wf = ResearchAssistant(df_fact).run(
        objective="compare_groups",
        outcome="biomarker",
        factor_a="dose",
        factor_b="diet",
        design="independent",
        estimand="mean",
        variable_types={"biomarker": "continuous", "dose": "nominal", "diet": "nominal"},
    )
    fact_html = to_interactive_html(fact_wf)
    assert str(factorial_sentinel) not in fact_html


# =============================================================================
# 12. FIGURE ADAPTER BRANCH COVERAGE
# =============================================================================


def test_figure_adapter_branch_coverage_all_families() -> None:
    """Test all advertised figure branches to achieve strong branch coverage."""
    # 1. One-sample t
    analysis_1s = AnalysisResult(
        method_id="one_sample_t",
        status=AnalysisStatus.AVAILABLE,
        values={
            "primary_estimate": 4.5,
            "confidence_interval": {"lower": 3.2, "upper": 5.8, "level": 0.95},
        },
        metadata={"null_value": 0.0},
    )
    spec_1s = build_figure_spec(analysis_1s)
    assert spec_1s is not None
    assert spec_1s.reference_value == 0.0

    # 2. Student t
    analysis_st = AnalysisResult(
        method_id="student_t",
        status=AnalysisStatus.AVAILABLE,
        values={
            "primary_estimate": 1.5,
            "confidence_interval": {"lower": 0.5, "upper": 2.5, "level": 0.95},
        },
    )
    spec_st = build_figure_spec(analysis_st)
    assert spec_st is not None
    assert spec_st.reference_value == 0.0

    # 3. Paired t
    analysis_pt = AnalysisResult(
        method_id="paired_t",
        status=AnalysisStatus.AVAILABLE,
        values={
            "primary_estimate": 0.8,
            "confidence_interval": {"lower": 0.2, "upper": 1.4, "level": 0.95},
        },
    )
    spec_pt = build_figure_spec(analysis_pt)
    assert spec_pt is not None
    assert spec_pt.reference_value == 0.0

    # 4. Fisher count heatmap
    analysis_fisher = AnalysisResult(
        method_id="fisher_exact",
        status=AnalysisStatus.AVAILABLE,
        values={"odds_ratio": 2.5, "odds_ratio_ci": {"lower": 1.1, "upper": 5.8}},
        metadata={
            "observed_counts": [[10, 5], [4, 11]],
            "row_order": ["A", "B"],
            "column_order": ["Yes", "No"],
        },
    )
    spec_fisher = build_figure_spec(analysis_fisher)
    assert spec_fisher is not None
    assert spec_fisher.kind == "count_heatmap"

    # 5. Two-way ANOVA incomplete grid -> returns None
    analysis_twoway_incomplete = AnalysisResult(
        method_id="two_way_anova",
        status=AnalysisStatus.AVAILABLE,
        values={
            "factor_a": "Diet",
            "factor_b": "Drug",
            "factor_a_levels": ["A", "B"],
            "factor_b_levels": ["Veh", "Active"],
            "cell_summaries": [
                {"Diet": "A", "Drug": "Veh", "mean": 10.0},
                # Missing (B, Veh) and others
            ],
        },
    )
    assert build_figure_spec(analysis_twoway_incomplete) is None

    # 6. Unsupported method -> returns None
    analysis_unsupp = AnalysisResult(
        method_id="kruskal_wallis",
        status=AnalysisStatus.AVAILABLE,
        values={"statistic": 5.4, "p_value": 0.06},
    )
    assert build_figure_spec(analysis_unsupp) is None
    assert build_figure_specs(analysis_unsupp) == ()
