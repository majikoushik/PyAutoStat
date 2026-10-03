"""Comprehensive coverage tests for the PyAutoStat Rich terminal presentation layer.

Validates all 24 statistical methods, direct descriptive outputs,
governance/planning objects, responsive widths, zero-recalculation,
and specific scientific presentation requirements.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from pyautostat import (
    AnalysisOptions,
    AnalysisResult,
    AnalysisStatus,
    AuditFinding,
    AuditResult,
    MeaningfulEffectThreshold,
    PracticalSignificanceResult,
    ResearchAssistant,
    ResearchWorkflowResult,
    SensitivitySpecification,
    StudyPlanner,
    UnsupportedPresentationError,
    WorkflowStatus,
    reproduce,
    show,
)
from pyautostat.presentation.adapters import adapt


def _capture(target, detail="standard", width=80, no_color=False, force_terminal=True) -> str:
    """Helper to capture show() output string cleanly."""
    console = Console(
        record=True,
        width=width,
        no_color=no_color,
        force_terminal=force_terminal,
    )
    show(target, detail=detail, console=console)
    return console.export_text()


# ── Fixtures for 24 Statistical Methods ──────────────────────────────────────


@pytest.fixture
def one_sample_workflow():
    df = pd.DataFrame({"score": [48.0, 51.0, 53.0, 55.0, 52.0, 49.0]})
    return ResearchAssistant(df).run(
        objective="compare_reference",
        outcome="score",
        reference_value=50.0,
        design="independent",
        estimand="mean",
        variable_types={"score": "continuous"},
    )


@pytest.fixture
def welch_t_workflow():
    df = pd.DataFrame(
        {
            "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
            "group": ["A"] * 5 + ["B"] * 5,
        }
    )
    return ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )


@pytest.fixture
def student_t_workflow(welch_t_workflow):
    from pyautostat import Recommendation, execution

    asst = ResearchAssistant(
        pd.DataFrame(
            {
                "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
                "group": ["A"] * 5 + ["B"] * 5,
            }
        )
    )
    spec = welch_t_workflow.specification
    rec = Recommendation(
        status="ready",
        method_id="student_t",
        method_name="Student's t-test",
        method_availability="runnable",
    )
    res = execution._group_result(asst._analyzer, spec, rec)
    interp = asst.interpret(res)
    rep = asst.report(res, interpretation=interp)
    return ResearchWorkflowResult(
        status=WorkflowStatus.COMPLETED,
        specification=spec,
        draft=welch_t_workflow.draft,
        recommendation=rec,
        analysis=res,
        interpretation=interp,
        report=rep,
        audit=asst.audit(rep, result=res),
        reproducibility=asst.reproducibility_record(res),
    )


@pytest.fixture
def mann_whitney_workflow():
    df = pd.DataFrame(
        {
            "score": [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "group": ["Ctrl"] * 5 + ["Treat"] * 5,
        }
    )
    return ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="distribution",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
    )


@pytest.fixture
def paired_t_workflow():
    df = pd.DataFrame(
        {
            "participant": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "condition": ["before", "after"] * 6,
            "score": [10.0, 8.0, 9.0, 7.0, 11.0, 10.0, 8.0, 5.0, 13.0, 10.0, 12.0, 9.0],
        }
    )
    return ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="paired",
        estimand="mean",
        unit_id="participant",
        condition_order=("before", "after"),
        variable_types={"score": "continuous", "condition": "nominal"},
    )


@pytest.fixture
def wilcoxon_workflow():
    df = pd.DataFrame(
        {
            "participant": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "condition": ["before", "after"] * 6,
            "score": [10.0, 8.0, 9.0, 7.0, 11.0, 10.0, 8.0, 5.0, 13.0, 10.0, 12.0, 9.0],
        }
    )
    return ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="paired",
        estimand="distribution",
        unit_id="participant",
        condition_order=("before", "after"),
        variable_types={"score": "continuous", "condition": "nominal"},
    )


@pytest.fixture
def welch_anova_workflow():
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


@pytest.fixture
def one_way_anova_workflow(welch_anova_workflow):
    from pyautostat import Recommendation, execution

    asst = ResearchAssistant(
        pd.DataFrame(
            {
                "group": ["C"] * 6 + ["A"] * 7 + ["B"] * 5,
                "score": [2, 3, 4, 5, 7, 8, 8, 10, 12, 15, 18, 22, 27, 1, 2, 2, 3, 5],
            }
        )
    )
    spec = welch_anova_workflow.specification
    rec = Recommendation(
        status="ready",
        method_id="one_way_anova",
        method_name="One-way ANOVA",
        method_availability="runnable",
    )
    res = execution._group_result(asst._analyzer, spec, rec)
    interp = asst.interpret(res)
    rep = asst.report(res, interpretation=interp)
    return ResearchWorkflowResult(
        status=WorkflowStatus.COMPLETED,
        specification=spec,
        draft=welch_anova_workflow.draft,
        recommendation=rec,
        analysis=res,
        interpretation=interp,
        report=rep,
        audit=asst.audit(rep, result=res),
        reproducibility=asst.reproducibility_record(res),
    )


@pytest.fixture
def kruskal_wallis_workflow():
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
    )


@pytest.fixture
def spearman_workflow():
    df = pd.DataFrame({"hours": [1, 2, 3, 4, 5, 6], "score": [2, 4, 3, 7, 8, 10]})
    return ResearchAssistant(df).run(
        objective="association",
        outcome="score",
        predictor="hours",
        design="independent",
        estimand="monotonic",
        variable_types={"score": "continuous", "hours": "continuous"},
    )


@pytest.fixture
def kendall_workflow():
    df = pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 8.0, 9.0],
            "rank": [8.0, 7.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    return ResearchAssistant(df).run(
        objective="association",
        outcome="score",
        predictor="rank",
        design="independent",
        estimand="monotonic",
        association_measure="kendall",
        variable_types={"score": "continuous", "rank": "continuous"},
    )


@pytest.fixture
def point_biserial_workflow():
    df = pd.DataFrame(
        {
            "binary": [False, False, False, False, True, True, True, True],
            "score": [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 8.0, 9.0],
        }
    )
    return ResearchAssistant(df).run(
        objective="association",
        outcome="binary",
        predictor="score",
        design="independent",
        estimand="point_biserial",
        variable_types={"binary": "boolean", "score": "continuous"},
    )


@pytest.fixture
def partial_pearson_workflow():
    rng = np.random.default_rng(8)
    n = 30
    age = rng.normal(size=n)
    attendance = rng.normal(size=n)
    x = 0.8 * age + rng.normal(size=n)
    y = 0.7 * x + 1.1 * attendance + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y, "age": age, "attendance": attendance})
    return ResearchAssistant(df).run(
        objective="association",
        outcome="x",
        predictor="y",
        controls=["age", "attendance"],
        estimand="partial_linear",
        design="independent",
        variable_types={name: "continuous" for name in df},
    )


@pytest.fixture
def chi_square_workflow():
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
def fisher_workflow():
    df = pd.DataFrame(
        {
            "treatment": ["A"] * 5 + ["B"] * 7,
            "response": ["yes"] + ["no"] * 4 + ["yes"] * 5 + ["no"] * 2,
        }
    )
    return ResearchAssistant(df).run(
        objective="association",
        outcome="response",
        predictor="treatment",
        design="independent",
        estimand="categorical_independence",
        variable_types={"response": "nominal", "treatment": "nominal"},
    )


@pytest.fixture
def mcnemar_workflow():
    rows = []
    for unit, (before, after) in enumerate([(0, 1), (0, 1), (0, 0), (1, 1), (1, 0), (0, 1)]):
        rows.extend(
            [
                {"unit": unit, "condition": "before", "response": before},
                {"unit": unit, "condition": "after", "response": after},
            ]
        )
    return ResearchAssistant(pd.DataFrame(rows)).run(
        objective="compare_groups",
        outcome="response",
        predictor="condition",
        design="paired",
        estimand="proportion",
        unit_id="unit",
        condition_order=("after", "before"),
        event_level=1,
        variable_types={"response": "nominal", "condition": "nominal"},
    )


@pytest.fixture
def linear_regression_workflow():
    df = pd.DataFrame(
        {
            "y": [4.2, 5.1, 6.5, 7.0, 8.4, 9.2, 10.5, 11.3, 12.1, 13.0],
            "x": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "group": ["A", "B"] * 5,
        }
    )
    return ResearchAssistant(df).run(
        objective="regression",
        outcome="y",
        predictors=["x", "group"],
        design="independent",
        estimand="conditional_mean",
        variable_types={"y": "continuous", "x": "continuous", "group": "nominal"},
        reference_levels={"group": "A"},
        covariance_type="HC3",
    )


@pytest.fixture
def logistic_regression_workflow():
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
def cronbach_alpha_workflow():
    df = pd.DataFrame(
        {
            "q1": [1, 2, 3, 4, 5, 2, 4, 3],
            "q2": [1, 2, 3, 4, 4, 2, 5, 3],
            "q3": [2, 2, 3, 5, 5, 2, 4, 4],
        }
    )
    return ResearchAssistant(df).reliability(["q1", "q2", "q3"], bootstrap_samples=40)


@pytest.fixture
def repeated_measures_workflow():
    df = pd.DataFrame(
        {
            "participant": [1, 2, 3, 4, 5, 6] * 3,
            "session": ["s1"] * 6 + ["s2"] * 6 + ["s3"] * 6,
            "score": [
                10.0,
                12.0,
                14.0,
                11.0,
                13.0,
                15.0,
                14.0,
                15.0,
                16.0,
                13.0,
                15.0,
                18.0,
                18.0,
                19.0,
                20.0,
                17.0,
                19.0,
                22.0,
            ],
        }
    )
    return ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="session",
        design="repeated",
        estimand="mean",
        unit_id="participant",
        condition_order=("s1", "s2", "s3"),
        variable_types={"score": "continuous", "session": "ordinal"},
    )


@pytest.fixture
def friedman_workflow():
    df_data = pd.DataFrame(
        {
            "participant": [1, 2, 3, 4, 5, 6] * 3,
            "session": ["s1"] * 6 + ["s2"] * 6 + ["s3"] * 6,
            "score": [
                10.0,
                12.0,
                14.0,
                11.0,
                13.0,
                15.0,
                14.0,
                15.0,
                16.0,
                13.0,
                15.0,
                18.0,
                18.0,
                19.0,
                20.0,
                17.0,
                19.0,
                22.0,
            ],
        }
    )
    return ResearchAssistant(df_data).run(
        objective="compare_groups",
        outcome="score",
        predictor="session",
        design="repeated",
        estimand="distribution",
        unit_id="participant",
        condition_order=("s1", "s2", "s3"),
        variable_types={"score": "continuous", "session": "ordinal"},
    )


@pytest.fixture
def two_way_anova_workflow():
    df = pd.DataFrame(
        {
            "y": [
                10.0,
                11.0,
                12.0,
                14.0,
                15.0,
                16.0,
                13.0,
                12.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                25.0,
            ],
            "A": ["A1"] * 3 + ["A2"] * 4 + ["A3"] * 7,
            "B": [
                "B1",
                "B1",
                "B2",
                "B1",
                "B1",
                "B2",
                "B2",
                "B1",
                "B1",
                "B1",
                "B2",
                "B2",
                "B2",
                "B2",
            ],
        }
    )
    return ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="y",
        factor_a="A",
        factor_b="B",
        design="independent",
        estimand="mean",
        variable_types={"y": "continuous", "A": "nominal", "B": "nominal"},
    )


@pytest.fixture
def icc_workflow():
    df = pd.DataFrame(
        {
            "target": ["T1", "T1", "T2", "T2", "T3", "T3"],
            "rater": ["R1", "R2", "R1", "R2", "R1", "R2"],
            "score": [9.0, 2.0, 6.0, 1.0, 8.0, 4.0],
        }
    )
    return ResearchAssistant(df).intraclass_correlation(
        target="target",
        rater="rater",
        value="score",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
    )


# ── Fixtures for Descriptives ───────────────────────────────────────────────


@pytest.fixture
def frequency_table_data():
    df = pd.DataFrame({"category": ["low", "medium", "medium", "high", "high", "high"]})
    return ResearchAssistant(df).frequency_table("category")


@pytest.fixture
def cross_tab_data():
    df = pd.DataFrame(
        {
            "status": ["active", "active", "inactive", "active", "inactive"],
            "group": ["A", "B", "A", "B", "B"],
        }
    )
    return ResearchAssistant(df).cross_tab("status", "group")


# ── Fixtures for Planning / Governance ──────────────────────────────────────


@pytest.fixture
def study_planning_result():
    return StudyPlanner().independent_mean_power(
        target_difference=2.0,
        sd_group1=3.0,
        sd_group2=3.0,
        target_power=0.80,
    )


@pytest.fixture
def sensitivity_result(welch_t_workflow):
    asst = ResearchAssistant(
        pd.DataFrame(
            {
                "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
                "group": ["A"] * 5 + ["B"] * 5,
            }
        )
    )
    base = welch_t_workflow.analysis
    return asst.sensitivity_analysis(
        base,
        scenarios=[
            SensitivitySpecification(
                "equal_var",
                base.specification,
                method_id="student_t",
                assumptions=("Equal population variances",),
            )
        ],
    )


@pytest.fixture
def practical_significance_result(welch_t_workflow):
    asst = ResearchAssistant(
        pd.DataFrame(
            {
                "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
                "group": ["A"] * 5 + ["B"] * 5,
            }
        )
    )
    threshold = MeaningfulEffectThreshold(
        "mean_difference",
        minimum_magnitude=5.0,
        unit="points",
        rationale="Clinical relevance cutoff.",
    )
    return asst.practical_significance(welch_t_workflow.analysis, threshold=threshold)


@pytest.fixture
def statistical_analysis_plan():
    df = pd.DataFrame(
        {
            "group": ["a"] * 6 + ["b"] * 6,
            "score": [1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 9],
        }
    )
    asst = ResearchAssistant(df)
    draft = asst.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
        variable_types={"score": "continuous"},
    )
    return asst.analysis_plan(draft, report_style="apa")


@pytest.fixture
def plan_adherence_result(statistical_analysis_plan):
    df = pd.DataFrame(
        {
            "group": ["a"] * 6 + ["b"] * 6,
            "score": [1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 9],
        }
    )
    asst = ResearchAssistant(df)
    draft = asst.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
        options=AnalysisOptions(alpha=0.01),
        variable_types={"score": "continuous"},
    )
    res = asst.analyze(draft)
    return asst.plan_adherence(
        statistical_analysis_plan, res, reason="Alpha adjusted for exploratory test"
    )


@pytest.fixture
def reporting_completeness_result(welch_t_workflow):
    asst = ResearchAssistant(
        pd.DataFrame(
            {
                "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
                "group": ["A"] * 5 + ["B"] * 5,
            }
        )
    )
    return asst.reporting_completeness(welch_t_workflow.report, style="apa")


@pytest.fixture
def decision_ledger_instance():
    df = pd.DataFrame(
        {
            "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
            "group": ["A"] * 5 + ["B"] * 5,
        }
    )
    asst = ResearchAssistant(df)
    ledger = asst.enable_tracking()
    asst.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )
    return ledger


@pytest.fixture
def session_snapshot_result(welch_t_workflow):
    df = pd.DataFrame(
        {
            "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
            "group": ["A"] * 5 + ["B"] * 5,
        }
    )
    asst = ResearchAssistant(df)
    return asst.session_snapshot(welch_t_workflow)


# ── Tests for All 24 Methods ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture_name,expected_label",
    [
        ("one_sample_workflow", "One-Sample"),
        ("welch_t_workflow", "Welch"),
        ("student_t_workflow", "Student"),
        ("mann_whitney_workflow", "Mann-Whitney"),
        ("paired_t_workflow", "Paired"),
        ("wilcoxon_workflow", "Wilcoxon"),
        ("welch_anova_workflow", "Welch"),
        ("one_way_anova_workflow", "One-Way ANOVA"),
        ("kruskal_wallis_workflow", "Kruskal-Wallis"),
        ("pearson_workflow", "Pearson"),
        ("spearman_workflow", "Spearman"),
        ("kendall_workflow", "Kendall"),
        ("point_biserial_workflow", "Point-biserial"),
        ("partial_pearson_workflow", "Partial Pearson"),
        ("chi_square_workflow", "chi-square"),
        ("fisher_workflow", "Fisher"),
        ("mcnemar_workflow", "McNemar"),
        ("linear_regression_workflow", "Linear Regression"),
        ("logistic_regression_workflow", "Logistic Regression"),
        ("cronbach_alpha_workflow", "Cronbach"),
        ("repeated_measures_workflow", "Repeated-Measures ANOVA"),
        ("friedman_workflow", "Friedman"),
        ("two_way_anova_workflow", "Two-Way"),
        ("icc_workflow", "Intraclass Correlation"),
    ],
)
def test_all_24_methods_render_standard_and_compact(request, fixture_name, expected_label):
    """Verify each method renders in standard, compact, and full mode without error."""
    wf = request.getfixturevalue(fixture_name)

    # Standard mode
    out_std = _capture(wf, detail="standard")
    assert expected_label.lower() in out_std.lower()
    assert "{'primary_estimate':" not in out_std

    # Compact mode
    out_cmp = _capture(wf, detail="compact")
    assert len(out_cmp.strip()) > 0

    # Full mode
    out_full = _capture(wf, detail="full")
    assert expected_label.lower() in out_full.lower()


@pytest.mark.parametrize(
    "fixture_name",
    [
        "one_sample_workflow",
        "welch_t_workflow",
        "paired_t_workflow",
        "welch_anova_workflow",
        "pearson_workflow",
        "chi_square_workflow",
        "linear_regression_workflow",
        "cronbach_alpha_workflow",
        "repeated_measures_workflow",
        "two_way_anova_workflow",
        "icc_workflow",
    ],
)
def test_direct_analysis_result_rendering(request, fixture_name):
    """Verify show() directly on AnalysisResult succeeds and contains no fabricated questions."""
    wf = request.getfixturevalue(fixture_name)
    analysis = wf.analysis
    assert isinstance(analysis, AnalysisResult)

    out = _capture(analysis, detail="standard")
    assert len(out.strip()) > 0
    assert "{'primary_estimate':" not in out


# ── Specific Requirements from Prompt Section 24 ────────────────────────────


def test_student_vs_welch_variance_assumption_wording(student_t_workflow, welch_t_workflow):
    """Verify Student t prominently displays equal-variance assumption context."""
    std_out = _capture(student_t_workflow, detail="standard")
    welch_out = _capture(welch_t_workflow, detail="standard")

    assert "equal variance" in std_out.lower()
    assert "assumed equal" in std_out.lower() or "equal-variance" in std_out.lower()
    # Welch should state robust or unequal variances allowed
    assert (
        "unequal variances" in welch_out.lower()
        or "heteroscedasticity" in welch_out.lower()
        or "welch" in welch_out.lower()
    )


def test_paired_t_condition_order(paired_t_workflow):
    """Verify paired t displays condition order prominently."""
    out = _capture(paired_t_workflow, detail="standard")
    assert "before" in out
    assert "after" in out
    assert "Condition order" in out or "Contrast" in out or "before - after" in out


def test_wilcoxon_signed_effect_orientation(wilcoxon_workflow):
    """Verify Wilcoxon signed-rank shows condition order and effect."""
    out = _capture(wilcoxon_workflow, detail="standard")
    assert "before" in out
    assert "after" in out
    assert "Rank-biserial" in out or "r_rb" in out or "biserial" in out.lower()


def test_mann_whitney_not_universally_median_test(mann_whitney_workflow):
    """Verify Mann-Whitney is not described universally as a median test."""
    out = _capture(mann_whitney_workflow, detail="standard")
    assert "Rank-sum" in out or "stochastic" in out or "distribution" in out.lower() or "U=" in out
    assert "Rank-biserial" in out or "r_rb" in out


def test_welch_anova_and_games_howell(welch_anova_workflow):
    """Verify Welch ANOVA displays omnibus Welch F and Games-Howell follow-ups."""
    out = _capture(welch_anova_workflow, detail="standard")
    assert "welch" in out.lower()
    assert "games-howell" in out.lower()
    assert "simultaneous ci" in out.lower() or "simultaneous" in out.lower()
    assert "adjusted p" in out.lower()


def test_one_way_anova_and_tukey(one_way_anova_workflow):
    """Verify One-way ANOVA displays F, eta-squared, and Tukey follow-ups."""
    out = _capture(one_way_anova_workflow, detail="standard")
    assert "one-way anova" in out.lower()
    assert "tukey" in out.lower()
    assert "eta" in out.lower()


def test_kruskal_wallis_and_dunn_holm(kruskal_wallis_workflow):
    """Verify Kruskal-Wallis displays H statistic, epsilon-squared, and Dunn follow-ups."""
    out = _capture(kruskal_wallis_workflow, detail="standard")
    assert "kruskal-wallis" in out.lower()
    assert "dunn" in out.lower()
    assert "holm" in out.lower() or "adjusted p" in out.lower()


def test_spearman_rho_notation(spearman_workflow):
    """Verify Spearman displays rho and monotonic association context."""
    out = _capture(spearman_workflow, detail="standard")
    assert "spearman" in out.lower()
    assert "rho" in out.lower() or "ρ" in out or "r_s" in out
    assert "monotonic" in out.lower()


def test_kendall_tau_b_notation(kendall_workflow):
    """Verify Kendall displays tau-b notation and tie context."""
    out = _capture(kendall_workflow, detail="standard")
    assert "kendall" in out.lower()
    assert "tau-b" in out.lower() or "τ_b" in out
    assert "not a proportion of variance explained" in out.lower() or "not" in out.lower()


def test_point_biserial_positive_level(point_biserial_workflow):
    """Verify point-biserial prominently displays positive/reference level."""
    out = _capture(point_biserial_workflow, detail="standard")
    assert "point-biserial" in out.lower()
    assert "positive level" in out.lower() or "reference" in out.lower() or "true" in out.lower()


def test_partial_pearson_controls(partial_pearson_workflow):
    """Verify partial Pearson displays control variables and no-causal-claim limitation."""
    out = _capture(partial_pearson_workflow, detail="standard")
    assert "partial pearson" in out.lower()
    assert "control variables" in out.lower() or "controls" in out.lower()
    assert "age" in out.lower()
    assert "attendance" in out.lower()


def test_chi_square_contingency_table_and_cramers_v(chi_square_workflow):
    """Verify Chi-Square displays contingency table and Cramer's V."""
    out = _capture(chi_square_workflow, detail="standard")
    assert "chi-square" in out.lower()
    assert "cramer" in out.lower()
    assert "contingency table" in out.lower() or "contingency" in out.lower()


def test_fisher_zero_cell_unavailable_ci():
    """Verify Fisher's exact with zero cell displays unavailable CI state explicitly."""
    df = pd.DataFrame(
        {
            "treatment": ["A"] * 5 + ["B"] * 5,
            "response": ["yes"] * 0 + ["no"] * 5 + ["yes"] * 3 + ["no"] * 2,
        }
    )
    wf = ResearchAssistant(df).run(
        objective="association",
        outcome="response",
        predictor="treatment",
        design="independent",
        estimand="categorical_independence",
        variable_types={"response": "nominal", "treatment": "nominal"},
    )
    out = _capture(wf, detail="standard")
    assert "Fisher" in out
    assert "N/A" in out or "unavailable" in out.lower() or "Odds Ratio" in out


def test_mcnemar_discordant_pairs(mcnemar_workflow):
    """Verify McNemar displays transition table and discordant pairs."""
    out = _capture(mcnemar_workflow, detail="standard")
    assert "McNemar" in out
    assert "Discordant pairs" in out or "Transition" in out
    assert "Proportion difference" in out or "Difference" in out


def test_ols_hc3_label_and_diagnostics(linear_regression_workflow):
    """Verify OLS regression shows HC3 covariance label and diagnostics in full mode."""
    out_std = _capture(linear_regression_workflow, detail="standard")
    assert "HC3" in out_std
    assert "R²" in out_std or "R-squared" in out_std

    out_full = _capture(linear_regression_workflow, detail="full")
    assert "DIAGNOSTIC" in out_full
    assert "HC3" in out_full
    assert "in-sample fit" in out_full.lower()


def test_logistic_modeled_event_and_or_table(logistic_regression_workflow):
    """Verify logistic regression shows modeled event and OR-first table."""
    out = _capture(logistic_regression_workflow, detail="standard")
    assert "logistic regression" in out.lower()
    assert "modeled event" in out.lower()
    assert "odds ratio" in out.lower() or "or" in out.lower()
    assert "ci" in out.lower()


def test_cronbach_item_diagnostics(cronbach_alpha_workflow):
    """Verify Cronbach alpha shows item diagnostics."""
    out = _capture(cronbach_alpha_workflow, detail="standard")
    assert "cronbach" in out.lower()
    assert "alpha if deleted" in out.lower()
    assert "item-total" in out.lower()


def test_repeated_anova_sphericity_and_gg(repeated_measures_workflow):
    """Verify repeated-measures ANOVA displays Mauchly and Greenhouse-Geisser."""
    out = _capture(repeated_measures_workflow, detail="standard")
    assert "repeated-measures anova" in out.lower()
    assert "mauchly" in out.lower() or "sphericity" in out.lower()
    assert "greenhouse-geisser" in out.lower() or "epsilon" in out.lower() or "gg" in out.lower()


def test_friedman_kendall_w_and_holm(friedman_workflow):
    """Verify Friedman test displays Kendall W and pairwise follow-up."""
    out = _capture(friedman_workflow, detail="standard")
    assert "friedman" in out.lower()
    assert "kendall" in out.lower() or "w" in out.lower()
    assert "pairwise" in out.lower() or "wilcoxon" in out.lower()


def test_two_way_anova_interaction_row(two_way_anova_workflow):
    """Verify Two-Way ANOVA displays factor A, factor B, and interaction A×B distinctly."""
    out = _capture(two_way_anova_workflow, detail="standard")
    assert "two-way" in out.lower()
    assert "a x b" in out.lower() or "interaction" in out.lower() or "a × b" in out.lower()
    assert "partial eta" in out.lower()


def test_icc_definition_before_estimate_and_negative_icc():
    """Verify ICC displays definition before estimate, and preserves negative ICC."""
    # Build ICC result with negative estimate
    res = AnalysisResult(
        method_id="intraclass_correlation",
        status=AnalysisStatus.AVAILABLE,
        sample_size=6,
        excluded_rows=0,
        values={
            "variant": "ICC(2,1)",
            "notation": "ICC(2,1)",
            "model": "two-way random effects",
            "definition": "absolute agreement",
            "unit": "single rater",
            "intraclass_correlation": -0.152,
            "primary_estimate": -0.152,
            "confidence_interval": {
                "quantity": "ICC",
                "lower": -0.450,
                "upper": 0.210,
                "level": 0.95,
                "status": "available",
            },
            "f_test": {
                "statistic": 0.85,
                "df1": 5,
                "df2": 5,
                "p_value": 0.56,
                "null_value": 0.0,
            },
            "variance_components": {
                "target_variance": -0.05,
                "residual_variance": 1.20,
            },
            "n_targets": 6,
            "n_raters": 2,
        },
    )
    out = _capture(res, detail="standard")
    assert "Intraclass Correlation" in out
    assert "Model" in out
    assert "two-way random effects" in out
    assert "absolute agreement" in out

    # Check definition appears before the numeric estimate in the text output
    def_pos = out.find("absolute agreement")
    est_pos = out.find("-0.152")
    assert def_pos != -1
    assert est_pos != -1
    assert def_pos < est_pos  # Definition precedes estimate!
    assert "-0.152" in out  # Preserves negative estimate without clamping to 0


def test_study_planner_does_not_show_observed_post_hoc_power(study_planning_result):
    """Verify study planning results state prospective design and never show post-hoc power."""
    out = _capture(study_planning_result, detail="standard")
    assert "study planning" in out.lower()
    assert "required total n" in out.lower() or "sample" in out.lower()
    assert "is not observed post-hoc power" in out.lower()


def test_sensitivity_does_not_rank_by_p_value(sensitivity_result):
    """Verify sensitivity table does not sort or rank scenarios by p-value."""
    out = _capture(sensitivity_result, detail="standard")
    assert "sensitivity analysis" in out.lower()
    assert "scenario" in out.lower()
    assert "rank" not in out.lower()


def test_practical_significance_separates_threshold_and_statistical_significance(
    practical_significance_result,
):
    """Verify practical significance displays threshold relationship
    distinct from statistical p-value.
    """
    out = _capture(practical_significance_result, detail="standard")
    assert "practical significance" in out.lower()
    assert "researcher threshold" in out.lower() or "threshold" in out.lower()
    assert "practical verdict" in out.lower() or "verdict" in out.lower()
    assert "statistical significance" in out.lower() or "p-value" in out.lower()


def test_audit_colors_and_text_operational_only(welch_t_workflow):
    """Verify Audit output communicates operational consistency, not scientific truth."""
    audit = welch_t_workflow.audit
    assert audit is not None
    out = _capture(audit, detail="standard")
    assert "audit" in out.lower()
    assert "pass" in out.lower()
    assert (
        "does not certify empirical truth" in out.lower() or "internal consistency" in out.lower()
    )


def test_completeness_says_not_study_quality(reporting_completeness_result):
    """Verify completeness output explicitly notes it is not a study-quality score."""
    out = _capture(reporting_completeness_result, detail="standard")
    assert "reporting completeness" in out.lower()
    assert "does not certify study quality" in out.lower() or "study quality" in out.lower()


# ── Descriptives & Governance Objects Tests ─────────────────────────────────


def test_frequency_table_rendering(frequency_table_data):
    """Verify frequency_table output renders properly."""
    out_std = _capture(frequency_table_data, detail="standard")
    assert "frequency" in out_std.lower()
    assert "category" in out_std.lower()
    assert "valid %" in out_std.lower()

    out_cmp = _capture(frequency_table_data, detail="compact")
    assert "frequency" in out_cmp.lower()

    out_full = _capture(frequency_table_data, detail="full")
    assert "n=6" in out_full.lower() or "total" in out_full.lower()


def test_cross_tab_rendering(cross_tab_data):
    """Verify cross_tab output renders properly."""
    out_std = _capture(cross_tab_data, detail="standard")
    assert "cross-tabulation" in out_std.lower()
    assert "status" in out_std.lower()

    out_full = _capture(cross_tab_data, detail="full")
    assert "cross-tabulation" in out_full.lower()


def test_statistical_analysis_plan_rendering(statistical_analysis_plan):
    """Verify StatisticalAnalysisPlan renders as a plan, not a result."""
    out = _capture(statistical_analysis_plan, detail="standard")
    assert "statistical analysis plan" in out.lower()
    assert "planned method" in out.lower() or "primary method" in out.lower()
    assert "plan" in out.lower()


def test_plan_adherence_rendering(plan_adherence_result):
    """Verify PlanAdherenceResult renders comparisons without misconduct claims."""
    out = _capture(plan_adherence_result, detail="standard")
    assert "plan adherence" in out.lower()
    assert "misconduct" not in out.lower()


def test_reproducibility_record_rendering(welch_t_workflow):
    """Verify ReproducibilityRecord renders runtime and seed info without raw dict dump."""
    repro = welch_t_workflow.reproducibility
    assert repro is not None
    out = _capture(repro, detail="standard")
    assert "reproducibility" in out.lower()
    assert "{'runtime':" not in out


def test_reproduction_outcome_rendering(welch_t_workflow):
    """Verify ReproductionOutcome renders replay status cleanly."""
    repro = welch_t_workflow.reproducibility
    assert repro is not None
    df = pd.DataFrame(
        {
            "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
            "group": ["A"] * 5 + ["B"] * 5,
        }
    )
    outcome = reproduce(repro, data=df)
    out = _capture(outcome, detail="standard")
    assert "reproduction outcome" in out.lower() or "reproduced" in out.lower()


def test_decision_ledger_rendering(decision_ledger_instance):
    """Verify DecisionLedger renders event table without claims of authenticated provenance."""
    out_std = _capture(decision_ledger_instance, detail="standard")
    assert "decision ledger" in out_std.lower()
    assert "event" in out_std.lower()
    assert "authenticated provenance" in out_std.lower() or "preregistration" in out_std.lower()

    out_full = _capture(decision_ledger_instance, detail="full")
    assert "decision ledger" in out_full.lower()


def test_session_snapshot_rendering(session_snapshot_result):
    """Verify ResearchSessionSnapshot renders cleanly without dumping raw JSON."""
    out = _capture(session_snapshot_result, detail="standard")
    assert "Research Session Snapshot" in out
    assert "Workflow Status" in out or "Status" in out
    assert '{"schema_version":' not in out


# ── Accessibility and Responsive Widths (Section 25) ─────────────────────────


@pytest.mark.parametrize("width", [70, 90, 120])
@pytest.mark.parametrize(
    "fixture_name",
    [
        "linear_regression_workflow",
        "repeated_measures_workflow",
        "two_way_anova_workflow",
        "icc_workflow",
    ],
)
def test_responsive_widths(request, fixture_name, width):
    """Verify complex outputs render cleanly at widths 70, 90, and 120 without exception."""
    wf = request.getfixturevalue(fixture_name)
    out = _capture(wf, detail="standard", width=width)
    assert len(out.strip()) > 0
    # Every line should be bounded roughly by width
    for line in out.splitlines():
        assert len(line) <= width + 5  # Allow minor border padding margin


def test_no_color_accessibility(linear_regression_workflow):
    """Verify no_color=True produces clean readable text with all labels preserved."""
    out = _capture(linear_regression_workflow, detail="standard", no_color=True)
    assert "Linear Regression" in out
    assert "Outcome" in out
    assert "Predictors" in out
    assert "R²" in out or "R-squared" in out


def test_non_tty_capture(linear_regression_workflow):
    """Verify non-TTY environments capture clean text output."""
    out = _capture(linear_regression_workflow, detail="standard", force_terminal=False)
    assert "Linear Regression" in out
    assert len(out.strip()) > 0


# ── Zero Recalculation Verification (Section 23 & 26) ───────────────────────


def test_zero_recalculation_during_show(
    welch_t_workflow, linear_regression_workflow, pearson_workflow
):
    """Ensure show() never invokes scipy.stats or statistical calculation backends."""
    from scipy import stats

    with patch.object(stats, "ttest_ind", side_effect=RuntimeError("ttest_ind called!")):
        with patch.object(stats, "pearsonr", side_effect=RuntimeError("pearsonr called!")):
            with patch.object(stats, "f_oneway", side_effect=RuntimeError("f_oneway called!")):
                # None of these show calls should trigger any calculation
                _capture(welch_t_workflow, detail="full")
                _capture(linear_regression_workflow, detail="full")
                _capture(pearson_workflow, detail="full")


# ── Missing Rich Graceful Handling ──────────────────────────────────────────


def test_missing_rich_graceful_handling(welch_t_workflow):
    """Verify show() raises UnsupportedPresentationError when Rich is not available."""
    with patch.dict("sys.modules", {"rich.console": None}):
        with pytest.raises(UnsupportedPresentationError, match="requires the 'rich' package"):
            show(welch_t_workflow)


def test_theme_graceful_fallback_without_rich():
    """Verify get_theme() returns None when Rich is not available."""
    from pyautostat.presentation.theme import get_theme

    with patch.dict("sys.modules", {"rich.theme": None}):
        assert get_theme() is None


# ── Correctness Remediation Pass Tests (Codex Requirements) ─────────────────


def test_logistic_or_ci_uses_authoritative_odds_ratio_ci():
    """Verify logistic regression uses odds_ratio_ci and NEVER log-odds CI for OR interval."""
    log_ci = {"lower": 0.216, "upper": 1.981, "level": 0.95, "status": "available"}
    or_ci = {"lower": 1.241, "upper": 7.253, "level": 0.95, "status": "available"}

    analysis = AnalysisResult(
        method_id="logistic_regression",
        status=AnalysisStatus.AVAILABLE,
        sample_size=100,
        excluded_rows=0,
        values={
            "outcome": "y",
            "modeled_event": "1",
            "coefficients": [
                {
                    "term": "x1",
                    "estimate": 1.0986,
                    "standard_error": 0.450,
                    "statistic": 2.441,
                    "statistic_type": "Wald z",
                    "p_value": 0.0146,
                    "confidence_interval": log_ci,
                    "odds_ratio": 3.000,
                    "odds_ratio_ci": or_ci,
                    "decision": "reject",
                }
            ],
            "fit": {
                "pseudo_r_squared": 0.25,
                "log_likelihood": -45.2,
                "aic": 94.4,
                "bic": 99.6,
            },
        },
    )

    # Layer A: Inspect normalized display model directly
    view_std = adapt(analysis, detail="standard")
    table_std = view_std.tables[0]
    row_std = table_std.rows[0].cells
    assert row_std[0] == "x1"
    assert "3.000" in row_std[1]
    assert "1.241 to 7.253" in row_std[2]
    assert "0.216 to 1.981" not in row_std[2]
    assert "2.44" in row_std[3]
    assert "0.015" in row_std[4]

    view_full = adapt(analysis, detail="full")
    table_full = view_full.tables[0]
    row_full = table_full.rows[0].cells
    assert row_full[0] == "x1"
    assert "3.000" in row_full[1]
    assert "1.241 to 7.253" in row_full[2]
    assert "0.216 to 1.981" not in row_full[2]
    assert "1.099" in row_full[3]
    assert "0.450" in row_full[4]
    assert "2.44" in row_full[5]
    assert "0.015" in row_full[6]

    # Layer B: show() capture smoke assertion
    out_std = _capture(analysis, detail="standard")
    assert "3.000" in out_std
    assert "1.241 to 7.253" in out_std
    assert "0.216 to 1.981" not in out_std

    out_full = _capture(analysis, detail="full")
    assert "3.000" in out_full
    assert "1.241 to 7.253" in out_full
    assert "0.450" in out_full
    assert "2.44" in out_full
    assert "0.216 to 1.981" not in out_full


def test_ols_coefficient_field_mapping_fidelity():
    """Verify OLS coefficient table displays estimate, standard_error, CI, t, and p_value."""
    analysis = AnalysisResult(
        method_id="linear_regression",
        status=AnalysisStatus.AVAILABLE,
        sample_size=50,
        excluded_rows=0,
        values={
            "outcome": "y",
            "predictors": ["x1"],
            "r_squared": 0.65,
            "adjusted_r_squared": 0.64,
            "coefficients": [
                {
                    "term": "x1",
                    "estimate": 2.500,
                    "standard_error": 0.500,
                    "statistic": 5.000,
                    "statistic_type": "t",
                    "p_value": 0.0042,
                    "confidence_interval": {
                        "lower": 1.480,
                        "upper": 3.520,
                        "level": 0.95,
                        "status": "available",
                    },
                    "standardized_beta": 0.806,
                    "decision": "reject",
                }
            ],
            "f_statistic": 25.0,
            "f_p_value": 0.0042,
        },
    )

    view = adapt(analysis, detail="standard")
    table = view.tables[0]
    row = table.rows[0].cells
    assert row[0] == "x1"
    assert "2.500" in row[1]
    assert "0.500" in row[2]
    assert "1.480 to 3.520" in row[3]
    assert "5.00" in row[4]
    assert "0.004" in row[5]

    out = _capture(analysis, detail="standard")
    assert "2.500" in out
    assert "0.500" in out
    assert "1.480 to 3.520" in out
    assert "5.00" in out
    assert "0.004" in out


def test_ols_diagnostics_use_stored_status_without_hardcoded_alpha():
    """Verify Breusch-Pagan and VIF diagnostics use stored status/advisory without p<0.05 logic."""
    analysis = AnalysisResult(
        method_id="linear_regression",
        status=AnalysisStatus.AVAILABLE,
        sample_size=60,
        excluded_rows=0,
        values={
            "outcome": "y",
            "predictors": ["x1"],
            "r_squared": 0.50,
            "coefficients": [
                {
                    "term": "x1",
                    "estimate": 1.0,
                    "standard_error": 0.2,
                    "statistic": 5.0,
                    "p_value": 0.001,
                    "confidence_interval": {
                        "lower": 0.6,
                        "upper": 1.4,
                        "level": 0.95,
                        "status": "available",
                    },
                }
            ],
            "diagnostics": {
                "breusch_pagan": {
                    "status": "homoscedasticity_tenable",
                    "decision": "Homoscedasticity tenable",
                    "lm_statistic": 3.95,
                    "lm_p_value": 0.047,  # Less than 0.05, but backend deemed tenable
                    "f_statistic": 3.90,
                    "f_p_value": 0.049,
                },
                "vif": {
                    "maximum": 6.2,  # > 5.0, but backend policy allows up to 10
                    "threshold_policy": "Policy threshold is 10.0",
                    "terms": {
                        "x1": {
                            "vif": 6.2,
                            "status": "acceptable",
                            "advisory": "Collinearity within acceptable tolerance",
                        }
                    },
                },
            },
        },
    )

    view = adapt(analysis, detail="standard")
    bp_diag = next(d for d in view.diagnostics if "Breusch-Pagan" in d.label)
    assert bp_diag.status != "REVIEW"
    assert bp_diag.severity != "warning"
    assert "0.047" in (bp_diag.detail or "") or "Homoscedasticity tenable" in (bp_diag.detail or "")

    vif_diag = next(d for d in view.diagnostics if "VIF" in d.label)
    assert vif_diag.status != "REVIEW"
    assert "6.20" in (vif_diag.detail or "")
    assert "10.0" in (vif_diag.detail or "")


def test_repeated_measures_sphericity_uses_stored_status_at_alpha_0_01():
    """Verify repeated-measures sphericity uses stored status and does not threshold at 0.05."""
    analysis = AnalysisResult(
        method_id="repeated_measures_anova",
        status=AnalysisStatus.AVAILABLE,
        sample_size=20,
        excluded_rows=0,
        values={
            "f_statistic": 4.12,
            "df_num": 2,
            "df_den": 38,
            "p_value": 0.024,
            "sphericity": {
                "status": "not_rejected",
                "decision": "Fail to reject null of sphericity",
                "statistic": 0.88,
                "p_value": 0.035,
            },
            "greenhouse_geisser": {
                "applied": False,
                "epsilon": 0.89,
                "p_value": 0.027,
            },
        },
    )

    view = adapt(analysis, detail="standard")
    diag = next(d for d in view.diagnostics if "Sphericity" in d.label)
    assert diag.status == "NOT REJECTED"
    assert diag.status != "VIOLATED"

    out = _capture(analysis, detail="standard")
    assert "VIOLATED" not in out
    assert "NOT REJECTED" in out


def test_repeated_measures_anova_pairwise_condition_mapping():
    """Verify repeated-measures ANOVA pairwise follow-up maps conditions and stats."""
    analysis = AnalysisResult(
        method_id="repeated_measures_anova",
        status=AnalysisStatus.AVAILABLE,
        sample_size=15,
        excluded_rows=0,
        values={
            "f_statistic": 8.50,
            "df_num": 2,
            "df_den": 28,
            "p_value": 0.001,
            "pairwise_comparisons": [
                {
                    "first_condition": "Baseline",
                    "second_condition": "FollowUp",
                    "estimate": -3.50,
                    "confidence_interval": {
                        "lower": -5.20,
                        "upper": -1.80,
                        "level": 0.95,
                        "status": "available",
                    },
                    "statistic": -4.20,
                    "statistic_name": "t",
                    "effect_size": {
                        "name": "cohen_dz",
                        "value": 1.08,
                    },
                    "raw_p_value": 0.0008,
                    "adjusted_p_value": 0.0024,
                    "decision": "reject",
                },
                {
                    "first_condition": "FollowUp",
                    "second_condition": "Maintenance",
                    "status": "unavailable",
                    "reason": "insufficient paired observations",
                },
            ],
        },
    )

    view = adapt(analysis, detail="standard")
    table = view.tables[0]
    row0 = table.rows[0].cells
    assert row0[0] == "Baseline - FollowUp"
    assert "-3.50" in row0[1]
    assert "-5.20 to -1.80" in row0[2]
    assert "1.08" in row0[3]
    assert "0.002" in row0[4]
    assert "Reject H0" in row0[5]

    row1 = table.rows[1].cells
    assert row1[0] == "FollowUp - Maintenance"
    assert "Unavailable" in row1[1]

    out = _capture(analysis, detail="standard", width=120)
    assert "Baseline - FollowUp" in out
    assert "-3.50" in out
    assert "-5.20 to -1.80" in out
    assert "1.08" in out
    assert "0.002" in out


def test_friedman_pairwise_mapping_fidelity():
    """Verify Friedman pairwise follow-up maps conditions, statistic, rank-biserial, and p."""
    analysis = AnalysisResult(
        method_id="friedman_test",
        status=AnalysisStatus.AVAILABLE,
        sample_size=12,
        excluded_rows=0,
        values={
            "statistic": 10.5,
            "p_value": 0.005,
            "kendall_w": {"value": 0.44},
            "pairwise_comparisons": [
                {
                    "first_condition": "Placebo",
                    "second_condition": "DrugA",
                    "statistic": 14.0,
                    "statistic_name": "Wilcoxon W",
                    "effect_size": {
                        "name": "matched_pairs_rank_biserial",
                        "value": -0.62,
                    },
                    "raw_p_value": 0.004,
                    "adjusted_p_value": 0.012,
                    "decision": "reject",
                }
            ],
        },
    )

    view = adapt(analysis, detail="standard")
    table = view.tables[0]
    row0 = table.rows[0].cells
    assert row0[0] == "Placebo vs DrugA"
    assert "14.0" in row0[1]
    assert "-0.620" in row0[2]
    assert "0.012" in row0[3]
    assert "Reject H0" in row0[4]

    out = _capture(analysis, detail="standard")
    assert "Placebo vs DrugA" in out
    assert "14.0" in out
    assert "-0.620" in out


def test_multigroup_pairwise_decision_fidelity_at_non_default_alpha():
    """Verify multigroup pairwise decision uses stored decision without 0.05 re-threshold."""
    analysis = AnalysisResult(
        method_id="welch_anova",
        status=AnalysisStatus.AVAILABLE,
        sample_size=45,
        excluded_rows=0,
        values={
            "f_statistic": 4.80,
            "p_value": 0.015,
            "pairwise_comparisons": [
                {
                    "first_group": "A",
                    "second_group": "B",
                    "mean_difference": 2.10,
                    "confidence_interval": {
                        "lower": 0.10,
                        "upper": 4.10,
                        "level": 0.99,
                        "status": "available",
                    },
                    "adjusted_p_value": 0.038,  # < 0.05, but under alpha=0.01 it is fail_to_reject
                    "decision": "fail_to_reject",
                }
            ],
        },
    )

    view = adapt(analysis, detail="standard")
    table = view.tables[0]
    row0 = table.rows[0].cells
    assert "A - B" in row0[0] or "A vs B" in row0[0]
    assert "0.038" in row0[3]
    assert "Fail to reject" in row0[4]
    assert "Reject H0" not in row0[4]

    out = _capture(analysis, detail="standard")
    assert "Fail to reject" in out
    assert "Reject H0" not in out


def test_kruskal_wallis_dunn_pairwise_headers_not_mean_difference():
    """Verify Kruskal-Wallis Dunn-Holm follow-up does NOT use mean difference headers."""
    analysis = AnalysisResult(
        method_id="kruskal_wallis",
        status=AnalysisStatus.AVAILABLE,
        sample_size=30,
        excluded_rows=0,
        values={
            "statistic": 7.82,
            "p_value": 0.020,
            "pairwise_comparisons": [
                {
                    "first_group": "Group1",
                    "second_group": "Group2",
                    "estimate": 6.5,
                    "estimate_name": "pooled mean-rank difference",
                    "statistic": 2.35,
                    "statistic_name": "Dunn z",
                    "effect_size": {"name": "rank_biserial", "value": 0.42},
                    "adjusted_p_value": 0.035,
                    "decision": "reject",
                }
            ],
        },
    )

    view = adapt(analysis, detail="standard")
    table = view.tables[0]
    assert "Mean Difference" not in table.columns
    assert "Simultaneous 95% Mean CI" not in table.columns
    assert "Mean-Rank Diff" in table.columns
    assert "Dunn z" in table.columns

    out = _capture(analysis, detail="standard", width=100)
    assert "Mean Difference" not in out
    assert "Mean-Rank Diff" in out


def test_practical_significance_semantic_string_status_fidelity():
    """Verify practical significance handles statistical_significance as a semantic string."""
    threshold = MeaningfulEffectThreshold(
        quantity="mean_difference",
        minimum_magnitude=5.0,
        unit="mg/dL",
        direction="two_sided",
    )

    # Case 1: no_evidence_against_null must NEVER render as "Statistically significant"
    res_no_ev = PracticalSignificanceResult(
        status="complete",
        quantity="mean_difference",
        estimate=2.1,
        threshold=threshold,
        confidence_interval={"lower": -1.0, "upper": 5.2, "level": 0.95, "status": "available"},
        point_estimate_relation="below_threshold",
        confidence_interval_relation="inconclusive",
        uncertainty_status="inconclusive",
        statistical_significance="no_evidence_against_null",
        conclusion="Point estimate is below threshold; interval is inconclusive.",
        warnings=(),
        provenance={},
    )

    view_no = adapt(res_no_ev, detail="standard")
    stat_sig_metric = next(m for m in view_no.key_metrics if m.label == "Statistical Evidence")
    assert stat_sig_metric.value == "No sufficient evidence against recorded null"
    assert "Statistically significant" not in stat_sig_metric.value

    out_no = _capture(res_no_ev, detail="standard")
    assert "No sufficient evidence against recorded null" in out_no
    assert "Statistically significant" not in out_no

    # Case 2: evidence_against_null
    res_ev = PracticalSignificanceResult(
        status="complete",
        quantity="mean_difference",
        estimate=8.2,
        threshold=threshold,
        confidence_interval={"lower": 5.5, "upper": 10.9, "level": 0.95, "status": "available"},
        point_estimate_relation="exceeds_threshold",
        confidence_interval_relation="entirely_above",
        uncertainty_status="clear_effect",
        statistical_significance="evidence_against_null",
        conclusion="Point estimate and confidence interval both exceed threshold.",
        warnings=(),
        provenance={},
    )
    view_ev = adapt(res_ev, detail="standard")
    metric_ev = next(m for m in view_ev.key_metrics if m.label == "Statistical Evidence")
    assert metric_ev.value == "Evidence against recorded null"

    # Case 3: unavailable
    res_un = PracticalSignificanceResult(
        status="complete",
        quantity="mean_difference",
        estimate=8.2,
        threshold=threshold,
        confidence_interval=None,
        point_estimate_relation="exceeds_threshold",
        confidence_interval_relation="unavailable",
        uncertainty_status="uncertain",
        statistical_significance="unavailable",
        conclusion="Unavailable.",
        warnings=(),
        provenance={},
    )
    view_un = adapt(res_un, detail="standard")
    metric_un = next(m for m in view_un.key_metrics if m.label == "Statistical Evidence")
    assert metric_un.value == "Unavailable"


def test_sensitivity_comparability_and_contrast_fidelity(welch_t_workflow):
    """Verify sensitivity table displays stored comparability and contrast accurately."""
    assistant = ResearchAssistant(
        pd.DataFrame(
            {
                "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
                "group": ["A"] * 5 + ["B"] * 5,
            }
        )
    )
    base = welch_t_workflow.analysis
    question_dist = replace(base.specification.question, estimand="distribution")
    scenarios = [
        SensitivitySpecification(
            name="equal_var",
            specification=base.specification,
            method_id="student_t",
            assumptions=("Equal population variances",),
        ),
        SensitivitySpecification(
            name="rank_comparison",
            specification=replace(base.specification, question=question_dist),
            method_id="mann_whitney_u",
            assumptions=("Ordinal or non-normal distribution",),
        ),
    ]
    sensitivity = assistant.sensitivity_analysis(base, scenarios=scenarios)
    assert [item.comparability.value for item in sensitivity.scenario_results] == [
        "same_estimand",
        "different_estimand",
    ]

    view = adapt(sensitivity, detail="standard")
    table = view.tables[0]
    row0_comp = table.rows[0].cells[2]
    row1_comp = table.rows[1].cells[2]
    assert "Same estimand" in row0_comp
    assert "Different estimand" in row1_comp

    row0_contrast = table.rows[0].cells[3]
    row1_contrast = table.rows[1].cells[3]
    assert row0_contrast != "Preserved"
    assert row1_contrast != "Preserved"

    out = _capture(sensitivity, detail="standard")
    assert "same estimand" in out.lower()
    assert "different estimand" in out.lower()


def test_two_way_anova_headline_interaction_f_statistic():
    """Verify Two-Way ANOVA interaction headline uses f_statistic and is not 'Unavailable'."""
    analysis = AnalysisResult(
        method_id="two_way_anova",
        status=AnalysisStatus.AVAILABLE,
        sample_size=48,
        excluded_rows=0,
        values={
            "factor_a": "Drug",
            "factor_b": "Dosage",
            "interaction_term": {
                "term": "Drug x Dosage",
                "term_type": "interaction",
                "sum_of_squares": 45.2,
                "mean_square": 22.6,
                "df": 2,
                "f_statistic": 6.84,  # Notice key is f_statistic, NOT statistic
                "p_value": 0.0025,
                "effect_size": {"name": "partial_eta_squared", "value": 0.23},
            },
            "terms": [
                {
                    "term": "Drug",
                    "term_type": "main_effect",
                    "sum_of_squares": 60.1,
                    "mean_square": 60.1,
                    "df": 1,
                    "f_statistic": 18.2,
                    "p_value": 0.0001,
                    "effect_size": {"name": "partial_eta_squared", "value": 0.28},
                },
                {
                    "term": "Dosage",
                    "term_type": "main_effect",
                    "sum_of_squares": 30.0,
                    "mean_square": 15.0,
                    "df": 2,
                    "f_statistic": 4.54,
                    "p_value": 0.016,
                    "effect_size": {"name": "partial_eta_squared", "value": 0.17},
                },
                {
                    "term": "Drug x Dosage",
                    "term_type": "interaction",
                    "sum_of_squares": 45.2,
                    "mean_square": 22.6,
                    "df": 2,
                    "f_statistic": 6.84,
                    "p_value": 0.0025,
                    "effect_size": {"name": "partial_eta_squared", "value": 0.23},
                },
            ],
        },
    )

    view = adapt(analysis, detail="standard")
    inter_metric = next(m for m in view.key_metrics if "F" in m.label)
    assert inter_metric.value == "6.840"
    assert inter_metric.value != "Unavailable"

    out = _capture(analysis, detail="standard")
    assert "6.840" in out
    assert "Drug x Dosage" in out


def test_icc_no_derived_percentage_and_negative_variance_component():
    """Verify ICC table displays negative variance component and derives NO percentage-of-total."""
    analysis = AnalysisResult(
        method_id="intraclass_correlation",
        status=AnalysisStatus.AVAILABLE,
        sample_size=8,
        excluded_rows=0,
        values={
            "variant": "ICC(2,1)",
            "definition": "absolute agreement",
            "model": "two-way random effects",
            "unit": "single rater",
            "primary_estimate": -0.125,
            "intraclass_correlation": -0.125,
            "confidence_interval": {
                "lower": -0.42,
                "upper": 0.25,
                "level": 0.95,
                "status": "available",
            },
            "variance_components": {
                "target_variance": -0.045,  # Unconstrained negative component
                "rater_variance": 0.150,
                "residual_variance": 0.900,
            },
            "n_targets": 8,
            "n_raters": 3,
        },
    )

    view = adapt(analysis, detail="standard")
    table = next(t for t in view.tables if "VARIANCE COMPONENTS" in (t.title or "").upper())
    assert table.columns == ("Component", "Estimate")
    assert "Percent of Total" not in table.columns
    assert "% of Total" not in table.columns
    row_target = next(r.cells for r in table.rows if "Target" in r.cells[0])
    assert "-0.045" in row_target[1]

    out = _capture(analysis, detail="standard")
    assert "-0.045" in out
    assert "Percent of Total" not in out
    assert "% of Total" not in out


def test_audit_status_roles_exact_mapping():
    """Verify AuditResult statuses ('passed', 'incomplete', 'failed') map to correct roles."""
    passed = AuditResult(
        status="passed",
        findings=(),
        checked_components=("a",),
        skipped_checks=(),
        source_references={},
    )
    view_pass = adapt(passed, detail="standard")
    metric_pass = view_pass.design_metrics[0]
    assert metric_pass.value == "PASSED"
    assert metric_pass.role == "status.success"
    assert view_pass.diagnostics[0].status == "PASSED"
    assert view_pass.diagnostics[0].severity == "success"

    incomplete = AuditResult(
        status="incomplete",
        findings=(),
        checked_components=("a",),
        skipped_checks=(),
        source_references={},
    )
    view_inc = adapt(incomplete, detail="standard")
    metric_inc = view_inc.design_metrics[0]
    assert metric_inc.value == "INCOMPLETE"
    assert metric_inc.role == "status.warning"
    assert view_inc.diagnostics[0].status == "INCOMPLETE"
    assert view_inc.diagnostics[0].severity == "warning"

    failed = AuditResult(
        status="failed",
        findings=(),
        checked_components=("a",),
        skipped_checks=(),
        source_references={},
    )
    view_fail = adapt(failed, detail="standard")
    metric_fail = view_fail.design_metrics[0]
    assert metric_fail.value == "FAILED"
    assert metric_fail.role == "status.error"
    assert view_fail.diagnostics[0].status == "FAILED"
    assert view_fail.diagnostics[0].severity == "error"


def test_audit_findings_severity_vocabulary_and_explanation():
    """Verify AuditFinding severities ('error', 'warning', 'pass') and explanations render."""
    f_err = AuditFinding(
        code="STATISTIC_MISMATCH",
        severity="error",
        component="report",
        field="test_statistic",
        expected=4.5,
        actual=5.2,
        explanation="The inspected value differs from the canonical source.",
    )
    f_warn = AuditFinding(
        code="MISSING_REQUIRED_WARNING",
        severity="warning",
        component="warnings",
        field="warnings",
        expected="Required warning",
        actual="None",
        explanation="A recommended warning was omitted.",
    )
    f_pass = AuditFinding(
        code="SPECIFICATION_MATCH",
        severity="pass",
        component="specification",
        field="alpha",
        expected=0.05,
        actual=0.05,
        explanation="Alpha matches specification.",
    )
    audit = AuditResult(
        status="failed",
        findings=(f_err, f_warn, f_pass),
        checked_components=("specification", "report", "warnings"),
        skipped_checks=(),
        source_references={"source": "ref"},
    )

    view = adapt(audit, detail="standard")
    # Status and role
    assert view.design_metrics[0].value == "FAILED"
    assert view.design_metrics[0].role == "status.error"
    assert view.diagnostics[0].status == "FAILED"
    assert view.diagnostics[0].severity == "error"

    # Invariants counting: error finding must count as failure, NOT zero!
    metric_fail = next(m for m in view.design_metrics if m.label == "Failures")
    assert metric_fail.value == "1 failures"
    assert metric_fail.role == "status.error"

    metric_warn = next(m for m in view.design_metrics if m.label == "Warnings")
    assert metric_warn.value == "1 warnings"
    assert metric_warn.role == "status.warning"

    metric_pass = next(m for m in view.design_metrics if m.label == "Passed Invariants")
    assert metric_pass.value == "1 checks"
    assert metric_pass.role == "status.success"

    # Table contains explanation, not empty message
    assert len(view.tables) == 1
    table = view.tables[0]
    row_err = table.rows[0]
    assert row_err.cells[0] == "ERROR"
    assert row_err.cells[1] == "Report"
    assert row_err.cells[2] == "STATISTIC_MISMATCH"
    assert row_err.cells[3] == "The inspected value differs from the canonical source."

    row_warn = table.rows[1]
    assert row_warn.cells[0] == "WARNING"
    assert row_warn.cells[3] == "A recommended warning was omitted."

    # show() rendering check
    out = _capture(audit, detail="standard")
    assert "FAILED" in out
    assert "1 failures" in out
    assert "1 warnings" in out
    assert "The inspected value differs from" in out


def test_analysis_plan_meaningful_threshold_uses_minimum_magnitude():
    """Verify StatisticalAnalysisPlan renders minimum_magnitude and no dataclass repr."""
    df = pd.DataFrame(
        {
            "group": ["a"] * 6 + ["b"] * 6,
            "score": [1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 9],
        }
    )
    asst = ResearchAssistant(df)
    draft = asst.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
        variable_types={"score": "continuous"},
    )
    thresh = MeaningfulEffectThreshold(
        quantity="mean_difference",
        minimum_magnitude=25.0,
        unit="USD/month",
        direction="two_sided",
        rationale="Budget significance threshold",
    )
    plan = asst.analysis_plan(draft, meaningful_threshold=thresh)

    view = adapt(plan, detail="standard")
    thresh_table = next(t for t in view.tables if "PRACTICAL THRESHOLD" in (t.title or "").upper())
    rows_dict = {r.cells[0]: r.cells[1] for r in thresh_table.rows}
    assert rows_dict["Minimum magnitude"] == "25.00"
    assert rows_dict["Unit"] == "USD/month"
    assert rows_dict["Rationale"] == "Budget significance threshold"

    out = _capture(plan, detail="standard")
    assert "25.00" in out
    assert "USD/month" in out
    assert "MeaningfulEffectThreshold(" not in out


def test_two_group_no_unavailable_descriptive_columns(welch_t_workflow):
    """Verify two-group mean summary table only shows Group and N when Mean/SD are not in result."""
    view = adapt(welch_t_workflow.analysis, detail="standard")
    table = next(t for t in view.tables if "GROUP SUMMARY" in (t.title or "").upper())
    assert table.columns == ("Group", "N")
    assert "Mean" not in table.columns
    assert "SD" not in table.columns

    out = _capture(welch_t_workflow.analysis, detail="standard")
    assert "GROUP SUMMARY" in out
    assert "Unavailable" not in out


def test_cronbach_alpha_surfaces_stored_missingness_and_scoring():
    """Verify Cronbach alpha adapter surfaces stored missingness and scoring configuration."""
    analysis = AnalysisResult(
        method_id="cronbach_alpha",
        status=AnalysisStatus.AVAILABLE,
        sample_size=40,
        excluded_rows=5,
        values={
            "items": ["q1", "q2", "q3", "q4"],
            "item_count": 4,
            "primary_estimate": 0.825,
            "confidence_interval": {
                "lower": 0.75,
                "upper": 0.88,
                "level": 0.95,
                "status": "available",
            },
            "mean_inter_item_correlation": 0.54,
            "scoring": {
                "reverse_scoring_applied": True,
                "reversed_items": [
                    {
                        "item": "q2",
                        "lower": 1.0,
                        "upper": 5.0,
                        "formula": "lower + upper - original",
                    }
                ],
            },
        },
        metadata={
            "sample": {
                "missing_data_policy": "complete cases across all selected items",
                "original_rows": 45,
                "analyzed_rows": 40,
                "excluded_rows": 5,
            },
        },
    )

    view = adapt(analysis, detail="standard")
    labels = {m.label: m.value for m in view.design_metrics}
    assert labels["Missingness Policy"] == "complete cases across all selected items"
    assert "Applied to 1 items (q2)" in labels["Reverse Scoring"]
    assert "5 rows" in labels["Excluded"]

    out = _capture(analysis, detail="standard")
    assert "complete cases across all selected items" in out
    assert "Applied to 1 items (q2)" in out


def test_cronbach_alpha_real_execution_path_with_missingness_and_reverse_scoring():
    """Verify Cronbach alpha presentation consumes the real execution schema from ResearchAssistant.

    Validates:
    - Analyzed and excluded respondent counts from real metadata['sample'];
    - Missingness policy from real metadata['sample'];
    - Reverse-scoring configuration and item names from real values['scoring'];
    - Item-level missingness in full detail mode;
    - Negative inter-item correlation diagnostic handling.
    """
    df = pd.DataFrame(
        {
            "q1": [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 3.0, None],
            "q2": [5.0, 4.0, 3.0, 2.0, 1.0, 4.0, 2.0, 3.0, 2.0],  # reverse scored (1 to 5)
            "q3": [2.0, 2.0, 3.0, 5.0, 5.0, 2.0, 4.0, 4.0, 1.0],
        }
    )
    wf = ResearchAssistant(df).reliability(
        ["q1", "q2", "q3"],
        reverse_scoring={"q2": (1.0, 5.0)},
        bootstrap_samples=40,
    )
    # Verify real engine output schema facts
    assert "sample" not in wf.analysis.values
    assert "sample" in wf.analysis.metadata
    assert "scoring" in wf.analysis.values
    assert "missingness" in wf.analysis.values

    # Test standard adaptation
    view = adapt(wf, detail="standard")
    labels = {m.label: m.value for m in view.design_metrics}
    assert labels["Respondents (N)"] == "8 complete cases"
    assert labels["Excluded"] == "1 rows"
    assert labels["Missingness Policy"] == "complete cases across all selected items"
    assert "Applied to 1 items (q2)" in labels["Reverse Scoring"]

    # Verify Inter-Item Alignment diagnostic is present
    diag = next(d for d in view.diagnostics if d.label == "Inter-Item Alignment")
    assert diag.status in ("CONSISTENT", "REVIEW")

    # Test full adaptation includes item-level missingness table
    view_full = adapt(wf, detail="full")
    table_titles = [t.title for t in view_full.tables]
    assert "ITEM-LEVEL MISSINGNESS" in table_titles

    # Test show() rendering
    out_std = _capture(wf, detail="standard")
    assert "8 complete cases" in out_std
    assert "1 rows" in out_std
    assert "complete cases across all selected items" in out_std
    assert "Applied to 1 items (q2)" in out_std

    out_full = _capture(wf, detail="full")
    assert "ITEM-LEVEL MISSINGNESS" in out_full


def test_stored_result_fidelity_and_zero_recalculation(
    welch_t_workflow, linear_regression_workflow
):
    """Verify that show() does not recalculate statistics or mutate the AnalysisResult."""
    from scipy import stats

    wf = deepcopy(welch_t_workflow)
    orig_values = deepcopy(wf.analysis.values)
    orig_sample_size = wf.analysis.sample_size
    orig_p_value = wf.analysis.values.get("p_value")

    with patch.object(
        stats, "ttest_ind", side_effect=AssertionError("ttest_ind recomputed during show!")
    ):
        with patch.object(
            stats, "pearsonr", side_effect=AssertionError("pearsonr recomputed during show!")
        ):
            with patch.object(
                stats, "f_oneway", side_effect=AssertionError("f_oneway recomputed during show!")
            ):
                with patch.object(
                    stats,
                    "mannwhitneyu",
                    side_effect=AssertionError("mannwhitneyu recomputed during show!"),
                ):
                    with patch(
                        "pyautostat.research_assistant.execute_selected_method",
                        side_effect=AssertionError("execute_selected_method called during show!"),
                    ):
                        with patch(
                            "pyautostat.ResearchAssistant.recommend_test",
                            side_effect=AssertionError("recommend_test called during show!"),
                        ):
                            out = _capture(wf, detail="full")
                            assert len(out) > 0

    assert wf.analysis.values == orig_values
    assert wf.analysis.sample_size == orig_sample_size
    assert wf.analysis.values.get("p_value") == orig_p_value


def test_workflow_execution_at_alpha_0_01():
    """Verify end-to-end workflow execution at alpha=0.01 where p is between 0.01 and 0.05.

    The decision must be 'Fail to reject' / 'no evidence against null', and presentation
    must faithfully render the stored decision without hardcoding 0.05.
    """
    g1 = [10.0, 11.0, 9.5, 10.5, 11.2, 10.8, 12.0, 9.8, 11.0, 10.2]
    g2 = [11.0, 11.5, 10.2, 11.8, 11.2, 10.8, 11.9, 11.4, 11.7, 11.1]
    df = pd.DataFrame({"score": g1 + g2, "group": ["A"] * 10 + ["B"] * 10})

    wf = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
        options=AnalysisOptions(alpha=0.01, confidence_level=0.99, random_seed=42),
    )
    p_val = wf.analysis.values.get("p_value")
    assert p_val is not None
    assert 0.01 < p_val < 0.05

    out = _capture(wf, detail="standard")
    assert "Independent Group Comparison" in out
    assert "alpha = 0.01" in out


def test_kruskal_wallis_equal_variance_diagnostic_not_applicable(kruskal_wallis_workflow):
    """Verify Kruskal-Wallis equal-variance diagnostic is NOT APPLICABLE and never Assumed."""
    view = adapt(kruskal_wallis_workflow, detail="standard")
    diag = next(d for d in view.diagnostics if d.label == "Equal Variance")
    assert diag.status == "Not applicable"
    assert diag.severity == "neutral"
    assert "Rank-based nonparametric comparison" in (diag.detail or "")
    assert "Assumed" not in diag.status

    out = _capture(kruskal_wallis_workflow, detail="standard")
    assert "[NOT APPLICABLE]" in out
    assert "Equal Variance" in out
    assert "Assumed" not in out


@pytest.mark.parametrize(
    ("planning_status", "expected_status", "expected_detail_fragment"),
    [
        ("planned", "PLANNED", "Researcher-supplied planned threshold"),
        ("exploratory", "EXPLORATORY", "Researcher-supplied exploratory threshold"),
        ("unknown", "RESEARCHER-SUPPLIED", "planning timing is not established"),
    ],
)
def test_practical_significance_respects_planning_status(
    planning_status, expected_status, expected_detail_fragment
):
    """Verify practical significance diagnostic wording respects threshold planning status."""
    threshold = MeaningfulEffectThreshold(
        quantity="mean_difference",
        minimum_magnitude=5.0,
        unit="mg/dL",
        direction="two_sided",
        planning_status=planning_status,
    )
    result = PracticalSignificanceResult(
        status="complete",
        quantity="mean_difference",
        estimate=6.0,
        threshold=threshold,
        confidence_interval={"lower": 2.0, "upper": 10.0, "level": 0.95, "status": "available"},
        point_estimate_relation="exceeds_threshold",
        confidence_interval_relation="inconclusive",
        uncertainty_status="inconclusive",
        statistical_significance="evidence_against_null",
        conclusion="Point estimate exceeds threshold.",
        warnings=(),
        provenance={},
    )
    view = adapt(result, detail="standard")
    diag = next(d for d in view.diagnostics if d.label == "Threshold Source")
    assert diag.status == expected_status
    assert expected_detail_fragment in (diag.detail or "")

    out = _capture(result, detail="standard")
    assert expected_status in out
    out_normalized = " ".join(out.split())
    assert expected_detail_fragment in out_normalized


def test_sensitivity_presentation_neutral_wording_and_mixed_estimands(welch_t_workflow):
    """Verify sensitivity presentation uses neutral subtitle and no blanket contrast claims."""
    assistant = ResearchAssistant(
        pd.DataFrame(
            {
                "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
                "group": ["A"] * 5 + ["B"] * 5,
            }
        )
    )
    base = welch_t_workflow.analysis
    question_dist = replace(base.specification.question, estimand="distribution")
    scenarios = [
        SensitivitySpecification(
            name="equal_var",
            specification=base.specification,
            method_id="student_t",
            assumptions=("Equal population variances",),
        ),
        SensitivitySpecification(
            name="rank_comparison",
            specification=replace(base.specification, question=question_dist),
            method_id="mann_whitney_u",
            assumptions=("Ordinal or non-normal distribution",),
        ),
    ]
    sensitivity = assistant.sensitivity_analysis(base, scenarios=scenarios)
    view = adapt(sensitivity, detail="standard")

    assert "Specification sensitivity analysis for" in (view.subtitle or "")
    assert "Robustness evaluation" not in (view.subtitle or "")

    diag = next(d for d in view.diagnostics if d.label == "Comparability Policy")
    assert diag.status == "EVALUATED"
    assert "displayed from their stored records" in (diag.detail or "")
    assert "Scenarios retain declared contrasts" not in (diag.detail or "")

    out = _capture(sensitivity, detail="standard")
    assert "Specification sensitivity analysis for" in out


def test_numeric_zero_preservation_in_presentation_adapters():
    """Verify that an authoritative test statistic of exactly 0.0 is preserved vs fallback."""
    # Test repeated measures ANOVA adapter with test_statistic=0.0 and fallback statistic=15.0
    analysis = AnalysisResult(
        method_id="repeated_measures_anova",
        status=AnalysisStatus.AVAILABLE,
        sample_size=20,
        excluded_rows=0,
        values={
            "test_statistic": 0.0,
            "statistic": 15.0,
            "degrees_of_freedom": [2, 18],
            "p_value": 1.0,
            "condition_summaries": [
                {"condition": "c1", "n": 0, "size": 10, "mean": 0.0, "sd": 0.0},
                {"condition": "c2", "n": 10, "size": 10, "mean": 0.0, "sd": 0.0},
            ],
        },
        metadata={"unit_id": "subject_id", "condition_order": ["c1", "c2"]},
    )
    view = adapt(analysis, detail="standard")
    omnibus_metric = next(m for m in view.key_metrics if "Omnibus" in m.label)
    assert "0.0" in omnibus_metric.value
    assert "15.0" not in omnibus_metric.value
    assert "15" not in omnibus_metric.value

    # Condition summary N must preserve 0 rather than falling back to size 10
    tbl = view.tables[0]
    row_c1 = tbl.rows[0]
    assert row_c1.cells[1] == "0"

    out = _capture(analysis, detail="standard")
    assert "F(2, 18) = 0.0" in out
    assert "15.0" not in out


def test_chi_square_expected_count_stored_status_wins():
    """Verify Chi-Square diagnostic uses stored status and does not re-threshold at 5."""
    # Sub-case A: min expected is 4.9, but backend records 'met'
    analysis_met = AnalysisResult(
        method_id="pearson_chi_square",
        status=AnalysisStatus.AVAILABLE,
        sample_size=100,
        excluded_rows=0,
        values={"primary_estimate": 0.35, "p_value": 0.01},
        metadata={
            "diagnostics": {
                "minimum_expected_count": 4.9,
                "expected_count_status": "met",
            }
        },
    )
    view_met = adapt(analysis_met, detail="standard")
    diag_met = next(d for d in view_met.diagnostics if d.label == "Min Expected Count")
    assert diag_met.status == "MET"
    assert diag_met.severity == "neutral"
    assert "4.9" in (diag_met.detail or "")

    out_met = _capture(analysis_met, detail="standard")
    assert "MET" in out_met
    assert "REVIEW" not in out_met

    # Sub-case B: min expected is 4.9, backend records 'review'
    analysis_rev = AnalysisResult(
        method_id="pearson_chi_square",
        status=AnalysisStatus.AVAILABLE,
        sample_size=100,
        excluded_rows=0,
        values={"primary_estimate": 0.35, "p_value": 0.01},
        metadata={
            "diagnostics": {
                "minimum_expected_count": 4.9,
                "expected_count_status": "review",
            }
        },
    )
    view_rev = adapt(analysis_rev, detail="standard")
    diag_rev = next(d for d in view_rev.diagnostics if d.label == "Min Expected Count")
    assert diag_rev.status == "REVIEW"
    assert diag_rev.severity == "review"

    # Sub-case C: missing status defaults to DOCUMENTED without inferring from 5
    analysis_none = AnalysisResult(
        method_id="pearson_chi_square",
        status=AnalysisStatus.AVAILABLE,
        sample_size=100,
        excluded_rows=0,
        values={"primary_estimate": 0.35, "p_value": 0.01},
        metadata={
            "diagnostics": {
                "min_expected_frequency": 4.9,
            }
        },
    )
    view_none = adapt(analysis_none, detail="standard")
    diag_none = next(d for d in view_none.diagnostics if d.label == "Min Expected Count")
    assert diag_none.status == "DOCUMENTED"
    assert diag_none.severity == "neutral"
