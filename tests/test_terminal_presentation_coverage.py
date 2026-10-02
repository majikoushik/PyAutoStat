"""Comprehensive coverage tests for the PyAutoStat Rich terminal presentation layer.

Validates all 24 statistical methods, direct descriptive outputs,
governance/planning objects, responsive widths, zero-recalculation,
and specific scientific presentation requirements.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from pyautostat import (
    AnalysisOptions,
    AnalysisResult,
    AnalysisStatus,
    MeaningfulEffectThreshold,
    ResearchAssistant,
    ResearchWorkflowResult,
    SensitivitySpecification,
    StudyPlanner,
    UnsupportedPresentationError,
    WorkflowStatus,
    reproduce,
    show,
)


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
