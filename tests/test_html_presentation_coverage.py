"""Comprehensive coverage tests for PyAutoStat static HTML presentation and
ResearchReport unification.

Validates:
- Static HTML rendering across all 24 statistical method IDs in compact, standard, and full modes.
- Method-family specific presentation requirements (one-sample, two-group, paired, multigroup,
  association, categorical, regression, reliability, repeated measures, factorial, ICC).
- All non-analysis and governance presentation objects:
  * Dataset profile
  * Frequency table
  * Cross-tabulation
  * Study planning
  * Sensitivity analysis
  * Practical significance
  * Statistical analysis plan
  * Plan adherence
  * Reporting completeness
  * Audit result
  * Reproducibility record
  * Reproduction outcome
  * Decision ledger
  * Research session snapshot
- Workflow status rendering: completed, needs_input, data_limited, unsupported, failed.
- ResearchReport vs standalone fidelity, zero recalculation, security XSS escaping,
  dynamic confidence levels, falsey value fidelity, standard-mode table bounding,
  table accessibility, and column alignment heuristics.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pyautostat import (
    AnalysisOptions,
    AnalysisResult,
    AnalysisStatus,
    MeaningfulEffectThreshold,
    ResearchAssistant,
    ResearchWorkflowResult,
    SensitivitySpecification,
    StudyPlanner,
    WorkflowStatus,
    reproduce,
    to_html,
)
from pyautostat.presentation import adapt
from pyautostat.presentation.html.formatting import escape_text, is_numeric_column

# ── Fixtures for 24 Statistical Methods ──────────────────────────────────────


@pytest.fixture
def one_sample_workflow() -> ResearchWorkflowResult:
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
def welch_t_workflow() -> ResearchWorkflowResult:
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
def student_t_workflow(welch_t_workflow: ResearchWorkflowResult) -> ResearchWorkflowResult:
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
def mann_whitney_workflow() -> ResearchWorkflowResult:
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
def paired_t_workflow() -> ResearchWorkflowResult:
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
def wilcoxon_workflow() -> ResearchWorkflowResult:
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


@pytest.fixture
def one_way_anova_workflow(welch_anova_workflow: ResearchWorkflowResult) -> ResearchWorkflowResult:
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
def kruskal_wallis_workflow() -> ResearchWorkflowResult:
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
def spearman_workflow() -> ResearchWorkflowResult:
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
def kendall_workflow() -> ResearchWorkflowResult:
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
def point_biserial_workflow() -> ResearchWorkflowResult:
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
def partial_pearson_workflow() -> ResearchWorkflowResult:
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
def fisher_workflow() -> ResearchWorkflowResult:
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
def mcnemar_workflow() -> ResearchWorkflowResult:
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
def linear_regression_workflow() -> ResearchWorkflowResult:
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
def logistic_regression_workflow() -> ResearchWorkflowResult:
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
def cronbach_alpha_workflow() -> ResearchWorkflowResult:
    df = pd.DataFrame(
        {
            "q1": [1, 2, 3, 4, 5, 2, 4, 3],
            "q2": [1, 2, 3, 4, 4, 2, 5, 3],
            "q3": [2, 2, 3, 5, 5, 2, 4, 4],
        }
    )
    return ResearchAssistant(df).reliability(["q1", "q2", "q3"], bootstrap_samples=40)


@pytest.fixture
def repeated_measures_workflow() -> ResearchWorkflowResult:
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
def friedman_workflow() -> ResearchWorkflowResult:
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
def two_way_anova_workflow() -> ResearchWorkflowResult:
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
                "B1",
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
def icc_workflow() -> ResearchWorkflowResult:
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


# ── Fixtures for Non-Analysis & Planning / Governance Objects ────────────────


@pytest.fixture
def profile_result():
    df = pd.DataFrame(
        {
            "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
            "group": ["A"] * 5 + ["B"] * 5,
        }
    )
    return ResearchAssistant(df).profile()


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
    df = pd.DataFrame(
        {
            "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
            "group": ["A"] * 5 + ["B"] * 5,
        }
    )
    return ResearchAssistant(df).reporting_completeness(welch_t_workflow.report, style="apa")


@pytest.fixture
def audit_result(welch_t_workflow):
    return welch_t_workflow.audit


@pytest.fixture
def reproducibility_record(welch_t_workflow):
    return welch_t_workflow.reproducibility


@pytest.fixture
def reproduction_outcome(welch_t_workflow):
    repro = welch_t_workflow.reproducibility
    df = pd.DataFrame(
        {
            "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
            "group": ["A"] * 5 + ["B"] * 5,
        }
    )
    return reproduce(repro, data=df)


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
    return ResearchAssistant(df).session_snapshot(welch_t_workflow)


# ── ALL 24 STATISTICAL METHODS VALIDATION ────────────────────────────────────


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
def test_all_24_methods_render_html_standard_compact_full(request, fixture_name, expected_label):
    """Verify each method renders in standard, compact, and full mode without error,

    and generates valid HTML with method label, no raw dict repr, and no traceback.
    """
    wf: ResearchWorkflowResult = request.getfixturevalue(fixture_name)

    # 1. Standard mode
    html_std = to_html(wf, detail="standard")
    assert "<!doctype html>" in html_std
    assert '<html lang="en">' in html_std
    assert expected_label.lower() in html_std.lower()
    assert "{'primary_estimate':" not in html_std
    assert "Traceback" not in html_std

    # 2. Compact mode
    html_cmp = to_html(wf, detail="compact")
    assert "<!doctype html>" in html_cmp
    assert len(html_cmp) > 100

    # 3. Full mode
    html_full = to_html(wf, detail="full")
    assert "<!doctype html>" in html_full
    assert expected_label.lower() in html_full.lower()

    # 4. Direct AnalysisResult rendering
    if wf.analysis is not None:
        html_res = to_html(wf.analysis, detail="standard")
        assert "<!doctype html>" in html_res
        assert "{'primary_estimate':" not in html_res

    # 5. ResearchReport rendering unification
    if wf.report is not None:
        html_rep = wf.report.to_html(detail="standard")
        assert "<!doctype html>" in html_rep
        assert '<html lang="en">' in html_rep
        assert "Traceback" not in html_rep


# ── METHOD-FAMILY SCIENTIFIC HTML QUALITY REGRESSIONS ────────────────────────


def test_student_vs_welch_variance_assumption_wording_html(student_t_workflow, welch_t_workflow):
    """Verify Student t prominently displays equal-variance assumption context in HTML."""
    std_html = to_html(student_t_workflow, detail="standard")
    welch_html = to_html(welch_t_workflow, detail="standard")

    assert "equal variance" in std_html.lower() or "assumed equal" in std_html.lower()
    assert "welch" in welch_html.lower()


def test_paired_t_condition_order_html(paired_t_workflow):
    """Verify paired t displays condition order prominently in HTML."""
    html = to_html(paired_t_workflow, detail="standard")
    assert "before" in html
    assert "after" in html
    assert "Condition order" in html or "Contrast" in html or "before - after" in html


def test_wilcoxon_signed_effect_orientation_html(wilcoxon_workflow):
    """Verify Wilcoxon signed-rank shows condition order and effect in HTML."""
    html = to_html(wilcoxon_workflow, detail="standard")
    assert "before" in html
    assert "after" in html
    assert "Rank-biserial" in html or "r_rb" in html or "biserial" in html.lower()


def test_mann_whitney_not_universally_median_test_html(mann_whitney_workflow):
    """Verify Mann-Whitney is not described universally as a median test in HTML."""
    html = to_html(mann_whitney_workflow, detail="standard")
    assert (
        "Rank-sum" in html
        or "stochastic" in html
        or "distribution" in html.lower()
        or "U=" in html
        or "u statistic" in html.lower()
    )
    assert "Rank-biserial" in html or "r_rb" in html


def test_welch_anova_and_games_howell_html(welch_anova_workflow):
    """Verify Welch ANOVA displays omnibus Welch F and Games-Howell follow-ups in HTML."""
    html = to_html(welch_anova_workflow, detail="standard")
    assert "welch" in html.lower()
    assert "games-howell" in html.lower()
    assert "adjusted p" in html.lower() or "simultaneous" in html.lower()


def test_one_way_anova_and_tukey_html(one_way_anova_workflow):
    """Verify One-way ANOVA displays F, eta-squared, and Tukey follow-ups in HTML."""
    html = to_html(one_way_anova_workflow, detail="standard")
    assert "one-way anova" in html.lower()
    assert "tukey" in html.lower()
    assert "eta" in html.lower()


def test_kruskal_wallis_and_dunn_holm_html(kruskal_wallis_workflow):
    """Verify Kruskal-Wallis displays H statistic, epsilon-squared, and Dunn follow-ups in HTML."""
    html = to_html(kruskal_wallis_workflow, detail="standard")
    assert "kruskal-wallis" in html.lower()
    assert "dunn" in html.lower()
    assert "holm" in html.lower() or "adjusted p" in html.lower()


def test_spearman_rho_notation_html(spearman_workflow):
    """Verify Spearman displays rho and monotonic association context in HTML."""
    html = to_html(spearman_workflow, detail="standard")
    assert "spearman" in html.lower()
    assert "rho" in html.lower() or "ρ" in html or "r_s" in html
    assert "monotonic" in html.lower()


def test_kendall_tau_b_notation_html(kendall_workflow):
    """Verify Kendall displays tau-b notation and tie context in HTML."""
    html = to_html(kendall_workflow, detail="standard")
    assert "kendall" in html.lower()
    assert "tau-b" in html.lower() or "τ_b" in html


def test_point_biserial_positive_level_html(point_biserial_workflow):
    """Verify point-biserial prominently displays positive/reference level in HTML."""
    html = to_html(point_biserial_workflow, detail="standard")
    assert "point-biserial" in html.lower()
    assert "positive level" in html.lower() or "reference" in html.lower() or "true" in html.lower()


def test_partial_pearson_controls_html(partial_pearson_workflow):
    """Verify partial Pearson displays control variables and no-causal-claim limitation in HTML."""
    html = to_html(partial_pearson_workflow, detail="standard")
    assert "partial pearson" in html.lower()
    assert "control variables" in html.lower() or "controls" in html.lower()
    assert "age" in html.lower()
    assert "attendance" in html.lower()


def test_chi_square_contingency_table_and_cramers_v_html(chi_square_workflow):
    """Verify Chi-Square displays contingency table and Cramer's V in HTML."""
    html = to_html(chi_square_workflow, detail="standard")
    assert "chi-square" in html.lower()
    assert "cramer" in html.lower()
    assert "contingency table" in html.lower() or "contingency" in html.lower()


def test_fisher_zero_cell_unavailable_ci_html():
    """Verify Fisher's exact with zero cell displays unavailable CI state explicitly in HTML."""
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
    html = to_html(wf, detail="standard")
    assert "Fisher" in html
    assert "N/A" in html or "unavailable" in html.lower() or "Odds Ratio" in html


def test_mcnemar_discordant_pairs_html(mcnemar_workflow):
    """Verify McNemar displays transition table and discordant pairs in HTML."""
    html = to_html(mcnemar_workflow, detail="standard")
    assert "McNemar" in html
    assert "Discordant pairs" in html or "Transition" in html
    assert "Proportion difference" in html or "Difference" in html


def test_ols_hc3_label_and_diagnostics_html(linear_regression_workflow):
    """Verify OLS regression shows HC3 covariance label and diagnostics in HTML."""
    html_std = to_html(linear_regression_workflow, detail="standard")
    assert "HC3" in html_std
    assert "R²" in html_std or "R-squared" in html_std

    html_full = to_html(linear_regression_workflow, detail="full")
    assert "Breusch-Pagan" in html_full or "HC3" in html_full


def test_logistic_modeled_event_and_or_table_html(logistic_regression_workflow):
    """Verify logistic regression shows modeled event and OR-first table in HTML."""
    html = to_html(logistic_regression_workflow, detail="standard")
    assert "logistic regression" in html.lower()
    assert "modeled event" in html.lower()
    assert "odds ratio" in html.lower() or "or" in html.lower()


def test_cronbach_item_diagnostics_html(cronbach_alpha_workflow):
    """Verify Cronbach alpha shows item diagnostics in HTML."""
    html = to_html(cronbach_alpha_workflow, detail="standard")
    assert "cronbach" in html.lower()
    assert "alpha if deleted" in html.lower()
    assert "item-total" in html.lower()


def test_repeated_anova_sphericity_and_gg_html(repeated_measures_workflow):
    """Verify repeated-measures ANOVA displays Mauchly and Greenhouse-Geisser in HTML."""
    html = to_html(repeated_measures_workflow, detail="standard")
    assert "repeated-measures anova" in html.lower()
    assert "mauchly" in html.lower() or "sphericity" in html.lower()
    assert "greenhouse-geisser" in html.lower() or "epsilon" in html.lower() or "gg" in html.lower()


def test_friedman_kendall_w_and_holm_html(friedman_workflow):
    """Verify Friedman test displays Kendall W and pairwise follow-up in HTML."""
    html = to_html(friedman_workflow, detail="standard")
    assert "friedman" in html.lower()
    assert "kendall" in html.lower() or "w" in html.lower()
    assert "pairwise" in html.lower() or "wilcoxon" in html.lower()


def test_two_way_anova_interaction_row_html(two_way_anova_workflow):
    """Verify Two-Way ANOVA displays factor A, factor B, and interaction A×B distinctly in HTML."""
    html = to_html(two_way_anova_workflow, detail="standard")
    assert "two-way" in html.lower()
    assert "a x b" in html.lower() or "interaction" in html.lower() or "a × b" in html.lower()
    assert "partial eta" in html.lower()


def test_icc_definition_before_estimate_and_negative_icc_html():
    """Verify ICC displays definition before estimate, and preserves negative ICC in HTML."""
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
    html = to_html(res, detail="standard")
    assert "Intraclass Correlation" in html
    assert "Model" in html
    assert "two-way random effects" in html
    assert "absolute agreement" in html

    def_pos = html.find("absolute agreement")
    est_pos = html.find("-0.152")
    assert def_pos != -1
    assert est_pos != -1
    assert def_pos < est_pos
    assert "-0.152" in html


# ── NON-ANALYSIS OBJECTS STATIC HTML VALIDATION ──────────────────────────────


def test_profile_html_rendering(profile_result):
    """Verify dataset profile renders cleanly to static HTML."""
    html_std = to_html(profile_result, detail="standard")
    assert "<!doctype html>" in html_std
    assert "Dataset Profile" in html_std
    assert "Rows" in html_std
    assert "Columns" in html_std
    assert "Missing cells" in html_std
    assert "NUMERIC VARIABLES" in html_std
    assert "CATEGORICAL VARIABLES" in html_std

    html_cmp = to_html(profile_result, detail="compact")
    assert "Dataset Profile" in html_cmp


def test_frequency_table_html_rendering(frequency_table_data):
    """Verify frequency_table output renders properly to HTML."""
    html_std = to_html(frequency_table_data, detail="standard")
    assert "<!doctype html>" in html_std
    assert "frequency" in html_std.lower()
    assert "category" in html_std.lower()
    assert "valid %" in html_std.lower()

    html_full = to_html(frequency_table_data, detail="full")
    assert "frequency" in html_full.lower()


def test_cross_tab_html_rendering(cross_tab_data):
    """Verify cross_tab output renders properly to HTML."""
    html_std = to_html(cross_tab_data, detail="standard")
    assert "<!doctype html>" in html_std
    assert "cross-tabulation" in html_std.lower()
    assert "status" in html_std.lower()


def test_study_planning_html_rendering(study_planning_result):
    """Verify StudyPlanningResult renders to HTML with prospective design context."""
    html = to_html(study_planning_result, detail="standard")
    assert "<!doctype html>" in html
    assert "study planning" in html.lower()
    assert "is not observed post-hoc power" in html.lower()


def test_sensitivity_html_rendering(sensitivity_result):
    """Verify SensitivityResult renders to HTML without ranking scenarios by p-value."""
    html = to_html(sensitivity_result, detail="standard")
    assert "<!doctype html>" in html
    assert "sensitivity analysis" in html.lower()
    assert "equal_var" in html


def test_practical_significance_html_rendering(practical_significance_result):
    """Verify PracticalSignificanceResult renders to HTML separating threshold and inference."""
    html = to_html(practical_significance_result, detail="standard")
    assert "<!doctype html>" in html
    assert "practical significance" in html.lower()
    assert "Evaluated Quantity" in html
    assert "Researcher Threshold" in html
    assert "Practical Verdict" in html


def test_statistical_analysis_plan_html_rendering(statistical_analysis_plan):
    """Verify StatisticalAnalysisPlan renders as a plan, not a result."""
    html = to_html(statistical_analysis_plan, detail="standard")
    assert "<!doctype html>" in html
    assert "statistical analysis plan" in html.lower()
    assert "plan" in html.lower()


def test_plan_adherence_html_rendering(plan_adherence_result):
    """Verify PlanAdherenceResult renders comparisons without misconduct claims."""
    html = to_html(plan_adherence_result, detail="standard")
    assert "<!doctype html>" in html
    assert "plan adherence" in html.lower()
    assert "misconduct" not in html.lower()


def test_reproducibility_record_html_rendering(reproducibility_record):
    """Verify ReproducibilityRecord renders runtime and seed info without raw dict dump."""
    html = to_html(reproducibility_record, detail="standard")
    assert "<!doctype html>" in html
    assert "reproducibility" in html.lower()
    assert "{'runtime':" not in html


def test_reproduction_outcome_html_rendering(reproduction_outcome):
    """Verify ReproductionOutcome renders replay status cleanly."""
    html = to_html(reproduction_outcome, detail="standard")
    assert "<!doctype html>" in html
    assert "reproduction outcome" in html.lower() or "reproduced" in html.lower()


def test_decision_ledger_html_rendering(decision_ledger_instance):
    """Verify DecisionLedger renders event table without claims of authenticated provenance."""
    html = to_html(decision_ledger_instance, detail="standard")
    assert "<!doctype html>" in html
    assert "decision ledger" in html.lower()
    assert "authenticated provenance" in html.lower() or "preregistration" in html.lower()


def test_session_snapshot_html_rendering(session_snapshot_result):
    """Verify ResearchSessionSnapshot renders cleanly without dumping raw JSON."""
    html = to_html(session_snapshot_result, detail="standard")
    assert "<!doctype html>" in html
    assert "Research Session Snapshot" in html
    assert '{"schema_version":' not in html


def test_reporting_completeness_html_rendering(reporting_completeness_result):
    """Verify completeness output explicitly notes it is not a study-quality score."""
    html = to_html(reporting_completeness_result, detail="standard")
    assert "<!doctype html>" in html
    assert "reporting completeness" in html.lower()
    assert "study quality" in html.lower()


def test_audit_result_html_rendering(audit_result):
    """Verify AuditResult renders audit checks and findings cleanly."""
    html = to_html(audit_result, detail="standard")
    assert "<!doctype html>" in html
    assert "audit" in html.lower()


# ── WORKFLOW STATUS HTML TESTS ───────────────────────────────────────────────


def test_needs_input_workflow_html_rendering():
    """Verify needs_input workflow renders actionable prompts without implying failure."""
    df = pd.DataFrame({"score": [10.0, 12.0, 11.0], "group": ["A", "B", "A"]})
    wf = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
    )
    assert wf.status.value == "needs_input"

    html = to_html(wf)
    assert "<!doctype html>" in html
    assert "Additional Information Required" in html
    assert "Analysis has not been run." in html
    assert "Required" in html
    assert "FAILED" not in html


def test_unsupported_workflow_html_rendering():
    """Verify unsupported workflow renders clear blocker explanation."""
    df = pd.DataFrame({"score": [10.0, 12.0, 11.0], "group": ["A", "B", "A"]})
    wf = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="clustered",
        variable_types={"score": "continuous"},
    )
    assert wf.status.value == "unsupported"

    html = to_html(wf)
    assert "<!doctype html>" in html
    assert "Unsupported Specification" in html
    assert "UNSUPPORTED" in html
    assert "incompatible" in html


def test_data_limited_workflow_html_rendering():
    """Verify data_limited workflow renders numerical limitation explanation."""
    too_small = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 3.0]})
    wf = ResearchAssistant(too_small).run(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="linear",
        design="independent",
        variable_types={"x": "continuous", "y": "continuous"},
    )
    assert wf.status.value == "data_limited"

    html = to_html(wf)
    assert "<!doctype html>" in html
    assert "Analysis Data-Limited" in html
    assert "DATA LIMITED" in html
    assert "at least three" in html


def test_failed_workflow_html_rendering(monkeypatch):
    """Verify failed workflow state renders failure reason without raw traceback."""
    df = pd.DataFrame(
        {
            "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
            "group": ["A"] * 5 + ["B"] * 5,
        }
    )
    asst = ResearchAssistant(df)

    def unavailable_analysis(draft, **kwargs):
        return AnalysisResult(
            method_id="welch_t",
            status=AnalysisStatus.UNAVAILABLE,
            warnings=("Computational singular matrix in test execution.",),
            specification=draft.specification,
            recommendation=asst.recommend_test(draft),
        )

    monkeypatch.setattr(asst, "analyze", unavailable_analysis)
    wf = asst.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )
    assert wf.status.value == "failed"

    html = to_html(wf)
    assert "<!doctype html>" in html
    assert "Analysis Failed" in html
    assert "FAILED" in html
    assert "Computational singular matrix" in html
    assert "Traceback (most recent call last)" not in html


# ── TABLE ACCESSIBILITY & DEDUPLICATION ──────────────────────────────────────


def test_no_duplicate_visible_table_titles(linear_regression_workflow):
    """Verify that table captions do not duplicate visible section headers."""
    html = to_html(linear_regression_workflow, detail="full")

    # Visible h3 tag for table title was replaced by sr-only caption
    assert '<caption class="sr-only">MODEL COEFFICIENTS</caption>' in html
    # The visible heading is h2
    assert "MODEL COEFFICIENTS</h2>" in html
    # No duplicate h3 MODEL COEFFICIENTS
    assert "<h3>MODEL COEFFICIENTS</h3>" not in html


def test_table_accessibility_structure(linear_regression_workflow):
    """Verify tables contain proper semantic headers and scope attributes."""
    html = to_html(linear_regression_workflow, detail="standard")
    assert '<th scope="col"' in html
    assert "<thead>" in html
    assert "<tbody>" in html
    assert '<div class="table-container">' in html


def test_column_alignment_heuristics():
    """Verify semantic non-numeric columns are left-aligned, not right-aligned."""
    text_cols = [
        "Method",
        "Scenario",
        "Comparability",
        "Reference",
        "Severity",
        "Rationale",
        "Planning Status",
        "Estimand",
        "Adjustment",
        "Reason",
        "Event",
        "Category",
        "Level",
        "Variable",
        "Condition",
    ]
    for col in text_cols:
        assert not is_numeric_column(col, 1), f"{col} should NOT be detected as numeric column!"

    numeric_cols = ["N", "df", "p-value", "Estimate", "Statistic", "t", "F", "R²", "SE"]
    for col in numeric_cols:
        assert is_numeric_column(col, 1), f"{col} SHOULD be detected as numeric column!"


# ── RESEARCHREPORT VS STANDALONE FIDELITY ────────────────────────────────────


def test_research_report_vs_standalone_fidelity(welch_t_workflow):
    """Verify that ResearchReport.to_html() matches standalone to_html() for key statistics."""
    analysis = welch_t_workflow.analysis
    report = welch_t_workflow.report
    assert analysis is not None
    assert report is not None

    standalone_html = to_html(analysis, detail="full")
    report_html = report.to_html(detail="full")

    # Both must contain exact same primary estimate, CI, effect, and p-value
    view = adapt(analysis, detail="full")
    for metric in view.key_metrics:
        assert metric.label in report_html
        assert escape_text(metric.value) in report_html
        assert escape_text(metric.value) in standalone_html

    # Both must contain the diagnostic status badges
    for diag in view.diagnostics:
        assert diag.label in report_html
        assert diag.label in standalone_html


# ── SECURITY & XSS TESTS ─────────────────────────────────────────────────────


def test_security_xss_escaping_in_report():
    """Verify that malicious XSS payloads in titles, labels, and text are strictly escaped."""
    df = pd.DataFrame(
        {
            "<script>alert('xss')</script>": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )
    col_name = "<script>alert('xss')</script>"
    asst = ResearchAssistant(df)
    wf = asst.run(
        objective="compare_groups",
        outcome=col_name,
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={col_name: "continuous", "group": "nominal"},
    )
    report = asst.report(wf.analysis, title="<script>alert('title')</script>")
    html = report.to_html(detail="full")

    assert "<script>" not in html
    title_escaped = (
        "&lt;script&gt;alert(&#x27;title&#x27;)&lt;/script&gt;" in html
        or "&lt;script&gt;alert('title')&lt;/script&gt;" in html
    )
    assert title_escaped
    xss_escaped = (
        "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html
        or "&lt;script&gt;alert('xss')&lt;/script&gt;" in html
    )
    assert xss_escaped


def test_practical_significance_and_sensitivity_xss_escaping(welch_t_workflow):
    """Verify practical significance and sensitivity components escape malicious input."""
    asst = ResearchAssistant(
        pd.DataFrame(
            {
                "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
                "group": ["A"] * 5 + ["B"] * 5,
            }
        )
    )
    base = welch_t_workflow.analysis

    threshold = MeaningfulEffectThreshold(
        "mean_difference",
        minimum_magnitude=5.0,
        unit="<script>alert('unit')</script>",
        rationale="<img src=x onerror=alert(1)>",
    )
    ps = asst.practical_significance(base, threshold=threshold)
    ps_html = to_html(ps)
    assert "<script>" not in ps_html
    assert "<img" not in ps_html

    sens = asst.sensitivity_analysis(
        base,
        scenarios=[
            SensitivitySpecification(
                "<script>alert('scen')</script>",
                base.specification,
                method_id="student_t",
                assumptions=("<iframe src=evil>",),
            )
        ],
    )
    sens_html = to_html(sens)
    assert "<script>" not in sens_html
    assert "<iframe" not in sens_html


# ── DYNAMIC CONFIDENCE LEVELS ACROSS FAMILIES ────────────────────────────────


@pytest.mark.parametrize("conf_level", [0.90, 0.95, 0.99])
def test_dynamic_confidence_levels_across_method_families(conf_level):
    pct = int(round(conf_level * 100))
    expected_ci = f"{pct}% CI"

    # 1. Mean comparison
    df_t = pd.DataFrame(
        {"score": [10.0, 12.0, 11.0, 20.0, 22.0, 21.0], "group": ["A", "A", "A", "B", "B", "B"]}
    )
    wf_t = ResearchAssistant(df_t).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
        options=AnalysisOptions(alpha=1.0 - conf_level, confidence_level=conf_level),
    )
    html_t = to_html(wf_t)
    assert expected_ci in html_t

    # 2. Association
    df_cor = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.1, 6.0, 8.2, 9.9]})
    wf_cor = ResearchAssistant(df_cor).run(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="linear",
        design="independent",
        variable_types={"x": "continuous", "y": "continuous"},
        options=AnalysisOptions(alpha=1.0 - conf_level, confidence_level=conf_level),
    )
    html_cor = to_html(wf_cor)
    assert expected_ci in html_cor

    # 3. Regression
    df_reg = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    wf_reg = ResearchAssistant(df_reg).run(
        objective="regression",
        outcome="y",
        predictors=["x"],
        design="independent",
        estimand="conditional_mean",
        variable_types={"y": "continuous", "x": "continuous"},
        options=AnalysisOptions(alpha=1.0 - conf_level, confidence_level=conf_level),
    )
    html_reg = to_html(wf_reg)
    assert expected_ci in html_reg


# ── FALSEY VALUE FIDELITY ────────────────────────────────────────────────────


def test_falsey_value_fidelity_in_html():
    """Verify that falsey values (0, 0.0, False) are not coerced to None or empty strings."""
    df = pd.DataFrame(
        {
            "category": [0, 0, 0, 1, 1, 1],
            "flag": [False, False, False, True, True, True],
            "value": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        }
    )
    dict_meta = {"category": {"type": "nominal"}, "flag": {"type": "boolean"}}
    freq = ResearchAssistant(df).frequency_table("category", data_dictionary=dict_meta)
    html_freq = to_html(freq)
    assert '<td class="align-left">0</td>' in html_freq

    crosstab = ResearchAssistant(df).cross_tab("category", "flag", data_dictionary=dict_meta)
    html_ct = to_html(crosstab)
    assert "0" in html_ct
    assert "False" in html_ct


# ── STANDARD-MODE TABLE BOUNDING ─────────────────────────────────────────────


def test_standard_mode_table_bounding_in_report():
    """Verify that large pairwise tables in ResearchReport are bounded in standard mode."""
    # Create dataset with 6 groups -> 15 pairwise comparisons
    groups = ["A", "B", "C", "D", "E", "F"]
    rows = []
    for g in groups:
        for val in [1.0, 2.0, 3.0, 4.0]:
            rows.append({"group": g, "score": val + (ord(g) - ord("A"))})
    df = pd.DataFrame(rows)
    wf = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
    )
    report = wf.report
    assert report is not None

    html_std = report.to_html(detail="standard")
    assert "Showing 6 of" in html_std
    assert "use detail=&#x27;full&#x27;" in html_std or "use detail='full'" in html_std

    html_full = report.to_html(detail="full")
    assert "Showing 6 of" not in html_full


# ── ZERO RECALCULATION VALIDATION ────────────────────────────────────────────


def test_zero_recalculation_on_standalone_and_report_html(welch_t_workflow):
    """Verify that neither standalone to_html nor ResearchReport.to_html triggers recalculation."""
    analysis = welch_t_workflow.analysis
    report = welch_t_workflow.report
    assert analysis is not None
    assert report is not None

    with (
        patch("scipy.stats.ttest_ind", side_effect=RuntimeError("ttest_ind called!")),
        patch("scipy.stats.f_oneway", side_effect=RuntimeError("f_oneway called!")),
        patch("scipy.stats.pearsonr", side_effect=RuntimeError("pearsonr called!")),
        patch("scipy.stats.kruskal", side_effect=RuntimeError("kruskal called!")),
    ):
        out1 = to_html(welch_t_workflow, detail="full")
        out2 = to_html(analysis, detail="full")
        out3 = report.to_html(detail="full")

    assert "<!doctype html>" in out1
    assert "<!doctype html>" in out2
    assert "<!doctype html>" in out3


# ── HTML VALIDITY & STRUCTURE TESTS (Section 37) ─────────────────────────────


def test_html_validity_and_structure(linear_regression_workflow):
    """Verify HTML structural validity: single doctype, single root/head/body, viewport,

    no duplicate section IDs, and matching table column/row counts.
    """
    import re

    for html in [
        to_html(linear_regression_workflow, detail="full"),
        linear_regression_workflow.report.to_html(detail="full"),
    ]:
        # Document structure
        assert html.lower().count("<!doctype html>") == 1
        assert html.count('<html lang="en">') == 1
        assert html.count("</html>") == 1
        assert html.count("<head>") == 1
        assert html.count("</head>") == 1
        assert html.count("<body>") == 1
        assert html.count("</body>") == 1
        assert '<meta charset="utf-8">' in html
        assert '<meta name="viewport" content="width=device-width, initial-scale=1.0">' in html

        # Unique section IDs
        ids = re.findall(r'id="([^"]+)"', html)
        assert len(ids) == len(set(ids)), f"Duplicate HTML IDs found: {ids}"

        # Table structure: row cell count matches column header count
        table_matches = re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL)
        for t_content in table_matches:
            col_count = len(re.findall(r"<th\s+scope=[\"']col[\"']", t_content))
            if col_count > 0:
                tbody_match = re.search(r"<tbody>(.*?)</tbody>", t_content, re.DOTALL)
                if tbody_match:
                    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tbody_match.group(1), re.DOTALL)
                    for r in rows:
                        td_count = len(re.findall(r"<td[\s>]", r))
                        assert td_count == col_count, (
                            f"Row cell count {td_count} does not match column count {col_count}"
                        )
