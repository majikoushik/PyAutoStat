"""Reusable factory functions producing schema-faithful workflows and analysis results.

Covers all 24 registered statistical methods.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from pyautostat import (
    ResearchAssistant,
    ResearchWorkflowResult,
    WorkflowStatus,
)


def make_one_sample_workflow() -> ResearchWorkflowResult:
    df = pd.DataFrame({"score": [48.0, 51.0, 53.0, 55.0, 52.0, 49.0]})
    return ResearchAssistant(df).run(
        objective="compare_reference",
        outcome="score",
        reference_value=50.0,
        design="independent",
        estimand="mean",
        variable_types={"score": "continuous"},
    )


def make_welch_t_workflow() -> ResearchWorkflowResult:
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


def make_student_t_workflow() -> ResearchWorkflowResult:
    from pyautostat import Recommendation, execution

    base_wf = make_welch_t_workflow()
    asst = ResearchAssistant(
        pd.DataFrame(
            {
                "score": [10.0, 12.0, 11.0, 13.0, 14.0, 20.0, 22.0, 21.0, 23.0, 24.0],
                "group": ["A"] * 5 + ["B"] * 5,
            }
        )
    )
    spec = base_wf.specification
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
        draft=base_wf.draft,
        recommendation=rec,
        analysis=res,
        interpretation=interp,
        report=rep,
        audit=asst.audit(rep, result=res),
        reproducibility=asst.reproducibility_record(res),
    )


def make_mann_whitney_workflow() -> ResearchWorkflowResult:
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


def make_paired_t_workflow() -> ResearchWorkflowResult:
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


def make_wilcoxon_workflow() -> ResearchWorkflowResult:
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


def make_welch_anova_workflow() -> ResearchWorkflowResult:
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


def make_one_way_anova_workflow() -> ResearchWorkflowResult:
    from pyautostat import Recommendation, execution

    base_wf = make_welch_anova_workflow()
    asst = ResearchAssistant(
        pd.DataFrame(
            {
                "group": ["C"] * 6 + ["A"] * 7 + ["B"] * 5,
                "score": [2, 3, 4, 5, 7, 8, 8, 10, 12, 15, 18, 22, 27, 1, 2, 2, 3, 5],
            }
        )
    )
    spec = base_wf.specification
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
        draft=base_wf.draft,
        recommendation=rec,
        analysis=res,
        interpretation=interp,
        report=rep,
        audit=asst.audit(rep, result=res),
        reproducibility=asst.reproducibility_record(res),
    )


def make_kruskal_wallis_workflow() -> ResearchWorkflowResult:
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


def make_pearson_workflow() -> ResearchWorkflowResult:
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


def make_spearman_workflow() -> ResearchWorkflowResult:
    df = pd.DataFrame({"hours": [1, 2, 3, 4, 5, 6], "score": [2, 4, 3, 7, 8, 10]})
    return ResearchAssistant(df).run(
        objective="association",
        outcome="score",
        predictor="hours",
        design="independent",
        estimand="monotonic",
        variable_types={"score": "continuous", "hours": "continuous"},
    )


def make_kendall_workflow() -> ResearchWorkflowResult:
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


def make_point_biserial_workflow() -> ResearchWorkflowResult:
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


def make_partial_pearson_workflow() -> ResearchWorkflowResult:
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


def make_chi_square_workflow() -> ResearchWorkflowResult:
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


def make_fisher_workflow() -> ResearchWorkflowResult:
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


def make_mcnemar_workflow() -> ResearchWorkflowResult:
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


def make_linear_regression_workflow() -> ResearchWorkflowResult:
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


def make_logistic_regression_workflow() -> ResearchWorkflowResult:
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


def make_cronbach_alpha_workflow() -> ResearchWorkflowResult:
    df = pd.DataFrame(
        {
            "q1": [1, 2, 3, 4, 5, 2, 4, 3],
            "q2": [1, 2, 3, 4, 4, 2, 5, 3],
            "q3": [2, 2, 3, 5, 5, 2, 4, 4],
        }
    )
    return ResearchAssistant(df).reliability(["q1", "q2", "q3"], bootstrap_samples=40)


def make_repeated_measures_workflow() -> ResearchWorkflowResult:
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


def make_friedman_workflow() -> ResearchWorkflowResult:
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
        estimand="distribution",
        unit_id="participant",
        condition_order=("s1", "s2", "s3"),
        variable_types={"score": "continuous", "session": "ordinal"},
    )


def make_two_way_anova_workflow() -> ResearchWorkflowResult:
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


def make_icc_workflow() -> ResearchWorkflowResult:
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


ALL_24_METHOD_FACTORIES: dict[str, tuple[Callable[[], ResearchWorkflowResult], str]] = {
    "welch_t": (make_welch_t_workflow, "Welch"),
    "student_t": (make_student_t_workflow, "Student"),
    "mann_whitney_u": (make_mann_whitney_workflow, "Mann-Whitney"),
    "one_sample_t": (make_one_sample_workflow, "One-Sample"),
    "paired_t": (make_paired_t_workflow, "Paired"),
    "wilcoxon_signed_rank": (make_wilcoxon_workflow, "Wilcoxon"),
    "welch_anova": (make_welch_anova_workflow, "Welch's ANOVA"),
    "one_way_anova": (make_one_way_anova_workflow, "One-Way ANOVA"),
    "kruskal_wallis": (make_kruskal_wallis_workflow, "Kruskal-Wallis"),
    "pearson_correlation": (make_pearson_workflow, "Pearson"),
    "spearman_correlation": (make_spearman_workflow, "Spearman"),
    "kendall_tau_b": (make_kendall_workflow, "Kendall"),
    "point_biserial_correlation": (make_point_biserial_workflow, "Point-biserial"),
    "partial_pearson_correlation": (make_partial_pearson_workflow, "Partial Pearson"),
    "pearson_chi_square": (make_chi_square_workflow, "chi-square"),
    "fisher_exact": (make_fisher_workflow, "Fisher"),
    "mcnemar": (make_mcnemar_workflow, "McNemar"),
    "linear_regression": (make_linear_regression_workflow, "Linear Regression"),
    "logistic_regression": (make_logistic_regression_workflow, "Logistic Regression"),
    "cronbach_alpha": (make_cronbach_alpha_workflow, "Cronbach"),
    "repeated_measures_anova": (make_repeated_measures_workflow, "Repeated-Measures ANOVA"),
    "friedman_test": (make_friedman_workflow, "Friedman"),
    "two_way_anova": (make_two_way_anova_workflow, "Two-Way"),
    "intraclass_correlation": (make_icc_workflow, "Intraclass Correlation"),
}
