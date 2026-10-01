"""Final scientific closure validation test across all supported major method families.

Verifies the end-to-end contract:
design -> estimand -> validation -> execution -> uncertainty ->
interpretation -> report -> audit -> reproducibility.

Covers:
1. Dataset profiling
2. Two-sample independent means (Welch t)
3. One-way multi-group means (Welch ANOVA)
4. Paired two-condition means (Paired t)
5. Paired ranks (Wilcoxon signed-rank)
6. Bivariate monotonic correlation (Spearman)
7. Exact contingency table (Fisher exact)
8. Ordinary least-squares regression with HC3 (Linear regression)
9. Binary logistic regression (Logistic regression)
10. Scale internal consistency (Cronbach's alpha)
11. One-way repeated-measures ANOVA (Greenhouse-Geisser correction)
12. Multi-condition repeated ranks (Friedman test)
13. Two-way factorial ANOVA (Type II and Type III)
14. Intraclass correlation coefficient (ICC across canonical variants)
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from pyautostat import (
    InsightEngine,
    ReportGenerator,
    ResearchAssistant,
    reproduce,
)


def test_closure_dataset_profiling():
    """Verify dataset profiling contract, JSON safety, and descriptive outputs."""
    df = pd.DataFrame(
        {
            "num": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, np.nan],
            "cat": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )
    assistant = ResearchAssistant(df)
    profile = assistant.profile()

    assert profile["overview"]["total_rows"] == 8
    assert profile["overview"]["total_columns"] == 2
    assert profile["descriptive"]["num"]["count"] == 7
    assert profile["descriptive"]["num"]["mean"] == pytest.approx(4.5)

    # JSON serialization safety via ReportGenerator
    report = ReportGenerator(profile, InsightEngine(profile).get_summary())
    dumped = report.to_json()
    assert len(dumped) > 100


def test_closure_independent_two_group_means():
    """Verify independent two-group comparison (Welch t) workflow."""
    df = pd.DataFrame(
        {
            "group": ["treat"] * 10 + ["ctrl"] * 10,
            "outcome": [12.0, 14.0, 13.0, 15.0, 16.0, 14.5, 13.5, 15.5, 14.0, 16.5]
            + [8.0, 9.0, 10.0, 8.5, 9.5, 11.0, 10.5, 9.0, 8.0, 10.0],
        }
    )
    wf = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="outcome",
        predictor="group",
        design="independent",
        estimand="mean",
        variable_types={"outcome": "continuous", "group": "nominal"},
    )
    assert wf.status.value == "completed"
    assert wf.analysis.method_id == "welch_t"

    # Uncertainty: lower <= upper
    ci = wf.analysis.values["confidence_interval"]
    assert ci["lower"] <= ci["upper"]
    assert "t interval" in ci["method"] or "welch" in ci["method"].lower()

    # Effect size CI
    es = wf.analysis.values["effect_size"]
    assert "value" in es and math.isfinite(es["value"])
    assert es["confidence_interval"]["lower"] <= es["confidence_interval"]["upper"]

    # Audit & Reproducibility
    assert wf.audit.status == "passed"
    replay = reproduce(wf.reproducibility, data=df)
    assert replay.status == "reproduced"

    # Report & JSON safety
    json.dumps(wf.to_dict(), allow_nan=False)
    assert "<!doctype html>" in wf.report.to_html(style="apa")


def test_closure_multigroup_anova():
    """Verify one-way multi-group comparison (Welch ANOVA) with Games-Howell."""
    df = pd.DataFrame(
        {
            "group": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
            "y": [5, 6, 7, 6, 5, 7, 8, 6]
            + [10, 11, 12, 10, 13, 11, 12, 11]
            + [15, 16, 17, 18, 16, 15, 17, 16],
        }
    )
    wf = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="y",
        predictor="group",
        design="independent",
        estimand="mean",
        variable_types={"y": "continuous", "group": "nominal"},
    )
    assert wf.status.value == "completed"
    assert wf.analysis.method_id in ("welch_anova", "one_way_anova")
    assert len(wf.analysis.values["pairwise_comparisons"]) == 3
    assert wf.audit.status == "passed"
    json.dumps(wf.to_dict(), allow_nan=False)


def test_closure_paired_comparison():
    """Verify paired comparison (paired t) with explicit condition ordering."""
    df = pd.DataFrame(
        {
            "sub": [1, 2, 3, 4, 5, 6, 7, 8] * 2,
            "time": ["pre"] * 8 + ["post"] * 8,
            "score": [20, 22, 19, 24, 21, 23, 20, 25] + [25, 27, 24, 28, 26, 29, 24, 30],
        }
    )
    wf = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="time",
        design="paired",
        estimand="mean",
        unit_id="sub",
        condition_order=("pre", "post"),
        variable_types={"score": "continuous", "time": "nominal"},
    )
    assert wf.status.value == "completed"
    assert wf.analysis.method_id == "paired_t"
    assert wf.analysis.values["primary_estimate"] < 0  # pre - post is negative
    assert wf.audit.status == "passed"
    replay = reproduce(wf.reproducibility, data=df)
    assert replay.status == "reproduced"
    json.dumps(wf.to_dict(), allow_nan=False)


def test_closure_paired_ranks():
    """Verify paired ranks comparison (Wilcoxon signed rank)."""
    df = pd.DataFrame(
        {
            "sub": [1, 2, 3, 4, 5, 6] * 2,
            "cond": ["c1"] * 6 + ["c2"] * 6,
            "val": [10, 15, 12, 18, 14, 16] + [12, 18, 15, 22, 17, 20],
        }
    )
    wf = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="val",
        predictor="cond",
        design="paired",
        estimand="distribution",
        unit_id="sub",
        condition_order=("c1", "c2"),
        variable_types={"val": "continuous", "cond": "nominal"},
    )
    assert wf.status.value == "completed"
    assert wf.analysis.method_id == "wilcoxon_signed_rank"
    assert wf.audit.status == "passed"
    json.dumps(wf.to_dict(), allow_nan=False)


def test_closure_correlation_and_association():
    """Verify Spearman correlation and Fisher's exact test."""
    df_corr = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "y": [2.0, 5.0, 4.0, 8.0, 7.0, 11.0, 10.0, 14.0],
        }
    )
    wf_corr = ResearchAssistant(df_corr).run(
        objective="association",
        outcome="y",
        predictor="x",
        design="independent",
        estimand="monotonic",
        variable_types={"x": "continuous", "y": "continuous"},
    )
    assert wf_corr.status.value == "completed"
    assert wf_corr.analysis.method_id == "spearman_correlation"
    assert wf_corr.audit.status == "passed"

    # Fisher exact
    df_cat = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "outcome": ["yes", "no", "no", "no", "no", "no"]
            + ["yes", "yes", "yes", "yes", "no", "no"],
        }
    )
    wf_fisher = ResearchAssistant(df_cat).run(
        objective="association",
        outcome="outcome",
        predictor="group",
        design="independent",
        estimand="categorical_independence",
        variable_types={"group": "nominal", "outcome": "nominal"},
    )
    assert wf_fisher.status.value in ("completed", "partial")
    assert wf_fisher.analysis.method_id == "fisher_exact"
    assert wf_fisher.audit.status == "passed"
    json.dumps(wf_fisher.to_dict(), allow_nan=False)


def test_closure_linear_and_logistic_regression():
    """Verify linear regression with HC3 and binary logistic regression."""
    df_reg = pd.DataFrame(
        {
            "y": [3.1, 4.8, 5.9, 7.2, 8.5, 9.9, 11.0, 12.5],
            "x": [1, 2, 3, 4, 5, 6, 7, 8],
            "grp": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )
    wf_lin = ResearchAssistant(df_reg).run(
        objective="regression",
        outcome="y",
        predictors=["x", "grp"],
        design="independent",
        estimand="conditional_mean",
        reference_levels={"grp": "A"},
        covariance_type="HC3",
        variable_types={"y": "continuous", "x": "continuous", "grp": "nominal"},
    )
    assert wf_lin.status.value == "completed"
    assert wf_lin.analysis.method_id == "linear_regression"
    assert wf_lin.analysis.values["covariance_type"] == "HC3"
    assert wf_lin.audit.status == "passed"
    replay = reproduce(wf_lin.reproducibility, data=df_reg)
    assert replay.status == "reproduced"
    json.dumps(wf_lin.to_dict(), allow_nan=False)

    # Logistic regression
    df_logit = pd.DataFrame(
        {
            "y": [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    wf_logit = ResearchAssistant(df_logit).run(
        objective="regression",
        outcome="y",
        predictors=["x"],
        design="independent",
        estimand="event_probability",
        event_level=1,
        variable_types={"y": "nominal", "x": "continuous"},
    )
    assert wf_logit.status.value == "completed"
    assert wf_logit.analysis.method_id == "logistic_regression"
    assert wf_logit.audit.status == "passed"
    json.dumps(wf_logit.to_dict(), allow_nan=False)


def test_closure_scale_reliability():
    """Verify Cronbach's alpha scale reliability."""
    df_alpha = pd.DataFrame(
        {
            "item1": [1, 2, 3, 4, 5, 4, 3, 5],
            "item2": [1, 2, 2, 4, 5, 3, 4, 4],
            "item3": [2, 1, 3, 5, 4, 4, 3, 5],
        }
    )
    wf_rel = ResearchAssistant(df_alpha).reliability(
        items=["item1", "item2", "item3"], bootstrap_samples=50
    )
    assert wf_rel.status.value == "completed"
    assert wf_rel.analysis.method_id == "cronbach_alpha"
    assert math.isfinite(wf_rel.analysis.values["cronbach_alpha"])
    assert wf_rel.audit.status == "passed"
    replay = reproduce(wf_rel.reproducibility, data=df_alpha)
    assert replay.status == "reproduced"
    json.dumps(wf_rel.to_dict(), allow_nan=False)


def test_closure_repeated_measures_and_friedman():
    """Verify repeated-measures ANOVA and Friedman test."""
    panel_df = pd.DataFrame(
        {
            "participant": [1, 2, 3, 4, 5] * 3,
            "cond": ["t1"] * 5 + ["t2"] * 5 + ["t3"] * 5,
            "score": [10.0, 12.0, 11.0, 14.0, 13.0]
            + [15.0, 17.0, 16.0, 18.0, 19.0]
            + [20.0, 22.0, 21.0, 25.0, 24.0],
        }
    )
    # Repeated-measures ANOVA
    wf_rm = ResearchAssistant(panel_df).run(
        objective="compare_groups",
        outcome="score",
        predictor="cond",
        design="repeated",
        estimand="mean",
        unit_id="participant",
        condition_order=("t1", "t2", "t3"),
        variable_types={"score": "continuous", "cond": "nominal"},
    )
    assert wf_rm.status.value == "completed"
    assert wf_rm.analysis.method_id == "repeated_measures_anova"
    assert wf_rm.audit.status == "passed"
    replay = reproduce(wf_rm.reproducibility, data=panel_df)
    assert replay.status == "reproduced"
    json.dumps(wf_rm.to_dict(), allow_nan=False)

    # Friedman test
    wf_fr = ResearchAssistant(panel_df).run(
        objective="compare_groups",
        outcome="score",
        predictor="cond",
        design="repeated",
        estimand="distribution",
        unit_id="participant",
        condition_order=("t1", "t2", "t3"),
        variable_types={"score": "continuous", "cond": "nominal"},
    )
    assert wf_fr.status.value == "completed"
    assert wf_fr.analysis.method_id == "friedman_test"
    assert wf_fr.audit.status == "passed"
    replay = reproduce(wf_fr.reproducibility, data=panel_df)
    assert replay.status == "reproduced"
    json.dumps(wf_fr.to_dict(), allow_nan=False)


def test_closure_two_way_factorial_anova():
    """Verify two-way factorial ANOVA with Type II and Type III sums of squares."""
    df_tw = pd.DataFrame(
        {
            "score": [12, 14, 15, 18, 19, 22, 14, 16, 17, 21, 23, 26],
            "A": ["a1", "a1", "a1", "a1", "a1", "a1", "a2", "a2", "a2", "a2", "a2", "a2"],
            "B": ["b1", "b1", "b1", "b2", "b2", "b2", "b1", "b1", "b1", "b2", "b2", "b2"],
        }
    )
    # Type II
    wf_tw2 = ResearchAssistant(df_tw).run(
        objective="compare_groups",
        outcome="score",
        factor_a="A",
        factor_b="B",
        design="independent",
        estimand="mean",
        sum_of_squares="type2",
        variable_types={"score": "continuous", "A": "nominal", "B": "nominal"},
    )
    assert wf_tw2.status.value == "completed"
    assert wf_tw2.analysis.method_id == "two_way_anova"
    assert wf_tw2.analysis.values["sum_of_squares_type"] == "type2"
    assert wf_tw2.audit.status == "passed"
    replay = reproduce(wf_tw2.reproducibility, data=df_tw)
    assert replay.status == "reproduced"
    json.dumps(wf_tw2.to_dict(), allow_nan=False)

    # Type III
    wf_tw3 = ResearchAssistant(df_tw).run(
        objective="compare_groups",
        outcome="score",
        factor_a="A",
        factor_b="B",
        design="independent",
        estimand="mean",
        sum_of_squares="type3",
        variable_types={"score": "continuous", "A": "nominal", "B": "nominal"},
    )
    assert wf_tw3.status.value == "completed"
    assert wf_tw3.analysis.values["sum_of_squares_type"] == "type3"
    assert wf_tw3.audit.status == "passed"


def test_closure_intraclass_correlation():
    """Verify ICC reliability framework across canonical variants and reporting."""
    df_icc = pd.DataFrame(
        {
            "target": ["T1", "T1", "T2", "T2", "T3", "T3", "T4", "T4"],
            "rater": ["R1", "R2", "R1", "R2", "R1", "R2", "R1", "R2"],
            "score": [9.0, 2.0, 6.0, 1.0, 8.0, 4.0, 7.0, 2.0],
        }
    )
    wf_icc = ResearchAssistant(df_icc).intraclass_correlation(
        target="target",
        rater="rater",
        value="score",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
    )
    assert wf_icc.status.value == "completed"
    assert wf_icc.analysis.method_id == "intraclass_correlation"
    assert len(wf_icc.analysis.values["all_variants"]) == 6
    assert wf_icc.audit.status == "passed"

    # Reproducibility
    replay = reproduce(wf_icc.reproducibility, data=df_icc)
    assert replay.status == "reproduced"

    # Serialization and reporting
    json.dumps(wf_icc.to_dict(), allow_nan=False)
    tables = wf_icc.report.to_csv_tables()
    assert "icc_summary" in tables
    assert "<!doctype html>" in wf_icc.report.to_html(style="apa")
