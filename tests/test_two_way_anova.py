"""Tests for two-way factorial ANOVA core engine, recommendations, and execution."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pyautostat import (
    ColumnNotFoundError,
    InsufficientDataError,
    InvalidDataError,
    Objective,
    ResearchAssistant,
    StatisticalAnalyzer,
    StudyDesign,
)
from pyautostat.question_builder import QuestionStatus, prepare_question
from pyautostat.two_way_anova import two_way_anova


@pytest.fixture
def balanced_2x2_df():
    """A balanced 2x2 factorial dataset with 10 observations per cell (N = 40)."""
    rng = np.random.default_rng(42)
    rows = []
    # Cell true means: A1:B1=10, A1:B2=15, A2:B1=20, A2:B2=35 (super-additive interaction)
    effects = {
        ("A1", "B1"): 10.0,
        ("A1", "B2"): 15.0,
        ("A2", "B1"): 20.0,
        ("A2", "B2"): 35.0,
    }
    for (a, b), mu in effects.items():
        for _ in range(10):
            val = float(mu + rng.normal(0, 2.0))
            rows.append({"factor_a": a, "factor_b": b, "outcome": val})
    return pd.DataFrame(rows)


@pytest.fixture
def balanced_3x2_df():
    """A balanced 3x2 factorial dataset with 5 observations per cell (N = 30)."""
    rng = np.random.default_rng(123)
    rows = []
    levels_a = ["ctrl", "low", "high"]
    levels_b = ["diet_std", "diet_keto"]
    for a in levels_a:
        for b in levels_b:
            base = 50.0 + (10.0 if a == "high" else (5.0 if a == "low" else 0.0))
            diet_eff = 8.0 if b == "diet_keto" else 0.0
            inter = 5.0 if (a == "high" and b == "diet_keto") else 0.0
            for _ in range(5):
                rows.append(
                    {
                        "dose": a,
                        "diet": b,
                        "biomarker": float(base + diet_eff + inter + rng.normal(0, 1.5)),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def unbalanced_2x2_df():
    """An unbalanced 2x2 factorial dataset with varying cell sizes (8, 12, 10, 14)."""
    rng = np.random.default_rng(999)
    rows = []
    cell_specs = [
        ("Placebo", "Low", 8, 10.0),
        ("Placebo", "High", 12, 14.0),
        ("Active", "Low", 10, 18.0),
        ("Active", "High", 14, 28.0),
    ]
    for drug, dose, count, mean in cell_specs:
        for _ in range(count):
            rows.append(
                {
                    "treatment": drug,
                    "dosage": dose,
                    "response": float(mean + rng.normal(0, 2.5)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core Engine Tests
# ---------------------------------------------------------------------------


def test_two_way_anova_balanced_type2_type3_equality(balanced_2x2_df):
    """On balanced designs, Type II and Type III SS must be numerically identical."""
    res_type2 = two_way_anova(
        balanced_2x2_df, "outcome", "factor_a", "factor_b", sum_of_squares="type2"
    )
    res_type3 = two_way_anova(
        balanced_2x2_df, "outcome", "factor_a", "factor_b", sum_of_squares="type3"
    )

    terms2 = {t["term"]: t for t in res_type2["terms"]}
    terms3 = {t["term"]: t for t in res_type3["terms"]}

    for term_name in ["factor_a", "factor_b", "factor_a:factor_b", "Residual"]:
        assert term_name in terms2
        assert term_name in terms3
        t2 = terms2[term_name]
        t3 = terms3[term_name]
        assert pytest.approx(t2["sum_squares"], rel=1e-8) == t3["sum_squares"]
        assert t2["df"] == t3["df"]
        assert pytest.approx(t2["mean_square"], rel=1e-8) == t3["mean_square"]
        if term_name != "Residual":
            assert pytest.approx(t2["f_statistic"], rel=1e-8) == t3["f_statistic"]
            assert pytest.approx(t2["p_value"], rel=1e-8) == t3["p_value"]
            eff2 = t2["effect_size"]["value"]
            eff3 = t3["effect_size"]["value"]
            assert pytest.approx(eff2, rel=1e-8) == eff3


def test_two_way_anova_unbalanced_type2_vs_type3(unbalanced_2x2_df):
    """On unbalanced designs, Type II and Type III main effects differ while interaction matches."""
    res_type2 = two_way_anova(
        unbalanced_2x2_df, "response", "treatment", "dosage", sum_of_squares="type2"
    )
    res_type3 = two_way_anova(
        unbalanced_2x2_df, "response", "treatment", "dosage", sum_of_squares="type3"
    )

    t2 = {t["term"]: t for t in res_type2["terms"]}
    t3 = {t["term"]: t for t in res_type3["terms"]}

    # Interaction and Residual SS must match exactly between Type II and Type III
    assert (
        pytest.approx(t2["treatment:dosage"]["sum_squares"], rel=1e-8)
        == (t3["treatment:dosage"]["sum_squares"])
    )
    assert pytest.approx(t2["Residual"]["sum_squares"], rel=1e-8) == (t3["Residual"]["sum_squares"])

    # In unbalanced designs, main effect sums of squares generally differ
    # (Type II does not adjust for interaction; Type III adjusts for interaction under sum-to-zero)
    # Both must be positive and have 1 df
    assert t2["treatment"]["sum_squares"] > 0
    assert t3["treatment"]["sum_squares"] > 0
    assert t2["treatment"]["df"] == 1
    assert t3["treatment"]["df"] == 1


def test_two_way_anova_direct_analyzer_api(balanced_3x2_df):
    """StatisticalAnalyzer.two_way_anova exposes complete result structure."""
    analyzer = StatisticalAnalyzer(balanced_3x2_df)
    res = analyzer.two_way_anova("biomarker", "dose", "diet", sum_of_squares="type2")

    assert res["method"] == "two_way_anova"
    assert res["sum_of_squares_type"] == "type2"
    assert res["sample_size"] == 30
    assert len(res["terms"]) == 4

    terms = {t["term"]: t for t in res["terms"]}
    assert terms["dose"]["df"] == 2
    assert terms["diet"]["df"] == 1
    assert terms["dose:diet"]["df"] == 2
    assert terms["Residual"]["df"] == 24

    # Check cell summaries
    assert len(res["cell_summaries"]) == 6
    for cell in res["cell_summaries"]:
        assert cell["sample_size"] == 5
        assert np.isfinite(cell["mean"])
        assert np.isfinite(cell["standard_deviation"])

    # Check estimated marginal means
    assert len(res["estimated_marginal_means"]["dose"]) == 3
    assert len(res["estimated_marginal_means"]["diet"]) == 2

    # Check diagnostics
    diag = res["diagnostics"]
    assert diag["residual_sample_size"] == 30
    assert diag["residual_degrees_of_freedom"] == 24
    assert diag["residual_sum_of_squares"] > 0
    assert diag["rmse"] > 0
    assert "normality" in diag
    assert "homoscedasticity" in diag


def test_two_way_anova_factor_order_invariance(balanced_2x2_df):
    """Swapping factor_a and factor_b yields identical term statistics."""
    res_ab = two_way_anova(balanced_2x2_df, "outcome", "factor_a", "factor_b")
    res_ba = two_way_anova(balanced_2x2_df, "outcome", "factor_b", "factor_a")

    t_ab = {t["term"]: t for t in res_ab["terms"]}
    t_ba = {t["term"]: t for t in res_ba["terms"]}

    # Main effects match
    assert (
        pytest.approx(t_ab["factor_a"]["sum_squares"], rel=1e-8)
        == (t_ba["factor_a"]["sum_squares"])
    )
    assert (
        pytest.approx(t_ab["factor_b"]["sum_squares"], rel=1e-8)
        == (t_ba["factor_b"]["sum_squares"])
    )
    # Interaction SS matches
    assert (
        pytest.approx(t_ab["factor_a:factor_b"]["sum_squares"], rel=1e-8)
        == (t_ba["factor_b:factor_a"]["sum_squares"])
    )
    # Residual SS matches
    assert (
        pytest.approx(t_ab["Residual"]["sum_squares"], rel=1e-8)
        == (t_ba["Residual"]["sum_squares"])
    )


def test_two_way_anova_missing_data_exclusion(balanced_2x2_df):
    """Missing rows in outcome, factor_a, or factor_b are excluded from complete-case analysis."""
    df_missing = balanced_2x2_df.copy()
    df_missing.loc[0, "outcome"] = np.nan
    df_missing.loc[1, "factor_a"] = None
    df_missing.loc[2, "factor_b"] = np.nan

    res = two_way_anova(df_missing, "outcome", "factor_a", "factor_b")
    assert res["sample_size"] == 37
    assert res["original_rows"] == 40
    assert res["excluded_rows"] == 3


def test_two_way_anova_column_names_with_spaces_and_special_chars():
    """Columns with spaces and special characters are handled correctly without collisions."""
    rng = np.random.default_rng(77)
    df = pd.DataFrame(
        {
            "Patient Outcome (pts)": rng.normal(50, 5, 20),
            "Drug Treatment / Arm": ["A", "B"] * 10,
            "Time Point [hr]": ["T1", "T1", "T2", "T2"] * 5,
        }
    )
    res = two_way_anova(
        df,
        outcome="Patient Outcome (pts)",
        factor_a="Drug Treatment / Arm",
        factor_b="Time Point [hr]",
    )
    assert res["sample_size"] == 20
    assert len(res["terms"]) == 4


# ---------------------------------------------------------------------------
# Degenerate Cases & Safeguards
# ---------------------------------------------------------------------------


def test_two_way_anova_column_not_found(balanced_2x2_df):
    """Requesting nonexistent outcome or factor raises ColumnNotFoundError."""
    with pytest.raises(ColumnNotFoundError):
        two_way_anova(balanced_2x2_df, "nonexistent", "factor_a", "factor_b")
    with pytest.raises(ColumnNotFoundError):
        two_way_anova(balanced_2x2_df, "outcome", "nonexistent", "factor_b")
    with pytest.raises(ColumnNotFoundError):
        two_way_anova(balanced_2x2_df, "outcome", "factor_a", "nonexistent")


def test_two_way_anova_fewer_than_two_levels():
    """A factor with only 1 level raises InvalidDataError."""
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "a": ["only_level", "only_level", "only_level", "only_level"],
            "b": ["B1", "B2", "B1", "B2"],
        }
    )
    with pytest.raises(InvalidDataError, match="at least 2 observed levels"):
        two_way_anova(df, "y", "a", "b")


def test_two_way_anova_empty_cell_raises_insufficient_data():
    """An empty cell in the factorial grid raises InsufficientDataError."""
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "a": ["A1", "A1", "A1", "A2", "A2"],
            "b": ["B1", "B1", "B2", "B1", "B1"],  # Cell (A2, B2) is missing!
        }
    )
    with pytest.raises(InsufficientDataError, match="Empty cell"):
        two_way_anova(df, "y", "a", "b")


def test_two_way_anova_saturated_model_zero_residual_df():
    """Having 1 observation per cell yields zero residual df and raises InsufficientDataError."""
    df = pd.DataFrame(
        {
            "y": [10.0, 15.0, 20.0, 25.0],
            "a": ["A1", "A1", "A2", "A2"],
            "b": ["B1", "B2", "B1", "B2"],
        }
    )
    with pytest.raises(InsufficientDataError, match="Residual degrees of freedom must be positive"):
        two_way_anova(df, "y", "a", "b")


def test_two_way_anova_constant_outcome_raises():
    """A constant outcome with zero variance raises InsufficientDataError."""
    df = pd.DataFrame(
        {
            "y": [5.0] * 8,
            "a": ["A1", "A1", "A1", "A1", "A2", "A2", "A2", "A2"],
            "b": ["B1", "B1", "B2", "B2", "B1", "B1", "B2", "B2"],
        }
    )
    with pytest.raises(InsufficientDataError, match="is constant"):
        two_way_anova(df, "y", "a", "b")


def test_two_way_anova_invalid_ss_type(balanced_2x2_df):
    """Unsupported SS type raises InvalidDataError."""
    with pytest.raises(InvalidDataError, match="sum_of_squares must be"):
        two_way_anova(balanced_2x2_df, "outcome", "factor_a", "factor_b", sum_of_squares="type1")


# ---------------------------------------------------------------------------
# Research Assistant & Guided Workflow Tests
# ---------------------------------------------------------------------------


def test_assistant_two_way_anova_method(balanced_2x2_df):
    """ResearchAssistant.two_way_anova executes the full research workflow."""
    ra = ResearchAssistant(balanced_2x2_df)
    res = ra.two_way_anova("outcome", "factor_a", "factor_b", sum_of_squares="type3")

    assert res.analysis.method_id == "two_way_anova"
    assert res.analysis.status.value == "available"
    assert res.audit.status == "passed"
    assert len(res.audit.findings) == 0
    assert res.report is not None

    terms = res.analysis.values["terms"]
    assert len(terms) == 4


def test_assistant_run_compare_groups_two_factors(balanced_3x2_df):
    """ResearchAssistant.run with two factors recommends and executes two_way_anova."""
    ra = ResearchAssistant(balanced_3x2_df)
    res = ra.run(
        objective=Objective.COMPARE_GROUPS,
        outcome="biomarker",
        factor_a="dose",
        factor_b="diet",
        design=StudyDesign.INDEPENDENT,
        estimand="mean",
    )
    assert res.recommendation.method_id == "two_way_anova"
    assert res.recommendation.status.value == "ready"
    assert res.analysis.method_id == "two_way_anova"
    assert res.analysis.status.value == "available"


def test_prepare_question_with_two_factors(balanced_2x2_df):
    """prepare_question correctly configures factors and sum_of_squares."""
    prep = prepare_question(
        balanced_2x2_df,
        objective=Objective.COMPARE_GROUPS,
        outcome="outcome",
        factor_a="factor_a",
        factor_b="factor_b",
        design=StudyDesign.INDEPENDENT,
        estimand="mean",
        sum_of_squares="type3",
    )
    assert prep.status == QuestionStatus.READY
    assert prep.specification.question.factor_a == "factor_a"
    assert prep.specification.question.factor_b == "factor_b"
    assert prep.specification.question.factors == ("factor_a", "factor_b")
    assert prep.specification.options.sum_of_squares == "type3"


def test_two_way_anova_json_serializable(balanced_2x2_df):
    """All structured outputs must strictly pass json.dumps with allow_nan=False."""
    res = two_way_anova(balanced_2x2_df, "outcome", "factor_a", "factor_b")
    dumped = json.dumps(res, allow_nan=False)
    assert dumped is not None
    reloaded = json.loads(dumped)
    assert reloaded["method"] == "two_way_anova"
