"""Tests for two-way factorial ANOVA follow-up contrasts and multiplicity adjustment."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pyautostat.two_way_anova import two_way_anova


@pytest.fixture
def clean_2x2_df():
    """A 2x2 factorial design with 10 rows per cell and known population means."""
    rng = np.random.default_rng(101)
    rows = []
    # Cell means: A1:B1=10, A1:B2=20 (diff= -10), A2:B1=15, A2:B2=35 (diff= -20)
    # Difference of differences: (-10) - (-20) = +10
    means = {
        ("A1", "B1"): 10.0,
        ("A1", "B2"): 20.0,
        ("A2", "B1"): 15.0,
        ("A2", "B2"): 35.0,
    }
    for (a, b), mu in means.items():
        for _ in range(10):
            rows.append({"a": a, "b": b, "y": float(mu + rng.normal(0, 1.0))})
    return pd.DataFrame(rows)


@pytest.fixture
def clean_3x2_df():
    """A 3x2 factorial design with 6 rows per cell."""
    rng = np.random.default_rng(202)
    rows = []
    for a in ["A1", "A2", "A3"]:
        for b in ["B1", "B2"]:
            base = 10.0 if a == "A1" else (20.0 if a == "A2" else 30.0)
            b_eff = 5.0 if b == "B2" else 0.0
            for _ in range(6):
                rows.append({"a": a, "b": b, "y": float(base + b_eff + rng.normal(0, 1.2))})
    return pd.DataFrame(rows)


def test_simple_effects_structure_and_counts(clean_3x2_df):
    """In a 3x2 design, simple effects and marginal comparisons have expected counts."""
    res = two_way_anova(clean_3x2_df, "y", "a", "b")
    followups = res["followups"]

    families = {f["family_type"] for f in followups}
    assert families == {
        "simple_a_within_b",
        "simple_b_within_a",
        "marginal_a",
        "marginal_b",
    }

    # Factor A has 3 levels -> 3 pairwise contrasts within each of 2 levels of B: 3 * 2 = 6
    simple_a = [f for f in followups if f["family_type"] == "simple_a_within_b"]
    assert len(simple_a) == 6

    # Factor B has 2 levels -> 1 pairwise contrast within each of 3 levels of A: 1 * 3 = 3
    simple_b = [f for f in followups if f["family_type"] == "simple_b_within_a"]
    assert len(simple_b) == 3

    # Marginal A: 3 comparisons
    marg_a = [f for f in followups if f["family_type"] == "marginal_a"]
    assert len(marg_a) == 3

    # Marginal B: 1 comparison
    marg_b = [f for f in followups if f["family_type"] == "marginal_b"]
    assert len(marg_b) == 1


def test_2x2_difference_of_differences_math(clean_2x2_df):
    """For a 2x2 design, t^2 for difference-of-differences equals interaction F statistic."""
    res = two_way_anova(clean_2x2_df, "y", "a", "b")

    dod = res["diff_of_diff"]
    assert dod is not None
    assert dod["family_type"] == "interaction_contrast"
    assert dod["family"] == "2x2 interaction difference-of-differences"

    # Find interaction term
    term_ab = next(t for t in res["terms"] if t["term_type"] == "interaction")
    f_stat = term_ab["f_statistic"]

    # t-statistic squared must equal F statistic of interaction in 2x2 factorial ANOVA!
    t_stat = dod["statistic"]
    assert pytest.approx(t_stat**2, rel=1e-6) == f_stat

    # Two-sided p-value must match interaction p-value
    assert pytest.approx(dod["raw_p_value"], rel=1e-6) == term_ab["p_value"]

    # Check SE equals sqrt(MSE * sum(1/n_i)) = sqrt(MSE * 4 / 10)
    mse = res["diagnostics"]["residual_mean_square"]
    expected_se = math.sqrt(mse * (1.0 / 10 + 1.0 / 10 + 1.0 / 10 + 1.0 / 10))
    assert pytest.approx(dod["standard_error"], rel=1e-8) == expected_se

    # Check CI bounds are symmetric around estimate
    ci = dod["confidence_interval"]
    assert ci["lower"] < dod["estimate"] < ci["upper"]
    half_width_low = dod["estimate"] - ci["lower"]
    half_width_high = ci["upper"] - dod["estimate"]
    assert pytest.approx(half_width_low, rel=1e-6) == half_width_high


def test_3x2_design_has_no_2x2_diff_of_diff(clean_3x2_df):
    """A 3x2 design does not have a single 2x2 difference-of-differences contrast."""
    res = two_way_anova(clean_3x2_df, "y", "a", "b")
    assert res["diff_of_diff"] is None


def test_holm_adjustment_invariants(clean_3x2_df):
    """Holm adjustment must preserve monotonicity, adjusted p >= raw p, and bounds in [0, 1]."""
    res = two_way_anova(clean_3x2_df, "y", "a", "b")
    followups = res["followups"]

    for family in ["simple_a_within_b", "simple_b_within_a", "marginal_a", "marginal_b"]:
        group = [f for f in followups if f["family_type"] == family]
        if not group:
            continue
        # Raw p vs adjusted p
        for f in group:
            raw_p = f["raw_p_value"]
            adj_p = f["adjusted_p_value"]
            assert 0.0 <= raw_p <= 1.0
            assert 0.0 <= adj_p <= 1.0
            assert adj_p >= raw_p or pytest.approx(adj_p, abs=1e-9) == raw_p

        # Monotonicity check
        sorted_by_raw = sorted(group, key=lambda x: x["raw_p_value"])
        adj_vals = [f["adjusted_p_value"] for f in sorted_by_raw]
        for i in range(len(adj_vals) - 1):
            assert adj_vals[i] <= adj_vals[i + 1] + 1e-9


def test_pointwise_ci_multiplicity_flag(clean_2x2_df):
    """Pointwise Student-t confidence intervals must have multiplicity_adjusted=False."""
    res = two_way_anova(clean_2x2_df, "y", "a", "b")
    for f in res["followups"]:
        ci = f["confidence_interval"]
        assert ci["status"] == "available"
        assert ci["multiplicity_adjusted"] is False
        assert ci["lower"] <= f["estimate"] <= ci["upper"]


def test_contrast_orientation_and_estimate(clean_2x2_df):
    """Contrast estimate must equal first level mean minus second level mean."""
    res = two_way_anova(clean_2x2_df, "y", "a", "b")
    cell_means = {(c["factor_a"], c["factor_b"]): c["mean"] for c in res["cell_summaries"]}

    # Test simple contrast of a within b='B1'
    simple_b1 = next(
        f
        for f in res["followups"]
        if f["family_type"] == "simple_a_within_b" and f.get("conditioning_level") == "B1"
    )
    expected_diff = cell_means[("A1", "B1")] - cell_means[("A2", "B1")]
    assert pytest.approx(simple_b1["estimate"], rel=1e-8) == expected_diff
