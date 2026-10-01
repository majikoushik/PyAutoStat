"""Numerical validation tests for two-way factorial ANOVA against reference benchmarks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pyautostat.two_way_anova import two_way_anova


@pytest.fixture
def balanced_2x2_df():
    """Balanced 2x2 factorial design with exactly known analytical sums of squares."""
    # 3 observations per cell (12 total):
    # Cell (A1, B1): 10, 11, 12 (mean = 11)
    # Cell (A1, B2): 20, 21, 22 (mean = 21)
    # Cell (A2, B1): 14, 15, 16 (mean = 15)
    # Cell (A2, B2): 30, 31, 32 (mean = 31)
    return pd.DataFrame(
        {
            "y": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 14.0, 15.0, 16.0, 30.0, 31.0, 32.0],
            "A": ["A1", "A1", "A1", "A1", "A1", "A1", "A2", "A2", "A2", "A2", "A2", "A2"],
            "B": ["B1", "B1", "B1", "B2", "B2", "B2", "B1", "B1", "B1", "B2", "B2", "B2"],
        }
    )


@pytest.fixture
def unbalanced_3x2_df():
    """Unbalanced 3x2 factorial design with 14 observations."""
    return pd.DataFrame(
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
            "A": [
                "A1",
                "A1",
                "A1",
                "A2",
                "A2",
                "A2",
                "A2",
                "A3",
                "A3",
                "A3",
                "A3",
                "A3",
                "A3",
                "A3",
            ],
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


def test_balanced_2x2_analytical_validation(balanced_2x2_df):
    """Verify analytical sums of squares, df, MS, F, and partial eta-sq in balanced design."""
    res_type2 = two_way_anova(balanced_2x2_df, "y", "A", "B", sum_of_squares="type2")
    res_type3 = two_way_anova(balanced_2x2_df, "y", "A", "B", sum_of_squares="type3")

    # In balanced orthogonal designs, Type II and Type III must be identical
    for t2, t3 in zip(res_type2["terms"], res_type3["terms"], strict=True):
        assert t2["term"] == t3["term"]
        assert pytest.approx(t2["sum_squares"], rel=1e-8) == t3["sum_squares"]
        assert t2["df"] == t3["df"]
        if t2["f_statistic"] is not None:
            assert pytest.approx(t2["f_statistic"], rel=1e-8) == t3["f_statistic"]
            assert pytest.approx(t2["p_value"], rel=1e-8) == t3["p_value"]

    # Verify analytical values
    terms = {t["term"]: t for t in res_type2["terms"]}

    term_a = terms["A"]
    term_b = terms["B"]
    term_ab = terms["A:B"]
    term_err = terms["Residual"]

    assert pytest.approx(term_a["sum_squares"], rel=1e-8) == 147.0
    assert term_a["df"] == 1
    assert pytest.approx(term_a["mean_square"], rel=1e-8) == 147.0
    assert pytest.approx(term_a["f_statistic"], rel=1e-8) == 147.0

    assert pytest.approx(term_b["sum_squares"], rel=1e-8) == 507.0
    assert term_b["df"] == 1
    assert pytest.approx(term_b["mean_square"], rel=1e-8) == 507.0
    assert pytest.approx(term_b["f_statistic"], rel=1e-8) == 507.0

    assert pytest.approx(term_ab["sum_squares"], rel=1e-8) == 27.0
    assert term_ab["df"] == 1
    assert pytest.approx(term_ab["mean_square"], rel=1e-8) == 27.0
    assert pytest.approx(term_ab["f_statistic"], rel=1e-8) == 27.0

    assert pytest.approx(term_err["sum_squares"], rel=1e-8) == 8.0
    assert term_err["df"] == 8
    assert pytest.approx(term_err["mean_square"], rel=1e-8) == 1.0

    # Orthogonal total sum of squares decomposition
    ss_total = (
        term_a["sum_squares"]
        + term_b["sum_squares"]
        + term_ab["sum_squares"]
        + term_err["sum_squares"]
    )
    y_raw = np.asarray(balanced_2x2_df["y"], dtype=float)
    expected_sst = float(np.sum((y_raw - np.mean(y_raw)) ** 2))
    assert pytest.approx(ss_total, rel=1e-8) == expected_sst

    # Partial eta-squared
    eta_a = term_a["effect_size"]["value"]
    eta_b = term_b["effect_size"]["value"]
    eta_ab = term_ab["effect_size"]["value"]

    assert pytest.approx(eta_a, rel=1e-6) == 147.0 / (147.0 + 8.0)
    assert pytest.approx(eta_b, rel=1e-6) == 507.0 / (507.0 + 8.0)
    assert pytest.approx(eta_ab, rel=1e-6) == 27.0 / (27.0 + 8.0)

    # Confidence intervals for partial eta-squared contain point estimates
    for t in [term_a, term_b, term_ab]:
        ci = t["effect_size"]["confidence_interval"]
        assert 0.0 <= ci["lower"] <= t["effect_size"]["value"] <= ci["upper"] < 1.0


def test_unbalanced_3x2_reference_validation(unbalanced_3x2_df):
    """Verify unbalanced 3x2 Type II and Type III against statsmodels and R car::Anova."""
    res_type2 = two_way_anova(unbalanced_3x2_df, "y", "A", "B", sum_of_squares="type2")
    res_type3 = two_way_anova(unbalanced_3x2_df, "y", "A", "B", sum_of_squares="type3")

    terms2 = {t["term"]: t for t in res_type2["terms"]}
    terms3 = {t["term"]: t for t in res_type3["terms"]}

    # Residuals are identical across Type II and Type III
    assert pytest.approx(terms2["Residual"]["sum_squares"], rel=1e-6) == 48.166667
    assert pytest.approx(terms3["Residual"]["sum_squares"], rel=1e-6) == 48.166667
    assert terms2["Residual"]["df"] == 8

    # Interaction term is identical across Type II and Type III
    assert pytest.approx(terms2["A:B"]["sum_squares"], rel=1e-5) == 22.593897
    assert pytest.approx(terms3["A:B"]["sum_squares"], rel=1e-5) == 22.593897
    assert pytest.approx(terms2["A:B"]["f_statistic"], rel=1e-5) == 1.876310
    assert pytest.approx(terms3["A:B"]["f_statistic"], rel=1e-5) == 1.876310
    assert pytest.approx(terms2["A:B"]["p_value"], rel=1e-4) == 0.214695

    # Type II Main Effects (conditional on other main effect, without interaction)
    assert pytest.approx(terms2["A"]["sum_squares"], rel=1e-5) == 141.810865
    assert terms2["A"]["df"] == 2
    assert pytest.approx(terms2["A"]["f_statistic"], rel=1e-5) == 11.776681
    assert pytest.approx(terms2["A"]["p_value"], rel=1e-4) == 0.004132

    assert pytest.approx(terms2["B"]["sum_squares"], rel=1e-5) == 33.953722
    assert terms2["B"]["df"] == 1
    assert pytest.approx(terms2["B"]["f_statistic"], rel=1e-5) == 5.639373
    assert pytest.approx(terms2["B"]["p_value"], rel=1e-4) == 0.044913

    # Type III Main Effects (sum-to-zero contrasts, conditional on all terms including AxB)
    assert pytest.approx(terms3["A"]["sum_squares"], rel=1e-5) == 137.241784
    assert terms3["A"]["df"] == 2
    assert pytest.approx(terms3["A"]["f_statistic"], rel=1e-5) == 11.397242
    assert pytest.approx(terms3["A"]["p_value"], rel=1e-4) == 0.004555

    assert pytest.approx(terms3["B"]["sum_squares"], rel=1e-5) == 16.657658
    assert terms3["B"]["df"] == 1
    assert pytest.approx(terms3["B"]["f_statistic"], rel=1e-5) == 2.766670
    assert pytest.approx(terms3["B"]["p_value"], rel=1e-4) == 0.134817


def test_factor_ordering_symmetry(unbalanced_3x2_df):
    """Factor declaring order (A, B vs B, A) yields symmetric results."""
    res_ab = two_way_anova(unbalanced_3x2_df, "y", "A", "B", sum_of_squares="type3")
    res_ba = two_way_anova(unbalanced_3x2_df, "y", "B", "A", sum_of_squares="type3")

    terms_ab = {t["term"]: t for t in res_ab["terms"]}
    terms_ba = {t["term"]: t for t in res_ba["terms"]}

    # Term A
    assert pytest.approx(terms_ab["A"]["sum_squares"], rel=1e-8) == terms_ba["A"]["sum_squares"]
    assert pytest.approx(terms_ab["A"]["f_statistic"], rel=1e-8) == terms_ba["A"]["f_statistic"]

    # Term B
    assert pytest.approx(terms_ab["B"]["sum_squares"], rel=1e-8) == terms_ba["B"]["sum_squares"]
    assert pytest.approx(terms_ab["B"]["f_statistic"], rel=1e-8) == terms_ba["B"]["f_statistic"]

    # Interaction
    term_ab_int = terms_ab["A:B"]
    term_ba_int = terms_ba["B:A"]
    assert pytest.approx(term_ab_int["sum_squares"], rel=1e-8) == term_ba_int["sum_squares"]
    assert pytest.approx(term_ab_int["f_statistic"], rel=1e-8) == term_ba_int["f_statistic"]


def test_json_serializability(unbalanced_3x2_df):
    """Two-way ANOVA output must be strictly JSON-serializable with no NaN or infinite values."""
    res2 = two_way_anova(unbalanced_3x2_df, "y", "A", "B", sum_of_squares="type2")
    res3 = two_way_anova(unbalanced_3x2_df, "y", "A", "B", sum_of_squares="type3")

    json_str2 = json.dumps(res2, allow_nan=False)
    json_str3 = json.dumps(res3, allow_nan=False)

    assert len(json_str2) > 0
    assert len(json_str3) > 0

    reloaded2 = json.loads(json_str2)
    reloaded3 = json.loads(json_str3)

    assert reloaded2["method"] == "two_way_anova"
    assert reloaded3["method"] == "two_way_anova"
