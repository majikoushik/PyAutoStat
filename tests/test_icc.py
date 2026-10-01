"""Tests for Intraclass Correlation Coefficient (ICC) canonical implementations.

Covers:
- Variant resolution (Shrout-Fleiss and McGraw-Wong mappings)
- Panel extraction, deterministic sorting, and complete-target filtering
- Missing data accounting and exclusion reporting
- ANOVA mean squares decomposition and method-of-moments variance components
- All 6 canonical ICC variants
- Non-clamping of negative sample ICC
- Input validation and degenerate data handling
- Direct API via StatisticalAnalyzer and ResearchAssistant
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyautostat import ResearchAssistant, StatisticalAnalyzer
from pyautostat.exceptions import ColumnNotFoundError, InsufficientDataError, InvalidDataError
from pyautostat.icc import (
    ICC_VARIANTS,
    icc_anova_components,
    icc_panel,
    intraclass_correlation,
    resolve_variant,
)


@pytest.fixture
def sf_table4_df() -> pd.DataFrame:
    """Shrout & Fleiss (1979) Table 4 data in long format.

    6 targets, 4 judges/raters.
    Target 1: [9, 2, 5, 8]
    Target 2: [6, 1, 3, 2]
    Target 3: [8, 4, 6, 8]
    Target 4: [7, 1, 2, 6]
    Target 5: [10, 5, 6, 9]
    Target 6: [6, 2, 4, 7]
    """
    targets = [1, 2, 3, 4, 5, 6]
    ratings = [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ]
    rows = []
    for t_idx, t in enumerate(targets):
        for j_idx, score in enumerate(ratings[t_idx]):
            rows.append({"target": f"T{t}", "rater": f"J{j_idx + 1}", "score": float(score)})
    return pd.DataFrame(rows)


@pytest.fixture
def panel_with_missing_df(sf_table4_df: pd.DataFrame) -> pd.DataFrame:
    """Panel with incomplete targets and missing cells."""
    df = sf_table4_df.copy()
    # Add an incomplete target with only 2 raters
    incomplete_rows = pd.DataFrame(
        [
            {"target": "T7", "rater": "J1", "score": 7.0},
            {"target": "T7", "rater": "J2", "score": 5.0},
            # Missing J3 and J4 for T7
            {"target": "T8", "rater": "J1", "score": np.nan},  # NaN rating
            {"target": "T8", "rater": "J2", "score": 4.0},
            {"target": "T8", "rater": "J3", "score": 5.0},
            {"target": "T8", "rater": "J4", "score": 6.0},
        ]
    )
    return pd.concat([df, incomplete_rows], ignore_index=True)


# ── Variant Resolution Tests ──────────────────────────────────────────────────


def test_resolve_variant_all_canonical():
    assert resolve_variant("one_way_random", "absolute_agreement", "single") == "icc_1_1"
    assert resolve_variant("one_way_random", None, "single") == "icc_1_1"
    assert resolve_variant("one_way_random", "absolute_agreement", "average") == "icc_1_k"
    assert resolve_variant("one_way_random", None, "average") == "icc_1_k"

    assert resolve_variant("two_way_random", "absolute_agreement", "single") == "icc_2_1"
    assert resolve_variant("two_way_random", "absolute_agreement", "average") == "icc_2_k"
    assert resolve_variant("two_way_random", "consistency", "single") == "icc_2_1_consistency"
    assert resolve_variant("two_way_random", "consistency", "average") == "icc_2_k_consistency"

    assert resolve_variant("two_way_mixed", "consistency", "single") == "icc_3_1"
    assert resolve_variant("two_way_mixed", "consistency", "average") == "icc_3_k"
    assert resolve_variant("two_way_mixed", "absolute_agreement", "single") == "icc_3_1_agreement"
    assert resolve_variant("two_way_mixed", "absolute_agreement", "average") == "icc_3_k_agreement"


def test_resolve_variant_invalid():
    with pytest.raises(InvalidDataError, match="model must be one of"):
        resolve_variant("three_way_random", "absolute_agreement", "single")

    with pytest.raises(InvalidDataError, match="definition must be one of"):
        resolve_variant("two_way_random", "approximate", "single")

    with pytest.raises(InvalidDataError, match="unit must be one of"):
        resolve_variant("two_way_random", "absolute_agreement", "median")

    with pytest.raises(InvalidDataError, match="One-way random model evaluates undifferentiated"):
        resolve_variant("one_way_random", "consistency", "single")


# ── Panel Construction & Validation Tests ─────────────────────────────────────


def test_icc_panel_complete(sf_table4_df: pd.DataFrame):
    panel = icc_panel(sf_table4_df, "target", "rater", "score")
    assert panel["n_targets"] == 6
    assert panel["n_raters"] == 4
    assert panel["matrix"].shape == (6, 4)
    assert panel["sample"]["original_rows"] == 24
    assert panel["sample"]["analyzed_rows"] == 24
    assert panel["sample"]["excluded_rows"] == 0
    assert panel["sample"]["excluded_targets"] == 0
    assert len(panel["targets"]) == 6
    assert len(panel["raters"]) == 4


def test_icc_panel_complete_target_filtering(panel_with_missing_df: pd.DataFrame):
    panel = icc_panel(panel_with_missing_df, "target", "rater", "score")
    # T7 (only 2 raters) and T8 (NaN rating on J1) must be excluded
    assert panel["n_targets"] == 6
    assert panel["n_raters"] == 4
    assert panel["sample"]["excluded_targets"] == 2
    assert panel["sample"]["excluded_rows"] == 6
    assert panel["sample"]["analyzed_rows"] == 24


def test_icc_panel_duplicate_detection(sf_table4_df: pd.DataFrame):
    dup_df = pd.concat([sf_table4_df, sf_table4_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(InsufficientDataError, match="Duplicate observations detected"):
        icc_panel(dup_df, "target", "rater", "score")


def test_icc_panel_insufficient_targets():
    df = pd.DataFrame(
        {
            "target": ["T1", "T1"],
            "rater": ["R1", "R2"],
            "score": [1.0, 2.0],
        }
    )
    with pytest.raises(InsufficientDataError, match="requires at least 2 targets"):
        icc_panel(df, "target", "rater", "score")


def test_icc_panel_insufficient_raters():
    df = pd.DataFrame(
        {
            "target": ["T1", "T2"],
            "rater": ["R1", "R1"],
            "score": [1.0, 2.0],
        }
    )
    with pytest.raises(InsufficientDataError, match="requires at least 2 raters"):
        icc_panel(df, "target", "rater", "score")


def test_icc_panel_invalid_column_names(sf_table4_df: pd.DataFrame):
    with pytest.raises(InvalidDataError, match="must be a non-empty string"):
        icc_panel(sf_table4_df, "", "rater", "score")

    with pytest.raises(InvalidDataError, match="must specify three different columns"):
        icc_panel(sf_table4_df, "target", "target", "score")

    with pytest.raises(ColumnNotFoundError, match="Column 'nonexistent' not found"):
        icc_panel(sf_table4_df, "nonexistent", "rater", "score")


def test_icc_panel_non_numeric_outcome():
    df = pd.DataFrame(
        {
            "target": ["T1", "T1", "T2", "T2"],
            "rater": ["R1", "R2", "R1", "R2"],
            "score": ["high", "low", "medium", "high"],
        }
    )
    with pytest.raises(InsufficientDataError, match="requires at least 2 complete targets"):
        icc_panel(df, "target", "rater", "score")


# ── ANOVA Components & Formulas ───────────────────────────────────────────────


def test_icc_anova_components(sf_table4_df: pd.DataFrame):
    panel = icc_panel(sf_table4_df, "target", "rater", "score")
    anova = icc_anova_components(panel["matrix"])

    # Basic sum of squares partition: SS_total = SS_targets + SS_raters + SS_error
    assert anova["ss_total"] == pytest.approx(
        anova["ss_targets"] + anova["ss_raters"] + anova["ss_error"], rel=1e-8
    )
    # Degrees of freedom partition
    assert anova["df_total"] == anova["df_targets"] + anova["df_raters"] + anova["df_error"]
    assert anova["df_targets"] == 5
    assert anova["df_raters"] == 3
    assert anova["df_error"] == 15
    assert anova["df_total"] == 23

    # Mean squares: MS = SS / df
    assert anova["ms_targets"] == pytest.approx(anova["ss_targets"] / anova["df_targets"])
    assert anova["ms_raters"] == pytest.approx(anova["ss_raters"] / anova["df_raters"])
    assert anova["ms_error"] == pytest.approx(anova["ss_error"] / anova["df_error"])

    # One-way components
    assert anova["ss_within"] == pytest.approx(anova["ss_raters"] + anova["ss_error"])
    assert anova["df_within"] == 18
    assert anova["ms_within"] == pytest.approx(anova["ss_within"] / anova["df_within"])

    # Variance components
    vc = anova["variance_components"]
    assert vc["residual_variance"] == anova["ms_error"]
    assert vc["target_variance"] == pytest.approx((anova["ms_targets"] - anova["ms_error"]) / 4)
    assert vc["rater_variance"] == pytest.approx((anova["ms_raters"] - anova["ms_error"]) / 6)


def test_constant_ratings_rejected():
    matrix = np.full((5, 3), 4.0)
    with pytest.raises(InvalidDataError, match="Total variance across all ratings is zero"):
        icc_anova_components(matrix)


# ── All 6 Canonical Variants Execution ────────────────────────────────────────


def test_all_six_canonical_variants_execution(sf_table4_df: pd.DataFrame):
    for variant_key in ("icc_1_1", "icc_1_k", "icc_2_1", "icc_2_k", "icc_3_1", "icc_3_k"):
        var_info = ICC_VARIANTS[variant_key]
        result = intraclass_correlation(
            sf_table4_df,
            target="target",
            rater="rater",
            outcome="score",
            model=var_info["model"],
            definition=var_info["definition"],
            unit=var_info["unit"],
        )
        assert result["method_id"] == "intraclass_correlation"
        assert result["variant"] == variant_key
        assert result["notation"] == var_info["notation"]
        assert np.isfinite(result["estimate"])
        ci = result["confidence_interval"]
        assert ci["status"] == "available"
        assert ci["lower"] <= ci["upper"]
        f_test = result["f_test"]
        assert f_test["statistic"] > 0
        assert 0 <= f_test["p_value"] <= 1
        assert len(result["all_variants"]) == 6


# ── Negative ICC Preserved (Never Clamped) ─────────────────────────────────────


def test_negative_icc_preserved_not_clamped():
    """Construct data where within-target disagreement exceeds between-target variance."""
    rows = []
    ratings = [
        [2.0, 10.0, 1.0, 7.0],
        [8.0, 3.0, 9.0, 4.0],
        [1.0, 9.0, 3.0, 7.0],
        [10.0, 2.0, 8.0, 4.0],
    ]
    for t_idx, row in enumerate(ratings):
        for r_idx, val in enumerate(row):
            rows.append({"target": f"T{t_idx}", "rater": f"R{r_idx}", "score": val})
    neg_df = pd.DataFrame(rows)

    result = intraclass_correlation(
        neg_df,
        target="target",
        rater="rater",
        outcome="score",
        model="one_way_random",
        unit="single",
    )
    assert result["estimate"] < 0.0
    # Must NOT be clamped to 0
    assert result["estimate"] != 0.0
    # Warning must be issued explaining negative estimate
    assert any("negative" in w.lower() for w in result["warnings"])


# ── StatisticalAnalyzer and ResearchAssistant APIs ────────────────────────────


def test_statistical_analyzer_icc_direct(sf_table4_df: pd.DataFrame):
    analyzer = StatisticalAnalyzer(sf_table4_df)
    res1 = analyzer.intraclass_correlation(
        target="target",
        rater="rater",
        value="score",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
    )
    assert res1["method_id"] == "intraclass_correlation"
    assert res1["variant"] == "icc_2_1"

    # Convenient alias .icc(...)
    res2 = analyzer.icc(
        target="target",
        rater="rater",
        value="score",
        model="two_way_mixed",
        definition="consistency",
        unit="single",
    )
    assert res2["method_id"] == "intraclass_correlation"
    assert res2["variant"] == "icc_3_1"


def test_research_assistant_icc_direct(sf_table4_df: pd.DataFrame):
    assistant = ResearchAssistant(sf_table4_df)
    workflow = assistant.intraclass_correlation(
        target="target",
        rater="rater",
        value="score",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
    )
    assert workflow.status.value == "completed"
    assert workflow.analysis.method_id == "intraclass_correlation"
    assert workflow.analysis.values["variant"] == "icc_2_1"

    # Alias .icc(...)
    workflow_alias = assistant.icc(
        target="target",
        rater="rater",
        value="score",
        model="two_way_mixed",
        definition="consistency",
        unit="average",
    )
    assert workflow_alias.status.value == "completed"
    assert workflow_alias.analysis.values["variant"] == "icc_3_k"
