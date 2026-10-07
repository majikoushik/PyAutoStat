"""Foundational reference validation cases for PyAutoStat (Methods 1-10)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from validation.references.foundational import (
    reference_chi_square,
    reference_fisher_exact,
    reference_mann_whitney_u,
    reference_one_sample_t,
    reference_paired_t,
    reference_pearson_correlation,
    reference_spearman_correlation,
    reference_student_t,
    reference_welch_t,
    reference_wilcoxon_signed_rank,
)


def get_foundational_cases() -> list[dict[str, Any]]:
    """Return the complete inventory of foundational reference validation cases."""
    """Return the complete inventory of reference validation cases."""
    cases: list[dict[str, Any]] = []

    # =========================================================================
    # 1. ONE-SAMPLE T-TEST
    # =========================================================================

    # Case 1: Standard positive difference
    x_1 = [12.0, 15.0, 14.0, 18.0, 16.0, 19.0, 17.0, 13.0]
    df_ost_1 = pd.DataFrame({"score": x_1})
    ref_1 = 10.0
    cases.append(
        {
            "case_id": "OST_01_BASIC_POS",
            "method_id": "one_sample_t",
            "scientific_target": (
                "One-sample mean comparison against reference (positive difference)"
            ),
            "description": (
                "8 observations tested against reference value 10.0 (sample mean 15.5 > 10.0)"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Student (1908); independent manual formulas",
            "df": df_ost_1,
            "run_kwargs": {
                "value_col": "score",
                "reference_value": ref_1,
                "confidence_level": 0.95,
            },
            "expected": reference_one_sample_t(x_1, ref_1, confidence_level=0.95),
            "missing_accounting": {"original_rows": 8, "analyzed_rows": 8, "excluded_rows": 0},
            "orientation": {"sample_mean_minus_reference": "positive"},
        }
    )

    # Case 2: Negative difference with missing values
    x_2_raw = [5.0, 4.0, np.nan, 6.0, 3.0, 5.0, 4.0, np.nan, 5.0]
    x_2_clean = [v for v in x_2_raw if not np.isnan(v)]
    df_ost_2 = pd.DataFrame({"score": x_2_raw})
    ref_2 = 8.0
    cases.append(
        {
            "case_id": "OST_02_MISSING_NEG",
            "method_id": "one_sample_t",
            "scientific_target": (
                "One-sample mean comparison with missing data (negative difference)"
            ),
            "description": (
                "9 rows (2 missing) tested against reference value 8.0 (sample mean ~4.57 < 8.0)"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Student (1908); complete-case sample accounting",
            "df": df_ost_2,
            "run_kwargs": {
                "value_col": "score",
                "reference_value": ref_2,
                "confidence_level": 0.95,
            },
            "expected": reference_one_sample_t(x_2_clean, ref_2, confidence_level=0.95),
            "missing_accounting": {"original_rows": 9, "analyzed_rows": 7, "excluded_rows": 2},
            "orientation": {"sample_mean_minus_reference": "negative"},
        }
    )

    # =========================================================================
    # 2. STUDENT'S TWO-SAMPLE T-TEST
    # =========================================================================

    # Case 1: Standard equal variance, positive difference
    g1_st1 = [20.0, 22.0, 19.0, 24.0, 21.0, 23.0]
    g2_st1 = [14.0, 16.0, 15.0, 18.0, 13.0, 17.0]
    df_st_1 = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "val": g1_st1 + g2_st1,
        }
    )
    cases.append(
        {
            "case_id": "STT_01_EQUAL_VAR_POS",
            "method_id": "student_t",
            "scientific_target": (
                "Two-sample Student's t-test with equal variance (positive difference)"
            ),
            "description": (
                "Two balanced groups (n1=6, n2=6) with equal spread; group A mean > group B mean"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Student (1908); pooled variance formula",
            "df": df_st_1,
            "run_kwargs": {
                "group_col": "group",
                "value_col": "val",
                "equal_var": True,
                "confidence_level": 0.95,
            },
            "expected": reference_student_t(g1_st1, g2_st1, confidence_level=0.95),
            "missing_accounting": {"original_rows": 12, "analyzed_rows": 12, "excluded_rows": 0},
            "orientation": {"first_minus_second": "positive"},
        }
    )

    # Case 2: Negative difference with missing rows and unbalanced n
    g1_st2_raw = [10.0, 12.0, np.nan, 11.0, 13.0, 10.0]
    g2_st2_raw = [18.0, 16.0, 19.0, np.nan, 17.0, 20.0, 18.0]
    g1_st2 = [v for v in g1_st2_raw if not np.isnan(v)]
    g2_st2 = [v for v in g2_st2_raw if not np.isnan(v)]
    df_st_2 = pd.DataFrame(
        {
            "group": ["T1"] * len(g1_st2_raw) + ["T2"] * len(g2_st2_raw),
            "val": g1_st2_raw + g2_st2_raw,
        }
    )
    cases.append(
        {
            "case_id": "STT_02_MISSING_NEG",
            "method_id": "student_t",
            "scientific_target": (
                "Two-sample Student's t-test with missing data (negative difference)"
            ),
            "description": (
                "Unbalanced groups with missing values (n1=5, n2=6 analyzed; 2 excluded); T1 < T2"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Student (1908); pooled variance with complete-case filtering",
            "df": df_st_2,
            "run_kwargs": {
                "group_col": "group",
                "value_col": "val",
                "equal_var": True,
                "confidence_level": 0.95,
            },
            "expected": reference_student_t(g1_st2, g2_st2, confidence_level=0.95),
            "missing_accounting": {"original_rows": 13, "analyzed_rows": 11, "excluded_rows": 2},
            "orientation": {"first_minus_second": "negative"},
        }
    )

    # =========================================================================
    # 3. WELCH'S TWO-SAMPLE T-TEST
    # =========================================================================

    # Case 1: Unequal variances, positive difference
    g1_w1 = [35.0, 42.0, 28.0, 50.0, 38.0, 45.0, 40.0]
    g2_w1 = [22.0, 24.0, 21.0, 25.0, 23.0]
    df_w_1 = pd.DataFrame(
        {
            "group": ["G1"] * len(g1_w1) + ["G2"] * len(g2_w1),
            "val": g1_w1 + g2_w1,
        }
    )
    cases.append(
        {
            "case_id": "WEL_01_UNEQUAL_VAR_POS",
            "method_id": "welch_t",
            "scientific_target": "Welch's t-test with unequal variances (positive difference)",
            "description": (
                "Heteroscedastic groups (n1=7 with large variance, "
                "n2=5 with small variance); G1 > G2"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Welch (1947); Welch-Satterthwaite approximation",
            "df": df_w_1,
            "run_kwargs": {
                "group_col": "group",
                "value_col": "val",
                "equal_var": False,
                "confidence_level": 0.95,
            },
            "expected": reference_welch_t(g1_w1, g2_w1, confidence_level=0.95),
            "missing_accounting": {"original_rows": 12, "analyzed_rows": 12, "excluded_rows": 0},
            "orientation": {"first_minus_second": "positive"},
        }
    )

    # Case 2: Welch t-test with missing data, negative difference
    g1_w2_raw = [15.0, np.nan, 14.0, 16.0, 13.0, 15.0]
    g2_w2_raw = [28.0, 32.0, np.nan, 35.0, 30.0, 34.0, 29.0]
    g1_w2 = [v for v in g1_w2_raw if not np.isnan(v)]
    g2_w2 = [v for v in g2_w2_raw if not np.isnan(v)]
    df_w_2 = pd.DataFrame(
        {
            "group": ["A"] * len(g1_w2_raw) + ["B"] * len(g2_w2_raw),
            "val": g1_w2_raw + g2_w2_raw,
        }
    )
    cases.append(
        {
            "case_id": "WEL_02_MISSING_NEG",
            "method_id": "welch_t",
            "scientific_target": "Welch's t-test with missing data (negative difference)",
            "description": (
                "Unbalanced heteroscedastic groups with missing rows (n1=5, n2=6 analyzed); A < B"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Welch (1947); complete-case Welch-Satterthwaite",
            "df": df_w_2,
            "run_kwargs": {
                "group_col": "group",
                "value_col": "val",
                "equal_var": False,
                "confidence_level": 0.95,
            },
            "expected": reference_welch_t(g1_w2, g2_w2, confidence_level=0.95),
            "missing_accounting": {"original_rows": 13, "analyzed_rows": 11, "excluded_rows": 2},
            "orientation": {"first_minus_second": "negative"},
        }
    )

    # =========================================================================
    # 4. PAIRED T-TEST
    # =========================================================================

    # Case 1: Standard paired positive difference
    u_p1 = [f"U{i}" for i in range(1, 8)]
    pre_p1 = [18.0, 22.0, 15.0, 20.0, 19.0, 24.0, 17.0]
    post_p1 = [14.0, 18.0, 13.0, 15.0, 16.0, 19.0, 14.0]
    df_p_1 = pd.DataFrame(
        {
            "unit": u_p1 * 2,
            "condition": ["Pre"] * 7 + ["Post"] * 7,
            "val": pre_p1 + post_p1,
        }
    )
    cases.append(
        {
            "case_id": "PAI_01_COMPLETE_POS",
            "method_id": "paired_t",
            "scientific_target": (
                "Paired t-test on complete pairs (positive paired difference Pre - Post)"
            ),
            "description": (
                "7 complete pairs, Pre minus Post is consistently positive (mean diff > 0)"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Fisher (1925); paired difference mean and SE",
            "df": df_p_1,
            "run_kwargs": {
                "unit_id": "unit",
                "condition_col": "condition",
                "value_col": "val",
                "condition_order": ("Pre", "Post"),
                "confidence_level": 0.95,
            },
            "expected": reference_paired_t(pre_p1, post_p1, confidence_level=0.95),
            "missing_accounting": {
                "original_rows": 14,
                "analyzed_rows": 14,
                "complete_pairs": 7,
                "excluded_rows": 0,
            },
            "orientation": {"first_condition_minus_second": "positive"},
        }
    )

    # Case 2: Incomplete pairs / missing condition (negative difference)
    # Unit 1-6 complete; Unit 7 missing Post; Unit 8 missing Pre
    u_p2_data = [
        {"unit": "U1", "condition": "Base", "val": 10.0},
        {"unit": "U1", "condition": "Treat", "val": 15.0},
        {"unit": "U2", "condition": "Base", "val": 12.0},
        {"unit": "U2", "condition": "Treat", "val": 18.0},
        {"unit": "U3", "condition": "Base", "val": 11.0},
        {"unit": "U3", "condition": "Treat", "val": 16.0},
        {"unit": "U4", "condition": "Base", "val": 14.0},
        {"unit": "U4", "condition": "Treat", "val": 19.0},
        {"unit": "U5", "condition": "Base", "val": 13.0},
        {"unit": "U5", "condition": "Treat", "val": 17.0},
        {"unit": "U6", "condition": "Base", "val": 12.0},
        {"unit": "U6", "condition": "Treat", "val": 18.0},
        # Incomplete units
        {"unit": "U7", "condition": "Base", "val": 15.0},
        {"unit": "U8", "condition": "Treat", "val": 22.0},
    ]
    df_p_2 = pd.DataFrame(u_p2_data)
    first_p2 = [10.0, 12.0, 11.0, 14.0, 13.0, 12.0]
    second_p2 = [15.0, 18.0, 16.0, 19.0, 17.0, 18.0]
    cases.append(
        {
            "case_id": "PAI_02_INCOMPLETE_NEG",
            "method_id": "paired_t",
            "scientific_target": (
                "Paired t-test with incomplete units (negative difference Base - Treat)"
            ),
            "description": (
                "8 units total (6 complete pairs, 2 incomplete excluded); Base - Treat < 0"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Fisher (1925); paired unit-matching exclusion rule",
            "df": df_p_2,
            "run_kwargs": {
                "unit_id": "unit",
                "condition_col": "condition",
                "value_col": "val",
                "condition_order": ("Base", "Treat"),
                "confidence_level": 0.95,
            },
            "expected": reference_paired_t(first_p2, second_p2, confidence_level=0.95),
            "missing_accounting": {
                "original_rows": 14,
                "analyzed_rows": 12,
                "complete_pairs": 6,
                "excluded_rows": 2,
                "incomplete_units": 2,
            },
            "orientation": {"first_condition_minus_second": "negative"},
        }
    )

    # =========================================================================
    # 5. MANN-WHITNEY U TEST
    # =========================================================================

    # Case 1: No ties, positive rank-biserial
    g1_mw1 = [12.0, 15.0, 18.0, 22.0, 25.0]
    g2_mw1 = [5.0, 8.0, 11.0, 14.0, 17.0]
    df_mw_1 = pd.DataFrame(
        {
            "group": ["A"] * len(g1_mw1) + ["B"] * len(g2_mw1),
            "val": g1_mw1 + g2_mw1,
        }
    )
    cases.append(
        {
            "case_id": "MWU_01_NO_TIES_POS",
            "method_id": "mann_whitney_u",
            "scientific_target": (
                "Mann-Whitney U with continuous distinct values (no ties, positive correlation)"
            ),
            "description": (
                "Two groups of 5 with no tied observations; group A values tend to exceed group B"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Mann & Whitney (1947); independent rank-sum formula",
            "df": df_mw_1,
            "run_kwargs": {
                "group_col": "group",
                "value_col": "val",
                "estimand": "distribution",
            },
            "expected": reference_mann_whitney_u(g1_mw1, g2_mw1),
            "missing_accounting": {"original_rows": 10, "analyzed_rows": 10, "excluded_rows": 0},
            "orientation": {"rank_biserial": "positive"},
        }
    )

    # Case 2: Tied ranks, negative rank-biserial, missing row
    g1_mw2_raw = [10.0, 12.0, 12.0, 15.0, np.nan, 15.0]
    g2_mw2_raw = [15.0, 18.0, 18.0, 20.0, 22.0, 25.0]
    g1_mw2 = [v for v in g1_mw2_raw if not np.isnan(v)]
    g2_mw2 = [v for v in g2_mw2_raw if not np.isnan(v)]
    df_mw_2 = pd.DataFrame(
        {
            "group": ["Low"] * len(g1_mw2_raw) + ["High"] * len(g2_mw2_raw),
            "val": g1_mw2_raw + g2_mw2_raw,
        }
    )
    cases.append(
        {
            "case_id": "MWU_02_TIED_RANKS_NEG",
            "method_id": "mann_whitney_u",
            "scientific_target": (
                "Mann-Whitney U with tied values and missing row (negative rank-biserial)"
            ),
            "description": (
                "Tied observations at 12.0, 15.0, 18.0; 1 missing row excluded; Low < High"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Mann & Whitney (1947); average rank assignment for ties",
            "df": df_mw_2,
            "run_kwargs": {
                "group_col": "group",
                "value_col": "val",
                "estimand": "distribution",
            },
            "expected": reference_mann_whitney_u(g1_mw2, g2_mw2),
            "missing_accounting": {"original_rows": 12, "analyzed_rows": 11, "excluded_rows": 1},
            "orientation": {"rank_biserial": "negative"},
        }
    )

    # =========================================================================
    # 6. WILCOXON SIGNED-RANK TEST
    # =========================================================================

    # Case 1: No zero differences, positive rank-biserial
    u_wx1 = [f"P{i}" for i in range(1, 7)]
    cond1_wx1 = [25.0, 30.0, 28.0, 35.0, 40.0, 32.0]
    cond2_wx1 = [18.0, 22.0, 24.0, 28.0, 31.0, 29.0]
    df_wx_1 = pd.DataFrame(
        {
            "unit": u_wx1 * 2,
            "cond": ["A"] * 6 + ["B"] * 6,
            "val": cond1_wx1 + cond2_wx1,
        }
    )
    cases.append(
        {
            "case_id": "WSR_01_NO_ZEROS_POS",
            "method_id": "wilcoxon_signed_rank",
            "scientific_target": (
                "Wilcoxon signed-rank test without zero differences (positive effect)"
            ),
            "description": "6 matched pairs where A > B across all pairs; no zero differences",
            "source_type": "manual_formula",
            "reference_citation": "Wilcoxon (1945); signed-rank sum and rank-biserial",
            "df": df_wx_1,
            "run_kwargs": {
                "unit_id": "unit",
                "condition_col": "cond",
                "value_col": "val",
                "condition_order": ("A", "B"),
            },
            "expected": reference_wilcoxon_signed_rank(cond1_wx1, cond2_wx1),
            "missing_accounting": {
                "original_rows": 12,
                "analyzed_rows": 12,
                "complete_pairs": 6,
                "excluded_rows": 0,
            },
            "orientation": {"rank_biserial": "positive"},
        }
    )

    # Case 2: Zero differences + tied differences (wilcox zero policy)
    # Pairs with some equal values: A - B includes 0s and ties
    u_wx2 = [f"S{i}" for i in range(1, 9)]
    cond1_wx2 = [15.0, 20.0, 18.0, 22.0, 14.0, 25.0, 19.0, 30.0]
    cond2_wx2 = [15.0, 24.0, 18.0, 26.0, 18.0, 21.0, 23.0, 28.0]
    # diffs: 0, -4, 0, -4, -4, +4, -4, +2
    # nonzero diffs: -4, -4, -4, +4, -4, +2
    df_wx_2 = pd.DataFrame(
        {
            "unit": u_wx2 * 2,
            "cond": ["Pre"] * 8 + ["Post"] * 8,
            "val": cond1_wx2 + cond2_wx2,
        }
    )
    cases.append(
        {
            "case_id": "WSR_02_ZEROS_AND_TIES",
            "method_id": "wilcoxon_signed_rank",
            "scientific_target": (
                "Wilcoxon signed-rank test with zero differences and ties (wilcox policy)"
            ),
            "description": (
                "8 pairs: 2 zero differences omitted from ranking; "
                "5 ties at |d|=4; Pre < Post overall"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Wilcoxon (1945); wilcox zero-difference exclusion policy",
            "df": df_wx_2,
            "run_kwargs": {
                "unit_id": "unit",
                "condition_col": "cond",
                "value_col": "val",
                "condition_order": ("Pre", "Post"),
            },
            "expected": reference_wilcoxon_signed_rank(cond1_wx2, cond2_wx2),
            "missing_accounting": {
                "original_rows": 16,
                "analyzed_rows": 16,
                "complete_pairs": 8,
                "excluded_rows": 0,
                "zero_differences": 2,
            },
            "orientation": {"rank_biserial": "negative"},
        }
    )

    # =========================================================================
    # 7. PEARSON CORRELATION
    # =========================================================================

    # Case 1: Positive linear correlation
    x_pc1 = [10.0, 12.0, 15.0, 18.0, 22.0, 25.0, 29.0, 32.0]
    y_pc1 = [22.0, 26.0, 31.0, 38.0, 44.0, 52.0, 58.0, 65.0]
    df_pc_1 = pd.DataFrame({"x": x_pc1, "y": y_pc1})
    cases.append(
        {
            "case_id": "PC_01_POSITIVE_LINEAR",
            "method_id": "pearson_correlation",
            "scientific_target": "Pearson correlation coefficient (positive linear association)",
            "description": "8 paired points with strong positive linear association; Fisher-z CI",
            "source_type": "manual_formula",
            "reference_citation": "Pearson (1895); Fisher (1921) z-transformation",
            "df": df_pc_1,
            "run_kwargs": {"outcome_col": "x", "predictor_col": "y", "confidence_level": 0.95},
            "expected": reference_pearson_correlation(x_pc1, y_pc1, confidence_level=0.95),
            "missing_accounting": {"original_rows": 8, "analyzed_rows": 8, "excluded_rows": 0},
            "orientation": {"pearson_r": "positive"},
        }
    )

    # Case 2: Negative linear correlation with missing data
    x_pc2_raw = [5.0, 8.0, np.nan, 12.0, 15.0, 18.0, 22.0, 25.0, np.nan]
    y_pc2_raw = [45.0, 40.0, 38.0, 32.0, 28.0, 22.0, 18.0, 12.0, np.nan]
    # Filter complete pairs
    pairs_pc2 = [
        (a, b)
        for a, b in zip(x_pc2_raw, y_pc2_raw, strict=True)
        if not (np.isnan(a) or np.isnan(b))
    ]
    x_pc2_clean = [p[0] for p in pairs_pc2]
    y_pc2_clean = [p[1] for p in pairs_pc2]
    df_pc_2 = pd.DataFrame({"x": x_pc2_raw, "y": y_pc2_raw})
    cases.append(
        {
            "case_id": "PC_02_NEGATIVE_MISSING",
            "method_id": "pearson_correlation",
            "scientific_target": (
                "Pearson correlation with missing pairs (negative linear association)"
            ),
            "description": "9 rows (2 missing pairs excluded, 7 complete); negative linear slope",
            "source_type": "manual_formula",
            "reference_citation": "Pearson (1895); complete-pair filtering",
            "df": df_pc_2,
            "run_kwargs": {"outcome_col": "x", "predictor_col": "y", "confidence_level": 0.95},
            "expected": reference_pearson_correlation(
                x_pc2_clean, y_pc2_clean, confidence_level=0.95
            ),
            "missing_accounting": {"original_rows": 9, "analyzed_rows": 7, "excluded_rows": 2},
            "orientation": {"pearson_r": "negative"},
        }
    )

    # =========================================================================
    # 8. SPEARMAN CORRELATION
    # =========================================================================

    # Case 1: Monotonic relationship without ties
    x_sc1 = [2.0, 5.0, 7.0, 9.0, 15.0, 20.0, 30.0]
    y_sc1 = [1.0, 4.0, 10.0, 18.0, 45.0, 90.0, 250.0]  # strictly monotonic nonlinear
    df_sc_1 = pd.DataFrame({"u": x_sc1, "v": y_sc1})
    cases.append(
        {
            "case_id": "SC_01_MONOTONIC_NO_TIES",
            "method_id": "spearman_correlation",
            "scientific_target": (
                "Spearman rank correlation for strictly monotonic nonlinear relationship"
            ),
            "description": "7 points with perfect monotonic order (rho = 1.0); no ties",
            "source_type": "manual_formula",
            "reference_citation": "Spearman (1904); Pearson correlation of ranks",
            "df": df_sc_1,
            "run_kwargs": {"outcome_col": "u", "predictor_col": "v"},
            "expected": reference_spearman_correlation(x_sc1, y_sc1),
            "missing_accounting": {"original_rows": 7, "analyzed_rows": 7, "excluded_rows": 0},
            "orientation": {"spearman_rho": "positive"},
        }
    )

    # Case 2: Negative rank correlation with tied values and missing pair
    x_sc2_raw = [10.0, 15.0, 15.0, 20.0, np.nan, 25.0, 30.0, 35.0]
    y_sc2_raw = [50.0, 45.0, 45.0, 38.0, 40.0, 30.0, 25.0, 20.0]
    pairs_sc2 = [
        (a, b)
        for a, b in zip(x_sc2_raw, y_sc2_raw, strict=True)
        if not (np.isnan(a) or np.isnan(b))
    ]
    x_sc2_clean = [p[0] for p in pairs_sc2]
    y_sc2_clean = [p[1] for p in pairs_sc2]
    df_sc_2 = pd.DataFrame({"score1": x_sc2_raw, "score2": y_sc2_raw})
    cases.append(
        {
            "case_id": "SC_02_TIED_NEGATIVE_MISSING",
            "method_id": "spearman_correlation",
            "scientific_target": "Spearman rank correlation with tied ranks and missing pair",
            "description": (
                "8 rows (1 missing pair, 7 analyzed); "
                "ties at 15.0 and 45.0; negative rank correlation"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Spearman (1904); average rank assignment for ties",
            "df": df_sc_2,
            "run_kwargs": {"outcome_col": "score1", "predictor_col": "score2"},
            "expected": reference_spearman_correlation(x_sc2_clean, y_sc2_clean),
            "missing_accounting": {"original_rows": 8, "analyzed_rows": 7, "excluded_rows": 1},
            "orientation": {"spearman_rho": "negative"},
        }
    )

    # =========================================================================
    # 9. PEARSON CHI-SQUARE TEST OF INDEPENDENCE
    # =========================================================================

    # Case 1: 2x2 contingency table (all expected >= 5)
    # Table:
    #             Success   Failure
    # Treatment   25        15       (row sum 40)
    # Control     15        25       (row sum 40)
    # Col sum     40        40       Total N = 80
    # Expected: all 20.0 >= 5
    table_chi1 = [[25, 15], [15, 25]]
    # Expand to a DataFrame of 80 rows
    rows_chi1 = []
    for _ in range(25):
        rows_chi1.append({"arm": "Treatment", "outcome": "Success"})
    for _ in range(15):
        rows_chi1.append({"arm": "Treatment", "outcome": "Failure"})
    for _ in range(15):
        rows_chi1.append({"arm": "Control", "outcome": "Success"})
    for _ in range(25):
        rows_chi1.append({"arm": "Control", "outcome": "Failure"})
    df_chi_1 = pd.DataFrame(rows_chi1)
    cases.append(
        {
            "case_id": "CHI_01_2X2_TABLE",
            "method_id": "pearson_chi_square",
            "scientific_target": (
                "Pearson chi-square test of independence (2x2 table, all expected >= 5)"
            ),
            "description": "80 observations in a 2x2 table; expected counts = 20.0 in all cells",
            "source_type": "manual_formula",
            "reference_citation": "Pearson (1900); uncorrected chi-square and Cramer's V",
            "df": df_chi_1,
            "run_kwargs": {"group_col": "arm", "outcome_col": "outcome"},
            "expected": reference_chi_square(table_chi1),
            "missing_accounting": {"original_rows": 80, "analyzed_rows": 80, "excluded_rows": 0},
            "orientation": {"contingency_dimension": "2x2"},
        }
    )

    # Case 2: 3x2 RxC table with missing rows (all expected >= 5)
    # Table:
    #         High   Low
    # GrpA    20     10    (30)
    # GrpB    15     15    (30)
    # GrpC    10     20    (30)
    # Total:  45     45    (90)
    # Expected: all 15.0 >= 5
    table_chi2 = [[20, 10], [15, 15], [10, 20]]
    rows_chi2 = []
    for _ in range(20):
        rows_chi2.append({"site": "GrpA", "level": "High"})
    for _ in range(10):
        rows_chi2.append({"site": "GrpA", "level": "Low"})
    for _ in range(15):
        rows_chi2.append({"site": "GrpB", "level": "High"})
    for _ in range(15):
        rows_chi2.append({"site": "GrpB", "level": "Low"})
    for _ in range(10):
        rows_chi2.append({"site": "GrpC", "level": "High"})
    for _ in range(20):
        rows_chi2.append({"site": "GrpC", "level": "Low"})
    # Add missing rows
    rows_chi2.append({"site": "GrpA", "level": None})
    rows_chi2.append({"site": None, "level": "High"})
    df_chi_2 = pd.DataFrame(rows_chi2)
    cases.append(
        {
            "case_id": "CHI_02_3X2_RXC_MISSING",
            "method_id": "pearson_chi_square",
            "scientific_target": (
                "Pearson chi-square test for RxC contingency table with missing rows"
            ),
            "description": (
                "92 rows (2 missing excluded, 90 analyzed); 3x2 table; all expected = 15.0 >= 5"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Pearson (1900); df = (r-1)(c-1) = 2",
            "df": df_chi_2,
            "run_kwargs": {"group_col": "site", "outcome_col": "level"},
            "expected": reference_chi_square(table_chi2),
            "missing_accounting": {"original_rows": 92, "analyzed_rows": 90, "excluded_rows": 2},
            "orientation": {"contingency_dimension": "3x2"},
        }
    )

    # =========================================================================
    # 10. FISHER'S EXACT TEST
    # =========================================================================

    # Case 1: Standard 2x2 contingency table
    # Table:
    #         Recovered   NotRecovered
    # Drug    12          3             (row sum 15)
    # Placebo 4           11            (row sum 15)
    # Odds ratio: (12 * 11) / (3 * 4) = 132 / 12 = 11.0
    table_fe1 = [[12, 3], [4, 11]]
    rows_fe1 = []
    for _ in range(12):
        rows_fe1.append({"drug": "Active", "status": "Recovered"})
    for _ in range(3):
        rows_fe1.append({"drug": "Active", "status": "NotRecovered"})
    for _ in range(4):
        rows_fe1.append({"drug": "Placebo", "status": "Recovered"})
    for _ in range(11):
        rows_fe1.append({"drug": "Placebo", "status": "NotRecovered"})
    df_fe_1 = pd.DataFrame(rows_fe1)
    cases.append(
        {
            "case_id": "FET_01_STANDARD_2X2",
            "method_id": "fisher_exact",
            "scientific_target": "Fisher's exact test for 2x2 table (unconditional odds ratio)",
            "description": "30 observations across 2x2 table with large odds ratio (OR = 11.0)",
            "source_type": "manual_formula",
            "reference_citation": "Fisher (1935); cross-product sample odds ratio",
            "df": df_fe_1,
            "run_kwargs": {"row_variable": "drug", "column_variable": "status"},
            "expected": reference_fisher_exact(table_fe1),
            "missing_accounting": {"original_rows": 30, "analyzed_rows": 30, "excluded_rows": 0},
            "orientation": {"odds_ratio": 11.0},
        }
    )

    # Case 2: 2x2 table with missing rows and odds ratio < 1.0
    # Table:
    #         Yes   No
    # Group1  3     9   (12)
    # Group2  8     4   (12)
    # Odds ratio: (3 * 4) / (9 * 8) = 12 / 72 = 0.16666666666666666
    table_fe2 = [[3, 9], [8, 4]]
    rows_fe2 = []
    for _ in range(3):
        rows_fe2.append({"exp": "E1", "resp": "R1"})
    for _ in range(9):
        rows_fe2.append({"exp": "E1", "resp": "R2"})
    for _ in range(8):
        rows_fe2.append({"exp": "E2", "resp": "R1"})
    for _ in range(4):
        rows_fe2.append({"exp": "E2", "resp": "R2"})
    # Missing rows
    rows_fe2.append({"exp": "E1", "resp": None})
    rows_fe2.append({"exp": None, "resp": "R1"})
    df_fe_2 = pd.DataFrame(rows_fe2)
    cases.append(
        {
            "case_id": "FET_02_MISSING_OR_LT1",
            "method_id": "fisher_exact",
            "scientific_target": "Fisher's exact test with missing values (odds ratio < 1.0)",
            "description": "26 rows (2 missing excluded, 24 analyzed); OR = 12/72 = 1/6 (~0.167)",
            "source_type": "manual_formula",
            "reference_citation": "Fisher (1935); complete-case sample odds ratio",
            "df": df_fe_2,
            "run_kwargs": {"row_variable": "exp", "column_variable": "resp"},
            "expected": reference_fisher_exact(table_fe2),
            "missing_accounting": {"original_rows": 26, "analyzed_rows": 24, "excluded_rows": 2},
            "orientation": {"odds_ratio": 1.0 / 6.0},
        }
    )

    return cases
