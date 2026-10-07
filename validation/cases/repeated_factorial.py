"""Reference validation cases for repeated-measures and factorial ANOVA.

Covers:
14. repeated_measures_anova (+ Mauchly & Greenhouse-Geisser)
15. friedman_test (+ Kendall's W & Wilcoxon-Holm)
16. two_way_anova (Balanced, Type II, and Type III sums of squares)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from validation.references.repeated_factorial import (
    reference_friedman_test,
    reference_repeated_measures_anova,
    reference_two_way_anova_balanced,
    reference_two_way_anova_model,
)


def get_repeated_factorial_cases() -> list[dict[str, Any]]:
    """Return reference validation cases for repeated and factorial designs."""
    cases: list[dict[str, Any]] = []

    # =========================================================================
    # 14. REPEATED-MEASURES ANOVA
    # =========================================================================

    # Case 1: Sphericity satisfied, complete panel of 6 subjects x 3 conditions
    # Sphericity holds when variance of pairwise differences is roughly constant
    panel_rma_1 = np.array(
        [
            [10.0, 15.0, 20.0],
            [12.0, 16.0, 22.0],
            [11.0, 14.0, 21.0],
            [9.0, 13.0, 18.0],
            [13.0, 18.0, 24.0],
            [10.5, 15.5, 20.5],
        ]
    )
    cond_labels_1 = ["T1", "T2", "T3"]
    rows_rma_1 = []
    for s_idx in range(6):
        for c_idx, c_name in enumerate(cond_labels_1):
            rows_rma_1.append(
                {
                    "subject": f"S{s_idx + 1}",
                    "condition": c_name,
                    "score": panel_rma_1[s_idx, c_idx],
                }
            )
    df_rma_1 = pd.DataFrame(rows_rma_1)

    cases.append(
        {
            "case_id": "RMA_01_SPHERICITY_SATISFIED",
            "method_id": "repeated_measures_anova",
            "scientific_target": "Repeated-measures ANOVA under satisfied sphericity",
            "description": "6 complete subjects observed across 3 ordered conditions with constant difference variances",
            "source_type": "manual_formula",
            "reference_citation": "Andy Field (2012); Box (1954); Mauchly (1940)",
            "df": df_rma_1,
            "run_kwargs": {
                "unit_id": "subject",
                "condition_col": "condition",
                "value_col": "score",
                "condition_order": ("T1", "T2", "T3"),
            },
            "expected": reference_repeated_measures_anova(panel_rma_1, cond_labels_1),
            "missing_accounting": {"original_rows": 18, "analyzed_rows": 18, "excluded_rows": 0},
            "orientation": {"condition_contrast": "ordered condition trajectory T1 -> T2 -> T3"},
        }
    )

    # Case 2: Sphericity violated with incomplete subject excluded
    # Subject 7 is incomplete (missing T3)
    panel_rma_2 = np.array(
        [
            [10.0, 12.0, 30.0],
            [11.0, 15.0, 42.0],
            [9.0, 11.0, 25.0],
            [14.0, 18.0, 50.0],
            [8.0, 9.0, 20.0],
        ]
    )
    cond_labels_2 = ["Pre", "Mid", "Post"]
    rows_rma_2 = []
    for s_idx in range(5):
        for c_idx, c_name in enumerate(cond_labels_2):
            rows_rma_2.append(
                {
                    "subject": f"Sub{s_idx + 1}",
                    "condition": c_name,
                    "score": panel_rma_2[s_idx, c_idx],
                }
            )
    # Incomplete subject
    rows_rma_2.append({"subject": "Sub6_Incomplete", "condition": "Pre", "score": 10.0})
    rows_rma_2.append({"subject": "Sub6_Incomplete", "condition": "Mid", "score": 12.0})
    df_rma_2 = pd.DataFrame(rows_rma_2)

    cases.append(
        {
            "case_id": "RMA_02_SPHERICITY_VIOLATED_MISSING",
            "method_id": "repeated_measures_anova",
            "scientific_target": "Repeated-measures ANOVA with Mauchly rejection, Greenhouse-Geisser correction, and missing unit accounting",
            "description": "5 complete subjects with high condition-3 variance violating sphericity plus 1 incomplete subject",
            "source_type": "manual_formula",
            "reference_citation": "Greenhouse & Geisser (1959); complete-unit panel filtering",
            "df": df_rma_2,
            "run_kwargs": {
                "unit_id": "subject",
                "condition_col": "condition",
                "value_col": "score",
                "condition_order": ("Pre", "Mid", "Post"),
            },
            "expected": reference_repeated_measures_anova(panel_rma_2, cond_labels_2),
            "missing_accounting": {"original_rows": 17, "analyzed_rows": 15, "excluded_rows": 2},
            "orientation": {"condition_contrast": "Pre -> Mid -> Post"},
        }
    )

    # =========================================================================
    # 15. FRIEDMAN TEST
    # =========================================================================

    # Case 1: No ties, 6 complete subjects x 3 conditions
    panel_frd_1 = np.array(
        [
            [1.0, 3.0, 6.0],
            [2.0, 4.0, 5.0],
            [1.5, 2.5, 7.0],
            [3.0, 5.0, 8.0],
            [2.2, 4.2, 6.5],
            [1.8, 3.8, 7.5],
        ]
    )
    cond_frd_1 = ["CondA", "CondB", "CondC"]
    rows_frd_1 = []
    for s_idx in range(6):
        for c_idx, c_name in enumerate(cond_frd_1):
            rows_frd_1.append(
                {
                    "participant": f"P{s_idx + 1}",
                    "condition": c_name,
                    "rating": panel_frd_1[s_idx, c_idx],
                }
            )
    df_frd_1 = pd.DataFrame(rows_frd_1)

    cases.append(
        {
            "case_id": "FRD_01_NO_TIES",
            "method_id": "friedman_test",
            "scientific_target": "Friedman non-parametric test and Kendall's W concordance without ties",
            "description": "6 participants measured across 3 conditions with strictly distinct within-subject ranks",
            "source_type": "manual_formula",
            "reference_citation": "Friedman (1937); Kendall & Babington Smith (1939)",
            "df": df_frd_1,
            "run_kwargs": {
                "unit_id": "participant",
                "condition_col": "condition",
                "value_col": "rating",
                "condition_order": ("CondA", "CondB", "CondC"),
            },
            "expected": reference_friedman_test(panel_frd_1, cond_frd_1),
            "missing_accounting": {"original_rows": 18, "analyzed_rows": 18, "excluded_rows": 0},
            "orientation": {"pairwise_contrast": "CondA vs CondB vs CondC"},
        }
    )

    # Case 2: Tied ranks within participants and incomplete participant
    panel_frd_2 = np.array(
        [
            [2.0, 2.0, 5.0],
            [1.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [2.0, 4.0, 6.0],
            [3.0, 3.0, 5.0],
        ]
    )
    cond_frd_2 = ["Baseline", "Step1", "Step2"]
    rows_frd_2 = []
    for s_idx in range(5):
        for c_idx, c_name in enumerate(cond_frd_2):
            rows_frd_2.append(
                {
                    "participant": f"Part{s_idx + 1}",
                    "condition": c_name,
                    "rating": panel_frd_2[s_idx, c_idx],
                }
            )
    # Incomplete participant
    rows_frd_2.append({"participant": "Part6_Incomplete", "condition": "Baseline", "rating": 3.0})
    df_frd_2 = pd.DataFrame(rows_frd_2)

    cases.append(
        {
            "case_id": "FRD_02_TIED_MISSING",
            "method_id": "friedman_test",
            "scientific_target": "Friedman test with within-subject ties and incomplete unit exclusion",
            "description": "5 complete participants with repeated tied values exercising Kendall W tie correction, plus 1 incomplete unit",
            "source_type": "manual_formula",
            "reference_citation": "Friedman (1937); complete-unit filtering",
            "df": df_frd_2,
            "run_kwargs": {
                "unit_id": "participant",
                "condition_col": "condition",
                "value_col": "rating",
                "condition_order": ("Baseline", "Step1", "Step2"),
            },
            "expected": reference_friedman_test(panel_frd_2, cond_frd_2),
            "missing_accounting": {"original_rows": 16, "analyzed_rows": 15, "excluded_rows": 1},
            "orientation": {"pairwise_contrast": "Baseline vs Step1 vs Step2"},
        }
    )

    # =========================================================================
    # 16. TWO-WAY FACTORIAL ANOVA
    # =========================================================================

    # Case 1: Balanced 2x2 Factorial Design (4 observations per cell, total N=16)
    data_2w_1 = [
        # A1, B1 (cell mean ~10)
        ("A1", "B1", 9.0),
        ("A1", "B1", 10.0),
        ("A1", "B1", 11.0),
        ("A1", "B1", 10.0),
        # A1, B2 (cell mean ~16)
        ("A1", "B2", 15.0),
        ("A1", "B2", 16.0),
        ("A1", "B2", 17.0),
        ("A1", "B2", 16.0),
        # A2, B1 (cell mean ~20)
        ("A2", "B1", 19.0),
        ("A2", "B1", 20.0),
        ("A2", "B1", 21.0),
        ("A2", "B1", 20.0),
        # A2, B2 (cell mean ~32)
        ("A2", "B2", 31.0),
        ("A2", "B2", 32.0),
        ("A2", "B2", 33.0),
        ("A2", "B2", 32.0),
    ]
    df_2w_1 = pd.DataFrame(data_2w_1, columns=["factor_a", "factor_b", "outcome"])
    cases.append(
        {
            "case_id": "TWA_01_BALANCED",
            "method_id": "two_way_anova",
            "scientific_target": "Two-Way Factorial ANOVA under balanced orthogonal design",
            "description": "2x2 factorial with 4 observations per cell (N=16) with main effects and interaction",
            "source_type": "manual_formula",
            "reference_citation": "Fisher (1925); Yates (1934)",
            "df": df_2w_1,
            "run_kwargs": {
                "outcome": "outcome",
                "factor_a": "factor_a",
                "factor_b": "factor_b",
                "sum_of_squares": "type2",
            },
            "expected": reference_two_way_anova_balanced(data_2w_1, ["A1", "A2"], ["B1", "B2"]),
            "missing_accounting": {"original_rows": 16, "analyzed_rows": 16, "excluded_rows": 0},
            "orientation": {"design": "2x2 factorial A1/A2 by B1/B2"},
        }
    )

    # Case 2: Unbalanced 2x2 Factorial under Type II Sum of Squares
    # Cell counts: A1/B1: 5, A1/B2: 4, A2/B1: 6, A2/B2: 5 (N=20)
    data_2w_2 = [
        ("A1", "B1", 8.0),
        ("A1", "B1", 10.0),
        ("A1", "B1", 9.0),
        ("A1", "B1", 11.0),
        ("A1", "B1", 10.0),
        ("A1", "B2", 14.0),
        ("A1", "B2", 16.0),
        ("A1", "B2", 15.0),
        ("A1", "B2", 17.0),
        ("A2", "B1", 18.0),
        ("A2", "B1", 20.0),
        ("A2", "B1", 19.0),
        ("A2", "B1", 21.0),
        ("A2", "B1", 20.0),
        ("A2", "B1", 22.0),
        ("A2", "B2", 30.0),
        ("A2", "B2", 32.0),
        ("A2", "B2", 31.0),
        ("A2", "B2", 33.0),
        ("A2", "B2", 34.0),
    ]
    df_2w_2 = pd.DataFrame(data_2w_2, columns=["factor_a", "factor_b", "outcome"])
    cases.append(
        {
            "case_id": "TWA_02_TYPE2_UNBALANCED",
            "method_id": "two_way_anova",
            "scientific_target": (
                "Two-Way Factorial ANOVA under Type II sum of squares with unequal cell counts"
            ),
            "description": (
                "Unbalanced 2x2 factorial (cells 5, 4, 6, 5) evaluated using Type II"
                " reduced-model comparison"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Yates (1934); Langsrud (2003)",
            "df": df_2w_2,
            "run_kwargs": {
                "outcome": "outcome",
                "factor_a": "factor_a",
                "factor_b": "factor_b",
                "sum_of_squares": "type2",
            },
            "expected": reference_two_way_anova_model(
                data_2w_2, ["A1", "A2"], ["B1", "B2"], sum_of_squares="type2"
            ),
            "missing_accounting": {"original_rows": 20, "analyzed_rows": 20, "excluded_rows": 0},
            "orientation": {"design": "2x2 unbalanced factorial under Type II"},
        }
    )

    # Case 3: Factorial under Type III Sum of Squares (with missing rows)
    data_2w_3_raw = list(data_2w_1)
    # Add missing rows
    data_2w_3_raw.append(("A1", "B1", np.nan))
    data_2w_3_raw.append(("A2", "B2", np.nan))
    df_2w_3 = pd.DataFrame(data_2w_3_raw, columns=["factor_a", "factor_b", "outcome"])
    cases.append(
        {
            "case_id": "TWA_03_TYPE3_SUM_CONTRASTS",
            "method_id": "two_way_anova",
            "scientific_target": (
                "Two-Way ANOVA under Type III sum of squares with sum-to-zero coding and"
                " missing values"
            ),
            "description": (
                "Balanced core design plus 2 missing rows evaluated under Type III"
                " full-model Wald contrasts"
            ),
            "source_type": "manual_formula",
            "reference_citation": "Yates (1934); Searle (1987)",
            "df": df_2w_3,
            "run_kwargs": {
                "outcome": "outcome",
                "factor_a": "factor_a",
                "factor_b": "factor_b",
                "sum_of_squares": "type3",
            },
            "expected": reference_two_way_anova_model(
                data_2w_1, ["A1", "A2"], ["B1", "B2"], sum_of_squares="type3"
            ),
            "missing_accounting": {"original_rows": 18, "analyzed_rows": 16, "excluded_rows": 2},
            "orientation": {"design": "2x2 factorial under Type III"},
        }
    )

    return cases
