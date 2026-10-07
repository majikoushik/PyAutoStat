"""Reference validation cases for extended association and paired categorical methods.

Covers:
19. kendall_tau_b (monotonic association with/without ties)
20. point_biserial_correlation (binary vs continuous)
21. partial_pearson_correlation (controlling for covariates)
22. mcnemar (paired nominal proportions)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from validation.references.association_categorical import (
    reference_kendall_tau_b,
    reference_mcnemar_exact,
    reference_partial_pearson,
    reference_point_biserial,
)


def get_association_categorical_cases() -> list[dict[str, Any]]:
    """Return reference validation cases for association and paired categorical methods."""
    cases: list[dict[str, Any]] = []

    # =========================================================================
    # 19. KENDALL TAU-B
    # =========================================================================

    # Case 1: Monotonic negative association without ties (N=8)
    x_ktb_1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y_ktb_1 = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    df_ktb_1 = pd.DataFrame({"score_x": x_ktb_1, "score_y": y_ktb_1})

    cases.append(
        {
            "case_id": "KTB_01_NO_TIES",
            "method_id": "kendall_tau_b",
            "scientific_target": "Kendall's tau-b rank association without ties",
            "description": "8 paired continuous observations with strictly inverse ordering (tau-b = -1.0)",
            "source_type": "manual_formula",
            "reference_citation": "Kendall (1938); pairwise concordance counting",
            "df": df_ktb_1,
            "run_kwargs": {
                "objective": "association",
                "outcome": "score_x",
                "predictor": "score_y",
                "design": "independent",
                "estimand": "monotonic",
                "association_measure": "kendall",
                "variable_types": {"score_x": "continuous", "score_y": "continuous"},
            },
            "expected": reference_kendall_tau_b(x_ktb_1, y_ktb_1),
            "missing_accounting": {"original_rows": 8, "analyzed_rows": 8, "excluded_rows": 0},
            "orientation": {"association": "strictly negative monotonic association"},
        }
    )

    # Case 2: Ties in both variables and missing rows
    x_ktb_2_raw = [1.0, 2.0, 2.0, 3.0, np.nan, 4.0, 4.0, 5.0, 5.0, 6.0]
    y_ktb_2_raw = [2.0, 1.0, 3.0, 3.0, 2.0, 4.0, 5.0, np.nan, 5.0, 6.0]
    clean_indices_ktb = [
        i
        for i in range(len(x_ktb_2_raw))
        if not np.isnan(x_ktb_2_raw[i]) and not np.isnan(y_ktb_2_raw[i])
    ]
    x_ktb_2_clean = [x_ktb_2_raw[i] for i in clean_indices_ktb]
    y_ktb_2_clean = [y_ktb_2_raw[i] for i in clean_indices_ktb]
    df_ktb_2 = pd.DataFrame({"var1": x_ktb_2_raw, "var2": y_ktb_2_raw})

    cases.append(
        {
            "case_id": "KTB_02_TIES_AND_MISSING",
            "method_id": "kendall_tau_b",
            "scientific_target": "Kendall's tau-b association with repeated ties and missing pairs",
            "description": "10 raw pairs (2 incomplete excluded) with multiple ties in x and y exercising tau-b denominator",
            "source_type": "manual_formula",
            "reference_citation": "Kendall (1938); complete-pair sample accounting",
            "df": df_ktb_2,
            "run_kwargs": {
                "objective": "association",
                "outcome": "var1",
                "predictor": "var2",
                "design": "independent",
                "estimand": "monotonic",
                "association_measure": "kendall",
                "variable_types": {"var1": "continuous", "var2": "continuous"},
            },
            "expected": reference_kendall_tau_b(x_ktb_2_clean, y_ktb_2_clean),
            "missing_accounting": {"original_rows": 10, "analyzed_rows": 8, "excluded_rows": 2},
            "orientation": {"association": "positive concordant trend"},
        }
    )

    # =========================================================================
    # 20. POINT-BISERIAL CORRELATION
    # =========================================================================

    # Case 1: Positive point-biserial association (N=10)
    bin_1 = [False] * 5 + [True] * 5
    score_1 = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 7.0, 8.0, 7.0, 9.0]
    df_pbc_1 = pd.DataFrame({"target": bin_1, "feature": score_1})

    cases.append(
        {
            "case_id": "PBC_01_POSITIVE",
            "method_id": "point_biserial_correlation",
            "scientific_target": "Point-biserial correlation for positive group difference",
            "description": "10 observations (5 False, 5 True) with distinct higher scores in True group",
            "source_type": "manual_formula",
            "reference_citation": "Lev (1949); Pearson product-moment equivalence",
            "df": df_pbc_1,
            "run_kwargs": {
                "objective": "association",
                "outcome": "target",
                "predictor": "feature",
                "design": "independent",
                "estimand": "point_biserial",
                "variable_types": {"target": "boolean", "feature": "continuous"},
            },
            "expected": reference_point_biserial(bin_1, score_1),
            "missing_accounting": {"original_rows": 10, "analyzed_rows": 10, "excluded_rows": 0},
            "orientation": {
                "point_biserial_r": "positive association (True corresponds to higher feature)"
            },
        }
    )

    # Case 2: Negative point-biserial association with missing data
    bin_2_raw = [True, True, True, True, False, False, False, False, True, False]
    score_2_raw = [2.0, 3.0, np.nan, 1.0, 8.0, 9.0, 7.0, np.nan, 2.5, 8.5]
    clean_indices_pbc = [i for i in range(len(bin_2_raw)) if not np.isnan(score_2_raw[i])]
    bin_2_clean = [bin_2_raw[i] for i in clean_indices_pbc]
    score_2_clean = [score_2_raw[i] for i in clean_indices_pbc]
    df_pbc_2 = pd.DataFrame({"flag": bin_2_raw, "val": score_2_raw})

    cases.append(
        {
            "case_id": "PBC_02_NEGATIVE_MISSING",
            "method_id": "point_biserial_correlation",
            "scientific_target": "Point-biserial correlation with negative direction and missing observations",
            "description": "10 raw rows (2 missing) where True group has systematically lower values",
            "source_type": "manual_formula",
            "reference_citation": "Lev (1949); complete-case sample accounting",
            "df": df_pbc_2,
            "run_kwargs": {
                "objective": "association",
                "outcome": "flag",
                "predictor": "val",
                "design": "independent",
                "estimand": "point_biserial",
                "variable_types": {"flag": "boolean", "val": "continuous"},
            },
            "expected": reference_point_biserial(bin_2_clean, score_2_clean),
            "missing_accounting": {"original_rows": 10, "analyzed_rows": 8, "excluded_rows": 2},
            "orientation": {
                "point_biserial_r": "negative association (True corresponds to lower values)"
            },
        }
    )

    # =========================================================================
    # 21. PARTIAL PEARSON CORRELATION
    # =========================================================================

    # Case 1: Single control variable (N=8)
    x_ppc_1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y_ppc_1 = [2.0, 3.0, 2.0, 5.0, 6.0, 5.0, 8.0, 9.0]
    z_ppc_1 = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    df_ppc_1 = pd.DataFrame({"outcome_x": x_ppc_1, "predictor_y": y_ppc_1, "control_z": z_ppc_1})

    cases.append(
        {
            "case_id": "PPC_01_SINGLE_CONTROL",
            "method_id": "partial_pearson_correlation",
            "scientific_target": "Partial Pearson correlation between x and y controlling for z",
            "description": "8 complete observations with single control covariate residualization",
            "source_type": "manual_formula",
            "reference_citation": "Yule (1907); OLS residualization",
            "df": df_ppc_1,
            "run_kwargs": {
                "objective": "association",
                "outcome": "outcome_x",
                "predictor": "predictor_y",
                "controls": ["control_z"],
                "design": "independent",
                "estimand": "partial_linear",
                "variable_types": {
                    "outcome_x": "continuous",
                    "predictor_y": "continuous",
                    "control_z": "continuous",
                },
            },
            "expected": reference_partial_pearson(
                x_ppc_1, y_ppc_1, np.array(z_ppc_1)[:, np.newaxis]
            ),
            "missing_accounting": {"original_rows": 8, "analyzed_rows": 8, "excluded_rows": 0},
            "orientation": {"partial_r": "positive association conditional on z"},
        }
    )

    # Case 2: Multiple controls with missing rows (N=12 raw, 2 missing)
    x_ppc_2_raw = [10.0, 12.0, np.nan, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0, np.nan, 35.0]
    y_ppc_2_raw = [20.0, 22.0, 24.0, 25.0, 27.0, 29.0, 31.0, np.nan, 35.0, 38.0, 40.0, 45.0]
    c1_ppc_2_raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    c2_ppc_2_raw = [5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0]

    clean_ppc_indices = [
        i
        for i in range(len(x_ppc_2_raw))
        if not np.isnan(x_ppc_2_raw[i]) and not np.isnan(y_ppc_2_raw[i])
    ]
    x_ppc_2_clean = [x_ppc_2_raw[i] for i in clean_ppc_indices]
    y_ppc_2_clean = [y_ppc_2_raw[i] for i in clean_ppc_indices]
    z_ppc_2_clean = np.column_stack(
        [
            [c1_ppc_2_raw[i] for i in clean_ppc_indices],
            [c2_ppc_2_raw[i] for i in clean_ppc_indices],
        ]
    )
    df_ppc_2 = pd.DataFrame(
        {"x": x_ppc_2_raw, "y": y_ppc_2_raw, "c1": c1_ppc_2_raw, "c2": c2_ppc_2_raw}
    )

    cases.append(
        {
            "case_id": "PPC_02_MULTIPLE_CONTROLS_MISSING",
            "method_id": "partial_pearson_correlation",
            "scientific_target": "Partial correlation with two control covariates and missing observation filtering",
            "description": "12 raw rows (3 rows excluded for missing values) controlling for c1 and c2",
            "source_type": "manual_formula",
            "reference_citation": "Yule (1907); complete-case sample accounting",
            "df": df_ppc_2,
            "run_kwargs": {
                "objective": "association",
                "outcome": "x",
                "predictor": "y",
                "controls": ["c1", "c2"],
                "design": "independent",
                "estimand": "partial_linear",
                "variable_types": {
                    "x": "continuous",
                    "y": "continuous",
                    "c1": "continuous",
                    "c2": "continuous",
                },
            },
            "expected": reference_partial_pearson(x_ppc_2_clean, y_ppc_2_clean, z_ppc_2_clean),
            "missing_accounting": {"original_rows": 12, "analyzed_rows": 9, "excluded_rows": 3},
            "orientation": {"partial_r": "association between x and y adjusted for c1 and c2"},
        }
    )

    # =========================================================================
    # 22. EXACT BINOMIAL MCNEMAR TEST
    # =========================================================================

    # Case 1: Asymmetric discordance (b=5, c=1, a=2, d=4, total=12 pairs)
    pairs_mcn_1 = [
        # (after, before)
        ("yes", "yes"),
        ("yes", "yes"),  # a=2
        ("yes", "no"),
        ("yes", "no"),
        ("yes", "no"),
        ("yes", "no"),
        ("yes", "no"),  # b=5
        ("no", "yes"),  # c=1
        ("no", "no"),
        ("no", "no"),
        ("no", "no"),
        ("no", "no"),  # d=4
    ]
    rows_mcn_1 = []
    for uid, (after_resp, before_resp) in enumerate(pairs_mcn_1, start=1):
        rows_mcn_1.append({"unit": uid, "time": "before", "response": before_resp})
        rows_mcn_1.append({"unit": uid, "time": "after", "response": after_resp})
    df_mcn_1 = pd.DataFrame(rows_mcn_1)

    # 2x2 transition table with row=after, col=before:
    # [[a=2, b=5], [c=1, d=4]]
    tbl_mcn_1 = [[2, 5], [1, 4]]

    cases.append(
        {
            "case_id": "MCN_01_ASYMMETRIC",
            "method_id": "mcnemar",
            "scientific_target": "Exact McNemar test with asymmetric discordant transitions",
            "description": "12 matched pairs with 5 positive transitions (no -> yes) vs 1 negative transition (yes -> no)",
            "source_type": "manual_formula",
            "reference_citation": "McNemar (1947); exact combinatorial binomial tail sum",
            "df": df_mcn_1,
            "run_kwargs": {
                "objective": "compare_groups",
                "outcome": "response",
                "predictor": "time",
                "design": "paired",
                "unit_id": "unit",
                "estimand": "proportion",
                "event_level": "yes",
                "condition_order": ("after", "before"),
                "variable_types": {"response": "nominal", "time": "nominal"},
            },
            "expected": reference_mcnemar_exact(tbl_mcn_1),
            "missing_accounting": {"original_rows": 24, "analyzed_rows": 24, "excluded_rows": 0},
            "orientation": {"transition": "after ('yes') vs before ('yes')"},
        }
    )

    # Case 2: Equal discordance (b=3, c=3) with an incomplete unit
    pairs_mcn_2 = [
        ("yes", "yes"),
        ("yes", "yes"),
        ("yes", "no"),
        ("yes", "no"),
        ("yes", "no"),  # b=3
        ("no", "yes"),
        ("no", "yes"),
        ("no", "yes"),  # c=3
        ("no", "no"),
        ("no", "no"),
    ]
    rows_mcn_2 = []
    for uid, (after_resp, before_resp) in enumerate(pairs_mcn_2, start=1):
        rows_mcn_2.append({"unit": uid, "time": "before", "response": before_resp})
        rows_mcn_2.append({"unit": uid, "time": "after", "response": after_resp})
    # Incomplete unit
    rows_mcn_2.append({"unit": 99, "time": "before", "response": "yes"})
    df_mcn_2 = pd.DataFrame(rows_mcn_2)

    tbl_mcn_2 = [[2, 3], [3, 2]]

    cases.append(
        {
            "case_id": "MCN_02_EQUAL_DISCORDANCE_MISSING",
            "method_id": "mcnemar",
            "scientific_target": "Exact McNemar test under equal discordance (p=1.0) with incomplete unit exclusion",
            "description": "10 complete matched pairs (b=3, c=3) plus 1 incomplete unit excluded",
            "source_type": "manual_formula",
            "reference_citation": "McNemar (1947); matched-pair complete unit filtering",
            "df": df_mcn_2,
            "run_kwargs": {
                "objective": "compare_groups",
                "outcome": "response",
                "predictor": "time",
                "design": "paired",
                "unit_id": "unit",
                "estimand": "proportion",
                "event_level": "yes",
                "condition_order": ("after", "before"),
                "variable_types": {"response": "nominal", "time": "nominal"},
            },
            "expected": reference_mcnemar_exact(tbl_mcn_2),
            "missing_accounting": {"original_rows": 21, "analyzed_rows": 20, "excluded_rows": 1},
            "orientation": {"transition": "after ('yes') vs before ('yes')"},
        }
    )

    return cases
