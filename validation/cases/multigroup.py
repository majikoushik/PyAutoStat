"""Reference validation cases for multigroup methods.

Covers:
11. welch_anova (+ Games-Howell)
12. one_way_anova (+ Tukey-Kramer)
13. kruskal_wallis (+ Dunn-Holm)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from validation.references.multigroup import (
    reference_kruskal_wallis,
    reference_one_way_anova,
    reference_welch_anova,
)


def get_multigroup_cases() -> list[dict[str, Any]]:
    """Return reference validation cases for multigroup methods."""
    cases: list[dict[str, Any]] = []

    # =========================================================================
    # 11. WELCH ANOVA
    # =========================================================================

    # Case 1: Heteroscedastic unequal n with 3 groups
    g_a1 = [10.0, 11.0, 12.0, 10.5, 11.5]  # n=5, mean=11.0, var=0.625
    g_b1 = [15.0, 18.0, 20.0, 16.0, 17.0, 19.0]  # n=6, mean=17.5, var=3.5
    g_c1 = [25.0, 30.0, 22.0, 28.0, 32.0, 26.0, 27.0]  # n=7, mean=27.14, var=11.14
    df_wan_1 = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 6 + ["C"] * 7,
            "score": g_a1 + g_b1 + g_c1,
        }
    )
    cases.append(
        {
            "case_id": "WAN_01_HETERO_UNEQUAL",
            "method_id": "welch_anova",
            "scientific_target": "Welch omnibus ANOVA and Games-Howell pairwise under heteroscedasticity",
            "description": "3 independent groups with unequal sample sizes (5, 6, 7) and distinct variances",
            "source_type": "manual_formula",
            "reference_citation": "Welch (1951); Games & Howell (1976)",
            "df": df_wan_1,
            "run_kwargs": {"group_col": "group", "value_col": "score"},
            "expected": reference_welch_anova([g_a1, g_b1, g_c1], ["A", "B", "C"]),
            "missing_accounting": {"original_rows": 18, "analyzed_rows": 18, "excluded_rows": 0},
            "orientation": {"pairwise_contrast": "first group minus second group"},
        }
    )

    # Case 2: Unequal n with missing values
    g_a2_raw = [4.0, 5.0, np.nan, 6.0, 5.5, 4.5]
    g_b2_raw = [8.0, 9.0, 10.0, 8.5, np.nan, 9.5]
    g_c2_raw = [2.0, 3.0, 2.5, 3.5, 1.5]
    g_a2 = [x for x in g_a2_raw if not np.isnan(x)]
    g_b2 = [x for x in g_b2_raw if not np.isnan(x)]
    g_c2 = [x for x in g_c2_raw if not np.isnan(x)]
    df_wan_2 = pd.DataFrame(
        {
            "group": ["G1"] * len(g_a2_raw) + ["G2"] * len(g_b2_raw) + ["G3"] * len(g_c2_raw),
            "score": g_a2_raw + g_b2_raw + g_c2_raw,
        }
    )
    cases.append(
        {
            "case_id": "WAN_02_MISSING_REVERSED",
            "method_id": "welch_anova",
            "scientific_target": "Welch ANOVA with missing rows and non-monotonic group ordering",
            "description": "17 raw rows (2 missing) across 3 groups with reversed mean pattern (G2 > G1 > G3)",
            "source_type": "manual_formula",
            "reference_citation": "Welch (1951); complete-case sample accounting",
            "df": df_wan_2,
            "run_kwargs": {"group_col": "group", "value_col": "score"},
            "expected": reference_welch_anova([g_a2, g_b2, g_c2], ["G1", "G2", "G3"]),
            "missing_accounting": {"original_rows": 17, "analyzed_rows": 15, "excluded_rows": 2},
            "orientation": {"pairwise_contrast": "first group minus second group"},
        }
    )

    # =========================================================================
    # 12. CLASSICAL ONE-WAY ANOVA
    # =========================================================================

    # Case 1: Balanced equal-n equal-variance
    g_owa_1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    g_owa_2 = [3.0, 4.0, 6.0, 7.0, 8.0]
    g_owa_3 = [8.0, 9.0, 10.0, 11.0, 13.0]
    df_owa_1 = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "score": g_owa_1 + g_owa_2 + g_owa_3,
        }
    )
    cases.append(
        {
            "case_id": "OWA_01_BALANCED",
            "method_id": "one_way_anova",
            "scientific_target": "Classical One-Way ANOVA and Tukey-Kramer post-hoc (balanced design)",
            "description": "15 observations in 3 equal groups of 5 with clear monotonic group separation",
            "source_type": "manual_formula",
            "reference_citation": "Fisher (1925); Tukey (1949)",
            "df": df_owa_1,
            "run_kwargs": {"group_col": "group", "value_col": "score"},
            "expected": reference_one_way_anova([g_owa_1, g_owa_2, g_owa_3], ["A", "B", "C"]),
            "missing_accounting": {"original_rows": 15, "analyzed_rows": 15, "excluded_rows": 0},
            "orientation": {"pairwise_contrast": "first group minus second group"},
        }
    )

    # Case 2: Unequal n with missing values
    g_owa_m1 = [12.0, 14.0, 15.0, 13.0, np.nan, 16.0]
    g_owa_m2 = [18.0, 20.0, 19.0, np.nan, 21.0, 22.0, 20.0]
    g_owa_m3 = [25.0, 26.0, 28.0, 27.0]
    g_clean_m1 = [x for x in g_owa_m1 if not np.isnan(x)]
    g_clean_m2 = [x for x in g_owa_m2 if not np.isnan(x)]
    g_clean_m3 = [x for x in g_owa_m3 if not np.isnan(x)]
    df_owa_2 = pd.DataFrame(
        {
            "group": ["Ctrl"] * len(g_owa_m1) + ["Low"] * len(g_owa_m2) + ["High"] * len(g_owa_m3),
            "score": g_owa_m1 + g_owa_m2 + g_owa_m3,
        }
    )
    cases.append(
        {
            "case_id": "OWA_02_UNEQUAL_MISSING",
            "method_id": "one_way_anova",
            "scientific_target": "One-Way ANOVA with unequal sample sizes and missing data",
            "description": "17 raw rows (2 missing) across 3 treatment conditions with complete-case exclusion",
            "source_type": "manual_formula",
            "reference_citation": "Fisher (1925); complete-case sample accounting",
            "df": df_owa_2,
            "run_kwargs": {"group_col": "group", "value_col": "score"},
            "expected": reference_one_way_anova(
                [g_clean_m1, g_clean_m2, g_clean_m3], ["Ctrl", "Low", "High"]
            ),
            "missing_accounting": {"original_rows": 17, "analyzed_rows": 15, "excluded_rows": 2},
            "orientation": {"pairwise_contrast": "first group minus second group"},
        }
    )

    # =========================================================================
    # 13. KRUSKAL-WALLIS
    # =========================================================================

    # Case 1: No ties
    g_kwa_1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    g_kwa_2 = [6.0, 7.0, 8.0, 9.0, 10.0]
    g_kwa_3 = [11.0, 12.0, 13.0, 14.0, 15.0]
    df_kwa_1 = pd.DataFrame(
        {
            "group": ["X"] * 5 + ["Y"] * 5 + ["Z"] * 5,
            "score": g_kwa_1 + g_kwa_2 + g_kwa_3,
        }
    )
    cases.append(
        {
            "case_id": "KWA_01_NO_TIES",
            "method_id": "kruskal_wallis",
            "scientific_target": "Kruskal-Wallis rank sum test and Dunn-Holm pairwise without ties",
            "description": "15 distinct observations across 3 groups with unique ranks 1..15",
            "source_type": "manual_formula",
            "reference_citation": "Kruskal & Wallis (1952); Dunn (1964); Holm (1979)",
            "df": df_kwa_1,
            "run_kwargs": {"group_col": "group", "value_col": "score"},
            "expected": reference_kruskal_wallis([g_kwa_1, g_kwa_2, g_kwa_3], ["X", "Y", "Z"]),
            "missing_accounting": {"original_rows": 15, "analyzed_rows": 15, "excluded_rows": 0},
            "orientation": {"pairwise_contrast": "first group minus second group rank difference"},
        }
    )

    # Case 2: Substantial ties and missing rows
    g_kwa_t1 = [1.0, 1.0, 2.0, 2.0, 3.0, np.nan]
    g_kwa_t2 = [2.0, 2.0, 3.0, 4.0, 4.0, 4.0]
    g_kwa_t3 = [4.0, 5.0, 5.0, 6.0, 6.0, np.nan]
    g_clean_t1 = [x for x in g_kwa_t1 if not np.isnan(x)]
    g_clean_t2 = [x for x in g_kwa_t2 if not np.isnan(x)]
    g_clean_t3 = [x for x in g_kwa_t3 if not np.isnan(x)]
    df_kwa_2 = pd.DataFrame(
        {
            "group": ["A"] * len(g_kwa_t1) + ["B"] * len(g_kwa_t2) + ["C"] * len(g_kwa_t3),
            "score": g_kwa_t1 + g_kwa_t2 + g_kwa_t3,
        }
    )
    cases.append(
        {
            "case_id": "KWA_02_TIED_RANKS",
            "method_id": "kruskal_wallis",
            "scientific_target": "Kruskal-Wallis test with fractional average tied ranks and missing data",
            "description": "18 rows (2 missing) with repeated score ties across groups exercising tie correction",
            "source_type": "manual_formula",
            "reference_citation": "Kruskal & Wallis (1952); fractional rank tie adjustment",
            "df": df_kwa_2,
            "run_kwargs": {"group_col": "group", "value_col": "score"},
            "expected": reference_kruskal_wallis(
                [g_clean_t1, g_clean_t2, g_clean_t3], ["A", "B", "C"]
            ),
            "missing_accounting": {"original_rows": 18, "analyzed_rows": 16, "excluded_rows": 2},
            "orientation": {"pairwise_contrast": "first group minus second group rank difference"},
        }
    )

    return cases
