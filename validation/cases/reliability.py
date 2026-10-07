"""Reference validation cases for scale reliability and inter-rater agreement.

Covers:
23. cronbach_alpha (scale reliability, item-total correlation, alpha-if-deleted)
24. intraclass_correlation (all 6 Shrout & Fleiss / McGraw & Wong ICC configurations)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from validation.references.reliability import (
    reference_cronbach_alpha,
    reference_icc,
)


def get_reliability_cases() -> list[dict[str, Any]]:
    """Return reference validation cases for reliability and agreement methods."""
    cases: list[dict[str, Any]] = []

    # =========================================================================
    # 23. CRONBACH'S ALPHA
    # =========================================================================

    # Case 1: Standard 4-item positive scale (N=8)
    mat_cra_1 = np.array(
        [
            [4.0, 5.0, 4.0, 5.0],
            [3.0, 4.0, 3.0, 4.0],
            [5.0, 5.0, 5.0, 5.0],
            [2.0, 2.0, 3.0, 2.0],
            [4.0, 4.0, 4.0, 4.0],
            [3.0, 3.0, 2.0, 3.0],
            [5.0, 4.0, 5.0, 5.0],
            [1.0, 2.0, 1.0, 2.0],
        ]
    )
    items_1 = ["item1", "item2", "item3", "item4"]
    df_cra_1 = pd.DataFrame(mat_cra_1, columns=items_1)

    cases.append(
        {
            "case_id": "CRA_01_STANDARD_SCALE",
            "method_id": "cronbach_alpha",
            "scientific_target": "Cronbach's alpha internal consistency and corrected item-total correlation",
            "description": "8 respondents x 4 items with high positive covariance across all items",
            "source_type": "manual_formula",
            "reference_citation": "Cronbach (1951); Nunnally & Bernstein (1994)",
            "df": df_cra_1,
            "run_kwargs": {
                "items": items_1,
            },
            "expected": reference_cronbach_alpha(mat_cra_1, items_1),
            "missing_accounting": {"original_rows": 8, "analyzed_rows": 8, "excluded_rows": 0},
            "orientation": {"internal_consistency": "positive unipolar scale"},
        }
    )

    # Case 2: Scale with missing respondent rows (N=10 raw, 2 missing)
    mat_cra_2_raw = [
        [1.0, 2.0, 1.0, 2.0],
        [np.nan, 3.0, 3.0, 4.0],  # missing item1
        [2.0, 3.0, 2.0, 3.0],
        [3.0, 4.0, 4.0, 4.0],
        [4.0, 4.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0],
        [3.0, 3.0, np.nan, 3.0],  # missing item3
        [2.0, 2.0, 3.0, 2.0],
        [4.0, 5.0, 4.0, 4.0],
        [1.0, 1.0, 2.0, 1.0],
    ]
    df_cra_2 = pd.DataFrame(mat_cra_2_raw, columns=["q1", "q2", "q3", "q4"])
    clean_cra_2 = [row for row in mat_cra_2_raw if not any(np.isnan(v) for v in row)]
    mat_cra_2_clean = np.array(clean_cra_2, dtype=float)

    cases.append(
        {
            "case_id": "CRA_02_MISSING_RESPONDENTS",
            "method_id": "cronbach_alpha",
            "scientific_target": "Cronbach's alpha with complete-respondent sample accounting",
            "description": "10 raw respondents (2 rows excluded for missing item values) with 4-item scale",
            "source_type": "manual_formula",
            "reference_citation": "Cronbach (1951); complete-case respondent filtering",
            "df": df_cra_2,
            "run_kwargs": {
                "items": ["q1", "q2", "q3", "q4"],
            },
            "expected": reference_cronbach_alpha(mat_cra_2_clean, ["q1", "q2", "q3", "q4"]),
            "missing_accounting": {"original_rows": 10, "analyzed_rows": 8, "excluded_rows": 2},
            "orientation": {"internal_consistency": "positive scale consistency"},
        }
    )

    # =========================================================================
    # 24. INTRACLASS CORRELATION
    # =========================================================================

    # Case 1: High agreement across all raters (6 targets x 3 raters)
    mat_icc_1 = np.array(
        [
            [1.0, 1.2, 1.0],
            [3.0, 3.1, 2.9],
            [5.0, 5.0, 5.2],
            [7.0, 6.9, 7.1],
            [9.0, 9.1, 8.9],
            [4.0, 4.0, 4.1],
        ],
        dtype=float,
    )
    rows_icc_1 = []
    for t in range(6):
        for r in range(3):
            rows_icc_1.append(
                {"target": f"T{t + 1}", "rater": f"R{r + 1}", "score": mat_icc_1[t, r]}
            )
    df_icc_1 = pd.DataFrame(rows_icc_1)

    cases.append(
        {
            "case_id": "ICC_01_HIGH_AGREEMENT",
            "method_id": "intraclass_correlation",
            "scientific_target": "Intraclass correlation across all 6 Shrout & Fleiss configurations under high agreement",
            "description": "6 targets rated by 3 raters with minimal measurement error; agreement and consistency both high",
            "source_type": "manual_formula",
            "reference_citation": "Shrout & Fleiss (1979); McGraw & Wong (1996)",
            "df": df_icc_1,
            "run_kwargs": {
                "target": "target",
                "rater": "rater",
                "value": "score",
            },
            "expected": reference_icc(mat_icc_1),
            "missing_accounting": {"original_rows": 18, "analyzed_rows": 18, "excluded_rows": 0},
            "orientation": {"agreement": "high rater consensus across targets"},
        }
    )

    # Case 2: Systematic rater shift distinguishing Consistency from Absolute Agreement
    # Rater 2 has a constant +5 point bias: consistency ICC(3,1) ~ 1.0, absolute agreement ICC(2,1) ~ 0.5
    mat_icc_2 = np.array(
        [
            [1.0, 6.0, 1.2],
            [3.0, 8.0, 3.1],
            [5.0, 10.0, 5.0],
            [7.0, 12.0, 6.9],
            [9.0, 14.0, 9.1],
            [4.0, 9.0, 4.0],
        ],
        dtype=float,
    )
    rows_icc_2 = []
    for t in range(6):
        for r in range(3):
            rows_icc_2.append(
                {"target": f"Target{t + 1}", "rater": f"Rater{r + 1}", "score": mat_icc_2[t, r]}
            )
    df_icc_2 = pd.DataFrame(rows_icc_2)

    cases.append(
        {
            "case_id": "ICC_02_SYSTEMATIC_RATER_SHIFT",
            "method_id": "intraclass_correlation",
            "scientific_target": "Intraclass correlation distinguishing absolute agreement from consistency under systematic rater bias",
            "description": "6 targets x 3 raters where Rater 2 has additive +5 offset; consistency ICC(3,1) ~ 1.0 while absolute ICC(2,1) ~ 0.5",
            "source_type": "manual_formula",
            "reference_citation": "Shrout & Fleiss (1979); McGraw & Wong (1996)",
            "df": df_icc_2,
            "run_kwargs": {
                "target": "target",
                "rater": "rater",
                "value": "score",
            },
            "expected": reference_icc(mat_icc_2),
            "missing_accounting": {"original_rows": 18, "analyzed_rows": 18, "excluded_rows": 0},
            "orientation": {
                "agreement": "consistency high, absolute agreement penalized by rater bias"
            },
        }
    )

    return cases
