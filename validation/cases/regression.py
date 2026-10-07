"""Reference validation cases for linear and logistic regression methods.

Covers:
17. linear_regression (continuous predictors, HC3 robust covariance, missing rows)
18. logistic_regression (binary Logit, odds ratios, McFadden pseudo-R2, missing rows)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from validation.references.regression import (
    reference_linear_regression,
    reference_logistic_regression,
)


def get_regression_cases() -> list[dict[str, Any]]:
    """Return reference validation cases for regression methods."""
    cases: list[dict[str, Any]] = []

    # =========================================================================
    # 17. OLS LINEAR REGRESSION
    # =========================================================================

    # Case 1: Continuous predictors with HC3 robust covariance
    x1_1 = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
        1.0,
        3.0,
        5.0,
        7.0,
        9.0,
    ]
    x2_1 = [
        5.0,
        1.0,
        4.0,
        2.0,
        6.0,
        3.0,
        7.0,
        2.0,
        8.0,
        4.0,
        7.0,
        3.0,
        6.0,
        2.0,
        5.0,
        8.0,
        4.0,
        7.0,
        3.0,
        6.0,
    ]
    # Fixed deterministic y based on seed 42
    rng_lr = np.random.default_rng(42)
    y_1 = [
        float(3.0 + 1.2 * x1_val - 0.7 * x2_val + err)
        for x1_val, x2_val, err in zip(x1_1, x2_1, rng_lr.normal(0, 0.5, 20), strict=True)
    ]
    df_lnr_1 = pd.DataFrame({"outcome": y_1, "pred1": x1_1, "pred2": x2_1})
    X_mat_1 = np.column_stack([np.ones(len(y_1)), x1_1, x2_1])

    cases.append(
        {
            "case_id": "LNR_01_CONTINUOUS_HC3",
            "method_id": "linear_regression",
            "scientific_target": "Multiple OLS linear regression with classical and HC3 robust standard errors",
            "description": "20 observations with 2 continuous predictors; beta solve, residuals, R2, and HC3 covariance",
            "source_type": "manual_formula",
            "reference_citation": "Legendre (1805); Gauss (1809); MacKinnon & White (1985) HC3",
            "df": df_lnr_1,
            "run_kwargs": {
                "outcome": "outcome",
                "predictors": ["pred1", "pred2"],
                "variable_types": {
                    "outcome": "continuous",
                    "pred1": "continuous",
                    "pred2": "continuous",
                },
                "covariance_type": "HC3",
            },
            "expected": reference_linear_regression(
                X_mat_1, np.array(y_1), ["Intercept", "pred1", "pred2"]
            ),
            "missing_accounting": {"original_rows": 20, "analyzed_rows": 20, "excluded_rows": 0},
            "orientation": {"slopes": "pred1 positive association, pred2 negative association"},
        }
    )

    # Case 2: Linear regression with missing data and classical covariance
    x1_2_raw = [
        2.0,
        3.0,
        5.0,
        np.nan,
        7.0,
        8.0,
        9.0,
        11.0,
        12.0,
        14.0,
        15.0,
        np.nan,
        18.0,
        19.0,
        20.0,
    ]
    y_2_raw = [
        5.2,
        6.8,
        11.4,
        9.0,
        14.7,
        17.3,
        18.9,
        np.nan,
        25.1,
        28.8,
        31.2,
        25.0,
        36.9,
        39.4,
        40.8,
    ]
    # Clean rows: both non-nan
    clean_indices = [
        i for i in range(len(x1_2_raw)) if not np.isnan(x1_2_raw[i]) and not np.isnan(y_2_raw[i])
    ]
    x1_2_clean = [x1_2_raw[i] for i in clean_indices]
    y_2_clean = [y_2_raw[i] for i in clean_indices]
    df_lnr_2 = pd.DataFrame({"y": y_2_raw, "x": x1_2_raw})
    X_mat_2 = np.column_stack([np.ones(len(y_2_clean)), x1_2_clean])

    cases.append(
        {
            "case_id": "LNR_02_CLASSICAL_MISSING",
            "method_id": "linear_regression",
            "scientific_target": "Simple OLS linear regression with classical standard errors and missing rows",
            "description": "15 raw rows (3 rows excluded for missing predictor or outcome) with complete-case fit",
            "source_type": "manual_formula",
            "reference_citation": "Gauss (1809); complete-case sample accounting",
            "df": df_lnr_2,
            "run_kwargs": {
                "outcome": "y",
                "predictors": ["x"],
                "variable_types": {"y": "continuous", "x": "continuous"},
                "covariance_type": "classical",
            },
            "expected": reference_linear_regression(
                X_mat_2, np.array(y_2_clean), ["Intercept", "x"]
            ),
            "missing_accounting": {"original_rows": 15, "analyzed_rows": 12, "excluded_rows": 3},
            "orientation": {"slope": "positive relationship x -> y"},
        }
    )

    # =========================================================================
    # 18. BINARY LOGISTIC REGRESSION
    # =========================================================================

    # Case 1: Single continuous predictor without separation (N=22)
    x_log_1 = [
        -2.5,
        -2.0,
        -1.8,
        -1.5,
        -1.2,
        -1.0,
        -0.8,
        -0.5,
        -0.3,
        -0.1,
        0.1,
        0.2,
        0.5,
        0.7,
        0.9,
        1.1,
        1.3,
        1.6,
        1.8,
        2.0,
        2.2,
        2.5,
    ]
    y_log_1 = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    event_labels_1 = ["yes" if v == 1 else "no" for v in y_log_1]
    df_log_1 = pd.DataFrame({"event": event_labels_1, "predictor": x_log_1})
    X_mat_log_1 = np.column_stack([np.ones(len(x_log_1)), x_log_1])

    cases.append(
        {
            "case_id": "LOG_01_SINGLE_CONTINUOUS",
            "method_id": "logistic_regression",
            "scientific_target": "Binary logistic regression MLE, log-likelihood, and odds ratios",
            "description": "22 observations with continuous predictor and overlapping binary response avoiding separation",
            "source_type": "manual_formula",
            "reference_citation": "Cox (1958); McFadden (1974); independent Newton-Raphson/IRLS",
            "df": df_log_1,
            "run_kwargs": {
                "objective": "regression",
                "outcome": "event",
                "predictors": ["predictor"],
                "design": "independent",
                "estimand": "event_probability",
                "event_level": "yes",
                "variable_types": {"event": "nominal", "predictor": "continuous"},
            },
            "expected": reference_logistic_regression(
                X_mat_log_1, np.array(y_log_1, dtype=float), ["Intercept", "predictor"]
            ),
            "missing_accounting": {"original_rows": 22, "analyzed_rows": 22, "excluded_rows": 0},
            "orientation": {
                "log_odds": "positive slope on predictor increasing probability of 'yes'"
            },
        }
    )

    # Case 2: Binary logistic regression with missing rows (N=24 raw, 2 missing)
    x_log_2_raw = [
        -2.0,
        -1.5,
        -1.2,
        np.nan,
        -0.8,
        -0.5,
        -0.2,
        0.0,
        0.3,
        0.6,
        0.9,
        1.2,
        1.5,
        1.8,
        2.1,
        -1.0,
        0.2,
        0.5,
        0.8,
        1.1,
        -1.8,
        np.nan,
        1.4,
        2.0,
    ]
    y_log_2_raw = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1]
    clean_log_indices = [
        i
        for i in range(len(x_log_2_raw))
        if not np.isnan(x_log_2_raw[i]) and y_log_2_raw[i] is not None
    ]
    x_log_2_clean = [x_log_2_raw[i] for i in clean_log_indices]
    y_log_2_clean = [y_log_2_raw[i] for i in clean_log_indices]
    event_labels_2_raw = [
        ("success" if v == 1 else "failure") if not np.isnan(x_log_2_raw[i]) else "failure"
        for i, v in enumerate(y_log_2_raw)
    ]
    # Put actual NaN in predictor
    df_log_2 = pd.DataFrame({"status": event_labels_2_raw, "dose": x_log_2_raw})
    X_mat_log_2 = np.column_stack([np.ones(len(x_log_2_clean)), x_log_2_clean])

    cases.append(
        {
            "case_id": "LOG_02_MISSING_DATA",
            "method_id": "logistic_regression",
            "scientific_target": "Binary logistic regression with complete-case observation filtering",
            "description": "24 raw observations (2 missing) evaluating complete-case MLE and McFadden pseudo-R2",
            "source_type": "manual_formula",
            "reference_citation": "Cox (1958); complete-case sample accounting",
            "df": df_log_2,
            "run_kwargs": {
                "objective": "regression",
                "outcome": "status",
                "predictors": ["dose"],
                "design": "independent",
                "estimand": "event_probability",
                "event_level": "success",
                "variable_types": {"status": "nominal", "dose": "continuous"},
            },
            "expected": reference_logistic_regression(
                X_mat_log_2, np.array(y_log_2_clean, dtype=float), ["Intercept", "dose"]
            ),
            "missing_accounting": {"original_rows": 24, "analyzed_rows": 22, "excluded_rows": 2},
            "orientation": {"log_odds": "higher dose increases probability of 'success'"},
        }
    )

    return cases
