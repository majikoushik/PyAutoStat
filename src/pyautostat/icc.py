"""Validated Intraclass Correlation Coefficient (ICC) engine for reliability analysis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import ColumnNotFoundError, InsufficientDataError, InvalidDataError

# Canonical mapping of ICC variants following Shrout & Fleiss (1979) and McGraw & Wong (1996)
ICC_VARIANTS: dict[str, dict[str, str]] = {
    "icc_1_1": {
        "variant": "icc_1_1",
        "notation": "ICC(1,1)",
        "mcgraw_wong_notation": "ICC(1,1)",
        "model": "one_way_random",
        "definition": "absolute_agreement",
        "unit": "single",
        "description": "One-way random-effects, single-measure ICC (ICC(1,1))",
    },
    "icc_1_k": {
        "variant": "icc_1_k",
        "notation": "ICC(1,k)",
        "mcgraw_wong_notation": "ICC(1,k)",
        "model": "one_way_random",
        "definition": "absolute_agreement",
        "unit": "average",
        "description": "One-way random-effects, average-measure ICC (ICC(1,k))",
    },
    "icc_2_1": {
        "variant": "icc_2_1",
        "notation": "ICC(2,1)",
        "mcgraw_wong_notation": "ICC(A,1)",
        "model": "two_way_random",
        "definition": "absolute_agreement",
        "unit": "single",
        "description": "Two-way random-effects, absolute-agreement, single-measure ICC (ICC(2,1))",
    },
    "icc_2_k": {
        "variant": "icc_2_k",
        "notation": "ICC(2,k)",
        "mcgraw_wong_notation": "ICC(A,k)",
        "model": "two_way_random",
        "definition": "absolute_agreement",
        "unit": "average",
        "description": "Two-way random-effects, absolute-agreement, average-measure ICC (ICC(2,k))",
    },
    "icc_3_1": {
        "variant": "icc_3_1",
        "notation": "ICC(3,1)",
        "mcgraw_wong_notation": "ICC(C,1)",
        "model": "two_way_mixed",
        "definition": "consistency",
        "unit": "single",
        "description": "Two-way mixed-effects, consistency, single-measure ICC (ICC(3,1))",
    },
    "icc_3_k": {
        "variant": "icc_3_k",
        "notation": "ICC(3,k)",
        "mcgraw_wong_notation": "ICC(C,k)",
        "model": "two_way_mixed",
        "definition": "consistency",
        "unit": "average",
        "description": "Two-way mixed-effects, consistency, average-measure ICC (ICC(3,k))",
    },
    "icc_2_1_consistency": {
        "variant": "icc_2_1_consistency",
        "notation": "McGraw-Wong C(2,1)",
        "mcgraw_wong_notation": "ICC(C,1)",
        "model": "two_way_random",
        "definition": "consistency",
        "unit": "single",
        "description": (
            "Two-way random-effects, consistency, single-measure ICC (McGraw-Wong C(2,1))"
        ),
    },
    "icc_2_k_consistency": {
        "variant": "icc_2_k_consistency",
        "notation": "McGraw-Wong C(2,k)",
        "mcgraw_wong_notation": "ICC(C,k)",
        "model": "two_way_random",
        "definition": "consistency",
        "unit": "average",
        "description": (
            "Two-way random-effects, consistency, average-measure ICC (McGraw-Wong C(2,k))"
        ),
    },
    "icc_3_1_agreement": {
        "variant": "icc_3_1_agreement",
        "notation": "McGraw-Wong A(3,1)",
        "mcgraw_wong_notation": "ICC(A,1)",
        "model": "two_way_mixed",
        "definition": "absolute_agreement",
        "unit": "single",
        "description": (
            "Two-way mixed-effects, absolute-agreement, single-measure ICC (McGraw-Wong A(3,1))"
        ),
    },
    "icc_3_k_agreement": {
        "variant": "icc_3_k_agreement",
        "notation": "McGraw-Wong A(3,k)",
        "mcgraw_wong_notation": "ICC(A,k)",
        "model": "two_way_mixed",
        "definition": "absolute_agreement",
        "unit": "average",
        "description": (
            "Two-way mixed-effects, absolute-agreement, average-measure ICC (McGraw-Wong A(3,k))"
        ),
    },
}

SUPPORTED_MODELS = {"one_way_random", "two_way_random", "two_way_mixed"}
SUPPORTED_DEFINITIONS = {"absolute_agreement", "consistency"}
SUPPORTED_UNITS = {"single", "average"}


def _check_probability(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or not 0 < float(value) < 1
    ):
        raise InvalidDataError(f"{name} must be a finite number strictly between 0 and 1.")
    return float(value)


def resolve_variant(model: str, definition: str | None, unit: str) -> str:
    """Resolve model, definition, and unit parameters to a canonical ICC variant identifier."""
    if not isinstance(model, str) or model not in SUPPORTED_MODELS:
        models_str = ", ".join(sorted(SUPPORTED_MODELS))
        raise InvalidDataError(f"model must be one of: {models_str}. Got {model!r}.")
    if model == "one_way_random" and (definition is None or definition == "absolute_agreement"):
        resolved_def = "absolute_agreement"
    elif not isinstance(definition, str) or definition not in SUPPORTED_DEFINITIONS:
        defs_str = ", ".join(sorted(SUPPORTED_DEFINITIONS))
        raise InvalidDataError(f"definition must be one of: {defs_str}. Got {definition!r}.")
    else:
        resolved_def = definition
    if not isinstance(unit, str) or unit not in SUPPORTED_UNITS:
        units_str = ", ".join(sorted(SUPPORTED_UNITS))
        raise InvalidDataError(f"unit must be one of: {units_str}. Got {unit!r}.")

    if model == "one_way_random":
        if resolved_def == "consistency":
            raise InvalidDataError(
                "One-way random model evaluates undifferentiated within-target error and does "
                "not support definition='consistency'. Use definition='absolute_agreement'."
            )
        return "icc_1_1" if unit == "single" else "icc_1_k"

    if model == "two_way_random":
        if resolved_def == "absolute_agreement":
            return "icc_2_1" if unit == "single" else "icc_2_k"
        return "icc_2_1_consistency" if unit == "single" else "icc_2_k_consistency"

    # two_way_mixed
    if resolved_def == "consistency":
        return "icc_3_1" if unit == "single" else "icc_3_k"
    return "icc_3_1_agreement" if unit == "single" else "icc_3_k_agreement"


def icc_panel(
    frame: pd.DataFrame,
    target_col: str,
    rater_col: str,
    value_col: str,
) -> dict[str, Any]:
    """Extract and validate a balanced, complete-target panel for ICC analysis.

    Enforces:
    - Non-empty column names and distinct target, rater, and value roles.
    - Quantitative, finite outcome values.
    - Rejection of duplicate target-rater observations (no silent averaging).
    - Complete-target filtering (targets missing any rater are excluded with explicit accounting).
    - At least 2 retained targets and at least 2 retained raters.
    - Deterministic target and rater ordering.
    """
    if not isinstance(frame, pd.DataFrame):
        raise InvalidDataError("frame must be a pandas DataFrame.")
    for name, col in (("target", target_col), ("rater", rater_col), ("value", value_col)):
        if not isinstance(col, str) or not col.strip():
            raise InvalidDataError(f"{name} column must be a non-empty string.")
        if col not in frame.columns:
            raise ColumnNotFoundError(f"Column '{col}' not found in DataFrame.")

    if len({target_col, rater_col, value_col}) != 3:
        raise InvalidDataError("target, rater, and value must specify three different columns.")

    raw_targets = list(pd.unique(frame[target_col].dropna()))
    raw_raters = list(pd.unique(frame[rater_col].dropna()))

    if len(raw_targets) < 2:
        raise InsufficientDataError(
            f"Intraclass correlation requires at least 2 targets; found {len(raw_targets)}."
        )
    if len(raw_raters) < 2:
        raise InsufficientDataError(
            f"Intraclass correlation requires at least 2 raters; found {len(raw_raters)}."
        )

    # Check for duplicates on valid pairs
    valid_pair_mask = frame[target_col].notna() & frame[rater_col].notna()
    if frame.loc[valid_pair_mask].duplicated(subset=[target_col, rater_col]).any():
        raise InsufficientDataError(
            "Duplicate observations detected for target-rater combinations. Replicate readings "
            "are not supported; observations must not be silently averaged."
        )

    # Ensure outcome is numeric
    outcome_series = pd.to_numeric(frame[value_col], errors="coerce")
    valid_rows_mask = frame[target_col].notna() & frame[rater_col].notna() & outcome_series.notna()

    clean_subset = frame.loc[valid_rows_mask].copy()
    clean_subset["_score_numeric"] = outcome_series.loc[valid_rows_mask]

    # Nonfinite values check
    if not np.isfinite(clean_subset["_score_numeric"]).all():
        raise InvalidDataError("Outcome values must be finite numeric values.")

    # Identify complete targets: targets having all raters
    all_raters_set = set(raw_raters)
    target_grouped_raters = clean_subset.groupby(target_col)[rater_col].agg(set)

    complete_target_ids = [
        target_id
        for target_id, r_set in target_grouped_raters.items()
        if all_raters_set.issubset(r_set)
    ]

    # Deterministic sorting
    try:
        sorted_targets = sorted(complete_target_ids)
    except TypeError:
        sorted_targets = sorted(complete_target_ids, key=str)

    try:
        sorted_raters = sorted(raw_raters)
    except TypeError:
        sorted_raters = sorted(raw_raters, key=str)

    n_retained_targets = len(sorted_targets)
    n_raters = len(sorted_raters)

    if n_retained_targets < 2:
        raise InsufficientDataError(
            f"Intraclass correlation requires at least 2 complete targets with ratings across "
            f"all {n_raters} raters; retained {n_retained_targets} complete targets."
        )

    retained_mask = clean_subset[target_col].isin(sorted_targets) & clean_subset[rater_col].isin(
        sorted_raters
    )
    retained_df = clean_subset.loc[retained_mask]

    # Construct the n x k matrix
    pivot = retained_df.pivot(index=target_col, columns=rater_col, values="_score_numeric")
    matrix = pivot.loc[sorted_targets, sorted_raters].to_numpy(dtype=float)

    total_rows = len(frame)
    analyzed_rows = n_retained_targets * n_raters
    excluded_rows = total_rows - analyzed_rows
    excluded_targets = len(raw_targets) - n_retained_targets

    sample_accounting = {
        "original_rows": total_rows,
        "analyzed_rows": analyzed_rows,
        "excluded_rows": excluded_rows,
        "original_targets": len(raw_targets),
        "retained_targets": n_retained_targets,
        "excluded_targets": excluded_targets,
        "raters": sorted_raters,
        "complete_panel_cells": analyzed_rows,
    }

    return {
        "matrix": matrix,
        "targets": sorted_targets,
        "raters": sorted_raters,
        "n_targets": n_retained_targets,
        "n_raters": n_raters,
        "sample": sample_accounting,
    }


def icc_anova_components(matrix: np.ndarray) -> dict[str, Any]:
    """Compute ANOVA components and mean squares for complete n x k matrix."""
    n, k = matrix.shape
    grand_mean = float(np.mean(matrix))
    target_means = np.mean(matrix, axis=1)
    rater_means = np.mean(matrix, axis=0)

    ss_total = float(np.sum((matrix - grand_mean) ** 2))
    ss_targets = float(k * np.sum((target_means - grand_mean) ** 2))
    ss_raters = float(n * np.sum((rater_means - grand_mean) ** 2))
    ss_error = float(ss_total - ss_targets - ss_raters)

    # Numerical precision cleanup for exact additive models
    if abs(ss_error) < 1e-12:
        ss_error = 0.0

    if ss_total <= 0.0:
        raise InvalidDataError(
            "Total variance across all ratings is zero; intraclass correlation is undefined for "
            "constant data."
        )

    df_targets = n - 1
    df_raters = k - 1
    df_error = (n - 1) * (k - 1)
    df_total = n * k - 1

    ms_targets = ss_targets / df_targets if df_targets > 0 else 0.0
    ms_raters = ss_raters / df_raters if df_raters > 0 else 0.0
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    # One-way components
    ss_between = ss_targets
    df_between = df_targets
    ms_between = ms_targets

    ss_within = float(ss_raters + ss_error)
    df_within = n * (k - 1)
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    # Variance components (method of moments)
    # Target variance (sigma_s^2)
    var_target = (ms_targets - ms_error) / k
    # Rater variance (sigma_r^2)
    var_rater = (ms_raters - ms_error) / n
    # Residual error variance (sigma_e^2)
    var_residual = ms_error
    # Total variance for absolute agreement
    var_total_agreement = var_target + var_rater + var_residual
    # Total variance for consistency
    var_total_consistency = var_target + var_residual

    # One-way variance components
    var_target_oneway = (ms_between - ms_within) / k
    var_within_oneway = ms_within

    return {
        "grand_mean": grand_mean,
        "target_means": target_means.tolist(),
        "rater_means": rater_means.tolist(),
        "ss_total": ss_total,
        "ss_targets": ss_targets,
        "ss_raters": ss_raters,
        "ss_error": ss_error,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "df_total": df_total,
        "df_targets": df_targets,
        "df_raters": df_raters,
        "df_error": df_error,
        "df_between": df_between,
        "df_within": df_within,
        "ms_targets": ms_targets,
        "ms_raters": ms_raters,
        "ms_error": ms_error,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "variance_components": {
            "target_variance": var_target,
            "rater_variance": var_rater,
            "residual_variance": var_residual,
            "total_variance_agreement": var_total_agreement,
            "total_variance_consistency": var_total_consistency,
            "target_variance_oneway": var_target_oneway,
            "within_variance_oneway": var_within_oneway,
        },
    }


def compute_icc_estimate(variant: str, n: int, k: int, anova: dict[str, Any]) -> float:
    """Compute point estimate for a specific ICC variant from ANOVA mean squares."""
    ms_r = anova["ms_targets"]
    ms_c = anova["ms_raters"]
    ms_e = anova["ms_error"]
    ms_b = anova["ms_between"]
    ms_w = anova["ms_within"]

    if variant == "icc_1_1":
        denom = ms_b + (k - 1) * ms_w
        if denom == 0.0:
            raise InvalidDataError("Denominator for ICC(1,1) is zero; estimate is undefined.")
        return float((ms_b - ms_w) / denom)

    if variant == "icc_1_k":
        if ms_b == 0.0:
            raise InvalidDataError("MS_between is zero; ICC(1,k) is undefined.")
        return float((ms_b - ms_w) / ms_b)

    if variant in {"icc_2_1", "icc_3_1_agreement"}:
        denom = ms_r + (k - 1) * ms_e + (k / n) * (ms_c - ms_e)
        if denom == 0.0:
            raise InvalidDataError("Denominator for ICC(2,1) is zero; estimate is undefined.")
        return float((ms_r - ms_e) / denom)

    if variant in {"icc_2_k", "icc_3_k_agreement"}:
        denom = ms_r + (ms_c - ms_e) / n
        if denom == 0.0:
            raise InvalidDataError("Denominator for ICC(2,k) is zero; estimate is undefined.")
        return float((ms_r - ms_e) / denom)

    if variant in {"icc_3_1", "icc_2_1_consistency"}:
        denom = ms_r + (k - 1) * ms_e
        if denom == 0.0:
            raise InvalidDataError("Denominator for ICC(3,1) is zero; estimate is undefined.")
        return float((ms_r - ms_e) / denom)

    if variant in {"icc_3_k", "icc_2_k_consistency"}:
        if ms_r == 0.0:
            raise InvalidDataError("MS_targets is zero; ICC(3,k) is undefined.")
        return float((ms_r - ms_e) / ms_r)

    raise InvalidDataError(f"Unsupported ICC variant {variant!r}.")


def compute_icc_f_test(
    variant: str,
    n: int,
    k: int,
    anova: dict[str, Any],
) -> dict[str, Any]:
    """Compute the hypothesis test (F-statistic, dfs, p-value) for the ICC estimate."""
    if variant in {"icc_1_1", "icc_1_k"}:
        ms_num = anova["ms_between"]
        ms_den = anova["ms_within"]
        df1 = anova["df_between"]
        df2 = anova["df_within"]
    else:
        # Two-way models (consistency and absolute agreement under H0: ICC = 0)
        ms_num = anova["ms_targets"]
        ms_den = anova["ms_error"]
        df1 = anova["df_targets"]
        df2 = anova["df_error"]

    if ms_den <= 0.0:
        f_stat = float("inf") if ms_num > 0.0 else float("nan")
        p_val = 0.0 if ms_num > 0.0 else float("nan")
    else:
        f_stat = float(ms_num / ms_den)
        p_val = float(stats.f.sf(f_stat, df1, df2))

    return {
        "statistic": f_stat,
        "df1": df1,
        "df2": df2,
        "p_value": p_val,
        "null_hypothesis": "Population ICC is zero (target variance is zero).",
        "alternative_hypothesis": "Population ICC is greater than zero.",
    }


def compute_icc_ci(
    variant: str,
    n: int,
    k: int,
    anova: dict[str, Any],
    estimate: float,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compute exact or Satterthwaite analytical F-distribution confidence intervals.

    Follows Shrout & Fleiss (1979) and McGraw & Wong (1996) corrected formulas.
    """
    alpha = 1.0 - confidence_level

    # Degenerate: perfect agreement without error
    if anova["ms_error"] == 0.0 and variant not in {"icc_1_1", "icc_1_k"}:
        if estimate == 1.0:
            return {
                "quantity": "intraclass_correlation",
                "method": "exact_analytical_f",
                "level": confidence_level,
                "lower": 1.0,
                "upper": 1.0,
                "status": "available",
            }

    if variant == "icc_1_1":
        ms_b = anova["ms_between"]
        ms_w = anova["ms_within"]
        df1 = anova["df_between"]
        df2 = anova["df_within"]
        if ms_w <= 0.0:
            return {
                "quantity": "intraclass_correlation",
                "method": "exact_f_inversion",
                "level": confidence_level,
                "lower": estimate,
                "upper": estimate,
                "status": "available",
            }
        f_stat = ms_b / ms_w
        f_l = f_stat / stats.f.ppf(1.0 - alpha / 2.0, df1, df2)
        f_u = f_stat * stats.f.ppf(1.0 - alpha / 2.0, df2, df1)
        lower = (f_l - 1.0) / (f_l + k - 1.0)
        upper = (f_u - 1.0) / (f_u + k - 1.0)
        return {
            "quantity": "intraclass_correlation",
            "method": "exact_f_inversion",
            "level": confidence_level,
            "lower": float(lower),
            "upper": float(upper),
            "status": "available",
        }

    if variant == "icc_1_k":
        ms_b = anova["ms_between"]
        ms_w = anova["ms_within"]
        df1 = anova["df_between"]
        df2 = anova["df_within"]
        if ms_w <= 0.0 or ms_b <= 0.0:
            return {
                "quantity": "intraclass_correlation",
                "method": "exact_f_inversion",
                "level": confidence_level,
                "lower": estimate,
                "upper": estimate,
                "status": "available",
            }
        f_stat = ms_b / ms_w
        f_l = f_stat / stats.f.ppf(1.0 - alpha / 2.0, df1, df2)
        f_u = f_stat * stats.f.ppf(1.0 - alpha / 2.0, df2, df1)
        lower = 1.0 - 1.0 / f_l
        upper = 1.0 - 1.0 / f_u
        return {
            "quantity": "intraclass_correlation",
            "method": "exact_f_inversion",
            "level": confidence_level,
            "lower": float(lower),
            "upper": float(upper),
            "status": "available",
        }

    if variant in {"icc_3_1", "icc_2_1_consistency"}:
        ms_r = anova["ms_targets"]
        ms_e = anova["ms_error"]
        df1 = anova["df_targets"]
        df2 = anova["df_error"]
        if ms_e <= 0.0:
            return {
                "quantity": "intraclass_correlation",
                "method": "exact_f_inversion",
                "level": confidence_level,
                "lower": estimate,
                "upper": estimate,
                "status": "available",
            }
        f_stat = ms_r / ms_e
        f_l = f_stat / stats.f.ppf(1.0 - alpha / 2.0, df1, df2)
        f_u = f_stat * stats.f.ppf(1.0 - alpha / 2.0, df2, df1)
        lower = (f_l - 1.0) / (f_l + k - 1.0)
        upper = (f_u - 1.0) / (f_u + k - 1.0)
        return {
            "quantity": "intraclass_correlation",
            "method": "exact_f_inversion",
            "level": confidence_level,
            "lower": float(lower),
            "upper": float(upper),
            "status": "available",
        }

    if variant in {"icc_3_k", "icc_2_k_consistency"}:
        ms_r = anova["ms_targets"]
        ms_e = anova["ms_error"]
        df1 = anova["df_targets"]
        df2 = anova["df_error"]
        if ms_e <= 0.0 or ms_r <= 0.0:
            return {
                "quantity": "intraclass_correlation",
                "method": "exact_f_inversion",
                "level": confidence_level,
                "lower": estimate,
                "upper": estimate,
                "status": "available",
            }
        f_stat = ms_r / ms_e
        f_l = f_stat / stats.f.ppf(1.0 - alpha / 2.0, df1, df2)
        f_u = f_stat * stats.f.ppf(1.0 - alpha / 2.0, df2, df1)
        lower = 1.0 - 1.0 / f_l
        upper = 1.0 - 1.0 / f_u
        return {
            "quantity": "intraclass_correlation",
            "method": "exact_f_inversion",
            "level": confidence_level,
            "lower": float(lower),
            "upper": float(upper),
            "status": "available",
        }

    # Two-way random absolute agreement: ICC(2,1) and ICC(2,k)
    # Using McGraw & Wong (1996) / Shrout & Fleiss (1979) Satterthwaite approximation
    ms_r = anova["ms_targets"]
    ms_c = anova["ms_raters"]
    ms_e = anova["ms_error"]

    # Point estimate for single absolute agreement
    single_estimate = compute_icc_estimate("icc_2_1", n, k, anova)

    f_j = ms_c / ms_e if ms_e > 0.0 else 1.0
    term_main = (
        k * single_estimate * f_j + n * (1.0 + (k - 1.0) * single_estimate) - k * single_estimate
    )

    v_num = (k - 1.0) * (n - 1.0) * (term_main**2)
    v_den = (n - 1.0) * (k**2) * (single_estimate**2) * (f_j**2) + (
        n * (1.0 + (k - 1.0) * single_estimate) - k * single_estimate
    ) ** 2

    v = v_num / v_den if v_den > 0.0 else float((n - 1.0) * (k - 1.0))
    if not math.isfinite(v) or v <= 0.0:
        v = float((n - 1.0) * (k - 1.0))

    f_2u = stats.f.ppf(1.0 - alpha / 2.0, n - 1, v)
    f_2l = stats.f.ppf(1.0 - alpha / 2.0, v, n - 1)

    den_l = f_2u * (k * ms_c + (k * n - k - n) * ms_e) + n * ms_r
    den_u = k * ms_c + (k * n - k - n) * ms_e + n * f_2l * ms_r

    l_2_1 = float(n * (ms_r - f_2u * ms_e) / den_l) if den_l != 0.0 else float("nan")
    u_2_1 = float(n * (f_2l * ms_r - ms_e) / den_u) if den_u != 0.0 else float("nan")

    if variant in {"icc_2_1", "icc_3_1_agreement"}:
        if math.isfinite(l_2_1) and math.isfinite(u_2_1) and l_2_1 <= u_2_1:
            return {
                "quantity": "intraclass_correlation",
                "method": "satterthwaite_f_inversion",
                "level": confidence_level,
                "lower": l_2_1,
                "upper": u_2_1,
                "status": "available",
            }
        return {
            "quantity": "intraclass_correlation",
            "method": "satterthwaite_f_inversion",
            "level": confidence_level,
            "lower": None,
            "upper": None,
            "status": "unavailable",
        }

    # For average measures ICC(2,k), apply Spearman-Brown step from single-measure limits
    # L(k) = k * L / (1 + (k - 1) * L)
    den_lk = 1.0 + (k - 1.0) * l_2_1
    den_uk = 1.0 + (k - 1.0) * u_2_1

    l_2_k = float(k * l_2_1 / den_lk) if den_lk > 0.0 else float("nan")
    u_2_k = float(k * u_2_1 / den_uk) if den_uk > 0.0 else float("nan")

    if math.isfinite(l_2_k) and math.isfinite(u_2_k) and l_2_k <= u_2_k:
        return {
            "quantity": "intraclass_correlation",
            "method": "satterthwaite_f_inversion",
            "level": confidence_level,
            "lower": l_2_k,
            "upper": u_2_k,
            "status": "available",
        }

    return {
        "quantity": "intraclass_correlation",
        "method": "satterthwaite_f_inversion",
        "level": confidence_level,
        "lower": None,
        "upper": None,
        "status": "unavailable",
    }


def intraclass_correlation(
    frame: pd.DataFrame,
    target: str,
    rater: str,
    outcome: str | None = None,
    *,
    value: str | None = None,
    model: str = "two_way_random",
    definition: str | None = "absolute_agreement",
    unit: str = "single",
    confidence_level: float = 0.95,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Execute complete Intraclass Correlation Coefficient (ICC) analysis.

    Parameters
    ----------
    frame : pd.DataFrame
        DataFrame containing target IDs, rater IDs, and quantitative scores.
    target : str
        Column identifying the subject/target being rated.
    rater : str
        Column identifying the rater/judge.
    outcome : str, optional
        Column containing the quantitative numeric score.
    value : str, optional
        Alias for outcome column.
    model : {"one_way_random", "two_way_random", "two_way_mixed"}, default "two_way_random"
        Rater sampling model.
    definition : {"absolute_agreement", "consistency"}, default "absolute_agreement"
        Agreement definition.
    unit : {"single", "average"}, default "single"
        Unit of analysis.
    confidence_level : float, default 0.95
        Confidence level for analytical interval.
    alpha : float, default 0.05
        Significance level for F hypothesis test.

    Returns
    -------
    dict[str, Any]
        Structured dictionary containing ICC estimate, CI, ANOVA table,
        variance components, F-tests, sample accounting, assumptions, and warnings.
    """
    outcome_col = outcome if outcome is not None else value
    if outcome_col is None:
        raise InvalidDataError(
            "An outcome column containing quantitative ratings must be specified."
        )
    if model == "one_way_random" and definition is None:
        definition = "absolute_agreement"

    conf_level = _check_probability(confidence_level, "confidence_level")
    _check_probability(alpha, "alpha")

    variant = resolve_variant(model, definition, unit)
    variant_info = ICC_VARIANTS[variant]

    panel = icc_panel(frame, target, rater, outcome_col)
    matrix = panel["matrix"]
    n_targets = panel["n_targets"]
    n_raters = panel["n_raters"]

    anova = icc_anova_components(matrix)
    estimate = compute_icc_estimate(variant, n_targets, n_raters, anova)
    f_test = compute_icc_f_test(variant, n_targets, n_raters, anova)
    ci = compute_icc_ci(variant, n_targets, n_raters, anova, estimate, conf_level)

    # Compute all canonical variants for complete diagnostic transparency
    all_variants_results = []
    for var_key in ("icc_1_1", "icc_1_k", "icc_2_1", "icc_2_k", "icc_3_1", "icc_3_k"):
        var_meta = ICC_VARIANTS[var_key]
        var_est = compute_icc_estimate(var_key, n_targets, n_raters, anova)
        var_f = compute_icc_f_test(var_key, n_targets, n_raters, anova)
        var_ci = compute_icc_ci(var_key, n_targets, n_raters, anova, var_est, conf_level)
        all_variants_results.append(
            {
                "variant": var_key,
                "notation": var_meta["notation"],
                "mcgraw_wong_notation": var_meta["mcgraw_wong_notation"],
                "model": var_meta["model"],
                "definition": var_meta["definition"],
                "unit": var_meta["unit"],
                "description": var_meta["description"],
                "estimate": var_est,
                "confidence_interval": var_ci,
                "f_test": var_f,
            }
        )

    warnings: list[str] = []
    if panel["sample"]["excluded_targets"] > 0:
        warnings.append(
            f"{panel['sample']['excluded_targets']} target(s) had incomplete ratings across raters "
            f"and were excluded from the complete-target panel."
        )

    if estimate < 0.0:
        warnings.append(
            f"Observed {variant_info['notation']} sample estimate is negative ({estimate:.4f}). "
            "Negative sample ICC indicates within-target disagreement/error exceeds between-target "
            "variation; it is not clamped to zero."
        )

    if anova["ms_error"] == 0.0 and variant not in {"icc_1_1", "icc_1_k"}:
        warnings.append(
            "Residual mean square error is zero, indicating perfect consistency across raters."
        )

    assumptions = (
        "Targets/subjects are independent random samples from the target population.",
        f"Raters follow {variant_info['model'].replace('_', ' ')} design.",
        (
            f"Reliability evaluated under {variant_info['definition'].replace('_', ' ')} "
            f"for {variant_info['unit']} ratings."
        ),
        "Measurements follow additive two-way ANOVA structure with normally distributed errors.",
    )

    limitations = (
        "ICC reflects relative reliability (ratio of variances); "
        "it depends heavily on sample target heterogeneity.",
        "High consistency ICC does not imply absolute agreement; systematic additive rater "
        "differences are ignored by consistency.",
        "Average-measure ICC reflects the reliability of the mean of k ratings, "
        "not individual single ratings.",
        "Failing to reject the null hypothesis does not certify unacceptable reliability, "
        "nor does p < alpha guarantee practical adequacy.",
        "Sample size requirements depend on rater count, target variability, "
        "and true population agreement.",
    )

    # Rater test
    f_rater = (
        float(anova["ms_raters"] / anova["ms_error"]) if anova["ms_error"] > 0 else float("inf")
    )
    p_rater = (
        float(stats.f.sf(f_rater, anova["df_raters"], anova["df_error"]))
        if anova["ms_error"] > 0
        else 0.0
    )

    return {
        "method_id": "intraclass_correlation",
        "variant": variant,
        "notation": variant_info["notation"],
        "mcgraw_wong_notation": variant_info["mcgraw_wong_notation"],
        "model": variant_info["model"],
        "definition": variant_info["definition"],
        "unit": variant_info["unit"],
        "description": variant_info["description"],
        "estimate": estimate,
        "confidence_interval": ci,
        "f_test": f_test,
        "rater_test": {
            "statistic": f_rater,
            "df1": anova["df_raters"],
            "df2": anova["df_error"],
            "p_value": p_rater,
        },
        "n_targets": n_targets,
        "n_raters": n_raters,
        "average_k": n_raters,
        "anova_table": anova,
        "variance_components": anova["variance_components"],
        "sample": panel["sample"],
        "all_variants": all_variants_results,
        "assumptions": assumptions,
        "warnings": warnings,
        "limitations": limitations,
    }
