"""Independent reference mathematical formulas for scale reliability and agreement.

Covers:
23. cronbach_alpha (item variances, total variance, corrected item-total, alpha-if-deleted)
24. intraclass_correlation (all 6 Shrout & Fleiss / McGraw & Wong ICC configurations)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

# =============================================================================
# 23. CRONBACH'S ALPHA
# =============================================================================


def reference_cronbach_alpha(
    items_matrix: np.ndarray,
    item_names: list[str],
) -> dict[str, Any]:
    """Level A independent manual calculation for Cronbach's alpha."""
    mat = np.asarray(items_matrix, dtype=float)
    n_respondents, k_items = mat.shape

    # Item variances (ddof=1)
    item_vars = [float(np.var(mat[:, j], ddof=1)) for j in range(k_items)]
    sum_item_vars = sum(item_vars)

    # Total scale score per respondent
    totals = np.sum(mat, axis=1)
    total_var = float(np.var(totals, ddof=1))

    # Cronbach's alpha
    alpha = (
        (k_items / (k_items - 1)) * (1.0 - sum_item_vars / total_var)
        if (total_var > 0 and k_items > 1)
        else 0.0
    )

    # Corrected item-total correlations & alpha-if-deleted
    item_stats = []
    for j in range(k_items):
        item_col = mat[:, j]
        rest_total = totals - item_col

        # Corrected item-total correlation: Pearson r between item and rest_total
        diff_item = item_col - np.mean(item_col)
        diff_rest = rest_total - np.mean(rest_total)
        cov_ir = np.sum(diff_item * diff_rest) / (n_respondents - 1)
        s_item = math.sqrt(item_vars[j])
        s_rest = math.sqrt(float(np.var(rest_total, ddof=1)))
        citc = cov_ir / (s_item * s_rest) if (s_item > 0 and s_rest > 0) else 0.0

        # Alpha if deleted
        if k_items > 2:
            sum_vars_del = sum_item_vars - item_vars[j]
            var_rest = s_rest**2
            alpha_del = (
                ((k_items - 1) / (k_items - 2)) * (1.0 - sum_vars_del / var_rest)
                if var_rest > 0
                else 0.0
            )
        else:
            alpha_del = None

        item_stats.append(
            {
                "item": item_names[j],
                "item_variance": item_vars[j],
                "corrected_item_total_correlation": citc,
                "alpha_if_deleted": alpha_del,
            }
        )

    # Inter-item correlation matrix
    corr_matrix = np.corrcoef(mat, rowvar=False)

    return {
        "n_respondents": n_respondents,
        "k_items": k_items,
        "item_variances": item_vars,
        "sum_item_variances": sum_item_vars,
        "total_variance": total_var,
        "cronbach_alpha": alpha,
        "item_statistics": item_stats,
        "inter_item_correlations": corr_matrix.tolist(),
    }


# =============================================================================
# 24. INTRACLASS CORRELATION (ALL 6 ICC CONFIGURATIONS)
# =============================================================================


def reference_icc(
    ratings_matrix: np.ndarray,
) -> dict[str, Any]:
    """Level A independent manual calculation for all 6 Shrout & Fleiss ICC configurations.

    Input shape: (n_targets, k_raters).
    """
    y = np.asarray(ratings_matrix, dtype=float)
    n, k = y.shape  # n targets, k raters

    grand_mean = float(np.mean(y))
    target_means = np.mean(y, axis=1)  # shape (n,)
    rater_means = np.mean(y, axis=0)  # shape (k,)

    ss_targets = float(k * np.sum((target_means - grand_mean) ** 2))
    ss_raters = float(n * np.sum((rater_means - grand_mean) ** 2))
    ss_error = float(
        np.sum((y - target_means[:, np.newaxis] - rater_means[np.newaxis, :] + grand_mean) ** 2)
    )

    ss_within = ss_raters + ss_error

    df_targets = n - 1
    df_raters = k - 1
    df_error = (n - 1) * (k - 1)
    df_within = n * (k - 1)

    # Mean squares
    bms = ss_targets / df_targets if df_targets > 0 else 0.0  # Between-targets MS
    jms = ss_raters / df_raters if df_raters > 0 else 0.0  # Between-raters / Judges MS
    ems = ss_error / df_error if df_error > 0 else 0.0  # Residual error MS
    wms = ss_within / df_within if df_within > 0 else 0.0  # Within-targets MS

    # Six ICC configurations:
    # 1. ICC(1,1) One-way random, single measure, absolute agreement
    icc_1_1 = (bms - wms) / (bms + (k - 1) * wms) if (bms + (k - 1) * wms) != 0 else 0.0
    f_1 = bms / wms if wms > 0 else 0.0
    p_1 = float(stats.f.sf(f_1, df_targets, df_within))

    # 2. ICC(1,k) One-way random, average measure, absolute agreement
    icc_1_k = (bms - wms) / bms if bms != 0 else 0.0

    # 3. ICC(2,1) Two-way random, single measure, absolute agreement
    denom_2_1 = bms + (k - 1) * ems + (k / n) * (jms - ems)
    icc_2_1 = (bms - ems) / denom_2_1 if denom_2_1 != 0 else 0.0
    f_2 = bms / ems if ems > 0 else 0.0
    p_2 = float(stats.f.sf(f_2, df_targets, df_error))

    # 4. ICC(2,k) Two-way random, average measure, absolute agreement
    denom_2_k = bms + (jms - ems) / n
    icc_2_k = (bms - ems) / denom_2_k if denom_2_k != 0 else 0.0

    # 5. ICC(3,1) Two-way mixed, single measure, consistency
    denom_3_1 = bms + (k - 1) * ems
    icc_3_1 = (bms - ems) / denom_3_1 if denom_3_1 != 0 else 0.0

    # 6. ICC(3,k) Two-way mixed, average measure, consistency
    icc_3_k = (bms - ems) / bms if bms != 0 else 0.0

    return {
        "n_targets": n,
        "k_raters": k,
        "grand_mean": grand_mean,
        "bms": bms,
        "jms": jms,
        "ems": ems,
        "wms": wms,
        "df_targets": df_targets,
        "df_raters": df_raters,
        "df_error": df_error,
        "df_within": df_within,
        "icc_1_1": icc_1_1,
        "icc_1_k": icc_1_k,
        "icc_2_1": icc_2_1,
        "icc_2_k": icc_2_k,
        "icc_3_1": icc_3_1,
        "icc_3_k": icc_3_k,
        "f_oneway": f_1,
        "p_oneway": p_1,
        "f_twoway": f_2,
        "p_twoway": p_2,
    }
