"""Independent reference mathematical formulas for PyAutoStat Phase 2 validation.

CRITICAL SCIENTIFIC PRINCIPLE:
This module does NOT import or reuse PyAutoStat internal calculation helpers.
All mathematical formulas are implemented independently to provide Level A
(independent reference) verification. Where direct SciPy library calls are used
for comparison, they are classified as Level B (external backend conformance).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


def compute_ranks(values: list[float] | np.ndarray) -> np.ndarray:
    """Compute 1-based fractional average ranks for tied values independently."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


# =============================================================================
# 1. ONE-SAMPLE T-TEST
# =============================================================================


def reference_one_sample_t(
    values: list[float] | np.ndarray,
    reference_value: float,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Level A independent manual calculation for one-sample t-test."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(np.sum(x) / n)
    diff = mean - float(reference_value)
    variance = float(np.sum((x - mean) ** 2) / (n - 1))
    sd = math.sqrt(variance)
    se = sd / math.sqrt(n)
    t_stat = diff / se if se > 0 else float("inf")
    df = n - 1
    alpha = 1.0 - confidence_level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df))
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se
    cohen_d = diff / sd if sd > 0 else 0.0

    # Level B backend cross-check
    scipy_res = stats.ttest_1samp(x, float(reference_value))
    backend_t = float(scipy_res.statistic)
    backend_p = float(scipy_res.pvalue)

    return {
        "n": n,
        "sample_mean": mean,
        "reference_value": float(reference_value),
        "mean_difference": diff,
        "sample_sd": sd,
        "standard_error": se,
        "degrees_of_freedom": df,
        "test_statistic": t_stat,
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohens_d": cohen_d,
        "backend_statistic": backend_t,
        "backend_p_value": backend_p,
    }


# =============================================================================
# 2. STUDENT'S TWO-SAMPLE T-TEST (EQUAL VARIANCE)
# =============================================================================


def reference_student_t(
    group1: list[float] | np.ndarray,
    group2: list[float] | np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Level A independent manual calculation for Student's two-sample t-test."""
    x1 = np.asarray(group1, dtype=float)
    x2 = np.asarray(group2, dtype=float)
    n1 = len(x1)
    n2 = len(x2)
    m1 = float(np.sum(x1) / n1)
    m2 = float(np.sum(x2) / n2)
    diff = m1 - m2
    v1 = float(np.sum((x1 - m1) ** 2) / (n1 - 1))
    v2 = float(np.sum((x2 - m2) ** 2) / (n2 - 1))
    df = n1 + n2 - 2
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    pooled_sd = math.sqrt(pooled_var)
    se = pooled_sd * math.sqrt(1.0 / n1 + 1.0 / n2)
    t_stat = diff / se if se > 0 else float("inf")
    alpha = 1.0 - confidence_level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df))
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se
    cohen_d = diff / pooled_sd if pooled_sd > 0 else 0.0

    # Level B backend cross-check
    scipy_res = stats.ttest_ind(x1, x2, equal_var=True)
    backend_t = float(scipy_res.statistic)
    backend_p = float(scipy_res.pvalue)

    return {
        "n1": n1,
        "n2": n2,
        "mean1": m1,
        "mean2": m2,
        "mean_difference": diff,
        "pooled_sd": pooled_sd,
        "standard_error": se,
        "degrees_of_freedom": df,
        "test_statistic": t_stat,
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohens_d": cohen_d,
        "backend_statistic": backend_t,
        "backend_p_value": backend_p,
    }


# =============================================================================
# 3. WELCH'S TWO-SAMPLE T-TEST (UNEQUAL VARIANCE)
# =============================================================================


def reference_welch_t(
    group1: list[float] | np.ndarray,
    group2: list[float] | np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Level A independent manual calculation for Welch's two-sample t-test."""
    x1 = np.asarray(group1, dtype=float)
    x2 = np.asarray(group2, dtype=float)
    n1 = len(x1)
    n2 = len(x2)
    m1 = float(np.sum(x1) / n1)
    m2 = float(np.sum(x2) / n2)
    diff = m1 - m2
    v1 = float(np.sum((x1 - m1) ** 2) / (n1 - 1))
    v2 = float(np.sum((x2 - m2) ** 2) / (n2 - 1))
    se_sq = (v1 / n1) + (v2 / n2)
    se = math.sqrt(se_sq)
    t_stat = diff / se if se > 0 else float("inf")

    # Welch-Satterthwaite degrees of freedom
    num = se_sq**2
    den = ((v1 / n1) ** 2 / (n1 - 1)) + ((v2 / n2) ** 2 / (n2 - 1))
    df = num / den

    alpha = 1.0 - confidence_level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df))
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se

    # Cohen's d uses pooled SD as specified in PyAutoStat contract
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    pooled_sd = math.sqrt(pooled_var)
    cohen_d = diff / pooled_sd if pooled_sd > 0 else 0.0

    # Level B backend cross-check
    scipy_res = stats.ttest_ind(x1, x2, equal_var=False)
    backend_t = float(scipy_res.statistic)
    backend_p = float(scipy_res.pvalue)

    return {
        "n1": n1,
        "n2": n2,
        "mean1": m1,
        "mean2": m2,
        "mean_difference": diff,
        "standard_error": se,
        "degrees_of_freedom": df,
        "test_statistic": t_stat,
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohens_d": cohen_d,
        "backend_statistic": backend_t,
        "backend_p_value": backend_p,
    }


# =============================================================================
# 4. PAIRED T-TEST
# =============================================================================


def reference_paired_t(
    first: list[float] | np.ndarray,
    second: list[float] | np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Level A independent manual calculation for paired-samples t-test."""
    x1 = np.asarray(first, dtype=float)
    x2 = np.asarray(second, dtype=float)
    d = x1 - x2
    n = len(d)
    mean_diff = float(np.sum(d) / n)
    var_d = float(np.sum((d - mean_diff) ** 2) / (n - 1))
    sd_d = math.sqrt(var_d)
    se = sd_d / math.sqrt(n)
    t_stat = mean_diff / se if se > 0 else float("inf")
    df = n - 1
    alpha = 1.0 - confidence_level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df))
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se
    cohen_dz = mean_diff / sd_d if sd_d > 0 else 0.0

    # Level B backend cross-check
    scipy_res = stats.ttest_rel(x1, x2)
    backend_t = float(scipy_res.statistic)
    backend_p = float(scipy_res.pvalue)

    return {
        "complete_pairs": n,
        "mean_paired_difference": mean_diff,
        "sd_differences": sd_d,
        "standard_error": se,
        "degrees_of_freedom": df,
        "test_statistic": t_stat,
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohens_dz": cohen_dz,
        "backend_statistic": backend_t,
        "backend_p_value": backend_p,
    }


# =============================================================================
# 5. MANN-WHITNEY U TEST
# =============================================================================


def reference_mann_whitney_u(
    group1: list[float] | np.ndarray,
    group2: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Level A independent manual calculation for Mann-Whitney U and rank-biserial."""
    x1 = np.asarray(group1, dtype=float)
    x2 = np.asarray(group2, dtype=float)
    n1 = len(x1)
    n2 = len(x2)
    combined = np.concatenate([x1, x2])
    ranks = compute_ranks(combined)
    r1_sum = float(np.sum(ranks[:n1]))
    u1 = float(r1_sum - n1 * (n1 + 1) / 2.0)
    u2 = float(n1 * n2 - u1)
    rank_biserial = float((2.0 * u1) / (n1 * n2) - 1.0)

    # Level B backend cross-check
    scipy_res = stats.mannwhitneyu(x1, x2, alternative="two-sided")
    backend_u = float(scipy_res.statistic)
    backend_p = float(scipy_res.pvalue)

    return {
        "n1": n1,
        "n2": n2,
        "u1": u1,
        "u2": u2,
        "rank_sum_1": r1_sum,
        "rank_biserial": rank_biserial,
        "backend_statistic": backend_u,
        "backend_p_value": backend_p,
    }


# =============================================================================
# 6. WILCOXON SIGNED-RANK TEST
# =============================================================================


def reference_wilcoxon_signed_rank(
    first: list[float] | np.ndarray,
    second: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Level A independent signed-rank bookkeeping with 'wilcox' zero policy."""
    x1 = np.asarray(first, dtype=float)
    x2 = np.asarray(second, dtype=float)
    d = x1 - x2
    n_pairs = len(d)
    zero_mask = d == 0
    zero_count = int(np.sum(zero_mask))
    nonzero_d = d[~zero_mask]
    n_nonzero = len(nonzero_d)

    abs_d = np.abs(nonzero_d)
    ranks = compute_ranks(abs_d)
    w_plus = float(np.sum(ranks[nonzero_d > 0]))
    w_minus = float(np.sum(ranks[nonzero_d < 0]))
    w_sum = w_plus + w_minus
    rank_biserial = (w_plus - w_minus) / w_sum if w_sum > 0 else 0.0

    # Level B backend cross-check
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scipy_res = stats.wilcoxon(
            d, zero_method="wilcox", correction=False, alternative="two-sided"
        )
    backend_stat = float(scipy_res.statistic)
    backend_p = float(scipy_res.pvalue)

    return {
        "complete_pairs": n_pairs,
        "zero_differences": zero_count,
        "nonzero_differences": n_nonzero,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "rank_biserial": rank_biserial,
        "backend_statistic": backend_stat,
        "backend_p_value": backend_p,
    }


# =============================================================================
# 7. PEARSON CORRELATION
# =============================================================================


def reference_pearson_correlation(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Level A independent manual covariance/SD and Fisher-z calculation."""
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    n = len(arr_x)
    mx = float(np.sum(arr_x) / n)
    my = float(np.sum(arr_y) / n)
    dx = arr_x - mx
    dy = arr_y - my
    ss_x = float(np.sum(dx**2))
    ss_y = float(np.sum(dy**2))
    ss_xy = float(np.sum(dx * dy))
    r = ss_xy / math.sqrt(ss_x * ss_y)

    # t relation and p-value
    df = n - 2
    t_stat = r * math.sqrt(df / (1.0 - r**2)) if abs(r) < 1.0 else float("inf")
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df)) if abs(r) < 1.0 else 0.0

    # Fisher z confidence interval
    z = 0.5 * math.log((1.0 + r) / (1.0 - r)) if abs(r) < 1.0 else 0.0
    se_z = 1.0 / math.sqrt(n - 3) if n > 3 else 0.0
    alpha = 1.0 - confidence_level
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_low = z - z_crit * se_z
    z_high = z + z_crit * se_z
    ci_lower = math.tanh(z_low)
    ci_upper = math.tanh(z_high)

    # Level B backend cross-check
    scipy_res = stats.pearsonr(arr_x, arr_y)
    backend_r = float(scipy_res.statistic)
    backend_p = float(scipy_res.pvalue)

    return {
        "n": n,
        "degrees_of_freedom": df,
        "pearson_r": r,
        "t_statistic": t_stat,
        "p_value": p_val,
        "fisher_z": z,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "backend_r": backend_r,
        "backend_p_value": backend_p,
    }


# =============================================================================
# 8. SPEARMAN CORRELATION
# =============================================================================


def reference_spearman_correlation(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Level A independent average ranking with Pearson formula on ranks."""
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    rx = compute_ranks(arr_x)
    ry = compute_ranks(arr_y)
    n = len(rx)
    mx = float(np.sum(rx) / n)
    my = float(np.sum(ry) / n)
    dx = rx - mx
    dy = ry - my
    ss_x = float(np.sum(dx**2))
    ss_y = float(np.sum(dy**2))
    ss_xy = float(np.sum(dx * dy))
    rho = ss_xy / math.sqrt(ss_x * ss_y)

    # Level B backend cross-check
    scipy_res = stats.spearmanr(arr_x, arr_y)
    backend_rho = float(scipy_res.statistic)
    backend_p = float(scipy_res.pvalue)

    return {
        "n": n,
        "spearman_rho": rho,
        "backend_statistic": backend_rho,
        "backend_p_value": backend_p,
    }


# =============================================================================
# 9. PEARSON CHI-SQUARE TEST OF INDEPENDENCE
# =============================================================================


def reference_chi_square(
    observed_table: list[list[int]] | np.ndarray,
) -> dict[str, Any]:
    """Level A independent expected counts, chi-square, df, and Cramer's V."""
    obs = np.asarray(observed_table, dtype=float)
    r, c = obs.shape
    row_sums = np.sum(obs, axis=1, keepdims=True)
    col_sums = np.sum(obs, axis=0, keepdims=True)
    total_n = float(np.sum(obs))
    expected = (row_sums @ col_sums) / total_n
    chi2 = float(np.sum((obs - expected) ** 2 / expected))
    df = (r - 1) * (c - 1)
    p_val = float(stats.chi2.sf(chi2, df))
    min_dim = min(r - 1, c - 1)
    cramers_v = math.sqrt(chi2 / (total_n * min_dim)) if min_dim > 0 and total_n > 0 else 0.0

    # Level B backend cross-check
    scipy_chi2, scipy_p, scipy_df, scipy_exp = stats.chi2_contingency(obs, correction=False)

    return {
        "total_n": int(total_n),
        "degrees_of_freedom": df,
        "expected_counts": expected.tolist(),
        "chi_square": chi2,
        "p_value": p_val,
        "cramers_v": cramers_v,
        "backend_chi2": float(scipy_chi2),
        "backend_p_value": float(scipy_p),
        "backend_df": int(scipy_df),
        "backend_expected": scipy_exp.tolist(),
    }


# =============================================================================
# 10. FISHER'S EXACT TEST
# =============================================================================


def reference_fisher_exact(
    table_2x2: list[list[int]] | np.ndarray,
) -> dict[str, Any]:
    """Level A odds ratio formula and Level B hypergeometric p-value cross-check."""
    t = np.asarray(table_2x2, dtype=float)
    a, b = t[0, 0], t[0, 1]
    c, d = t[1, 0], t[1, 1]
    total_n = int(a + b + c + d)
    odds_ratio = (a * d) / (b * c) if (b * c) > 0 else None

    # Level B backend cross-check
    scipy_res = stats.fisher_exact(table_2x2, alternative="two-sided")
    backend_or = float(scipy_res.statistic)
    backend_p = float(scipy_res.pvalue)

    return {
        "total_n": total_n,
        "odds_ratio": odds_ratio,
        "backend_odds_ratio": backend_or,
        "backend_p_value": backend_p,
    }
