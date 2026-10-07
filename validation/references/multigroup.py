"""Independent reference mathematical formulas for multigroup PyAutoStat methods.

Covers:
11. welch_anova (+ Games-Howell pairwise)
12. one_way_anova (+ Tukey-Kramer pairwise)
13. kruskal_wallis (+ Dunn-Holm pairwise)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats
from validation.references.foundational import compute_ranks


def adjust_holm(p_values: list[float]) -> list[float]:
    """Independent implementation of the Holm (1979) step-down p-value adjustment."""
    m = len(p_values)
    if m <= 1:
        return list(p_values)

    # Sort indices by raw p-value
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted: list[tuple[int, float]] = []

    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        multiplier = m - rank
        adj = min(1.0, p_val * multiplier)
        adj = max(running_max, adj)
        running_max = adj
        adjusted.append((orig_idx, adj))

    # Restore original order
    adjusted.sort(key=lambda x: x[0])
    return [p for _, p in adjusted]


# =============================================================================
# 11. WELCH'S ANOVA AND GAMES-HOWELL
# =============================================================================


def reference_welch_anova(
    groups: list[list[float] | np.ndarray],
    group_labels: list[str],
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Level A independent manual calculation for Welch ANOVA and Games-Howell."""
    k = len(groups)
    sizes = [len(g) for g in groups]
    means = [float(np.mean(g)) for g in groups]
    variances = [float(np.var(g, ddof=1)) for g in groups]
    sds = [math.sqrt(v) for v in variances]

    weights = [n / v for n, v in zip(sizes, variances, strict=True)]
    w_sum = sum(weights)
    weighted_mean = sum(w * m for w, m in zip(weights, means, strict=True)) / w_sum

    lambda_term = sum(
        ((1.0 - w / w_sum) ** 2) / (n - 1) for w, n in zip(weights, sizes, strict=True)
    )

    ss_num = sum(w * (m - weighted_mean) ** 2 for w, m in zip(weights, means, strict=True))
    f_num = ss_num / (k - 1)
    f_denom = 1.0 + (2.0 * (k - 2) / (k**2 - 1)) * lambda_term
    f_stat = f_num / f_denom

    df1 = float(k - 1)
    df2 = float((k**2 - 1) / (3.0 * lambda_term))
    p_val = float(stats.f.sf(f_stat, df1, df2))

    # Games-Howell pairwise
    pairwise = []
    for i in range(k):
        for j in range(i + 1, k):
            g1, g2 = group_labels[i], group_labels[j]
            m1, m2 = means[i], means[j]
            v1, v2 = variances[i], variances[j]
            n1, n2 = sizes[i], sizes[j]
            diff = m1 - m2
            se = math.sqrt(v1 / n1 + v2 / n2)
            gh_df = ((v1 / n1 + v2 / n2) ** 2) / (
                ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
            )
            q_stat = abs(diff) * math.sqrt(2.0) / se
            p_gh = float(stats.studentized_range.sf(q_stat, k, gh_df))
            pairwise.append(
                {
                    "group1": g1,
                    "group2": g2,
                    "mean_difference": diff,
                    "standard_error": se,
                    "degrees_of_freedom": gh_df,
                    "q_statistic": q_stat,
                    "p_value": p_gh,
                }
            )

    return {
        "k": k,
        "group_sizes": sizes,
        "group_means": means,
        "group_variances": variances,
        "group_sds": sds,
        "weights": weights,
        "weighted_mean": weighted_mean,
        "f_statistic": f_stat,
        "df1": df1,
        "df2": df2,
        "p_value": p_val,
        "pairwise": pairwise,
    }


# =============================================================================
# 12. CLASSICAL ONE-WAY ANOVA AND TUKEY-KRAMER
# =============================================================================


def reference_one_way_anova(
    groups: list[list[float] | np.ndarray],
    group_labels: list[str],
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Level A independent manual calculation for One-Way ANOVA and Tukey-Kramer."""
    k = len(groups)
    sizes = [len(g) for g in groups]
    n_total = sum(sizes)
    means = [float(np.mean(g)) for g in groups]

    all_vals = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    grand_mean = float(np.mean(all_vals))

    ss_between = sum(n * (m - grand_mean) ** 2 for n, m in zip(sizes, means, strict=True))
    ss_within = sum(
        float(np.sum((np.asarray(g) - m) ** 2)) for g, m in zip(groups, means, strict=True)
    )
    ss_total = float(np.sum((all_vals - grand_mean) ** 2))

    df_between = k - 1
    df_within = n_total - k
    df_total = n_total - 1

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    f_stat = ms_between / ms_within if ms_within > 0 else float("inf")
    p_val = float(stats.f.sf(f_stat, df_between, df_within))

    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    # Tukey-Kramer pairwise
    pairwise = []
    for i in range(k):
        for j in range(i + 1, k):
            g1, g2 = group_labels[i], group_labels[j]
            m1, m2 = means[i], means[j]
            n1, n2 = sizes[i], sizes[j]
            diff = m1 - m2
            se_tk = math.sqrt(ms_within * (1.0 / n1 + 1.0 / n2))
            q_stat = math.sqrt(2.0) * abs(diff) / se_tk if se_tk > 0 else 0.0
            p_tk = float(stats.studentized_range.sf(q_stat, k, df_within))
            pairwise.append(
                {
                    "group1": g1,
                    "group2": g2,
                    "mean_difference": diff,
                    "standard_error": se_tk,
                    "q_statistic": q_stat,
                    "p_value": p_tk,
                }
            )

    # SciPy one-way ANOVA cross-check (Level B)
    scipy_f, scipy_p = stats.f_oneway(*groups)

    return {
        "k": k,
        "n_total": n_total,
        "grand_mean": grand_mean,
        "group_means": means,
        "group_sizes": sizes,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,
        "df_between": df_between,
        "df_within": df_within,
        "df_total": df_total,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "f_statistic": f_stat,
        "p_value": p_val,
        "eta_squared": eta_squared,
        "pairwise": pairwise,
        "backend_statistic": float(scipy_f),
        "backend_p_value": float(scipy_p),
    }


# =============================================================================
# 13. KRUSKAL-WALLIS AND DUNN-HOLM
# =============================================================================


def reference_kruskal_wallis(
    groups: list[list[float] | np.ndarray],
    group_labels: list[str],
) -> dict[str, Any]:
    """Level A independent manual calculation for Kruskal-Wallis and Dunn-Holm."""
    k = len(groups)
    sizes = [len(g) for g in groups]
    n_total = sum(sizes)

    all_vals = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    ranks = compute_ranks(all_vals)

    # Split ranks back to groups
    group_ranks: list[np.ndarray] = []
    idx = 0
    for n in sizes:
        group_ranks.append(ranks[idx : idx + n])
        idx += n

    rank_sums = [float(np.sum(gr)) for gr in group_ranks]
    mean_ranks = [rs / n for rs, n in zip(rank_sums, sizes, strict=True)]

    # Ties accounting
    _, counts = np.unique(all_vals, return_counts=True)
    tie_counts = counts[counts > 1]
    tie_correction = 1.0 - float(np.sum(tie_counts**3 - tie_counts)) / float(n_total**3 - n_total)

    h_num = (12.0 / (n_total * (n_total + 1))) * sum(
        (rs**2) / n for rs, n in zip(rank_sums, sizes, strict=True)
    ) - 3.0 * (n_total + 1)
    h_stat = h_num / tie_correction if tie_correction > 0 else h_num
    df = k - 1
    p_val = float(stats.chi2.sf(h_stat, df))

    epsilon_squared = (h_stat - k + 1) / (n_total - k) if (n_total > k) else 0.0

    # Dunn pairwise tests
    tie_term = (
        float(np.sum(tie_counts**3 - tie_counts)) / (12.0 * (n_total - 1)) if n_total > 1 else 0.0
    )
    pooled_se_base = (n_total * (n_total + 1)) / 12.0 - tie_term

    pairwise = []
    raw_p_values = []
    for i in range(k):
        for j in range(i + 1, k):
            g1, g2 = group_labels[i], group_labels[j]
            mr1, mr2 = mean_ranks[i], mean_ranks[j]
            n1, n2 = sizes[i], sizes[j]
            diff = mr1 - mr2
            se_dunn = math.sqrt(pooled_se_base * (1.0 / n1 + 1.0 / n2))
            z_stat = diff / se_dunn if se_dunn > 0 else 0.0
            p_dunn = float(2.0 * stats.norm.sf(abs(z_stat)))
            raw_p_values.append(p_dunn)
            pairwise.append(
                {
                    "group1": g1,
                    "group2": g2,
                    "mean_rank_difference": diff,
                    "standard_error": se_dunn,
                    "z_statistic": z_stat,
                    "raw_p_value": p_dunn,
                }
            )

    # Holm adjustment
    adj_p_values = adjust_holm(raw_p_values)
    for pw, adj_p in zip(pairwise, adj_p_values, strict=True):
        pw["adjusted_p_value"] = adj_p

    # SciPy Kruskal-Wallis cross-check (Level B)
    scipy_h, scipy_p = stats.kruskal(*groups)

    return {
        "k": k,
        "n_total": n_total,
        "group_sizes": sizes,
        "rank_sums": rank_sums,
        "mean_ranks": mean_ranks,
        "tie_correction": tie_correction,
        "h_statistic": h_stat,
        "degrees_of_freedom": df,
        "p_value": p_val,
        "epsilon_squared": epsilon_squared,
        "pairwise": pairwise,
        "backend_statistic": float(scipy_h),
        "backend_p_value": float(scipy_p),
    }
