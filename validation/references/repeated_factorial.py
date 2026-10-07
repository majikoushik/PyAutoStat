"""Independent reference mathematical formulas for repeated-measures and factorial ANOVA.

Covers:
14. repeated_measures_anova (+ Greenhouse-Geisser & Mauchly sphericity)
15. friedman_test (+ Kendall's W & pairwise Wilcoxon-Holm)
16. two_way_anova (Balanced Factorial, Type II, and Type III sums of squares)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy import stats
from validation.references.foundational import compute_ranks
from validation.references.multigroup import adjust_holm

# =============================================================================
# 14. REPEATED-MEASURES ANOVA
# =============================================================================


def reference_helmert_contrasts(k: int) -> np.ndarray:
    """Construct an orthonormal k x (k-1) Helmert contrast matrix independently."""
    matrix = np.zeros((k, k - 1), dtype=float)
    for j in range(1, k):
        scale = 1.0 / math.sqrt(j * (j + 1))
        matrix[:j, j - 1] = scale
        matrix[j, j - 1] = -float(j) * scale
    return matrix


def reference_repeated_measures_anova(
    panel: np.ndarray,
    condition_labels: list[str],
) -> dict[str, Any]:
    """Level A independent manual calculation for repeated-measures ANOVA."""
    n, k = panel.shape  # n subjects, k conditions

    grand_mean = float(np.mean(panel))
    unit_means = np.mean(panel, axis=1)  # shape (n,)
    cond_means = np.mean(panel, axis=0)  # shape (k,)

    ss_total = float(np.sum((panel - grand_mean) ** 2))
    ss_subject = float(k * np.sum((unit_means - grand_mean) ** 2))
    ss_condition = float(n * np.sum((cond_means - grand_mean) ** 2))
    ss_error = float(
        np.sum((panel - cond_means[np.newaxis, :] - unit_means[:, np.newaxis] + grand_mean) ** 2)
    )

    df_condition = k - 1
    df_error = (n - 1) * (k - 1)
    df_subject = n - 1

    ms_condition = ss_condition / df_condition
    ms_error = ss_error / df_error

    f_stat = ms_condition / ms_error
    p_uncorrected = float(stats.f.sf(f_stat, df_condition, df_error))
    partial_eta_sq = ss_condition / (ss_condition + ss_error)

    # Orthonormal Helmert contrast space for sphericity
    cov_matrix = np.cov(panel, rowvar=False, ddof=1)
    m_helmert = reference_helmert_contrasts(k)
    sigma_c = m_helmert.T @ cov_matrix @ m_helmert  # shape (k-1, k-1)

    tr_c = float(np.trace(sigma_c))
    tr_c2 = float(np.trace(sigma_c @ sigma_c))
    det_c = float(np.linalg.det(sigma_c))

    # Greenhouse-Geisser epsilon
    lower_bound_eps = 1.0 / (k - 1)
    raw_eps = (tr_c**2) / ((k - 1) * tr_c2) if tr_c2 > 0 else lower_bound_eps
    epsilon_gg = float(min(1.0, max(lower_bound_eps, raw_eps)))
    corrected_df_num = df_condition * epsilon_gg
    corrected_df_den = df_error * epsilon_gg
    p_corrected = float(stats.f.sf(f_stat, corrected_df_num, corrected_df_den))

    # Mauchly's sphericity test (Box / Anderson approximation)
    d = k - 1
    df_mauchly = (d * (d + 1)) // 2 - 1
    mean_diag = tr_c / d
    w_mauchly = float(det_c / (mean_diag**d)) if mean_diag > 0 else 0.0
    w_mauchly = float(min(1.0, max(0.0, w_mauchly)))

    df_resid = n - 1
    d_factor = 1.0 - (2.0 * d**2 + d + 2.0) / (6.0 * d * df_resid)
    chi2_mauchly = float(-df_resid * d_factor * np.log(max(w_mauchly, 1e-15)))
    p_mauchly = float(stats.chi2.sf(chi2_mauchly, df_mauchly))

    return {
        "n_subjects": n,
        "k_conditions": k,
        "grand_mean": grand_mean,
        "condition_means": cond_means.tolist(),
        "unit_means": unit_means.tolist(),
        "ss_total": ss_total,
        "ss_condition": ss_condition,
        "ss_subject": ss_subject,
        "ss_error": ss_error,
        "df_condition": df_condition,
        "df_error": df_error,
        "df_subject": df_subject,
        "ms_condition": ms_condition,
        "ms_error": ms_error,
        "f_statistic": f_stat,
        "p_uncorrected": p_uncorrected,
        "partial_eta_squared": partial_eta_sq,
        "epsilon_gg": epsilon_gg,
        "corrected_df_num": corrected_df_num,
        "corrected_df_den": corrected_df_den,
        "p_corrected": p_corrected,
        "w_mauchly": w_mauchly,
        "chi2_mauchly": chi2_mauchly,
        "df_mauchly": df_mauchly,
        "p_mauchly": p_mauchly,
    }


# =============================================================================
# 15. FRIEDMAN TEST AND KENDALL'S W
# =============================================================================


def reference_friedman_test(
    panel: np.ndarray,
    condition_labels: list[str],
) -> dict[str, Any]:
    """Level A independent manual rank calculation for Friedman test."""
    n, k = panel.shape  # n subjects, k conditions

    # Rank within each subject
    ranked_rows = np.empty_like(panel, dtype=float)
    tie_adjustment_sum = 0.0
    for s in range(n):
        row = panel[s, :]
        r = compute_ranks(row)
        ranked_rows[s, :] = r
        _, counts = np.unique(row, return_counts=True)
        ties = counts[counts > 1]
        tie_adjustment_sum += float(np.sum(ties**3 - ties))

    # Condition rank sums
    col_rank_sums = np.sum(ranked_rows, axis=0)

    # Q statistic
    q_num = (12.0 / (n * k * (k + 1))) * float(np.sum(col_rank_sums**2)) - 3.0 * n * (k + 1)
    tie_denom = 1.0 - tie_adjustment_sum / float(n * (k**3 - k)) if (k > 1) else 1.0
    q_stat = q_num / tie_denom if (tie_denom > 0) else q_num

    df = k - 1
    p_val = float(stats.chi2.sf(q_stat, df))

    # Kendall's W = Q / (n * (k - 1))
    kendall_w = q_stat / (n * (k - 1)) if (n * (k - 1) > 0) else 0.0
    kendall_w = float(min(1.0, max(0.0, kendall_w)))

    # Pairwise Wilcoxon matched-pairs with Holm adjustment
    pairwise = []
    raw_p_values = []
    for i in range(k):
        for j in range(i + 1, k):
            c1, c2 = condition_labels[i], condition_labels[j]
            diffs = panel[:, i] - panel[:, j]
            non_zero = diffs[diffs != 0]
            if len(non_zero) > 0:
                abs_diffs = np.abs(non_zero)
                ranks = compute_ranks(abs_diffs)
                w_plus = float(np.sum(ranks[non_zero > 0]))
                w_minus = float(np.sum(ranks[non_zero < 0]))
                total_w = w_plus + w_minus
                pair_stat = min(w_plus, w_minus)
                rank_biserial = (w_plus - w_minus) / total_w if total_w > 0 else 0.0
                scipy_res = stats.wilcoxon(
                    panel[:, i], panel[:, j], zero_method="wilcox", alternative="two-sided"
                )
                p_pw = float(scipy_res.pvalue)
            else:
                pair_stat = 0.0
                rank_biserial = 0.0
                p_pw = 1.0
            raw_p_values.append(p_pw)
            pairwise.append(
                {
                    "condition1": c1,
                    "condition2": c2,
                    "statistic": pair_stat,
                    "rank_biserial": rank_biserial,
                    "raw_p_value": p_pw,
                }
            )

    adj_p_values = adjust_holm(raw_p_values)
    for pw, adj_p in zip(pairwise, adj_p_values, strict=True):
        pw["adjusted_p_value"] = adj_p

    # SciPy Friedman cross-check (Level B)
    scipy_res = stats.friedmanchisquare(*[panel[:, c] for c in range(k)])

    return {
        "n_subjects": n,
        "k_conditions": k,
        "condition_rank_sums": col_rank_sums.tolist(),
        "q_statistic": q_stat,
        "degrees_of_freedom": df,
        "p_value": p_val,
        "kendall_w": kendall_w,
        "pairwise": pairwise,
        "backend_statistic": float(scipy_res.statistic),
        "backend_p_value": float(scipy_res.pvalue),
    }


# =============================================================================
# 16. TWO-WAY FACTORIAL ANOVA (BALANCED, TYPE II, TYPE III)
# =============================================================================


def reference_two_way_anova_balanced(
    data: list[tuple[str, str, float]],
    factor_a_levels: list[str],
    factor_b_levels: list[str],
) -> dict[str, Any]:
    """Level A independent manual calculation for balanced 2-way factorial ANOVA."""
    a_levels = list(factor_a_levels)
    b_levels = list(factor_b_levels)
    a = len(a_levels)
    b = len(b_levels)

    # Collect cell observations
    cell_data: dict[tuple[str, str], list[float]] = {
        (la, lb): [] for la in a_levels for lb in b_levels
    }
    for fa, fb, y in data:
        cell_data[(fa, fb)].append(y)

    n_cell = len(cell_data[(a_levels[0], b_levels[0])])
    # Verify balance
    for cell_vals in cell_data.values():
        if len(cell_vals) != n_cell:
            raise ValueError("reference_two_way_anova_balanced requires equal cell sizes")

    n_total = a * b * n_cell

    # Cell means
    cell_means = {k: float(np.mean(v)) for k, v in cell_data.items()}

    # Marginal means
    marginal_a = {la: float(np.mean([cell_means[(la, lb)] for lb in b_levels])) for la in a_levels}
    marginal_b = {lb: float(np.mean([cell_means[(la, lb)] for la in a_levels])) for lb in b_levels}
    grand_mean = float(np.mean(list(cell_means.values())))

    # Sum of squares components
    ss_a = n_cell * b * sum((marginal_a[la] - grand_mean) ** 2 for la in a_levels)
    ss_b = n_cell * a * sum((marginal_b[lb] - grand_mean) ** 2 for lb in b_levels)

    ss_cells = n_cell * sum(
        (cell_means[(la, lb)] - grand_mean) ** 2 for la in a_levels for lb in b_levels
    )
    ss_ab = ss_cells - ss_a - ss_b

    ss_residual = sum(
        sum((y - cell_means[(la, lb)]) ** 2 for y in cell_data[(la, lb)])
        for la in a_levels
        for lb in b_levels
    )
    ss_total = ss_a + ss_b + ss_ab + ss_residual

    df_a = a - 1
    df_b = b - 1
    df_ab = (a - 1) * (b - 1)
    df_residual = a * b * (n_cell - 1)
    df_total = n_total - 1

    ms_a = ss_a / df_a
    ms_b = ss_b / df_b
    ms_ab = ss_ab / df_ab
    ms_residual = ss_residual / df_residual

    f_a = ms_a / ms_residual
    f_b = ms_b / ms_residual
    f_ab = ms_ab / ms_residual

    p_a = float(stats.f.sf(f_a, df_a, df_residual))
    p_b = float(stats.f.sf(f_b, df_b, df_residual))
    p_ab = float(stats.f.sf(f_ab, df_ab, df_residual))

    eta_p_a = ss_a / (ss_a + ss_residual)
    eta_p_b = ss_b / (ss_b + ss_residual)
    eta_p_ab = ss_ab / (ss_ab + ss_residual)

    return {
        "n_total": n_total,
        "n_cell": n_cell,
        "grand_mean": grand_mean,
        "cell_means": cell_means,
        "marginal_a": marginal_a,
        "marginal_b": marginal_b,
        "ss_a": ss_a,
        "ss_b": ss_b,
        "ss_ab": ss_ab,
        "ss_residual": ss_residual,
        "ss_total": ss_total,
        "df_a": df_a,
        "df_b": df_b,
        "df_ab": df_ab,
        "df_residual": df_residual,
        "df_total": df_total,
        "ms_a": ms_a,
        "ms_b": ms_b,
        "ms_ab": ms_ab,
        "ms_residual": ms_residual,
        "f_a": f_a,
        "f_b": f_b,
        "f_ab": f_ab,
        "p_a": p_a,
        "p_b": p_b,
        "p_ab": p_ab,
        "partial_eta_a": eta_p_a,
        "partial_eta_b": eta_p_b,
        "partial_eta_ab": eta_p_ab,
    }


def reference_two_way_anova_model(
    data: list[tuple[str, str, float]],
    factor_a_levels: list[str],
    factor_b_levels: list[str],
    sum_of_squares: str = "type2",
) -> dict[str, Any]:
    """Level A / Level B independent model comparison for Type II and Type III."""
    a_levels = list(factor_a_levels)
    b_levels = list(factor_b_levels)
    n = len(data)

    # Build sum-contrast coding matrix (Level A independent algebra)
    def make_sum_contrast(vals: list[str], levels: list[str]) -> np.ndarray:
        m = len(levels) - 1
        x = np.zeros((len(vals), m), dtype=float)
        ref_level = levels[-1]
        for row_i, v in enumerate(vals):
            if v == ref_level:
                x[row_i, :] = -1.0
            else:
                try:
                    c = levels.index(v)
                    x[row_i, c] = 1.0
                except ValueError:
                    pass
        return x

    vals_a = [d[0] for d in data]
    vals_b = [d[1] for d in data]
    y = np.array([d[2] for d in data], dtype=float)

    xa = make_sum_contrast(vals_a, a_levels)
    xb = make_sum_contrast(vals_b, b_levels)

    # Interaction columns: elementwise products
    xab_list = []
    for i in range(xa.shape[1]):
        for j in range(xb.shape[1]):
            xab_list.append(xa[:, i] * xb[:, j])
    xab = np.column_stack(xab_list)

    const = np.ones((n, 1), dtype=float)
    x_full = np.hstack([const, xa, xb, xab])
    x_add = np.hstack([const, xa, xb])
    x_a = np.hstack([const, xa])
    x_b = np.hstack([const, xb])

    # OLS fits
    m_full = sm.OLS(y, x_full).fit()
    df_residual = n - x_full.shape[1]
    ss_residual = float(m_full.ssr)
    ms_residual = ss_residual / df_residual

    df_a = len(a_levels) - 1
    df_b = len(b_levels) - 1
    df_ab = df_a * df_b

    if sum_of_squares == "type2":
        m_add = sm.OLS(y, x_add).fit()
        m_a = sm.OLS(y, x_a).fit()
        m_b = sm.OLS(y, x_b).fit()

        ss_a = max(0.0, float(m_b.ssr - m_add.ssr))
        ss_b = max(0.0, float(m_a.ssr - m_add.ssr))
        ss_ab = max(0.0, float(m_add.ssr - m_full.ssr))

        f_a = (ss_a / df_a) / ms_residual
        f_b = (ss_b / df_b) / ms_residual
        f_ab = (ss_ab / df_ab) / ms_residual

        p_a = float(stats.f.sf(f_a, df_a, df_residual))
        p_b = float(stats.f.sf(f_b, df_b, df_residual))
        p_ab = float(stats.f.sf(f_ab, df_ab, df_residual))

    else:  # type3
        p = x_full.shape[1]
        idx_a = list(range(1, 1 + xa.shape[1]))
        idx_b = list(range(1 + xa.shape[1], 1 + xa.shape[1] + xb.shape[1]))
        idx_ab = list(range(1 + xa.shape[1] + xb.shape[1], p))

        def _wald(indices: list[int]) -> tuple[float, float, float]:
            L = np.zeros((len(indices), p), dtype=float)
            for r, idx in enumerate(indices):
                L[r, idx] = 1.0
            test = m_full.f_test(L)
            f_val = max(0.0, float(test.fvalue))
            p_val = float(test.pvalue)
            ss_val = max(0.0, f_val * float(test.df_num) * ms_residual)
            return ss_val, f_val, p_val

        ss_a, f_a, p_a = _wald(idx_a)
        ss_b, f_b, p_b = _wald(idx_b)
        ss_ab, f_ab, p_ab = _wald(idx_ab)

    eta_p_a = ss_a / (ss_a + ss_residual)
    eta_p_b = ss_b / (ss_b + ss_residual)
    eta_p_ab = ss_ab / (ss_ab + ss_residual)

    return {
        "n_total": n,
        "ss_a": ss_a,
        "ss_b": ss_b,
        "ss_ab": ss_ab,
        "ss_residual": ss_residual,
        "df_a": df_a,
        "df_b": df_b,
        "df_ab": df_ab,
        "df_residual": df_residual,
        "f_a": f_a,
        "f_b": f_b,
        "f_ab": f_ab,
        "p_a": p_a,
        "p_b": p_b,
        "p_ab": p_ab,
        "partial_eta_a": eta_p_a,
        "partial_eta_b": eta_p_b,
        "partial_eta_ab": eta_p_ab,
    }
