"""Independent reference formulas for extended association and paired categorical tests.

Covers:
19. kendall_tau_b (pairwise concordance and tie accounting)
20. point_biserial_correlation (closed-form group formula & Pearson equivalence)
21. partial_pearson_correlation (OLS residualization and degrees of freedom)
22. mcnemar (combinatorial exact binomial calculation)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

# =============================================================================
# 19. KENDALL TAU-B
# =============================================================================


def reference_kendall_tau_b(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Level A independent manual calculation for Kendall's tau-b."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = len(xa)

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    ties_both = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx = xa[i] - xa[j]
            dy = ya[i] - ya[j]

            if dx == 0 and dy == 0:
                ties_both += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1

    denom_x = concordant + discordant + ties_x
    denom_y = concordant + discordant + ties_y
    denom = math.sqrt(denom_x * denom_y) if (denom_x > 0 and denom_y > 0) else 0.0

    tau_b = (concordant - discordant) / denom if denom > 0 else 0.0

    # Level B backend cross-check with SciPy
    scipy_res = stats.kendalltau(xa, ya, variant="b")

    return {
        "n": n,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "ties_x_only": ties_x,
        "ties_y_only": ties_y,
        "ties_both": ties_both,
        "tau_b": tau_b,
        "backend_statistic": float(scipy_res.statistic),
        "backend_p_value": float(scipy_res.pvalue),
    }


# =============================================================================
# 20. POINT-BISERIAL CORRELATION
# =============================================================================


def reference_point_biserial(
    binary: list[bool | int] | np.ndarray,
    continuous: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Level A independent calculation using both closed-form group moments and Pearson r."""
    b_arr = np.asarray(binary, dtype=float)
    c_arr = np.asarray(continuous, dtype=float)
    n = len(b_arr)

    # Group separation
    g0 = c_arr[b_arr == 0.0]
    g1 = c_arr[b_arr == 1.0]
    n0 = len(g0)
    n1 = len(g1)

    if n0 == 0 or n1 == 0:
        raise ValueError("Both binary groups must contain observations")

    m0 = float(np.mean(g0))
    m1 = float(np.mean(g1))
    sy = float(np.std(c_arr, ddof=1))

    # Closed-form Lev (1949) formula: (m1 - m0) / sy * sqrt(n0 * n1 / (n * (n - 1)))
    r_pb = ((m1 - m0) / sy) * math.sqrt((n0 * n1) / (n * (n - 1))) if (sy > 0 and n > 1) else 0.0

    # Direct Pearson correlation verification on dummy-coded 0/1
    diff_b = b_arr - np.mean(b_arr)
    diff_c = c_arr - np.mean(c_arr)
    cov_bc = np.sum(diff_b * diff_c) / (n - 1)
    sb = np.std(b_arr, ddof=1)
    r_pearson = cov_bc / (sb * sy) if (sb > 0 and sy > 0) else 0.0

    df = n - 2
    denom_t = math.sqrt(max(0.0, 1.0 - r_pb**2))
    t_stat = r_pb * math.sqrt(df) / denom_t if denom_t > 0 else 0.0
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df))

    # Level B backend cross-check with SciPy
    scipy_res = stats.pointbiserialr(b_arr.astype(bool), c_arr)

    return {
        "n": n,
        "n0": n0,
        "n1": n1,
        "mean0": m0,
        "mean1": m1,
        "sd_total": sy,
        "point_biserial_r": r_pb,
        "pearson_r_equivalence": r_pearson,
        "degrees_of_freedom": df,
        "test_statistic": t_stat,
        "p_value": p_val,
        "backend_statistic": float(scipy_res.statistic),
        "backend_p_value": float(scipy_res.pvalue),
    }


# =============================================================================
# 21. PARTIAL PEARSON CORRELATION
# =============================================================================


def reference_partial_pearson(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    controls: list[list[float]] | np.ndarray,
) -> dict[str, Any]:
    """Level A independent calculation using OLS residualization on control covariates."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    z = np.asarray(controls, dtype=float)
    n = len(xa)

    # If z is 1D, reshape to (n, 1)
    if z.ndim == 1:
        z = z[:, np.newaxis]

    k_ctrl = z.shape[1]

    # Add constant column for OLS regressions
    z_design = np.column_stack([np.ones(n, dtype=float), z])

    # OLS residualization for x
    beta_x = np.linalg.solve(z_design.T @ z_design, z_design.T @ xa)
    resid_x = xa - z_design @ beta_x

    # OLS residualization for y
    beta_y = np.linalg.solve(z_design.T @ z_design, z_design.T @ ya)
    resid_y = ya - z_design @ beta_y

    # Pearson correlation on residuals
    rx = resid_x - np.mean(resid_x)
    ry = resid_y - np.mean(resid_y)
    cov_xy = np.sum(rx * ry)
    var_x = np.sum(rx**2)
    var_y = np.sum(ry**2)

    r_partial = cov_xy / math.sqrt(var_x * var_y) if (var_x > 0 and var_y > 0) else 0.0

    df = n - 2 - k_ctrl
    denom_t = math.sqrt(max(0.0, 1.0 - r_partial**2))
    t_stat = r_partial * math.sqrt(df) / denom_t if denom_t > 0 else 0.0
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df))

    return {
        "n": n,
        "k_controls": k_ctrl,
        "degrees_of_freedom": df,
        "partial_r": r_partial,
        "test_statistic": t_stat,
        "p_value": p_val,
    }


# =============================================================================
# 22. EXACT BINOMIAL MCNEMAR TEST
# =============================================================================


def reference_mcnemar_exact(
    table_2x2: list[list[int]] | np.ndarray,
) -> dict[str, Any]:
    """Level A independent combinatorial calculation for exact binomial McNemar."""
    t = np.asarray(table_2x2, dtype=int)
    # Conventional transition table:
    #   [ [a, b],
    #     [c, d] ]
    # where b and c are discordant cells.
    a = int(t[0, 0])
    b = int(t[0, 1])
    c = int(t[1, 0])
    d = int(t[1, 1])
    n_pairs = a + b + c + d
    n_disc = b + c

    if n_disc == 0:
        p_val = 1.0
    else:
        # Independent exact two-sided binomial calculation:
        # sum over all outcomes with prob <= p(observed)
        k_min = min(b, c)
        prob_sum = 0.0
        for k in range(k_min + 1):
            comb = math.comb(n_disc, k)
            prob_sum += comb * (0.5**n_disc)
        p_val = min(1.0, 2.0 * prob_sum)

    # Proportion difference: (b - c) / n_pairs
    diff_prop = (b - c) / n_pairs if n_pairs > 0 else 0.0

    # Level B backend cross-check with SciPy binomtest
    scipy_res = (
        stats.binomtest(min(b, c), n_disc, p=0.5, alternative="two-sided") if n_disc > 0 else None
    )
    backend_p = float(scipy_res.pvalue) if scipy_res else 1.0

    return {
        "n_pairs": n_pairs,
        "discordant_b": b,
        "discordant_c": c,
        "n_discordant": n_disc,
        "difference_in_proportions": diff_prop,
        "p_value": p_val,
        "backend_p_value": backend_p,
    }
