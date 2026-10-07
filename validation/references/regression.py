"""Independent reference mathematical formulas for linear and logistic regression.

Covers:
17. linear_regression (OLS, HC3 robust covariance, diagnostics)
18. logistic_regression (Newton-Raphson/IRLS binary Logit, Wald inference, model fit)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy import stats

# =============================================================================
# 17. OLS LINEAR REGRESSION AND HC3
# =============================================================================


def reference_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Level A independent manual linear algebra calculation for OLS and HC3."""
    # Ensure constant is included if not already present
    if not np.allclose(X[:, 0], 1.0):
        X_mat = np.column_stack([np.ones(len(y), dtype=float), X])
        names = ["Intercept"] + list(feature_names)
    else:
        X_mat = np.asarray(X, dtype=float)
        names = list(feature_names)

    n, p = X_mat.shape
    y_vec = np.asarray(y, dtype=float)

    # Beta = (X'X)^(-1) X'y via independent solve
    xtx = X_mat.T @ X_mat
    xty = X_mat.T @ y_vec
    beta = np.linalg.solve(xtx, xty)

    y_hat = X_mat @ beta
    residuals = y_vec - y_hat
    sse = float(np.sum(residuals**2))

    y_mean = float(np.mean(y_vec))
    sst = float(np.sum((y_vec - y_mean) ** 2))
    ssm = sst - sse

    r_squared = 1.0 - (sse / sst) if sst > 0 else 0.0
    df_model = float(p - 1)
    df_resid = float(n - p)
    adj_r_squared = 1.0 - ((sse / df_resid) / (sst / (n - 1))) if (df_resid > 0 and n > 1) else 0.0

    ms_model = ssm / df_model if df_model > 0 else 0.0
    ms_resid = sse / df_resid if df_resid > 0 else 0.0
    f_stat = ms_model / ms_resid if ms_resid > 0 else 0.0
    f_p_val = float(stats.f.sf(f_stat, df_model, df_resid))

    # Classical OLS covariance matrix: s^2 * (X'X)^(-1)
    inv_xtx = np.linalg.inv(xtx)
    cov_classical = ms_resid * inv_xtx
    se_classical = np.sqrt(np.maximum(0.0, np.diag(cov_classical)))
    t_classical = np.where(se_classical > 0, beta / se_classical, 0.0)
    alpha = 1.0 - confidence_level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_resid))
    p_classical = 2.0 * stats.t.sf(np.abs(t_classical), df_resid)

    # Standardized coefficients (excluding intercept)
    sy = math.sqrt(sst / (n - 1)) if n > 1 else 1.0
    std_beta = [None]  # intercept
    for j in range(1, p):
        sx = math.sqrt(float(np.var(X_mat[:, j], ddof=1)))
        std_beta.append(float(beta[j] * sx / sy) if sy > 0 else 0.0)

    # HC3 Robust Covariance:
    # h_ii = x_i' (X'X)^(-1) x_i
    hat_diag = np.diag(X_mat @ inv_xtx @ X_mat.T)
    omega_hc3 = np.diag((residuals**2) / ((1.0 - hat_diag) ** 2))
    cov_hc3 = inv_xtx @ (X_mat.T @ omega_hc3 @ X_mat) @ inv_xtx
    se_hc3 = np.sqrt(np.maximum(0.0, np.diag(cov_hc3)))
    t_hc3 = np.where(se_hc3 > 0, beta / se_hc3, 0.0)
    p_hc3 = 2.0 * stats.t.sf(np.abs(t_hc3), df_resid)

    # HC3 Wald F-statistic for model slopes
    if df_model > 0 and len(beta) > 1:
        cov_slopes_hc3 = cov_hc3[1:, 1:]
        b_slopes = beta[1:]
        try:
            inv_cov_slopes = np.linalg.inv(cov_slopes_hc3)
            wald_hc3 = float(b_slopes @ inv_cov_slopes @ b_slopes)
            f_hc3 = wald_hc3 / float(df_model)
            f_p_hc3 = float(stats.f.sf(f_hc3, df_model, df_resid))
        except Exception:
            f_hc3 = f_stat
            f_p_hc3 = f_p_val
    else:
        f_hc3 = f_stat
        f_p_hc3 = f_p_val

    # Breusch-Pagan test
    # Auxiliary regression of squared residuals on X
    e2 = residuals**2
    e2_mean = float(np.mean(e2))
    sst_e2 = float(np.sum((e2 - e2_mean) ** 2))
    beta_aux = np.linalg.solve(xtx, X_mat.T @ e2)
    e2_hat = X_mat @ beta_aux
    sse_aux = float(np.sum((e2 - e2_hat) ** 2))
    r2_aux = 1.0 - (sse_aux / sst_e2) if sst_e2 > 0 else 0.0
    bp_lm = float(n * r2_aux)
    bp_lm_p = float(stats.chi2.sf(bp_lm, df_model))
    bp_f = (r2_aux / df_model) / ((1.0 - r2_aux) / df_resid) if (1.0 - r2_aux > 0) else 0.0
    bp_f_p = float(stats.f.sf(bp_f, df_model, df_resid))

    # Condition number
    cond_num = float(np.linalg.cond(X_mat))

    # Backend OLS cross-check (Level B)
    sm_model = sm.OLS(y_vec, X_mat).fit()
    sm_hc3 = sm.OLS(y_vec, X_mat).fit(cov_type="HC3")

    return {
        "n": n,
        "p": p,
        "feature_names": names,
        "beta": beta.tolist(),
        "residuals": residuals.tolist(),
        "sse": sse,
        "sst": sst,
        "r_squared": r_squared,
        "adjusted_r_squared": adj_r_squared,
        "df_model": df_model,
        "df_resid": df_resid,
        "f_statistic": f_stat,
        "f_p_value": f_p_val,
        "f_statistic_hc3": f_hc3,
        "f_p_value_hc3": f_p_hc3,
        "se_classical": se_classical.tolist(),
        "t_classical": t_classical.tolist(),
        "p_classical": p_classical.tolist(),
        "ci_lower_classical": (beta - t_crit * se_classical).tolist(),
        "ci_upper_classical": (beta + t_crit * se_classical).tolist(),
        "standardized_beta": std_beta,
        "se_hc3": se_hc3.tolist(),
        "t_hc3": t_hc3.tolist(),
        "p_hc3": p_hc3.tolist(),
        "bp_lm": bp_lm,
        "bp_lm_p": bp_lm_p,
        "bp_f": bp_f,
        "bp_f_p": bp_f_p,
        "condition_number": cond_num,
        "backend_r_squared": float(sm_model.rsquared),
        "backend_beta": sm_model.params.tolist(),
        "backend_se_hc3": sm_hc3.bse.tolist(),
    }


# =============================================================================
# 18. BINARY LOGISTIC REGRESSION (IRLS / NEWTON-RAPHSON)
# =============================================================================


def reference_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    confidence_level: float = 0.95,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Level A independent manual Newton-Raphson/IRLS solver for binary logistic regression."""
    # Ensure constant column
    if not np.allclose(X[:, 0], 1.0):
        X_mat = np.column_stack([np.ones(len(y), dtype=float), X])
        names = ["Intercept"] + list(feature_names)
    else:
        X_mat = np.asarray(X, dtype=float)
        names = list(feature_names)

    n, p = X_mat.shape
    y_vec = np.asarray(y, dtype=float)

    # Initialize beta = 0
    beta = np.zeros(p, dtype=float)

    # Newton-Raphson iterations
    converged = False
    for _ in range(max_iter):
        eta = X_mat @ beta
        eta = np.clip(eta, -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-eta))

        # Gradient: X' (y - prob)
        grad = X_mat.T @ (y_vec - prob)

        # Hessian: -X' W X
        w = prob * (1.0 - prob)
        w = np.clip(w, 1e-12, None)
        H = -(X_mat.T * w) @ X_mat

        delta = np.linalg.solve(-H, grad)
        beta += delta

        if np.max(np.abs(delta)) < tol:
            converged = True
            break

    # Final probabilities
    eta = np.clip(X_mat @ beta, -30.0, 30.0)
    prob = 1.0 / (1.0 + np.exp(-eta))
    prob = np.clip(prob, 1e-15, 1.0 - 1e-15)

    # Log-likelihood
    llf = float(np.sum(y_vec * np.log(prob) + (1.0 - y_vec) * np.log(1.0 - prob)))

    # Null model log-likelihood
    y_bar = float(np.mean(y_vec))
    y_bar = np.clip(y_bar, 1e-15, 1.0 - 1e-15)
    ll_null = float(n * (y_bar * math.log(y_bar) + (1.0 - y_bar) * math.log(1.0 - y_bar)))

    # LR test
    lr_stat = float(2.0 * (llf - ll_null))
    lr_df = float(p - 1)
    lr_p = float(stats.chi2.sf(lr_stat, lr_df))

    # McFadden pseudo-R2
    mcfadden_r2 = 1.0 - (llf / ll_null) if ll_null != 0 else 0.0

    # Information criteria
    aic = 2.0 * p - 2.0 * llf
    bic = float(p * math.log(n) - 2.0 * llf)

    # Odds ratios
    odds_ratios = np.exp(beta).tolist()

    # Wald standard errors from inverted observed Fisher information (-H)^(-1)
    w = prob * (1.0 - prob)
    fisher_info = (X_mat.T * w) @ X_mat
    cov_mat = np.linalg.inv(fisher_info)
    se_wald = np.sqrt(np.diag(cov_mat))
    z_wald = beta / se_wald
    p_wald = 2.0 * stats.norm.sf(np.abs(z_wald))

    alpha = 1.0 - confidence_level
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_lower = (beta - z_crit * se_wald).tolist()
    ci_upper = (beta + z_crit * se_wald).tolist()
    or_ci_lower = np.exp(beta - z_crit * se_wald).tolist()
    or_ci_upper = np.exp(beta + z_crit * se_wald).tolist()

    # Level B backend cross-check with statsmodels Logit
    sm_logit = sm.Logit(y_vec, X_mat).fit(disp=0)

    return {
        "n": n,
        "p": p,
        "feature_names": names,
        "converged": converged,
        "beta": beta.tolist(),
        "log_likelihood": llf,
        "null_log_likelihood": ll_null,
        "lr_statistic": lr_stat,
        "lr_df": lr_df,
        "lr_p_value": lr_p,
        "mcfadden_r2": mcfadden_r2,
        "aic": aic,
        "bic": bic,
        "odds_ratios": odds_ratios,
        "se_wald": se_wald.tolist(),
        "z_wald": z_wald.tolist(),
        "p_wald": p_wald.tolist(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "or_ci_lower": or_ci_lower,
        "or_ci_upper": or_ci_upper,
        "backend_beta": sm_logit.params.tolist(),
        "backend_llf": float(sm_logit.llf),
    }
