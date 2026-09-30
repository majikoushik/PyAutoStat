"""Statistical uncertainty and confidence interval estimators for effect quantities.

This module provides validated analytical and deterministic bootstrap confidence
interval routines for effect quantities shipped by PyAutoStat, ensuring exact
estimator-to-CI parameter matching, finite numbers, and strict serialization.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats


def _confidence_interval_record(
    lower: float | None,
    upper: float | None,
    level: float,
    method: str,
    quantity: str,
    *,
    status: str = "available",
    reason: str | None = None,
    sidedness: str = "two-sided",
    multiplicity_adjusted: bool = False,
    requested_resamples: int | None = None,
    valid_resamples: int | None = None,
    invalid_resamples: int | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Build a standardized, JSON-serializable confidence interval record."""
    if status != "available":
        record: dict[str, Any] = {
            "lower": None,
            "upper": None,
            "level": float(level),
            "method": method,
            "quantity": quantity,
            "status": status,
            "reason": reason,
            "sidedness": sidedness,
            "multiplicity_adjusted": multiplicity_adjusted,
        }
    else:
        assert lower is not None and upper is not None
        low_val = float(lower)
        upp_val = float(upper)
        if not (math.isfinite(low_val) and math.isfinite(upp_val)):
            return {
                "lower": None,
                "upper": None,
                "level": float(level),
                "method": method,
                "quantity": quantity,
                "status": "uncomputable",
                "reason": "Computed interval bounds were nonfinite.",
                "sidedness": sidedness,
                "multiplicity_adjusted": multiplicity_adjusted,
            }
        record = {
            "lower": low_val,
            "upper": upp_val,
            "level": float(level),
            "method": method,
            "quantity": quantity,
            "status": "available",
            "reason": None,
            "sidedness": sidedness,
            "multiplicity_adjusted": multiplicity_adjusted,
        }

    if requested_resamples is not None:
        record["requested_resamples"] = int(requested_resamples)
        record["valid_resamples"] = int(valid_resamples or 0)
        record["invalid_resamples"] = int(invalid_resamples if invalid_resamples is not None else 0)
        record["random_seed"] = random_seed

    return record


def noncentral_t_confidence_limits(
    t_stat: float, df: float, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Invert the noncentral t cumulative distribution to find NCP confidence limits.

    Solves for delta_L and delta_U such that:
      P(T <= t_stat | df, delta_L) = 1 - alpha / 2
      P(T <= t_stat | df, delta_U) = alpha / 2
    for alpha = 1 - confidence_level.

    Uses the exact symmetry of noncentral t: T ~ nct(df, delta) <=> -T ~ nct(df, -delta).
    """
    alpha = 1.0 - confidence_level
    target_low = 1.0 - alpha / 2.0
    target_high = alpha / 2.0

    sign = 1.0 if t_stat >= 0.0 else -1.0
    t_abs = abs(t_stat)

    def f_low(delta: float) -> float:
        return float(stats.nct.cdf(t_abs, df, delta)) - target_low

    def f_high(delta: float) -> float:
        return float(stats.nct.cdf(t_abs, df, delta)) - target_high

    # f_low bracket search: CDF is strictly decreasing in delta.
    # As delta -> -inf, CDF -> 1.0, so f_low -> alpha/2 > 0.
    # As delta -> +inf, CDF -> 0.0, so f_low -> -(1 - alpha/2) < 0.
    d_left = t_abs - 2.0
    step = 2.0
    while f_low(d_left) < 0:
        d_left -= step
        step *= 1.5

    d_right = t_abs + 2.0
    step = 2.0
    while f_low(d_right) > 0:
        d_right += step
        step *= 1.5

    delta_l = optimize.brentq(f_low, d_left, d_right)

    # f_high bracket search
    d_left = t_abs - 2.0
    step = 2.0
    while f_high(d_left) < 0:
        d_left -= step
        step *= 1.5

    d_right = t_abs + 2.0
    step = 2.0
    while f_high(d_right) > 0:
        d_right += step
        step *= 1.5

    delta_u = optimize.brentq(f_high, d_left, d_right)

    if sign >= 0.0:
        return float(delta_l), float(delta_u)
    return float(-delta_u), float(-delta_l)


def paired_cohen_dz_ci(
    mean_diff: float, sd_diff: float, n: int, confidence_level: float = 0.95
) -> dict[str, Any]:
    """Exact noncentral-t confidence interval for paired Cohen's dz."""
    method = "exact noncentral-t inversion"
    quantity = "Cohen's dz"

    if n < 2:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="unavailable",
            reason="Cohen's dz confidence interval requires at least 2 complete pairs.",
        )
    if not math.isfinite(mean_diff) or not math.isfinite(sd_diff) or sd_diff <= 0.0:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="uncomputable",
            reason="Sample SD of paired differences is nonpositive or nonfinite.",
        )

    dz = mean_diff / sd_diff
    t_val = dz * math.sqrt(n)
    df = n - 1

    try:
        ncp_l, ncp_u = noncentral_t_confidence_limits(t_val, df, confidence_level)
        dz_l = ncp_l / math.sqrt(n)
        dz_u = ncp_u / math.sqrt(n)
        return _confidence_interval_record(
            dz_l, dz_u, confidence_level, method, quantity, status="available"
        )
    except Exception as exc:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="uncomputable",
            reason=f"Noncentral-t root finding failed: {exc}",
        )


def one_sample_cohen_d_ci(
    mean_val: float, ref_val: float, sd_val: float, n: int, confidence_level: float = 0.95
) -> dict[str, Any]:
    """Exact noncentral-t confidence interval for one-sample Cohen's d."""
    method = "exact noncentral-t inversion"
    quantity = "one-sample Cohen's d"

    if n < 2:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="unavailable",
            reason="One-sample Cohen's d confidence interval requires at least 2 observations.",
        )
    if (
        not math.isfinite(mean_val)
        or not math.isfinite(ref_val)
        or not math.isfinite(sd_val)
        or sd_val <= 0.0
    ):
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="uncomputable",
            reason="Sample standard deviation is nonpositive or nonfinite.",
        )

    d_val = (mean_val - ref_val) / sd_val
    t_val = d_val * math.sqrt(n)
    df = n - 1

    try:
        ncp_l, ncp_u = noncentral_t_confidence_limits(t_val, df, confidence_level)
        d_l = ncp_l / math.sqrt(n)
        d_u = ncp_u / math.sqrt(n)
        return _confidence_interval_record(
            d_l, d_u, confidence_level, method, quantity, status="available"
        )
    except Exception as exc:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="uncomputable",
            reason=f"Noncentral-t root finding failed: {exc}",
        )


def fisher_z_correlation_ci(
    r: float, n: int, confidence_level: float = 0.95, quantity: str = "Pearson r"
) -> dict[str, Any]:
    """Analytical Fisher z-transformation confidence interval for linear correlation."""
    method = "Fisher-z asymptotic normal confidence interval"

    if n <= 3:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="unavailable",
            reason="Fisher-z correlation interval requires at least 4 observations (n > 3).",
        )
    if not math.isfinite(r) or not -1.0 <= r <= 1.0:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="unavailable",
            reason="Correlation coefficient is nonfinite or outside [-1, 1].",
        )

    if r >= 1.0 - 1e-12:
        return _confidence_interval_record(
            max(-1.0, 1.0 - 1e-12),
            1.0,
            confidence_level,
            method,
            quantity,
            status="available",
        )
    if r <= -1.0 + 1e-12:
        return _confidence_interval_record(
            -1.0,
            min(1.0, -1.0 + 1e-12),
            confidence_level,
            method,
            quantity,
            status="available",
        )

    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    alpha = 1.0 - confidence_level
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    zl = z - z_crit * se
    zu = z + z_crit * se
    lower = math.tanh(zl)
    upper = math.tanh(zu)

    lower = max(-1.0, min(1.0, lower))
    upper = max(-1.0, min(1.0, upper))

    return _confidence_interval_record(
        lower, upper, confidence_level, method, quantity, status="available"
    )


def fisher_exact_sample_or_ci(
    observed_counts: list[list[int]] | np.ndarray, confidence_level: float = 0.95
) -> dict[str, Any]:
    """Asymptotic log-Wald confidence interval for the sample odds ratio estimator.

    Targets the sample odds-ratio estimator ad/(bc) for an ordered 2x2 table [[a, b], [c, d]].
    Unavailable when any cell count is zero; no silent continuity corrections are applied.
    """
    method = "log-Wald confidence interval for sample odds-ratio estimator"
    quantity = "sample odds ratio"

    counts = np.asarray(observed_counts)
    if counts.shape != (2, 2):
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="unavailable",
            reason="Sample odds-ratio interval is supported only for 2x2 contingency tables.",
        )

    a, b = float(counts[0, 0]), float(counts[0, 1])
    c, d = float(counts[1, 0]), float(counts[1, 1])

    if a <= 0 or b <= 0 or c <= 0 or d <= 0:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="unavailable",
            reason=(
                "Log-Wald interval for the sample odds ratio is unavailable when "
                "contingency table contains zero cells."
            ),
        )

    or_val = (a * d) / (b * c)
    se_log = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    alpha = 1.0 - confidence_level
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    log_or = math.log(or_val)
    lower = math.exp(log_or - z_crit * se_log)
    upper = math.exp(log_or + z_crit * se_log)

    return _confidence_interval_record(
        lower, upper, confidence_level, method, quantity, status="available"
    )


def repeated_measures_partial_eta_squared_ci(
    f_stat: float, df_cond: float, df_error: float, confidence_level: float = 0.95
) -> dict[str, Any]:
    """Exact noncentral-F pivot confidence interval for repeated-measures partial eta-squared.

    Uses uncorrected observed F and uncorrected condition and error degrees of freedom
    to invert the noncentral-F distribution into noncentrality limits lambda_L and lambda_U,
    which are transformed via lambda / (lambda + df_error) to partial eta-squared bounds.
    """
    method = "noncentral-F inversion (uncorrected F and df)"
    quantity = "partial eta-squared"

    if (
        not math.isfinite(f_stat)
        or f_stat < 0.0
        or not math.isfinite(df_cond)
        or df_cond <= 0.0
        or not math.isfinite(df_error)
        or df_error <= 0.0
    ):
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="uncomputable",
            reason="F-statistic or degrees of freedom are nonpositive or nonfinite.",
            sidedness="two-sided",
        )

    alpha = 1.0 - confidence_level
    target_low = 1.0 - alpha / 2.0
    target_high = alpha / 2.0

    try:
        # Lower lambda bound
        if stats.f.cdf(f_stat, df_cond, df_error) <= target_low:
            lambda_l = 0.0
        else:

            def f_l(lam: float) -> float:
                return float(stats.ncf.cdf(f_stat, df_cond, df_error, lam)) - target_low

            lam_r = max(10.0, f_stat * df_cond)
            while f_l(lam_r) > 0:
                lam_r *= 2.0
            lambda_l = float(optimize.brentq(f_l, 0.0, lam_r))

        # Upper lambda bound
        def f_u(lam: float) -> float:
            return float(stats.ncf.cdf(f_stat, df_cond, df_error, lam)) - target_high

        lam_r = max(10.0, f_stat * df_cond * 2.0)
        while f_u(lam_r) > 0:
            lam_r *= 2.0
        lambda_u = float(optimize.brentq(f_u, 0.0, lam_r))

        lower = lambda_l / (lambda_l + df_error)
        upper = lambda_u / (lambda_u + df_error)

        lower = max(0.0, min(1.0, lower))
        upper = max(0.0, min(1.0, upper))

        return _confidence_interval_record(
            lower,
            upper,
            confidence_level,
            method,
            quantity,
            status="available",
            sidedness="two-sided",
        )
    except Exception as exc:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="uncomputable",
            reason=f"Noncentral-F root finding failed: {exc}",
            sidedness="two-sided",
        )


def matched_pairs_rank_biserial_bootstrap_ci(
    differences: np.ndarray | list[float],
    confidence_level: float = 0.95,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Deterministic pair-level percentile bootstrap for matched-pairs rank-biserial correlation."""
    method = "paired-observation percentile bootstrap"
    quantity = "matched-pairs rank-biserial correlation"
    seed = 0 if random_state is None else random_state

    diffs = np.asarray(differences, dtype=float)
    n = len(diffs)
    if n < 2:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="unavailable",
            reason="Bootstrap rank-biserial interval requires at least 2 complete pairs.",
            requested_resamples=bootstrap_samples,
            valid_resamples=0,
            invalid_resamples=bootstrap_samples,
            random_seed=seed,
        )

    rng = np.random.default_rng(seed)
    reps: list[float] = []

    for _ in range(bootstrap_samples):
        indices = rng.integers(0, n, size=n)
        sample = diffs[indices]
        nonzero = sample[sample != 0.0]
        if len(nonzero) == 0:
            continue
        abs_nz = np.abs(nonzero)
        ranks = stats.rankdata(abs_nz, method="average")
        pos = float(ranks[nonzero > 0.0].sum())
        neg = float(ranks[nonzero < 0.0].sum())
        denom = pos + neg
        if denom > 0.0:
            val = (pos - neg) / denom
            if math.isfinite(val):
                reps.append(float(val))

    min_valid = max(50, bootstrap_samples // 2)
    if len(reps) < min_valid:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="unavailable",
            reason=(
                f"Fewer than half of requested bootstrap resamples were valid "
                f"({len(reps)} / {bootstrap_samples}; minimum {min_valid})."
            ),
            requested_resamples=bootstrap_samples,
            valid_resamples=len(reps),
            invalid_resamples=bootstrap_samples - len(reps),
            random_seed=seed,
        )

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    lower = max(-1.0, min(1.0, float(lower)))
    upper = max(-1.0, min(1.0, float(upper)))

    return _confidence_interval_record(
        lower,
        upper,
        confidence_level,
        method,
        quantity,
        status="available",
        requested_resamples=bootstrap_samples,
        valid_resamples=len(reps),
        invalid_resamples=bootstrap_samples - len(reps),
        random_seed=seed,
    )


def friedman_kendall_w_bootstrap_ci(
    matrix: np.ndarray,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Deterministic participant/unit block bootstrap for Kendall's W rank concordance."""
    method = "participant-block percentile bootstrap"
    quantity = "Kendall's W"
    seed = 0 if random_state is None else random_state

    n, k = matrix.shape
    if n < 3 or k < 3:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="unavailable",
            reason=(
                "Friedman Kendall's W bootstrap interval requires at least 3 units "
                "and 3 conditions."
            ),
            requested_resamples=bootstrap_samples,
            valid_resamples=0,
            invalid_resamples=bootstrap_samples,
            random_seed=seed,
        )

    rng = np.random.default_rng(seed)
    reps: list[float] = []

    for _ in range(bootstrap_samples):
        indices = rng.integers(0, n, size=n)
        sample = matrix[indices, :]
        try:
            q_b, _ = stats.friedmanchisquare(*[sample[:, j] for j in range(k)])
            w_b = float(q_b / (n * (k - 1)))
            if math.isfinite(w_b) and 0.0 <= w_b <= 1.0:
                reps.append(w_b)
        except Exception:
            continue

    min_valid = max(50, bootstrap_samples // 2)
    if len(reps) < min_valid:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="unavailable",
            reason=(
                f"Fewer than half of requested bootstrap resamples were valid "
                f"({len(reps)} / {bootstrap_samples}; minimum {min_valid})."
            ),
            requested_resamples=bootstrap_samples,
            valid_resamples=len(reps),
            invalid_resamples=bootstrap_samples - len(reps),
            random_seed=seed,
        )

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    lower = max(0.0, min(1.0, float(lower)))
    upper = max(0.0, min(1.0, float(upper)))

    return _confidence_interval_record(
        lower,
        upper,
        confidence_level,
        method,
        quantity,
        status="available",
        requested_resamples=bootstrap_samples,
        valid_resamples=len(reps),
        invalid_resamples=bootstrap_samples - len(reps),
        random_seed=seed,
    )


def dunn_pairwise_rank_biserial_bootstrap_ci(
    group1: np.ndarray | list[float],
    group2: np.ndarray | list[float],
    confidence_level: float = 0.95,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Deterministic within-group bootstrap for Dunn pairwise rank-biserial correlation."""
    method = "independent within-group percentile bootstrap"
    quantity = "pairwise rank-biserial correlation"
    seed = 0 if random_state is None else random_state

    x = np.asarray(group1, dtype=float)
    y = np.asarray(group2, dtype=float)
    n1, n2 = len(x), len(y)

    if n1 < 2 or n2 < 2:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="unavailable",
            reason=(
                "Pairwise rank-biserial bootstrap interval requires at least 2 observations "
                "per group."
            ),
            multiplicity_adjusted=False,
            requested_resamples=bootstrap_samples,
            valid_resamples=0,
            invalid_resamples=bootstrap_samples,
            random_seed=seed,
        )

    rng = np.random.default_rng(seed)
    reps: list[float] = []

    for _ in range(bootstrap_samples):
        idx_x = rng.integers(0, n1, size=n1)
        idx_y = rng.integers(0, n2, size=n2)
        x_b = x[idx_x]
        y_b = y[idx_y]
        combined = np.concatenate([x_b, y_b])
        ranks = stats.rankdata(combined, method="average")
        u_val = float(np.sum(ranks[:n1]) - n1 * (n1 + 1) / 2.0)
        denom = n1 * n2
        if denom > 0:
            rb_b = (2.0 * u_val / denom) - 1.0
            if math.isfinite(rb_b):
                reps.append(float(rb_b))

    min_valid = max(50, bootstrap_samples // 2)
    if len(reps) < min_valid:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="unavailable",
            reason=(
                f"Fewer than half of requested bootstrap resamples were valid "
                f"({len(reps)} / {bootstrap_samples}; minimum {min_valid})."
            ),
            multiplicity_adjusted=False,
            requested_resamples=bootstrap_samples,
            valid_resamples=len(reps),
            invalid_resamples=bootstrap_samples - len(reps),
            random_seed=seed,
        )

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    lower = max(-1.0, min(1.0, float(lower)))
    upper = max(-1.0, min(1.0, float(upper)))

    return _confidence_interval_record(
        lower,
        upper,
        confidence_level,
        method,
        quantity,
        status="available",
        multiplicity_adjusted=False,
        requested_resamples=bootstrap_samples,
        valid_resamples=len(reps),
        invalid_resamples=bootstrap_samples - len(reps),
        random_seed=seed,
    )


def ols_r_squared_case_bootstrap_ci(
    y: np.ndarray,
    X: np.ndarray,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Deterministic case-resampling percentile bootstrap for in-sample R-squared."""
    method = "case-resampling percentile bootstrap CI for in-sample R-squared"
    quantity = "R-squared"
    seed = 0 if random_state is None else random_state

    n = len(y)
    p = X.shape[1] if X.ndim > 1 else 1

    if n <= p:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="unavailable",
            reason="In-sample R-squared bootstrap requires degrees of freedom (n > p).",
            requested_resamples=bootstrap_samples,
            valid_resamples=0,
            invalid_resamples=bootstrap_samples,
            random_seed=seed,
        )

    rng = np.random.default_rng(seed)
    reps: list[float] = []

    for _ in range(bootstrap_samples):
        indices = rng.integers(0, n, size=n)
        y_b = y[indices]
        X_b = X[indices, :] if X.ndim > 1 else X[indices]
        if np.var(y_b) <= 0.0:
            continue
        if np.linalg.matrix_rank(X_b) < p:
            continue
        try:
            # Solve OLS coefficients
            beta, residuals, rank, s = np.linalg.lstsq(X_b, y_b, rcond=None)
            y_pred = X_b @ beta
            ss_tot = float(np.sum((y_b - np.mean(y_b)) ** 2))
            ss_res = float(np.sum((y_b - y_pred) ** 2))
            if ss_tot > 0.0:
                r2 = 1.0 - (ss_res / ss_tot)
                r2 = max(0.0, min(1.0, r2))
                if math.isfinite(r2):
                    reps.append(r2)
        except Exception:
            continue

    min_valid = max(50, bootstrap_samples // 2)
    if len(reps) < min_valid:
        return _confidence_interval_record(
            None,
            None,
            confidence_level,
            method,
            quantity,
            status="unavailable",
            reason=(
                f"Fewer than half of requested bootstrap resamples were valid "
                f"({len(reps)} / {bootstrap_samples}; minimum {min_valid})."
            ),
            requested_resamples=bootstrap_samples,
            valid_resamples=len(reps),
            invalid_resamples=bootstrap_samples - len(reps),
            random_seed=seed,
        )

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    lower = max(0.0, min(1.0, float(lower)))
    upper = max(0.0, min(1.0, float(upper)))

    return _confidence_interval_record(
        lower,
        upper,
        confidence_level,
        method,
        quantity,
        status="available",
        requested_resamples=bootstrap_samples,
        valid_resamples=len(reps),
        invalid_resamples=bootstrap_samples - len(reps),
        random_seed=seed,
    )
