"""Validated two-way factorial ANOVA engine for independent observations."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .exceptions import ColumnNotFoundError, InsufficientDataError, InvalidDataError
from .multigroup import adjust_pvalues
from .uncertainty import two_way_anova_partial_eta_squared_ci


def _check_level(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or not 0 < float(value) < 1
    ):
        raise InvalidDataError("confidence_level must be a finite number strictly between 0 and 1.")
    return float(value)


def _check_alpha(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or not 0 < float(value) < 1
    ):
        raise InvalidDataError("alpha must be a finite number strictly between 0 and 1.")
    return float(value)


def _label(value: Any) -> str | int | float | bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidDataError("Categorical factor levels must be finite.")
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _extract_levels(series: pd.Series) -> list[Any]:
    """Extract deterministic, ordered factor levels."""
    observed = list(pd.unique(series))
    if isinstance(series.dtype, pd.CategoricalDtype):
        ordered = [item for item in series.cat.categories if item in observed]
        ordered.extend(item for item in observed if item not in ordered)
        return [_label(item) for item in ordered]
    try:
        sorted_observed = sorted(observed)
        return [_label(item) for item in sorted_observed]
    except TypeError:
        sorted_observed = sorted(observed, key=lambda x: str(x))
        return [_label(item) for item in sorted_observed]


def _build_sum_contrasts(series: pd.Series, levels: list[Any]) -> np.ndarray:
    """Build sum-to-zero (effect) contrasts: 1 for level_k, -1 for last level, 0 otherwise."""
    cols = []
    last_level = levels[-1]
    for lvl in levels[:-1]:
        col = np.where(series == lvl, 1.0, np.where(series == last_level, -1.0, 0.0))
        cols.append(col)
    return np.column_stack(cols)


def two_way_anova(
    frame: pd.DataFrame,
    outcome: str,
    factor_a: str,
    factor_b: str,
    *,
    sum_of_squares: str = "type2",
    alpha: float = 0.05,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Execute two-way factorial ANOVA for independent observations.

    Parameters
    ----------
    frame : pd.DataFrame
        DataFrame containing outcome and factors.
    outcome : str
        Continuous quantitative dependent variable.
    factor_a : str
        First categorical independent factor.
    factor_b : str
        Second categorical independent factor.
    sum_of_squares : {"type2", "type3"}, default "type2"
        Sums of squares formulation:
        - "type2": Hierarchical sums of squares testing each main effect conditional
          on the other main effect, without the interaction term.
        - "type3": Fully conditional sums of squares testing each term conditional
          on all other terms (including interaction), using sum-to-zero contrast coding.
    alpha : float, default 0.05
        Significance threshold for hypothesis decisions.
    confidence_level : float, default 0.95
        Confidence level for partial eta-squared intervals and contrast intervals.
    """
    for name in (outcome, factor_a, factor_b):
        if not isinstance(name, str) or not name.strip():
            raise InvalidDataError("Column names must be non-empty strings.")
        if name not in frame.columns:
            raise ColumnNotFoundError(f"'{name}' is not a column in this DataFrame.")

    if len({outcome, factor_a, factor_b}) != 3:
        raise InvalidDataError("outcome, factor_a, and factor_b must name three distinct columns.")

    if sum_of_squares not in {"type2", "type3"}:
        raise InvalidDataError("sum_of_squares must be 'type2' or 'type3'.")

    dec_alpha = _check_alpha(alpha)
    conf_level = _check_level(confidence_level)

    # Complete cases
    columns = [outcome, factor_a, factor_b]
    usable = frame[columns].dropna().copy()
    original_rows = int(len(frame))
    analyzed_rows = int(len(usable))
    excluded_rows = original_rows - analyzed_rows

    if analyzed_rows < 4:
        raise InsufficientDataError(
            f"Two-way ANOVA requires at least 4 complete cases; {analyzed_rows} were found."
        )

    if not pd.api.types.is_numeric_dtype(usable[outcome]):
        raise InvalidDataError(f"'{outcome}' must be a real numeric column.")

    try:
        y = np.asarray(usable[outcome], dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidDataError(f"'{outcome}' contains nonfinite or non-numeric values.") from exc

    if not np.isfinite(y).all():
        raise InvalidDataError(f"'{outcome}' contains nonfinite values.")

    if float(np.ptp(y)) == 0.0:
        raise InsufficientDataError(f"Outcome '{outcome}' is constant across all usable rows.")

    # Factor levels
    levels_a = _extract_levels(usable[factor_a])
    levels_b = _extract_levels(usable[factor_b])

    if len(levels_a) < 2:
        raise InsufficientDataError(
            f"Factor '{factor_a}' must have at least 2 observed levels; found {len(levels_a)}."
        )
    if len(levels_b) < 2:
        raise InsufficientDataError(
            f"Factor '{factor_b}' must have at least 2 observed levels; found {len(levels_b)}."
        )

    num_levels_a = len(levels_a)
    num_levels_b = len(levels_b)
    num_cells = num_levels_a * num_levels_b

    # Check cell structure
    cell_values: list[np.ndarray] = []
    cell_summaries: list[dict[str, Any]] = []
    empty_cells: list[dict[str, Any]] = []

    for a_lvl in levels_a:
        for b_lvl in levels_b:
            mask = (usable[factor_a] == a_lvl) & (usable[factor_b] == b_lvl)
            cell_y = y[mask]
            count = len(cell_y)
            if count == 0:
                empty_cells.append({"factor_a": a_lvl, "factor_b": b_lvl})
                cell_summaries.append(
                    {
                        "factor_a": a_lvl,
                        "factor_b": b_lvl,
                        "sample_size": 0,
                        "mean": None,
                        "standard_deviation": None,
                        "standard_error": None,
                    }
                )
            else:
                cell_values.append(cell_y)
                mean_val = float(np.mean(cell_y))
                sd_val = float(np.std(cell_y, ddof=1)) if count > 1 else 0.0
                se_val = float(sd_val / math.sqrt(count)) if count > 0 else None
                cell_summaries.append(
                    {
                        "factor_a": a_lvl,
                        "factor_b": b_lvl,
                        "factor_a_level": a_lvl,
                        "factor_b_level": b_lvl,
                        "sample_size": count,
                        "n": count,
                        "mean": mean_val,
                        "standard_deviation": sd_val,
                        "standard_error": se_val,
                    }
                )

    if empty_cells:
        cell_desc = ", ".join(f"({c['factor_a']!r}, {c['factor_b']!r})" for c in empty_cells)
        raise InsufficientDataError(
            f"Empty cells detected in factorial design: {cell_desc}. "
            "Fully crossed two-way factorial ANOVA requires all cells to contain observations."
        )

    df_residual = analyzed_rows - num_cells
    if df_residual <= 0:
        raise InsufficientDataError(
            f"Residual degrees of freedom must be positive. Analyzed rows ({analyzed_rows}) "
            f"must exceed the number of factor cells ({num_cells}); residual df is {df_residual}."
        )

    # Build sum-contrast matrices
    XA = _build_sum_contrasts(usable[factor_a], levels_a)
    XB = _build_sum_contrasts(usable[factor_b], levels_b)

    # Interaction columns: elementwise products of XA and XB columns
    XAB_list = []
    for i in range(XA.shape[1]):
        for j in range(XB.shape[1]):
            XAB_list.append(XA[:, i] * XB[:, j])
    XAB = np.column_stack(XAB_list)

    const = np.ones((analyzed_rows, 1), dtype=float)
    X_full = np.hstack([const, XA, XB, XAB])
    X_add = np.hstack([const, XA, XB])
    X_a_only = np.hstack([const, XA])
    X_b_only = np.hstack([const, XB])

    # Fit OLS models
    model_full = sm.OLS(y, X_full).fit()
    ss_error = float(model_full.ssr)
    ms_error = ss_error / df_residual
    rmse = math.sqrt(ms_error) if ms_error >= 0.0 else 0.0

    if ss_error <= 0.0 or ms_error <= 0.0:
        raise InsufficientDataError(
            "Within-cell residual variance is zero or negative; F-tests cannot be computed."
        )

    df_a = num_levels_a - 1
    df_b = num_levels_b - 1
    df_ab = df_a * df_b

    # Terms calculation
    if sum_of_squares == "type2":
        model_add = sm.OLS(y, X_add).fit()
        model_a = sm.OLS(y, X_a_only).fit()
        model_b = sm.OLS(y, X_b_only).fit()

        ss_a = max(0.0, float(model_b.ssr - model_add.ssr))
        ss_b = max(0.0, float(model_a.ssr - model_add.ssr))
        ss_ab = max(0.0, float(model_add.ssr - model_full.ssr))

        ms_a = ss_a / df_a
        f_a = ms_a / ms_error
        p_a = float(stats.f.sf(f_a, df_a, df_residual))

        ms_b = ss_b / df_b
        f_b = ms_b / ms_error
        p_b = float(stats.f.sf(f_b, df_b, df_residual))

        ms_ab = ss_ab / df_ab
        f_ab = ms_ab / ms_error
        p_ab = float(stats.f.sf(f_ab, df_ab, df_residual))

    else:  # type3
        p = X_full.shape[1]
        idx_a = list(range(1, 1 + XA.shape[1]))
        idx_b = list(range(1 + XA.shape[1], 1 + XA.shape[1] + XB.shape[1]))
        idx_ab = list(range(1 + XA.shape[1] + XB.shape[1], p))

        def _wald(indices: list[int]) -> tuple[float, float, float, float]:
            L = np.zeros((len(indices), p), dtype=float)
            for r, idx in enumerate(indices):
                L[r, idx] = 1.0
            test = model_full.f_test(L)
            f_val = max(0.0, float(test.fvalue))
            p_val = float(test.pvalue)
            num_df = float(test.df_num)
            ss_val = max(0.0, f_val * num_df * ms_error)
            return ss_val, num_df, f_val, p_val

        ss_a, _, f_a, p_a = _wald(idx_a)
        ms_a = ss_a / df_a

        ss_b, _, f_b, p_b = _wald(idx_b)
        ms_b = ss_b / df_b

        ss_ab, _, f_ab, p_ab = _wald(idx_ab)
        ms_ab = ss_ab / df_ab

    # Partial eta-squared effect sizes and CIs
    eta_p_a = max(0.0, min(1.0, ss_a / (ss_a + ss_error)))
    eta_p_b = max(0.0, min(1.0, ss_b / (ss_b + ss_error)))
    eta_p_ab = max(0.0, min(1.0, ss_ab / (ss_ab + ss_error)))

    ci_a = two_way_anova_partial_eta_squared_ci(f_a, df_a, df_residual, conf_level)
    ci_b = two_way_anova_partial_eta_squared_ci(f_b, df_b, df_residual, conf_level)
    ci_ab = two_way_anova_partial_eta_squared_ci(f_ab, df_ab, df_residual, conf_level)

    terms = [
        {
            "term": factor_a,
            "term_type": "main_effect",
            "sum_squares": ss_a,
            "sum_of_squares": ss_a,
            "df": df_a,
            "mean_square": ms_a,
            "f_statistic": f_a,
            "p_value": p_a,
            "effect_size": {
                "name": "partial_eta_squared",
                "value": eta_p_a,
                "confidence_interval": ci_a,
                "status": "available",
            },
            "decision": "reject" if p_a < dec_alpha else "fail_to_reject",
        },
        {
            "term": factor_b,
            "term_type": "main_effect",
            "sum_squares": ss_b,
            "sum_of_squares": ss_b,
            "df": df_b,
            "mean_square": ms_b,
            "f_statistic": f_b,
            "p_value": p_b,
            "effect_size": {
                "name": "partial_eta_squared",
                "value": eta_p_b,
                "confidence_interval": ci_b,
                "status": "available",
            },
            "decision": "reject" if p_b < dec_alpha else "fail_to_reject",
        },
        {
            "term": f"{factor_a}:{factor_b}",
            "term_type": "interaction",
            "sum_squares": ss_ab,
            "sum_of_squares": ss_ab,
            "df": df_ab,
            "mean_square": ms_ab,
            "f_statistic": f_ab,
            "p_value": p_ab,
            "effect_size": {
                "name": "partial_eta_squared",
                "value": eta_p_ab,
                "confidence_interval": ci_ab,
                "status": "available",
            },
            "decision": "reject" if p_ab < dec_alpha else "fail_to_reject",
        },
        {
            "term": "Residual",
            "term_type": "residual",
            "sum_squares": ss_error,
            "sum_of_squares": ss_error,
            "df": df_residual,
            "mean_square": ms_error,
            "f_statistic": None,
            "p_value": None,
            "effect_size": None,
            "decision": None,
        },
    ]

    # Estimated marginal means (unweighted least-squares means)
    cell_map = {(c["factor_a"], c["factor_b"]): c for c in cell_summaries}

    emm_a = []
    for a_lvl in levels_a:
        means = [cell_map[(a_lvl, b_lvl)]["mean"] for b_lvl in levels_b]
        emm_val = float(np.mean(means))
        # Variance of unweighted mean of cell means: (1 / J^2) * sum_j (MSE / n_ij)
        var_emm = (ms_error / (num_levels_b**2)) * sum(
            1.0 / cell_map[(a_lvl, b_lvl)]["sample_size"] for b_lvl in levels_b
        )
        se_emm = math.sqrt(var_emm) if var_emm > 0.0 else 0.0
        emm_a.append(
            {"factor": factor_a, "level": a_lvl, "estimate": emm_val, "standard_error": se_emm}
        )

    emm_b = []
    for b_lvl in levels_b:
        means = [cell_map[(a_lvl, b_lvl)]["mean"] for a_lvl in levels_a]
        emm_val = float(np.mean(means))
        var_emm = (ms_error / (num_levels_a**2)) * sum(
            1.0 / cell_map[(a_lvl, b_lvl)]["sample_size"] for a_lvl in levels_a
        )
        se_emm = math.sqrt(var_emm) if var_emm > 0.0 else 0.0
        emm_b.append(
            {"factor": factor_b, "level": b_lvl, "estimate": emm_val, "standard_error": se_emm}
        )

    # Follow-ups
    critical_t = float(stats.t.ppf(1.0 - (1.0 - conf_level) / 2.0, df_residual))

    # Family 1: Simple effects of A within each level of B
    simple_a_records = []
    simple_a_raw_p: list[float] = []
    for b_lvl in levels_b:
        for a1, a2 in combinations(levels_a, 2):
            c1 = cell_map[(a1, b_lvl)]
            c2 = cell_map[(a2, b_lvl)]
            diff = c1["mean"] - c2["mean"]
            se = math.sqrt(ms_error * (1.0 / c1["sample_size"] + 1.0 / c2["sample_size"]))
            t_val = diff / se if se > 0.0 else 0.0
            raw_p = float(2.0 * stats.t.sf(abs(t_val), df_residual))
            margin = critical_t * se
            simple_a_raw_p.append(raw_p)
            simple_a_records.append(
                {
                    "contrast_id": f"{a1}_vs_{a2}_at_{b_lvl}",
                    "family": f"Simple effects of {factor_a} within {factor_b}",
                    "family_type": "simple_a_within_b",
                    "condition": b_lvl,
                    "conditioning_level": b_lvl,
                    "first": a1,
                    "second": a2,
                    "contrast": f"{a1} minus {a2} at {factor_b}={b_lvl}",
                    "estimate": diff,
                    "standard_error": se,
                    "statistic": t_val,
                    "t_statistic": t_val,
                    "degrees_of_freedom": float(df_residual),
                    "raw_p_value": raw_p,
                    "confidence_interval": {
                        "lower": diff - margin,
                        "upper": diff + margin,
                        "level": conf_level,
                        "method": "pointwise Student-t interval",
                        "multiplicity_adjusted": False,
                        "status": "available",
                    },
                }
            )

    adj_simple_a = adjust_pvalues(simple_a_raw_p, method="holm")
    for rec, adj_p in zip(simple_a_records, adj_simple_a, strict=True):
        rec["adjusted_p_value"] = adj_p
        rec["adjustment_method"] = "Holm step-down procedure"
        rec["comparison_count"] = len(simple_a_records)
        rec["decision"] = "reject" if adj_p < dec_alpha else "fail_to_reject"

    # Family 2: Simple effects of B within each level of A
    simple_b_records = []
    simple_b_raw_p: list[float] = []
    for a_lvl in levels_a:
        for b1, b2 in combinations(levels_b, 2):
            c1 = cell_map[(a_lvl, b1)]
            c2 = cell_map[(a_lvl, b2)]
            diff = c1["mean"] - c2["mean"]
            se = math.sqrt(ms_error * (1.0 / c1["sample_size"] + 1.0 / c2["sample_size"]))
            t_val = diff / se if se > 0.0 else 0.0
            raw_p = float(2.0 * stats.t.sf(abs(t_val), df_residual))
            margin = critical_t * se
            simple_b_raw_p.append(raw_p)
            simple_b_records.append(
                {
                    "contrast_id": f"{b1}_vs_{b2}_at_{a_lvl}",
                    "family": f"Simple effects of {factor_b} within {factor_a}",
                    "family_type": "simple_b_within_a",
                    "condition": a_lvl,
                    "conditioning_level": a_lvl,
                    "first": b1,
                    "second": b2,
                    "contrast": f"{b1} minus {b2} at {factor_a}={a_lvl}",
                    "estimate": diff,
                    "standard_error": se,
                    "statistic": t_val,
                    "t_statistic": t_val,
                    "degrees_of_freedom": float(df_residual),
                    "raw_p_value": raw_p,
                    "confidence_interval": {
                        "lower": diff - margin,
                        "upper": diff + margin,
                        "level": conf_level,
                        "method": "pointwise Student-t interval",
                        "multiplicity_adjusted": False,
                        "status": "available",
                    },
                }
            )

    adj_simple_b = adjust_pvalues(simple_b_raw_p, method="holm")
    for rec, adj_p in zip(simple_b_records, adj_simple_b, strict=True):
        rec["adjusted_p_value"] = adj_p
        rec["adjustment_method"] = "Holm step-down procedure"
        rec["comparison_count"] = len(simple_b_records)
        rec["decision"] = "reject" if adj_p < dec_alpha else "fail_to_reject"

    # Family 3: Main-effect marginal comparisons for factor A (if > 2 levels, or pairwise)
    marginal_a_records = []
    marginal_a_raw_p: list[float] = []
    for a1, a2 in combinations(levels_a, 2):
        # Difference in EMMs:
        m1 = next(item["estimate"] for item in emm_a if item["level"] == a1)
        m2 = next(item["estimate"] for item in emm_a if item["level"] == a2)
        diff = m1 - m2
        # Variance = (MSE / J^2) * sum_j (1/n_1j + 1/n_2j)
        var_diff = (ms_error / (num_levels_b**2)) * sum(
            1.0 / cell_map[(a1, b_lvl)]["sample_size"] + 1.0 / cell_map[(a2, b_lvl)]["sample_size"]
            for b_lvl in levels_b
        )
        se = math.sqrt(var_diff) if var_diff > 0.0 else 0.0
        t_val = diff / se if se > 0.0 else 0.0
        raw_p = float(2.0 * stats.t.sf(abs(t_val), df_residual))
        margin = critical_t * se
        marginal_a_raw_p.append(raw_p)
        marginal_a_records.append(
            {
                "contrast_id": f"{a1}_vs_{a2}_marginal",
                "family": f"Marginal means for {factor_a}",
                "family_type": "marginal_a",
                "condition": None,
                "first": a1,
                "second": a2,
                "contrast": f"{a1} minus {a2} (marginal mean)",
                "estimate": diff,
                "standard_error": se,
                "statistic": t_val,
                "t_statistic": t_val,
                "degrees_of_freedom": float(df_residual),
                "raw_p_value": raw_p,
                "confidence_interval": {
                    "lower": diff - margin,
                    "upper": diff + margin,
                    "level": conf_level,
                    "method": "pointwise Student-t interval",
                    "multiplicity_adjusted": False,
                    "status": "available",
                },
            }
        )

    adj_marginal_a = adjust_pvalues(marginal_a_raw_p, method="holm")
    for rec, adj_p in zip(marginal_a_records, adj_marginal_a, strict=True):
        rec["adjusted_p_value"] = adj_p
        rec["adjustment_method"] = "Holm step-down procedure"
        rec["comparison_count"] = len(marginal_a_records)
        rec["decision"] = "reject" if adj_p < dec_alpha else "fail_to_reject"

    # Family 4: Main-effect marginal comparisons for factor B
    marginal_b_records = []
    marginal_b_raw_p: list[float] = []
    for b1, b2 in combinations(levels_b, 2):
        m1 = next(item["estimate"] for item in emm_b if item["level"] == b1)
        m2 = next(item["estimate"] for item in emm_b if item["level"] == b2)
        diff = m1 - m2
        var_diff = (ms_error / (num_levels_a**2)) * sum(
            1.0 / cell_map[(a_lvl, b1)]["sample_size"] + 1.0 / cell_map[(a_lvl, b2)]["sample_size"]
            for a_lvl in levels_a
        )
        se = math.sqrt(var_diff) if var_diff > 0.0 else 0.0
        t_val = diff / se if se > 0.0 else 0.0
        raw_p = float(2.0 * stats.t.sf(abs(t_val), df_residual))
        margin = critical_t * se
        marginal_b_raw_p.append(raw_p)
        marginal_b_records.append(
            {
                "contrast_id": f"{b1}_vs_{b2}_marginal",
                "family": f"Marginal means for {factor_b}",
                "family_type": "marginal_b",
                "condition": None,
                "first": b1,
                "second": b2,
                "contrast": f"{b1} minus {b2} (marginal mean)",
                "estimate": diff,
                "standard_error": se,
                "statistic": t_val,
                "t_statistic": t_val,
                "degrees_of_freedom": float(df_residual),
                "raw_p_value": raw_p,
                "confidence_interval": {
                    "lower": diff - margin,
                    "upper": diff + margin,
                    "level": conf_level,
                    "method": "pointwise Student-t interval",
                    "multiplicity_adjusted": False,
                    "status": "available",
                },
            }
        )

    adj_marginal_b = adjust_pvalues(marginal_b_raw_p, method="holm")
    for rec, adj_p in zip(marginal_b_records, adj_marginal_b, strict=True):
        rec["adjusted_p_value"] = adj_p
        rec["adjustment_method"] = "Holm step-down procedure"
        rec["comparison_count"] = len(marginal_b_records)
        rec["decision"] = "reject" if adj_p < dec_alpha else "fail_to_reject"

    # Family 5: Difference-of-differences for 2x2 design
    diff_of_diff = None
    if num_levels_a == 2 and num_levels_b == 2:
        a1, a2 = levels_a[0], levels_a[1]
        b1, b2 = levels_b[0], levels_b[1]
        # (mean_A2B2 - mean_A1B2) - (mean_A2B1 - mean_A1B1)
        m_22 = cell_map[(a2, b2)]["mean"]
        m_12 = cell_map[(a1, b2)]["mean"]
        m_21 = cell_map[(a2, b1)]["mean"]
        m_11 = cell_map[(a1, b1)]["mean"]
        dod_est = (m_22 - m_12) - (m_21 - m_11)
        dod_se = math.sqrt(
            ms_error
            * (
                1.0 / cell_map[(a2, b2)]["sample_size"]
                + 1.0 / cell_map[(a1, b2)]["sample_size"]
                + 1.0 / cell_map[(a2, b1)]["sample_size"]
                + 1.0 / cell_map[(a1, b1)]["sample_size"]
            )
        )
        dod_t = dod_est / dod_se if dod_se > 0.0 else 0.0
        dod_raw_p = float(2.0 * stats.t.sf(abs(dod_t), df_residual))
        dod_margin = critical_t * dod_se
        diff_of_diff = {
            "contrast_id": f"diff_of_diff_{a1}_{a2}_by_{b1}_{b2}",
            "family": "2x2 interaction difference-of-differences",
            "family_type": "interaction_contrast",
            "contrast": f"({a2} - {a1} at {b2}) minus ({a2} - {a1} at {b1})",
            "estimate": dod_est,
            "standard_error": dod_se,
            "statistic": dod_t,
            "t_statistic": dod_t,
            "degrees_of_freedom": float(df_residual),
            "raw_p_value": dod_raw_p,
            "adjusted_p_value": dod_raw_p,
            "adjustment_method": "single contrast",
            "confidence_interval": {
                "lower": dod_est - dod_margin,
                "upper": dod_est + dod_margin,
                "level": conf_level,
                "method": "Student-t interval",
                "multiplicity_adjusted": False,
                "status": "available",
            },
            "decision": "reject" if dod_raw_p < dec_alpha else "fail_to_reject",
        }

    all_followups = [
        *simple_a_records,
        *simple_b_records,
        *marginal_a_records,
        *marginal_b_records,
    ]
    if diff_of_diff is not None:
        all_followups.append(diff_of_diff)

    # Diagnostics
    residuals = np.asarray(model_full.resid, dtype=float)

    # Normality diagnostic (Shapiro-Wilk for small sample, normaltest for larger)
    normality_diag: dict[str, Any] = {}
    if 3 <= analyzed_rows <= 5000:
        s_stat, s_p = stats.shapiro(residuals)
        normality_diag["shapiro_wilk"] = {
            "statistic": float(s_stat),
            "p_value": float(s_p),
            "verdict": "fail_to_reject" if s_p >= 0.05 else "reject",
        }
    if analyzed_rows >= 8:
        try:
            n_stat, n_p = stats.normaltest(residuals)
            normality_diag["dagostino_pearson"] = {
                "statistic": float(n_stat),
                "p_value": float(n_p),
                "verdict": "fail_to_reject" if n_p >= 0.05 else "reject",
            }
        except Exception:
            pass

    # Homoscedasticity diagnostic across cells (Levene / Brown-Forsythe)
    homoscedasticity_diag: dict[str, Any] = {}
    if len(cell_values) >= 2 and all(len(c) >= 2 for c in cell_values):
        try:
            l_stat, l_p = stats.levene(*cell_values, center="median")
            homoscedasticity_diag["levene_median"] = {
                "statistic": float(l_stat),
                "p_value": float(l_p),
                "verdict": "fail_to_reject" if l_p >= 0.05 else "reject",
            }
        except Exception:
            pass

    diagnostics = {
        "residual_sample_size": analyzed_rows,
        "residual_degrees_of_freedom": float(df_residual),
        "residual_sum_of_squares": ss_error,
        "residual_mean_square": ms_error,
        "rmse": rmse,
        "normality": normality_diag,
        "homoscedasticity": homoscedasticity_diag,
        "cell_counts": {
            f"{c['factor_a']}:{c['factor_b']}": c["sample_size"] for c in cell_summaries
        },
        "empty_cells": empty_cells,
        "design_matrix_rank": int(model_full.df_model + 1),
    }

    # Warnings & limitations
    warnings: list[str] = []
    cell_sizes = [c["sample_size"] for c in cell_summaries]
    is_balanced = len(set(cell_sizes)) == 1

    if not is_balanced:
        warnings.append(
            f"Unbalanced cell sizes detected across {factor_a} x {factor_b} "
            f"(min={min(cell_sizes)}, max={max(cell_sizes)}). "
            f"Sums of squares evaluated using {sum_of_squares.upper()} convention."
        )

    if min(cell_sizes) < 3:
        warnings.append(
            f"One or more cells have very small sample size (min={min(cell_sizes)}), which may "
            "reduce statistical power and sensitivity of residual diagnostics."
        )

    if (
        normality_diag.get("shapiro_wilk", {}).get("verdict") == "reject"
        or normality_diag.get("dagostino_pearson", {}).get("verdict") == "reject"
    ):
        warnings.append(
            "Advisory diagnostic provides evidence against residual normality; interpret "
            "small-sample inference with caution."
        )

    if homoscedasticity_diag.get("levene_median", {}).get("verdict") == "reject":
        warnings.append(
            "Advisory Levene test provides evidence against equal cell variances; classical "
            "F-test assumes homoscedastic errors."
        )

    return {
        "method": "two_way_anova",
        "sum_of_squares_type": sum_of_squares,
        "outcome": outcome,
        "factor_a": factor_a,
        "factor_b": factor_b,
        "factor_a_levels": levels_a,
        "factor_b_levels": levels_b,
        "sample": {
            "original_rows": original_rows,
            "analyzed_rows": analyzed_rows,
            "excluded_rows": excluded_rows,
            "balanced": is_balanced,
        },
        "sample_size": analyzed_rows,
        "analyzed_rows": analyzed_rows,
        "original_rows": original_rows,
        "excluded_rows": excluded_rows,
        "terms": terms,
        "cell_summaries": cell_summaries,
        "estimated_marginal_means": {
            factor_a: emm_a,
            factor_b: emm_b,
        },
        "followups": all_followups,
        "diff_of_diff": diff_of_diff,
        "diagnostics": diagnostics,
        "warnings": warnings,
        "alpha": dec_alpha,
        "confidence_level": conf_level,
    }
