"""Validated ordinary least-squares regression and diagnostic records."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

from .exceptions import InsufficientDataError, InvalidDataError
from .uncertainty import ols_r_squared_case_bootstrap_ci

_NUMERIC_TYPES = {"continuous_numerical", "discrete_numerical"}
_CATEGORICAL_TYPES = {"nominal_categorical", "ordinal_categorical", "boolean"}


def _finite(value: Any, label: str, *, probability: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InsufficientDataError(f"Regression produced an invalid {label}.") from exc
    if not math.isfinite(number) or (probability and not 0 <= number <= 1):
        raise InsufficientDataError(f"Regression produced an invalid {label}.")
    return number


def _label(value: Any) -> str | int | float | bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidDataError("Categorical levels must be finite.")
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _levels(series: pd.Series, declared: dict[str, Any]) -> list[Any]:
    observed = list(pd.unique(series))
    declared_order = declared.get("ordinal_order") or declared.get("allowed_values")
    if declared_order:
        ordered = [item for item in declared_order if item in observed]
        ordered.extend(item for item in observed if item not in ordered)
        return ordered
    if isinstance(series.dtype, pd.CategoricalDtype):
        ordered = [item for item in series.cat.categories if item in observed]
        ordered.extend(item for item in observed if item not in ordered)
        return ordered
    return observed


def build_design_matrix(
    frame: pd.DataFrame,
    outcome: str,
    predictors: tuple[str, ...],
    variable_types: dict[str, str],
    *,
    reference_levels: dict[str, Any] | None = None,
    data_dictionary: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one complete-case treatment-coded design and its public metadata."""
    if not predictors:
        raise InvalidDataError("Regression requires at least one predictor.")
    columns = [outcome, *predictors]
    usable = frame[columns].dropna().copy()
    if len(usable) < 3:
        raise InsufficientDataError("Regression requires at least three complete cases.")
    outcome_type = variable_types.get(outcome)
    if outcome_type != "continuous_numerical" or not pd.api.types.is_numeric_dtype(usable[outcome]):
        raise InvalidDataError("Guided OLS regression requires a continuous numerical outcome.")
    y = np.asarray(usable[outcome], dtype=float)
    if not np.isfinite(y).all():
        raise InvalidDataError("The complete-case outcome contains nonfinite values.")
    if float(np.ptp(y)) == 0:
        raise InsufficientDataError("The outcome is constant in the complete-case sample.")

    references = reference_levels or {}
    dictionary = data_dictionary or {}
    unknown_references = set(references) - set(predictors)
    if unknown_references:
        raise InvalidDataError(
            f"reference_levels names non-predictors: {sorted(unknown_references)!r}."
        )
    arrays: list[np.ndarray] = [np.ones(len(usable), dtype=float)]
    terms: list[dict[str, Any]] = [
        {
            "term": "Intercept",
            "predictor": None,
            "kind": "intercept",
            "level": None,
            "reference_level": None,
        }
    ]
    coding: list[dict[str, Any]] = []
    for predictor in predictors:
        kind = variable_types.get(predictor)
        series = usable[predictor]
        if kind in _NUMERIC_TYPES:
            if not pd.api.types.is_numeric_dtype(series):
                raise InvalidDataError(
                    f"Predictor {predictor!r} is declared numerical but is not numeric."
                )
            values = np.asarray(series, dtype=float)
            if not np.isfinite(values).all():
                raise InvalidDataError(f"Predictor {predictor!r} contains nonfinite values.")
            if float(np.ptp(values)) == 0:
                raise InsufficientDataError(
                    f"Predictor {predictor!r} has no variation in the analyzed sample."
                )
            arrays.append(values)
            terms.append(
                {
                    "term": predictor,
                    "predictor": predictor,
                    "kind": "continuous" if kind == "continuous_numerical" else "discrete",
                    "level": None,
                    "reference_level": None,
                }
            )
            coding.append(
                {
                    "predictor": predictor,
                    "type": kind,
                    "coding": "original numeric units",
                    "terms": [predictor],
                    "reference_level": None,
                }
            )
            if predictor in references:
                raise InvalidDataError(
                    f"reference_levels[{predictor!r}] is invalid because the predictor "
                    "is numerical."
                )
            continue
        if kind not in _CATEGORICAL_TYPES:
            raise InvalidDataError(
                f"Predictor {predictor!r} has unsupported analytical type {kind!r}."
            )
        levels = _levels(series, dictionary.get(predictor, {}))
        if len(levels) < 2:
            raise InsufficientDataError(
                f"Categorical predictor {predictor!r} has fewer than two observed "
                "complete-case levels."
            )
        reference = references.get(predictor, levels[0])
        if reference not in levels:
            raise InvalidDataError(
                f"Reference level {reference!r} is not observed for predictor {predictor!r}."
            )
        comparison_levels = [level for level in levels if level != reference]
        encoded_terms = []
        for level in comparison_levels:
            term_name = f"{predictor}[{_label(level)!s} vs {_label(reference)!s}]"
            arrays.append(np.asarray(series == level, dtype=float))
            terms.append(
                {
                    "term": term_name,
                    "predictor": predictor,
                    "kind": "categorical",
                    "level": _label(level),
                    "reference_level": _label(reference),
                }
            )
            encoded_terms.append(term_name)
        coding.append(
            {
                "predictor": predictor,
                "type": kind,
                "coding": "treatment",
                "observed_levels": [_label(item) for item in levels],
                "reference_level": _label(reference),
                "terms": encoded_terms,
                "ordinal_policy": (
                    "treated categorically; no equal spacing assumed"
                    if kind == "ordinal_categorical"
                    else None
                ),
            }
        )

    matrix = np.column_stack(arrays)
    if not np.isfinite(matrix).all():
        raise InvalidDataError("The regression design matrix contains nonfinite values.")
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != matrix.shape[1]:
        raise InsufficientDataError(
            "The regression design matrix is not full rank. At least one term is perfectly "
            "determined by the others; PyAutoStat will not silently remove a predictor."
        )
    residual_df = len(usable) - matrix.shape[1]
    if residual_df <= 0:
        raise InsufficientDataError(
            "Regression needs positive residual degrees of freedom after fitting all terms."
        )
    return {
        "y": y,
        "x": matrix,
        "terms": terms,
        "coding": coding,
        "outcome": outcome,
        "predictors": list(predictors),
        "original_rows": int(len(frame)),
        "analyzed_rows": int(len(usable)),
        "excluded_rows": int(len(frame) - len(usable)),
        "complete_case_columns": columns,
        "rank": rank,
        "parameter_count": int(matrix.shape[1]),
        "residual_df": int(residual_df),
        "analysis_positions": list(np.flatnonzero(frame[columns].notna().all(axis=1))),
    }


def fit_ols(
    frame: pd.DataFrame,
    outcome: str,
    predictors: tuple[str, ...],
    variable_types: dict[str, str],
    *,
    covariance_type: str = "classical",
    reference_levels: dict[str, Any] | None = None,
    data_dictionary: dict[str, dict[str, Any]] | None = None,
    confidence_level: float = 0.95,
    alpha: float = 0.05,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Fit OLS once and return strict JSON-ready coefficients and diagnostics."""
    covariance = covariance_type.upper()
    if covariance not in {"CLASSICAL", "HC3"}:
        raise InvalidDataError("covariance_type must be 'classical' or 'HC3'.")
    design = build_design_matrix(
        frame,
        outcome,
        predictors,
        variable_types,
        reference_levels=reference_levels,
        data_dictionary=data_dictionary,
    )
    try:
        base = sm.OLS(design["y"], design["x"], hasconst=True).fit()
        fitted = base if covariance == "CLASSICAL" else base.get_robustcov_results(cov_type="HC3")
    except Exception as exc:
        raise InsufficientDataError(f"The OLS backend could not fit this model: {exc}") from exc
    interval = np.asarray(fitted.conf_int(alpha=1 - confidence_level), dtype=float)
    arrays = [
        np.asarray(fitted.params, dtype=float),
        np.asarray(fitted.bse, dtype=float),
        np.asarray(fitted.tvalues, dtype=float),
        np.asarray(fitted.pvalues, dtype=float),
        interval,
    ]
    if any(not np.isfinite(item).all() for item in arrays):
        raise InsufficientDataError("OLS coefficient inference produced nonfinite values.")
    y_sd = float(np.std(design["y"], ddof=1))
    coefficients = []
    for index, term in enumerate(design["terms"]):
        beta = None
        beta_status = "not_applicable"
        if term["kind"] == "continuous":
            x_sd = float(np.std(design["x"][:, index], ddof=1))
            beta = float(fitted.params[index]) * x_sd / y_sd
            beta_status = "available"
        coefficients.append(
            {
                **term,
                "term_id": f"term_{index}",
                "term_label": term["term"],
                "term_type": term["kind"],
                "covariance_type": "classical" if covariance == "CLASSICAL" else "HC3",
                "estimate": _finite(fitted.params[index], "coefficient"),
                "standard_error": _finite(fitted.bse[index], "coefficient standard error"),
                "statistic": _finite(fitted.tvalues[index], "coefficient t statistic"),
                "p_value": _finite(fitted.pvalues[index], "coefficient p-value", probability=True),
                "confidence_interval": {
                    "lower": _finite(interval[index, 0], "coefficient interval"),
                    "upper": _finite(interval[index, 1], "coefficient interval"),
                    "level": confidence_level,
                    "method": f"{covariance} covariance t interval",
                    "quantity": "regression coefficient",
                },
                "standardized_beta": beta,
                "standardized_beta_status": beta_status,
                "decision": "reject" if float(fitted.pvalues[index]) < alpha else "fail_to_reject",
            }
        )

    vif = []
    for index, term in enumerate(design["terms"][1:], start=1):
        value = float(variance_inflation_factor(design["x"], index))
        if not math.isfinite(value):
            advisory = "nonfinite"
        elif value >= 10:
            advisory = "strong_collinearity_signal"
        elif value >= 5:
            advisory = "elevated_collinearity_signal"
        else:
            advisory = "no_large_signal"
        vif.append(
            {
                "term": term["term"],
                "predictor": term["predictor"],
                "value": value if math.isfinite(value) else None,
                "status": "available" if math.isfinite(value) else "nonfinite",
                "advisory": advisory,
            }
        )
    lm, lm_p, bp_f, bp_f_p = het_breuschpagan(base.resid, design["x"])
    jb, jb_p, skew, kurtosis = jarque_bera(base.resid)
    influence = OLSInfluence(base)
    cooks = np.asarray(influence.cooks_distance[0], dtype=float)
    leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
    studentized = np.asarray(influence.resid_studentized_external, dtype=float)
    if not all(np.isfinite(item).all() for item in (cooks, leverage, studentized)):
        raise InsufficientDataError("Influence diagnostics produced nonfinite values.")
    n = design["analyzed_rows"]
    p = design["parameter_count"]
    cook_threshold = 4 / n
    leverage_threshold = 2 * p / n
    influence_flags = (
        (cooks > cook_threshold) | (leverage > leverage_threshold) | (np.abs(studentized) > 3)
    )
    model_f = None if fitted.fvalue is None else _finite(fitted.fvalue, "model F statistic")
    model_p = (
        None
        if fitted.f_pvalue is None
        else _finite(fitted.f_pvalue, "model F p-value", probability=True)
    )
    warnings = []
    if design["residual_df"] < 5:
        warnings.append("Residual degrees of freedom are very low; inference is unstable.")
    if covariance == "CLASSICAL" and float(lm_p) < alpha:
        warnings.append(
            "Breusch-Pagan flagged non-constant residual variance; classical standard errors "
            "may be sensitive. HC3 is an explicit alternative and was not selected automatically."
        )
    if int(np.sum(influence_flags)):
        warnings.append(
            "Influence diagnostics flagged observations for review; no rows were removed."
        )
    r2_ci = ols_r_squared_case_bootstrap_ci(
        design["y"],
        design["x"],
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )
    return {
        "method": "ordinary least squares linear regression",
        "outcome": outcome,
        "predictors": list(predictors),
        "target": "conditional_mean",
        "covariance_type": "classical" if covariance == "CLASSICAL" else "HC3",
        "intercept": True,
        "sample": {
            "original_rows": design["original_rows"],
            "analyzed_rows": n,
            "excluded_rows": design["excluded_rows"],
            "complete_case_columns": design["complete_case_columns"],
            "missing_data_policy": "complete cases across outcome and all predictors",
        },
        "design_matrix": {
            "term_names": [term["term"] for term in design["terms"]],
            "term_mapping": design["terms"],
            "coding": design["coding"],
            "rank": design["rank"],
            "parameter_count": p,
            "full_rank": True,
            "matrix_values_included": False,
        },
        "coefficients": coefficients,
        "model_fit": {
            "analyzed_rows": n,
            "parameter_count": p,
            "model_rank": design["rank"],
            "covariance_type": "classical" if covariance == "CLASSICAL" else "HC3",
            "r_squared": _finite(base.rsquared, "R-squared"),
            "r_squared_confidence_interval": r2_ci,
            "adjusted_r_squared": _finite(base.rsquared_adj, "adjusted R-squared"),
            "model_f_statistic": model_f,
            "model_f_p_value": model_p,
            "model_degrees_of_freedom": _finite(base.df_model, "model degrees of freedom"),
            "residual_degrees_of_freedom": _finite(base.df_resid, "residual degrees of freedom"),
            "residual_sum_of_squares": _finite(base.ssr, "residual sum of squares"),
            "residual_standard_error": _finite(
                math.sqrt(base.ssr / base.df_resid), "residual standard error"
            ),
            "rmse": _finite(math.sqrt(float(np.mean(np.square(base.resid)))), "RMSE"),
        },
        "diagnostics": {
            "vif": {
                "terms": vif,
                "maximum": max(
                    (item["value"] for item in vif if item["value"] is not None), default=None
                ),
                "threshold_policy": (
                    "VIF values around 5 or 10 are review heuristics, not pass/fail rules."
                ),
            },
            "breusch_pagan": {
                "status": "rejected" if float(lm_p) < alpha else "not_rejected",
                "lm_statistic": _finite(lm, "Breusch-Pagan LM statistic"),
                "lm_p_value": _finite(lm_p, "Breusch-Pagan LM p-value", probability=True),
                "f_statistic": _finite(bp_f, "Breusch-Pagan F statistic"),
                "f_p_value": _finite(bp_f_p, "Breusch-Pagan F p-value", probability=True),
                "interpretation_policy": "diagnostic evidence only; covariance was not changed",
            },
            "residual_normality": {
                "method": "Jarque-Bera",
                "status": "rejected" if float(jb_p) < alpha else "not_rejected",
                "statistic": _finite(jb, "Jarque-Bera statistic"),
                "p_value": _finite(jb_p, "Jarque-Bera p-value", probability=True),
                "skewness": _finite(skew, "residual skewness"),
                "kurtosis": _finite(kurtosis, "residual kurtosis"),
                "interpretation_policy": "diagnostic evidence; not an automatic model invalidation",
            },
            "influence": {
                "status": "review" if int(np.sum(influence_flags)) else "no_flags",
                "cook_threshold": cook_threshold,
                "leverage_threshold": leverage_threshold,
                "studentized_residual_threshold": 3.0,
                "flagged_count": int(np.sum(influence_flags)),
                "cook_flagged_count": int(np.sum(cooks > cook_threshold)),
                "leverage_flagged_count": int(np.sum(leverage > leverage_threshold)),
                "studentized_flagged_count": int(np.sum(np.abs(studentized) > 3)),
                "maximum_cooks_distance": float(np.max(cooks)),
                "maximum_leverage": float(np.max(leverage)),
                "maximum_absolute_studentized_residual": float(np.max(np.abs(studentized))),
                "row_identifiers_included": False,
                "rows_removed": 0,
                "interpretation_policy": (
                    "Thresholds are review heuristics; observations are never removed "
                    "automatically."
                ),
            },
            "condition_number": _finite(base.condition_number, "condition number"),
        },
        "warnings": warnings,
    }
