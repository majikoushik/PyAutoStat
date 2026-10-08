"""Validated binary logistic regression and partial Pearson backends.

This module reuses the Phase 4 design-matrix infrastructure (``build_design_matrix``
categorical coding, reference-level handling, etc.) for the logistic regression case
and provides a standalone partial Pearson correlation backend.  It never runs an OLS
fit as a primary computation or produces predictive model evaluations.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .exceptions import InsufficientDataError, InvalidDataError
from .regression import _finite, _label, build_design_matrix
from .usability import invalid_level_message

# ---------------------------------------------------------------------------
# Binary outcome helper
# ---------------------------------------------------------------------------


def _validate_binary_outcome(
    series: pd.Series,
    outcome: str,
    event_level: Any,
) -> tuple[Any, Any]:
    """Return (non_event_level, event_level) for the binary outcome.

    ``event_level`` must be supplied by the workflow (or derived only by its
    documented Boolean/declared-0/1 rules); this backend never chooses an event
    from an arbitrary ordering of the observed values.
    """
    clean = series.dropna()
    unique_vals = list(pd.unique(clean))
    if len(unique_vals) != 2:
        raise InsufficientDataError(
            f"Binary logistic regression requires exactly two observed levels in {outcome!r}; "
            f"{len(unique_vals)} were found."
        )
    if event_level not in unique_vals:
        raise InvalidDataError(
            invalid_level_message("event_level", event_level, outcome, unique_vals)
        )
    non_event = [value for value in unique_vals if value != event_level][0]
    return non_event, event_level


def _binary_y(series: pd.Series, non_event: Any, event_level: Any) -> np.ndarray:
    """Encode the binary outcome as 0/1 float (caller guarantees no NaN rows)."""
    y = (series == event_level).astype(float).values
    if not np.isin(y, [0.0, 1.0]).all():
        raise InsufficientDataError("Binary outcome encoding produced non-0/1 values.")
    return y


# ---------------------------------------------------------------------------
# Design matrix builder for logistic regression
# ---------------------------------------------------------------------------


def build_logistic_design_matrix(
    frame: pd.DataFrame,
    outcome: str,
    predictors: tuple[str, ...],
    variable_types: dict[str, str],
    *,
    event_level: Any,
    reference_levels: dict[str, Any] | None = None,
    data_dictionary: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a complete-case binary design matrix reusing Phase 4 predictor logic.

    The outcome column is encoded as 0/1 using the explicit ``event_level``;
    this backend never chooses an event from an arbitrary ordering.

    Parameters
    ----------
    frame : pd.DataFrame
        Source dataset; never mutated.
    outcome : str
        Binary outcome column name.
    predictors : tuple of str
        Ordered predictor column names (at least one required).
    variable_types : dict
        Analytical types from the profiler; must include all predictors.
    reference_levels : dict, optional
        Reference level overrides for categorical predictors.
    data_dictionary : dict, optional
        Researcher declarations used by the shared predictor-coding builder.

    Returns
    -------
    dict
        Keys include ``y`` (0/1 ndarray), ``x`` (design matrix ndarray),
        ``terms``, ``coding``, sample counts, ``event_level``,
        ``non_event_level``, and ``event_orientation_warning``.
    """
    if not predictors:
        raise InvalidDataError("Binary logistic regression requires at least one predictor.")
    columns = [outcome, *predictors]
    usable = frame[columns].dropna().copy()
    if len(usable) < 10:
        raise InsufficientDataError(
            "Binary logistic regression requires at least ten complete cases."
        )

    non_event, event_level = _validate_binary_outcome(usable[outcome], outcome, event_level)
    y = _binary_y(usable[outcome], non_event, event_level)

    event_count = int(y.sum())
    non_event_count = int(len(y) - event_count)
    if event_count < 2 or non_event_count < 2:
        raise InsufficientDataError(
            "Each binary outcome category must have at least two observed cases."
        )

    helper_outcome = "__pyautostat_binary_design_outcome__"
    while helper_outcome in usable.columns:
        helper_outcome += "_"
    shared_frame = usable.copy()
    shared_frame[helper_outcome] = np.arange(len(shared_frame), dtype=float)
    shared_types = {**variable_types, helper_outcome: "continuous_numerical"}
    shared = build_design_matrix(
        shared_frame,
        helper_outcome,
        predictors,
        shared_types,
        reference_levels=reference_levels,
        data_dictionary=data_dictionary,
    )
    matrix = shared["x"]
    terms = shared["terms"]
    coding = shared["coding"]
    rank = shared["rank"]
    residual_df = shared["residual_df"]

    return {
        "y": y,
        "x": matrix,
        "terms": terms,
        "coding": coding,
        "outcome": outcome,
        "predictors": list(predictors),
        "event_level": _label(event_level),
        "non_event_level": _label(non_event),
        "event_count": event_count,
        "non_event_count": non_event_count,
        "original_rows": int(len(frame)),
        "analyzed_rows": int(len(usable)),
        "excluded_rows": int(len(frame) - len(usable)),
        "complete_case_columns": columns,
        "rank": rank,
        "parameter_count": int(matrix.shape[1]),
        "residual_df": int(residual_df),
        "event_orientation_warning": None,
    }


# ---------------------------------------------------------------------------
# Logistic regression fitter
# ---------------------------------------------------------------------------


def fit_logit(
    frame: pd.DataFrame,
    outcome: str,
    predictors: tuple[str, ...],
    variable_types: dict[str, str],
    *,
    event_level: Any,
    covariance_type: str = "classical",
    reference_levels: dict[str, Any] | None = None,
    data_dictionary: dict[str, dict[str, Any]] | None = None,
    confidence_level: float = 0.95,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit one binary logistic regression model and return a JSON-ready result.

    Satisfies the complete method contract:

    * **Estimand**: conditional log-odds of the declared event level.
    * **Validation**: binary outcome (exactly 2 levels, ≥ 2 per class, ≥ 10
      complete cases); predictors follow Phase 4 rules.
    * **Execution**: statsmodels Logit with MLE; Wald inference.
    * **Uncertainty**: Wald confidence intervals on log-odds scale; exponentiated
      for odds-ratio CIs.
    * **Effect / Estimate**: odds ratios with Wald intervals per coefficient.
    * **Interpretation**: odds ratios only; no direct probability statements;
      no causal language; McFadden pseudo-R² is a likelihood-based fit index,
      not comparable with OLS R².
    * **Assumptions / Diagnostics**: separation check (large-z flag),
      convergence, event rate.
    * **Reporting**: all estimates, SE, Wald z, p, CI, OR, OR-CI, and
      McFadden pseudo-R².
    * **Audit**: all numerical sources identified.
    * **Reproducibility**: deterministic MLE; no stochastic components.
    """
    if not 0 < confidence_level < 1:
        raise InvalidDataError("confidence_level must be strictly between 0 and 1.")
    if not 0 < alpha < 1:
        raise InvalidDataError("alpha must be strictly between 0 and 1.")
    if covariance_type not in {"classical", "HC3"}:
        raise InvalidDataError("covariance_type must be 'classical' or 'HC3'.")

    design = build_logistic_design_matrix(
        frame,
        outcome,
        predictors,
        variable_types,
        event_level=event_level,
        reference_levels=reference_levels,
        data_dictionary=data_dictionary,
    )
    y = design["y"]
    x = design["x"]
    n = design["analyzed_rows"]
    p = design["parameter_count"]

    fit_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = sm.Logit(y, x)
            result = model.fit(
                disp=0,
                cov_type="nonrobust" if covariance_type == "classical" else "HC3",
            )
        for w in caught:
            msg = str(w.message)
            if any(kw in msg.lower() for kw in ("converge", "iteration limit", "singular")):
                fit_warnings.append(
                    "Logistic regression MLE convergence warning; estimates may be unreliable."
                )
    except (PerfectSeparationError, np.linalg.LinAlgError) as exc:
        raise InsufficientDataError(
            "The outcome is perfectly or nearly separated by the predictors. Ordinary "
            "maximum-likelihood logistic coefficients are not reliably estimable."
        ) from exc
    except Exception as exc:
        raise InsufficientDataError(
            f"The logistic regression backend could not fit this model: {exc}"
        ) from exc

    if not result.mle_retvals.get("converged", True):
        raise InsufficientDataError(
            "The logistic optimization did not converge; complete or quasi-separation may be "
            "present, so coefficient inference is unavailable."
        )

    half_alpha = (1 - confidence_level) / 2
    ci_array = np.asarray(result.conf_int(alpha=2 * half_alpha), dtype=float)
    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(result.bse, dtype=float)
    tvalues = np.asarray(result.tvalues, dtype=float)
    pvalues = np.asarray(result.pvalues, dtype=float)

    if any(not np.isfinite(item).all() for item in [params, bse, tvalues, pvalues, ci_array]):
        raise InsufficientDataError(
            "Logistic regression coefficient inference produced nonfinite values. "
            "Quasi-complete separation is the most likely cause; inspect the data."
        )

    separation_signal = bool(
        np.any(np.abs(params) > 25)
        or np.any(bse > 100)
        or np.any(~np.isfinite(np.asarray(result.predict(), dtype=float)))
    )
    if separation_signal:
        raise InsufficientDataError(
            "The outcome is perfectly or nearly separated by the predictors. Ordinary "
            "maximum-likelihood logistic coefficients are not reliably estimable."
        )

    vif_records = []
    for index, term in enumerate(design["terms"]):
        if index == 0:
            continue
        try:
            vif = float(variance_inflation_factor(x, index))
        except (ValueError, np.linalg.LinAlgError):
            vif = math.nan
        if not math.isfinite(vif):
            advisory = "nonfinite"
        elif vif >= 10:
            advisory = "strong_collinearity_signal"
        elif vif >= 5:
            advisory = "elevated_collinearity_signal"
        else:
            advisory = "no_large_signal"
        vif_records.append(
            {
                "term": term["term"],
                "predictor": term["predictor"],
                "value": vif if math.isfinite(vif) else None,
                "status": "available" if math.isfinite(vif) else "nonfinite",
                "advisory": advisory,
            }
        )
    condition_number = _finite(np.linalg.cond(x), "condition number")

    coefficients = []
    for index, term in enumerate(design["terms"]):
        log_or = float(params[index])
        se = float(bse[index])
        z = float(tvalues[index])
        pv = float(pvalues[index])
        ci_lower_log = float(ci_array[index, 0])
        ci_upper_log = float(ci_array[index, 1])
        coefficients.append(
            {
                **term,
                "term_id": f"term_{index}",
                "term_label": term["term"],
                "term_type": term["kind"],
                "estimate": log_or,
                "standard_error": se,
                "statistic": z,
                "statistic_type": "Wald z",
                "p_value": pv,
                "confidence_interval": {
                    "lower": ci_lower_log,
                    "upper": ci_upper_log,
                    "level": confidence_level,
                    "method": "Wald z interval on log-odds scale",
                    "quantity": "log odds ratio",
                },
                "odds_ratio": float(math.exp(log_or)),
                "odds_ratio_ci": {
                    "lower": float(math.exp(ci_lower_log)),
                    "upper": float(math.exp(ci_upper_log)),
                    "level": confidence_level,
                    "method": "Wald z interval exponentiated",
                    "quantity": "odds ratio",
                },
                "decision": "reject" if pv < alpha else "fail_to_reject",
                "covariance_type": covariance_type,
            }
        )

    llf = _finite(result.llf, "log-likelihood")
    llnull = _finite(result.llnull, "null log-likelihood")
    mcfadden_r2: float | None = None
    if llnull != 0 and math.isfinite(llf) and math.isfinite(llnull):
        candidate = 1.0 - llf / llnull
        mcfadden_r2 = candidate if math.isfinite(candidate) else None

    lr_df = p - 1
    lr_statistic = float(-2.0 * (llnull - llf))
    from scipy import stats as _scipy_stats

    lr_p_value: float | None = (
        float(_scipy_stats.chi2.sf(lr_statistic, lr_df)) if lr_df > 0 else None
    )

    event_rate = float(design["event_count"]) / n
    model_warnings = list(dict.fromkeys(fit_warnings))
    if design["event_orientation_warning"]:
        model_warnings.append(design["event_orientation_warning"])
    if event_rate < 0.1 or event_rate > 0.9:
        model_warnings.append(
            f"The event rate is {event_rate:.1%}; highly imbalanced outcomes may make "
            "coefficient estimates sensitive."
        )
    if design["residual_df"] < 5:
        model_warnings.append(
            "Residual degrees of freedom are very low; coefficient inference may be unstable."
        )

    aic_val: float | None = None
    bic_val: float | None = None
    if hasattr(result, "aic") and result.aic is not None:
        try:
            aic_val = float(result.aic)
            if not math.isfinite(aic_val):
                aic_val = None
        except (TypeError, ValueError):
            pass
    if hasattr(result, "bic") and result.bic is not None:
        try:
            bic_val = float(result.bic)
            if not math.isfinite(bic_val):
                bic_val = None
        except (TypeError, ValueError):
            pass

    return {
        "method": "binary logistic regression",
        "outcome": outcome,
        "predictors": list(predictors),
        "target": "conditional_log_odds",
        "event_level": design["event_level"],
        "non_event_level": design["non_event_level"],
        "event_count": design["event_count"],
        "non_event_count": design["non_event_count"],
        "event_rate": event_rate,
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
            "model_rank": design["rank"],
            "covariance_type": covariance_type,
            "full_rank": True,
            "matrix_values_included": False,
        },
        "coefficients": coefficients,
        "model_fit": {
            "analyzed_rows": n,
            "parameter_count": p,
            "log_likelihood": llf,
            "null_log_likelihood": llnull,
            "lr_statistic": lr_statistic,
            "lr_degrees_of_freedom": lr_df,
            "lr_p_value": lr_p_value,
            "mcfadden_r2": mcfadden_r2,
            "aic": aic_val,
            "bic": bic_val,
            "residual_degrees_of_freedom": int(design["residual_df"]),
            "model_degrees_of_freedom": int(lr_df),
        },
        "diagnostics": {
            "separation_signal": separation_signal,
            "converged": bool(result.mle_retvals.get("converged", True)),
            "event_rate": event_rate,
            "independence_policy": "Declared design; not verified from values.",
            "iterations": int(result.mle_retvals.get("iterations", 0)),
            "covariance_type": covariance_type,
            "vif": {
                "terms": vif_records,
                "maximum": max(
                    (item["value"] for item in vif_records if item["value"] is not None),
                    default=None,
                ),
                "threshold_policy": (
                    "VIF values around 5 or 10 are review heuristics, not pass/fail rules."
                ),
            },
            "condition_number": condition_number,
        },
        "warnings": model_warnings,
    }


# ---------------------------------------------------------------------------
# Partial Pearson correlation backend
# ---------------------------------------------------------------------------


def partial_pearson_correlation(
    frame: pd.DataFrame,
    first: str,
    second: str,
    controls: tuple[str, ...],
    *,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Partial Pearson correlation between *first* and *second* holding *controls* fixed.

    Derived from OLS residuals: both target variables are regressed on an
    intercept and the controls; the partial correlation is the Pearson
    correlation of their residuals.  This is mathematically equivalent to the
    standard partial correlation formula.

    Satisfies the complete method contract:

    * **Estimand**: linear association between *first* and *second* after
      partialling out *controls*.
    * **Validation**: all columns must be numeric; at least (k + 3) complete
      cases where k = number of controls.
    * **Execution**: OLS residualisation; Pearson r of residuals; t test.
    * **Uncertainty**: complete-row percentile bootstrap with both adjustment
      models refitted in every valid draw.
    * **Effect / Estimate**: partial r.
    * **Interpretation**: partial correlation alone does not establish causal
      control or independent causal effects.
    * **Assumptions / Diagnostics**: numeric columns, full-rank control matrix,
      variation in both residuals.
    * **Reporting**: partial r, t, df, p, CI.
    * **Audit**: all numerical sources identified.
    * **Reproducibility**: deterministic local random generator with its seed
      and requested/valid resample counts recorded.

    Parameters
    ----------
    frame : pd.DataFrame
        Source dataset; never mutated.
    first : str
        First primary variable column.
    second : str
        Second primary variable column.
    controls : tuple of str
        One or more control variable columns.
    confidence_level : float
        Percentile-bootstrap CI level; default 0.95.

    Returns
    -------
    dict
        JSON-safe result record.
    """
    if not 0 < confidence_level < 1:
        raise InvalidDataError("confidence_level must be strictly between 0 and 1.")
    if not controls:
        raise InvalidDataError("At least one control variable is required.")
    if len(set(controls)) != len(controls):
        raise InvalidDataError("Control variables must not contain duplicates.")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 100
    ):
        raise InvalidDataError("bootstrap_samples must be an integer of at least 100.")
    if random_state is not None and (
        isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0
    ):
        raise InvalidDataError("random_state must be a nonnegative integer or None.")
    if first == second:
        raise InvalidDataError("The two primary variables must be different columns.")
    overlap = {first, second} & set(controls)
    if overlap:
        raise InvalidDataError(
            f"Control variables must differ from the primary variables; "
            f"overlap: {sorted(overlap)!r}."
        )
    all_cols = [first, second, *controls]
    for col in all_cols:
        if col not in frame.columns:
            from .exceptions import ColumnNotFoundError

            raise ColumnNotFoundError(f"Column {col!r} does not exist.")
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise InvalidDataError(f"Column {col!r} must be numeric for partial correlation.")

    usable = frame[all_cols].dropna()
    n = len(usable)
    k = len(controls)
    if n < k + 3:
        raise InsufficientDataError(
            f"Partial Pearson correlation needs at least {k + 3} complete observations "
            f"(number_of_controls + 3)."
        )

    x_controls = np.column_stack(
        [np.ones(n)] + [np.asarray(usable[c], dtype=float) for c in controls]
    )
    if not np.isfinite(x_controls).all():
        raise InvalidDataError("Control variables contain nonfinite values.")
    rank = int(np.linalg.matrix_rank(x_controls))
    if rank != x_controls.shape[1]:
        raise InsufficientDataError(
            "The control variable design matrix is not full rank; at least one control "
            "is a perfect linear combination of others."
        )

    def _resid(target: np.ndarray) -> np.ndarray:
        try:
            fit = sm.OLS(target, x_controls).fit()
        except Exception as exc:
            raise InsufficientDataError(
                f"Partial correlation residualisation failed: {exc}"
            ) from exc
        r = np.asarray(fit.resid, dtype=float)
        if not np.isfinite(r).all():
            raise InsufficientDataError("Residualisation produced nonfinite values.")
        return r

    e1 = _resid(np.asarray(usable[first], dtype=float))
    e2 = _resid(np.asarray(usable[second], dtype=float))

    target_x = np.asarray(usable[first], dtype=float)
    target_y = np.asarray(usable[second], dtype=float)

    def _residual_varies(residual: np.ndarray, target: np.ndarray) -> bool:
        tolerance = np.finfo(float).eps ** 0.75 * max(1.0, float(np.ptp(target)))
        return bool(float(np.ptp(residual)) > tolerance)

    if not _residual_varies(e1, target_x) or not _residual_varies(e2, target_y):
        raise InsufficientDataError(
            "A residualised variable has zero variance; partial correlation is undefined."
        )

    pr = float(np.corrcoef(e1, e2)[0, 1])
    if not math.isfinite(pr) or not -1 <= pr <= 1:
        raise InsufficientDataError("Partial correlation coefficient is invalid.")

    df = n - k - 2
    if df <= 0:
        raise InsufficientDataError(
            "Insufficient residual degrees of freedom for partial correlation inference."
        )

    denom = math.sqrt(max(0.0, 1 - pr**2))
    from scipy import stats as _scipy_stats

    if denom == 0:
        t_stat = None
        p_value = 0.0
    else:
        t_stat = pr * math.sqrt(df) / denom
        p_value = float(2.0 * _scipy_stats.t.sf(abs(t_stat), df))
    if not math.isfinite(p_value) or not 0 <= p_value <= 1:
        raise InsufficientDataError("Partial correlation p-value is invalid.")

    seed = 0 if random_state is None else random_state
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, n, size=n)
        control_draw = x_controls[indices]
        if np.linalg.matrix_rank(control_draw) != control_draw.shape[1]:
            continue
        residual_x = np.asarray(sm.OLS(target_x[indices], control_draw).fit().resid)
        residual_y = np.asarray(sm.OLS(target_y[indices], control_draw).fit().resid)
        if not _residual_varies(residual_x, target_x[indices]) or not _residual_varies(
            residual_y, target_y[indices]
        ):
            continue
        candidate = float(np.corrcoef(residual_x, residual_y)[0, 1])
        if math.isfinite(candidate):
            estimates.append(candidate)
    interval: dict[str, Any] | None = None
    if len(estimates) >= max(50, bootstrap_samples // 2):
        tail = (1 - confidence_level) / 2
        lower, upper = np.quantile(estimates, [tail, 1 - tail])
        interval = {
            "lower": float(lower),
            "upper": float(upper),
            "level": confidence_level,
            "method": "complete-row percentile bootstrap with model refitting",
            "quantity": "partial Pearson r",
            "requested_resamples": bootstrap_samples,
            "valid_resamples": len(estimates),
            "random_seed": seed,
        }

    return {
        "test": "Partial Pearson correlation",
        "statistic": t_stat,
        "p_value": p_value,
        "partial_r": pr,
        "degrees_of_freedom": df,
        "controls": list(controls),
        "confidence_interval": interval,
        "effect_size": {
            "name": "partial Pearson r",
            "value": pr,
            "confidence_interval": interval,
        },
        "sample_size": n,
        "excluded_rows": int(len(frame) - n),
        "alternative": "two-sided",
        "limitation": (
            "Adjustment describes association conditional on the included control variables. "
            "It does not establish that confounding has been removed."
        ),
        "control_design": {
            "intercept": True,
            "effective_control_terms": k,
            "rank": rank,
            "coding": "original numeric units",
        },
        "bootstrap": {
            "method": "complete-row percentile bootstrap with model refitting",
            "requested_resamples": bootstrap_samples,
            "valid_resamples": len(estimates),
            "random_seed": seed,
            "confidence_level": confidence_level,
        },
        "warnings": (
            [] if interval is not None else ["Partial-correlation bootstrap CI is unavailable."]
        ),
    }
