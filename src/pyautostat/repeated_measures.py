"""Validated repeated-measures backends for 3+ condition within-subject panels."""

from __future__ import annotations

import inspect
import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import ColumnNotFoundError, InsufficientDataError, InvalidTestError
from .inference import _scipy_result_value
from .multigroup import adjust_pvalues
from .uncertainty import (
    friedman_kendall_w_bootstrap_ci,
    matched_pairs_rank_biserial_bootstrap_ci,
    paired_cohen_dz_ci,
    repeated_measures_partial_eta_squared_ci,
)
from .usability import invalid_condition_order_message, observed_levels


def _helmert_contrasts(k: int) -> np.ndarray:
    """Construct a normalized k x (k-1) Helmert contrast matrix.

    Satisfies M.T @ M = I_(k-1) and M.T @ 1 = 0.
    """
    if k < 2:
        raise InvalidTestError("Helmert contrasts require at least two conditions.")
    matrix = np.zeros((k, k - 1), dtype=float)
    for j in range(1, k):
        scale = 1.0 / math.sqrt(j * (j + 1))
        matrix[:j, j - 1] = scale
        matrix[j, j - 1] = -float(j) * scale
    return matrix


def repeated_panel(
    frame: pd.DataFrame,
    unit_id: str,
    condition_col: str,
    value_col: str,
    condition_order: tuple[Any, ...] | list[Any] | None,
    *,
    numeric: bool = True,
) -> dict[str, Any]:
    """Extract and validate a balanced complete-unit repeated panel from long-form data.

    Row order is never treated as unit or condition identity.
    """
    if not isinstance(frame, pd.DataFrame):
        raise InvalidTestError("frame must be a pandas DataFrame.")
    for name in (unit_id, condition_col, value_col):
        if not isinstance(name, str) or not name.strip():
            raise InvalidTestError("Column names must be non-empty strings.")
        if name not in frame.columns:
            raise ColumnNotFoundError(f"'{name}' is not a column in this DataFrame.")
    if len({unit_id, condition_col, value_col}) != 3:
        raise InvalidTestError("unit_id, condition_col, and value_col must be different columns.")

    if condition_order is None:
        raise InsufficientDataError(
            "Repeated-measures analysis requires an explicit condition_order with at least "
            "three conditions; order is not inferred alphabetically."
        )
    order = list(condition_order)
    if len(order) < 3:
        raise InsufficientDataError(
            "Repeated-measures analysis requires at least 3 declared conditions in condition_order."
        )
    if len(set(order)) != len(order):
        raise InsufficientDataError("condition_order contains duplicate condition labels.")

    observed_raw = pd.unique(frame[condition_col].dropna())
    observed_set = set(observed_raw.tolist())
    order_set = set(order)
    if not order_set.issubset(observed_set):
        missing_from_data = [item for item in order if item not in observed_set]
        raise InsufficientDataError(
            invalid_condition_order_message(missing_from_data, condition_col, observed_raw)
            + " The declared labels were not observed in data."
        )
    extra_in_data = [item for item in observed_raw if item not in order_set]
    if extra_in_data:
        raise InsufficientDataError(
            f"Data contains undeclared condition levels {observed_levels(list(extra_in_data))}. "
            f"Observed levels for {condition_col!r}: {observed_levels(observed_raw)}. Filter "
            "the dataset or declare every level in `condition_order`; its order defines the "
            "signed contrasts."
        )

    missing_unit_rows = int(frame[unit_id].isna().sum())

    usable_mask = (
        frame[unit_id].notna() & frame[condition_col].isin(order) & frame[value_col].notna()
    )
    usable = frame.loc[usable_mask, [unit_id, condition_col, value_col]]

    if usable.duplicated(subset=[unit_id, condition_col]).any():
        raise InsufficientDataError(
            "Repeated-measures analysis requires one usable outcome per unit "
            "per declared condition."
        )

    if numeric and not pd.api.types.is_numeric_dtype(usable[value_col]):
        try:
            usable[value_col] = pd.to_numeric(usable[value_col])
        except (ValueError, TypeError) as exc:
            raise InvalidTestError(
                f"Outcome column {value_col!r} must contain real numeric values."
            ) from exc

    pivot = usable.pivot(index=unit_id, columns=condition_col, values=value_col)
    complete = pivot.dropna(subset=order)
    n_complete = len(complete)

    if n_complete < 3:
        raise InsufficientDataError(
            f"Repeated-measures analysis requires at least three complete units across all "
            f"declared conditions; only {n_complete} complete units observed."
        )

    panel_values = complete[order].to_numpy(dtype=float if numeric else object, copy=True)
    if numeric and not np.isfinite(panel_values).all():
        raise InvalidTestError("Repeated panel contains nonfinite outcome values.")

    if numeric:
        ptp = float(np.ptp(panel_values))
        if not math.isfinite(ptp) or ptp == 0.0:
            raise InsufficientDataError(
                "Repeated-measures analysis requires variability in the outcome variable."
            )

    total_units = int(frame[unit_id].dropna().nunique())
    incomplete_units = total_units - n_complete
    k = len(order)

    condition_summaries = []
    for j, cond_label in enumerate(order):
        col_vals = panel_values[:, j]
        if numeric:
            mean_val = float(np.mean(col_vals))
            sd_val = float(np.std(col_vals, ddof=1)) if n_complete > 1 else 0.0
            med_val = float(np.median(col_vals))
            q25 = float(np.percentile(col_vals, 25))
            q75 = float(np.percentile(col_vals, 75))
            iqr_val = float(q75 - q25)
            summary = {
                "condition": cond_label,
                "n": n_complete,
                "mean": mean_val,
                "standard_deviation": sd_val,
                "median": med_val,
                "q25": q25,
                "q75": q75,
                "iqr": iqr_val,
            }
        else:
            summary = {
                "condition": cond_label,
                "n": n_complete,
            }
        condition_summaries.append(summary)

    return {
        "panel": panel_values,
        "condition_order": order,
        "condition_count": k,
        "complete_units": n_complete,
        "total_units": total_units,
        "incomplete_units": incomplete_units,
        "unit_ids": [str(item) for item in complete.index],
        "analyzed_rows": n_complete * k,
        "excluded_rows": int(len(frame) - n_complete * k),
        "missing_unit_rows": missing_unit_rows,
        "condition_summaries": condition_summaries,
        "unit_id": unit_id,
        "condition_col": condition_col,
        "value_col": value_col,
    }


def friedman_test(
    frame: pd.DataFrame,
    unit_id: str,
    condition_col: str,
    value_col: str,
    condition_order: tuple[Any, ...] | list[Any] | None,
    *,
    alpha: float = 0.05,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Execute the Friedman repeated-rank omnibus test with complete Wilcoxon-Holm follow-up."""
    if not 0 < alpha < 1:
        raise InvalidTestError("alpha must be strictly between 0 and 1.")
    if not 0 < confidence_level < 1:
        raise InvalidTestError("confidence_level must be strictly between 0 and 1.")

    panel_info = repeated_panel(
        frame,
        unit_id,
        condition_col,
        value_col,
        condition_order,
        numeric=True,
    )
    panel = panel_info["panel"]
    n = panel_info["complete_units"]
    order = panel_info["condition_order"]
    k = panel_info["condition_count"]

    columns = [panel[:, j] for j in range(k)]
    friedman_res = stats.friedmanchisquare(*columns)
    stat = float(getattr(friedman_res, "statistic", friedman_res[0]))
    p_val = float(getattr(friedman_res, "pvalue", friedman_res[1]))

    if not math.isfinite(stat) or not math.isfinite(p_val) or not 0 <= p_val <= 1:
        raise InsufficientDataError("Friedman backend returned an invalid statistic or p-value.")

    df_omnibus = k - 1
    w_raw = stat / (n * (k - 1)) if n > 0 and k > 1 else 0.0
    kendall_w = float(min(1.0, max(0.0, w_raw)))

    # Pairwise post-hoc comparisons using paired Wilcoxon signed-rank tests
    pair_records: list[dict[str, Any]] = []
    pair_combos = list(combinations(range(k), 2))

    for i, j in pair_combos:
        first_cond = order[i]
        second_cond = order[j]
        diff = panel[:, i] - panel[:, j]
        nonzero = diff[diff != 0]

        pair_stat: float | None = None
        pair_p: float | None = None
        pair_effect: float | None = None
        pair_status = "available"
        pair_reason: str | None = None

        if len(nonzero) == 0:
            pair_status = "unavailable"
            pair_reason = (
                "All complete paired differences are zero; signed-rank inference is undefined."
            )
        elif len(nonzero) < 2:
            pair_status = "unavailable"
            pair_reason = (
                "Wilcoxon signed-rank inference needs at least two nonzero paired differences."
            )
        else:
            try:
                method_parameter = (
                    "method" if "method" in inspect.signature(stats.wilcoxon).parameters else "mode"
                )
                method_value = "approx" if len(nonzero) > 50 else "auto"
                options: dict[str, Any] = {
                    "zero_method": "wilcox",
                    "correction": False,
                    "alternative": "two-sided",
                    method_parameter: method_value,
                }
                w_res = stats.wilcoxon(diff, **options)
                raw_stat = _scipy_result_value(w_res, "statistic", index=0)
                raw_p = _scipy_result_value(w_res, "pvalue", index=1)
                if not math.isfinite(raw_stat) or not math.isfinite(raw_p) or not 0 <= raw_p <= 1:
                    pair_status = "unavailable"
                    pair_reason = "Wilcoxon returned an invalid statistic or p-value."
                else:
                    pair_stat = float(raw_stat)
                    pair_p = float(raw_p)
                    ranks = stats.rankdata(np.abs(nonzero), method="average")
                    positive = float(ranks[nonzero > 0].sum())
                    negative = float(ranks[nonzero < 0].sum())
                    denom = positive + negative
                    pair_effect = float((positive - negative) / denom) if denom > 0 else 0.0
            except Exception as exc:
                pair_status = "unavailable"
                pair_reason = f"Wilcoxon computation failed: {exc}"

        pair_ci = (
            matched_pairs_rank_biserial_bootstrap_ci(
                diff,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                random_state=random_state,
            )
            if pair_status == "available"
            else None
        )
        pair_records.append(
            {
                "contrast_id": f"{first_cond}_vs_{second_cond}",
                "first_condition": first_cond,
                "second_condition": second_cond,
                "contrast": {
                    "definition": "first condition minus second condition",
                    "first": first_cond,
                    "second": second_cond,
                },
                "orientation": f"{first_cond} minus {second_cond}",
                "estimate_name": "matched-pairs rank-biserial correlation",
                "estimate": pair_effect,
                "statistic_name": "Wilcoxon signed-rank statistic",
                "statistic": pair_stat,
                "degrees_of_freedom": None,
                "raw_p_value": pair_p,
                "adjusted_p_value": None,
                "adjustment_method": "holm",
                "multiplicity_adjustment": "holm",
                "alpha": alpha,
                "decision": "unavailable" if pair_p is None else "fail_to_reject",
                "confidence_interval": pair_ci,
                "effect_size": {
                    "name": "matched-pairs rank-biserial correlation",
                    "value": pair_effect,
                    "definition": (
                        "Positive minus negative signed-rank sums divided by their total; "
                        "positive values indicate higher ranks in the first condition."
                    ),
                    "confidence_interval": pair_ci,
                },
                "n_pairs": n,
                "family_size": len(pair_combos),
                "method_id": "wilcoxon_signed_rank",
                "status": pair_status,
                "reason": pair_reason,
            }
        )

    if all(rec.get("raw_p_value") is not None for rec in pair_records):
        raw_p_values = [float(rec["raw_p_value"]) for rec in pair_records]
        adjusted_p_values = adjust_pvalues(raw_p_values, method="holm")
        for rec, adj_p in zip(pair_records, adjusted_p_values, strict=True):
            rec["adjusted_p_value"] = float(adj_p)
            rec["decision"] = "reject" if adj_p < alpha else "fail_to_reject"
        multiplicity_status = "available"
        multiplicity_reason = None
    else:
        for rec in pair_records:
            rec["adjusted_p_value"] = None
            rec["decision"] = "unavailable"
        multiplicity_status = "unavailable"
        multiplicity_reason = (
            "Multiplicity adjustment is unavailable because one or more planned pairwise "
            "contrasts could not be evaluated; the planned family size is preserved."
        )

    w_ci = friedman_kendall_w_bootstrap_ci(
        panel,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )

    return {
        "method": "Friedman test for repeated ranks",
        "statistic": stat,
        "degrees_of_freedom": df_omnibus,
        "p_value": p_val,
        "complete_units": n,
        "confidence_interval": w_ci,
        "effect_size": {
            "name": "Kendall's W",
            "value": kendall_w,
            "definition": (
                "Kendall's coefficient of concordance W = Q / (n * (k - 1)); "
                "describes overall within-unit rank agreement across conditions on a 0 to 1 scale."
            ),
            "confidence_interval": w_ci,
        },
        "conditions": order,
        "condition_order": order,
        "condition_count": k,
        "condition_summaries": panel_info["condition_summaries"],
        "pairwise_comparisons": pair_records,
        "multiplicity": {
            "method": "holm",
            "number_of_comparisons": len(pair_records),
            "alpha": alpha,
            "decision_basis": "Holm-adjusted p-value < alpha",
            "status": multiplicity_status,
            "reason": multiplicity_reason,
        },
        "sample": {
            "original_rows": panel_info["analyzed_rows"] + panel_info["excluded_rows"],
            "analyzed_rows": panel_info["analyzed_rows"],
            "excluded_rows": panel_info["excluded_rows"],
            "total_units": panel_info["total_units"],
            "complete_units": n,
            "incomplete_units": panel_info["incomplete_units"],
            "missing_unit_rows": panel_info["missing_unit_rows"],
            "complete_unit_policy": "complete observations across all declared repeated conditions",
        },
        "target": "repeated rank distribution",
        "assumptions": (
            "Same units observed under three or more ordered conditions",
            "Independent observational units across subjects",
            "Ordinal or continuous outcome supporting ranking within unit",
        ),
        "limitations": (
            "Friedman evaluates within-unit ranks, not mean differences or independent medians.",
            "Complete-unit exclusion omits units with missing repeated observations; "
            "missingness is not modeled.",
            "Observed omnibus significance does not imply that every pairwise contrast differs.",
            "Observed differences do not establish causal effects.",
        ),
        "warnings": [],
    }


def repeated_measures_anova(
    frame: pd.DataFrame,
    unit_id: str,
    condition_col: str,
    value_col: str,
    condition_order: tuple[Any, ...] | list[Any] | None,
    *,
    alpha: float = 0.05,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Execute repeated-measures ANOVA with Mauchly's sphericity and GG correction."""
    if not 0 < alpha < 1:
        raise InvalidTestError("alpha must be strictly between 0 and 1.")
    if not 0 < confidence_level < 1:
        raise InvalidTestError("confidence_level must be strictly between 0 and 1.")

    panel_info = repeated_panel(
        frame,
        unit_id,
        condition_col,
        value_col,
        condition_order,
        numeric=True,
    )
    panel = panel_info["panel"]
    n = panel_info["complete_units"]
    order = panel_info["condition_order"]
    k = panel_info["condition_count"]

    if n <= 2:
        raise InsufficientDataError("Repeated-measures ANOVA requires at least 3 complete units.")

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

    ms_condition = ss_condition / df_condition if df_condition > 0 else 0.0
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    if ms_error <= 0.0 or not math.isfinite(ms_error):
        raise InsufficientDataError(
            "Repeated-measures ANOVA error variance is zero or nonfinite; cannot compute F ratio."
        )

    f_stat = float(ms_condition / ms_error)
    p_uncorrected = float(stats.f.sf(f_stat, df_condition, df_error))

    denom_eta = ss_condition + ss_error
    partial_eta_sq = float(ss_condition / denom_eta) if denom_eta > 0 else 0.0
    partial_eta_sq = float(min(1.0, max(0.0, partial_eta_sq)))

    # Sphericity evaluation via orthonormal Helmert contrasts
    cov_matrix = np.cov(panel, rowvar=False, ddof=1)
    m_helmert = _helmert_contrasts(k)
    sigma_c = m_helmert.T @ cov_matrix @ m_helmert  # shape (k-1, k-1)

    tr_c = float(np.trace(sigma_c))
    tr_c2 = float(np.trace(sigma_c @ sigma_c))
    det_c = float(np.linalg.det(sigma_c))

    # Greenhouse-Geisser epsilon
    lower_bound_eps = 1.0 / (k - 1)
    if tr_c2 > 1e-15 and math.isfinite(tr_c2) and tr_c > 0:
        raw_eps = (tr_c**2) / ((k - 1) * tr_c2)
        epsilon_gg = float(min(1.0, max(lower_bound_eps, raw_eps)))
    else:
        epsilon_gg = float(lower_bound_eps)

    # Mauchly's sphericity test
    # Evaluates H0: orthonormal contrast covariance matrix Sigma_c is proportional to identity.
    # Uses Box (1954) / Anderson (1958, 2003) asymptotic chi-square approximation with the
    # second-order correction term w2, matching SPSS and R mauchly.test:
    #   d = k - 1 (dimension of orthonormal contrast space)
    #   df_mauchly = d*(d+1)/2 - 1 = k*(k-1)/2 - 1
    #   df_resid = n - 1
    #   f = 1 - (2*d^2 + d + 2) / (6*d*df_resid)
    #   chi2 = -df_resid * f * ln(W)
    #   w2 = (d+2)*(d-1)*(d-2)*(2*d^3 + 6*d^2 + 3*k + 2) / (288 * (df_resid * d * f)^2)
    #   p = p1 + w2 * (p2 - p1), where p1 ~ chi2(df), p2 ~ chi2(df + 4)
    # For k=3 (d=2), (d-2)=0 so w2=0 and p reproduces the standard first-order chi2.sf.
    d = k - 1
    df_mauchly = (d * (d + 1)) // 2 - 1
    if det_c > 1e-15 and tr_c > 1e-15 and math.isfinite(det_c):
        mean_diag = tr_c / d
        w_mauchly = float(det_c / (mean_diag**d))
        w_mauchly = float(min(1.0, max(0.0, w_mauchly)))
        df_resid = n - 1
        d_factor = 1.0 - (2.0 * d**2 + d + 2.0) / (6.0 * d * df_resid)
        chi2_mauchly = float(-df_resid * d_factor * np.log(max(w_mauchly, 1e-15)))
        p1 = float(stats.chi2.sf(chi2_mauchly, df_mauchly))
        denom_w2 = 288.0 * ((df_resid * d * d_factor) ** 2)
        if denom_w2 > 0 and d > 2:
            w2 = (d + 2) * (d - 1) * (d - 2) * (2 * d**3 + 6 * d**2 + 3 * k + 2) / denom_w2
            p2 = float(stats.chi2.sf(chi2_mauchly, df_mauchly + 4))
            p_mauchly = float(min(1.0, max(0.0, p1 + w2 * (p2 - p1))))
        else:
            p_mauchly = p1
        sphericity_status = "not_rejected" if p_mauchly >= alpha else "rejected"
    else:
        w_mauchly = None
        chi2_mauchly = None
        p_mauchly = None
        sphericity_status = "uncomputable"

    # Greenhouse-Geisser correction calculation
    df_condition_corrected = float(epsilon_gg * df_condition)
    df_error_corrected = float(epsilon_gg * df_error)
    p_corrected = float(stats.f.sf(f_stat, df_condition_corrected, df_error_corrected))

    # Primary inference policy
    if sphericity_status == "rejected":
        primary_p = p_corrected
        primary_df_num = df_condition_corrected
        primary_df_den = df_error_corrected
        correction_applied = "Greenhouse-Geisser"
        primary_inference_rule = (
            "Mauchly's test rejected sphericity; Greenhouse-Geisser corrected degrees of "
            "freedom and p-value are reported as primary."
        )
    elif sphericity_status == "not_rejected":
        primary_p = p_uncorrected
        primary_df_num = float(df_condition)
        primary_df_den = float(df_error)
        correction_applied = "none"
        primary_inference_rule = (
            "Mauchly's test did not provide evidence against sphericity; uncorrected degrees "
            "of freedom are retained as primary."
        )
    else:
        # uncomputable sphericity (e.g. singular contrast covariance)
        primary_p = p_corrected
        primary_df_num = df_condition_corrected
        primary_df_den = df_error_corrected
        correction_applied = "Greenhouse-Geisser"
        primary_inference_rule = (
            "Sphericity could not be computed reliably; Greenhouse-Geisser correction is applied "
            "as a conservative safeguard."
        )

    # Pairwise paired t-tests on the omnibus-complete panel
    pair_records: list[dict[str, Any]] = []
    pair_combos = list(combinations(range(k), 2))
    t_crit = float(stats.t.ppf((1.0 + confidence_level) / 2.0, df=n - 1))

    for i, j in pair_combos:
        first_cond = order[i]
        second_cond = order[j]
        diff = panel[:, i] - panel[:, j]
        diff_mean = float(np.mean(diff))
        diff_sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        se = float(diff_sd / math.sqrt(n)) if n > 0 else 0.0

        if diff_sd > 0 and se > 0 and math.isfinite(diff_sd) and math.isfinite(se):
            t_num = float(diff_mean / se)
            t_val: float | None = t_num
            pair_p: float | None = float(2.0 * stats.t.sf(abs(t_num), df=n - 1))
            ci_lower = float(diff_mean - t_crit * se)
            ci_upper = float(diff_mean + t_crit * se)
            ci_method = "analytical paired t confidence interval"
            cohen_dz: float | None = float(diff_mean / diff_sd)
            pair_status = "available"
            pair_reason: str | None = None
        else:
            t_val = None
            pair_p = None
            ci_lower = diff_mean
            ci_upper = diff_mean
            ci_method = "analytical paired t confidence interval (degenerate zero-variance sample)"
            cohen_dz = None
            pair_status = "unavailable"
            pair_reason = (
                "Paired differences have zero variance: Cohen's dz and a finite paired t "
                "statistic are unavailable."
            )

        pair_dz_ci = (
            paired_cohen_dz_ci(diff_mean, diff_sd, n, confidence_level=confidence_level)
            if (pair_status == "available" and cohen_dz is not None and diff_sd > 0)
            else None
        )
        pair_records.append(
            {
                "contrast_id": f"{first_cond}_vs_{second_cond}",
                "first_condition": first_cond,
                "second_condition": second_cond,
                "contrast": {
                    "definition": "first condition minus second condition",
                    "first": first_cond,
                    "second": second_cond,
                },
                "orientation": f"{first_cond} minus {second_cond}",
                "estimate_name": "mean difference",
                "estimate": diff_mean,
                "mean_difference": diff_mean,
                "statistic_name": "paired t statistic",
                "statistic": t_val,
                "degrees_of_freedom": n - 1 if t_val is not None else None,
                "standard_error": se if se > 0 else 0.0,
                "raw_p_value": pair_p,
                "adjusted_p_value": None,
                "adjustment_method": "holm",
                "multiplicity_adjustment": "holm",
                "alpha": alpha,
                "decision": "unavailable" if pair_p is None else "fail_to_reject",
                "confidence_interval": {
                    "lower": ci_lower,
                    "upper": ci_upper,
                    "level": confidence_level,
                    "method": ci_method,
                    "quantity": "mean difference",
                    "multiplicity_adjusted": False,
                },
                "effect_size": {
                    "name": "Cohen's dz",
                    "value": cohen_dz,
                    "definition": (
                        "Mean paired difference divided by the standard deviation of paired "
                        "differences."
                    ),
                    "confidence_interval": pair_dz_ci,
                },
                "n_pairs": n,
                "family_size": len(pair_combos),
                "method_id": "paired_t",
                "status": pair_status,
                "reason": pair_reason,
            }
        )

    if all(rec.get("raw_p_value") is not None for rec in pair_records):
        raw_p_values = [float(rec["raw_p_value"]) for rec in pair_records]
        adjusted_p_values = adjust_pvalues(raw_p_values, method="holm")
        for rec, adj_p in zip(pair_records, adjusted_p_values, strict=True):
            rec["adjusted_p_value"] = float(adj_p)
            rec["decision"] = "reject" if adj_p < alpha else "fail_to_reject"
        multiplicity_status = "available"
        multiplicity_reason = None
    else:
        for rec in pair_records:
            rec["adjusted_p_value"] = None
            rec["decision"] = "unavailable"
        multiplicity_status = "unavailable"
        multiplicity_reason = (
            "Multiplicity adjustment is unavailable because one or more planned pairwise "
            "contrasts could not be evaluated; the planned family size is preserved."
        )

    return {
        "method": "One-way repeated-measures ANOVA",
        "statistic": f_stat,
        "anova_table": {
            "ss_condition": ss_condition,
            "ss_error": ss_error,
            "ss_subject": ss_subject,
            "ss_total": ss_total,
            "df_condition": df_condition,
            "df_error": df_error,
            "df_subject": df_subject,
            "ms_condition": ms_condition,
            "ms_error": ms_error,
            "f_statistic": f_stat,
            "p_value": p_uncorrected,
        },
        "degrees_of_freedom": {
            "condition": df_condition,
            "error": df_error,
            "subject": df_subject,
            "corrected_condition": df_condition_corrected,
            "corrected_error": df_error_corrected,
        },
        "uncorrected_p_value": p_uncorrected,
        "corrected_p_value": p_corrected,
        "primary_p_value": primary_p,
        "p_value": primary_p,
        "complete_units": n,
        "conditions": order,
        "primary_degrees_of_freedom": {
            "numerator": primary_df_num,
            "denominator": primary_df_den,
        },
        "correction_applied": correction_applied,
        "primary_inference_rule": primary_inference_rule,
        "sums_of_squares": {
            "condition": ss_condition,
            "error": ss_error,
            "subject": ss_subject,
            "total": ss_total,
        },
        "mean_squares": {
            "condition": ms_condition,
            "error": ms_error,
        },
        "effect_size": {
            "name": "partial eta-squared",
            "value": partial_eta_sq,
            "definition": (
                "SS_condition / (SS_condition + SS_error); the proportion of within-subject "
                "variance attributable to condition differences."
            ),
            "confidence_interval": repeated_measures_partial_eta_squared_ci(
                f_stat, float(df_condition), float(df_error), confidence_level=confidence_level
            ),
        },
        "confidence_interval": repeated_measures_partial_eta_squared_ci(
            f_stat, float(df_condition), float(df_error), confidence_level=confidence_level
        ),
        "sphericity": {
            "test_name": "Mauchly's test of sphericity",
            "statistic": w_mauchly,
            "mauchly_w": w_mauchly,
            "chi2_statistic": chi2_mauchly,
            "degrees_of_freedom": df_mauchly,
            "df": df_mauchly,
            "p_value": p_mauchly,
            "alpha": alpha,
            "status": sphericity_status,
            "sphericity_not_rejected": (sphericity_status == "not_rejected"),
            # Compatibility alias for unreleased intermediate callers;
            # does not imply proof of sphericity.
            "sphericity_supported": (sphericity_status == "not_rejected"),
            "decision": (
                "Sphericity assumption violated; correction recommended."
                if sphericity_status == "rejected"
                else "Mauchly's test did not provide evidence against sphericity."
                if sphericity_status == "not_rejected"
                else "Sphericity could not be computed reliably."
            ),
        },
        "greenhouse_geisser": {
            "applied": (correction_applied == "Greenhouse-Geisser"),
            "epsilon": epsilon_gg,
            "lower_bound": lower_bound_eps,
            "upper_bound": 1.0,
            "corrected_numerator_df": df_condition_corrected,
            "corrected_denominator_df": df_error_corrected,
            "corrected_df_num": df_condition_corrected,
            "corrected_df_den": df_error_corrected,
            "corrected_p_value": p_corrected,
        },
        "primary_inference": "greenhouse_geisser"
        if correction_applied == "Greenhouse-Geisser"
        else "uncorrected",
        "condition_order": order,
        "condition_count": k,
        "condition_summaries": panel_info["condition_summaries"],
        "pairwise_comparisons": pair_records,
        "multiplicity": {
            "method": "holm",
            "number_of_comparisons": len(pair_records),
            "alpha": alpha,
            "decision_basis": "Holm-adjusted p-value < alpha",
            "status": multiplicity_status,
            "reason": multiplicity_reason,
        },
        "sample": {
            "original_rows": panel_info["analyzed_rows"] + panel_info["excluded_rows"],
            "analyzed_rows": panel_info["analyzed_rows"],
            "excluded_rows": panel_info["excluded_rows"],
            "total_units": panel_info["total_units"],
            "complete_units": n,
            "incomplete_units": panel_info["incomplete_units"],
            "missing_unit_rows": panel_info["missing_unit_rows"],
            "complete_unit_policy": "complete observations across all declared repeated conditions",
        },
        "target": "repeated condition means",
        "assumptions": (
            "Same units observed under three or more ordered conditions",
            "Independent observational units across subjects",
            "Continuous outcome variable",
            "Multivariate normality of within-subject contrast errors",
            "Sphericity (equal pairwise difference variances across condition pairs)",
        ),
        "limitations": (
            "Omnibus F test indicates whether condition means differ overall; "
            "pairwise contrasts identify specific pairwise differences.",
            "Complete-unit exclusion omits units with missing repeated observations; "
            "missingness is not modeled.",
            "Pairwise confidence intervals are per-contrast intervals, not simultaneous bands.",
            "Repeated-measures ANOVA does not establish causality.",
        ),
        "warnings": [],
    }
