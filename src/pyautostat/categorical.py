"""Categorical descriptions and independent-sample association tests."""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .exceptions import (
    ColumnNotFoundError,
    InsufficientDataError,
    InvalidDataError,
    InvalidTestError,
)


def _json_scalar(value):
    """Return a stable JSON scalar for an observed category label."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _validate_columns(df, first, second=None):
    names = (first,) if second is None else (first, second)
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise InvalidDataError("Categorical column names must be non-empty strings.")
        if name not in df.columns:
            raise ColumnNotFoundError(f"'{name}' is not a column in this DataFrame.")
    if second is not None and first == second:
        raise InvalidDataError("Cross-tabulation requires two different columns.")


def _observed_levels(series, explicit_order=None, *, respect_dtype_order=True):
    observed = list(pd.unique(series.dropna()))
    preferred = explicit_order
    source = "declared_order"
    if (
        preferred is None
        and respect_dtype_order
        and isinstance(series.dtype, pd.CategoricalDtype)
        and series.dtype.ordered
    ):
        preferred = list(series.dtype.categories)
        source = "ordered_categorical_dtype"
    if preferred is None:
        return observed, "first_observed"
    ordered = [level for level in preferred if any(level == value for value in observed)]
    extras = [value for value in observed if not any(value == level for level in ordered)]
    ordered.extend(extras)
    if source == "declared_order" and extras:
        source = "declared_order_incomplete"
    return ordered, source


def contingency_counts(
    df,
    row_variable,
    column_variable,
    *,
    row_order=None,
    column_order=None,
    respect_dtype_order=True,
):
    """Build one complete-case contingency table for description and inference."""
    _validate_columns(df, row_variable, column_variable)
    usable = df[[row_variable, column_variable]].dropna()
    if usable.empty:
        raise InsufficientDataError(
            "Cross-tabulation has no complete rows after excluding missing values."
        )
    row_levels, row_ordering = _observed_levels(
        usable[row_variable], row_order, respect_dtype_order=respect_dtype_order
    )
    column_levels, column_ordering = _observed_levels(
        usable[column_variable], column_order, respect_dtype_order=respect_dtype_order
    )
    row_codes = pd.Categorical(usable[row_variable], categories=row_levels).codes
    column_codes = pd.Categorical(usable[column_variable], categories=column_levels).codes
    observed = np.zeros((len(row_levels), len(column_levels)), dtype=int)
    np.add.at(observed, (row_codes, column_codes), 1)
    return {
        "usable": usable,
        "row_levels_raw": row_levels,
        "column_levels_raw": column_levels,
        "row_levels": [_json_scalar(value) for value in row_levels],
        "column_levels": [_json_scalar(value) for value in column_levels],
        "counts": observed,
        "row_ordering": row_ordering,
        "column_ordering": column_ordering,
    }


def frequency_table(df, column, *, analytical_type, explicit_order=None):
    """Return a JSON-safe categorical frequency table using valid-value percentages."""
    _validate_columns(df, column)
    series = df[column]
    valid = series.dropna()
    if valid.empty:
        raise InsufficientDataError(
            f"Frequency table for '{column}' has no non-missing observations."
        )
    levels, ordering = _observed_levels(series, explicit_order)
    counts = [int((valid == level).sum()) for level in levels]
    if ordering == "first_observed":
        ranked = sorted(zip(levels, counts, strict=True), key=lambda item: -item[1])
        levels = [item[0] for item in ranked]
        counts = [item[1] for item in ranked]
        ordering = "descending_frequency_first_observed_ties"
    valid_n = int(len(valid))
    total_n = int(len(series))
    include_cumulative = analytical_type == "ordinal_categorical" and ordering in {
        "declared_order",
        "ordered_categorical_dtype",
    }
    cumulative = 0.0
    rows = []
    for level, count in zip(levels, counts, strict=True):
        percent = float(count / valid_n * 100)
        row = {
            "level": _json_scalar(level),
            "count": count,
            "percent": percent,
            "total_percent": float(count / total_n * 100),
        }
        if include_cumulative:
            cumulative += percent
            row["cumulative_percent"] = cumulative
        rows.append(row)
    return {
        "column": column,
        "analytical_type": analytical_type,
        "levels": rows,
        "valid_n": valid_n,
        "missing_n": total_n - valid_n,
        "total_n": total_n,
        "excluded_rows": total_n - valid_n,
        "percentage_denominator": "non-missing observations",
        "missing_values_as_category": False,
        "ordering": ordering,
        "cumulative_percentage_status": (
            "available"
            if include_cumulative
            else "omitted_unordered"
            if analytical_type == "ordinal_categorical"
            else "not_applicable"
        ),
        "high_cardinality": len(rows) > 20,
        "level_count": len(rows),
    }


def cross_tabulation(
    df,
    row_variable,
    column_variable,
    *,
    row_order=None,
    column_order=None,
):
    """Return counts and explicitly named row, column, and total percentages."""
    table = contingency_counts(
        df,
        row_variable,
        column_variable,
        row_order=row_order,
        column_order=column_order,
    )
    counts = table["counts"]
    valid_n = int(counts.sum())
    row_totals = counts.sum(axis=1, keepdims=True)
    column_totals = counts.sum(axis=0, keepdims=True)
    row_percent = np.divide(
        counts * 100.0,
        row_totals,
        out=np.zeros(counts.shape, dtype=float),
        where=row_totals != 0,
    )
    column_percent = np.divide(
        counts * 100.0,
        column_totals,
        out=np.zeros(counts.shape, dtype=float),
        where=column_totals != 0,
    )
    return {
        "row_variable": row_variable,
        "column_variable": column_variable,
        "row_levels": table["row_levels"],
        "column_levels": table["column_levels"],
        "counts": counts.tolist(),
        "row_percent": row_percent.tolist(),
        "column_percent": column_percent.tolist(),
        "total_percent": (counts / valid_n * 100.0).tolist(),
        "original_rows": int(len(df)),
        "valid_n": valid_n,
        "excluded_rows": int(len(df) - valid_n),
        "missing_policy": "complete cases for the two selected columns",
        "missing_values_as_category": False,
        "row_ordering": table["row_ordering"],
        "column_ordering": table["column_ordering"],
        "shape": [int(counts.shape[0]), int(counts.shape[1])],
        "cell_count": int(counts.size),
        "large_table": counts.size > 100,
    }


def _interval(estimates, level, requested, random_state):
    if len(estimates) < max(50, requested // 2):
        return None
    alpha = (1 - level) / 2
    lower, upper = np.quantile(estimates, [alpha, 1 - alpha])
    return {
        "level": level,
        "lower": float(lower),
        "upper": float(upper),
        "method": "observation-row percentile bootstrap",
        "requested_resamples": requested,
        "random_seed": random_state,
        "valid_resamples": len(estimates),
    }


def _cohens_h(first, second):
    return float(2 * (np.arcsin(np.sqrt(first)) - np.arcsin(np.sqrt(second))))


def categorical_association(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    *,
    success_value=None,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
) -> dict:
    """Pearson chi-square with Cramer's V for two categorical columns.

    Pairwise missing rows are excluded. Group and outcome order follows first
    appearance. Cohen's h is available only for a 2x2 table and an explicit
    success_value; positive h means the first group has a higher success rate.
    Expected counts must all be at least five for the chi-square approximation.
    """
    if not isinstance(group_col, str) or not group_col.strip():
        raise InvalidTestError("group_col must be a non-empty column name string.")
    if not isinstance(outcome_col, str) or not outcome_col.strip():
        raise InvalidTestError("outcome_col must be a non-empty column name string.")
    if group_col == outcome_col:
        raise InvalidTestError("group_col and outcome_col must name different columns.")
    for name in (group_col, outcome_col):
        if name not in df.columns:
            raise ColumnNotFoundError(f"'{name}' is not a column in this DataFrame.")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float, np.integer, np.floating))
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise InvalidTestError("confidence_level must be a finite number between 0 and 1.")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, (int, np.integer))
        or (bootstrap_samples != 0 and bootstrap_samples < 100)
    ):
        raise InvalidTestError("bootstrap_samples must be 0 or an integer of at least 100.")
    if random_state is not None and (
        isinstance(random_state, bool)
        or not isinstance(random_state, (int, np.integer))
        or random_state < 0
    ):
        raise InvalidTestError("random_state must be a nonnegative integer or None.")

    # Preserve this established inferential API's first-appearance contrast order.
    table = contingency_counts(df, group_col, outcome_col, respect_dtype_order=False)
    usable = table["usable"]
    groups = pd.Index(table["row_levels_raw"])
    outcomes = pd.Index(table["column_levels_raw"])
    group_codes = pd.Categorical(usable[group_col], categories=groups).codes
    outcome_codes = pd.Categorical(usable[outcome_col], categories=outcomes).codes
    if len(groups) < 2 or len(outcomes) < 2:
        raise InsufficientDataError(
            "Chi-square association needs at least 2 observed categories in each column "
            "after excluding missing rows."
        )
    if success_value is not None and (len(groups) != 2 or len(outcomes) != 2):
        raise InvalidTestError("Cohen's h needs exactly 2 groups and 2 outcome categories.")
    if success_value is not None and (
        not pd.api.types.is_scalar(success_value) or pd.isna(success_value)
    ):
        raise InvalidTestError("success_value must be one non-missing scalar outcome category.")
    observed = table["counts"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        chi2, p_value, degrees_of_freedom, expected = chi2_contingency(observed, correction=False)
    if (
        caught
        or not np.isfinite(chi2)
        or not np.isfinite(p_value)
        or not np.isfinite(expected).all()
    ):
        raise InsufficientDataError("Chi-square returned an undefined result; review the table.")
    if np.any(expected < 5):
        raise InsufficientDataError(
            "This library requires expected counts of at least 5 in every cell as a "
            "conservative chi-square approximation policy. Use a justified method for sparse data."
        )
    n = int(observed.sum())
    cramer_v = float(np.sqrt(chi2 / (n * min(len(groups) - 1, len(outcomes) - 1))))

    success_index = None
    if success_value is not None:
        matches = [index for index, value in enumerate(outcomes) if value == success_value]
        if len(matches) != 1:
            raise InvalidTestError("success_value must match one observed outcome category.")
        success_index = matches[0]

    v_estimates = []
    h_estimates = []
    if bootstrap_samples:
        rng = np.random.default_rng(random_state)
        for _ in range(bootstrap_samples):
            indices = rng.integers(0, n, size=n)
            sampled = np.bincount(
                group_codes[indices] * len(outcomes) + outcome_codes[indices],
                minlength=observed.size,
            ).reshape(observed.shape)
            if np.any(sampled.sum(axis=0) == 0) or np.any(sampled.sum(axis=1) == 0):
                continue
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                sampled_chi2 = chi2_contingency(sampled, correction=False).statistic
            if caught or not np.isfinite(sampled_chi2):
                continue
            estimate = np.sqrt(sampled_chi2 / (n * min(len(groups) - 1, len(outcomes) - 1)))
            if np.isfinite(estimate):
                v_estimates.append(float(estimate))
            if success_index is not None:
                rates = sampled[:, success_index] / sampled.sum(axis=1)
                h_estimate = _cohens_h(*rates)
                if np.isfinite(h_estimate):
                    h_estimates.append(h_estimate)

    result = {
        "test": "Pearson chi-square independence",
        "statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(degrees_of_freedom),
        "groups": list(groups),
        "outcomes": list(outcomes),
        "observed_counts": observed.tolist(),
        "expected_counts": expected.tolist(),
        "sample_size": n,
        "excluded_rows": len(df) - n,
        "assumptions": {
            "independent_observations": "Required by study design; cannot be verified from values.",
            "minimum_expected_count": float(expected.min()),
            "expected_count_status": "met",
            "bootstrap": {
                "method": "observation-row percentile bootstrap",
                "requested_resamples": int(bootstrap_samples),
                "valid_resamples_cramers_v": len(v_estimates),
                "valid_resamples_cohens_h": len(h_estimates) if success_index is not None else None,
                "random_seed": random_state,
            },
            "warnings": [],
        },
        "effect_size": {
            "name": "Cramer's V",
            "value": cramer_v,
            "confidence_interval": (
                _interval(
                    v_estimates, float(confidence_level), int(bootstrap_samples), random_state
                )
                if bootstrap_samples
                else None
            ),
        },
    }
    if success_index is not None:
        rates = observed[:, success_index] / observed.sum(axis=1)
        result["cohens_h"] = {
            "success_value": success_value,
            "group_proportions": [float(value) for value in rates],
            "value": _cohens_h(*rates),
            "confidence_interval": (
                _interval(
                    h_estimates, float(confidence_level), int(bootstrap_samples), random_state
                )
                if bootstrap_samples
                else None
            ),
        }
    if bootstrap_samples and result["effect_size"]["confidence_interval"] is None:
        result["assumptions"]["warnings"].append(
            "Cramer's V interval unavailable: too few valid bootstrap resamples."
        )
    if (
        success_index is not None
        and bootstrap_samples
        and result["cohens_h"]["confidence_interval"] is None
    ):
        result["assumptions"]["warnings"].append(
            "Cohen's h interval unavailable: too few valid bootstrap resamples."
        )
    return result
