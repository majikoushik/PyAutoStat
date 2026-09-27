"""Validated numerical backends for the Phase 2 basic-inference methods."""

from __future__ import annotations

import inspect
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .categorical import contingency_counts
from .exceptions import ColumnNotFoundError, InsufficientDataError, InvalidTestError


def _scipy_result_value(result: Any, *names: str, index: int) -> float:
    """Read a SciPy result across named-result variants used by supported releases."""
    for name in names:
        if hasattr(result, name):
            return float(getattr(result, name))
    return float(result[index])


def _column(frame: pd.DataFrame, name: str, *, numeric: bool = False) -> pd.Series:
    if not isinstance(name, str) or not name.strip():
        raise InvalidTestError("Column names must be non-empty strings.")
    if name not in frame.columns:
        raise ColumnNotFoundError(f"'{name}' is not a column in this DataFrame.")
    series = frame[name]
    if numeric and not pd.api.types.is_numeric_dtype(series):
        raise InvalidTestError(f"'{name}' must be a real numeric column.")
    return series


def _confidence_level(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or not 0 < float(value) < 1
    ):
        raise InvalidTestError("confidence_level must be a finite number between 0 and 1.")
    return float(value)


def _bootstrap_options(samples: Any, random_state: Any) -> tuple[int, int | None]:
    if (
        isinstance(samples, bool)
        or not isinstance(samples, (int, np.integer))
        or (int(samples) != 0 and int(samples) < 100)
    ):
        raise InvalidTestError("bootstrap_samples must be 0 or an integer of at least 100.")
    if random_state is not None and (
        isinstance(random_state, bool)
        or not isinstance(random_state, (int, np.integer))
        or int(random_state) < 0
    ):
        raise InvalidTestError("random_state must be a nonnegative integer or None.")
    return int(samples), None if random_state is None else int(random_state)


def _finite_numeric(series: pd.Series, name: str) -> np.ndarray:
    try:
        values = np.asarray(series.dropna(), dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidTestError(f"'{name}' cannot be represented as finite real values.") from exc
    if not np.isfinite(values).all():
        raise InvalidTestError(f"'{name}' contains nonfinite values.")
    return values


def one_sample_t_test(
    frame: pd.DataFrame,
    value_col: str,
    reference_value: float,
    *,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Two-sided one-sample mean inference, oriented observed minus reference."""
    series = _column(frame, value_col, numeric=True)
    if (
        isinstance(reference_value, bool)
        or not isinstance(reference_value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(reference_value))
    ):
        raise InvalidTestError("reference_value must be a finite numeric value.")
    level = _confidence_level(confidence_level)
    values = _finite_numeric(series, value_col)
    if len(values) < 2:
        raise InsufficientDataError("A one-sample t-test needs at least two observed values.")
    scale = float(np.max(np.abs(values)))
    mean = float(np.mean(values / scale) * scale) if scale else 0.0
    sd = float(np.std(values / scale, ddof=1) * scale) if scale else 0.0
    difference = mean - float(reference_value)
    if not all(math.isfinite(item) for item in (mean, sd, difference)):
        raise InsufficientDataError(
            "The sample mean, spread, or reference contrast is not representable; rescale data."
        )
    n = int(len(values))
    standard_error = sd / math.sqrt(n)
    backend_warnings: list[str] = []
    statistic: float | None
    p_value: float | None
    interval: dict[str, Any] | None
    effect: float | None
    if sd == 0:
        statistic = None
        p_value = None
        interval = {
            "lower": difference,
            "upper": difference,
            "level": level,
            "method": "analytical one-sample t interval (degenerate zero-variance sample)",
        }
        effect = None
        backend_warnings.append(
            "The sample has zero variance: Cohen's d and a finite t statistic are unavailable."
        )
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            ttest_options = (
                {"alternative": "two-sided"}
                if "alternative" in inspect.signature(stats.ttest_1samp).parameters
                else {}
            )
            test = stats.ttest_1samp(values, float(reference_value), **ttest_options)
        statistic = _scipy_result_value(test, "statistic", index=0)
        p_value = _scipy_result_value(test, "pvalue", index=1)
        critical = float(stats.t.ppf((1 + level) / 2, n - 1))
        margin = critical * standard_error
        effect = difference / sd
        interval = {
            "lower": difference - margin,
            "upper": difference + margin,
            "level": level,
            "method": "analytical one-sample t interval",
        }
        if caught:
            backend_warnings.append(
                "SciPy reported numerical precision loss; review the one-sample result."
            )
        numeric = (
            statistic,
            p_value,
            critical,
            margin,
            effect,
            interval["lower"],
            interval["upper"],
        )
        if not all(math.isfinite(item) for item in numeric) or not 0 <= p_value <= 1:
            raise InsufficientDataError(
                "The one-sample t calculation produced a nonfinite result; rescale data."
            )
    return {
        "test": "One-sample t-test",
        "statistic": statistic,
        "p_value": p_value,
        "degrees_of_freedom": n - 1,
        "sample_mean": mean,
        "reference_value": float(reference_value),
        "mean_difference": difference,
        "standard_error": standard_error,
        "sample_standard_deviation": sd,
        "confidence_interval": interval,
        "effect_size": {"name": "one-sample Cohen's d", "value": effect},
        "sample_size": n,
        "excluded_rows": int(len(frame) - n),
        "alternative": "two-sided",
        "warnings": backend_warnings,
    }


def paired_values(
    frame: pd.DataFrame,
    unit_id: str,
    condition_col: str,
    value_col: str,
    condition_order: tuple[Any, Any] | list[Any] | None,
) -> dict[str, Any]:
    """Construct explicit complete pairs; row order is never used as pair identity."""
    for name in (unit_id, condition_col):
        _column(frame, name)
    _column(frame, value_col, numeric=True)
    if len({unit_id, condition_col, value_col}) != 3:
        raise InvalidTestError("unit_id, condition_col, and value_col must be different columns.")
    usable = frame[[unit_id, condition_col, value_col]].dropna()
    if usable.duplicated([unit_id, condition_col]).any():
        raise InsufficientDataError(
            "A unit has multiple usable observations in one condition; aggregate explicitly."
        )
    observed = list(pd.unique(frame[condition_col].dropna()))
    order = list(condition_order or tuple(observed))
    if len(observed) != 2 or len(order) != 2 or set(order) != set(observed):
        raise InsufficientDataError("Paired analysis requires exactly two ordered conditions.")
    pivot = usable.pivot(index=unit_id, columns=condition_col, values=value_col)
    complete = pivot.dropna(subset=order)
    first = _finite_numeric(complete[order[0]], value_col)
    second = _finite_numeric(complete[order[1]], value_col)
    if len(first) < 2:
        raise InsufficientDataError("At least two finite complete pairs are required.")
    complete_pairs = int(len(first))
    total_units = int(frame[unit_id].dropna().nunique())
    return {
        "first": first,
        "second": second,
        "differences": first - second,
        "condition_order": order,
        "complete_pairs": complete_pairs,
        "analyzed_rows": 2 * complete_pairs,
        "excluded_rows": int(len(frame) - 2 * complete_pairs),
        "total_units": total_units,
        "incomplete_units": total_units - complete_pairs,
        "missing_unit_rows": int(frame[unit_id].isna().sum()),
    }


def paired_wilcoxon(
    frame: pd.DataFrame,
    unit_id: str,
    condition_col: str,
    value_col: str,
    condition_order: tuple[Any, Any] | list[Any] | None,
) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank inference with the ``wilcox`` zero policy."""
    pairs = paired_values(frame, unit_id, condition_col, value_col, condition_order)
    differences = pairs["differences"]
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        raise InsufficientDataError(
            "All complete paired differences are zero; signed-rank inference is undefined."
        )
    if len(nonzero) < 2:
        raise InsufficientDataError(
            "Wilcoxon signed-rank inference needs at least two nonzero paired differences."
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        method_parameter = (
            "method" if "method" in inspect.signature(stats.wilcoxon).parameters else "mode"
        )
        options: dict[str, Any] = {
            "zero_method": "wilcox",
            "correction": False,
            "alternative": "two-sided",
            method_parameter: "auto",
        }
        result = stats.wilcoxon(differences, **options)
    statistic = _scipy_result_value(result, "statistic", index=0)
    p_value = _scipy_result_value(result, "pvalue", index=1)
    if not math.isfinite(statistic) or not math.isfinite(p_value) or not 0 <= p_value <= 1:
        raise InsufficientDataError("Wilcoxon returned an invalid statistic or p-value.")
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative
    effect = (positive - negative) / denominator
    return {
        "test": "Wilcoxon signed-rank test",
        "statistic": statistic,
        "p_value": p_value,
        "effect_size": {
            "name": "matched-pairs rank-biserial correlation",
            "value": float(effect),
            "confidence_interval": None,
        },
        "positive_rank_sum": positive,
        "negative_rank_sum": negative,
        "zero_differences": int(np.count_nonzero(differences == 0)),
        "nonzero_differences": int(len(nonzero)),
        "zero_method": "wilcox",
        "alternative": "two-sided",
        "method": "auto",
        "method_parameter": method_parameter,
        "method_detail": (
            "SciPy auto-selected its supported exact or approximate calculation; "
            "no exactness claim is made."
        ),
        "warnings": [str(item.message) for item in caught],
        **pairs,
    }


def spearman_correlation(
    frame: pd.DataFrame,
    first: str,
    second: str,
    *,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Spearman inference with a paired-observation percentile bootstrap interval."""
    _column(frame, first, numeric=True)
    _column(frame, second, numeric=True)
    if first == second:
        raise InvalidTestError("Spearman variables must be different columns.")
    level = _confidence_level(confidence_level)
    requested, seed = _bootstrap_options(bootstrap_samples, random_state)
    pair = frame[[first, second]].dropna()
    if len(pair) < 3:
        raise InsufficientDataError("Spearman inference needs at least three complete pairs.")
    x = _finite_numeric(pair[first], first)
    y = _finite_numeric(pair[second], second)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        raise InsufficientDataError("Spearman correlation needs variation in both variables.")
    spearman_options = (
        {"alternative": "two-sided"}
        if "alternative" in inspect.signature(stats.spearmanr).parameters
        else {}
    )
    result = stats.spearmanr(x, y, **spearman_options)
    rho = _scipy_result_value(result, "statistic", "correlation", index=0)
    p_value = _scipy_result_value(result, "pvalue", index=1)
    if not math.isfinite(rho) or not math.isfinite(p_value) or not -1 <= rho <= 1:
        raise InsufficientDataError("Spearman returned an invalid coefficient or p-value.")
    estimates: list[float] = []
    if requested:
        rng = np.random.default_rng(seed)
        for _ in range(requested):
            indices = rng.integers(0, len(pair), size=len(pair))
            sampled_x, sampled_y = x[indices], y[indices]
            if np.unique(sampled_x).size < 2 or np.unique(sampled_y).size < 2:
                continue
            bootstrap_result = stats.spearmanr(sampled_x, sampled_y)
            estimate = _scipy_result_value(bootstrap_result, "statistic", "correlation", index=0)
            if math.isfinite(estimate):
                estimates.append(estimate)
    interval = None
    if requested and len(estimates) >= max(50, requested // 2):
        alpha = (1 - level) / 2
        lower, upper = np.quantile(estimates, [alpha, 1 - alpha])
        interval = {
            "lower": float(lower),
            "upper": float(upper),
            "level": level,
            "method": "paired-observation percentile bootstrap",
            "requested_resamples": requested,
            "valid_resamples": len(estimates),
            "random_seed": seed,
        }
    warning_list = []
    if requested and interval is None:
        warning_list.append(
            "Spearman interval unavailable: fewer than half of requested bootstrap resamples "
            "were valid (minimum 50)."
        )
    return {
        "test": "Spearman rank correlation",
        "statistic": rho,
        "p_value": p_value,
        "effect_size": {"name": "Spearman rho", "value": rho, "confidence_interval": interval},
        "confidence_interval": interval,
        "sample_size": int(len(pair)),
        "excluded_rows": int(len(frame) - len(pair)),
        "alternative": "two-sided",
        "ties": {
            "first_has_ties": bool(pd.Series(x).duplicated().any()),
            "second_has_ties": bool(pd.Series(y).duplicated().any()),
        },
        "bootstrap": {
            "method": "paired-observation percentile bootstrap",
            "requested_resamples": requested,
            "valid_resamples": len(estimates),
            "random_seed": seed,
        },
        "warnings": warning_list,
    }


def fisher_exact_test(
    frame: pd.DataFrame,
    row_variable: str,
    column_variable: str,
) -> dict[str, Any]:
    """Two-sided Fisher exact inference for one observed 2x2 table."""
    table = contingency_counts(frame, row_variable, column_variable)
    observed = table["counts"]
    if observed.shape != (2, 2):
        raise InsufficientDataError("Fisher's exact test is supported only for 2x2 tables.")
    result = stats.fisher_exact(observed, alternative="two-sided")
    raw_odds_ratio = _scipy_result_value(result, "statistic", "oddsratio", index=0)
    p_value = _scipy_result_value(result, "pvalue", index=1)
    if not math.isfinite(p_value) or not 0 <= p_value <= 1:
        raise InsufficientDataError("Fisher's exact test returned an invalid p-value.")
    odds_ratio = raw_odds_ratio if math.isfinite(raw_odds_ratio) else None
    odds_status = (
        "finite"
        if odds_ratio is not None
        else "positive_infinity"
        if raw_odds_ratio > 0
        else "undefined"
    )
    n = int(observed.sum())
    warnings_list = []
    if odds_ratio is None:
        warnings_list.append(
            "The sample odds ratio is nonfinite because of zero cells; it is recorded as "
            f"{odds_status} rather than serialized as a nonfinite number."
        )
    return {
        "test": "Fisher's exact test",
        "statistic": odds_ratio,
        "p_value": p_value,
        "odds_ratio": odds_ratio,
        "odds_ratio_status": odds_status,
        "odds_ratio_definition": (
            "SciPy's unconditional sample odds ratio for the ordered 2x2 table: "
            "(row1,col1 * row2,col2) / (row1,col2 * row2,col1)."
        ),
        "confidence_interval": None,
        "row_levels": table["row_levels"],
        "column_levels": table["column_levels"],
        "row_ordering": table["row_ordering"],
        "column_ordering": table["column_ordering"],
        "observed_counts": observed.tolist(),
        "sample_size": n,
        "excluded_rows": int(len(frame) - n),
        "alternative": "two-sided",
        "warnings": warnings_list,
    }
