"""Validated numerical backends for Phase 6 association methods."""

from __future__ import annotations

import inspect
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import ColumnNotFoundError, InsufficientDataError, InvalidDataError
from .inference import paired_values


def _label(value: Any) -> str | int | float | bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _confidence_level(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or not 0 < float(value) < 1
    ):
        raise InvalidDataError("confidence_level must be a finite number between 0 and 1.")
    return float(value)


def _scipy_statistic(res: Any) -> float:
    return float(getattr(res, "statistic", getattr(res, "correlation", res[0])))


def _scipy_pvalue(res: Any) -> float:
    return float(getattr(res, "pvalue", res[1]))


def _bootstrap_options(samples: Any, random_state: Any) -> tuple[int, int]:
    if (
        isinstance(samples, bool)
        or not isinstance(samples, (int, np.integer))
        or int(samples) < 100
    ):
        raise InvalidDataError("bootstrap_samples must be an integer of at least 100.")
    if random_state is not None and (
        isinstance(random_state, bool)
        or not isinstance(random_state, (int, np.integer))
        or int(random_state) < 0
    ):
        raise InvalidDataError("random_state must be a nonnegative integer or None.")
    return int(samples), 0 if random_state is None else int(random_state)


def _column(frame: pd.DataFrame, name: str, *, numeric: bool = False) -> pd.Series:
    if not isinstance(name, str) or not name.strip():
        raise InvalidDataError("Column names must be non-empty strings.")
    if name not in frame.columns:
        raise ColumnNotFoundError(f"{name!r} is not a column in this DataFrame.")
    series = frame[name]
    if numeric and not pd.api.types.is_numeric_dtype(series):
        raise InvalidDataError(f"{name!r} must be a real numeric column.")
    return series


def _finite(values: pd.Series, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidDataError(f"{name!r} cannot be represented as finite values.") from exc
    if not np.isfinite(result).all():
        raise InvalidDataError(f"{name!r} contains nonfinite values.")
    return result


def _binary_levels(values: np.ndarray, event_level: Any, variable: str) -> tuple[Any, Any]:
    levels = list(pd.unique(pd.Series(values).dropna()))
    if len(levels) != 2:
        raise InsufficientDataError(
            f"{variable!r} must have exactly two usable levels; {len(levels)} were found."
        )
    matches = [value for value in levels if value == event_level]
    if not matches:
        raise InvalidDataError(
            f"event_level {event_level!r} is not observed in binary variable {variable!r}."
        )
    event = matches[0]
    non_event = next(value for value in levels if value != event)
    return _label(non_event), _label(event)


def _bootstrap_interval(
    estimates: list[float],
    level: float,
    requested: int,
    seed: int,
    quantity: str,
    method: str = "paired-observation percentile bootstrap",
) -> dict[str, Any] | None:
    if len(estimates) < max(50, requested // 2):
        return None
    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "level": level,
        "method": method,
        "quantity": quantity,
        "requested_resamples": requested,
        "valid_resamples": len(estimates),
        "random_seed": seed,
    }


def mcnemar_test(
    frame: pd.DataFrame,
    unit_id: str,
    condition_col: str,
    outcome_col: str,
    condition_order: tuple[Any, Any] | list[Any] | None,
    *,
    event_level: Any,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Exact two-sided McNemar inference from explicit long-format unit pairs."""
    level = _confidence_level(confidence_level)
    requested, seed = _bootstrap_options(bootstrap_samples, random_state)
    pairs = paired_values(
        frame, unit_id, condition_col, outcome_col, condition_order, numeric=False
    )
    first = pairs["first"]
    second = pairs["second"]
    non_event, event = _binary_levels(np.concatenate([first, second]), event_level, outcome_col)
    first_event = first == event
    second_event = second == event
    a = int(np.sum(first_event & second_event))
    b = int(np.sum(first_event & ~second_event))
    c = int(np.sum(~first_event & second_event))
    d = int(np.sum(~first_event & ~second_event))
    n = int(len(first))
    discordant = b + c
    p_value = (
        1.0
        if discordant == 0
        else float(stats.binomtest(b, discordant, 0.5, alternative="two-sided").pvalue)
    )
    first_rate = float(np.mean(first_event))
    second_rate = float(np.mean(second_event))
    difference = first_rate - second_rate
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(requested):
        indices = rng.integers(0, n, size=n)
        estimates.append(float(np.mean(first_event[indices]) - np.mean(second_event[indices])))
    interval = _bootstrap_interval(
        estimates,
        level,
        requested,
        seed,
        "paired proportion difference",
        "paired-unit percentile bootstrap",
    )
    if c == 0 and b > 0:
        matched_or, matched_status = None, "positive_infinity"
    elif b == 0 and c > 0:
        matched_or, matched_status = 0.0, "zero"
    elif b == 0 and c == 0:
        matched_or, matched_status = None, "undefined"
    else:
        matched_or, matched_status = float(b / c), "finite"
    order = pairs["condition_order"]
    order_labels = [_label(value) for value in order]
    table = {
        "rows": {"condition": order_labels[0], "event": event, "non_event": non_event},
        "columns": {"condition": order_labels[1], "event": event, "non_event": non_event},
        "first_event_second_event": a,
        "first_event_second_non_event": b,
        "first_non_event_second_event": c,
        "first_non_event_second_non_event": d,
        "discordant_b": b,
        "discordant_c": c,
        "total_discordant": discordant,
        "total_pairs": n,
    }
    return {
        "test": "Exact two-sided McNemar test",
        "statistic": float(min(b, c)),
        "statistic_type": "smaller discordant-cell count",
        "p_value": p_value,
        "method": "exact binomial McNemar",
        "alternative": "two-sided",
        "transition_table": table,
        "event_level": event,
        "non_event_level": non_event,
        "condition_order": order_labels,
        "first_event_proportion": first_rate,
        "second_event_proportion": second_rate,
        "paired_proportion_difference": difference,
        "confidence_interval": interval,
        "matched_odds_ratio": {"value": matched_or, "status": matched_status},
        "sample": {
            "original_rows": int(len(frame)),
            "analyzed_rows": int(pairs["analyzed_rows"]),
            "excluded_rows": int(pairs["excluded_rows"]),
            "total_units": int(pairs["total_units"]),
            "complete_pairs": n,
            "incomplete_units": int(pairs["incomplete_units"]),
            "missing_unit_rows": int(pairs["missing_unit_rows"]),
        },
        "bootstrap": {
            "method": "paired-unit percentile bootstrap",
            "requested_resamples": requested,
            "valid_resamples": len(estimates),
            "random_seed": seed,
            "confidence_level": level,
        },
        "warnings": [],
    }


def point_biserial_correlation(
    frame: pd.DataFrame,
    continuous_col: str,
    binary_col: str,
    *,
    positive_level: Any,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Point-biserial correlation with explicit coding and row-pair bootstrap CI."""
    level = _confidence_level(confidence_level)
    requested, seed = _bootstrap_options(bootstrap_samples, random_state)
    _column(frame, continuous_col, numeric=True)
    _column(frame, binary_col)
    if continuous_col == binary_col:
        raise InvalidDataError("Point-biserial variables must be different.")
    pair = frame[[continuous_col, binary_col]].dropna()
    n = int(len(pair))
    if n < 3:
        raise InsufficientDataError("Point-biserial correlation needs three complete pairs.")
    negative, positive = _binary_levels(pair[binary_col].to_numpy(), positive_level, binary_col)
    encoded = (pair[binary_col] == positive).astype(float).to_numpy()
    continuous = _finite(pair[continuous_col], continuous_col)
    if np.unique(continuous).size < 2:
        raise InsufficientDataError("The continuous variable must vary.")
    result = stats.pointbiserialr(encoded, continuous)
    coefficient = _scipy_statistic(result)
    p_value = _scipy_pvalue(result)
    if not math.isfinite(coefficient) or not math.isfinite(p_value):
        raise InsufficientDataError("Point-biserial inference returned a nonfinite value.")
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(requested):
        indices = rng.integers(0, n, size=n)
        bx, by = encoded[indices], continuous[indices]
        if np.unique(bx).size != 2 or np.unique(by).size < 2:
            continue
        candidate = _scipy_statistic(stats.pointbiserialr(bx, by))
        if math.isfinite(candidate):
            estimates.append(candidate)
    interval = _bootstrap_interval(estimates, level, requested, seed, "point-biserial r")
    group_0 = continuous[encoded == 0]
    group_1 = continuous[encoded == 1]
    warning_messages = (
        [] if interval is not None else ["Point-biserial bootstrap interval is unavailable."]
    )
    return {
        "test": "Point-biserial correlation",
        "statistic": coefficient,
        "p_value": p_value,
        "point_biserial_r": coefficient,
        "degrees_of_freedom": n - 2,
        "confidence_interval": interval,
        "effect_size": {
            "name": "point-biserial r",
            "value": coefficient,
            "confidence_interval": interval,
        },
        "binary_encoding": {
            "level_0": negative,
            "level_1": positive,
            "negative_level": negative,
            "positive_level": positive,
            "encoding_note": f"0 = {negative!r}; 1 = declared positive level {positive!r}.",
        },
        "group_sizes": {str(negative): int(len(group_0)), str(positive): int(len(group_1))},
        "group_means": {
            str(negative): float(np.mean(group_0)),
            str(positive): float(np.mean(group_1)),
        },
        "sample_size": n,
        "excluded_rows": int(len(frame) - n),
        "alternative": "two-sided",
        "bootstrap": {
            "method": "paired-observation percentile bootstrap",
            "requested_resamples": requested,
            "valid_resamples": len(estimates),
            "random_seed": seed,
            "confidence_level": level,
        },
        "warnings": warning_messages,
    }


def kendall_tau_b(
    frame: pd.DataFrame,
    first: str,
    second: str,
    *,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Kendall tau-b inference with paired-observation bootstrap uncertainty."""
    level = _confidence_level(confidence_level)
    requested, seed = _bootstrap_options(bootstrap_samples, random_state)
    _column(frame, first, numeric=True)
    _column(frame, second, numeric=True)
    if first == second:
        raise InvalidDataError("Kendall variables must be different.")
    pair = frame[[first, second]].dropna()
    n = int(len(pair))
    if n < 3:
        raise InsufficientDataError("Kendall tau-b needs at least three complete pairs.")
    x, y = _finite(pair[first], first), _finite(pair[second], second)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        raise InsufficientDataError("Kendall tau-b requires variation in both variables.")
    options: dict[str, Any] = {"variant": "b"}
    if "method" in inspect.signature(stats.kendalltau).parameters:
        options["method"] = "auto"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = stats.kendalltau(x, y, **options)
    coefficient = _scipy_statistic(result)
    p_value = _scipy_pvalue(result)
    if not math.isfinite(coefficient) or not math.isfinite(p_value):
        raise InsufficientDataError("Kendall tau-b inference returned a nonfinite value.")
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(requested):
        indices = rng.integers(0, n, size=n)
        bx, by = x[indices], y[indices]
        if np.unique(bx).size < 2 or np.unique(by).size < 2:
            continue
        candidate = _scipy_statistic(stats.kendalltau(bx, by, **options))
        if math.isfinite(candidate):
            estimates.append(candidate)
    interval = _bootstrap_interval(estimates, level, requested, seed, "Kendall tau-b")
    warning_messages = [
        str(item.message) for item in caught if issubclass(item.category, RuntimeWarning)
    ]
    if interval is None:
        warning_messages.append("Kendall tau-b bootstrap interval is unavailable.")
    ties = {
        "first_tied_observations": int(n - pd.Series(x).nunique()),
        "second_tied_observations": int(n - pd.Series(y).nunique()),
        "variant": "b",
    }
    return {
        "test": "Kendall's tau-b",
        "statistic": coefficient,
        "p_value": p_value,
        "tau_b": coefficient,
        "degrees_of_freedom": None,
        "confidence_interval": interval,
        "effect_size": {
            "name": "Kendall's tau-b",
            "value": coefficient,
            "definition": "Pairwise concordance minus discordance, adjusted for ties.",
            "confidence_interval": interval,
        },
        "sample_size": n,
        "excluded_rows": int(len(frame) - n),
        "alternative": "two-sided",
        "ties": ties,
        "bootstrap": {
            "method": "paired-observation percentile bootstrap",
            "requested_resamples": requested,
            "valid_resamples": len(estimates),
            "random_seed": seed,
            "confidence_level": level,
        },
        "warnings": warning_messages,
    }
