"""Validated numerical calculations for multi-item scale internal consistency."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import InsufficientDataError, InvalidDataError


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InsufficientDataError(f"Reliability analysis produced an invalid {label}.") from exc
    if not math.isfinite(number):
        raise InsufficientDataError(f"Reliability analysis produced an invalid {label}.")
    return number


def cronbach_alpha(matrix: np.ndarray) -> float:
    """Return sample-variance Cronbach alpha without clipping its sign."""
    if matrix.ndim != 2 or matrix.shape[1] < 2 or matrix.shape[0] < 2:
        raise InsufficientDataError(
            "Cronbach's alpha requires at least two items and two complete respondents."
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        item_variances = np.var(matrix, axis=0, ddof=1)
        total_variance = float(np.var(np.sum(matrix, axis=1), ddof=1))
        if not np.isfinite(item_variances).all() or not math.isfinite(total_variance):
            raise InsufficientDataError("Item or total-score variance is not finite.")
        if total_variance <= 0:
            raise InsufficientDataError(
                "The complete-case total score has zero variance, so Cronbach's alpha is undefined."
            )
        k = matrix.shape[1]
        alpha = k / (k - 1) * (1 - float(np.sum(item_variances)) / total_variance)
    return _finite(alpha, "Cronbach alpha")


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[float | None, str, str | None]:
    if float(np.ptp(left)) == 0:
        return None, "unavailable", "The focal item is constant in the analyzed sample."
    if float(np.ptp(right)) == 0:
        return None, "unavailable", "The comparison score is constant in the analyzed sample."
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(value):
        return None, "unavailable", "The correlation is numerically nonfinite."
    return value, "available", None


def _bootstrap_interval(
    matrix: np.ndarray,
    *,
    confidence_level: float,
    bootstrap_samples: int,
    random_state: int | None,
) -> dict[str, Any]:
    rng = np.random.default_rng(0 if random_state is None else random_state)
    values: list[float] = []
    n = matrix.shape[0]
    for _ in range(bootstrap_samples):
        indexes = rng.integers(0, n, size=n)
        try:
            values.append(cronbach_alpha(matrix[indexes, :]))
        except InsufficientDataError:
            continue
    minimum_valid = max(50, bootstrap_samples // 2)
    if len(values) < minimum_valid:
        return {
            "status": "unavailable",
            "lower": None,
            "upper": None,
            "level": confidence_level,
            "method": "respondent-row percentile bootstrap",
            "quantity": "Cronbach's alpha",
            "requested_resamples": bootstrap_samples,
            "valid_resamples": len(values),
            "minimum_valid_resamples": minimum_valid,
            "random_state": 0 if random_state is None else random_state,
            "reason": "Too few finite bootstrap alpha replicates were available.",
        }
    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(np.asarray(values), [tail, 1 - tail])
    return {
        "status": "available",
        "lower": _finite(lower, "bootstrap interval bound"),
        "upper": _finite(upper, "bootstrap interval bound"),
        "level": confidence_level,
        "method": "respondent-row percentile bootstrap",
        "quantity": "Cronbach's alpha",
        "requested_resamples": bootstrap_samples,
        "valid_resamples": len(values),
        "minimum_valid_resamples": minimum_valid,
        "random_state": 0 if random_state is None else random_state,
        "reason": None,
    }


def calculate_reliability(
    frame: pd.DataFrame,
    items: tuple[str, ...] | list[str],
    *,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
    reverse_scoring: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Calculate alpha and item diagnostics on one validated complete-case matrix."""
    if isinstance(items, (str, bytes)) or not isinstance(items, (tuple, list)):
        raise InvalidDataError("Reliability items must be an ordered sequence of column names.")
    items = tuple(items)
    if len(items) < 2:
        raise InvalidDataError("Reliability analysis requires at least two selected items.")
    if len(set(items)) != len(items):
        raise InvalidDataError("Reliability item names must not contain duplicates.")
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise InvalidDataError("Reliability item names must be nonempty strings.")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not math.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise InvalidDataError("confidence_level must be strictly between zero and one.")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int):
        raise InvalidDataError("bootstrap_samples must be a positive integer.")
    if bootstrap_samples < 1:
        raise InvalidDataError("bootstrap_samples must be a positive integer.")
    if random_state is not None and (
        isinstance(random_state, bool) or not isinstance(random_state, int)
    ):
        raise InvalidDataError("random_state must be an integer or None.")
    missing_columns = [item for item in items if item not in frame.columns]
    if missing_columns:
        raise InvalidDataError(f"Reliability items do not exist: {missing_columns!r}.")
    if any(not pd.api.types.is_numeric_dtype(frame[item]) for item in items):
        nonnumeric = [item for item in items if not pd.api.types.is_numeric_dtype(frame[item])]
        raise InvalidDataError(
            "Reliability items require explicit numeric scores; no label scoring is inferred. "
            f"Non-numeric items: {nonnumeric!r}."
        )
    if reverse_scoring is not None and not isinstance(reverse_scoring, dict):
        raise InvalidDataError("reverse_scoring must map item names to (lower, upper).")
    scoring = reverse_scoring or {}
    unknown_scoring = set(scoring) - set(items)
    if unknown_scoring:
        raise InvalidDataError(
            f"reverse_scoring names items outside the selected scale: {sorted(unknown_scoring)!r}."
        )
    working = frame.loc[:, list(items)].copy()
    reverse_metadata = []
    for item, bounds in scoring.items():
        if (
            not isinstance(bounds, (tuple, list))
            or len(bounds) != 2
            or any(
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or not math.isfinite(bound)
                for bound in bounds
            )
        ):
            raise InvalidDataError("Every reverse_scoring value must be two finite numeric bounds.")
        lower, upper = float(bounds[0]), float(bounds[1])
        if lower >= upper:
            raise InvalidDataError("Reverse-scoring lower bounds must be below upper bounds.")
        observed = working[item].dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise InvalidDataError(f"Item {item!r} contains nonfinite values.")
        if np.any(observed < lower) or np.any(observed > upper):
            raise InvalidDataError(
                f"Item {item!r} contains values outside reverse-scoring bounds [{lower}, {upper}]."
            )
        working[item] = lower + upper - working[item]
        reverse_metadata.append(
            {
                "item": item,
                "lower": lower,
                "upper": upper,
                "formula": "lower + upper - original",
            }
        )
    missingness = [
        {
            "item": item,
            "valid_count": int(frame[item].notna().sum()),
            "missing_count": int(frame[item].isna().sum()),
            "missing_percentage": float(100 * frame[item].isna().mean()) if len(frame) else 0.0,
        }
        for item in items
    ]
    complete = working.dropna()
    if len(complete) < 2:
        raise InsufficientDataError(
            "Reliability analysis requires at least two respondents complete on every selected "
            "item."
        )
    matrix = complete.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise InvalidDataError("Complete-case item scores contain nonfinite values.")
    alpha = cronbach_alpha(matrix)
    n, k = matrix.shape
    constant_items = [items[index] for index in range(k) if float(np.ptp(matrix[:, index])) == 0]
    warnings = []
    if constant_items:
        warnings.append(
            "Constant items were retained because alpha remains mathematically defined, but "
            f"their item correlations are unavailable: {constant_items!r}."
        )
    if k == 2:
        warnings.append(
            "This scale contains only two items. Alpha is closely tied to their inter-item "
            "correlation and gives limited information about broader internal consistency."
        )
    if alpha < 0:
        warnings.append(
            "Cronbach's alpha is negative, which is a strong review cue; review item scoring "
            "direction and whether the supplied items are intended to form one positively "
            "aligned scale."
        )
    interval = _bootstrap_interval(
        matrix,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )
    if interval["status"] != "available":
        warnings.append(str(interval["reason"]))

    totals = np.sum(matrix, axis=1)
    item_statistics = []
    for index, item in enumerate(items):
        values = matrix[:, index]
        corrected, corrected_status, corrected_reason = _correlation(values, totals - values)
        if k == 2:
            deleted = None
            deleted_status = "not_applicable"
            deleted_reason = "Removing one item would leave a one-item scale."
            delta = None
        else:
            try:
                deleted = cronbach_alpha(np.delete(matrix, index, axis=1))
                deleted_status = "available"
                deleted_reason = None
                delta = deleted - alpha
            except InsufficientDataError as exc:
                deleted = None
                deleted_status = "unavailable"
                deleted_reason = str(exc)
                delta = None
        missing = missingness[index]
        item_statistics.append(
            {
                "item": item,
                **missing,
                "analyzed_count": n,
                "mean": _finite(np.mean(values), "item mean"),
                "standard_deviation": _finite(np.std(values, ddof=1), "item standard deviation"),
                "minimum": _finite(np.min(values), "item minimum"),
                "maximum": _finite(np.max(values), "item maximum"),
                "corrected_item_total_correlation": corrected,
                "corrected_item_total_status": corrected_status,
                "corrected_item_total_reason": corrected_reason,
                "corrected_total_excludes_focal_item": True,
                "alpha_if_deleted": deleted,
                "alpha_if_deleted_status": deleted_status,
                "alpha_if_deleted_reason": deleted_reason,
                "delta_from_full_alpha": delta,
                "deleted_item": item,
                "remaining_item_count": k - 1,
            }
        )

    correlation_values: list[list[float | None]] = []
    valid_off_diagonal: list[float] = []
    negative_pairs: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(items):
        row: list[float | None] = []
        for right_index, right_name in enumerate(items):
            value, status, _ = _correlation(matrix[:, left_index], matrix[:, right_index])
            row.append(value if status == "available" else None)
            if left_index < right_index and value is not None:
                valid_off_diagonal.append(value)
                if value < 0:
                    negative_pairs.append(
                        {"first_item": left_name, "second_item": right_name, "correlation": value}
                    )
        correlation_values.append(row)
    most_negative = (
        min(negative_pairs, key=lambda pair: float(pair["correlation"])) if negative_pairs else None
    )
    excluded = len(frame) - n
    return {
        "method": "Cronbach's alpha",
        "target": "internal consistency of the researcher-declared item set",
        "items": list(items),
        "item_count": k,
        "sample": {
            "original_rows": int(len(frame)),
            "analyzed_rows": n,
            "excluded_rows": int(excluded),
            "excluded_percentage": float(100 * excluded / len(frame)) if len(frame) else 0.0,
            "missing_data_policy": "complete cases across all selected items",
            "imputation_applied": False,
        },
        "cronbach_alpha": alpha,
        "confidence_interval": interval,
        "item_statistics": item_statistics,
        "inter_item_correlations": {"items": list(items), "values": correlation_values},
        "mean_inter_item_correlation": (
            float(np.mean(valid_off_diagonal)) if valid_off_diagonal else None
        ),
        "negative_inter_item_correlations": {
            "count": len(negative_pairs),
            "most_negative_pair": most_negative,
            "pairs": negative_pairs,
        },
        "missingness": missingness,
        "scoring": {
            "reverse_scoring_applied": bool(reverse_metadata),
            "reversed_items": reverse_metadata,
            "automatic_reverse_scoring": False,
            "source_dataframe_modified": False,
            "ordinal_note": (
                "Supplied numeric item scores are treated as quantitative values; ordinal or "
                "polychoric reliability is not implemented."
            ),
        },
        "formula": {
            "name": "Cronbach's alpha",
            "definition": "k/(k-1) * (1 - sum(item variances)/variance(total score))",
            "variance_ddof": 1,
            "version": "pyautostat-cronbach-alpha-v1",
        },
        "warnings": warnings,
    }
