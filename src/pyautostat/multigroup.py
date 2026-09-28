"""Validated omnibus and pairwise backends for independent multi-group analyses."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import ColumnNotFoundError, InsufficientDataError, InvalidTestError


def _confidence_level(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or not 0 < float(value) < 1
    ):
        raise InvalidTestError("confidence_level must be a finite number between 0 and 1.")
    return float(value)


def _alpha(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or not 0 < float(value) < 1
    ):
        raise InvalidTestError("alpha must be a finite number between 0 and 1.")
    return float(value)


def _groups(frame: pd.DataFrame, group_col: str, value_col: str) -> dict[str, Any]:
    for name in (group_col, value_col):
        if not isinstance(name, str) or not name.strip():
            raise InvalidTestError("Column names must be non-empty strings.")
        if name not in frame.columns:
            raise ColumnNotFoundError(f"'{name}' is not a column in this DataFrame.")
    if group_col == value_col:
        raise InvalidTestError("group_col and value_col must name different columns.")
    if not pd.api.types.is_numeric_dtype(frame[value_col]):
        raise InvalidTestError(f"'{value_col}' must be a real numeric column.")
    usable = frame[[group_col, value_col]].dropna()
    labels = list(pd.unique(usable[group_col]))
    if len(labels) < 3:
        raise InsufficientDataError("Multi-group analysis requires at least three usable groups.")
    values: list[np.ndarray] = []
    for label in labels:
        try:
            group = np.asarray(usable.loc[usable[group_col] == label, value_col], dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidTestError(
                f"'{value_col}' cannot be represented as finite real values."
            ) from exc
        if len(group) < 2:
            raise InsufficientDataError(
                "Each group needs at least two complete observations for multi-group analysis."
            )
        if not np.isfinite(group).all():
            raise InvalidTestError(f"'{value_col}' contains nonfinite values.")
        values.append(group)
    all_values = np.concatenate(values)
    reference = float(all_values[0])
    centered = all_values - reference
    scale = float(np.max(np.abs(centered)))
    if not math.isfinite(scale):
        raise InsufficientDataError("Outcome values cannot be represented on a stable scale.")
    if scale == 0:
        raise InsufficientDataError("Multi-group inference needs at least two distinct outcomes.")
    normalized = [(group - reference) / scale for group in values]
    summaries = []
    for label, raw, scaled in zip(labels, values, normalized, strict=True):
        summary = {
            "group": label,
            "sample_size": int(len(raw)),
            "mean": float(np.mean(scaled) * scale + reference),
            "standard_deviation": float(np.std(scaled, ddof=1) * scale),
            "median": float(np.median(scaled) * scale + reference),
        }
        if not all(math.isfinite(summary[key]) for key in ("mean", "standard_deviation", "median")):
            raise InsufficientDataError(
                "A group summary is nonfinite at this numerical scale; rescale the outcome."
            )
        summaries.append(summary)
    return {
        "labels": labels,
        "values": values,
        "normalized": normalized,
        "reference": reference,
        "scale": scale,
        "summaries": summaries,
        "sample_size": int(len(usable)),
        "excluded_rows": int(len(frame) - len(usable)),
    }


def adjust_pvalues(p_values: list[float], method: str = "holm") -> list[float]:
    """Adjust a complete family of p-values while preserving its original order."""
    if method not in {"holm", "bonferroni"}:
        raise InvalidTestError("adjustment method must be 'holm' or 'bonferroni'.")
    if not p_values:
        return []
    values = [float(value) for value in p_values]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise InvalidTestError("p-values must be finite numbers between 0 and 1.")
    count = len(values)
    if method == "bonferroni":
        return [min(1.0, count * value) for value in values]
    order = sorted(range(count), key=lambda index: (values[index], index))
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _pair_record(
    *,
    procedure: str,
    first: Any,
    second: Any,
    estimate_name: str,
    estimate: float,
    statistic_name: str,
    statistic: float,
    raw_p_value: float | None,
    adjusted_p_value: float,
    adjustment_method: str,
    comparison_count: int,
    confidence_interval: dict[str, Any] | None,
    effect_size: dict[str, Any] | None,
    n1: int,
    n2: int,
    standard_error: float,
    degrees_of_freedom: float | None,
    alpha: float,
) -> dict[str, Any]:
    return {
        "procedure": procedure,
        "group1": first,
        "group2": second,
        "contrast": {
            "definition": "first group minus second group",
            "first_group": first,
            "second_group": second,
        },
        "estimate_name": estimate_name,
        "estimate": float(estimate),
        "statistic_name": statistic_name,
        "statistic": float(statistic),
        "raw_p_value": None if raw_p_value is None else float(raw_p_value),
        "adjusted_p_value": float(adjusted_p_value),
        "adjustment_method": adjustment_method,
        "comparison_count": comparison_count,
        "confidence_interval": confidence_interval,
        "effect_size": effect_size,
        "sample_sizes": {"group1": n1, "group2": n2},
        "standard_error": float(standard_error),
        "degrees_of_freedom": degrees_of_freedom,
        "alpha": alpha,
        "decision": "reject" if adjusted_p_value < alpha else "fail_to_reject",
        "warnings": [],
    }


def _pairwise_summary(procedure: str, records: list[dict[str, Any]]) -> str:
    rejected = sum(item["decision"] == "reject" for item in records)
    return (
        f"{procedure} calculated all {len(records)} pairwise comparisons regardless of the "
        f"omnibus decision; {rejected} rejected after the recorded multiplicity control."
    )


def games_howell(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    *,
    confidence_level: float = 0.95,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """All-pairs Games-Howell comparisons using the studentized-range distribution."""
    grouped = _groups(frame, group_col, value_col)
    level, decision_alpha = _confidence_level(confidence_level), _alpha(alpha)
    labels, groups = grouped["labels"], grouped["normalized"]
    means = [float(np.mean(group)) for group in groups]
    variances = [float(np.var(group, ddof=1)) for group in groups]
    if any(not math.isfinite(value) or value <= 0 for value in variances):
        raise InsufficientDataError(
            "Games-Howell needs positive, finite within-group variance in every group."
        )
    scale = grouped["scale"]
    count = len(labels) * (len(labels) - 1) // 2
    records = []
    for first, second in combinations(range(len(labels)), 2):
        n1, n2 = len(groups[first]), len(groups[second])
        component1, component2 = variances[first] / n1, variances[second] / n2
        variance = component1 + component2
        df = variance**2 / (component1**2 / (n1 - 1) + component2**2 / (n2 - 1))
        difference = means[first] - means[second]
        standard_error = math.sqrt(variance)
        q_value = math.sqrt(2) * abs(difference) / standard_error
        raw_p_value = float(2 * stats.t.sf(abs(difference) / standard_error, df))
        p_value = float(stats.studentized_range.sf(q_value, len(labels), df))
        critical = float(stats.studentized_range.ppf(level, len(labels), df))
        if not all(
            math.isfinite(value) for value in (df, q_value, raw_p_value, p_value, critical)
        ) or not (0 <= raw_p_value <= 1 and 0 <= p_value <= 1):
            raise InsufficientDataError("Games-Howell produced a nonfinite pairwise result.")
        margin = critical * standard_error / math.sqrt(2) * scale
        raw_difference = difference * scale
        records.append(
            _pair_record(
                procedure="games_howell",
                first=labels[first],
                second=labels[second],
                estimate_name="mean difference",
                estimate=raw_difference,
                statistic_name="studentized range q",
                statistic=q_value,
                raw_p_value=raw_p_value,
                adjusted_p_value=p_value,
                adjustment_method="Games-Howell studentized-range familywise inference",
                comparison_count=count,
                confidence_interval={
                    "lower": raw_difference - margin,
                    "upper": raw_difference + margin,
                    "level": level,
                    "method": "Games-Howell simultaneous studentized-range interval",
                },
                effect_size=None,
                n1=n1,
                n2=n2,
                standard_error=standard_error * scale,
                degrees_of_freedom=float(df),
                alpha=decision_alpha,
            )
        )
    return {
        "procedure": "Games-Howell",
        "group_order": labels,
        "group_summaries": grouped["summaries"],
        "comparisons": records,
        "comparison_count": count,
        "multiplicity_control": "studentized-range familywise inference",
        "confidence_level": level,
        "alpha": decision_alpha,
        "sample_size": grouped["sample_size"],
        "excluded_rows": grouped["excluded_rows"],
        "interpretation": _pairwise_summary("Games-Howell", records),
    }


def welch_anova(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    *,
    confidence_level: float = 0.95,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Welch's one-way ANOVA with an unconditional Games-Howell follow-up family."""
    grouped = _groups(frame, group_col, value_col)
    groups, labels = grouped["normalized"], grouped["labels"]
    means = np.asarray([np.mean(group) for group in groups], dtype=float)
    variances = np.asarray([np.var(group, ddof=1) for group in groups], dtype=float)
    sizes = np.asarray([len(group) for group in groups], dtype=float)
    if np.any(~np.isfinite(variances)) or np.any(variances <= 0):
        raise InsufficientDataError(
            "Welch ANOVA needs positive, finite within-group variance in every group."
        )
    weights = sizes / variances
    weight_sum = float(np.sum(weights))
    weighted_mean = float(np.sum(weights * means) / weight_sum)
    group_count = len(groups)
    correction_term = float(np.sum((1 - weights / weight_sum) ** 2 / (sizes - 1)))
    denominator = 1 + 2 * (group_count - 2) * correction_term / (group_count**2 - 1)
    statistic = float(np.sum(weights * (means - weighted_mean) ** 2) / (group_count - 1))
    statistic /= denominator
    df1 = float(group_count - 1)
    df2 = float((group_count**2 - 1) / (3 * correction_term))
    p_value = float(stats.f.sf(statistic, df1, df2))
    if not all(math.isfinite(value) for value in (statistic, df2, p_value)):
        raise InsufficientDataError("Welch ANOVA produced a nonfinite result; rescale the data.")
    pairwise = games_howell(
        frame,
        group_col,
        value_col,
        confidence_level=confidence_level,
        alpha=alpha,
    )
    result = {
        "test": "Welch one-way ANOVA",
        "statistic": statistic,
        "p_value": p_value,
        "degrees_of_freedom": [df1, df2],
        "groups": labels,
        "group_summaries": grouped["summaries"],
        "effect_size": {
            "name": "global standardized effect",
            "value": None,
            "status": "not_applicable",
            "reason": (
                "No global standardized effect is reported for Welch ANOVA; group means and "
                "Games-Howell mean differences are the estimands."
            ),
            "confidence_interval": None,
        },
        "confidence_interval": None,
        "pairwise_method": pairwise["procedure"],
        "multiplicity_control": pairwise["multiplicity_control"],
        "pairwise_comparisons": pairwise["comparisons"],
        "sample_size": grouped["sample_size"],
        "excluded_rows": grouped["excluded_rows"],
        "group_sizes": [
            {"group": summary["group"], "size": summary["sample_size"]}
            for summary in grouped["summaries"]
        ],
        "assumptions": {
            "independent_observations": "Required; cannot be verified from numerical values.",
            "equal_variances": "Not required by Welch ANOVA or Games-Howell.",
            "warnings": ["Independence must be confirmed from the study design."],
            "requested_test_type": "auto",
            "selected_test_type": "welch_anova",
            "selection_reason": "Selected for an independent multi-group mean estimand.",
            "estimand": "mean",
        },
    }
    result["interpretation"] = f"Welch one-way ANOVA p = {p_value:.4g}. " + _pairwise_summary(
        "Games-Howell", pairwise["comparisons"]
    )
    return result


def tukey_hsd(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    *,
    confidence_level: float = 0.95,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """All-pairs Tukey-Kramer HSD for a classical equal-variance ANOVA model."""
    grouped = _groups(frame, group_col, value_col)
    level, decision_alpha = _confidence_level(confidence_level), _alpha(alpha)
    labels, groups = grouped["labels"], grouped["normalized"]
    sizes = [len(group) for group in groups]
    means = [float(np.mean(group)) for group in groups]
    residual_df = sum(sizes) - len(groups)
    mse = sum((len(group) - 1) * float(np.var(group, ddof=1)) for group in groups) / residual_df
    if not math.isfinite(mse) or mse <= 0:
        raise InsufficientDataError(
            "Tukey HSD needs positive, finite pooled within-group variance."
        )
    critical = float(stats.studentized_range.ppf(level, len(groups), residual_df))
    count = len(labels) * (len(labels) - 1) // 2
    scale = grouped["scale"]
    records = []
    for first, second in combinations(range(len(labels)), 2):
        se = math.sqrt(mse * (1 / sizes[first] + 1 / sizes[second]))
        difference = means[first] - means[second]
        q_value = math.sqrt(2) * abs(difference) / se
        raw_p_value = float(2 * stats.t.sf(abs(difference) / se, residual_df))
        p_value = float(stats.studentized_range.sf(q_value, len(groups), residual_df))
        if not all(
            math.isfinite(value) for value in (q_value, raw_p_value, p_value, critical)
        ) or not (0 <= raw_p_value <= 1 and 0 <= p_value <= 1):
            raise InsufficientDataError("Tukey HSD produced a nonfinite pairwise result.")
        raw_difference = difference * scale
        margin = critical * se / math.sqrt(2) * scale
        records.append(
            _pair_record(
                procedure="tukey_hsd",
                first=labels[first],
                second=labels[second],
                estimate_name="mean difference",
                estimate=raw_difference,
                statistic_name="studentized range q",
                statistic=q_value,
                raw_p_value=raw_p_value,
                adjusted_p_value=p_value,
                adjustment_method="Tukey studentized-range familywise inference",
                comparison_count=count,
                confidence_interval={
                    "lower": raw_difference - margin,
                    "upper": raw_difference + margin,
                    "level": level,
                    "method": "Tukey-Kramer simultaneous studentized-range interval",
                },
                effect_size=None,
                n1=sizes[first],
                n2=sizes[second],
                standard_error=se * scale,
                degrees_of_freedom=float(residual_df),
                alpha=decision_alpha,
            )
        )
    return {
        "procedure": "Tukey HSD (Tukey-Kramer for unequal sample sizes)",
        "group_order": labels,
        "group_summaries": grouped["summaries"],
        "comparisons": records,
        "comparison_count": count,
        "multiplicity_control": "studentized-range familywise inference",
        "confidence_level": level,
        "alpha": decision_alpha,
        "sample_size": grouped["sample_size"],
        "excluded_rows": grouped["excluded_rows"],
        "interpretation": _pairwise_summary("Tukey HSD", records),
    }


def dunn(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    *,
    adjustment: str = "holm",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """All-pairs Dunn comparisons with a tie correction and central p adjustment."""
    grouped = _groups(frame, group_col, value_col)
    decision_alpha = _alpha(alpha)
    labels, groups = grouped["labels"], grouped["normalized"]
    if any(len(group) < 5 for group in groups):
        raise InsufficientDataError(
            "Dunn follow-up needs at least five observations per group for its normal "
            "approximation."
        )
    pooled = np.concatenate(groups)
    ranks = stats.rankdata(pooled, method="average")
    _, tie_counts = np.unique(pooled, return_counts=True)
    size = len(pooled)
    tie_correction = 1 - float(np.sum(tie_counts**3 - tie_counts)) / (size**3 - size)
    if tie_correction <= 0:
        raise InsufficientDataError("Dunn comparisons are undefined when all outcomes are tied.")
    base_variance = size * (size + 1) / 12 * tie_correction
    offsets = np.cumsum([0, *[len(group) for group in groups]])
    mean_ranks = [float(np.mean(ranks[offsets[i] : offsets[i + 1]])) for i in range(len(groups))]
    pair_data = []
    raw_p_values = []
    for first, second in combinations(range(len(labels)), 2):
        se = math.sqrt(base_variance * (1 / len(groups[first]) + 1 / len(groups[second])))
        difference = mean_ranks[first] - mean_ranks[second]
        z_value = difference / se
        raw_p = float(2 * stats.norm.sf(abs(z_value)))
        if not all(math.isfinite(value) for value in (se, difference, z_value, raw_p)):
            raise InsufficientDataError("Dunn's test produced a nonfinite pairwise result.")
        pair_data.append((first, second, difference, se, z_value))
        raw_p_values.append(raw_p)
    adjusted = adjust_pvalues(raw_p_values, adjustment)
    records = []
    count = len(pair_data)
    for data, raw_p, adjusted_p in zip(pair_data, raw_p_values, adjusted, strict=True):
        first, second, difference, se, z_value = data
        x, y = groups[first], groups[second]
        combined_ranks = stats.rankdata(np.concatenate([x, y]), method="average")
        u_first = float(np.sum(combined_ranks[: len(x)]) - len(x) * (len(x) + 1) / 2)
        rank_biserial = 2 * u_first / (len(x) * len(y)) - 1
        records.append(
            _pair_record(
                procedure="dunn",
                first=labels[first],
                second=labels[second],
                estimate_name="pooled mean-rank difference",
                estimate=difference,
                statistic_name="Dunn z",
                statistic=z_value,
                raw_p_value=raw_p,
                adjusted_p_value=adjusted_p,
                adjustment_method=adjustment,
                comparison_count=count,
                confidence_interval=None,
                effect_size={
                    "name": "pairwise rank-biserial correlation",
                    "value": float(rank_biserial),
                    "confidence_interval": None,
                    "uncertainty_status": "unavailable",
                    "reason": (
                        "No validated pairwise effect interval is calculated for Dunn comparisons."
                    ),
                },
                n1=len(x),
                n2=len(y),
                standard_error=se,
                degrees_of_freedom=None,
                alpha=decision_alpha,
            )
        )
    return {
        "procedure": "Dunn test",
        "group_order": labels,
        "group_summaries": grouped["summaries"],
        "comparisons": records,
        "comparison_count": count,
        "multiplicity_control": adjustment,
        "tie_correction": tie_correction,
        "alpha": decision_alpha,
        "sample_size": grouped["sample_size"],
        "excluded_rows": grouped["excluded_rows"],
        "interpretation": _pairwise_summary(f"Dunn-{adjustment}", records),
    }
