"""Deterministic narration of already-computed statistical effects.

This module contains presentation rules only. It does not calculate effect
sizes or confidence intervals and it does not select statistical methods.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _EffectRule:
    display_name: str
    concept: str
    bands: tuple[tuple[float, str], ...] | None
    signed: bool = False
    nonnegative: bool = False


_COHENS_D_BANDS = (
    (2.0, "very large"),
    (0.8, "large"),
    (0.5, "medium"),
    (0.2, "small"),
    (0.0, "negligible"),
)
_ETA_SQUARED_BANDS = (
    (0.14, "large"),
    (0.06, "medium"),
    (0.01, "small"),
    (0.0, "negligible"),
)
_CORRELATION_BANDS = (
    (0.5, "large"),
    (0.3, "medium"),
    (0.1, "small"),
    (0.0, "negligible"),
)

_RULES = {
    "cohens_d": _EffectRule(
        "Cohen's d", "standardized mean separation", _COHENS_D_BANDS, signed=True
    ),
    "cohens_dz": _EffectRule(
        "Cohen's dz", "standardized paired-mean separation", _COHENS_D_BANDS, signed=True
    ),
    "rank_biserial": _EffectRule(
        "Rank-biserial correlation", "rank-based group separation", None, signed=True
    ),
    "eta_squared": _EffectRule(
        "Eta-squared",
        "proportion-style variance association for ANOVA",
        _ETA_SQUARED_BANDS,
        nonnegative=True,
    ),
    "epsilon_squared": _EffectRule(
        "Epsilon-squared",
        "non-parametric effect measure associated with Kruskal-Wallis",
        None,
        nonnegative=True,
    ),
    "cramers_v": _EffectRule(
        "Cramer's V", "categorical association strength", _CORRELATION_BANDS, nonnegative=True
    ),
    "pearson_r": _EffectRule(
        "Pearson r", "linear association strength", _CORRELATION_BANDS, signed=True
    ),
}

_MEASURE_ALIASES = {
    "cohens_d": "cohens_d",
    "cohen's d": "cohens_d",
    "cohens_dz": "cohens_dz",
    "cohen's dz": "cohens_dz",
    "rank_biserial": "rank_biserial",
    "rank-biserial correlation": "rank_biserial",
    "eta_squared": "eta_squared",
    "eta-squared": "eta_squared",
    "epsilon_squared": "epsilon_squared",
    "epsilon-squared": "epsilon_squared",
    "epsilon-squared (rank)": "epsilon_squared",
    "cramers_v": "cramers_v",
    "cramer's v": "cramers_v",
    "pearson_r": "pearson_r",
    "pearson r": "pearson_r",
}

_D_CONSEQUENCES = {
    "very large": (
        "The groups differ by more than 2 standard deviations — a difference of this size is "
        "usually visible in raw data."
    ),
    "large": (
        "A large effect that is likely to be practically meaningful in most research contexts."
    ),
    "medium": "A medium effect — noticeable in careful observation.",
    "small": "A small effect — real but subtle; it may require a large sample to detect reliably.",
    "negligible": (
        "Below conventional detection thresholds; the groups are nearly indistinguishable on "
        "this measure."
    ),
}

ASSUMPTION_SEVERITY_DESCRIPTIONS = {
    "CRITICAL": "The result cannot be trusted without resolving this.",
    "WARNING": "This may materially affect interpretation.",
    "CAUTION": "Worth noting; low risk given context.",
    "INFO": "For transparency only; no action required.",
}

_NONTRIVIAL_MAGNITUDES = {"large", "very large", "moderate", "strong"}
_KNOWN_MAGNITUDES = _NONTRIVIAL_MAGNITUDES | {"negligible", "small", "medium"}

_POINT_RELATIONS = {
    "positive_meaningful_region",
    "negative_meaningful_region",
    "below_meaningful_magnitude",
    "meets_positive_threshold",
    "below_positive_threshold",
    "meets_negative_threshold",
    "above_negative_threshold",
    "meets_nonnegative_threshold",
    "below_nonnegative_threshold",
}
_INTERVAL_RELATIONS = {
    "entirely_positive_meaningful",
    "entirely_negative_meaningful",
    "entirely_within_negligible_region",
    "spans_both_directions",
    "crosses_meaningful_boundary",
    "entirely_above_positive_threshold",
    "entirely_below_positive_threshold",
    "crosses_positive_threshold",
    "entirely_below_negative_threshold",
    "entirely_above_negative_threshold",
    "crosses_negative_threshold",
    "entirely_at_or_above_nonnegative_threshold",
    "entirely_below_nonnegative_threshold",
    "crosses_nonnegative_threshold",
    "unavailable",
}
_POINT_CONFIRMING_INTERVALS = {
    "positive_meaningful_region": {"entirely_positive_meaningful"},
    "negative_meaningful_region": {"entirely_negative_meaningful"},
    "below_meaningful_magnitude": {"entirely_within_negligible_region"},
    "meets_positive_threshold": {"entirely_above_positive_threshold"},
    "below_positive_threshold": {"entirely_below_positive_threshold"},
    "meets_negative_threshold": {"entirely_below_negative_threshold"},
    "above_negative_threshold": {"entirely_above_negative_threshold"},
    "meets_nonnegative_threshold": {"entirely_at_or_above_nonnegative_threshold"},
    "below_nonnegative_threshold": {"entirely_below_nonnegative_threshold"},
}
_MEANINGFUL_POINTS = {
    "positive_meaningful_region",
    "negative_meaningful_region",
    "meets_positive_threshold",
    "meets_negative_threshold",
    "meets_nonnegative_threshold",
}
_CROSSING_INTERVALS = {
    "spans_both_directions",
    "crosses_meaningful_boundary",
    "crosses_positive_threshold",
    "crosses_negative_threshold",
    "crosses_nonnegative_threshold",
}


def _interval_table_entry(point: str, interval: str) -> tuple[str, str]:
    if interval == "unavailable":
        return (
            "POINT ESTIMATE ONLY",
            "The point estimate was compared with the declared threshold, but no confidence "
            "interval is available to assess precision around that judgment.",
        )
    if interval in _POINT_CONFIRMING_INTERVALS[point]:
        if point in _MEANINGFUL_POINTS:
            return (
                "CONFIRMED ABOVE THRESHOLD",
                "The observed effect exceeds the declared threshold and the entire confidence "
                "interval remains in the corresponding meaningful region. The conclusion is "
                "strongly supported by the recorded interval.",
            )
        return (
            "BELOW THRESHOLD",
            "Both the point estimate and the entire confidence interval fall below the declared "
            "meaningful effect. The effect is not practically meaningful by the stated criterion.",
        )
    if interval in _CROSSING_INTERVALS:
        if point in _MEANINGFUL_POINTS:
            return (
                "LIKELY ABOVE THRESHOLD",
                "The point estimate exceeds the threshold, but the confidence interval crosses "
                "a meaningful boundary. The effect is plausibly above the threshold, but "
                "uncertainty remains; consider whether greater precision is needed.",
            )
        return (
            "INCONCLUSIVE",
            "The point estimate is below the threshold, but the confidence interval reaches or "
            "crosses a meaningful region. Precision is insufficient to decide from the recorded "
            "interval.",
        )
    return (
        "INCONCLUSIVE",
        "The point estimate and confidence interval occupy different threshold regions. Report "
        "both recorded relations; the practical-significance judgment is inconclusive.",
    )


_INTERVAL_VERDICT_TABLE = {
    (point, interval): _interval_table_entry(point, interval)
    for point in _POINT_RELATIONS
    for interval in _INTERVAL_RELATIONS
}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _canonical_measure(measure: Any) -> str | None:
    if not isinstance(measure, str) or not measure.strip():
        return None
    return _MEASURE_ALIASES.get(measure.strip().lower())


def _magnitude_label(rule: _EffectRule, value: float) -> str | None:
    if rule.bands is None:
        return None
    magnitude = abs(value)
    for cutoff, label in rule.bands:
        if magnitude >= cutoff:
            return label
    return None


def _p_value_text(value: float) -> str:
    if value == 0:
        return "p < 0.001 (computational zero)"
    return f"p = {_fmt(value)}"


def hypothesis_verdict(
    p: float | None,
    alpha: float | None,
    effect_val: float | None,
    measure: str,
    magnitude_label: str | None = None,
    n: int | None = None,
) -> str:
    """Narrate the recorded p-value/effect quadrant without recomputing either value."""
    p_value = _finite(p)
    alpha_value = _finite(alpha)
    if p_value is None or alpha_value is None or not 0 <= p_value <= 1 or not 0 < alpha_value < 1:
        return "Hypothesis verdict unavailable: a finite p-value and valid alpha are required."

    canonical = _canonical_measure(measure)
    effect = _finite(effect_val)
    significant = p_value < alpha_value
    p_text = _p_value_text(p_value)
    significance_text = (
        f"{p_text}. The result is statistically significant at alpha = {_fmt(alpha_value)}"
        if significant
        else f"{p_text}. The result does not reach significance at alpha = {_fmt(alpha_value)}"
    )
    if canonical is None or effect is None:
        return (
            f"{significance_text}. The effect magnitude is unavailable, so the four-quadrant "
            "verdict cannot be assigned."
        )
    rule = _RULES[canonical]
    label = magnitude_label or _magnitude_label(rule, effect)
    if label is None or label not in _KNOWN_MAGNITUDES:
        return (
            f"{significance_text}. {rule.display_name} = {_fmt(effect)}, but no supported "
            "magnitude label is available; the effect was not classified as trivial."
        )

    effect_text = f"{rule.display_name} = {_fmt(effect)}"
    nontrivial = label in _NONTRIVIAL_MAGNITUDES
    if significant and nontrivial:
        return (
            f"{significance_text}, and the effect size is {label} ({effect_text}). Both "
            "significance and conventional magnitude support a non-trivial difference."
        )
    if significant:
        sample_text = (
            f" With n = {n:,}, even a trivial effect can produce a small p-value."
            if isinstance(n, int) and not isinstance(n, bool) and n > 0
            else " A small p-value can occur even when the observed effect is limited."
        )
        return (
            f"{significance_text}, but the effect is {label} ({effect_text}).{sample_text} "
            "Check the confidence interval and a researcher-declared practical threshold before "
            "acting. Statistical significance does not establish effect size, practical "
            "importance, or replication."
        )
    if nontrivial:
        return (
            f"{significance_text}, yet the observed {effect_text} ({label}) is non-trivial. "
            "The data may be under-powered for this effect; consider prospective power analysis "
            "rather than treating non-significance as proof of no effect."
        )
    return (
        f"{significance_text}, and the observed effect is {label} ({effect_text}). There is "
        "little evidence here of a non-trivial difference, although this is not an equivalence "
        "test."
    )


def assumption_grade(
    assumption: str,
    status: str,
    *,
    n: int | None = None,
    p_value: float | None = None,
    group: Any = None,
    method_id: str | None = None,
) -> tuple[str, str]:
    """Return a deterministic ``(severity, message)`` for a recorded assumption state."""
    assumption_key = assumption.strip().lower() if isinstance(assumption, str) else "unknown"
    status_key = status.strip().lower() if isinstance(status, str) else "unknown"
    checked_n = n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else None
    checked_p = _finite(p_value)
    p_text = f" ({_p_value_text(checked_p)})" if checked_p is not None else ""
    group_text = f" for group {group!r}" if group is not None else ""

    if assumption_key == "normality":
        if status_key == "rejected":
            if checked_n is None:
                severity = "WARNING"
                message = (
                    f"The normality diagnostic rejected normality{group_text}{p_text}; the group "
                    "sample size is unavailable, so large-sample robustness cannot be assessed."
                )
            elif checked_n >= 30:
                severity = "INFO"
                message = (
                    f"The normality diagnostic rejected normality{group_text}{p_text}, but "
                    f"n = {checked_n}. Large-group central-limit robustness generally reduces "
                    "the risk for mean inference; inspect severe skew or outliers separately."
                )
            elif checked_n >= 15:
                severity = "CAUTION"
                message = (
                    f"The normality diagnostic rejected normality{group_text}{p_text}; n = "
                    f"{checked_n}. This is a moderate-risk issue: inspect the distribution and "
                    "consider whether a method targeting a different estimand is appropriate."
                )
            else:
                severity = "WARNING"
                message = (
                    f"The normality diagnostic rejected normality{group_text}{p_text}; n = "
                    f"{checked_n}. This small group may make parametric mean inference unreliable."
                )
        elif status_key == "not_rejected":
            severity = "INFO"
            message = (
                f"Normality was not rejected{group_text}{p_text}; this does not establish "
                "normality."
            )
        else:
            severity = "CAUTION"
            message = f"Normality diagnostic{group_text}: {status_key}; normality was not verified."
    elif assumption_key == "equal_variance":
        if status_key == "rejected" and method_id == "welch_t":
            severity = "INFO"
            detail = (
                f" Levene's test reported {_p_value_text(checked_p)}."
                if checked_p is not None
                else ""
            )
            message = (
                "The equal-variance diagnostic rejected equal variances. Welch's correction was "
                f"used, so equal variances were not assumed.{detail}"
            )
        elif status_key == "rejected":
            severity = "WARNING"
            detail = (
                f" Levene's test reported {_p_value_text(checked_p)}."
                if checked_p is not None
                else ""
            )
            message = f"The equal-variance diagnostic rejected equal variances.{detail}"
        elif status_key == "not_rejected":
            severity = "INFO"
            message = (
                "The equal-variance diagnostic did not reject equal variances; this does not "
                "prove equal population variances."
            )
        elif status_key == "required" and method_id in {"student_t", "one_way_anova"}:
            severity = "CAUTION"
            message = (
                "This pooled method assumes equal population variances; Levene's test cannot "
                "prove this condition."
            )
        elif status_key == "not_required" and method_id == "welch_t":
            severity = "INFO"
            message = (
                "Welch's test does not require equal population variances; independence and "
                "appropriate mean-inference conditions still matter."
            )
        else:
            severity = "CAUTION"
            message = f"Equal-variance diagnostic: {status_key}."
    elif assumption_key == "independence":
        severity = "CAUTION"
        message = (
            "Independence must be verified from the study design; numerical values cannot "
            "establish independent observations."
        )
    elif assumption_key == "paired_structure":
        severity = "CAUTION"
        message = (
            "The paired t-test models within-unit differences; complete pairs must be independent "
            "across units and suitable for mean inference."
        )
    elif status_key in {"critical", "failed", "violated"}:
        severity = "CRITICAL"
        message = f"The recorded {assumption_key.replace('_', ' ')} condition was {status_key}."
    elif status_key in {"rejected", "warning"}:
        severity = "WARNING"
        message = f"The recorded {assumption_key.replace('_', ' ')} condition was {status_key}."
    elif status_key in {"unknown", "unverified", "required"}:
        severity = "CAUTION"
        message = f"The recorded {assumption_key.replace('_', ' ')} condition is {status_key}."
    else:
        severity = "INFO"
        message = f"The recorded {assumption_key.replace('_', ' ')} condition is {status_key}."
    return severity, f"{message} {ASSUMPTION_SEVERITY_DESCRIPTIONS[severity]}"


def _confidence_interval_width(estimate: float, lower: float, upper: float) -> str | None:
    """Classify recorded interval width relative to a recorded estimate."""
    if not all(math.isfinite(item) for item in (estimate, lower, upper)) or lower > upper:
        return None
    width = upper - lower
    magnitude = abs(estimate)
    if magnitude == 0:
        return (
            "narrow (precise estimate)"
            if width == 0
            else "wide — the true effect is poorly constrained"
        )
    ratio = width / magnitude
    if ratio < 0.5:
        return "narrow (precise estimate)"
    if ratio < 2.0:
        return "moderate width"
    return "wide — the true effect is poorly constrained"


def _interval_parts(
    ci: tuple[float | None, float | None] | Mapping[str, Any] | None,
    confidence_level: float | None,
) -> tuple[float, float, float | None, str | None] | None:
    if ci is None:
        return None
    method: str | None = None
    if isinstance(ci, Mapping):
        lower = _finite(ci.get("lower"))
        upper = _finite(ci.get("upper"))
        recorded_level = _finite(ci.get("level"))
        if recorded_level is not None:
            confidence_level = recorded_level
        recorded_method = ci.get("method")
        if isinstance(recorded_method, str) and recorded_method.strip():
            method = recorded_method.strip()
    elif isinstance(ci, tuple) and len(ci) == 2:
        lower = _finite(ci[0])
        upper = _finite(ci[1])
    else:
        return None
    if lower is None or upper is None or lower > upper:
        return None
    level = _finite(confidence_level)
    if level is not None and not 0 < level < 1:
        level = None
    return lower, upper, level, method


def _confidence_interval_narrative(
    estimate: float,
    ci: tuple[float | None, float | None] | Mapping[str, Any] | None,
    confidence_level: float | None = None,
) -> str | None:
    """Narrate valid recorded CI bounds without calculating an interval."""
    parts = _interval_parts(ci, confidence_level)
    if parts is None:
        return None
    lower, upper, level, method = parts
    width = _confidence_interval_width(estimate, lower, upper)
    if width is None:
        return None
    level_text = f"{_fmt(100 * level)}% " if level is not None else ""
    method_text = "bootstrap " if method is not None and "bootstrap" in method.lower() else ""
    interval = f"The {level_text}{method_text}confidence interval [{_fmt(lower)}, {_fmt(upper)}]"
    if width == "moderate width":
        return f"{interval} has {width}."
    return f"{interval} is {width}."


def _ratio_sentence(estimate: float, threshold: float) -> str:
    if threshold == 0:
        return "The declared threshold is zero, so a finite magnitude ratio is not reported."
    ratio = abs(estimate) / threshold
    if ratio >= 5:
        return f"The observed effect is {ratio:.1f}x the declared threshold."
    if ratio >= 2:
        return f"The observed effect is {ratio:.1f} times the threshold — clearly above it."
    if ratio >= 1:
        return f"The observed effect just exceeds the threshold (ratio = {ratio:.2f})."
    return f"The observed effect is {ratio:.2f}x the threshold — below it."


def interval_verdict(
    point_relation: str,
    interval_relation: str,
    estimate: float | None,
    threshold: float,
    *,
    quantity: str,
    unit: str | None = None,
    confidence_interval: Mapping[str, Any] | None = None,
    direction: str = "two_sided",
    orientation: str | None = None,
) -> str:
    """Narrate existing practical-significance relations and their recorded values."""
    estimate_value = _finite(estimate)
    threshold_value = _finite(threshold)
    key = (point_relation, interval_relation)
    if estimate_value is None or threshold_value is None or threshold_value < 0:
        return "VERDICT: UNAVAILABLE. A finite estimate and nonnegative threshold are required."
    entry = _INTERVAL_VERDICT_TABLE.get(key)
    if entry is None:
        return (
            "VERDICT: UNAVAILABLE. The recorded practical-significance relations are unsupported; "
            "no narrative conclusion was assigned."
        )
    label, narrative = entry
    quantity_text = quantity.replace("_", " ") if isinstance(quantity, str) else "effect"
    unit_text = f" {unit}" if isinstance(unit, str) and unit.strip() else ""
    direction_text = (
        f" in the declared {direction.replace('_', ' ')} direction"
        if direction in {"positive", "negative", "nonnegative"}
        else ""
    )
    parts = [
        f"VERDICT: {label}.",
        f"The declared minimum meaningful {quantity_text} is {_fmt(threshold_value)}{unit_text}"
        f"{direction_text}.",
        f"The observed {quantity_text} is {_fmt(estimate_value)}{unit_text}.",
        _ratio_sentence(estimate_value, threshold_value),
        narrative,
    ]
    if orientation is not None and orientation.strip():
        parts.append(f"The recorded contrast is {orientation.strip()}.")
    if confidence_interval is not None:
        lower = _finite(confidence_interval.get("lower"))
        upper = _finite(confidence_interval.get("upper"))
        level = _finite(confidence_interval.get("level"))
        if lower is not None and upper is not None and lower <= upper:
            level_text = f"{_fmt(100 * level)}% " if level is not None and 0 < level < 1 else ""
            parts.append(
                f"The recorded {level_text}confidence interval is "
                f"[{_fmt(lower)}, {_fmt(upper)}]{unit_text}."
            )
    parts.append(
        "This judgment applies only to the researcher-declared threshold; it does not establish "
        "causation, equivalence, or replication."
    )
    return " ".join(parts)


def sensitivity_verdict(
    base_decision: str | None,
    scenario_decisions: Sequence[str | None],
    *,
    same_estimand: bool,
    completed_scenarios: int | None = None,
    mixed_estimands: bool = False,
    alpha: float | None = None,
) -> str:
    """Narrate stored sensitivity decisions without comparing unlike effect estimates."""
    allowed = {"reject null", "fail to reject null"}
    completed = (
        completed_scenarios
        if isinstance(completed_scenarios, int) and completed_scenarios >= 0
        else len(scenario_decisions)
    )
    if completed == 0:
        return (
            "UNAVAILABLE: No sensitivity scenario completed; hypothesis-decision consistency "
            "cannot be assessed."
        )
    if not same_estimand or not scenario_decisions:
        return (
            "DESCRIPTIVE: No completed alternative shares the primary estimand and comparison "
            "identity. Numerical p-values and effect estimates cannot be directly compared."
        )
    decisions = [base_decision, *scenario_decisions]
    if any(item not in allowed for item in decisions):
        return (
            "UNAVAILABLE: At least one same-estimand analysis has no valid hypothesis decision; "
            "consistency cannot be classified."
        )
    alpha_value = _finite(alpha)
    alpha_text = (
        f" at alpha = {_fmt(alpha_value)}"
        if alpha_value is not None and 0 < alpha_value < 1
        else " at the declared alpha level"
    )
    if len(set(decisions)) != 1:
        return (
            "INCONSISTENT: Methods disagree on the hypothesis decision. The hypothesis decision "
            "differs across analyses. Report all results, effect sizes, and assumptions, and "
            "consider which assumptions are most appropriate."
        )
    decision = decisions[0]
    agreement = (
        f"All comparable analyses reject the null hypothesis{alpha_text}."
        if decision == "reject null"
        else f"All comparable analyses fail to reject the null hypothesis{alpha_text}."
    )
    if mixed_estimands:
        return (
            "CONSISTENT across methods (note: not all test the same estimand). "
            f"{agreement} Direct numerical comparison across different estimands is not "
            "appropriate. This descriptive agreement does not by itself establish robustness."
        )
    return (
        f"ROBUST: All comparable methods agree on the hypothesis decision. {agreement} The "
        "conclusion is consistent across methods. This descriptive agreement does not by itself "
        "establish general robustness, equivalence, or practical importance."
    )


def effect_narrative(
    measure: str,
    value: float | None,
    n: int | None = None,
    ci: tuple[float | None, float | None] | Mapping[str, Any] | None = None,
    *,
    confidence_level: float | None = None,
    orientation: str | None = None,
    definition: str | None = None,
) -> str:
    """Return deterministic prose for an already-computed effect-size record.

    ``measure`` accepts PyAutoStat's public quantity identifiers (for example,
    ``"cohens_d"``) and the display names stored in analysis results. ``value``
    and any confidence-interval bounds must already have been computed by the
    statistical layer. Invalid, unavailable, and unsupported inputs produce an
    explanatory string rather than an invented magnitude or an exception.
    """
    canonical = _canonical_measure(measure)
    if canonical is None:
        shown = f" {measure!r}" if isinstance(measure, str) and measure else ""
        return f"Effect narrative is unavailable because effect measure{shown} is unsupported."
    rule = _RULES[canonical]
    estimate = _finite(value)
    if estimate is None:
        return f"{rule.display_name} is unavailable; no effect magnitude was narrated."
    if (
        (rule.nonnegative and estimate < 0)
        or (canonical in {"rank_biserial", "pearson_r"} and abs(estimate) > 1)
        or (canonical in {"eta_squared", "cramers_v"} and estimate > 1)
    ):
        return f"{rule.display_name} is outside its valid range; no effect magnitude was narrated."

    label = _magnitude_label(rule, estimate)
    if label is None:
        opening = (
            f"{rule.display_name} = {_fmt(estimate)}. PyAutoStat does not assign a conventional "
            "magnitude label to this measure."
        )
    else:
        opening = f"{rule.display_name} = {_fmt(estimate)}, conventionally labelled '{label}'."

    details: list[str] = [opening]
    if canonical in {"cohens_d", "cohens_dz"} and label is not None:
        consequence = _D_CONSEQUENCES[label]
        if canonical == "cohens_dz":
            consequence = consequence.replace("groups", "paired conditions")
        details.append(consequence)
    else:
        details.append(f"This measure describes {rule.concept}.")

    if definition is not None and definition.strip():
        details.append(definition.strip().rstrip(".") + ".")
    if orientation is not None and orientation.strip() and rule.signed:
        details.append(f"Its sign follows {orientation.strip()}.")
    if canonical == "rank_biserial":
        details.append(
            "Its sign describes the observed rank ordering; it is not a median difference."
        )
    elif canonical == "pearson_r":
        direction = "positive" if estimate > 0 else "negative" if estimate < 0 else "zero"
        details.append(
            f"The observed linear association is {direction}; a zero value does not establish "
            "population independence."
        )
    elif canonical == "cramers_v":
        details.append("The measure is nonnegative and has no direction.")

    if isinstance(n, int) and not isinstance(n, bool) and n > 0:
        details.append(f"The recorded effect sample size is n = {n}.")
    interval_text = _confidence_interval_narrative(estimate, ci, confidence_level)
    if interval_text is not None:
        details.append(interval_text)
    return " ".join(details)


__all__ = [
    "ASSUMPTION_SEVERITY_DESCRIPTIONS",
    "assumption_grade",
    "effect_narrative",
    "hypothesis_verdict",
    "interval_verdict",
    "sensitivity_verdict",
]
