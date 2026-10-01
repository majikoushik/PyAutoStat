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
    "paired_rank_biserial": _EffectRule(
        "Matched-pairs rank-biserial correlation",
        "signed paired-rank separation",
        None,
        signed=True,
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
    "spearman_rho": _EffectRule(
        "Spearman rho", "monotonic rank-association strength", _CORRELATION_BANDS, signed=True
    ),
    "point_biserial_r": _EffectRule(
        "Point-biserial r", "binary-continuous association", _CORRELATION_BANDS, signed=True
    ),
    "partial_pearson_r": _EffectRule(
        "Partial Pearson r", "linearly adjusted association", _CORRELATION_BANDS, signed=True
    ),
    "kendall_tau_b": _EffectRule(
        "Kendall's tau-b", "pairwise ordinal concordance adjusted for ties", None, signed=True
    ),
    "paired_proportion_difference": _EffectRule(
        "Paired proportion difference",
        "first-minus-second paired event probability",
        None,
        signed=True,
    ),
    "odds_ratio": _EffectRule("Sample odds ratio", "2x2 odds association", None),
    "partial_eta_squared": _EffectRule(
        "Partial eta-squared",
        "term variance association relative to term plus residual variation",
        _ETA_SQUARED_BANDS,
        nonnegative=True,
    ),
    "kendalls_w": _EffectRule(
        "Kendall's W",
        "repeated-rank concordance across conditions",
        None,
        nonnegative=True,
    ),
}

_MEASURE_ALIASES = {
    "cohens_d": "cohens_d",
    "cohen's d": "cohens_d",
    "one-sample cohen's d": "cohens_d",
    "cohens_dz": "cohens_dz",
    "cohen's dz": "cohens_dz",
    "rank_biserial": "rank_biserial",
    "rank-biserial correlation": "rank_biserial",
    "paired_rank_biserial": "paired_rank_biserial",
    "matched-pairs rank-biserial correlation": "paired_rank_biserial",
    "eta_squared": "eta_squared",
    "eta-squared": "eta_squared",
    "partial_eta_squared": "partial_eta_squared",
    "partial eta-squared": "partial_eta_squared",
    "kendalls_w": "kendalls_w",
    "kendall's w": "kendalls_w",
    "Kendall's W": "kendalls_w",
    "epsilon_squared": "epsilon_squared",
    "epsilon-squared": "epsilon_squared",
    "epsilon-squared (rank)": "epsilon_squared",
    "cramers_v": "cramers_v",
    "cramer's v": "cramers_v",
    "pearson_r": "pearson_r",
    "pearson r": "pearson_r",
    "spearman_rho": "spearman_rho",
    "spearman rho": "spearman_rho",
    "point_biserial_r": "point_biserial_r",
    "point-biserial r": "point_biserial_r",
    "partial_pearson_r": "partial_pearson_r",
    "partial pearson r": "partial_pearson_r",
    "kendall_tau_b": "kendall_tau_b",
    "kendall's tau-b": "kendall_tau_b",
    "paired_proportion_difference": "paired_proportion_difference",
    "paired proportion difference": "paired_proportion_difference",
    "odds_ratio": "odds_ratio",
    "sample odds ratio": "odds_ratio",
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
        if method_id == "wilcoxon_signed_rank":
            message = (
                "The Wilcoxon signed-rank test models ordered within-unit differences; complete "
                "pairs must be independent across units and suitable for signed-rank inference."
            )
        else:
            message = (
                "The paired t-test models within-unit differences; complete pairs must be "
                "independent across units and suitable for mean inference."
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
        or (
            canonical
            in {
                "rank_biserial",
                "paired_rank_biserial",
                "pearson_r",
                "spearman_rho",
                "point_biserial_r",
                "partial_pearson_r",
                "kendall_tau_b",
                "paired_proportion_difference",
            }
            and abs(estimate) > 1
        )
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
    elif canonical == "paired_rank_biserial":
        details.append(
            "Its sign follows the declared paired contrast; it is not universally a median "
            "difference."
        )
    elif canonical in {"pearson_r", "spearman_rho", "point_biserial_r", "partial_pearson_r"}:
        direction = "positive" if estimate > 0 else "negative" if estimate < 0 else "zero"
        relationship = {
            "pearson_r": "linear",
            "spearman_rho": "monotonic rank-order",
            "point_biserial_r": "binary-continuous",
            "partial_pearson_r": "linearly adjusted",
        }[canonical]
        details.append(
            f"The observed {relationship} association is {direction}; a zero value does not "
            "establish population independence."
        )
        if canonical == "point_biserial_r":
            details.append(
                "Equivalently, it is standardized separation in the continuous variable across "
                "the explicitly coded binary groups."
            )
        if canonical == "partial_pearson_r":
            details.append("The adjustment does not establish that confounding has been removed.")
    elif canonical == "kendall_tau_b":
        direction = "positive" if estimate > 0 else "negative" if estimate < 0 else "zero"
        details.append(
            f"The observed pairwise ordinal concordance is {direction}; tau-b is not percent "
            "variance explained."
        )
    elif canonical == "paired_proportion_difference":
        details.append(
            "Its sign follows the declared first-condition minus second-condition order; no "
            "universal magnitude threshold is applied."
        )
    elif canonical == "cramers_v":
        details.append("The measure is nonnegative and has no direction.")
    elif canonical == "odds_ratio":
        details.append(
            "Values above 1 and below 1 depend on the recorded row and column level order; "
            "the measure does not establish causation."
        )

    if isinstance(n, int) and not isinstance(n, bool) and n > 0:
        details.append(f"The recorded effect sample size is n = {n}.")
    interval_text = _confidence_interval_narrative(estimate, ci, confidence_level)
    if interval_text is not None:
        details.append(interval_text)
    return " ".join(details)


_SIZE_COMMENTS = (
    (
        5000,
        "This is a large dataset; many standard methods can have high power when their design "
        "and effect-size assumptions are appropriate.",
    ),
    (
        500,
        "This is a medium-sized dataset; standard methods are often appropriate when their "
        "assumptions match the study design.",
    ),
    (100, "This is a moderate sample; use care with methods that require large groups."),
    (30, "This is a small sample; method assumptions and uncertainty need particular attention."),
    (0, "This is a very small dataset; most inferential methods will be unreliable."),
)

_STORY_CORRELATION_THRESHOLD = 0.70
_COLUMN_SKEW_THRESHOLDS = (0.5, 1.0, 2.0)
_MEAN_MEDIAN_SMALL_STANDARDIZED_GAP = 0.25
_INSIGHT_EXAMPLE_LIMIT = 3
_ACTION_LIMIT = 5

_INSIGHT_CONSEQUENCES = {
    "Multicollinearity": {
        "regression": (
            "Highly correlated predictors can inflate standard errors and make individual "
            "regression coefficients unstable."
        ),
        "exploration": (
            "These variables carry overlapping linear information, so retaining all of them "
            "may obscure the main patterns."
        ),
        "general": (
            "Strong association can make it difficult to distinguish the variables' separate "
            "contributions in a joint analysis."
        ),
    },
    "Missing Data": {
        "general": (
            "Missing observations can change the available sample and should be handled through "
            "an explicit, documented analysis policy."
        )
    },
    "Outliers": {
        "general": (
            "Flagged values may affect summaries and model estimates, but their legitimacy "
            "cannot be determined from the values alone."
        )
    },
    "Normality": {
        "general": (
            "A rejected diagnostic qualifies methods that rely on distributional assumptions; "
            "it does not by itself select a different estimand or test."
        )
    },
    "Distribution Shape": {
        "general": (
            "Distribution shape can affect summaries and method assumptions and should be "
            "considered alongside the sample size and research target."
        )
    },
    "Data Quality": {
        "general": (
            "Recorded quality issues should be reviewed before inferential analysis; the profile "
            "does not alter or remove source rows."
        )
    },
}

_INSIGHT_ACTIONS = {
    "Multicollinearity": (
        "Review each strongly correlated pair and choose variables according to the research "
        "question before fitting a joint model."
    ),
    "Missing Data": "Define and document a missing-data policy before analysis.",
    "Outliers": "Verify flagged values and document any exclusion decision.",
    "Normality": (
        "Inspect the recorded diagnostics and choose a method that preserves the estimand."
    ),
    "Distribution Shape": "Review distribution shape when selecting summaries and methods.",
    "Data Quality": "Resolve or document the recorded quality issues before analysis.",
}

_INSIGHT_SEVERITY_PRIORITY = {"high": 0, "medium": 1, "low": 2}
_INSIGHT_CATEGORY_PRIORITY = {
    "Data Quality": 0,
    "Missing Data": 1,
    "Outliers": 2,
    "Multicollinearity": 3,
    "Normality": 4,
    "Distribution Shape": 5,
}


def _profile_overview(profile: Mapping[str, Any]) -> tuple[int | None, int | None]:
    overview = profile.get("overview")
    if not isinstance(overview, Mapping):
        return None, None
    rows = overview.get("total_rows")
    columns = overview.get("total_columns")
    safe_rows = rows if isinstance(rows, int) and not isinstance(rows, bool) and rows >= 0 else None
    safe_columns = (
        columns
        if isinstance(columns, int) and not isinstance(columns, bool) and columns >= 0
        else None
    )
    return safe_rows, safe_columns


def dataset_opening(profile: Mapping[str, Any]) -> str:
    """Narrate the recorded dataset dimensions without recalculating the profile."""
    if not isinstance(profile, Mapping):
        return "Dataset dimensions are unavailable from the recorded profile."
    rows, columns = _profile_overview(profile)
    if rows is None or columns is None:
        return "Dataset dimensions are unavailable from the recorded profile."
    size_comment = next(comment for minimum, comment in _SIZE_COMMENTS if rows >= minimum)
    return (
        f"The dataset contains {rows:,} row{'s' if rows != 1 else ''} across {columns:,} "
        f"column{'s' if columns != 1 else ''}. {size_comment}"
    )


def _top_correlation_pair(profile: Mapping[str, Any]) -> tuple[str, str, float] | None:
    correlation = profile.get("correlation")
    if not isinstance(correlation, Mapping):
        return None
    pearson = correlation.get("pearson")
    matrix = pearson.get("matrix") if isinstance(pearson, Mapping) else None
    if not isinstance(matrix, Mapping):
        return None
    candidates: dict[tuple[str, str], float] = {}
    for first, row in matrix.items():
        if not isinstance(first, str) or not isinstance(row, Mapping):
            continue
        for second, raw_value in row.items():
            if not isinstance(second, str) or first == second:
                continue
            value = _finite(raw_value)
            if value is None or abs(value) > 1:
                continue
            first_name, second_name = sorted((first, second))
            pair = (first_name, second_name)
            previous = candidates.get(pair)
            if previous is None or abs(value) > abs(previous):
                candidates[pair] = value
    eligible = [
        (first, second, value)
        for (first, second), value in candidates.items()
        if abs(value) >= _STORY_CORRELATION_THRESHOLD
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda item: (-abs(item[2]), item[0], item[1]))


def _correlation_story(profile: Mapping[str, Any]) -> str:
    pair = _top_correlation_pair(profile)
    if pair is None:
        return (
            "No eligible Pearson pair reached the recorded strong-correlation threshold "
            f"(|r| >= {_STORY_CORRELATION_THRESHOLD:.2f})."
        )
    first, second, value = pair
    direction = "positive" if value > 0 else "negative"
    return (
        f"'{first}' and '{second}' have the strongest eligible linear association "
        f"(r = {_fmt(value)}, {direction}). Review whether both variables provide distinct "
        "information for the intended analysis. Correlation does not establish causation."
    )


def _missing_columns(profile: Mapping[str, Any]) -> list[tuple[str, int, float]]:
    missing = profile.get("missing_data")
    by_column = missing.get("by_column") if isinstance(missing, Mapping) else None
    if not isinstance(by_column, Mapping):
        return []
    entries: list[tuple[str, int, float, int]] = []
    for position, (column, raw) in enumerate(by_column.items()):
        if not isinstance(column, str) or not isinstance(raw, Mapping):
            continue
        count = raw.get("count")
        percentage = _finite(raw.get("percentage"))
        if (
            isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
            and percentage is not None
            and percentage >= 0
        ):
            entries.append((column, count, percentage, position))
    ranked = sorted(entries, key=lambda item: (-item[2], item[3]))
    return [(column, count, percentage) for column, count, percentage, _ in ranked]


def _quality_issues(profile: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    quality = profile.get("data_quality")
    raw_issues = quality.get("issues") if isinstance(quality, Mapping) else None
    if not isinstance(raw_issues, Sequence) or isinstance(raw_issues, str):
        return []
    priority = {"high": 0, "review": 1, "info": 2}
    issues = [item for item in raw_issues if isinstance(item, Mapping)]
    return [
        item
        for _, item in sorted(
            enumerate(issues),
            key=lambda pair: (priority.get(str(pair[1].get("severity", "")), 3), pair[0]),
        )
    ]


def _data_quality_story(profile: Mapping[str, Any]) -> str:
    missing = profile.get("missing_data")
    quality = profile.get("data_quality")
    if not isinstance(missing, Mapping) and not isinstance(quality, Mapping):
        return "Data-quality metadata are unavailable in the recorded profile."
    completeness = _finite(quality.get("completeness")) if isinstance(quality, Mapping) else None
    duplicate_count = quality.get("duplicate_rows") if isinstance(quality, Mapping) else None
    duplicate_pct = (
        _finite(quality.get("duplicate_rows_percentage")) if isinstance(quality, Mapping) else None
    )
    missing_columns = _missing_columns(profile)
    parts: list[str] = []
    if completeness is not None and 0 <= completeness <= 1:
        parts.append(f"The profile is {_fmt(completeness * 100)}% complete.")
    if not missing_columns:
        total_missing = missing.get("total_missing_cells") if isinstance(missing, Mapping) else None
        if total_missing == 0:
            parts.append("No missing values were recorded.")
        elif isinstance(missing, Mapping):
            parts.append("No column-level missingness issue was recorded.")
    elif len(missing_columns) == 1:
        column, count, percentage = missing_columns[0]
        parts.append(
            f"Missingness is concentrated in '{column}': {count:,} value(s), "
            f"or {_fmt(percentage)}% of its rows, are missing."
        )
    else:
        shown = "; ".join(
            f"'{column}' ({count:,}, {_fmt(percentage)}%)"
            for column, count, percentage in missing_columns[:3]
        )
        extra = f"; plus {len(missing_columns) - 3} more" if len(missing_columns) > 3 else ""
        parts.append(f"Multiple columns contain missing values: {shown}{extra}.")
    if (
        isinstance(duplicate_count, int)
        and not isinstance(duplicate_count, bool)
        and duplicate_count > 0
    ):
        pct_text = f" ({_fmt(duplicate_pct)}%)" if duplicate_pct is not None else ""
        parts.append(
            f"The profile recorded {duplicate_count:,} duplicate row(s){pct_text}; confirm "
            "whether they are genuine duplicates before changing the data."
        )
    elif duplicate_count == 0:
        parts.append("No duplicate rows were recorded.")
    already_narrated = {"missing_values", "all_missing", "exact_duplicate_rows"}
    additional = next(
        (
            issue
            for issue in _quality_issues(profile)
            if issue.get("code") not in already_narrated
            and issue.get("section") in {"data_dictionary", "data_quality", "variable_intelligence"}
        ),
        None,
    )
    if additional is not None:
        message = additional.get("message")
        if isinstance(message, str) and message.strip():
            parts.append(f"Additional recorded issue: {message.strip()}")
    return " ".join(parts) if parts else "No major recorded data-quality issue was identified."


def _distribution_counts(profile: Mapping[str, Any]) -> tuple[int, int, int]:
    descriptive = profile.get("descriptive")
    normality = profile.get("normality")
    columns = list(descriptive) if isinstance(descriptive, Mapping) else []
    if not columns:
        overview = profile.get("overview")
        numeric_count = (
            overview.get("analytical_numeric_columns") if isinstance(overview, Mapping) else None
        )
        count = numeric_count if isinstance(numeric_count, int) and numeric_count >= 0 else 0
        return count, 0, count
    rejected = 0
    unavailable = 0
    for column in columns:
        tests = normality.get(column) if isinstance(normality, Mapping) else None
        statuses = (
            [
                item.get("status")
                for item in tests.values()
                if isinstance(item, Mapping) and isinstance(item.get("status"), str)
            ]
            if isinstance(tests, Mapping)
            else []
        )
        if any(status == "rejected" for status in statuses):
            rejected += 1
        elif not statuses:
            unavailable += 1
    return len(columns), rejected, unavailable


def _distribution_story(profile: Mapping[str, Any]) -> str:
    total, rejected, unavailable = _distribution_counts(profile)
    if total == 0:
        return "No analytical numeric variables are available for distribution diagnostics."
    available = total - unavailable
    if available == 0:
        opening = f"Normality diagnostics are unavailable for all {total} numeric variable(s)."
    elif rejected == 0:
        opening = (
            f"The recorded diagnostics did not reject normality for any of the {available} "
            "evaluated numeric variable(s); non-rejection does not prove normality."
        )
    elif rejected == available:
        opening = (
            f"At least one recorded normality diagnostic rejected normality for all {available} "
            "evaluated numeric variable(s)."
        )
    elif rejected > available / 2:
        opening = (
            f"At least one recorded normality diagnostic rejected normality for most evaluated "
            f"numeric variables ({rejected} of {available})."
        )
    else:
        opening = (
            f"At least one recorded normality diagnostic rejected normality for some evaluated "
            f"numeric variables ({rejected} of {available})."
        )
    if unavailable:
        opening += f" Diagnostics were unavailable for {unavailable} additional variable(s)."
    return opening + " These diagnostics qualify method assumptions but do not select an estimand."


def _prioritised_actions(profile: Mapping[str, Any], *, limit: int = 3) -> tuple[str, ...]:
    candidates: list[tuple[int, int, str]] = []
    order = 0
    quality_issue = next(
        (
            issue
            for issue in _quality_issues(profile)
            if issue.get("severity") in {"high", "review"}
            and issue.get("code") not in {"missing_values", "all_missing", "exact_duplicate_rows"}
            and issue.get("section") in {"data_dictionary", "data_quality", "variable_intelligence"}
            and isinstance(issue.get("recommendation"), str)
        ),
        None,
    )
    if quality_issue is not None:
        quality_priority = -1 if quality_issue.get("severity") == "high" else 1
        candidates.append((quality_priority, order, quality_issue["recommendation"].strip()))
        order += 1
    missing_columns = _missing_columns(profile)
    if missing_columns:
        severe = [entry for entry in missing_columns if entry[2] >= 20]
        selected = severe or missing_columns
        names = ", ".join(f"'{column}'" for column, _, _ in selected[:3])
        priority = 0 if severe else 2
        candidates.append(
            (
                priority,
                order,
                f"Define and document a missing-data policy for {names} before analysis.",
            )
        )
        order += 1
    outliers = profile.get("outliers")
    flagged_columns: list[str] = []
    if isinstance(outliers, Mapping):
        for column, methods in outliers.items():
            iqr = methods.get("iqr") if isinstance(methods, Mapping) else None
            percentage = _finite(iqr.get("percentage")) if isinstance(iqr, Mapping) else None
            if isinstance(column, str) and percentage is not None and percentage > 5:
                flagged_columns.append(column)
    if flagged_columns:
        names = ", ".join(f"'{column}'" for column in flagged_columns[:3])
        candidates.append(
            (
                2,
                order,
                f"Review IQR-flagged observations in {names}; do not remove values without a "
                "documented reason.",
            )
        )
        order += 1
    quality = profile.get("data_quality")
    duplicate_count = quality.get("duplicate_rows") if isinstance(quality, Mapping) else None
    if isinstance(duplicate_count, int) and duplicate_count > 0:
        candidates.append(
            (
                1,
                order,
                f"Verify the {duplicate_count:,} recorded duplicate row(s) before deciding "
                "whether any should be removed.",
            )
        )
        order += 1
    pair = _top_correlation_pair(profile)
    if pair is not None:
        candidates.append(
            (
                3,
                order,
                f"Review whether '{pair[0]}' and '{pair[1]}' both provide distinct information "
                "for the intended analysis.",
            )
        )
        order += 1
    total, rejected, unavailable = _distribution_counts(profile)
    if rejected:
        candidates.append(
            (
                4,
                order,
                f"Review the rejected normality diagnostics for {rejected} numeric variable(s) "
                "when choosing methods.",
            )
        )
        order += 1
    if total and unavailable == total:
        candidates.append(
            (5, order, "Record that normality diagnostics were unavailable before inferential use.")
        )
    actions: list[str] = []
    for _, _, action in sorted(candidates, key=lambda item: (item[0], item[1])):
        if action not in actions:
            actions.append(action)
        if len(actions) >= limit:
            break
    return tuple(actions)


def _dataset_story(profile: Mapping[str, Any]) -> str:
    """Assemble an opt-in dataset story from one recorded profile."""
    if not isinstance(profile, Mapping):
        return "DATASET STORY\n=============\n\nThe recorded profile is unavailable."
    actions = _prioritised_actions(profile)
    lines = [
        "DATASET STORY",
        "=============",
        "",
        dataset_opening(profile),
        "",
        "KEY FINDING",
        _correlation_story(profile),
        "",
        "DATA QUALITY",
        _data_quality_story(profile),
        "",
        "DISTRIBUTION",
        _distribution_story(profile),
        "",
        "CATEGORICAL DISTRIBUTIONS",
        _categorical_distribution_story(profile),
        "",
        "RECOMMENDED FIRST STEPS",
    ]
    if actions:
        lines.extend(f"  {index}. {action}" for index, action in enumerate(actions, start=1))
    else:
        lines.append("  No profile-driven corrective action was identified.")
    return "\n".join(lines)


def _skewness_story(skewness: float | None) -> str:
    value = _finite(skewness)
    if value is None:
        return "Skewness is unavailable."
    magnitude = abs(value)
    if magnitude < _COLUMN_SKEW_THRESHOLDS[0]:
        label = "approximately symmetric"
    elif magnitude < _COLUMN_SKEW_THRESHOLDS[1]:
        label = f"mild {'right' if value > 0 else 'left'} skew"
    elif magnitude < _COLUMN_SKEW_THRESHOLDS[2]:
        label = f"moderate {'right' if value > 0 else 'left'} skew"
    else:
        label = f"strong {'right' if value > 0 else 'left'} skew"
    return f"The distribution shows {label} (skewness = {_fmt(value)})."


def _mean_median_story(stats: Mapping[str, Any]) -> str:
    mean = _finite(stats.get("mean"))
    median = _finite(stats.get("median"))
    if mean is None or median is None:
        return "The mean-to-median relationship is unavailable."
    if math.isclose(mean, median, rel_tol=1e-9, abs_tol=1e-12):
        return "The mean and median are effectively equal in the recorded statistics."
    minimum = _finite(stats.get("min"))
    maximum = _finite(stats.get("max"))
    recorded_range = maximum - minimum if maximum is not None and minimum is not None else None
    spread = next(
        (
            value
            for value in (
                _finite(stats.get("std")),
                _finite(stats.get("iqr")),
                recorded_range,
            )
            if value is not None and value > 0
        ),
        None,
    )
    if spread is None:
        return (
            "The mean and median differ, but no finite positive spread is available to scale "
            "the difference."
        )
    standardized_gap = abs(mean - median) / spread
    if standardized_gap <= _MEAN_MEDIAN_SMALL_STANDARDIZED_GAP:
        return (
            "The mean and median differ slightly relative to the recorded spread "
            f"(scaled gap = {_fmt(standardized_gap)})."
        )
    return (
        "The mean and median differ materially relative to the recorded spread "
        f"(scaled gap = {_fmt(standardized_gap)}), which is consistent with an asymmetric center."
    )


def percentile_narrative(stats: Mapping[str, Any]) -> str:
    """Narrate stored percentiles without recalculating them."""
    if not isinstance(stats, Mapping) or not isinstance(stats.get("percentiles"), Mapping):
        return "Percentile summary is unavailable."
    percentiles = stats["percentiles"]
    parts = []
    median = _finite(percentiles.get("p50"))
    lower = _finite(percentiles.get("p25"))
    upper = _finite(percentiles.get("p75"))
    p05 = _finite(percentiles.get("p05"))
    p95 = _finite(percentiles.get("p95"))
    if median is not None:
        parts.append(f"The 50th percentile (median) is {_fmt(median)}.")
    if lower is not None and upper is not None:
        parts.append(
            "The middle 50% of recorded values lies approximately between "
            f"{_fmt(lower)} and {_fmt(upper)}."
        )
    if p05 is not None:
        parts.append(f"The 5th percentile is {_fmt(p05)}.")
    if p95 is not None:
        parts.append(f"The 95th percentile is {_fmt(p95)}.")
    return " ".join(parts) if parts else "Requested percentiles are unavailable."


def coefficient_of_variation_narrative(stats: Mapping[str, Any]) -> str:
    """Narrate a stored, safeguarded coefficient of variation."""
    detail = stats.get("coefficient_of_variation_details") if isinstance(stats, Mapping) else None
    if not isinstance(detail, Mapping):
        return "Coefficient of variation details are unavailable."
    caveat = detail.get("applicability")
    if detail.get("status") == "available" and _finite(detail.get("percent")) is not None:
        text = (
            "The sample standard deviation is "
            f"{_fmt(float(detail['percent']))}% of the absolute mean."
        )
    else:
        reason = str(detail.get("reason") or "the calculation is not stable")
        text = f"The coefficient of variation is unavailable ({reason.replace('_', ' ')})."
    if isinstance(caveat, str) and caveat:
        text += " " + caveat
    return text


def frequency_narrative(result: Mapping[str, Any], *, limit: int = 10) -> str:
    """Narrate a structured categorical frequency table with bounded detail."""
    if not isinstance(result, Mapping) or not isinstance(result.get("levels"), Sequence):
        return "Frequency narrative unavailable."
    rows = [row for row in result["levels"] if isinstance(row, Mapping)]
    column = str(result.get("column") or "Category")
    valid_n = result.get("valid_n")
    total_n = result.get("total_n")
    missing_n = result.get("missing_n")
    lines = [f"{column.upper()} DISTRIBUTION"]
    for row in rows[:limit]:
        percent = _finite(row.get("percent"))
        shown = f"{percent:.1f}%" if percent is not None else "percentage unavailable"
        lines.append(f"{row.get('level')}: {row.get('count')} ({shown})")
    if len(rows) > limit:
        lines.append(f"{len(rows) - limit} additional level(s) are retained in the table.")
    if rows:
        maximum = max(int(row.get("count", 0)) for row in rows)
        modes = [row for row in rows if row.get("count") == maximum]
        if len(modes) == 1:
            lines.append(
                f"{modes[0].get('level')} is the most common observed category, representing "
                f"{float(modes[0].get('percent', 0)):.1f}% of {valid_n:,} valid observations."
            )
        else:
            labels = ", ".join(str(row.get("level")) for row in modes[:limit])
            suffix = f" and {len(modes) - limit} more" if len(modes) > limit else ""
            lines.append(f"The largest count is tied across {labels}{suffix} ({maximum} each).")
    if result.get("cumulative_percentage_status") == "omitted_unordered":
        lines.append(
            "Cumulative percentages are omitted because no complete ordinal order was supplied."
        )
    if isinstance(total_n, int) and isinstance(missing_n, int):
        missing_percent = missing_n / total_n * 100 if total_n else 0.0
        lines.append(f"Missing values: {missing_n:,} of {total_n:,} rows ({missing_percent:.1f}%).")
    return "\n".join(lines)


def crosstab_narrative(result: Mapping[str, Any]) -> str:
    """Narrate selected observed cross-tab cells without inferential language."""
    if not isinstance(result, Mapping):
        return "Cross-tabulation narrative unavailable."
    rows = result.get("row_levels")
    columns = result.get("column_levels")
    percentages = result.get("row_percent")
    if not isinstance(rows, Sequence) or not isinstance(columns, Sequence):
        return "Cross-tabulation narrative unavailable."
    candidates = []
    if isinstance(percentages, Sequence):
        for row_index, values in enumerate(percentages):
            if not isinstance(values, Sequence):
                continue
            for column_index, value in enumerate(values):
                number = _finite(value)
                if number is not None:
                    candidates.append((number, row_index, column_index))
    if candidates:
        value, row_index, column_index = max(
            candidates, key=lambda item: (item[0], -item[1], -item[2])
        )
        lead = (
            f"Within observed {result.get('row_variable')}={rows[row_index]}, "
            f"{result.get('column_variable')}={columns[column_index]} accounts for "
            f"{value:.1f}% of valid rows in that row category."
        )
    else:
        lead = "No finite row percentage is available to summarize."
    size_note = (
        f" The full {len(rows)} by {len(columns)} table is retained in the structured result."
        if result.get("large_table")
        else ""
    )
    excluded = result.get("excluded_rows")
    exclusion_note = (
        f" {excluded:,} row(s) with a missing value in either variable were excluded."
        if isinstance(excluded, int) and excluded
        else " No rows were excluded for missing values in the selected pair."
    )
    return (
        lead + size_note + exclusion_note + (" The table describes the observed distribution only.")
    )


def _categorical_distribution_story(profile: Mapping[str, Any]) -> str:
    summaries = profile.get("categorical_summary")
    intelligence = profile.get("variable_intelligence")
    if not isinstance(summaries, Mapping):
        return "No categorical distribution summary is available."
    findings = []
    for column, summary in summaries.items():
        info = intelligence.get(column, {}) if isinstance(intelligence, Mapping) else {}
        if isinstance(info, Mapping) and (
            info.get("suggested_role") == "identifier" or info.get("suggested_type") == "identifier"
        ):
            continue
        frequencies = summary.get("frequencies") if isinstance(summary, Mapping) else None
        if isinstance(frequencies, Sequence) and frequencies:
            top = frequencies[0]
            findings.append(
                f"For {column!r}, {top.get('value')!r} is the most frequent recorded level "
                f"({float(top.get('percentage', 0)):.1f}% of non-missing observations)."
            )
        if len(findings) == 2:
            break
    return " ".join(findings) if findings else "No observed categorical levels were available."


def column_story(
    column_name: str,
    stats: Mapping[str, Any],
    unit: str | None = None,
) -> str:
    """Narrate an existing numeric descriptive-statistics record."""
    if not isinstance(column_name, str) or not column_name:
        return "Column narrative unavailable: a column name is required."
    if not isinstance(stats, Mapping):
        return f"{column_name}: descriptive statistics are unavailable."
    count = stats.get("count")
    safe_count = (
        count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else None
    )
    values = {
        key: _finite(stats.get(key)) for key in ("mean", "median", "std", "min", "max", "skewness")
    }
    if safe_count == 0 or all(value is None for value in values.values()):
        return f"{column_name}: no finite non-missing numerical values are available to narrate."
    suffix = f" {unit.strip()}" if isinstance(unit, str) and unit.strip() else ""
    mean_text = (
        f"Mean = {_fmt(values['mean'])}{suffix}"
        if values["mean"] is not None
        else "Mean unavailable"
    )
    median_text = (
        f"median = {_fmt(values['median'])}{suffix}"
        if values["median"] is not None
        else "median unavailable"
    )
    if values["std"] == 0 or (
        values["min"] is not None and values["max"] is not None and values["min"] == values["max"]
    ):
        spread_text = "The recorded values are constant; standard deviation is 0."
    else:
        sd_text = (
            f"SD = {_fmt(values['std'])}{suffix}"
            if values["std"] is not None
            else "SD is unavailable"
        )
        range_text = (
            f" Values range from {_fmt(values['min'])}{suffix} to {_fmt(values['max'])}{suffix}."
            if values["min"] is not None and values["max"] is not None
            else " The recorded range is unavailable."
        )
        spread_text = sd_text + "." + range_text
    count_text = f" Based on {safe_count:,} non-missing value(s)." if safe_count is not None else ""
    return "\n".join(
        [
            f"{column_name}:",
            f"Central tendency: {mean_text}; {median_text}.{count_text}",
            f"Spread: {spread_text}",
            f"Percentiles: {percentile_narrative(stats)}",
            f"Relative spread: {coefficient_of_variation_narrative(stats)}",
            f"Shape: {_skewness_story(values['skewness'])}",
            f"Note: {_mean_median_story(stats)}",
        ]
    )


def _detail_note(category: str, detail: Mapping[str, Any]) -> str | None:
    if category == "Multicollinearity":
        pair = detail.get("pair")
        value = _finite(detail.get("correlation"))
        if (
            isinstance(pair, Sequence)
            and not isinstance(pair, str)
            and len(pair) == 2
            and all(isinstance(item, str) for item in pair)
            and value is not None
        ):
            return f"'{pair[0]}' and '{pair[1]}' (r = {_fmt(value)})"
    note = detail.get("note")
    if isinstance(note, str) and note.strip():
        return note.strip().rstrip(".")
    column = detail.get("column")
    if isinstance(column, str):
        return f"'{column}'"
    return None


def insight_narrative(
    category: str,
    findings: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> str:
    """Assemble connected prose from existing structured insight findings."""
    details = [findings] if isinstance(findings, Mapping) else list(findings)
    clean_details = [item for item in details if isinstance(item, Mapping)]
    context_map = context if isinstance(context, Mapping) else {}
    severity = context_map.get("severity")
    severity_text = severity.upper() if isinstance(severity, str) else "INFO"
    objective = context_map.get("objective")
    objective_key = objective if isinstance(objective, str) else "general"
    if category == "Multicollinearity":
        clean_details.sort(
            key=lambda item: (
                -abs(_finite(item.get("correlation")) or 0),
                tuple(item.get("pair", ())),
            )
        )
    header = f"[{severity_text}] {category.upper()} - {len(clean_details)} finding(s)"
    notes = [note for item in clean_details if (note := _detail_note(category, item)) is not None]
    paragraphs: list[str] = [header]
    if notes:
        paragraphs.append(f"Lead finding: {notes[0]}.")
        if len(notes) > 1:
            examples = "; ".join(notes[1:_INSIGHT_EXAMPLE_LIMIT])
            remaining = len(notes) - 1
            extra = f" Examples: {examples}." if examples else ""
            paragraphs.append(f"Other findings ({remaining} more).{extra}")
    else:
        finding = context_map.get("finding")
        if isinstance(finding, str) and finding.strip():
            paragraphs.append(finding.strip())
        else:
            paragraphs.append("No detailed finding text is available.")
    consequence_map = _INSIGHT_CONSEQUENCES.get(category, {})
    consequence = consequence_map.get(objective_key) or consequence_map.get("general")
    if consequence:
        paragraphs.append(consequence)
    recommendations = context_map.get("recommendation")
    action = (
        next(
            (item.strip() for item in recommendations if isinstance(item, str) and item.strip()),
            None,
        )
        if isinstance(recommendations, Sequence) and not isinstance(recommendations, str)
        else None
    )
    action = action or _INSIGHT_ACTIONS.get(category)
    if action:
        paragraphs.append(f"Recommended action: {action}")
    return "\n\n".join(paragraphs)


def _ranked_insight_actions(
    insights: Sequence[Mapping[str, Any]], *, limit: int = _ACTION_LIMIT
) -> tuple[str, ...]:
    """Rank existing insight recommendations by explicit severity/category rules."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        return ()
    ranked = sorted(
        enumerate(insights),
        key=lambda item: (
            _INSIGHT_SEVERITY_PRIORITY.get(str(item[1].get("severity", "")).lower(), 3),
            _INSIGHT_CATEGORY_PRIORITY.get(str(item[1].get("category", "")), 99),
            item[0],
        ),
    )
    actions: list[str] = []
    for _, insight in ranked:
        recommendations = insight.get("recommendation")
        if not isinstance(recommendations, Sequence) or isinstance(recommendations, str):
            continue
        for raw_action in recommendations:
            if not isinstance(raw_action, str) or not raw_action.strip():
                continue
            action = raw_action.strip()
            if action not in actions:
                actions.append(action)
            if len(actions) >= limit:
                return tuple(actions)
    return tuple(actions)


def _recommendation_value(recommendation: Any, name: str, default: Any = None) -> Any:
    if isinstance(recommendation, Mapping):
        return recommendation.get(name, default)
    return getattr(recommendation, name, default)


def _recommendation_variables(recommendation: Any) -> tuple[str, ...]:
    trace = _recommendation_value(recommendation, "decision_trace", ())
    if isinstance(trace, Sequence) and not isinstance(trace, (str, bytes)):
        for item in trace:
            if not isinstance(item, Mapping) or item.get("key") != "variables":
                continue
            value = item.get("value")
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return tuple(str(name) for name in value)
    context = _recommendation_value(recommendation, "context", {})
    types = context.get("variable_types") if isinstance(context, Mapping) else None
    return tuple(str(name) for name in types) if isinstance(types, Mapping) else ()


def _quoted_variables(variables: tuple[str, ...]) -> str:
    if not variables:
        return "the selected variables"
    if len(variables) == 1:
        return repr(variables[0])
    return " and ".join(repr(item) for item in variables[:2])


def _diagnostic_value(diagnostics: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(diagnostics, Mapping):
        return None
    if key in diagnostics:
        return diagnostics[key]
    assumptions = diagnostics.get("assumptions")
    return assumptions.get(key) if isinstance(assumptions, Mapping) else None


def _why_this_recommendation(
    method_id: str,
    recommendation: Any,
    diagnostics: Mapping[str, Any] | None,
) -> str:
    context = _recommendation_value(recommendation, "context", {})
    context = context if isinstance(context, Mapping) else {}
    variables = _recommendation_variables(recommendation)
    selected = _quoted_variables(variables)
    group_sizes = context.get("group_sizes")
    group_count = len(group_sizes) if isinstance(group_sizes, Sequence) else None
    complete_pairs = context.get("complete_pairs")
    contingency_shape = context.get("contingency_shape")
    minimum_expected = _finite(context.get("minimum_expected_count"))

    if method_id == "dataset_profile":
        return (
            "The declared objective is descriptive, so the dataset profile summarizes the "
            "recorded data without converting the question into an inferential hypothesis test."
        )
    if method_id == "welch_t":
        opening = (
            f"The specification compares {selected} across two independent groups and declares "
            "a population-mean estimand. Welch's test estimates that mean contrast without "
            "requiring equal population variances."
        )
        variance_status = _diagnostic_value(diagnostics, "equal_variance_status")
        levene_p = _finite(_diagnostic_value(diagnostics, "levene_p_value"))
        if variance_status == "rejected":
            diagnostic = " An available equal-variance diagnostic was rejected"
            if levene_p is not None:
                diagnostic += f" (Levene {_p_value_text(levene_p)})"
            return opening + diagnostic + "; Welch's correction remains appropriate."
        return opening
    if method_id == "welch_anova":
        count_text = f"{group_count} " if group_count is not None else "three or more "
        return (
            f"The specification compares {selected} across {count_text}independent groups and "
            "declares a population-mean estimand. Welch ANOVA does not require equal population "
            "variances, and Games-Howell supplies every multiplicity-controlled pairwise mean "
            "contrast regardless of the omnibus decision."
        )
    if method_id == "one_sample_t":
        reference = _finite(context.get("reference_value"))
        reference_text = f" {_fmt(reference)}" if reference is not None else ""
        return (
            f"The specification declares one continuous outcome, a population-mean estimand, "
            f"and the reference value{reference_text}. The one-sample t-test targets the signed "
            "observed-mean-minus-reference contrast."
        )
    if method_id == "paired_t":
        order = context.get("condition_order")
        order_text = (
            f" The signed contrast follows {order[0]!r} minus {order[1]!r}."
            if isinstance(order, Sequence)
            and not isinstance(order, (str, bytes))
            and len(order) == 2
            else ""
        )
        pair_text = (
            f" {complete_pairs} complete pair(s) were identified for recommendation checks."
            if isinstance(complete_pairs, int) and not isinstance(complete_pairs, bool)
            else ""
        )
        return (
            f"The specification declares paired observations for {selected}, a two-condition "
            "mean estimand, and an explicit unit identifier. The paired t-test targets the "
            f"population mean within-unit difference.{order_text}{pair_text}"
        )
    if method_id == "wilcoxon_signed_rank":
        order = context.get("condition_order")
        order_text = (
            f" The signed contrast follows {order[0]!r} minus {order[1]!r}."
            if isinstance(order, Sequence)
            and not isinstance(order, (str, bytes))
            and len(order) == 2
            else ""
        )
        return (
            f"The specification declares paired observations for {selected}, an ordered "
            "rank-distribution target, and an explicit unit identifier. Wilcoxon evaluates "
            f"signed ranks of nonzero paired differences.{order_text}"
        )
    if method_id == "repeated_measures_anova":
        conditions = context.get("condition_order")
        cond_count = (
            len(conditions)
            if isinstance(conditions, Sequence) and not isinstance(conditions, (str, bytes))
            else 3
        )
        return (
            f"The specification observes the same units across {cond_count} ordered conditions "
            f"for {selected} with a declared population-mean estimand. One-way repeated-measures "
            "ANOVA evaluates whether condition population means differ, with sphericity "
            "evaluated via Mauchly's test, Greenhouse-Geisser correction applied when indicated, "
            "and complete Holm-adjusted paired-t follow-up comparisons."
        )
    if method_id == "two_way_anova":
        factor_a = context.get("factor_a")
        factor_b = context.get("factor_b")
        factors_text = (
            f"two categorical factors ({factor_a!r} and {factor_b!r})"
            if factor_a and factor_b
            else "two categorical factors"
        )
        return (
            f"The specification declares one continuous outcome ({selected}) and {factors_text} "
            "for independent observations. Two-way factorial ANOVA evaluates main effects and "
            "their interaction under a full factorial fixed-effects model with explicit "
            "sums-of-squares accounting and follow-up contrasts."
        )
    if method_id == "friedman_test":
        conditions = context.get("condition_order")
        cond_count = (
            len(conditions)
            if isinstance(conditions, Sequence) and not isinstance(conditions, (str, bytes))
            else 3
        )
        return (
            f"The specification observes the same units across {cond_count} ordered "
            f"conditions for {selected} with a declared repeated rank/distribution estimand. "
            "The Friedman test evaluates whether within-unit ranks differ across conditions, "
            "reports Kendall's W concordance, and provides complete Holm-adjusted paired "
            "Wilcoxon signed-rank follow-up."
        )
    if method_id == "mann_whitney_u":
        return (
            f"The specification compares {selected} across two independent groups and declares "
            "an ordered distribution estimand. Mann-Whitney compares rank distributions; it is "
            "not being used as a universal test of medians."
        )
    if method_id == "kruskal_wallis":
        count_text = f"{group_count} " if group_count is not None else "three or more "
        return (
            f"The specification compares {selected} across {count_text}independent groups and "
            "declares an ordered distribution estimand. Kruskal-Wallis compares their rank "
            "distributions, and Dunn-Holm supplies every multiplicity-controlled pairwise rank "
            "contrast regardless of the omnibus decision."
        )
    if method_id == "pearson_chi_square":
        shape_text = (
            f" The recorded contingency table has shape {contingency_shape[0]} by "
            f"{contingency_shape[1]}."
            if isinstance(contingency_shape, Sequence)
            and not isinstance(contingency_shape, (str, bytes))
            and len(contingency_shape) == 2
            else ""
        )
        expected_text = (
            f" Its smallest expected count is {_fmt(minimum_expected)}, which meets the current "
            "minimum-count policy."
            if minimum_expected is not None
            else ""
        )
        return (
            f"Both {selected} are categorical, and the declared target is categorical "
            "independence. Pearson's chi-square test assesses evidence of association without "
            f"assigning direction or causation.{shape_text}{expected_text}"
        )
    if method_id == "fisher_exact":
        expected_text = (
            f" The smallest expected count is {_fmt(minimum_expected)}, below the current "
            "chi-square threshold."
            if minimum_expected is not None
            else ""
        )
        return (
            f"Both {selected} are categorical and form a 2x2 table for the declared "
            "independence target. Fisher's exact test retains that target when sparse expected "
            f"counts block chi-square under package policy.{expected_text}"
        )
    if method_id == "pearson_correlation":
        pair_text = (
            f" The recommendation checks recorded {complete_pairs} complete pair(s)."
            if isinstance(complete_pairs, int) and not isinstance(complete_pairs, bool)
            else ""
        )
        return (
            f"Both {selected} are declared quantitative variables, and the estimand is linear "
            "association. Pearson correlation targets that linear relationship; it does not "
            f"establish causation.{pair_text}"
        )
    if method_id == "spearman_correlation":
        return (
            f"Both {selected} are ordered numeric variables, and the declared estimand is "
            "monotonic association. Spearman correlation targets rank-order association; it "
            "does not assert linearity or causation."
        )
    rationale = _recommendation_value(recommendation, "rationale")
    return (
        str(rationale)
        if isinstance(rationale, str) and rationale.strip()
        else "No supported researcher-readable rationale is available for this method."
    )


def _why_not_recommendation(method_id: str, recommendation: Any) -> tuple[tuple[str, str], ...]:
    alternatives = _recommendation_value(recommendation, "alternatives", ())
    recorded: dict[str, Mapping[str, Any]] = {}
    if isinstance(alternatives, Sequence) and not isinstance(alternatives, (str, bytes)):
        recorded = {
            str(item.get("method_id")): item
            for item in alternatives
            if isinstance(item, Mapping) and item.get("method_id")
        }
    if method_id == "dataset_profile":
        return (
            (
                "AN INFERENTIAL TEST",
                "The descriptive objective does not declare an inferential estimand or null "
                "hypothesis.",
            ),
        )
    if method_id == "welch_t":
        return (
            (
                "STUDENT'S T-TEST",
                str(
                    recorded.get("student_t", {}).get("reason")
                    or "It adds an equal-population-variance assumption that the declared mean "
                    "contrast does not require."
                ),
            ),
            (
                "MANN-WHITNEY",
                str(
                    recorded.get("mann_whitney_u", {}).get("reason")
                    or "It targets rank distributions rather than the declared population means."
                ),
            ),
        )
    if method_id == "one_sample_t":
        return (
            (
                "AN INDEPENDENT-SAMPLES TEST",
                "There is one outcome sample and one declared numeric reference, not two "
                "observed groups.",
            ),
        )
    if method_id == "paired_t":
        return (
            (
                "AN INDEPENDENT-SAMPLES TEST",
                "Independent-samples methods discard the declared within-unit pairing and "
                "target a different sampling structure.",
            ),
        )
    if method_id == "repeated_measures_anova":
        return (
            (
                "FRIEDMAN TEST",
                "Targets within-unit rank distributions rather than the declared "
                "population-mean pattern.",
            ),
            (
                "WELCH ANOVA",
                "Independent-groups ANOVA ignores within-subject dependence across the "
                "repeated observations.",
            ),
        )
    if method_id == "two_way_anova":
        return (
            (
                "ONE-WAY ANOVA",
                "Collapses or ignores one of the declared factors, discarding the factorial "
                "structure and interaction.",
            ),
            (
                "REPEATED-MEASURES ANOVA",
                "Assumes repeated measurements on the same units rather than independent "
                "observations across the factorial cells.",
            ),
        )
    if method_id == "friedman_test":
        return (
            (
                "REPEATED-MEASURES ANOVA",
                "Targets repeated condition means rather than the declared rank/distribution "
                "estimand; mean models are not substituted solely from diagnostics.",
            ),
            (
                "KRUSKAL-WALLIS",
                "Independent-groups Kruskal-Wallis ignores within-subject dependence across "
                "repeated conditions.",
            ),
        )
    if method_id == "wilcoxon_signed_rank":
        return (
            (
                "PAIRED T-TEST",
                str(
                    recorded.get("paired_t", {}).get("reason")
                    or "It targets the population mean paired difference rather than the "
                    "declared rank-distribution target."
                ),
            ),
        )
    if method_id == "mann_whitney_u":
        return (
            (
                "WELCH'S T-TEST",
                str(
                    recorded.get("welch_t", {}).get("reason")
                    or "It targets population means rather than the declared rank-distribution "
                    "contrast."
                ),
            ),
        )
    if method_id == "kruskal_wallis":
        return (
            (
                "ONE-WAY ANOVA",
                str(
                    recorded.get("one_way_anova", {}).get("reason")
                    or "It targets population means and requires additional mean-inference "
                    "assumptions."
                ),
            ),
        )
    if method_id == "pearson_chi_square":
        return (
            (
                "PEARSON CORRELATION",
                "Correlation targets linear association between quantitative measurements; it "
                "is not a substitute for categorical independence.",
            ),
        )
    if method_id == "fisher_exact":
        return (
            (
                "PEARSON CHI-SQUARE",
                str(
                    recorded.get("pearson_chi_square", {}).get("reason")
                    or "At least one expected count is below PyAutoStat's chi-square threshold."
                ),
            ),
        )
    if method_id == "pearson_correlation":
        return (
            (
                "SPEARMAN CORRELATION",
                str(
                    recorded.get("spearman_correlation", {}).get("reason")
                    or "It targets monotonic rank association rather than the declared linear "
                    "association target."
                ),
            ),
        )
    if method_id == "spearman_correlation":
        return (
            (
                "PEARSON CORRELATION",
                str(
                    recorded.get("pearson_correlation", {}).get("reason")
                    or "It targets linear association rather than the declared monotonic "
                    "rank-association target."
                ),
            ),
        )
    return ()


def _verification_notes(method_id: str, recommendation: Any) -> tuple[str, ...]:
    context = _recommendation_value(recommendation, "context", {})
    context = context if isinstance(context, Mapping) else {}
    if method_id == "dataset_profile":
        return (
            "Verify variable roles, units, and missing-value codes from collection documentation; "
            "values alone cannot establish their meaning.",
            "Confirm that the observed rows are appropriate for the descriptive population of "
            "interest.",
        )
    if method_id == "paired_t":
        unit_id = context.get("unit_id")
        order = context.get("condition_order")
        unit_text = (
            f"Verify that {unit_id!r} correctly identifies the same unit across conditions and "
            "that different units are independent."
            if unit_id is not None
            else "Verify that the unit identifier correctly links the same unit across "
            "conditions and that different units are independent."
        )
        order_text = (
            f"Verify that the declared condition order {order[0]!r} minus {order[1]!r} matches "
            "the intended signed contrast."
            if isinstance(order, Sequence)
            and not isinstance(order, (str, bytes))
            and len(order) == 2
            else "Verify the intended condition order before interpreting the sign of the "
            "paired difference."
        )
        return (
            unit_text,
            order_text,
            "Verify sampling relevance and whether paired differences are suitable for "
            "population-mean inference; these facts are not established by the values.",
        )
    if method_id == "wilcoxon_signed_rank":
        return (
            "Verify that the unit identifier links the same or matched unit across both "
            "conditions and that different pairs are independent.",
            "Verify that the declared first-minus-second condition order matches the intended "
            "effect direction.",
            "For a location-shift interpretation, review whether the paired-difference "
            "distribution is suitably symmetric; do not assume a universal median test.",
            "Zero paired differences are omitted from ranks under the recorded 'wilcox' policy.",
        )
    notes = [
        "Verify from the study design that observational units are independent; numerical "
        "values cannot establish independence.",
        "Verify that sampling, measurement, and exclusions support inference to the intended "
        "population; representativeness is not data-tested here.",
    ]
    if method_id == "welch_t":
        notes.append(
            "Review group distributions, influential observations, and mean-inference "
            "conditions; Welch's correction addresses unequal variances, not every design or "
            "distribution issue."
        )
    elif method_id == "one_sample_t":
        notes.append(
            "Verify that the reference value is scientifically justified and that independent "
            "observations and distributional conditions support population-mean inference."
        )
    elif method_id in {"mann_whitney_u", "kruskal_wallis"}:
        notes.append(
            "Verify that outcome ordering is scientifically meaningful and interpret the result "
            "as a rank-distribution comparison, not automatically as a median difference."
        )
    elif method_id == "pearson_chi_square":
        minimum = _finite(context.get("minimum_expected_count"))
        if minimum is not None:
            notes.append(
                f"The recommendation check recorded a minimum expected count of {_fmt(minimum)}; "
                "retain this limitation if the analyzed table changes."
            )
        notes.append(
            "Verify that categories and each observational unit were defined independently of "
            "the analyzed outcome."
        )
    elif method_id == "fisher_exact":
        notes.append(
            "Verify the 2x2 category definitions, ordered levels, observational independence, "
            "and whether the table margins support the intended exact-test interpretation."
        )
    elif method_id == "pearson_correlation":
        notes.append(
            "Review linearity, influential observations, and scale suitability; the "
            "recommendation does not infer these scientific conditions or causation."
        )
    elif method_id == "spearman_correlation":
        notes.append(
            "Verify meaningful ordering, observational independence, and a scientifically "
            "relevant monotonic target; ties are permitted but should be reported."
        )
    elif method_id == "repeated_measures_anova":
        notes.extend(
            [
                "Confirm that the unit identifier uniquely identifies repeated units.",
                "Confirm that the declared condition order reflects the intended structure.",
                "Verify that complete-case analysis across all conditions is acceptable.",
                "Check that observational units are independent across subjects.",
            ]
        )
    elif method_id == "two_way_anova":
        notes.extend(
            [
                "Verify that observational units are genuinely independent across all cells.",
                "Confirm that all factor combinations are represented with sufficient data.",
                "Review the selected sums-of-squares convention for unbalanced data.",
                "Check residual normality and homoscedasticity diagnostics.",
            ]
        )
    elif method_id == "friedman_test":
        notes.extend(
            [
                "Confirm that the unit identifier uniquely identifies repeated units.",
                "Confirm that the declared condition order reflects the intended structure.",
                "Verify that the outcome supports meaningful ordinal ranking within each unit.",
                "Verify that complete-case analysis across all conditions is acceptable.",
            ]
        )
    return tuple(notes)


def recommendation_rationale(
    recommendation: Any,
    diagnostics: Mapping[str, Any] | None = None,
) -> str:
    """Explain a ready recommendation from its stored context and optional diagnostics.

    The explanation is presentation-only: it does not select a method or run a
    diagnostic. A diagnostic is mentioned only when its recorded values are
    explicitly supplied by the caller.
    """
    status = _recommendation_value(recommendation, "status")
    status_value = getattr(status, "value", status)
    method_id = _recommendation_value(recommendation, "method_id")
    if status_value != "ready" or not isinstance(method_id, str):
        rationale = _recommendation_value(recommendation, "rationale")
        return "RECOMMENDATION NOT READY\n\n" + (
            str(rationale)
            if isinstance(rationale, str) and rationale.strip()
            else "No statistical method is ready for explanation."
        )
    method_name = _recommendation_value(recommendation, "method_name")
    method_label = method_name if isinstance(method_name, str) and method_name else method_id
    lines = [
        "RECOMMENDED TEST:",
        method_label,
        "",
        "WHY THIS TEST?",
        _why_this_recommendation(method_id, recommendation, diagnostics),
    ]
    for label, explanation in _why_not_recommendation(method_id, recommendation):
        lines.extend(("", f"WHY NOT {label}?", explanation))
    lines.extend(("", "WHAT YOU NEED TO VERIFY"))
    lines.extend(f"- {note}" for note in _verification_notes(method_id, recommendation))
    return "\n".join(lines)


def executive_summary(
    *,
    profile: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    analyses: Sequence[Mapping[str, Any]] = (),
    limitations: Sequence[str] = (),
    practical_significance: str | None = None,
    sensitivity: str | None = None,
) -> tuple[str, ...]:
    """Build format-neutral executive-summary paragraphs from stored report values."""
    paragraphs: list[str] = []
    overview = profile.get("overview") if isinstance(profile, Mapping) else None
    rows = overview.get("total_rows") if isinstance(overview, Mapping) else None
    columns = overview.get("total_columns") if isinstance(overview, Mapping) else None
    if rows is None and isinstance(dataset, Mapping):
        rows = dataset.get("original_rows")
    analyzed = dataset.get("analyzed_rows") if isinstance(dataset, Mapping) else None
    count = len(analyses)
    if isinstance(rows, int) and isinstance(columns, int):
        scale = f"{rows:,} row(s) across {columns:,} column(s)"
    elif isinstance(rows, int):
        scale = f"{rows:,} recorded row(s)"
    elif isinstance(analyzed, int):
        scale = f"{analyzed:,} analyzed row(s)"
    else:
        scale = "the recorded dataset"
    if count:
        paragraphs.append(
            f"This report summarizes {scale} and contains {count} statistical "
            f"{'analysis' if count == 1 else 'analyses'}."
        )
    else:
        paragraphs.append(
            f"This report summarizes {scale}. No inferential test is represented; the report "
            "is descriptive."
        )
    for analysis in analyses:
        method = analysis.get("method_name")
        finding = analysis.get("finding")
        if not isinstance(method, str) or not method.strip():
            method = "Recorded analysis"
        if isinstance(finding, str) and finding.strip():
            paragraphs.append(f"{method}: {finding.strip()}")
        else:
            paragraphs.append(
                f"{method}: the stored result does not support a significance/effect summary."
            )
    if isinstance(profile, Mapping):
        paragraphs.append("Data quality: " + _data_quality_story(profile))
    first_limitation = next(
        (item.strip() for item in limitations if isinstance(item, str) and item.strip()), None
    )
    if first_limitation is not None:
        paragraphs.append(f"Important diagnostic or limitation: {first_limitation}")
    if isinstance(practical_significance, str) and practical_significance.strip():
        paragraphs.append("Practical significance was assessed: " + practical_significance.strip())
    if isinstance(sensitivity, str) and sensitivity.strip():
        paragraphs.append("Sensitivity analysis was performed: " + sensitivity.strip())
    return tuple(paragraphs)


def reliability_narrative(
    alpha: float,
    item_count: int,
    respondent_count: int,
    interval: Mapping[str, Any] | None,
) -> str:
    """Describe an alpha estimate without imposing a universal quality cutoff."""
    text = (
        f"Cronbach's alpha was {_fmt(alpha)} for {item_count} researcher-declared items "
        f"among {respondent_count} complete respondents."
    )
    if isinstance(interval, Mapping) and interval.get("status") == "available":
        lower = _finite(interval.get("lower"))
        upper = _finite(interval.get("upper"))
        level = _finite(interval.get("level"))
        if lower is not None and upper is not None and level is not None:
            text += (
                f" The {_fmt(level * 100)}% respondent-row bootstrap interval was "
                f"[{_fmt(lower)}, {_fmt(upper)}]."
            )
    return text


def reliability_item_narrative(item_statistics: Sequence[Mapping[str, Any]]) -> str:
    """Summarize diagnostic cues while leaving item decisions to the researcher."""
    negative = [
        str(item.get("item"))
        for item in item_statistics
        if (_finite(item.get("corrected_item_total_correlation")) or 0) < 0
    ]
    available_deltas = [
        (str(item.get("item")), _finite(item.get("delta_from_full_alpha")))
        for item in item_statistics
    ]
    increases = [
        (name, value) for name, value in available_deltas if value is not None and value > 0
    ]
    parts = []
    if negative:
        parts.append(
            "Negative corrected item-total correlations were observed for "
            + ", ".join(repr(item) for item in negative)
            + "."
        )
    if increases:
        name, value = max(increases, key=lambda pair: pair[1])
        parts.append(
            f"The largest recorded alpha increase on deletion was {_fmt(value)} for {name!r}."
        )
    if not parts:
        parts.append("No negative corrected item-total correlation was recorded.")
    parts.append("These are review cues, not instructions to delete or reverse-score an item.")
    return " ".join(parts)


def reliability_diagnostics_narrative(
    mean_inter_item: float | None,
    negative_pair_count: int,
) -> str:
    """Describe the recorded inter-item correlation diagnostics."""
    mean_text = (
        f"The mean available unique inter-item correlation was {_fmt(mean_inter_item)}. "
        if mean_inter_item is not None
        else "The mean inter-item correlation was unavailable. "
    )
    return (
        mean_text
        + f"{negative_pair_count} negative unique inter-item "
        + ("pair was" if negative_pair_count == 1 else "pairs were")
        + " recorded."
    )


__all__ = [
    "ASSUMPTION_SEVERITY_DESCRIPTIONS",
    "assumption_grade",
    "coefficient_of_variation_narrative",
    "column_story",
    "crosstab_narrative",
    "dataset_opening",
    "effect_narrative",
    "executive_summary",
    "frequency_narrative",
    "hypothesis_verdict",
    "insight_narrative",
    "interval_verdict",
    "percentile_narrative",
    "recommendation_rationale",
    "reliability_diagnostics_narrative",
    "reliability_item_narrative",
    "reliability_narrative",
    "sensitivity_verdict",
]
