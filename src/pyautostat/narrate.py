"""Deterministic narration of already-computed statistical effects.

This module contains presentation rules only. It does not calculate effect
sizes or confidence intervals and it does not select statistical methods.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
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


__all__ = ["effect_narrative"]
