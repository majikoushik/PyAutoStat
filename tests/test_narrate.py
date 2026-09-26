"""Deterministic narration consumes recorded effects without calculating them."""

import math

import pandas as pd
import pytest

from pyautostat import ResearchAssistant, effect_narrative
from pyautostat.narrate import _confidence_interval_narrative, _confidence_interval_width


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (0.0, "negligible"),
        (0.1999, "negligible"),
        (0.2, "small"),
        (0.4999, "small"),
        (0.5, "medium"),
        (0.7999, "medium"),
        (0.8, "large"),
        (1.9999, "large"),
        (2.0, "very large"),
        (2.5, "very large"),
    ],
)
def test_cohens_d_boundaries(value, label):
    narrative = effect_narrative("cohens_d", value)
    assert f"labelled '{label}'" in narrative
    assert f"Cohen's d = {value:.4g}" in narrative


@pytest.mark.parametrize(
    ("value", "label"),
    [(-0.2, "small"), (-0.5, "medium"), (-0.8, "large"), (-2.5, "very large")],
)
def test_cohens_d_uses_absolute_magnitude_but_preserves_sign(value, label):
    narrative = effect_narrative("Cohen's d", value)
    assert f"Cohen's d = {value:.4g}" in narrative
    assert f"labelled '{label}'" in narrative


def test_enhancement_plan_example_includes_effect_sample_and_ci_commentary():
    narrative = effect_narrative("cohens_d", 2.5, n=100, ci=(2.1, 2.9))
    assert "very large" in narrative
    assert "2.5" in narrative
    assert "n = 100" in narrative
    assert "confidence interval [2.1, 2.9]" in narrative
    assert "narrow (precise estimate)" in narrative


@pytest.mark.parametrize(
    ("measure", "value", "expected"),
    [
        ("eta_squared", 0.1, "labelled 'medium'"),
        ("cramers_v", 0.4, "labelled 'medium'"),
        ("pearson_r", -0.4, "labelled 'medium'"),
        ("rank_biserial", -0.4, "does not assign a conventional magnitude label"),
        ("epsilon_squared", 0.4, "does not assign a conventional magnitude label"),
        ("Cohen's dz", -0.8, "labelled 'large'"),
    ],
)
def test_supported_effects_reuse_existing_magnitude_policy(measure, value, expected):
    assert expected in effect_narrative(measure, value)


def test_measure_specific_explanations_preserve_direction_and_scope():
    rank = effect_narrative("rank_biserial", -0.4, orientation="'A' minus 'B'")
    pearson = effect_narrative("pearson_r", 0.0)
    cramer = effect_narrative("cramers_v", 0.3)
    epsilon = effect_narrative("epsilon_squared", 0.2)
    assert "'A' minus 'B'" in rank
    assert "not a median difference" in rank
    assert "zero" in pearson
    assert "does not establish population independence" in pearson
    assert "nonnegative" in cramer and "no direction" in cramer
    assert "Kruskal-Wallis" in epsilon


@pytest.mark.parametrize("value", [None, math.nan, math.inf, -math.inf])
def test_unavailable_or_nonfinite_effect_does_not_become_zero(value):
    narrative = effect_narrative("cohens_d", value)
    assert "unavailable" in narrative
    assert "Cohen's d = 0" not in narrative
    assert "negligible" not in narrative


def test_unsupported_and_out_of_range_effects_are_safe():
    assert "unsupported" in effect_narrative("odds_ratio", 1.2)
    assert "outside its valid range" in effect_narrative("pearson_r", 1.1)
    assert "outside its valid range" in effect_narrative("eta_squared", -0.1)


@pytest.mark.parametrize(
    ("estimate", "interval", "expected"),
    [
        (1.0, (0.9, 1.1), "narrow (precise estimate)"),
        (1.0, (0.25, 1.25), "moderate width"),
        (1.0, (-1.0, 2.0), "wide — the true effect is poorly constrained"),
        (-1.0, (-1.1, -0.9), "narrow (precise estimate)"),
        (0.0, (0.0, 0.0), "narrow (precise estimate)"),
        (0.0, (-0.1, 0.1), "wide — the true effect is poorly constrained"),
    ],
)
def test_confidence_interval_width_classification(estimate, interval, expected):
    lower, upper = interval
    assert _confidence_interval_width(estimate, lower, upper) == expected
    assert expected in effect_narrative("cohens_d", estimate, ci=interval)


def test_confidence_interval_uses_recorded_level_and_method():
    ci = {
        "lower": 0.8,
        "upper": 1.2,
        "level": 0.9,
        "method": "independent within-group percentile bootstrap",
    }
    narrative = effect_narrative("cohens_d", 1.0, ci=ci)
    assert "90% bootstrap confidence interval" in narrative
    assert "95%" not in narrative


@pytest.mark.parametrize(
    "ci",
    [
        (None, 1.0),
        (0.0, None),
        (math.nan, 1.0),
        (0.0, math.inf),
        (2.0, 1.0),
        {"lower": 0.0},
        "not an interval",
    ],
)
def test_invalid_or_incomplete_ci_is_omitted_safely(ci):
    narrative = effect_narrative("cohens_d", 1.0, ci=ci)
    assert "confidence interval" not in narrative


def test_ci_helper_does_not_invent_confidence_level():
    narrative = _confidence_interval_narrative(1.0, (0.8, 1.2))
    assert narrative is not None
    assert "confidence interval" in narrative
    assert "%" not in narrative


def test_identical_inputs_produce_identical_text():
    arguments = {
        "measure": "cohens_d",
        "value": -0.8,
        "n": 25,
        "ci": {"lower": -1.0, "upper": -0.6, "level": 0.95},
        "orientation": "first minus second",
    }
    assert effect_narrative(**arguments) == effect_narrative(**arguments)


def test_interpretation_integration_adds_rich_effect_narrative():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 8 + ["B"] * 8,
            "score": [10.0, 11, 12, 13, 14, 15, 16, 17, 1.0, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    assistant = ResearchAssistant(frame)
    result = assistant.analyze(
        assistant.prepare_question(
            objective="compare_groups",
            outcome="score",
            predictor="group",
            estimand="mean",
            design="independent",
            variable_types={"score": "continuous", "group": "nominal"},
        )
    )
    interpretation = assistant.interpret(result)
    assert "conventionally labelled" in interpretation.effect_interpretation
    assert "pooled sample SD" in interpretation.effect_interpretation
    assert "'A' minus 'B'" in interpretation.effect_interpretation
    assert "Relative to Cohen's d" in interpretation.effect_interpretation
