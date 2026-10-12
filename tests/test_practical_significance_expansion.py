"""Tests for PR-6 Part C practical-significance quantity expansion and validations."""

import pytest

from pyautostat.exceptions import InvalidDataError
from pyautostat.practical_significance import (
    MeaningfulEffectThreshold,
    practical_significance,
)
from pyautostat.results import AnalysisResult, AnalysisStatus


def _dummy_result(
    method_id: str,
    primary_estimate: float | None = None,
    estimate_name: str | None = None,
    ci: dict | None = None,
    effect_size: dict | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        method_id=method_id,
        status=AnalysisStatus.AVAILABLE,
        sample_size=30,
        excluded_rows=0,
        values={
            "primary_estimate": primary_estimate,
            "estimate_name": estimate_name,
            "p_value": 0.01,
            "confidence_interval": ci if primary_estimate is not None else None,
            "effect_size": effect_size,
        },
        assumptions=(),
        warnings=(),
        metadata={"method_name": method_id},
    )


def test_wilcoxon_matched_rank_biserial_threshold():
    effect = {
        "name": "matched-pairs rank-biserial correlation",
        "value": 0.45,
        "confidence_interval": {
            "quantity": "matched-pairs rank-biserial correlation",
            "lower": 0.20,
            "upper": 0.65,
            "level": 0.95,
            "method": "bootstrap",
        },
    }
    res = _dummy_result("wilcoxon_signed_rank", effect_size=effect)
    threshold = MeaningfulEffectThreshold("matched_pairs_rank_biserial", 0.3, direction="two_sided")
    ps = practical_significance(res, threshold)
    assert ps.status == "complete"
    assert ps.point_estimate_relation == "positive_meaningful_region"
    assert ps.confidence_interval_relation == "crosses_meaningful_boundary"


def test_point_biserial_threshold():
    res = _dummy_result(
        "point_biserial_correlation",
        primary_estimate=0.50,
        estimate_name="point-biserial r",
        ci={
            "quantity": "point-biserial r",
            "lower": 0.35,
            "upper": 0.65,
            "level": 0.95,
            "method": "analytical",
        },
    )
    threshold = MeaningfulEffectThreshold("point_biserial_r", 0.30, direction="two_sided")
    ps = practical_significance(res, threshold)
    assert ps.status == "complete"
    assert ps.point_estimate_relation == "positive_meaningful_region"
    assert ps.confidence_interval_relation == "entirely_positive_meaningful"


def test_kendall_tau_b_threshold():
    res = _dummy_result(
        "kendall_tau_b",
        primary_estimate=0.40,
        estimate_name="Kendall's tau-b",
        ci={
            "quantity": "Kendall's tau-b",
            "lower": 0.25,
            "upper": 0.55,
            "level": 0.95,
            "method": "analytical",
        },
    )
    threshold = MeaningfulEffectThreshold("kendall_tau_b", 0.20, direction="positive")
    ps = practical_significance(res, threshold)
    assert ps.status == "complete"
    assert ps.point_estimate_relation == "meets_positive_threshold"
    assert ps.confidence_interval_relation == "entirely_above_positive_threshold"


def test_partial_pearson_threshold():
    res = _dummy_result(
        "partial_pearson_correlation",
        primary_estimate=-0.35,
        estimate_name="partial Pearson r",
        ci={
            "quantity": "partial Pearson r",
            "lower": -0.50,
            "upper": -0.15,
            "level": 0.95,
            "method": "analytical",
        },
    )
    threshold = MeaningfulEffectThreshold("partial_pearson_r", 0.20, direction="negative")
    ps = practical_significance(res, threshold)
    assert ps.status == "complete"
    assert ps.point_estimate_relation == "meets_negative_threshold"
    assert ps.confidence_interval_relation == "crosses_negative_threshold"


def test_paired_proportion_difference_threshold():
    res = _dummy_result(
        "mcnemar",
        primary_estimate=0.25,
        estimate_name="paired proportion difference",
        ci={
            "quantity": "paired proportion difference",
            "lower": 0.10,
            "upper": 0.40,
            "level": 0.95,
            "method": "analytical",
        },
    )
    threshold = MeaningfulEffectThreshold(
        "paired_proportion_difference", 0.15, direction="two_sided"
    )
    ps = practical_significance(res, threshold)
    assert ps.status == "complete"
    assert ps.point_estimate_relation == "positive_meaningful_region"


def test_kendalls_w_threshold():
    effect = {
        "name": "Kendall's W",
        "value": 0.60,
        "confidence_interval": {
            "quantity": "Kendall's W",
            "lower": 0.45,
            "upper": 0.75,
            "level": 0.95,
            "method": "bootstrap",
        },
    }
    res = _dummy_result("friedman_test", effect_size=effect)
    threshold = MeaningfulEffectThreshold("kendalls_w", 0.50, direction="nonnegative")
    ps = practical_significance(res, threshold)
    assert ps.status == "complete"
    assert ps.point_estimate_relation == "meets_nonnegative_threshold"


def test_estimate_domain_validation():
    # Correlation estimate > 1 is invalid
    res = _dummy_result(
        "point_biserial_correlation",
        primary_estimate=1.25,
        estimate_name="point-biserial r",
    )
    threshold = MeaningfulEffectThreshold("point_biserial_r", 0.20)
    with pytest.raises(InvalidDataError, match=r"outside \[-1, 1\]"):
        practical_significance(res, threshold)

    # Kendall's W > 1 is invalid
    res_w = _dummy_result("friedman_test", effect_size={"name": "Kendall's W", "value": 1.5})
    thresh_w = MeaningfulEffectThreshold("kendalls_w", 0.20, direction="nonnegative")
    with pytest.raises(InvalidDataError, match=r"outside \[0, 1\]"):
        practical_significance(res_w, thresh_w)


def test_ci_domain_validation():
    res = _dummy_result(
        "kendall_tau_b",
        primary_estimate=0.40,
        estimate_name="Kendall's tau-b",
        ci={
            "quantity": "Kendall's tau-b",
            "lower": -0.20,
            "upper": 1.20,  # Invalid upper bound > 1
            "level": 0.95,
            "method": "analytical",
        },
    )
    threshold = MeaningfulEffectThreshold("kendall_tau_b", 0.20)
    with pytest.raises(InvalidDataError, match=r"interval is outside \[-1, 1\]"):
        practical_significance(res, threshold)


def test_missing_ci_falls_back_to_partial_assessment():
    res = _dummy_result(
        "point_biserial_correlation",
        primary_estimate=0.45,
        estimate_name="point-biserial r",
        ci=None,
    )
    threshold = MeaningfulEffectThreshold("point_biserial_r", 0.30)
    ps = practical_significance(res, threshold)
    assert ps.status == "partial"
    assert ps.confidence_interval_relation == "unavailable"
    assert ps.uncertainty_status == "confidence_interval_unavailable"
    assert ps.point_estimate_relation == "positive_meaningful_region"


def test_wrong_quantity_raises_invalid_data_error():
    res = _dummy_result(
        "point_biserial_correlation",
        primary_estimate=0.45,
        estimate_name="point-biserial r",
    )
    threshold = MeaningfulEffectThreshold("mean_difference", 5.0)
    with pytest.raises(InvalidDataError, match="The analysis does not provide quantity"):
        practical_significance(res, threshold)


def test_unsupported_odds_ratio_clear_explanation():
    with pytest.raises(
        InvalidDataError,
        match="odds-ratio meaningfulness requires a reference-centered ratio contract around OR=1",
    ):
        MeaningfulEffectThreshold("odds_ratio", 1.5)


def test_unsupported_coefficient_vector_clear_explanation():
    with pytest.raises(InvalidDataError, match="without an explicit researcher-selected term"):
        MeaningfulEffectThreshold("coefficient_vector", 0.5)


def test_unsupported_reliability_clear_explanation():
    with pytest.raises(
        InvalidDataError,
        match="Reliability coefficients require a directional minimum-level contract",
    ):
        MeaningfulEffectThreshold("cronbach_alpha", 0.70)
