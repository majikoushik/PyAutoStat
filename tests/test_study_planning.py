"""Prospective planning numerical and validation contracts."""

import json
import math

import pytest
from scipy import stats

from pyautostat import InvalidDataError, StudyPlanner, StudyPlanningResult


def _welch_power(n1, n2, difference, sd1, sd2, alpha):
    first = sd1**2 / n1
    second = sd2**2 / n2
    se = math.sqrt(first + second)
    df = (first + second) ** 2 / (first**2 / (n1 - 1) + second**2 / (n2 - 1))
    critical = stats.t.ppf(1 - alpha / 2, df)
    ncp = difference / se
    return stats.nct.cdf(-critical, df, ncp) + stats.nct.sf(critical, df, ncp)


def test_independent_welch_power_reference_and_minimum_integer_solution():
    result = StudyPlanner().independent_mean_power(
        target_difference=5,
        sd_group1=10,
        sd_group2=12,
        alpha=0.05,
        target_power=0.80,
        allocation_ratio=1,
    )
    assert result.status == "available"
    assert (result.required_n1, result.required_n2, result.total_required_n) == (78, 78, 156)
    assert result.achieved_power == pytest.approx(_welch_power(78, 78, 5, 10, 12, 0.05), rel=1e-13)
    assert _welch_power(77, 77, 5, 10, 12, 0.05) < 0.80
    assert "approximation" in result.calculation_method


def test_independent_planning_supports_unequal_sd_and_allocation():
    result = StudyPlanner().independent_mean_power(
        target_difference=-5,
        sd_group1=10,
        sd_group2=12,
        target_power=0.80,
        allocation_ratio=2,
    )
    assert (result.required_n1, result.required_n2) == (55, 110)
    assert result.achieved_allocation_ratio == 2
    assert result.assumptions["target_difference"] == 5


def test_independent_precision_reference_and_minimum_solution():
    result = StudyPlanner().independent_mean_precision(
        sd_group1=10,
        sd_group2=12,
        confidence_level=0.95,
        target_half_width=3,
    )
    assert (result.required_n1, result.required_n2) == (106, 106)
    assert result.achieved_half_width == pytest.approx(2.9914518891987667)
    first, second = 10**2 / 105, 12**2 / 105
    df = (first + second) ** 2 / (first**2 / 104 + second**2 / 104)
    previous = stats.t.ppf(0.975, df) * math.sqrt(first + second)
    assert previous > 3


def test_paired_power_and_precision_return_complete_pairs():
    planner = StudyPlanner()
    power = planner.paired_mean_power(
        target_mean_difference=2,
        sd_difference=5,
        alpha=0.05,
        target_power=0.80,
    )
    precision = planner.paired_mean_precision(
        sd_difference=5,
        confidence_level=0.95,
        target_half_width=2,
    )
    assert power.required_pairs == 52
    assert power.total_required_n is None
    assert power.achieved_power == pytest.approx(0.8077878088760594)
    assert precision.required_pairs == 27
    assert precision.total_required_n is None
    assert precision.achieved_half_width == pytest.approx(1.977934124546103)
    assert "complete pairs" in power.approximation_notes[0]


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("independent_mean_power", {"target_difference": 0, "sd_group1": 1, "sd_group2": 1}),
        ("independent_mean_power", {"target_difference": True, "sd_group1": 1, "sd_group2": 1}),
        ("independent_mean_power", {"target_difference": 1, "sd_group1": 0, "sd_group2": 1}),
        ("independent_mean_power", {"target_difference": 1, "sd_group1": 1, "sd_group2": math.inf}),
        (
            "independent_mean_power",
            {"target_difference": 1, "sd_group1": 1, "sd_group2": 1, "alpha": 1},
        ),
        ("independent_mean_precision", {"sd_group1": 1, "sd_group2": 1, "target_half_width": 0}),
        ("paired_mean_power", {"target_mean_difference": math.nan, "sd_difference": 1}),
        ("paired_mean_precision", {"sd_difference": 1, "target_half_width": 1, "max_pairs": True}),
    ],
)
def test_invalid_planning_inputs_are_rejected(method, kwargs):
    with pytest.raises(InvalidDataError):
        getattr(StudyPlanner(), method)(**kwargs)


def test_bounded_search_returns_actionable_unavailable_result():
    result = StudyPlanner().paired_mean_power(
        target_mean_difference=0.01,
        sd_difference=10,
        target_power=0.99,
        max_pairs=3,
    )
    assert result.status == "unavailable"
    assert "max_pairs=3" in result.warnings[0]


def test_study_planning_result_is_strict_json_and_validates_status():
    result = StudyPlanner().paired_mean_precision(sd_difference=2, target_half_width=1)
    assert json.loads(result.to_json())["schema_version"] == 1
    with pytest.raises(InvalidDataError, match="status"):
        StudyPlanningResult("bad", "power", "paired", "difference", {})
