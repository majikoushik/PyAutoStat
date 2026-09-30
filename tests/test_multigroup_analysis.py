"""Numerical and integration coverage for complete independent multi-group analysis."""

import json
import math

import pandas as pd
import pytest
from scipy import stats

from pyautostat import (
    AnalysisOptions,
    ResearchAssistant,
    SensitivitySpecification,
    StatisticalAnalyzer,
    reproduce,
)
from pyautostat.exceptions import InsufficientDataError
from pyautostat.multigroup import adjust_pvalues


@pytest.fixture
def unequal_groups():
    return pd.DataFrame(
        {
            "group": ["C"] * 6 + ["A"] * 7 + ["B"] * 5,
            "score": [2, 3, 4, 5, 7, 8, 8, 10, 12, 15, 18, 22, 27, 1, 2, 2, 3, 5],
        }
    )


def test_welch_anova_matches_independent_formula_and_preserves_order(unequal_groups):
    raw = StatisticalAnalyzer(unequal_groups).welch_anova("group", "score")
    groups = [
        unequal_groups.loc[unequal_groups.group == label, "score"].to_numpy(dtype=float)
        for label in ("C", "A", "B")
    ]
    sizes = [len(group) for group in groups]
    means = [group.mean() for group in groups]
    variances = [group.var(ddof=1) for group in groups]
    weights = [size / variance for size, variance in zip(sizes, variances, strict=True)]
    weighted_mean = sum(weight * mean for weight, mean in zip(weights, means, strict=True)) / sum(
        weights
    )
    k = len(groups)
    term = sum(
        (1 - weight / sum(weights)) ** 2 / (size - 1)
        for weight, size in zip(weights, sizes, strict=True)
    )
    expected_f = (
        sum(
            weight * (mean - weighted_mean) ** 2
            for weight, mean in zip(weights, means, strict=True)
        )
        / (k - 1)
        / (1 + 2 * (k - 2) * term / (k**2 - 1))
    )
    expected_df2 = (k**2 - 1) / (3 * term)
    assert raw["groups"] == ["C", "A", "B"]
    assert raw["statistic"] == pytest.approx(expected_f)
    assert raw["degrees_of_freedom"] == pytest.approx([2, expected_df2])
    assert raw["p_value"] == pytest.approx(stats.f.sf(expected_f, 2, expected_df2))
    assert raw["effect_size"]["status"] == "not_applicable"


def test_games_howell_contract_is_complete_oriented_and_simultaneous(unequal_groups):
    raw = StatisticalAnalyzer(unequal_groups).games_howell("group", "score")
    assert [(item["group1"], item["group2"]) for item in raw["comparisons"]] == [
        ("C", "A"),
        ("C", "B"),
        ("A", "B"),
    ]
    for item in raw["comparisons"]:
        assert item["contrast"]["definition"] == "first group minus second group"
        assert 0 <= item["raw_p_value"] <= 1
        assert item["adjusted_p_value"] >= item["raw_p_value"]
        assert 0 <= item["adjusted_p_value"] <= 1
        assert item["comparison_count"] == 3
        assert item["confidence_interval"]["lower"] <= item["estimate"]
        assert item["confidence_interval"]["upper"] >= item["estimate"]
        assert item["standard_error"] > 0
        assert item["degrees_of_freedom"] > 0


def test_tukey_hsd_matches_scipy_equal_size_reference():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "score": [1, 2, 3, 4, 5, 3, 4, 6, 7, 8, 8, 9, 10, 11, 13],
        }
    )
    raw = StatisticalAnalyzer(frame).tukey_hsd("group", "score")
    if hasattr(stats, "tukey_hsd"):
        reference = stats.tukey_hsd(
            *(frame.loc[frame.group == label, "score"] for label in ("A", "B", "C"))
        )
        assert raw["comparisons"][0]["adjusted_p_value"] == pytest.approx(reference.pvalue[0, 1])
        assert raw["comparisons"][1]["adjusted_p_value"] == pytest.approx(reference.pvalue[0, 2])
        assert raw["comparisons"][2]["adjusted_p_value"] == pytest.approx(reference.pvalue[1, 2])
    classical = StatisticalAnalyzer(frame).hypothesis_tests("group", "score", test_type="anova")
    assert classical["pairwise_method"].startswith("Tukey HSD")
    assert len(classical["pairwise_comparisons"]) == 3


def test_dunn_uses_tie_correction_pairwise_effect_and_holm():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "score": [1, 1, 2, 2, 3, 2, 2, 3, 4, 4, 4, 5, 5, 6, 6],
        }
    )
    raw = StatisticalAnalyzer(frame).dunn("group", "score")
    assert 0 < raw["tie_correction"] < 1
    adjusted = adjust_pvalues([item["raw_p_value"] for item in raw["comparisons"]])
    assert [item["adjusted_p_value"] for item in raw["comparisons"]] == pytest.approx(adjusted)
    for item in raw["comparisons"]:
        assert item["confidence_interval"] is not None
        assert (
            item["confidence_interval"]["method"] == "independent within-group percentile bootstrap"
        )
        assert item["confidence_interval"]["multiplicity_adjusted"] is False
        assert -1 <= item["effect_size"]["value"] <= 1
        assert item["effect_size"]["uncertainty_status"] == "available"


def test_holm_is_monotone_and_restores_input_order():
    assert adjust_pvalues([0.04, 0.001, 0.03, 0.20]) == pytest.approx([0.09, 0.004, 0.09, 0.20])
    assert adjust_pvalues([0.04, 0.001], "bonferroni") == pytest.approx([0.08, 0.002])


def test_pairwise_family_is_calculated_when_omnibus_is_not_significant():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
            "score": [1, 2, 3, 4, 5, 6, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
        }
    )
    raw = StatisticalAnalyzer(frame).welch_anova("group", "score")
    assert raw["p_value"] > 0.05
    assert len(raw["pairwise_comparisons"]) == 3


def test_guided_pairwise_decisions_use_declared_alpha(unequal_groups):
    workflow = ResearchAssistant(unequal_groups).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
        options=AnalysisOptions(alpha=0.01),
        variable_types={"score": "continuous", "group": "nominal"},
    )
    assert all(item["alpha"] == 0.01 for item in workflow.analysis.values["pairwise_comparisons"])


def test_welch_and_classical_anova_sensitivity_compare_group_mean_vectors(unequal_groups):
    assistant = ResearchAssistant(unequal_groups)
    base = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
        variable_types={"score": "continuous", "group": "nominal"},
    ).analysis
    sensitivity = assistant.sensitivity_analysis(
        base,
        scenarios=[
            SensitivitySpecification(
                "classical equal-variance model",
                base.specification,
                method_id="one_way_anova",
                assumptions=("Equal population variances",),
            )
        ],
    )
    scenario = sensitivity.scenario_results[0]
    assert scenario.status.value == "completed"
    assert scenario.comparability.value == "same_estimand"
    assert scenario.comparison["available"] is True
    assert scenario.comparison["comparison_quantity"] == "group mean vector"


def test_constant_group_is_blocked_for_variance_weighted_methods():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "score": [1] * 5 + list(range(5)) + list(range(5, 10)),
        }
    )
    analyzer = StatisticalAnalyzer(frame)
    with pytest.raises(InsufficientDataError, match="positive, finite"):
        analyzer.welch_anova("group", "score")
    with pytest.raises(InsufficientDataError, match="positive, finite"):
        analyzer.games_howell("group", "score")


@pytest.mark.parametrize(
    "estimand,method", [("mean", "welch_anova"), ("distribution", "kruskal_wallis")]
)
def test_guided_workflow_report_audit_and_replay_round_trip(unequal_groups, estimand, method):
    original = unequal_groups.copy(deep=True)
    workflow = ResearchAssistant(unequal_groups).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand=estimand,
        variable_types={"score": "continuous", "group": "nominal"},
    )
    assert workflow.status.value in {"completed", "partial"}
    assert workflow.analysis.method_id == method
    assert len(workflow.analysis.values["pairwise_comparisons"]) == 3
    assert workflow.audit.status == "passed"
    assert "pairwise_comparisons" in workflow.report.to_csv_tables()
    assert "PAIRWISE FOLLOW-UP" in workflow.explain()
    serialized = json.loads(workflow.to_json())
    assert len(serialized["analysis"]["values"]["pairwise_comparisons"]) == 3
    reproduction = reproduce(workflow.reproducibility, data=unequal_groups)
    assert reproduction.status == "reproduced"
    pd.testing.assert_frame_equal(unequal_groups, original)


def test_extreme_shared_scale_remains_finite():
    base = 1e200
    frame = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "score": [
                base * factor
                for factor in (
                    1,
                    1.01,
                    1.02,
                    1.03,
                    1.04,
                    1.1,
                    1.12,
                    1.14,
                    1.16,
                    1.18,
                    0.9,
                    0.93,
                    0.96,
                    0.99,
                    1.02,
                )
            ],
        }
    )
    result = StatisticalAnalyzer(frame).welch_anova("group", "score")
    assert math.isfinite(result["statistic"])
    assert all(math.isfinite(item["estimate"]) for item in result["pairwise_comparisons"])
