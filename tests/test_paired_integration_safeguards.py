"""Targeted regression tests for paired integration safeguards."""

from dataclasses import replace

import pandas as pd
import pytest

from pyautostat import (
    MeaningfulEffectThreshold,
    ResearchAssistant,
    SensitivitySpecification,
)


@pytest.fixture
def paired_case():
    frame = pd.DataFrame(
        {
            "participant": [1, 1, 2, 2, 3, 3, 4, 4],
            "alternate_id": [101, 101, 102, 102, 103, 103, 104, 104],
            "condition": ["before", "after"] * 4,
            "score": [12.0, 9.0, 10.0, 8.0, 15.0, 11.0, 9.0, 8.0],
        }
    )
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="paired",
        estimand="mean",
        unit_id="participant",
        condition_order=("before", "after"),
        variable_types={"score": "continuous"},
    )
    return assistant, draft, assistant.analyze(draft)


def test_paired_sensitivity_identity_includes_unit_and_oriented_contrast(paired_case):
    assistant, _, base = paired_case
    same = SensitivitySpecification("same pairing", base.specification, method_id="paired_t")
    changed_unit = SensitivitySpecification(
        "different pairing",
        replace(base.specification, unit_id="alternate_id"),
        method_id="paired_t",
    )
    reversed_contrast = SensitivitySpecification(
        "reversed contrast",
        replace(base.specification, condition_order=("after", "before")),
        method_id="paired_t",
    )

    result = assistant.sensitivity_analysis(base, scenarios=[same, changed_unit, reversed_contrast])

    assert result.scenario_results[0].comparability == "same_estimand"
    assert result.scenario_results[1].status == "incompatible"
    assert result.scenario_results[1].comparability == "incompatible"
    assert "unit_id" in result.scenario_results[1].error
    assert result.scenario_results[2].status == "completed"
    assert result.scenario_results[2].comparability == "incompatible"
    assert result.scenario_results[2].comparison is None


def test_matching_positive_paired_threshold_uses_declared_contrast(paired_case):
    assistant, _, result = paired_case
    assessment = assistant.practical_significance(
        result,
        threshold=MeaningfulEffectThreshold(
            "mean_difference",
            1,
            direction="positive",
            contrast_order=("before", "after"),
        ),
    )
    assert assessment.status == "complete"
    assert assessment.point_estimate_relation == "meets_positive_threshold"
    assert assessment.provenance["result_contrast_order"] == ("before", "after")


def test_reversed_directional_paired_threshold_is_not_reused(paired_case):
    assistant, _, result = paired_case
    assessment = assistant.practical_significance(
        result,
        threshold=MeaningfulEffectThreshold(
            "mean_difference",
            1,
            direction="positive",
            contrast_order=("after", "before"),
        ),
    )
    assert assessment.status == "unavailable"
    assert assessment.point_estimate_relation == "unavailable"
    assert assessment.estimate is None
    assert "reversed paired contrast" in assessment.warnings[0]


def test_directional_paired_threshold_requires_an_orientation(paired_case):
    assistant, _, result = paired_case
    assessment = assistant.practical_significance(
        result,
        threshold=MeaningfulEffectThreshold("mean_difference", 1, direction="negative"),
    )
    assert assessment.status == "unavailable"
    assert "requires an explicit contrast_order" in assessment.warnings[0]


def test_two_sided_paired_magnitude_threshold_is_orientation_invariant(paired_case):
    assistant, draft, forward = paired_case
    reverse = assistant.analyze(
        assistant.update_question(draft, condition_order=("after", "before"))
    )
    threshold = MeaningfulEffectThreshold(
        "mean_difference",
        1,
        direction="two_sided",
        contrast_order=("before", "after"),
    )
    forward_assessment = assistant.practical_significance(forward, threshold=threshold)
    reverse_assessment = assistant.practical_significance(reverse, threshold=threshold)
    assert forward_assessment.status == reverse_assessment.status == "complete"
    assert forward_assessment.point_estimate_relation == "positive_meaningful_region"
    assert reverse_assessment.point_estimate_relation == "negative_meaningful_region"


def test_matching_negative_paired_threshold_uses_reversed_result_contrast(paired_case):
    assistant, draft, _ = paired_case
    reverse = assistant.analyze(
        assistant.update_question(draft, condition_order=("after", "before"))
    )
    assessment = assistant.practical_significance(
        reverse,
        threshold=MeaningfulEffectThreshold(
            "mean_difference",
            1,
            direction="negative",
            contrast_order=("after", "before"),
        ),
    )
    assert assessment.status == "complete"
    assert assessment.point_estimate_relation == "meets_negative_threshold"


@pytest.mark.parametrize(
    "coded_rows",
    [
        [(999, "before", 8.0), (999, "after", 7.0)],
        [
            (999, "before", 8.0),
            (999, "after", 7.0),
            (999, "before", 12.0),
            (999, "after", 9.0),
        ],
    ],
    ids=["would-create-pair", "would-merge-pairs"],
)
def test_declared_missing_code_unit_identifier_blocks_pairing(coded_rows):
    rows = [(1, "before", 5.0), (1, "after", 3.0), *coded_rows]
    frame = pd.DataFrame(rows, columns=["participant", "condition", "score"])
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="paired",
        estimand="mean",
        unit_id="participant",
        condition_order=("before", "after"),
        data_dictionary={
            "participant": {"type": "identifier", "missing_codes": [999]},
            "score": {"type": "continuous"},
        },
    )

    recommendation = assistant.recommend_test(draft)
    analysis = assistant.analyze(draft)

    assert recommendation.status == "unsupported"
    assert "'participant' contains" in recommendation.blockers[0]
    assert "Normalize them explicitly" in recommendation.blockers[0]
    assert analysis.status == "unavailable"
    assert any("missing-code" in warning for warning in analysis.warnings)


def test_plan_adherence_compares_performed_sensitivity_and_threshold():
    frame = pd.DataFrame(
        {
            "group": ["a"] * 6 + ["b"] * 6,
            "score": [1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 9],
        }
    )
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
        variable_types={"score": "continuous"},
    )
    result = assistant.analyze(draft)
    scenario = SensitivitySpecification(
        "equal variance",
        result.specification,
        method_id="student_t",
        assumptions=("Equal population variances",),
    )
    threshold = MeaningfulEffectThreshold("mean_difference", 1)
    plan = assistant.analysis_plan(
        draft,
        sensitivity_scenarios=[scenario],
        meaningful_threshold=threshold,
    )
    sensitivity = assistant.sensitivity_analysis(result, scenarios=[scenario])
    practical = assistant.practical_significance(result, threshold=threshold)

    adherence = assistant.plan_adherence(
        plan,
        result,
        sensitivity=sensitivity,
        practical_significance=practical,
    )
    fields = {item["field"]: item for item in adherence.comparisons}
    assert adherence.status == "matched"
    assert fields["sensitivity_scenarios"]["status"] == "matched"
    assert fields["meaningful_threshold"]["status"] == "matched"
    assert adherence.reason is None
    assert adherence.to_dict()["misconduct_inference"] is False

    changed_practical = assistant.practical_significance(
        result, threshold=MeaningfulEffectThreshold("mean_difference", 2)
    )
    changed = assistant.plan_adherence(
        plan,
        result,
        sensitivity=sensitivity,
        practical_significance=changed_practical,
    )
    changed_fields = {item["field"]: item for item in changed.comparisons}
    assert changed.status == "changed"
    assert changed_fields["meaningful_threshold"]["status"] == "changed"
    assert changed.reason is None


def test_plan_adherence_marks_unprovided_followups_not_recorded(paired_case):
    assistant, draft, result = paired_case
    scenario = SensitivitySpecification("repeat", result.specification, method_id="paired_t")
    threshold = MeaningfulEffectThreshold("mean_difference", 1)
    plan = assistant.analysis_plan(
        draft,
        sensitivity_scenarios=[scenario],
        meaningful_threshold=threshold,
    )

    adherence = assistant.plan_adherence(plan, result)
    fields = {item["field"]: item for item in adherence.comparisons}
    assert fields["sensitivity_scenarios"]["status"] == "not_recorded"
    assert fields["meaningful_threshold"]["status"] == "not_recorded"
