"""Phase 12 explicit paired-data contract and numerical validation."""

import json
import math

import pandas as pd
import pytest
from scipy import stats

from pyautostat import (
    AnalysisSpecification,
    InvalidDataError,
    ResearchAssistant,
    ResearchQuestion,
    StudyDesign,
    reproduce,
)


@pytest.fixture
def paired_frame():
    return pd.DataFrame(
        {
            "participant": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            "condition": ["before", "after"] * 5 + ["before"],
            "score": [10.0, 8.0, 9.0, 7.0, 11.0, 10.0, 8.0, 5.0, 13.0, 10.0, 20.0],
        }
    )


def _draft(assistant, **overrides):
    arguments = {
        "objective": "compare_groups",
        "outcome": "score",
        "predictor": "condition",
        "design": "paired",
        "estimand": "mean",
        "unit_id": "participant",
        "condition_order": ("before", "after"),
        "variable_types": {"score": "continuous"},
    }
    arguments.update(overrides)
    return assistant.prepare_question(**arguments)


def test_paired_numerics_orientation_effect_and_sample_accounting(paired_frame):
    assistant = ResearchAssistant(paired_frame)
    result = assistant.analyze(_draft(assistant))
    before = [10, 9, 11, 8, 13]
    after = [8, 7, 10, 5, 10]
    reference = stats.ttest_rel(before, after)
    differences = [left - right for left, right in zip(before, after, strict=True)]
    mean = sum(differences) / len(differences)
    sd = math.sqrt(sum((value - mean) ** 2 for value in differences) / 4)
    margin = stats.t.ppf(0.975, 4) * sd / math.sqrt(5)
    assert result.method_id == "paired_t"
    assert result.values["test_statistic"] == pytest.approx(reference.statistic)
    assert result.values["p_value"] == pytest.approx(reference.pvalue)
    assert result.values["primary_estimate"] == pytest.approx(mean)
    assert result.values["effect_size"]["value"] == pytest.approx(mean / sd)
    assert result.values["confidence_interval"]["lower"] == pytest.approx(mean - margin)
    assert result.values["confidence_interval"]["upper"] == pytest.approx(mean + margin)
    assert result.metadata["sample"]["complete_pairs"] == 5
    assert result.metadata["sample"]["incomplete_units"] == 1
    assert result.sample_size == 10
    assert result.excluded_rows == 1
    assert result.metadata["contrast"]["definition"] == "first condition minus second condition"


def test_reversing_condition_order_reverses_signed_quantities(paired_frame):
    assistant = ResearchAssistant(paired_frame)
    forward = assistant.analyze(_draft(assistant))
    reverse = assistant.analyze(_draft(assistant, condition_order=("after", "before")))
    assert reverse.values["primary_estimate"] == pytest.approx(-forward.values["primary_estimate"])
    assert reverse.values["test_statistic"] == pytest.approx(-forward.values["test_statistic"])
    assert reverse.values["effect_size"]["value"] == pytest.approx(
        -forward.values["effect_size"]["value"]
    )
    assert reverse.values["p_value"] == pytest.approx(forward.values["p_value"])


def test_complete_paired_workflow_report_audit_and_replay(paired_frame):
    assistant = ResearchAssistant(paired_frame)
    workflow = assistant.run(draft=_draft(assistant))
    assert workflow.status == "completed"
    assert workflow.audit.status == "passed"
    assert workflow.report.to_dict()["sections"]["dataset"]["complete_pairs"] == 5
    serialized = json.dumps(workflow.to_dict())
    assert 'participant": 1' not in serialized
    assert 'participant": 6' not in serialized
    assert reproduce(workflow.reproducibility, data=paired_frame).status == "reproduced"


def test_missing_unit_identifier_requests_input_without_inference(paired_frame):
    draft = ResearchAssistant(paired_frame).prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="paired",
        estimand="mean",
    )
    assert draft.status == "needs_input"
    assert draft.questions[0].field == "unit_id"


def test_duplicate_unit_condition_is_blocked(paired_frame):
    duplicate = pd.concat([paired_frame, paired_frame.iloc[[0]]], ignore_index=True)
    recommendation = ResearchAssistant(duplicate).recommend_test(
        _draft(ResearchAssistant(duplicate))
    )
    assert recommendation.status == "unsupported"
    assert "multiple usable observations" in recommendation.blockers[0]


def test_three_conditions_and_constant_differences_are_blocked():
    three = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "condition": ["a", "b", "c"] * 2,
            "score": [1, 2, 3, 2, 3, 4],
        }
    )
    assistant = ResearchAssistant(three)
    result = assistant.recommend_test(
        assistant.prepare_question(
            objective="compare_groups",
            outcome="score",
            predictor="condition",
            design="paired",
            estimand="mean",
            unit_id="id",
            variable_types={"score": "continuous"},
        )
    )
    assert result.status == "unsupported"
    assert "exactly two" in result.blockers[0]

    constant = pd.DataFrame(
        {"id": [1, 1, 2, 2], "condition": ["a", "b"] * 2, "score": [3, 1, 4, 2]}
    )
    assistant = ResearchAssistant(constant)
    result = assistant.recommend_test(
        assistant.prepare_question(
            objective="compare_groups",
            outcome="score",
            predictor="condition",
            design="paired",
            estimand="mean",
            unit_id="id",
            variable_types={"score": "continuous"},
        )
    )
    assert result.status == "unsupported"
    assert "nonzero variation" in result.blockers[0]


def test_paired_specification_schema_three_and_old_schemas_remain_readable():
    paired = AnalysisSpecification(
        question=ResearchQuestion(
            objective="compare_groups", outcome="score", predictor="condition", estimand="mean"
        ),
        design=StudyDesign.PAIRED,
        unit_id="participant",
        condition_order=("before", "after"),
    )
    payload = paired.to_dict()
    assert payload["schema_version"] == 3
    assert AnalysisSpecification.from_dict(payload).to_dict() == payload
    old = AnalysisSpecification().to_dict()
    assert old["schema_version"] == 1
    assert AnalysisSpecification.from_dict(old).to_dict() == old
    with pytest.raises(InvalidDataError, match="require schema_version 3"):
        AnalysisSpecification.from_dict({**old, "unit_id": "id"})
    with pytest.raises(InvalidDataError, match="scalar"):
        AnalysisSpecification(
            question=paired.question,
            design="paired",
            unit_id="participant",
            condition_order=(["before"], ["after"]),
        )


def test_numeric_condition_labels_serialize_and_design_change_clears_pairing():
    frame = pd.DataFrame(
        {"id": [1, 1, 2, 2, 3, 3], "condition": [0, 1] * 3, "score": [3, 1, 5, 2, 6, 4]}
    )
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="paired",
        estimand="mean",
        unit_id="id",
        variable_types={"score": "continuous", "condition": "nominal"},
    )
    assert assistant.recommend_test(draft).to_dict()["context"]["condition_order"] == ["0", "1"]
    revised = assistant.update_question(draft, design="independent")
    assert revised.specification.unit_id is None
    assert revised.specification.condition_order is None
