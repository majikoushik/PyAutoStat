"""Phase 4 question intake: scientific meaning stays explicit and serializable."""

import json

import pandas as pd
import pytest

from pyautostat import (
    AnalysisSpecification,
    ColumnNotFoundError,
    InvalidDataError,
    QuestionDraft,
    ResearchAssistant,
    StatisticalAnalyzer,
)


@pytest.fixture
def assistant():
    return ResearchAssistant(
        pd.DataFrame(
            {
                "score": [10.5, 11.5, 12.5, 13.5],
                "group": ["A", "A", "B", "B"],
                "hours": [1.1, 2.2, 3.3, 4.4],
            }
        )
    )


def fields(draft):
    return [item.field for item in draft.questions]


def test_descriptive_needs_no_inferential_details(assistant):
    draft = assistant.prepare_question(objective="descriptive")
    assert draft.status == "ready"
    assert draft.questions == ()
    assert draft.specification.design.value == "unknown"
    assert draft.specification.question.predictor is None
    assert "method_id" not in draft.to_dict()


def test_comparison_requires_only_target_and_design(assistant):
    draft = assistant.prepare_question(
        objective="compare_groups", outcome="score", predictor="group"
    )
    assert draft.status == "needs_input"
    assert fields(draft) == ["estimand", "design"]
    assert [item.field for item in draft.missing_information] == fields(draft)
    completed = assistant.update_question(draft, estimand="mean", design="independent")
    assert completed.status == "ready"
    assert completed.specification.question.estimand == "mean"
    assert completed.specification.design.value == "independent"
    assert completed.availability == {
        "columns": ["score", "group"],
        "available_rows": 4,
        "excluded_rows": 0,
        "total_rows": 4,
    }


@pytest.mark.parametrize(
    ("arguments", "missing"),
    [
        ({"predictor": "group", "estimand": "mean", "design": "independent"}, "outcome"),
        ({"outcome": "score", "estimand": "mean", "design": "independent"}, "predictor"),
        ({"outcome": "score", "predictor": "group", "estimand": "mean"}, "design"),
        ({"outcome": "score", "predictor": "group", "design": "independent"}, "estimand"),
    ],
)
def test_each_missing_comparison_field_is_requested(assistant, arguments, missing):
    draft = assistant.prepare_question(objective="compare_groups", **arguments)
    assert draft.status == "needs_input"
    assert fields(draft) == [missing]


def test_association_requires_two_variables_and_observation_design(assistant):
    draft = assistant.prepare_question(objective="association", outcome="hours", predictor="score")
    assert fields(draft) == ["design"]
    ready = assistant.update_question(draft, design="independent")
    assert ready.status == "ready"
    assert ready.specification.question.estimand is None
    assert "method_id" not in ready.to_dict()
    missing = assistant.prepare_question(
        objective="association", outcome="hours", design="independent"
    )
    assert fields(missing) == ["predictor"]
    options = next(item for item in missing.questions if item.field == "predictor")
    assert {item["value"] for item in options.to_dict()["options"]} == {"score", "group", "hours"}


def test_unknown_column_and_same_column_are_actionable(assistant):
    with pytest.raises(ColumnNotFoundError, match="unknown.*Available columns"):
        assistant.prepare_question(objective="association", outcome="score", predictor="unknown")
    with pytest.raises(InvalidDataError, match="different columns"):
        assistant.prepare_question(objective="association", outcome="score", predictor="score")
    with pytest.raises(InvalidDataError, match="non-empty string"):
        assistant.prepare_question(objective="association", outcome=" ")


def test_declared_type_overrides_integer_category_ambiguity():
    assistant = ResearchAssistant(
        pd.DataFrame({"rating": [1, 2, 3, 1], "group": ["A", "A", "B", "B"]})
    )
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="rating",
        predictor="group",
        design="independent",
        estimand="distribution",
    )
    assert fields(draft) == ["variable_types.rating"]
    assert draft.variable_suggestions["rating"]["type_source"] == "suggested"
    revised = assistant.update_question(draft, variable_types={"rating": "ordinal"})
    assert revised.status == "ready"
    assert revised.variable_suggestions["rating"]["suggested_type"] == "ordinal_categorical"
    assert revised.specification.data_dictionary == {"rating": {"type": "ordinal"}}
    assert revised.specification.to_dict()["schema_version"] == 2
    with pytest.raises(InvalidDataError, match="not numeric"):
        assistant.update_question(draft, variable_types={"group": "continuous"})


def test_profile_metadata_reused_but_copied():
    assistant = ResearchAssistant(
        pd.DataFrame({"rating": [1, 2, 3, 1], "group": ["A", "A", "B", "B"]})
    )
    profile = assistant.profile(data_dictionary={"rating": {"type": "ordinal"}})
    profile["data_dictionary"]["rating"]["type"] = "identifier"
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="rating",
        predictor="group",
        estimand="distribution",
        design="independent",
    )
    assert draft.status == "ready"
    assert draft.specification.data_dictionary == {"rating": {"type": "ordinal"}}


def test_identifier_warning_and_ordered_category():
    frame = pd.DataFrame(
        {
            "student_id": [101, 102, 103, 104],
            "rating": pd.Categorical(["low", "high", "low", "high"], ordered=True),
        }
    )
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="association",
        outcome="student_id",
        predictor="rating",
        design="independent",
    )
    assert "variable_types.student_id" in fields(draft)
    assert any("identifier" in warning for warning in draft.warnings)
    assert draft.variable_suggestions["rating"]["suggested_type"] == "ordinal_categorical"


def test_data_limits_and_overlapping_missing_counts():
    assistant = ResearchAssistant(
        pd.DataFrame(
            {
                "x": [1.1, None, None, 1.1],
                "group": ["A", None, "B", "B"],
            }
        )
    )
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="x",
        predictor="group",
        design="independent",
        estimand="mean",
    )
    assert draft.availability["excluded_rows"] == 2
    assert draft.availability["available_rows"] == 2
    assert draft.status == "ready"
    assert any("constant" in item for item in draft.warnings)
    all_missing = ResearchAssistant(pd.DataFrame({"x": [None, None], "group": ["A", "B"]}))
    blocked = all_missing.prepare_question(
        objective="compare_groups",
        outcome="x",
        predictor="group",
        design="independent",
        estimand="mean",
    )
    assert blocked.status == "data_limited"
    assert any("no observed values" in item for item in blocked.blockers)
    one_group = ResearchAssistant(pd.DataFrame({"x": [1.0, 2.0], "group": ["A", "A"]}))
    blocked = one_group.prepare_question(
        objective="compare_groups",
        outcome="x",
        predictor="group",
        design="independent",
        estimand="mean",
    )
    assert blocked.status == "data_limited"
    assert any("fewer than two" in item for item in blocked.blockers)


@pytest.mark.parametrize("design", ["paired", "repeated", "clustered"])
def test_designs_are_representable_without_execution(assistant, design):
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design=design,
        estimand="mean",
    )
    assert draft.status == "ready"
    assert draft.specification.design.value == design


def test_unknown_design_stays_unknown_and_options_have_stable_values(assistant):
    draft = assistant.prepare_question(
        objective="compare_groups", outcome="score", predictor="group", estimand="mean"
    )
    assert draft.specification.design.value == "unknown"
    options = next(item for item in draft.questions if item.field == "design").to_dict()["options"]
    assert {item["value"] for item in options} == {
        "independent",
        "paired",
        "repeated",
        "clustered",
        "unknown",
    }
    still_unknown = assistant.update_question(draft, design="unknown")
    assert still_unknown.status == "needs_input"


def test_progressive_update_and_objective_change(assistant):
    draft = assistant.prepare_question(
        objective="compare_groups", outcome="score", predictor="group"
    )
    halfway = assistant.update_question(draft, estimand="mean")
    assert fields(halfway) == ["design"]
    descriptive = assistant.update_question(halfway, objective="descriptive")
    assert descriptive.status == "ready"
    assert descriptive.specification.question.predictor is None
    assert descriptive.specification.question.estimand is None
    assert descriptive.specification.design.value == "unknown"
    association = assistant.update_question(descriptive, objective="association")
    assert fields(association) == ["outcome", "predictor", "design"]
    assert halfway.specification.question.estimand == "mean"


def test_invalid_update_preserves_original_draft(assistant):
    draft = assistant.prepare_question(objective="descriptive")
    before = draft.to_dict()
    with pytest.raises(ColumnNotFoundError):
        assistant.update_question(draft, objective="association", outcome="not_a_column")
    assert draft.to_dict() == before
    with pytest.raises(InvalidDataError, match="Unknown question updates"):
        assistant.update_question(draft, secret="x")
    with pytest.raises(InvalidDataError, match="design"):
        assistant.update_question(draft, objective="association", design="")


def test_specification_round_trip_and_legacy_schema(assistant):
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
        variable_types={"score": "continuous"},
        data_dictionary={"score": {"type": "continuous", "missing_codes": [999]}},
    )
    payload = draft.specification.to_dict()
    assert payload["schema_version"] == 2
    assert json.loads(json.dumps(draft.to_dict(), allow_nan=False))["status"] == "ready"
    restored = AnalysisSpecification.from_dict(json.loads(json.dumps(payload)))
    assert assistant.prepare_question(specification=restored).specification.to_dict() == payload
    legacy = assistant.prepare_question(objective="descriptive").specification.to_dict()
    assert legacy["schema_version"] == 1
    assert AnalysisSpecification.from_dict(legacy).to_dict() == legacy
    with pytest.raises(InvalidDataError, match="schema_version"):
        AnalysisSpecification.from_dict({**payload, "schema_version": 3})
    with pytest.raises(InvalidDataError, match="requires schema_version 2"):
        AnalysisSpecification.from_dict({**payload, "schema_version": 1})
    with pytest.raises(InvalidDataError, match="requires a data_dictionary"):
        AnalysisSpecification.from_dict({**payload, "data_dictionary": None})


def test_legacy_text_metadata_survives_phase4_migration(assistant):
    legacy = AnalysisSpecification(variable_metadata={"score": "test score"})
    original = legacy.to_dict()
    assert assistant.prepare_question(specification=legacy).specification.to_dict() == original
    migrated = assistant.prepare_question(
        specification=legacy, data_dictionary={"score": {"type": "continuous"}}
    ).specification
    assert migrated.to_dict()["schema_version"] == 2
    assert migrated.variable_metadata == {"score": "test score"}
    assert AnalysisSpecification.from_dict(migrated.to_dict()).to_dict() == migrated.to_dict()


def test_assistant_uses_its_construction_snapshot():
    frame = pd.DataFrame({"score": [1.1, 2.2], "group": ["A", "B"]})
    assistant = ResearchAssistant(frame)
    frame.loc[0, "score"] = None
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
    )
    assert draft.availability["available_rows"] == 2
    assert frame["score"].isna().sum() == 1


def test_original_frame_and_existing_profile_and_phase2_method_are_unchanged():
    frame = pd.DataFrame(
        {"group": ["A"] * 8 + ["B"] * 8, "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 2}
    )
    before = frame.copy(deep=True)
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
        variable_types={"score": "continuous"},
    )
    assert draft.status == "ready"
    assert assistant.profile()["overview"]["total_rows"] == 16
    assert (
        StatisticalAnalyzer(frame).hypothesis_tests(
            "group", "score", test_type="auto", estimand="mean", bootstrap_samples=0
        )["test"]
        == "t-test"
    )
    pd.testing.assert_frame_equal(frame, before)


def test_builder_does_not_run_statistical_tests(monkeypatch, assistant):
    def forbidden(*args, **kwargs):
        raise AssertionError("hypothesis test was called")

    monkeypatch.setattr(StatisticalAnalyzer, "hypothesis_tests", forbidden)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
    )
    assert draft.status == "ready"


def test_draft_contract_rejects_false_completion(assistant):
    spec = assistant.prepare_question(objective="descriptive").specification

    def make(status, *, questions=(), missing=(), blockers=()):
        return QuestionDraft(spec, status, missing, questions, (), blockers, {}, None)

    with pytest.raises(InvalidDataError, match="ready draft"):
        make("ready", blockers=("No usable rows",))
    with pytest.raises(InvalidDataError, match="needs_input"):
        make("needs_input")
    with pytest.raises(InvalidDataError, match="blocked draft"):
        make("unsupported")
    with pytest.raises(InvalidDataError, match="status"):
        make("fictional")


def test_invalid_builder_arguments_are_actionable(assistant):
    with pytest.raises(InvalidDataError, match="objective"):
        assistant.prepare_question(objective="predict")
    with pytest.raises(InvalidDataError, match="options"):
        assistant.prepare_question(objective="descriptive", options={"alpha": 0.05})
    with pytest.raises(InvalidDataError, match="variable_types"):
        assistant.prepare_question(objective="descriptive", variable_types=["continuous"])
    with pytest.raises(InvalidDataError, match="boolean"):
        assistant.prepare_question(
            objective="descriptive", data_dictionary={"score": {"type": "boolean"}}
        )
