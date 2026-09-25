"""Phase 12 analysis plans, provenance, adherence, and session payloads."""

import json

import pandas as pd
import pytest

from pyautostat import (
    AnalysisOptions,
    AnalysisSpecification,
    InvalidDataError,
    MeaningfulEffectThreshold,
    ResearchAssistant,
    ResearchQuestion,
    SensitivitySpecification,
    StatisticalAnalysisPlan,
    StudyPlanner,
)


@pytest.fixture
def assistant():
    return ResearchAssistant(
        pd.DataFrame(
            {
                "group": ["a"] * 6 + ["b"] * 6,
                "score": [1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 9],
            }
        )
    )


def _draft(assistant, *, options=None):
    return assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
        options=options,
        variable_types={"score": "continuous"},
    )


def test_analysis_plan_preserves_contract_without_executing(monkeypatch, assistant):
    draft = _draft(assistant)
    monkeypatch.setattr(
        assistant,
        "analyze",
        lambda *args, **kwargs: pytest.fail("plan creation must not execute an analysis"),
    )
    assistant.declare_planning("planned", reason="Defined before outcomes were analyzed")
    scenario = SensitivitySpecification(
        name="equal variance",
        specification=draft.specification,
        method_id="student_t",
        assumptions=("Equal population variances",),
        planning_status="planned",
    )
    threshold = MeaningfulEffectThreshold(
        quantity="mean_difference",
        minimum_magnitude=1.5,
        direction="two_sided",
        unit="points",
        rationale="Minimum decision-relevant difference",
        planning_status="planned",
    )
    plan = assistant.analysis_plan(
        draft,
        sensitivity_scenarios=[scenario],
        meaningful_threshold=threshold,
        multiplicity_policy="none_planned",
        report_style="apa",
    )
    payload = plan.to_dict()
    assert payload["status"] == "ready"
    assert payload["objective"] == "compare_groups"
    assert payload["primary_method_id"] == "welch_t"
    assert payload["estimand"] == "mean"
    assert payload["alpha"] == 0.05
    assert payload["confidence_level"] == 0.95
    assert payload["missing_data_policy"] == "analysis-specific complete cases"
    assert payload["outlier_rule"] == "No automatic outlier deletion."
    assert payload["researcher_planning_declaration"] == "planned"
    assert payload["planned_report_style"] == "apa"
    assert payload["planned_sensitivity_scenarios"][0]["name"] == "equal variance"
    assert payload["meaningful_threshold"]["minimum_magnitude"] == 1.5
    assert payload["created_after_analysis"] is False
    assert payload["provenance"]["external_preregistration_verified"] is False


def test_plan_round_trip_needs_input_and_historical_timing(assistant):
    incomplete = assistant.prepare_question(
        objective="compare_groups", outcome="score", predictor="group", estimand="mean"
    )
    pending = assistant.analysis_plan(incomplete)
    assert pending.status == "needs_input"
    assert (
        StatisticalAnalysisPlan.from_dict(json.loads(pending.to_json())).to_dict()
        == pending.to_dict()
    )
    result = assistant.analyze(_draft(assistant))
    assert result.status == "available"
    late = assistant.analysis_plan(_draft(assistant))
    assert late.created_after_analysis is True
    assert late.to_dict()["provenance"]["timing"] == "after_analysis"


def test_plan_revision_and_adherence_are_recorded_without_misconduct_claim(assistant):
    ledger = assistant.enable_tracking(clock=lambda: "2026-01-01T00:00:00Z")
    first = assistant.analysis_plan(_draft(assistant))
    revised = assistant.analysis_plan(
        _draft(assistant),
        previous_plan=first,
        reason="Changed intended presentation",
        report_style="ieee",
    )
    events = ledger.to_dict()["events"]
    update = next(item for item in events if item["event_type"] == "analysis_plan_updated")
    assert update["previous_state"] == first.to_dict()
    assert update["new_state"] == revised.to_dict()
    assert "planned_report_style" in update["metadata"]["changed_fields"]
    result = assistant.analyze(_draft(assistant, options=AnalysisOptions(alpha=0.01)))
    adherence = assistant.plan_adherence(first, result, reason="Alpha revised before execution")
    assert adherence.status == "changed"
    assert adherence.to_dict()["misconduct_inference"] is False
    alpha = next(item for item in adherence.comparisons if item["field"] == "alpha")
    assert alpha["status"] == "changed"


def test_invalid_plan_options_are_rejected(assistant):
    draft = _draft(assistant)
    with pytest.raises(InvalidDataError, match="multiplicity"):
        assistant.analysis_plan(draft, multiplicity_policy="bonferroni")
    with pytest.raises(InvalidDataError, match="requires multiplicity_method"):
        assistant.analysis_plan(draft, multiplicity_policy="planned_method")
    with pytest.raises(InvalidDataError, match="report_style"):
        assistant.analysis_plan(draft, report_style="journal")
    with pytest.raises(InvalidDataError, match="QuestionDraft"):
        assistant.analysis_plan({})


def test_unsupported_multiplicity_method_is_recorded_without_substitution(assistant):
    plan = assistant.analysis_plan(
        _draft(assistant),
        multiplicity_policy="planned_method",
        multiplicity_method="researcher-specified Holm procedure",
    )
    assert plan.to_dict()["multiplicity_method"] == "researcher-specified Holm procedure"
    assert any("not executed" in item for item in plan.limitations)


def test_session_snapshot_is_json_safe_and_advertises_only_state_valid_actions(assistant):
    plan = assistant.analysis_plan(_draft(assistant))
    workflow = assistant.run(draft=_draft(assistant))
    study = StudyPlanner().independent_mean_precision(sd_group1=2, sd_group2=3, target_half_width=1)
    completeness = assistant.reporting_completeness(workflow.report, style="ieee")
    snapshot = assistant.session_snapshot(
        workflow,
        analysis_plan=plan,
        study_planning=study,
        reporting_completeness=completeness,
    )
    payload = json.loads(snapshot.to_json())
    assert payload["schema_version"] == 1
    assert "run_analysis" not in payload["available_actions"]
    assert "run_sensitivity" in payload["available_actions"]
    assert payload["capabilities"]["gui_framework"] is None
    assert payload["capabilities"]["methods"]
    assert payload["analysis_plan"]["primary_method_id"] == "welch_t"
    assert "DataFrame" not in snapshot.to_json()


def test_assistant_study_planner_records_only_the_prospective_result(assistant):
    ledger = assistant.enable_tracking(clock=lambda: "2026-01-01T00:00:00Z")
    result = assistant.study_planner().paired_mean_precision(sd_difference=3, target_half_width=1)
    event = ledger.to_dict()["events"][-1]
    assert event["event_type"] == "study_planning_completed"
    assert event["metadata"]["status"] == result.status
    assert "analysis" not in event["references"]


def test_pending_snapshot_keeps_machine_renderable_question(assistant):
    workflow = assistant.run(
        objective="compare_groups", outcome="score", predictor="group", estimand="mean"
    )
    payload = assistant.session_snapshot(workflow).to_dict()
    assert payload["questions"][0]["field"] == "design"
    assert payload["questions"][0]["input_type"] == "select"
    assert "provide_missing_information" in payload["available_actions"]
    assert "run_analysis" not in payload["available_actions"]
    with pytest.raises(InvalidDataError, match="ResearchWorkflowResult"):
        assistant.session_snapshot({})


def test_plan_adherence_reports_not_recorded_fields():
    specification = AnalysisSpecification(question=ResearchQuestion(objective="descriptive"))
    assistant = ResearchAssistant(pd.DataFrame({"x": [1, 2, 3]}))
    plan = assistant.analysis_plan(specification)
    result = assistant.analyze(specification=specification)
    adherence = assistant.plan_adherence(plan, result)
    outcome = next(item for item in adherence.comparisons if item["field"] == "outcome")
    assert outcome["status"] == "not_recorded"
