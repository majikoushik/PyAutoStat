"""Phase 10 end-to-end checks for the controlled MVP workflow."""

import json
import math
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from pyautostat import (
    AnalysisResult,
    AnalysisSpecification,
    AnalysisStatus,
    AuditFinding,
    AuditResult,
    InvalidDataError,
    ResearchAssistant,
    ResearchWorkflowResult,
    WorkflowStatus,
)


@pytest.fixture
def comparison_frame():
    return pd.DataFrame(
        {
            "exam_score": [70.0, 72, 68, 71, 69, 74, 82, 85, 80, 84, 83, 81, None],
            "teaching_method": ["A"] * 6 + ["B"] * 7,
            "private_id": [f"participant-{index}" for index in range(13)],
        }
    )


def _mean_arguments():
    return {
        "objective": "compare_groups",
        "outcome": "exam_score",
        "predictor": "teaching_method",
        "estimand": "mean",
        "design": "independent",
        "variable_types": {"exam_score": "continuous"},
    }


def test_guided_mean_workflow_preserves_one_canonical_numerical_result(comparison_frame):
    original = comparison_frame.copy(deep=True)
    workflow = ResearchAssistant(comparison_frame).run(**_mean_arguments())

    assert isinstance(workflow, ResearchWorkflowResult)
    assert workflow.status is WorkflowStatus.COMPLETED
    assert workflow.recommendation.method_id == workflow.analysis.method_id == "welch_t"
    assert workflow.analysis.sample_size == 12
    assert workflow.analysis.excluded_rows == 1
    assert workflow.analysis.metadata["group_order"] == ["A", "B"]
    assert workflow.analysis.values["primary_estimate"] == pytest.approx(-11.833333333333329)
    assert workflow.analysis.values["effect_size"]["value"] < 0
    interval = workflow.analysis.values["confidence_interval"]
    assert interval["quantity"] == "mean difference"
    assert interval["lower"] < interval["upper"]
    assert workflow.interpretation.metadata["primary_estimate"] == pytest.approx(
        workflow.analysis.values["primary_estimate"]
    )
    assert workflow.report.to_dict()["analysis"] == workflow.analysis.to_dict()
    assert workflow.audit.status == "passed"
    assert workflow.reproducibility.method_id == "welch_t"
    assert workflow.reproducibility.to_dict()["stochastic"]["effective_seed"] == 0
    pd.testing.assert_frame_equal(comparison_frame, original)


def test_missing_design_is_actionable_and_can_continue_without_reentry(comparison_frame):
    assistant = ResearchAssistant(comparison_frame)
    incomplete = assistant.run(
        objective="compare_groups",
        outcome="exam_score",
        predictor="teaching_method",
        estimand="mean",
        variable_types={"exam_score": "continuous"},
    )

    assert incomplete.status is WorkflowStatus.NEEDS_INPUT
    assert incomplete.analysis is incomplete.report is None
    design = next(item for item in incomplete.draft.questions if item.field == "design")
    assert dict(design.options)["independent"] == "Independent observations"
    assert "cannot be established" in design.explanation

    revised = assistant.update_question(incomplete.draft, design="independent")
    completed = assistant.run(draft=revised)
    assert completed.status is WorkflowStatus.COMPLETED
    assert completed.specification.question.outcome == "exam_score"
    assert completed.specification.question.estimand == "mean"


def test_supplied_draft_and_specification_are_supported_but_cannot_conflict(comparison_frame):
    assistant = ResearchAssistant(comparison_frame)
    draft = assistant.prepare_question(**_mean_arguments())
    assert assistant.run(draft=draft).analysis.method_id == "welch_t"
    assert assistant.run(specification=draft.specification).analysis.method_id == "welch_t"
    with pytest.raises(InvalidDataError, match="cannot be combined"):
        assistant.run(draft=draft, design="independent")
    with pytest.raises(InvalidDataError, match="either draft or specification"):
        assistant.run(draft=draft, specification=draft.specification)
    with pytest.raises(InvalidDataError, match="AnalysisSpecification"):
        assistant.run(specification={})
    with pytest.raises(InvalidDataError, match="QuestionDraft"):
        assistant.run(draft={})


@pytest.mark.parametrize("argument", ["include_profile", "audit", "fingerprint", "include_figures"])
def test_workflow_controls_require_booleans(comparison_frame, argument):
    with pytest.raises(InvalidDataError, match=f"{argument} must be a Boolean"):
        ResearchAssistant(comparison_frame).run(**{**_mean_arguments(), argument: "yes"})


@pytest.mark.parametrize("design", ["paired", "repeated", "clustered"])
def test_unsupported_dependent_designs_never_execute(comparison_frame, monkeypatch, design):
    assistant = ResearchAssistant(comparison_frame)
    monkeypatch.setattr(
        assistant,
        "analyze",
        lambda *args, **kwargs: pytest.fail("an independent analysis must not execute"),
    )
    workflow = assistant.run(**{**_mean_arguments(), "design": design})
    assert workflow.status is WorkflowStatus.UNSUPPORTED
    assert workflow.analysis is workflow.report is None
    assert workflow.blockers


def test_three_group_mean_target_is_not_switched_to_rank_test(monkeypatch):
    frame = pd.DataFrame({"score": list(range(18)), "group": ["A"] * 6 + ["B"] * 6 + ["C"] * 6})
    assistant = ResearchAssistant(frame)
    monkeypatch.setattr(
        assistant,
        "analyze",
        lambda *args, **kwargs: pytest.fail("an unsupported analysis must not execute"),
    )
    workflow = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )
    assert workflow.status is WorkflowStatus.UNSUPPORTED
    assert workflow.specification.question.estimand == "mean"
    assert workflow.recommendation.method_id is None
    assert "Kruskal" not in " ".join(workflow.blockers)


@pytest.mark.parametrize(
    "frame, expected_text",
    [
        (
            pd.DataFrame({"score": [1.0, 2.0], "group": ["A", "A"]}),
            "fewer than two observed categories",
        ),
        (
            pd.DataFrame({"score": [float("nan"), float("nan")], "group": ["A", "B"]}),
            "no observed values",
        ),
    ],
)
def test_data_limits_stop_before_recommendation(frame, expected_text):
    workflow = ResearchAssistant(frame).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )
    assert workflow.status is WorkflowStatus.DATA_LIMITED
    assert workflow.recommendation is workflow.analysis is workflow.report is None
    assert expected_text in " ".join(workflow.blockers)


@pytest.mark.parametrize(
    "frame, arguments",
    [
        (
            pd.DataFrame({"score": [1.0, 1, 2, 2], "group": ["A", "A", "B", "B"]}),
            dict(
                objective="compare_groups",
                outcome="score",
                predictor="group",
                estimand="mean",
                design="independent",
                variable_types={"score": "continuous"},
            ),
        ),
        (
            pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 3.0]}),
            dict(
                objective="association",
                outcome="x",
                predictor="y",
                estimand="linear",
                design="independent",
                variable_types={"x": "continuous", "y": "continuous"},
            ),
        ),
        (
            pd.DataFrame({"x": ["a", "a", "b", "b"], "y": ["u", "v", "u", "v"]}),
            dict(
                objective="association",
                outcome="x",
                predictor="y",
                estimand="categorical_independence",
                design="independent",
                variable_types={"x": "nominal", "y": "nominal"},
            ),
        ),
    ],
)
def test_recommendation_stage_data_limits_have_a_distinct_status(frame, arguments):
    workflow = ResearchAssistant(frame).run(**arguments)
    assert workflow.status is WorkflowStatus.DATA_LIMITED
    assert workflow.recommendation is not None
    assert workflow.analysis is workflow.report is None


def test_ambiguous_numeric_outcome_requests_a_measurement_declaration():
    frame = pd.DataFrame({"score_code": [1, 2] * 6, "group": ["A"] * 6 + ["B"] * 6})
    workflow = ResearchAssistant(frame).run(
        objective="compare_groups",
        outcome="score_code",
        predictor="group",
        estimand="mean",
        design="independent",
    )
    assert workflow.status is WorkflowStatus.NEEDS_INPUT
    assert "variable_types.score_code" in {item.field for item in workflow.missing_information}


def test_mixed_type_association_requests_objective_clarification():
    frame = pd.DataFrame({"score": range(12), "group": ["A"] * 6 + ["B"] * 6})
    workflow = ResearchAssistant(frame).run(
        objective="association",
        outcome="score",
        predictor="group",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
    )
    assert workflow.status is WorkflowStatus.NEEDS_INPUT
    assert workflow.recommendation is not None
    assert workflow.missing_information[0].field == "objective"
    assert workflow.analysis is None


@pytest.mark.parametrize(
    "frame, arguments, method, expected_status",
    [
        (
            pd.DataFrame({"score": list(range(16)), "group": ["A"] * 8 + ["B"] * 8}),
            dict(
                objective="compare_groups",
                outcome="score",
                predictor="group",
                estimand="distribution",
                design="independent",
                variable_types={"score": "continuous"},
            ),
            "mann_whitney_u",
            "completed",
        ),
        (
            pd.DataFrame({"score": list(range(18)), "group": ["A"] * 6 + ["B"] * 6 + ["C"] * 6}),
            dict(
                objective="compare_groups",
                outcome="score",
                predictor="group",
                estimand="distribution",
                design="independent",
                variable_types={"score": "continuous"},
            ),
            "kruskal_wallis",
            "completed",
        ),
        (
            pd.DataFrame({"x": range(10), "y": [1.0, 2, 4, 3, 5, 7, 6, 8, 10, 9]}),
            dict(
                objective="association",
                outcome="x",
                predictor="y",
                estimand="linear",
                design="independent",
                variable_types={"x": "continuous", "y": "continuous"},
            ),
            "pearson_correlation",
            "partial",
        ),
        (
            pd.DataFrame(
                {
                    "method": ["A"] * 20 + ["B"] * 20,
                    "choice": ["yes"] * 12 + ["no"] * 8 + ["yes"] * 8 + ["no"] * 12,
                }
            ),
            dict(
                objective="association",
                outcome="choice",
                predictor="method",
                estimand="categorical_independence",
                design="independent",
                variable_types={"choice": "nominal", "method": "nominal"},
            ),
            "pearson_chi_square",
            "completed",
        ),
    ],
)
def test_supported_guided_method_matrix(frame, arguments, method, expected_status):
    workflow = ResearchAssistant(frame).run(**arguments)
    assert workflow.status.value == expected_status
    assert workflow.recommendation.method_id == workflow.analysis.method_id == method
    assert workflow.report is not None
    assert workflow.audit.status == "passed"


def test_descriptive_run_reuses_its_profile_and_optional_profile_is_not_duplicated(
    comparison_frame, monkeypatch
):
    descriptive = ResearchAssistant(comparison_frame).run(objective="descriptive")
    assert descriptive.status is WorkflowStatus.COMPLETED
    assert descriptive.analysis.method_id == "dataset_profile"
    assert descriptive.profile == descriptive.analysis.values["profile"]

    assistant = ResearchAssistant(comparison_frame)
    calls = 0
    original = assistant.profile

    def counted_profile(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(assistant, "profile", counted_profile)
    inferential = assistant.run(include_profile=True, **_mean_arguments())
    assert inferential.profile is not None
    assert calls == 1


def test_analysis_executes_once_and_downstream_stages_do_not_replay(comparison_frame, monkeypatch):
    assistant = ResearchAssistant(comparison_frame)
    executions = 0
    original = assistant._analyzer.hypothesis_tests

    def counted_backend(*args, **kwargs):
        nonlocal executions
        executions += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(assistant._analyzer, "hypothesis_tests", counted_backend)
    workflow = assistant.run(**_mean_arguments())
    assert workflow.status is WorkflowStatus.COMPLETED
    assert executions == 1


def test_audit_controls_workflow_status_without_claiming_a_pass(comparison_frame, monkeypatch):
    disabled = ResearchAssistant(comparison_frame).run(audit=False, **_mean_arguments())
    assert disabled.status is WorkflowStatus.PARTIAL
    assert disabled.audit is None
    assert any("no audit was performed" in item for item in disabled.warnings)

    assistant = ResearchAssistant(comparison_frame)
    failed = AuditResult(
        "failed",
        (
            AuditFinding(
                "VALUE_MISMATCH",
                "error",
                "report",
                "results.primary_estimate",
                -1.0,
                1.0,
                "The report contradicts the canonical estimate.",
            ),
        ),
        ("analysis", "report"),
        (),
        {},
    )
    monkeypatch.setattr(assistant, "audit", lambda *args, **kwargs: failed)
    workflow = assistant.run(**_mean_arguments())
    assert workflow.status is WorkflowStatus.FAILED
    assert workflow.audit is failed
    assert "contradicts" in workflow.blockers[0]

    incomplete_audit = AuditResult("incomplete", (), ("analysis",), ("exports",), {})
    monkeypatch.setattr(assistant, "audit", lambda *args, **kwargs: incomplete_audit)
    incomplete = assistant.run(**_mean_arguments())
    assert incomplete.status is WorkflowStatus.PARTIAL
    assert incomplete.audit is incomplete_audit
    assert any("audit was incomplete" in item for item in incomplete.warnings)


def test_unavailable_execution_stops_before_interpretation(comparison_frame, monkeypatch):
    assistant = ResearchAssistant(comparison_frame)

    def unavailable(draft, **kwargs):
        return AnalysisResult(
            method_id="welch_t",
            status=AnalysisStatus.UNAVAILABLE,
            warnings=("The backend returned a nonfinite result.",),
            specification=draft.specification,
            recommendation=assistant.recommend_test(draft),
        )

    monkeypatch.setattr(assistant, "analyze", unavailable)
    monkeypatch.setattr(
        assistant,
        "interpret",
        lambda *args: pytest.fail("an unavailable execution must not be interpreted"),
    )
    workflow = assistant.run(**_mean_arguments())
    assert workflow.status is WorkflowStatus.FAILED
    assert workflow.analysis.status is AnalysisStatus.UNAVAILABLE
    assert workflow.interpretation is workflow.report is None


def test_tracking_records_each_actual_stage_once(comparison_frame):
    assistant = ResearchAssistant(comparison_frame)
    ledger = assistant.enable_tracking(clock=lambda: "phase-10-test")
    workflow = assistant.run(**_mean_arguments())
    assert workflow.status is WorkflowStatus.COMPLETED
    assert [item["event_type"] for item in ledger.events] == [
        "question_prepared",
        "method_recommended",
        "analysis_executed",
        "interpretation_generated",
        "report_generated",
        "audit_performed",
    ]
    assert ResearchAssistant(comparison_frame).run(**_mean_arguments()).draft is not None


def test_workflow_is_json_safe_private_and_does_not_write_files(comparison_frame, tmp_path):
    before = set(tmp_path.iterdir())
    workflow = ResearchAssistant(comparison_frame).run(**_mean_arguments())
    payload = workflow.to_dict()
    json.dumps(payload, allow_nan=False)
    assert json.loads(workflow.to_json())["schema_version"] == 1
    assert "participant-0" not in workflow.to_json()
    assert set(tmp_path.iterdir()) == before
    assert "private_id" not in workflow.reproducibility.to_json()


def test_existing_phase7_interval_regressions_survive_integrated_contract(comparison_frame):
    workflow = ResearchAssistant(comparison_frame).run(**_mean_arguments())
    altered = deepcopy(workflow.analysis.values["effect_size"]["confidence_interval"])
    assert altered["method"] == "independent within-group percentile bootstrap"
    # A bootstrap interval's validity is based on its finite ordered bounds, not containment.
    assert all(math.isfinite(altered[key]) for key in ("lower", "upper"))
    assert altered["lower"] <= altered["upper"]


def test_schema_and_public_exports_are_additive():
    assert AnalysisSpecification().to_dict()["schema_version"] == 1
    assert WorkflowStatus("completed") is WorkflowStatus.COMPLETED


def test_workflow_schema_rejects_inconsistent_statuses(comparison_frame):
    valid = ResearchAssistant(comparison_frame).run(**_mean_arguments())
    with pytest.raises(InvalidDataError, match="workflow specification"):
        replace(valid, specification={})
    with pytest.raises(InvalidDataError, match="workflow draft"):
        replace(valid, draft={})
    with pytest.raises(InvalidDataError, match="invalid record"):
        replace(valid, missing_information=("design",))
    with pytest.raises(InvalidDataError, match="identify missing information"):
        replace(valid, status="needs_input", missing_information=())
    with pytest.raises(InvalidDataError, match="explain the blocker"):
        replace(valid, status="unsupported", blockers=())
    with pytest.raises(InvalidDataError, match="requires its computed outputs"):
        replace(valid, status="partial", report=None)
