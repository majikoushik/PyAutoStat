"""JSON-only adapter contract for future notebook, CLI, or GUI clients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .analysis_plan import StatisticalAnalysisPlan
from .completeness import ReportingCompletenessResult
from .exceptions import InvalidDataError
from .practical_significance import PracticalSignificanceResult
from .recommendation import METHOD_CAPABILITIES
from .research_report import REPORT_STYLES
from .sensitivity import SensitivityResult
from .specifications import _json_value
from .study_planning import STUDY_PLANNING_CAPABILITIES, StudyPlanningResult
from .workflow import ResearchWorkflowResult


def capability_payload() -> dict[str, Any]:
    methods = [
        {
            "method_id": item.identifier,
            "name": item.name,
            "objective": item.objective,
            "target": item.target,
            "designs": item.designs,
            "availability": item.availability,
        }
        for item in METHOD_CAPABILITIES.values()
    ]
    return _json_value(
        {
            "objectives": ["descriptive", "compare_groups", "association"],
            "designs": ["independent", "paired", "repeated", "clustered"],
            "methods": methods,
            "report_styles": REPORT_STYLES,
            "exports": ["json", "csv", "html", "markdown", "latex"],
            "study_planning": STUDY_PLANNING_CAPABILITIES,
            "sensitivity": True,
            "practical_significance": True,
            "gui_framework": None,
        }
    )


@dataclass(frozen=True)
class ResearchSessionSnapshot:
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, dict) or self.payload.get("schema_version") != 1:
            raise InvalidDataError("ResearchSessionSnapshot schema_version must be 1.")
        object.__setattr__(self, "payload", _json_value(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self.payload)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)


def build_session_snapshot(
    workflow: ResearchWorkflowResult,
    *,
    sensitivity: SensitivityResult | None = None,
    practical_significance: PracticalSignificanceResult | None = None,
    analysis_plan: StatisticalAnalysisPlan | None = None,
    study_planning: StudyPlanningResult | None = None,
    reporting_completeness: ReportingCompletenessResult | None = None,
) -> ResearchSessionSnapshot:
    if not isinstance(workflow, ResearchWorkflowResult):
        raise InvalidDataError("session snapshot requires a ResearchWorkflowResult.")
    expected_types = (
        ("sensitivity", sensitivity, SensitivityResult),
        ("practical_significance", practical_significance, PracticalSignificanceResult),
        ("analysis_plan", analysis_plan, StatisticalAnalysisPlan),
        ("study_planning", study_planning, StudyPlanningResult),
        ("reporting_completeness", reporting_completeness, ReportingCompletenessResult),
    )
    for name, value, expected in expected_types:
        if value is not None and not isinstance(value, expected):
            raise InvalidDataError(f"{name} has an invalid session snapshot type.")
    workflow_payload = workflow.to_dict()
    questions = workflow_payload.get("draft", {}).get("questions", [])
    blockers = list(workflow_payload.get("blockers", []))
    warnings = list(workflow_payload.get("warnings", []))
    status = workflow_payload["status"]
    actions = ["update_specification", "create_analysis_plan", "open_study_planner"]
    if status == "needs_input":
        actions.insert(0, "provide_missing_information")
    if status in {"completed", "partial"}:
        actions.extend(
            [
                "run_sensitivity",
                "assess_practical_significance",
                "generate_report",
                "save_report",
                "audit_report",
                "create_reproducibility_record",
            ]
        )
    elif status not in {"data_limited", "unsupported", "failed"} and not questions:
        actions.append("run_analysis")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "capabilities": capability_payload(),
        "workflow": workflow_payload,
        "questions": questions,
        "available_actions": actions,
        "warnings": warnings,
        "blockers": blockers,
    }
    optional = {
        "analysis_plan": analysis_plan,
        "study_planning": study_planning,
        "sensitivity": sensitivity,
        "practical_significance": practical_significance,
        "reporting_completeness": reporting_completeness,
    }
    for key, value in optional.items():
        if value is not None:
            payload[key] = value.to_dict()
    if workflow.report is not None:
        payload["reporting"] = workflow.report.to_dict()
    if workflow.audit is not None:
        payload["audit"] = workflow.audit.to_dict()
    if workflow.reproducibility is not None:
        payload["reproducibility"] = workflow.reproducibility.to_dict()
    return ResearchSessionSnapshot(_json_value(payload))
