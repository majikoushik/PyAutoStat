"""Structured result for the integrated, deterministic research workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .audit import AuditResult
from .exceptions import InvalidDataError
from .interpretation import InterpretationResult
from .question_builder import QuestionDraft
from .reproducibility import ReproducibilityRecord
from .research_report import ResearchReport
from .results import AnalysisResult, MissingInformation, Recommendation
from .specifications import AnalysisSpecification, _json_value


class WorkflowStatus(str, Enum):
    """Outcome of the integrated workflow, distinct from component statuses."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    NEEDS_INPUT = "needs_input"
    DATA_LIMITED = "data_limited"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


@dataclass(frozen=True)
class ResearchWorkflowResult:
    """One inspectable view of a guided workflow without embedding source data."""

    status: WorkflowStatus
    specification: AnalysisSpecification
    draft: QuestionDraft
    recommendation: Recommendation | None = None
    analysis: AnalysisResult | None = None
    interpretation: InterpretationResult | None = None
    report: ResearchReport | None = None
    audit: AuditResult | None = None
    reproducibility: ReproducibilityRecord | None = None
    profile: dict[str, Any] | None = None
    missing_information: tuple[MissingInformation, ...] = ()
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "status", WorkflowStatus(self.status))
        except (TypeError, ValueError) as exc:
            raise InvalidDataError("Research workflow status is invalid.") from exc
        if not isinstance(self.specification, AnalysisSpecification):
            raise InvalidDataError("workflow specification must be an AnalysisSpecification.")
        if not isinstance(self.draft, QuestionDraft):
            raise InvalidDataError("workflow draft must be a QuestionDraft.")
        if any(not isinstance(item, MissingInformation) for item in self.missing_information):
            raise InvalidDataError("workflow missing_information contains an invalid record.")
        if self.status is WorkflowStatus.NEEDS_INPUT and not self.missing_information:
            raise InvalidDataError("A needs_input workflow must identify missing information.")
        if (
            self.status
            in {
                WorkflowStatus.DATA_LIMITED,
                WorkflowStatus.UNSUPPORTED,
                WorkflowStatus.FAILED,
            }
            and not self.blockers
        ):
            raise InvalidDataError("A blocked or failed workflow must explain the blocker.")
        if self.status in {WorkflowStatus.COMPLETED, WorkflowStatus.PARTIAL} and (
            self.analysis is None or self.interpretation is None or self.report is None
        ):
            raise InvalidDataError("A completed or partial workflow requires its computed outputs.")

    def to_dict(self) -> dict[str, Any]:
        """Return schema-versioned JSON-safe metadata; source rows are never included."""
        return _json_value(
            {
                "schema_version": 1,
                "status": self.status.value,
                "specification": self.specification.to_dict(),
                "draft": self.draft.to_dict(),
                "recommendation": self.recommendation.to_dict()
                if self.recommendation is not None
                else None,
                "analysis": self.analysis.to_dict() if self.analysis is not None else None,
                "interpretation": self.interpretation.to_dict()
                if self.interpretation is not None
                else None,
                "report": self.report.to_dict() if self.report is not None else None,
                "audit": self.audit.to_dict() if self.audit is not None else None,
                "reproducibility": self.reproducibility.to_dict()
                if self.reproducibility is not None
                else None,
                "profile": self.profile,
                "missing_information": [item.to_dict() for item in self.missing_information],
                "blockers": self.blockers,
                "warnings": self.warnings,
            }
        )

    def to_json(self) -> str:
        """Serialize without nonfinite JSON constants."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
