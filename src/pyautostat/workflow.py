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

    def explain(self) -> str:
        """Return a human-readable summary of the entire workflow result.

        Aggregates status, method, hypothesis decision, effect size, confidence
        interval, assumptions, and limitations from the already-computed
        ``interpretation`` and ``analysis`` attributes into one readable text
        block.  No new statistics are calculated.

        Example usage::

            workflow = assistant.run(objective='compare_groups', ...)
            print(workflow.explain())

        Returns
        -------
        str
            A multi-section plain-text summary suitable for printing.
        """
        sep = "=" * 68
        thin = "-" * 68
        lines: list[str] = [sep]

        # ── Status & method ──────────────────────────────────────────────────
        status_val = self.status.value.upper().replace("_", " ")
        method_label = "Not yet determined"
        if self.analysis is not None:
            method_label = self.analysis.method_label
        elif self.recommendation is not None and self.recommendation.method_label:
            method_label = self.recommendation.method_label

        spec = self.specification
        outcome = spec.question.outcome or "not provided"
        predictor = spec.question.predictor or "not provided"
        lines.append(f" ANALYSIS RESULT | {method_label}")
        lines.append(sep)
        lines.append(f" Status    : {status_val}")
        lines.append(f" Outcome   : {outcome}")
        lines.append(f" Predictor : {predictor}")
        if self.analysis is not None and self.analysis.sample_size is not None:
            excl = self.analysis.excluded_rows or 0
            lines.append(
                f" Sample    : {self.analysis.sample_size} rows analysed"
                + (f" ({excl} excluded)" if excl else "")
            )

        # ── Needs-input / blocked ────────────────────────────────────────────
        if self.missing_information:
            lines.append(thin)
            lines.append(" MISSING INFORMATION - the analysis cannot run yet")
            questions: dict[str, Any] = {item.field: item for item in self.draft.questions}
            if self.recommendation is not None:
                questions.update(
                    {
                        item["field"]: item
                        for item in self.recommendation.questions
                        if isinstance(item, dict) and isinstance(item.get("field"), str)
                    }
                )
            for item in self.missing_information:
                question = questions.get(item.field)
                lines.append(f"   Field '{item.field}': {item.message}")
                if question is not None:
                    explanation = (
                        question.get("explanation")
                        if isinstance(question, dict)
                        else question.explanation
                    )
                    options = (
                        tuple(
                            (option.get("value"), option.get("label"))
                            for option in question.get("options", ())
                            if isinstance(option, dict)
                        )
                        if isinstance(question, dict)
                        else question.options
                    )
                    if explanation:
                        lines.append(f"     Why: {explanation}")
                    if options:
                        choices = ", ".join(f"{value!r} ({label})" for value, label in options)
                        lines.append(f"     Choices: {choices}")
            fix_args = ", ".join(f"{item.field}=<value>" for item in self.missing_information)
            lines.append(f"   Fix directly: assistant.run(..., {fix_args})")
            lines.append(
                "   Or update the draft: revised = assistant.update_question(result.draft, ...)"
            )
            lines.append("                        assistant.run(draft=revised)")

        if self.blockers:
            lines.append(thin)
            lines.append(" BLOCKED")
            for blocker in self.blockers:
                lines.append(f"   BLOCKER: {blocker}")

        # ── Interpretation sections ──────────────────────────────────────────
        interp = self.interpretation
        if interp is not None:
            if interp.hypothesis_interpretation:
                lines.append(thin)
                lines.append(" HYPOTHESIS TEST")
                lines.append(f"   {interp.hypothesis_interpretation.strip()}")
            if interp.effect_interpretation:
                lines.append(thin)
                lines.append(" EFFECT SIZE")
                lines.append(f"   {interp.effect_interpretation.strip()}")
            if interp.uncertainty_interpretation:
                lines.append(thin)
                lines.append(" CONFIDENCE INTERVAL")
                lines.append(f"   {interp.uncertainty_interpretation.strip()}")
            if interp.assumption_notes:
                lines.append(thin)
                lines.append(" ASSUMPTIONS")
                for note in interp.assumption_notes:
                    lines.append(f"   - {note}")
            if interp.limitations:
                lines.append(thin)
                lines.append(" LIMITATIONS")
                for lim in interp.limitations:
                    lines.append(f"   - {lim}")

        # ── Warnings ─────────────────────────────────────────────────────────
        if self.warnings:
            lines.append(thin)
            lines.append(" WARNINGS")
            for warning in self.warnings:
                lines.append(f"   ! {warning}")

        lines.append(sep)
        return "\n".join(lines)

    def __str__(self) -> str:
        """Delegate to :meth:`explain` for natural ``print()`` behaviour."""
        return self.explain()
