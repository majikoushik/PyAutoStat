"""Public entry point for profiling, question intake, and method recommendations."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from typing import Any, cast

import pandas as pd

from .analyzer import StatisticalAnalyzer
from .audit import AuditResult, StatisticalResultAuditor
from .decision_ledger import DecisionLedger
from .exceptions import InvalidDataError
from .execution import execute_specification
from .interpretation import InterpretationEngine, InterpretationResult
from .profiling import complete_case_count
from .provenance import content_reference
from .question_builder import QuestionDraft, prepare_question
from .recommendation import recommend_from_draft
from .reproducibility import ReproducibilityRecord
from .research_report import ResearchReport, build_research_report
from .results import AnalysisResult, Recommendation
from .specifications import (
    AnalysisOptions,
    AnalysisSpecification,
    Objective,
    ResearchQuestion,
    StudyDesign,
)
from .workflow import ResearchWorkflowResult, WorkflowStatus


class ResearchAssistant:
    """Coordinate profiling, question intake, and deterministic recommendations.

    Construction uses StatisticalAnalyzer's validation and private DataFrame copy.
    It does not run the profiling calculations until :meth:`profile` is called.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._analyzer = StatisticalAnalyzer(df)
        self._data_dictionary: dict | None = None
        self._ledger: DecisionLedger | None = None
        self._planning_status = "unknown"

    def enable_tracking(
        self, *, clock: Callable[[], datetime | str] | None = None
    ) -> DecisionLedger:
        """Start observing subsequent actions; no earlier decisions are reconstructed."""
        if self._ledger is None:
            self._ledger = DecisionLedger(clock=clock)
        return self._ledger

    @property
    def decision_ledger(self) -> DecisionLedger | None:
        return self._ledger

    @property
    def planning_status(self) -> str:
        return self._planning_status

    def declare_planning(self, status: str, *, reason: str | None = None) -> None:
        """Record a researcher declaration, never independent preregistration proof."""
        if status not in {"planned", "exploratory", "unknown"}:
            raise InvalidDataError("planning status must be planned, exploratory, or unknown.")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise InvalidDataError("A supplied planning reason must be non-empty text.")
        previous = self._planning_status
        if self._ledger is not None:
            self._ledger._record(
                "planning_declared",
                source="researcher",
                previous_state=previous,
                new_state=status,
                reason=reason,
            )
        self._planning_status = status

    def profile(
        self, *, data_dictionary=None, histogram_bins: int = 20, include_row_positions: bool = False
    ) -> dict:
        """Profile a DataFrame; optional declarations never mutate source values."""
        result = self._analyzer.analyze_all(
            data_dictionary=data_dictionary,
            histogram_bins=histogram_bins,
            include_row_positions=include_row_positions,
        )
        self._data_dictionary = deepcopy(result["data_dictionary"])
        return result

    def complete_case_count(self, columns: list[str]) -> dict:
        """Count rows available for a specified set of columns."""
        return complete_case_count(self._analyzer.df, columns)

    def run(
        self,
        *,
        objective: str | Objective | None = None,
        outcome: str | None = None,
        predictor: str | None = None,
        design: str | StudyDesign | None = None,
        estimand: str | None = None,
        description: str | None = None,
        options: AnalysisOptions | None = None,
        data_dictionary: dict | None = None,
        variable_types: dict[str, str] | None = None,
        draft: QuestionDraft | None = None,
        specification: AnalysisSpecification | None = None,
        include_profile: bool = False,
        audit: bool = True,
        fingerprint: bool = True,
        title: str | None = None,
        include_figures: bool = False,
    ) -> ResearchWorkflowResult:
        """Run one supported workflow or return the exact information needed next.

        Scientific design facts are never inferred. Supplying a draft or specification
        is mutually exclusive with raw question arguments. No files are written and no
        reproduction is attempted.
        """
        for name, value in (
            ("include_profile", include_profile),
            ("audit", audit),
            ("fingerprint", fingerprint),
            ("include_figures", include_figures),
        ):
            if not isinstance(value, bool):
                raise InvalidDataError(f"{name} must be a Boolean.")
        if draft is not None and specification is not None:
            raise InvalidDataError("Provide either draft or specification, not both.")
        raw_values = (
            objective,
            outcome,
            predictor,
            design,
            estimand,
            description,
            options,
            data_dictionary,
            variable_types,
        )
        if (draft is not None or specification is not None) and any(
            value is not None for value in raw_values
        ):
            raise InvalidDataError(
                "A supplied draft or specification cannot be combined with raw question "
                "arguments. Update the draft explicitly before running it."
            )
        if draft is not None:
            if not isinstance(draft, QuestionDraft):
                raise InvalidDataError("draft must be a QuestionDraft.")
            selected = prepare_question(self._analyzer.df, specification=draft.specification)
        elif specification is not None:
            selected = prepare_question(self._analyzer.df, specification=specification)
        else:
            selected = self.prepare_question(
                objective=objective,
                outcome=outcome,
                predictor=predictor,
                design=design,
                estimand=estimand,
                description=description,
                options=options,
                data_dictionary=data_dictionary,
                variable_types=variable_types,
            )

        profile = None
        is_descriptive = selected.specification.question.objective is Objective.DESCRIPTIVE
        if include_profile and not is_descriptive:
            profile = self.profile(data_dictionary=selected.specification.data_dictionary)

        def stop(
            status: WorkflowStatus,
            *,
            recommendation: Recommendation | None = None,
            analysis: AnalysisResult | None = None,
            missing_information=(),
            blockers=(),
            warnings=(),
        ) -> ResearchWorkflowResult:
            return ResearchWorkflowResult(
                status=status,
                specification=selected.specification,
                draft=selected,
                recommendation=recommendation,
                analysis=analysis,
                profile=profile,
                missing_information=tuple(missing_information),
                blockers=tuple(blockers),
                warnings=tuple(dict.fromkeys(warnings)),
            )

        if selected.status.value == "needs_input":
            return stop(
                WorkflowStatus.NEEDS_INPUT,
                missing_information=selected.missing_information,
                warnings=selected.warnings,
            )
        if selected.status.value == "data_limited":
            return stop(
                WorkflowStatus.DATA_LIMITED,
                blockers=selected.blockers,
                warnings=selected.warnings,
            )
        if selected.status.value == "unsupported":
            return stop(
                WorkflowStatus.UNSUPPORTED,
                blockers=selected.blockers,
                warnings=selected.warnings,
            )

        recommendation = self.recommend_test(selected)
        if recommendation.status.value == "needs_input":
            return stop(
                WorkflowStatus.NEEDS_INPUT,
                recommendation=recommendation,
                missing_information=recommendation.missing_information,
                warnings=(*selected.warnings, *recommendation.warnings),
            )
        if recommendation.status.value == "unsupported":
            blocked_status = (
                WorkflowStatus.DATA_LIMITED
                if _is_data_limited_recommendation(recommendation)
                else WorkflowStatus.UNSUPPORTED
            )
            return stop(
                blocked_status,
                recommendation=recommendation,
                blockers=recommendation.blockers,
                warnings=(*selected.warnings, *recommendation.warnings),
            )

        analysis = self.analyze(selected)
        if analysis.status.value != "available":
            return stop(
                WorkflowStatus.FAILED,
                recommendation=recommendation,
                analysis=analysis,
                blockers=analysis.warnings
                or ("The selected method could not produce a usable numerical result.",),
                warnings=(*selected.warnings, *recommendation.warnings, *analysis.warnings),
            )

        interpretation = self.interpret(analysis)
        report = self.report(
            analysis,
            interpretation=interpretation,
            title=title,
            include_figures=include_figures,
        )
        audit_result = self.audit(report, result=analysis) if audit else None
        reproducibility = self.reproducibility_record(analysis, fingerprint=fingerprint)
        if is_descriptive:
            profile_value = analysis.values.get("profile")
            profile = profile_value if isinstance(profile_value, dict) else None

        warnings = list(
            dict.fromkeys(
                [
                    *selected.warnings,
                    *recommendation.warnings,
                    *analysis.warnings,
                    *interpretation.warnings,
                    *report.to_dict()["warnings"],
                    *reproducibility.to_dict().get("warnings", []),
                ]
            )
        )
        blockers: tuple[str, ...] = ()
        if audit_result is not None and audit_result.status == "failed":
            status = WorkflowStatus.FAILED
            blockers = tuple(finding.explanation for finding in audit_result.findings)
        elif interpretation.status.value == "unavailable" or report.status == "unavailable":
            status = WorkflowStatus.FAILED
            blockers = ("The computed result could not be interpreted into a usable report.",)
        elif (
            interpretation.status.value == "partial"
            or report.status == "partial"
            or audit_result is None
            or audit_result.status == "incomplete"
        ):
            status = WorkflowStatus.PARTIAL
            if audit_result is None:
                warnings.append("Report auditing was disabled; no audit was performed.")
            elif audit_result.status == "incomplete":
                warnings.append("The report audit was incomplete; inspect its skipped checks.")
        else:
            status = WorkflowStatus.COMPLETED
        return ResearchWorkflowResult(
            status=status,
            specification=selected.specification,
            draft=selected,
            recommendation=recommendation,
            analysis=analysis,
            interpretation=interpretation,
            report=report,
            audit=audit_result,
            reproducibility=reproducibility,
            profile=profile,
            blockers=blockers,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def prepare_question(
        self,
        *,
        objective: str | Objective | None = None,
        outcome: str | None = None,
        predictor: str | None = None,
        design: str | StudyDesign | None = None,
        estimand: str | None = None,
        description: str | None = None,
        options: AnalysisOptions | None = None,
        data_dictionary: dict | None = None,
        variable_types: dict[str, str] | None = None,
        specification: AnalysisSpecification | None = None,
    ) -> QuestionDraft:
        """Prepare a serializable question; return focused requests for missing facts."""
        draft = prepare_question(
            self._analyzer.df,
            objective=objective,
            outcome=outcome,
            predictor=predictor,
            design=design,
            estimand=estimand,
            description=description,
            options=options,
            data_dictionary=(
                data_dictionary
                if data_dictionary is not None
                else self._data_dictionary
                if specification is None
                else None
            ),
            variable_types=variable_types,
            specification=specification,
        )
        if self._ledger is not None:
            payload = draft.specification.to_dict()
            self._ledger._record(
                "question_prepared",
                new_state=payload,
                references={"specification": content_reference("specification", payload)},
                metadata={"draft_status": draft.status.value},
            )
        return draft

    def update_question(
        self, draft: QuestionDraft, *, reason: str | None = None, **changes: Any
    ) -> QuestionDraft:
        """Reconstruct and revalidate an immutable draft after explicit answers."""
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise InvalidDataError("A supplied revision reason must be non-empty text.")
        if not isinstance(draft, QuestionDraft):
            raise InvalidDataError("draft must be a QuestionDraft.")
        allowed = {
            "objective",
            "outcome",
            "predictor",
            "design",
            "estimand",
            "description",
            "options",
            "data_dictionary",
            "variable_types",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise InvalidDataError(f"Unknown question updates: {sorted(unknown)}.")
        old = draft.specification
        previous = old.question
        new_objective: Objective | None
        if "objective" in changes and changes["objective"] is not None:
            try:
                new_objective = Objective(changes["objective"])
            except (ValueError, TypeError) as exc:
                raise InvalidDataError(
                    "objective must be descriptive, compare_groups, or association."
                ) from exc
        else:
            new_objective = cast(Objective | None, changes.get("objective", previous.objective))
        switched = new_objective != previous.objective
        values: dict[str, Any] = {
            "objective": new_objective,
            "outcome": previous.outcome,
            "predictor": previous.predictor,
            "estimand": previous.estimand,
            "description": previous.description,
        }
        selected_design: StudyDesign | str = old.design
        if switched:
            values["predictor"] = None
            values["estimand"] = None
            selected_design = StudyDesign.UNKNOWN
            if new_objective == Objective.DESCRIPTIVE:
                values["outcome"] = None
        for key in ("outcome", "predictor", "estimand", "description"):
            if key in changes:
                values[key] = changes[key]
        if "design" in changes:
            selected_design = (
                StudyDesign.UNKNOWN if changes["design"] is None else changes["design"]
            )
        if new_objective == Objective.DESCRIPTIVE and any(
            key in changes for key in ("predictor", "estimand", "design")
        ):
            raise InvalidDataError(
                "Descriptive questions do not use predictor, estimand, or design."
            )
        revised = AnalysisSpecification(
            question=ResearchQuestion(**values),
            design=cast(StudyDesign, selected_design),
            options=changes.get("options", old.options),
            variable_metadata=old.variable_metadata,
            data_dictionary=changes.get("data_dictionary", old.data_dictionary),
        )
        updated = prepare_question(
            self._analyzer.df,
            specification=revised,
            variable_types=changes.get("variable_types"),
        )
        if self._ledger is not None:
            before = old.to_dict()
            after = updated.specification.to_dict()
            if before != after:
                changed = _changed_fields(before, after)
                self._ledger._record(
                    "specification_updated",
                    source="researcher",
                    previous_state=before,
                    new_state=after,
                    reason=reason,
                    references={"specification": content_reference("specification", after)},
                    metadata={"changed_fields": changed},
                )
        return updated

    def recommend_test(
        self,
        draft: QuestionDraft | None = None,
        *,
        specification: AnalysisSpecification | None = None,
    ) -> Recommendation:
        """Revalidate a question, then recommend a capability without executing it."""
        if (draft is None) == (specification is None):
            raise InvalidDataError("Provide either one QuestionDraft or specification.")
        if draft is not None and not isinstance(draft, QuestionDraft):
            raise InvalidDataError("draft must be a QuestionDraft.")
        selected_spec = draft.specification if draft is not None else specification
        validated = prepare_question(self._analyzer.df, specification=selected_spec)
        recommendation = recommend_from_draft(self._analyzer.df, validated)
        if self._ledger is not None:
            spec_payload = validated.specification.to_dict()
            rec_payload = recommendation.to_dict()
            self._ledger._record(
                "method_recommended",
                references={
                    "specification": content_reference("specification", spec_payload),
                    "recommendation": content_reference("recommendation", rec_payload),
                },
                metadata={
                    "status": recommendation.status.value,
                    "method_id": recommendation.method_id,
                },
            )
        return recommendation

    def analyze(
        self,
        draft: QuestionDraft | None = None,
        *,
        specification: AnalysisSpecification | None = None,
    ) -> AnalysisResult:
        """Execute a freshly validated question using its selected existing backend."""
        if (draft is None) == (specification is None):
            raise InvalidDataError("Provide either one QuestionDraft or specification.")
        if draft is not None and not isinstance(draft, QuestionDraft):
            raise InvalidDataError("draft must be a QuestionDraft.")
        selected_spec = draft.specification if draft is not None else specification
        assert selected_spec is not None
        result = execute_specification(self._analyzer, selected_spec)
        if self._ledger is not None:
            references = {"analysis": content_reference("analysis", result.to_dict())}
            if result.specification is not None:
                references["specification"] = content_reference(
                    "specification", result.specification.to_dict()
                )
            if result.recommendation is not None:
                references["recommendation"] = content_reference(
                    "recommendation", result.recommendation.to_dict()
                )
            self._ledger._record(
                "analysis_executed",
                references=references,
                metadata={
                    "status": result.status.value,
                    "method_id": result.method_id,
                    "analyzed_rows": result.sample_size,
                    "excluded_rows": result.excluded_rows,
                },
            )
        return result

    def interpret(self, result: AnalysisResult) -> InterpretationResult:
        """Explain a completed analysis from its recorded values and specification."""
        interpretation = InterpretationEngine().interpret(result)
        if self._ledger is not None:
            self._ledger._record(
                "interpretation_generated",
                references={
                    "analysis": content_reference("analysis", result.to_dict()),
                    "interpretation": content_reference("interpretation", interpretation.to_dict()),
                },
                metadata={"status": interpretation.status.value},
            )
        return interpretation

    def report(
        self,
        result: AnalysisResult,
        *,
        interpretation: InterpretationResult | None = None,
        title: str | None = None,
        include_figures: bool = False,
    ) -> ResearchReport:
        """Assemble a general research report from recorded analysis and interpretation."""
        report = build_research_report(
            result, interpretation=interpretation, title=title, include_figures=include_figures
        )
        if self._ledger is not None:
            report_reference = content_reference("report", report.to_dict())
            self._ledger._record(
                "report_generated",
                references={
                    "analysis": content_reference("analysis", result.to_dict()),
                    "interpretation": content_reference(
                        "interpretation", report.to_dict()["interpretation"]
                    ),
                    "report": report_reference,
                },
                metadata={"status": report.status},
            )
            ledger = self._ledger

            def on_save(format_name: str) -> None:
                ledger._record(
                    "report_exported",
                    references={"report": report_reference},
                    metadata={"format": format_name},
                )

            report._on_save = on_save
        return report

    def audit(
        self,
        report: ResearchReport,
        *,
        result: AnalysisResult | None = None,
        exports: dict[str, Any] | None = None,
    ) -> AuditResult:
        """Check recorded analysis, report and supplied exports without rerunning tests."""
        audit = StatisticalResultAuditor().audit(report, result=result, exports=exports)
        if self._ledger is not None:
            self._ledger._record(
                "audit_performed",
                references={
                    **audit.source_references,
                    "audit": content_reference("audit", audit.to_dict()),
                },
                metadata={"status": audit.status},
            )
        return audit

    def reproducibility_record(
        self, result: AnalysisResult, *, fingerprint: bool = True
    ) -> ReproducibilityRecord:
        """Capture replay metadata for the assistant's current data, on request."""
        return ReproducibilityRecord.from_result(
            result,
            data=self._analyzer.df,
            fingerprint=fingerprint,
            planning_status=self._planning_status,
        )


def _changed_fields(before: dict[str, Any], after: dict[str, Any], prefix: str = "") -> list[str]:
    fields: list[str] = []
    for key in sorted(before.keys() | after.keys()):
        path = f"{prefix}.{key}" if prefix else key
        left, right = before.get(key), after.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            fields.extend(_changed_fields(left, right, path))
        elif left != right:
            fields.append(path)
    return fields


_DATA_LIMIT_MESSAGES = (
    "fewer than two groups",
    "needs at least two usable outcomes",
    "not representable at this scale",
    "zero representable within-group variation",
    "no observed variation",
    "requires at least five usable observations",
    "at least two complete, varying numeric pairs",
    "at least three complete pairs",
    "variation in both variables",
    "at least two observed categories",
    "expected cell count below 5",
)


def _is_data_limited_recommendation(recommendation: Recommendation) -> bool:
    """Separate observed data insufficiency from unsupported scientific requests."""
    messages = " ".join(recommendation.blockers).lower()
    return any(fragment in messages for fragment in _DATA_LIMIT_MESSAGES)
