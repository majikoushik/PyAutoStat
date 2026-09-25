"""Public entry point for profiling, question intake, and method recommendations."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from typing import Any, cast

import pandas as pd

from .analysis_plan import (
    AnalysisPlanStatus,
    PlanAdherenceResult,
    StatisticalAnalysisPlan,
    compare_plan_to_result,
    planned_interval_quantity,
    planned_quantity,
)
from .analyzer import StatisticalAnalyzer
from .audit import AuditResult, StatisticalResultAuditor
from .completeness import ReportingCompletenessResult, assess_reporting_completeness
from .decision_ledger import DecisionLedger
from .exceptions import InvalidDataError, PyAutoStatError
from .execution import execute_selected_method, execute_specification
from .interpretation import InterpretationEngine, InterpretationResult
from .practical_significance import (
    MeaningfulEffectThreshold,
    PracticalSignificanceResult,
    assess_practical_significance,
)
from .profiling import complete_case_count
from .provenance import content_reference, dataset_fingerprint
from .question_builder import QuestionDraft, prepare_question
from .recommendation import recommend_from_draft
from .reproducibility import ReproducibilityRecord
from .research_report import ResearchReport, build_research_report
from .results import AnalysisResult, Recommendation
from .sensitivity import (
    Comparability,
    ScenarioStatus,
    SensitivityResult,
    SensitivityScenarioResult,
    SensitivitySpecification,
    SensitivityStatus,
    classify_comparability,
    compare_same_estimand,
    estimate_quantity,
    scenario_values,
)
from .session import ResearchSessionSnapshot, build_session_snapshot
from .specifications import (
    AnalysisOptions,
    AnalysisSpecification,
    Objective,
    ResearchQuestion,
    StudyDesign,
)
from .study_planning import StudyPlanner, StudyPlanningResult
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
        self._last_meaningful_threshold: dict[str, Any] | None = None
        self._analysis_has_executed = False

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

    def study_planner(self) -> StudyPlanner:
        """Return a prospective planner that does not inspect this assistant's data."""

        def record(result: StudyPlanningResult) -> None:
            if self._ledger is not None:
                payload = result.to_dict()
                self._ledger._record(
                    "study_planning_completed",
                    source="researcher",
                    references={"study_planning": content_reference("study_planning", payload)},
                    metadata={
                        "status": result.status,
                        "planning_type": result.planning_type,
                        "method_family": result.method_family,
                    },
                )

        return StudyPlanner(on_result=record)

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
        unit_id: str | None = None,
        condition_order: tuple[Any, Any] | None = None,
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
            unit_id,
            condition_order,
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
                unit_id=unit_id,
                condition_order=condition_order,
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
        unit_id: str | None = None,
        condition_order: tuple[Any, Any] | None = None,
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
            unit_id=unit_id,
            condition_order=condition_order,
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
            "unit_id",
            "condition_order",
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
            selected_unit_id = None
            selected_condition_order = None
            if new_objective == Objective.DESCRIPTIVE:
                values["outcome"] = None
        else:
            selected_unit_id = old.unit_id
            selected_condition_order = old.condition_order
        for key in ("outcome", "predictor", "estimand", "description"):
            if key in changes:
                values[key] = changes[key]
        if "design" in changes:
            selected_design = (
                StudyDesign.UNKNOWN if changes["design"] is None else changes["design"]
            )
            try:
                revised_design = StudyDesign(selected_design)
            except (TypeError, ValueError) as exc:
                raise InvalidDataError(
                    "design must be unknown, independent, paired, repeated, or clustered."
                ) from exc
            if revised_design is not StudyDesign.PAIRED:
                selected_unit_id = None
                selected_condition_order = None
        if changes.get("unit_id", selected_unit_id) is None:
            selected_condition_order = None
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
            unit_id=changes.get("unit_id", selected_unit_id),
            condition_order=changes.get("condition_order", selected_condition_order),
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

    def analysis_plan(
        self,
        draft_or_specification: QuestionDraft | AnalysisSpecification,
        *,
        sensitivity_scenarios: list[SensitivitySpecification]
        | tuple[SensitivitySpecification, ...]
        | None = None,
        meaningful_threshold: MeaningfulEffectThreshold | None = None,
        multiplicity_policy: str = "not_applicable",
        multiplicity_method: str | None = None,
        report_style: str = "general",
        previous_plan: StatisticalAnalysisPlan | None = None,
        reason: str | None = None,
    ) -> StatisticalAnalysisPlan:
        """Create or revise a plan without executing any numerical analysis."""
        if isinstance(draft_or_specification, QuestionDraft):
            draft = prepare_question(
                self._analyzer.df, specification=draft_or_specification.specification
            )
        elif isinstance(draft_or_specification, AnalysisSpecification):
            draft = prepare_question(self._analyzer.df, specification=draft_or_specification)
        else:
            raise InvalidDataError(
                "analysis_plan requires a QuestionDraft or AnalysisSpecification."
            )
        scenarios = tuple(sensitivity_scenarios or ())
        if any(not isinstance(item, SensitivitySpecification) for item in scenarios):
            raise InvalidDataError(
                "sensitivity_scenarios must contain SensitivitySpecification records."
            )
        if meaningful_threshold is not None and not isinstance(
            meaningful_threshold, MeaningfulEffectThreshold
        ):
            raise InvalidDataError(
                "meaningful_threshold must be a MeaningfulEffectThreshold or None."
            )
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise InvalidDataError("A supplied plan-revision reason must be non-empty text.")
        recommendation = None
        if draft.status.value == "ready":
            recommendation = recommend_from_draft(self._analyzer.df, draft)
        status = (
            AnalysisPlanStatus.NEEDS_INPUT
            if draft.status.value == "needs_input"
            else AnalysisPlanStatus.READY
            if recommendation is not None and recommendation.status.value == "ready"
            else AnalysisPlanStatus.UNSUPPORTED
        )
        method_id = (
            recommendation.method_id
            if recommendation is not None and recommendation.status.value == "ready"
            else None
        )
        warnings = [*draft.warnings]
        limitations = [
            "A local plan is not proof of external preregistration or study validity.",
            "No automatic outlier deletion or missing-data modeling is planned.",
        ]
        if multiplicity_policy == "planned_method":
            limitations.append(
                "The named multiplicity procedure is recorded but is not executed by PyAutoStat."
            )
        rationale = None
        if recommendation is not None:
            rationale = recommendation.rationale
            warnings.extend(recommendation.warnings)
            limitations.extend(recommendation.blockers)
        else:
            limitations.extend(draft.blockers)
            limitations.extend(item.message for item in draft.missing_information)
        plan = StatisticalAnalysisPlan(
            specification=draft.specification,
            status=status,
            primary_method_id=method_id,
            method_rationale=rationale,
            effect_quantity=planned_quantity(method_id),
            confidence_interval_quantity=planned_interval_quantity(method_id),
            missing_data_policy="analysis-specific complete cases",
            exclusion_rule=(
                "Only rows missing variables required by the selected analysis are excluded."
            ),
            outlier_rule="No automatic outlier deletion.",
            sensitivity_scenarios=scenarios,
            meaningful_threshold=meaningful_threshold,
            multiplicity_policy=multiplicity_policy,
            multiplicity_method=multiplicity_method,
            report_style=report_style,
            planning_status=self._planning_status,
            created_after_analysis=self._analysis_has_executed,
            warnings=tuple(dict.fromkeys(warnings)),
            limitations=tuple(dict.fromkeys(limitations)),
            provenance={
                "tracking_enabled": self._ledger is not None,
                "local_record_only": True,
                "external_preregistration_verified": False,
                "timing": "after_analysis"
                if self._analysis_has_executed
                else "before_observed_analysis_in_this_assistant",
            },
        )
        if previous_plan is not None and not isinstance(previous_plan, StatisticalAnalysisPlan):
            raise InvalidDataError("previous_plan must be a StatisticalAnalysisPlan or None.")
        if self._ledger is not None:
            payload = plan.to_dict()
            previous = previous_plan.to_dict() if previous_plan is not None else None
            self._ledger._record(
                "analysis_plan_updated" if previous is not None else "analysis_plan_created",
                source="researcher",
                previous_state=previous,
                new_state=payload,
                reason=reason,
                references={"analysis_plan": content_reference("analysis_plan", payload)},
                metadata={
                    "status": plan.status.value,
                    "changed_fields": _changed_fields(previous, payload)
                    if previous is not None
                    else [],
                },
            )
        return plan

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
        self._analysis_has_executed = result.method_id != "unselected"
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

    def sensitivity_analysis(
        self,
        result: AnalysisResult,
        *,
        scenarios: list[SensitivitySpecification] | tuple[SensitivitySpecification, ...],
    ) -> SensitivityResult:
        """Execute exactly the declared scenarios, once each, in supplied order."""
        if not isinstance(result, AnalysisResult) or result.specification is None:
            raise InvalidDataError("sensitivity_analysis requires a result with its specification.")
        if result.status.value != "available":
            raise InvalidDataError("sensitivity_analysis requires an available base result.")
        if not isinstance(scenarios, (list, tuple)) or not scenarios:
            raise InvalidDataError("scenarios must be a non-empty list or tuple.")
        if any(not isinstance(item, SensitivitySpecification) for item in scenarios):
            raise InvalidDataError("Every scenario must be a SensitivitySpecification.")
        names = [item.name for item in scenarios]
        if len(set(names)) != len(names):
            raise InvalidDataError("Sensitivity scenario names must be unique.")
        original_rows = result.metadata.get("sample", {}).get("original_rows")
        if original_rows is not None and original_rows != len(self._analyzer.df):
            raise InvalidDataError(
                "The base result row count does not match this assistant's dataset."
            )
        fingerprint_warning = None
        try:
            fingerprint = dataset_fingerprint(self._analyzer.df)
        except InvalidDataError as exc:
            fingerprint = None
            fingerprint_warning = f"Dataset fingerprint unavailable: {exc}"
        base_reference = content_reference("analysis", result.to_dict())
        plan = [item.to_dict() for item in scenarios]
        if self._ledger is not None:
            self._ledger._record(
                "sensitivity_plan_created",
                source="researcher",
                new_state=plan,
                references={"base_analysis": base_reference},
                metadata={
                    "scenario_order": names,
                    "note": "Local planning record; not external preregistration.",
                },
            )
        outputs: list[SensitivityScenarioResult] = []
        warnings: list[str] = [fingerprint_warning] if fingerprint_warning is not None else []
        base_spec = result.specification
        for scenario in scenarios:
            requested = scenario.method_id
            if self._ledger is not None:
                self._ledger._record(
                    "sensitivity_scenario_attempted",
                    source="researcher",
                    new_state=scenario.to_dict(),
                    references={"base_analysis": base_reference},
                    metadata={"name": scenario.name, "planning_status": scenario.planning_status},
                )
            incompatible_reason = _scenario_incompatibility(base_spec, scenario.specification)
            if incompatible_reason is not None:
                output = SensitivityScenarioResult(
                    name=scenario.name,
                    specification=scenario,
                    requested_method_id=requested,
                    method_id=None,
                    status=ScenarioStatus.INCOMPATIBLE,
                    comparability=Comparability.INCOMPATIBLE,
                    warnings=(incompatible_reason,),
                    error=incompatible_reason,
                )
                outputs.append(output)
                warnings.append(f"{scenario.name}: {incompatible_reason}")
                self._record_sensitivity_outcome(output, base_reference)
                continue
            method_id = requested
            if method_id is None:
                scenario_draft = prepare_question(
                    self._analyzer.df, specification=scenario.specification
                )
                recommendation = recommend_from_draft(self._analyzer.df, scenario_draft)
                method_id = (
                    recommendation.method_id if recommendation.status.value == "ready" else None
                )
                if method_id is None:
                    reason = "; ".join(recommendation.blockers) or (
                        recommendation.rationale or "No runnable method was selected."
                    )
                    output = SensitivityScenarioResult(
                        name=scenario.name,
                        specification=scenario,
                        requested_method_id=None,
                        method_id=None,
                        status=ScenarioStatus.UNAVAILABLE,
                        comparability=Comparability.UNAVAILABLE,
                        warnings=tuple(recommendation.warnings),
                        error=reason,
                    )
                    outputs.append(output)
                    warnings.append(f"{scenario.name}: {reason}")
                    self._record_sensitivity_outcome(output, base_reference)
                    continue
            if method_id in {"student_t", "one_way_anova"} and not any(
                "equal" in item.lower() and "variance" in item.lower()
                for item in scenario.assumptions
            ):
                reason = (
                    f"{method_id} requires an explicit equal-population-variance assumption "
                    "in the sensitivity scenario."
                )
                output = SensitivityScenarioResult(
                    name=scenario.name,
                    specification=scenario,
                    requested_method_id=requested,
                    method_id=method_id,
                    status=ScenarioStatus.INCOMPATIBLE,
                    comparability=Comparability.INCOMPATIBLE,
                    warnings=(reason,),
                    error=reason,
                )
                outputs.append(output)
                warnings.append(f"{scenario.name}: {reason}")
                self._record_sensitivity_outcome(output, base_reference)
                continue
            try:
                analysis = execute_selected_method(
                    self._analyzer, scenario.specification, method_id
                )
                if analysis.status.value != "available":
                    reason = analysis.warnings[-1] if analysis.warnings else "Scenario unavailable."
                    output = SensitivityScenarioResult(
                        name=scenario.name,
                        specification=scenario,
                        requested_method_id=requested,
                        method_id=method_id,
                        status=ScenarioStatus.UNAVAILABLE,
                        comparability=Comparability.UNAVAILABLE,
                        warnings=analysis.warnings,
                        error=reason,
                        analysis=analysis,
                    )
                else:
                    comparability = classify_comparability(result, analysis)
                    comparison = (
                        compare_same_estimand(result, analysis)
                        if comparability is Comparability.SAME_ESTIMAND
                        else None
                    )
                    if (
                        comparability is Comparability.SAME_ESTIMAND
                        and comparison is not None
                        and not comparison.get("available", False)
                        and "contrast" in str(comparison.get("reason", "")).lower()
                    ):
                        comparability = Comparability.INCOMPATIBLE
                    scenario_warning = list(analysis.warnings)
                    if comparability is Comparability.DIFFERENT_ESTIMAND:
                        scenario_warning.append(
                            "This scenario targets a different estimand and is not a direct "
                            "robustness replication of the base analysis."
                        )
                    elif comparability is Comparability.INCOMPATIBLE:
                        scenario_warning.append(
                            "This scenario changes the paired scientific comparison identity "
                            "or contrast orientation and is not a same-estimand robustness "
                            "comparison."
                        )
                    values = scenario_values(analysis)
                    missing_numerical = [
                        label
                        for label, value in (
                            ("primary estimate", values["primary_estimate"]),
                            ("p-value", values["p_value"]),
                        )
                        if value is None
                    ]
                    if missing_numerical:
                        reason = (
                            "Scenario returned no finite "
                            + " or ".join(missing_numerical)
                            + "; the numerical result is unreliable."
                        )
                        output = SensitivityScenarioResult(
                            name=scenario.name,
                            specification=scenario,
                            requested_method_id=requested,
                            method_id=analysis.method_id,
                            status=ScenarioStatus.FAILED,
                            comparability=Comparability.UNAVAILABLE,
                            warnings=tuple(dict.fromkeys([*scenario_warning, reason])),
                            error=reason,
                            analysis=analysis,
                            **values,
                        )
                    else:
                        output = SensitivityScenarioResult(
                            name=scenario.name,
                            specification=scenario,
                            requested_method_id=requested,
                            method_id=analysis.method_id,
                            status=ScenarioStatus.COMPLETED,
                            comparability=comparability,
                            comparison=comparison,
                            warnings=tuple(dict.fromkeys(scenario_warning)),
                            analysis=analysis,
                            **values,
                        )
                outputs.append(output)
                warnings.extend(f"{scenario.name}: {item}" for item in output.warnings)
                self._record_sensitivity_outcome(output, base_reference)
            except (PyAutoStatError, ValueError, TypeError, OverflowError, RuntimeError) as exc:
                output = SensitivityScenarioResult(
                    name=scenario.name,
                    specification=scenario,
                    requested_method_id=requested,
                    method_id=method_id,
                    status=ScenarioStatus.FAILED,
                    comparability=Comparability.UNAVAILABLE,
                    error=str(exc),
                    warnings=(f"Scenario execution failed: {exc}",),
                )
                outputs.append(output)
                warnings.extend(f"{scenario.name}: {item}" for item in output.warnings)
                self._record_sensitivity_outcome(output, base_reference)
        completed = sum(item.status is ScenarioStatus.COMPLETED for item in outputs)
        status = (
            SensitivityStatus.COMPLETE
            if completed == len(outputs)
            else SensitivityStatus.PARTIAL
            if completed
            else SensitivityStatus.UNAVAILABLE
        )
        same = [item for item in outputs if item.comparability is Comparability.SAME_ESTIMAND]
        different = [
            item for item in outputs if item.comparability is Comparability.DIFFERENT_ESTIMAND
        ]
        summary = {
            "base_method_id": result.method_id,
            "base_estimate_quantity": estimate_quantity(result.method_id),
            "declared_scenario_count": len(outputs),
            "completed_scenario_count": completed,
            "same_estimand_scenarios": len(same),
            "different_estimand_scenarios": len(different),
            "same_estimand_direction_consistent": (
                all(bool((item.comparison or {}).get("direction_consistent")) for item in same)
                if same
                else None
            ),
            "all_same_estimand_intervals_available": (
                all((item.comparison or {}).get("interval_overlap") is not None for item in same)
                if same
                else None
            ),
            "note": (
                "Comparisons are descriptive across the declared scenarios. P-values do not "
                "select, rank, or replace the primary analysis."
            ),
        }
        return SensitivityResult(
            base_result=result,
            scenario_results=tuple(outputs),
            status=status,
            comparison_summary=summary,
            warnings=tuple(dict.fromkeys(warnings)),
            provenance={
                "tracking_enabled": self._ledger is not None,
                "local_record_only": True,
                "external_preregistration_verified": False,
                "scenario_planning_statuses": [item.planning_status for item in scenarios],
            },
            reproducibility={
                "schema_version": 1,
                "base_analysis_reference": base_reference,
                "dataset_fingerprint": fingerprint,
                "scenario_order": names,
                "scenarios": plan,
                "actual_methods": [item.method_id for item in outputs],
                "scenario_statuses": [item.status.value for item in outputs],
                "random_seeds": [
                    item.specification.specification.options.random_seed for item in outputs
                ],
                "automatic_replay": False,
            },
        )

    def _record_sensitivity_outcome(
        self, output: SensitivityScenarioResult, base_reference: str
    ) -> None:
        if self._ledger is None:
            return
        event = (
            "sensitivity_scenario_completed"
            if output.status is ScenarioStatus.COMPLETED
            else "sensitivity_scenario_failed"
            if output.status is ScenarioStatus.FAILED
            else "sensitivity_scenario_unavailable"
        )
        self._ledger._record(
            event,
            references={
                "base_analysis": base_reference,
                "scenario": content_reference("sensitivity_scenario", output.to_dict()),
            },
            metadata={
                "name": output.name,
                "status": output.status.value,
                "comparability": output.comparability.value,
                "method_id": output.method_id,
            },
        )

    def practical_significance(
        self,
        result: AnalysisResult,
        *,
        threshold: MeaningfulEffectThreshold,
    ) -> PracticalSignificanceResult:
        """Assess one available quantity against a researcher-supplied threshold."""
        if not isinstance(threshold, MeaningfulEffectThreshold):
            raise InvalidDataError("threshold must be a MeaningfulEffectThreshold.")
        payload = threshold.to_dict()
        if self._ledger is not None:
            self._ledger._record(
                "meaningful_threshold_declared",
                source="researcher",
                previous_state=self._last_meaningful_threshold,
                new_state=payload,
                reason=threshold.rationale,
                references={"analysis": content_reference("analysis", result.to_dict())},
                metadata={
                    "planning_status": threshold.planning_status,
                    "note": (
                        "Researcher declaration recorded locally; timing is not external proof."
                    ),
                },
            )
        self._last_meaningful_threshold = payload
        return assess_practical_significance(
            result, threshold, planning_status=threshold.planning_status
        )

    def plan_adherence(
        self,
        plan: StatisticalAnalysisPlan,
        result: AnalysisResult,
        *,
        reason: str | None = None,
        sensitivity: SensitivityResult | None = None,
        practical_significance: PracticalSignificanceResult | None = None,
    ) -> PlanAdherenceResult:
        """Compare recorded plan fields with a later result without judging conduct."""
        comparison = compare_plan_to_result(
            plan,
            result,
            reason=reason,
            sensitivity=sensitivity,
            practical_significance=practical_significance,
        )
        if self._ledger is not None:
            payload = comparison.to_dict()
            self._ledger._record(
                "plan_adherence_compared",
                references={
                    "analysis_plan": content_reference("analysis_plan", plan.to_dict()),
                    "analysis": content_reference("analysis", result.to_dict()),
                    "plan_adherence": content_reference("plan_adherence", payload),
                    **(
                        {"sensitivity": content_reference("sensitivity", sensitivity.to_dict())}
                        if sensitivity is not None
                        else {}
                    ),
                    **(
                        {
                            "practical_significance": content_reference(
                                "practical_significance", practical_significance.to_dict()
                            )
                        }
                        if practical_significance is not None
                        else {}
                    ),
                },
                metadata={"status": comparison.status},
            )
        return comparison

    def reporting_completeness(
        self, report: ResearchReport, *, style: str = "general"
    ) -> ReportingCompletenessResult:
        """Assess applicable reporting fields without scoring research quality."""
        result = assess_reporting_completeness(report, style=style)
        if self._ledger is not None:
            payload = result.to_dict()
            self._ledger._record(
                "reporting_completeness_assessed",
                references={
                    "report": content_reference("report", report.to_dict()),
                    "reporting_completeness": content_reference("reporting_completeness", payload),
                },
                metadata={"status": result.status, "style": result.style},
            )
        return result

    def session_snapshot(
        self,
        workflow: ResearchWorkflowResult,
        *,
        sensitivity: SensitivityResult | None = None,
        practical_significance: PracticalSignificanceResult | None = None,
        analysis_plan: StatisticalAnalysisPlan | None = None,
        study_planning: StudyPlanningResult | None = None,
        reporting_completeness: ReportingCompletenessResult | None = None,
    ) -> ResearchSessionSnapshot:
        """Serialize current records for UI-independent adapters without rerunning work."""
        snapshot = build_session_snapshot(
            workflow,
            sensitivity=sensitivity,
            practical_significance=practical_significance,
            analysis_plan=analysis_plan,
            study_planning=study_planning,
            reporting_completeness=reporting_completeness,
        )
        if self._ledger is not None:
            payload = snapshot.to_dict()
            self._ledger._record(
                "session_snapshot_created",
                references={"session_snapshot": content_reference("session_snapshot", payload)},
                metadata={"workflow_status": workflow.status.value},
            )
        return snapshot

    def report(
        self,
        result: AnalysisResult,
        *,
        interpretation: InterpretationResult | None = None,
        sensitivity: SensitivityResult | None = None,
        practical_significance: PracticalSignificanceResult | None = None,
        title: str | None = None,
        include_figures: bool = False,
    ) -> ResearchReport:
        """Assemble a general research report from recorded analysis and interpretation."""
        report = build_research_report(
            result,
            interpretation=interpretation,
            sensitivity=sensitivity,
            practical_significance=practical_significance,
            title=title,
            include_figures=include_figures,
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
        sensitivity: SensitivityResult | None = None,
        practical_significance: PracticalSignificanceResult | None = None,
        exports: dict[str, Any] | None = None,
    ) -> AuditResult:
        """Check recorded analysis, report and supplied exports without rerunning tests."""
        audit = StatisticalResultAuditor().audit(
            report,
            result=result,
            sensitivity=sensitivity,
            practical_significance=practical_significance,
            exports=exports,
        )
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
        self,
        result: AnalysisResult,
        *,
        fingerprint: bool = True,
        sensitivity: SensitivityResult | None = None,
        practical_significance: PracticalSignificanceResult | None = None,
    ) -> ReproducibilityRecord:
        """Capture replay metadata for the assistant's current data, on request."""
        return ReproducibilityRecord.from_result(
            result,
            data=self._analyzer.df,
            fingerprint=fingerprint,
            planning_status=self._planning_status,
            sensitivity=sensitivity,
            practical_significance=practical_significance,
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


def _scenario_incompatibility(
    base: AnalysisSpecification, scenario: AnalysisSpecification
) -> str | None:
    """Reject changes that would no longer be an analysis of the same question roles."""
    left, right = base.question, scenario.question
    if left.objective != right.objective:
        return "The scenario changes the research objective."
    if left.outcome != right.outcome or left.predictor != right.predictor:
        return "The scenario changes the outcome or predictor role."
    if base.design != scenario.design:
        return (
            "The scenario changes the declared study design; no independent-analysis method "
            "was substituted."
        )
    if base.design is StudyDesign.PAIRED and base.unit_id != scenario.unit_id:
        return (
            "The scenario changes the paired unit_id, so it does not preserve the scientific "
            "pairing definition."
        )
    return None
