"""Workflow and analysis-result routing adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...results import AnalysisResult
from ...workflow import ResearchWorkflowResult, WorkflowStatus
from ..models import (
    DisplayDiagnostic,
    DisplayMetric,
    TerminalView,
)
from .association import (
    adapt_kendall_tau_b,
    adapt_partial_pearson,
    adapt_pearson,
    adapt_point_biserial,
    adapt_spearman,
)
from .categorical import (
    adapt_fisher_exact,
    adapt_mcnemar,
    adapt_pearson_chi_square,
)
from .factorial import adapt_two_way_anova
from .means import (
    adapt_mann_whitney_u,
    adapt_one_sample_t,
    adapt_student_t,
    adapt_welch_t,
)
from .multigroup import (
    adapt_kruskal_wallis,
    adapt_one_way_anova,
    adapt_welch_anova,
)
from .paired import (
    adapt_paired_t,
    adapt_wilcoxon_signed_rank,
)
from .regression import (
    adapt_linear_regression,
    adapt_logistic_regression,
)
from .reliability import (
    adapt_cronbach_alpha,
    adapt_intraclass_correlation,
)
from .repeated import (
    adapt_friedman_test,
    adapt_repeated_measures_anova,
)

METHOD_ADAPTERS: dict[str, Callable[[Any, str], TerminalView]] = {
    "welch_t": adapt_welch_t,
    "student_t": adapt_student_t,
    "mann_whitney_u": adapt_mann_whitney_u,
    "one_sample_t": adapt_one_sample_t,
    "paired_t": adapt_paired_t,
    "wilcoxon_signed_rank": adapt_wilcoxon_signed_rank,
    "welch_anova": adapt_welch_anova,
    "one_way_anova": adapt_one_way_anova,
    "kruskal_wallis": adapt_kruskal_wallis,
    "pearson_correlation": adapt_pearson,
    "spearman_correlation": adapt_spearman,
    "kendall_tau_b": adapt_kendall_tau_b,
    "point_biserial_correlation": adapt_point_biserial,
    "partial_pearson_correlation": adapt_partial_pearson,
    "pearson_chi_square": adapt_pearson_chi_square,
    "fisher_exact": adapt_fisher_exact,
    "mcnemar": adapt_mcnemar,
    "linear_regression": adapt_linear_regression,
    "logistic_regression": adapt_logistic_regression,
    "cronbach_alpha": adapt_cronbach_alpha,
    "repeated_measures_anova": adapt_repeated_measures_anova,
    "friedman_test": adapt_friedman_test,
    "two_way_anova": adapt_two_way_anova,
    "intraclass_correlation": adapt_intraclass_correlation,
}


class UnsupportedPresentationError(Exception):
    """Raised when an object or statistical method is not supported by the presentation layer."""


def adapt_workflow(
    workflow: ResearchWorkflowResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a ResearchWorkflowResult into a TerminalView."""
    if workflow.status not in {WorkflowStatus.COMPLETED, WorkflowStatus.PARTIAL}:
        return adapt_workflow_status(workflow, detail=detail)

    if workflow.analysis is None:
        return adapt_workflow_status(workflow, detail=detail)

    method_id = workflow.analysis.method_id
    adapter_fn = METHOD_ADAPTERS.get(method_id)
    if adapter_fn is not None:
        return adapter_fn(workflow, detail)

    raise UnsupportedPresentationError(
        f"Method '{method_id}' is not supported by the presentation layer."
    )


def adapt_analysis_result(
    analysis: AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt an AnalysisResult directly into a TerminalView."""
    method_id = analysis.method_id
    adapter_fn = METHOD_ADAPTERS.get(method_id)
    if adapter_fn is not None:
        return adapter_fn(analysis, detail)

    raise UnsupportedPresentationError(
        f"AnalysisResult with method '{method_id}' is not supported by the presentation layer."
    )


def adapt_workflow_status(
    workflow: ResearchWorkflowResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a pending, blocked, or failed workflow into a clean TerminalView."""
    status = workflow.status
    draft = workflow.draft
    spec = workflow.specification or (draft.specification if draft else None)

    design_metrics_list = []
    if spec:
        if spec.question.outcome:
            design_metrics_list.append(DisplayMetric("Outcome", str(spec.question.outcome)))
        if spec.question.predictor:
            design_metrics_list.append(DisplayMetric("Predictor", str(spec.question.predictor)))
        if spec.question.estimand:
            design_metrics_list.append(DisplayMetric("Estimand", str(spec.question.estimand)))
        if spec.design:
            design_metrics_list.append(DisplayMetric("Design", str(spec.design)))

    diagnostics_list: list[DisplayDiagnostic] = []
    if status is WorkflowStatus.NEEDS_INPUT:
        title = "Additional Information Required"
        subtitle = "Analysis has not been run."
        family = "family.status"
        for item in workflow.missing_information:
            diagnostics_list.append(
                DisplayDiagnostic(
                    label=f"Field '{item.field}'",
                    status="Required",
                    detail=item.message,
                    severity="missing",
                )
            )
        n_req = len(workflow.missing_information)
        compact_text = f"Workflow: Needs Input | {n_req} fields required"

    elif status is WorkflowStatus.DATA_LIMITED:
        title = "Analysis Data-Limited"
        subtitle = "Current data do not satisfy the method's numerical requirements."
        family = "family.status"
        for blocker in workflow.blockers:
            diagnostics_list.append(
                DisplayDiagnostic(
                    label="Blocker",
                    status="Data Limited",
                    detail=blocker,
                    severity="warning",
                )
            )
        first_blocker = workflow.blockers[0] if workflow.blockers else ""
        compact_text = f"Workflow: Data Limited | {first_blocker}"

    elif status is WorkflowStatus.UNSUPPORTED:
        title = "Unsupported Specification"
        subtitle = "The declared design or target is not currently implemented."
        family = "family.status"
        for blocker in workflow.blockers:
            diagnostics_list.append(
                DisplayDiagnostic(
                    label="Blocker",
                    status="Unsupported",
                    detail=blocker,
                    severity="warning",
                )
            )
        compact_text = "Workflow: Unsupported | No substitute statistical method was run"

    else:  # FAILED
        title = "Analysis Failed"
        subtitle = "A computational or validation failure occurred."
        family = "status.error"
        for blocker in workflow.blockers:
            diagnostics_list.append(
                DisplayDiagnostic(
                    label="Failure",
                    status="Failed",
                    detail=blocker,
                    severity="error",
                )
            )
        compact_text = f"Workflow: Failed | {workflow.blockers[0] if workflow.blockers else ''}"

    return TerminalView(
        title=title,
        subtitle=subtitle,
        family=family,
        design_metrics=tuple(design_metrics_list),
        key_metrics=(),
        tables=(),
        diagnostics=tuple(diagnostics_list),
        interpretation=None,
        limitations=(),
        warnings=workflow.warnings,
        metadata={"workflow_status": status.value, "draft": draft.to_dict()},
        compact_text=compact_text,
    )
