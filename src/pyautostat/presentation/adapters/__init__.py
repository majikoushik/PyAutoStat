"""Presentation adapters package.

Transforms PyAutoStat domain and result records into normalized TerminalView display models.
"""

from __future__ import annotations

from typing import Any

from ...analysis_plan import PlanAdherenceResult, StatisticalAnalysisPlan
from ...audit import AuditResult
from ...completeness import ReportingCompletenessResult
from ...decision_ledger import DecisionLedger
from ...practical_significance import PracticalSignificanceResult
from ...reproducibility import ReproducibilityRecord, ReproductionOutcome
from ...results import AnalysisResult
from ...sensitivity import SensitivityResult
from ...session import ResearchSessionSnapshot
from ...study_planning import StudyPlanningResult
from ...workflow import ResearchWorkflowResult
from ..models import TerminalView
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
from .descriptive import (
    adapt_cross_tab,
    adapt_frequency_table,
    adapt_profile,
    is_cross_tab,
    is_frequency_table,
    is_profile,
)
from .factorial import adapt_two_way_anova
from .governance import (
    adapt_audit,
    adapt_decision_ledger,
    adapt_plan_adherence,
    adapt_reporting_completeness,
    adapt_reproducibility,
    adapt_session_snapshot,
    adapt_statistical_analysis_plan,
)
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
from .planning import (
    adapt_practical_significance,
    adapt_sensitivity,
    adapt_study_planning,
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
from .workflow import (
    METHOD_ADAPTERS,
    UnsupportedPresentationError,
    adapt_analysis_result,
    adapt_workflow,
    adapt_workflow_status,
)


def adapt(target: Any, detail: str = "standard") -> TerminalView:
    """Convert any supported PyAutoStat structured result into a normalized TerminalView."""
    if detail not in {"compact", "standard", "full"}:
        raise ValueError(
            f"Invalid detail mode '{detail}'. Expected 'compact', 'standard', or 'full'."
        )

    if isinstance(target, ResearchWorkflowResult):
        return adapt_workflow(target, detail=detail)

    if isinstance(target, AnalysisResult):
        return adapt_analysis_result(target, detail=detail)

    if isinstance(target, StudyPlanningResult):
        return adapt_study_planning(target, detail=detail)

    if isinstance(target, SensitivityResult):
        return adapt_sensitivity(target, detail=detail)

    if isinstance(target, PracticalSignificanceResult):
        return adapt_practical_significance(target, detail=detail)

    if isinstance(target, StatisticalAnalysisPlan):
        return adapt_statistical_analysis_plan(target, detail=detail)

    if isinstance(target, PlanAdherenceResult):
        return adapt_plan_adherence(target, detail=detail)

    if isinstance(target, ReportingCompletenessResult):
        return adapt_reporting_completeness(target, detail=detail)

    if isinstance(target, AuditResult):
        return adapt_audit(target, detail=detail)

    if isinstance(target, (ReproducibilityRecord, ReproductionOutcome)):
        return adapt_reproducibility(target, detail=detail)

    if isinstance(target, DecisionLedger):
        return adapt_decision_ledger(target, detail=detail)

    if isinstance(target, ResearchSessionSnapshot):
        return adapt_session_snapshot(target, detail=detail)

    if isinstance(target, dict):
        if is_profile(target):
            return adapt_profile(target, detail=detail)
        if is_frequency_table(target):
            return adapt_frequency_table(target, detail=detail)
        if is_cross_tab(target):
            return adapt_cross_tab(target, detail=detail)

    raise TypeError(
        f"Unsupported target type for show(): {type(target).__name__}. "
        "Expected ResearchWorkflowResult, AnalysisResult, dataset profile, "
        "governance result, or study planning object."
    )


__all__ = [
    "METHOD_ADAPTERS",
    "UnsupportedPresentationError",
    "adapt",
    "adapt_analysis_result",
    "adapt_audit",
    "adapt_cronbach_alpha",
    "adapt_cross_tab",
    "adapt_decision_ledger",
    "adapt_fisher_exact",
    "adapt_frequency_table",
    "adapt_friedman_test",
    "adapt_intraclass_correlation",
    "adapt_kendall_tau_b",
    "adapt_kruskal_wallis",
    "adapt_linear_regression",
    "adapt_logistic_regression",
    "adapt_mann_whitney_u",
    "adapt_mcnemar",
    "adapt_one_sample_t",
    "adapt_one_way_anova",
    "adapt_paired_t",
    "adapt_partial_pearson",
    "adapt_pearson",
    "adapt_pearson_chi_square",
    "adapt_plan_adherence",
    "adapt_point_biserial",
    "adapt_practical_significance",
    "adapt_profile",
    "adapt_repeated_measures_anova",
    "adapt_reporting_completeness",
    "adapt_reproducibility",
    "adapt_sensitivity",
    "adapt_session_snapshot",
    "adapt_spearman",
    "adapt_statistical_analysis_plan",
    "adapt_student_t",
    "adapt_study_planning",
    "adapt_two_way_anova",
    "adapt_welch_anova",
    "adapt_welch_t",
    "adapt_wilcoxon_signed_rank",
    "adapt_workflow",
    "adapt_workflow_status",
    "is_cross_tab",
    "is_frequency_table",
    "is_profile",
]
