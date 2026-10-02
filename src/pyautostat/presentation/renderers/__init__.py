"""Registry and routing for PyAutoStat presentation renderers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

from ..models import TerminalView
from .association import AssociationRenderer
from .base import BaseRenderer
from .categorical import CategoricalRenderer
from .descriptive import DescriptiveTableRenderer
from .factorial import FactorialRenderer
from .governance import (
    AnalysisPlanRenderer,
    AuditRenderer,
    CompletenessRenderer,
    DecisionLedgerRenderer,
    PlanAdherenceRenderer,
    ReproducibilityRenderer,
    SessionRenderer,
)
from .icc import ICCRenderer
from .logistic import LogisticRenderer
from .multigroup import MultiGroupRenderer
from .one_sample import OneSampleRenderer
from .paired import PairedRenderer
from .paired_categorical import PairedCategoricalRenderer
from .planning import (
    PlanningRenderer,
    PracticalSignificanceRenderer,
    SensitivityRenderer,
)
from .profile import ProfileRenderer
from .regression import RegressionRenderer
from .reliability import ReliabilityRenderer
from .repeated import RepeatedMeasuresRenderer
from .two_group import TwoGroupRenderer
from .workflow_status import WorkflowStatusRenderer

RENDERER_REGISTRY: dict[str, type[BaseRenderer]] = {
    # Two-group independent comparisons
    "welch_t": TwoGroupRenderer,
    "student_t": TwoGroupRenderer,
    "mann_whitney_u": TwoGroupRenderer,
    # One-sample comparisons
    "one_sample_t": OneSampleRenderer,
    # Paired two-condition comparisons
    "paired_t": PairedRenderer,
    "wilcoxon_signed_rank": PairedRenderer,
    # Multi-group comparisons
    "welch_anova": MultiGroupRenderer,
    "one_way_anova": MultiGroupRenderer,
    "kruskal_wallis": MultiGroupRenderer,
    # Association analyses
    "pearson_correlation": AssociationRenderer,
    "spearman_correlation": AssociationRenderer,
    "kendall_tau_b": AssociationRenderer,
    "point_biserial_correlation": AssociationRenderer,
    "partial_pearson_correlation": AssociationRenderer,
    # Categorical associations
    "pearson_chi_square": CategoricalRenderer,
    "fisher_exact": CategoricalRenderer,
    # Paired categorical
    "mcnemar": PairedCategoricalRenderer,
    # Regression models
    "linear_regression": RegressionRenderer,
    "logistic_regression": LogisticRenderer,
    # Scale and rater reliability
    "cronbach_alpha": ReliabilityRenderer,
    "intraclass_correlation": ICCRenderer,
    # Repeated measures
    "repeated_measures_anova": RepeatedMeasuresRenderer,
    "friedman_test": RepeatedMeasuresRenderer,
    # Factorial designs
    "two_way_anova": FactorialRenderer,
}


def get_renderer(view: TerminalView, console: Console, detail: str = "standard") -> BaseRenderer:
    """Return an instantiated renderer for the given TerminalView."""
    if view.family == "family.profile":
        return ProfileRenderer(view, console, detail=detail)

    if view.family == "family.descriptive":
        return DescriptiveTableRenderer(view, console, detail=detail)

    if "workflow_status" in view.metadata:
        return WorkflowStatusRenderer(view, console, detail=detail)

    # Governance and planning result types
    if "planning_result" in view.metadata:
        return PlanningRenderer(view, console, detail=detail)

    if "sensitivity_result" in view.metadata:
        return SensitivityRenderer(view, console, detail=detail)

    if "practical_significance_result" in view.metadata:
        return PracticalSignificanceRenderer(view, console, detail=detail)

    if "plan" in view.metadata:
        return AnalysisPlanRenderer(view, console, detail=detail)

    if "adherence_result" in view.metadata:
        return PlanAdherenceRenderer(view, console, detail=detail)

    if "completeness_result" in view.metadata:
        return CompletenessRenderer(view, console, detail=detail)

    if "audit_result" in view.metadata or view.family == "family.audit":
        return AuditRenderer(view, console, detail=detail)

    if "record" in view.metadata or view.family == "family.reproducibility":
        return ReproducibilityRenderer(view, console, detail=detail)

    if "ledger" in view.metadata:
        return DecisionLedgerRenderer(view, console, detail=detail)

    if "snapshot" in view.metadata:
        return SessionRenderer(view, console, detail=detail)

    method_id = view.metadata.get("method_id")
    if method_id and method_id in RENDERER_REGISTRY:
        renderer_cls = RENDERER_REGISTRY[method_id]
        return renderer_cls(view, console, detail=detail)

    raise ValueError(
        f"No renderer registered for view family '{view.family}' or method '{method_id}'."
    )


__all__ = [
    "AnalysisPlanRenderer",
    "AssociationRenderer",
    "AuditRenderer",
    "BaseRenderer",
    "CategoricalRenderer",
    "CompletenessRenderer",
    "DecisionLedgerRenderer",
    "DescriptiveTableRenderer",
    "FactorialRenderer",
    "ICCRenderer",
    "LogisticRenderer",
    "MultiGroupRenderer",
    "OneSampleRenderer",
    "PairedCategoricalRenderer",
    "PairedRenderer",
    "PlanAdherenceRenderer",
    "PlanningRenderer",
    "PracticalSignificanceRenderer",
    "ProfileRenderer",
    "RENDERER_REGISTRY",
    "RegressionRenderer",
    "ReliabilityRenderer",
    "RepeatedMeasuresRenderer",
    "ReproducibilityRenderer",
    "SensitivityRenderer",
    "SessionRenderer",
    "TwoGroupRenderer",
    "WorkflowStatusRenderer",
    "get_renderer",
]
