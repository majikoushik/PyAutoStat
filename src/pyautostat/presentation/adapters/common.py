"""Common helper utilities for presentation adapters."""

from __future__ import annotations

from typing import Any

from ...results import AnalysisResult
from ...workflow import ResearchWorkflowResult
from ..models import (
    DisplayDiagnostic,
)


def first_present(record: dict[str, Any] | Any, *keys: str, default: Any = None) -> Any:
    """Return the first non-None value among keys from a dictionary or object.

    Unlike boolean 'or', this safely preserves 0, 0.0, and False.
    """
    if isinstance(record, dict):
        for key in keys:
            if key in record and record[key] is not None:
                return record[key]
        return default
    for key in keys:
        if hasattr(record, key):
            val = getattr(record, key)
            if val is not None:
                return val
    return default


def extract_context(
    target: ResearchWorkflowResult | AnalysisResult,
) -> tuple[
    AnalysisResult,
    ResearchWorkflowResult | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
]:
    """Extract standard analysis components from workflow or direct AnalysisResult.

    Returns: (analysis, workflow, specification, interpretation, recommendation, audit)
    """
    if isinstance(target, ResearchWorkflowResult):
        assert target.analysis is not None
        return (
            target.analysis,
            target,
            target.specification,
            target.interpretation,
            target.recommendation,
            target.audit,
        )
    return (
        target,
        None,
        target.specification,
        None,
        target.recommendation,
        None,
    )


def extract_diagnostics(
    analysis: AnalysisResult,
    diagnostics_dict: dict[str, Any] | None = None,
) -> list[DisplayDiagnostic]:
    """Extract diagnostics into normalized DisplayDiagnostic records."""
    diagnostics_list: list[DisplayDiagnostic] = []
    diag_source = (
        diagnostics_dict if diagnostics_dict is not None else analysis.metadata.get("diagnostics")
    )

    if isinstance(diag_source, dict):
        for k, v in diag_source.items():
            label = str(k).replace("_", " ").title()
            if isinstance(v, dict):
                status = str(v.get("status", "Documented")).upper()
                msg = v.get("message") or v.get("detail") or str(v)
                sev = (
                    "review"
                    if "reject" in status.lower() or "violation" in status.lower()
                    else "neutral"
                )
            else:
                msg = str(v)
                status = "DOCUMENTED"
                sev = "neutral"
            diagnostics_list.append(
                DisplayDiagnostic(
                    label=label,
                    status=status,
                    detail=msg,
                    severity=sev,
                )
            )

    for assumption in analysis.assumptions:
        diagnostics_list.append(
            DisplayDiagnostic(
                label="Assumption",
                status="REVIEW",
                detail=(
                    f"Required conditions: {assumption}. Calculation alone does not verify them."
                ),
                severity="review",
            )
        )

    return diagnostics_list


def build_metadata_dict(
    analysis: AnalysisResult,
    workflow: ResearchWorkflowResult | None,
) -> dict[str, Any]:
    """Build shared metadata dictionary for TerminalView."""
    meta: dict[str, Any] = {
        "method_id": analysis.method_id,
        "method_name": analysis.metadata.get("method_name") or analysis.method_id,
        "rationale": workflow.recommendation.rationale
        if workflow and workflow.recommendation
        else None,
        "audit_status": workflow.audit.status if workflow and workflow.audit else None,
        "sample_size": analysis.sample_size,
        "excluded_rows": analysis.excluded_rows,
        "numerical_source": analysis.metadata.get("numerical_source"),
    }
    if workflow and workflow.reproducibility:
        meta["reproducibility"] = {
            "method_id": analysis.method_id,
            "seed": workflow.reproducibility.payload.get("random_seed")
            if hasattr(workflow.reproducibility, "payload")
            else None,
        }
    return meta


def format_interpretation_text(interp: Any, detail: str = "standard") -> str | None:
    """Format stored deterministic interpretation based on detail level."""
    if interp is None:
        return None
    if isinstance(interp, str):
        return interp
    parts: list[str] = []
    hyp = getattr(interp, "hypothesis_interpretation", None)
    if hyp:
        parts.append(str(hyp).strip())
    if detail == "full":
        eff = getattr(interp, "effect_interpretation", None)
        if eff:
            parts.append(str(eff).strip())
        unc = getattr(interp, "uncertainty_interpretation", None)
        if unc:
            parts.append(str(unc).strip())
    if parts:
        return "\n\n".join(parts)
    return getattr(interp, "summary", None) or getattr(interp, "conclusion", None)
