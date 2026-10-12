"""Canonical assumption metadata and execution diagnostic linkage.

Protects against conflating pre-execution design requirements, structural data checks,
post-execution inferential diagnostics, and researcher domain review. Non-rejection of
a null hypothesis by a diagnostic test is never described as proving an assumption.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .specifications import AnalysisSpecification, StudyDesign, _json_value


class AssessmentMode:
    """Canonical lifecycle modes for statistical assumption checks."""

    RESEARCHER_DESIGN_FACT = "researcher_design_fact"
    PRE_EXECUTION_DATA_CHECK = "pre_execution_data_check"
    POST_EXECUTION_DIAGNOSTIC = "post_execution_diagnostic"
    RESEARCHER_REVIEW = "researcher_review"
    NOT_ASSESSABLE = "not_assessable"


class AssumptionStatus:
    """Lifecycle-safe statuses for assumption records."""

    CONFIRMED = "confirmed"
    UNRESOLVED = "unresolved"
    CHECKED = "checked"
    NOT_ASSESSED = "not_assessed"
    DIAGNOSTIC_AVAILABLE = "diagnostic_available"
    EVIDENCE_AGAINST = "evidence_against"
    NO_STRONG_EVIDENCE_AGAINST = "no_strong_evidence_against"
    REQUIRES_REVIEW = "requires_review"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class AssumptionCheck:
    """Canonical structured representation of an analytical assumption check.

    Distinguishes pre-execution design declarations, structural data checks,
    post-execution diagnostics, and uncheckable researcher review items.
    """

    assumption_id: str
    description: str
    assessment_mode: str
    status: str
    diagnostic_ids: tuple[str, ...] = ()
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return _json_value(
            {
                "schema_version": 1,
                "assumption_id": self.assumption_id,
                "description": self.description,
                "assessment_mode": self.assessment_mode,
                "status": self.status,
                "diagnostic_ids": list(self.diagnostic_ids),
                "message": self.message,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssumptionCheck:
        """Construct an AssumptionCheck from a dictionary."""
        return cls(
            assumption_id=str(data["assumption_id"]),
            description=str(data["description"]),
            assessment_mode=str(data["assessment_mode"]),
            status=str(data["status"]),
            diagnostic_ids=tuple(data.get("diagnostic_ids", ())),
            message=data.get("message"),
        )


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return cleaned.strip("_")


def build_recommendation_assumptions(
    method_id: str | None,
    spec: AnalysisSpecification,
    capability: Any,
) -> list[AssumptionCheck]:
    """Construct structured assumption records at the recommendation stage.

    No inferential diagnostic is run during recommendation.
    """
    if capability is None or not hasattr(capability, "assumptions"):
        return []

    checks: list[AssumptionCheck] = []

    # 1. Primary design fact (Independence / Pairing / Repeated units)
    design = spec.design
    if design == StudyDesign.INDEPENDENT:
        checks.append(
            AssumptionCheck(
                assumption_id="independent_observations",
                description="Independent observations",
                assessment_mode=AssessmentMode.RESEARCHER_DESIGN_FACT,
                status=AssumptionStatus.CONFIRMED,
                message=(
                    "Declared independent study design; "
                    "not statistically proven from observed values."
                ),
            )
        )
    elif design == StudyDesign.PAIRED:
        checks.append(
            AssumptionCheck(
                assumption_id="independent_pairs",
                description="Independent pairs",
                assessment_mode=AssessmentMode.RESEARCHER_DESIGN_FACT,
                status=AssumptionStatus.CONFIRMED,
                message=(
                    "Declared paired study design; pair independence is not verified from values."
                ),
            )
        )
    elif design == StudyDesign.REPEATED:
        checks.append(
            AssumptionCheck(
                assumption_id="independent_units",
                description="Units independent of other units",
                assessment_mode=AssessmentMode.RESEARCHER_DESIGN_FACT,
                status=AssumptionStatus.CONFIRMED,
                message=(
                    "Declared repeated-measures design; "
                    "unit independence is not verified from values."
                ),
            )
        )

    # 2. Method-specific required assumptions from capability
    for raw_desc in capability.assumptions:
        desc_lower = raw_desc.lower()

        # Skip duplicate independence since already handled above
        if (
            "independent observations" in desc_lower
            or "independent pairs" in desc_lower
            or "units independent" in desc_lower
            or "independent observational units" in desc_lower
            or "independent observational pairs" in desc_lower
        ):
            continue

        if (
            "equal population variances" in desc_lower
            or "equal within-cell residual variance" in desc_lower
        ):
            checks.append(
                AssumptionCheck(
                    assumption_id="homogeneity_of_variance",
                    description=raw_desc,
                    assessment_mode=AssessmentMode.POST_EXECUTION_DIAGNOSTIC,
                    status=AssumptionStatus.NOT_ASSESSED,
                    diagnostic_ids=("levene",),
                    message=(
                        "Equal variance condition to be evaluated via "
                        "post-execution Levene diagnostic."
                    ),
                )
            )
        elif "sphericity" in desc_lower:
            checks.append(
                AssumptionCheck(
                    assumption_id="sphericity",
                    description=raw_desc,
                    assessment_mode=AssessmentMode.POST_EXECUTION_DIAGNOSTIC,
                    status=AssumptionStatus.NOT_ASSESSED,
                    diagnostic_ids=("mauchly",),
                    message=(
                        "Sphericity condition to be evaluated via "
                        "post-execution Mauchly diagnostic."
                    ),
                )
            )
        elif (
            "normal model errors" in desc_lower
            or "normality" in desc_lower
            or "residual normality" in desc_lower
            or "mean-inference conditions" in desc_lower
        ):
            checks.append(
                AssumptionCheck(
                    assumption_id="normality",
                    description=raw_desc,
                    assessment_mode=AssessmentMode.POST_EXECUTION_DIAGNOSTIC,
                    status=AssumptionStatus.NOT_ASSESSED,
                    diagnostic_ids=("shapiro_residuals",)
                    if "residual" in desc_lower or "model" in desc_lower
                    else ("shapiro",),
                    message="Normality condition to be evaluated via post-execution diagnostic.",
                )
            )
            if "sampling" in desc_lower:
                checks.append(
                    AssumptionCheck(
                        assumption_id="sampling_intent",
                        description="Appropriate sampling and collection intent",
                        assessment_mode=AssessmentMode.RESEARCHER_REVIEW,
                        status=AssumptionStatus.REQUIRES_REVIEW,
                        message=(
                            "Appropriate sampling and collection intent require "
                            "researcher domain verification."
                        ),
                    )
                )
        elif "expected frequencies" in desc_lower or "expected cell count" in desc_lower:
            checks.append(
                AssumptionCheck(
                    assumption_id="expected_frequencies",
                    description=raw_desc,
                    assessment_mode=AssessmentMode.PRE_EXECUTION_DATA_CHECK,
                    status=AssumptionStatus.CHECKED,
                    diagnostic_ids=("expected_frequencies",),
                    message=(
                        "Expected cell count requirement evaluated "
                        "from observed contingency margins."
                    ),
                )
            )
        elif "cells populated" in desc_lower or "fully crossed" in desc_lower:
            checks.append(
                AssumptionCheck(
                    assumption_id="populated_cells",
                    description=raw_desc,
                    assessment_mode=AssessmentMode.PRE_EXECUTION_DATA_CHECK,
                    status=AssumptionStatus.CHECKED,
                    message="All factorial cells confirmed populated.",
                )
            )
        elif "one observation per unit" in desc_lower or "complete panel" in desc_lower:
            checks.append(
                AssumptionCheck(
                    assumption_id="complete_panel",
                    description=raw_desc,
                    assessment_mode=AssessmentMode.PRE_EXECUTION_DATA_CHECK,
                    status=AssumptionStatus.CHECKED,
                    message="Repeated-measures unit accounting verified across conditions.",
                )
            )
        elif "full-rank" in desc_lower:
            checks.append(
                AssumptionCheck(
                    assumption_id="full_rank_design",
                    description=raw_desc,
                    assessment_mode=AssessmentMode.PRE_EXECUTION_DATA_CHECK,
                    status=AssumptionStatus.CHECKED,
                    message="Predictor design matrix full-rank status verified.",
                )
            )
        elif "paired or matched units" in desc_lower:
            checks.append(
                AssumptionCheck(
                    assumption_id="paired_units",
                    description=raw_desc,
                    assessment_mode=AssessmentMode.RESEARCHER_DESIGN_FACT,
                    status=AssumptionStatus.CONFIRMED,
                    message="Explicit paired or matched units declared by researcher.",
                )
            )
        elif "two declared categorical factors" in desc_lower:
            checks.append(
                AssumptionCheck(
                    assumption_id="categorical_factors",
                    description=raw_desc,
                    assessment_mode=AssessmentMode.RESEARCHER_DESIGN_FACT,
                    status=AssumptionStatus.CONFIRMED,
                    message="Researcher declared two categorical factors.",
                )
            )
        elif "binary outcome with researcher-declared event level" in desc_lower:
            checks.append(
                AssumptionCheck(
                    assumption_id="declared_event_level",
                    description=raw_desc,
                    assessment_mode=AssessmentMode.RESEARCHER_DESIGN_FACT,
                    status=AssumptionStatus.CONFIRMED,
                    message="Binary event level declared by researcher.",
                )
            )
        elif "sampling" in desc_lower:
            checks.append(
                AssumptionCheck(
                    assumption_id="sampling_conditions",
                    description=raw_desc,
                    assessment_mode=AssessmentMode.RESEARCHER_REVIEW,
                    status=AssumptionStatus.REQUIRES_REVIEW,
                    message=(
                        "Appropriate sampling and collection intent require "
                        "researcher domain verification."
                    ),
                )
            )
        else:
            checks.append(
                AssumptionCheck(
                    assumption_id=_slug(raw_desc),
                    description=raw_desc,
                    assessment_mode=AssessmentMode.RESEARCHER_REVIEW,
                    status=AssumptionStatus.REQUIRES_REVIEW,
                    message="Substantive assumption requires researcher domain review.",
                )
            )

    return checks


def reconcile_assumptions(
    assumption_checks: Sequence[AssumptionCheck],
    analysis_result: Any,
) -> list[AssumptionCheck]:
    """Reconcile pre-execution assumption checks with actual post-execution diagnostics.

    Preserves historical pre-execution metadata, updates post-execution diagnostics
    with lifecycle-safe statuses (evidence_against or no_strong_evidence_against),
    and ensures non-rejection does not claim that an assumption is statistically proven.
    """
    if not assumption_checks:
        return []

    values: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    alpha_val = 0.05

    if analysis_result is not None:
        if isinstance(analysis_result, dict):
            if "diagnostics" in analysis_result and isinstance(
                analysis_result["diagnostics"], dict
            ):
                diagnostics = analysis_result["diagnostics"]
            elif (
                "values" in analysis_result
                and isinstance(analysis_result["values"], dict)
                and "diagnostics" in analysis_result["values"]
            ):
                diagnostics = analysis_result["values"]["diagnostics"]
            else:
                diagnostics = analysis_result
            values = (
                analysis_result.get("values", {})
                if isinstance(analysis_result.get("values"), dict)
                else analysis_result
            )
            metadata = (
                analysis_result.get("metadata", {})
                if isinstance(analysis_result.get("metadata"), dict)
                else {}
            )
        else:
            val_attr = getattr(analysis_result, "values", None)
            meta_attr = getattr(analysis_result, "metadata", None)
            values = val_attr if isinstance(val_attr, dict) else {}
            metadata = meta_attr if isinstance(meta_attr, dict) else {}
            diag_val = values.get("diagnostics") or metadata.get("diagnostics")
            if isinstance(diag_val, dict):
                diagnostics = diag_val
            spec = getattr(analysis_result, "specification", None)
            options = getattr(spec, "options", None)
            alpha_val = getattr(options, "alpha", 0.05) if options is not None else 0.05

    reconciled: list[AssumptionCheck] = []

    for check in assumption_checks:
        if check.assessment_mode != AssessmentMode.POST_EXECUTION_DIAGNOSTIC:
            reconciled.append(check)
            continue

        # Reconcile post-execution diagnostics
        if check.assumption_id == "homogeneity_of_variance":
            levene_p = None
            if isinstance(diagnostics, dict):
                levene_p = diagnostics.get("levene_p_value")
                if levene_p is None and "levene" in diagnostics:
                    lev = diagnostics["levene"]
                    if isinstance(lev, dict):
                        levene_p = lev.get("p_value")
                if levene_p is None and "homoscedasticity" in diagnostics:
                    homo = diagnostics["homoscedasticity"]
                    if isinstance(homo, dict) and "levene_median" in homo:
                        levene_p = homo["levene_median"].get("p_value")
                if levene_p is None and "breusch_pagan" in diagnostics:
                    bp = diagnostics["breusch_pagan"]
                    if isinstance(bp, dict):
                        levene_p = bp.get("lm_p_value")

            if levene_p is not None:
                if float(levene_p) < alpha_val:
                    reconciled.append(
                        AssumptionCheck(
                            assumption_id=check.assumption_id,
                            description=check.description,
                            assessment_mode=check.assessment_mode,
                            status=AssumptionStatus.EVIDENCE_AGAINST,
                            diagnostic_ids=check.diagnostic_ids or ("levene",),
                            message=(
                                "Advisory Levene diagnostic provides evidence against "
                                f"equal variance (p = {levene_p:.4g} < {alpha_val}); "
                                "method was retained as declared."
                            ),
                        )
                    )
                else:
                    reconciled.append(
                        AssumptionCheck(
                            assumption_id=check.assumption_id,
                            description=check.description,
                            assessment_mode=check.assessment_mode,
                            status=AssumptionStatus.NO_STRONG_EVIDENCE_AGAINST,
                            diagnostic_ids=check.diagnostic_ids or ("levene",),
                            message=(
                                "The diagnostic did not detect strong evidence against "
                                f"equal variance (p = {levene_p:.4g} >= {alpha_val}); "
                                "equal variance is not proven."
                            ),
                        )
                    )
            else:
                reconciled.append(check)

        elif check.assumption_id == "sphericity":
            sphericity = (
                values.get("sphericity")
                or metadata.get("sphericity")
                or diagnostics.get("sphericity")
                or diagnostics.get("mauchly")
            )
            if isinstance(sphericity, dict):
                p_mauchly = sphericity.get("p_value")
                if p_mauchly is not None:
                    if float(p_mauchly) < alpha_val:
                        reconciled.append(
                            AssumptionCheck(
                                assumption_id=check.assumption_id,
                                description=check.description,
                                assessment_mode=check.assessment_mode,
                                status=AssumptionStatus.EVIDENCE_AGAINST,
                                diagnostic_ids=("mauchly",),
                                message=(
                                    f"Mauchly's test indicates evidence against sphericity "
                                    f"(p = {p_mauchly:.4g}); "
                                    "Greenhouse-Geisser correction was applied."
                                ),
                            )
                        )
                    else:
                        reconciled.append(
                            AssumptionCheck(
                                assumption_id=check.assumption_id,
                                description=check.description,
                                assessment_mode=check.assessment_mode,
                                status=AssumptionStatus.NO_STRONG_EVIDENCE_AGAINST,
                                diagnostic_ids=("mauchly",),
                                message=(
                                    "Mauchly's test did not detect strong evidence against "
                                    f"sphericity (p = {p_mauchly:.4g}); sphericity is not proven."
                                ),
                            )
                        )
                else:
                    reconciled.append(check)
            else:
                reconciled.append(check)

        elif check.assumption_id == "normality":
            norm_diag = None
            if isinstance(diagnostics, dict):
                norm_diag = (
                    diagnostics.get("normality")
                    or diagnostics.get("residual_normality")
                    or diagnostics.get("shapiro")
                )
                if norm_diag is None and "shapiro_residuals" in diagnostics:
                    norm_diag = diagnostics["shapiro_residuals"]

            has_evidence_against = False
            found = False

            if isinstance(norm_diag, list):
                # e.g. group normality tests
                found = True
                for item in norm_diag:
                    if isinstance(item, dict):
                        p = item.get("p_value")
                        if p is not None and float(p) < alpha_val:
                            has_evidence_against = True
                            break
            elif isinstance(norm_diag, dict):
                found = True
                p = norm_diag.get("p_value")
                if p is None and "shapiro_wilk" in norm_diag:
                    sw = norm_diag["shapiro_wilk"]
                    if isinstance(sw, dict):
                        p = sw.get("p_value")
                if p is not None and float(p) < alpha_val:
                    has_evidence_against = True
                elif norm_diag.get("verdict") == "reject" or norm_diag.get("status") == "rejected":
                    has_evidence_against = True

            if found:
                if has_evidence_against:
                    reconciled.append(
                        AssumptionCheck(
                            assumption_id=check.assumption_id,
                            description=check.description,
                            assessment_mode=check.assessment_mode,
                            status=AssumptionStatus.EVIDENCE_AGAINST,
                            diagnostic_ids=check.diagnostic_ids or ("shapiro",),
                            message=(
                                "Advisory diagnostic provides evidence against the assessed "
                                "normality condition; method was retained as declared."
                            ),
                        )
                    )
                else:
                    reconciled.append(
                        AssumptionCheck(
                            assumption_id=check.assumption_id,
                            description=check.description,
                            assessment_mode=check.assessment_mode,
                            status=AssumptionStatus.NO_STRONG_EVIDENCE_AGAINST,
                            diagnostic_ids=check.diagnostic_ids or ("shapiro",),
                            message=(
                                "The diagnostic did not detect strong evidence against the "
                                "assessed normality condition; normality is not "
                                "statistically proven."
                            ),
                        )
                    )
            else:
                reconciled.append(check)
        else:
            reconciled.append(check)

    return reconciled


def assumptions_to_legacy_dict(checks: Sequence[AssumptionCheck]) -> list[dict[str, Any]]:
    """Convert canonical AssumptionCheck records to the legacy dict structure."""
    return [
        {
            "assumption": (
                check.assumption_id
                if check.assumption_id
                in {"independent_observations", "independent_pairs", "independent_units"}
                else check.description
            ),
            "assumption_id": check.assumption_id,
            "description": check.description,
            "category": (
                "researcher_design_fact"
                if check.assessment_mode == AssessmentMode.RESEARCHER_DESIGN_FACT
                else (
                    "data_checkable"
                    if check.assessment_mode == AssessmentMode.PRE_EXECUTION_DATA_CHECK
                    else "not_fully_checkable"
                )
            ),
            "assessment_mode": check.assessment_mode,
            "status": check.status,
            "diagnostic_ids": list(check.diagnostic_ids),
            "message": check.message,
        }
        for check in checks
    ]
