"""Adapters for paired two-condition inferential workflows."""

from __future__ import annotations

from ...results import AnalysisResult
from ...workflow import ResearchWorkflowResult
from ..formatting import (
    format_confidence_interval,
    format_confidence_level_label,
    format_effect,
    format_number,
    format_p_value,
    format_sample_size,
    format_statistic,
)
from ..models import (
    DisplayDiagnostic,
    DisplayMetric,
    TerminalView,
)
from .common import (
    build_metadata_dict,
    extract_context,
    extract_diagnostics,
    first_present,
    format_interpretation_text,
    resolve_confidence_level,
)


def adapt_paired_t(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a paired-samples t-test workflow."""
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    outcome = (spec.question.outcome if spec and spec.question else None) or "Outcome"
    condition = (spec.question.predictor if spec and spec.question else None) or "Condition"
    unit_id = analysis.metadata.get("unit_id") or "unit_id"
    unit = (spec.data_dictionary or {}).get(outcome, {}).get("unit") if spec else None
    conf_level = resolve_confidence_level(analysis, spec)

    # Condition order & contrast
    condition_order = analysis.metadata.get("condition_order") or analysis.metadata.get(
        "group_order", []
    )
    if len(condition_order) >= 2:
        contrast_str = f"'{condition_order[0]}' - '{condition_order[1]}'"
    else:
        contrast_str = "Unavailable"

    sample_meta = analysis.metadata.get("sample", {})
    complete_pairs = first_present(sample_meta, "complete_pairs", default=analysis.sample_size)
    incomplete_units = first_present(sample_meta, "incomplete_units", default=0)

    design_metrics = (
        DisplayMetric("Method", "Paired-samples t-test", role="method"),
        DisplayMetric("Estimand", "Population mean paired difference"),
        DisplayMetric("Pairing Unit", str(unit_id)),
        DisplayMetric("Contrast", contrast_str),
        DisplayMetric("Complete Pairs", f"{format_sample_size(complete_pairs)} pairs"),
        DisplayMetric("Incomplete Units", f"{format_sample_size(incomplete_units)} units"),
    )

    diff = analysis.values.get("primary_estimate")
    diff_str = format_number(diff, decimals=2, unit=unit)
    ci_dict = analysis.values.get("confidence_interval")
    ci_str = format_confidence_interval(ci_dict, decimals=2)
    p_val = analysis.values.get("p_value")
    p_str = format_p_value(p_val)
    stat = analysis.values.get("test_statistic")
    df = analysis.values.get("degrees_of_freedom")

    ci_label = format_confidence_level_label(ci_dict, confidence_level=conf_level)

    key_metrics_list = [
        DisplayMetric("Mean difference", diff_str, role="result.estimate"),
        DisplayMetric(ci_label, ci_str, role="result.ci"),
    ]

    effect_obj = analysis.values.get("effect_size")
    effect_val = None
    if isinstance(effect_obj, dict):
        effect_val = effect_obj.get("value")
        effect_str = format_effect(effect_val, decimals=2)
        effect_ci = effect_obj.get("confidence_interval")
        effect_ci_str = format_confidence_interval(effect_ci, decimals=2)
        effect_ci_label = format_confidence_level_label(
            effect_ci, confidence_level=conf_level, prefix="Effect"
        )
        if effect_val is not None:
            key_metrics_list.append(DisplayMetric("Cohen's dz", effect_str, role="result.effect"))
        if effect_ci is not None:
            key_metrics_list.append(DisplayMetric(effect_ci_label, effect_ci_str, role="result.ci"))

    key_metrics_list.append(DisplayMetric("p-value", p_str, role="result.evidence"))
    if detail == "full" and stat is not None:
        key_metrics_list.append(DisplayMetric("Test statistic", format_statistic("t", stat, df=df)))

    cond_detail = (
        f"Condition '{condition_order[0]}' minus '{condition_order[1]}'."
        if len(condition_order) >= 2
        else "Condition contrast order unavailable."
    )
    diagnostics_list = [
        DisplayDiagnostic(
            label="Condition Order",
            status="PRESERVED",
            detail=cond_detail,
            severity="neutral",
        ),
        DisplayDiagnostic(
            label="Pairing",
            status="COMPLETE UNITS",
            detail=f"{complete_pairs} matched pairs retained for analysis.",
            severity="neutral",
        ),
    ]
    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("practical importance" in lim for lim in limitations):
        limitations.append(
            "Statistical significance alone does not establish practical importance."
        )
    if not any("exchangeable" in lim for lim in limitations):
        limitations.append(
            "Paired inference assumes independent pairing units and "
            "exchangeable paired differences."
        )

    compact_text = (
        f"Paired t-test | N={format_sample_size(complete_pairs)} pairs | "
        f"diff={diff_str} | {ci_label} {ci_str.replace(' to ', '...')}"
    )
    if effect_val is not None:
        compact_text += f" | dz={format_effect(effect_val, decimals=2)}"
    compact_text += f" | p={p_str}"

    metadata = build_metadata_dict(analysis, workflow)
    metadata["condition_order"] = condition_order

    return TerminalView(
        title="Paired Two-Condition Mean Comparison",
        subtitle=f"{outcome} by {condition} (within {unit_id})",
        family="family.mean",
        design_metrics=design_metrics,
        key_metrics=tuple(key_metrics_list),
        tables=(),
        diagnostics=tuple(diagnostics_list),
        interpretation=interpretation_text,
        limitations=tuple(limitations),
        warnings=analysis.warnings,
        metadata=metadata,
        compact_text=compact_text,
    )


def adapt_wilcoxon_signed_rank(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a Wilcoxon signed-rank test workflow."""
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    outcome = (spec.question.outcome if spec and spec.question else None) or "Outcome"
    condition = (spec.question.predictor if spec and spec.question else None) or "Condition"
    unit_id = analysis.metadata.get("unit_id") or "unit_id"
    conf_level = resolve_confidence_level(analysis, spec)

    condition_order = analysis.metadata.get("condition_order") or analysis.metadata.get(
        "group_order", []
    )
    if len(condition_order) >= 2:
        contrast_str = f"'{condition_order[0]}' vs '{condition_order[1]}'"
    else:
        contrast_str = "Unavailable"

    sample_meta = analysis.metadata.get("sample", {})
    complete_pairs = first_present(sample_meta, "complete_pairs", default=analysis.sample_size)
    nonzero_diffs = first_present(sample_meta, "nonzero_differences", default="Unavailable")
    zero_diffs = first_present(sample_meta, "zero_differences", default=0)

    design_metrics = (
        DisplayMetric("Method", "Wilcoxon signed-rank test", role="method"),
        DisplayMetric("Estimand", "Paired-difference signed-rank distribution"),
        DisplayMetric("Pairing Unit", str(unit_id)),
        DisplayMetric("Contrast", contrast_str),
        DisplayMetric("Complete Pairs", f"{format_sample_size(complete_pairs)} pairs"),
        DisplayMetric("Non-zero Diffs", f"{format_sample_size(nonzero_diffs)} pairs"),
        DisplayMetric("Zero Diffs", f"{format_sample_size(zero_diffs)} pairs"),
    )

    stat = analysis.values.get("test_statistic")
    stat_str = format_number(stat, decimals=1)
    p_val = analysis.values.get("p_value")
    p_str = format_p_value(p_val)
    effect_obj = analysis.values.get("effect_size")
    effect_val = None
    ci_dict = None
    if isinstance(effect_obj, dict):
        effect_val = effect_obj.get("value")
        ci_dict = effect_obj.get("confidence_interval")
    if ci_dict is None:
        ci_dict = analysis.values.get("confidence_interval")

    ci_str = format_confidence_interval(ci_dict, decimals=3)
    ci_label = format_confidence_level_label(ci_dict, confidence_level=conf_level, prefix="Effect")
    short_ci_label = format_confidence_level_label(ci_dict, confidence_level=conf_level)

    key_metrics_list = [
        DisplayMetric("Wilcoxon W", stat_str, role="result.estimate"),
    ]
    if effect_val is not None:
        key_metrics_list.append(
            DisplayMetric(
                "Rank-biserial r", format_effect(effect_val, decimals=3), role="result.effect"
            )
        )
    if ci_dict is not None:
        key_metrics_list.append(DisplayMetric(ci_label, ci_str, role="result.ci"))
    key_metrics_list.append(DisplayMetric("p-value", p_str, role="result.evidence"))

    wilc_cond_detail = (
        f"Positive ranks favor condition '{condition_order[0]}'."
        if len(condition_order) >= 1
        else "Condition order unavailable."
    )
    diagnostics_list = [
        DisplayDiagnostic(
            label="Zero Differences",
            status="POLICY",
            detail=f"{zero_diffs} zero differences omitted from ranks (wilcox policy).",
            severity="neutral",
        ),
        DisplayDiagnostic(
            label="Condition Order",
            status="PRESERVED",
            detail=wilc_cond_detail,
            severity="neutral",
        ),
    ]
    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("universally a test of medians" in lim for lim in limitations):
        limitations.append(
            "Wilcoxon signed-rank test evaluates the distribution of paired differences; "
            "it is not universally a test of medians."
        )
    if not any("practical importance" in lim for lim in limitations):
        limitations.append(
            "Statistical significance alone does not establish practical importance."
        )

    compact_text = (
        f"Wilcoxon signed-rank | N={format_sample_size(complete_pairs)} pairs | W={stat_str}"
    )
    if effect_val is not None:
        compact_text += f" | r_rb={format_effect(effect_val, decimals=3)}"
    if ci_dict is not None:
        compact_text += f" | {short_ci_label} {ci_str.replace(' to ', '...')}"
    compact_text += f" | p={p_str}"

    metadata = build_metadata_dict(analysis, workflow)
    metadata["condition_order"] = condition_order

    return TerminalView(
        title="Paired Signed-Rank Comparison",
        subtitle=f"{outcome} by {condition} (within {unit_id})",
        family="family.rank",
        design_metrics=design_metrics,
        key_metrics=tuple(key_metrics_list),
        tables=(),
        diagnostics=tuple(diagnostics_list),
        interpretation=interpretation_text,
        limitations=tuple(limitations),
        warnings=analysis.warnings,
        metadata=metadata,
        compact_text=compact_text,
    )
