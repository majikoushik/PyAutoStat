"""Adapters for independent group comparisons and one-sample mean inference."""

from __future__ import annotations

from ...results import AnalysisResult
from ...workflow import ResearchWorkflowResult
from ..formatting import (
    format_confidence_interval,
    format_effect,
    format_number,
    format_p_value,
    format_sample_size,
    format_statistic,
)
from ..models import (
    DisplayDiagnostic,
    DisplayMetric,
    DisplayRow,
    DisplayTable,
    TerminalView,
)
from .common import (
    build_metadata_dict,
    extract_context,
    extract_diagnostics,
    format_interpretation_text,
)


def adapt_welch_t(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a Welch independent-samples t-test workflow."""
    return _adapt_two_group_mean(target, detail=detail, method_id="welch_t")


def adapt_student_t(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a Student independent-samples t-test workflow."""
    return _adapt_two_group_mean(target, detail=detail, method_id="student_t")


def _adapt_two_group_mean(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
    method_id: str = "welch_t",
) -> TerminalView:
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    outcome = (spec.question.outcome if spec and spec.question else None) or "Outcome"
    predictor = (spec.question.predictor if spec and spec.question else None) or "Predictor"
    unit = (spec.data_dictionary or {}).get(outcome, {}).get("unit") if spec else None

    # Contrast
    contrast_dict = analysis.metadata.get("contrast")
    groups = analysis.metadata.get("group_order", [])
    if isinstance(contrast_dict, dict) and "first" in contrast_dict and "second" in contrast_dict:
        contrast_str = f"'{contrast_dict['first']}' - '{contrast_dict['second']}'"
    elif len(groups) == 2:
        contrast_str = f"'{groups[0]}' - '{groups[1]}'"
    else:
        contrast_str = "Unavailable"

    is_welch = method_id == "welch_t"
    method_name = (
        "Welch's independent-samples t-test" if is_welch else "Student's independent-samples t-test"
    )

    sample_size = analysis.sample_size
    excluded_rows = analysis.excluded_rows or 0
    design_metrics = (
        DisplayMetric("Method", method_name, role="method"),
        DisplayMetric("Estimand", "Population mean difference"),
        DisplayMetric("Contrast", contrast_str),
        DisplayMetric("Sample", f"{format_sample_size(sample_size)} rows"),
        DisplayMetric("Excluded", f"{format_sample_size(excluded_rows)} rows"),
    )

    # Group summary table
    tables: list[DisplayTable] = []
    group_sizes = analysis.metadata.get("sample", {}).get("group_sizes", [])
    if group_sizes:
        rows: list[DisplayRow] = []
        for g_info in group_sizes:
            g_name = str(g_info.get("group", ""))
            g_n = format_sample_size(g_info.get("size"))
            rows.append(DisplayRow((g_name, g_n)))
        tables.append(
            DisplayTable(
                title="GROUP SUMMARY",
                columns=("Group", "N"),
                rows=tuple(rows),
            )
        )

    # Key results
    diff = analysis.values.get("primary_estimate")
    diff_str = format_number(diff, decimals=2, unit=unit)
    ci_dict = analysis.values.get("confidence_interval")
    ci_str = format_confidence_interval(ci_dict, decimals=2)
    effect_dict = analysis.values.get("effect_size", {})
    effect_val = effect_dict.get("value")
    effect_str = format_effect(effect_val, decimals=2)
    effect_ci = effect_dict.get("confidence_interval")
    effect_ci_str = format_confidence_interval(effect_ci, decimals=2)
    p_val = analysis.values.get("p_value")
    p_str = format_p_value(p_val)
    stat = analysis.values.get("test_statistic")
    df = analysis.values.get("degrees_of_freedom")

    key_metrics_list = [
        DisplayMetric("Mean difference", diff_str, role="result.estimate"),
        DisplayMetric("95% CI", ci_str, role="result.ci"),
        DisplayMetric("Cohen's d", effect_str, role="result.effect"),
        DisplayMetric("Effect 95% CI", effect_ci_str, role="result.ci"),
        DisplayMetric("p-value", p_str, role="result.evidence"),
    ]
    if detail == "full" and stat is not None:
        key_metrics_list.append(DisplayMetric("Test statistic", format_statistic("t", stat, df=df)))

    # Diagnostics
    diagnostics_list = [
        DisplayDiagnostic(
            label="Equal variance",
            status="Not assumed" if is_welch else "Assumed (pooled)",
            detail="Welch test does not assume equal population variances."
            if is_welch
            else "Student t assumes equal population variances (pooled SD).",
            severity="neutral" if is_welch else "review",
        ),
        DisplayDiagnostic(
            label="Missingness",
            status="Complete cases",
            detail=f"{excluded_rows} missing or excluded rows.",
            severity="neutral",
        ),
    ]
    diagnostics_list.extend(extract_diagnostics(analysis))

    # Narrative interpretation
    interpretation_text = format_interpretation_text(interp, detail=detail)

    # Limitations
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("practical importance" in lim for lim in limitations):
        limitations.append(
            "Statistical significance alone does not establish practical importance."
        )
    if not is_welch:
        limitations.append("Equal-variance Student t is sensitive to variance heterogeneity.")

    # Compact text
    compact_text = (
        f"{'Welch' if is_welch else 'Student'} t-test | "
        f"N={format_sample_size(sample_size)} | "
        f"diff={diff_str} | 95% CI {ci_str.replace(' to ', '...')} | "
        f"d={effect_str} | p={p_str}"
    )

    metadata = build_metadata_dict(analysis, workflow)
    metadata["contrast"] = contrast_str

    return TerminalView(
        title="Independent Group Comparison",
        subtitle=f"{outcome} by {predictor}",
        family="family.mean",
        design_metrics=design_metrics,
        key_metrics=tuple(key_metrics_list),
        tables=tuple(tables),
        diagnostics=tuple(diagnostics_list),
        interpretation=interpretation_text,
        limitations=tuple(limitations),
        warnings=analysis.warnings,
        metadata=metadata,
        compact_text=compact_text,
    )


def adapt_mann_whitney_u(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a Mann-Whitney U rank comparison workflow."""
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    outcome = (spec.question.outcome if spec and spec.question else None) or "Outcome"
    predictor = (spec.question.predictor if spec and spec.question else None) or "Predictor"

    groups = analysis.metadata.get("group_order", [])
    contrast_dict = analysis.metadata.get("contrast")
    if isinstance(contrast_dict, dict) and "first" in contrast_dict and "second" in contrast_dict:
        contrast_str = f"'{contrast_dict['first']}' vs '{contrast_dict['second']}'"
    elif len(groups) == 2:
        contrast_str = f"'{groups[0]}' vs '{groups[1]}'"
    else:
        contrast_str = "Unavailable"

    sample_size = analysis.sample_size
    excluded_rows = analysis.excluded_rows or 0
    design_metrics = (
        DisplayMetric("Method", "Mann-Whitney U rank test", role="method"),
        DisplayMetric("Estimand", "Stochastic superiority / rank distribution difference"),
        DisplayMetric("Contrast", contrast_str),
        DisplayMetric("Sample", f"{format_sample_size(sample_size)} rows"),
        DisplayMetric("Excluded", f"{format_sample_size(excluded_rows)} rows"),
    )

    # Group summaries table
    tables: list[DisplayTable] = []
    group_sizes = analysis.metadata.get("sample", {}).get("group_sizes", [])
    if group_sizes:
        rows: list[DisplayRow] = []
        for g_info in group_sizes:
            g_name = str(g_info.get("group", ""))
            g_n = format_sample_size(g_info.get("size"))
            rows.append(DisplayRow((g_name, g_n)))
        tables.append(
            DisplayTable(
                title="GROUP SAMPLE SIZES",
                columns=("Group", "N"),
                rows=tuple(rows),
            )
        )

    # Key results
    u_stat = analysis.values.get("test_statistic")
    u_str = format_number(u_stat, decimals=1)
    p_val = analysis.values.get("p_value")
    p_str = format_p_value(p_val)
    effect_dict = analysis.values.get("effect_size", {})
    effect_val = effect_dict.get("value")
    effect_str = format_effect(effect_val, decimals=3)
    ci_dict = effect_dict.get("confidence_interval") or analysis.values.get("confidence_interval")
    ci_str = format_confidence_interval(ci_dict, decimals=3)

    key_metrics_list = [
        DisplayMetric("Mann-Whitney U", u_str, role="result.estimate"),
        DisplayMetric("Rank-biserial r", effect_str, role="result.effect"),
        DisplayMetric("Effect 95% CI", ci_str, role="result.ci"),
        DisplayMetric("p-value", p_str, role="result.evidence"),
    ]

    diagnostics_list = [
        DisplayDiagnostic(
            label="Rank Model",
            status="Nonparametric",
            detail="Evaluates relative ranks; does not require outcome normality.",
            severity="neutral",
        ),
    ]
    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("universally a test of medians" in lim for lim in limitations):
        limitations.append(
            "Mann-Whitney U evaluates rank distributions; it is not universally a test of medians."
        )
    if not any("practical importance" in lim for lim in limitations):
        limitations.append(
            "Statistical significance alone does not establish practical importance."
        )

    compact_text = (
        f"Mann-Whitney U | N={format_sample_size(sample_size)} | "
        f"U={u_str} | r_rb={effect_str} | 95% CI {ci_str.replace(' to ', '...')} | p={p_str}"
    )

    return TerminalView(
        title="Independent Group Rank Comparison",
        subtitle=f"{outcome} by {predictor}",
        family="family.rank",
        design_metrics=design_metrics,
        key_metrics=tuple(key_metrics_list),
        tables=tuple(tables),
        diagnostics=tuple(diagnostics_list),
        interpretation=interpretation_text,
        limitations=tuple(limitations),
        warnings=analysis.warnings,
        metadata=build_metadata_dict(analysis, workflow),
        compact_text=compact_text,
    )


def adapt_one_sample_t(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a one-sample t-test workflow."""
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    outcome = (spec.question.outcome if spec and spec.question else None) or "Outcome"
    ref_val = analysis.values.get("reference_value")
    ref_str = format_number(ref_val, decimals=2)
    unit = (spec.data_dictionary or {}).get(outcome, {}).get("unit") if spec else None

    sample_size = analysis.sample_size
    excluded_rows = analysis.excluded_rows or 0
    sample_mean = analysis.values.get("sample_mean")
    sample_mean_str = format_number(sample_mean, decimals=2, unit=unit)

    design_metrics = (
        DisplayMetric("Method", "One-sample t-test", role="method"),
        DisplayMetric("Estimand", "Population mean relative to reference"),
        DisplayMetric("Outcome", outcome),
        DisplayMetric("Reference Value", ref_str),
        DisplayMetric("Sample Mean", sample_mean_str),
        DisplayMetric("Sample", f"{format_sample_size(sample_size)} rows"),
        DisplayMetric("Excluded", f"{format_sample_size(excluded_rows)} rows"),
    )

    diff = analysis.values.get("primary_estimate")
    diff_str = format_number(diff, decimals=2, unit=unit)
    ci_dict = analysis.values.get("confidence_interval")
    ci_str = format_confidence_interval(ci_dict, decimals=2)
    effect_dict = analysis.values.get("effect_size", {})
    effect_val = effect_dict.get("value")
    effect_str = format_effect(effect_val, decimals=2)
    effect_ci = effect_dict.get("confidence_interval")
    effect_ci_str = format_confidence_interval(effect_ci, decimals=2)
    p_val = analysis.values.get("p_value")
    p_str = format_p_value(p_val)
    stat = analysis.values.get("test_statistic")
    df = analysis.values.get("degrees_of_freedom")

    key_metrics_list = [
        DisplayMetric("Observed - Reference", diff_str, role="result.estimate"),
        DisplayMetric("95% CI", ci_str, role="result.ci"),
        DisplayMetric("Cohen's d", effect_str, role="result.effect"),
        DisplayMetric("Effect 95% CI", effect_ci_str, role="result.ci"),
        DisplayMetric("p-value", p_str, role="result.evidence"),
    ]
    if detail == "full" and stat is not None:
        key_metrics_list.append(DisplayMetric("Test statistic", format_statistic("t", stat, df=df)))

    diagnostics_list = [
        DisplayDiagnostic(
            label="Contrast",
            status="PRESERVED",
            detail=f"Observed sample mean ({sample_mean_str}) minus reference ({ref_str}).",
            severity="neutral",
        )
    ]
    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("practical importance" in lim for lim in limitations):
        limitations.append(
            "Statistical significance alone does not establish practical importance."
        )

    compact_text = (
        f"One-sample t-test | N={format_sample_size(sample_size)} | "
        f"diff={diff_str} | 95% CI {ci_str.replace(' to ', '...')} | "
        f"d={effect_str} | p={p_str}"
    )

    return TerminalView(
        title="One-Sample Mean Comparison",
        subtitle=f"{outcome} vs {ref_str}",
        family="family.mean",
        design_metrics=design_metrics,
        key_metrics=tuple(key_metrics_list),
        tables=(),
        diagnostics=tuple(diagnostics_list),
        interpretation=interpretation_text,
        limitations=tuple(limitations),
        warnings=analysis.warnings,
        metadata=build_metadata_dict(analysis, workflow),
        compact_text=compact_text,
    )
