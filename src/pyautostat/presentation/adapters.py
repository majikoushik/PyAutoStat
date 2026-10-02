"""Adapters from PyAutoStat domain/result objects to normalized TerminalViews.

These adapters read authoritative stored results. They NEVER recompute or recalculate
inferential statistics, confidence intervals, p-values, or estimands.
"""

from __future__ import annotations

from typing import Any

from ..workflow import ResearchWorkflowResult, WorkflowStatus
from .formatting import (
    format_confidence_interval,
    format_effect,
    format_memory_bytes,
    format_number,
    format_p_value,
    format_sample_size,
)
from .models import (
    DisplayDiagnostic,
    DisplayMetric,
    DisplayRow,
    DisplayTable,
    TerminalView,
)


class UnsupportedPresentationError(Exception):
    """Raised when an object or statistical method is not supported by the presentation layer."""


def is_profile(target: Any) -> bool:
    """Return True if target looks like a DatasetProfiler result dictionary."""
    if not isinstance(target, dict):
        return False
    return "overview" in target and ("descriptive" in target or "data_quality" in target)


def adapt(target: Any, detail: str = "standard") -> TerminalView:
    """Convert an authoritative PyAutoStat result into a normalized TerminalView."""
    if detail not in {"compact", "standard", "full"}:
        raise ValueError(
            f"Invalid detail mode '{detail}'. Expected 'compact', 'standard', or 'full'."
        )

    if isinstance(target, ResearchWorkflowResult):
        return adapt_workflow(target, detail=detail)

    if is_profile(target):
        return adapt_profile(target, detail=detail)

    raise TypeError(
        f"Unsupported target type for show(): {type(target).__name__}. "
        "Expected ResearchWorkflowResult or dataset profile mapping."
    )


def adapt_workflow(workflow: ResearchWorkflowResult, detail: str = "standard") -> TerminalView:
    """Adapt a ResearchWorkflowResult into a TerminalView."""
    if workflow.status not in {WorkflowStatus.COMPLETED, WorkflowStatus.PARTIAL}:
        return adapt_workflow_status(workflow, detail=detail)

    if workflow.analysis is None:
        return adapt_workflow_status(workflow, detail=detail)

    method_id = workflow.analysis.method_id
    if method_id == "welch_t":
        return adapt_welch_t(workflow, detail=detail)
    elif method_id == "pearson_correlation":
        return adapt_pearson(workflow, detail=detail)

    raise UnsupportedPresentationError(
        f"Method '{method_id}' is not supported in the initial presentation layer pilot. "
        "Supported pilot methods: 'welch_t', 'pearson_correlation'."
    )


def adapt_welch_t(workflow: ResearchWorkflowResult, detail: str = "standard") -> TerminalView:
    """Adapt a Welch independent-samples t-test workflow."""
    analysis = workflow.analysis
    assert analysis is not None
    interp = workflow.interpretation
    spec = workflow.specification

    outcome = spec.question.outcome or "Outcome"
    predictor = spec.question.predictor or "Predictor"
    unit = (spec.data_dictionary or {}).get(outcome, {}).get("unit")

    # Contrast
    contrast_dict = analysis.metadata.get("contrast")
    groups = analysis.metadata.get("group_order", [])
    if isinstance(contrast_dict, dict) and "first" in contrast_dict and "second" in contrast_dict:
        contrast_str = f"'{contrast_dict['first']}' - '{contrast_dict['second']}'"
    elif len(groups) == 2:
        contrast_str = f"'{groups[0]}' - '{groups[1]}'"
    else:
        contrast_str = "Unavailable"

    # Design metrics
    sample_size = analysis.sample_size
    excluded_rows = analysis.excluded_rows or 0
    design_metrics = (
        DisplayMetric("Method", "Welch's independent-samples t-test", role="method"),
        DisplayMetric("Estimand", "Population mean difference"),
        DisplayMetric("Contrast", contrast_str),
        DisplayMetric("Sample", f"{format_sample_size(sample_size)} rows"),
        DisplayMetric("Excluded", f"{format_sample_size(excluded_rows)} rows"),
    )

    # Key result metrics
    diff_val = analysis.values.get("primary_estimate")
    diff_str = format_number(diff_val, decimals=2, unit=unit)

    ci_dict = analysis.values.get("confidence_interval")
    conf_level = int(ci_dict.get("level", 0.95) * 100) if isinstance(ci_dict, dict) else 95
    ci_str = format_confidence_interval(ci_dict, decimals=2)

    eff_dict = analysis.values.get("effect_size", {})
    d_val = eff_dict.get("value")
    d_str = format_effect(d_val, decimals=2)

    eff_ci = eff_dict.get("confidence_interval")
    eff_ci_str = format_confidence_interval(eff_ci, decimals=2)

    p_val = analysis.values.get("p_value")
    p_str = format_p_value(p_val)

    key_metrics = (
        DisplayMetric("Mean difference", diff_str, role="result.estimate"),
        DisplayMetric(f"{conf_level}% CI", ci_str, role="result.ci"),
        DisplayMetric("Cohen's d", d_str, role="result.effect"),
        DisplayMetric("Effect 95% CI", eff_ci_str, role="result.effect"),
        DisplayMetric("p-value", p_str, role="result.evidence"),
    )

    # Tables: Group Summary
    # Check if group_summaries is present in analysis.values
    group_summaries = analysis.values.get("group_summaries")
    table_rows: list[DisplayRow] = []
    if isinstance(group_summaries, list) and group_summaries:
        for item in group_summaries:
            if isinstance(item, dict):
                g_name = str(item.get("group") or item.get("condition") or "")
                g_n = format_sample_size(item.get("sample_size") or item.get("n"))
                g_mean = format_number(item.get("mean"), decimals=2, unit=unit)
                g_sd = format_number(item.get("standard_deviation"), decimals=2, unit=unit)
                table_rows.append(DisplayRow((g_name, g_n, g_mean, g_sd)))
    else:
        # Fall back to metadata group sizes
        group_sizes = analysis.metadata.get("sample", {}).get("group_sizes")
        if isinstance(group_sizes, list):
            for g in group_sizes:
                if isinstance(g, dict):
                    table_rows.append(
                        DisplayRow(
                            (
                                str(g.get("group", "")),
                                format_sample_size(g.get("size")),
                                "Unavailable",
                                "Unavailable",
                            )
                        )
                    )
        elif isinstance(group_sizes, dict):
            for k, v in group_sizes.items():
                table_rows.append(
                    DisplayRow((str(k), format_sample_size(v), "Unavailable", "Unavailable"))
                )

    tables = (
        DisplayTable(
            title="GROUP SUMMARY",
            columns=("Group", "N", "Mean", "SD"),
            rows=tuple(table_rows),
        ),
    )

    # Diagnostics
    diagnostics_list: list[DisplayDiagnostic] = [
        DisplayDiagnostic(
            label="Equal variance",
            status="Not assumed",
            detail="Welch test does not assume equal population variances.",
            severity="neutral",
        ),
        DisplayDiagnostic(
            label="Missingness",
            status="Complete cases" if not excluded_rows else "Rows excluded",
            detail=f"{format_sample_size(excluded_rows)} missing or excluded rows.",
            severity="neutral" if not excluded_rows else "review",
        ),
    ]

    # Additional assumption notes from interpretation
    if interp and interp.assumption_notes:
        for note in interp.assumption_notes:
            sev = "warning" if "[CAUTION]" in note or "[WARNING]" in note else "neutral"
            clean_note = (
                note.replace("[INFO]", "").replace("[CAUTION]", "").replace("[WARNING]", "").strip()
            )
            diagnostics_list.append(
                DisplayDiagnostic(
                    label="Assumption",
                    status="Review" if sev == "warning" else "Documented",
                    detail=clean_note,
                    severity=sev,
                )
            )

    # Interpretation
    interpretation_text: str | None = None
    if interp:
        parts: list[str] = []
        if interp.hypothesis_interpretation:
            parts.append(interp.hypothesis_interpretation.strip())
        if detail == "full":
            if interp.effect_interpretation:
                parts.append(interp.effect_interpretation.strip())
            if interp.uncertainty_interpretation:
                parts.append(interp.uncertainty_interpretation.strip())
        interpretation_text = "\n\n".join(parts) if parts else interp.summary

    # Limitations & warnings
    limitations = tuple(interp.limitations) if interp else ()
    warnings = tuple(workflow.warnings)

    # Compact text
    ci_compact = format_confidence_interval(ci_dict, decimals=2, ellipsis_sep=True)
    compact_text = (
        f"Welch t-test | N={format_sample_size(sample_size)} | "
        f"diff={diff_str} | 95% CI {ci_compact} | d={d_str} | p={p_str}"
    )

    return TerminalView(
        title="Independent Group Comparison",
        subtitle=f"{outcome} by {predictor}",
        family="family.mean",
        design_metrics=design_metrics,
        key_metrics=key_metrics,
        tables=tables,
        diagnostics=tuple(diagnostics_list),
        interpretation=interpretation_text,
        limitations=limitations,
        warnings=warnings,
        metadata={
            "degrees_of_freedom": analysis.values.get("degrees_of_freedom"),
            "test_statistic": analysis.values.get("test_statistic"),
            "method_id": "welch_t",
            "method_name": analysis.method_label,
            "rationale": workflow.recommendation.rationale if workflow.recommendation else None,
            "audit_status": workflow.audit.status if workflow.audit else None,
            "reproducibility": workflow.reproducibility.to_dict()
            if workflow.reproducibility
            else None,
        },
        compact_text=compact_text,
    )


def adapt_pearson(workflow: ResearchWorkflowResult, detail: str = "standard") -> TerminalView:
    """Adapt a Pearson correlation workflow."""
    analysis = workflow.analysis
    assert analysis is not None
    interp = workflow.interpretation
    spec = workflow.specification

    outcome = spec.question.outcome or "Variable 1"
    predictor = spec.question.predictor or "Variable 2"

    sample_size = analysis.sample_size
    excluded_rows = analysis.excluded_rows or 0

    design_metrics = (
        DisplayMetric("Method", "Pearson correlation", role="method"),
        DisplayMetric("Estimand", "Linear association"),
        DisplayMetric("N", f"{format_sample_size(sample_size)} paired observations"),
        DisplayMetric("Missing paired rows", f"{format_sample_size(excluded_rows)} rows"),
    )

    r_val = analysis.values.get("primary_estimate")
    r_str = format_number(r_val, decimals=3)

    ci_dict = analysis.values.get("confidence_interval")
    ci_str = format_confidence_interval(ci_dict, decimals=3, bracket=True)

    p_val = analysis.values.get("p_value")
    p_str = format_p_value(p_val)

    key_metrics = (
        DisplayMetric("Correlation", f"r = {r_str}", role="result.estimate"),
        DisplayMetric("95% CI", ci_str, role="result.ci"),
        DisplayMetric("p-value", p_str, role="result.evidence"),
    )

    diagnostics_list: list[DisplayDiagnostic] = [
        DisplayDiagnostic(
            label="Pairwise complete N",
            status=format_sample_size(sample_size),
            detail="Valid pairs with non-missing values for both variables.",
            severity="neutral",
        ),
        DisplayDiagnostic(
            label="Linear target",
            status="Specified",
            detail="Measures linear association; sensitive to extreme values and curvature.",
            severity="neutral",
        ),
    ]

    if interp and interp.assumption_notes:
        for note in interp.assumption_notes:
            sev = "warning" if "[CAUTION]" in note or "[WARNING]" in note else "neutral"
            clean_note = (
                note.replace("[INFO]", "").replace("[CAUTION]", "").replace("[WARNING]", "").strip()
            )
            diagnostics_list.append(
                DisplayDiagnostic(
                    label="Assumption",
                    status="Review" if sev == "warning" else "Documented",
                    detail=clean_note,
                    severity=sev,
                )
            )

    interpretation_text: str | None = None
    if interp:
        parts: list[str] = []
        if interp.hypothesis_interpretation:
            parts.append(interp.hypothesis_interpretation.strip())
        if detail == "full":
            if interp.effect_interpretation:
                parts.append(interp.effect_interpretation.strip())
            if interp.uncertainty_interpretation:
                parts.append(interp.uncertainty_interpretation.strip())
        interpretation_text = "\n\n".join(parts) if parts else interp.summary

    limitations = (
        tuple(interp.limitations)
        if interp and interp.limitations
        else ("Observed association alone does not establish causation.",)
    )
    warnings = tuple(workflow.warnings)

    compact_text = (
        f"Pearson r | N={format_sample_size(sample_size)} | r={r_str} | 95% CI {ci_str} | p={p_str}"
    )

    return TerminalView(
        title="Association Analysis",
        subtitle=f"{outcome} <-> {predictor}",
        family="family.association",
        design_metrics=design_metrics,
        key_metrics=key_metrics,
        tables=(),
        diagnostics=tuple(diagnostics_list),
        interpretation=interpretation_text,
        limitations=limitations,
        warnings=warnings,
        metadata={
            "method_id": "pearson_correlation",
            "method_name": analysis.method_label,
            "rationale": workflow.recommendation.rationale if workflow.recommendation else None,
            "audit_status": workflow.audit.status if workflow.audit else None,
            "reproducibility": workflow.reproducibility.to_dict()
            if workflow.reproducibility
            else None,
        },
        compact_text=compact_text,
    )


def adapt_profile(profile: dict[str, Any], detail: str = "standard") -> TerminalView:
    """Adapt a dataset profile dictionary into a TerminalView."""
    overview = profile.get("overview", {})
    total_rows = overview.get("total_rows", 0)
    total_cols = overview.get("total_columns", 0)

    missing = profile.get("missing_data", {})
    missing_cells = missing.get("total_missing_cells", overview.get("missing_cells", 0))
    missing_pct = missing.get(
        "overall_missing_percentage", overview.get("missing_cell_percentage", 0.0)
    )

    quality = profile.get("data_quality", {})
    duplicate_rows = quality.get("duplicate_rows", overview.get("duplicate_rows", 0))
    mem_bytes = overview.get(
        "memory_usage_bytes", profile.get("resource_info", {}).get("estimated_memory_bytes")
    )

    design_metrics = (
        DisplayMetric("Rows", format_sample_size(total_rows)),
        DisplayMetric("Columns", format_sample_size(total_cols)),
        DisplayMetric("Missing cells", f"{format_sample_size(missing_cells)} ({missing_pct:.1f}%)"),
        DisplayMetric("Duplicate rows", format_sample_size(duplicate_rows)),
        DisplayMetric("Memory", format_memory_bytes(mem_bytes)),
    )

    # Tables
    tables: list[DisplayTable] = []

    # 1. Numeric Variables
    descriptive = profile.get("descriptive", {})
    missing_by_col = missing.get("by_column", {})
    num_rows: list[DisplayRow] = []

    num_items = list(descriptive.items())
    max_num = len(num_items) if detail == "full" else min(10, len(num_items))
    for col, stats in num_items[:max_num]:
        if isinstance(stats, dict):
            n = format_sample_size(stats.get("count"))
            mean = format_number(stats.get("mean"))
            median = format_number(stats.get("median"))
            sd = format_number(stats.get("std"))
            miss_count = format_sample_size(missing_by_col.get(col, {}).get("count", 0))
            num_rows.append(DisplayRow((str(col), n, mean, median, sd, miss_count)))

    if num_rows:
        num_title = "NUMERIC VARIABLES"
        if detail != "full" and len(num_items) > max_num:
            num_title += f" (showing {max_num} of {len(num_items)})"
        tables.append(
            DisplayTable(
                title=num_title,
                columns=("Variable", "N", "Mean", "Median", "SD", "Missing"),
                rows=tuple(num_rows),
            )
        )

    # 2. Categorical Variables
    categorical = profile.get("categorical_summary", {})
    cat_rows: list[DisplayRow] = []
    cat_items = list(categorical.items())
    max_cat = len(cat_items) if detail == "full" else min(10, len(cat_items))

    for col, info in cat_items[:max_cat]:
        if isinstance(info, dict):
            levels = format_sample_size(info.get("observed_categories"))
            modes = info.get("mode_values", [])
            most_common = str(modes[0]) if modes else "None"
            # Truncate long mode strings if necessary
            if len(most_common) > 25:
                most_common = most_common[:22] + "..."
            miss_count = format_sample_size(info.get("missing_count", 0))
            cat_rows.append(DisplayRow((str(col), levels, most_common, miss_count)))

    if cat_rows:
        cat_title = "CATEGORICAL VARIABLES"
        if detail != "full" and len(cat_items) > max_cat:
            cat_title += f" (showing {max_cat} of {len(cat_items)})"
        tables.append(
            DisplayTable(
                title=cat_title,
                columns=("Variable", "Levels", "Most common", "Missing"),
                rows=tuple(cat_rows),
            )
        )

    # Review Cues / Issues
    issues = quality.get("issues", [])
    diagnostics_list: list[DisplayDiagnostic] = []
    max_issues = len(issues) if detail == "full" else min(8, len(issues))
    for issue in issues[:max_issues]:
        if isinstance(issue, dict):
            sev = issue.get("severity", "review")
            col = issue.get("column") or issue.get("section") or "Data Quality"
            msg = issue.get("message", "")
            diagnostics_list.append(
                DisplayDiagnostic(
                    label=str(col),
                    status="Review" if sev == "review" else "Info",
                    detail=msg,
                    severity=sev,
                )
            )

    compact_text = (
        f"Dataset Profile | {total_rows:,} rows x {total_cols:,} vars | "
        f"missing={missing_cells:,} ({missing_pct:.1f}%) | duplicates={duplicate_rows:,}"
    )

    return TerminalView(
        title="Dataset Profile",
        subtitle=f"{total_rows:,} rows x {total_cols:,} variables",
        family="family.profile",
        design_metrics=design_metrics,
        key_metrics=(),
        tables=tuple(tables),
        diagnostics=tuple(diagnostics_list),
        interpretation=None,
        limitations=(),
        warnings=(),
        metadata={"total_rows": total_rows, "total_columns": total_cols},
        compact_text=compact_text,
    )


def adapt_workflow_status(
    workflow: ResearchWorkflowResult, detail: str = "standard"
) -> TerminalView:
    """Adapt a non-completed or blocked workflow status."""
    status = workflow.status
    spec = workflow.specification
    q = spec.question

    known_metrics: list[DisplayMetric] = []
    if q.outcome:
        known_metrics.append(DisplayMetric("Outcome", str(q.outcome)))
    if q.predictor:
        known_metrics.append(DisplayMetric("Predictor", str(q.predictor)))
    if q.estimand:
        est_val = getattr(q.estimand, "value", q.estimand)
        known_metrics.append(DisplayMetric("Estimand", str(est_val)))
    if spec.design:
        des_val = getattr(spec.design, "value", spec.design)
        known_metrics.append(DisplayMetric("Design", str(des_val)))

    diagnostics_list: list[DisplayDiagnostic] = []
    warnings_list = list(workflow.warnings)

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
        design_metrics=tuple(known_metrics),
        key_metrics=(),
        tables=(),
        diagnostics=tuple(diagnostics_list),
        interpretation=None,
        limitations=(),
        warnings=tuple(warnings_list),
        metadata={"workflow_status": status.value},
        compact_text=compact_text,
    )
