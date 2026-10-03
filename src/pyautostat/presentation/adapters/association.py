"""Adapters for bivariate and partial association analyses."""

from __future__ import annotations

from ...results import AnalysisResult
from ...workflow import ResearchWorkflowResult
from ..formatting import (
    format_confidence_interval,
    format_confidence_level_label,
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
    first_present,
    format_interpretation_text,
    resolve_confidence_level,
)


def adapt_pearson(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a Pearson correlation workflow."""
    return _adapt_association(target, detail=detail, method_id="pearson_correlation")


def adapt_spearman(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a Spearman rank correlation workflow."""
    return _adapt_association(target, detail=detail, method_id="spearman_correlation")


def adapt_kendall_tau_b(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a Kendall's tau-b rank concordance workflow."""
    return _adapt_association(target, detail=detail, method_id="kendall_tau_b")


def adapt_point_biserial(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a point-biserial correlation workflow."""
    return _adapt_association(target, detail=detail, method_id="point_biserial_correlation")


def adapt_partial_pearson(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a partial Pearson correlation workflow."""
    return _adapt_association(target, detail=detail, method_id="partial_pearson_correlation")


def _adapt_association(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
    method_id: str = "pearson_correlation",
) -> TerminalView:
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    outcome = (spec.question.outcome if spec and spec.question else None) or "Variable A"
    predictor = (spec.question.predictor if spec and spec.question else None) or "Variable B"
    conf_level = resolve_confidence_level(analysis, spec)

    sample_size = analysis.sample_size
    excluded_rows = first_present(analysis, "excluded_rows", default=0)

    design_metrics_list = []
    if method_id == "pearson_correlation":
        method_name = "Pearson correlation"
        estimand = "Linear association"
        est_label = "Correlation"
        symbol = "r"
        family = "family.association"
    elif method_id == "spearman_correlation":
        method_name = "Spearman rank correlation"
        estimand = "Monotonic association"
        est_label = "Spearman rho"
        symbol = "rho"
        family = "family.rank"
    elif method_id == "kendall_tau_b":
        method_name = "Kendall's tau-b"
        estimand = "Rank concordance"
        est_label = "Kendall's tau-b"
        symbol = "tau_b"
        family = "family.rank"
    elif method_id == "point_biserial_correlation":
        method_name = "Point-biserial correlation"
        estimand = "Dichotomous-continuous association"
        est_label = "Point-biserial r"
        symbol = "r_pb"
        family = "family.association"
    else:  # partial_pearson_correlation
        method_name = "Partial Pearson correlation"
        estimand = "Linear association adjusted for covariates"
        est_label = "Partial r"
        symbol = "r_partial"
        family = "family.association"

    design_metrics_list.append(DisplayMetric("Method", method_name, role="method"))
    design_metrics_list.append(DisplayMetric("Estimand", estimand))
    design_metrics_list.append(
        DisplayMetric("N", f"{format_sample_size(sample_size)} paired observations")
    )
    design_metrics_list.append(
        DisplayMetric("Missing paired rows", f"{format_sample_size(excluded_rows)} rows")
    )

    # Extra design metrics for point-biserial and partial Pearson
    tables: list[DisplayTable] = []
    if method_id == "point_biserial_correlation":
        pos_level = analysis.metadata.get("positive_level")
        bin_var = analysis.metadata.get("binary_variable")
        design_metrics_list.append(DisplayMetric("Binary Target", str(bin_var)))
        design_metrics_list.append(DisplayMetric("Positive Level", f"'{pos_level}' (+1 coding)"))

        # Show binary groups summary if available
        group_sizes = analysis.values.get("group_sizes")
        group_means = analysis.values.get("group_means")
        if isinstance(group_sizes, dict) and isinstance(group_means, dict):
            rows = []
            for lvl, count in group_sizes.items():
                mean_val = format_number(group_means.get(lvl), decimals=2)
                rows.append(DisplayRow((str(lvl), format_sample_size(count), mean_val)))
            tables.append(
                DisplayTable(
                    title="BINARY LEVEL BREAKDOWN", columns=("Level", "N", "Mean"), rows=tuple(rows)
                )
            )

    elif method_id == "partial_pearson_correlation":
        controls = analysis.metadata.get("controls", [])
        controls_str = ", ".join(str(c) for c in controls)
        design_metrics_list.append(DisplayMetric("Controls", controls_str or "None"))

    # Key results
    r_val = analysis.values.get("primary_estimate")
    r_str = format_number(r_val, decimals=3)
    p_val = analysis.values.get("p_value")
    p_str = format_p_value(p_val)
    ci_dict = analysis.values.get("confidence_interval")
    ci_str = format_confidence_interval(ci_dict, decimals=3, bracket=True)

    ci_label = format_confidence_level_label(ci_dict, confidence_level=conf_level)

    key_metrics_list = [
        DisplayMetric(est_label, f"{symbol} = {r_str}", role="result.estimate"),
        DisplayMetric(ci_label, ci_str, role="result.ci"),
        DisplayMetric("p-value", p_str, role="result.evidence"),
    ]

    stat = analysis.values.get("test_statistic")
    df = analysis.values.get("degrees_of_freedom")
    if detail == "full" and stat is not None and df is not None:
        key_metrics_list.append(DisplayMetric("Test statistic", format_statistic("t", stat, df=df)))

    # Diagnostics
    diagnostics_list = [
        DisplayDiagnostic(
            label="Pairwise complete N",
            status=format_sample_size(sample_size),
            detail="Valid pairs with non-missing values for both variables.",
            severity="neutral",
        ),
    ]
    if method_id == "pearson_correlation":
        diagnostics_list.append(
            DisplayDiagnostic(
                label="Linear target",
                status="SPECIFIED",
                detail="Measures linear association; sensitive to extreme values and curvature.",
                severity="neutral",
            )
        )
    elif method_id in ("spearman_correlation", "kendall_tau_b"):
        ties_info = analysis.metadata.get("ties")
        detail_msg = (
            f"Ties accounted for: {ties_info}" if ties_info else "Rank-based monotonic target."
        )
        diagnostics_list.append(
            DisplayDiagnostic(
                label="Monotonic target",
                status="RANK MODEL",
                detail=detail_msg,
                severity="neutral",
            )
        )
    elif method_id == "partial_pearson_correlation":
        n_ctrls = len(analysis.metadata.get("controls", []))
        diagnostics_list.append(
            DisplayDiagnostic(
                label="OLS Conditioning",
                status="RESIDUALISED",
                detail=f"Adjusted for {n_ctrls} control variables via OLS residualisation.",
                severity="neutral",
            )
        )

    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("establish causation" in lim for lim in limitations):
        limitations.append("Observed association alone does not establish causation.")
    if not any("practical importance" in lim for lim in limitations):
        limitations.append(
            "Statistical significance alone does not establish practical importance."
        )
    if method_id == "kendall_tau_b":
        limitations.append(
            "Kendall's tau-b is a rank concordance metric; "
            "it does NOT represent proportion of variance explained."
        )
    elif method_id == "partial_pearson_correlation":
        limitations.append(
            "Conditioning on covariates via OLS residualisation is NOT causal deconfounding."
        )

    comp_label = "Pearson r" if method_id == "pearson_correlation" else method_name
    compact_text = (
        f"{comp_label} | N={format_sample_size(sample_size)} | "
        f"{symbol}={r_str} | {ci_label} {ci_str} | p={p_str}"
    )

    return TerminalView(
        title="Association Analysis",
        subtitle=f"{outcome} <-> {predictor}",
        family=family,
        design_metrics=tuple(design_metrics_list),
        key_metrics=tuple(key_metrics_list),
        tables=tuple(tables),
        diagnostics=tuple(diagnostics_list),
        interpretation=interpretation_text,
        limitations=tuple(limitations),
        warnings=analysis.warnings,
        metadata=build_metadata_dict(analysis, workflow),
        compact_text=compact_text,
    )
