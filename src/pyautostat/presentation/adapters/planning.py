"""Adapters for study planning, sensitivity analysis, and practical significance."""

from __future__ import annotations

from ...practical_significance import PracticalSignificanceResult
from ...sensitivity import SensitivityResult
from ...study_planning import StudyPlanningResult
from ..formatting import (
    format_confidence_interval,
    format_number,
    format_percent,
    format_sample_size,
)
from ..models import (
    DisplayDiagnostic,
    DisplayMetric,
    DisplayRow,
    DisplayTable,
    TerminalView,
)


def adapt_study_planning(
    result: StudyPlanningResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a StudyPlanningResult into a TerminalView."""
    conf_pct = (
        format_percent(result.confidence_level * 100)
        if result.confidence_level is not None
        else "Unavailable"
    )
    design_metrics = (
        DisplayMetric(
            "Planning Type", str(result.planning_type).replace("_", " ").title(), role="method"
        ),
        DisplayMetric("Method Family", str(result.method_family).replace("_", " ").title()),
        DisplayMetric("Target Quantity", str(result.target_quantity)),
        DisplayMetric("Alpha", format_number(result.alpha, decimals=3)),
        DisplayMetric("Confidence Level", conf_pct),
    )

    key_metrics_list = []
    if result.target_power is not None:
        key_metrics_list.append(
            DisplayMetric(
                "Target Power",
                format_percent(
                    result.target_power * 100 if result.target_power is not None else None
                ),
                role="result.ci",
            )
        )
    if result.achieved_power is not None:
        key_metrics_list.append(
            DisplayMetric(
                "Achieved Power",
                format_percent(
                    result.achieved_power * 100 if result.achieved_power is not None else None
                ),
                role="result.ci",
            )
        )
    if result.target_half_width is not None:
        key_metrics_list.append(
            DisplayMetric("Target Half-Width", format_number(result.target_half_width, decimals=3))
        )

    if result.total_required_n is not None:
        key_metrics_list.append(
            DisplayMetric(
                "Required Total N",
                format_sample_size(result.total_required_n),
                role="result.estimate",
            )
        )
        n1_str = format_sample_size(result.required_n1)
        n2_str = format_sample_size(result.required_n2)
        key_metrics_list.append(
            DisplayMetric(
                "Required Group Ns",
                f"n1={n1_str}, n2={n2_str}",
            )
        )
    if result.required_pairs is not None:
        key_metrics_list.append(
            DisplayMetric(
                "Required Complete Pairs",
                format_sample_size(result.required_pairs),
                role="result.estimate",
            )
        )

    tables: list[DisplayTable] = []
    if result.assumptions:
        cols = ("Parameter / Assumption", "Supplied Value")
        rows = [
            DisplayRow((str(k).replace("_", " ").title(), str(v)))
            for k, v in result.assumptions.items()
        ]
        tables.append(
            DisplayTable(title="RESEARCHER-SUPPLIED ASSUMPTIONS", columns=cols, rows=tuple(rows))
        )

    diagnostics_list = [
        DisplayDiagnostic(
            label="Prospective Intent",
            status="PRE-COLLECTION",
            detail=result.calculation_method
            or "Analytical sample-size formula under stated assumptions.",
            severity="neutral",
        )
    ]

    limitations = [
        "Prospective study planning provides theoretical sample-size requirements; "
        "it is NOT observed post-hoc power.",
        "Calculated sample size is only valid if researcher-supplied design assumptions hold.",
    ]

    compact_text = (
        f"Study Planning | {result.method_family} | "
        f"Total N={format_sample_size(result.total_required_n)} | "
        f"Alpha={format_number(result.alpha, decimals=3)}"
    )

    return TerminalView(
        title="Prospective Study Planning",
        subtitle=f"{result.method_family} ({result.planning_type})",
        family="family.planning",
        design_metrics=design_metrics,
        key_metrics=tuple(key_metrics_list),
        tables=tuple(tables),
        diagnostics=tuple(diagnostics_list),
        interpretation="\n\n".join(result.approximation_notes)
        if result.approximation_notes
        else None,
        limitations=tuple(limitations),
        warnings=result.warnings,
        metadata={"planning_result": result},
        compact_text=compact_text,
    )


def adapt_sensitivity(
    result: SensitivityResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a SensitivityResult into a TerminalView."""
    base = result.base_result
    primary_method = base.method_id if hasattr(base, "method_id") else "Primary Method"

    design_metrics = (
        DisplayMetric("Primary Method", str(primary_method), role="method"),
        DisplayMetric("Sensitivity Status", str(result.status).upper()),
        DisplayMetric("Scenarios Run", f"{len(result.scenario_results)} alternative scenarios"),
    )

    tables: list[DisplayTable] = []
    cols = ("Scenario", "Method", "Estimand Match", "Contrast Match", "Status", "Conclusion")
    rows: list[DisplayRow] = []
    for sc in result.scenario_results:
        sc_name = getattr(sc, "name", "")
        m_name = getattr(sc, "method_id", getattr(sc, "requested_method_id", "Same")) or "Same"
        comp_val = getattr(sc, "comparability", None)
        comp_str = (
            str(comp_val.value)
            if (comp_val is not None and hasattr(comp_val, "value"))
            else str(comp_val or "")
        )
        est_match = "Yes" if "same_estimand" in comp_str else ("No" if comp_str else "Unknown")
        con_match = "Preserved"
        st = str(getattr(sc, "status", "")).upper()
        if hasattr(getattr(sc, "status", None), "value"):
            st = str(sc.status.value).upper()
        comp_dict = getattr(sc, "comparison", {}) or {}
        conc = str(comp_dict.get("decision", comp_dict.get("conclusion", "Evaluated")))
        rows.append(DisplayRow((sc_name, m_name, est_match, con_match, st, conc)))
    tables.append(
        DisplayTable(title="SENSITIVITY SCENARIO COMPARISON", columns=cols, rows=tuple(rows))
    )

    diagnostics_list = [
        DisplayDiagnostic(
            label="Comparability Policy",
            status="EVALUATED",
            detail=(
                "Scenarios retain declared contrasts and identify estimand differences explicitly."
            ),
            severity="neutral",
        )
    ]

    limitations = [
        "Sensitivity analysis explores robustness under alternative specifications; "
        "do not select a scenario by favorable p-value.",
        "A nonsignificant superiority test across scenarios is not an equivalence test.",
    ]

    summary_text = result.comparison_summary if isinstance(result.comparison_summary, str) else None
    compact_text = (
        f"Sensitivity Analysis | Primary={primary_method} | "
        f"Scenarios={len(result.scenario_results)} | Status={result.status}"
    )

    return TerminalView(
        title="Sensitivity Analysis",
        subtitle=f"Robustness evaluation for {primary_method}",
        family="family.planning",
        design_metrics=design_metrics,
        key_metrics=(),
        tables=tuple(tables),
        diagnostics=tuple(diagnostics_list),
        interpretation=summary_text,
        limitations=tuple(limitations),
        warnings=result.warnings,
        metadata={"sensitivity_result": result},
        compact_text=compact_text,
    )


def adapt_practical_significance(
    result: PracticalSignificanceResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a PracticalSignificanceResult into a TerminalView."""
    est_str = format_number(result.estimate, decimals=3)
    ci_str = format_confidence_interval(result.confidence_interval, decimals=3)
    if hasattr(result.threshold, "minimum_magnitude"):
        unit_str = f" {result.threshold.unit}" if getattr(result.threshold, "unit", None) else ""
        thresh_str = f"{format_number(result.threshold.minimum_magnitude, decimals=3)}{unit_str}"
    else:
        thresh_val = getattr(result.threshold, "threshold_value", str(result.threshold))
        thresh_str = format_number(thresh_val, decimals=3)

    design_metrics = (
        DisplayMetric("Evaluated Quantity", str(result.quantity)),
        DisplayMetric("Observed Estimate", est_str, role="result.estimate"),
        DisplayMetric("95% CI", ci_str, role="result.ci"),
        DisplayMetric("Researcher Threshold", thresh_str, role="method"),
    )

    stat_sig = result.statistical_significance
    stat_sig_str = "Statistically significant" if stat_sig else "Not statistically significant"

    key_metrics_list = [
        DisplayMetric("Practical Verdict", str(result.status).upper(), role="result.estimate"),
        DisplayMetric("Point vs Threshold", str(result.point_estimate_relation)),
        DisplayMetric(
            "CI vs Threshold", str(result.confidence_interval_relation), role="result.ci"
        ),
        DisplayMetric("Statistical Evidence", stat_sig_str, role="result.evidence"),
    ]

    diagnostics_list = [
        DisplayDiagnostic(
            label="Threshold Source",
            status="RESEARCHER-SUPPLIED",
            detail="Threshold declared a priori; not derived from observed distribution.",
            severity="neutral",
        ),
        DisplayDiagnostic(
            label="Inference Distinction",
            status="SEPARATED",
            detail=(
                "Statistical significance and practical meaningfulness are evaluated separately."
            ),
            severity="neutral",
        ),
    ]

    limitations = [
        "Statistical significance is not effect magnitude or practical importance.",
        "Practical threshold relationships reflect researcher-declared bounds; "
        "they do not establish causal impact.",
    ]

    compact_text = (
        f"Practical Significance | {result.quantity}={est_str} | "
        f"Threshold={thresh_str} | Verdict={result.status}"
    )

    return TerminalView(
        title="Practical Significance Assessment",
        subtitle=f"{result.quantity} vs Threshold ({thresh_str})",
        family="family.planning",
        design_metrics=design_metrics,
        key_metrics=tuple(key_metrics_list),
        tables=(),
        diagnostics=tuple(diagnostics_list),
        interpretation=result.conclusion,
        limitations=tuple(limitations),
        warnings=result.warnings,
        metadata={"practical_significance_result": result},
        compact_text=compact_text,
    )
