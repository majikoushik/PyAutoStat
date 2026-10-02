"""Adapters for two-way factorial ANOVA."""

from __future__ import annotations

from ...results import AnalysisResult
from ...workflow import ResearchWorkflowResult
from ..formatting import (
    format_confidence_interval,
    format_df,
    format_effect,
    format_number,
    format_p_value,
    format_sample_size,
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


def adapt_two_way_anova(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt an independent two-way factorial ANOVA workflow."""
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    outcome = analysis.values.get("outcome") or "Outcome"
    factor_a = analysis.values.get("factor_a") or "Factor A"
    factor_b = analysis.values.get("factor_b") or "Factor B"
    levels_a = analysis.values.get("factor_a_levels") or []
    levels_b = analysis.values.get("factor_b_levels") or []
    ss_type = analysis.values.get("sum_of_squares_type") or "Type III"

    sample_size = analysis.sample_size
    excluded_rows = analysis.excluded_rows or 0

    design_metrics = (
        DisplayMetric("Method", "Two-way factorial ANOVA", role="method"),
        DisplayMetric("Estimand", "Main factor effects and interaction effect"),
        DisplayMetric(
            "Factors", f"{factor_a} ({len(levels_a)} levels) x {factor_b} ({len(levels_b)} levels)"
        ),
        DisplayMetric("SS Type", str(ss_type)),
        DisplayMetric("Sample", f"{format_sample_size(sample_size)} observations"),
        DisplayMetric("Excluded", f"{format_sample_size(excluded_rows)} rows"),
    )

    terms = analysis.values.get("terms") or []
    interaction_term = next((t for t in terms if t.get("term_type") == "interaction"), None)
    if interaction_term:
        f_int = interaction_term.get("statistic")
        p_int = interaction_term.get("p_value")
        eff_dict = interaction_term.get("effect_size") or {}
        eta_int = eff_dict.get("value")
        ci_int = eff_dict.get("confidence_interval")
        f_str = format_number(f_int, decimals=3)
        p_str = format_p_value(p_int)
        eta_str = format_effect(eta_int, decimals=3)
        ci_str = format_confidence_interval(ci_int, decimals=3)
    else:
        f_str = "Unavailable"
        p_str = "Unavailable"
        eta_str = "Unavailable"
        ci_str = "Unavailable"

    key_metrics_list = [
        DisplayMetric(f"{factor_a} x {factor_b} F", f_str, role="result.estimate"),
        DisplayMetric("Interaction p-value", p_str, role="result.evidence"),
        DisplayMetric("Interaction partial eta2", eta_str, role="result.effect"),
        DisplayMetric("Interaction 95% CI", ci_str, role="result.ci"),
    ]

    # Tables: ANOVA effects table & Cell summaries
    tables: list[DisplayTable] = []
    if terms:
        cols = ("Effect / Term", "Sum of Sq", "df", "Mean Sq", "F", "p-value", "partial eta2")
        rows = []
        for t in terms:
            t_name = str(t.get("term", ""))
            if t.get("term_type") == "interaction":
                t_name = f"{t_name} [Interaction]"
            ss = format_number(t.get("sum_of_squares"), decimals=2)
            df = format_df(t.get("df"))
            ms = format_number(t.get("mean_square"), decimals=2)
            f_val = format_number(t.get("f_statistic"), decimals=2)
            p_val = format_p_value(t.get("p_value"))
            term_eff = t.get("effect_size") or {}
            eta = format_effect(term_eff.get("value"), decimals=3)
            rows.append(DisplayRow((t_name, ss, df, ms, f_val, p_val, eta)))
        tables.append(DisplayTable(title="ANOVA EFFECTS TABLE", columns=cols, rows=tuple(rows)))

    cell_sums = analysis.values.get("cell_summaries")
    if isinstance(cell_sums, list) and cell_sums:
        c_cols = (str(factor_a), str(factor_b), "N", "Mean", "SD")
        c_rows = [
            DisplayRow(
                (
                    str(c.get("level_a", c.get(factor_a, ""))),
                    str(c.get("level_b", c.get(factor_b, ""))),
                    format_sample_size(c.get("n", c.get("size"))),
                    format_number(c.get("mean"), decimals=2),
                    format_number(c.get("sd", c.get("standard_deviation")), decimals=2),
                )
            )
            for c in cell_sums
        ]
        tables.append(DisplayTable(title="CELL SUMMARIES", columns=c_cols, rows=tuple(c_rows)))

    diagnostics_list = [
        DisplayDiagnostic(
            label="Interaction Assessment",
            status="DOCUMENTED",
            detail=(
                f"Interaction p = {p_str}. "
                "Inspect simple effects if interaction is statistically detectable."
            ),
            severity="neutral",
        )
    ]
    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("main factor effects" in lim for lim in limitations):
        limitations.append(
            "In the presence of an interaction, main factor effects must be "
            "interpreted with caution across levels."
        )
    if not any("statistically detectable interaction" in lim for lim in limitations):
        limitations.append(
            "Absence of a statistically detectable interaction is not proof "
            "that the factors are purely additive."
        )

    compact_text = (
        f"Two-way ANOVA | {factor_a} x {factor_b} | "
        f"Interaction F={f_str} | eta_p2={eta_str} | p={p_str}"
    )

    metadata = build_metadata_dict(analysis, workflow)

    return TerminalView(
        title="Two-Way Factorial ANOVA",
        subtitle=f"{outcome} by {factor_a} x {factor_b}",
        family="family.factorial",
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
