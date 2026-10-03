"""Adapters for categorical association and paired binary analyses."""

from __future__ import annotations

from ...results import AnalysisResult
from ...workflow import ResearchWorkflowResult
from ..formatting import (
    format_confidence_interval,
    format_effect,
    format_number,
    format_odds_ratio,
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
)


def adapt_pearson_chi_square(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a Pearson chi-square independence test workflow."""
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    outcome = (spec.question.outcome if spec and spec.question else None) or "Outcome"
    predictor = (spec.question.predictor if spec and spec.question else None) or "Predictor"

    sample_size = analysis.sample_size
    excluded_rows = first_present(analysis, "excluded_rows", default=0)

    design_metrics = (
        DisplayMetric("Method", "Pearson chi-square test of independence", role="method"),
        DisplayMetric("Estimand", "Categorical independence / association"),
        DisplayMetric("Variables", f"{outcome} x {predictor}"),
        DisplayMetric("Sample", f"{format_sample_size(sample_size)} observations"),
        DisplayMetric("Excluded", f"{format_sample_size(excluded_rows)} rows"),
    )

    stat = analysis.values.get("test_statistic")
    df = analysis.values.get("degrees_of_freedom")
    p_val = analysis.values.get("p_value")
    p_str = format_p_value(p_val)
    stat_str = format_statistic("Chi-square", stat, df=df)

    v_val = analysis.values.get("primary_estimate")
    v_str = format_effect(v_val, decimals=3)
    ci_dict = analysis.values.get("confidence_interval")
    ci_str = format_confidence_interval(ci_dict, decimals=3)

    key_metrics_list = [
        DisplayMetric("Test statistic", stat_str, role="result.evidence"),
        DisplayMetric("p-value", p_str, role="result.evidence"),
        DisplayMetric("Cramer's V", v_str, role="result.effect"),
        DisplayMetric("Effect 95% CI", ci_str, role="result.ci"),
    ]

    # Contingency Table
    tables: list[DisplayTable] = []
    observed = analysis.metadata.get("observed_counts")
    groups = analysis.metadata.get("group_order", [])
    outcomes = analysis.metadata.get("outcome_order", [])
    if isinstance(observed, list) and groups and outcomes:
        cols = [f"{predictor} \\ {outcome}"] + [str(o) for o in outcomes] + ["Total"]
        t_rows: list[DisplayRow] = []
        for g_idx, g_name in enumerate(groups):
            row_counts = observed[g_idx]
            total_r = sum(row_counts)
            row_cells = (
                [str(g_name)]
                + [format_sample_size(c) for c in row_counts]
                + [format_sample_size(total_r)]
            )
            t_rows.append(DisplayRow(tuple(row_cells)))
        tables.append(
            DisplayTable(
                title="CONTINGENCY TABLE (OBSERVED COUNTS)", columns=tuple(cols), rows=tuple(t_rows)
            )
        )

    # Expected count diagnostics
    diagnostics_list: list[DisplayDiagnostic] = []
    diag_data = analysis.metadata.get("diagnostics", {})
    if isinstance(diag_data, dict):
        min_exp = first_present(diag_data, "minimum_expected_count", "min_expected_frequency")
        if min_exp is not None:
            min_exp_str = format_number(min_exp, decimals=2)
            stored_status = first_present(diag_data, "expected_count_status", "status")
            if stored_status is not None:
                status = str(stored_status).upper()
                severity = (
                    "review"
                    if status in ("REVIEW", "VIOLATED", "WARNING", "UNMET", "FAIL", "FAILED")
                    else "neutral"
                )
            else:
                status = "DOCUMENTED"
                severity = "neutral"
            diagnostics_list.append(
                DisplayDiagnostic(
                    label="Min Expected Count",
                    status=status,
                    detail=f"Minimum expected cell count is {min_exp_str} (standard rule >= 5).",
                    severity=severity,
                )
            )

    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("departures from independence" in lim for lim in limitations):
        limitations.append(
            "Chi-square tests evaluate departures from independence; "
            "they do not establish direction or causality."
        )
    if not any("practical importance" in lim for lim in limitations):
        limitations.append(
            "Statistical significance alone does not establish practical importance."
        )

    compact_text = (
        f"Pearson chi-square | N={format_sample_size(sample_size)} | "
        f"{stat_str} | V={v_str} | p={p_str}"
    )

    return TerminalView(
        title="Categorical Association Analysis",
        subtitle=f"{outcome} by {predictor}",
        family="family.categorical",
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


def adapt_fisher_exact(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a Fisher's exact 2x2 test workflow."""
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    outcome = (spec.question.outcome if spec and spec.question else None) or "Outcome"
    predictor = (spec.question.predictor if spec and spec.question else None) or "Predictor"

    sample_size = analysis.sample_size
    excluded_rows = first_present(analysis, "excluded_rows", default=0)

    design_metrics = (
        DisplayMetric("Method", "Fisher's exact test (2x2)", role="method"),
        DisplayMetric("Estimand", "Exact binary independence and odds ratio"),
        DisplayMetric("Sample", f"{format_sample_size(sample_size)} observations"),
        DisplayMetric("Excluded", f"{format_sample_size(excluded_rows)} rows"),
    )

    odds_ratio = analysis.values.get("primary_estimate")
    or_str = format_odds_ratio(odds_ratio, decimals=3)
    p_val = analysis.values.get("p_value")
    p_str = format_p_value(p_val)
    ci_dict = analysis.values.get("confidence_interval")
    if ci_dict is None:
        ci_str = "Unavailable (zero cell observed; no ad-hoc pseudocount added)"
    else:
        ci_str = format_confidence_interval(ci_dict, decimals=3)

    key_metrics_list = [
        DisplayMetric("Odds ratio", or_str, role="result.estimate"),
        DisplayMetric("95% CI", ci_str, role="result.ci"),
        DisplayMetric("p-value (two-sided)", p_str, role="result.evidence"),
    ]

    tables: list[DisplayTable] = []
    observed = analysis.metadata.get("observed_counts")
    rows_lvl = analysis.metadata.get("row_order", ["Row 1", "Row 2"])
    cols_lvl = analysis.metadata.get("column_order", ["Col 1", "Col 2"])
    if isinstance(observed, list) and len(observed) == 2:
        cols = [f"{outcome} \\ {predictor}"] + [str(c) for c in cols_lvl]
        t_rows = [
            DisplayRow(
                (
                    str(rows_lvl[0]),
                    format_sample_size(observed[0][0]),
                    format_sample_size(observed[0][1]),
                )
            ),
            DisplayRow(
                (
                    str(rows_lvl[1]),
                    format_sample_size(observed[1][0]),
                    format_sample_size(observed[1][1]),
                )
            ),
        ]
        tables.append(
            DisplayTable(title="2x2 CONTINGENCY TABLE", columns=tuple(cols), rows=tuple(t_rows))
        )

    diagnostics_list = [
        DisplayDiagnostic(
            label="Conditioning",
            status="EXACT",
            detail="Conditioned on both fixed marginal totals under hypergeometric distribution.",
            severity="neutral",
        )
    ]
    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("marginal" in lim for lim in limitations):
        limitations.append(
            "Fisher's exact test evaluates binary independence conditional on marginals; "
            "it does not establish causation."
        )
    if not any("Zero-cell" in lim for lim in limitations):
        limitations.append(
            "Zero-cell counts result in unavailable analytical confidence intervals; "
            "no ad-hoc pseudocounts are added."
        )

    compact_text = f"Fisher's exact | N={format_sample_size(sample_size)} | OR={or_str} | p={p_str}"

    return TerminalView(
        title="Exact 2x2 Binary Association",
        subtitle=f"{outcome} by {predictor}",
        family="family.categorical",
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


def adapt_mcnemar(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a McNemar paired binary test workflow."""
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    outcome = (spec.question.outcome if spec and spec.question else None) or "Outcome"
    condition = (spec.question.predictor if spec and spec.question else None) or "Condition"
    unit_id = analysis.metadata.get("unit_id") or "unit_id"
    cond_order = analysis.metadata.get("condition_order", ["Condition 1", "Condition 2"])

    sample_size = analysis.sample_size
    excluded_rows = first_present(analysis, "excluded_rows", default=0)

    design_metrics = (
        DisplayMetric("Method", "McNemar paired binary test", role="method"),
        DisplayMetric("Estimand", "Paired marginal proportion difference"),
        DisplayMetric("Pairing Unit", str(unit_id)),
        DisplayMetric("Condition Order", f"'{cond_order[0]}' vs '{cond_order[1]}'"),
        DisplayMetric("Sample", f"{format_sample_size(sample_size)} observations"),
        DisplayMetric("Excluded", f"{format_sample_size(excluded_rows)} rows"),
    )

    diff = analysis.values.get("primary_estimate")
    diff_str = format_number(diff, decimals=3)
    ci_dict = analysis.values.get("confidence_interval")
    ci_str = format_confidence_interval(ci_dict, decimals=3)
    p_val = analysis.values.get("p_value")
    p_str = format_p_value(p_val)
    matched_or = analysis.values.get("matched_odds_ratio")
    or_str = format_odds_ratio(matched_or, decimals=3)

    key_metrics_list = [
        DisplayMetric("Proportion difference", diff_str, role="result.estimate"),
        DisplayMetric("95% CI", ci_str, role="result.ci"),
        DisplayMetric("Matched Odds Ratio", or_str, role="result.effect"),
        DisplayMetric("p-value (exact)", p_str, role="result.evidence"),
    ]

    # Transition table
    tables: list[DisplayTable] = []
    trans = analysis.values.get("transition_table")
    if isinstance(trans, dict):
        b_cell = first_present(trans, "discordant_b", "b", default=0)
        c_cell = first_present(trans, "discordant_c", "c", default=0)
        concord_a = first_present(trans, "concordant_a", "a", default=0)
        concord_d = first_present(trans, "concordant_d", "d", default=0)
        cols = ("Pair Transition", f"{cond_order[1]}: Event", f"{cond_order[1]}: Non-event")
        t_rows = [
            DisplayRow(
                (
                    f"{cond_order[0]}: Event",
                    format_sample_size(concord_a),
                    format_sample_size(b_cell),
                )
            ),
            DisplayRow(
                (
                    f"{cond_order[0]}: Non-event",
                    format_sample_size(c_cell),
                    format_sample_size(concord_d),
                )
            ),
        ]
        tables.append(
            DisplayTable(title="PAIRED 2x2 TRANSITION TABLE", columns=cols, rows=tuple(t_rows))
        )

    trans_dict = trans if isinstance(trans, dict) else {}
    disc_b = trans_dict.get("discordant_b", 0)
    disc_c = trans_dict.get("discordant_c", 0)
    diagnostics_list = [
        DisplayDiagnostic(
            label="Discordant Pairs",
            status="COUNTED",
            detail=f"b={disc_b} vs c={disc_c} discordant transitions.",
            severity="neutral",
        ),
        DisplayDiagnostic(
            label="Paired Design",
            status="PAIRED UNITS",
            detail="Evaluates within-unit binary transitions; not independent proportions.",
            severity="neutral",
        ),
    ]
    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("marginal homogeneity" in lim for lim in limitations):
        limitations.append(
            "McNemar test evaluates marginal homogeneity of paired binary data; "
            "it must not be interpreted as independent proportions."
        )
    if not any("practical importance" in lim for lim in limitations):
        limitations.append(
            "Statistical significance alone does not establish practical importance."
        )

    compact_text = (
        f"McNemar paired binary | N={format_sample_size(sample_size)} | "
        f"diff={diff_str} | 95% CI {ci_str} | p={p_str}"
    )

    return TerminalView(
        title="Paired Categorical Association",
        subtitle=f"{outcome} by {condition} (within {unit_id})",
        family="family.categorical",
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
