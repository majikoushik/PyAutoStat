"""Adapters for scale and rater reliability (Cronbach's alpha and ICC)."""

from __future__ import annotations

from ...results import AnalysisResult
from ...workflow import ResearchWorkflowResult
from ..formatting import (
    format_confidence_interval,
    format_confidence_level_label,
    format_df,
    format_number,
    format_p_value,
    format_percent,
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


def adapt_cronbach_alpha(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a Cronbach's alpha internal consistency scale workflow."""
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    conf_level = resolve_confidence_level(analysis, spec)
    items = analysis.values.get("items") or []
    item_count = analysis.values.get("item_count") or len(items)

    sample_info = analysis.metadata.get("sample") or analysis.values.get("sample", {})
    sample_size = first_present(sample_info, "analyzed_rows", default=analysis.sample_size)
    excluded_rows = first_present(sample_info, "excluded_rows", default=analysis.excluded_rows)
    missing_policy = sample_info.get("missing_data_policy")
    scoring_info = analysis.values.get("scoring", {})
    rev_applied = scoring_info.get("reverse_scoring_applied", False)
    rev_items = scoring_info.get("reversed_items")

    item_summary = ", ".join(str(it) for it in items[:5])
    if len(items) > 5:
        item_summary += "..."
    design_metrics_list = [
        DisplayMetric("Method", "Cronbach's alpha internal consistency", role="method"),
        DisplayMetric("Scale Items", f"{item_count} items ({item_summary})"),
        DisplayMetric("Respondents (N)", f"{format_sample_size(sample_size)} complete cases"),
        DisplayMetric("Excluded", f"{format_sample_size(excluded_rows)} rows"),
    ]
    if missing_policy:
        design_metrics_list.append(DisplayMetric("Missingness Policy", str(missing_policy)))
    if rev_applied:
        if isinstance(rev_items, list) and rev_items:
            rev_names = ", ".join(
                str(it.get("item", it)) if isinstance(it, dict) else str(it) for it in rev_items
            )
            design_metrics_list.append(
                DisplayMetric("Reverse Scoring", f"Applied to {len(rev_items)} items ({rev_names})")
            )
        elif isinstance(rev_items, dict) and rev_items:
            rev_names = ", ".join(str(k) for k in rev_items.keys())
            design_metrics_list.append(
                DisplayMetric("Reverse Scoring", f"Applied to {len(rev_items)} items ({rev_names})")
            )
        else:
            design_metrics_list.append(DisplayMetric("Reverse Scoring", "Applied"))
    else:
        design_metrics_list.append(
            DisplayMetric("Reverse Scoring", "None (original item scales preserved)")
        )
    design_metrics = tuple(design_metrics_list)

    alpha_val = analysis.values.get("primary_estimate")
    alpha_str = format_number(alpha_val, decimals=3)
    ci_dict = analysis.values.get("confidence_interval")
    ci_str = format_confidence_interval(ci_dict, decimals=3)
    ci_label = format_confidence_level_label(
        ci_dict, confidence_level=conf_level, suffix="Bootstrap CI"
    )
    mean_inter = analysis.values.get("mean_inter_item_correlation")
    mean_inter_str = format_number(mean_inter, decimals=3)

    key_metrics_list = [
        DisplayMetric("Cronbach's alpha", alpha_str, role="result.estimate"),
        DisplayMetric(ci_label, ci_str, role="result.ci"),
        DisplayMetric("Mean Inter-Item r", mean_inter_str, role="result.effect"),
    ]

    # Item statistics table
    tables: list[DisplayTable] = []
    item_stats = analysis.values.get("item_statistics")
    if isinstance(item_stats, list) and item_stats:
        cols = ("Item", "Mean", "SD", "Item-Total r", "Alpha if Deleted")
        rows = []
        for it in item_stats:
            name = str(it.get("item", it.get("name", "")))
            m = format_number(it.get("mean"), decimals=2)
            sd = format_number(first_present(it, "sd", "standard_deviation"), decimals=2)
            citr = format_number(
                first_present(it, "corrected_item_total_correlation", "item_total_correlation"),
                decimals=3,
            )
            aid = format_number(it.get("alpha_if_deleted"), decimals=3)
            rows.append(DisplayRow((name, m, sd, citr, aid)))
        tables.append(DisplayTable(title="ITEM-LEVEL DIAGNOSTICS", columns=cols, rows=tuple(rows)))

    # Inter-item correlation summary and item missingness in full mode
    if detail == "full":
        missingness = analysis.values.get("missingness")
        if isinstance(missingness, list) and missingness:
            m_cols = ("Item", "Valid Count", "Missing Count", "Missing %")
            m_rows = [
                DisplayRow(
                    (
                        str(m.get("item", "")),
                        format_sample_size(m.get("valid_count")),
                        format_sample_size(m.get("missing_count")),
                        format_percent(m.get("missing_percentage")),
                    )
                )
                for m in missingness
            ]
            tables.append(
                DisplayTable(title="ITEM-LEVEL MISSINGNESS", columns=m_cols, rows=tuple(m_rows))
            )

        inter_corr = analysis.values.get("inter_item_correlations")
        if isinstance(inter_corr, dict) and "items" in inter_corr and "values" in inter_corr:
            items_list = list(inter_corr["items"])
            matrix_vals = inter_corr["values"]
            matrix_cols = tuple(["Item"] + items_list)
            i_rows = []
            for idx, it in enumerate(items_list):
                row_vals = [it] + [
                    format_number(matrix_vals[idx][j], decimals=2)
                    if idx < len(matrix_vals) and j < len(matrix_vals[idx])
                    else ""
                    for j in range(len(items_list))
                ]
                i_rows.append(DisplayRow(tuple(row_vals)))
            tables.append(
                DisplayTable(
                    title="INTER-ITEM CORRELATION MATRIX", columns=matrix_cols, rows=tuple(i_rows)
                )
            )

    diagnostics_list = [
        DisplayDiagnostic(
            label="Scale Dimensionality",
            status="ADVISORY",
            detail=(
                "Alpha reflects internal consistency under tau-equivalence; "
                "it does not prove unidimensionality."
            ),
            severity="neutral",
        )
    ]
    neg_info = analysis.values.get("negative_inter_item_correlations")
    if isinstance(neg_info, dict):
        neg_count = neg_info.get("count", 0)
        if neg_count > 0:
            most_neg = neg_info.get("most_negative_pair")
            if isinstance(most_neg, dict):
                p1 = most_neg.get("first_item", "item1")
                p2 = most_neg.get("second_item", "item2")
                r_val = format_number(most_neg.get("correlation"), decimals=2)
                detail_text = (
                    f"{neg_count} negative inter-item correlation(s) detected "
                    f"(most negative: {p1} and {p2}, r={r_val}); review item scoring direction."
                )
            else:
                detail_text = (
                    f"{neg_count} negative inter-item correlation(s) detected; "
                    "review item scoring direction."
                )
            diagnostics_list.append(
                DisplayDiagnostic(
                    label="Inter-Item Alignment",
                    status="REVIEW",
                    detail=detail_text,
                    severity="review",
                )
            )
        else:
            diagnostics_list.append(
                DisplayDiagnostic(
                    label="Inter-Item Alignment",
                    status="CONSISTENT",
                    detail="All inter-item correlations are non-negative in the analyzed sample.",
                    severity="neutral",
                )
            )
    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("unidimensionality" in lim for lim in limitations):
        limitations.append(
            "High Cronbach's alpha does not prove unidimensionality or construct validity."
        )
    if not any("mechanically delete" in lim for lim in limitations):
        limitations.append(
            "Do not mechanically delete items solely to maximize alpha without "
            "theoretical justification."
        )

    ci_short_label = format_confidence_level_label(ci_dict, confidence_level=conf_level)
    compact_text = (
        f"Cronbach's alpha | Items={item_count} | N={format_sample_size(sample_size)} | "
        f"alpha={alpha_str} | {ci_short_label} {ci_str}"
    )

    return TerminalView(
        title="Scale Reliability Analysis",
        subtitle=f"{item_count} items (N={format_sample_size(sample_size)})",
        family="family.reliability",
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


def adapt_intraclass_correlation(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt an Intraclass Correlation Coefficient (ICC) rater reliability workflow."""
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    conf_level = resolve_confidence_level(analysis, spec)

    # MANDATORY: Extract canonical definition fields first
    variant = analysis.values.get("variant") or analysis.values.get("notation") or "ICC"
    model = analysis.values.get("model") or "two_way_random"
    definition = analysis.values.get("definition") or "absolute_agreement"
    unit = analysis.values.get("unit") or "single"
    desc = analysis.values.get("description") or f"{model}, {definition}, {unit}"

    n_targets = analysis.values.get("n_targets")
    n_raters = analysis.values.get("n_raters")
    sample_size = analysis.sample_size
    excluded_rows = first_present(analysis, "excluded_rows", default=0)

    design_metrics = (
        DisplayMetric("Canonical Form", str(variant), role="method"),
        DisplayMetric("Model", str(model)),
        DisplayMetric("Definition", str(definition)),
        DisplayMetric("Measure Unit", f"{unit} measure"),
        DisplayMetric(
            "Targets x Raters",
            f"{format_sample_size(n_targets)} targets x {format_sample_size(n_raters)} raters",
        ),
        DisplayMetric("Complete Panel", f"{format_sample_size(sample_size)} ratings"),
        DisplayMetric("Excluded Rows", f"{format_sample_size(excluded_rows)} rows"),
    )

    # Primary estimate - preserving negative values exactly!
    icc_val = analysis.values.get("primary_estimate")
    icc_str = format_number(icc_val, decimals=3)
    ci_dict = analysis.values.get("confidence_interval")
    ci_str = format_confidence_interval(ci_dict, decimals=3)
    ci_label = format_confidence_level_label(ci_dict, confidence_level=conf_level)

    f_test = analysis.values.get("f_test", {})
    f_stat = f_test.get("statistic")
    f_df = [f_test.get("df1"), f_test.get("df2")]
    f_p = f_test.get("p_value")
    f_str = format_statistic("F", f_stat, df=f_df)
    f_p_str = format_p_value(f_p)

    key_metrics_list = [
        DisplayMetric(f"{variant} Estimate", icc_str, role="result.estimate"),
        DisplayMetric(ci_label, ci_str, role="result.ci"),
        DisplayMetric("Target F-test", f_str, role="result.evidence"),
        DisplayMetric("F-test p-value", f_p_str, role="result.evidence"),
    ]

    # Tables: ANOVA Mean Squares & Variance Components
    tables: list[DisplayTable] = []
    anova_tbl = analysis.values.get("anova_table")
    if isinstance(anova_tbl, dict):
        cols = ("Source", "Sum of Squares", "df", "Mean Square")
        rows = []
        for src, data in anova_tbl.items():
            if isinstance(data, dict):
                src_name = str(src).replace("_", " ").title()
                ss = format_number(first_present(data, "sum_of_squares", "ss"), decimals=2)
                df = format_df(first_present(data, "degrees_of_freedom", "df"))
                ms = format_number(first_present(data, "mean_square", "ms"), decimals=2)
                rows.append(DisplayRow((src_name, ss, df, ms)))
        if rows:
            tables.append(DisplayTable(title="ANOVA MEAN SQUARES", columns=cols, rows=tuple(rows)))

    var_comp = analysis.values.get("variance_components")
    if isinstance(var_comp, dict) and var_comp:
        v_cols = ("Component", "Estimate")
        rows = []
        for comp, val in var_comp.items():
            c_name = str(comp).replace("_", " ").title()
            v_str = format_number(val, decimals=4)
            rows.append(DisplayRow((c_name, v_str)))
        if rows:
            tables.append(
                DisplayTable(title="VARIANCE COMPONENTS", columns=v_cols, rows=tuple(rows))
            )

    diagnostics_list = [
        DisplayDiagnostic(
            label="ICC Definition",
            status="CANONICAL",
            detail=str(desc),
            severity="neutral",
        ),
        DisplayDiagnostic(
            label="Negative Estimates",
            status="PRESERVED",
            detail=(
                "Negative estimates indicate within-target noise exceeds "
                "between-target variance; never clamped."
            ),
            severity="neutral",
        ),
    ]
    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("significant F test" in lim for lim in limitations):
        limitations.append(
            "A statistically significant F test does NOT prove acceptable "
            "or practically adequate reliability."
        )
    if not any("relative reliability" in lim for lim in limitations):
        limitations.append(
            "ICC evaluates relative reliability across sample targets; "
            "it depends heavily on sample target heterogeneity."
        )
    if not any("Negative sample estimates" in lim for lim in limitations):
        limitations.append(
            "Negative sample estimates are preserved and indicate that "
            "rater or measurement noise dominates."
        )

    compact_text = (
        f"ICC | {variant} ({model}, {definition}, {unit}) | "
        f"Targets={format_sample_size(n_targets)}, Raters={format_sample_size(n_raters)} | "
        f"ICC={icc_str} | {ci_label} {ci_str} | {f_str} | p={f_p_str}"
    )

    metadata = build_metadata_dict(analysis, workflow)
    metadata["variant"] = variant

    return TerminalView(
        title="Intraclass Correlation (ICC) Reliability",
        subtitle=f"{variant}: {desc}",
        family="family.reliability",
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
