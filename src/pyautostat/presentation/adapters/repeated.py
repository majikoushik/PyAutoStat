"""Adapters for repeated measures ANOVA and Friedman tests."""

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


def adapt_repeated_measures_anova(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a one-way repeated-measures ANOVA workflow."""
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    outcome = (spec.question.outcome if spec and spec.question else None) or "Outcome"
    condition = (spec.question.predictor if spec and spec.question else None) or "Condition"
    unit_id = analysis.metadata.get("unit_id") or "unit_id"
    cond_order = analysis.metadata.get("condition_order", [])
    conf_level = resolve_confidence_level(analysis, spec)

    sample_size = analysis.sample_size
    excluded_rows = first_present(analysis, "excluded_rows", default=0)

    c_summary = ", ".join(str(c) for c in cond_order[:4])
    if len(cond_order) > 4:
        c_summary += "..."
    design_metrics = (
        DisplayMetric("Method", "Repeated-Measures ANOVA", role="method"),
        DisplayMetric("Estimand", "Equality of repeated-condition population means"),
        DisplayMetric("Pairing Unit", str(unit_id)),
        DisplayMetric("Conditions", f"{len(cond_order)} conditions ({c_summary})"),
        DisplayMetric("Sample (Rows)", f"{format_sample_size(sample_size)} observations"),
        DisplayMetric("Excluded Rows", f"{format_sample_size(excluded_rows)} rows"),
    )

    stat = first_present(analysis.values, "test_statistic", "statistic")
    df = analysis.values.get("degrees_of_freedom")
    p_val = analysis.values.get("p_value")
    p_str = format_p_value(p_val)
    stat_str = format_statistic("F", stat, df=df)

    key_metrics_list = [
        DisplayMetric("Omnibus F-test", stat_str, role="result.estimate"),
        DisplayMetric("Primary p-value", p_str, role="result.evidence"),
    ]

    effect_obj = analysis.values.get("effect_size")
    eta_str = "—"
    if isinstance(effect_obj, dict):
        eta_val = effect_obj.get("value")
        eta_str = format_effect(eta_val, decimals=3)
        eta_ci = effect_obj.get("confidence_interval")
        eta_ci_str = format_confidence_interval(eta_ci, decimals=3)
        eta_ci_label = format_confidence_level_label(
            eta_ci, confidence_level=conf_level, prefix="Effect"
        )
        if eta_val is not None:
            key_metrics_list.append(
                DisplayMetric("Partial eta-squared", eta_str, role="result.effect")
            )
        if eta_ci is not None:
            key_metrics_list.append(DisplayMetric(eta_ci_label, eta_ci_str, role="result.ci"))

    # Tables: Condition summaries & Pairwise comparisons
    tables: list[DisplayTable] = []
    cond_sums = analysis.values.get("condition_summaries")
    if isinstance(cond_sums, list) and cond_sums:
        cols = ("Condition", "N", "Mean", "SD")
        rows = [
            DisplayRow(
                (
                    str(c.get("condition", c.get("level", ""))),
                    format_sample_size(first_present(c, "n", "size")),
                    format_number(c.get("mean"), decimals=2),
                    format_number(first_present(c, "sd", "standard_deviation"), decimals=2),
                )
            )
            for c in cond_sums
        ]
        tables.append(DisplayTable(title="CONDITION SUMMARIES", columns=cols, rows=tuple(rows)))

    pairwise = analysis.values.get("pairwise_comparisons")
    if isinstance(pairwise, list) and pairwise:
        pair_ci = pairwise[0].get("confidence_interval") if pairwise else None
        p_ci_label = format_confidence_level_label(pair_ci, confidence_level=conf_level)
        p_cols = ("Contrast", "Difference", p_ci_label, "Cohen's dz", "Adjusted p", "Decision")
        p_rows = []
        for p in pairwise:
            c1 = first_present(p, "first_condition", "condition1", default="")
            c2 = first_present(p, "second_condition", "condition2", default="")
            contrast = f"{c1} - {c2}" if (c1 != "" and c2 != "") else "-"
            if p.get("status") == "unavailable":
                diff = "Unavailable"
                ci = "Unavailable"
                dz = "Unavailable"
                adj_p = "Unavailable"
                decision = str(p.get("reason", "Unavailable"))
            else:
                diff = format_number(first_present(p, "estimate", "mean_difference"), decimals=2)
                ci = format_confidence_interval(p.get("confidence_interval"), decimals=2)
                eff_dict = p.get("effect_size") or {}
                dz_val = eff_dict.get("value") if isinstance(eff_dict, dict) else p.get("cohen_dz")
                dz = format_number(dz_val, decimals=2)
                adj_p = format_p_value(p.get("adjusted_p_value"))
                dec_raw = p.get("decision")
                decision = (
                    "Reject H0"
                    if dec_raw == "reject"
                    else ("Fail to reject" if dec_raw == "fail_to_reject" else str(dec_raw or ""))
                )
            p_rows.append(DisplayRow((contrast, diff, ci, dz, adj_p, decision)))
        tables.append(
            DisplayTable(title="PAIRWISE FOLLOW-UP COMPARISONS", columns=p_cols, rows=tuple(p_rows))
        )

    # Sphericity Diagnostics
    diagnostics_list: list[DisplayDiagnostic] = []
    sphericity = analysis.values.get("sphericity")
    if isinstance(sphericity, dict):
        mauchly_p = sphericity.get("p_value")
        w_stat_str = format_number(first_present(sphericity, "statistic", "mauchly_w"), decimals=3)
        sph_status_raw = sphericity.get("status")
        sph_status = (
            "VIOLATED"
            if sph_status_raw == "rejected"
            else (
                "NOT REJECTED"
                if sph_status_raw == "not_rejected"
                else str(sph_status_raw or "DOCUMENTED").upper()
            )
        )
        sph_decision = sphericity.get("decision", "")
        sph_detail = f"W = {w_stat_str}, p = {format_p_value(mauchly_p)}."
        if sph_decision:
            sph_detail += f" {sph_decision}"
        diagnostics_list.append(
            DisplayDiagnostic(
                label="Mauchly's Sphericity",
                status=sph_status,
                detail=sph_detail,
                severity="warning" if sph_status_raw == "rejected" else "neutral",
            )
        )
    gg = analysis.values.get("greenhouse_geisser")
    if isinstance(gg, dict):
        eps = gg.get("epsilon")
        diagnostics_list.append(
            DisplayDiagnostic(
                label="Greenhouse-Geisser",
                status="APPLIED"
                if analysis.values.get("primary_inference") == "greenhouse_geisser"
                else "DOCUMENTED",
                detail=f"Epsilon = {format_number(eps, decimals=3)}.",
                severity="neutral",
            )
        )

    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("sphericity" in lim for lim in limitations):
        limitations.append(
            "Repeated-measures ANOVA requires sphericity or degrees-of-freedom correction."
        )
    if not any("practical importance" in lim for lim in limitations):
        limitations.append(
            "Statistical significance alone does not establish practical importance."
        )

    compact_text = (
        f"Repeated-measures ANOVA | Conditions={len(cond_order)} | "
        f"{stat_str} | eta_p2={eta_str} | p={p_str}"
    )

    metadata = build_metadata_dict(analysis, workflow)
    metadata["multiplicity"] = analysis.metadata.get("multiplicity_control")

    return TerminalView(
        title="Repeated-Measures ANOVA",
        subtitle=f"{outcome} across {condition} (within {unit_id})",
        family="family.repeated",
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


def adapt_friedman_test(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a Friedman nonparametric repeated-measures rank test workflow."""
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    outcome = (spec.question.outcome if spec and spec.question else None) or "Outcome"
    condition = (spec.question.predictor if spec and spec.question else None) or "Condition"
    unit_id = analysis.metadata.get("unit_id") or "unit_id"
    cond_order = analysis.metadata.get("condition_order", [])

    sample_size = analysis.sample_size
    excluded_rows = first_present(analysis, "excluded_rows", default=0)

    c_summary = ", ".join(str(c) for c in cond_order[:4])
    if len(cond_order) > 4:
        c_summary += "..."
    design_metrics = (
        DisplayMetric("Method", "Friedman rank sum test", role="method"),
        DisplayMetric("Estimand", "Within-unit repeated-condition rank distribution"),
        DisplayMetric("Pairing Unit", str(unit_id)),
        DisplayMetric("Conditions", f"{len(cond_order)} conditions ({c_summary})"),
        DisplayMetric("Sample (Rows)", f"{format_sample_size(sample_size)} observations"),
        DisplayMetric("Excluded Rows", f"{format_sample_size(excluded_rows)} rows"),
    )

    conf_level = resolve_confidence_level(analysis, spec)
    stat = first_present(analysis.values, "test_statistic", "statistic")
    df = analysis.values.get("degrees_of_freedom")
    p_val = analysis.values.get("p_value")
    p_str = format_p_value(p_val)
    stat_str = format_statistic("Friedman Q", stat, df=df)

    key_metrics_list = [
        DisplayMetric("Friedman Q Test", stat_str, role="result.estimate"),
        DisplayMetric("p-value", p_str, role="result.evidence"),
    ]

    effect_obj = analysis.values.get("effect_size")
    w_str = "—"
    if isinstance(effect_obj, dict):
        w_val = effect_obj.get("value")
        w_str = format_effect(w_val, decimals=3)
        w_ci = effect_obj.get("confidence_interval")
        w_ci_str = format_confidence_interval(w_ci, decimals=3)
        w_ci_label = format_confidence_level_label(
            w_ci, confidence_level=conf_level, prefix="Effect"
        )
        if w_val is not None:
            key_metrics_list.append(DisplayMetric("Kendall's W", w_str, role="result.effect"))
        if w_ci is not None:
            key_metrics_list.append(DisplayMetric(w_ci_label, w_ci_str, role="result.ci"))

    # Tables: Condition summaries (medians/IQRs) & Pairwise comparisons
    tables: list[DisplayTable] = []
    cond_sums = analysis.values.get("condition_summaries")
    if isinstance(cond_sums, list) and cond_sums:
        cols = ("Condition", "N", "Median", "IQR")
        rows = [
            DisplayRow(
                (
                    str(first_present(c, "condition", "level", default="")),
                    format_sample_size(first_present(c, "n", "size")),
                    format_number(c.get("median"), decimals=2),
                    format_number(c.get("iqr"), decimals=2),
                )
            )
            for c in cond_sums
        ]
        tables.append(
            DisplayTable(title="CONDITION RANK SUMMARIES", columns=cols, rows=tuple(rows))
        )

    pairwise = analysis.values.get("pairwise_comparisons")
    if isinstance(pairwise, list) and pairwise:
        p_cols = ("Contrast", "Wilcoxon W", "Rank-biserial r", "Adjusted p", "Decision")
        p_rows = []
        for p in pairwise:
            c1 = first_present(p, "first_condition", "condition1", default="")
            c2 = first_present(p, "second_condition", "condition2", default="")
            contrast = f"{c1} vs {c2}"
            if p.get("status") == "unavailable":
                w_sub = "Unavailable"
                rb = "Unavailable"
                adj_p = "Unavailable"
                decision = str(p.get("reason", "Unavailable"))
            else:
                w_sub = format_number(p.get("statistic"), decimals=1)
                eff_dict = p.get("effect_size") or {}
                rb_val = (
                    eff_dict.get("value")
                    if isinstance(eff_dict, dict)
                    else p.get("rank_biserial_r")
                )
                rb = format_number(rb_val, decimals=3)
                adj_p = format_p_value(p.get("adjusted_p_value"))
                dec_raw = p.get("decision")
                decision = (
                    "Reject H0"
                    if dec_raw == "reject"
                    else ("Fail to reject" if dec_raw == "fail_to_reject" else str(dec_raw or ""))
                )
            p_rows.append(DisplayRow((contrast, w_sub, rb, adj_p, decision)))
        tables.append(
            DisplayTable(title="PAIRWISE WILCOXON FOLLOW-UP", columns=p_cols, rows=tuple(p_rows))
        )

    diagnostics_list = [
        DisplayDiagnostic(
            label="Rank Model",
            status="NONPARAMETRIC",
            detail=(
                "Evaluates within-unit ranks across repeated conditions; "
                "does not require outcome normality."
            ),
            severity="neutral",
        )
    ]
    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("universally a test of medians" in lim for lim in limitations):
        limitations.append(
            "Friedman test evaluates within-unit rank distributions; "
            "it is not universally a test of medians."
        )
    if not any("practical importance" in lim for lim in limitations):
        limitations.append(
            "Statistical significance alone does not establish practical importance."
        )

    compact_text = (
        f"Friedman test | Conditions={len(cond_order)} | {stat_str} | W={w_str} | p={p_str}"
    )

    metadata = build_metadata_dict(analysis, workflow)
    metadata["multiplicity"] = analysis.metadata.get("multiplicity_control")

    return TerminalView(
        title="Friedman Repeated-Measures Rank Test",
        subtitle=f"{outcome} across {condition} (within {unit_id})",
        family="family.rank",
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
