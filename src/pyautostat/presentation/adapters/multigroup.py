"""Adapters for independent multi-group comparisons (ANOVA and Kruskal-Wallis)."""

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
    first_present,
    format_interpretation_text,
)


def adapt_welch_anova(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a Welch one-way ANOVA workflow."""
    return _adapt_multigroup(target, detail=detail, method_id="welch_anova")


def adapt_one_way_anova(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a classical one-way ANOVA workflow."""
    return _adapt_multigroup(target, detail=detail, method_id="one_way_anova")


def adapt_kruskal_wallis(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a Kruskal-Wallis rank ANOVA workflow."""
    return _adapt_multigroup(target, detail=detail, method_id="kruskal_wallis")


def _adapt_multigroup(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
    method_id: str = "welch_anova",
) -> TerminalView:
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    outcome = (spec.question.outcome if spec and spec.question else None) or "Outcome"
    predictor = (spec.question.predictor if spec and spec.question else None) or "Group"

    sample_size = analysis.sample_size
    excluded_rows = first_present(analysis, "excluded_rows", default=0)

    if method_id == "welch_anova":
        method_name = "Welch's one-way ANOVA"
        estimand = "Equality of population group means"
        stat_name = "Welch F"
        family = "family.mean"
    elif method_id == "one_way_anova":
        method_name = "One-way ANOVA (equal variance)"
        estimand = "Equality of population group means"
        stat_name = "F"
        family = "family.mean"
    else:  # kruskal_wallis
        method_name = "Kruskal-Wallis rank sum test"
        estimand = "Equality of population rank distributions"
        stat_name = "H"
        family = "family.rank"

    design_metrics = (
        DisplayMetric("Method", method_name, role="method"),
        DisplayMetric("Estimand", estimand),
        DisplayMetric("Sample", f"{format_sample_size(sample_size)} rows"),
        DisplayMetric("Excluded", f"{format_sample_size(excluded_rows)} rows"),
    )

    stat = analysis.values.get("test_statistic")
    df = analysis.values.get("degrees_of_freedom")
    p_val = analysis.values.get("p_value")
    p_str = format_p_value(p_val)
    stat_str = format_statistic(stat_name, stat, df=df)

    key_metrics_list = [
        DisplayMetric("Omnibus Test", stat_str, role="result.estimate"),
        DisplayMetric("p-value", p_str, role="result.evidence"),
    ]

    effect_dict = analysis.values.get("effect_size", {})
    if effect_dict and effect_dict.get("value") is not None:
        eff_name = effect_dict.get("name", "Effect size")
        eff_val = effect_dict.get("value")
        eff_str = format_effect(eff_val, decimals=3)
        key_metrics_list.append(DisplayMetric(eff_name, eff_str, role="result.effect"))
        eff_ci = effect_dict.get("confidence_interval")
        if eff_ci:
            ci_str = format_confidence_interval(eff_ci, decimals=3)
            key_metrics_list.append(DisplayMetric("Effect 95% CI", ci_str, role="result.ci"))

    # Tables: Group summaries and pairwise comparisons
    tables: list[DisplayTable] = []
    group_sums = analysis.values.get("group_summaries")
    if isinstance(group_sums, list) and group_sums:
        if method_id == "kruskal_wallis":
            cols = ("Group", "N", "Median", "IQR")
            rows = [
                DisplayRow(
                    (
                        str(g.get("group", "")),
                        format_sample_size(first_present(g, "size", "n")),
                        format_number(g.get("median"), decimals=2),
                        format_number(g.get("iqr"), decimals=2),
                    )
                )
                for g in group_sums
            ]
        else:
            cols = ("Group", "N", "Mean", "SD")
            rows = [
                DisplayRow(
                    (
                        str(g.get("group", "")),
                        format_sample_size(first_present(g, "sample_size", "size", "n")),
                        format_number(g.get("mean"), decimals=2),
                        format_number(first_present(g, "sd", "standard_deviation"), decimals=2),
                    )
                )
                for g in group_sums
            ]
        tables.append(DisplayTable(title="GROUP SUMMARY", columns=cols, rows=tuple(rows)))

    # Pairwise table
    pairwise = analysis.values.get("pairwise_comparisons")
    if isinstance(pairwise, list) and pairwise:
        pair_cols: tuple[str, ...] = ()
        p_rows: list[DisplayRow] = []
        if method_id == "kruskal_wallis":
            pair_cols = (
                "Contrast",
                "Mean-Rank Diff",
                "Dunn z",
                "Rank-biserial r",
                "Adjusted p",
                "Decision",
            )
            for pair in pairwise:
                contrast_obj = pair.get("contrast")
                g1 = pair.get("group1") or pair.get("first_group")
                g2 = pair.get("group2") or pair.get("second_group")
                if isinstance(contrast_obj, dict):
                    g1 = g1 or contrast_obj.get("first_group") or contrast_obj.get("first")
                    g2 = g2 or contrast_obj.get("second_group") or contrast_obj.get("second")
                    contrast = f"{g1} vs {g2}" if (g1 and g2) else "-"
                elif isinstance(contrast_obj, str) and contrast_obj:
                    contrast = contrast_obj
                else:
                    contrast = f"{g1} vs {g2}" if (g1 and g2) else "-"
                diff_val = format_number(pair.get("estimate"), decimals=2)
                z_val = format_number(pair.get("statistic"), decimals=2)
                eff_dict = pair.get("effect_size") or {}
                rb_val = format_effect(eff_dict.get("value"), decimals=3)
                adj_p = format_p_value(pair.get("adjusted_p_value"))
                dec_raw = pair.get("decision")
                decision = (
                    "Reject H0"
                    if dec_raw == "reject"
                    else ("Fail to reject" if dec_raw == "fail_to_reject" else str(dec_raw or ""))
                )
                p_rows.append(DisplayRow((contrast, diff_val, z_val, rb_val, adj_p, decision)))
            p_title = "DUNN-HOLM PAIRWISE COMPARISONS"
        else:
            pair_cols = (
                "Contrast",
                "Difference",
                "95% Simultaneous CI",
                "Adjusted p",
                "Decision",
            )
            for pair in pairwise:
                contrast_obj = pair.get("contrast")
                g1 = pair.get("group1") or pair.get("first_group")
                g2 = pair.get("group2") or pair.get("second_group")
                if isinstance(contrast_obj, dict):
                    g1 = g1 or contrast_obj.get("first_group") or contrast_obj.get("first")
                    g2 = g2 or contrast_obj.get("second_group") or contrast_obj.get("second")
                    contrast = f"{g1} - {g2}" if (g1 and g2) else "-"
                elif isinstance(contrast_obj, str) and contrast_obj:
                    contrast = contrast_obj
                else:
                    contrast = f"{g1} - {g2}" if (g1 and g2) else "-"
                diff_val = format_number(
                    first_present(pair, "estimate", "mean_difference"), decimals=2
                )
                ci_str = format_confidence_interval(pair.get("confidence_interval"), decimals=2)
                adj_p = format_p_value(pair.get("adjusted_p_value"))
                dec_raw = pair.get("decision")
                decision = (
                    "Reject H0"
                    if dec_raw == "reject"
                    else ("Fail to reject" if dec_raw == "fail_to_reject" else str(dec_raw or ""))
                )
                p_rows.append(DisplayRow((contrast, diff_val, ci_str, adj_p, decision)))
            p_title = (
                "GAMES-HOWELL PAIRWISE COMPARISONS"
                if method_id == "welch_anova"
                else "TUKEY-KRAMER PAIRWISE COMPARISONS"
            )
        tables.append(DisplayTable(title=p_title, columns=pair_cols, rows=tuple(p_rows)))

    if method_id == "welch_anova":
        var_status = "Not assumed"
        var_detail = (
            "Welch ANOVA does not assume equal population variances; Games-Howell follow-up."
        )
        var_severity = "neutral"
    elif method_id == "one_way_anova":
        var_status = "Assumed"
        var_detail = "Equal group variances assumed."
        var_severity = "review"
    else:  # kruskal_wallis
        var_status = "Not applicable"
        var_detail = (
            "Rank-based nonparametric comparison; the classical equal-variance ANOVA "
            "assumption is not invoked."
        )
        var_severity = "neutral"

    diagnostics_list = [
        DisplayDiagnostic(
            label="Equal Variance",
            status=var_status,
            detail=var_detail,
            severity=var_severity,
        )
    ]
    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("Omnibus statistical significance" in lim for lim in limitations):
        limitations.append(
            "Omnibus statistical significance does not identify which specific groups "
            "differ without post-hoc analysis."
        )
    if not any("practical importance" in lim for lim in limitations):
        limitations.append(
            "Statistical significance alone does not establish practical importance."
        )

    compact_text = f"{method_name} | N={format_sample_size(sample_size)} | {stat_str} | p={p_str}"

    metadata = build_metadata_dict(analysis, workflow)
    metadata["multiplicity"] = analysis.metadata.get("multiplicity_control")

    return TerminalView(
        title="Multi-Group Comparison",
        subtitle=f"{outcome} by {predictor}",
        family=family,
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
