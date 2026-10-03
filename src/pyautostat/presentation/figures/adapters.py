"""Adapters extracting FigureSpec models from authoritative PyAutoStat results.

Figure adapters extract display-ready numeric parameters (estimates, confidence intervals,
counts, cell summaries) without performing any inferential recalculation, model refitting,
or p-value calculation.
"""

from __future__ import annotations

import math
from typing import Any

from ...results import AnalysisResult
from ...workflow import ResearchWorkflowResult
from ..adapters.common import extract_context, first_present, resolve_confidence_level
from .models import FigureSeries, FigureSpec


def _extract_ci(val: Any) -> tuple[float | None, float | None]:
    """Extract lower and upper bounds from a confidence interval mapping or sequence."""
    if isinstance(val, dict):
        lower = val.get("lower")
        upper = val.get("upper")
        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            if math.isfinite(lower) and math.isfinite(upper):
                return float(lower), float(upper)
    elif isinstance(val, (tuple, list)) and len(val) == 2:
        l_val, u_val = val[0], val[1]
        if isinstance(l_val, (int, float)) and isinstance(u_val, (int, float)):
            if math.isfinite(l_val) and math.isfinite(u_val):
                return float(l_val), float(u_val)
    return None, None


def _format_ci_level_label(conf_level: float | None) -> str:
    """Format confidence level percentage for figure titles."""
    if conf_level is not None and 0 < conf_level < 1:
        pct = round(conf_level * 100)
        return f"{pct:g}%" if pct == conf_level * 100 else f"{conf_level * 100:.1f}%"
    return "95%"


def build_figure_spec(target: Any) -> FigureSpec | None:
    """Build an authoritative FigureSpec from a supported PyAutoStat analysis or report.

    Parameters
    ----------
    target : Any
        An AnalysisResult, ResearchWorkflowResult, or ResearchReport.

    Returns
    -------
    FigureSpec or None
        Authoritative figure specification, or None if no scientifically appropriate
        figure is available for the given method or result state.
    """
    if target is None:
        return None

    # Handle ResearchReport
    if hasattr(target, "_source_result"):
        source_res = getattr(target, "_source_result", None)
        if source_res is not None:
            return build_figure_spec(source_res)
        payload = getattr(target, "_payload", None)
        if isinstance(payload, dict):
            return _build_figure_spec_from_payload(payload)
        return None

    # Extract standard context
    if isinstance(target, ResearchWorkflowResult):
        if target.analysis is None:
            return None
        analysis, workflow, spec, interp, rec, audit = extract_context(target)
    elif isinstance(target, AnalysisResult):
        analysis, workflow, spec, interp, rec, audit = extract_context(target)
    elif hasattr(target, "method_id") and hasattr(target, "values"):
        analysis = target
        spec = getattr(target, "specification", None)
    else:
        return None

    if (
        analysis.status != "available"
        and getattr(analysis.status, "value", str(analysis.status)) != "available"
    ):
        return None

    method_id = analysis.method_id
    values = analysis.values or {}
    metadata = analysis.metadata or {}
    conf_level = resolve_confidence_level(analysis, spec)
    ci_label = _format_ci_level_label(conf_level)

    # 1. Independent two-group mean comparisons (Welch t, Student t)
    if method_id in ("welch_t", "student_t"):
        diff = values.get("primary_estimate")
        if diff is None or not isinstance(diff, (int, float)) or not math.isfinite(diff):
            return None
        lower, upper = _extract_ci(values.get("confidence_interval"))
        contrast_dict = metadata.get("contrast")
        groups = metadata.get("group_order", [])
        if (
            isinstance(contrast_dict, dict)
            and "first" in contrast_dict
            and "second" in contrast_dict
        ):
            contrast_label = f"'{contrast_dict['first']}' - '{contrast_dict['second']}'"
        elif len(groups) == 2:
            contrast_label = f"'{groups[0]}' - '{groups[1]}'"
        else:
            contrast_label = "Mean difference"

        unit = (
            (spec.data_dictionary or {}).get(spec.question.outcome, {}).get("unit")
            if spec and spec.question
            else None
        )
        x_label = f"Mean Difference ({unit})" if unit else "Mean Difference"
        method_title = "Welch t-test" if method_id == "welch_t" else "Student t-test"

        return FigureSpec(
            kind="estimate_ci",
            title=f"Mean difference and {ci_label} confidence interval",
            subtitle=method_title,
            x_label=x_label,
            y_label=None,
            reference_value=0.0,
            series=(
                FigureSeries(
                    label=contrast_label,
                    estimate=float(diff),
                    lower=lower,
                    upper=upper,
                ),
            ),
            note="Reference line at 0 indicates no difference between groups.",
            placement="KEY RESULTS",
        )

    # 2. One-sample mean comparison
    if method_id == "one_sample_t":
        diff = values.get("primary_estimate")
        if diff is None or not isinstance(diff, (int, float)) or not math.isfinite(diff):
            return None
        lower, upper = _extract_ci(values.get("confidence_interval"))
        outcome = (spec.question.outcome if spec and spec.question else None) or "Sample Mean"
        ref_val = spec.question.reference_value if spec and spec.question else 0.0

        return FigureSpec(
            kind="estimate_ci",
            title=f"Mean difference from reference and {ci_label} confidence interval",
            subtitle=f"One-Sample t-test (Reference = {ref_val})",
            x_label="Mean Difference",
            y_label=None,
            reference_value=0.0,
            series=(
                FigureSeries(
                    label=f"{outcome} - {ref_val}",
                    estimate=float(diff),
                    lower=lower,
                    upper=upper,
                ),
            ),
            note=f"Reference line at 0 indicates sample mean equals reference value ({ref_val}).",
            placement="KEY RESULTS",
        )

    # 3. Paired-samples t-test
    if method_id == "paired_t":
        diff = values.get("primary_estimate")
        if diff is None or not isinstance(diff, (int, float)) or not math.isfinite(diff):
            return None
        lower, upper = _extract_ci(values.get("confidence_interval"))
        contrast_dict = metadata.get("contrast")
        conds = metadata.get("condition_order", [])
        if (
            isinstance(contrast_dict, dict)
            and "first" in contrast_dict
            and "second" in contrast_dict
        ):
            contrast_label = f"'{contrast_dict['first']}' - '{contrast_dict['second']}'"
        elif len(conds) == 2:
            contrast_label = f"'{conds[0]}' - '{conds[1]}'"
        else:
            contrast_label = "Paired difference"

        return FigureSpec(
            kind="estimate_ci",
            title=f"Paired mean difference and {ci_label} confidence interval",
            subtitle="Paired-Samples t-test",
            x_label="Mean Difference",
            y_label=None,
            reference_value=0.0,
            series=(
                FigureSeries(
                    label=contrast_label,
                    estimate=float(diff),
                    lower=lower,
                    upper=upper,
                ),
            ),
            note="Reference line at 0 indicates no mean difference across paired conditions.",
            placement="KEY RESULTS",
        )

    # 4. Correlation / Bivariate association
    if method_id in (
        "pearson_correlation",
        "spearman_correlation",
        "kendall_tau_b",
        "point_biserial_correlation",
        "partial_pearson_correlation",
    ):
        r_val = values.get("primary_estimate")
        if r_val is None or not isinstance(r_val, (int, float)) or not math.isfinite(r_val):
            return None
        lower, upper = _extract_ci(values.get("confidence_interval"))
        outcome = (spec.question.outcome if spec and spec.question else None) or "Outcome"
        predictor = (spec.question.predictor if spec and spec.question else None) or "Predictor"
        pair_label = f"{predictor} & {outcome}"

        name_map = {
            "pearson_correlation": ("Pearson correlation", "r"),
            "spearman_correlation": ("Spearman rank correlation", "rho"),
            "kendall_tau_b": ("Kendall's tau-b", "tau"),
            "point_biserial_correlation": ("Point-biserial correlation", "r_pb"),
            "partial_pearson_correlation": ("Partial Pearson correlation", "r_partial"),
        }
        method_name, symbol = name_map[method_id]

        return FigureSpec(
            kind="estimate_ci",
            title=f"{method_name} estimate and {ci_label} confidence interval",
            subtitle=f"{predictor} vs {outcome}",
            x_label=f"Correlation Coefficient ({symbol})",
            y_label=None,
            reference_value=0.0,
            series=(
                FigureSeries(
                    label=pair_label,
                    estimate=float(r_val),
                    lower=lower,
                    upper=upper,
                ),
            ),
            note="Reference line at 0 indicates no linear or monotonic association.",
            placement="KEY RESULTS",
        )

    # 5. Reliability: ICC and Cronbach alpha
    if method_id in ("icc", "intraclass_correlation"):
        icc_val = values.get("primary_estimate")
        if icc_val is None or not isinstance(icc_val, (int, float)) or not math.isfinite(icc_val):
            return None
        lower, upper = _extract_ci(values.get("confidence_interval"))
        icc_type = metadata.get("icc_type") or "ICC"
        return FigureSpec(
            kind="estimate_ci",
            title=f"Intraclass correlation ({icc_type}) and {ci_label} confidence interval",
            subtitle="Reliability Analysis",
            x_label="ICC Estimate",
            y_label=None,
            reference_value=None,
            series=(
                FigureSeries(
                    label=str(icc_type),
                    estimate=float(icc_val),
                    lower=lower,
                    upper=upper,
                ),
            ),
            note="Values close to 1 indicate high inter-rater or intra-rater agreement.",
            placement="KEY RESULTS",
        )

    if method_id == "cronbach_alpha":
        alpha = values.get("primary_estimate")
        if alpha is None or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
            return None
        lower, upper = _extract_ci(values.get("confidence_interval"))
        return FigureSpec(
            kind="estimate_ci",
            title=f"Cronbach's alpha estimate and {ci_label} confidence interval",
            subtitle="Internal Consistency Reliability",
            x_label="Cronbach's Alpha",
            y_label=None,
            reference_value=None,
            series=(
                FigureSeries(
                    label="Internal Consistency",
                    estimate=float(alpha),
                    lower=lower,
                    upper=upper,
                ),
            ),
            note="Values above 0.70 generally indicate acceptable internal consistency.",
            placement="KEY RESULTS",
        )

    # 6. OLS Linear Regression: Coefficient forest
    if method_id in ("ols_linear_regression", "linear_regression"):
        coefs = values.get("coefficients", [])
        if not isinstance(coefs, list) or not coefs:
            return None
        series_list: list[FigureSeries] = []
        for c in coefs:
            if not isinstance(c, dict):
                continue
            term = str(c.get("term", c.get("term_label", c.get("name", ""))))
            est = c.get("estimate")
            if est is None or not isinstance(est, (int, float)) or not math.isfinite(est):
                continue
            lower, upper = _extract_ci(c.get("confidence_interval"))
            series_list.append(
                FigureSeries(
                    label=term,
                    estimate=float(est),
                    lower=lower,
                    upper=upper,
                    metadata={
                        "standard_error": c.get("standard_error"),
                        "statistic": c.get("statistic"),
                        "p_value": c.get("p_value"),
                        "standardized_estimate": c.get("standardized_estimate"),
                    },
                )
            )
        if not series_list:
            return None
        outcome = values.get("outcome") or "Outcome"
        return FigureSpec(
            kind="forest",
            title=f"Coefficient estimates and {ci_label} confidence intervals",
            subtitle=f"Ordinary Least Squares Regression ({outcome})",
            x_label="Coefficient Estimate",
            y_label="Model Term",
            reference_value=0.0,
            series=tuple(series_list),
            note="Reference line at 0 indicates no linear association.",
            placement="MODEL COEFFICIENTS",
        )

    # 7. Logistic Regression: Odds-ratio forest
    if method_id == "logistic_regression":
        coefs = values.get("coefficients", [])
        if not isinstance(coefs, list) or not coefs:
            return None
        series_list = []
        all_positive = True
        for c in coefs:
            if not isinstance(c, dict):
                continue
            term = str(c.get("term", c.get("term_label", c.get("name", ""))))
            or_val = c.get("odds_ratio")
            if or_val is None or not isinstance(or_val, (int, float)) or not math.isfinite(or_val):
                continue
            lower, upper = _extract_ci(c.get("odds_ratio_ci"))
            if (
                or_val <= 0
                or (lower is not None and lower <= 0)
                or (upper is not None and upper <= 0)
            ):
                all_positive = False
            series_list.append(
                FigureSeries(
                    label=term,
                    estimate=float(or_val),
                    lower=lower,
                    upper=upper,
                    metadata={
                        "raw_estimate": c.get("estimate"),
                        "standard_error": c.get("standard_error"),
                        "statistic": c.get("statistic"),
                        "p_value": c.get("p_value"),
                    },
                )
            )
        if not series_list:
            return None
        event_lvl = first_present(values, "event_level", default="1")
        outcome = values.get("outcome") or "Outcome"
        return FigureSpec(
            kind="odds_ratio_forest",
            title=f"Predictor odds ratios and {ci_label} confidence intervals",
            subtitle=f"Binary Logistic Regression ({outcome}, event='{event_lvl}')",
            x_label="Odds Ratio (log scale)" if all_positive else "Odds Ratio",
            y_label="Predictor",
            reference_value=1.0,
            series=tuple(series_list),
            note="Reference line at 1.0 indicates no change in odds.",
            placement="PREDICTOR ODDS RATIOS",
            layout_hints={"log_x": all_positive},
        )

    # 8. ANOVA pairwise comparisons: Pairwise forest
    if method_id in ("welch_anova", "one_way_anova"):
        pairwise = values.get("pairwise_comparisons", [])
        if not isinstance(pairwise, list) or not pairwise:
            return None
        series_list = []
        for pair in pairwise:
            if not isinstance(pair, dict):
                continue
            contrast_obj = pair.get("contrast")
            g1 = first_present(pair, "group1", "first_group", "first")
            g2 = first_present(pair, "group2", "second_group", "second")
            if g1 is None and isinstance(contrast_obj, dict):
                g1 = first_present(contrast_obj, "first_group", "first")
            if g2 is None and isinstance(contrast_obj, dict):
                g2 = first_present(contrast_obj, "second_group", "second")
            if isinstance(contrast_obj, str) and contrast_obj:
                contrast_str = contrast_obj
            elif g1 is not None and g2 is not None:
                contrast_str = f"{g1} - {g2}"
            else:
                contrast_str = "Contrast"

            diff = first_present(pair, "estimate", "mean_difference")
            if diff is None or not isinstance(diff, (int, float)) or not math.isfinite(diff):
                continue
            lower, upper = _extract_ci(pair.get("confidence_interval"))
            series_list.append(
                FigureSeries(
                    label=contrast_str,
                    estimate=float(diff),
                    lower=lower,
                    upper=upper,
                    metadata={
                        "adjusted_p_value": pair.get("adjusted_p_value"),
                        "decision": pair.get("decision"),
                    },
                )
            )
        if not series_list:
            return None
        method_title = (
            "Games-Howell Post-hoc" if method_id == "welch_anova" else "Tukey-Kramer Post-hoc"
        )
        return FigureSpec(
            kind="pairwise_forest",
            title=f"Pairwise mean differences and simultaneous {ci_label} confidence intervals",
            subtitle=method_title,
            x_label="Mean Difference",
            y_label="Contrast",
            reference_value=0.0,
            series=tuple(series_list),
            note="Reference line at 0 indicates equality of population group means.",
            placement="PAIRWISE",
        )

    # Kruskal-Wallis intentionally omits pairwise forest if no authoritative CI is stored
    if method_id == "kruskal_wallis":
        return None

    # 9. Pearson chi-square: Contingency count heatmap
    if method_id in ("chi_square_independence", "pearson_chi_square"):
        observed = metadata.get("observed_counts")
        groups = metadata.get("group_order", [])
        outcomes = metadata.get("outcome_order", [])
        if not (
            isinstance(observed, list) and groups and outcomes and len(observed) == len(groups)
        ):
            return None
        series_list = []
        for g_idx, g_name in enumerate(groups):
            row = observed[g_idx]
            if not isinstance(row, (list, tuple)) or len(row) != len(outcomes):
                return None
            series_list.append(
                FigureSeries(
                    label=str(g_name),
                    categories=tuple(str(o) for o in outcomes),
                    values=tuple(float(c) for c in row),
                )
            )
        outcome_name = (spec.question.outcome if spec and spec.question else None) or "Outcome"
        predictor_name = (
            spec.question.predictor if spec and spec.question else None
        ) or "Predictor"
        return FigureSpec(
            kind="count_heatmap",
            title="Observed contingency counts",
            subtitle=f"{predictor_name} x {outcome_name}",
            x_label=str(outcome_name),
            y_label=str(predictor_name),
            series=tuple(series_list),
            note="Cell values display observed frequencies across categories.",
            placement="CONTINGENCY TABLE",
        )

    # 10. Two-way Factorial ANOVA: Observed cell profiles
    if method_id in ("twoway_anova", "two_way_anova"):
        cell_sums = values.get("cell_summaries")
        if not isinstance(cell_sums, list) or not cell_sums:
            return None
        factor_a = values.get("factor_a") or "Factor A"
        factor_b = values.get("factor_b") or "Factor B"
        levels_a = values.get("factor_a_levels") or []
        levels_b = values.get("factor_b_levels") or []

        if not levels_a or not levels_b:
            # Extract distinct levels preserving source encounter order
            seen_a: list[str] = []
            seen_b: list[str] = []
            for c in cell_sums:
                la = str(c.get("level_a", c.get(factor_a, "")))
                lb = str(c.get("level_b", c.get(factor_b, "")))
                if la and la not in seen_a:
                    seen_a.append(la)
                if lb and lb not in seen_b:
                    seen_b.append(lb)
            levels_a = seen_a
            levels_b = seen_b

        series_list = []
        for b_lvl in levels_b:
            means: list[float] = []
            for a_lvl in levels_a:
                match = next(
                    (
                        c
                        for c in cell_sums
                        if str(c.get("level_a", c.get(factor_a, ""))) == str(a_lvl)
                        and str(c.get("level_b", c.get(factor_b, ""))) == str(b_lvl)
                    ),
                    None,
                )
                if match is None or match.get("mean") is None:
                    # Incomplete cell grid; omit figure rather than fabricate
                    return None
                m_val = match.get("mean")
                if not isinstance(m_val, (int, float)) or not math.isfinite(m_val):
                    return None
                means.append(float(m_val))

            series_list.append(
                FigureSeries(
                    label=str(b_lvl),
                    categories=tuple(str(a) for a in levels_a),
                    values=tuple(means),
                )
            )

        return FigureSpec(
            kind="cell_profile",
            title="Observed cell means by factor levels",
            subtitle=f"{factor_a} x {factor_b}",
            x_label=str(factor_a),
            y_label="Observed Cell Mean",
            series=tuple(series_list),
            note="Profiles connect observed sample cell means; no CI bands are inferred.",
            placement="CELL SUMMARIES",
        )

    # 11. Fisher exact test: 2x2 contingency count heatmap or OR estimate
    if method_id == "fisher_exact":
        observed = metadata.get("observed_counts")
        row_order = metadata.get("row_order", [])
        col_order = metadata.get("column_order", [])
        if (
            isinstance(observed, list)
            and row_order
            and col_order
            and len(observed) == len(row_order)
        ):
            series_list = [
                FigureSeries(
                    label=str(r),
                    categories=tuple(str(c) for c in col_order),
                    values=tuple(float(v) for v in observed[idx]),
                )
                for idx, r in enumerate(row_order)
            ]
            return FigureSpec(
                kind="count_heatmap",
                title="Observed contingency counts",
                subtitle="Fisher's Exact Test (2x2)",
                x_label="Column Category",
                y_label="Row Category",
                series=tuple(series_list),
                note="Cell values display observed frequencies in 2x2 contingency table.",
                placement="CONTINGENCY TABLE",
            )
        # Fallback to OR if stored
        or_val = values.get("primary_estimate")
        if or_val is not None and isinstance(or_val, (int, float)) and math.isfinite(or_val):
            low, up = _extract_ci(values.get("confidence_interval"))
            return FigureSpec(
                kind="estimate_ci",
                title=f"Odds ratio estimate and {ci_label} confidence interval",
                subtitle="Fisher's Exact Test (2x2)",
                x_label="Odds Ratio",
                y_label=None,
                reference_value=1.0,
                series=(
                    FigureSeries(
                        label="Odds ratio",
                        estimate=float(or_val),
                        lower=low,
                        upper=up,
                    ),
                ),
                note="Reference line at 1.0 indicates equal odds.",
                placement="KEY RESULTS",
            )

    return None


def _build_figure_spec_from_payload(payload: dict[str, Any]) -> FigureSpec | None:
    """Build FigureSpec directly from a ResearchReport canonical payload."""
    sections = payload.get("sections", {})
    if not isinstance(sections, dict):
        return None
    methods_sec = sections.get("methods", {})
    method_id = methods_sec.get("method_id")
    results_sec = sections.get("results", {})
    if not isinstance(results_sec, dict) or not method_id:
        return None

    # Scalar difference / correlation
    diff = results_sec.get("primary_estimate")
    ci = results_sec.get("confidence_interval")
    if (
        diff is not None
        and isinstance(diff, (int, float))
        and math.isfinite(diff)
        and ci is not None
    ):
        low, up = _extract_ci(ci)
        ref_val = 0.0
        method_name = methods_sec.get("method_name") or method_id
        return FigureSpec(
            kind="estimate_ci",
            title="Estimate and 95% confidence interval",
            subtitle=str(method_name),
            x_label="Estimate",
            y_label=None,
            reference_value=ref_val,
            series=(
                FigureSeries(
                    label="Estimate",
                    estimate=float(diff),
                    lower=low,
                    upper=up,
                ),
            ),
            note="Derived from authoritative report result.",
            placement="KEY RESULTS",
        )

    return None


__all__ = [
    "build_figure_spec",
]
