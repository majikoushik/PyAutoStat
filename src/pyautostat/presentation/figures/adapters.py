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
from ..formatting import confidence_interval_phrase
from .models import (
    PLACEMENT_CELL_SUMMARY,
    PLACEMENT_COEFFICIENTS,
    PLACEMENT_CONTINGENCY,
    PLACEMENT_KEY_RESULTS,
    PLACEMENT_PAIRWISE,
    FigureSeries,
    FigureSpec,
)


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


def build_figure_spec(target: Any) -> FigureSpec | None:
    """Build an authoritative FigureSpec from a supported PyAutoStat analysis or report.

    Parameters
    ----------
    target : Any
        An AnalysisResult, ResearchWorkflowResult, or ResearchReport.

    Returns
    -------
    FigureSpec or None
        First authoritative figure specification, or None if no scientifically
        appropriate figure is available for the given method or result state.
    """
    specs = build_figure_specs(target)
    return specs[0] if specs else None


def build_figure_specs(target: Any) -> tuple[FigureSpec, ...]:
    """Build authoritative FigureSpecs from a supported PyAutoStat analysis or report.

    Parameters
    ----------
    target : Any
        An AnalysisResult, ResearchWorkflowResult, or ResearchReport.

    Returns
    -------
    tuple[FigureSpec, ...]
        Authoritative figure specifications, or an empty tuple if no scientifically
        appropriate figures are available for the given method or result state.
    """
    if target is None:
        return ()

    # Handle ResearchReport
    if hasattr(target, "_source_result"):
        source_res = getattr(target, "_source_result", None)
        if source_res is not None:
            return build_figure_specs(source_res)
        payload = getattr(target, "_payload", None)
        if isinstance(payload, dict):
            spec = _build_figure_spec_from_payload(payload)
            return (spec,) if spec is not None else ()
        return ()

    # PresentationView alone does not have raw numerical records from which to reconstruct figures
    from ..models import PresentationView

    if isinstance(target, PresentationView):
        return ()

    # Extract standard context
    if isinstance(target, ResearchWorkflowResult):
        if target.analysis is None:
            return ()
        analysis, workflow, spec, interp, rec, audit = extract_context(target)
    elif isinstance(target, AnalysisResult):
        analysis, workflow, spec, interp, rec, audit = extract_context(target)
    elif hasattr(target, "method_id") and hasattr(target, "values"):
        analysis = target
        spec = getattr(target, "specification", None)
    else:
        return ()

    if (
        analysis.status != "available"
        and getattr(analysis.status, "value", str(analysis.status)) != "available"
    ):
        return ()

    method_id = analysis.method_id
    values = analysis.values or {}
    metadata = analysis.metadata or {}
    conf_level = resolve_confidence_level(analysis, spec)

    # 1. Independent two-group mean comparisons (Welch t, Student t)
    if method_id in ("welch_t", "student_t"):
        diff = values.get("primary_estimate")
        if diff is None or not isinstance(diff, (int, float)) or not math.isfinite(diff):
            return ()
        ci_dict = values.get("confidence_interval")
        lower, upper = _extract_ci(ci_dict)
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
        ci_phrase = confidence_interval_phrase(ci_dict, confidence_level=conf_level)

        return (
            FigureSpec(
                kind="estimate_ci",
                title=f"Mean difference and {ci_phrase}",
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
                placement=PLACEMENT_KEY_RESULTS,
            ),
        )

    # 2. One-sample mean comparison
    if method_id == "one_sample_t":
        diff = values.get("primary_estimate")
        if diff is None or not isinstance(diff, (int, float)) or not math.isfinite(diff):
            return ()
        ci_dict = values.get("confidence_interval")
        lower, upper = _extract_ci(ci_dict)
        outcome = (spec.question.outcome if spec and spec.question else None) or "Sample Mean"
        ref_val = spec.question.reference_value if spec and spec.question else 0.0
        ci_phrase = confidence_interval_phrase(ci_dict, confidence_level=conf_level)

        return (
            FigureSpec(
                kind="estimate_ci",
                title=f"Mean difference from reference and {ci_phrase}",
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
                note=(
                    f"Reference line at 0 indicates sample mean equals reference value ({ref_val})."
                ),
                placement=PLACEMENT_KEY_RESULTS,
            ),
        )

    # 3. Paired-samples t-test
    if method_id == "paired_t":
        diff = values.get("primary_estimate")
        if diff is None or not isinstance(diff, (int, float)) or not math.isfinite(diff):
            return ()
        ci_dict = values.get("confidence_interval")
        lower, upper = _extract_ci(ci_dict)
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

        ci_phrase = confidence_interval_phrase(ci_dict, confidence_level=conf_level)

        return (
            FigureSpec(
                kind="estimate_ci",
                title=f"Paired mean difference and {ci_phrase}",
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
                placement=PLACEMENT_KEY_RESULTS,
            ),
        )

    # 4. Correlation / Bivariate association
    corr_methods = {
        "pearson_correlation": (
            "Pearson correlation",
            "r",
            "Reference line at 0 marks the null value for Pearson r.",
        ),
        "spearman_correlation": (
            "Spearman rank correlation",
            "rho",
            "Reference line at 0 marks the null value for Spearman rho.",
        ),
        "kendall_tau_b": (
            "Kendall's tau-b",
            "tau",
            "Reference line at 0 marks the null value for Kendall tau-b.",
        ),
        "point_biserial_correlation": (
            "Point-biserial correlation",
            "r_pb",
            "Reference line at 0 marks the null value for the point-biserial coefficient.",
        ),
        "point_biserial": (
            "Point-biserial correlation",
            "r_pb",
            "Reference line at 0 marks the null value for the point-biserial coefficient.",
        ),
        "partial_pearson_correlation": (
            "Partial Pearson correlation",
            "r_partial",
            "Reference line at 0 marks the null value for the partial correlation coefficient.",
        ),
        "partial_pearson": (
            "Partial Pearson correlation",
            "r_partial",
            "Reference line at 0 marks the null value for the partial correlation coefficient.",
        ),
    }
    if method_id in corr_methods:
        r_val = values.get("primary_estimate")
        if r_val is None or not isinstance(r_val, (int, float)) or not math.isfinite(r_val):
            return ()
        ci_dict = values.get("confidence_interval")
        lower, upper = _extract_ci(ci_dict)
        outcome = (spec.question.outcome if spec and spec.question else None) or "Outcome"
        predictor = (spec.question.predictor if spec and spec.question else None) or "Predictor"
        pair_label = f"{predictor} & {outcome}"

        method_name, symbol, note_text = corr_methods[method_id]
        ci_phrase = confidence_interval_phrase(ci_dict, confidence_level=conf_level)

        return (
            FigureSpec(
                kind="estimate_ci",
                title=f"{method_name} estimate and {ci_phrase}",
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
                note=note_text,
                placement=PLACEMENT_KEY_RESULTS,
            ),
        )

    # 5. Reliability: ICC and Cronbach alpha
    if method_id in ("icc", "intraclass_correlation"):
        icc_val = first_present(values, "primary_estimate", "icc", "intraclass_correlation")
        if icc_val is None or not isinstance(icc_val, (int, float)) or not math.isfinite(icc_val):
            return ()
        ci_dict = first_present(values, "confidence_interval")
        lower, upper = _extract_ci(ci_dict)
        icc_type = first_present(metadata, "icc_type", "variant", default="ICC")
        ci_phrase = confidence_interval_phrase(ci_dict, confidence_level=conf_level)
        return (
            FigureSpec(
                kind="estimate_ci",
                title=f"Intraclass correlation ({icc_type}) and {ci_phrase}",
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
                note=(
                    "Interpretation depends on the declared ICC model, definition, "
                    "measurement unit, and study context."
                ),
                placement=PLACEMENT_KEY_RESULTS,
            ),
        )

    if method_id == "cronbach_alpha":
        alpha = first_present(values, "primary_estimate", "cronbach_alpha", "raw_alpha")
        if alpha is None or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
            return ()
        ci_dict = first_present(values, "confidence_interval", "raw_alpha_ci")
        lower, upper = _extract_ci(ci_dict)
        ci_phrase = confidence_interval_phrase(ci_dict, confidence_level=conf_level)
        return (
            FigureSpec(
                kind="estimate_ci",
                title=f"Cronbach's alpha estimate and {ci_phrase}",
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
                note="Figure displays the stored reliability estimate and confidence interval.",
                placement=PLACEMENT_KEY_RESULTS,
            ),
        )

    # 6. OLS Linear Regression: Coefficient forest
    if method_id in ("ols_linear_regression", "linear_regression"):
        coefs = values.get("coefficients", [])
        if not isinstance(coefs, list) or not coefs:
            return ()
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
            return ()
        outcome = values.get("outcome") or "Outcome"
        first_ci = next(
            (
                c.get("confidence_interval")
                for c in coefs
                if isinstance(c, dict) and isinstance(c.get("confidence_interval"), dict)
            ),
            None,
        )
        ci_phrase = confidence_interval_phrase(first_ci, confidence_level=conf_level, plural=True)
        return (
            FigureSpec(
                kind="forest",
                title=f"Coefficient estimates and {ci_phrase}",
                subtitle=f"Ordinary Least Squares Regression ({outcome})",
                x_label="Coefficient Estimate",
                y_label="Model Term",
                reference_value=0.0,
                series=tuple(series_list),
                note="Reference line at 0 indicates no linear association.",
                placement=PLACEMENT_COEFFICIENTS,
            ),
        )

    # 7. Logistic Regression: Odds-ratio forest
    if method_id == "logistic_regression":
        coefs = values.get("coefficients", [])
        if not isinstance(coefs, list) or not coefs:
            return ()
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
            return ()
        event_lvl = first_present(values, "event_level", default="1")
        outcome = values.get("outcome") or "Outcome"
        first_ci = next(
            (
                c.get("odds_ratio_ci")
                for c in coefs
                if isinstance(c, dict) and isinstance(c.get("odds_ratio_ci"), dict)
            ),
            None,
        )
        ci_phrase = confidence_interval_phrase(first_ci, confidence_level=conf_level, plural=True)
        return (
            FigureSpec(
                kind="odds_ratio_forest",
                title=f"Predictor odds ratios and {ci_phrase}",
                subtitle=f"Binary Logistic Regression ({outcome}, event='{event_lvl}')",
                x_label="Odds Ratio (log scale)" if all_positive else "Odds Ratio",
                y_label="Predictor",
                reference_value=1.0,
                series=tuple(series_list),
                note="Reference line at 1.0 indicates no change in odds.",
                placement=PLACEMENT_COEFFICIENTS,
                layout_hints={"log_x": all_positive},
            ),
        )

    # 8. ANOVA pairwise comparisons: Pairwise forest
    if method_id in ("welch_anova", "one_way_anova"):
        pairwise = values.get("pairwise_comparisons", [])
        if not isinstance(pairwise, list) or not pairwise:
            return ()
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
            return ()
        method_title = (
            "Games-Howell Post-hoc" if method_id == "welch_anova" else "Tukey-Kramer Post-hoc"
        )
        pairwise_levels: list[float | None] = []
        for p in pairwise:
            if isinstance(p, dict):
                p_ci = p.get("confidence_interval")
                if isinstance(p_ci, dict) and "level" in p_ci:
                    lvl = p_ci.get("level")
                    if isinstance(lvl, (int, float)) and math.isfinite(lvl) and 0 < lvl < 1:
                        pairwise_levels.append(float(lvl))
                    else:
                        pairwise_levels.append(None)
                else:
                    pairwise_levels.append(None)

        if (
            pairwise_levels
            and all(lvl is not None for lvl in pairwise_levels)
            and len(set(pairwise_levels)) == 1
            and pairwise_levels[0] is not None
        ):
            lvl_pct = f"{pairwise_levels[0] * 100:g}% "
            title = f"Pairwise mean differences and {lvl_pct}simultaneous confidence intervals"
        else:
            title = "Pairwise mean differences and simultaneous confidence intervals"

        return (
            FigureSpec(
                kind="pairwise_forest",
                title=title,
                subtitle=method_title,
                x_label="Mean Difference",
                y_label="Contrast",
                reference_value=0.0,
                series=tuple(series_list),
                note="Reference line at 0 indicates equality of population group means.",
                placement=PLACEMENT_PAIRWISE,
            ),
        )

    # Kruskal-Wallis intentionally omits pairwise forest if no authoritative CI is stored
    if method_id == "kruskal_wallis":
        return ()

    # 9. Pearson chi-square: Contingency count heatmap
    if method_id in ("chi_square_independence", "pearson_chi_square"):
        observed = metadata.get("observed_counts")
        groups = metadata.get("group_order", [])
        outcomes = metadata.get("outcome_order", [])
        if not (
            isinstance(observed, list) and groups and outcomes and len(observed) == len(groups)
        ):
            return ()
        series_list = []
        for g_idx, g_name in enumerate(groups):
            row = observed[g_idx]
            if not isinstance(row, (list, tuple)) or len(row) != len(outcomes):
                return ()
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
        return (
            FigureSpec(
                kind="count_heatmap",
                title="Observed contingency counts",
                subtitle=f"{predictor_name} x {outcome_name}",
                x_label=str(outcome_name),
                y_label=str(predictor_name),
                series=tuple(series_list),
                note="Cell values display observed frequencies across categories.",
                placement=PLACEMENT_CONTINGENCY,
            ),
        )

    # 10. Two-way Factorial ANOVA: Observed cell profiles
    if method_id in ("twoway_anova", "two_way_anova"):
        cell_sums = values.get("cell_summaries")
        if not isinstance(cell_sums, list) or not cell_sums:
            return ()
        factor_a = values.get("factor_a") or "Factor A"
        factor_b = values.get("factor_b") or "Factor B"
        levels_a = values.get("factor_a_levels") or []
        levels_b = values.get("factor_b_levels") or []

        def _cell_level(c_dict: dict[str, Any], prefix: str, factor_col: str) -> str:
            for k in (f"level_{prefix}", f"factor_{prefix}_level", f"factor_{prefix}", factor_col):
                if k in c_dict and c_dict[k] is not None:
                    return str(c_dict[k])
            return ""

        if not levels_a or not levels_b:
            # Extract distinct levels preserving source encounter order
            seen_a: list[str] = []
            seen_b: list[str] = []
            for c in cell_sums:
                la = _cell_level(c, "a", factor_a)
                lb = _cell_level(c, "b", factor_b)
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
                        if _cell_level(c, "a", factor_a) == str(a_lvl)
                        and _cell_level(c, "b", factor_b) == str(b_lvl)
                    ),
                    None,
                )
                if match is None or match.get("mean") is None:
                    # Incomplete cell grid; omit figure rather than fabricate
                    return ()
                m_val = match.get("mean")
                if not isinstance(m_val, (int, float)) or not math.isfinite(m_val):
                    return ()
                means.append(float(m_val))

            series_list.append(
                FigureSeries(
                    label=str(b_lvl),
                    categories=tuple(str(a) for a in levels_a),
                    values=tuple(means),
                )
            )

        return (
            FigureSpec(
                kind="cell_profile",
                title="Observed cell means by factor levels",
                subtitle=f"{factor_a} x {factor_b}",
                x_label=str(factor_a),
                y_label="Observed Cell Mean",
                series=tuple(series_list),
                note="Profiles connect observed sample cell means; no CI bands are inferred.",
                placement=PLACEMENT_CELL_SUMMARY,
            ),
        )

    # 11. Fisher exact test: 2x2 contingency count heatmap or OR estimate
    if method_id == "fisher_exact":
        observed = first_present(metadata, "observed_counts", "contingency_table")
        row_order = first_present(metadata, "row_order", "row_labels", default=[])
        col_order = first_present(
            metadata, "column_order", "column_labels", "col_labels", default=[]
        )
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
            return (
                FigureSpec(
                    kind="count_heatmap",
                    title="Observed contingency counts",
                    subtitle="Fisher's Exact Test (2x2)",
                    x_label="Column Category",
                    y_label="Row Category",
                    series=tuple(series_list),
                    note="Cell values display observed frequencies in 2x2 contingency table.",
                    placement=PLACEMENT_CONTINGENCY,
                ),
            )
        # Fallback to OR if stored
        or_val = first_present(values, "primary_estimate", "odds_ratio")
        if or_val is not None and isinstance(or_val, (int, float)) and math.isfinite(or_val):
            ci_dict = first_present(values, "confidence_interval", "odds_ratio_ci")
            low, up = _extract_ci(ci_dict)
            ci_phrase = confidence_interval_phrase(ci_dict, confidence_level=conf_level)
            return (
                FigureSpec(
                    kind="estimate_ci",
                    title=f"Odds ratio estimate and {ci_phrase}",
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
                    placement=PLACEMENT_KEY_RESULTS,
                ),
            )

    return ()


def _build_figure_spec_from_payload(payload: dict[str, Any]) -> FigureSpec | None:
    """Build FigureSpec directly from a ResearchReport canonical payload."""
    sections_obj = payload.get("sections")
    sections_dict: dict[str, Any] = sections_obj if isinstance(sections_obj, dict) else {}
    m_obj = sections_dict.get("methods")
    methods_sec: dict[str, Any] = m_obj if isinstance(m_obj, dict) else {}
    r_obj = sections_dict.get("results")
    results_sec: dict[str, Any] = r_obj if isinstance(r_obj, dict) else {}
    a_obj = payload.get("analysis")
    analysis_dict: dict[str, Any] = a_obj if isinstance(a_obj, dict) else {}
    v_obj = analysis_dict.get("values")
    values_dict: dict[str, Any] = v_obj if isinstance(v_obj, dict) else {}

    method_id = (
        methods_sec.get("method_id") or analysis_dict.get("method_id") or payload.get("method_id")
    )
    if not method_id or not isinstance(method_id, str):
        return None

    conf_level = methods_sec.get("confidence_level") or analysis_dict.get("confidence_level")
    ci = results_sec.get("confidence_interval") or values_dict.get("confidence_interval")
    est = results_sec.get("primary_estimate") or values_dict.get("primary_estimate")
    if est is None or not isinstance(est, (int, float)) or not math.isfinite(est) or ci is None:
        return None

    low, up = _extract_ci(ci)
    ci_phrase = confidence_interval_phrase(ci, confidence_level=conf_level)

    # 1. Independent or paired mean differences -> reference 0.0
    if method_id in ("welch_t", "student_t", "paired_t"):
        contrast_obj = payload.get("sections", {}).get("dataset", {}).get(
            "contrast"
        ) or payload.get("sections", {}).get("research_question", {}).get("condition_order")
        if isinstance(contrast_obj, dict) and "first" in contrast_obj and "second" in contrast_obj:
            contrast_label = f"'{contrast_obj['first']}' - '{contrast_obj['second']}'"
        elif isinstance(contrast_obj, (list, tuple)) and len(contrast_obj) == 2:
            contrast_label = f"'{contrast_obj[0]}' - '{contrast_obj[1]}'"
        elif isinstance(contrast_obj, str) and contrast_obj:
            contrast_label = contrast_obj
        else:
            contrast_label = "Mean difference"

        method_title = (
            "Paired t-test"
            if method_id == "paired_t"
            else ("Welch t-test" if method_id == "welch_t" else "Student t-test")
        )
        title_prefix = "Paired mean difference" if method_id == "paired_t" else "Mean difference"
        note_text = (
            "Reference line at 0 indicates no mean difference across paired conditions."
            if method_id == "paired_t"
            else "Reference line at 0 indicates no difference between groups."
        )

        return FigureSpec(
            kind="estimate_ci",
            title=f"{title_prefix} and {ci_phrase}",
            subtitle=method_title,
            x_label="Mean Difference",
            y_label=None,
            reference_value=0.0,
            series=(
                FigureSeries(
                    label=contrast_label,
                    estimate=float(est),
                    lower=low,
                    upper=up,
                ),
            ),
            note=note_text,
            placement=PLACEMENT_KEY_RESULTS,
        )

    # 2. One-sample mean difference -> reference 0.0
    if method_id == "one_sample_t":
        ref_val = methods_sec.get("reference_value", 0.0)
        outcome = (
            payload.get("sections", {}).get("research_question", {}).get("outcome") or "Sample Mean"
        )
        return FigureSpec(
            kind="estimate_ci",
            title=f"Mean difference from reference and {ci_phrase}",
            subtitle=f"One-Sample t-test (Reference = {ref_val})",
            x_label="Mean Difference",
            y_label=None,
            reference_value=0.0,
            series=(
                FigureSeries(
                    label=f"{outcome} - {ref_val}",
                    estimate=float(est),
                    lower=low,
                    upper=up,
                ),
            ),
            note=f"Reference line at 0 indicates sample mean equals reference value ({ref_val}).",
            placement=PLACEMENT_KEY_RESULTS,
        )

    # 3. Bivariate correlation -> reference 0.0
    corr_notes = {
        "pearson_correlation": (
            "Pearson correlation",
            "r",
            "Reference line at 0 marks the null value for Pearson r.",
        ),
        "spearman_correlation": (
            "Spearman rank correlation",
            "rho",
            "Reference line at 0 marks the null value for Spearman rho.",
        ),
        "kendall_tau_b": (
            "Kendall's tau-b",
            "tau",
            "Reference line at 0 marks the null value for Kendall tau-b.",
        ),
        "point_biserial_correlation": (
            "Point-biserial correlation",
            "r_pb",
            "Reference line at 0 marks the null value for the point-biserial coefficient.",
        ),
        "point_biserial": (
            "Point-biserial correlation",
            "r_pb",
            "Reference line at 0 marks the null value for the point-biserial coefficient.",
        ),
        "partial_pearson_correlation": (
            "Partial Pearson correlation",
            "r_partial",
            "Reference line at 0 marks the null value for the partial correlation coefficient.",
        ),
        "partial_pearson": (
            "Partial Pearson correlation",
            "r_partial",
            "Reference line at 0 marks the null value for the partial correlation coefficient.",
        ),
    }
    if method_id in corr_notes:
        method_name, symbol, note_str = corr_notes[method_id]
        outcome = (
            payload.get("sections", {}).get("research_question", {}).get("outcome") or "Outcome"
        )
        predictor = (
            payload.get("sections", {}).get("research_question", {}).get("predictor") or "Predictor"
        )
        return FigureSpec(
            kind="estimate_ci",
            title=f"{method_name} estimate and {ci_phrase}",
            subtitle=f"{predictor} vs {outcome}",
            x_label=f"Correlation Coefficient ({symbol})",
            y_label=None,
            reference_value=0.0,
            series=(
                FigureSeries(
                    label=f"{predictor} & {outcome}",
                    estimate=float(est),
                    lower=low,
                    upper=up,
                ),
            ),
            note=note_str,
            placement=PLACEMENT_KEY_RESULTS,
        )

    # 4. Odds ratio (Fisher's exact test or logistic regression OR) -> reference 1.0
    if method_id in ("fisher_exact", "logistic_regression"):
        subtitle = (
            "Logistic Regression"
            if method_id == "logistic_regression"
            else "Fisher's Exact Test (2x2)"
        )
        return FigureSpec(
            kind="estimate_ci",
            title=f"Odds ratio estimate and {ci_phrase}",
            subtitle=subtitle,
            x_label="Odds Ratio",
            y_label=None,
            reference_value=1.0,
            series=(
                FigureSeries(
                    label="Odds ratio",
                    estimate=float(est),
                    lower=low,
                    upper=up,
                ),
            ),
            note="Reference line at 1.0 indicates equal odds.",
            placement=PLACEMENT_KEY_RESULTS,
        )

    # 5. Reliability -> no null reference line
    if method_id in ("icc", "intraclass_correlation"):
        variant = results_sec.get("variant") or "ICC"
        return FigureSpec(
            kind="estimate_ci",
            title=f"Intraclass correlation ({variant}) and {ci_phrase}",
            subtitle="Reliability Analysis",
            x_label="ICC Estimate",
            y_label=None,
            reference_value=None,
            series=(
                FigureSeries(
                    label=str(variant),
                    estimate=float(est),
                    lower=low,
                    upper=up,
                ),
            ),
            note=(
                "Interpretation depends on the declared ICC model, definition, "
                "measurement unit, and study context."
            ),
            placement=PLACEMENT_KEY_RESULTS,
        )

    if method_id == "cronbach_alpha":
        return FigureSpec(
            kind="estimate_ci",
            title=f"Cronbach's alpha estimate and {ci_phrase}",
            subtitle="Internal Consistency Reliability",
            x_label="Cronbach's Alpha",
            y_label=None,
            reference_value=None,
            series=(
                FigureSeries(
                    label="Internal Consistency",
                    estimate=float(est),
                    lower=low,
                    upper=up,
                ),
            ),
            note="Figure displays the stored reliability estimate and confidence interval.",
            placement=PLACEMENT_KEY_RESULTS,
        )

    return None


__all__ = [
    "build_figure_spec",
    "build_figure_specs",
]
