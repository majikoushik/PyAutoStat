"""Adapters for linear and logistic regression models."""

from __future__ import annotations

from ...results import AnalysisResult
from ...workflow import ResearchWorkflowResult
from ..formatting import (
    format_confidence_interval,
    format_confidence_level_label,
    format_number,
    format_odds_ratio,
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


def adapt_linear_regression(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt an OLS linear regression workflow."""
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    conf_level = resolve_confidence_level(analysis, spec)
    outcome = analysis.values.get("outcome") or "Outcome"
    predictors = analysis.values.get("predictors") or []
    cov_type = analysis.values.get("covariance_type") or "nonrobust"

    sample_size = analysis.sample_size
    excluded_rows = first_present(analysis, "excluded_rows", default=0)

    pred_summary = ", ".join(str(p) for p in predictors[:4])
    if len(predictors) > 4:
        pred_summary += "..."
    design_metrics = (
        DisplayMetric("Method", "Ordinary Least Squares (OLS) Regression", role="method"),
        DisplayMetric("Outcome", str(outcome)),
        DisplayMetric("Predictors", f"{len(predictors)} predictors ({pred_summary})"),
        DisplayMetric("Covariance Type", str(cov_type)),
        DisplayMetric("Sample", f"{format_sample_size(sample_size)} observations"),
        DisplayMetric("Excluded", f"{format_sample_size(excluded_rows)} rows"),
    )

    fit_obj = analysis.values.get("model_fit")
    fit = fit_obj if isinstance(fit_obj, dict) else {}
    r2 = fit.get("r_squared")
    adj_r2 = fit.get("adjusted_r_squared")
    r2_ci = fit.get("r_squared_confidence_interval")
    f_stat = fit.get("model_f_statistic")
    f_df = [fit.get("model_degrees_of_freedom"), fit.get("residual_degrees_of_freedom")]
    f_p = fit.get("model_f_p_value")
    resid_se = fit.get("residual_standard_error")

    r2_str = format_number(r2, decimals=4)
    adj_r2_str = format_number(adj_r2, decimals=4)
    ci_str = format_confidence_interval(r2_ci, decimals=4)
    r2_ci_label = format_confidence_level_label(
        r2_ci, confidence_level=conf_level, prefix="R-squared"
    )
    f_str = format_statistic("F", f_stat, df=f_df)
    f_p_str = format_p_value(f_p)

    key_metrics_list = [
        DisplayMetric("R-squared", r2_str, role="result.estimate"),
        DisplayMetric("Adjusted R-squared", adj_r2_str, role="result.estimate"),
        DisplayMetric(r2_ci_label, ci_str, role="result.ci"),
        DisplayMetric("Model F-test", f_str, role="result.evidence"),
        DisplayMetric("Model p-value", f_p_str, role="result.evidence"),
    ]
    if resid_se is not None:
        key_metrics_list.append(DisplayMetric("Residual SE", format_number(resid_se, decimals=3)))

    # Coefficients table
    tables: list[DisplayTable] = []
    raw_coefs = analysis.values.get("coefficients", [])
    if isinstance(raw_coefs, list) and raw_coefs:
        coef_ci_sample = (
            raw_coefs[0].get("confidence_interval") if isinstance(raw_coefs[0], dict) else None
        )
        coef_ci_label = format_confidence_level_label(coef_ci_sample, confidence_level=conf_level)
        cols = ("Term", "Estimate", "Std Error", coef_ci_label, "t", "p-value")
        rows: list[DisplayRow] = []
        for c in raw_coefs:
            term = str(c.get("term", c.get("term_label", c.get("name", ""))))
            est = format_number(c.get("estimate"), decimals=3)
            se = format_number(c.get("standard_error"), decimals=3)
            ci = format_confidence_interval(c.get("confidence_interval"), decimals=3)
            t_val = format_number(c.get("statistic"), decimals=2)
            p_val = format_p_value(c.get("p_value"))
            rows.append(DisplayRow((term, est, se, ci, t_val, p_val)))
        tables.append(DisplayTable(title="MODEL COEFFICIENTS", columns=cols, rows=tuple(rows)))

    # Diagnostics
    diagnostics_list: list[DisplayDiagnostic] = []
    diag_data = analysis.values.get("diagnostics", {})
    if isinstance(diag_data, dict):
        bp = diag_data.get("breusch_pagan", {})
        if isinstance(bp, dict) and bp:
            bp_status_raw = bp.get("status")
            bp_status = "REVIEW" if bp_status_raw == "rejected" else "DOCUMENTED"
            bp_lm_p = first_present(bp, "lm_p_value", "p_value")
            bp_policy = bp.get(
                "interpretation_policy", "diagnostic evidence only; covariance was not changed"
            )
            bp_detail = (
                f"LM stat = {format_number(bp.get('lm_statistic'), decimals=2)}, "
                f"p = {format_p_value(bp_lm_p)}; {bp_policy}."
            )
            diagnostics_list.append(
                DisplayDiagnostic(
                    label="Breusch-Pagan (Heteroskedasticity)",
                    status=bp_status,
                    detail=bp_detail,
                    severity="review" if bp_status_raw == "rejected" else "neutral",
                )
            )

        vif_data = diag_data.get("vif", {})
        if isinstance(vif_data, dict) and vif_data:
            max_vif = vif_data.get("maximum")
            vif_terms = vif_data.get("terms", [])
            has_signal = any(
                isinstance(t, dict)
                and t.get("advisory")
                in ("elevated_collinearity_signal", "strong_collinearity_signal", "nonfinite")
                for t in vif_terms
            )
            vif_status = "REVIEW" if has_signal else "DOCUMENTED"
            policy_text = vif_data.get(
                "threshold_policy",
                "VIF values around 5 or 10 are review heuristics, not pass/fail rules.",
            )
            vif_str = format_number(max_vif, decimals=2)
            vif_detail = f"Maximum VIF is {vif_str}. {policy_text}"
            diagnostics_list.append(
                DisplayDiagnostic(
                    label="Collinearity (VIF)",
                    status=vif_status,
                    detail=vif_detail,
                    severity="review" if has_signal else "neutral",
                )
            )

        norm = diag_data.get("residual_normality", {})
        if isinstance(norm, dict) and norm:
            norm_status_raw = norm.get("status")
            norm_status = "REVIEW" if norm_status_raw == "rejected" else "DOCUMENTED"
            norm_detail = (
                f"{norm.get('method', 'Residual test')} statistic = "
                f"{format_number(norm.get('statistic'), decimals=2)}, "
                f"p = {format_p_value(norm.get('p_value'))}."
            )
            diagnostics_list.append(
                DisplayDiagnostic(
                    label="Residual Normality",
                    status=norm_status,
                    detail=norm_detail,
                    severity="review" if norm_status_raw == "rejected" else "neutral",
                )
            )

        influence = diag_data.get("influence", {})
        if isinstance(influence, dict) and influence:
            inf_status_raw = influence.get("status")
            inf_status = "REVIEW" if inf_status_raw == "review" else "DOCUMENTED"
            flagged = influence.get("flagged_count", 0)
            policy = influence.get(
                "interpretation_policy",
                "Thresholds are review heuristics; observations are never removed automatically.",
            )
            inf_detail = f"{flagged} observations flagged by review heuristics. {policy}"
            diagnostics_list.append(
                DisplayDiagnostic(
                    label="Influence Diagnostics",
                    status=inf_status,
                    detail=inf_detail,
                    severity="review" if inf_status_raw == "review" else "neutral",
                )
            )

    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("statistical associations" in lim for lim in limitations):
        limitations.append(
            "Regression coefficients reflect statistical associations in the fitted model; "
            "they do not establish causal relationships."
        )
    if not any("predictive validity" in lim for lim in limitations):
        limitations.append(
            "In-sample fit (R-squared) does not establish out-of-sample predictive validity."
        )

    compact_text = (
        f"OLS Regression | N={format_sample_size(sample_size)} | "
        f"R2={r2_str} (adj={adj_r2_str}) | {f_str} | p={f_p_str}"
    )

    metadata = build_metadata_dict(analysis, workflow)
    reg_spec_obj = analysis.metadata.get("regression_specification")
    metadata["reference_levels"] = (
        reg_spec_obj.get("reference_levels") if isinstance(reg_spec_obj, dict) else None
    )

    return TerminalView(
        title="Linear Regression Model (OLS)",
        subtitle=f"{outcome} ~ {len(predictors)} predictors",
        family="family.regression",
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


def adapt_logistic_regression(
    target: ResearchWorkflowResult | AnalysisResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a binary logistic regression workflow."""
    analysis, workflow, spec, interp, rec, audit = extract_context(target)
    conf_level = resolve_confidence_level(analysis, spec)
    outcome = analysis.values.get("outcome") or "Outcome"
    predictors = analysis.values.get("predictors") or []
    event_lvl = first_present(analysis.values, "event_level", default="1")
    non_event_lvl = first_present(analysis.values, "non_event_level", default="0")
    event_count = analysis.values.get("event_count")
    event_rate = analysis.values.get("event_rate")

    sample_size = analysis.sample_size
    excluded_rows = first_present(analysis, "excluded_rows", default=0)

    ev_cnt_str = format_sample_size(event_count)
    ev_pct_str = format_percent(event_rate * 100 if event_rate is not None else None)
    design_metrics = (
        DisplayMetric("Method", "Binary Logistic Regression (Logit)", role="method"),
        DisplayMetric("Modeled Event", f"'{event_lvl}' (vs '{non_event_lvl}')"),
        DisplayMetric("Event Accounting", f"{ev_cnt_str} events ({ev_pct_str})"),
        DisplayMetric("Predictors", f"{len(predictors)} predictors"),
        DisplayMetric("Sample", f"{format_sample_size(sample_size)} observations"),
        DisplayMetric("Excluded", f"{format_sample_size(excluded_rows)} rows"),
    )

    fit_obj = analysis.values.get("model_fit")
    fit = fit_obj if isinstance(fit_obj, dict) else {}
    lr_stat = fit.get("lr_statistic")
    lr_df = fit.get("lr_degrees_of_freedom")
    lr_p = fit.get("lr_p_value")
    mcfadden = fit.get("mcfadden_r2")
    aic = fit.get("aic")

    mcfadden_str = format_number(mcfadden, decimals=4)
    lr_str = format_statistic("LR chi-square", lr_stat, df=lr_df)
    lr_p_str = format_p_value(lr_p)

    key_metrics_list = [
        DisplayMetric("Likelihood Ratio Test", lr_str, role="result.evidence"),
        DisplayMetric("Model p-value", lr_p_str, role="result.evidence"),
        DisplayMetric("McFadden pseudo-R2", mcfadden_str, role="result.estimate"),
    ]
    if aic is not None:
        key_metrics_list.append(DisplayMetric("AIC", format_number(aic, decimals=1)))

    # Coefficients table: OR-first in standard mode!
    tables: list[DisplayTable] = []
    raw_coefs = analysis.values.get("coefficients", [])
    if isinstance(raw_coefs, list) and raw_coefs:
        or_ci_sample = raw_coefs[0].get("odds_ratio_ci") if isinstance(raw_coefs[0], dict) else None
        wald_ci_label = format_confidence_level_label(
            or_ci_sample, confidence_level=conf_level, suffix="Wald CI"
        )
        cols: tuple[str, ...]
        if detail == "full":
            cols = ("Term", "Odds Ratio", wald_ci_label, "raw beta", "SE", "z", "p-value")
            rows = []
            for c in raw_coefs:
                term = str(c.get("term", c.get("term_label", c.get("name", ""))))
                or_val = format_odds_ratio(c.get("odds_ratio"))
                or_ci = format_confidence_interval(c.get("odds_ratio_ci"), decimals=3)
                beta = format_number(c.get("estimate"), decimals=3)
                se = format_number(c.get("standard_error"), decimals=3)
                z_val = format_number(c.get("statistic"), decimals=2)
                p_val = format_p_value(c.get("p_value"))
                rows.append(DisplayRow((term, or_val, or_ci, beta, se, z_val, p_val)))
        else:
            cols = ("Term", "Odds Ratio", wald_ci_label, "z", "p-value")
            rows = []
            for c in raw_coefs:
                term = str(c.get("term", c.get("term_label", c.get("name", ""))))
                or_val = format_odds_ratio(c.get("odds_ratio"))
                or_ci = format_confidence_interval(c.get("odds_ratio_ci"), decimals=3)
                z_val = format_number(c.get("statistic"), decimals=2)
                p_val = format_p_value(c.get("p_value"))
                rows.append(DisplayRow((term, or_val, or_ci, z_val, p_val)))
        tables.append(DisplayTable(title="PREDICTOR ODDS RATIOS", columns=cols, rows=tuple(rows)))

    diagnostics_list = [
        DisplayDiagnostic(
            label="Event Orientation",
            status="PRESERVED",
            detail=f"Modeling event '{event_lvl}' relative to baseline '{non_event_lvl}'.",
            severity="neutral",
        )
    ]
    diagnostics_list.extend(extract_diagnostics(analysis))

    interpretation_text = format_interpretation_text(interp, detail=detail)
    limitations = list(interp.limitations) if interp and hasattr(interp, "limitations") else []
    if not any("multiplicative factor" in lim for lim in limitations):
        limitations.append(
            "Odds ratio is a multiplicative factor on the odds; "
            "it is not a constant probability difference."
        )
    if not any("statistical association" in lim for lim in limitations):
        limitations.append(
            "Logistic regression coefficients reflect statistical association, not causal impact."
        )

    compact_text = (
        f"Logistic Regression | N={format_sample_size(sample_size)} | "
        f"Event='{event_lvl}' | McFadden R2={mcfadden_str} | {lr_str} | p={lr_p_str}"
    )

    metadata = build_metadata_dict(analysis, workflow)
    reg_spec_obj = analysis.metadata.get("regression_specification")
    metadata["reference_levels"] = (
        reg_spec_obj.get("reference_levels") if isinstance(reg_spec_obj, dict) else None
    )

    return TerminalView(
        title="Logistic Regression Model (Logit)",
        subtitle=f"P({outcome} == '{event_lvl}') ~ {len(predictors)} predictors",
        family="family.regression",
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
