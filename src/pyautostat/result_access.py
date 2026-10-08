"""Centralized result-access and deterministic scientific statement formatting.

This internal module provides non-serialized, read-only convenience accessors,
primary result views, tabular DataFrame converters, and deterministic APA-oriented
statistical statement generation.

Zero-recalculation guarantee:
Presentation and convenience layers must never recalculate statistics.
They read stored AnalysisResult values only.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .results import AnalysisResult


AMBIGUOUS_SCALAR_METHODS = {
    "two_way_anova",
    "linear_regression",
    "logistic_regression",
    "cronbach_alpha",
}

STATISTIC_LABELS: dict[str, str] = {
    "welch_t": "t",
    "student_t": "t",
    "one_sample_t": "t",
    "paired_t": "t",
    "mann_whitney_u": "U",
    "wilcoxon_signed_rank": "W",
    "welch_anova": "F",
    "one_way_anova": "F",
    "kruskal_wallis": "H",
    "repeated_measures_anova": "F",
    "friedman_test": "Q",
    "pearson_correlation": "r",
    "spearman_correlation": "r_s",
    "kendall_tau_b": "tau",
    "point_biserial_correlation": "r_pb",
    "partial_pearson_correlation": "r_partial",
    "pearson_chi_square": "chi^2",
    "fisher_exact": "OR",
    "mcnemar": "diff",
    "intraclass_correlation": "F",
}


def get_statistic_label(method_id: str) -> str | None:
    return STATISTIC_LABELS.get(method_id)


def get_statistic(result: AnalysisResult) -> float | None:
    if result.method_id in AMBIGUOUS_SCALAR_METHODS:
        return None
    vals = result.values or {}
    stat = (
        vals.get("f_test", {}).get("statistic")
        if result.method_id == "intraclass_correlation"
        else vals.get("test_statistic")
    )
    return stat if isinstance(stat, (int, float)) and not math.isnan(stat) else None


def get_p_value(result: AnalysisResult) -> float | None:
    if result.method_id in AMBIGUOUS_SCALAR_METHODS:
        return None
    vals = result.values or {}
    pval = (
        vals.get("f_test", {}).get("p_value")
        if result.method_id == "intraclass_correlation"
        else vals.get("p_value")
    )
    return pval if isinstance(pval, (int, float)) and not math.isnan(pval) else None


def get_degrees_of_freedom(result: AnalysisResult) -> int | float | tuple[Any, ...] | None:
    mid = result.method_id
    vals = result.values or {}
    if mid in ("two_way_anova", "linear_regression", "logistic_regression", "cronbach_alpha"):
        return None
    df = vals.get("degrees_of_freedom")
    if isinstance(df, (list, tuple)) and len(df) == 2:
        return (df[0], df[1])
    if isinstance(df, (int, float)) and not math.isnan(df):
        return df
    if mid == "intraclass_correlation":
        f_t = vals.get("f_test", {})
        if isinstance(f_t, Mapping) and "df1" in f_t and "df2" in f_t:
            return (f_t["df1"], f_t["df2"])
    return None


def get_estimate(result: AnalysisResult) -> float | None:
    est = (result.values or {}).get("primary_estimate")
    return est if isinstance(est, (int, float)) and not math.isnan(est) else None


def get_confidence_interval(result: AnalysisResult) -> dict[str, Any] | None:
    ci = (result.values or {}).get("confidence_interval")
    return copy.deepcopy(ci) if isinstance(ci, dict) else None


def get_effect_size(result: AnalysisResult) -> dict[str, Any] | None:
    eff = (result.values or {}).get("effect_size")
    return copy.deepcopy(eff) if isinstance(eff, dict) else None


def get_effect_size_ci(result: AnalysisResult) -> dict[str, Any] | None:
    vals = result.values or {}
    eff = vals.get("effect_size")
    ci = eff.get("confidence_interval") if isinstance(eff, Mapping) else None
    if ci is None:
        ci = vals.get("effect_size_confidence_interval")
    return copy.deepcopy(ci) if isinstance(ci, dict) else None


def get_sample_accounting(result: AnalysisResult) -> dict[str, Any]:
    meta = result.metadata or {}
    s_meta = meta.get("sample", {}) if isinstance(meta.get("sample"), Mapping) else {}
    vals = result.values or {}
    acc: dict[str, Any] = {}
    if result.sample_size is not None:
        acc["sample_size"] = result.sample_size
    if result.excluded_rows is not None:
        acc["excluded_rows"] = result.excluded_rows

    tracked_keys = (
        "original_rows",
        "analyzed_rows",
        "excluded_rows",
        "sample_size",
        "group_sizes",
        "total_units",
        "complete_pairs",
        "complete_units",
        "incomplete_units",
        "excluded_units",
        "missing_unit_rows",
        "nonzero_differences",
        "zero_differences",
        "effective_pair_count",
        "effective_n",
        "targets",
        "n_targets",
        "raters",
        "n_raters",
    )
    for k in tracked_keys:
        v = s_meta.get(k)
        if v is None:
            v = meta.get(k)
        if v is None and isinstance(vals, Mapping):
            v = vals.get(k)
        if v is not None and k not in acc:
            acc[k] = copy.deepcopy(v)
    return acc


def get_primary_result(result: AnalysisResult) -> dict[str, Any]:
    mid = result.method_id
    vals = result.values or {}
    if getattr(result, "status", None) and result.status.value != "available":
        return {
            "method_id": mid,
            "status": result.status.value,
            "reason": "Analysis result is not available",
        }

    base: dict[str, Any] = {
        "method_id": mid,
        "test_name": result.method_label,
        "statistic": get_statistic(result),
        "statistic_label": get_statistic_label(mid),
        "degrees_of_freedom": get_degrees_of_freedom(result),
        "p_value": get_p_value(result),
        "estimate": get_estimate(result),
        "estimate_label": vals.get("estimate_name"),
        "confidence_interval": get_confidence_interval(result),
        "effect_size": get_effect_size(result),
        "effect_size_confidence_interval": get_effect_size_ci(result),
        "sample_accounting": get_sample_accounting(result),
    }

    if mid == "two_way_anova":
        base["terms"] = copy.deepcopy(vals.get("terms", {}))
        base["cell_summaries"] = copy.deepcopy(vals.get("cell_summaries", {}))
        base["estimated_marginal_means"] = copy.deepcopy(vals.get("estimated_marginal_means", {}))
        base["followups"] = copy.deepcopy(vals.get("followups", {}))
    elif mid in ("linear_regression", "logistic_regression"):
        base["model_fit"] = copy.deepcopy(vals.get("model_fit", {}))
        base["coefficients"] = copy.deepcopy(vals.get("coefficients", []))
        base["diagnostics"] = copy.deepcopy(vals.get("diagnostics", {}))
    elif mid in (
        "welch_anova",
        "one_way_anova",
        "kruskal_wallis",
        "repeated_measures_anova",
        "friedman_test",
    ):
        if "pairwise_comparisons" in vals:
            base["pairwise_comparisons"] = copy.deepcopy(vals["pairwise_comparisons"])
        gs = vals.get("group_summaries") or vals.get("condition_summaries")
        if gs is not None:
            base["group_summaries"] = copy.deepcopy(gs)
    elif mid == "intraclass_correlation":
        base["variant"] = vals.get("variant")
        base["notation"] = vals.get("notation")
        base["all_variants"] = copy.deepcopy(vals.get("all_variants", []))

    return base


def result_to_dataframe(result: AnalysisResult, section: str = "primary") -> pd.DataFrame:
    sec = section.strip().lower()
    mid = result.method_id
    vals = result.values or {}

    if sec == "primary":
        data: dict[str, Any] = {
            "method_id": [mid],
            "test_name": [result.method_label],
            "statistic": [get_statistic(result)],
            "degrees_of_freedom": [get_degrees_of_freedom(result)],
            "p_value": [get_p_value(result)],
            "estimate": [get_estimate(result)],
            "sample_size": [result.sample_size],
        }
        eff = get_effect_size(result)
        if eff:
            data["effect_size_name"] = [eff.get("name")]
            data["effect_size_value"] = [eff.get("value")]
        return pd.DataFrame(data)

    if sec == "coefficients" and mid in ("linear_regression", "logistic_regression"):
        return pd.DataFrame(copy.deepcopy(vals.get("coefficients", [])))
    if sec == "terms" and mid == "two_way_anova":
        terms = vals.get("terms", [])
        return (
            pd.DataFrame(list(terms.values()))
            if isinstance(terms, dict)
            else pd.DataFrame(copy.deepcopy(terms))
        )
    if sec == "comparisons" and "pairwise_comparisons" in vals:
        return pd.DataFrame(copy.deepcopy(vals["pairwise_comparisons"]))

    valid = ["primary"]
    if mid in ("linear_regression", "logistic_regression"):
        valid.append("coefficients")
    if mid == "two_way_anova":
        valid.append("terms")
    if "pairwise_comparisons" in vals:
        valid.append("comparisons")
    avail = ", ".join(valid)
    raise ValueError(
        f"Unsupported section '{section}' for method '{mid}'. Available sections: {avail}."
    )


def format_apa_number(val: Any, bounded_unit: bool = False, decimals: int = 2) -> str:
    if val is None or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
        return "Unavailable"
    txt = f"{val:.{decimals}f}"
    return txt.replace("0.", ".", 1) if bounded_unit and abs(val) < 1.0 else txt


def format_apa_p_value(p: Any) -> str:
    if p is None or not isinstance(p, (int, float)) or math.isnan(p) or math.isinf(p):
        return "p = Unavailable"
    if p < 0.001:
        return "p < .001"
    txt = f"{p:.3f}".replace("0.", ".", 1)
    return f"p = {txt}"


def format_apa_ci(ci: Mapping[str, Any] | None, bounded_unit: bool = False) -> str:
    if not isinstance(ci, Mapping):
        return ""
    low, upp = ci.get("lower"), ci.get("upper")
    if low is None or upp is None:
        return ""
    lvl = ci.get("level", 0.95)
    lvl_str = f"{int(lvl * 100)}%" if isinstance(lvl, (int, float)) else "95%"
    l_str = format_apa_number(low, bounded_unit)
    u_str = format_apa_number(upp, bounded_unit)
    return f"{lvl_str} CI [{l_str}, {u_str}]"


def generate_statement(result: AnalysisResult, style: str = "apa") -> str:
    mid = result.method_id
    vals = result.values or {}
    t_name = result.method_label
    if style.lower() != "apa":
        s_val = get_statistic(result)
        p_val = get_p_value(result)
        e_val = get_estimate(result)
        return f"{t_name}: statistic = {s_val}, p = {p_val}, estimate = {e_val}."

    p_str = format_apa_p_value(get_p_value(result))
    ci_str = format_apa_ci(vals.get("confidence_interval"))
    eff = vals.get("effect_size")
    eff_val = format_apa_number(eff.get("value")) if isinstance(eff, Mapping) else "Unavailable"
    eff_name = eff.get("name", "effect") if isinstance(eff, Mapping) else "effect"

    if mid == "two_way_anova":
        ss_type = vals.get("sum_of_squares_type", "II")
        terms = vals.get("terms", [])
        if isinstance(terms, Mapping):
            terms = list(terms.values())
        resid_term = next((t for t in terms if t.get("term_type") == "residual"), None)
        df_resid = resid_term.get("df") if resid_term else None
        df_resid_str = (
            f"{df_resid:.1f}"
            if isinstance(df_resid, float) and not df_resid.is_integer()
            else str(int(df_resid))
            if isinstance(df_resid, (int, float))
            else None
        )
        lines = [f"{t_name} terms (Type {ss_type} SS):"]
        for t in terms:
            if t.get("term_type") == "residual":
                continue
            df_num = t.get("df")
            df_num_str = (
                f"{df_num:.1f}"
                if isinstance(df_num, float) and not df_num.is_integer()
                else str(int(df_num))
                if isinstance(df_num, (int, float))
                else "Unavailable"
            )
            df_s = f"{df_num_str}, {df_resid_str}" if df_resid_str else df_num_str
            eff_entry = t.get("effect_size")
            eta_val = eff_entry.get("value") if isinstance(eff_entry, Mapping) else None
            eta_s = format_apa_number(eta_val, bounded_unit=True)
            eta_clause = f", partial eta^2 = {eta_s}" if eta_s != "Unavailable" else ""
            t_f = format_apa_number(t.get("f_statistic"))
            t_p = format_apa_p_value(t.get("p_value"))
            lines.append(f"  - {t.get('term', 'term')}: F({df_s}) = {t_f}, {t_p}{eta_clause}.")
        return "\n".join(lines)

    if mid in ("linear_regression", "logistic_regression"):
        fit = vals.get("model_fit", {})
        is_log = mid == "logistic_regression"
        if is_log:
            df_lr = fit.get("lr_degrees_of_freedom") or fit.get("model_degrees_of_freedom")
            df_str = str(int(df_lr)) if df_lr is not None else "Unavailable"
            pr2 = format_apa_number(fit.get("mcfadden_r2"), bounded_unit=True)
            f_s = format_apa_number(fit.get("lr_statistic"))
            p_s = format_apa_p_value(fit.get("lr_p_value"))
            event_clause = ""
            if vals.get("event_level") is not None:
                event_clause = f" (event: {vals.get('event_level')})"
            lines = [
                f"Logistic regression model fit{event_clause}: LR chi^2({df_str}) = {f_s}, {p_s}, "
                f"McFadden pseudo-R^2 = {pr2}."
            ]
        else:
            cov = " (HC3 robust standard errors)" if vals.get("covariance_type") == "HC3" else ""
            df_m, df_r = fit.get("model_degrees_of_freedom"), fit.get("residual_degrees_of_freedom")
            df_str = f"{int(df_m)}, {int(df_r)}" if df_m and df_r else "Unavailable"
            r2 = format_apa_number(fit.get("r_squared"), bounded_unit=True)
            ar2 = format_apa_number(fit.get("adjusted_r_squared"), bounded_unit=True)
            f_s = format_apa_number(fit.get("model_f_statistic"))
            p_s = format_apa_p_value(fit.get("model_f_p_value"))
            lines = [
                f"Linear regression model fit{cov}: F({df_str}) = {f_s}, {p_s}, "
                f"R^2 = {r2}, adjusted R^2 = {ar2}."
            ]

        for c in vals.get("coefficients", []):
            ci_c = format_apa_ci(c.get("odds_ratio_ci") if is_log else c.get("confidence_interval"))
            ci_part = f", {ci_c}" if ci_c else ""
            b_s = format_apa_number(c.get("estimate"))
            se_s = format_apa_number(c.get("standard_error"))
            stat_name = "z" if is_log else "t"
            stat_s = format_apa_number(c.get("statistic"))
            cp_s = format_apa_p_value(c.get("p_value"))
            or_part = f", OR = {format_apa_number(c.get('odds_ratio'))}" if is_log else ""
            lines.append(
                f"  - {c.get('term', 'term')}: B = {b_s}, SE = {se_s}, "
                f"{stat_name} = {stat_s}, {cp_s}{or_part}{ci_part}."
            )
        return "\n".join(lines)

    return _format_canonical_statement(result, mid, vals, t_name, p_str, ci_str, eff_name, eff_val)


def _format_canonical_statement(
    result: AnalysisResult,
    mid: str,
    vals: Mapping[str, Any],
    t_name: str,
    p_str: str,
    ci_str: str,
    eff_name: str,
    eff_val: str,
) -> str:
    if mid in ("welch_t", "student_t", "one_sample_t", "paired_t"):
        df = get_degrees_of_freedom(result)
        df_str = (
            f"{df:.1f}"
            if isinstance(df, float) and not df.is_integer()
            else str(int(df))
            if isinstance(df, (int, float))
            else "Unavailable"
        )
        eff_part = f", {eff_name} = {eff_val}" if eff_val != "Unavailable" else ""
        ci_part = f", {ci_str} for the mean difference." if ci_str else "."
        pref = "Welch's independent-samples t-test" if mid == "welch_t" else t_name
        stat_s = format_apa_number(vals.get("test_statistic"))
        return f"{pref}: t({df_str}) = {stat_s}, {p_str}{eff_part}{ci_part}"

    if mid == "mann_whitney_u":
        u_s = format_apa_number(vals.get("test_statistic"))
        r_s = format_apa_number(
            vals.get("effect_size", {}).get("value")
            if isinstance(vals.get("effect_size"), Mapping)
            else None,
            bounded_unit=True,
        )
        return f"Mann-Whitney U test: U = {u_s}, {p_str}, rank-biserial r = {r_s}."

    if mid == "wilcoxon_signed_rank":
        w_s = format_apa_number(vals.get("test_statistic"))
        r_s = format_apa_number(
            vals.get("effect_size", {}).get("value")
            if isinstance(vals.get("effect_size"), Mapping)
            else None,
            bounded_unit=True,
        )
        return (
            f"Wilcoxon signed-rank test: W = {w_s}, {p_str}, matched-pairs rank-biserial r = {r_s}."
        )

    if mid in ("welch_anova", "one_way_anova"):
        dfs = get_degrees_of_freedom(result)
        if isinstance(dfs, (tuple, list)) and len(dfs) == 2:
            d1, d2 = dfs[0], dfs[1]
            d1_str = f"{d1:.1f}" if isinstance(d1, float) and not d1.is_integer() else str(int(d1))
            d2_str = f"{d2:.1f}" if isinstance(d2, float) and not d2.is_integer() else str(int(d2))
            df_str = f"{d1_str}, {d2_str}"
        else:
            df_str = "Unavailable"
        eff_part = f", {eff_name} = {eff_val}" if eff_val != "Unavailable" else ""
        f_s = format_apa_number(vals.get("test_statistic"))
        return f"{t_name}: omnibus F({df_str}) = {f_s}, {p_str}{eff_part}."

    if mid == "kruskal_wallis":
        df_k = get_degrees_of_freedom(result)
        h_df = str(int(df_k)) if isinstance(df_k, (int, float)) else "Unavailable"
        h_s = format_apa_number(vals.get("test_statistic"))
        return f"Kruskal-Wallis test: H({h_df}) = {h_s}, {p_str}, epsilon^2 = {eff_val}."

    if mid == "repeated_measures_anova":
        dfs = get_degrees_of_freedom(result)
        if isinstance(dfs, (tuple, list)) and len(dfs) == 2:
            d1, d2 = dfs[0], dfs[1]
            d1_str = f"{d1:.1f}" if isinstance(d1, float) and not d1.is_integer() else str(int(d1))
            d2_str = f"{d2:.1f}" if isinstance(d2, float) and not d2.is_integer() else str(int(d2))
            df_str = f"{d1_str}, {d2_str}"
        else:
            df_str = "Unavailable"
        eff_part = f", partial eta^2 = {eff_val}" if eff_val != "Unavailable" else ""
        eps = format_apa_number(vals.get("greenhouse_geisser_epsilon"), bounded_unit=True)
        gg_applied = (
            vals.get("sphericity_correction_applied")
            or vals.get("primary_inference") == "Greenhouse-Geisser"
        )
        gg = f" (Greenhouse-Geisser epsilon = {eps} applied)" if gg_applied else ""
        f_s = format_apa_number(vals.get("test_statistic"))
        return f"Repeated-measures ANOVA: F({df_str}) = {f_s}, {p_str}{eff_part}{gg}."

    if mid == "friedman_test":
        df_f = get_degrees_of_freedom(result)
        df_str = str(int(df_f)) if isinstance(df_f, (int, float)) else "Unavailable"
        eff_part = f", {eff_name} = {eff_val}" if eff_val != "Unavailable" else ""
        q_s = format_apa_number(vals.get("test_statistic"))
        return f"Friedman test: Q({df_str}) = {q_s}, {p_str}{eff_part}."

    if mid in (
        "pearson_correlation",
        "spearman_correlation",
        "kendall_tau_b",
        "point_biserial_correlation",
        "partial_pearson_correlation",
    ):
        df = get_degrees_of_freedom(result)
        df_str = f"({int(df)})" if isinstance(df, (int, float)) else ""
        sym = {
            "pearson_correlation": "r",
            "spearman_correlation": "r_s",
            "kendall_tau_b": "tau",
            "point_biserial_correlation": "r_pb",
        }.get(mid, "r_partial")
        r_s = format_apa_number(vals.get("primary_estimate"), bounded_unit=True)
        ci_c = format_apa_ci(vals.get("confidence_interval"), bounded_unit=True)
        ci_part = f", {ci_c}." if ci_c else "."
        return f"{t_name}: {sym}{df_str} = {r_s}, {p_str}{ci_part}"

    if mid == "pearson_chi_square":
        df_c = get_degrees_of_freedom(result)
        chi_df = str(int(df_c)) if isinstance(df_c, (int, float)) else "Unavailable"
        n_s = result.sample_size
        chi_s = format_apa_number(vals.get("test_statistic"))
        v_s = format_apa_number(vals.get("primary_estimate"), bounded_unit=True)
        return (
            f"Pearson's chi-square test: chi^2({chi_df}, N = {n_s}) = {chi_s}, "
            f"{p_str}, Cramer's V = {v_s}."
        )

    if mid == "fisher_exact":
        or_s = format_apa_number(vals.get("primary_estimate"))
        ci_part = f", {ci_str}." if ci_str else "."
        return f"Fisher's exact test: {p_str}, sample OR = {or_s}{ci_part}"

    if mid == "mcnemar":
        diff_s = format_apa_number(vals.get("primary_estimate"), bounded_unit=True)
        ci_c = format_apa_ci(vals.get("confidence_interval"), bounded_unit=True)
        ci_part = f", {ci_c}." if ci_c else "."
        return f"Exact McNemar test: {p_str}, paired proportion difference = {diff_s}{ci_part}"

    if mid == "cronbach_alpha":
        k_items = vals.get("item_count") or len(vals.get("items", []))
        a_s = format_apa_number(vals.get("primary_estimate"), bounded_unit=True)
        ci_part = f", {ci_str}." if ci_str else "."
        return (
            f"Cronbach's alpha: alpha = {a_s}, k = {k_items} items, "
            f"N = {result.sample_size} respondents{ci_part}"
        )

    if mid == "intraclass_correlation":
        dfs = get_degrees_of_freedom(result)
        if isinstance(dfs, (tuple, list)) and len(dfs) == 2:
            df_s = f"{int(dfs[0])}, {int(dfs[1])}"
        else:
            df_s = "Unavailable"
        icc_s = format_apa_number(vals.get("primary_estimate"), bounded_unit=True)
        f_s = format_apa_number(get_statistic(result))
        fp_s = format_apa_p_value(get_p_value(result))
        ci_part = f", {ci_str}" if ci_str else ""
        notat = vals.get("notation", "ICC")
        desc = vals.get("description", "intraclass correlation")
        return (
            f"Intraclass correlation {notat} ({desc}): ICC = {icc_s}{ci_part}, "
            f"F({df_s}) = {f_s}, {fp_s}."
        )

    stat_v = get_statistic(result)
    p_v = get_p_value(result)
    est_v = get_estimate(result)
    return f"{t_name}: statistic = {stat_v}, p = {p_v}, estimate = {est_v}."


def generate_workflow_statement(workflow: Any) -> str:
    analysis = getattr(workflow, "analysis", None)
    if analysis is None:
        blockers = getattr(workflow, "blockers", ())
        fallback_msg = "Workflow has not completed statistical execution."
        blocker_msg = "; ".join(blockers) if blockers else fallback_msg
        return f"Statement unavailable: {blocker_msg}"
    return generate_statement(analysis, style="apa")
