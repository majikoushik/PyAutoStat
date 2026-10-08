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

# Every registered statistical method has one deliberately small brief layout.
# Keeping this explicit makes additions fail visibly instead of silently inheriting
# an ill-fitting statistical summary.
BRIEF_METHOD_CATEGORIES: dict[str, str] = {
    "welch_t": "mean_comparison",
    "student_t": "mean_comparison",
    "one_sample_t": "mean_comparison",
    "paired_t": "mean_comparison",
    "mann_whitney_u": "rank_comparison",
    "wilcoxon_signed_rank": "rank_comparison",
    "welch_anova": "multigroup",
    "one_way_anova": "multigroup",
    "kruskal_wallis": "multigroup",
    "repeated_measures_anova": "repeated_measures",
    "friedman_test": "repeated_measures",
    "pearson_correlation": "association",
    "spearman_correlation": "association",
    "kendall_tau_b": "association",
    "point_biserial_correlation": "association",
    "partial_pearson_correlation": "association",
    "pearson_chi_square": "categorical",
    "fisher_exact": "categorical",
    "mcnemar": "categorical",
    "two_way_anova": "factorial",
    "linear_regression": "linear_model",
    "logistic_regression": "logistic_model",
    "cronbach_alpha": "reliability",
    "intraclass_correlation": "reliability",
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
        gg_map = vals.get("greenhouse_geisser")
        gg_eps = (
            gg_map.get("epsilon")
            if isinstance(gg_map, Mapping)
            else vals.get("greenhouse_geisser_epsilon")
        )
        eps = format_apa_number(gg_eps, bounded_unit=True)
        primary_inf = str(vals.get("primary_inference", "")).lower()
        gg_applied = (
            (isinstance(gg_map, Mapping) and gg_map.get("applied") is True)
            or vals.get("sphericity_correction_applied") is True
            or primary_inf in ("greenhouse_geisser", "greenhouse-geisser")
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


def _brief_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _brief_number(value: Any, *, bounded_unit: bool = False) -> str | None:
    if not _brief_finite(value):
        return None
    return format_apa_number(value, bounded_unit=bounded_unit)


def _brief_df(value: Any) -> str | None:
    if not _brief_finite(value):
        return None
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:.2f}"


def _brief_p(value: Any) -> str | None:
    return format_apa_p_value(value) if _brief_finite(value) else None


def _brief_ci(value: Any, *, bounded_unit: bool = False) -> str | None:
    if not isinstance(value, Mapping):
        return None
    if value.get("status") == "unavailable":
        return None
    if not _brief_finite(value.get("lower")) or not _brief_finite(value.get("upper")):
        return None
    return format_apa_ci(value, bounded_unit=bounded_unit) or None


def _brief_text(value: Any) -> str:
    """Flatten stored display text so a brief always remains one paragraph."""
    return " ".join(str(value).split())


def _brief_name(value: Any) -> str:
    return _brief_text(value).replace("_", " ")


def _brief_unavailable(result: AnalysisResult, label: str) -> str:
    reason = (result.values or {}).get("reason")
    if reason is None and result.warnings:
        reason = result.warnings[0]
    suffix = f" ({_brief_text(reason).rstrip('.')})" if reason else ""
    return f"{label} not available{suffix}"


def _brief_orientation(result: AnalysisResult) -> str | None:
    metadata = result.metadata or {}
    contrast = metadata.get("contrast")
    if isinstance(contrast, Mapping):
        first, second = contrast.get("first"), contrast.get("second")
        if first is not None and second is not None:
            return f"{_brief_text(first)} minus {_brief_text(second)}"
    order = metadata.get("condition_order") or metadata.get("group_order")
    if isinstance(order, (list, tuple)) and len(order) == 2:
        return f"{_brief_text(order[0])} minus {_brief_text(order[1])}"
    return None


def _brief_sample(result: AnalysisResult) -> str | None:
    accounting = get_sample_accounting(result)
    parts: list[str] = []
    if accounting.get("sample_size") is not None:
        parts.append(f"N = {accounting['sample_size']}")
    if accounting.get("excluded_rows") is not None:
        parts.append(f"excluded rows = {accounting['excluded_rows']}")
    for key, label in (
        ("complete_pairs", "complete pairs"),
        ("complete_units", "complete units"),
        ("incomplete_units", "incomplete units"),
        ("excluded_units", "excluded units"),
        ("missing_unit_rows", "missing-unit rows"),
        ("nonzero_differences", "nonzero differences"),
        ("n_targets", "targets"),
        ("n_raters", "raters"),
    ):
        if accounting.get(key) is not None:
            parts.append(f"{label} = {accounting[key]}")
    return ", ".join(parts) if parts else None


def _brief_effect(values: Mapping[str, Any]) -> str | None:
    effect = values.get("effect_size")
    if not isinstance(effect, Mapping) or effect.get("status") == "not_applicable":
        return None
    number = _brief_number(effect.get("value"), bounded_unit=True)
    if number is None:
        return None
    return f"{_brief_name(effect.get('name', 'effect'))} = {number}"


def _brief_pairwise_count(values: Mapping[str, Any], metadata: Mapping[str, Any]) -> str | None:
    comparisons = values.get("pairwise_comparisons")
    if isinstance(comparisons, (list, tuple)):
        return f"pairwise comparisons recorded = {len(comparisons)}"
    count = metadata.get("pairwise_comparison_count")
    return f"pairwise comparisons recorded = {count}" if count is not None else None


def _brief_join(label: str, clauses: list[str | None]) -> str:
    available = [clause for clause in clauses if clause]
    return f"{_brief_text(label)}: " + "; ".join(available) + "."


def _brief_simple(result: AnalysisResult) -> str:
    values = result.values or {}
    orientation = _brief_orientation(result)
    estimate = _brief_number(values.get("primary_estimate"))
    estimate_name = _brief_name(values.get("estimate_name", "estimate"))
    if estimate is None:
        estimate_clause = _brief_unavailable(result, estimate_name)
    else:
        estimate_clause = f"{estimate_name} = {estimate}"
        if orientation:
            estimate_clause += f" ({orientation})"
    ci = _brief_ci(values.get("confidence_interval"))
    statistic = _brief_number(values.get("test_statistic"))
    df = _brief_df(values.get("degrees_of_freedom"))
    test_clause = f"t({df}) = {statistic}" if df and statistic else None
    p_clause = _brief_p(values.get("p_value"))
    reference = values.get("reference_value")
    reference_clause = None
    if result.method_id == "one_sample_t" and reference is not None:
        reference_clause = f"reference = {_brief_text(reference)}"
    return _brief_join(
        result.method_label,
        [
            estimate_clause,
            reference_clause,
            ci,
            test_clause,
            p_clause,
            _brief_effect(values),
            _brief_sample(result),
        ],
    )


def _brief_rank(result: AnalysisResult) -> str:
    values = result.values or {}
    symbol = "U" if result.method_id == "mann_whitney_u" else "W"
    statistic = _brief_number(values.get("test_statistic"))
    stat_clause = f"{symbol} = {statistic}" if statistic else f"{symbol} not available"
    orientation = _brief_orientation(result)
    orientation_clause = f"orientation = {orientation}" if orientation else None
    return _brief_join(
        result.method_label,
        [
            stat_clause,
            _brief_p(values.get("p_value")),
            _brief_effect(values),
            orientation_clause,
            _brief_sample(result),
        ],
    )


def _brief_multigroup(result: AnalysisResult, *, repeated: bool = False) -> str:
    values, metadata = result.values or {}, result.metadata or {}
    raw_df = values.get("degrees_of_freedom")
    dfs = raw_df if isinstance(raw_df, (list, tuple)) else (raw_df,)
    df_text = ", ".join(filter(None, (_brief_df(value) for value in dfs)))
    symbol = (
        "H"
        if result.method_id == "kruskal_wallis"
        else "Q"
        if result.method_id == "friedman_test"
        else "F"
    )
    statistic = _brief_number(values.get("test_statistic"))
    stat_clause = f"{symbol}({df_text}) = {statistic}" if df_text and statistic else None
    clauses: list[str | None] = [
        stat_clause,
        _brief_p(values.get("p_value")),
        _brief_effect(values),
    ]
    gg = values.get("greenhouse_geisser")
    if repeated and isinstance(gg, Mapping) and gg.get("applied") is True:
        epsilon = _brief_number(gg.get("epsilon"), bounded_unit=True)
        clauses.append(
            "Greenhouse-Geisser correction applied" + (f" (epsilon = {epsilon})" if epsilon else "")
        )
    clauses.extend([_brief_pairwise_count(values, metadata), _brief_sample(result)])
    return _brief_join(result.method_label, clauses)


def _brief_association(result: AnalysisResult) -> str:
    values, metadata = result.values or {}, result.metadata or {}
    label = {
        "pearson_correlation": "Pearson linear correlation",
        "spearman_correlation": "Spearman rank correlation",
        "kendall_tau_b": "Kendall's tau-b correlation",
    }.get(result.method_id, result.method_label)
    symbol = {
        "pearson_correlation": "r",
        "spearman_correlation": "r_s",
        "kendall_tau_b": "tau-b",
        "point_biserial_correlation": "r_pb",
        "partial_pearson_correlation": "partial r",
    }[result.method_id]
    estimate = _brief_number(values.get("primary_estimate"), bounded_unit=True)
    df = _brief_df(values.get("degrees_of_freedom"))
    estimate_clause = (
        f"{symbol}{f'({df})' if df else ''} = {estimate}"
        if estimate
        else _brief_unavailable(result, symbol)
    )
    orientation_clause = None
    if (
        result.method_id == "point_biserial_correlation"
        and metadata.get("positive_level") is not None
    ):
        orientation_clause = f"positive level = {_brief_text(metadata['positive_level'])}"
    elif result.method_id == "partial_pearson_correlation":
        controls = values.get("controls") or metadata.get("controls")
        if isinstance(controls, (list, tuple)):
            orientation_clause = "controls = " + ", ".join(_brief_text(item) for item in controls)
    return _brief_join(
        label,
        [
            estimate_clause,
            _brief_ci(values.get("confidence_interval"), bounded_unit=True),
            _brief_p(values.get("p_value")),
            orientation_clause,
            _brief_sample(result),
        ],
    )


def _brief_categorical(result: AnalysisResult) -> str:
    values, metadata = result.values or {}, result.metadata or {}
    mid = result.method_id
    if mid == "pearson_chi_square":
        df = _brief_df(values.get("degrees_of_freedom"))
        statistic = _brief_number(values.get("test_statistic"))
        estimate = _brief_number(values.get("primary_estimate"), bounded_unit=True)
        clauses = [
            f"chi^2({df}) = {statistic}" if df and statistic else None,
            _brief_p(values.get("p_value")),
            f"Cramer's V = {estimate}" if estimate else "Cramer's V not available",
            _brief_sample(result),
        ]
    elif mid == "fisher_exact":
        odds_ratio = _brief_number(values.get("primary_estimate"))
        ci = _brief_ci(values.get("confidence_interval"))
        p_value = _brief_p(values.get("p_value"))
        clauses = [
            f"unconditional sample OR = {odds_ratio}"
            if odds_ratio
            else _brief_unavailable(result, "unconditional sample OR"),
            f"asymptotic log-Wald {ci}" if ci else "sample-OR interval not available",
            f"exact two-sided {p_value}" if p_value else None,
            _brief_sample(result),
        ]
    else:
        estimate = _brief_number(values.get("primary_estimate"), bounded_unit=True)
        order = values.get("condition_order") or metadata.get("condition_order")
        event = (
            values.get("event_level")
            if values.get("event_level") is not None
            else metadata.get("event_level")
        )
        orientation = None
        if isinstance(order, (list, tuple)) and len(order) == 2:
            orientation = f"{_brief_text(order[0])} minus {_brief_text(order[1])}"
        clauses = [
            f"paired proportion difference = {estimate}"
            if estimate
            else _brief_unavailable(result, "paired proportion difference"),
            f"orientation = {orientation}" if orientation else None,
            f"event = {_brief_text(event)}" if event is not None else None,
            _brief_ci(values.get("confidence_interval"), bounded_unit=True),
            _brief_p(values.get("p_value")),
            _brief_sample(result),
        ]
    label = "Exact McNemar test" if mid == "mcnemar" else result.method_label
    return _brief_join(label, clauses)


def _brief_factorial(result: AnalysisResult) -> str:
    values = result.values or {}
    terms = values.get("terms", [])
    term_rows = (
        list(terms.values())
        if isinstance(terms, Mapping)
        else list(terms)
        if isinstance(terms, (list, tuple))
        else []
    )
    residual = next(
        (
            row
            for row in term_rows
            if isinstance(row, Mapping) and row.get("term_type") == "residual"
        ),
        None,
    )
    residual_df = _brief_df(residual.get("df")) if isinstance(residual, Mapping) else None
    clauses: list[str | None] = []
    for term in term_rows:
        if not isinstance(term, Mapping) or term.get("term_type") == "residual":
            continue
        term_df = _brief_df(term.get("df"))
        statistic = _brief_number(term.get("f_statistic"))
        effect = term.get("effect_size")
        eta = (
            _brief_number(effect.get("value"), bounded_unit=True)
            if isinstance(effect, Mapping)
            else None
        )
        detail = (
            f"{_brief_text(term.get('term', 'term'))}: F({term_df}, {residual_df}) = {statistic}"
        )
        p_value = _brief_p(term.get("p_value"))
        if p_value:
            detail += f", {p_value}"
        if eta:
            detail += f", partial eta^2 = {eta}"
        clauses.append(detail)
    ss_type = _brief_text(values.get("sum_of_squares_type", "stored"))
    ss_type = {
        "type2": "Type II",
        "type3": "Type III",
        "2": "Type II",
        "3": "Type III",
    }.get(ss_type.lower(), ss_type)
    return _brief_join(
        f"{result.method_label} ({ss_type} sums of squares)",
        [*clauses, _brief_sample(result)],
    )


def _brief_nonintercept_count(values: Mapping[str, Any]) -> int:
    coefficients = values.get("coefficients", [])
    if not isinstance(coefficients, (list, tuple)):
        return 0
    return sum(
        1
        for coefficient in coefficients
        if isinstance(coefficient, Mapping)
        and coefficient.get("term_type") != "intercept"
        and str(coefficient.get("term", "")).lower() not in {"intercept", "const"}
    )


def _brief_model(result: AnalysisResult, *, logistic: bool) -> str:
    values = result.values or {}
    fit = values.get("model_fit", {})
    fit = fit if isinstance(fit, Mapping) else {}
    coefficient_count = _brief_nonintercept_count(values)
    if logistic:
        df = _brief_df(fit.get("lr_degrees_of_freedom"))
        statistic = _brief_number(fit.get("lr_statistic"))
        pseudo_r2 = _brief_number(fit.get("mcfadden_r2"), bounded_unit=True)
        event = values.get("event_level")
        clauses = [
            f"event = {_brief_text(event)}" if event is not None else None,
            f"LR chi^2({df}) = {statistic}" if df and statistic else None,
            _brief_p(fit.get("lr_p_value")),
            f"McFadden pseudo-R^2 = {pseudo_r2}" if pseudo_r2 else None,
            f"non-intercept coefficients = {coefficient_count}",
            _brief_sample(result),
        ]
    else:
        df_model = _brief_df(fit.get("model_degrees_of_freedom"))
        df_residual = _brief_df(fit.get("residual_degrees_of_freedom"))
        statistic = _brief_number(fit.get("model_f_statistic"))
        r_squared = _brief_number(fit.get("r_squared"), bounded_unit=True)
        adjusted = _brief_number(fit.get("adjusted_r_squared"), bounded_unit=True)
        covariance = values.get("covariance_type") or fit.get("covariance_type")
        clauses = [
            f"covariance = {_brief_text(covariance)}" if covariance is not None else None,
            f"F({df_model}, {df_residual}) = {statistic}"
            if df_model and df_residual and statistic
            else None,
            _brief_p(fit.get("model_f_p_value")),
            f"R^2 = {r_squared}" if r_squared else None,
            f"adjusted R^2 = {adjusted}" if adjusted else None,
            f"non-intercept coefficients = {coefficient_count}",
            _brief_sample(result),
        ]
    return _brief_join(result.method_label, clauses)


def _brief_reliability(result: AnalysisResult) -> str:
    values = result.values or {}
    if result.method_id == "cronbach_alpha":
        alpha = _brief_number(values.get("primary_estimate"), bounded_unit=True)
        item_count = values.get("item_count")
        clauses = [
            f"alpha = {alpha}" if alpha else _brief_unavailable(result, "alpha"),
            f"items = {item_count}" if item_count is not None else None,
            _brief_ci(values.get("confidence_interval"), bounded_unit=True),
            _brief_sample(result),
        ]
    else:
        icc = _brief_number(values.get("primary_estimate"), bounded_unit=True)
        f_test = values.get("f_test", {})
        f_test = f_test if isinstance(f_test, Mapping) else {}
        notation = _brief_text(values.get("notation", "ICC"))
        definition = _brief_name(values.get("definition", ""))
        model = _brief_name(values.get("model", ""))
        unit = _brief_name(values.get("unit", ""))
        df1, df2 = _brief_df(f_test.get("df1")), _brief_df(f_test.get("df2"))
        statistic = _brief_number(f_test.get("statistic"))
        descriptor = ", ".join(item for item in (model, definition, unit) if item)
        clauses = [
            f"{notation} = {icc}" if icc else _brief_unavailable(result, notation),
            descriptor or None,
            _brief_ci(values.get("confidence_interval"), bounded_unit=True),
            f"F({df1}, {df2}) = {statistic}" if df1 and df2 and statistic else None,
            _brief_p(f_test.get("p_value")),
            _brief_sample(result),
        ]
    return _brief_join(result.method_label, clauses)


def _format_completed_brief(result: AnalysisResult) -> str:
    category = BRIEF_METHOD_CATEGORIES.get(result.method_id)
    if category == "mean_comparison":
        return _brief_simple(result)
    if category == "rank_comparison":
        return _brief_rank(result)
    if category == "multigroup":
        return _brief_multigroup(result)
    if category == "repeated_measures":
        return _brief_multigroup(result, repeated=True)
    if category == "association":
        return _brief_association(result)
    if category == "categorical":
        return _brief_categorical(result)
    if category == "factorial":
        return _brief_factorial(result)
    if category == "linear_model":
        return _brief_model(result, logistic=False)
    if category == "logistic_model":
        return _brief_model(result, logistic=True)
    if category == "reliability":
        return _brief_reliability(result)
    return _brief_join(
        result.method_label or result.method_id,
        ["structured result available", _brief_sample(result), "use explain() for details"],
    )


def _with_brief_warning(workflow: Any, text: str, *, include_warnings: bool) -> str:
    warnings = getattr(workflow, "warnings", ())
    if not include_warnings or not warnings:
        return text
    first = _brief_text(warnings[0]).rstrip(".")
    remainder = len(warnings) - 1
    suffix = f" (+{remainder} more)" if remainder else ""
    return f"{text.rstrip('.')}; Warning: {first}{suffix}."


def generate_workflow_brief(workflow: Any, *, include_warnings: bool = True) -> str:
    """Return one deterministic paragraph made only from stored workflow records."""
    raw_status = getattr(workflow, "status", None)
    status = getattr(raw_status, "value", raw_status)
    blockers = getattr(workflow, "blockers", ())
    analysis = getattr(workflow, "analysis", None)

    if status == "needs_input":
        missing: tuple[Any, ...] = tuple(getattr(workflow, "missing_information", ()))
        if missing:
            first = missing[0]
            remainder = len(missing) - 1
            more = (
                f" (+{remainder} more required field{'s' if remainder != 1 else ''})"
                if remainder
                else ""
            )
            text = f"Input needed — {_brief_text(first.field)}: {_brief_text(first.message)}{more}."
        else:
            text = "Input needed — required workflow information is missing."
    elif status == "data_limited":
        reason = (
            _brief_text(blockers[0]) if blockers else "the recorded data do not support analysis"
        )
        text = f"Data limited — {reason} No statistical analysis was completed."
    elif status == "unsupported":
        reason = _brief_text(blockers[0]) if blockers else "the requested design is unsupported"
        text = f"Unsupported workflow — {reason} No substitute method was run."
    elif status == "failed":
        reason = _brief_text(blockers[0]) if blockers else "the recorded workflow failed"
        if analysis is not None:
            text = (
                f"Workflow failed after analysis — {reason} "
                f"Computed method: {_brief_text(analysis.method_label)}."
            )
        else:
            text = f"Workflow failed — {reason}"
    elif analysis is None:
        text = "Workflow has not completed statistical execution."
    else:
        text = _format_completed_brief(analysis)
        if status == "partial":
            text = f"Partial workflow — {text}"
    return _with_brief_warning(workflow, text, include_warnings=include_warnings)


def generate_workflow_statement(workflow: Any) -> str:
    analysis = getattr(workflow, "analysis", None)
    if analysis is None:
        blockers = getattr(workflow, "blockers", ())
        fallback_msg = "Workflow has not completed statistical execution."
        blocker_msg = "; ".join(blockers) if blockers else fallback_msg
        return f"Statement unavailable: {blocker_msg}"
    return generate_statement(analysis, style="apa")
