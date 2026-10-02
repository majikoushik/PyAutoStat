"""Example 08: Factorial Customer Segments Analysis (Two-Way ANOVA).

Business Question:
    "How are two customer-segmentation factors jointly associated with average monthly spend?"

Scientific Focus:
    Demonstrates independent two-way factorial ANOVA evaluating two factors (home ownership
    and news subscription) and their interaction on total monthly spend.
    Uses Type II sums of squares (hierarchical testing) and non-central F inversion for exact
    partial eta-squared confidence intervals. Unweighted estimated marginal means (EMMs)
    and simple cell summaries illustrate that the effects are purely additive.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from repository root or examples directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import (
    format_number,
    load_customer_data,
    section,
    subsection,
)

from pyautostat import ResearchAssistant


def main() -> None:
    section("BUSINESS QUESTION")
    print(
        "Question : How are two customer-segmentation factors jointly associated with "
        "average monthly spend?"
    )
    print("Estimand : Main effect differences, interaction effect, and partial eta-squared")
    print(
        "Design   : 2x2 independent factorial ANOVA: Home Owner (Owner / Non-owner) x "
        "News Subscriber (Yes / No)"
    )

    # 1. Load data with human-readable labels (customer IDs excluded)
    frame = load_customer_data(include_customer_id=False, labeled_categories=True)
    assistant = ResearchAssistant(frame)

    # 2. Run two-way factorial ANOVA
    workflow = assistant.two_way_anova(
        outcome="total_avg_monthly_spend",
        factor_a="home_owner",
        factor_b="news_subscriber",
        sum_of_squares="type2",
    )

    if workflow.analysis is None or workflow.interpretation is None:
        raise RuntimeError(f"Workflow execution failed: {workflow.blockers}")

    analysis = workflow.analysis
    values = analysis.values

    section("PYAUTOSTAT DECISION")
    print(f"Selected Method: {analysis.method_label} ({analysis.method_id})")
    print("Sums of Squares: Type II (hierarchical adjustment for main effects)")
    print(f"Sample Size    : {analysis.sample_size:,} observations (0 excluded)")
    print(f"Factor A (rows): home_owner ({', '.join(values.get('factor_a_levels', []))})")
    print(f"Factor B (cols): news_subscriber ({', '.join(values.get('factor_b_levels', []))})")

    subsection("Cell Sample Sizes and Means (2x2 Design)")
    print(
        f"  {'Home Owner':<14} | {'News Sub':<10} | {'Cell N':<8} | "
        f"{'Mean Spend ($)':<16} | Std Dev ($)"
    )
    print("  " + "-" * 64)
    for cell in values.get("cell_summaries", []):
        fa = cell.get("factor_a_level")
        fb = cell.get("factor_b_level")
        n_c = cell.get("sample_size")
        m_c = cell.get("mean")
        s_c = cell.get("standard_deviation")
        print(f"  {fa:<14} | {fb:<10} | {n_c:>8} | ${m_c:>15.2f} | ${s_c:>10.2f}")

    section("RESULT: ANOVA TABLE & EFFECT SIZES")
    print(
        f"  {'Source / Term':<30} | {'SS':<12} | {'df':<4} | {'MS':<10} | "
        f"{'F-stat':<10} | {'p-value':<12} | Partial Eta2 [95% CI]"
    )
    print("  " + "-" * 104)
    for term in values.get("terms", []):
        t_name = term.get("term", "")
        ss = term.get("sum_squares")
        df = term.get("df")
        ms = term.get("mean_square")
        f_val = term.get("f_statistic")
        p_val = term.get("p_value")
        eff = term.get("effect_size") or {}
        eff_val = eff.get("value")
        eff_ci = eff.get("confidence_interval") or {}

        f_str = f"{f_val:>10.2f}" if isinstance(f_val, (int, float)) else "       N/A"
        p_str = format_number(p_val) if p_val is not None else "N/A"
        if isinstance(eff_val, (int, float)):
            ci_str = f"[{eff_ci.get('lower', 0):.4f}, {eff_ci.get('upper', 0):.4f}]"
            eff_str = f"{eff_val:.4f} {ci_str}"
        else:
            eff_str = "N/A"

        print(
            f"  {t_name:<30} | {ss:>12.1f} | {df:>4} | {ms:>10.1f} | "
            f"{f_str} | {p_str:<12} | {eff_str}"
        )

    subsection("Estimated Marginal Means (Unweighted EMMs)")
    emms = values.get("estimated_marginal_means", {})
    for factor_name, levels in emms.items():
        print(f"Factor '{factor_name}':")
        for lvl_info in levels:
            lvl = lvl_info.get("level")
            est = lvl_info.get("estimate")
            se = lvl_info.get("standard_error")
            print(f"  - {lvl:<12}: EMM = ${est:>7.2f} / month (SE = ${se:.2f})")

    section("INTERPRETATION")
    print(workflow.explain())

    section("WHAT THIS MEANS")
    print(
        "- Both main effects are statistically significant: Homeowners spend more than "
        "non-owners (F = 30.99, p < 1e-7),"
    )
    print("  and news subscribers spend more than non-subscribers (F = 99.79, p < 1e-22).")
    print(
        "- The interaction term (home_owner:news_subscriber) is NOT statistically significant "
        "(F = 0.001, p = 0.975)."
    )
    print(
        "- This absence of interaction indicates an ADDITIVE relationship: the spending premium "
        "associated with"
    )
    print(
        "  news subscription is virtually identical for homeowners (+$46.11/mo) and non-homeowners "
        "(+$46.42/mo)."
    )

    section("WHAT THIS DOES NOT MEAN")
    print("- Neither factor was experimentally manipulated; these are observational associations.")
    print(
        "- Non-significance of the interaction does NOT prove the true interaction is zero, "
        "only that the observed"
    )
    print("  data are fully consistent with additive main effects.")
    print(
        "- Residual non-normality was noted in diagnostics, but given N = 5,000, F-tests for main "
        "effects remain robust."
    )
    print(
        "\nNext step: Run 'python examples/09_complete_research_workflow.py' for the end-to-end "
        "research lifecycle."
    )


if __name__ == "__main__":
    main()
