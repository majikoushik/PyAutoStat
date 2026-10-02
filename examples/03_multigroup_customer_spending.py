"""Example 03: Multi-Group Customer Spending Comparison.

Business Question:
    "Does average monthly spending differ across customer job categories?"

Scientific Focus:
    Demonstrates guided multi-group mean inference across 3+ independent groups.
    Welch one-way ANOVA handles unequal group variances, followed by Games-Howell
    simultaneous pairwise comparisons for multiplicity control without silent estimand drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from repository root or examples directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import (
    DATA_DICTIONARY,
    format_currency,
    format_number,
    load_customer_data,
    section,
    subsection,
)

from pyautostat import AnalysisOptions, ResearchAssistant

RANDOM_SEED = 42


def main() -> None:
    section("BUSINESS QUESTION")
    print("Question : Does average monthly spending differ across customer job categories?")
    print("Estimand : Population group means and pairwise mean differences")
    print("Design   : Independent groups (6 job categories across 5,000 customers)")

    # 1. Load cleaned customer data (customer IDs excluded)
    frame = load_customer_data(include_customer_id=False)
    assistant = ResearchAssistant(frame)

    # 2. Run guided workflow targeting the declared population mean
    workflow = assistant.run(
        objective="compare_groups",
        outcome="total_avg_monthly_spend",
        predictor="job_category",
        estimand="mean",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )

    if workflow.analysis is None or workflow.interpretation is None:
        raise RuntimeError(f"Workflow execution failed: {workflow.blockers}")

    analysis = workflow.analysis
    values = analysis.values

    section("PYAUTOSTAT DECISION")
    print(f"Selected Method: {analysis.method_label} ({analysis.method_id})")
    print(f"Rows Analyzed  : {analysis.sample_size:,}")
    print(f"Rows Excluded  : {analysis.excluded_rows}")
    print(f"Groups (k = 6) : {', '.join(values.get('groups', []))}")

    subsection("Recommendation Rationale")
    if workflow.recommendation:
        print(workflow.recommendation.rationale_text.strip())

    section("RESULT: OMNIBUS TEST")
    f_stat = values.get("test_statistic")
    p_val = values.get("p_value")
    df = values.get("degrees_of_freedom")
    effect = values.get("effect_size") or {}

    print(f"Welch F Statistic       : {format_number(f_stat)}")
    print(f"Degrees of Freedom (df) : {format_number(df)}")
    print(f"p-value                 : {format_number(p_val)}")
    eff_name = effect.get("name", "omega-squared")
    print(
        f"Effect Size ({eff_name})  : {format_number(effect.get('value'))} "
        f"({effect.get('interpretation', 'small')})"
    )

    subsection("Group Summaries")
    for group_info in values.get("group_summaries", []):
        g_name = group_info["group"]
        g_n = group_info["sample_size"]
        g_mean = group_info["mean"]
        g_std = group_info["standard_deviation"]
        g_med = group_info["median"]
        print(
            f"  - {g_name:<14}: N={g_n:>5}, Mean=${g_mean:>7.2f}, "
            f"SD=${g_std:>7.2f}, Median=${g_med:>7.2f}"
        )

    subsection("Games-Howell Pairwise Follow-Ups (Sample of Comparisons)")
    pairwise = values.get("pairwise_comparisons", [])
    print(f"Total pairwise comparisons evaluated: {len(pairwise)}")
    print(
        f"  {'Contrast':<28} | {'Diff ($)':<10} | {'95% Simultaneous CI':<24} | "
        f"{'Adj p-val':<12} | Decision"
    )
    print("  " + "-" * 88)
    for comp in pairwise[:6]:
        label = f"{comp['group1']} vs {comp['group2']}"
        diff_val = comp["estimate"]
        ci_dict = comp.get("confidence_interval") or {}
        ci_str = f"[{ci_dict.get('lower', 0):.2f}, {ci_dict.get('upper', 0):.2f}]"
        adj_p = format_number(comp.get("adjusted_p_value"))
        decision = comp.get("decision", "fail_to_reject")
        print(f"  {label:<28} | {diff_val:>10.2f} | {ci_str:<24} | {adj_p:<12} | {decision}")

    section("INTERPRETATION")
    print(workflow.explain())

    section("WHAT THIS MEANS")
    print(
        "- The omnibus Welch ANOVA tests whether all 6 population means are equal without "
        "assuming equal variances."
    )
    print(
        "- Pairwise Games-Howell adjustments protect the familywise error rate across all 15 "
        "contrasts."
    )
    summaries = values.get("group_summaries", [])
    if summaries:
        sorted_groups = sorted(summaries, key=lambda g: g.get("mean", 0.0))
        lowest = sorted_groups[0]
        highest = sorted_groups[-1]
        low_grp = lowest["group"]
        low_m = format_currency(lowest["mean"])
        hi_grp = highest["group"]
        hi_m = format_currency(highest["mean"])
        print(
            f"- Across the {len(summaries)} categories, {low_grp} employees show the lowest "
            f"observed average spend ({low_m}/month), whereas {hi_grp} shows "
            f"the highest ({hi_m}/month)."
        )
        print(
            "- Simultaneous Games-Howell adjustments identify which specific category pairs differ "
            "reliably after familywise error control."
        )

    section("WHAT THIS DOES NOT MEAN")
    print("- Job category is an observational classification, not an experimental assignment.")
    print("- Spending differences across job categories do NOT mean job category causes spending.")
    print(
        "- Failing to find significant differences between certain pairs does NOT prove those "
        "means are equivalent."
    )
    print(
        "\nNext step: Run 'python examples/04_customer_value_association.py' for categorical "
        "association."
    )


if __name__ == "__main__":
    main()
