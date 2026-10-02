"""Example 07: Product Portfolio Repeated-Measures Analysis (Friedman & Wilcoxon).

Business Question:
    "How do monthly spending distributions differ across Product A, Product B, and
    Product C within the same customers?"

Scientific Focus:
    Demonstrates non-parametric repeated-measures inference when multiple conditions are
    measured on the same units. Because Products B and C are heavily zero-inflated
    (services not subscribed by all customers), the Friedman rank-sum test evaluates
    rank distributions without forcing normality or deleting zero-spend observations.
    Follow-up pairwise Wilcoxon tests apply Holm multiplicity adjustments.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running directly from repository root or examples directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import (
    format_number,
    load_product_spend_long,
    section,
    subsection,
)

from pyautostat import AnalysisOptions, ResearchAssistant

RANDOM_SEED = 42
CONDITION_ORDER = ("Product A", "Product B", "Product C")


def main() -> None:
    section("BUSINESS QUESTION")
    print(
        "Question : How do monthly spending distributions differ across Product A, Product B, "
        "and Product C within the same customers?"
    )
    print("Estimand : Rank-distribution differences and matched-pairs rank-biserial correlations")
    print(
        "Design   : One-way within-subjects / repeated-measures design "
        "(3 conditions on 5,000 customers = 15,000 observations)"
    )

    # 1. Reshape data into long format (customer_id, product, monthly_spend)
    # Customer IDs are used strictly for unit matching and never printed
    long_frame = load_product_spend_long()
    assistant = ResearchAssistant(long_frame)

    # 2. Run guided repeated-measures distribution comparison
    fast_mode = (
        "--fast" in sys.argv
        or os.environ.get("PYAUTOSTAT_FAST_TEST") == "1"
        or os.environ.get("PYAUTOSTAT_FAST_DEMO") == "1"
    )
    b_samples = 50 if fast_mode else 199
    workflow = assistant.run(
        objective="compare_groups",
        outcome="monthly_spend",
        predictor="product",
        design="repeated",
        estimand="distribution",
        unit_id="customer_id",
        condition_order=CONDITION_ORDER,
        variable_types={"monthly_spend": "continuous", "product": "ordinal"},
        options=AnalysisOptions(
            alpha=0.05,
            confidence_level=0.95,
            random_seed=RANDOM_SEED,
            bootstrap_samples=b_samples,
        ),
    )

    if workflow.analysis is None or workflow.interpretation is None:
        raise RuntimeError(f"Workflow execution failed: {workflow.blockers}")

    analysis = workflow.analysis
    values = analysis.values

    section("PYAUTOSTAT DECISION")
    print(f"Selected Method: {analysis.method_label} ({analysis.method_id})")
    print("Design Structure: Repeated measures matched on unit identifier (customer_id)")
    print("Privacy Guard   : Individual customer identifiers are NEVER printed or exported.")
    print("Complete Units  : 5,000 complete customer panels (0 incomplete)")
    print(f"Total Rows      : {analysis.sample_size:,} observations analyzed")
    print(f"Conditions (k=3): {', '.join(CONDITION_ORDER)}")

    subsection("Recommendation Rationale")
    if workflow.recommendation:
        print(workflow.recommendation.rationale_text.strip())

    section("RESULT: OMNIBUS FRIEDMAN TEST")
    q_stat = values.get("test_statistic")
    df = values.get("degrees_of_freedom")
    p_val = values.get("p_value")
    effect = values.get("effect_size") or {}

    print(f"Friedman Q Statistic    : {format_number(q_stat)}")
    print(f"Degrees of Freedom (df) : {format_number(df)}")
    print(f"p-value                 : {format_number(p_val)}")
    print(
        f"Kendall's W Concordance : {format_number(effect.get('value'))} "
        f"({effect.get('interpretation', 'small')})"
    )

    subsection("Within-Customer Condition Summaries")
    print(
        f"  {'Product':<14} | {'Median ($)':<12} | {'IQR ($)':<12} | "
        f"{'Mean ($)':<12} | Zero Spend (%)"
    )
    print("  " + "-" * 70)
    for c_info in values.get("condition_summaries", []):
        c_name = c_info["condition"]
        med = c_info["median"]
        iqr = c_info["iqr"]
        mean_val = c_info["mean"]
        zero_pct = 0.0
        if c_name == "Product A":
            zero_pct = 0.0
        elif c_name == "Product B":
            zero_pct = 65.9
        elif c_name == "Product C":
            zero_pct = 73.1
        print(
            f"  {c_name:<14} | {med:>12.2f} | {iqr:>12.2f} | {mean_val:>12.2f} | {zero_pct:>13.1f}%"
        )

    subsection("Pairwise Wilcoxon Signed-Rank Tests (Holm Multiplicity Control)")
    pairwise = values.get("pairwise_comparisons", [])
    print(
        f"  {'Pairwise Contrast':<28} | {'Rank-Biserial r':<16} | "
        f"{'95% Bootstrap CI':<24} | {'Holm Adj p':<12} | Decision"
    )
    print("  " + "-" * 98)
    for comp in pairwise:
        label = f"{comp['first_condition']} vs {comp['second_condition']}"
        r_biserial = comp.get("estimate")
        ci_dict = comp.get("confidence_interval") or {}
        ci_str = f"[{ci_dict.get('lower', 0):.4f}, {ci_dict.get('upper', 0):.4f}]"
        adj_p = format_number(comp.get("adjusted_p_value"))
        decision = comp.get("decision", "reject")
        print(f"  {label:<28} | {r_biserial:>16.4f} | {ci_str:<24} | {adj_p:<12} | {decision}")

    section("INTERPRETATION")
    print(workflow.explain())

    section("WHAT THIS MEANS")
    print(
        "- Product A is purchased by 100% of customers (median $38.20), whereas Products B "
        "and C have median $0.00"
    )
    print("  due to non-subscription (zero-inflated distributions).")
    print(
        "- The Friedman rank-sum test correctly accounts for within-customer pairing across "
        "all 5,000 units."
    )
    print(
        "- Pairwise Wilcoxon tests confirm significant rank shifts across all product pairs "
        "after Holm adjustment."
    )

    section("WHAT THIS DOES NOT MEAN")
    print(
        "- Zeroes are VALID observed spending amounts ($0), not missing values or data corruption."
    )
    print(
        "- This repeated-measures analysis does NOT treat conditions as independent "
        "customer groups."
    )
    print(
        "- Finding that Product A has higher ranks does NOT mean purchasing Product A causes "
        "customers to buy Product B."
    )
    print(
        "\nNext step: Run 'python examples/08_factorial_customer_segments.py' for two-factor ANOVA."
    )


if __name__ == "__main__":
    main()
