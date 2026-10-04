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
    Canonical Rich presentation renders the omnibus test, condition summaries, and
    pairwise contrast table.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from repository root or examples directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import (
    announce_fast_mode_if_active,
    load_product_spend_long,
    resolve_bootstrap_samples,
    section,
)

from pyautostat import AnalysisOptions, ResearchAssistant, show

RANDOM_SEED = 42
CONDITION_ORDER = ("Product A", "Product B", "Product C")


def main() -> None:
    announce_fast_mode_if_active()
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
    b_samples = resolve_bootstrap_samples(default=499, fast_count=50)
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

    # 3. Canonical Rich Terminal Presentation (detail="full")
    show(workflow, detail="full")

    # 4. Contextual tutorial takeaways
    section("WHAT THIS MEANS")
    print(
        "- The Friedman rank-sum test accounts for within-customer pairing across "
        "all complete customer panels without requiring normality."
    )
    print(
        "- Follow-up pairwise Wilcoxon signed-rank tests evaluate condition rank shifts "
        "after Holm multiplicity adjustment."
    )
    print("- Kendall's W concordance measures overall agreement across conditions.")

    section("WHAT THIS DOES NOT MEAN")
    print(
        "- Zeroes are VALID observed spending amounts ($0), not missing values or data corruption."
    )
    print(
        "- This repeated-measures analysis does NOT treat conditions as independent "
        "customer groups; matching on customer_id is required."
    )
    print(
        "- Finding that Product A has higher ranks does NOT establish that purchasing Product A "
        "causes customers to buy Product B or C."
    )
    print(
        "\nNext step: Run 'python examples/08_factorial_customer_segments.py' for two-factor ANOVA."
    )


if __name__ == "__main__":
    main()
