"""Example 02: Two-Group Customer Segment Spending Comparison.

Business Question:
    "Do news subscribers and non-subscribers differ in average total monthly spend?"

Scientific Focus:
    Demonstrates guided two-group mean inference. Method selection preserves the declared
    population-mean estimand without silently switching to a rank test because of skewness.
    Canonical Rich presentation renders the design, contrast, estimates, diagnostics,
    and interpretation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from repository root or examples directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import (
    DATA_DICTIONARY,
    load_customer_data,
    section,
)

from pyautostat import AnalysisOptions, ResearchAssistant, show

RANDOM_SEED = 42


def main() -> None:
    section("BUSINESS QUESTION")
    print(
        "Question : Do news subscribers and non-subscribers differ in average total monthly spend?"
    )
    print("Estimand : Population mean difference ('No' minus 'Yes')")
    print("Design   : Independent groups (each row represents a distinct customer)")

    # 1. Load cleaned customer data (customer IDs excluded)
    frame = load_customer_data(include_customer_id=False)
    assistant = ResearchAssistant(frame)

    # 2. Run the guided workflow with explicit research specification
    workflow = assistant.run(
        objective="compare_groups",
        outcome="total_avg_monthly_spend",
        predictor="news_subscriber",
        estimand="mean",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )

    if workflow.analysis is None or workflow.interpretation is None:
        raise RuntimeError(f"Workflow execution failed: {workflow.blockers}")

    # 3. Canonical Rich Terminal Presentation
    show(workflow, detail="standard")

    # 4. Contextual tutorial takeaways
    section("WHAT THIS MEANS")
    print(
        "- In this dataset, non-subscribers spent approximately $47.70 less per month "
        "than subscribers (95% CI [-$56.86, -$38.55])."
    )
    print(
        "- Welch independent-samples t-test was selected because it preserves the declared "
        "mean estimand without assuming equal population variances."
    )

    section("WHAT THIS DOES NOT MEAN")
    print(
        "- This observational difference does NOT establish that news subscription "
        "causes higher spending."
    )
    print(
        "- Statistical significance alone does not establish practical importance; "
        "with N = 5,000, even small effects achieve small p-values."
    )
    print(
        "- Non-normality in spending distributions does not invalidate mean inference given "
        "N > 2,000 per group, and PyAutoStat never switches estimands behind the researcher's back."
    )
    print("\nNext step: Run 'python examples/03_multigroup_customer_spending.py' for 3+ groups.")


if __name__ == "__main__":
    main()
