"""Example 03: Multi-Group Customer Spending Comparison.

Business Question:
    "Does average monthly spending differ across customer job categories?"

Scientific Focus:
    Demonstrates guided multi-group mean inference across 3+ independent groups.
    Welch one-way ANOVA handles unequal group variances, followed by Games-Howell
    simultaneous pairwise comparisons for multiplicity control without silent estimand drift.
    Canonical Rich presentation renders the omnibus test, group summaries, and
    pairwise contrast table.
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

    # 3. Canonical Rich Terminal Presentation (detail="full")
    show(workflow, detail="full")

    # 4. Contextual tutorial takeaways
    section("WHAT THIS MEANS")
    print(
        "- The omnibus Welch one-way ANOVA tests whether all 6 population means are equal "
        "without assuming equal group variances."
    )
    print(
        "- Simultaneous Games-Howell adjustments protect the familywise error rate across "
        "all pairwise contrasts while accommodating unequal variances and unequal group sizes."
    )
    print(
        "- Pairwise contrasts identify which specific category pairs differ reliably after "
        "multiplicity control rather than relying on unadjusted comparisons."
    )

    section("WHAT THIS DOES NOT MEAN")
    print("- Job category is an observational classification, not an experimental assignment.")
    print("- Spending differences across job categories do NOT mean job category causes spending.")
    print(
        "- Failing to find significant differences between certain pairs does NOT prove those "
        "means are equivalent."
    )
    print(
        "- ANOVA on means addresses population mean differences, not median or "
        "distribution shape shifts."
    )
    print(
        "\nNext step: Run 'python examples/04_customer_value_association.py' for categorical "
        "association."
    )


if __name__ == "__main__":
    main()
