"""Example 04: Customer Value Categorical Association (Chi-Square & Cramer's V).

Business Question:
    "Is home-ownership status associated with high-value-customer segment membership?"

Scientific Focus:
    Demonstrates categorical independence testing on a 2x2 contingency table without
    spend leakage. Home ownership is an external demographic factor, not a mathematical
    component of the high-value definition. Expected counts, Cramer's V effect size,
    and bootstrap uncertainty are reported alongside the chi-square statistic.
    Canonical Rich presentation renders the contingency table, test statistics, and interpretation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from repository root or examples directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import (
    load_customer_data,
    section,
)

from pyautostat import AnalysisOptions, ResearchAssistant, show

RANDOM_SEED = 42


def main() -> None:
    section("BUSINESS QUESTION")
    print(
        "Question : Is home-ownership status associated with high-value-customer segment "
        "membership?"
    )
    print("Estimand : Categorical independence and association strength (Cramer's V)")
    print("Design   : Independent observations, 2x2 cross-classification (5,000 customers)")

    # 1. Load data with human-readable labels (Owner / Non-owner, High Value / Standard)
    # Customer IDs are excluded by default
    frame = load_customer_data(include_customer_id=False, labeled_categories=True)
    assistant = ResearchAssistant(frame)

    # 2. Run categorical independence workflow
    workflow = assistant.run(
        objective="association",
        outcome="high_value_customer",
        predictor="home_owner",
        estimand="categorical_independence",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
    )

    if workflow.analysis is None or workflow.interpretation is None:
        raise RuntimeError(f"Workflow execution failed: {workflow.blockers}")

    # 3. Canonical Rich Terminal Presentation
    show(workflow, detail="standard")

    # 4. Contextual tutorial takeaways
    section("WHAT THIS MEANS")
    print(
        "- The Pearson chi-square test of independence evaluates whether segment membership "
        "is statistically independent of home-ownership status."
    )
    print("- Cramer's V represents association magnitude separately from statistical significance.")
    print(
        "- With large samples, small associations can produce small p-values, so effect magnitude "
        "remains essential for interpretation."
    )

    section("WHAT THIS DOES NOT MEAN")
    print("- Home ownership does NOT cause customer value; both are shaped by income and wealth.")
    print(
        "- Statistical significance does NOT imply home ownership is an effective standalone "
        "customer targeting filter."
    )
    print(
        "- Sparse 2x2 exact tests (Fisher) are fully supported in PyAutoStat, but were not "
        "triggered here because expected cell counts well exceed minimum sample thresholds."
    )
    print(
        "\nNext step: Run 'python examples/05_spend_drivers_regression.py' for multivariable "
        "regression."
    )


if __name__ == "__main__":
    main()
