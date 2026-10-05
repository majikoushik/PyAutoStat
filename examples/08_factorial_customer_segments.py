"""Example 08: Factorial Customer Segments Analysis (Two-Way ANOVA).

Business Question:
    "How are two customer-segmentation factors jointly associated with average monthly spend?"

Scientific Focus:
    Demonstrates independent two-way factorial ANOVA evaluating two factors (home ownership
    and news subscription) and their interaction on total monthly spend.
    Uses Type II sums of squares (hierarchical testing) and non-central F inversion for exact
    partial eta-squared confidence intervals. Unweighted estimated marginal means (EMMs)
    and cell summaries illustrate an approximately additive observed pattern without
    detectable interaction.
    Canonical Rich presentation renders the ANOVA effects table, cell summaries, and diagnostics.
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

from pyautostat import ResearchAssistant, show


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

    # 3. Canonical Rich Terminal Presentation (detail="full")
    show(workflow, detail="full")

    # 4. Contextual tutorial takeaways
    section("WHAT THIS MEANS")
    print(
        "- Two-way factorial ANOVA evaluates main effects for home ownership and news subscription "
        "along with their potential interaction on monthly spending."
    )
    print(
        "- The main-effect rows evaluate average associations for each factor while accounting "
        "for the other factor."
    )
    print(
        "- The interaction row evaluates whether the association of one factor differs across "
        "levels of the other."
    )
    print("- Interpret evidence for those effects directly from the Rich ANOVA table above.")
    print(
        "- Cell Sample Sizes and Means show an approximately additive pattern across "
        "customer subgroups without strong non-parallel profile lines."
    )

    section("WHAT THIS DOES NOT MEAN")
    print("- Neither factor was experimentally manipulated; these are observational associations.")
    print(
        "- Non-significance of the interaction does NOT prove that a population interaction "
        "is absent; absence of evidence is not evidence of exact additivity."
    )
    print(
        "- Visual line nonparallelism in sample profiles is not by itself inferential "
        "proof of an interaction."
    )
    print(
        "- Residual non-normality was noted in diagnostics, but given N = 5,000, F-tests for main "
        "effects have well-controlled asymptotic error rates."
    )
    print(
        "\nNext step: Run 'python examples/09_complete_research_workflow.py' for the end-to-end "
        "research lifecycle."
    )


if __name__ == "__main__":
    main()
