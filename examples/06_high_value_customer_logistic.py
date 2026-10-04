"""Example 06: High-Value Customer Logistic Regression Without Data Leakage.

Business Question:
    "Which non-spend customer characteristics are associated with membership in the
    high-value segment?"

Scientific Focus:
    Demonstrates inferential binary logistic regression without structural leakage.
    High-value status is perfectly separated by total monthly spend in this demonstration
    dataset (cutoff ~ $275). Using spend variables as predictors would therefore create
    severe target leakage/circularity. Spend variables and direct structural flags
    (Streaming, Wireless) are intentionally excluded. Evaluates odds ratios, Wald
    confidence intervals, and McFadden's pseudo-R2 strictly as inferential associations,
    not predictive AutoML.
    Canonical Rich presentation renders model fit, likelihood ratio test, odds ratios,
    and diagnostics.
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
    verify_dataset_integrity,
)

from pyautostat import AnalysisOptions, ResearchAssistant, show

RANDOM_SEED = 42

# Only non-spend, non-leakage candidate predictors
PREDICTORS = [
    "household_income",
    "education_years",
    "brand_tenure_months",
    "home_owner",
    "news_subscriber",
    "age",
]


def main() -> None:
    section("BUSINESS QUESTION")
    print(
        "Question : Which non-spend customer characteristics are associated with membership "
        "in the high-value segment?"
    )
    print("Estimand : Log-odds coefficients (beta) and Odds Ratios (OR = exp(beta))")
    print(
        "Design   : Observational cross-sectional binary logistic regression "
        "(5,000 customers, event=High Value)"
    )

    # 1. Programmatically verify the structural relation between target and spend
    integrity = verify_dataset_integrity()
    std_max = integrity["standard_customer_max_spend"]
    hvc_min = integrity["high_value_customer_min_spend"]
    separation = integrity["high_value_spend_separation"]

    section("STRUCTURAL LEAKAGE SAFEGUARD")
    print(
        f"Data Check: Standard customer max spend = ${std_max:.2f}; "
        f"High-value min spend = ${hvc_min:.2f}."
    )
    print(
        "Separation: High-value status is perfectly separated by total monthly spend in this "
        f"demonstration dataset (separation holds: {separation})."
    )
    print(
        "Safeguard : Using spend variables as predictors would therefore create "
        "severe target leakage/circularity."
    )
    print("            Therefore, all spend fields (Total Spend, Product A/B/C spends) and")
    print("            structurally tied flags (Streaming, Wireless) are INTENTIONALLY EXCLUDED.")
    print(
        "            Using them would create 100% circular tautology rather than genuine insight."
    )

    # 2. Load data (customer IDs excluded)
    frame = load_customer_data(include_customer_id=False)
    assistant = ResearchAssistant(frame)

    # 3. Run binary logistic regression
    workflow = assistant.run(
        objective="regression",
        outcome="high_value_customer",
        predictors=PREDICTORS,
        estimand="event_probability",
        design="independent",
        event_level=1,
        reference_levels={"home_owner": 0, "news_subscriber": "No"},
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )

    if workflow.analysis is None or workflow.interpretation is None:
        raise RuntimeError(f"Workflow execution failed: {workflow.blockers}")

    # 4. Canonical Rich Terminal Presentation (detail="full" for odds ratios, CIs, Wald tests)
    show(workflow, detail="full")

    # 5. Contextual tutorial takeaways
    section("WHAT THIS MEANS")
    print(
        "- Binary logistic regression models the log-odds of high-value customer status "
        "as a linear function of non-leakage customer characteristics."
    )
    print(
        "- Odds Ratios (OR = exp(beta)) represent multiplicative factors on the odds of being "
        "a high-value customer for a one-unit change in the predictor, holding other variables "
        "constant."
    )
    print(
        "- Even after rigorously excluding spend leakage, education years and brand tenure are "
        "strongly associated with high-value segment membership."
    )

    section("WHAT THIS DOES NOT MEAN")
    print("- An Odds Ratio is NOT a constant difference in probability.")
    print("- Odds are p / (1 - p); doubling the odds does not mean doubling the probability.")
    print(
        "- This model establishes conditional epidemiological associations, "
        "NOT causal interventions."
    )
    print(
        "- McFadden's pseudo-R2 indicates relative log-likelihood improvement, "
        "and is not comparable to OLS R-squared."
    )
    print(
        "- This is an inferential model; no claim is made regarding out-of-sample predictive "
        "accuracy."
    )
    print(
        "\nNext step: Run 'python examples/07_product_portfolio_repeated_measures.py' for "
        "within-customer repeated analysis."
    )


if __name__ == "__main__":
    main()
