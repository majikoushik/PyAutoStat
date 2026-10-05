"""Example 05: Customer Characteristics Associated with Monthly Spend (Multiple OLS with HC3).

Business Question:
    "Which non-product customer characteristics are conditionally associated with total monthly
    spend?"

Scientific Focus:
    Demonstrates multiple linear regression without structural data leakage. Spend components
    (Product A/B/C) and direct structural flags (Streaming, Wireless) are excluded.
    HC3 heteroscedasticity-consistent covariance estimates reduce reliance on the equal-variance
    assumption without altering point estimates. Model diagnostics (VIF, Breusch-Pagan) inform
    interpretation without triggering post-hoc variable selection or row deletion. Canonical Rich
    presentation renders model fit, coefficient tables, diagnostics, and interpretation.

    Note: "Spend drivers" is business shorthand only; the fitted OLS model estimates
    conditional associations, not causal drivers.
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
        "Question : Which non-product customer characteristics are conditionally associated "
        "with total monthly spend?"
    )
    print("Estimand : Conditional mean regression coefficients (beta) and in-sample R-squared")
    print(
        "Design   : Cross-sectional multiple OLS regression with HC3 robust covariance "
        "(5,000 customers)"
    )
    print(
        "Note     : 'Spend drivers' is business shorthand; the fitted OLS model estimates "
        "conditional associations, not causal effects."
    )

    # 1. Load cleaned customer data (customer IDs excluded)
    frame = load_customer_data(include_customer_id=False)
    assistant = ResearchAssistant(frame)

    # 2. Run multiple regression with explicit reference levels and HC3 covariance
    workflow = assistant.run(
        objective="regression",
        outcome="total_avg_monthly_spend",
        predictors=PREDICTORS,
        estimand="conditional_mean",
        design="independent",
        reference_levels={"home_owner": 0, "news_subscriber": "No"},
        covariance_type="HC3",
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
        "- Ordinary least-squares linear regression models customer monthly spend conditional on "
        "the researcher-specified set of non-leakage predictors."
    )
    print(
        "- HC3 provides heteroskedasticity-consistent covariance estimates and standard errors, "
        "reducing reliance on the constant-variance assumption without altering point estimates."
    )
    print(
        "- Standardized beta coefficients allow comparative assessment of relative "
        "association strength across predictors measured on different scales."
    )

    section("WHAT THIS DOES NOT MEAN")
    print("- These coefficients are CONDITIONAL associations, not causal effects.")
    print(
        "- 'Spend drivers' is business shorthand; regression does not establish causal mechanisms."
    )
    print("- R-squared reflects in-sample fit; it is not out-of-sample predictive accuracy.")
    print("- No automated stepwise variable selection or p-hacking was performed.")
    print(
        "- Outliers or influential points were NOT automatically pruned to artificially inflate R2."
    )
    print(
        "\nNext step: Run 'python examples/06_high_value_customer_logistic.py' for binary "
        "logistic modeling."
    )


if __name__ == "__main__":
    main()
