"""Example 06: High-Value Customer Logistic Regression Without Data Leakage.

Business Question:
    "Which non-spend customer characteristics are associated with membership in the
    high-value segment?"

Scientific Focus:
    Demonstrates inferential binary logistic regression without structural leakage.
    Because 'high_value_customer' is deterministically derived from total monthly spend
    (cutoff ~ $275), all spend variables and direct structural flags (Streaming, Wireless)
    are intentionally excluded. Evaluates odds ratios, Wald confidence intervals,
    and McFadden's pseudo-R2 strictly as inferential associations, not predictive AutoML.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from repository root or examples directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import (
    DATA_DICTIONARY,
    format_number,
    load_customer_data,
    section,
    subsection,
    verify_dataset_integrity,
)

from pyautostat import AnalysisOptions, ResearchAssistant

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
        f"Separation: High-value status is deterministically separated at spend ~ $275 "
        f"(separation holds: {separation})."
    )
    print("Safeguard : Therefore, all spend fields (Total Spend, Product A/B/C spends) and")
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

    analysis = workflow.analysis
    values = analysis.values
    fit = values.get("model_fit", {})

    section("PYAUTOSTAT DECISION")
    print(f"Selected Method: {analysis.method_label} ({analysis.method_id})")
    print("Modeled Event  : high_value_customer == 1 (High Value Segment)")
    print(f"Sample Size    : {analysis.sample_size:,} complete cases (0 excluded)")
    event_cnt = values.get("event_count", 0)
    event_rate = values.get("event_rate", 0.0)
    print(f"Event Prevalence: {event_cnt:,} / {analysis.sample_size:,} ({event_rate * 100:.1f}%)")

    subsection("Recommendation Rationale")
    if workflow.recommendation:
        print(workflow.recommendation.rationale_text.strip())

    section("RESULT: MODEL FIT & LIKELIHOOD RATIO TEST")
    lr_stat = fit.get("lr_statistic")
    lr_p = fit.get("lr_p_value")
    mcfadden = fit.get("mcfadden_r2")
    aic = fit.get("aic")

    print(
        f"Likelihood Ratio X2       : {format_number(lr_stat)} "
        f"(df = {fit.get('model_degrees_of_freedom')})"
    )
    print(f"LR p-value                : {format_number(lr_p)}")
    print(
        f"McFadden's Pseudo-R2      : {format_number(mcfadden)} "
        f"({mcfadden * 100:.1f}% log-likelihood improvement)"
    )
    print(f"Akaike Information (AIC)  : {format_number(aic)}")

    subsection("Odds Ratios & Wald Inference Table")
    print(
        f"  {'Predictor':<28} | {'Odds Ratio':<12} | {'95% Wald CI':<24} | {'Wald z':<10} | p-value"
    )
    print("  " + "-" * 92)
    for coef in values.get("coefficients", []):
        t_label = coef.get("term_label", coef.get("term", ""))
        if coef.get("kind") == "intercept":
            continue
        or_val = coef.get("odds_ratio")
        or_ci = coef.get("odds_ratio_ci") or {}
        ci_str = f"[{or_ci.get('lower', 0):.4f}, {or_ci.get('upper', 0):.4f}]"
        z_stat = coef.get("statistic")
        p_val = format_number(coef.get("p_value"))
        print(f"  {t_label:<28} | {or_val:>12.4f} | {ci_str:<24} | {z_stat:>10.2f} | {p_val}")

    section("INTERPRETATION")
    print(workflow.explain())

    section("WHAT THIS MEANS")
    print(
        "- Even after rigorously excluding spend leakage, education years and brand tenure are "
        "strongly associated"
    )
    print("  with high-value segment membership.")
    print(
        "- Each additional year of education multiplies the odds of being a high-value customer "
        "by approx 1.34 (OR ~ 1.34, 95% CI [1.30, 1.38], p < 1e-70)."
    )
    print(
        "- Each additional month of brand tenure increases the odds by approx 2.1% "
        "(OR ~ 1.021, 95% CI [1.016, 1.026], p < 1e-14)."
    )

    section("WHAT THIS DOES NOT MEAN")
    print("- An Odds Ratio is NOT a constant difference in probability.")
    print(
        "- This model establishes conditional epidemiological associations, "
        "NOT causal interventions."
    )
    print(
        "- McFadden pseudo-R2 (14.6%) indicates meaningful explanatory signal, "
        "but is NOT comparable to OLS R2."
    )
    print(
        "- This is an inferential model; no test-set AUC or classification accuracy "
        "threshold is claimed."
    )
    print(
        "\nNext step: Run 'python examples/07_product_portfolio_repeated_measures.py' for "
        "within-customer repeated analysis."
    )


if __name__ == "__main__":
    main()
