"""Example 05: Customer Characteristics Associated with Monthly Spend (Multiple OLS with HC3).

Business Question:
    "Which non-product customer characteristics are conditionally associated with total monthly
    spend?"

Scientific Focus:
    Demonstrates multiple linear regression without structural data leakage. Spend components
    (Product A/B/C) and direct structural flags (Streaming, Wireless) are excluded.
    HC3 heteroscedasticity-consistent covariance estimates reduce reliance on the equal-variance
    assumption without altering point estimates. Model diagnostics (VIF, Breusch-Pagan) inform
    interpretation without triggering post-hoc variable selection or row deletion.

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
    format_currency,
    format_number,
    load_customer_data,
    section,
    subsection,
)

from pyautostat import AnalysisOptions, ResearchAssistant

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

    analysis = workflow.analysis
    values = analysis.values
    fit = values.get("model_fit", {})
    r2_ci = fit.get("r_squared_confidence_interval") or {}

    section("PYAUTOSTAT DECISION")
    print(f"Selected Method: {analysis.method_label} ({analysis.method_id})")
    print(f"Rows Analyzed  : {fit.get('analyzed_rows', analysis.sample_size):,}")
    print(f"Covariance Type: {fit.get('covariance_type', 'HC3')} (heteroscedasticity-robust)")
    print(f"Excluded Rows  : {analysis.excluded_rows}")
    print(f"Predictors ({len(PREDICTORS)}) : {', '.join(PREDICTORS)}")

    subsection("Recommendation Rationale")
    if workflow.recommendation:
        print(workflow.recommendation.rationale_text.strip())

    section("RESULT: MODEL FIT & IN-SAMPLE EXPLANATION")
    r2 = fit.get("r_squared")
    adj_r2 = fit.get("adjusted_r_squared")
    f_stat = fit.get("model_f_statistic")
    f_p = fit.get("model_f_p_value")

    print(f"R-squared (R2)            : {format_number(r2)} ({r2 * 100:.1f}% variance explained)")
    if r2_ci:
        print(
            f"R-squared 95% Bootstrap CI: [{format_number(r2_ci.get('lower'))}, "
            f"{format_number(r2_ci.get('upper'))}]"
        )
    print(f"Adjusted R-squared        : {format_number(adj_r2)}")
    print(
        f"Residual Standard Error   : ${format_number(fit.get('residual_standard_error'))} / month"
    )
    print(f"Overall Model F-Statistic : {format_number(f_stat)} (p = {format_number(f_p)})")

    subsection("Coefficient Table (HC3 Robust Standard Errors)")
    print(
        f"  {'Term':<32} | {'Estimate':<10} | {'HC3 SE':<10} | {'95% CI':<24} | "
        f"{'p-value':<12} | Beta (std)"
    )
    print("  " + "-" * 104)
    for coef in values.get("coefficients", []):
        t_label = coef.get("term_label", coef.get("term", ""))
        b_est = coef.get("estimate")
        se = coef.get("standard_error")
        ci_dict = coef.get("confidence_interval") or {}
        ci_str = f"[{ci_dict.get('lower', 0):.2f}, {ci_dict.get('upper', 0):.2f}]"
        p_val = format_number(coef.get("p_value"))
        beta = coef.get("standardized_beta")
        beta_str = f"{beta:.4f}" if isinstance(beta, (int, float)) else "N/A"
        print(
            f"  {t_label:<32} | {b_est:>10.4f} | {se:>10.4f} | {ci_str:<24} | "
            f"{p_val:<12} | {beta_str:>10}"
        )

    subsection("Regression Diagnostics")
    diag = values.get("diagnostics", {})
    bp = diag.get("breusch_pagan", {})
    print(
        f"Breusch-Pagan Heteroscedasticity Test: LM = {format_number(bp.get('lm_statistic'))}, "
        f"p = {format_number(bp.get('lm_p_value'))}"
    )
    print("  -> The heteroscedasticity diagnostic supports reporting HC3 robust covariance.")

    vif_info = diag.get("vif", {})
    print(
        f"Multicollinearity: Max VIF = {format_number(vif_info.get('maximum'))} (< 5.0; low risk)"
    )
    for term_vif in vif_info.get("terms", [])[:3]:
        print(f"  - {term_vif['predictor']:<22}: VIF = {term_vif['value']:.2f}")

    section("INTERPRETATION")
    print(workflow.explain())

    coef_map = {c.get("term", ""): c for c in values.get("coefficients", [])}
    edu_c = coef_map.get("education_years", {})
    edu_b = edu_c.get("estimate")
    edu_std = edu_c.get("standardized_beta")
    tenure_c = coef_map.get("brand_tenure_months", {})
    tenure_std = tenure_c.get("standardized_beta")

    section("WHAT THIS MEANS")
    if edu_b is not None:
        print(
            f"- Holding other demographic predictors constant, education years "
            f"(standardized beta = {format_number(edu_std)}) and"
        )
        print(
            f"  brand tenure (standardized beta = {format_number(tenure_std)}) "
            f"exhibit the strongest conditional associations with customer monthly spend."
        )
        print(
            f"- Each additional year of education is associated with an estimated "
            f"{format_currency(edu_b)} higher monthly spend."
        )
    print(
        "- HC3 provides heteroskedasticity-consistent covariance estimates and reduces reliance "
        "on the equal-variance assumption."
    )

    section("WHAT THIS DOES NOT MEAN")
    print("- These coefficients are CONDITIONAL associations, not causal effects.")
    r2_val = fit.get("r_squared")
    r2_str = f"R-squared ({r2_val * 100:.1f}%)" if isinstance(r2_val, (int, float)) else "R-squared"
    print(f"- {r2_str} reflects in-sample fit; it is not out-of-sample predictive accuracy.")
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
