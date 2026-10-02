"""Example 02: Two-Group Customer Segment Spending Comparison.

Business Question:
    "Do news subscribers and non-subscribers differ in average total monthly spend?"

Scientific Focus:
    Demonstrates guided two-group mean inference. Method selection preserves the declared
    population-mean estimand without silently switching to a rank test because of skewness.
    Effect size, bootstrap uncertainty, and assumption diagnostics complement the p-value.
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

    analysis = workflow.analysis
    values = analysis.values

    section("PYAUTOSTAT DECISION")
    print(f"Selected Method: {analysis.method_label} ({analysis.method_id})")
    print(f"Rows Analyzed  : {analysis.sample_size:,}")
    print(f"Rows Excluded  : {analysis.excluded_rows}")
    print(f"Groups Compared: {values.get('groups', ['No', 'Yes'])}")

    subsection("Recommendation Rationale")
    if workflow.recommendation:
        print(workflow.recommendation.rationale_text.strip())

    section("RESULT")
    est = values.get("primary_estimate")
    ci = values.get("confidence_interval") or {}
    effect = values.get("effect_size") or {}
    effect_ci = effect.get("confidence_interval") or {}
    p_val = values.get("p_value")

    print(f"Primary Estimate (Mean Diff) : {format_currency(est)} / month")
    print(
        f"95% Confidence Interval      : [{format_currency(ci.get('lower'))}, "
        f"{format_currency(ci.get('upper'))}]"
    )
    eff_name = effect.get("name", "Cohen d")
    eff_interp = effect.get("interpretation", "small")
    print(f"Effect Size ({eff_name:7})       : {format_number(effect.get('value'))} ({eff_interp})")
    if effect_ci:
        print(
            f"Effect Size 95% Bootstrap CI : [{format_number(effect_ci.get('lower'))}, "
            f"{format_number(effect_ci.get('upper'))}]"
        )
    print(f"p-value                      : {format_number(p_val)}")
    print(f"Test Statistic (t)           : {format_number(values.get('test_statistic'))}")
    print(f"Degrees of Freedom (df)      : {format_number(values.get('degrees_of_freedom'))}")

    section("INTERPRETATION")
    print(workflow.explain())

    grp_means = frame.groupby("news_subscriber")["total_avg_monthly_spend"].mean()
    mean_no = grp_means.get("No", 0.0)
    mean_yes = grp_means.get("Yes", 0.0)

    section("WHAT THIS MEANS")
    print(
        f"- In this dataset, non-subscribers averaged {format_currency(mean_no)}/month versus "
        f"{format_currency(mean_yes)}/month for news subscribers."
    )
    print(
        f"- The observed mean difference ('No' minus 'Yes') is {format_currency(est)}/month "
        f"(95% CI [{format_currency(ci.get('lower'))}, {format_currency(ci.get('upper'))}])."
    )
    if isinstance(est, (int, float)):
        diff_mag = format_currency(-est if est < 0.0 else est)
        direction_word = "less" if est < 0.0 else "more"
        print(
            f"- Non-subscribers spent approximately {diff_mag} {direction_word} "
            "per month than subscribers."
        )
    print(
        "- Welch's t-test was selected because it preserves the declared mean estimand without "
        "assuming equal population variances."
    )

    section("WHAT THIS DOES NOT MEAN")
    print(
        "- This observational difference does NOT establish that news subscription "
        "causes higher spending."
    )
    print(
        "- Highly statistically significant p-values result partly from the large sample "
        "(N = 5,000);"
    )
    eff_val = format_number(effect.get("value"))
    print(
        f"  the effect size (Cohen's d = {eff_val}) is small-to-moderate and requires practical "
        "evaluation."
    )
    print(
        "- Non-normality in spending distributions does not invalidate mean inference given "
        "N > 2,000 per group,"
    )
    print("  and PyAutoStat never switches estimands behind the researcher's back.")
    print("\nNext step: Run 'python examples/03_multigroup_customer_spending.py' for 3+ groups.")


if __name__ == "__main__":
    main()
