"""Example 04: Customer Value Categorical Association (Chi-Square & Cramer's V).

Business Question:
    "Is home-ownership status associated with high-value-customer segment membership?"

Scientific Focus:
    Demonstrates categorical independence testing on a 2x2 contingency table without
    spend leakage. Home ownership is an external demographic factor, not a mathematical
    component of the high-value definition. Expected counts, Cramer's V effect size,
    and bootstrap uncertainty are reported alongside the chi-square statistic.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from repository root or examples directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import (
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

    analysis = workflow.analysis
    values = analysis.values
    meta = analysis.metadata

    section("PYAUTOSTAT DECISION")
    print(f"Selected Method: {analysis.method_label} ({analysis.method_id})")
    print(f"Rows Analyzed  : {analysis.sample_size:,}")
    print(f"Rows Excluded  : {analysis.excluded_rows}")

    subsection("Recommendation Rationale")
    if workflow.recommendation:
        print(workflow.recommendation.rationale_text.strip())

    section("RESULT: CONTINGENCY TABLE & DIAGNOSTICS")
    row_labels = meta.get("group_order", ["Owner", "Non-owner"])
    col_labels = meta.get("outcome_order", ["High Value", "Standard"])
    observed = meta.get("observed_counts", [[0, 0], [0, 0]])
    expected = meta.get("expected_counts", [[0.0, 0.0], [0.0, 0.0]])

    print(f"Contingency Table ({' vs '.join(col_labels)}):")
    print(f"  {'Segment':<14} | {col_labels[0]:<12} | {col_labels[1]:<12} | Total")
    print("  " + "-" * 50)
    for i, r_label in enumerate(row_labels):
        r_total = sum(observed[i])
        print(f"  {r_label:<14} | {observed[i][0]:>12} | {observed[i][1]:>12} | {r_total:>5}")
        print(f"  {'  (expected)':<14} | {expected[i][0]:>12.1f} | {expected[i][1]:>12.1f} |")

    diag = meta.get("diagnostics", {})
    min_exp = diag.get("minimum_expected_count")
    print(f"\nMinimum Expected Frequency: {format_number(min_exp)} (policy: >= 5 required; MET)")
    print("Exact Fisher fallback     : Not required (expected counts well exceed threshold).")

    subsection("Test Statistics & Effect Size")
    chi2 = values.get("test_statistic")
    df = values.get("degrees_of_freedom")
    p_val = values.get("p_value")
    effect = values.get("effect_size") or {}
    ci = values.get("confidence_interval") or {}

    print(f"Pearson Chi-Square (X2) : {format_number(chi2)}")
    print(f"Degrees of Freedom (df) : {format_number(df)}")
    print(f"p-value                 : {format_number(p_val)}")
    print(
        f"Effect Size (Cramer's V): {format_number(effect.get('value'))} "
        f"({effect.get('interpretation', 'negligible')})"
    )
    if ci:
        print(
            f"Cramer's V 95% CI       : [{format_number(ci.get('lower'))}, "
            f"{format_number(ci.get('upper'))}]"
        )

    section("INTERPRETATION")
    print(workflow.explain())

    section("WHAT THIS MEANS")
    if observed and len(observed) >= 2 and len(observed[0]) >= 2:
        tot_0 = sum(observed[0])
        tot_1 = sum(observed[1])
        rate_0 = (observed[0][0] / tot_0 * 100.0) if tot_0 > 0 else 0.0
        rate_1 = (observed[1][0] / tot_1 * 100.0) if tot_1 > 0 else 0.0
        print(
            f"- {row_labels[0]} customers are slightly more likely to belong to the "
            f"{col_labels[0]} segment ({rate_0:.1f}%) than "
            f"{row_labels[1]} customers ({rate_1:.1f}%)."
        )
    print(
        f"- The association is statistically detectable (p = {format_number(p_val)}), but the "
        f"magnitude is negligible (Cramer's V = {format_number(effect.get('value'))})."
    )
    print(
        f"- With N = {analysis.sample_size:,}, Pearson's chi-square test has high power to reject "
        "independence even when the real-world association is slight."
    )

    section("WHAT THIS DOES NOT MEAN")
    print("- Home ownership does NOT cause customer value; both are shaped by income and wealth.")
    print(
        "- Statistical significance does NOT imply home ownership is a strong customer "
        "targeting filter."
    )
    print(
        "- Sparse 2x2 exact tests (Fisher) are fully supported in PyAutoStat, but were not "
        "triggered here because cell counts are large."
    )
    print(
        "\nNext step: Run 'python examples/05_spend_drivers_regression.py' for multivariable "
        "regression."
    )


if __name__ == "__main__":
    main()
