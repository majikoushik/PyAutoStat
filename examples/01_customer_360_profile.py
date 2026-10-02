"""Example 01: Customer 360 Profile and Data Understanding.

Business Question:
    "What does this customer base look like, and what should an analyst investigate first?"

Scientific Focus:
    Demonstrates that PyAutoStat begins with systematic dataset understanding and
    diagnostics before inferential testing. Outlier and normality flags are review cues,
    not instructions to delete observations or manipulate estimands.
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
)

from pyautostat import InsightEngine, ResearchAssistant

KEY_NUMERIC_VARIABLES = (
    "age",
    "household_income",
    "total_avg_monthly_spend",
    "car_value",
)


def main() -> None:
    section("BUSINESS QUESTION")
    print(
        "Question : What does this customer base look like, and what should we investigate first?"
    )
    print("Objective: Dataset understanding, distribution profiling, and data-quality screening")
    print("Estimand : Descriptive distribution summaries and diagnostic review cues")
    print("Design   : Observational cross-sectional customer cohort (demonstration dataset)")

    # 1. Load cleaned customer data (customer IDs excluded by default)
    frame = load_customer_data(include_customer_id=False)
    assistant = ResearchAssistant(frame)

    section("PYAUTOSTAT DECISION & PROFILE SUMMARY")
    print(f"Rows loaded     : {len(frame):,}")
    print(f"Columns analyzed: {len(frame.columns)}")
    print(
        "Customer ID     : Intentionally excluded from descriptive reporting to protect identity."
    )
    print("Data Dictionary : Declared variable types and measurement units applied.")

    # 2. Executive summary via assistant.summarize()
    subsection("Structured Overview")
    print(assistant.summarize(data_dictionary=DATA_DICTIONARY))

    # 3. Numeric Distribution and Percentile Behavior
    profile = assistant.profile(data_dictionary=DATA_DICTIONARY)
    subsection("Percentile Behavior: Total Average Monthly Spend")
    spend_desc = profile.get("descriptive", {}).get("total_avg_monthly_spend", {})
    percentiles = spend_desc.get("percentiles", {})
    print(f"  Mean   : ${format_number(spend_desc.get('mean'))} / month")
    print(f"  Std Dev: ${format_number(spend_desc.get('std'))}")
    print(
        f"  p05: ${format_number(percentiles.get('p05'))}  |  "
        f"p25: ${format_number(percentiles.get('p25'))}  |  "
        f"p50 (Median): ${format_number(percentiles.get('p50'))}  |  "
        f"p75: ${format_number(percentiles.get('p75'))}  |  "
        f"p95: ${format_number(percentiles.get('p95'))}"
    )
    print(
        f"  Skewness: {format_number(spend_desc.get('skewness'))} (positive skew, long upper tail)"
    )

    # 4. Outlier and Normality Review Cues
    subsection("Distribution Shape & Outlier Review Cues")
    for column in KEY_NUMERIC_VARIABLES:
        col_desc = profile.get("descriptive", {}).get(column, {})
        norm_tests = profile.get("normality", {}).get(column, {})
        norm_verdict = next(
            (
                norm_tests[t]["verdict"]
                for t in ("d_agostino_pearson", "shapiro_wilk")
                if norm_tests.get(t, {}).get("verdict")
            ),
            "Normality test unavailable",
        )
        iqr_info = profile.get("outliers", {}).get(column, {}).get("iqr", {})
        iqr_count = iqr_info.get("count", 0)
        iqr_pct = iqr_info.get("percentage", 0.0)
        print(
            f"  {column:26}: Mean={format_number(col_desc.get('mean')):>10}, "
            f"IQR Flags={iqr_count:>4} ({iqr_pct:>4.1f}%), {norm_verdict}"
        )

    print(
        "\n  Note: Outlier flags are review cues to inspect data collection and distribution tails."
    )
    print("  PyAutoStat does NOT automatically delete outliers or silently trim data.")

    # 5. Categorical Frequencies & Segment Cross-Tabs
    subsection("Categorical Segmentation")
    job_freq = assistant.frequency_table("job_category")
    print("Job category distribution:")
    for item in job_freq.get("levels", []):
        print(f"  - {item['level']:<16}: {item['count']:>5} ({item['percent']:.1f}%)")

    home_val_tab = assistant.cross_tab(
        "home_owner", "high_value_customer", data_dictionary=DATA_DICTIONARY
    )
    print(
        "\nCross-tab: Home Owner (0=No, 1=Yes) vs High Value Customer (0=Standard, 1=High Value):"
    )
    print(f"  Counts: {home_val_tab.get('counts')}")
    print(f"  Row % : {home_val_tab.get('row_percent')}")

    # 6. Prioritized Automated Insights
    section("PRIORITIZED INSIGHTS")
    engine = InsightEngine(profile)
    summary = engine.get_summary()
    counts = {
        "high": summary.get("high_severity", 0),
        "medium": summary.get("medium_severity", 0),
        "low": summary.get("low_severity", 0),
    }
    print(f"Screening findings by severity: {counts}")
    for insight in summary.get("insights", [])[:5]:
        print(f"\n[{insight['severity'].upper()}] {insight['category']}")
        print(f"  Finding       : {insight['finding']}")
        for rec in insight.get("recommendation", ())[:2]:
            print(f"  Recommendation: {rec}")

    section("WHAT THIS DOES NOT MEAN")
    print("- Descriptive correlations and distribution cues do NOT establish causal mechanisms.")
    print("- High skew or normality rejections do NOT justify arbitrary row trimming.")
    print("- Discovered associations require explicit research questions, declared estimands,")
    print("  and rigorous inferential testing.")
    print(
        "\nNext step: Run 'python examples/02_compare_customer_segments.py' "
        "for two-group inference."
    )


if __name__ == "__main__":
    main()
