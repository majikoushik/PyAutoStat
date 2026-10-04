"""Example 01: Customer 360 Profile and Data Understanding.

Business Question:
    "What does this customer base look like, and what should an analyst investigate first?"

Scientific Focus:
    Demonstrates that PyAutoStat begins with systematic dataset understanding and
    diagnostics before inferential testing. Canonical Rich presentation formats dataset
    profiles, frequency distributions, and cross-tabulations. Outlier and normality flags
    are review cues, not instructions to delete observations or manipulate estimands.
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
    subsection,
)

from pyautostat import InsightEngine, ResearchAssistant, show


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

    # 2. Canonical Dataset Profile via show(profile)
    profile = assistant.profile(data_dictionary=DATA_DICTIONARY)
    show(profile)

    # 3. Categorical Frequencies & Segment Cross-Tabs via show()
    subsection("Categorical Frequency Distribution: Job Category")
    job_freq = assistant.frequency_table("job_category")
    show(job_freq)

    subsection("Cross-Tabulation: Home Ownership vs High-Value Customer")
    home_val_tab = assistant.cross_tab(
        "home_owner",
        "high_value_customer",
        data_dictionary=DATA_DICTIONARY,
    )
    show(home_val_tab)

    # 4. Outlier & Normality Guidance
    subsection("Diagnostic Review Cue Guidance")
    print(
        "Note on Diagnostics:"
        "\n- Outlier flags are review cues to inspect data collection and distribution tails."
        "\n- PyAutoStat does NOT automatically delete outliers or silently trim data."
        "\n- Rejection of normality diagnostics does not imply row deletion or automatic"
        "\n  switching to a rank test when the research question targets the population mean."
    )

    # 5. Prioritized Automated Insights
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
