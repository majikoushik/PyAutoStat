"""Example 10: Rich Terminal Presentation for PyAutoStat.

Demonstrates the terminal presentation layer (`show`) across detail levels:
1. Dataset profile (Standard & Compact)
2. Welch independent-samples t-test workflow (Standard, Full, & Compact)
3. Pearson linear correlation workflow (Standard & Compact)

Using the realistic CustomerDataset (5,000 customers, customer IDs excluded).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing local example utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import DATA_DICTIONARY, load_customer_data, section, subsection

from pyautostat import AnalysisOptions, ResearchAssistant, show

RANDOM_SEED = 42


def main() -> None:
    section("PyAutoStat Rich Terminal Presentation")

    # 1. Load customer data (customer IDs excluded)
    df = load_customer_data(include_customer_id=False)
    assistant = ResearchAssistant(df)

    # -------------------------------------------------------------------------
    # 1. Dataset Profile (Standard Detail)
    # -------------------------------------------------------------------------
    subsection("1. DATASET PROFILE PRESENTATION (STANDARD DETAIL)")
    profile = assistant.profile(data_dictionary=DATA_DICTIONARY)
    show(profile)

    # -------------------------------------------------------------------------
    # 2. Welch Independent-Samples t-test (Standard Detail)
    # -------------------------------------------------------------------------
    subsection("2. WELCH t-TEST WORKFLOW (STANDARD DETAIL)")
    welch_workflow = assistant.run(
        objective="compare_groups",
        outcome="total_avg_monthly_spend",
        predictor="news_subscriber",
        estimand="mean",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    show(welch_workflow, detail="standard")

    # -------------------------------------------------------------------------
    # 3. Welch t-test (Full Detail Mode with Diagnostics & Metadata)
    # -------------------------------------------------------------------------
    subsection("3. WELCH t-TEST WORKFLOW (FULL DETAIL MODE)")
    show(welch_workflow, detail="full")

    # -------------------------------------------------------------------------
    # 4. Pearson Correlation (Standard Detail)
    # -------------------------------------------------------------------------
    subsection("4. PEARSON CORRELATION WORKFLOW (STANDARD DETAIL)")
    pearson_workflow = assistant.run(
        objective="association",
        outcome="monthly_spend_product_a",
        predictor="monthly_spend_product_b",
        estimand="linear",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    show(pearson_workflow, detail="standard")

    # -------------------------------------------------------------------------
    # 5. Compact Detail Mode Showcase
    # -------------------------------------------------------------------------
    subsection("5. COMPACT DETAIL MODE SHOWCASE")
    print("Profile compact:")
    show(profile, detail="compact")
    print("\nWelch t-test compact:")
    show(welch_workflow, detail="compact")
    print("\nPearson r compact:")
    show(pearson_workflow, detail="compact")


if __name__ == "__main__":
    main()
