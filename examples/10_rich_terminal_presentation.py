"""Example 10: Rich-based Terminal Presentation Pilot for PyAutoStat.

Demonstrates the new terminal presentation layer (`show`) on:
1. Dataset profile
2. Welch independent-samples t-test workflow
3. Pearson correlation workflow

Using the realistic CustomerDataset (5,000 customers, customer IDs excluded).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing local example utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import DATA_DICTIONARY, load_customer_data

from pyautostat import AnalysisOptions, ResearchAssistant, show

RANDOM_SEED = 42


def main() -> None:
    print("=" * 72)
    print(" PyAutoStat Rich Terminal Presentation Pilot")
    print("=" * 72)

    # 1. Load customer data (customer IDs excluded)
    df = load_customer_data(include_customer_id=False)
    assistant = ResearchAssistant(df)

    # -------------------------------------------------------------------------
    # Pilot 1: Dataset Profile
    # -------------------------------------------------------------------------
    print("\n>>> 1. DATASET PROFILE PRESENTATION\n")
    profile = assistant.profile(data_dictionary=DATA_DICTIONARY)
    show(profile)

    # -------------------------------------------------------------------------
    # Pilot 2: Welch Independent-Samples t-test
    # -------------------------------------------------------------------------
    print("\n>>> 2. WELCH t-TEST WORKFLOW PRESENTATION\n")
    welch_workflow = assistant.run(
        objective="compare_groups",
        outcome="total_avg_monthly_spend",
        predictor="news_subscriber",
        estimand="mean",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    show(welch_workflow)

    # -------------------------------------------------------------------------
    # Pilot 3: Pearson Correlation
    # -------------------------------------------------------------------------
    print("\n>>> 3. PEARSON CORRELATION WORKFLOW PRESENTATION\n")
    pearson_workflow = assistant.run(
        objective="association",
        outcome="monthly_spend_product_a",
        predictor="monthly_spend_product_b",
        estimand="linear",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    show(pearson_workflow)

    # -------------------------------------------------------------------------
    # Compact Mode Showcase
    # -------------------------------------------------------------------------
    print("\n>>> 4. COMPACT DETAIL MODE SHOWCASE\n")
    print("Profile compact:")
    show(profile, detail="compact")
    print("\nWelch t-test compact:")
    show(welch_workflow, detail="compact")
    print("\nPearson r compact:")
    show(pearson_workflow, detail="compact")


if __name__ == "__main__":
    main()
