"""Example 14: Interactive HTML & Scientific Figure Foundation Showcase.

Demonstrates the optional interactive HTML presentation layer and Plotly-backed
scientific figures across representative statistical workflows:
1. Welch independent-samples t-test (Estimate + CI figure)
2. Pearson linear correlation (Estimate + CI figure)
3. Ordinary Least Squares (OLS) linear regression (Coefficient forest plot)
4. Binary logistic regression (Predictor odds-ratio forest plot)
5. Pearson chi-square test (Contingency count heatmap)
6. Canonical ResearchReport interactive export

Usage:
    python examples/14_interactive_html_reporting.py [--output-dir /path/to/output]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Allow importing local example utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import DATA_DICTIONARY, load_customer_data, section

from pyautostat import AnalysisOptions, ResearchAssistant, save_interactive_html, show

RANDOM_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PyAutoStat interactive HTML reports with scientific figures."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save HTML files. If omitted, a temporary directory is used.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory()
        out_dir = Path(temp_dir.name)

    section("PyAutoStat Interactive HTML & Scientific Figure Foundation Showcase")
    print(f"Target output directory: {out_dir}")

    # Load customer dataset (customer IDs excluded to preserve privacy)
    df = load_customer_data(include_customer_id=False)
    assistant = ResearchAssistant(df)

    # -------------------------------------------------------------------------
    # 1. Welch Independent-Samples t-test (Source Analysis)
    # -------------------------------------------------------------------------
    welch_workflow = assistant.run(
        objective="compare_groups",
        outcome="total_avg_monthly_spend",
        predictor="news_subscriber",
        estimand="mean",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )

    section("SOURCE ANALYSIS")
    show(welch_workflow, detail="standard")

    section("GENERATED ARTIFACTS")
    welch_path = save_interactive_html(
        welch_workflow,
        out_dir / "welch_interactive.html",
        detail="standard",
        title="Customer Spending by News Subscription (Welch t-test)",
        include_figures=True,
        overwrite=True,
    )
    print(f"  ✓ Interactive HTML: {welch_path.name} ({welch_path.stat().st_size:,} bytes)")

    # -------------------------------------------------------------------------
    # 2. Pearson Linear Correlation (Estimate + CI)
    # -------------------------------------------------------------------------
    pearson_workflow = assistant.run(
        objective="association",
        outcome="monthly_spend_product_a",
        predictor="monthly_spend_product_b",
        estimand="linear",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    pearson_path = save_interactive_html(
        pearson_workflow,
        out_dir / "pearson_interactive.html",
        detail="standard",
        title="Product Spend Correlation (Pearson r)",
        include_figures=True,
        overwrite=True,
    )
    print(f"  ✓ Interactive HTML: {pearson_path.name} ({pearson_path.stat().st_size:,} bytes)")

    # -------------------------------------------------------------------------
    # 3. OLS Linear Regression (Coefficient Forest Plot)
    # -------------------------------------------------------------------------
    ols_workflow = assistant.run(
        objective="regression",
        outcome="total_avg_monthly_spend",
        predictors=["household_income", "education_years", "brand_tenure_months"],
        design="independent",
        estimand="conditional_mean",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    ols_path = save_interactive_html(
        ols_workflow,
        out_dir / "ols_interactive.html",
        detail="standard",
        title="Spend Drivers Regression (OLS)",
        include_figures=True,
        overwrite=True,
    )
    print(f"  ✓ Interactive HTML: {ols_path.name} ({ols_path.stat().st_size:,} bytes)")

    # -------------------------------------------------------------------------
    # 4. Binary Logistic Regression (Odds-Ratio Forest Plot)
    # -------------------------------------------------------------------------
    logistic_workflow = assistant.run(
        objective="regression",
        outcome="high_value_customer",
        predictors=["household_income", "education_years", "brand_tenure_months"],
        design="independent",
        estimand="event_probability",
        event_level=1,
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    logistic_path = save_interactive_html(
        logistic_workflow,
        out_dir / "logistic_interactive.html",
        detail="standard",
        title="High-Value Customer Predictors (Logistic Regression)",
        include_figures=True,
        overwrite=True,
    )
    print(f"  ✓ Interactive HTML: {logistic_path.name} ({logistic_path.stat().st_size:,} bytes)")

    # -------------------------------------------------------------------------
    # 5. Pearson Chi-Square Test (Contingency Count Heatmap)
    # -------------------------------------------------------------------------
    chi2_workflow = assistant.run(
        objective="association",
        outcome="news_subscriber",
        predictor="gender",
        design="independent",
        estimand="categorical_independence",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    chi2_path = save_interactive_html(
        chi2_workflow,
        out_dir / "chi_square_interactive.html",
        detail="standard",
        title="Gender vs News Subscription (Chi-Square Independence)",
        include_figures=True,
        overwrite=True,
    )
    print(f"  ✓ Interactive HTML: {chi2_path.name} ({chi2_path.stat().st_size:,} bytes)")

    # -------------------------------------------------------------------------
    # 6. Canonical ResearchReport Interactive Export
    # -------------------------------------------------------------------------
    report = welch_workflow.report
    rep_path = report.save_interactive_html(
        out_dir / "research_report_interactive.html",
        include_figures=True,
        overwrite=True,
    )
    print(f"  ✓ Interactive HTML: {rep_path.name} ({rep_path.stat().st_size:,} bytes)")

    print(f"\nAll interactive HTML reports generated successfully in {out_dir}.")
    print("Note: Output files are self-contained and require no external network calls.")


if __name__ == "__main__":
    main()
