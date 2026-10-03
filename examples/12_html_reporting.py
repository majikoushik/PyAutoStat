"""Example 12: Modern HTML and Research Report Presentation Showcase.

Demonstrates the self-contained, reproducible HTML presentation system across
four representative statistical analysis families and a canonical ResearchReport:
1. Welch independent-samples t-test
2. Pearson linear correlation
3. Ordinary Least Squares (OLS) linear regression
4. Kruskal-Wallis rank-based multi-group comparison
5. Canonical ResearchReport HTML export

Usage:
    python examples/12_html_reporting.py [--output-dir /path/to/output]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Allow importing local example utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import DATA_DICTIONARY, load_customer_data

from pyautostat import AnalysisOptions, ResearchAssistant, save_html

RANDOM_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PyAutoStat modern HTML reports for representative customer analyses."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save generated HTML files. If omitted, a temporary directory is used.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory()
        out_dir = Path(temp_dir.name)

    print("=" * 76)
    print(" PyAutoStat Modern HTML & Research Report Presentation Showcase")
    print(f" Target output directory: {out_dir}")
    print("=" * 76)

    # 1. Load customer dataset
    df = load_customer_data(include_customer_id=False)
    assistant = ResearchAssistant(df)

    # -------------------------------------------------------------------------
    # 1. Welch Independent-Samples t-test
    # -------------------------------------------------------------------------
    print("\n[1/5] Generating Welch t-test HTML report...")
    welch_workflow = assistant.run(
        objective="compare_groups",
        outcome="total_avg_monthly_spend",
        predictor="news_subscriber",
        estimand="mean",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    welch_path = save_html(
        welch_workflow,
        out_dir / "welch.html",
        detail="standard",
        title="Customer Spending by News Subscription (Welch t-test)",
        overwrite=True,
    )
    print(f"  -> Written: {welch_path} ({welch_path.stat().st_size:,} bytes)")

    # -------------------------------------------------------------------------
    # 2. Pearson Correlation
    # -------------------------------------------------------------------------
    print("\n[2/5] Generating Pearson correlation HTML report...")
    pearson_workflow = assistant.run(
        objective="association",
        outcome="monthly_spend_product_a",
        predictor="monthly_spend_product_b",
        estimand="linear",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    pearson_path = save_html(
        pearson_workflow,
        out_dir / "pearson.html",
        detail="standard",
        title="Product Spend Correlation (Pearson r)",
        overwrite=True,
    )
    print(f"  -> Written: {pearson_path} ({pearson_path.stat().st_size:,} bytes)")

    # -------------------------------------------------------------------------
    # 3. OLS Linear Regression
    # -------------------------------------------------------------------------
    print("\n[3/5] Generating OLS regression HTML report...")
    ols_workflow = assistant.run(
        objective="regression",
        outcome="total_avg_monthly_spend",
        predictors=["household_income", "car_value"],
        design="independent",
        estimand="conditional_mean",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        covariance_type="HC3",
        data_dictionary=DATA_DICTIONARY,
    )
    ols_path = save_html(
        ols_workflow,
        out_dir / "ols.html",
        detail="full",
        title="Customer Spend Drivers (OLS Regression with HC3 Covariance)",
        overwrite=True,
    )
    print(f"  -> Written: {ols_path} ({ols_path.stat().st_size:,} bytes)")

    # -------------------------------------------------------------------------
    # 4. Kruskal-Wallis Multi-Group Comparison
    # -------------------------------------------------------------------------
    print("\n[4/5] Generating Kruskal-Wallis rank ANOVA HTML report...")
    kruskal_workflow = assistant.run(
        objective="compare_groups",
        outcome="total_avg_monthly_spend",
        predictor="job_category",
        estimand="distribution",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    kruskal_path = save_html(
        kruskal_workflow,
        out_dir / "kruskal.html",
        detail="full",
        title="Monthly Spend Across Job Categories (Kruskal-Wallis & Dunn-Holm)",
        overwrite=True,
    )
    print(f"  -> Written: {kruskal_path} ({kruskal_path.stat().st_size:,} bytes)")

    # -------------------------------------------------------------------------
    # 5. Canonical ResearchReport
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating canonical ResearchReport HTML...")
    assert welch_workflow.report is not None
    report_path = welch_workflow.report.save_html(
        out_dir / "research_report.html",
        style="general",
        detail="standard",
        title="Canonical Customer Research Report",
        overwrite=True,
    )
    print(f"  -> Written: {report_path} ({report_path.stat().st_size:,} bytes)")

    print("\n" + "=" * 76)
    print(" All 5 demonstration HTML reports generated successfully!")
    print(f" Directory: {out_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
