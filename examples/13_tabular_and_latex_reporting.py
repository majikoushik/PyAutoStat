"""Example 13: Tabular (CSV), LaTeX, and Markdown Research Report Showcase.

Demonstrates extracting structured tabular data and publication-oriented markup
formats from canonical PyAutoStat ResearchReport objects:
1. Extracting individual structured CSV tables with `report.to_csv_tables()`
2. Exporting journal-ready LaTeX documents with `report.to_latex()` and `report.save_latex()`
3. Generating structured Markdown reports with `report.to_markdown()`
4. Formula-injection protection for spreadsheet data and safe LaTeX character escaping

Usage:
    python examples/13_tabular_and_latex_reporting.py [--output-dir /path/to/output]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Allow importing local example utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import DATA_DICTIONARY, load_customer_data, section

from pyautostat import AnalysisOptions, ResearchAssistant, show

RANDOM_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PyAutoStat CSV tables, LaTeX, and Markdown exports for customer analyses."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save generated report files. If omitted, a temporary directory is used.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory()
        out_dir = Path(temp_dir.name)

    section("PyAutoStat Tabular, LaTeX, and Markdown Export Showcase")
    print(f"Target output directory: {out_dir}")

    # 1. Load customer dataset (customer IDs excluded)
    df = load_customer_data(include_customer_id=False)
    assistant = ResearchAssistant(df)

    # -------------------------------------------------------------------------
    # 1. Two-Group Mean Comparison (Welch t-test)
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

    section("SOURCE ANALYSIS: WELCH INDEPENDENT T-TEST")
    show(welch_workflow, detail="standard")

    report = welch_workflow.report
    assert report is not None

    section("STRUCTURED TABULAR EXPORT (CSV TABLES)")
    csv_dir = out_dir / "csv_tables"
    csv_dir.mkdir(parents=True, exist_ok=True)

    csv_tables = report.to_csv_tables()
    for table_id, csv_content in csv_tables.items():
        table_path = csv_dir / f"{table_id}.csv"
        table_path.write_text(csv_content, encoding="utf-8")
        print(f"  * CSV table: {table_path.name} ({len(csv_content):,} chars)")

    section("PUBLICATION-READY LATEX EXPORT")
    latex_path = report.save_latex(
        out_dir / "customer_spending_welch.tex",
        style="apa",
        overwrite=True,
    )
    print(f"  * LaTeX: {latex_path.name} ({latex_path.stat().st_size:,} bytes)")

    section("STRUCTURED MARKDOWN EXPORT")
    md_content = report.to_markdown(style="apa")
    md_path = out_dir / "customer_spending_welch.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"  * Markdown: {md_path.name} ({md_path.stat().st_size:,} bytes)")

    # -------------------------------------------------------------------------
    # 2. Linear Regression (OLS with HC3 Covariance)
    # -------------------------------------------------------------------------
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

    ols_report = ols_workflow.report
    assert ols_report is not None

    ols_latex_path = ols_report.save_latex(
        out_dir / "customer_spend_drivers_ols.tex",
        style="apa",
        overwrite=True,
    )
    print(f"  * Regression LaTeX: {ols_latex_path.name} ({ols_latex_path.stat().st_size:,} bytes)")

    ols_tables = ols_report.to_csv_tables()
    for table_id, csv_content in ols_tables.items():
        table_path = csv_dir / f"ols_{table_id}.csv"
        table_path.write_text(csv_content, encoding="utf-8")
        print(f"  * Regression CSV table: {table_path.name} ({len(csv_content):,} chars)")

    print(f"\nAll tabular, LaTeX, and Markdown exports completed successfully in {out_dir}.")


if __name__ == "__main__":
    main()
