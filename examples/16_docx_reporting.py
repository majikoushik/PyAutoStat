"""Example 16: Editable Word / DOCX Research Reporting Showcase.

Demonstrates the optional Microsoft Word (.docx) export layer across representative
statistical workflows:
1. Welch independent-samples t-test (Standard detail, A4 portrait)
2. OLS linear regression (Full detail with model diagnostics, A4 portrait)
3. Binary logistic regression (Standard detail, A4 portrait)
4. Canonical ResearchReport DOCX (Full detail executive summary & reproducibility)
5. Wide-table regression analysis in landscape orientation (Letter format)

Usage:
    python examples/16_docx_reporting.py [--output-dir /path/to/output]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Allow importing local example utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import DATA_DICTIONARY, load_customer_data

from pyautostat import AnalysisOptions, ResearchAssistant, save_docx

RANDOM_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PyAutoStat editable Microsoft Word (.docx) research reports."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save DOCX files. If omitted, a temporary directory is used.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory()
        out_dir = Path(temp_dir.name)

    print("=" * 76)
    print(" PyAutoStat Editable Word / DOCX Research Reporting Showcase")
    print(f" Target output directory: {out_dir.resolve()}")
    print("=" * 76)

    # Load customer dataset (customer IDs excluded to preserve privacy)
    df = load_customer_data(include_customer_id=False)
    assistant = ResearchAssistant(df)

    # 1. Welch t-test (Standard detail, A4 Portrait)
    print("\n[1/5] Generating Welch t-test DOCX report (A4 portrait)...")
    welch_workflow = assistant.run(
        objective="compare_groups",
        outcome="total_avg_monthly_spend",
        predictor="news_subscriber",
        estimand="mean",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    welch_docx = out_dir / "welch.docx"
    save_docx(
        welch_workflow,
        welch_docx,
        detail="standard",
        title="Customer Spending by News Subscription (Welch t-test)",
        page_size="A4",
        landscape=False,
        page_numbers=True,
        overwrite=True,
    )
    print(f"  -> Written: {welch_docx} ({welch_docx.stat().st_size:,} bytes)")

    # 2. OLS linear regression (Full detail, A4 Portrait)
    print("\n[2/5] Generating OLS regression full DOCX report (A4 portrait)...")
    ols_workflow = assistant.run(
        objective="regression",
        outcome="total_avg_monthly_spend",
        predictors=["household_income", "education_years", "brand_tenure_months"],
        design="independent",
        estimand="conditional_mean",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    ols_docx = out_dir / "ols_full.docx"
    save_docx(
        ols_workflow,
        ols_docx,
        detail="full",
        title="Spend Drivers Regression (OLS Full Detail)",
        page_size="A4",
        landscape=False,
        page_numbers=True,
        overwrite=True,
    )
    print(f"  -> Written: {ols_docx} ({ols_docx.stat().st_size:,} bytes)")

    # 3. Binary logistic regression (Standard detail, A4 Portrait)
    print("\n[3/5] Generating Logistic regression DOCX report (A4 portrait)...")
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
    logistic_docx = out_dir / "logistic.docx"
    save_docx(
        logistic_workflow,
        logistic_docx,
        detail="standard",
        title="High-Value Customer Predictors (Logistic Regression)",
        page_size="A4",
        landscape=False,
        page_numbers=True,
        overwrite=True,
    )
    print(f"  -> Written: {logistic_docx} ({logistic_docx.stat().st_size:,} bytes)")

    # 4. Canonical ResearchReport DOCX (Full detail)
    print("\n[4/5] Generating canonical ResearchReport full DOCX export...")
    report = welch_workflow.report
    report_docx = out_dir / "research_report_full.docx"
    report.save_docx(
        report_docx,
        detail="full",
        page_size="A4",
        page_numbers=True,
        overwrite=True,
    )
    print(f"  -> Written: {report_docx} ({report_docx.stat().st_size:,} bytes)")

    # 5. Wide-table regression analysis in landscape orientation (Letter format)
    print("\n[5/5] Generating wide-table analysis in landscape orientation (Letter)...")
    landscape_docx = out_dir / "wide_landscape.docx"
    save_docx(
        ols_workflow,
        landscape_docx,
        detail="full",
        title="Comprehensive Regression Diagnostics (Landscape)",
        page_size="Letter",
        landscape=True,
        page_numbers=True,
        overwrite=True,
    )
    print(f"  -> Written: {landscape_docx} ({landscape_docx.stat().st_size:,} bytes)")

    print("\n" + "=" * 76)
    print(" All editable Microsoft Word (.docx) reports generated successfully!")
    print(f" Location: {out_dir.resolve()}")
    print(" Note: Output files are valid, self-contained OpenXML packages.")
    print("=" * 76)


if __name__ == "__main__":
    main()
