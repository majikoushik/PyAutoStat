"""Example 15: Publication-Ready PDF Export & Print Fidelity Showcase.

Demonstrates the optional PDF export layer and browser-print fidelity across
representative statistical workflows:
1. Welch independent-samples t-test (Static PDF, A4 portrait)
2. OLS linear regression (Full detail PDF with diagnostics, A4 portrait)
3. Binary logistic regression (Figure-enabled PDF with odds-ratio forest)
4. Pearson chi-square test (Figure-enabled PDF with contingency heatmap)
5. Canonical ResearchReport PDF (Full detail snapshot export)
6. Wide-table analysis in landscape orientation (Letter format)

Usage:
    python examples/15_pdf_reporting.py [--output-dir /path/to/output]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Allow importing local example utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import DATA_DICTIONARY, load_customer_data

from pyautostat import AnalysisOptions, ResearchAssistant, save_pdf

RANDOM_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PyAutoStat publication-ready PDF reports."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save PDF files. If omitted, a temporary directory is used.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory()
        out_dir = Path(temp_dir.name)

    print("=" * 76)
    print(" PyAutoStat Publication-Ready PDF Export & Print Fidelity Showcase")
    print(f" Target output directory: {out_dir.resolve()}")
    print("=" * 76)

    # Load customer dataset (customer IDs excluded to preserve privacy)
    df = load_customer_data(include_customer_id=False)
    assistant = ResearchAssistant(df)

    # 1. Welch t-test (Static PDF, A4 Portrait)
    print("\n[1/6] Generating Welch t-test static PDF report (A4 portrait)...")
    welch_workflow = assistant.run(
        objective="compare_groups",
        outcome="total_avg_monthly_spend",
        predictor="news_subscriber",
        estimand="mean",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    welch_pdf = out_dir / "welch_static.pdf"
    save_pdf(
        welch_workflow,
        welch_pdf,
        detail="standard",
        title="Customer Spending by News Subscription (Welch t-test)",
        page_size="A4",
        landscape=False,
        page_numbers=True,
        overwrite=True,
    )
    print(f"  -> Written: {welch_pdf} ({welch_pdf.stat().st_size:,} bytes)")

    # 2. OLS linear regression (Full detail PDF, A4 Portrait)
    print("\n[2/6] Generating OLS regression full PDF report (A4 portrait)...")
    ols_workflow = assistant.run(
        objective="regression",
        outcome="total_avg_monthly_spend",
        predictors=["household_income", "education_years", "brand_tenure_months"],
        design="independent",
        estimand="conditional_mean",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    ols_pdf = out_dir / "ols_full.pdf"
    save_pdf(
        ols_workflow,
        ols_pdf,
        detail="full",
        title="Spend Drivers Regression (OLS)",
        page_size="A4",
        landscape=False,
        page_numbers=True,
        overwrite=True,
    )
    print(f"  -> Written: {ols_pdf} ({ols_pdf.stat().st_size:,} bytes)")

    # 3. Binary logistic regression (Figure-enabled PDF)
    print("\n[3/6] Generating Logistic regression PDF with odds-ratio figure...")
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
    logistic_pdf = out_dir / "logistic_figures.pdf"
    save_pdf(
        logistic_workflow,
        logistic_pdf,
        detail="standard",
        title="High-Value Customer Predictors (Logistic Regression)",
        include_figures=True,
        page_size="A4",
        landscape=False,
        page_numbers=True,
        overwrite=True,
    )
    print(f"  -> Written: {logistic_pdf} ({logistic_pdf.stat().st_size:,} bytes)")

    # 4. Pearson chi-square test (Figure-enabled PDF with heatmap)
    print("\n[4/6] Generating Pearson chi-square PDF with heatmap figure...")
    chi2_workflow = assistant.run(
        objective="association",
        outcome="news_subscriber",
        predictor="gender",
        design="independent",
        estimand="categorical_independence",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    chi_sq_pdf = out_dir / "chi_square_figures.pdf"
    save_pdf(
        chi2_workflow,
        chi_sq_pdf,
        detail="standard",
        title="Gender vs News Subscription (Chi-Square Independence)",
        include_figures=True,
        page_size="A4",
        landscape=False,
        page_numbers=True,
        overwrite=True,
    )
    print(f"  -> Written: {chi_sq_pdf} ({chi_sq_pdf.stat().st_size:,} bytes)")

    # 5. Canonical ResearchReport PDF
    print("\n[5/6] Generating canonical ResearchReport full PDF export...")
    report = welch_workflow.report
    report_pdf = out_dir / "research_report_full.pdf"
    report.save_pdf(
        report_pdf,
        detail="full",
        page_size="A4",
        page_numbers=True,
        overwrite=True,
    )
    print(f"  -> Written: {report_pdf} ({report_pdf.stat().st_size:,} bytes)")

    # 6. Wide-table analysis in landscape orientation (Letter format)
    print("\n[6/6] Generating wide-table analysis in landscape orientation (Letter)...")
    landscape_pdf = out_dir / "wide_landscape.pdf"
    save_pdf(
        ols_workflow,
        landscape_pdf,
        detail="full",
        title="Comprehensive Regression Diagnostics (Landscape)",
        page_size="Letter",
        landscape=True,
        page_numbers=True,
        overwrite=True,
    )
    print(f"  -> Written: {landscape_pdf} ({landscape_pdf.stat().st_size:,} bytes)")

    print("\n" + "=" * 76)
    print(" All publication-ready PDF reports generated successfully!")
    print(f" Location: {out_dir.resolve()}")
    print(" Note: Output files are self-contained and generated offline.")
    print("=" * 76)


if __name__ == "__main__":
    main()
