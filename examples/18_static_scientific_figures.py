"""Example 18: Static Scientific Figure Export and DOCX Embedding.

Demonstrates publication-quality static image export (PNG, SVG, PDF) and DOCX
figure embedding from authoritative FigureSpec objects without statistical recalculation.

Shows:
1. Welch t-test mean difference and CI (PNG).
2. Pearson correlation estimate and CI (SVG).
3. OLS linear regression coefficient forest (PNG).
4. Logistic regression odds ratio forest (PNG).
5. Chi-square contingency count heatmap (PNG).
6. Two-way factorial ANOVA cell profile (PNG).
7. Editable DOCX report with embedded scientific figures.

Usage:
    python examples/18_static_scientific_figures.py [--output-dir /path/to/output]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Allow importing local example utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import load_customer_data

from pyautostat import (
    AnalysisOptions,
    ResearchAssistant,
    save_docx,
    save_static_figures,
)

RANDOM_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality static scientific figures and DOCX reports."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save generated figure artifacts and DOCX reports.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory()
        out_dir = Path(temp_dir.name)

    print("=" * 76)
    print(" PyAutoStat Static Scientific Figure Export & DOCX Embedding Showcase")
    print("=" * 76)
    print(f"Target directory: {out_dir.resolve()}\n")

    df = load_customer_data(include_customer_id=False)
    assistant = ResearchAssistant(df)

    # 1. Welch t-test (PNG)
    print("1. Generating Welch t-test Estimate & CI Figure (PNG)...")
    wf_welch = assistant.run(
        objective="compare_groups",
        outcome="total_avg_monthly_spend",
        predictor="gender",
        estimand="mean",
        design="independent",
        variable_types={"total_avg_monthly_spend": "continuous", "gender": "nominal"},
        options=AnalysisOptions(confidence_level=0.95),
    )
    welch_paths = save_static_figures(
        wf_welch,
        out_dir / "welch",
        format="png",
        overwrite=True,
    )
    for p in welch_paths:
        print(f"   Saved: {p.name} ({p.stat().st_size:,} bytes)")

    # 2. Pearson correlation (SVG)
    print("\n2. Generating Pearson Correlation Estimate & CI Figure (SVG)...")
    wf_pearson = assistant.run(
        objective="association",
        outcome="total_avg_monthly_spend",
        predictor="household_income",
        estimand="linear",
        design="independent",
        variable_types={
            "household_income": "continuous",
            "total_avg_monthly_spend": "continuous",
        },
        options=AnalysisOptions(confidence_level=0.95),
    )
    pearson_paths = save_static_figures(
        wf_pearson,
        out_dir / "pearson",
        format="svg",
        overwrite=True,
    )
    for p in pearson_paths:
        print(f"   Saved: {p.name} ({p.stat().st_size:,} bytes)")

    # 3. OLS linear regression coefficient forest (PNG)
    print("\n3. Generating OLS Regression Coefficient Forest Figure (PNG)...")
    wf_ols = assistant.run(
        objective="regression",
        outcome="total_avg_monthly_spend",
        predictors=["household_income", "age", "education_years"],
        design="independent",
        estimand="conditional_mean",
        variable_types={
            "total_avg_monthly_spend": "continuous",
            "household_income": "continuous",
            "age": "continuous",
            "education_years": "continuous",
        },
        options=AnalysisOptions(confidence_level=0.95),
    )
    ols_paths = save_static_figures(
        wf_ols,
        out_dir / "ols",
        format="png",
        overwrite=True,
    )
    for p in ols_paths:
        print(f"   Saved: {p.name} ({p.stat().st_size:,} bytes)")

    # 4. Logistic regression odds ratio forest (PNG)
    print("\n4. Generating Logistic Regression Odds Ratio Forest Figure (PNG)...")
    wf_logit = assistant.run(
        objective="regression",
        outcome="high_value_customer",
        predictors=["household_income", "age"],
        design="independent",
        estimand="event_probability",
        event_level=1,
        variable_types={
            "high_value_customer": "nominal",
            "household_income": "continuous",
            "age": "continuous",
        },
        options=AnalysisOptions(confidence_level=0.95),
    )
    logit_paths = save_static_figures(
        wf_logit,
        out_dir / "logistic",
        format="png",
        overwrite=True,
    )
    for p in logit_paths:
        print(f"   Saved: {p.name} ({p.stat().st_size:,} bytes)")

    # 5. Chi-square contingency heatmap (PNG)
    print("\n5. Generating Chi-Square Contingency Count Heatmap Figure (PNG)...")
    wf_chi2 = assistant.run(
        objective="association",
        outcome="high_value_customer",
        predictor="car_ownership",
        estimand="categorical_independence",
        design="independent",
        variable_types={"car_ownership": "nominal", "high_value_customer": "nominal"},
    )
    chi2_paths = save_static_figures(
        wf_chi2,
        out_dir / "chi_square",
        format="png",
        overwrite=True,
    )
    for p in chi2_paths:
        print(f"   Saved: {p.name} ({p.stat().st_size:,} bytes)")

    # 6. Two-way ANOVA cell profile (PNG)
    print("\n6. Generating Two-Way ANOVA Observed Cell Profile Figure (PNG)...")
    wf_twoway = assistant.run(
        objective="compare_groups",
        outcome="total_avg_monthly_spend",
        factor_a="gender",
        factor_b="marital_status",
        design="independent",
        estimand="mean",
        variable_types={
            "total_avg_monthly_spend": "continuous",
            "gender": "nominal",
            "marital_status": "nominal",
        },
    )
    twoway_paths = save_static_figures(
        wf_twoway,
        out_dir / "two_way_anova",
        format="png",
        overwrite=True,
    )
    for p in twoway_paths:
        print(f"   Saved: {p.name} ({p.stat().st_size:,} bytes)")

    # 7. Editable Microsoft Word (.docx) report with embedded figures
    print("\n7. Generating Editable Word Report (.docx) with Embedded Figures...")
    docx_path = out_dir / "spend_drivers_research_report.docx"
    save_docx(
        wf_ols,
        docx_path,
        include_figures=True,
        title="Customer Spend Drivers Analysis",
        style="apa",
        overwrite=True,
    )
    print(f"   Saved DOCX: {docx_path.name} ({docx_path.stat().st_size:,} bytes)")

    print("\n" + "=" * 76)
    print(" Static scientific figure generation and DOCX embedding completed.")
    print("=" * 76)


if __name__ == "__main__":
    main()
