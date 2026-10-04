"""The published customer analytics examples should run and preserve the scientific contract."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

# All public example scripts must use CustomerDataset.csv
PUBLIC_SCRIPTS = (
    "01_customer_360_profile.py",
    "02_compare_customer_segments.py",
    "03_multigroup_customer_spending.py",
    "04_customer_value_association.py",
    "05_spend_drivers_regression.py",
    "06_high_value_customer_logistic.py",
    "07_product_portfolio_repeated_measures.py",
    "08_factorial_customer_segments.py",
    "09_complete_research_workflow.py",
)

ALL_EXAMPLE_FILES = (
    *PUBLIC_SCRIPTS,
    "_customer_data.py",
    "run_all.py",
)

SAMPLE_CUSTOMER_IDS = (
    "0002-GTOKLU-YVY",
    "0003-RLTRGE-IW2",
    "0003-UTGKPR-PRU",
    "0008-ZIQQOT-SGB",
    "0012-CIVYLF-839",
)


def _run_example(filename: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYAUTOSTAT_FAST_TEST"] = "1"
    return subprocess.run(
        [sys.executable, str(EXAMPLES / filename), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=180,
    )


def test_customer_data_loader_contract():
    """Verify CustomerDataset.csv loading, shape, types, and verified structural invariants."""
    sys.path.insert(0, str(EXAMPLES))
    from _customer_data import load_customer_data, load_product_spend_long

    # 1. Anonymized view
    df_clean = load_customer_data(include_customer_id=False)
    assert df_clean.shape == (5000, 39)
    assert "customer_id" not in df_clean.columns

    # 2. View with customer_id for matching
    df_full = load_customer_data(include_customer_id=True)
    assert df_full.shape == (5000, 40)
    assert "customer_id" in df_full.columns
    assert df_full["customer_id"].is_unique
    assert df_full["customer_id"].notna().all()

    # 3. Numeric currency conversions
    for col in (
        "monthly_spend_product_a",
        "monthly_spend_product_b",
        "monthly_spend_product_c",
        "total_avg_monthly_spend",
        "cumulative_spend_product_a",
        "cumulative_spend_product_b",
        "cumulative_spend_product_c",
    ):
        assert pd.api.types.is_numeric_dtype(df_full[col]), f"{col} must be numeric"

    # 4. Total Avg Monthly Spend is exact sum of Product A, B, and C
    spend_diff = (
        df_full["total_avg_monthly_spend"]
        - (
            df_full["monthly_spend_product_a"]
            + df_full["monthly_spend_product_b"]
            + df_full["monthly_spend_product_c"]
        )
    ).abs()
    assert (spend_diff < 1e-5).all(), "Total spend must equal sum of product spends"

    # 5. Zero-inflation structural relationships
    assert (
        (df_full["monthly_spend_product_b"] > 0) == (df_full["streaming_services"] == "Yes")
    ).all(), "Product B spend exists iff customer has streaming services"
    assert (
        (df_full["monthly_spend_product_c"] > 0) == (df_full["wireless_internet"] == "Yes")
    ).all(), "Product C spend exists iff customer has wireless internet"

    # 6. High-value customer deterministic separation threshold (~$275)
    max_standard = df_full.loc[df_full["high_value_customer"] == 0, "total_avg_monthly_spend"].max()
    min_high_val = df_full.loc[df_full["high_value_customer"] == 1, "total_avg_monthly_spend"].min()
    assert max_standard < min_high_val, "High-value customer must separate cleanly by spend"
    assert 274.0 < max_standard < 275.1
    assert 275.1 < min_high_val < 276.0

    # 7. Long-form product spend repeated panel
    long_panel = load_product_spend_long()
    assert long_panel.shape == (15000, 3)
    assert set(long_panel.columns) == {"customer_id", "product", "monthly_spend"}
    assert set(long_panel["product"].unique()) == {"Product A", "Product B", "Product C"}
    assert long_panel["customer_id"].nunique() == 5000
    assert long_panel["monthly_spend"].notna().all()


@pytest.mark.parametrize(
    ("filename", "expected_content"),
    (
        (
            "01_customer_360_profile.py",
            [
                "BUSINESS QUESTION",
                "PYAUTOSTAT DECISION & PROFILE SUMMARY",
                "WHAT THIS DOES NOT MEAN",
            ],
        ),
        (
            "02_compare_customer_segments.py",
            ["BUSINESS QUESTION", "Welch independent-samples t-test", "INTERPRETATION"],
        ),
        (
            "03_multigroup_customer_spending.py",
            ["BUSINESS QUESTION", "Welch one-way ANOVA", "Games-Howell"],
        ),
        (
            "04_customer_value_association.py",
            ["BUSINESS QUESTION", "Pearson chi-square test of independence", "Cramer's V"],
        ),
        (
            "05_spend_drivers_regression.py",
            ["BUSINESS QUESTION", "Ordinary least-squares linear regression", "HC3"],
        ),
        (
            "06_high_value_customer_logistic.py",
            ["BUSINESS QUESTION", "STRUCTURAL LEAKAGE SAFEGUARD", "Binary logistic regression"],
        ),
        (
            "07_product_portfolio_repeated_measures.py",
            ["BUSINESS QUESTION", "Friedman rank-sum test", "Kendall's W"],
        ),
        (
            "08_factorial_customer_segments.py",
            ["BUSINESS QUESTION", "Two-way factorial ANOVA", "Cell Sample Sizes and Means"],
        ),
    ),
)
def test_introductory_examples_run_from_repository_root(
    filename: str, expected_content: list[str]
) -> None:
    result = _run_example(filename)

    assert result.returncode == 0, f"Script {filename} failed with stderr:\n{result.stderr}"
    for expected in expected_content:
        assert expected in result.stdout, f"Expected '{expected}' missing from {filename}"
    for sample_id in SAMPLE_CUSTOMER_IDS:
        assert sample_id not in result.stdout, f"Customer ID {sample_id} leaked in {filename}"
    assert not re.search(r"\b\d{4}-[A-Z0-9]{6}-[A-Z0-9]{3}\b", result.stdout)


def test_complete_research_workflow_runs_and_writes_canonical_exports(tmp_path: Path) -> None:
    output_dir = tmp_path / "customer_reports"
    result = _run_example(
        "09_complete_research_workflow.py", "--output-dir", str(output_dir), "--fast"
    )

    assert result.returncode == 0, f"Example 09 failed with stderr:\n{result.stderr}"

    for heading in (
        "1. STRUCTURED CLARIFICATION: UNKNOWN DESIGN STAYS UNKNOWN",
        "2. PLAN THE ESTIMAND, SENSITIVITY SCENARIOS, AND MEANINGFUL EFFECT",
        "3. EXECUTE ONCE AND INTERPRET RECORDED VALUES",
        "4. COMPARE DECLARED SENSITIVITY SCENARIOS",
        "5. KEEP PRACTICAL IMPORTANCE SEPARATE FROM THE P-VALUE",
        "6. REPORT, AUDIT, REPLAY, AND SERIALIZE THE SESSION",
        "7. EXPLICIT PAIRED ANALYSIS AND PROSPECTIVE PAIRED PLANNING",
        "SUMMARY OF GENERATED ARTIFACTS",
    ):
        assert heading in result.stdout

    for evidence in (
        "Initial workflow status: needs_input",
        "Plan created after analysis: False",
        "comparability: same_estimand",
        "comparability: different_estimand",
        "Plan adherence: matched",
        "Audit: passed",
        "Same-data replay: reproduced",
        "Identifier values are used for matching",
    ):
        assert evidence in result.stdout

    for sample_id in SAMPLE_CUSTOMER_IDS:
        assert sample_id not in result.stdout

    expected_files = {
        "customer_analysis.html",
        "customer_analysis.md",
        "customer_analysis.json",
        "decision_ledger.json",
        "session_snapshot.json",
    }
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    assert (output_dir / "customer_analysis_tables").is_dir()

    report_json = json.loads((output_dir / "customer_analysis.json").read_text(encoding="utf-8"))
    assert report_json["schema_version"] == 2
    assert report_json["status"] == "complete"

    snapshot_json = json.loads((output_dir / "session_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot_json["schema_version"] == 1

    ledger_json = json.loads((output_dir / "decision_ledger.json").read_text(encoding="utf-8"))
    assert "events" in ledger_json
    assert len(ledger_json["events"]) > 0

    html_content = (output_dir / "customer_analysis.html").read_text(encoding="utf-8")
    for sample_id in SAMPLE_CUSTOMER_IDS:
        assert sample_id not in html_content
    assert not re.search(r"\b\d{4}-[A-Z0-9]{6}-[A-Z0-9]{3}\b", html_content)


def test_public_examples_use_only_csv_dataset() -> None:
    """Ensure no public example script or examples README references CustomerDataset.xlsx."""
    for script_name in ALL_EXAMPLE_FILES:
        content = (EXAMPLES / script_name).read_text(encoding="utf-8")
        assert "CustomerDataset.xlsx" not in content, (
            f"{script_name} must not reference CustomerDataset.xlsx"
        )

    readme_content = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert "CustomerDataset.xlsx" not in readme_content


def test_scientific_copy_guardrails() -> None:
    """Assert example narrative adheres to scientific guardrails and avoids misleading claims."""
    text = "\n".join(
        (EXAMPLES / name).read_text(encoding="utf-8")
        for name in (*PUBLIC_SCRIPTS, "_customer_data.py", "run_all.py", "README.md")
    ).lower()

    assert "picks the appropriate test based on your data" not in text
    assert "based on normality and sample size" not in text
    assert "if both methods agree the conclusion is more robust" not in text
    assert "bootstrap confidence intervals — for effect sizes, always" not in text


def test_bundled_workbook_matches_documented_shape_and_has_unique_pairing_ids() -> None:
    """Legacy check verifying the optional workbook artifact preserves structure if present."""
    pytest.importorskip("openpyxl")
    workbook_path = EXAMPLES / "CustomerDataset.xlsx"
    if not workbook_path.exists():
        pytest.skip("CustomerDataset.xlsx not present")
    frame = pd.read_excel(workbook_path)

    assert frame.shape == (5000, 40)
    assert frame["CustomerID"].notna().all()
    assert frame["CustomerID"].is_unique
    assert {
        "Gender",
        "TotalAvgMonthlySpend",
        "MonthlySpend_ProductA",
        "MonthlySpend_ProductB",
    } <= set(frame)


def test_example_02_signed_contrast_orientation() -> None:
    """Example 02 must preserve signed contrast orientation without abs() inversion."""
    result = _run_example("02_compare_customer_segments.py")
    assert result.returncode == 0, f"Example 02 failed:\n{result.stderr}"

    # Output must state negative difference: Non-subscribers spend less
    assert "-$47.70" in result.stdout or "-47.70" in result.stdout
    assert "[-$56.86, -$38.55]" in result.stdout or "[-56.86, -38.55]" in result.stdout
    assert "spent approximately $47.70 less per month than subscribers" in result.stdout

    # Verify no abs() inversion or reversed bound order
    assert "abs(" not in (EXAMPLES / "02_compare_customer_segments.py").read_text(encoding="utf-8")


def test_example_05_predictor_leakage_invariants() -> None:
    """Example 05 must not include direct spend components or total spend as predictors."""
    sys.path.insert(0, str(EXAMPLES))
    import importlib

    e05 = importlib.import_module("05_spend_drivers_regression")
    cdata = importlib.import_module("_customer_data")

    predictors = set(e05.PREDICTORS)
    spend_components = set(cdata.SPEND_COMPONENT_COLUMNS)

    # 1. Disjoint from direct spend components
    assert predictors.isdisjoint(spend_components), (
        f"Example 05 predictors contain spend components: {predictors & spend_components}"
    )

    # 2. Total spend must not predict itself
    assert "total_avg_monthly_spend" not in predictors

    # 3. Structural spend proxies (streaming / wireless) must be excluded
    assert "streaming_services" not in predictors
    assert "wireless_internet" not in predictors


def test_example_06_predictor_leakage_invariants() -> None:
    """Example 06 must not include spend variables predicting high-value status."""
    sys.path.insert(0, str(EXAMPLES))
    import importlib

    e06 = importlib.import_module("06_high_value_customer_logistic")
    cdata = importlib.import_module("_customer_data")

    predictors = set(e06.PREDICTORS)
    forbidden = set(cdata.HIGH_VALUE_FORBIDDEN_PREDICTORS)

    # 1. Disjoint from forbidden predictors
    assert predictors.isdisjoint(forbidden), (
        f"Example 06 predictors contain forbidden leakage variables: {predictors & forbidden}"
    )

    # 2. Target must not predict itself
    assert "high_value_customer" not in predictors


def test_run_all_fast_mode(tmp_path: Path) -> None:
    """run_all.py --fast must succeed in isolated subprocess, report pass, and not leak IDs."""
    test_out = tmp_path / "fast_reports"
    cmd = [
        sys.executable,
        str(EXAMPLES / "run_all.py"),
        "--fast",
        "--output-dir",
        str(test_out),
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=300,
    )

    assert proc.returncode == 0, f"run_all.py --fast failed with stderr:\n{proc.stderr}"
    assert "RESULT: ALL 9 EXAMPLES PASSED." in proc.stdout
    assert "Traceback" not in proc.stdout
    assert "Traceback" not in proc.stderr

    for sample_id in SAMPLE_CUSTOMER_IDS:
        assert sample_id not in proc.stdout

    # Reports directory in examples/ must not have been created or modified
    assert not (EXAMPLES / "reports").exists(), "run_all.py must not write to examples/reports/"


def test_fast_mode_disclosure() -> None:
    """When fast mode is active, an explicit disclaimer must be printed."""
    result = _run_example("07_product_portfolio_repeated_measures.py", "--fast")
    assert result.returncode == 0
    assert "FAST DEMO/CI MODE: bootstrap intervals use reduced resamples" in result.stdout


def test_expanded_scientific_guardrails() -> None:
    """Ensure public example copy avoids unjustified causal and certainty claims."""
    text = "\n".join(
        (EXAMPLES / name).read_text(encoding="utf-8")
        for name in (*PUBLIC_SCRIPTS, "_customer_data.py", "run_all.py", "README.md")
    )

    for forbidden in (
        "purely additive",
        "pure additive",
        "proves additive",
        "interaction is zero",
        "guarantees that standard errors",
        "ensures that standard errors",
        "validates hc3",
    ):
        assert forbidden not in text.lower(), (
            f"Forbidden phrase '{forbidden}' found in example text"
        )


def test_canonical_show_used_across_examples() -> None:
    """Ensure public examples import and delegate statistical presentation to canonical show()."""
    for script_name in PUBLIC_SCRIPTS:
        content = (EXAMPLES / script_name).read_text(encoding="utf-8")
        assert "show" in content, f"{script_name} must import show"
        assert "show(" in content, f"{script_name} must call show()"

    # Specific detail mode policy assertions
    e02 = (EXAMPLES / "02_compare_customer_segments.py").read_text(encoding="utf-8")
    assert 'show(workflow, detail="standard")' in e02

    e05 = (EXAMPLES / "05_spend_drivers_regression.py").read_text(encoding="utf-8")
    assert 'show(workflow, detail="full")' in e05

    e09 = (EXAMPLES / "09_complete_research_workflow.py").read_text(encoding="utf-8")
    for expected_target in (
        "show(incomplete)",
        'show(plan, detail="standard")',
        'show(workflow, detail="full")',
        'show(sensitivity, detail="standard")',
        'show(practical, detail="standard")',
        'show(adherence, detail="standard")',
        'show(audit, detail="standard")',
        'show(record, detail="compact")',
        'show(replay, detail="compact")',
        'show(completeness, detail="compact")',
        'show(planning, detail="standard")',
        'show(snapshot, detail="compact")',
        'show(assistant.decision_ledger, detail="compact")',
        'show(paired, detail="standard")',
    ):
        assert expected_target in e09, (
            f"Missing lifecycle show call in Example 09: {expected_target}"
        )


def test_obsolete_manual_result_tables_removed() -> None:
    """Verify deprecated manual statistical table formatting has been excised from examples."""
    e01 = (EXAMPLES / "01_customer_360_profile.py").read_text(encoding="utf-8")
    assert "home_val_tab.get('counts')" not in e01
    assert "home_val_tab.get('row_percent')" not in e01

    e02 = (EXAMPLES / "02_compare_customer_segments.py").read_text(encoding="utf-8")
    assert "Primary Estimate (Mean Diff) :" not in e02
    assert "Test Statistic (t)           :" not in e02

    e03 = (EXAMPLES / "03_multigroup_customer_spending.py").read_text(encoding="utf-8")
    assert "Games-Howell Pairwise Follow-Ups (Sample of Comparisons)" not in e03
    assert "Welch F Statistic       :" not in e03

    e04 = (EXAMPLES / "04_customer_value_association.py").read_text(encoding="utf-8")
    assert "Pearson Chi-Square (X2) :" not in e04
    assert "Minimum Expected Frequency:" not in e04

    e05 = (EXAMPLES / "05_spend_drivers_regression.py").read_text(encoding="utf-8")
    assert "Coefficient Table (HC3 Robust Standard Errors)" not in e05
    assert "Overall Model F-Statistic :" not in e05

    e06 = (EXAMPLES / "06_high_value_customer_logistic.py").read_text(encoding="utf-8")
    assert "Odds Ratios & Wald Inference Table" not in e06
    assert "Likelihood Ratio X2       :" not in e06

    e07 = (EXAMPLES / "07_product_portfolio_repeated_measures.py").read_text(encoding="utf-8")
    assert "Within-Customer Condition Summaries" not in e07
    assert "Friedman Q Statistic    :" not in e07

    e08 = (EXAMPLES / "08_factorial_customer_segments.py").read_text(encoding="utf-8")
    assert "Cell Sample Sizes and Means (2x2 Design)" not in e08
    assert "Source / Term" not in e08
