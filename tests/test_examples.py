"""The published examples should run and preserve the package's scientific contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SCRIPTS = (
    "01_quick_start.py",
    "02_hypothesis_testing.py",
    "03_advanced_workflow.py",
)


def _run_example(filename: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(EXAMPLES / filename), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


@pytest.mark.parametrize(
    ("filename", "expected"),
    (
        ("01_quick_start.py", "PRIORITIZED INSIGHTS"),
        ("02_hypothesis_testing.py", "TRACEABLE SUMMARY"),
    ),
)
def test_introductory_examples_run_from_repository_root(filename, expected):
    result = _run_example(filename)

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert "0002-GTOKLU-YVY" not in result.stdout


def test_advanced_example_runs_full_lifecycle_and_writes_canonical_exports(tmp_path):
    output_dir = tmp_path / "advanced"
    result = _run_example("03_advanced_workflow.py", "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr
    for heading in (
        "STRUCTURED CLARIFICATION",
        "PLAN THE ESTIMAND",
        "COMPARE DECLARED SENSITIVITY",
        "PRACTICAL IMPORTANCE",
        "REPORT, AUDIT, REPLAY",
        "EXPLICIT PAIRED ANALYSIS",
    ):
        assert heading in result.stdout
    for evidence in (
        "Initial workflow status: needs_input",
        "Plan created after analysis: False",
        "comparability: same_estimand",
        "comparability: different_estimand",
        "Same-data replay: reproduced",
        "Identifier values are used for matching",
    ):
        assert evidence in result.stdout
    assert "0002-GTOKLU-YVY" not in result.stdout

    expected_files = {
        "customer_analysis.html",
        "customer_analysis.md",
        "customer_analysis.json",
        "decision_ledger.json",
        "session_snapshot.json",
    }
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    assert (output_dir / "customer_analysis_tables").is_dir()

    report = json.loads((output_dir / "customer_analysis.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert "CustomerID" not in (output_dir / "customer_analysis.html").read_text(encoding="utf-8")
    snapshot = json.loads((output_dir / "session_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["schema_version"] == 1


def test_bundled_workbook_matches_documented_shape_and_has_unique_pairing_ids():
    pytest.importorskip("openpyxl")
    frame = pd.read_excel(EXAMPLES / "CustomerDataset.xlsx")

    assert frame.shape == (5000, 40)
    assert frame["CustomerID"].notna().all()
    assert frame["CustomerID"].is_unique
    assert {
        "Gender",
        "TotalAvgMonthlySpend",
        "MonthlySpend_ProductA",
        "MonthlySpend_ProductB",
    } <= set(frame)


def test_example_copy_does_not_claim_diagnostics_change_the_estimand():
    text = "\n".join(
        (EXAMPLES / name).read_text(encoding="utf-8") for name in (*SCRIPTS, "README.md")
    ).lower()

    assert "picks the appropriate test based on your data" not in text
    assert "based on normality and sample size" not in text
    assert "if both methods agree the conclusion is more robust" not in text
    assert "bootstrap confidence intervals — for effect sizes, always" not in text
