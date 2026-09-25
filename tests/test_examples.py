"""The published showcase should run from the repository root without input files."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "example_usage.py"
RENAMED_EXAMPLES = (
    "reproducibility_example.py",
    "sensitivity_and_practical_significance_example.py",
    "planning_and_paired_analysis_example.py",
)


def _run_showcase(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )


@pytest.mark.parametrize("filename", RENAMED_EXAMPLES)
def test_feature_named_examples_run_from_repository_root(filename):
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / filename)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )

    assert result.returncode == 0, result.stderr


def test_showcase_runs_every_feature_and_writes_reports(tmp_path):
    output_dir = tmp_path / "showcase"
    result = _run_showcase(
        "--sample-size", "60", "--output-dir", str(output_dir), "--skip-interactive"
    )

    assert result.returncode == 0, result.stderr
    for heading in (
        "analyze_all(): every analysis section",
        "detect_column_types() and suggest_column_roles()",
        "InsightEngine: severity-rated findings",
        "hypothesis_tests(): selection",
        "categorical_association(): chi-square",
        "ReportGenerator: dict, JSON, CSV, HTML",
        "Bad-data and error-handling examples",
    ):
        assert heading in result.stdout
    for error in (
        "InvalidDataError",
        "ColumnNotFoundError",
        "InsufficientGroupsError",
        "InsufficientDataError",
        "InvalidTestError",
        "ReportError",
    ):
        assert f"{error}:" in result.stdout
    assert "Interactive HTML skipped" in result.stdout
    assert not (output_dir / "interactive.html").exists()
    assert (output_dir / "analysis.html").exists()
    assert set(path.name for path in (output_dir / "csv").iterdir()) == {
        "descriptive_stats.csv",
        "outliers.csv",
        "missing_data.csv",
        "insights.csv",
        "hypothesis_tests.csv",
    }
    report = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
    assert len(report["hypothesis_tests"]) == 8
    assert "analysis_warnings" in report["analysis"]


def test_showcase_rejects_too_small_sample(tmp_path):
    result = _run_showcase("--sample-size", "10", "--output-dir", str(tmp_path / "unused"))
    assert result.returncode != 0
    assert "--sample-size must be at least 60" in result.stderr
    assert not (tmp_path / "unused").exists()


def test_showcase_writes_interactive_report_when_plotly_available(tmp_path):
    pytest.importorskip("plotly")
    output_dir = tmp_path / "showcase"
    result = _run_showcase("--sample-size", "60", "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr
    interactive_html = (output_dir / "interactive.html").read_text(encoding="utf-8")
    assert "Correlation Heatmap" in interactive_html
    assert "Plotly.newPlot" in interactive_html
    assert "to_interactive_html() returned" in result.stdout
    assert "Open interactive.html" in result.stdout
