"""Unit and API tests for PyAutoStat research export bundle creation and saving."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from pyautostat import (
    AnalysisOptions,
    ReportError,
    ResearchAssistant,
    ResearchWorkflowResult,
    save_bundle,
    to_bundle,
    verify_bundle,
)
from pyautostat.bundle.assembler import normalize_requested_formats
from pyautostat.bundle.models import (
    ALL_BUNDLE_FORMATS,
    BUNDLE_SCHEMA_VERSION,
    CANONICAL_FORMAT_ORDER,
    DEFAULT_BUNDLE_FORMATS,
)
from pyautostat.presentation import DisplayMetric, PresentationView


@pytest.fixture
def welch_workflow() -> ResearchWorkflowResult:
    df = pd.DataFrame(
        {
            "group": ["A"] * 15 + ["B"] * 15,
            "score": [10.0, 11.0, 10.5, 12.0, 11.5] * 3 + [14.0, 15.0, 14.5, 16.0, 15.5] * 3,
        }
    )
    return ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
        options=AnalysisOptions(confidence_level=0.95),
    )


def test_bundle_constants():
    assert BUNDLE_SCHEMA_VERSION == 1
    assert DEFAULT_BUNDLE_FORMATS == ("html", "json", "csv")
    assert "html" in ALL_BUNDLE_FORMATS
    assert "docx" in ALL_BUNDLE_FORMATS
    assert "pdf" in ALL_BUNDLE_FORMATS
    assert "latex" in ALL_BUNDLE_FORMATS
    assert CANONICAL_FORMAT_ORDER[0] == "html"
    assert CANONICAL_FORMAT_ORDER[-1] == "latex"


def test_normalize_requested_formats():
    assert normalize_requested_formats(None) == ("html", "json", "csv")
    assert normalize_requested_formats(["json", "html"]) == ("html", "json")
    assert normalize_requested_formats(["csv", "csv", "json"]) == ("json", "csv")

    with pytest.raises(ReportError, match="At least one format"):
        normalize_requested_formats([])

    with pytest.raises(ReportError, match="Unsupported bundle format"):
        normalize_requested_formats(["html", "xlsx"])

    with pytest.raises(ReportError, match="Invalid bundle format"):
        normalize_requested_formats(["html", "   "])


def test_to_bundle_workflow_returns_valid_zip_bytes(welch_workflow: ResearchWorkflowResult):
    bundle_bytes = to_bundle(welch_workflow)
    assert isinstance(bundle_bytes, bytes)
    assert bundle_bytes.startswith(b"PK")

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        names = zf.namelist()
        assert "pyautostat_bundle/manifest.json" in names
        assert "pyautostat_bundle/README.md" in names
        assert "pyautostat_bundle/report/report.html" in names
        assert "pyautostat_bundle/data/report.json" in names
        assert "pyautostat_bundle/provenance/analysis.json" in names


def test_save_bundle_creates_zip_and_parent_dirs(
    welch_workflow: ResearchWorkflowResult, tmp_path: Path
):
    dest = tmp_path / "nested" / "exports" / "analysis_bundle.zip"
    saved = save_bundle(welch_workflow, dest)

    assert saved == dest.resolve()
    assert dest.is_file()
    assert dest.read_bytes().startswith(b"PK")


def test_save_bundle_enforces_zip_extension(welch_workflow: ResearchWorkflowResult, tmp_path: Path):
    with pytest.raises(ReportError, match="must have a '.zip' extension"):
        save_bundle(welch_workflow, tmp_path / "bundle.tar")

    with pytest.raises(ReportError, match="must have a '.zip' extension"):
        save_bundle(welch_workflow, tmp_path / "bundle")


def test_save_bundle_overwrite_protection(welch_workflow: ResearchWorkflowResult, tmp_path: Path):
    dest = tmp_path / "bundle.zip"
    save_bundle(welch_workflow, dest)

    with pytest.raises(ReportError, match="already exists and overwrite=False"):
        save_bundle(welch_workflow, dest, overwrite=False)

    # Overwrite=True succeeds
    saved = save_bundle(welch_workflow, dest, overwrite=True)
    assert saved.is_file()


def test_research_report_convenience_methods(
    welch_workflow: ResearchWorkflowResult, tmp_path: Path
):
    report = welch_workflow.report
    assert report is not None

    bundle_bytes = report.to_bundle(formats=("html", "json"))
    assert bundle_bytes.startswith(b"PK")

    saved_formats: list[str] = []
    report._on_save = lambda fmt: saved_formats.append(fmt)

    dest = tmp_path / "report_bundle.zip"
    saved = report.save_bundle(dest, formats=("html", "json"))
    assert saved.is_file()
    assert saved_formats == ["bundle"]

    # Save via top-level function delegating to report
    dest2 = tmp_path / "report_bundle2.zip"
    save_bundle(report, dest2)
    assert dest2.is_file()
    assert saved_formats == ["bundle", "bundle"]


def test_analysis_result_direct_bundle(welch_workflow: ResearchWorkflowResult):
    analysis = welch_workflow.analysis
    assert analysis is not None

    # Supported formats for direct AnalysisResult
    bundle_bytes = to_bundle(analysis, formats=("html", "json"))
    assert bundle_bytes.startswith(b"PK")

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        names = zf.namelist()
        assert "pyautostat_bundle/manifest.json" in names
        assert "pyautostat_bundle/README.md" in names
        assert "pyautostat_bundle/report/report.html" in names
        assert "pyautostat_bundle/data/analysis.json" in names
        assert "pyautostat_bundle/provenance/analysis.json" in names

    # Incompatible format request
    with pytest.raises(ReportError, match="requires a ResearchReport"):
        to_bundle(analysis, formats=("html", "csv"))

    with pytest.raises(ReportError, match="requires a ResearchReport"):
        to_bundle(analysis, formats=("markdown",))


def test_presentation_view_direct_bundle():
    view = PresentationView(
        title="Custom Presentation View",
        subtitle=None,
        family="comparison",
        key_metrics=(DisplayMetric(label="Metric A", value="1.23"),),
    )

    # HTML is supported
    bundle_bytes = to_bundle(view, formats=("html",))
    assert bundle_bytes.startswith(b"PK")

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        names = zf.namelist()
        assert "pyautostat_bundle/manifest.json" in names
        assert "pyautostat_bundle/report/report.html" in names

    # Unsupported format for PresentationView
    with pytest.raises(ReportError, match="not supported for PresentationView"):
        to_bundle(view, formats=("json",))


def test_bundle_options_propagation(welch_workflow: ResearchWorkflowResult):
    bundle_bytes = to_bundle(
        welch_workflow,
        formats=("html", "json"),
        detail="compact",
        style="apa",
        title="Custom Bundle Title",
    )
    check = verify_bundle(bundle_bytes)
    assert check.valid

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        readme = zf.read("pyautostat_bundle/README.md").decode("utf-8")
        assert "Detail Mode:** compact" in readme
        assert "Style Mode:** apa" in readme
