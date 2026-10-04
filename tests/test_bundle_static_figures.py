"""Tests for bundling standalone static scientific figures and all-format integrity verification."""

from __future__ import annotations

import io
import json
import zipfile
from unittest.mock import patch

import pandas as pd
import pytest
from method_factories import make_welch_t_workflow

from pyautostat import (
    AnalysisOptions,
    ResearchAssistant,
    save_bundle,
    to_bundle,
    verify_bundle,
)


def test_bundle_default_does_not_include_static_figures():
    wf = make_welch_t_workflow()

    with patch("pyautostat.presentation.figures.static.check_chrome_available") as mock_chrome:
        bundle_bytes = to_bundle(wf)
        mock_chrome.assert_not_called()

        with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
            fig_members = [n for n in zf.namelist() if "figures/" in n]
            assert len(fig_members) == 0

        res = verify_bundle(bundle_bytes)
        assert res.valid is True


def test_bundle_packaging_static_png_figures():
    wf = make_welch_t_workflow()
    bundle_bytes = to_bundle(wf, include_static_figures=True, static_figure_format="png")

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        fig_members = [n for n in zf.namelist() if n.startswith("pyautostat_bundle/figures/")]
        assert len(fig_members) >= 1
        assert any(n.endswith(".png") for n in fig_members)

        # Validate PNG signature
        for f in fig_members:
            data = zf.read(f)
            assert data.startswith(b"\x89PNG\r\n\x1a\n")

        # Validate manifest records
        manifest_raw = zf.read("pyautostat_bundle/manifest.json").decode("utf-8")
        manifest = json.loads(manifest_raw)
        assert manifest["include_static_figures"] is True

        fig_records = [r for r in manifest["files"] if r["role"] == "scientific_figure"]
        assert len(fig_records) == len(fig_members)
        assert all(r["format"] == "png" for r in fig_records)

    # Full cryptographic verification
    result = verify_bundle(bundle_bytes)
    assert result.valid is True
    assert result.errors == ()


def test_bundle_packaging_static_svg_figures_with_privacy():
    sentinel_id = "SECRET_BUNDLE_PARTICIPANT_9999"
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "score": [10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            "raw_notes": [f"{sentinel_id}_{i}" for i in range(10)],
        }
    )

    assistant = ResearchAssistant(df)
    wf = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
        options=AnalysisOptions(confidence_level=0.95),
    )

    bundle_bytes = to_bundle(wf, include_static_figures=True, static_figure_format="svg")
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        fig_members = [n for n in zf.namelist() if n.startswith("pyautostat_bundle/figures/")]
        assert len(fig_members) >= 1
        for f in fig_members:
            svg_text = zf.read(f).decode("utf-8")
            assert sentinel_id not in svg_text
            assert "<svg" in svg_text

    result = verify_bundle(bundle_bytes)
    assert result.valid is True


try:
    from pyautostat.presentation.pdf import check_playwright_available

    check_playwright_available()
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        browser.close()
    HAS_CHROMIUM = True
except Exception:
    HAS_CHROMIUM = False


def test_bundle_multi_format_with_static_figures(tmp_path):
    """Test multi-format bundle containing HTML, interactive HTML, DOCX, CSV, JSON, Markdown,
    LaTeX, and static figures.
    """
    wf = make_welch_t_workflow()
    formats = (
        "html",
        "interactive_html",
        "docx",
        "json",
        "csv",
        "markdown",
        "latex",
    )

    bundle_path = tmp_path / "research_bundle_figures.zip"
    saved = save_bundle(
        wf,
        bundle_path,
        formats=formats,
        include_figures=True,
        include_static_figures=True,
        static_figure_format="png",
        overwrite=True,
    )

    assert saved.exists()
    result = verify_bundle(saved)
    assert result.valid is True
    assert result.errors == ()

    with zipfile.ZipFile(saved) as zf:
        names = zf.namelist()
        assert "pyautostat_bundle/report/report.html" in names
        assert "pyautostat_bundle/report/report.docx" in names
        assert "pyautostat_bundle/data/report.json" in names
        assert any(n.startswith("pyautostat_bundle/figures/") for n in names)


@pytest.mark.skipif(
    not HAS_CHROMIUM, reason="Playwright/Chromium not installed for PDF bundle export"
)
def test_bundle_all_publication_formats_with_static_figures(tmp_path):
    """Test full publication bundle with HTML, interactive HTML, PDF, DOCX, and static figures."""
    wf = make_welch_t_workflow()
    all_formats = (
        "html",
        "interactive_html",
        "pdf",
        "docx",
        "json",
        "csv",
        "markdown",
        "latex",
    )

    bundle_path = tmp_path / "complete_research_bundle.zip"
    saved = save_bundle(
        wf,
        bundle_path,
        formats=all_formats,
        include_figures=True,
        include_static_figures=True,
        static_figure_format="png",
        overwrite=True,
    )

    assert saved.exists()
    result = verify_bundle(saved)
    assert result.valid is True
    assert result.errors == ()
    assert result.checked_files >= 10

    with zipfile.ZipFile(saved) as zf:
        names = zf.namelist()
        assert "pyautostat_bundle/report/report.html" in names
        assert "pyautostat_bundle/report/report.pdf" in names
        assert "pyautostat_bundle/report/report.docx" in names
        assert "pyautostat_bundle/data/report.json" in names
        assert any(n.startswith("pyautostat_bundle/figures/") for n in names)
