"""Tests for static scientific figure export public APIs, formats, dimensions, and errors."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pypdf
import pytest
from method_factories import (
    make_welch_t_workflow,
)

from pyautostat import (
    ReportError,
    ResearchAssistant,
    save_static_figures,
    to_static_figures,
)
from pyautostat.presentation.figures.models import StaticFigureArtifact
from pyautostat.presentation.figures.static import (
    validate_static_dimensions,
)


def test_no_figures_target_returns_empty_tuple_without_chrome_requirement():
    """Methods or targets without figure specs must return () without checking Chrome."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    asst = ResearchAssistant(df)
    profile = asst.profile()

    with patch("pyautostat.presentation.figures.static.check_chrome_available") as mock_check:
        figs = to_static_figures(profile)
        assert figs == ()
        mock_check.assert_not_called()

        paths = save_static_figures(profile, "nonexistent_dir/")
        assert paths == ()
        mock_check.assert_not_called()


def test_invalid_detail_raises_value_error():
    wf = make_welch_t_workflow()
    with pytest.raises(ValueError, match="Invalid detail mode 'invalid'"):
        to_static_figures(wf, detail="invalid")


def test_invalid_format_raises_report_error():
    wf = make_welch_t_workflow()
    with pytest.raises(ReportError, match="Unsupported static figure format: 'jpeg'"):
        to_static_figures(wf, format="jpeg")
    with pytest.raises(ReportError, match="Unsupported static figure format: 'eps'"):
        to_static_figures(wf, format="eps")


def test_dimension_validation_safeguards():
    # Valid dimensions
    w, h, scale = validate_static_dimensions(800, 600, 2.0)
    assert (w, h, scale) == (800, 600, 2.0)

    # Boolean rejected as int
    with pytest.raises(ReportError, match="width must be a positive integer"):
        validate_static_dimensions(True, 600, 2.0)
    with pytest.raises(ReportError, match="height must be a positive integer"):
        validate_static_dimensions(800, False, 2.0)

    # Negative / zero dimensions
    with pytest.raises(ReportError, match="width must be a positive integer"):
        validate_static_dimensions(0, 600, 2.0)
    with pytest.raises(ReportError, match="height must be a positive integer"):
        validate_static_dimensions(800, -50, 2.0)

    # Upper safeguard
    with pytest.raises(ReportError, match="width exceeds maximum allowed dimension"):
        validate_static_dimensions(15000, 600, 2.0)
    with pytest.raises(ReportError, match="height exceeds maximum allowed dimension"):
        validate_static_dimensions(800, 15000, 2.0)

    # Scale validation
    with pytest.raises(ReportError, match="scale must be a finite positive number"):
        validate_static_dimensions(800, 600, 0.0)
    with pytest.raises(ReportError, match="scale must be a finite positive number"):
        validate_static_dimensions(800, 600, -1.0)
    with pytest.raises(ReportError, match="scale exceeds maximum allowed limit"):
        validate_static_dimensions(800, 600, 10.0)


def test_welch_t_static_figure_png():
    wf = make_welch_t_workflow()
    figs = to_static_figures(wf, format="png")
    assert len(figs) >= 1
    fig = figs[0]
    assert isinstance(fig, StaticFigureArtifact)
    assert fig.index == 1
    assert fig.format == "png"
    assert fig.filename == "figure_01_estimate_ci.png"
    assert fig.placement == "key_results"
    assert fig.data.startswith(b"\x89PNG\r\n\x1a\n")


def test_welch_t_static_figure_svg():
    wf = make_welch_t_workflow()
    figs = to_static_figures(wf, format="svg")
    assert len(figs) >= 1
    fig = figs[0]
    assert fig.format == "svg"
    assert fig.filename == "figure_01_estimate_ci.svg"
    svg_text = fig.data.decode("utf-8")
    assert "<svg" in svg_text
    assert "</svg>" in svg_text


def test_welch_t_static_figure_pdf():
    wf = make_welch_t_workflow()
    figs = to_static_figures(wf, format="pdf")
    assert len(figs) >= 1
    fig = figs[0]
    assert fig.format == "pdf"
    assert fig.filename == "figure_01_estimate_ci.pdf"
    assert fig.data.startswith(b"%PDF")
    reader = pypdf.PdfReader(io.BytesIO(fig.data))
    assert len(reader.pages) >= 1


def test_save_static_figures_overwrite_protection(tmp_path: Path):
    wf = make_welch_t_workflow()
    fig_dir = tmp_path / "figs"

    saved_paths = save_static_figures(wf, fig_dir, format="png", overwrite=False)
    assert len(saved_paths) >= 1
    assert all(p.exists() for p in saved_paths)
    assert all(p.suffix == ".png" for p in saved_paths)

    # Attempting to save again without overwrite must raise ReportError
    with pytest.raises(ReportError, match="already exists and overwrite=False"):
        save_static_figures(wf, fig_dir, format="png", overwrite=False)

    # With overwrite=True, succeeds
    saved_again = save_static_figures(wf, fig_dir, format="png", overwrite=True)
    assert len(saved_again) == len(saved_paths)


def test_missing_dependencies_actionable_errors():
    from choreographer.errors import ChromeNotFoundError

    from pyautostat.presentation.figures.static import (
        check_chrome_available,
    )

    wf = make_welch_t_workflow()

    with patch(
        "pyautostat.presentation.figures.static.check_figure_dependencies",
        side_effect=ReportError(
            "Static figure export requires the optional figure dependency.\n"
            'Install: pip install "pyautostat[figures]"'
        ),
    ):
        with pytest.raises(
            ReportError, match="Static figure export requires the optional figure dependency"
        ):
            to_static_figures(wf)

    with patch(
        "choreographer.browsers.chromium.Chromium.find_browser",
        side_effect=ChromeNotFoundError("No Chrome"),
    ):
        with pytest.raises(ReportError, match="plotly_get_chrome"):
            check_chrome_available()


def test_production_figure_pdf_validation_does_not_require_pypdf(monkeypatch: pytest.MonkeyPatch):
    """Production PDF byte validation must not import or require pypdf."""
    import sys

    from pyautostat.presentation.figures.static import validate_rendered_bytes

    # Block pypdf import
    monkeypatch.setitem(sys.modules, "pypdf", None)

    # Valid %PDF header succeeds without requiring pypdf
    validate_rendered_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF", "pdf")

    # Invalid header raises ReportError
    with pytest.raises(ReportError, match="Generated figure PDF data is missing %PDF header"):
        validate_rendered_bytes(b"NOT_A_PDF_STREAM", "pdf")


def test_check_chrome_available_preflight_variants(tmp_path: Path):
    """Test find_browser return value validation in check_chrome_available."""
    from choreographer.errors import ChromeNotFoundError

    from pyautostat.presentation.figures.static import check_chrome_available

    # 1. find_browser raises ChromeNotFoundError -> ReportError
    with patch(
        "choreographer.browsers.chromium.Chromium.find_browser",
        side_effect=ChromeNotFoundError("Browser not found"),
    ):
        with pytest.raises(ReportError, match="plotly_get_chrome"):
            check_chrome_available()

    # 2. find_browser returns None -> ReportError
    with patch(
        "choreographer.browsers.chromium.Chromium.find_browser",
        return_value=None,
    ):
        with pytest.raises(ReportError, match="plotly_get_chrome"):
            check_chrome_available()

    # 3. find_browser returns nonexistent path -> ReportError
    nonexistent = str(tmp_path / "nonexistent" / "chrome.exe")
    with patch(
        "choreographer.browsers.chromium.Chromium.find_browser",
        return_value=nonexistent,
    ):
        with pytest.raises(ReportError, match="plotly_get_chrome"):
            check_chrome_available()

    # 4. find_browser returns existing path -> succeeds
    real_mock_file = tmp_path / "mock_chrome.exe"
    real_mock_file.write_text("binary")
    with patch(
        "choreographer.browsers.chromium.Chromium.find_browser",
        return_value=str(real_mock_file),
    ):
        # Must succeed without error
        check_chrome_available()
