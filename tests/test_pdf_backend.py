"""Unit tests for PyAutoStat PDF generation backend and settings."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from pyautostat import ReportError
from pyautostat.presentation.pdf.backend import check_playwright_available, html_to_pdf_bytes
from pyautostat.presentation.pdf.settings import (
    get_footer_template,
    normalize_page_size,
)


def test_normalize_page_size():
    assert normalize_page_size("a4") == "A4"
    assert normalize_page_size("A4") == "A4"
    assert normalize_page_size(" A4  ") == "A4"
    assert normalize_page_size("letter") == "Letter"
    assert normalize_page_size("Letter") == "Letter"
    assert normalize_page_size(" LETTER ") == "Letter"

    with pytest.raises(ReportError, match="Invalid page size"):
        normalize_page_size("")

    with pytest.raises(ReportError, match="Invalid page size"):
        normalize_page_size(None)

    with pytest.raises(ReportError, match="Unsupported page size 'tabloid'"):
        normalize_page_size("tabloid")


def test_footer_template():
    active_footer = get_footer_template(page_numbers=True)
    assert "pageNumber" in active_footer
    assert "totalPages" in active_footer
    assert "color: #64748b" in active_footer

    disabled_footer = get_footer_template(page_numbers=False)
    assert disabled_footer == ""


def test_check_playwright_available_missing():
    with patch("importlib.util.find_spec", return_value=None):
        with pytest.raises(ReportError) as exc_info:
            check_playwright_available()
        msg = str(exc_info.value)
        assert 'pip install "pyautostat[pdf]"' in msg
        assert "python -m playwright install chromium" in msg


def test_check_playwright_available_present():
    with patch("importlib.util.find_spec", return_value=MagicMock()):
        # Should not raise
        check_playwright_available()


def test_html_to_pdf_bytes_chromium_missing_actionable_error():
    mock_playwright = MagicMock()

    # Playwright's Error subclass
    class PlaywrightError(Exception):
        pass

    launch_error = PlaywrightError(
        "Executable doesn't exist at C:\\path\\to\\chromium.exe\n"
        "Please run the following command to download new browsers:\n\n"
        "playwright install"
    )
    mock_playwright.chromium.launch.side_effect = launch_error

    mock_sync_playwright = MagicMock()
    mock_sync_playwright.return_value.__enter__.return_value = mock_playwright

    with patch("pyautostat.presentation.pdf.backend.check_playwright_available"):
        with patch.dict(
            sys.modules,
            {
                "playwright.sync_api": MagicMock(
                    sync_playwright=mock_sync_playwright,
                    Error=PlaywrightError,
                )
            },
        ):
            with pytest.raises(ReportError) as exc_info:
                html_to_pdf_bytes("<html><body>Hello</body></html>")

            msg = str(exc_info.value)
            assert "Chromium browser binary is missing." in msg
            assert "python -m playwright install chromium" in msg
            assert "C:\\path\\to\\chromium.exe" not in msg  # No raw traceback leakage


def test_html_to_pdf_bytes_figure_timeout_error():
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    mock_page.wait_for_function.side_effect = TimeoutError("Wait timeout")

    mock_playwright = MagicMock()
    mock_playwright.chromium.launch.return_value = mock_browser

    mock_sync_playwright = MagicMock()
    mock_sync_playwright.return_value.__enter__.return_value = mock_playwright

    with patch("pyautostat.presentation.pdf.backend.check_playwright_available"):
        with patch.dict(
            sys.modules,
            {
                "playwright.sync_api": MagicMock(
                    sync_playwright=mock_sync_playwright,
                    Error=Exception,
                )
            },
        ):
            with pytest.raises(ReportError) as exc_info:
                html_to_pdf_bytes(
                    "<html><body>Figure</body></html>",
                    wait_for_figures=True,
                )

            assert "Interactive figures did not finish rendering before PDF export" in str(
                exc_info.value
            )
            mock_browser.close.assert_called_once()


def test_html_to_pdf_bytes_figure_render_failure():
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    # wait_for_function passes, but page.evaluate detects data-pyautostat-error
    mock_page.wait_for_function.return_value = True
    mock_page.evaluate.return_value = True

    mock_playwright = MagicMock()
    mock_playwright.chromium.launch.return_value = mock_browser

    mock_sync_playwright = MagicMock()
    mock_sync_playwright.return_value.__enter__.return_value = mock_playwright

    with patch("pyautostat.presentation.pdf.backend.check_playwright_available"):
        with patch.dict(
            sys.modules,
            {
                "playwright.sync_api": MagicMock(
                    sync_playwright=mock_sync_playwright,
                    Error=Exception,
                )
            },
        ):
            with pytest.raises(ReportError) as exc_info:
                html_to_pdf_bytes(
                    "<html><body>Figure</body></html>",
                    wait_for_figures=True,
                )

            assert "Interactive figures encountered an error while rendering" in str(exc_info.value)
            mock_browser.close.assert_called_once()


def test_html_to_pdf_bytes_invalid_pdf_signature():
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    # Return invalid signature
    mock_page.pdf.return_value = b"NOT-A-PDF-DOCUMENT"

    mock_playwright = MagicMock()
    mock_playwright.chromium.launch.return_value = mock_browser

    mock_sync_playwright = MagicMock()
    mock_sync_playwright.return_value.__enter__.return_value = mock_playwright

    with patch("pyautostat.presentation.pdf.backend.check_playwright_available"):
        with patch.dict(
            sys.modules,
            {
                "playwright.sync_api": MagicMock(
                    sync_playwright=mock_sync_playwright,
                    Error=Exception,
                )
            },
        ):
            with pytest.raises(ReportError, match="valid %PDF signature"):
                html_to_pdf_bytes("<html><body>Hello</body></html>")

            mock_browser.close.assert_called_once()
