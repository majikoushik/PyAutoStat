"""Browser-print PDF generation backend using Playwright and Chromium."""

from __future__ import annotations

import importlib.util
from typing import Any, cast

from ...exceptions import ReportError
from .settings import (
    DEFAULT_MARGINS,
    HEADER_TEMPLATE,
    get_footer_template,
    normalize_page_size,
)


def check_playwright_available() -> None:
    """Verify that the optional Playwright library is installed."""
    if importlib.util.find_spec("playwright") is None:
        raise ReportError(
            "PDF export requires the optional PDF dependency.\n\n"
            "Install:\n"
            '    pip install "pyautostat[pdf]"\n\n'
            "Then install Chromium:\n"
            "    python -m playwright install chromium"
        )


def html_to_pdf_bytes(
    html: str,
    *,
    page_size: str = "A4",
    landscape: bool = False,
    page_numbers: bool = True,
    wait_for_figures: bool = False,
    figure_timeout_seconds: float = 15.0,
) -> bytes:
    """Convert self-contained HTML document into PDF bytes using Chromium print renderer.

    Parameters
    ----------
    html : str
        Self-contained HTML report string.
    page_size : str, default="A4"
        Standard page format ('A4' or 'Letter', case-insensitive).
    landscape : bool, default=False
        Whether to print in landscape orientation.
    page_numbers : bool, default=True
        Whether to print running page numbers in the footer.
    wait_for_figures : bool, default=False
        Whether to wait for Plotly interactive figures to resolve before printing.
    figure_timeout_seconds : float, default=15.0
        Maximum time in seconds to wait for figures to finish rendering.

    Returns
    -------
    bytes
        Binary PDF data starting with the '%PDF' signature.
    """
    norm_size = normalize_page_size(page_size)
    check_playwright_available()

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ReportError(
            "PDF export requires the optional PDF dependency.\n\n"
            "Install:\n"
            '    pip install "pyautostat[pdf]"\n\n'
            "Then install Chromium:\n"
            "    python -m playwright install chromium"
        ) from exc

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
        except PlaywrightError as exc:
            msg = str(exc)
            if (
                "Executable doesn't exist" in msg
                or "playwright install" in msg
                or "download new browsers" in msg
            ):
                raise ReportError(
                    "Chromium browser binary is missing.\n\n"
                    "Install it with:\n"
                    "    python -m playwright install chromium"
                ) from exc
            raise ReportError(f"Failed to launch Chromium browser for PDF export: {exc}") from exc

        try:
            context = browser.new_context()
            page = context.new_page()

            # Enforce offline generation: block all external http/https requests
            page.route(
                "**/*",
                lambda route: (
                    route.abort()
                    if route.request.url.startswith(("http://", "https://"))
                    else route.continue_()
                ),
            )

            page.set_content(html, wait_until="load")

            if wait_for_figures:
                try:
                    page.wait_for_function(
                        """() => {
                            const expected = window.__pyautostatFiguresExpected || 0;
                            if (expected === 0) {
                                const graphs = document.querySelectorAll(
                                    '.pyautostat-plotly-graph'
                                );
                                if (graphs.length === 0) return true;
                                return Array.from(graphs).every(
                                    el => el.getAttribute('data-pyautostat-rendered') === 'true'
                                );
                            }
                            const rendered = window.__pyautostatFiguresRendered || 0;
                            return rendered >= expected;
                        }""",
                        timeout=figure_timeout_seconds * 1000,
                    )
                except Exception as exc:
                    raise ReportError(
                        "Interactive figures did not finish rendering before PDF export.\n"
                        "Try exporting without figures or inspect the interactive HTML output."
                    ) from exc

                has_render_error = page.evaluate(
                    """() => {
                        const graphs = document.querySelectorAll(
                            '.pyautostat-plotly-graph[data-pyautostat-error]'
                        );
                        return graphs.length > 0;
                    }"""
                )
                if has_render_error:
                    raise ReportError(
                        "Interactive figures encountered an error while rendering before PDF "
                        "export.\n"
                        "Try exporting without figures or inspect the interactive HTML output."
                    )

            footer = get_footer_template(page_numbers)
            pdf_bytes = page.pdf(
                format=norm_size,
                landscape=landscape,
                display_header_footer=page_numbers,
                header_template=HEADER_TEMPLATE,
                footer_template=footer,
                margin=cast(Any, DEFAULT_MARGINS),
                print_background=True,
            )

            if not pdf_bytes.startswith(b"%PDF"):
                raise ReportError("Generated output does not start with valid %PDF signature.")

            return bytes(pdf_bytes)

        finally:
            browser.close()
