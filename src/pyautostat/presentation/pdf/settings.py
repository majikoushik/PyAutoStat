"""Settings and page format definitions for publication-ready PDF export."""

from __future__ import annotations

from typing import Any

from ...exceptions import ReportError

VALID_PAGE_SIZES: dict[str, str] = {
    "a4": "A4",
    "letter": "Letter",
}

DEFAULT_MARGINS: dict[str, str] = {
    "top": "20mm",
    "bottom": "20mm",
    "left": "15mm",
    "right": "15mm",
}

HEADER_TEMPLATE: str = '<div style="font-size: 8pt; width: 100%;"></div>'


def normalize_page_size(page_size: Any) -> str:
    """Validate and normalize case-insensitive PDF page size."""
    if not isinstance(page_size, str) or not page_size.strip():
        raise ReportError(
            f"Invalid page size {page_size!r}. Expected a non-empty string such as 'A4' "
            "or 'Letter'."
        )
    normalized = page_size.strip().lower()
    if normalized not in VALID_PAGE_SIZES:
        valid_options = ", ".join(repr(v) for v in sorted(VALID_PAGE_SIZES.values()))
        raise ReportError(
            f"Unsupported page size {page_size!r}. Supported formats are: {valid_options}."
        )
    return VALID_PAGE_SIZES[normalized]


def get_footer_template(page_numbers: bool = True) -> str:
    """Return Chromium print footer template with accessible page numbers."""
    if not page_numbers:
        return ""
    return (
        "<div style=\"font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
        "Helvetica, Arial, sans-serif; font-size: 8pt; color: #64748b; width: 100%; "
        'text-align: center; padding-bottom: 4mm;">'
        'Page <span class="pageNumber"></span> of <span class="totalPages"></span>'
        "</div>"
    )
