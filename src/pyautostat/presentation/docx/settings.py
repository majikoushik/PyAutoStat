"""Page setup and geometry settings for DOCX research reports."""

from __future__ import annotations

from typing import Any

from ...exceptions import ReportError

VALID_PAGE_SIZES: dict[str, str] = {
    "a4": "A4",
    "letter": "Letter",
}

DEFAULT_MARGINS_MM: float = 20.0


def normalize_page_size(page_size: Any) -> str:
    """Validate and normalize case-insensitive DOCX page size."""
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


def apply_section_geometry(
    section: Any,
    page_size: str,
    landscape: bool,
    margins_mm: float = DEFAULT_MARGINS_MM,
) -> None:
    """Apply standard dimensions, orientation, and margins to a Word document section."""
    from docx.enum.section import WD_ORIENT
    from docx.shared import Inches, Length, Mm

    norm_size = normalize_page_size(page_size)
    width: Length
    height: Length
    if norm_size == "A4":
        width = Mm(210)
        height = Mm(297)
    else:  # Letter
        width = Inches(8.5)
        height = Inches(11.0)

    if landscape:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width = height
        section.page_height = width
    else:
        section.orientation = WD_ORIENT.PORTRAIT
        section.page_width = width
        section.page_height = height

    margin_val = Mm(margins_mm)
    section.top_margin = margin_val
    section.bottom_margin = margin_val
    section.left_margin = margin_val
    section.right_margin = margin_val
