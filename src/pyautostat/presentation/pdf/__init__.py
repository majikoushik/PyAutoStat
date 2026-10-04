"""PDF export layer for PyAutoStat research workflows and reports."""

from .api import save_pdf, to_pdf
from .backend import check_playwright_available, html_to_pdf_bytes

__all__ = [
    "check_playwright_available",
    "html_to_pdf_bytes",
    "save_pdf",
    "to_pdf",
]
