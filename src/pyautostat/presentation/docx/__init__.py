"""Publication-ready editable Microsoft Word (DOCX) presentation package."""

from __future__ import annotations

from .api import check_docx_available, save_docx, to_docx

__all__ = [
    "check_docx_available",
    "save_docx",
    "to_docx",
]
