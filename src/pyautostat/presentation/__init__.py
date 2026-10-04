"""PyAutoStat Rich-based terminal presentation layer."""

from __future__ import annotations

from .adapters import UnsupportedPresentationError, adapt
from .api import show
from .docx import save_docx, to_docx
from .figures import FigureSeries, FigureSpec, build_figure_spec, build_figure_specs
from .html import HtmlRenderer, save_html, save_interactive_html, to_html, to_interactive_html
from .models import (
    DisplayDiagnostic,
    DisplayMetric,
    DisplayRow,
    DisplayTable,
    PresentationView,
    TerminalView,
)
from .pdf import save_pdf, to_pdf
from .theme import PYAUTOSTAT_THEME

__all__ = [
    "DisplayDiagnostic",
    "DisplayMetric",
    "DisplayRow",
    "DisplayTable",
    "FigureSeries",
    "FigureSpec",
    "HtmlRenderer",
    "PYAUTOSTAT_THEME",
    "PresentationView",
    "TerminalView",
    "UnsupportedPresentationError",
    "adapt",
    "build_figure_spec",
    "build_figure_specs",
    "save_docx",
    "save_html",
    "save_interactive_html",
    "save_pdf",
    "show",
    "to_docx",
    "to_html",
    "to_interactive_html",
    "to_pdf",
]
