"""PyAutoStat Rich-based terminal presentation layer."""

from __future__ import annotations

from .adapters import UnsupportedPresentationError, adapt
from .api import show
from .figures import FigureSeries, FigureSpec, build_figure_spec
from .html import HtmlRenderer, save_html, save_interactive_html, to_html, to_interactive_html
from .models import (
    DisplayDiagnostic,
    DisplayMetric,
    DisplayRow,
    DisplayTable,
    PresentationView,
    TerminalView,
)
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
    "save_html",
    "save_interactive_html",
    "show",
    "to_html",
    "to_interactive_html",
]
