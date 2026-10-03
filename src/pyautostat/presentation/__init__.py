"""PyAutoStat Rich-based terminal presentation layer."""

from __future__ import annotations

from .adapters import UnsupportedPresentationError, adapt
from .api import show
from .html import HtmlRenderer, save_html, to_html
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
    "HtmlRenderer",
    "PYAUTOSTAT_THEME",
    "PresentationView",
    "TerminalView",
    "UnsupportedPresentationError",
    "adapt",
    "save_html",
    "show",
    "to_html",
]
