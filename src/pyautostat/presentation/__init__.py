"""PyAutoStat Rich-based terminal presentation layer."""

from __future__ import annotations

from .adapters import UnsupportedPresentationError
from .api import show
from .models import (
    DisplayDiagnostic,
    DisplayMetric,
    DisplayRow,
    DisplayTable,
    TerminalView,
)
from .theme import PYAUTOSTAT_THEME

__all__ = [
    "DisplayDiagnostic",
    "DisplayMetric",
    "DisplayRow",
    "DisplayTable",
    "PYAUTOSTAT_THEME",
    "TerminalView",
    "UnsupportedPresentationError",
    "show",
]
