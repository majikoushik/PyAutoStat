"""Plotly-independent scientific figure specification layer for PyAutoStat."""

from __future__ import annotations

from .adapters import build_figure_spec
from .models import FigureSeries, FigureSpec

__all__ = [
    "FigureSeries",
    "FigureSpec",
    "build_figure_spec",
]
