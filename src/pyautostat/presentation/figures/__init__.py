"""Plotly-independent scientific figure specification layer for PyAutoStat."""

from __future__ import annotations

from .adapters import build_figure_spec, build_figure_specs
from .api import save_static_figures, to_static_figures
from .models import FigureSeries, FigureSpec, StaticFigureArtifact

__all__ = [
    "FigureSeries",
    "FigureSpec",
    "StaticFigureArtifact",
    "build_figure_spec",
    "build_figure_specs",
    "save_static_figures",
    "to_static_figures",
]
