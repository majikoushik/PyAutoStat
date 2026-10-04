"""Plotly-independent scientific figure specification layer for PyAutoStat."""

from __future__ import annotations

from .adapters import build_figure_spec, build_figure_specs
from .api import save_static_figures, to_static_figures
from .models import (
    CANONICAL_PLACEMENTS,
    PLACEMENT_CELL_SUMMARY,
    PLACEMENT_COEFFICIENTS,
    PLACEMENT_CONTINGENCY,
    PLACEMENT_KEY_RESULTS,
    PLACEMENT_PAIRWISE,
    FigureSeries,
    FigureSpec,
    StaticFigureArtifact,
    normalize_figure_placement,
)

__all__ = [
    "CANONICAL_PLACEMENTS",
    "FigureSeries",
    "FigureSpec",
    "PLACEMENT_CELL_SUMMARY",
    "PLACEMENT_COEFFICIENTS",
    "PLACEMENT_CONTINGENCY",
    "PLACEMENT_KEY_RESULTS",
    "PLACEMENT_PAIRWISE",
    "StaticFigureArtifact",
    "build_figure_spec",
    "build_figure_specs",
    "normalize_figure_placement",
    "save_static_figures",
    "to_static_figures",
]
