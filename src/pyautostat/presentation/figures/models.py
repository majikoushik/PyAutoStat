"""Figure specification models for PyAutoStat scientific figures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FigureSeries:
    """A series of points, intervals, categories, or values in a FigureSpec.

    Contains display-ready, pre-calculated numeric values extracted from
    authoritative stored results. No inferential recalculation or raw
    observations are stored here.
    """

    label: str
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    x: tuple[float, ...] = ()
    y: tuple[float, ...] = ()
    categories: tuple[str, ...] = ()
    values: tuple[float, ...] = ()
    role: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


PLACEMENT_KEY_RESULTS = "key_results"
PLACEMENT_COEFFICIENTS = "coefficients"
PLACEMENT_PAIRWISE = "pairwise"
PLACEMENT_CONTINGENCY = "contingency"
PLACEMENT_CELL_SUMMARY = "cell_summary"

CANONICAL_PLACEMENTS: tuple[str, ...] = (
    PLACEMENT_KEY_RESULTS,
    PLACEMENT_COEFFICIENTS,
    PLACEMENT_PAIRWISE,
    PLACEMENT_CONTINGENCY,
    PLACEMENT_CELL_SUMMARY,
)

_LEGACY_PLACEMENT_MAP: dict[str, str] = {
    "KEY RESULTS": PLACEMENT_KEY_RESULTS,
    "KEY_RESULTS": PLACEMENT_KEY_RESULTS,
    "RESULTS": PLACEMENT_KEY_RESULTS,
    "MODEL COEFFICIENTS": PLACEMENT_COEFFICIENTS,
    "COEFFICIENTS": PLACEMENT_COEFFICIENTS,
    "PREDICTOR ODDS RATIOS": PLACEMENT_COEFFICIENTS,
    "PAIRWISE": PLACEMENT_PAIRWISE,
    "PAIRWISE COMPARISONS": PLACEMENT_PAIRWISE,
    "CONTINGENCY": PLACEMENT_CONTINGENCY,
    "CONTINGENCY TABLE": PLACEMENT_CONTINGENCY,
    "CELL SUMMARIES": PLACEMENT_CELL_SUMMARY,
    "CELL_SUMMARIES": PLACEMENT_CELL_SUMMARY,
    "CELL SUMMARY": PLACEMENT_CELL_SUMMARY,
    "CELL_SUMMARY": PLACEMENT_CELL_SUMMARY,
}


def normalize_figure_placement(value: str | None) -> str:
    """Normalize a figure placement label to a canonical placement identifier.

    Canonical identifiers:
    - 'key_results': Associated with key metrics / results section.
    - 'coefficients': Associated with model coefficients / estimates table.
    - 'pairwise': Associated with pairwise comparisons table.
    - 'contingency': Associated with contingency / crosstab table.
    - 'cell_summary': Associated with factorial cell summary table.
    """
    if not value:
        return PLACEMENT_KEY_RESULTS
    stripped = value.strip()
    val_upper = stripped.upper()
    if val_upper in _LEGACY_PLACEMENT_MAP:
        return _LEGACY_PLACEMENT_MAP[val_upper]
    val_lower = stripped.lower()
    if val_lower in CANONICAL_PLACEMENTS:
        return val_lower
    return val_lower


@dataclass(frozen=True)
class FigureSpec:
    """A Plotly-independent, immutable scientific figure specification.

    Represents the visual and numerical structure of a supplementary
    scientific figure. Fully decoupled from Plotly and any concrete rendering engine.
    """

    kind: str
    title: str
    subtitle: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    reference_value: float | None = None
    series: tuple[FigureSeries, ...] = ()
    note: str | None = None
    placement: str = PLACEMENT_KEY_RESULTS
    layout_hints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StaticFigureArtifact:
    """An immutable static scientific figure artifact rendered from a FigureSpec.

    Contains rendered byte payload (e.g. PNG, SVG, PDF), library-controlled safe filename,
    dimensions, and canonical placement metadata ('key_results', 'coefficients',
    'pairwise', 'contingency', 'cell_summary'). Excludes all raw dataset rows.
    """

    index: int
    kind: str
    title: str
    placement: str
    format: str
    data: bytes
    filename: str
    width: int | None = None
    height: int | None = None


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
    "normalize_figure_placement",
]
