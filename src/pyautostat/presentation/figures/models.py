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
    placement: str = "key_results"
    layout_hints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StaticFigureArtifact:
    """An immutable static scientific figure artifact rendered from a FigureSpec.

    Contains rendered byte payload (e.g. PNG, SVG, PDF), library-controlled safe filename,
    dimensions, and placement metadata. Excludes all raw dataset rows.
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
    "FigureSeries",
    "FigureSpec",
    "StaticFigureArtifact",
]
