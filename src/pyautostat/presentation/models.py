"""Normalized terminal view models for PyAutoStat presentation.

These dataclasses decouple presentation rendering from statistical result structures.
They contain only display-safe, pre-formatted values derived from authoritative stored records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DisplayMetric:
    """A single labeled display metric with an optional semantic role and note."""

    label: str
    value: str
    role: str = "default"
    note: str | None = None


@dataclass(frozen=True)
class DisplayRow:
    """A row of formatted cell values for terminal table display."""

    cells: tuple[str, ...]


@dataclass(frozen=True)
class DisplayTable:
    """A tabular section with header columns and rows."""

    title: str | None
    columns: tuple[str, ...]
    rows: tuple[DisplayRow, ...]


@dataclass(frozen=True)
class DisplayDiagnostic:
    """A diagnostic cue or condition check for display."""

    label: str
    status: str
    detail: str | None = None
    severity: str = "neutral"


@dataclass(frozen=True)
class PresentationView:
    """Universal normalized presentation model passed to terminal and HTML renderers."""

    title: str
    subtitle: str | None
    family: str
    design_metrics: tuple[DisplayMetric, ...] = ()
    key_metrics: tuple[DisplayMetric, ...] = ()
    tables: tuple[DisplayTable, ...] = ()
    diagnostics: tuple[DisplayDiagnostic, ...] = ()
    interpretation: str | None = None
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    compact_text: str | None = None


# Backward-compatible alias for terminal renderers
TerminalView = PresentationView


__all__ = [
    "DisplayDiagnostic",
    "DisplayMetric",
    "DisplayRow",
    "DisplayTable",
    "PresentationView",
    "TerminalView",
]
