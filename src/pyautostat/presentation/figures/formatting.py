"""Formatting helpers for figure hover labels and scientific annotations."""

from __future__ import annotations

import math
from typing import Any


def format_hover_number(value: Any, decimals: int = 3) -> str:
    """Format numeric values for hover tooltips."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return "N/A"
        return f"{value:.{decimals}f}"
    return str(value)


def format_hover_ci(lower: Any, upper: Any, decimals: int = 3) -> str:
    """Format lower and upper CI bounds for hover tooltips."""
    low_str = format_hover_number(lower, decimals=decimals)
    up_str = format_hover_number(upper, decimals=decimals)
    return f"[{low_str}, {up_str}]"


__all__ = [
    "format_hover_ci",
    "format_hover_number",
]
