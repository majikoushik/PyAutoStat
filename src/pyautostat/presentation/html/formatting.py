"""HTML text escaping and cell alignment utilities for presentation."""

from __future__ import annotations

import html
import math
from typing import Any

NON_NUMERIC_COLUMNS = {
    "Levels",
    "Most common",
    "Most Common",
    "Variable",
    "Term",
    "Group",
    "Pair",
    "Item",
    "Condition",
    "Component",
    "Contrast",
    "Decision",
    "Status",
    "Measure",
    "Model",
    "Definition",
    "Notation",
    "Unit",
    "Source",
    "Variant",
    "Description",
}


def escape_text(value: Any, quote: bool = False) -> str:
    """Safely escape text for inclusion in HTML elements or attributes."""
    if value is None:
        return "Not available"
    return html.escape(str(value), quote=quote)


def is_numeric_column(col_name: str, col_idx: int) -> bool:
    """Determine if a table column represents numeric data and should be right-aligned."""
    if col_idx == 0:
        return False
    return col_name not in NON_NUMERIC_COLUMNS


def format_display_value(value: Any) -> str:
    """Format and escape an arbitrary statistical or descriptive value for display."""
    if value is None:
        return "Not available"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "Not available"
        return escape_text(f"{value:.4g}")
    return escape_text(str(value))
