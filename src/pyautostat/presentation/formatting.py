"""Formatters for terminal display values.

All formatters operate purely on stored numerical/categorical values without
recalculating inferential statistics. They guarantee display safety and consistent notation.
"""

from __future__ import annotations

import math
from typing import Any


def format_number(
    val: Any,
    decimals: int = 2,
    unit: str | None = None,
    default: str = "Unavailable",
) -> str:
    """Format a numerical value preserving sign and precision."""
    if val is None or isinstance(val, bool):
        return default
    if not isinstance(val, (int, float)):
        return str(val)
    if not math.isfinite(val):
        return default

    is_negative = val < 0
    abs_val = abs(val)

    if unit and ("$" in unit or "USD" in unit):
        # Currency formatting preserving negative sign correctly: e.g. -$47.70
        prefix = "-$" if is_negative else "$"
        return f"{prefix}{abs_val:,.{decimals}f}"

    if isinstance(val, int):
        num_str = f"{val:,}"
    else:
        num_str = f"{val:.{decimals}f}"

    if unit:
        return f"{num_str} {unit}"
    return num_str


def format_p_value(p: Any, default: str = "Unavailable") -> str:
    """Format a p-value into readable scientific presentation form."""
    if p is None or isinstance(p, bool):
        return default
    if not isinstance(p, (int, float)):
        return str(p)
    if not math.isfinite(p):
        return default

    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_confidence_interval(
    ci: Any,
    decimals: int = 2,
    default: str = "Unavailable",
    bracket: bool = False,
    ellipsis_sep: bool = False,
) -> str:
    """Format confidence interval bounds without reordering bounds."""
    if not isinstance(ci, dict):
        return default

    lower = ci.get("lower")
    upper = ci.get("upper")
    if lower is None or upper is None:
        return default
    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
        return default
    if not math.isfinite(lower) or not math.isfinite(upper):
        return default

    l_str = format_number(lower, decimals=decimals)
    u_str = format_number(upper, decimals=decimals)

    if ellipsis_sep:
        return f"{l_str}...{u_str}"
    if bracket:
        return f"[{l_str}, {u_str}]"
    return f"{l_str} to {u_str}"


def format_effect(
    effect: Any,
    decimals: int = 2,
    default: str = "Unavailable",
) -> str:
    """Format an effect size value from a scalar or effect dictionary."""
    if isinstance(effect, dict):
        val = effect.get("value")
    else:
        val = effect
    if val is None or not isinstance(val, (int, float)) or not math.isfinite(val):
        return default
    return format_number(val, decimals=decimals)


def format_sample_size(n: Any, default: str = "Unavailable") -> str:
    """Format sample sizes with thousands separators."""
    if n is None or isinstance(n, bool):
        return default
    if isinstance(n, (int, float)) and math.isfinite(n):
        return f"{int(n):,}"
    return str(n)


def format_percent(
    pct: Any,
    decimals: int = 1,
    default: str = "Unavailable",
) -> str:
    """Format percentage values."""
    if pct is None or isinstance(pct, bool):
        return default
    if isinstance(pct, (int, float)) and math.isfinite(pct):
        return f"{pct:.{decimals}f}%"
    return default


def format_memory_bytes(
    num_bytes: Any,
    default: str = "Unavailable",
) -> str:
    """Format byte counts into human-readable memory strings."""
    if num_bytes is None or isinstance(num_bytes, bool):
        return default
    if not isinstance(num_bytes, (int, float)) or not math.isfinite(num_bytes):
        return default
    if num_bytes < 1024:
        return f"{int(num_bytes)} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KiB"
    if num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MiB"
    return f"{num_bytes / (1024**3):.2f} GiB"


def format_status(status: str) -> str:
    """Format a status identifier into a readable uppercase token."""
    return status.upper().replace("_", " ")
