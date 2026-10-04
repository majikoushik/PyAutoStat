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


def format_df(df: Any, default: str = "Unavailable") -> str:
    """Format degrees of freedom as an integer, float, or tuple string."""
    if df is None or isinstance(df, bool):
        return default
    if isinstance(df, (int, float)):
        if not math.isfinite(df):
            return default
        return str(int(df)) if df == int(df) else f"{df:.2f}"
    if isinstance(df, (list, tuple)):
        clean_parts = [format_df(part, default="") for part in df]
        if any(not p for p in clean_parts):
            return default
        return f"({', '.join(clean_parts)})"
    return str(df)


def format_statistic(
    name: str,
    val: Any,
    df: Any = None,
    decimals: int = 3,
    default: str = "Unavailable",
) -> str:
    """Format a test statistic with optional degrees of freedom."""
    num_str = format_number(val, decimals=decimals, default=default)
    if num_str == default:
        return default
    if df is not None:
        df_str = format_df(df, default="")
        if df_str:
            return f"{name}{df_str} = {num_str}"
    return f"{name} = {num_str}"


def format_odds_ratio(
    val: Any,
    decimals: int = 3,
    default: str = "Unavailable",
) -> str:
    """Format an odds ratio value, explicitly handling infinite values."""
    if val is None or isinstance(val, bool):
        return default
    if isinstance(val, (int, float)):
        if math.isinf(val):
            return "Infinite" if val > 0 else "-Infinite"
        if not math.isfinite(val):
            return default
        return f"{val:.{decimals}f}"
    return str(val)


def format_ratio(
    val: Any,
    decimals: int = 2,
    default: str = "Unavailable",
) -> str:
    """Format a ratio value."""
    if val is None or isinstance(val, bool):
        return default
    if isinstance(val, (int, float)) and math.isfinite(val):
        return f"{val:.{decimals}f}"
    return str(val)


def format_probability(
    val: Any,
    decimals: int = 3,
    default: str = "Unavailable",
) -> str:
    """Format a probability or rate value strictly bounded in [0, 1]."""
    if val is None or isinstance(val, bool):
        return default
    if isinstance(val, (int, float)) and math.isfinite(val):
        return f"{val:.{decimals}f}"
    return str(val)


def format_boolean_status(
    val: Any,
    true_label: str = "Yes",
    false_label: str = "No",
    default: str = "Unavailable",
) -> str:
    """Format a boolean flag into human-readable text."""
    if val is True:
        return true_label
    if val is False:
        return false_label
    return default


def format_confidence_level_label(
    ci: Any = None,
    *,
    confidence_level: float | None = None,
    prefix: str | None = None,
    suffix: str = "CI",
) -> str:
    """Format a dynamic confidence interval label reflecting stored confidence level.

    Follows precedence:
    1. Stored 'level' in the interval object/dictionary itself.
    2. Explicit fallback 'confidence_level' (e.g. from analysis specification or metadata).
    3. Neutral fallback label without assuming 95%.
    """

    def _parse_level(val: Any) -> float | None:
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
            if 0 < val < 1:
                return float(val)
            return None
        elif isinstance(val, str):
            s = val.strip().rstrip("%").strip()
            try:
                num = float(s)
                if 0 < num < 1:
                    return num
                if 1 < num < 100:
                    return num / 100.0
            except ValueError:
                return None
        return None

    level: float | None = None
    if isinstance(ci, dict):
        level = _parse_level(ci.get("level"))
    elif ci is not None:
        level = _parse_level(getattr(ci, "level", None))

    if level is None and confidence_level is not None:
        level = _parse_level(confidence_level)

    parts: list[str] = []
    if prefix:
        parts.append(prefix)

    if level is not None:
        pct = level * 100
        parts.append(f"{pct:g}%")

    if suffix:
        parts.append(suffix)

    result = " ".join(parts).strip()
    return result if result else "CI"


def confidence_interval_phrase(
    ci: Any = None,
    *,
    confidence_level: float | None = None,
    plural: bool = False,
) -> str:
    """Format a dynamic confidence interval phrase for titles and descriptions.

    Follows precedence:
    1. Stored 'level' in the interval object/dictionary itself.
    2. Explicit fallback 'confidence_level' (e.g. from analysis or specification).
    3. Neutral fallback ("confidence interval" / "confidence intervals") without assuming 95%.
    """
    suffix = "confidence intervals" if plural else "confidence interval"
    return format_confidence_level_label(
        ci,
        confidence_level=confidence_level,
        suffix=suffix,
    )


def resolve_table_row_limit(table: Any, detail: str = "standard") -> int | None:
    """Determine the maximum number of rows to display for a table across formats.

    Guarantees stable source order and identical standard-mode row selection
    between HTML and DOCX without ranking by p-value or effect size.

    Rules:
    - full detail: always None (show all rows)
    - standard detail: 6 rows if the table title indicates pairwise comparison
      or the table has more than 10 rows; otherwise None.
    - compact detail: 6 rows if pairwise or long; otherwise None.
    """
    if detail == "full":
        return None

    rows = getattr(table, "rows", None)
    title = getattr(table, "title", None) or ""
    num_rows = len(rows) if rows is not None else 0
    is_pairwise = "PAIRWISE" in title.upper()
    is_long = num_rows > 10

    if detail in ("standard", "compact"):
        if is_pairwise or is_long:
            return 6
        return None

    return None
