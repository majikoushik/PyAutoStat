"""Pure formatting helpers for beginner-facing validation and type guidance."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from difflib import get_close_matches
from typing import Any

COLUMN_DISPLAY_LIMIT = 12
LEVEL_DISPLAY_LIMIT = 10
SUPPORTED_VARIABLE_TYPES = (
    "continuous",
    "discrete",
    "nominal",
    "ordinal",
    "identifier",
    "boolean",
)

_TYPE_LABELS = {
    "continuous_numerical": "continuous numerical",
    "discrete_numerical": "discrete numerical",
    "nominal_categorical": "nominal categorical",
    "ordinal_categorical": "ordinal categorical",
    "identifier": "identifier",
    "boolean": "boolean",
    "datetime": "datetime-like",
    "unknown": "unknown",
}


def closest_column(requested: Any, columns: Iterable[Any]) -> Any | None:
    """Return one credible string-label match without altering the requested value."""
    if not isinstance(requested, str):
        return None
    candidates = [column for column in columns if isinstance(column, str)]
    matches = get_close_matches(requested, candidates, n=1, cutoff=0.72)
    return matches[0] if matches else None


def bounded_values(values: Sequence[Any], *, limit: int, noun: str) -> str:
    """Represent ordered values without dumping an unbounded schema or category list."""
    shown = list(values[:limit])
    rendered = ", ".join(repr(value) for value in shown)
    suffix = len(values) - len(shown)
    if suffix:
        rendered = f"{rendered} (+{suffix} more {noun})"
    return f"[{rendered}]"


def available_columns(columns: Iterable[Any]) -> str:
    """Return a bounded, original-order representation of DataFrame columns."""
    return bounded_values(list(columns), limit=COLUMN_DISPLAY_LIMIT, noun="columns")


def missing_column_message(argument: str, requested: Any, columns: Iterable[Any]) -> str:
    """Explain one invalid column argument and, where credible, suggest one correction."""
    ordered = list(columns)
    message = f"Column {requested!r} supplied for `{argument}` was not found."
    suggestion = closest_column(requested, ordered)
    if suggestion is not None:
        message += f" Did you mean {suggestion!r}?"
    if argument in {"variable_types", "data_dictionary", "reference_levels"}:
        action = f"Change the `{argument}` dictionary key to an existing column."
    else:
        action = f"Change `{argument}=` to an existing column."
    return f"{message} {action} Available columns: {available_columns(ordered)}."


def observed_levels(values: Iterable[Any]) -> str:
    """Represent observed category values in source order and with visible types."""
    return bounded_values(list(values), limit=LEVEL_DISPLAY_LIMIT, noun="levels")


def invalid_level_message(
    argument: str,
    requested: Any,
    column: str,
    levels: Iterable[Any],
) -> str:
    """Explain an invalid level without coercing or silently replacing it."""
    return (
        f"`{argument}` requested {requested!r} for column {column!r}, but that value is not "
        f"observed. Observed levels: {observed_levels(levels)}. Change `{argument}` to one of "
        "these exact values."
    )


def invalid_condition_order_message(
    invalid: Iterable[Any], column: str, levels: Iterable[Any]
) -> str:
    """Explain invalid condition labels and the scientific meaning of their order."""
    return (
        f"`condition_order` contains unobserved label(s) {observed_levels(list(invalid))} for "
        f"column {column!r}. Observed levels: {observed_levels(list(levels))}. Change "
        "`condition_order` to the exact observed labels; its order defines the signed contrast."
    )


def format_type_guidance(
    variable_suggestions: Mapping[str, Any],
    analytical_variable_types: Mapping[str, Any] | None,
    data_dictionary: Mapping[str, Any] | None,
) -> str:
    """Format stored type evidence without inspecting data or recalculating anything."""
    if not variable_suggestions:
        return "No analytical variables are selected, so no variable-type guidance is available."

    analytical = analytical_variable_types or {}
    dictionary = data_dictionary or {}
    supported = ", ".join(SUPPORTED_VARIABLE_TYPES)
    lines: list[str] = []
    for column, raw_info in variable_suggestions.items():
        info = raw_info if isinstance(raw_info, Mapping) else {}
        stored_type = analytical.get(column, info.get("suggested_type", "unknown"))
        label = _TYPE_LABELS.get(str(stored_type), str(stored_type).replace("_", " "))
        declared_entry = dictionary.get(column, {})
        declared = info.get("type_source") == "declared" or (
            isinstance(declared_entry, Mapping)
            and ("type" in declared_entry or "ordinal_order" in declared_entry)
        )
        if declared:
            lines.append(f"{column}: {label} - declared by the researcher.")
            continue

        dtype = info.get("observed_dtype", "unknown dtype")
        if info.get("suggested_type") == "identifier":
            lines.append(
                f"{column}: identifier - inferred advisory from {dtype} values and "
                f"identifier-like evidence; confirmation is required for "
                f"`variable_types.{column}` when requested by the workflow. This hint does "
                "not establish analytical meaning."
            )
        elif info.get("ambiguous_numeric_category"):
            lines.append(
                f"{column}: ambiguous numeric category - confirmation required for "
                f"`variable_types.{column}`. The {dtype} values have an inferred advisory "
                f"type of {label}, but could encode categories. Use "
                f"`variable_types={{'{column}': 'ordinal'}}` if the codes are ordered "
                f"categories, or choose the scientific meaning from: {supported}."
            )
        else:
            lines.append(f"{column}: {label} - inferred advisory from {dtype} values.")
    return "\n".join(lines)
