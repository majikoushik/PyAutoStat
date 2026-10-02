"""Adapters for descriptive dataset profiles, frequency tables, and cross-tabulations."""

from __future__ import annotations

from typing import Any

from ..formatting import (
    format_memory_bytes,
    format_number,
    format_percent,
    format_sample_size,
)
from ..models import (
    DisplayDiagnostic,
    DisplayMetric,
    DisplayRow,
    DisplayTable,
    TerminalView,
)


def is_profile(target: Any) -> bool:
    """Return True if target looks like a DatasetProfiler result dictionary."""
    if not isinstance(target, dict):
        return False
    return "overview" in target and ("descriptive" in target or "data_quality" in target)


def is_frequency_table(target: Any) -> bool:
    """Return True if target looks like a frequency_table dictionary."""
    if not isinstance(target, dict):
        return False
    return "column" in target and "levels" in target and "valid_n" in target


def is_cross_tab(target: Any) -> bool:
    """Return True if target looks like a cross_tabulation dictionary."""
    if not isinstance(target, dict):
        return False
    return "row_variable" in target and "column_variable" in target and "counts" in target


def adapt_profile(profile: dict[str, Any], detail: str = "standard") -> TerminalView:
    """Adapt a dataset profile dictionary into a TerminalView."""
    overview = profile.get("overview", {})
    total_rows = overview.get("total_rows", 0)
    total_cols = overview.get("total_columns", 0)

    missing = profile.get("missing_data", {})
    missing_cells = missing.get("total_missing_cells", overview.get("missing_cells", 0))
    missing_pct = missing.get(
        "overall_missing_percentage", overview.get("missing_cell_percentage", 0.0)
    )

    quality = profile.get("data_quality", {})
    duplicate_rows = quality.get("duplicate_rows", overview.get("duplicate_rows", 0))
    mem_bytes = overview.get(
        "memory_usage_bytes", profile.get("resource_info", {}).get("estimated_memory_bytes")
    )

    design_metrics = (
        DisplayMetric("Rows", format_sample_size(total_rows)),
        DisplayMetric("Columns", format_sample_size(total_cols)),
        DisplayMetric("Missing cells", f"{format_sample_size(missing_cells)} ({missing_pct:.1f}%)"),
        DisplayMetric("Duplicate rows", format_sample_size(duplicate_rows)),
        DisplayMetric("Memory", format_memory_bytes(mem_bytes)),
    )

    # Tables
    tables: list[DisplayTable] = []

    # 1. Numeric Variables
    descriptive = profile.get("descriptive", {})
    missing_by_col = missing.get("by_column", {})
    num_rows: list[DisplayRow] = []

    num_items = list(descriptive.items())
    max_num = len(num_items) if detail == "full" else min(10, len(num_items))
    for col, stats in num_items[:max_num]:
        if isinstance(stats, dict):
            n = format_sample_size(stats.get("count"))
            mean = format_number(stats.get("mean"))
            median = format_number(stats.get("median"))
            sd = format_number(stats.get("std"))
            miss_count = format_sample_size(missing_by_col.get(col, {}).get("count", 0))
            num_rows.append(DisplayRow((str(col), n, mean, median, sd, miss_count)))

    if num_rows:
        num_title = "NUMERIC VARIABLES"
        if detail != "full" and len(num_items) > max_num:
            num_title += f" (showing {max_num} of {len(num_items)})"
        tables.append(
            DisplayTable(
                title=num_title,
                columns=("Variable", "N", "Mean", "Median", "SD", "Missing"),
                rows=tuple(num_rows),
            )
        )

    # 2. Categorical Variables
    categorical = profile.get("categorical_summary", {})
    cat_rows: list[DisplayRow] = []
    cat_items = list(categorical.items())
    max_cat = len(cat_items) if detail == "full" else min(10, len(cat_items))

    for col, info in cat_items[:max_cat]:
        if isinstance(info, dict):
            levels = format_sample_size(info.get("observed_categories"))
            modes = info.get("mode_values", [])
            most_common = str(modes[0]) if modes else "None"
            # Truncate long mode strings if necessary
            if len(most_common) > 25:
                most_common = most_common[:22] + "..."
            miss_count = format_sample_size(info.get("missing_count", 0))
            cat_rows.append(DisplayRow((str(col), levels, most_common, miss_count)))

    if cat_rows:
        cat_title = "CATEGORICAL VARIABLES"
        if detail != "full" and len(cat_items) > max_cat:
            cat_title += f" (showing {max_cat} of {len(cat_items)})"
        tables.append(
            DisplayTable(
                title=cat_title,
                columns=("Variable", "Levels", "Most common", "Missing"),
                rows=tuple(cat_rows),
            )
        )

    # Review Cues / Issues
    issues = quality.get("issues", [])
    diagnostics_list: list[DisplayDiagnostic] = []
    max_issues = len(issues) if detail == "full" else min(8, len(issues))
    for issue in issues[:max_issues]:
        if isinstance(issue, dict):
            sev = issue.get("severity", "review")
            col = issue.get("column") or issue.get("section") or "Data Quality"
            msg = issue.get("message", "")
            diagnostics_list.append(
                DisplayDiagnostic(
                    label=str(col),
                    status="Review" if sev == "review" else "Info",
                    detail=msg,
                    severity=sev,
                )
            )

    compact_text = (
        f"Dataset Profile | {total_rows:,} rows x {total_cols:,} vars | "
        f"missing={missing_cells:,} ({missing_pct:.1f}%) | duplicates={duplicate_rows:,}"
    )

    return TerminalView(
        title="Dataset Profile",
        subtitle=f"{total_rows:,} rows x {total_cols:,} variables",
        family="family.profile",
        design_metrics=design_metrics,
        key_metrics=(),
        tables=tuple(tables),
        diagnostics=tuple(diagnostics_list),
        interpretation=None,
        limitations=(),
        warnings=(),
        metadata={"total_rows": total_rows, "total_columns": total_cols},
        compact_text=compact_text,
    )


def adapt_frequency_table(freq_dict: dict[str, Any], detail: str = "standard") -> TerminalView:
    """Adapt a frequency_table dictionary into a TerminalView."""
    column = freq_dict.get("column", "Variable")
    a_type = freq_dict.get("analytical_type", "categorical")
    valid_n = freq_dict.get("valid_n", 0)
    missing_n = freq_dict.get("missing_n", 0)
    total_n = freq_dict.get("total_n", valid_n + missing_n)
    levels = freq_dict.get("levels", [])

    design_metrics = (
        DisplayMetric("Variable", str(column)),
        DisplayMetric("Analytical Type", str(a_type)),
        DisplayMetric("Valid Observations", f"{format_sample_size(valid_n)} rows"),
        DisplayMetric("Missing Values", f"{format_sample_size(missing_n)} rows"),
        DisplayMetric("Total Rows", f"{format_sample_size(total_n)} rows"),
    )

    has_cumulative = any("cumulative_percent" in row for row in levels)
    cols: tuple[str, ...]
    if has_cumulative:
        cols = ("Level / Category", "Count", "Valid %", "Total %", "Cumulative %")
    else:
        cols = ("Level / Category", "Count", "Valid %", "Total %")

    max_rows = None if detail == "full" else 15
    rows_to_use = levels if max_rows is None else levels[:max_rows]

    display_rows: list[DisplayRow] = []
    for r in rows_to_use:
        lvl = str(r.get("level", ""))
        cnt = format_sample_size(r.get("count"))
        pct = format_percent(r.get("percent"))
        tpct = format_percent(r.get("total_percent"))
        if has_cumulative:
            cpct = format_percent(r.get("cumulative_percent"))
            display_rows.append(DisplayRow((lvl, cnt, pct, tpct, cpct)))
        else:
            display_rows.append(DisplayRow((lvl, cnt, pct, tpct)))

    tables = [
        DisplayTable(
            title=f"FREQUENCY TABLE: {column}"
            if len(levels) <= 15 or detail == "full"
            else f"FREQUENCY TABLE: {column} (first 15)",
            columns=cols,
            rows=tuple(display_rows),
        )
    ]

    diagnostics = []
    if freq_dict.get("high_cardinality"):
        diagnostics.append(
            DisplayDiagnostic(
                label="High Cardinality",
                status="ADVISORY",
                detail=f"Variable has {len(levels)} levels; inspect aggregation before modeling.",
                severity="review",
            )
        )

    narrative = freq_dict.get("narrative")
    compact_text = (
        f"Frequency Table | {column} | Levels={len(levels)} | "
        f"N={format_sample_size(valid_n)} | Missing={format_sample_size(missing_n)}"
    )

    return TerminalView(
        title="Categorical Frequency Distribution",
        subtitle=f"{column} ({len(levels)} levels)",
        family="family.descriptive",
        design_metrics=design_metrics,
        key_metrics=(),
        tables=tuple(tables),
        diagnostics=tuple(diagnostics),
        interpretation=narrative,
        limitations=(),
        warnings=(),
        metadata=freq_dict,
        compact_text=compact_text,
    )


def adapt_cross_tab(crosstab_dict: dict[str, Any], detail: str = "standard") -> TerminalView:
    """Adapt a cross_tabulation dictionary into a TerminalView."""
    row_var = crosstab_dict.get("row_variable", "Row")
    col_var = crosstab_dict.get("column_variable", "Column")
    valid_n = crosstab_dict.get("valid_n", 0)
    excluded = crosstab_dict.get("excluded_rows", 0)

    design_metrics = (
        DisplayMetric("Row Variable", str(row_var)),
        DisplayMetric("Column Variable", str(col_var)),
        DisplayMetric("Valid Paired N", f"{format_sample_size(valid_n)} observations"),
        DisplayMetric("Excluded / Missing", f"{format_sample_size(excluded)} rows"),
    )

    row_levels = crosstab_dict.get("row_levels", [])
    col_levels = crosstab_dict.get("column_levels", [])
    counts = crosstab_dict.get("counts", [])

    tables: list[DisplayTable] = []
    if counts and row_levels and col_levels:
        cols = [f"{row_var} \\ {col_var}"] + [str(c) for c in col_levels] + ["Total"]
        t_rows: list[DisplayRow] = []
        for r_idx, r_name in enumerate(row_levels):
            row_vals = counts[r_idx]
            tot = sum(row_vals)
            cells = (
                [str(r_name)]
                + [format_sample_size(c) for c in row_vals]
                + [format_sample_size(tot)]
            )
            t_rows.append(DisplayRow(tuple(cells)))
        tables.append(
            DisplayTable(title="CROSS-TABULATION (COUNTS)", columns=tuple(cols), rows=tuple(t_rows))
        )

    if detail == "full":
        # Row percentages table
        row_pct = crosstab_dict.get("row_percent", [])
        if row_pct:
            cols = [f"{row_var} \\ {col_var}"] + [str(c) for c in col_levels]
            t_rows = []
            for r_idx, r_name in enumerate(row_levels):
                cells = [str(r_name)] + [format_percent(p) for p in row_pct[r_idx]]
                t_rows.append(DisplayRow(tuple(cells)))
            tables.append(
                DisplayTable(title="ROW PERCENTAGES (%)", columns=tuple(cols), rows=tuple(t_rows))
            )

    narrative = crosstab_dict.get("narrative")
    compact_text = (
        f"Cross-Tab | {row_var} x {col_var} | N={format_sample_size(valid_n)} | "
        f"Shape={len(row_levels)}x{len(col_levels)}"
    )

    return TerminalView(
        title="Cross-Tabulation",
        subtitle=f"{row_var} x {col_var}",
        family="family.descriptive",
        design_metrics=design_metrics,
        key_metrics=(),
        tables=tuple(tables),
        diagnostics=(),
        interpretation=narrative,
        limitations=(),
        warnings=(),
        metadata=crosstab_dict,
        compact_text=compact_text,
    )
