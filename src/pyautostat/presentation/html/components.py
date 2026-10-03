"""Reusable HTML presentation components for PyAutoStat reports.

Pure functions producing semantic, accessible, well-escaped HTML strings
from normalized presentation models.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from ..models import (
    DisplayDiagnostic,
    DisplayMetric,
    DisplayRow,
    DisplayTable,
)
from .formatting import escape_text, format_display_value, is_numeric_column
from .theme import get_theme_css


def render_page(
    title: str,
    body_html: str,
    *,
    style: str = "general",
    css: str | None = None,
) -> str:
    """Render a complete, self-contained HTML5 document wrapper."""
    theme_css = css if css is not None else get_theme_css()
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"  <title>{escape_text(title, quote=True)}</title>\n"
        f"  <style>{theme_css}</style>\n"
        "</head>\n"
        "<body>\n"
        '  <main class="report-container">\n'
        f"{body_html}\n"
        "  </main>\n"
        "</body>\n"
        "</html>\n"
    )


def render_header(
    title: str,
    subtitle: str | None = None,
    *,
    brand: str = "PyAutoStat Analysis Report",
) -> str:
    """Render the top report title panel with branding and optional subtitle."""
    subtitle_html = (
        f'  <p class="report-subtitle">{escape_text(subtitle)}</p>\n' if subtitle else ""
    )
    return (
        '<header class="report-header">\n'
        f'  <div class="report-brand">{escape_text(brand)}</div>\n'
        f'  <h1 class="report-title">{escape_text(title)}</h1>\n'
        f"{subtitle_html}"
        "</header>"
    )


def render_section(
    title: str,
    content_html: str,
    *,
    section_id: str | None = None,
    css_class: str = "",
) -> str:
    """Render a titled report section. Returns empty string if content is empty."""
    if not content_html or not content_html.strip():
        return ""
    id_attr = f' id="{escape_text(section_id, quote=True)}"' if section_id else ""
    class_attr = f"report-section {css_class}".strip()
    return (
        f'<section class="{class_attr}"{id_attr}>\n'
        f'  <h2 class="section-title">{escape_text(title)}</h2>\n'
        f"{content_html}\n"
        "</section>"
    )


def render_metric_cards(metrics: Sequence[DisplayMetric]) -> str:
    """Render key result metrics as a responsive card grid."""
    if not metrics:
        return ""

    cards: list[str] = []
    for m in metrics:
        role_class = "role-default"
        if m.role == "result.estimate":
            role_class = "role-estimate"
        elif m.role == "result.ci":
            role_class = "role-ci"
        elif m.role == "result.effect":
            role_class = "role-effect"
        elif m.role == "result.evidence":
            role_class = "role-evidence"

        val_unavail = m.value in ("Unavailable", "Not available")
        val_class = "metric-value value-unavailable" if val_unavail else "metric-value"
        note_html = (
            f'\n    <span class="metric-note">{escape_text(m.note)}</span>' if m.note else ""
        )

        cards.append(
            f'  <div class="metric-card {role_class}">\n'
            f'    <span class="metric-label">{escape_text(m.label)}</span>\n'
            f'    <span class="{val_class}">{escape_text(m.value)}</span>'
            f"{note_html}\n"
            "  </div>"
        )

    return '<div class="metric-grid">\n' + "\n".join(cards) + "\n</div>"


def render_design_grid(metrics: Sequence[DisplayMetric]) -> str:
    """Render analysis design and sample facts as a structured definition grid."""
    if not metrics:
        return ""

    items: list[str] = []
    for m in metrics:
        items.append(
            '  <div class="design-item">\n'
            f'    <dt class="design-label">{escape_text(m.label)}</dt>\n'
            f'    <dd class="design-value">{escape_text(m.value)}</dd>\n'
            "  </div>"
        )

    return '<dl class="design-grid">\n' + "\n".join(items) + "\n</dl>"


def render_table(table: DisplayTable, *, max_rows: int | None = None) -> str:
    """Render a semantic table with header scopes, responsive wrapper, and right-aligned numbers."""
    if not table or not table.rows:
        return ""

    th_cells: list[str] = []
    for i, col in enumerate(table.columns):
        align = "align-right" if is_numeric_column(col, i) else "align-left"
        th_cells.append(f'<th scope="col" class="{align}">{escape_text(col)}</th>')

    th_joined = "\n      ".join(th_cells)
    thead_html = f"  <thead>\n    <tr>\n      {th_joined}\n    </tr>\n  </thead>"

    total_rows = len(table.rows)
    rows_to_render = table.rows[:max_rows] if max_rows is not None else table.rows

    tbody_rows: list[str] = []
    for row in rows_to_render:
        td_cells: list[str] = []
        for i, cell in enumerate(row.cells):
            col_name = table.columns[i] if i < len(table.columns) else ""
            align = "align-right" if is_numeric_column(col_name, i) else "align-left"
            cell_unavail = cell in ("Unavailable", "Not available")
            td_class = f"{align} cell-unavailable" if cell_unavail else align
            td_cells.append(f'<td class="{td_class}">{escape_text(cell)}</td>')
        tbody_rows.append("    <tr>\n      " + "\n      ".join(td_cells) + "\n    </tr>")

    tbody_html = "  <tbody>\n" + "\n".join(tbody_rows) + "\n  </tbody>"

    note_html = ""
    if max_rows is not None and total_rows > max_rows:
        note_html = (
            f'\n  <p class="table-note">Showing {max_rows} of {total_rows} rows; '
            "use detail='full' for the complete table.</p>"
        )

    caption_html = (
        f'  <caption class="sr-only">{escape_text(table.title)}</caption>\n' if table.title else ""
    )

    return (
        '<div class="table-container">\n'
        '<div class="table-responsive">\n'
        '<table class="styled-table">\n'
        f"{caption_html}"
        f"{thead_html}\n"
        f"{tbody_html}\n"
        "</table>\n"
        "</div>"
        f"{note_html}\n"
        "</div>"
    )


def render_diagnostics(diagnostics: Sequence[DisplayDiagnostic]) -> str:
    """Render diagnostic status cues with explicit text badges."""
    if not diagnostics:
        return ""

    items: list[str] = []
    for diag in diagnostics:
        status_text = diag.status.upper()
        severity = diag.severity.lower()
        if severity == "success":
            badge_class = "status-success"
        elif severity == "error":
            badge_class = "status-error"
        elif severity in ("warning", "review"):
            badge_class = "status-warning" if severity == "warning" else "status-review"
        elif severity == "missing":
            badge_class = "status-missing"
        else:
            badge_class = "status-neutral"

        detail_text = escape_text(diag.detail) if diag.detail else ""
        escaped_st = escape_text(status_text)
        items.append(
            '  <div class="diagnostic-item">\n'
            f'    <span class="diagnostic-status {badge_class}">[{escaped_st}]</span>\n'
            f'    <span class="diagnostic-label">{escape_text(diag.label)}</span>\n'
            f'    <span class="diagnostic-detail">{detail_text}</span>\n'
            "  </div>"
        )

    return '<div class="diagnostic-list">\n' + "\n".join(items) + "\n</div>"


def render_interpretation(text: str | None) -> str:
    """Render deterministic interpretation narrative paragraphs."""
    if not text or not text.strip():
        return ""

    sanitized = (
        text.replace("\u2014", "--")
        .replace("\u2013", "-")
        .replace("\ufffd", "-")
        .replace("\u2212", "-")
    )
    paragraphs = [p.strip() for p in sanitized.split("\n\n") if p.strip()]
    if not paragraphs:
        return ""

    p_html = "\n".join(f"  <p>{escape_text(p)}</p>" for p in paragraphs)
    return f'<div class="interpretation-text">\n{p_html}\n</div>'


def render_limitations(limitations: Sequence[str]) -> str:
    """Render methodological limitations as an accessible bulleted list."""
    if not limitations:
        return ""

    li_html = "\n".join(f"    <li>{escape_text(lim)}</li>" for lim in limitations)
    return f'<div class="limitations-block">\n  <ul>\n{li_html}\n  </ul>\n</div>'


def render_warnings(warnings: Sequence[str]) -> str:
    """Render analysis warnings as a distinct caution block."""
    if not warnings:
        return ""

    li_html = "\n".join(f"    <li>{escape_text(w)}</li>" for w in warnings)
    return f'<div class="warnings-block">\n  <ul>\n{li_html}\n  </ul>\n</div>'


def render_analysis_record(metadata: dict[str, Any], detail: str = "standard") -> str:
    """Render structured analysis record and reproducibility metadata."""
    if not metadata:
        return ""

    items: list[str] = []
    # Key fields to render cleanly if present
    field_labels = [
        ("method_id", "Method Identifier"),
        ("status", "Analysis Status"),
        ("covariance_type", "Covariance Type"),
        ("multiplicity", "Multiplicity Control"),
        ("audit_status", "Consistency Audit"),
        ("package_version", "Package Version"),
    ]

    for key, label in field_labels:
        if key in metadata and metadata[key] is not None:
            val = metadata[key]
            items.append(
                '  <div class="record-item">\n'
                f'    <dt class="record-label">{escape_text(label)}</dt>\n'
                f'    <dd class="record-value">{escape_text(str(val))}</dd>\n'
                "  </div>"
            )

    # Reproducibility sub-dict
    repro = metadata.get("reproducibility")
    if isinstance(repro, dict):
        stoch = repro.get("stochastic", {})
        if isinstance(stoch, dict):
            seed = stoch.get("effective_seed")
            if seed is not None:
                items.append(
                    '  <div class="record-item">\n'
                    '    <dt class="record-label">Random Seed</dt>\n'
                    f'    <dd class="record-value">{escape_text(str(seed))}</dd>\n'
                    "  </div>"
                )
            resamples = stoch.get("bootstrap_resamples")
            if resamples is not None:
                items.append(
                    '  <div class="record-item">\n'
                    '    <dt class="record-label">Bootstrap Resamples</dt>\n'
                    f'    <dd class="record-value">{escape_text(str(resamples))}</dd>\n'
                    "  </div>"
                )
        spec_ref = repro.get("specification_reference")
        if spec_ref:
            items.append(
                '  <div class="record-item">\n'
                '    <dt class="record-label">Specification Reference</dt>\n'
                f'    <dd class="record-value">{escape_text(str(spec_ref))}</dd>\n'
                "  </div>"
            )

    if not items:
        return ""

    return '<dl class="record-grid">\n' + "\n".join(items) + "\n</dl>"


def _display_helper(value: Any, *, p_value: bool = False) -> str:
    """Format an arbitrary value for report display with small p-value handling."""
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return "Not available"
        if p_value and value == 0:
            return "p < 0.001 (computational zero)"
        return f"{value:.4g}" if isinstance(value, float) else str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return str(value)


def report_table_to_display_table(table_payload: dict[str, Any]) -> DisplayTable:
    """Convert a canonical report table payload dictionary into a DisplayTable."""
    title = table_payload.get("title")
    columns = tuple(str(c) for c in table_payload.get("columns", ()))
    rows: list[DisplayRow] = []
    for row in table_payload.get("rows", ()):
        cells: list[str] = []
        for item in row:
            val = item.get("value") if isinstance(item, dict) else item
            is_pval = isinstance(item, dict) and item.get("source") == "analysis.values.p_value"
            cells.append(_display_helper(val, p_value=is_pval))
        rows.append(DisplayRow(tuple(cells)))
    return DisplayTable(title=title, columns=columns, rows=tuple(rows))


def render_report_status(status: str) -> str:
    """Render report execution status with an accessible semantic badge."""
    status_lower = status.lower()
    if status_lower in ("completed", "available", "passed", "reproduced"):
        badge_class = "status-success"
    elif status_lower in ("needs_input", "draft", "warning", "review"):
        badge_class = "status-warning"
    elif status_lower in ("failed", "error", "unsupported"):
        badge_class = "status-error"
    else:
        badge_class = "status-neutral"
    status_upper = status.upper()
    return (
        '<p class="report-status"><strong>Report status:</strong> '
        f'<span class="diagnostic-status {badge_class}">[{escape_text(status_upper)}]</span> '
        f"{escape_text(status)}</p>"
    )


def render_executive_summary(paragraphs: Sequence[str]) -> str:
    """Render executive summary section with calm visual styling."""
    if not paragraphs:
        return ""
    paras_html = "\n".join(
        f"  <p>{escape_text(p, quote=True)}</p>" for p in paragraphs if p and p.strip()
    )
    return (
        '<section class="executive-summary">\n'
        "  <h2>Executive Summary</h2>\n"
        f"{paras_html}\n"
        "</section>"
    )


def render_report_section_dl(
    section_data: dict[str, Any],
    *,
    p_value: bool = False,
) -> str:
    """Render a structured definition list for canonical report sections."""
    if not section_data:
        return ""

    items: list[str] = []
    for label, val in section_data.items():
        if val is None or val == [] or val == {}:
            continue
        is_pval = p_value and (label == "p_value" or "p_value" in label)
        disp_val = _display_helper(val, p_value=is_pval)
        items.append(
            f"  <dt><strong>{escape_text(label)}</strong></dt><dd>{escape_text(disp_val)}</dd>"
        )

    if not items:
        return ""

    joined = "\n".join(items)
    return f"<dl>\n{joined}\n</dl>"


def render_practical_significance(practical: dict[str, Any] | Any) -> str:
    """Render practical significance findings, thresholds, and interval relations."""
    if not practical:
        return ""

    data = practical.to_dict() if hasattr(practical, "to_dict") else dict(practical)

    threshold_val = data.get("threshold")
    threshold_str = str(threshold_val) if threshold_val is not None else None
    estimate_val = data.get("estimate")
    estimate_str = format_display_value(estimate_val) if estimate_val is not None else None
    ci_val = data.get("confidence_interval")

    metrics: list[DisplayMetric] = []
    if threshold_str:
        metrics.append(DisplayMetric("Practical Threshold", threshold_str))
    if estimate_str:
        metrics.append(DisplayMetric("Observed Estimate", estimate_str))
    if isinstance(ci_val, dict):
        lower = ci_val.get("lower")
        upper = ci_val.get("upper")
        conf = ci_val.get("confidence_level", 0.95)
        pct = int(round(conf * 100)) if conf < 1.0 else int(conf)
        if lower is not None and upper is not None:
            metrics.append(DisplayMetric(f"{pct}% CI", f"[{lower:.4g}, {upper:.4g}]"))

    cards_html = render_metric_cards(metrics) if metrics else ""

    verdict = data.get("verdict")
    status_badge = ""
    if verdict:
        v_str = str(verdict).lower()
        if "meaningful" in v_str or "meets" in v_str or "superior" in v_str:
            badge_class = "status-success"
        elif "negligible" in v_str:
            badge_class = "status-warning"
        elif "inconclusive" in v_str:
            badge_class = "status-neutral"
        else:
            badge_class = "status-neutral"
        status_badge = (
            f'<span class="diagnostic-status {badge_class}">'
            f"[{escape_text(str(verdict).upper())}]</span>"
        )

    conclusion = data.get("conclusion")
    conc_html = (
        f'<p class="practical-conclusion">{status_badge} {escape_text(conclusion)}</p>'
        if conclusion
        else ""
    )

    dl_items = {}
    for k in (
        "quantity",
        "planning_status",
        "point_estimate_relationship",
        "confidence_interval_relationship",
        "statistical_evidence",
    ):
        if k in data and data[k] is not None:
            dl_items[k] = data[k]

    dl_html = render_report_section_dl(dl_items)
    parts = [p for p in (cards_html, conc_html, dl_html) if p]
    return "\n".join(parts)


def render_sensitivity(sensitivity: dict[str, Any] | Any) -> str:
    """Render sensitivity analysis scenario comparison metadata."""
    if not sensitivity:
        return ""

    data = sensitivity.to_dict() if hasattr(sensitivity, "to_dict") else dict(sensitivity)
    comparison_note = data.get("comparison_note")
    note_html = (
        f'<p class="sensitivity-note">{escape_text(comparison_note)}</p>' if comparison_note else ""
    )

    dl_items = {}
    for k in ("baseline_method", "scenarios_evaluated", "estimand_stability"):
        if k in data and data[k] is not None:
            dl_items[k] = data[k]

    dl_html = render_report_section_dl(dl_items)
    parts = [p for p in (note_html, dl_html) if p]
    return "\n".join(parts)
