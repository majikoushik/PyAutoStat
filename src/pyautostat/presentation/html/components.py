"""Reusable HTML presentation components for PyAutoStat reports.

Pure functions producing semantic, accessible, well-escaped HTML strings
from normalized presentation models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..models import (
    DisplayDiagnostic,
    DisplayMetric,
    DisplayTable,
)
from .formatting import escape_text, is_numeric_column
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
        f'<h3 class="table-caption">{escape_text(table.title)}</h3>\n' if table.title else ""
    )

    return (
        '<div class="table-container">\n'
        f"{caption_html}"
        "<table>\n"
        f"{thead_html}\n"
        f"{tbody_html}\n"
        f"</table>{note_html}\n"
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
