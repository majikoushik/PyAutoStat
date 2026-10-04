"""Reusable semantic Word document components for PyAutoStat presentations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..html.formatting import is_numeric_column
from ..models import DisplayDiagnostic, DisplayMetric, DisplayTable
from .ooxml import (
    set_cell_shading,
    set_keep_with_next,
    set_row_cant_split,
    set_table_borders,
    set_table_cell_margins,
    set_table_header_row,
)


def clean_text_repr(s: Any) -> str:
    """Clean stringified Python dict/list reprs into reader-facing text."""
    if not isinstance(s, str):
        return format_display_value(s)
    s_strip = s.strip()
    if (s_strip.startswith("{") and s_strip.endswith("}")) or (
        s_strip.startswith("[") and s_strip.endswith("]")
    ):
        try:
            import ast

            parsed = ast.literal_eval(s_strip)
            return format_display_value(parsed)
        except Exception:
            pass
    if "['" in s_strip or '["' in s_strip:
        for delim in ("['", '["'):
            if delim in s_strip:
                idx = s_strip.find(delim)
                prefix = s_strip[:idx].strip()
                list_part = s_strip[idx:]
                try:
                    import ast

                    parsed = ast.literal_eval(list_part)
                    clean_list = format_display_value(parsed)
                    return f"{prefix} {clean_list}".strip()
                except Exception:
                    pass
    if "{'" in s_strip or '{"' in s_strip:
        for delim in ("{'", '{"'):
            if delim in s_strip:
                idx = s_strip.find(delim)
                prefix = s_strip[:idx].strip()
                dict_part = s_strip[idx:]
                try:
                    import ast

                    parsed = ast.literal_eval(dict_part)
                    clean_dict = format_display_value(parsed)
                    return f"{prefix} {clean_dict}".strip()
                except Exception:
                    pass
    return s_strip if s_strip else "—"


def format_display_value(val: Any) -> str:
    """Format an arbitrary value for reader-facing presentation without raw Python repr."""
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, (list, tuple, set)):
        if not val:
            return "—"
        items = [format_display_value(x) for x in val if x is not None]
        return ", ".join(items) if items else "—"
    if isinstance(val, dict):
        if not val:
            return "—"
        pairs = []
        for k, v in val.items():
            if v is None or v == "" or v == [] or v == {}:
                continue
            k_clean = str(k).replace("_", " ").title()
            v_clean = format_display_value(v)
            pairs.append(f"{k_clean}: {v_clean}")
        return "; ".join(pairs) if pairs else "—"
    if isinstance(val, str):
        return clean_text_repr(val)
    s = str(val).strip()
    return s if s else "—"


def format_cell_text(val: Any) -> str:
    """Format a cell value while strictly preserving falsey numeric and boolean data."""
    return format_display_value(val)


def add_report_header(
    doc: Any,
    title: str,
    subtitle: str | None = None,
    style_mode: str = "general",
) -> None:
    """Add standardized document title and subtitle blocks."""
    p_title = doc.add_paragraph(title, style="PyAutoStat Title")
    set_keep_with_next(p_title)

    if subtitle:
        p_sub = doc.add_paragraph(subtitle, style="PyAutoStat Subtitle")
        set_keep_with_next(p_sub)


def add_status(doc: Any, status: str) -> None:
    """Add a structured report status indication."""
    p = doc.add_paragraph()
    set_keep_with_next(p)
    r_label = p.add_run("Report Status: ")
    r_label.bold = True
    r_val = p.add_run(status.capitalize())
    if status.lower() == "complete":
        from docx.shared import RGBColor

        r_val.font.color.rgb = RGBColor(22, 101, 52)  # green
    elif status.lower() in ("partial", "needs_input", "data_limited"):
        from docx.shared import RGBColor

        r_val.font.color.rgb = RGBColor(146, 64, 14)  # amber


def add_section_heading(doc: Any, heading: str, level: int = 1) -> None:
    """Add an accessible section heading with keep-with-next pagination protection."""
    h = doc.add_heading(heading, level=level)
    set_keep_with_next(h)


def add_metric_grid(doc: Any, metrics: Sequence[DisplayMetric]) -> None:
    """Render primary key metrics as an editable Word card/grid table."""
    if not metrics:
        return

    n_metrics = len(metrics)
    cols = min(n_metrics, 4)
    rows_count = (n_metrics + cols - 1) // cols

    table = doc.add_table(rows=rows_count, cols=cols)
    set_table_borders(table, color_hex="E2E8F0", sz="4", val="single")
    set_table_cell_margins(table, top_dxa=100, bottom_dxa=100, left_dxa=140, right_dxa=140)

    for i, m in enumerate(metrics):
        r_idx = i // cols
        c_idx = i % cols
        cell = table.cell(r_idx, c_idx)
        set_cell_shading(cell, "F8FAFC")

        p_lbl = cell.paragraphs[0]
        p_lbl.style = "PyAutoStat Metric Label"
        p_lbl.text = m.label

        cell.add_paragraph(clean_text_repr(m.value), style="PyAutoStat Metric Value")
        if m.note:
            p_note = cell.add_paragraph(clean_text_repr(m.note))
            p_note.paragraph_format.space_before = 0
            p_note.paragraph_format.space_after = 0
            if p_note.runs:
                from docx.shared import Pt, RGBColor

                p_note.runs[0].font.size = Pt(8.5)
                p_note.runs[0].font.color.rgb = RGBColor(100, 116, 139)

    for row in table.rows:
        set_row_cant_split(row)

    # Empty spacer paragraph after table
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = 6


def add_design_table(doc: Any, metrics: Sequence[DisplayMetric]) -> None:
    """Render research design facts as a clean two-column definition table."""
    if not metrics:
        return

    table = doc.add_table(rows=len(metrics) + 1, cols=2)
    set_table_borders(table, color_hex="CBD5E1", sz="4", val="single")
    set_table_cell_margins(table, top_dxa=80, bottom_dxa=80, left_dxa=120, right_dxa=120)

    # Header
    hdr = table.rows[0]
    set_table_header_row(hdr)
    set_row_cant_split(hdr)

    hdr.cells[0].paragraphs[0].text = "Design Factor"
    hdr.cells[0].paragraphs[0].style = "PyAutoStat Table Header"
    set_cell_shading(hdr.cells[0], "F1F5F9")

    hdr.cells[1].paragraphs[0].text = "Specification / Fact"
    hdr.cells[1].paragraphs[0].style = "PyAutoStat Table Header"
    set_cell_shading(hdr.cells[1], "F1F5F9")

    for i, m in enumerate(metrics, start=1):
        row = table.rows[i]
        set_row_cant_split(row)

        c0 = row.cells[0].paragraphs[0]
        c0.text = m.label
        c0.runs[0].bold = True if c0.runs else False

        c1 = row.cells[1].paragraphs[0]
        c1.text = clean_text_repr(m.value)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = 6


def add_display_table(
    doc: Any,
    table: DisplayTable,
    *,
    max_rows: int | None = None,
) -> None:
    """Render a statistical display table into an editable Word table."""
    if not table or not table.columns:
        return

    from docx.enum.text import WD_ALIGN_PARAGRAPH

    if table.title:
        p_title = doc.add_paragraph(table.title, style="PyAutoStat Table Header")
        set_keep_with_next(p_title)

    total_rows = len(table.rows)
    rendered_rows = table.rows[:max_rows] if max_rows is not None else table.rows

    doc_table = doc.add_table(rows=len(rendered_rows) + 1, cols=len(table.columns))
    set_table_borders(doc_table, color_hex="CBD5E1", sz="4", val="single")
    set_table_cell_margins(doc_table, top_dxa=80, bottom_dxa=80, left_dxa=120, right_dxa=120)

    # Header Row
    hdr_row = doc_table.rows[0]
    set_table_header_row(hdr_row)
    set_row_cant_split(hdr_row)

    for c_idx, col_name in enumerate(table.columns):
        cell = hdr_row.cells[c_idx]
        set_cell_shading(cell, "F1F5F9")
        p = cell.paragraphs[0]
        p.style = "PyAutoStat Table Header"
        p.text = col_name
        p.alignment = (
            WD_ALIGN_PARAGRAPH.RIGHT
            if is_numeric_column(col_name, c_idx)
            else WD_ALIGN_PARAGRAPH.LEFT
        )

    # Data Rows
    for r_idx, row_data in enumerate(rendered_rows, start=1):
        row = doc_table.rows[r_idx]
        set_row_cant_split(row)
        for c_idx, cell_val in enumerate(row_data.cells):
            if c_idx >= len(table.columns):
                break
            col_name = table.columns[c_idx]
            cell = row.cells[c_idx]
            p = cell.paragraphs[0]
            p.text = format_cell_text(cell_val)
            p.alignment = (
                WD_ALIGN_PARAGRAPH.RIGHT
                if is_numeric_column(col_name, c_idx)
                else WD_ALIGN_PARAGRAPH.LEFT
            )

    if max_rows is not None and total_rows > max_rows:
        p_note = doc.add_paragraph(
            f"Showing {max_rows} of {total_rows} rows; use detail='full' for the complete table."
        )
        p_note.paragraph_format.space_before = 2
        p_note.paragraph_format.space_after = 6
    else:
        p_sp = doc.add_paragraph()
        p_sp.paragraph_format.space_after = 6


def add_diagnostics(doc: Any, diagnostics: Sequence[DisplayDiagnostic]) -> None:
    """Render statistical condition diagnostics as an editable table."""
    if not diagnostics:
        return

    table = doc.add_table(rows=len(diagnostics) + 1, cols=3)
    set_table_borders(table, color_hex="CBD5E1", sz="4", val="single")
    set_table_cell_margins(table, top_dxa=80, bottom_dxa=80, left_dxa=120, right_dxa=120)

    hdr = table.rows[0]
    set_table_header_row(hdr)
    set_row_cant_split(hdr)

    for idx, name in enumerate(("Diagnostic Check", "Status", "Details")):
        cell = hdr.cells[idx]
        set_cell_shading(cell, "F1F5F9")
        p = cell.paragraphs[0]
        p.style = "PyAutoStat Table Header"
        p.text = name

    for i, diag in enumerate(diagnostics, start=1):
        row = table.rows[i]
        set_row_cant_split(row)

        c0 = row.cells[0].paragraphs[0]
        c0.text = diag.label
        if c0.runs:
            c0.runs[0].bold = True

        c1 = row.cells[1].paragraphs[0]
        c1.text = diag.status.capitalize()
        status_lower = diag.status.lower()
        if c1.runs:
            from docx.shared import RGBColor

            if "pass" in status_lower or "met" in status_lower:
                c1.runs[0].font.color.rgb = RGBColor(22, 101, 52)
            elif "fail" in status_lower or "reject" in status_lower:
                c1.runs[0].font.color.rgb = RGBColor(185, 28, 28)
            else:
                c1.runs[0].font.color.rgb = RGBColor(146, 64, 14)

        c2 = row.cells[2].paragraphs[0]
        c2.text = clean_text_repr(diag.detail) if diag.detail else "—"

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = 6


def add_interpretation(doc: Any, interpretation: str) -> None:
    """Render deterministic statistical interpretation text."""
    if not interpretation:
        return
    p = doc.add_paragraph(clean_text_repr(interpretation), style="PyAutoStat Body")
    p.paragraph_format.space_after = 8


def add_bullet_section(doc: Any, items: Sequence[str]) -> None:
    """Render bullet items using true Word bullet list style."""
    if not items:
        p = doc.add_paragraph("None recorded.", style="PyAutoStat Body")
        p.paragraph_format.space_after = 6
        return

    for item in items:
        p = doc.add_paragraph(clean_text_repr(item), style="List Bullet")
        p.paragraph_format.space_after = 2

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = 4


def add_executive_summary(doc: Any, summary: dict[str, Any] | str) -> None:
    """Render executive summary block."""
    if isinstance(summary, str):
        doc.add_paragraph(summary, style="PyAutoStat Body")
        return

    if not summary:
        return

    for k, v in summary.items():
        if v is None or v == "" or v == [] or v == {}:
            continue
        p = doc.add_paragraph()
        r_k = p.add_run(f"{k.replace('_', ' ').title()}: ")
        r_k.bold = True
        p.add_run(format_display_value(v))
        p.paragraph_format.space_after = 3


def add_analysis_record(doc: Any, metadata: dict[str, Any]) -> None:
    """Render audit and reproducibility metadata table in full detail mode."""
    if not metadata:
        return

    table = doc.add_table(rows=len(metadata) + 1, cols=2)
    set_table_borders(table, color_hex="CBD5E1", sz="4", val="single")
    set_table_cell_margins(table, top_dxa=80, bottom_dxa=80, left_dxa=120, right_dxa=120)

    hdr = table.rows[0]
    set_table_header_row(hdr)
    set_row_cant_split(hdr)

    hdr.cells[0].paragraphs[0].text = "Record Property"
    hdr.cells[0].paragraphs[0].style = "PyAutoStat Table Header"
    set_cell_shading(hdr.cells[0], "F1F5F9")

    hdr.cells[1].paragraphs[0].text = "Recorded Value"
    hdr.cells[1].paragraphs[0].style = "PyAutoStat Table Header"
    set_cell_shading(hdr.cells[1], "F1F5F9")

    for i, (k, v) in enumerate(metadata.items(), start=1):
        row = table.rows[i]
        set_row_cant_split(row)

        c0 = row.cells[0].paragraphs[0]
        c0.text = str(k).replace("_", " ").title()
        if c0.runs:
            c0.runs[0].bold = True

        c1 = row.cells[1].paragraphs[0]
        c1.text = str(v)
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = 6


def add_figure_image(
    doc: Any,
    artifact: Any,
    *,
    printable_width: Any = None,
    note: str | None = None,
) -> None:
    """Embed a static PNG figure and caption into an OpenXML Word document."""
    import io

    from docx.enum.text import WD_ALIGN_PARAGRAPH

    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = 6
    p_img.paragraph_format.space_after = 2

    run = p_img.add_run()
    img_stream = io.BytesIO(artifact.data)
    if printable_width is not None:
        run.add_picture(img_stream, width=printable_width)
    else:
        run.add_picture(img_stream)

    set_keep_with_next(p_img)

    caption_text = f"Figure {artifact.index}. {artifact.title}."
    if note:
        caption_text += f" {note}"
    p_cap = doc.add_paragraph(caption_text, style="PyAutoStat Figure Caption")
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
