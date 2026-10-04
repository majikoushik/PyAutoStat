"""Typography and style management for PyAutoStat DOCX presentations."""

from __future__ import annotations

from typing import Any


def setup_document_styles(doc: Any, style_mode: str = "general") -> None:
    """Configure built-in and named PyAutoStat document styles."""
    from docx.enum.style import WD_STYLE_TYPE
    from docx.shared import Pt, RGBColor

    is_apa = style_mode == "apa"
    is_ieee = style_mode == "ieee"

    # Font family selection
    font_name = "Times New Roman" if (is_apa or is_ieee) else "Calibri"

    # Default colors (accessible dark slate)
    primary_color = RGBColor(15, 23, 42)  # #0f172a
    secondary_color = RGBColor(71, 85, 105)  # #475569
    body_color = RGBColor(30, 41, 59)  # #1e293b
    warning_color = RGBColor(146, 64, 14)  # #92400e
    muted_color = RGBColor(100, 116, 139)  # #64748b

    # Base Normal / Body style
    normal = doc.styles["Normal"]
    normal.font.name = font_name
    normal.font.size = Pt(12) if is_apa else (Pt(10) if is_ieee else Pt(10.5))
    normal.font.color.rgb = body_color
    normal.paragraph_format.line_spacing = 1.5 if is_apa else 1.15
    normal.paragraph_format.space_after = Pt(4)

    # Configure Headings
    h1 = doc.styles["Heading 1"]
    h1.font.name = font_name
    h1.font.size = Pt(12 if is_apa else (12 if is_ieee else 15))
    h1.font.bold = True
    h1.font.color.rgb = primary_color
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(4)
    h1.paragraph_format.keep_with_next = True

    h2 = doc.styles["Heading 2"]
    h2.font.name = font_name
    h2.font.size = Pt(12 if is_apa else (11 if is_ieee else 13))
    h2.font.bold = True
    h2.font.color.rgb = primary_color
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(3)
    h2.paragraph_format.keep_with_next = True

    h3 = doc.styles["Heading 3"]
    h3.font.name = font_name
    h3.font.size = Pt(12 if is_apa else (10 if is_ieee else 11))
    h3.font.bold = True
    h3.font.italic = bool(is_apa)
    h3.font.color.rgb = secondary_color
    h3.paragraph_format.space_before = Pt(8)
    h3.paragraph_format.space_after = Pt(2)
    h3.paragraph_format.keep_with_next = True

    def _get_or_add_style(name: str, style_type: Any) -> Any:
        try:
            return doc.styles[name]
        except KeyError:
            s = doc.styles.add_style(name, style_type)
            s.base_style = doc.styles["Normal"]
            return s

    # Named Styles for PyAutoStat Semantic Elements
    # 1. PyAutoStat Title
    st_title = _get_or_add_style("PyAutoStat Title", WD_STYLE_TYPE.PARAGRAPH)
    st_title.font.name = font_name
    st_title.font.size = Pt(12 if is_apa else (20 if is_ieee else 22))
    st_title.font.bold = True
    st_title.font.color.rgb = primary_color
    st_title.paragraph_format.space_after = Pt(4)
    st_title.paragraph_format.keep_with_next = True

    # 2. PyAutoStat Subtitle
    st_subtitle = _get_or_add_style("PyAutoStat Subtitle", WD_STYLE_TYPE.PARAGRAPH)
    st_subtitle.font.name = font_name
    st_subtitle.font.size = Pt(11 if is_apa else (10 if is_ieee else 11.5))
    st_subtitle.font.color.rgb = secondary_color
    st_subtitle.paragraph_format.space_after = Pt(12)
    st_subtitle.paragraph_format.keep_with_next = True

    # 3. PyAutoStat Body
    st_body = _get_or_add_style("PyAutoStat Body", WD_STYLE_TYPE.PARAGRAPH)
    st_body.font.name = font_name
    st_body.font.size = Pt(12 if is_apa else (10 if is_ieee else 10.5))
    st_body.font.color.rgb = body_color
    st_body.paragraph_format.space_after = Pt(4)

    # 4. PyAutoStat Metric Label
    st_ml = _get_or_add_style("PyAutoStat Metric Label", WD_STYLE_TYPE.PARAGRAPH)
    st_ml.font.name = font_name
    st_ml.font.size = Pt(9.5)
    st_ml.font.bold = True
    st_ml.font.color.rgb = muted_color
    st_ml.paragraph_format.space_after = Pt(1)
    st_ml.paragraph_format.keep_with_next = True

    # 5. PyAutoStat Metric Value
    st_mv = _get_or_add_style("PyAutoStat Metric Value", WD_STYLE_TYPE.PARAGRAPH)
    st_mv.font.name = font_name
    st_mv.font.size = Pt(12)
    st_mv.font.bold = True
    st_mv.font.color.rgb = primary_color
    st_mv.paragraph_format.space_after = Pt(2)

    # 6. PyAutoStat Table Header
    st_th = _get_or_add_style("PyAutoStat Table Header", WD_STYLE_TYPE.PARAGRAPH)
    st_th.font.name = font_name
    st_th.font.size = Pt(10)
    st_th.font.bold = True
    st_th.font.color.rgb = primary_color
    st_th.paragraph_format.space_after = Pt(2)

    # 7. PyAutoStat Diagnostic
    st_diag = _get_or_add_style("PyAutoStat Diagnostic", WD_STYLE_TYPE.PARAGRAPH)
    st_diag.font.name = font_name
    st_diag.font.size = Pt(10)
    st_diag.font.color.rgb = body_color
    st_diag.paragraph_format.space_after = Pt(2)

    # 8. PyAutoStat Warning
    st_warn = _get_or_add_style("PyAutoStat Warning", WD_STYLE_TYPE.PARAGRAPH)
    st_warn.font.name = font_name
    st_warn.font.size = Pt(10)
    st_warn.font.bold = True
    st_warn.font.color.rgb = warning_color
    st_warn.paragraph_format.space_after = Pt(2)

    # 9. PyAutoStat Figure Caption
    st_cap = _get_or_add_style("PyAutoStat Figure Caption", WD_STYLE_TYPE.PARAGRAPH)
    st_cap.font.name = font_name
    st_cap.font.size = Pt(10 if is_apa else (9 if is_ieee else 9.5))
    st_cap.font.italic = bool(is_apa)
    st_cap.font.color.rgb = secondary_color
    st_cap.paragraph_format.space_before = Pt(4)
    st_cap.paragraph_format.space_after = Pt(10)
