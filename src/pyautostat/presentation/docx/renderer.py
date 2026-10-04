"""Document renderer transforming PresentationView models into editable Word documents."""

from __future__ import annotations

from typing import Any

from ...exceptions import ReportError
from ..formatting import resolve_table_row_limit
from ..models import PresentationView
from .components import (
    add_analysis_record,
    add_bullet_section,
    add_design_table,
    add_diagnostics,
    add_display_table,
    add_interpretation,
    add_metric_grid,
    add_report_header,
    add_section_heading,
)
from .ooxml import add_page_number_fields
from .settings import apply_section_geometry, normalize_page_size
from .styles import setup_document_styles

VALID_DETAILS = {"compact", "standard", "full"}
VALID_STYLES = {"general", "apa", "ieee"}


class DocxRenderer:
    """Renders standalone PyAutoStat presentation views as professional Word documents."""

    def __init__(
        self,
        view: PresentationView,
        *,
        detail: str = "standard",
        title: str | None = None,
        style: str = "general",
        page_size: str = "A4",
        landscape: bool = False,
        page_numbers: bool = True,
    ) -> None:
        if detail not in VALID_DETAILS:
            raise ValueError(
                f"Invalid detail mode '{detail}'. Expected 'compact', 'standard', or 'full'."
            )
        if style not in VALID_STYLES:
            raise ReportError("style must be general, apa, or ieee.")
        if title is not None and (not isinstance(title, str) or not title.strip()):
            raise ReportError("title must be a non-empty string when provided.")

        self.view = view
        self.detail = detail
        self.title = title.strip() if title else view.title
        self.style = style
        self.page_size = normalize_page_size(page_size)
        self.landscape = landscape
        self.page_numbers = page_numbers

    def render(self) -> Any:
        """Construct and populate a python-docx Document instance."""
        import docx
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        doc = docx.Document()

        # Set Core Document Metadata
        doc.core_properties.author = "PyAutoStat"
        doc.core_properties.title = self.title
        doc.core_properties.subject = "PyAutoStat Research Report"

        # Apply Styles & Page Geometry
        setup_document_styles(doc, style_mode=self.style)
        apply_section_geometry(
            doc.sections[0],
            page_size=self.page_size,
            landscape=self.landscape,
        )

        # Running Header
        header = doc.sections[0].header
        p_head = header.paragraphs[0]
        p_head.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_head.text = f"{self.title} | PyAutoStat"
        if p_head.runs:
            from docx.shared import Pt, RGBColor

            p_head.runs[0].font.size = Pt(8.5)
            p_head.runs[0].font.color.rgb = RGBColor(148, 163, 184)

        # Running Footer with Page Numbers
        if self.page_numbers:
            footer = doc.sections[0].footer
            p_foot = footer.paragraphs[0]
            p_foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_page_number_fields(p_foot)
            if p_foot.runs:
                from docx.shared import Pt, RGBColor

                for r in p_foot.runs:
                    r.font.size = Pt(8.5)
                    r.font.color.rgb = RGBColor(148, 163, 184)

        # Document Header
        add_report_header(doc, self.title, self.view.subtitle, style_mode=self.style)

        # Compact Mode
        if self.detail == "compact":
            if self.view.key_metrics:
                add_metric_grid(doc, self.view.key_metrics)
            if self.view.compact_text:
                doc.add_paragraph(self.view.compact_text, style="PyAutoStat Body")
            return doc

        # Key Results Section
        if self.view.key_metrics:
            add_section_heading(doc, "Key Results", level=1)
            add_metric_grid(doc, self.view.key_metrics)

        # Design & Sample Facts
        if self.view.design_metrics:
            add_section_heading(doc, "Research Design & Sample", level=1)
            add_design_table(doc, self.view.design_metrics)

        # Display Tables
        if self.view.tables:
            for table in self.view.tables:
                max_r = resolve_table_row_limit(table, detail=self.detail)
                add_display_table(doc, table, max_rows=max_r)

        # Assumption Diagnostics
        if self.view.diagnostics:
            add_section_heading(doc, "Assumption Diagnostics", level=1)
            add_diagnostics(doc, self.view.diagnostics)

        # Statistical Interpretation
        if self.view.interpretation:
            add_section_heading(doc, "Statistical Interpretation", level=1)
            add_interpretation(doc, self.view.interpretation)

        # Limitations
        if self.view.limitations:
            add_section_heading(doc, "Limitations", level=1)
            add_bullet_section(doc, self.view.limitations)

        # Warnings
        if self.view.warnings:
            add_section_heading(doc, "Warnings", level=1)
            add_bullet_section(doc, self.view.warnings)

        # Full Mode: Analysis Record & Reproducibility Metadata
        if self.detail == "full" and self.view.metadata:
            add_section_heading(doc, "Analysis Record & Reproducibility", level=1)
            add_analysis_record(doc, self.view.metadata)

        return doc
