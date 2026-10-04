"""Renderer transforming canonical ResearchReport models into editable Word documents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...exceptions import ReportError
from ..adapters import UnsupportedPresentationError, adapt
from ..formatting import resolve_table_row_limit
from ..html.components import report_table_to_display_table
from ..models import DisplayMetric, PresentationView
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
    format_display_value,
)
from .ooxml import add_page_number_fields
from .settings import apply_section_geometry, normalize_page_size
from .styles import setup_document_styles

if TYPE_CHECKING:
    from ...research_report import ResearchReport

VALID_DETAILS = {"compact", "standard", "full"}
VALID_STYLES = {"general", "apa", "ieee"}


def _render_structured_section(doc: Any, sec_data: dict[str, Any]) -> None:
    """Render a structured report section with no raw Python dict/list repr."""
    if not isinstance(sec_data, dict) or not sec_data:
        return

    pending_metrics: list[DisplayMetric] = []

    def flush_metrics() -> None:
        if pending_metrics:
            add_design_table(doc, list(pending_metrics))
            pending_metrics.clear()

    for k, v in sec_data.items():
        if v is None or v == "" or v == [] or v == {}:
            continue
        label = str(k).replace("_", " ").title()

        if isinstance(v, (list, tuple)) and any(isinstance(x, str) and len(x) > 35 for x in v):
            flush_metrics()
            add_bullet_section(doc, [format_display_value(x) for x in v])
        else:
            pending_metrics.append(DisplayMetric(label, format_display_value(v)))

    flush_metrics()


class ResearchReportDocxRenderer:
    """Renderer for complete ResearchReport models using shared presentation components."""

    def __init__(
        self,
        report: ResearchReport,
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

        self.report = report
        self.detail = detail
        self.title_override = title.strip() if title else None
        self.style = style
        self.page_size = normalize_page_size(page_size)
        self.landscape = landscape
        self.page_numbers = page_numbers

    def render(self) -> Any:
        """Construct and populate a python-docx Document instance from a ResearchReport."""
        import docx
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        from ...research_report import (
            _build_executive_summary,
            _concise_result,
            _report_sections,
        )

        data = self.report._payload
        report_title = (
            self.title_override
            if self.title_override
            else str(data.get("title", "Research Report"))
        )

        doc = docx.Document()

        # Core Metadata
        doc.core_properties.author = "PyAutoStat"
        doc.core_properties.title = report_title
        doc.core_properties.subject = "PyAutoStat Research Report"

        # Apply Styles & Geometry
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
        p_head.text = f"{report_title} | PyAutoStat"
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

        # Source PresentationView (safe reuse of validated values, zero recalculation)
        source_result = getattr(self.report, "_source_result", None)
        source_view: PresentationView | None = None
        if source_result is not None:
            try:
                source_view = adapt(source_result, detail=self.detail)
            except UnsupportedPresentationError:
                source_view = None

        # Title and Subtitle Block
        status = str(data.get("status", "complete"))
        add_report_header(
            doc,
            report_title,
            subtitle=f"Report status: {status}",
            style_mode=self.style,
        )

        # Concise Result Summary (APA / IEEE oriented)
        concise = _concise_result(data, self.style)
        if concise:
            p_concise = doc.add_paragraph(concise, style="PyAutoStat Body")
            if p_concise.runs:
                p_concise.runs[0].italic = True

        # Compact Mode returns early
        if self.detail == "compact":
            if source_view and source_view.key_metrics:
                add_metric_grid(doc, source_view.key_metrics)
            elif "sections" in data and "results" in data["sections"]:
                res_metrics = [
                    DisplayMetric(str(k).replace("_", " ").title(), format_display_value(v))
                    for k, v in data["sections"]["results"].items()
                    if v is not None and v != "" and v != [] and v != {}
                ]
                if res_metrics:
                    add_metric_grid(doc, res_metrics)
            return doc

        # Executive Summary
        summary_items = _build_executive_summary(data)
        if summary_items:
            add_section_heading(doc, "Executive Summary", level=1)
            add_bullet_section(doc, summary_items)

        # Canonical Structured Sections
        sections_dict: dict[str, Any] = data.get("sections", {})
        for key, heading in _report_sections(data):
            sec_data = sections_dict.get(key, {})
            core_keys = ("research_question", "results", "diagnostics", "interpretation")
            if not sec_data and key not in core_keys:
                continue

            add_section_heading(doc, heading, level=1)

            if key == "research_question":
                if source_view and source_view.design_metrics:
                    add_design_table(doc, source_view.design_metrics)
                else:
                    _render_structured_section(doc, sec_data)

            elif key == "results":
                if source_view and source_view.key_metrics:
                    add_metric_grid(doc, source_view.key_metrics)
                else:
                    items = [
                        DisplayMetric(str(k).replace("_", " ").title(), format_display_value(v))
                        for k, v in sec_data.items()
                        if v is not None and v != "" and v != [] and v != {}
                    ]
                    if items:
                        add_metric_grid(doc, items)

            elif key == "diagnostics":
                if source_view and source_view.diagnostics:
                    add_diagnostics(doc, source_view.diagnostics)
                else:
                    _render_structured_section(doc, sec_data)

            elif key == "interpretation":
                if source_view and source_view.interpretation:
                    add_interpretation(doc, source_view.interpretation)
                elif isinstance(sec_data, dict):
                    interp_text = " ".join(
                        f"{str(k).replace('_', ' ').title()}: {format_display_value(v)}"
                        for k, v in sec_data.items()
                        if v is not None and v != "" and v != [] and v != {}
                    )
                    add_interpretation(doc, interp_text)
                elif isinstance(sec_data, str):
                    add_interpretation(doc, sec_data)

            else:
                # Other structured sections (dataset, methods, sensitivity, practical significance)
                if isinstance(sec_data, dict):
                    _render_structured_section(doc, sec_data)

        # Embedded Tables
        for table_dict in data.get("tables", []):
            disp_table = report_table_to_display_table(table_dict)
            max_r = resolve_table_row_limit(disp_table, detail=self.detail)
            add_display_table(doc, disp_table, max_rows=max_r)

        # Limitations
        limitations = data.get("limitations") or ["None recorded."]
        add_section_heading(doc, "Limitations", level=1)
        add_bullet_section(doc, limitations)

        # Warnings
        warnings = data.get("warnings") or ["None recorded."]
        add_section_heading(doc, "Warnings", level=1)
        add_bullet_section(doc, warnings)

        # Full Mode: Analysis Record & Reproducibility
        if self.detail == "full":
            add_section_heading(doc, "Analysis Record & Reproducibility", level=1)
            meta: dict[str, Any] = {}
            if source_view and source_view.metadata:
                meta.update(source_view.metadata)
            if "analysis" in data and isinstance(data["analysis"], dict):
                an_dict = data["analysis"]
                meta.setdefault("method_id", an_dict.get("method_id"))
                meta.setdefault("status", an_dict.get("status"))
                if "metadata" in an_dict and isinstance(an_dict["metadata"], dict):
                    meta.update(an_dict["metadata"])
            from ... import __version__

            meta.setdefault("package_version", data.get("package_version") or __version__)
            if "audit_status" in data:
                meta["audit_status"] = data["audit_status"]
            add_analysis_record(doc, meta)

        return doc
