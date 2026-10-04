"""Unified HTML report renderer for PyAutoStat ResearchReport models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ...exceptions import ReportError
from ..adapters import UnsupportedPresentationError, adapt
from ..figures.models import FigureSpec
from ..formatting import resolve_table_row_limit
from ..models import PresentationView
from .components import (
    render_analysis_record,
    render_design_grid,
    render_diagnostics,
    render_executive_summary,
    render_header,
    render_interpretation,
    render_limitations,
    render_metric_cards,
    render_page,
    render_practical_significance,
    render_report_section_dl,
    render_report_status,
    render_section,
    render_sensitivity,
    render_table,
    render_warnings,
    report_table_to_display_table,
)
from .formatting import escape_text
from .plotly_renderer import get_plotly_bundle, render_figure_html, render_noscript_banner
from .templates import section_anchor, styled_heading

if TYPE_CHECKING:
    from ...research_report import ResearchReport

VALID_DETAILS = {"compact", "standard", "full"}
VALID_STYLES = {"general", "apa", "ieee"}


class ResearchReportHtmlRenderer:
    """Renderer for complete ResearchReport models using shared presentation components."""

    def __init__(
        self,
        report: ResearchReport,
        *,
        detail: str = "standard",
        title: str | None = None,
        style: str = "general",
        figure_spec: FigureSpec | None = None,
        figure_specs: Sequence[FigureSpec] | None = None,
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
        self.title_override = title
        self.style = style
        if figure_specs is not None:
            self.figure_specs: tuple[FigureSpec, ...] = tuple(figure_specs)
        elif figure_spec is not None:
            self.figure_specs = (figure_spec,)
        else:
            self.figure_specs = ()
        self.figure_spec = self.figure_specs[0] if self.figure_specs else None

    def render(self) -> str:
        """Render the ResearchReport according to requested detail mode."""
        from ...research_report import (
            _build_executive_summary,
            _concise_result,
            _report_sections,
        )

        data = self.report._payload
        report_title = (
            self.title_override.strip()
            if self.title_override and self.title_override.strip()
            else str(data.get("title", "Research Report"))
        )

        # Obtain source PresentationView when safe, with zero recalculation
        source_result = getattr(self.report, "_source_result", None)
        source_view: PresentationView | None = None
        if source_result is not None:
            try:
                source_view = adapt(source_result, detail=self.detail)
            except UnsupportedPresentationError:
                source_view = None

        sec_idx = 1
        body_parts: list[str] = [
            render_header(report_title, brand="PyAutoStat Research Report"),
            render_report_status(str(data.get("status", "unknown"))),
        ]

        # Executive Summary
        source_ps = getattr(self.report, "_source_practical_significance", None)
        source_sens = getattr(self.report, "_source_sensitivity", None)
        practical_verdict = source_ps.verdict if source_ps is not None else None
        sensitivity_verdict = source_sens.verdict if source_sens is not None else None

        summary_paragraphs = _build_executive_summary(
            data,
            practical_verdict=practical_verdict,
            sensitivity_verdict=sensitivity_verdict,
        )
        if summary_paragraphs:
            body_parts.append(render_executive_summary(summary_paragraphs))

        # Oriented summary (APA / IEEE)
        summary = _concise_result(data, self.style)
        if summary is not None:
            body_parts.append(f'<p class="oriented-summary">{escape_text(summary)}</p>')

        rendered_figures = [
            (spec, render_figure_html(spec, figure_idx=i + 1))
            for i, spec in enumerate(self.figure_specs)
        ]
        inserted_indices: set[int] = set()

        # Compact mode returns early
        if self.detail == "compact":
            for idx, (spec, fig_html) in enumerate(rendered_figures):
                if spec.kind == "estimate_ci":
                    body_parts.append(fig_html)
                    inserted_indices.add(idx)
            if inserted_indices:
                body_parts.insert(1, render_noscript_banner())
                extra_head = (
                    get_plotly_bundle() + '\n<script type="text/javascript">\n'
                    f"  window.__pyautostatFiguresExpected = {len(inserted_indices)};\n"
                    "  window.__pyautostatFiguresRendered = 0;\n"
                    "</script>"
                )
            else:
                extra_head = ""
            body_html = "\n\n".join(part for part in body_parts if part)
            return render_page(report_title, body_html, style=self.style, extra_head=extra_head)

        # Standard & Full Modes: Render canonical sections
        sections_dict: dict[str, Any] = data.get("sections", {})
        for key, heading in _report_sections(data):
            sec_heading = styled_heading(heading, sec_idx, self.style)
            sec_idx += 1
            sec_id = section_anchor(heading)
            section_content = self._render_section_content(
                key, sections_dict.get(key, {}), source_view, source_ps, source_sens
            )
            if section_content:
                body_parts.append(render_section(sec_heading, section_content, section_id=sec_id))
                if key == "results":
                    for idx, (spec, fig_html) in enumerate(rendered_figures):
                        if idx not in inserted_indices and spec.placement == "KEY RESULTS":
                            body_parts.append(fig_html)
                            inserted_indices.add(idx)

        # Tables
        for table_dict in data.get("tables", []):
            t_title = table_dict.get("title") or "Table"
            disp_table = report_table_to_display_table(table_dict)
            max_r = resolve_table_row_limit(disp_table, detail=self.detail)
            heading = styled_heading(t_title, sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_table(disp_table, max_rows=max_r),
                    section_id=section_anchor(t_title),
                )
            )
            for idx, (spec, fig_html) in enumerate(rendered_figures):
                if (
                    idx not in inserted_indices
                    and spec.placement
                    and spec.placement.upper() in t_title.upper()
                ):
                    body_parts.append(fig_html)
                    inserted_indices.add(idx)

        for idx, (_spec, fig_html) in enumerate(rendered_figures):
            if idx not in inserted_indices:
                body_parts.append(fig_html)
                inserted_indices.add(idx)

        # Limitations & Warnings
        limitations = data.get("limitations", [])
        lim_heading = styled_heading("Limitations", sec_idx, self.style)
        sec_idx += 1
        lim_html = render_limitations(limitations if limitations else ["None recorded."])
        body_parts.append(
            f'<section class="report-section caution"><h2>{escape_text(lim_heading)}</h2>\n'
            f"{lim_html}\n</section>"
        )

        warnings = data.get("warnings", [])
        warn_heading = styled_heading("Warnings", sec_idx, self.style)
        sec_idx += 1
        warn_html = render_warnings(warnings if warnings else ["None recorded."])
        body_parts.append(
            f'<section class="report-section caution"><h2>{escape_text(warn_heading)}</h2>\n'
            f"{warn_html}\n</section>"
        )

        # Full Mode: Analysis record & reproducibility
        if self.detail == "full":
            rec_heading = styled_heading("Analysis Record & Reproducibility", sec_idx, self.style)
            sec_idx += 1
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
            rec_html = render_analysis_record(meta, detail="full")
            if rec_html:
                body_parts.append(
                    render_section(rec_heading, rec_html, section_id="analysis-record")
                )

        if inserted_indices:
            body_parts.insert(1, render_noscript_banner())
            extra_head = (
                get_plotly_bundle() + '\n<script type="text/javascript">\n'
                f"  window.__pyautostatFiguresExpected = {len(inserted_indices)};\n"
                "  window.__pyautostatFiguresRendered = 0;\n"
                "</script>"
            )
        else:
            extra_head = ""

        body_html = "\n\n".join(part for part in body_parts if part)
        return render_page(report_title, body_html, style=self.style, extra_head=extra_head)

    def _render_section_content(
        self,
        key: str,
        section_data: dict[str, Any],
        source_view: PresentationView | None,
        source_ps: Any,
        source_sens: Any,
    ) -> str:
        """Render inner HTML content for a specific canonical section."""
        if key == "research_question":
            if source_view and source_view.design_metrics:
                grid = render_design_grid(source_view.design_metrics)
                dl = render_report_section_dl(section_data)
                return f"{grid}\n{dl}" if dl else grid
            return render_report_section_dl(section_data)

        if key == "results":
            cards = (
                render_metric_cards(source_view.key_metrics)
                if (source_view and source_view.key_metrics)
                else ""
            )
            dl = render_report_section_dl(section_data, p_value=True)
            return f"{cards}\n{dl}" if cards else dl

        if key == "diagnostics":
            diags = (
                render_diagnostics(source_view.diagnostics)
                if (source_view and source_view.diagnostics)
                else ""
            )
            dl = render_report_section_dl(section_data)
            return f"{diags}\n{dl}" if diags else dl

        if key == "interpretation":
            if source_view and source_view.interpretation:
                return render_interpretation(source_view.interpretation)
            return render_report_section_dl(section_data)

        if key == "practical_significance":
            return render_practical_significance(source_ps or section_data)

        if key == "sensitivity_analysis":
            return render_sensitivity(source_sens or section_data)

        return render_report_section_dl(section_data)
