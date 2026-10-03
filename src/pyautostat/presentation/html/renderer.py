"""HTML renderer for PyAutoStat normalized presentation models."""

from __future__ import annotations

from ..figures.models import FigureSpec
from ..models import DisplayMetric, PresentationView
from .components import (
    render_analysis_record,
    render_design_grid,
    render_diagnostics,
    render_header,
    render_interpretation,
    render_limitations,
    render_metric_cards,
    render_page,
    render_section,
    render_table,
    render_warnings,
)
from .formatting import escape_text
from .plotly_renderer import get_plotly_bundle, render_figure_html, render_noscript_banner
from .templates import section_anchor, styled_heading


class HtmlRenderer:
    """Renderer converting PresentationView models into polished, standalone HTML reports."""

    def __init__(
        self,
        view: PresentationView,
        *,
        detail: str = "standard",
        title: str | None = None,
        style: str = "general",
        figure_spec: FigureSpec | None = None,
    ) -> None:
        self.view = view
        self.detail = detail
        self.title_override = title
        self.style = style
        self.figure_spec = figure_spec

    @property
    def display_title(self) -> str:
        if self.title_override is not None and self.title_override.strip():
            return self.title_override.strip()
        return self.view.title

    def render(self) -> str:
        """Render HTML document according to requested detail mode."""
        if self.detail == "compact":
            return self.render_compact()
        if self.detail == "full":
            return self.render_full()
        return self.render_standard()

    def render_compact(self) -> str:
        """Render a concise, high-level summary report."""
        sec_idx = 1
        body_parts = [render_header(self.display_title, self.view.subtitle)]

        figure_html = ""
        plotly_bundle = ""
        figure_inserted = False
        if self.figure_spec is not None and self.figure_spec.kind == "estimate_ci":
            figure_html = render_figure_html(self.figure_spec, figure_idx=1)
            plotly_bundle = get_plotly_bundle()

        # Key results
        if self.view.key_metrics:
            heading = styled_heading("KEY RESULTS", sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_metric_cards(self.view.key_metrics),
                    section_id="key-results",
                )
            )
            if figure_html:
                body_parts.append(figure_html)
                figure_inserted = True

        # Compact interpretation
        interp_text = self.view.compact_text or self.view.interpretation
        if interp_text:
            first_para = interp_text.strip().split("\n\n")[0]
            heading = styled_heading("SUMMARY FINDING", sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_interpretation(first_para),
                    section_id="interpretation",
                )
            )

        if figure_inserted:
            body_parts.insert(1, render_noscript_banner())

        body_html = "\n\n".join(part for part in body_parts if part)
        return render_page(
            self.display_title, body_html, style=self.style, extra_head=plotly_bundle
        )

    def render_standard(self) -> str:
        """Render standard comprehensive report."""
        sec_idx = 1
        body_parts = [render_header(self.display_title, self.view.subtitle)]

        figure_html = ""
        plotly_bundle = ""
        figure_inserted = False
        if self.figure_spec is not None:
            figure_html = render_figure_html(self.figure_spec, figure_idx=1)
            plotly_bundle = get_plotly_bundle()

        # APA / IEEE concise prefix if available in metadata
        apa_summary = self.view.metadata.get("apa_summary")
        if self.style == "apa" and apa_summary:
            body_parts.append(
                f'<p class="oriented-summary">APA-oriented summary: {escape_text(apa_summary)}</p>'
            )

        # 1. Analysis Design
        if self.view.design_metrics:
            design_title = "ANALYSIS DESIGN"
            if self.view.family == "family.regression":
                design_title = "MODEL SPECIFICATION"
            heading = styled_heading(design_title, sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_design_grid(self.view.design_metrics),
                    section_id=section_anchor(design_title),
                )
            )

        # 2. Key Results / Model Fit / Omnibus
        if self.view.key_metrics:
            results_title = "KEY RESULTS"
            if self.view.family == "family.regression":
                results_title = "MODEL FIT"
            elif self.view.family == "family.rank" and "Multi-Group" in self.view.title:
                results_title = "OMNIBUS KEY RESULT"
            heading = styled_heading(results_title, sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_metric_cards(self.view.key_metrics),
                    section_id=section_anchor(results_title),
                )
            )
            if (
                figure_html
                and not figure_inserted
                and self.figure_spec is not None
                and self.figure_spec.placement == "KEY RESULTS"
            ):
                body_parts.append(figure_html)
                figure_inserted = True

        # 3. Summary Tables (Pairwise comparisons bounded to max 6 rows in standard mode)
        if self.view.tables:
            for table_model in self.view.tables:
                table_title = table_model.title or "SUMMARY TABLE"
                max_r = 6 if "PAIRWISE" in table_title.upper() else None
                heading = styled_heading(table_title, sec_idx, self.style)
                sec_idx += 1
                body_parts.append(
                    render_section(
                        heading,
                        render_table(table_model, max_rows=max_r),
                        section_id=section_anchor(table_title),
                    )
                )
                if (
                    figure_html
                    and not figure_inserted
                    and self.figure_spec is not None
                    and self.figure_spec.placement.upper() in table_title.upper()
                ):
                    body_parts.append(figure_html)
                    figure_inserted = True

        if figure_html and not figure_inserted:
            body_parts.append(figure_html)
            figure_inserted = True

        # 4. Interpretation
        if self.view.interpretation:
            heading = styled_heading("INTERPRETATION", sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_interpretation(self.view.interpretation),
                    section_id="interpretation",
                )
            )

        # 5. Diagnostics
        if self.view.diagnostics:
            diag_title = "DIAGNOSTIC CONTEXT"
            heading = styled_heading(diag_title, sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_diagnostics(self.view.diagnostics),
                    section_id="diagnostics",
                )
            )

        # 6. Limitations
        if self.view.limitations:
            heading = styled_heading("IMPORTANT LIMITATIONS", sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_limitations(self.view.limitations),
                    section_id="limitations",
                )
            )

        # 7. Warnings
        if self.view.warnings:
            heading = styled_heading("WARNINGS", sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_warnings(self.view.warnings),
                    section_id="warnings",
                )
            )

        if figure_inserted:
            body_parts.insert(1, render_noscript_banner())

        body_html = "\n\n".join(part for part in body_parts if part)
        return render_page(
            self.display_title, body_html, style=self.style, extra_head=plotly_bundle
        )

    def render_full(self) -> str:
        """Render complete, detailed report with untruncated tables and governance records."""
        sec_idx = 1
        body_parts = [render_header(self.display_title, self.view.subtitle)]

        figure_html = ""
        plotly_bundle = ""
        figure_inserted = False
        if self.figure_spec is not None:
            figure_html = render_figure_html(self.figure_spec, figure_idx=1)
            plotly_bundle = get_plotly_bundle()

        apa_summary = self.view.metadata.get("apa_summary")
        if self.style == "apa" and apa_summary:
            body_parts.append(
                f'<p class="oriented-summary">APA-oriented summary: {escape_text(apa_summary)}</p>'
            )

        # 1. Analysis Design
        if self.view.design_metrics:
            design_title = "ANALYSIS DESIGN"
            if self.view.family == "family.regression":
                design_title = "MODEL SPECIFICATION"
            heading = styled_heading(design_title, sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_design_grid(self.view.design_metrics),
                    section_id=section_anchor(design_title),
                )
            )

        # 2. Key Results
        if self.view.key_metrics:
            results_title = "KEY RESULTS"
            if self.view.family == "family.regression":
                results_title = "MODEL FIT"
            elif self.view.family == "family.rank" and "Multi-Group" in self.view.title:
                results_title = "OMNIBUS KEY RESULT"
            heading = styled_heading(results_title, sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_metric_cards(self.view.key_metrics),
                    section_id=section_anchor(results_title),
                )
            )
            if (
                figure_html
                and not figure_inserted
                and self.figure_spec is not None
                and self.figure_spec.placement == "KEY RESULTS"
            ):
                body_parts.append(figure_html)
                figure_inserted = True

        # 3. Tables (Untruncated)
        if self.view.tables:
            for table_model in self.view.tables:
                table_title = table_model.title or "SUMMARY TABLE"
                heading = styled_heading(table_title, sec_idx, self.style)
                sec_idx += 1
                body_parts.append(
                    render_section(
                        heading,
                        render_table(table_model, max_rows=None),
                        section_id=section_anchor(table_title),
                    )
                )
                if (
                    figure_html
                    and not figure_inserted
                    and self.figure_spec is not None
                    and self.figure_spec.placement.upper() in table_title.upper()
                ):
                    body_parts.append(figure_html)
                    figure_inserted = True

        if figure_html and not figure_inserted:
            body_parts.append(figure_html)
            figure_inserted = True

        # 4. Interpretation
        if self.view.interpretation:
            heading = styled_heading("INTERPRETATION", sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_interpretation(self.view.interpretation),
                    section_id="interpretation",
                )
            )

        # 5. Diagnostics
        if self.view.diagnostics:
            diag_title = "DIAGNOSTIC CONTEXT"
            heading = styled_heading(diag_title, sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_diagnostics(self.view.diagnostics),
                    section_id="diagnostics",
                )
            )

        # 6. Sample Accounting (Full Mode)
        sample_acc = self.view.metadata.get("sample_accounting")
        if isinstance(sample_acc, dict) and sample_acc:
            heading = styled_heading("SAMPLE ACCOUNTING", sec_idx, self.style)
            sec_idx += 1
            acc_metrics = [
                DisplayMetric(str(k).replace("_", " ").title(), str(v))
                for k, v in sample_acc.items()
            ]
            body_parts.append(
                render_section(
                    heading,
                    render_design_grid(acc_metrics),
                    section_id="sample-accounting",
                )
            )

        # 7. Reference Levels (Full Mode)
        ref_levels = self.view.metadata.get("reference_levels")
        if isinstance(ref_levels, dict) and ref_levels:
            heading = styled_heading("REFERENCE LEVELS", sec_idx, self.style)
            sec_idx += 1
            ref_metrics = [DisplayMetric(str(k), f"Reference: {v}") for k, v in ref_levels.items()]
            body_parts.append(
                render_section(
                    heading,
                    render_design_grid(ref_metrics),
                    section_id="reference-levels",
                )
            )

        # 8. Recommendation Rationale (Full Mode)
        rationale = self.view.metadata.get("rationale")
        if rationale:
            heading = styled_heading("RECOMMENDATION RATIONALE", sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    f'<p class="diagnostic-detail">{escape_text(rationale)}</p>',
                    section_id="recommendation-rationale",
                )
            )

        # 9. Audit Verification (Full Mode)
        audit_status = self.view.metadata.get("audit_status")
        if audit_status:
            heading = styled_heading("AUDIT VERIFICATION", sec_idx, self.style)
            sec_idx += 1
            status_upper = str(audit_status).upper()
            badge_class = (
                "status-success" if status_upper in ("PASS", "PASSED") else "status-warning"
            )
            escaped_status = escape_text(status_upper)
            audit_html = (
                '<div class="diagnostic-item">\n'
                f'  <span class="diagnostic-status {badge_class}">[{escaped_status}]</span>\n'
                '  <span class="diagnostic-label">Consistency Audit</span>\n'
                f'  <span class="diagnostic-detail">Result verification: {escaped_status}</span>\n'
                "</div>"
            )
            body_parts.append(render_section(heading, audit_html, section_id="audit-verification"))

        # 10. Reproducibility Metadata (Full Mode)
        repro = self.view.metadata.get("reproducibility")
        if isinstance(repro, dict) and repro:
            heading = styled_heading("REPRODUCIBILITY METADATA", sec_idx, self.style)
            sec_idx += 1
            repro_metrics: list[DisplayMetric] = []
            stoch = repro.get("stochastic", {})
            if isinstance(stoch, dict):
                seed = stoch.get("effective_seed")
                if seed is not None:
                    repro_metrics.append(DisplayMetric("Random Seed", str(seed)))
                resamples = stoch.get("bootstrap_resamples")
                if resamples is not None:
                    repro_metrics.append(DisplayMetric("Bootstrap Resamples", str(resamples)))
            spec_ref = repro.get("specification_reference")
            if spec_ref:
                repro_metrics.append(DisplayMetric("Specification Reference", str(spec_ref)))
            if repro_metrics:
                body_parts.append(
                    render_section(
                        heading,
                        render_design_grid(repro_metrics),
                        section_id="reproducibility-metadata",
                    )
                )

        # 11. Multiplicity Control (Full Mode)
        multiplicity = self.view.metadata.get("multiplicity")
        if multiplicity:
            heading = styled_heading("MULTIPLICITY CONTROL", sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    f'<p class="diagnostic-detail">Policy: {escape_text(str(multiplicity))}</p>',
                    section_id="multiplicity-control",
                )
            )

        # 12. Limitations
        if self.view.limitations:
            heading = styled_heading("IMPORTANT LIMITATIONS", sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_limitations(self.view.limitations),
                    section_id="limitations",
                )
            )

        # 13. Warnings
        if self.view.warnings:
            heading = styled_heading("WARNINGS", sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_warnings(self.view.warnings),
                    section_id="warnings",
                )
            )

        # 14. Analysis Record / General Metadata
        if self.view.metadata:
            heading = styled_heading("ANALYSIS RECORD", sec_idx, self.style)
            sec_idx += 1
            body_parts.append(
                render_section(
                    heading,
                    render_analysis_record(self.view.metadata, detail="full"),
                    section_id="analysis-record",
                )
            )

        if figure_inserted:
            body_parts.insert(1, render_noscript_banner())

        body_html = "\n\n".join(part for part in body_parts if part)
        return render_page(
            self.display_title, body_html, style=self.style, extra_head=plotly_bundle
        )
