"""Renderer for direct descriptive outputs (frequency tables and cross-tabs)."""

from __future__ import annotations

from .base import BaseRenderer


class DescriptiveTableRenderer(BaseRenderer):
    """Rich presentation renderer for frequency tables and cross-tabulations."""

    def render_standard(self) -> None:
        self.render_header()

        self.render_section_heading("TABLE ACCOUNTING")
        lbl_w = 20 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        if self.view.tables:
            for table_model in self.view.tables:
                self.render_section_heading(table_model.title or "DESCRIPTIVE TABLE")
                self.render_data_table(table_model)

        if self.view.interpretation:
            self.render_section_heading("NARRATIVE SUMMARY")
            self.render_interpretation(self.view.interpretation)

        if self.view.diagnostics:
            self.render_section_heading("REVIEW CUES")
            self.render_diagnostics(self.view.diagnostics)

    def render_full(self) -> None:
        self.render_standard()
