"""Renderer for dataset profiling presentation."""

from __future__ import annotations

from .base import BaseRenderer


class ProfileRenderer(BaseRenderer):
    """Rich presentation renderer for dataset profile results."""

    def render_standard(self) -> None:
        self.render_header()

        self.render_section_heading("DATA QUALITY")
        lbl_w = 18 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        for table_model in self.view.tables:
            self.render_section_heading(table_model.title or "VARIABLES")
            self.render_data_table(table_model)

        if self.view.diagnostics:
            self.render_section_heading("REVIEW CUES")
            self.render_diagnostics(self.view.diagnostics)

    def render_full(self) -> None:
        # Full mode renders all tables and full issues via view model configuration
        self.render_standard()
