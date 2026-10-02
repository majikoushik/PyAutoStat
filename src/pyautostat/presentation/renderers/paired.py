"""Renderer for paired two-condition comparisons (paired t-test, Wilcoxon signed-rank)."""

from __future__ import annotations

from .base import BaseRenderer


class PairedRenderer(BaseRenderer):
    """Rich presentation renderer for paired two-condition workflows."""

    def render_standard(self) -> None:
        self.render_header()

        self.render_section_heading("ANALYSIS DESIGN")
        lbl_w = 20 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        if self.view.tables:
            for table_model in self.view.tables:
                self.render_section_heading(table_model.title or "CONDITION SUMMARY")
                self.render_data_table(table_model)

        self.render_section_heading("KEY RESULT")
        self.render_key_results_table(self.view.key_metrics)

        if self.view.interpretation:
            self.render_section_heading("INTERPRETATION")
            self.render_interpretation(self.view.interpretation)

        if self.view.diagnostics:
            self.render_section_heading("DIAGNOSTIC CONTEXT")
            self.render_diagnostics(self.view.diagnostics)

        if self.view.limitations:
            self.render_section_heading("IMPORTANT LIMITATIONS")
            self.render_limitations(self.view.limitations)

        if self.view.warnings:
            self.render_section_heading("WARNINGS")
            self.render_warnings(self.view.warnings)

    def render_full(self) -> None:
        self.render_standard()
        self.render_recommendation_rationale()
        self.render_audit_status()
        self.render_reproducibility_metadata()
