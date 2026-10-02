"""Renderer for Intraclass Correlation Coefficient (ICC) reliability."""

from __future__ import annotations

from .base import BaseRenderer


class ICCRenderer(BaseRenderer):
    """Rich presentation renderer for ICC rater reliability workflows."""

    def render_standard(self) -> None:
        self.render_header()

        # MANDATORY: Canonical definition displayed before estimate!
        self.render_section_heading("CANONICAL ICC DEFINITION & DESIGN")
        lbl_w = 20 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        self.render_section_heading("RELIABILITY ESTIMATE")
        self.render_key_results_table(self.view.key_metrics)

        if self.view.tables:
            for table_model in self.view.tables:
                self.render_section_heading(table_model.title or "ANOVA DECOMPOSITION")
                self.render_data_table(table_model)

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
