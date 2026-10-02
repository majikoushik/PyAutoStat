"""Renderers for study planning, sensitivity analysis, and practical significance."""

from __future__ import annotations

from .base import BaseRenderer


class PlanningRenderer(BaseRenderer):
    """Rich presentation renderer for prospective study planning."""

    def render_standard(self) -> None:
        self.render_header()

        self.render_section_heading("PLANNING SPECIFICATION")
        lbl_w = 20 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        self.render_section_heading("SAMPLE SIZE REQUIREMENTS")
        self.render_key_results_table(self.view.key_metrics)

        if self.view.tables:
            for table_model in self.view.tables:
                self.render_section_heading(table_model.title or "ASSUMPTIONS")
                self.render_data_table(table_model)

        if self.view.interpretation:
            self.render_section_heading("APPROXIMATION NOTES")
            self.render_interpretation(self.view.interpretation)

        if self.view.diagnostics:
            self.render_section_heading("PLANNING CONTEXT")
            self.render_diagnostics(self.view.diagnostics)

        if self.view.limitations:
            self.render_section_heading("IMPORTANT LIMITATIONS")
            self.render_limitations(self.view.limitations)

        if self.view.warnings:
            self.render_section_heading("WARNINGS")
            self.render_warnings(self.view.warnings)

    def render_full(self) -> None:
        self.render_standard()


class SensitivityRenderer(BaseRenderer):
    """Rich presentation renderer for sensitivity scenario comparisons."""

    def render_standard(self) -> None:
        self.render_header()

        self.render_section_heading("SENSITIVITY DESIGN")
        lbl_w = 20 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        if self.view.tables:
            for table_model in self.view.tables:
                self.render_section_heading(table_model.title or "SCENARIO COMPARISONS")
                self.render_data_table(table_model)

        if self.view.interpretation:
            self.render_section_heading("ROBUSTNESS SUMMARY")
            self.render_interpretation(self.view.interpretation)

        if self.view.diagnostics:
            self.render_section_heading("COMPARABILITY CONTEXT")
            self.render_diagnostics(self.view.diagnostics)

        if self.view.limitations:
            self.render_section_heading("IMPORTANT LIMITATIONS")
            self.render_limitations(self.view.limitations)

        if self.view.warnings:
            self.render_section_heading("WARNINGS")
            self.render_warnings(self.view.warnings)

    def render_full(self) -> None:
        self.render_standard()


class PracticalSignificanceRenderer(BaseRenderer):
    """Rich presentation renderer for practical significance thresholds."""

    def render_standard(self) -> None:
        self.render_header()

        self.render_section_heading("EVALUATION DESIGN")
        lbl_w = 22 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        self.render_section_heading("PRACTICAL SIGNIFICANCE VERDICT")
        self.render_key_results_table(self.view.key_metrics)

        if self.view.interpretation:
            self.render_section_heading("CONCLUSION")
            self.render_interpretation(self.view.interpretation)

        if self.view.diagnostics:
            self.render_section_heading("EVALUATION CONTEXT")
            self.render_diagnostics(self.view.diagnostics)

        if self.view.limitations:
            self.render_section_heading("IMPORTANT LIMITATIONS")
            self.render_limitations(self.view.limitations)

        if self.view.warnings:
            self.render_section_heading("WARNINGS")
            self.render_warnings(self.view.warnings)

    def render_full(self) -> None:
        self.render_standard()
