"""Renderer for association analysis (Pearson correlation)."""

from __future__ import annotations

from .base import BaseRenderer


class AssociationRenderer(BaseRenderer):
    """Rich presentation renderer for linear association workflows."""

    def render_standard(self) -> None:
        self.render_header()

        self.render_section_heading("ANALYSIS DESIGN")
        lbl_w = 20 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        if self.view.tables:
            for table_model in self.view.tables:
                self.render_section_heading(table_model.title or "BREAKDOWN")
                self.render_data_table(table_model)

        self.render_section_heading("KEY RESULT")
        self.render_key_results_table(self.view.key_metrics)

        if self.view.interpretation:
            self.render_section_heading("INTERPRETATION")
            self.render_interpretation(self.view.interpretation)

        if self.view.diagnostics:
            self.render_section_heading("DIAGNOSTIC / CONTEXT")
            self.render_diagnostics(self.view.diagnostics)

        if self.view.limitations:
            self.render_section_heading("IMPORTANT LIMITATIONS")
            self.render_limitations(self.view.limitations)

        if self.view.warnings:
            self.render_section_heading("WARNINGS")
            self.render_warnings(self.view.warnings)

    def render_full(self) -> None:
        self.render_standard()

        rationale = self.view.metadata.get("rationale")
        if rationale:
            self.render_section_heading("RECOMMENDATION RATIONALE")
            self.console.print(rationale, style="muted")

        audit_status = self.view.metadata.get("audit_status")
        if audit_status:
            self.render_section_heading("AUDIT VERIFICATION")
            status_style = "status.success" if audit_status == "passed" else "status.warning"
            self.console.print(f"Status: [{status_style}]{audit_status.upper()}[/{status_style}]")

        repro = self.view.metadata.get("reproducibility")
        if repro:
            self.render_section_heading("REPRODUCIBILITY METADATA")
            stoch = repro.get("stochastic", {})
            self.console.print(f"Random seed         : {stoch.get('effective_seed', 'None')}")
            spec_ref = repro.get("specification_reference")
            if spec_ref:
                self.console.print(f"Specification ref   : {spec_ref[:36]}...")
