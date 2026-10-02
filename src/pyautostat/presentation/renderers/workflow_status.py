"""Renderer for non-completed or blocked workflow status presentation."""

from __future__ import annotations

from .base import BaseRenderer


class WorkflowStatusRenderer(BaseRenderer):
    """Rich presentation renderer for incomplete, blocked, or failed workflows."""

    def render_standard(self) -> None:
        self.render_header()

        status = self.view.metadata.get("workflow_status")

        lbl_w = 16 if not self.is_narrow else None

        if status == "needs_input":
            self.render_section_heading("REQUIRED INFORMATION")
            self.render_diagnostics(self.view.diagnostics)

            if self.view.design_metrics:
                self.render_section_heading("ALREADY KNOWN")
                self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

            self.console.print()
            self.console.print(
                "Supply missing design facts to proceed: assistant.run(..., field=<value>)",
                style="muted",
            )

        elif status in {"data_limited", "unsupported"}:
            self.render_section_heading("BLOCKERS")
            self.render_diagnostics(self.view.diagnostics)

            if status == "unsupported":
                self.console.print("No substitute statistical method was run.", style="muted")

            if self.view.design_metrics:
                self.render_section_heading("SPECIFICATION")
                self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        else:  # failed
            self.render_section_heading("FAILURE DETAILS")
            self.render_diagnostics(self.view.diagnostics)

            if self.view.design_metrics:
                self.render_section_heading("SPECIFICATION")
                self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        if self.view.warnings:
            self.render_section_heading("WARNINGS")
            self.render_warnings(self.view.warnings)

    def render_full(self) -> None:
        self.render_standard()
