"""Renderers for research governance, audit, plans, completeness, and reproducibility."""

from __future__ import annotations

from .base import BaseRenderer


class AnalysisPlanRenderer(BaseRenderer):
    """Rich presentation renderer for statistical analysis plans."""

    def render_standard(self) -> None:
        self.render_header()

        self.render_section_heading("PLAN SPECIFICATION")
        lbl_w = 20 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        if self.view.tables:
            for table_model in self.view.tables:
                self.render_section_heading(table_model.title or "PLAN ELEMENT")
                self.render_data_table(table_model)

        if self.view.interpretation:
            self.render_section_heading("PLANNED METHOD RATIONALE")
            self.render_interpretation(self.view.interpretation)

        if self.view.diagnostics:
            self.render_section_heading("GOVERNANCE CONTEXT")
            self.render_diagnostics(self.view.diagnostics)

        if self.view.limitations:
            self.render_section_heading("IMPORTANT LIMITATIONS")
            self.render_limitations(self.view.limitations)

        if self.view.warnings:
            self.render_section_heading("WARNINGS")
            self.render_warnings(self.view.warnings)

    def render_full(self) -> None:
        self.render_standard()


class PlanAdherenceRenderer(BaseRenderer):
    """Rich presentation renderer for plan adherence results."""

    def render_standard(self) -> None:
        self.render_header()

        self.render_section_heading("ADHERENCE SUMMARY")
        lbl_w = 20 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        if self.view.tables:
            for table_model in self.view.tables:
                self.render_section_heading(table_model.title or "ADHERENCE TABLE")
                self.render_data_table(table_model)

        if self.view.interpretation:
            self.render_section_heading("REASON / DETAILS")
            self.render_interpretation(self.view.interpretation)

        if self.view.limitations:
            self.render_section_heading("IMPORTANT LIMITATIONS")
            self.render_limitations(self.view.limitations)

    def render_full(self) -> None:
        self.render_standard()


class CompletenessRenderer(BaseRenderer):
    """Rich presentation renderer for reporting completeness results."""

    def render_standard(self) -> None:
        self.render_header()

        self.render_section_heading("COMPLETENESS ACCOUNTING")
        lbl_w = 20 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        if self.view.tables:
            for table_model in self.view.tables:
                self.render_section_heading(table_model.title or "CHECKLIST ITEMS")
                self.render_data_table(table_model)

        if self.view.limitations:
            self.render_section_heading("IMPORTANT LIMITATIONS")
            self.render_limitations(self.view.limitations)

    def render_full(self) -> None:
        self.render_standard()


class AuditRenderer(BaseRenderer):
    """Rich presentation renderer for scientific consistency audits."""

    def render_standard(self) -> None:
        self.render_header()

        self.render_section_heading("AUDIT SUMMARY")
        lbl_w = 22 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        if self.view.tables:
            for table_model in self.view.tables:
                self.render_section_heading(table_model.title or "AUDIT FINDINGS")
                self.render_data_table(table_model)

        if self.view.diagnostics:
            self.render_section_heading("AUDIT INVARIANTS")
            self.render_diagnostics(self.view.diagnostics)

        if self.view.limitations:
            self.render_section_heading("IMPORTANT LIMITATIONS")
            self.render_limitations(self.view.limitations)

    def render_full(self) -> None:
        self.render_standard()


class ReproducibilityRenderer(BaseRenderer):
    """Rich presentation renderer for reproducibility and replay outcomes."""

    def render_standard(self) -> None:
        self.render_header()

        self.render_section_heading("REPRODUCIBILITY METADATA")
        lbl_w = 22 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        if self.view.tables:
            for table_model in self.view.tables:
                self.render_section_heading(table_model.title or "DISCREPANCIES")
                self.render_data_table(table_model)

        if self.view.limitations:
            self.render_section_heading("IMPORTANT LIMITATIONS")
            self.render_limitations(self.view.limitations)

    def render_full(self) -> None:
        self.render_standard()


class DecisionLedgerRenderer(BaseRenderer):
    """Rich presentation renderer for decision ledgers."""

    def render_standard(self) -> None:
        self.render_header()

        self.render_section_heading("LEDGER OVERVIEW")
        lbl_w = 20 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        if self.view.tables:
            for table_model in self.view.tables:
                self.render_section_heading(table_model.title or "EVENT HISTORY")
                self.render_data_table(table_model)

        if self.view.limitations:
            self.render_section_heading("IMPORTANT LIMITATIONS")
            self.render_limitations(self.view.limitations)

    def render_full(self) -> None:
        self.render_standard()


class SessionRenderer(BaseRenderer):
    """Rich presentation renderer for session snapshots."""

    def render_standard(self) -> None:
        self.render_header()

        self.render_section_heading("SESSION OVERVIEW")
        lbl_w = 20 if not self.is_narrow else None
        self.render_metrics_grid(self.view.design_metrics, label_width=lbl_w)

        if self.view.tables:
            for table_model in self.view.tables:
                self.render_section_heading(table_model.title or "SESSION CONTENTS")
                self.render_data_table(table_model)

        if self.view.limitations:
            self.render_section_heading("IMPORTANT LIMITATIONS")
            self.render_limitations(self.view.limitations)

        if self.view.warnings:
            self.render_section_heading("WARNINGS")
            self.render_warnings(self.view.warnings)

    def render_full(self) -> None:
        self.render_standard()
