"""Base Rich renderer with reusable layout components for PyAutoStat presentation.

Provides consistent styling, headers, tables, diagnostics, and responsiveness
across all analysis families.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from rich import box
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ..models import (
    DisplayDiagnostic,
    DisplayMetric,
    DisplayTable,
    TerminalView,
)


class BaseRenderer:
    """Base class for all terminal presentation renderers."""

    def __init__(self, view: TerminalView, console: Console, detail: str = "standard") -> None:
        self.view = view
        self.console = console
        self.detail = detail

    @property
    def width(self) -> int:
        """Effective console width."""
        return self.console.width

    @property
    def is_narrow(self) -> bool:
        """Whether terminal is narrow (< 80 columns)."""
        return self.width < 80

    def render(self) -> None:
        """Render the view model according to detail level."""
        if self.detail == "compact":
            self.render_compact()
        elif self.detail == "full":
            self.render_full()
        else:
            self.render_standard()

    def render_compact(self) -> None:
        """Render compact one-line output."""
        if self.view.compact_text:
            self.console.print(self.view.compact_text, highlight=False)
        else:
            self.console.print(f"{self.view.title} | {self.view.subtitle or ''}", highlight=False)

    def render_standard(self) -> None:
        """Render standard view."""
        raise NotImplementedError

    def render_full(self) -> None:
        """Render full view."""
        self.render_standard()

    # -- Shared reusable components ------------------------------------------

    def render_header(self) -> None:
        """Render top title panel."""
        family_style = self.view.family
        title_text = Text(self.view.title, style=family_style)
        subtitle_text = Text(self.view.subtitle, style="bold") if self.view.subtitle else None

        content: RenderableType
        if subtitle_text:
            grid = Table.grid(padding=(0, 0))
            grid.add_row(subtitle_text)
            content = grid
        else:
            content = ""

        panel = Panel(
            content,
            title=title_text,
            title_align="left",
            border_style=family_style,
            box=box.ROUNDED,
            expand=not self.is_narrow,
            padding=(0, 1),
        )
        self.console.print(panel)

    def render_section_heading(self, title: str) -> None:
        """Render a clean section divider or heading."""
        rule = Rule(title=Text(title, style="section"), style="section", align="left")
        self.console.print()
        self.console.print(rule)

    def render_metrics_grid(
        self,
        metrics: Sequence[DisplayMetric],
        label_width: int | None = None,
    ) -> None:
        """Render a clean key-value table without borders."""
        if not metrics:
            return

        table = Table.grid(padding=(0, 2))
        table.add_column("Label", style="label", no_wrap=not self.is_narrow, width=label_width)
        table.add_column("Value", no_wrap=False)

        for m in metrics:
            val_style = m.role if m.role != "default" else ""
            val_text = Text(m.value, style=val_style)
            if m.note:
                val_text.append(f" ({m.note})", style="muted")
            table.add_row(m.label, val_text)

        self.console.print(table)

    def render_key_results_table(
        self,
        metrics: Sequence[DisplayMetric],
    ) -> None:
        """Render primary key result metrics."""
        self.render_metrics_grid(metrics, label_width=22 if not self.is_narrow else None)

    def render_data_table(self, table_model: DisplayTable) -> None:
        """Render a tabular dataset preview or group summary."""
        t = Table(
            title=f"[bold]{table_model.title}[/bold]" if table_model.title else None,
            title_justify="left",
            box=box.SIMPLE_HEAD,
            header_style="bold",
            show_header=True,
            show_edge=False,
            expand=False,
            padding=(0, 1),
        )

        for i, col in enumerate(table_model.columns):
            # Right-align columns that look numeric (not first column)
            non_numeric = {"Levels", "Most common", "Most Common"}
            justify: Literal["left", "right"] = (
                "right" if i > 0 and col not in non_numeric else "left"
            )
            t.add_column(col, justify=justify, no_wrap=not self.is_narrow)

        for row in table_model.rows:
            # Highlight Unavailable cells with muted style
            styled_cells: list[RenderableType] = []
            for cell in row.cells:
                if cell == "Unavailable":
                    styled_cells.append(Text(cell, style="muted"))
                else:
                    styled_cells.append(cell)
            t.add_row(*styled_cells)

        self.console.print(t)

    def render_diagnostics(
        self,
        diagnostics: Sequence[DisplayDiagnostic],
    ) -> None:
        """Render diagnostic and assumption cues with explicit status text."""
        if not diagnostics:
            return

        t = Table.grid(padding=(0, 2))
        t.add_column("Status", no_wrap=True)
        t.add_column("Label", style="label", width=22 if not self.is_narrow else None)
        t.add_column("Detail")

        for diag in diagnostics:
            status_style = "status.review"
            status_text = diag.status.upper()
            if diag.severity == "success":
                status_style = "status.success"
            elif diag.severity == "error":
                status_style = "status.error"
            elif diag.severity == "warning" or diag.severity == "review":
                status_style = "status.warning"
            elif diag.severity == "missing":
                status_style = "status.missing"
            elif diag.severity == "neutral":
                status_style = "muted"

            badge = Text(f"[{status_text}]", style=status_style)
            t.add_row(badge, diag.label, diag.detail or "")

        self.console.print(t)

    def render_interpretation(self, text: str | None) -> None:
        """Render deterministic interpretation narrative."""
        if not text:
            return
        sanitized = (
            text.replace("\u2014", "--")
            .replace("\u2013", "-")
            .replace("\ufffd", "-")
            .replace("\u2212", "-")
        )
        for paragraph in sanitized.split("\n\n"):
            cleaned = paragraph.strip()
            if cleaned:
                self.console.print(cleaned, style="default")

    def render_limitations(self, limitations: Sequence[str]) -> None:
        """Render methodological limitations."""
        if not limitations:
            return
        for lim in limitations:
            self.console.print(f"* [limitation]{lim}[/limitation]")

    def render_warnings(self, warnings: Sequence[str]) -> None:
        """Render analysis warnings."""
        if not warnings:
            return
        for w in warnings:
            self.console.print(f"[!] [status.warning]{w}[/status.warning]")
