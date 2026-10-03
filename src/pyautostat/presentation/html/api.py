"""Public HTML presentation API for PyAutoStat."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...exceptions import ReportError
from ...research_report import ResearchReport
from ..adapters import adapt
from ..models import PresentationView
from .renderer import HtmlRenderer

VALID_DETAILS = {"compact", "standard", "full"}
VALID_STYLES = {"general", "apa", "ieee"}


def to_html(
    target: Any,
    *,
    detail: str = "standard",
    title: str | None = None,
    style: str = "general",
) -> str:
    """Render a polished, self-contained HTML report of a PyAutoStat result or workflow.

    Parameters
    ----------
    target : Any
        A supported PyAutoStat object (e.g. ResearchWorkflowResult, AnalysisResult,
        ResearchReport, or PresentationView).
    detail : {"compact", "standard", "full"}, default="standard"
        Detail level for the HTML report.
    title : str | None, optional
        Custom title override for the report header.
    style : {"general", "apa", "ieee"}, default="general"
        Reporting style convention.

    Returns
    -------
    str
        Self-contained HTML5 document string.
    """
    if detail not in VALID_DETAILS:
        raise ValueError(
            f"Invalid detail mode '{detail}'. Expected 'compact', 'standard', or 'full'."
        )
    if style not in VALID_STYLES:
        raise ReportError("style must be general, apa, or ieee.")
    if title is not None and (not isinstance(title, str) or not title.strip()):
        raise ReportError("title must be a non-empty string when provided.")

    if isinstance(target, ResearchReport):
        return target.to_html(style=style, detail=detail, title=title)

    if isinstance(target, PresentationView):
        view = target
        if title is not None:
            view = PresentationView(
                title=title.strip(),
                subtitle=view.subtitle,
                family=view.family,
                design_metrics=view.design_metrics,
                key_metrics=view.key_metrics,
                tables=view.tables,
                diagnostics=view.diagnostics,
                interpretation=view.interpretation,
                limitations=view.limitations,
                warnings=view.warnings,
                metadata=view.metadata,
                compact_text=view.compact_text,
            )
        renderer = HtmlRenderer(view, detail=detail, title=title, style=style)
        return renderer.render()

    view = adapt(target, detail=detail)
    if title is not None:
        view = PresentationView(
            title=title.strip(),
            subtitle=view.subtitle,
            family=view.family,
            design_metrics=view.design_metrics,
            key_metrics=view.key_metrics,
            tables=view.tables,
            diagnostics=view.diagnostics,
            interpretation=view.interpretation,
            limitations=view.limitations,
            warnings=view.warnings,
            metadata=view.metadata,
            compact_text=view.compact_text,
        )
    renderer = HtmlRenderer(view, detail=detail, title=title, style=style)
    return renderer.render()


def save_html(
    target: Any,
    path: str | Path,
    *,
    detail: str = "standard",
    title: str | None = None,
    style: str = "general",
    overwrite: bool = False,
) -> Path:
    """Save a polished, self-contained HTML report to a file with overwrite protection.

    Parameters
    ----------
    target : Any
        A supported PyAutoStat object (e.g. ResearchWorkflowResult, AnalysisResult,
        ResearchReport, or PresentationView).
    path : str or Path
        Destination file path.
    detail : {"compact", "standard", "full"}, default="standard"
        Detail level for the HTML report.
    title : str | None, optional
        Custom title override for the report header.
    style : {"general", "apa", "ieee"}, default="general"
        Reporting style convention.
    overwrite : bool, default=False
        Whether to overwrite an existing destination file.

    Returns
    -------
    Path
        Path to the saved HTML report file.
    """
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise ReportError(f"Report destination already exists: {destination}.")

    if isinstance(target, ResearchReport):
        return target.save_html(
            destination,
            style=style,
            detail=detail,
            title=title,
            overwrite=overwrite,
        )

    content = to_html(target, detail=detail, title=title, style=style)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return destination
    except ReportError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ReportError(f"Could not write report to {path!r}: {exc}") from exc
