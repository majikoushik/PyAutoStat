"""Public PDF export API for PyAutoStat research workflows and reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...exceptions import ReportError
from ...research_report import ResearchReport
from ..figures.adapters import build_figure_specs
from ..html.api import to_html, to_interactive_html
from ..html.plotly_renderer import check_plotly_available
from .backend import html_to_pdf_bytes
from .settings import normalize_page_size

VALID_DETAILS = {"compact", "standard", "full"}
VALID_STYLES = {"general", "apa", "ieee"}


def to_pdf(
    target: Any,
    *,
    detail: str = "standard",
    title: str | None = None,
    style: str = "general",
    include_figures: bool = False,
    page_size: str = "A4",
    landscape: bool = False,
    page_numbers: bool = True,
) -> bytes:
    """Render a publication-ready PDF document from a PyAutoStat result or workflow.

    The PDF export layer reuses the canonical HTML report presentation and prints it
    via headless Chromium. It does not recalculate statistics or infer new results.

    Parameters
    ----------
    target : Any
        A supported PyAutoStat object (e.g. ResearchWorkflowResult, AnalysisResult,
        ResearchReport, or PresentationView).
    detail : {"compact", "standard", "full"}, default="standard"
        Detail level for the report.
    title : str | None, optional
        Custom title override for the report header.
    style : {"general", "apa", "ieee"}, default="general"
        Reporting style convention ('general', 'apa', or 'ieee').
    include_figures : bool, default=False
        Whether to generate and embed supplementary interactive figures in the PDF.
        When False (default), or when the target produces no figures, Plotly is not
        required.
    page_size : str, default="A4"
        Standard page format ('A4' or 'Letter', case-insensitive).
    landscape : bool, default=False
        Whether to print in landscape orientation (useful for wide tables).
    page_numbers : bool, default=True
        Whether to include running page numbers in the footer.

    Returns
    -------
    bytes
        Binary PDF data starting with the '%PDF' signature.
    """
    if detail not in VALID_DETAILS:
        raise ValueError(
            f"Invalid detail mode '{detail}'. Expected 'compact', 'standard', or 'full'."
        )
    if style not in VALID_STYLES:
        raise ReportError("style must be general, apa, or ieee.")
    if title is not None and (not isinstance(title, str) or not title.strip()):
        raise ReportError("title must be a non-empty string when provided.")
    if not isinstance(include_figures, bool):
        raise ReportError("include_figures must be a Boolean.")
    if not isinstance(landscape, bool):
        raise ReportError("landscape must be a Boolean.")
    if not isinstance(page_numbers, bool):
        raise ReportError("page_numbers must be a Boolean.")

    norm_size = normalize_page_size(page_size)

    if not include_figures:
        html = to_html(target, detail=detail, title=title, style=style)
        return html_to_pdf_bytes(
            html,
            page_size=norm_size,
            landscape=landscape,
            page_numbers=page_numbers,
            wait_for_figures=False,
        )

    # Figures requested: inspect whether authoritative FigureSpecs exist
    figure_specs = build_figure_specs(target)
    if not figure_specs:
        # Table-only method or PresentationView: render static report without requiring Plotly
        html = to_html(target, detail=detail, title=title, style=style)
        return html_to_pdf_bytes(
            html,
            page_size=norm_size,
            landscape=landscape,
            page_numbers=page_numbers,
            wait_for_figures=False,
        )

    # Figures exist: require Plotly and wait for rendering
    check_plotly_available()
    html = to_interactive_html(
        target,
        detail=detail,
        title=title,
        style=style,
        include_figures=True,
    )
    return html_to_pdf_bytes(
        html,
        page_size=norm_size,
        landscape=landscape,
        page_numbers=page_numbers,
        wait_for_figures=True,
    )


def save_pdf(
    target: Any,
    path: str | Path,
    *,
    detail: str = "standard",
    title: str | None = None,
    style: str = "general",
    include_figures: bool = False,
    page_size: str = "A4",
    landscape: bool = False,
    page_numbers: bool = True,
    overwrite: bool = False,
) -> Path:
    """Save a publication-ready PDF document to a file with overwrite protection.

    Parameters
    ----------
    target : Any
        A supported PyAutoStat object (e.g. ResearchWorkflowResult, AnalysisResult,
        ResearchReport, or PresentationView).
    path : str or Path
        Destination file path. Must end with '.pdf'.
    detail : {"compact", "standard", "full"}, default="standard"
        Detail level for the report.
    title : str | None, optional
        Custom title override for the report header.
    style : {"general", "apa", "ieee"}, default="general"
        Reporting style convention ('general', 'apa', or 'ieee').
    include_figures : bool, default=False
        Whether to generate and embed supplementary figures in the PDF.
    page_size : str, default="A4"
        Standard page format ('A4' or 'Letter', case-insensitive).
    landscape : bool, default=False
        Whether to print in landscape orientation.
    page_numbers : bool, default=True
        Whether to include running page numbers in the footer.
    overwrite : bool, default=False
        Whether to overwrite an existing destination file.

    Returns
    -------
    Path
        Path to the saved PDF file.
    """
    destination = Path(path)
    if destination.suffix.lower() != ".pdf":
        raise ReportError("PDF export destination must have a .pdf extension.")
    if destination.exists() and not overwrite:
        raise ReportError(f"Report destination already exists: {destination}.")

    if isinstance(target, ResearchReport):
        return target.save_pdf(
            destination,
            style=style,
            detail=detail,
            title=title,
            include_figures=include_figures,
            page_size=page_size,
            landscape=landscape,
            page_numbers=page_numbers,
            overwrite=overwrite,
        )

    pdf_bytes = to_pdf(
        target,
        detail=detail,
        title=title,
        style=style,
        include_figures=include_figures,
        page_size=page_size,
        landscape=landscape,
        page_numbers=page_numbers,
    )

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(pdf_bytes)
        return destination
    except ReportError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ReportError(f"Could not write PDF report to {path!r}: {exc}") from exc
