"""Public DOCX export API for PyAutoStat research workflows and reports."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from typing import Any

from ...exceptions import ReportError
from ...research_report import ResearchReport
from ..adapters import adapt
from ..models import PresentationView
from .renderer import DocxRenderer
from .report_renderer import ResearchReportDocxRenderer
from .settings import normalize_page_size

VALID_DETAILS = {"compact", "standard", "full"}
VALID_STYLES = {"general", "apa", "ieee"}


def check_docx_available() -> None:
    """Verify that python-docx optional dependency is installed."""
    if importlib.util.find_spec("docx") is None:
        raise ReportError(
            "DOCX export requires the optional Word-report dependency.\n\n"
            "Install:\n"
            '    pip install "pyautostat[docx]"\n'
        )


def to_docx(
    target: Any,
    *,
    detail: str = "standard",
    title: str | None = None,
    style: str = "general",
    page_size: str = "A4",
    landscape: bool = False,
    page_numbers: bool = True,
    include_figures: bool = False,
) -> bytes:
    """Render an editable Microsoft Word (.docx) document from a PyAutoStat result or report.

    The DOCX export layer renders directly from canonical presentation semantics without
    recalculating statistics or reinterpreting results.

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
    page_size : str, default="A4"
        Standard page format ('A4' or 'Letter', case-insensitive).
    landscape : bool, default=False
        Whether to format in landscape orientation (useful for wide tables).
    page_numbers : bool, default=True
        Whether to include dynamic Word page number fields in the footer.
    include_figures : bool, default=False
        Whether to embed canonical scientific figures (PNG format) into the Word report.

    Returns
    -------
    bytes
        Binary OpenXML DOCX data starting with the 'PK' zip signature.
    """
    if detail not in VALID_DETAILS:
        raise ValueError(
            f"Invalid detail mode '{detail}'. Expected 'compact', 'standard', or 'full'."
        )
    if style not in VALID_STYLES:
        raise ReportError("style must be general, apa, or ieee.")
    if title is not None and (not isinstance(title, str) or not title.strip()):
        raise ReportError("title must be a non-empty string when provided.")
    if not isinstance(landscape, bool):
        raise ReportError("landscape must be a Boolean.")
    if not isinstance(page_numbers, bool):
        raise ReportError("page_numbers must be a Boolean.")
    if not isinstance(include_figures, bool):
        raise ReportError("include_figures must be a Boolean.")

    norm_size = normalize_page_size(page_size)
    check_docx_available()

    figures: tuple[Any, ...] = ()
    if include_figures:
        from ..figures.api import to_static_figures

        figures = to_static_figures(target, format="png", detail=detail, scale=2.0)

    renderer: DocxRenderer | ResearchReportDocxRenderer
    if isinstance(target, ResearchReport):
        renderer = ResearchReportDocxRenderer(
            target,
            detail=detail,
            title=title,
            style=style,
            figures=figures,
            page_size=norm_size,
            landscape=landscape,
            page_numbers=page_numbers,
        )
    elif isinstance(target, PresentationView):
        renderer = DocxRenderer(
            target,
            detail=detail,
            title=title,
            style=style,
            figures=figures,
            page_size=norm_size,
            landscape=landscape,
            page_numbers=page_numbers,
        )
    else:
        view = adapt(target, detail=detail)
        renderer = DocxRenderer(
            view,
            detail=detail,
            title=title,
            style=style,
            figures=figures,
            page_size=norm_size,
            landscape=landscape,
            page_numbers=page_numbers,
        )

    doc = renderer.render()
    buffer = io.BytesIO()
    doc.save(buffer)
    docx_bytes = buffer.getvalue()

    if not docx_bytes.startswith(b"PK"):
        raise ReportError("Generated output does not start with valid OpenXML PK signature.")

    return docx_bytes


def save_docx(
    target: Any,
    path: str | Path,
    *,
    detail: str = "standard",
    title: str | None = None,
    style: str = "general",
    page_size: str = "A4",
    landscape: bool = False,
    page_numbers: bool = True,
    include_figures: bool = False,
    overwrite: bool = False,
) -> Path:
    """Save an editable Microsoft Word (.docx) report with overwrite protection.

    Parameters
    ----------
    target : Any
        A supported PyAutoStat object (e.g. ResearchWorkflowResult, AnalysisResult,
        ResearchReport, or PresentationView).
    path : str or Path
        Destination file path. Must end with '.docx'.
    detail : {"compact", "standard", "full"}, default="standard"
        Detail level for the report.
    title : str | None, optional
        Custom title override for the report header.
    style : {"general", "apa", "ieee"}, default="general"
        Reporting style convention ('general', 'apa', or 'ieee').
    page_size : str, default="A4"
        Standard page format ('A4' or 'Letter', case-insensitive).
    landscape : bool, default=False
        Whether to format in landscape orientation.
    page_numbers : bool, default=True
        Whether to include running page numbers in the footer.
    include_figures : bool, default=False
        Whether to embed canonical scientific figures (PNG format) into the Word report.
    overwrite : bool, default=False
        Whether to overwrite an existing destination file.

    Returns
    -------
    Path
        Path to the saved DOCX file.
    """
    destination = Path(path)
    if destination.suffix.lower() != ".docx":
        raise ReportError("DOCX export destination must have a .docx extension.")
    if destination.exists() and not overwrite:
        raise ReportError(f"Report destination already exists: {destination}.")

    if isinstance(target, ResearchReport):
        return target.save_docx(
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

    docx_bytes = to_docx(
        target,
        detail=detail,
        title=title,
        style=style,
        page_size=page_size,
        landscape=landscape,
        page_numbers=page_numbers,
        include_figures=include_figures,
    )

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(docx_bytes)
        return destination
    except ReportError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ReportError(f"Could not write DOCX report to {path!r}: {exc}") from exc
