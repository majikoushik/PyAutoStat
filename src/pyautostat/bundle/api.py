"""Public interface for creating, saving, and verifying research export bundles."""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any

from ..exceptions import ReportError
from ..research_report import ResearchReport
from ..workflow import ResearchWorkflowResult
from .assembler import BundleAssembler
from .models import BundleOptions


def to_bundle(
    target: Any,
    *,
    formats: Sequence[str] | None = None,
    detail: str = "full",
    style: str = "general",
    title: str | None = None,
    include_figures: bool = False,
    page_size: str = "A4",
    landscape: bool = False,
    page_numbers: bool = True,
) -> bytes:
    """Package a ResearchReport, ResearchWorkflowResult, or AnalysisResult into a verifiable bundle.

    Returns raw in-memory ZIP archive bytes.
    """
    options = BundleOptions(
        detail=detail,
        style=style,
        title=title,
        include_figures=include_figures,
        page_size=page_size,
        landscape=landscape,
        page_numbers=page_numbers,
    )
    assembler = BundleAssembler(target, formats=formats, options=options)
    return assembler.assemble()


def save_bundle(
    target: Any,
    path: str | pathlib.Path,
    *,
    formats: Sequence[str] | None = None,
    detail: str = "full",
    style: str = "general",
    title: str | None = None,
    include_figures: bool = False,
    page_size: str = "A4",
    landscape: bool = False,
    page_numbers: bool = True,
    overwrite: bool = False,
) -> pathlib.Path:
    """Assemble and write a research export bundle to a local ZIP file.

    Parameters
    ----------
    target : ResearchReport | ResearchWorkflowResult | AnalysisResult
        Statistical analysis artifact to package.
    path : str | pathlib.Path
        Output destination file path; must end with '.zip'.
    formats : Sequence[str], optional
        Requested export formats. Defaults to ('html', 'json', 'csv').
    detail : str, default 'full'
        Detail mode ('compact', 'standard', 'full').
    style : str, default 'general'
        Reporting style ('general', 'apa', 'ieee').
    title : str, optional
        Custom document title overriding default.
    include_figures : bool, default False
        Whether to generate figures for interactive HTML and PDF.
    page_size : str, default 'A4'
        Page geometry ('A4' or 'Letter').
    landscape : bool, default False
        Whether to format page orientation as landscape.
    page_numbers : bool, default True
        Whether to include page numbers in printed/exported documents.
    overwrite : bool, default False
        Whether to overwrite an existing destination file.

    Returns
    -------
    pathlib.Path
        Resolved path to the saved bundle archive.
    """
    path_obj = pathlib.Path(path)
    if path_obj.suffix.lower() != ".zip":
        raise ReportError(f"Bundle destination path must have a '.zip' extension, got: {path!r}")

    if path_obj.exists() and not overwrite:
        raise ReportError(f"Destination file {str(path_obj)!r} already exists and overwrite=False.")

    # If target is a ResearchReport, delegate directly to report.save_bundle
    # to ensure a single audit entry
    if isinstance(target, ResearchReport):
        return target.save_bundle(
            path_obj,
            formats=formats,
            detail=detail,
            style=style,
            title=title,
            include_figures=include_figures,
            page_size=page_size,
            landscape=landscape,
            page_numbers=page_numbers,
            overwrite=overwrite,
        )

    bundle_bytes = to_bundle(
        target,
        formats=formats,
        detail=detail,
        style=style,
        title=title,
        include_figures=include_figures,
        page_size=page_size,
        landscape=landscape,
        page_numbers=page_numbers,
    )

    path_obj.parent.mkdir(parents=True, exist_ok=True)
    try:
        path_obj.write_bytes(bundle_bytes)
    except OSError as exc:
        raise ReportError(f"Failed to write bundle archive to {str(path_obj)!r}: {exc}") from exc

    # Record save action for workflow if attached
    if isinstance(target, ResearchWorkflowResult) and target.report is not None:
        on_save = getattr(target.report, "_on_save", None)
        if on_save is not None:
            on_save("bundle")

    return path_obj.resolve()
