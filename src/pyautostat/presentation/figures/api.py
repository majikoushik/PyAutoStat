"""Public static scientific figure export API for PyAutoStat research workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...exceptions import ReportError
from .adapters import build_figure_specs
from .models import StaticFigureArtifact
from .static import (
    check_chrome_available,
    generate_static_artifact,
    validate_static_dimensions,
    validate_static_format,
)

VALID_DETAILS = {"compact", "standard", "full"}


def to_static_figures(
    target: Any,
    *,
    format: str = "png",
    detail: str = "standard",
    width: int | None = None,
    height: int | None = None,
    scale: float = 2.0,
) -> tuple[StaticFigureArtifact, ...]:
    """Export publication-ready static figures from a PyAutoStat workflow, result, or report.

    Renders immutable FigureSpec models into standalone image artifacts (PNG, SVG, or figure-PDF).
    Does not recalculate statistics or expose raw dataset rows.

    Parameters
    ----------
    target : Any
        A supported PyAutoStat object (e.g. ResearchWorkflowResult, AnalysisResult,
        or ResearchReport).
    format : {"png", "svg", "pdf"}, default="png"
        Desired static image export format.
    detail : {"compact", "standard", "full"}, default="standard"
        Detail level for the analysis context.
    width : int or None, optional
        Target image width in pixels. If None, centralized default is used.
    height : int or None, optional
        Target image height in pixels. If None, centralized default is used.
    scale : float, default=2.0
        Resolution scale multiplier (e.g. 2.0 for high-DPI/retina publication quality).

    Returns
    -------
    tuple[StaticFigureArtifact, ...]
        Ordered sequence of rendered static figure artifacts. If the target has no
        figure specifications, returns an empty tuple without requiring Kaleido or Chrome.
    """
    if detail not in VALID_DETAILS:
        raise ValueError(
            f"Invalid detail mode '{detail}'. Expected 'compact', 'standard', or 'full'."
        )

    norm_fmt = validate_static_format(format)
    val_w, val_h, val_scale = validate_static_dimensions(width, height, scale)

    # 1. Build FigureSpecs from canonical target
    specs = build_figure_specs(target)

    # 2. If no figure specs exist, return empty tuple without requiring Kaleido or browser
    if not specs:
        return ()

    # 3. Check optional dependencies and browser availability
    check_chrome_available()

    # 4. Render figures in deterministic order
    artifacts: list[StaticFigureArtifact] = []
    for idx, spec in enumerate(specs, start=1):
        artifact = generate_static_artifact(
            spec,
            index=idx,
            fmt=norm_fmt,
            width=val_w,
            height=val_h,
            scale=val_scale,
        )
        artifacts.append(artifact)

    return tuple(artifacts)


def save_static_figures(
    target: Any,
    directory: str | Path,
    *,
    format: str = "png",
    detail: str = "standard",
    width: int | None = None,
    height: int | None = None,
    scale: float = 2.0,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Save publication-ready static figures to a target directory with overwrite protection.

    Parameters
    ----------
    target : Any
        A supported PyAutoStat object (e.g. ResearchWorkflowResult, AnalysisResult,
        or ResearchReport).
    directory : str or Path
        Target directory to save figure artifacts.
    format : {"png", "svg", "pdf"}, default="png"
        Desired static image export format.
    detail : {"compact", "standard", "full"}, default="standard"
        Detail level for the analysis context.
    width : int or None, optional
        Target image width in pixels.
    height : int or None, optional
        Target image height in pixels.
    scale : float, default=2.0
        Resolution scale multiplier.
    overwrite : bool, default=False
        Whether to overwrite existing files.

    Returns
    -------
    tuple[Path, ...]
        Tuple of paths to saved static figure files.
    """
    artifacts = to_static_figures(
        target,
        format=format,
        detail=detail,
        width=width,
        height=height,
        scale=scale,
    )

    if not artifacts:
        return ()

    dest_dir = Path(directory)
    target_paths = [dest_dir / art.filename for art in artifacts]

    # Preflight collisions before writing any file
    if not overwrite:
        for p in target_paths:
            if p.exists():
                raise ReportError(f"Destination file already exists and overwrite=False: {p}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    for art, p in zip(artifacts, target_paths, strict=True):
        p.write_bytes(art.data)

    return tuple(target_paths)


__all__ = [
    "save_static_figures",
    "to_static_figures",
]
