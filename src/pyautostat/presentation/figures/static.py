"""Static scientific figure exporter for publication-quality PNG, SVG, and figure-PDF artifacts.

Converts FigureSpec models into standalone image artifacts via Plotly and Kaleido.
Does not recalculate statistics, refit models, or serialize raw dataset rows.
"""

from __future__ import annotations

import io
import math
import re
import xml.etree.ElementTree as ET
from typing import Any

from ...exceptions import ReportError
from .models import FigureSpec, StaticFigureArtifact
from .plotly import spec_to_plotly_figure

SUPPORTED_STATIC_FORMATS: tuple[str, ...] = ("png", "svg", "pdf")
DEFAULT_STATIC_FORMAT: str = "png"
MAX_DIMENSION: int = 10000
MAX_SCALE: float = 8.0
PNG_SIGNATURE: bytes = b"\x89PNG\r\n\x1a\n"


def check_figure_dependencies() -> None:
    """Verify that Plotly and Kaleido optional dependencies are installed."""
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ReportError(
            "Static figure export requires the optional figure dependency.\n"
            'Install: pip install "pyautostat[figures]"'
        ) from exc

    try:
        import kaleido  # noqa: F401
    except ImportError as exc:
        raise ReportError(
            "Static figure export requires the optional figure dependency.\n"
            'Install: pip install "pyautostat[figures]"'
        ) from exc


def check_chrome_available() -> None:
    """Verify that a compatible Chrome or Chromium browser is available for Kaleido."""
    check_figure_dependencies()
    chrome_not_found_errors: tuple[type[Exception], ...]
    try:
        from choreographer.errors import ChromeNotFoundError as ChoreographerError

        chrome_not_found_errors = (ChoreographerError,)
    except ImportError:
        try:
            from kaleido.errors import ChromeNotFoundError as KaleidoError

            chrome_not_found_errors = (KaleidoError,)
        except ImportError:
            chrome_not_found_errors = (FileNotFoundError,)

    try:
        import choreographer.browsers.chromium as c

        c.Chromium.find_browser(skip_local=False)
    except chrome_not_found_errors as exc:
        raise ReportError(
            "Static figure export requires a compatible Chrome or Chromium browser.\n"
            "Install one normally, or run: plotly_get_chrome"
        ) from exc
    except Exception as exc:
        if "chrom" in str(exc).lower() and "not found" in str(exc).lower():
            raise ReportError(
                "Static figure export requires a compatible Chrome or Chromium browser.\n"
                "Install one normally, or run: plotly_get_chrome"
            ) from exc


def validate_static_format(format_name: Any) -> str:
    """Validate and normalize requested static image format."""
    if not isinstance(format_name, str) or not format_name.strip():
        raise ReportError(
            f"Invalid format {format_name!r}. Expected one of: "
            + ", ".join(repr(f) for f in SUPPORTED_STATIC_FORMATS)
        )
    fmt = format_name.strip().lower()
    if fmt not in SUPPORTED_STATIC_FORMATS:
        raise ReportError(
            f"Unsupported static figure format: {format_name!r}. "
            f"Supported formats are: {', '.join(repr(f) for f in SUPPORTED_STATIC_FORMATS)}."
        )
    return fmt


def get_default_dimensions(spec: FigureSpec) -> tuple[int, int]:
    """Compute centralized, row-aware layout width and height in pixels for a FigureSpec."""
    if spec.kind == "estimate_ci":
        n_rows = len(spec.series)
        width = 1000
        height = max(450, 150 + n_rows * 60)
        return width, min(height, 3000)

    if spec.kind in ("forest", "odds_ratio_forest", "pairwise_forest"):
        n_rows = len(spec.series)
        width = 1100
        height = max(450, 180 + n_rows * 35)
        return width, min(height, 5000)

    if spec.kind == "count_heatmap":
        n_y = len(spec.series)
        n_x = len(spec.series[0].categories) if spec.series else 1
        width = max(800, min(1400, 300 + n_x * 120))
        height = max(500, min(1200, 250 + n_y * 80))
        return width, height

    if spec.kind == "cell_profile":
        return 1000, 600

    return 1000, 600


def validate_static_dimensions(
    width: Any,
    height: Any,
    scale: Any,
) -> tuple[int | None, int | None, float]:
    """Validate user-provided width, height, and scale values."""
    validated_width: int | None = None
    validated_height: int | None = None
    validated_scale: float = 2.0

    if width is not None:
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ReportError(f"width must be a positive integer, got: {width!r}")
        if width > MAX_DIMENSION:
            raise ReportError(
                f"width exceeds maximum allowed dimension ({width} > {MAX_DIMENSION})"
            )
        validated_width = width

    if height is not None:
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ReportError(f"height must be a positive integer, got: {height!r}")
        if height > MAX_DIMENSION:
            raise ReportError(
                f"height exceeds maximum allowed dimension ({height} > {MAX_DIMENSION})"
            )
        validated_height = height

    if scale is not None:
        if (
            not isinstance(scale, (int, float))
            or isinstance(scale, bool)
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ReportError(f"scale must be a finite positive number, got: {scale!r}")
        if scale > MAX_SCALE:
            raise ReportError(f"scale exceeds maximum allowed limit ({scale} > {MAX_SCALE})")
        validated_scale = float(scale)

    return validated_width, validated_height, validated_scale


def validate_rendered_bytes(data: bytes, fmt: str) -> None:
    """Validate format integrity and security constraints of rendered image bytes."""
    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
        raise ReportError(f"Static figure export produced empty {fmt.upper()} payload.")

    if fmt == "png":
        if not data.startswith(PNG_SIGNATURE):
            raise ReportError("Generated PNG data is corrupted or missing standard PNG header.")

    elif fmt == "svg":
        try:
            svg_text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReportError(f"Generated SVG data is not valid UTF-8: {exc}") from exc

        if "<svg" not in svg_text:
            raise ReportError("Generated SVG data is missing root <svg> element.")

        try:
            root = ET.fromstring(svg_text)
        except Exception as exc:
            raise ReportError(f"Generated SVG is not valid XML: {exc}") from exc

        # Check for dangerous tags or event handler attributes
        for elem in root.iter():
            tag = elem.tag.split("}")[-1].lower() if "}" in elem.tag else elem.tag.lower()
            if tag in ("script", "object", "embed", "iframe"):
                raise ReportError(
                    f"Security error: SVG contains potentially dangerous tag <{tag}>."
                )
            for attr in elem.attrib:
                attr_lower = attr.lower()
                if attr_lower.startswith("on") or "javascript:" in str(elem.attrib[attr]).lower():
                    raise ReportError(
                        f"Security error: SVG contains executable attribute {attr!r}."
                    )

    elif fmt == "pdf":
        if not data.startswith(b"%PDF"):
            raise ReportError("Generated figure PDF data is missing %PDF header.")
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(data))
            if len(reader.pages) < 1:
                raise ReportError("Generated figure PDF contains 0 pages.")
        except Exception as exc:
            raise ReportError(f"Generated figure PDF is invalid or unreadable: {exc}") from exc


def render_static_figure_bytes(
    fig: Any,
    *,
    fmt: str = "png",
    width: int,
    height: int,
    scale: float = 2.0,
) -> bytes:
    """Render a Plotly figure to static image bytes using Kaleido."""
    check_figure_dependencies()

    try:
        from kaleido.errors import ChromeNotFoundError
    except ImportError:
        ChromeNotFoundError = None  # type: ignore[assignment, misc]

    try:
        img_bytes = fig.to_image(
            format=fmt,
            width=width,
            height=height,
            scale=scale,
        )
    except Exception as exc:
        exc_str = str(exc).lower()
        if (
            (ChromeNotFoundError and isinstance(exc, ChromeNotFoundError))
            or "plotly_get_chrome" in exc_str
            or ("chrome" in exc_str and "not found" in exc_str)
        ):
            raise ReportError(
                "Static figure export requires a compatible Chrome or Chromium browser.\n"
                "Install one normally, or run: plotly_get_chrome"
            ) from exc
        if "kaleido" in exc_str:
            raise ReportError(
                "Static figure export requires the optional figure dependency.\n"
                'Install: pip install "pyautostat[figures]"'
            ) from exc
        raise ReportError(f"Static figure export failed: {exc}") from exc

    validate_rendered_bytes(img_bytes, fmt)
    return img_bytes


def generate_static_artifact(
    spec: FigureSpec,
    index: int,
    *,
    fmt: str = "png",
    width: int | None = None,
    height: int | None = None,
    scale: float = 2.0,
) -> StaticFigureArtifact:
    """Render a FigureSpec into an immutable StaticFigureArtifact."""
    fig = spec_to_plotly_figure(spec)
    if fig is None:
        raise ReportError(f"Unsupported FigureSpec kind: {spec.kind!r}")

    default_w, default_h = get_default_dimensions(spec)
    w = width or default_w
    h = height or default_h

    data = render_static_figure_bytes(fig, fmt=fmt, width=w, height=h, scale=scale)

    kind_slug = re.sub(r"[^a-zA-Z0-9_]", "_", spec.kind).strip("_").lower()
    filename = f"figure_{index:02d}_{kind_slug}.{fmt}"

    return StaticFigureArtifact(
        index=index,
        kind=spec.kind,
        title=spec.title,
        placement=spec.placement,
        format=fmt,
        data=data,
        filename=filename,
        width=w,
        height=h,
    )


__all__ = [
    "check_chrome_available",
    "check_figure_dependencies",
    "generate_static_artifact",
    "get_default_dimensions",
    "render_static_figure_bytes",
    "validate_rendered_bytes",
    "validate_static_dimensions",
    "validate_static_format",
]
