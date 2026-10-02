"""Public terminal presentation API for PyAutoStat."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

from .adapters import UnsupportedPresentationError, adapt


def show(
    target: Any,
    *,
    detail: str = "standard",
    console: Console | None = None,
) -> None:
    """Render a polished terminal presentation of a PyAutoStat result or profile.

    Parameters
    ----------
    target : Any
        A supported PyAutoStat result object (e.g. ResearchWorkflowResult for Welch t-test
        or Pearson correlation, or a dataset profile dictionary).
    detail : {"compact", "standard", "full"}, default="standard"
        Detail level for terminal display.
    console : Console | None, optional
        Rich Console instance to write to. If None, a standard Console using PYAUTOSTAT_THEME
        is used.
    """
    if detail not in {"compact", "standard", "full"}:
        raise ValueError(
            f"Invalid detail mode '{detail}'. Expected 'compact', 'standard', or 'full'."
        )

    try:
        from rich.console import Console as RichConsole
    except ImportError as exc:
        raise UnsupportedPresentationError(
            "Terminal presentation requires the 'rich' package. Install it with: pip install rich"
        ) from exc

    from .renderers import get_renderer
    from .theme import get_theme

    view = adapt(target, detail=detail)
    theme = get_theme()

    if console is None:
        active_console = RichConsole(theme=theme)
        renderer = get_renderer(view, active_console, detail=detail)
        renderer.render()
    else:
        if theme is not None:
            console.push_theme(theme)
        try:
            renderer = get_renderer(view, console, detail=detail)
            renderer.render()
        finally:
            if theme is not None:
                console.pop_theme()


__all__ = ["UnsupportedPresentationError", "show"]
