"""Registry and routing for PyAutoStat presentation renderers."""

from __future__ import annotations

from rich.console import Console

from ..models import TerminalView
from .association import AssociationRenderer
from .base import BaseRenderer
from .profile import ProfileRenderer
from .two_group import TwoGroupRenderer
from .workflow_status import WorkflowStatusRenderer

RENDERER_REGISTRY: dict[str, type[BaseRenderer]] = {
    "welch_t": TwoGroupRenderer,
    "pearson_correlation": AssociationRenderer,
}


def get_renderer(view: TerminalView, console: Console, detail: str = "standard") -> BaseRenderer:
    """Return an instantiated renderer for the given TerminalView."""
    if view.family == "family.profile":
        return ProfileRenderer(view, console, detail=detail)

    if "workflow_status" in view.metadata:
        return WorkflowStatusRenderer(view, console, detail=detail)

    method_id = view.metadata.get("method_id")
    if method_id and method_id in RENDERER_REGISTRY:
        renderer_cls = RENDERER_REGISTRY[method_id]
        return renderer_cls(view, console, detail=detail)

    raise ValueError(
        f"No renderer registered for view family '{view.family}' or method '{method_id}'."
    )


__all__ = [
    "AssociationRenderer",
    "BaseRenderer",
    "ProfileRenderer",
    "RENDERER_REGISTRY",
    "TwoGroupRenderer",
    "WorkflowStatusRenderer",
    "get_renderer",
]
