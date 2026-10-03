"""HTML presentation package for PyAutoStat."""

from __future__ import annotations

from .api import save_html, to_html
from .renderer import HtmlRenderer

__all__ = [
    "HtmlRenderer",
    "save_html",
    "to_html",
]
