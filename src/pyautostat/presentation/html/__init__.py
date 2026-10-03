"""HTML presentation package for PyAutoStat."""

from __future__ import annotations

from .api import save_html, save_interactive_html, to_html, to_interactive_html
from .renderer import HtmlRenderer
from .report_renderer import ResearchReportHtmlRenderer

__all__ = [
    "HtmlRenderer",
    "ResearchReportHtmlRenderer",
    "save_html",
    "save_interactive_html",
    "to_html",
    "to_interactive_html",
]
