"""HTML presentation package for PyAutoStat."""

from __future__ import annotations

from .api import save_html, to_html
from .renderer import HtmlRenderer
from .report_renderer import ResearchReportHtmlRenderer

__all__ = [
    "HtmlRenderer",
    "ResearchReportHtmlRenderer",
    "save_html",
    "to_html",
]
