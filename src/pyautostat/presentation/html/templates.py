"""Template and section formatting helpers for HTML presentation."""

from __future__ import annotations


def styled_heading(heading: str, index: int | None = None, style: str = "general") -> str:
    """Format section heading with optional IEEE numbering."""
    if style == "ieee" and index is not None:
        return f"{index}. {heading}"
    return heading


def section_anchor(title: str) -> str:
    """Generate a clean URL-safe HTML id attribute from a section title."""
    return title.lower().replace(" ", "-").replace("'", "").replace('"', "")
