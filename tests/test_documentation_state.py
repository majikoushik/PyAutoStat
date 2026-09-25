"""Regression checks for the permanent post-roadmap documentation layout."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_development_roadmap_is_retired_without_stale_markdown_links():
    assert not (ROOT / "DEVELOPMENT_ROADMAP.md").exists()
    references = []
    markdown_files = [
        *ROOT.glob("*.md"),
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "examples").rglob("*.md"),
    ]
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        if "DEVELOPMENT_ROADMAP" in text:
            references.append(path.relative_to(ROOT).as_posix())
    assert references == []


def test_active_documentation_has_no_numbered_phase_language():
    active_documents = [
        ROOT / "AGENTS.md",
        ROOT / "API_REFERENCE.md",
        ROOT / "CHANGELOG.md",
        ROOT / "PRODUCT_VISION.md",
        ROOT / "README.md",
        ROOT / "ROADMAP.md",
        *(ROOT / "docs").glob("*.md"),
    ]
    stale = []
    for path in active_documents:
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bphase\s+\d+\b", text, flags=re.IGNORECASE):
            stale.append(path.relative_to(ROOT).as_posix())
    assert stale == []


def test_readme_presents_beginner_workflow_before_advanced_topics():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    sections = [
        "## Installation",
        "## Quick start: profile a DataFrame",
        "## Guided analysis",
        "## When information is missing",
        "## Advanced workflows",
    ]
    positions = [readme.index(section) for section in sections]
    assert positions == sorted(positions)


def test_roadmap_is_future_facing_and_covers_research_lifecycle():
    roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    assert "# PyAutoStat roadmap" in roadmap
    assert "Design and planning" in roadmap
    assert "Data" in roadmap
    assert "Analysis" in roadmap
    assert "Interpretation" in roadmap
    assert "Reporting" in roadmap
    assert "Reproducibility" in roadmap
    assert re.search(r"\bphase\s+\d+\b", roadmap, flags=re.IGNORECASE) is None
