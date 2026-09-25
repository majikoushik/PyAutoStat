"""Regression checks for the permanent post-roadmap documentation layout."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _local_markdown_targets(path: Path) -> list[Path]:
    targets = []
    for raw_target in re.findall(r"\[[^]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        target = raw_target.strip("<>").split("#", maxsplit=1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        targets.append((path.parent / target).resolve())
    return targets


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


def test_capability_documents_are_consolidated_and_indexed():
    assert not (ROOT / "docs" / "CONTROLLED_MVP.md").exists()
    assert not (ROOT / "docs" / "FINAL_CAPABILITY_MATRIX.md").exists()
    assert (ROOT / "docs" / "CAPABILITIES.md").is_file()
    assert (ROOT / "docs" / "README.md").is_file()


def test_internal_documentation_links_resolve():
    documents = [
        *ROOT.glob("*.md"),
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / "examples").glob("*.md"),
    ]
    missing = {
        path.relative_to(ROOT).as_posix(): [
            target.as_posix() for target in _local_markdown_targets(path) if not target.exists()
        ]
        for path in documents
    }
    assert {path: targets for path, targets in missing.items() if targets} == {}


def test_example_commands_and_feature_oriented_filenames_are_current():
    guide = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    commands = re.findall(r"python (examples/[^\s`]+\.py)", guide)
    assert commands
    assert all((ROOT / command).is_file() for command in commands)
    assert not list((ROOT / "examples").glob("*phase*.py"))
    assert not list((ROOT / "tests").glob("test_phase*.py"))


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
