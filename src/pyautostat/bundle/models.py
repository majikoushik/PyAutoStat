"""Data models, options, and schemas for research export bundles."""

from __future__ import annotations

from dataclasses import dataclass

BUNDLE_SCHEMA_VERSION: int = 1

DEFAULT_BUNDLE_FORMATS: tuple[str, ...] = ("html", "json", "csv")

ALL_BUNDLE_FORMATS: tuple[str, ...] = (
    "html",
    "interactive_html",
    "pdf",
    "docx",
    "json",
    "csv",
    "markdown",
    "latex",
)

CANONICAL_FORMAT_ORDER: tuple[str, ...] = (
    "html",
    "interactive_html",
    "pdf",
    "docx",
    "json",
    "csv",
    "markdown",
    "latex",
)


@dataclass(frozen=True)
class BundleOptions:
    """Configuration options for research bundle generation."""

    detail: str = "full"
    style: str = "general"
    title: str | None = None
    include_figures: bool = False
    page_size: str = "A4"
    landscape: bool = False
    page_numbers: bool = True


@dataclass(frozen=True)
class BundleVerificationResult:
    """Immutable result of verifying a research bundle's manifest and artifact integrity."""

    valid: bool
    schema_version: int | None
    checked_files: int
    missing_files: tuple[str, ...]
    unexpected_files: tuple[str, ...]
    checksum_mismatches: tuple[str, ...]
    size_mismatches: tuple[str, ...]
    errors: tuple[str, ...]
