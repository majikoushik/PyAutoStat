"""PyAutoStat research export bundle module.

Packages canonical reports, tabular exports, machine-readable records, and execution
provenance into verifiable, reproducible ZIP archives.
"""

from __future__ import annotations

from .api import save_bundle, to_bundle
from .models import (
    ALL_BUNDLE_FORMATS,
    BUNDLE_SCHEMA_VERSION,
    DEFAULT_BUNDLE_FORMATS,
    BundleOptions,
    BundleVerificationResult,
)
from .verification import verify_bundle

__all__ = [
    "ALL_BUNDLE_FORMATS",
    "BUNDLE_SCHEMA_VERSION",
    "DEFAULT_BUNDLE_FORMATS",
    "BundleOptions",
    "BundleVerificationResult",
    "save_bundle",
    "to_bundle",
    "verify_bundle",
]
