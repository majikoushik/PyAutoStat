"""Manifest generation, record modeling, and SHA-256 hashing for bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .. import __version__
from .models import BUNDLE_SCHEMA_VERSION


def compute_sha256(data: bytes) -> str:
    """Compute standard SHA-256 hexadecimal digest for raw byte payload."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class BundleFileRecord:
    """Metadata record for a single artifact included in the research bundle."""

    path: str
    role: str
    format: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def build_manifest_dict(
    *,
    target_type: str,
    report_status: str,
    method_id: str | None,
    detail: str,
    style: str,
    include_figures: bool,
    files: list[BundleFileRecord],
) -> dict[str, Any]:
    """Assemble a deterministic manifest dictionary adhering to BUNDLE_SCHEMA_VERSION."""
    sorted_files = sorted(files, key=lambda r: r.path)
    return {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "pyautostat_version": __version__,
        "target_type": target_type,
        "report_status": report_status,
        "method_id": method_id,
        "detail": detail,
        "style": style,
        "include_figures": include_figures,
        "files": [f.to_dict() for f in sorted_files],
    }


def serialize_manifest(manifest_dict: dict[str, Any]) -> bytes:
    """Serialize manifest to deterministic formatted JSON bytes."""
    return json.dumps(manifest_dict, indent=2, ensure_ascii=False).encode("utf-8")
