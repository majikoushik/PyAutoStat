"""Security and path-safety controls for research export bundles."""

from __future__ import annotations

import posixpath
import re
import zipfile
from typing import Any

from ..exceptions import ReportError

MAX_MEMBER_COUNT: int = 500
MAX_TOTAL_UNCOMPRESSED_BYTES: int = 100 * 1024 * 1024  # 100 MB
MAX_SINGLE_MEMBER_BYTES: int = 50 * 1024 * 1024  # 50 MB

_SAFE_TABLE_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_\-]")


def sanitize_table_id(table_id: Any) -> str:
    """Sanitize table identifier for safe, traversal-free filenames.

    Only alphanumeric characters, underscores, and dashes are retained.
    """
    if not table_id:
        return "table"
    cleaned = _SAFE_TABLE_ID_PATTERN.sub("_", str(table_id)).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned if cleaned else "table"


def validate_zip_member_path(path: str) -> None:
    """Validate internal ZIP member path to prevent ZIP-slip and path traversal.

    Raises ReportError if path is absolute, contains backslashes, drive letters,
    or directory traversal segments ('..').
    """
    if not isinstance(path, str) or not path.strip():
        raise ReportError("ZIP member path cannot be empty.")

    if "\\" in path:
        raise ReportError(f"ZIP member path must use forward slashes, got: {path!r}")

    # Check for Windows drive prefix (e.g. C:)
    if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        raise ReportError(f"ZIP member path cannot contain drive specifications: {path!r}")

    if path.startswith("/"):
        raise ReportError(f"ZIP member path cannot be absolute: {path!r}")

    if "//" in path:
        raise ReportError(f"ZIP member path contains empty segment: {path!r}")

    stripped = path.rstrip("/")
    segments = stripped.split("/")
    for seg in segments:
        if seg == "..":
            raise ReportError(f"ZIP member path cannot contain traversal segment '..': {path!r}")
        if seg == ".":
            raise ReportError(f"ZIP member path cannot contain relative segment '.': {path!r}")
        if not seg:
            raise ReportError(f"ZIP member path contains empty segment: {path!r}")


def check_zip_resource_safety(archive: zipfile.ZipFile) -> None:
    """Verify archive resource limits and member paths before extraction or reading.

    Raises ReportError if member count, single member size, or total uncompressed
    size exceeds conservative safety limits, or if any member path violates path safety.
    """
    infolist = archive.infolist()
    if len(infolist) > MAX_MEMBER_COUNT:
        raise ReportError(
            f"Bundle exceeds maximum allowed member count ({len(infolist)} > {MAX_MEMBER_COUNT})."
        )

    seen_paths: set[str] = set()
    total_bytes = 0

    for info in infolist:
        name = info.filename
        validate_zip_member_path(name)

        normalized = posixpath.normpath(name.rstrip("/"))
        if normalized in seen_paths:
            raise ReportError(f"Duplicate normalized member path detected in bundle: {name!r}")
        seen_paths.add(normalized)

        if info.file_size > MAX_SINGLE_MEMBER_BYTES:
            raise ReportError(
                f"Bundle member {name!r} exceeds maximum single file size "
                f"({info.file_size} > {MAX_SINGLE_MEMBER_BYTES} bytes)."
            )

        total_bytes += info.file_size
        if total_bytes > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ReportError(
                f"Bundle total uncompressed size exceeds maximum allowed limit "
                f"({total_bytes} > {MAX_TOTAL_UNCOMPRESSED_BYTES} bytes)."
            )
