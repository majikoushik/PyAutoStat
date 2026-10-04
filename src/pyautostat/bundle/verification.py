"""Bundle verification engine checking manifest consistency, artifact hashes, and archive safety."""

from __future__ import annotations

import io
import json
import pathlib
import zipfile
from typing import Any

from ..exceptions import ReportError
from .manifest import compute_sha256
from .models import BUNDLE_SCHEMA_VERSION, BundleVerificationResult
from .safety import check_zip_resource_safety


def verify_bundle(bundle_input: bytes | str | pathlib.Path) -> BundleVerificationResult:
    """Verify integrity and structure of a research export bundle.

    Accepts bundle bytes, a Path object, or a local file path string.
    Does not accept URLs or perform network operations.

    Raises ReportError if:
    - input is an unsupported type or network URL;
    - file cannot be found or is not a valid ZIP;
    - safety constraints (traversal, resource limits) are violated;
    - canonical manifest is missing or contains malformed JSON;
    - manifest specifies an unsupported schema version.

    Returns BundleVerificationResult indicating valid status, checked files,
    missing files, unexpected files, checksum mismatches, and size mismatches.
    """
    if isinstance(bundle_input, str):
        if (
            bundle_input.startswith("http://")
            or bundle_input.startswith("https://")
            or bundle_input.startswith("ftp://")
        ):
            raise ReportError(f"URLs are not supported for bundle verification: {bundle_input!r}")
        path_obj = pathlib.Path(bundle_input)
        if not path_obj.is_file():
            raise ReportError(f"Bundle file not found: {bundle_input!r}")
        try:
            zip_source: io.BytesIO | pathlib.Path = path_obj
            zf = zipfile.ZipFile(zip_source, mode="r")
        except (zipfile.BadZipFile, OSError) as exc:
            raise ReportError(f"Input is not a valid ZIP archive: {exc}") from exc
    elif isinstance(bundle_input, pathlib.Path):
        if not bundle_input.is_file():
            raise ReportError(f"Bundle file not found: {bundle_input!r}")
        try:
            zf = zipfile.ZipFile(bundle_input, mode="r")
        except (zipfile.BadZipFile, OSError) as exc:
            raise ReportError(f"Input is not a valid ZIP archive: {exc}") from exc
    elif isinstance(bundle_input, (bytes, bytearray)):
        if not bundle_input.startswith(b"PK"):
            raise ReportError("Input is not a valid ZIP archive (missing PK signature).")
        try:
            zf = zipfile.ZipFile(io.BytesIO(bundle_input), mode="r")
        except (zipfile.BadZipFile, OSError) as exc:
            raise ReportError(f"Input is not a valid ZIP archive: {exc}") from exc
    else:
        raise ReportError(
            f"Unsupported bundle_input type: {type(bundle_input).__name__}. "
            "Expected bytes, str, or pathlib.Path."
        )

    with zf:
        # 1. Resource and path-safety checks
        check_zip_resource_safety(zf)

        # 2. Check canonical manifest
        manifest_path = "pyautostat_bundle/manifest.json"
        if manifest_path not in zf.namelist():
            raise ReportError(f"Bundle manifest '{manifest_path}' is missing.")

        try:
            manifest_raw = zf.read(manifest_path).decode("utf-8")
            manifest_data: dict[str, Any] = json.loads(manifest_raw)
        except Exception as exc:
            raise ReportError(
                f"Bundle manifest '{manifest_path}' is malformed JSON: {exc}"
            ) from exc

        if not isinstance(manifest_data, dict):
            raise ReportError(f"Bundle manifest '{manifest_path}' must be a JSON object.")

        schema_version = manifest_data.get("bundle_schema_version")
        if schema_version != BUNDLE_SCHEMA_VERSION:
            raise ReportError(
                f"Unsupported bundle schema version: {schema_version!r}. "
                f"Supported schema version is {BUNDLE_SCHEMA_VERSION}."
            )

        declared_files_list = manifest_data.get("files")
        if not isinstance(declared_files_list, list):
            raise ReportError("Manifest 'files' property must be a list.")

        REQUIRED_FIELDS = ("path", "role", "format", "size_bytes", "sha256")
        HEX_DIGITS = set("0123456789abcdefABCDEF")

        declared_files: dict[str, dict[str, Any]] = {}
        for idx, entry in enumerate(declared_files_list):
            if not isinstance(entry, dict):
                raise ReportError(
                    f"Manifest 'files' entry at index {idx} must be a JSON object/mapping, "
                    f"got {type(entry).__name__}."
                )

            for req in REQUIRED_FIELDS:
                if req not in entry:
                    raise ReportError(
                        f"Manifest file entry at index {idx} is missing required field {req!r}."
                    )

            path = entry["path"]
            if not isinstance(path, str) or not path.strip():
                raise ReportError(
                    f"Manifest file entry at index {idx} has invalid path: "
                    "must be a non-empty string."
                )

            from .safety import validate_zip_member_path

            validate_zip_member_path(path)

            if path == manifest_path or path == "manifest.json" or path.endswith("/manifest.json"):
                raise ReportError(
                    f"Manifest file entry at index {idx} may not declare the manifest itself: "
                    f"{path!r}."
                )

            if path in declared_files:
                raise ReportError(f"Manifest contains duplicate file path declaration: {path!r}.")

            role = entry["role"]
            if not isinstance(role, str) or not role.strip():
                raise ReportError(
                    f"Manifest file entry for {path!r} has invalid role: "
                    "must be a non-empty string."
                )

            fmt = entry["format"]
            if not isinstance(fmt, str) or not fmt.strip():
                raise ReportError(
                    f"Manifest file entry for {path!r} has invalid format: "
                    "must be a non-empty string."
                )

            size_bytes = entry["size_bytes"]
            if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
                raise ReportError(
                    f"Manifest file entry for {path!r} has invalid size_bytes: "
                    f"must be an integer >= 0, got {size_bytes!r}."
                )

            sha256 = entry["sha256"]
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or not all(c in HEX_DIGITS for c in sha256)
            ):
                raise ReportError(
                    f"Manifest file entry for {path!r} has invalid sha256 digest: "
                    f"must be a 64-character hexadecimal string, got {sha256!r}."
                )

            declared_files[path] = entry

        # 3. Check actual member files (excluding directory entries and manifest itself)
        actual_members = set(
            name for name in zf.namelist() if not name.endswith("/") and name != manifest_path
        )

        missing_files = tuple(sorted(p for p in declared_files if p not in actual_members))
        unexpected_files = tuple(sorted(p for p in actual_members if p not in declared_files))

        checksum_mismatches: list[str] = []
        size_mismatches: list[str] = []
        errors: list[str] = []
        checked_files = 0

        for path, meta in declared_files.items():
            if path in actual_members:
                content = zf.read(path)
                checked_files += 1

                exp_size = meta.get("size_bytes")
                if exp_size is not None and len(content) != exp_size:
                    size_mismatches.append(path)

                exp_hash = meta.get("sha256")
                actual_hash = compute_sha256(content)
                if exp_hash is not None and actual_hash != exp_hash:
                    checksum_mismatches.append(path)

        if missing_files:
            joined_missing = ", ".join(missing_files)
            errors.append(f"Missing {len(missing_files)} declared files: {joined_missing}")
        if unexpected_files:
            joined_unexpected = ", ".join(unexpected_files)
            errors.append(f"Found {len(unexpected_files)} undeclared files: {joined_unexpected}")
        if checksum_mismatches:
            joined_checksums = ", ".join(checksum_mismatches)
            errors.append(
                f"Checksum mismatch in {len(checksum_mismatches)} files: {joined_checksums}"
            )
        if size_mismatches:
            joined_sizes = ", ".join(size_mismatches)
            errors.append(f"Size mismatch in {len(size_mismatches)} files: {joined_sizes}")

        valid = (
            len(missing_files) == 0
            and len(unexpected_files) == 0
            and len(checksum_mismatches) == 0
            and len(size_mismatches) == 0
        )

        return BundleVerificationResult(
            valid=valid,
            schema_version=schema_version,
            checked_files=checked_files,
            missing_files=missing_files,
            unexpected_files=unexpected_files,
            checksum_mismatches=tuple(sorted(checksum_mismatches)),
            size_mismatches=tuple(sorted(size_mismatches)),
            errors=tuple(errors),
        )
