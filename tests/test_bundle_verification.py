"""Tests for research export bundle integrity verification and tampering detection.

Checks corruptions, extra files, missing files, and error handling.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from pyautostat import (
    BUNDLE_SCHEMA_VERSION,
    AnalysisOptions,
    BundleVerificationResult,
    ReportError,
    ResearchAssistant,
    to_bundle,
    verify_bundle,
)


@pytest.fixture
def sample_bundle_bytes() -> bytes:
    df = pd.DataFrame(
        {
            "group": ["Ctrl"] * 10 + ["Trt"] * 10,
            "outcome": [5.1, 5.3, 5.2, 5.0, 5.4] * 2 + [7.1, 7.5, 7.2, 7.0, 7.3] * 2,
        }
    )
    workflow = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="outcome",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"outcome": "continuous", "group": "nominal"},
        options=AnalysisOptions(confidence_level=0.95),
    )
    return to_bundle(workflow, formats=("html", "json", "csv"))


def test_verify_valid_bundle_from_bytes(sample_bundle_bytes: bytes):
    result = verify_bundle(sample_bundle_bytes)

    assert isinstance(result, BundleVerificationResult)
    assert result.valid is True
    assert result.schema_version == BUNDLE_SCHEMA_VERSION
    assert result.checked_files > 0
    assert result.missing_files == ()
    assert result.unexpected_files == ()
    assert result.checksum_mismatches == ()
    assert result.size_mismatches == ()
    assert result.errors == ()


def test_verify_valid_bundle_from_path(sample_bundle_bytes: bytes, tmp_path: Path):
    bundle_path = tmp_path / "valid_bundle.zip"
    bundle_path.write_bytes(sample_bundle_bytes)

    # Test Path object
    res_path = verify_bundle(bundle_path)
    assert res_path.valid is True

    # Test string path
    res_str = verify_bundle(str(bundle_path))
    assert res_str.valid is True


def test_verify_rejects_urls():
    with pytest.raises(ReportError, match="URLs are not supported"):
        verify_bundle("https://example.com/bundle.zip")

    with pytest.raises(ReportError, match="URLs are not supported"):
        verify_bundle("http://localhost:8000/bundle.zip")


def test_verify_rejects_missing_file_or_invalid_type(tmp_path: Path):
    with pytest.raises(ReportError, match="Bundle file not found"):
        verify_bundle(tmp_path / "nonexistent.zip")

    with pytest.raises(ReportError, match="Unsupported bundle_input type"):
        verify_bundle(12345)  # type: ignore[arg-type]


def test_verify_rejects_corrupted_or_non_zip_bytes():
    with pytest.raises(ReportError, match="missing PK signature"):
        verify_bundle(b"NOT_A_ZIP_FILE_DATA")

    with pytest.raises(ReportError, match="not a valid ZIP archive"):
        verify_bundle(b"PK\x03\x04corrupted_payload")


def test_verify_rejects_missing_manifest(sample_bundle_bytes: bytes):
    # Repack ZIP omitting manifest.json
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(sample_bundle_bytes)) as zf_in:
        with zipfile.ZipFile(buf, mode="w") as zf_out:
            for item in zf_in.infolist():
                if item.filename != "pyautostat_bundle/manifest.json":
                    zf_out.writestr(item, zf_in.read(item.filename))

    with pytest.raises(ReportError, match="manifest 'pyautostat_bundle/manifest.json' is missing"):
        verify_bundle(buf.getvalue())


def test_verify_rejects_malformed_manifest_json(sample_bundle_bytes: bytes):
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(sample_bundle_bytes)) as zf_in:
        with zipfile.ZipFile(buf, mode="w") as zf_out:
            for item in zf_in.infolist():
                if item.filename == "pyautostat_bundle/manifest.json":
                    zf_out.writestr(item, b"{invalid_json: true,")
                else:
                    zf_out.writestr(item, zf_in.read(item.filename))

    with pytest.raises(ReportError, match="malformed JSON"):
        verify_bundle(buf.getvalue())


def test_verify_rejects_unsupported_schema_version(sample_bundle_bytes: bytes):
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(sample_bundle_bytes)) as zf_in:
        with zipfile.ZipFile(buf, mode="w") as zf_out:
            for item in zf_in.infolist():
                if item.filename == "pyautostat_bundle/manifest.json":
                    manifest = json.loads(zf_in.read(item.filename).decode("utf-8"))
                    manifest["bundle_schema_version"] = 99
                    zf_out.writestr(item, json.dumps(manifest).encode("utf-8"))
                else:
                    zf_out.writestr(item, zf_in.read(item.filename))

    with pytest.raises(ReportError, match="Unsupported bundle schema version: 99"):
        verify_bundle(buf.getvalue())


def test_verify_detects_tampered_member_content(sample_bundle_bytes: bytes):
    # Modify 1 byte in report.html
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(sample_bundle_bytes)) as zf_in:
        with zipfile.ZipFile(buf, mode="w") as zf_out:
            for item in zf_in.infolist():
                data = zf_in.read(item.filename)
                if item.filename == "pyautostat_bundle/report/report.html":
                    # Flip one byte while keeping length identical
                    tampered = bytearray(data)
                    tampered[10] = ord("X") if tampered[10] != ord("X") else ord("Y")
                    data = bytes(tampered)
                zf_out.writestr(item, data)

    res = verify_bundle(buf.getvalue())
    assert res.valid is False
    assert "pyautostat_bundle/report/report.html" in res.checksum_mismatches
    assert any("Checksum mismatch" in err for err in res.errors)


def test_verify_detects_size_mismatch(sample_bundle_bytes: bytes):
    # Truncate report.html
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(sample_bundle_bytes)) as zf_in:
        with zipfile.ZipFile(buf, mode="w") as zf_out:
            for item in zf_in.infolist():
                data = zf_in.read(item.filename)
                if item.filename == "pyautostat_bundle/report/report.html":
                    data = data[: len(data) - 10]
                zf_out.writestr(item, data)

    res = verify_bundle(buf.getvalue())
    assert res.valid is False
    assert "pyautostat_bundle/report/report.html" in res.size_mismatches
    assert "pyautostat_bundle/report/report.html" in res.checksum_mismatches
    assert any("Size mismatch" in err for err in res.errors)


def test_verify_detects_deleted_member_file(sample_bundle_bytes: bytes):
    # Remove data/report.json from ZIP without updating manifest
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(sample_bundle_bytes)) as zf_in:
        with zipfile.ZipFile(buf, mode="w") as zf_out:
            for item in zf_in.infolist():
                if item.filename != "pyautostat_bundle/data/report.json":
                    zf_out.writestr(item, zf_in.read(item.filename))

    res = verify_bundle(buf.getvalue())
    assert res.valid is False
    assert "pyautostat_bundle/data/report.json" in res.missing_files
    assert any("Missing" in err for err in res.errors)


def test_verify_detects_undeclared_extra_file(sample_bundle_bytes: bytes):
    # Add an undeclared extra file to the archive
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(sample_bundle_bytes)) as zf_in:
        with zipfile.ZipFile(buf, mode="w") as zf_out:
            for item in zf_in.infolist():
                zf_out.writestr(item, zf_in.read(item.filename))
            zf_out.writestr("pyautostat_bundle/extra_file.txt", b"Unauthorized extra payload")

    res = verify_bundle(buf.getvalue())
    assert res.valid is False
    assert "pyautostat_bundle/extra_file.txt" in res.unexpected_files
    assert any("Found 1 undeclared files" in err for err in res.errors)


def _repack_with_manifest(sample_bundle_bytes: bytes, manifest_modifier) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(sample_bundle_bytes)) as zf_in:
        with zipfile.ZipFile(buf, mode="w") as zf_out:
            for item in zf_in.infolist():
                if item.filename == "pyautostat_bundle/manifest.json":
                    manifest = json.loads(zf_in.read(item.filename).decode("utf-8"))
                    manifest_modifier(manifest)
                    zf_out.writestr(item, json.dumps(manifest).encode("utf-8"))
                else:
                    zf_out.writestr(item, zf_in.read(item.filename))
    return buf.getvalue()


def test_verify_rejects_duplicate_manifest_path(sample_bundle_bytes: bytes):
    def modify(manifest):
        entry = manifest["files"][0]
        manifest["files"].append(entry.copy())

    bundle = _repack_with_manifest(sample_bundle_bytes, modify)
    with pytest.raises(ReportError, match="duplicate file path declaration"):
        verify_bundle(bundle)


def test_verify_rejects_missing_sha256(sample_bundle_bytes: bytes):
    def modify(manifest):
        del manifest["files"][0]["sha256"]

    bundle = _repack_with_manifest(sample_bundle_bytes, modify)
    with pytest.raises(ReportError, match="missing required field 'sha256'"):
        verify_bundle(bundle)


def test_verify_rejects_missing_size_bytes(sample_bundle_bytes: bytes):
    def modify(manifest):
        del manifest["files"][0]["size_bytes"]

    bundle = _repack_with_manifest(sample_bundle_bytes, modify)
    with pytest.raises(ReportError, match="missing required field 'size_bytes'"):
        verify_bundle(bundle)


def test_verify_rejects_malformed_sha256(sample_bundle_bytes: bytes):
    def modify(manifest):
        manifest["files"][0]["sha256"] = "not_a_valid_64_char_hex_digest"

    bundle = _repack_with_manifest(sample_bundle_bytes, modify)
    with pytest.raises(ReportError, match="invalid sha256 digest"):
        verify_bundle(bundle)


def test_verify_rejects_negative_or_bool_size(sample_bundle_bytes: bytes):
    def modify_negative(manifest):
        manifest["files"][0]["size_bytes"] = -5

    bundle1 = _repack_with_manifest(sample_bundle_bytes, modify_negative)
    with pytest.raises(ReportError, match="invalid size_bytes"):
        verify_bundle(bundle1)

    def modify_bool(manifest):
        manifest["files"][0]["size_bytes"] = True

    bundle2 = _repack_with_manifest(sample_bundle_bytes, modify_bool)
    with pytest.raises(ReportError, match="invalid size_bytes"):
        verify_bundle(bundle2)


def test_verify_rejects_non_dict_entry(sample_bundle_bytes: bytes):
    def modify(manifest):
        manifest["files"].append("not_a_dict_entry")

    bundle = _repack_with_manifest(sample_bundle_bytes, modify)
    with pytest.raises(ReportError, match="must be a JSON object/mapping"):
        verify_bundle(bundle)


def test_verify_rejects_unsafe_declared_path(sample_bundle_bytes: bytes):
    def modify(manifest):
        manifest["files"][0]["path"] = "../traversal/path.txt"

    bundle = _repack_with_manifest(sample_bundle_bytes, modify)
    with pytest.raises(ReportError, match="traversal segment"):
        verify_bundle(bundle)


def test_verify_rejects_manifest_self_declaration(sample_bundle_bytes: bytes):
    def modify(manifest):
        manifest["files"].append(
            {
                "path": "pyautostat_bundle/manifest.json",
                "role": "metadata",
                "format": "json",
                "size_bytes": 100,
                "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            }
        )

    bundle = _repack_with_manifest(sample_bundle_bytes, modify)
    with pytest.raises(ReportError, match="may not declare the manifest itself"):
        verify_bundle(bundle)
