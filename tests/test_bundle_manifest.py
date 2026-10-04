"""Tests for research export bundle manifest schema, contents, and SHA-256 accuracy."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pandas as pd
import pytest

from pyautostat import (
    BUNDLE_SCHEMA_VERSION,
    AnalysisOptions,
    ResearchAssistant,
    ResearchWorkflowResult,
    to_bundle,
)
from pyautostat import __version__ as package_version


@pytest.fixture
def workflow_with_audit() -> ResearchWorkflowResult:
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "val": [1.0, 2.0, 3.0, 4.0, 5.0] * 2 + [3.0, 4.0, 5.0, 6.0, 7.0] * 2,
        }
    )
    assistant = ResearchAssistant(df)
    return assistant.run(
        objective="compare_groups",
        outcome="val",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"val": "continuous", "group": "nominal"},
        options=AnalysisOptions(confidence_level=0.95),
    )


def test_manifest_schema_and_version_separation(workflow_with_audit: ResearchWorkflowResult):
    bundle_bytes = to_bundle(workflow_with_audit)

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        manifest_raw = zf.read("pyautostat_bundle/manifest.json").decode("utf-8")
        manifest = json.loads(manifest_raw)

    assert manifest["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION
    assert manifest["bundle_schema_version"] == 1
    assert manifest["pyautostat_version"] == package_version
    assert manifest["target_type"] == "ResearchWorkflowResult"
    assert manifest["report_status"] == "complete"
    assert manifest["method_id"] == "welch_t"
    assert manifest["detail"] == "full"
    assert manifest["style"] == "general"
    assert manifest["include_figures"] is False
    assert isinstance(manifest["files"], list)


def test_manifest_does_not_hash_itself(workflow_with_audit: ResearchWorkflowResult):
    bundle_bytes = to_bundle(workflow_with_audit)

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        manifest = json.loads(zf.read("pyautostat_bundle/manifest.json").decode("utf-8"))

    file_paths = [f["path"] for f in manifest["files"]]
    assert "pyautostat_bundle/manifest.json" not in file_paths
    assert "pyautostat_bundle/README.md" in file_paths


def test_manifest_files_are_sorted_by_path(workflow_with_audit: ResearchWorkflowResult):
    bundle_bytes = to_bundle(workflow_with_audit, formats=("html", "json", "csv", "markdown"))

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        manifest = json.loads(zf.read("pyautostat_bundle/manifest.json").decode("utf-8"))

    file_paths = [f["path"] for f in manifest["files"]]
    assert file_paths == sorted(file_paths)


def test_manifest_file_checksums_and_sizes_match_actual_bytes(
    workflow_with_audit: ResearchWorkflowResult,
):
    bundle_bytes = to_bundle(workflow_with_audit, formats=("html", "json", "csv"))

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        manifest = json.loads(zf.read("pyautostat_bundle/manifest.json").decode("utf-8"))

        for record in manifest["files"]:
            path = record["path"]
            assert path in zf.namelist(), f"Declared path {path} not found in ZIP"

            actual_bytes = zf.read(path)
            assert len(actual_bytes) == record["size_bytes"], f"Size mismatch for {path}"

            expected_sha256 = hashlib.sha256(actual_bytes).hexdigest()
            assert record["sha256"] == expected_sha256, f"SHA-256 digest mismatch for {path}"


def test_manifest_records_expected_file_roles(workflow_with_audit: ResearchWorkflowResult):
    bundle_bytes = to_bundle(
        workflow_with_audit, formats=("html", "json", "csv", "markdown", "latex")
    )

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        manifest = json.loads(zf.read("pyautostat_bundle/manifest.json").decode("utf-8"))

    roles_by_path = {f["path"]: f["role"] for f in manifest["files"]}

    assert roles_by_path["pyautostat_bundle/README.md"] == "documentation"
    assert roles_by_path["pyautostat_bundle/report/report.html"] == "human_readable_report"
    assert roles_by_path["pyautostat_bundle/report/report.md"] == "markdown_report"
    assert roles_by_path["pyautostat_bundle/report/report.tex"] == "latex_report"
    assert roles_by_path["pyautostat_bundle/data/report.json"] == "canonical_report_data"
    assert roles_by_path["pyautostat_bundle/provenance/analysis.json"] == "analysis_record"
    assert roles_by_path["pyautostat_bundle/provenance/audit.json"] == "audit_record"
    assert (
        roles_by_path["pyautostat_bundle/provenance/reproducibility.json"]
        == "reproducibility_record"
    )

    # CSV tables check
    for path, role in roles_by_path.items():
        if path.startswith("pyautostat_bundle/tables/"):
            assert role == "tabular_export"


def test_manifest_does_not_contain_secrets_or_source_paths(
    workflow_with_audit: ResearchWorkflowResult,
):
    bundle_bytes = to_bundle(workflow_with_audit)

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        manifest_raw = zf.read("pyautostat_bundle/manifest.json").decode("utf-8")

    # Ensure no absolute paths or Windows drive letters leaked into manifest
    assert "C:\\" not in manifest_raw
    assert "c:/" not in manifest_raw.lower()
    assert "/Users/" not in manifest_raw
    assert "/home/" not in manifest_raw
