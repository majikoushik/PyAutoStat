"""Security, resource limit, privacy sentinel, and zero-recalculation tests for bundles."""

from __future__ import annotations

import io
import zipfile
from unittest.mock import patch

import pandas as pd
import pytest
import scipy.stats

from pyautostat import (
    AnalysisOptions,
    ReportError,
    ResearchAssistant,
    to_bundle,
    verify_bundle,
)
from pyautostat.bundle.safety import (
    MAX_MEMBER_COUNT,
    MAX_SINGLE_MEMBER_BYTES,
    check_zip_resource_safety,
    sanitize_table_id,
    validate_zip_member_path,
)


def test_sanitize_table_id():
    assert sanitize_table_id("clean_table_1") == "clean_table_1"
    assert sanitize_table_id("../../etc/passwd") == "etc_passwd"
    assert sanitize_table_id(r"C:\Windows\System32") == "C_Windows_System32"
    assert sanitize_table_id("group:mean (diff)") == "group_mean_diff"
    assert sanitize_table_id("") == "table"
    assert sanitize_table_id(None) == "table"
    assert sanitize_table_id("___") == "table"


def test_validate_zip_member_path_rejects_unsafe_paths():
    # Empty
    with pytest.raises(ReportError, match="cannot be empty"):
        validate_zip_member_path("")

    # Backslashes
    with pytest.raises(ReportError, match="must use forward slashes"):
        validate_zip_member_path(r"pyautostat_bundle\report.html")

    # Windows drive
    with pytest.raises(ReportError, match="cannot contain drive specifications"):
        validate_zip_member_path("C:/bundle/file.txt")

    # Absolute path
    with pytest.raises(ReportError, match="cannot be absolute"):
        validate_zip_member_path("/etc/shadow")

    # Directory traversal
    with pytest.raises(ReportError, match="traversal segment"):
        validate_zip_member_path("pyautostat_bundle/../evil.sh")

    with pytest.raises(ReportError, match="traversal segment"):
        validate_zip_member_path("../manifest.json")

    # Valid relative paths
    validate_zip_member_path("pyautostat_bundle/report/report.html")
    validate_zip_member_path("pyautostat_bundle/manifest.json")


def test_zip_resource_safety_exceeding_member_count():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        for i in range(MAX_MEMBER_COUNT + 1):
            zf.writestr(f"pyautostat_bundle/file_{i}.txt", b"x")

    with pytest.raises(ReportError, match="maximum allowed member count"):
        verify_bundle(buf.getvalue())


def test_zip_resource_safety_exceeding_single_file_size():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr("pyautostat_bundle/manifest.json", b"{}")

    # Patch archive infolist to simulate a member exceeding MAX_SINGLE_MEMBER_BYTES
    with zipfile.ZipFile(buf, mode="r") as zf:
        fake_info = zipfile.ZipInfo("pyautostat_bundle/huge.bin")
        fake_info.file_size = MAX_SINGLE_MEMBER_BYTES + 1
        with patch.object(zf, "infolist", return_value=[fake_info]):
            with pytest.raises(ReportError, match="exceeds maximum single file size"):
                check_zip_resource_safety(zf)


def test_zip_resource_safety_exceeding_total_uncompressed():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr("pyautostat_bundle/manifest.json", b"{}")

    with zipfile.ZipFile(buf, mode="r") as zf:
        # 3 parts of 40MB each = 120MB > 100MB, while each part is 40MB < 50MB
        fake_infos = []
        for i in range(3):
            info = zipfile.ZipInfo(f"pyautostat_bundle/part{i}.bin")
            info.file_size = 40 * 1024 * 1024
            fake_infos.append(info)
        with patch.object(zf, "infolist", return_value=fake_infos):
            with pytest.raises(ReportError, match="total uncompressed size exceeds maximum"):
                check_zip_resource_safety(zf)


def test_zip_resource_safety_duplicate_normalized_member():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr("pyautostat_bundle/dir/", b"")
        zf.writestr("pyautostat_bundle/dir", b"content")

    with pytest.raises(ReportError, match="Duplicate normalized member path"):
        with zipfile.ZipFile(buf, mode="r") as zf:
            check_zip_resource_safety(zf)


def test_privacy_sentinel_zero_raw_data_leakage():
    # Unique sentinel tokens representing private participant rows
    sentinels = [
        "SENTINEL_PATIENT_ALPHA_99812",
        "SENTINEL_PATIENT_BETA_77341",
        "SENTINEL_SSN_000_12_3456",
        "SECRET_CLINICAL_TRIAL_ID_9999",
    ]

    df = pd.DataFrame(
        {
            "patient_id": sentinels * 5,
            "group": ["Placebo"] * 10 + ["Drug"] * 10,
            "score": [12.0, 13.5, 11.8, 14.0] * 5,
        }
    )

    assistant = ResearchAssistant(df)
    workflow = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
        options=AnalysisOptions(confidence_level=0.95),
    )

    # Generate full bundle with all available text/export formats
    bundle_bytes = to_bundle(
        workflow,
        formats=("html", "json", "csv", "markdown", "latex", "docx"),
    )

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        for member_name in zf.namelist():
            # Check member name itself
            for sentinel in sentinels:
                assert sentinel not in member_name

            raw_bytes = zf.read(member_name)
            # Decode if possible
            try:
                text_content = raw_bytes.decode("utf-8")
                for sentinel in sentinels:
                    assert sentinel not in text_content, (
                        f"Privacy leak: sentinel '{sentinel}' found in member '{member_name}'"
                    )
            except UnicodeDecodeError:
                # Binary files like DOCX - inspect internal XML parts
                if member_name.endswith(".docx"):
                    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as docx_zf:
                        for docx_part in docx_zf.namelist():
                            if docx_part.endswith(".xml"):
                                xml_text = docx_zf.read(docx_part).decode("utf-8")
                                for sentinel in sentinels:
                                    msg = (
                                        f"Privacy leak: sentinel '{sentinel}' found in "
                                        f"docx XML '{docx_part}'"
                                    )
                                    assert sentinel not in xml_text, msg


def test_zero_statistical_recalculation_during_bundle_assembly():
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "score": [1.0, 2.0, 3.0, 4.0, 5.0] * 2 + [5.0, 6.0, 7.0, 8.0, 9.0] * 2,
        }
    )

    workflow = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
        options=AnalysisOptions(confidence_level=0.95),
    )

    def forbid_statistical_computation(*args, **kwargs):
        raise RuntimeError("Forbidden: statistical computation was invoked during bundle assembly!")

    # Monkeypatch statistical engines and scipy test runners
    with patch.object(ResearchAssistant, "analyze", side_effect=forbid_statistical_computation):
        with patch.object(scipy.stats, "ttest_ind", side_effect=forbid_statistical_computation):
            with patch.object(
                scipy.stats, "mannwhitneyu", side_effect=forbid_statistical_computation
            ):
                with patch.object(
                    scipy.stats, "pearsonr", side_effect=forbid_statistical_computation
                ):
                    # Assemble all formats
                    bundle_bytes = to_bundle(
                        workflow,
                        formats=("html", "docx", "json", "csv", "markdown", "latex"),
                    )

    # Verification must succeed and archive must be valid
    result = verify_bundle(bundle_bytes)
    assert result.valid is True
    assert result.checked_files > 0
