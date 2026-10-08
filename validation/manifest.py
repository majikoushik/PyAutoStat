"""Manifest generation and deterministic summary routines for reference validation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import statsmodels
from validation.models import FieldValidationResult

import pyautostat

NUMERICAL_BASELINE_SHA = "817bb9c5454c266462a457690f3673099546b49f"
FRAMEWORK_BASELINE_SHA = "8fce4e9087236446eb4238387ffc04a9e33930c4"

STATISTICAL_RUNTIME_PATHS = [
    "src/pyautostat/categorical.py",
    "src/pyautostat/execution.py",
    "src/pyautostat/icc.py",
    "src/pyautostat/inference.py",
    "src/pyautostat/inference_extended.py",
    "src/pyautostat/logistic_regression.py",
    "src/pyautostat/method_contracts.py",
    "src/pyautostat/multigroup.py",
    "src/pyautostat/regression.py",
    "src/pyautostat/reliability.py",
    "src/pyautostat/repeated_measures.py",
    "src/pyautostat/two_way_anova.py",
    "src/pyautostat/uncertainty.py",
]


def compute_validation_framework_content_sha256(root: Path | None = None) -> str:
    """Deterministically compute SHA-256 content fingerprint over validation framework source files."""
    if root is None:
        root = Path(__file__).resolve().parent
    val_files: list[tuple[str, Path]] = []
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(root).as_posix()
        val_files.append((rel, p))
    val_files.sort(key=lambda x: x[0])

    h = hashlib.sha256()
    for rel, path in val_files:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        h.update(raw)
        h.update(b"\0")
    return h.hexdigest()


def _get_git_metadata() -> dict[str, Any]:
    """Retrieve Git provenance metadata without exposing personal filesystem paths."""
    rev_sha: str | None = None
    is_dirty: bool | None = None
    invariance_status: str = "unavailable"

    try:
        # Current HEAD revision
        res_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_head.returncode == 0:
            rev_sha = res_head.stdout.strip() or None

        # Worktree dirty status
        res_status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_status.returncode == 0:
            is_dirty = bool(res_status.stdout.strip())

        # Check diff of statistical runtime source files against numerical baseline SHA
        res_diff = subprocess.run(
            ["git", "diff", "--name-only", NUMERICAL_BASELINE_SHA, "--", *STATISTICAL_RUNTIME_PATHS],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_diff.returncode == 0:
            changed_files = res_diff.stdout.strip()
            invariance_status = "changed" if changed_files else "unchanged"
        else:
            invariance_status = "unavailable"
    except Exception:
        rev_sha = "unavailable"
        is_dirty = None
        invariance_status = "unavailable"

    return {
        "validation_framework_revision_sha": rev_sha or "unavailable",
        "git_worktree_dirty": is_dirty,
        "source_code_invariance_status": invariance_status,
        "validation_framework_content_sha256": compute_validation_framework_content_sha256(),
    }


def build_manifest_v2(
    cases: list[dict[str, Any]],
    field_results: list[FieldValidationResult],
) -> dict[str, Any]:
    """Generate Schema Version 2 manifest containing field-level expected values."""
    results_by_case: dict[str, list[FieldValidationResult]] = {}
    for r in field_results:
        results_by_case.setdefault(r.case_id, []).append(r)

    manifest_cases = []
    for c in cases:
        cid = c["case_id"]
        c_results = results_by_case.get(cid, [])
        fields_entries = [r.to_manifest_field_dict() for r in c_results]

        manifest_cases.append(
            {
                "case_id": c["case_id"],
                "method_id": c["method_id"],
                "scientific_target": c["scientific_target"],
                "reference_provenance": c["reference_citation"],
                "orientation_definition": c.get("orientation", {}),
                "sample_accounting": c.get("missing_accounting", {}),
                "fields": fields_entries,
            }
        )

    return {
        "schema_version": 2,
        "cases": manifest_cases,
    }


def write_manifest(
    manifest_data: dict[str, Any],
    output_path: str | Path = "validation/reference_manifest.json",
) -> None:
    """Save manifest to disk with UTF-8 formatting and no NaN values."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)


def build_validation_summary(
    cases: list[dict[str, Any]],
    field_results: list[FieldValidationResult],
    summary_stats: dict[str, Any],
) -> dict[str, Any]:
    """Generate deterministic validation summary artifact."""
    methods_seen = sorted({c["method_id"] for c in cases})

    # Group results by method
    method_summaries: dict[str, dict[str, Any]] = {}
    for mid in methods_seen:
        m_results = [r for r in field_results if r.method_id == mid]
        m_cases = [c for c in cases if c["method_id"] == mid]
        m_a = sum(1 for r in m_results if r.evidence_level == "A" and r.status == "pass")
        m_b = sum(1 for r in m_results if r.evidence_level == "B" and r.status == "pass")
        m_c = sum(1 for r in m_results if r.evidence_level == "C" and r.status == "pass")
        m_d = sum(1 for r in m_results if r.status == "deferred")
        m_disc = sum(1 for r in m_results if r.status == "discrepancy")

        method_summaries[mid] = {
            "cases_count": len(m_cases),
            "fields_compared_count": len(m_results),
            "level_a_passes": m_a,
            "level_b_passes": m_b,
            "level_c_passes": m_c,
            "level_d_deferred": m_d,
            "discrepancies_count": m_disc,
            "status": "passed" if m_disc == 0 else "failed",
        }

    # Extract all deferred quantities
    deferred_records = []
    seen_def = set()
    for r in field_results:
        if r.status == "deferred":
            key = (r.method_id, r.field)
            if key not in seen_def:
                seen_def.add(key)
                deferred_records.append(
                    {
                        "method_id": r.method_id,
                        "field": r.field,
                        "evidence_level": "D",
                        "reason": r.notes or "Explicitly deferred in current validation tranche",
                    }
                )

    git_meta = _get_git_metadata()

    return {
        "schema_version": 1,
        "package": {
            "name": "pyautostat",
            "version": pyautostat.__version__,
            "numerical_source_baseline_sha": NUMERICAL_BASELINE_SHA,
            "validation_framework_baseline_sha": FRAMEWORK_BASELINE_SHA,
            "validation_framework_revision_sha": git_meta["validation_framework_revision_sha"],
            "validation_framework_content_sha256": git_meta["validation_framework_content_sha256"],
            "git_worktree_dirty": git_meta["git_worktree_dirty"],
            "source_code_invariance_status": git_meta["source_code_invariance_status"],
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__,
            "statsmodels_version": statsmodels.__version__,
        },
        "totals": {
            "methods_count": len(methods_seen),
            "cases_count": len(cases),
            "fields_compared_count": len(field_results),
            "level_a_passes": summary_stats.get("level_a_passes", 0),
            "level_b_passes": summary_stats.get("level_b_passes", 0),
            "level_c_passes": summary_stats.get("level_c_passes", 0),
            "deferred_count": summary_stats.get("deferred_count", 0),
            "discrepancies_count": summary_stats.get("discrepancies_count", 0),
            "strict_mode": summary_stats.get("strict_mode", False),
            "overall_status": summary_stats.get("overall_status", "pass"),
        },
        "per_method_summary": method_summaries,
        "deferred_quantities": deferred_records,
    }


def write_summary(
    summary_data: dict[str, Any],
    output_path: str | Path = "validation/reference_validation_summary.json",
) -> None:
    """Save deterministic summary artifact to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
