"""Example 17: Reproducible Research Export Bundles.

Demonstrates packaging canonical research reports, tables, machine-readable records,
and execution provenance into verifiable ZIP bundles.
Shows:
1. Generating a lightweight default bundle (HTML + JSON + CSV tables + Provenance).
2. Generating a full publication bundle with all supported formats.
3. Cryptographic SHA-256 integrity verification with `verify_bundle`.
4. Inspecting member artifacts, sizes, and manifest metadata.

Usage:
    python examples/17_research_export_bundle.py [--output-dir /path/to/output]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

# Allow importing local example utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _customer_data import DATA_DICTIONARY, load_customer_data, section

from pyautostat import (
    AnalysisOptions,
    ResearchAssistant,
    save_bundle,
    show,
    verify_bundle,
)
from pyautostat.presentation.docx import check_docx_available
from pyautostat.presentation.pdf import check_playwright_available

RANDOM_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and verify PyAutoStat research export bundles."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save bundle files. If omitted, a temporary directory is used.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory()
        out_dir = Path(temp_dir.name)

    section("PyAutoStat Reproducible Research Export Bundle Showcase")
    print(f"Target output directory: {out_dir.resolve()}")

    # Load customer dataset (customer IDs excluded to preserve privacy)
    df = load_customer_data(include_customer_id=False)
    assistant = ResearchAssistant(df)

    workflow = assistant.run(
        objective="compare_groups",
        outcome="total_avg_monthly_spend",
        predictor="news_subscriber",
        estimand="mean",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )

    section("SOURCE ANALYSIS")
    show(workflow, detail="standard")

    section("GENERATED ARTIFACTS")
    # 1. Lightweight Bundle (Default: HTML + JSON + CSV)
    lightweight_path = out_dir / "lightweight_research_bundle.zip"
    save_bundle(
        workflow,
        lightweight_path,
        formats=("html", "json", "csv"),
        detail="full",
        overwrite=True,
    )
    print(
        f"  ✓ Bundle (Lightweight): {lightweight_path.name} "
        f"({lightweight_path.stat().st_size:,} bytes)"
    )

    # Verify lightweight bundle
    res_light = verify_bundle(lightweight_path)
    status_str = "VALID" if res_light.valid else "INVALID"
    print(f"    Integrity verification: {status_str} ({res_light.checked_files} files checked)")
    assert res_light.valid, f"Verification failed: {res_light.errors}"

    print("    Archive contents:")
    with zipfile.ZipFile(lightweight_path) as zf:
        for info in zf.infolist():
            print(f"      - {info.filename} ({info.file_size:,} bytes)")

    # 2. Publication Bundle with Optional Formats
    formats = ["html", "json", "csv", "markdown", "latex"]

    try:
        check_docx_available()
        formats.append("docx")
    except Exception:
        pass

    try:
        check_playwright_available()
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            browser.close()
        formats.append("pdf")
    except Exception:
        pass

    publication_path = out_dir / "publication_research_bundle.zip"
    save_bundle(
        workflow,
        publication_path,
        formats=formats,
        detail="full",
        overwrite=True,
    )
    print(
        f"\n  ✓ Bundle (Publication): {publication_path.name} "
        f"({publication_path.stat().st_size:,} bytes)"
    )

    # Verify publication bundle
    res_pub = verify_bundle(publication_path)
    pub_status = "VALID" if res_pub.valid else "INVALID"
    print(f"    Integrity verification: {pub_status} ({res_pub.checked_files} files checked)")
    assert res_pub.valid, f"Verification failed: {res_pub.errors}"

    print("    Archive contents:")
    with zipfile.ZipFile(publication_path) as zf:
        for info in zf.infolist():
            print(f"      - {info.filename} ({info.file_size:,} bytes)")

    print("\nBundle generation and cryptographic verification completed successfully.")


if __name__ == "__main__":
    main()
