# Research Export Bundles

PyAutoStat research export bundles provide a reproducible, integrity-verifiable packaging mechanism for statistical analysis artifacts. They bundle human-readable reports, machine-readable structured records, presentation tables, and execution provenance into an offline ZIP archive without introducing a separate statistical rendering engine or recalculating statistics.

---

## 1. Architecture & Design Principles

```text
                  Authoritative Analysis / ResearchReport
                                    │
                                    ▼
                        Canonical Exporters
           (HTML, Interactive HTML, PDF, DOCX, JSON, CSV, Markdown, LaTeX)
                                    │
                                    ▼
                             Bundle Assembler
                                    │
                                    ▼
                ZIP Archive + Deterministic Manifest + SHA-256 Checksums
```

1. **Zero Statistical Recalculation**: Bundles package already-authoritative outputs. They never rerun, refit, or reinterpret statistical computations.
2. **Reuse Canonical Exporters**: All formats are generated through PyAutoStat's canonical renderers (`to_html`, `to_pdf`, `to_docx`, `to_json`, `to_csv_tables`, etc.). Each requested format is rendered exactly once during bundle assembly.
3. **No Raw Data (Hard Safeguard)**: The bundle contains derived presentation artifacts, aggregate summary tables, and recorded execution provenance. Raw dataset observations, participant identifiers, and raw data rows are strictly excluded. (If dataset profiling was separately executed with `include_row_positions=True`, integer row index offsets remain confined to that profile diagnostic dictionary and are never converted into raw row records in bundles).
4. **Integrity Verification vs. Digital Signatures**: Cryptographic SHA-256 checksums verify archive integrity against transmission corruption, incomplete writes, or unauthorized file alteration. Checksums are **not** digital certificates, legal electronic signatures, or identity authentication.

---

## 2. Public Bundle APIs

PyAutoStat exposes three top-level functions and corresponding convenience methods on `ResearchReport`:

```python
from pyautostat import to_bundle, save_bundle, verify_bundle, BundleVerificationResult
```

### In-Memory Archive (`to_bundle`)

```python
bundle_bytes = to_bundle(
    target,
    formats=("html", "json", "csv"),
    detail="full",
    style="general",
    title="Clinical Trial Analysis",
    include_figures=False,
)
```

Returns standard in-memory ZIP bytes (`bytes` starting with `b"PK"`).

### File Persistence (`save_bundle`)

```python
saved_path = save_bundle(
    target,
    "exports/analysis_bundle.zip",
    formats=("html", "docx", "json", "csv"),
    detail="full",
    overwrite=False,
)
```

- Enforces a `.zip` destination filename extension.
- Creates parent directories automatically.
- Enforces overwrite protection (`overwrite=False` by default; raises `ReportError` if the file exists).
- Records `"bundle"` export in the session audit ledger exactly once.

### Convenience Methods on `ResearchReport`

```python
# In-memory bytes
bundle_bytes = report.to_bundle(formats=("html", "json"))

# Save to disk
saved_path = report.save_bundle("my_report_bundle.zip", overwrite=True)
```

---

## 3. Supported Formats & Lightweight Defaults

The default format tuple is deliberately lightweight and free of heavy external dependencies:

```python
formats = ("html", "json", "csv")
```

| Format Identifier | File Location in Bundle | Role | Dependencies |
| :--- | :--- | :--- | :--- |
| `html` *(default)* | `report/report.html` | `human_readable_report` | Core (pandas, SciPy) |
| `json` *(default)* | `data/report.json` | `canonical_report_data` | Core |
| `csv` *(default)* | `tables/<table_id>.csv` | `tabular_export` | Core |
| `markdown` | `report/report.md` | `markdown_report` | Core |
| `latex` | `report/report.tex` | `latex_report` | Core |
| `interactive_html`| `report/report_interactive.html` | `interactive_report` | Core / `plotly` (only if figures requested) |
| `docx` | `report/report.docx` | `editable_report` | `python-docx` (`pip install "pyautostat[docx]"`) |
| `pdf` | `report/report.pdf` | `human_readable_report` | Playwright & Chromium (`pip install "pyautostat[pdf]"`) |

---

## 4. Target Compatibility Matrix

PyAutoStat supports multiple analysis target objects:

| Target Type | Supported Formats | Included Provenance |
| :--- | :--- | :--- |
| `ResearchWorkflowResult` (`completed` or `partial`) | All 8 formats | `analysis.json`, `audit.json`, `reproducibility.json` |
| `ResearchReport` | All 8 formats | `analysis.json` (+ session/audit if attached) |
| `AnalysisResult` | `html`, `pdf`, `docx`, `json` | `analysis.json` |
| `PresentationView` | `html`, `pdf`, `docx` | None (presentation-only) |

*Notes on Target Compatibility:*
- **Workflow Status Requirement**: Research bundles require a computed `completed` or `partial` workflow with statistical results. Attempting to create a bundle from a workflow in a blocked, pending, or failed state (`needs_input`, `data_limited`, `unsupported`, `failed`) raises an explanatory `ReportError`. To inspect or share blocked workflows without full multi-format bundle packaging, use `show(workflow)`, `save_html(workflow, ...)`, or `save_docx(workflow, ...)`.
- **Presentation-Only Targets**: Attempting to request report-semantic formats (such as `csv`, `markdown`, or `latex`) for an `AnalysisResult` or `PresentationView` raises an explanatory `ReportError`.

---

## 5. Archive Layout

All member paths use forward slashes (`/`) and adhere to a deterministic hierarchy:

```text
pyautostat_bundle/
    README.md
    manifest.json

    report/
        report.html
        report_interactive.html
        report.pdf
        report.docx
        report.md
        report.tex

    data/
        report.json

    tables/
        <sanitized_table_id>.csv

    provenance/
        analysis.json
        reproducibility.json
        audit.json
        session.json
```

---

## 6. Manifest Schema (`BUNDLE_SCHEMA_VERSION = 1`)

The bundle manifest is located at `pyautostat_bundle/manifest.json`. It provides structured metadata, target details, and SHA-256 digests for all archive members:

```json
{
  "bundle_schema_version": 1,
  "pyautostat_version": "0.5.0",
  "target_type": "ResearchWorkflowResult",
  "report_status": "complete",
  "method_id": "welch_t",
  "detail": "full",
  "style": "general",
  "include_figures": false,
  "files": [
    {
      "path": "pyautostat_bundle/README.md",
      "role": "documentation",
      "format": "markdown",
      "size_bytes": 1420,
      "sha256": "4b92b678c1a..."
    },
    {
      "path": "pyautostat_bundle/report/report.html",
      "role": "human_readable_report",
      "format": "html",
      "size_bytes": 12840,
      "sha256": "e3b0c44298f..."
    }
  ]
}
```

### Manifest Invariants
- Manifest does not record itself in the `files` array.
- The `files` list is lexicographically sorted by `path`.
- No sensitive environment secrets, absolute filesystem paths, or raw dataset rows are recorded.

---

## 7. Verification Engine (`verify_bundle`)

The verification engine inspects an archive without unzipping files to disk:

```python
from pyautostat import verify_bundle

result = verify_bundle("exports/analysis_bundle.zip")
if result.valid:
    print(f"Verified {result.checked_files} files against manifest.")
else:
    print("Verification failed!")
    print("Errors:", result.errors)
    print("Checksum mismatches:", result.checksum_mismatches)
    print("Size mismatches:", result.size_mismatches)
    print("Missing files:", result.missing_files)
    print("Unexpected files:", result.unexpected_files)
```

### Verification Result Model

```python
@dataclass(frozen=True)
class BundleVerificationResult:
    valid: bool
    schema_version: int | None
    checked_files: int
    missing_files: tuple[str, ...]
    unexpected_files: tuple[str, ...]
    checksum_mismatches: tuple[str, ...]
    size_mismatches: tuple[str, ...]
    errors: tuple[str, ...]
```

### Tamper Detection Capabilities
1. **Byte modification**: If any declared file is modified, its SHA-256 digest will not match the manifest digest (`valid=False`, file listed in `checksum_mismatches`).
2. **Size modification**: Truncated or padded files are flagged in `size_mismatches`.
3. **File deletion**: Declared files missing from the archive are flagged in `missing_files`.
4. **Undeclared members**: Any unexpected extra files added to the archive invalidate the bundle and appear in `unexpected_files`.
5. **Structural corruption**: Non-ZIP data, missing manifest, malformed manifest JSON, or unsupported schema versions raise `ReportError`.

---

## 8. Safety & Resource Limits

To defend against decompression bombs, path traversal attacks, and malformed inputs, the bundle engine enforces conservative bounds before reading or unpacking members:

- **ZIP-Slip Traversal Prevention**: Members containing `..`, absolute paths (`/`), Windows drive letters (`C:`), backslashes (`\`), or empty segments are rejected.
- **Table ID Sanitization**: CSV table identifiers are sanitized to alphanumeric, underscore, and dash characters only.
- **Maximum Member Count**: 500 members.
- **Maximum Single Member Size**: 50 MB uncompressed.
- **Maximum Total Uncompressed Size**: 100 MB.
- **Duplicate Normalized Path Rejection**: Detects conflicting normalized paths (e.g. `dir` and `dir/`).

---

## 9. Privacy & Data Governance

PyAutoStat research bundles package **derived** statistical analyses, aggregate summary tables, and recorded provenance.

- Raw participant records and DataFrame rows are **never** included.
- No `include_data=True` flag or raw CSV export option exists.
- Research data sharing should be conducted through designated data repositories with proper de-identification, access controls, and participant consent.

---

## 10. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `ReportError: Destination file ... already exists and overwrite=False.` | File exists at target path. | Set `overwrite=True` or choose a distinct filename. |
| `ReportError: Package python-docx is required...` | `docx` format requested but dependency missing. | Install with `pip install "pyautostat[docx]"`. |
| `ReportError: PDF export requires the optional PDF dependency...` | `pdf` format requested but Playwright is missing. | Install with `pip install "pyautostat[pdf]"` and run `python -m playwright install chromium`. |
| `ReportError: Format 'csv' requires a ResearchReport...` | Tabular export requested for an `AnalysisResult` target. | Run through `ResearchAssistant.run(...)` or package the workflow report. |
| `verify_bundle: Checksum mismatch in ...` | Member bytes were altered after bundle generation. | Re-generate bundle from source or verify file transfer integrity. |
| `verify_bundle: Found 1 undeclared files ...` | An external file was injected into the ZIP archive. | Only bundle members recorded in `manifest.json` are permitted. |
