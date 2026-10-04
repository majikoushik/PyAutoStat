# Editable Microsoft Word (.docx) Research Reports

PyAutoStat provides an editable Microsoft Word (`.docx`) export layer that transforms structured, authoritative statistical results and research workflows into native OpenXML documents.

The DOCX export layer is strictly a **renderer of existing presentation semantics**. It renders directly from canonical presentation objects (`PresentationView` and `ResearchReport`) without recalculating statistics, parsing HTML back into data, altering estimands, or inventing evidence.

---

## 1. Architecture

Rather than round-tripping through HTML or embedding rasterized pages, PyAutoStat maps canonical presentation semantics directly into OpenXML elements using `python-docx`:

```text
    Statistical Engine
           ↓
    Authoritative Result (AnalysisResult / ResearchWorkflowResult / ResearchReport)
           ↓
    PresentationView / Canonical Report Payload
           ↓
    DOCX Renderer (OpenXML Structure, Named Styles, Tables, Page Setup)
           ↓
    Editable .docx Research Report (Bytes or File)
```

### Core Principles

1. **Native OpenXML Documents**: Output is real Microsoft Word `.docx` containing editable paragraphs, headings, bullet lists, and tables—never images, rasterized pages, or HTML approximations.
2. **Single Source of Presentation Truth**: The Word document renders the exact same tables, KPI grids, diagnostic statuses, and executive summaries produced by terminal, HTML, and PDF presentation layers.
3. **Zero Statistical Recalculation**: Generating a DOCX report never re-runs hypothesis tests, model fittings, bootstrap algorithms, or sensitivity analyses.
4. **No Raw Data Leakage**: In accordance with PyAutoStat privacy safeguards, raw row-level observations are never serialized or embedded in generated `.docx` files.
5. **No External Service or Office Dependency**: Documents are built completely in Python using standard OpenXML schemas. No running instance of Microsoft Word or LibreOffice is required.

---

## 2. Installation

DOCX export is an optional capability powered by `python-docx`.

### Installing DOCX Support

```bash
# Install PyAutoStat with the [docx] extra
python -m pip install "pyautostat[docx]"
```

If DOCX export functions are called without the optional dependency installed, PyAutoStat raises an actionable `ReportError` explaining how to install `pyautostat[docx]`. Base `pyautostat` imports and workflows remain fully functional without this extra.

---

## 3. Public API

```python
from pyautostat import save_docx, to_docx

# Render in-memory DOCX bytes
docx_bytes = to_docx(workflow)

# Save directly to file with overwrite protection
saved_path = save_docx(
    workflow,
    "reports/spending_analysis.docx",
    detail="standard",
    style="general",
    page_size="A4",
    landscape=False,
    page_numbers=True,
    overwrite=True,
)
```

Direct `AnalysisResult`, `PresentationView`, and `ResearchReport` targets are also supported:

```python
# Direct analysis result
docx_bytes = to_docx(workflow.analysis, detail="full")

# Canonical ResearchReport snapshot
report = workflow.report
docx_bytes = report.to_docx(page_size="Letter", landscape=True)
report.save_docx("reports/research_report.docx", overwrite=True)
```

---

## 4. Options and Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `detail` | `str` | `"standard"` | Detail mode: `"compact"`, `"standard"`, or `"full"`. |
| `title` | `str \| None` | `None` | Optional custom document title override. |
| `style` | `str` | `"general"` | Semantic typographic style mode: `"general"`, `"apa"`, or `"ieee"`. |
| `page_size` | `str` | `"A4"` | Standard page dimension: `"A4"` (210 × 297 mm) or `"Letter"` (8.5 × 11 in). |
| `landscape` | `bool` | `False` | When `True`, swaps section width and height for wide landscape tables. |
| `page_numbers` | `bool` | `True` | When `True`, inserts dynamic `PAGE` / `NUMPAGES` field codes in the footer. |
| `overwrite` | `bool` | `False` | Overwrite safety for `save_docx()` (raises `ReportError` if target exists and `overwrite=False`). |

---

## 5. Document Structure and Styles

### Named Typography Styles

PyAutoStat registers clean, semantic named styles in every generated document:
- `PyAutoStat Title`: Document title with restrained spacing.
- `PyAutoStat Subtitle`: Subtitle, status indicator, or concise narrative summary.
- `PyAutoStat Body`: Primary narrative and descriptive paragraphs.
- `PyAutoStat Metric Label`: Bold lead-in label for key metrics.
- `PyAutoStat Metric Value`: Metric value and uncertainty bounds.
- `PyAutoStat Table Header`: Styled table column headings.
- `PyAutoStat Diagnostic`: Diagnostic checks and verification summaries.
- `PyAutoStat Warning`: Prominent warning messages.

### Typographic Presets

- **General** (default): Modern, clean presentation using standard document fonts (Calibri) with clear section separation.
- **APA**: Academic research style using Times New Roman 12 pt, 1.15 line spacing, and restrained tabular headers.
- **IEEE**: Technical research style using Times New Roman with compact table formatting and numbered sections.

*(Note: Style presets provide restrained semantic formatting; they do not certify journal compliance.)*

---

## 6. Table Layout and OpenXML Enhancements

### Editable Tables
All statistical metrics, contingency counts, ANOVA tables, regression coefficients, and diagnostic summaries are converted into true Word tables (`w:tbl`) with editable cells.

### Repeating Table Headers (`tblHeader`)
The first row of every statistical display table is tagged with OpenXML `w:tblHeader`. When a large table spans multiple pages, Word automatically repeats the header row at the top of each page.

### Splitting and Pagination (`cantSplit`)
Data rows include `w:cantSplit` to prevent individual table rows from breaking awkwardly across page boundaries, while allowing the table itself to paginate naturally.

### Number Alignment
Statistical estimates, test statistics, degrees of freedom, p-values, and confidence intervals are right-aligned to enhance numerical scannability.

---

## 7. Dynamic Page Fields

When `page_numbers=True`, footers include dynamic Word field codes:

`Page [PAGE] of [NUMPAGES]`

These fields automatically recalculate inside Microsoft Word, LibreOffice, or Google Docs when the document is edited or repaginated. When `page_numbers=False`, footer page fields are omitted.

---

## 8. ResearchReport DOCX

A `ResearchReport` exported to DOCX includes full canonical sections:
1. **Header & Status**: Report title and execution status.
2. **Executive Summary**: Key findings and highlighted contrasts.
3. **Research Question & Design**: Estimand, variables, study design, and sample sizes.
4. **Dataset & Sample**: Included observations, missingness, and variable types.
5. **Methods**: Selected statistical method, rationale, and mathematical formula.
6. **Results**: Key metric table, primary estimates, dynamic confidence intervals, and full tabular output.
7. **Diagnostics**: Assumption evaluations and model checks.
8. **Interpretation**: Qualified narrative interpretation.
9. **Practical Significance** *(if assessed)*: Threshold comparisons and equivalence decisions.
10. **Sensitivity Analysis** *(if conducted)*: Scenario stability and robustness comparisons.
11. **Limitations**: Methodological constraints and boundaries.
12. **Warnings**: Diagnostic failures or advisory notices.
13. **Analysis Record** *(in full detail mode)*: Deterministic reproducibility audit metadata.

Primary analysis values in report DOCX exports strictly match standalone DOCX exports.

---

## 9. Current Limitations & Client Expectations

- **Figures**: Figures are out of scope for DOCX export in this release. Statistical tables and structured text convey the complete analysis.
- **Client Rendering & Pagination**: Microsoft Word, LibreOffice Writer, and Google Docs use different layout and font-substitution engines. While document structure, styles, tables, and field codes are strictly compliant OpenXML, exact line breaks and pagination may vary slightly between word processors.
- **Templates**: Custom third-party `.docx` templates are not supported in this release. All documents use PyAutoStat's canonical semantic styling.
