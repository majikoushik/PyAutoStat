# Publication-Ready PDF Export and Print Fidelity

PyAutoStat provides a publication-ready PDF export layer that transforms structured, authoritative statistical results and research workflows into high-fidelity PDF documents.

The PDF export layer is strictly an **export format** of the existing canonical HTML and research report architecture. It never recalculates statistics, selects methods, alters estimands, reverses contrasts, changes alpha, or invents evidence.

---

## 1. Architecture

Rather than implementing a separate rendering engine (such as ReportLab) that would duplicate tabular layouts, diagnostic cards, and styling tokens, PyAutoStat uses headless Chromium browser printing:

```text
    Statistical Engine
           ↓
    Authoritative Result (AnalysisResult / ResearchWorkflowResult / ResearchReport)
           ↓
    PresentationView / Canonical Report Payload
           ↓
    Existing Static or Interactive HTML Renderer
           ↓
    Headless Chromium Print Renderer (Playwright)
           ↓
    Publication-Ready PDF Bytes or File
```

### Core Principles

1. **Single Source of Presentation Truth**: The PDF document displays the exact same tables, KPI cards, diagnostic statuses, and executive summaries as the HTML report.
2. **Zero Statistical Recalculation**: Generating a PDF never re-runs hypothesis tests, model fittings, bootstrap algorithms, or sensitivity analyses.
3. **No Raw Data Leakage**: In accordance with PyAutoStat privacy guarantees, raw row-level observations are never serialized or embedded in generated PDFs.
4. **100% Offline & Network-Blocked**: The browser context strictly aborts external network requests (`http://`, `https://`, fonts, trackers). No internet connection is needed or used during PDF generation.
5. **Native Selectable Text**: PDFs contain native vector text and semantic table structures—never rasterized full-page screenshots.

---

## 2. Installation

PDF export is an optional capability powered by Playwright and headless Chromium.

### Installing PDF Support

```bash
# Install PyAutoStat with the [pdf] extra
python -m pip install "pyautostat[pdf]"

# Install the Chromium browser binary
python -m playwright install chromium
```

### Figure-Enabled PDF Support

To include interactive figures in PDF exports, install both `[report]` (Plotly) and `[pdf]` (Playwright):

```bash
python -m pip install "pyautostat[report,pdf]"
python -m playwright install chromium
```

---

## 3. Public API

```python
from pyautostat import to_pdf, save_pdf

# Render in-memory PDF bytes
pdf_bytes = to_pdf(workflow)

# Save directly to file with overwrite protection
saved_path = save_pdf(
    workflow,
    "reports/spending_analysis.pdf",
    detail="standard",
    page_size="A4",
    landscape=False,
    page_numbers=True,
    overwrite=True,
)
```

Direct `AnalysisResult` and `ResearchReport` snapshots are also supported:

```python
# Direct analysis result
pdf_bytes = to_pdf(workflow.analysis, detail="full")

# Canonical ResearchReport snapshot
report = workflow.report
pdf_bytes = report.to_pdf(page_size="Letter", landscape=True)
report.save_pdf("reports/research_report.pdf", overwrite=True)
```

---

## 4. Options and Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `detail` | `str` | `"standard"` | Detail mode: `"compact"`, `"standard"`, or `"full"`. |
| `title` | `str \| None` | `None` | Custom title override for the report header. |
| `style` | `str` | `"general"` | Formatting convention: `"general"`, `"apa"`, or `"ieee"`. |
| `include_figures` | `bool` | `False` | Whether to include supplementary scientific figures. |
| `page_size` | `str` | `"A4"` | Standard paper size (`"A4"` or `"Letter"`, case-insensitive). |
| `landscape` | `bool` | `False` | Whether to print in landscape orientation (ideal for wide tables). |
| `page_numbers` | `bool` | `True` | Whether to include running `"Page X of Y"` footer numbers. |
| `overwrite` | `bool` | `False` | Whether to overwrite an existing destination file (`save_pdf` only). |

---

## 5. Static vs. Figure-Enabled PDF

### Static PDF (Default: `include_figures=False`)
- **Fastest and Most Lightweight**: Does not require Plotly or figure initialization.
- **Scientifically Complete**: All numerical results, estimates, intervals, $p$-values, effect sizes, tables, and diagnostics are presented in semantic text and tables.
- **Offline & Deterministic**: Zero JavaScript execution required for static reports.

### Figure-Enabled PDF (`include_figures=True`)
- Reuses the existing `FigureSpec` layer and Plotly rendering from modern interactive HTML.
- **Deterministic Readiness Signal**: The PDF engine waits for the `Plotly.newPlot` promise to resolve on each chart element (`data-pyautostat-rendered="true"`), rather than relying on arbitrary sleep timers.
- **Fail-Safe**: If figures were explicitly requested but fail to render before timeout, an actionable `ReportError` is raised rather than printing blank chart boxes.
- **No-Figure Fallback**: If a method is intentionally table-only (e.g., Kruskal-Wallis) or if a target has no figures, static PDF is rendered cleanly without requiring Plotly.

---

## 6. Layout, Print CSS, and Page Breaks

The print stylesheet (`@media print`) enforces clean scientific publication standards:

- **Page Breaks**: Metric cards, diagnostic badges, figure containers, and section headings use `break-inside: avoid;` and `break-after: avoid;` to prevent awkward splitting.
- **Multipage Tables**: Tables allow natural page breaks (`table { break-inside: auto; }`) while keeping individual table rows intact (`tr { break-inside: avoid; }`). Table headers repeat on subsequent pages (`thead { display: table-header-group; }`).
- **Landscape Mode**: Callers can pass `landscape=True` for reports with wide tables, such as comprehensive OLS regression models or large post-hoc pairwise comparison matrices.
- **Page Numbers**: Running footers display `"Page <current> of <total>"` in a clean, subdued font that avoids content overlap.

---

## 7. Troubleshooting

### `ReportError: PDF export requires the optional PDF dependency.`
Playwright is not installed in the current Python environment.
**Solution**: Run `python -m pip install "pyautostat[pdf]"`.

### `ReportError: Chromium browser binary is missing.`
Playwright is installed, but the Chromium headless browser has not been downloaded.
**Solution**: Run `python -m playwright install chromium`.

### `ReportError: Interactive figures did not finish rendering before PDF export.`
Figures did not finish rendering before the timeout.
**Solution**: Export with `include_figures=False`, or verify that Plotly is working properly by exporting to interactive HTML first (`save_interactive_html`).

---

## 8. CI and Testing Strategy

- **Cross-Platform Unit Tests**: Argument validation, error taxonomy, overwrite safety, page size normalization, and mock backend behavior run across all Python versions and platforms in the standard CI matrix without requiring Chromium.
- **Dedicated Real-Browser Integration Job**: A dedicated CI workflow job installs Chromium (`playwright install --with-deps chromium`) and runs end-to-end integration tests verifying PDF generation, `%PDF` byte signatures, text extractability, and offline request blocking on Linux.
