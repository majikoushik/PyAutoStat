# PyAutoStat Reporting, Export, Presentation, and Cross-Format Consistency Audit

This document provides a systematic engineering and usability audit of PyAutoStat's presentation, reporting, and export architecture across all supported media. It establishes the cross-format semantic contract, audits each exporter against verified implementation facts, records target compatibility and dependency boundaries, and catalogs findings and priority actions.

---

## 1. Overview Matrix

The matrix below benchmarks all twelve presentation and export channels in PyAutoStat.

| Channel / Medium | Public Entry Point | Accepted Targets | In-Memory Return Type | File-Writing Function | Return Type (File) | Optional Extra | Detail Modes | Style Modes | Figure Support | Status Support | Verdict | Priority |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Terminal View** | `show(target, ...)` | `ResearchWorkflowResult`, `AnalysisResult`, profile `dict`, `ResearchSessionSnapshot`, planning objects, `PresentationView` | `None` (stdout/Rich) | N/A | N/A | None (core uses `rich`) | compact, standard, full | Preserved via Rich tokens | None (text/tables) | Full (`completed`, `partial`, `needs_input`, `data_limited`, `unsupported`, `failed`) | KEEP | P1 |
| **Static HTML** | `to_html(target, ...)` | `ResearchWorkflowResult`, `AnalysisResult`, `ResearchReport`, `PresentationView` | `str` (HTML5) | `save_html(target, path, ...)` | `Path` | None (pure Python) | compact, standard, full | general, apa, ieee | Supplementary only (no inline canvas) | Full (clean status report on blocked) | KEEP | P1 |
| **Interactive HTML** | `to_interactive_html(target, ...)` | `ResearchWorkflowResult`, `AnalysisResult`, `ResearchReport`, `PresentationView` | `str` (HTML5 + Plotly JS) | `save_interactive_html(target, path, ...)` | `Path` | `[report]` (`plotly>=5`) | compact, standard, full | general, apa, ieee | Interactive Plotly charts (offline embedded) | Full (falls back to static status if blocked) | KEEP | P1 |
| **PDF Document** | `to_pdf(target, ...)` | `ResearchWorkflowResult`, `AnalysisResult`, `ResearchReport`, `PresentationView` | `bytes` (`%PDF`) | `save_pdf(target, path, ...)` | `Path` | `[pdf]` (`playwright>=1.40`, Chromium) | compact, standard, full | general, apa, ieee | Optional (`include_figures=True`) | Full (prints canonical HTML status) | KEEP | P1 |
| **Microsoft Word (DOCX)** | `to_docx(target, ...)` | `ResearchWorkflowResult`, `AnalysisResult`, `ResearchReport`, `PresentationView` | `bytes` (OpenXML) | `save_docx(target, path, ...)` | `Path` | `[docx]` (`python-docx>=1.2`), `[figures]` if embedding images | compact, standard, full | general, apa, ieee | Optional (`include_figures=True`, embeds PNGs + captions) | Full (clean status OpenXML report) | KEEP | P1 |
| **Static Figures** | `to_static_figures(target, ...)` | `ResearchWorkflowResult`, `AnalysisResult`, `ResearchReport` | `tuple[StaticFigureArtifact, ...]` | `save_static_figures(target, dir, ...)` | `tuple[Path, ...]` | `[figures]` (`plotly>=6.1`, `kaleido>=1,<2`) | compact, standard, full | Neutral typography | Standalone PNG, SVG, or vector PDF | Returns empty tuple `()` when uncomputable | KEEP | P1 |
| **ResearchReport Native** | `report.to_*()` | `ResearchReport` instance | `dict`, `str`, `dict[str, str]`, `bytes` | `report.save_*()` | `Path` | Matching format extras | compact, standard, full | general, apa, ieee | Format-dependent | Full (reflects report status) | KEEP | P1 |
| **Research Bundle** | `to_bundle(target, ...)` | `ResearchWorkflowResult`, `ResearchReport`, `AnalysisResult`, `PresentationView` | `bytes` (ZIP) | `save_bundle(target, path, ...)` | `Path` | Dependent on requested formats | compact, standard, full | general, apa, ieee | Optional standalone and/or embedded | Validates completed/partial results | KEEP / POLISH | P1 |
| **Structured JSON** | `to_json()`, `workflow.to_json()` | `ResearchWorkflowResult`, `ResearchReport`, `AnalysisResult`, specifications | `str` (JSON UTF-8) | Bundle or custom file write | N/A | None (standard library `json`) | Complete structured payload | N/A (schema-versioned) | N/A | Full (preserves blockers and required fields) | KEEP | P1 |
| **Tabular CSV** | `report.to_csv_tables()` | `ResearchReport`, `ResearchWorkflowResult` (via bundle) | `dict[str, str]` (table ID → CSV string) | Packaged in bundle `pyautostat_bundle/tables/` | N/A | None (standard library `csv`) | Full tabular rows | Standard tabular columns | N/A | Generated when tables exist | KEEP | P1 |
| **Markdown Report** | `report.to_markdown(...)` | `ResearchReport`, `ResearchWorkflowResult` (via bundle) | `str` (GFM markdown) | Packaged in bundle `pyautostat_bundle/report/` | N/A | None | Complete sections | general, apa, ieee | N/A | Full | KEEP | P1 |
| **LaTeX Report** | `report.to_latex(...)` | `ResearchReport`, `ResearchWorkflowResult` (via bundle) | `str` (LaTeX source) | Packaged in bundle `pyautostat_bundle/report/` | N/A | None | Complete sections | general, apa, ieee | N/A | Full | KEEP | P1 |

---

## 2. Cross-Format Semantic Parity Contract

To protect scientific integrity, PyAutoStat enforces a strict semantic parity contract across all presentation and export media. The goal is not pixel-identical rendering, but complete scientific alignment:

1. **Estimand and Target Identity**: Every exporter must represent the exact same scientific estimand (e.g. difference of population means, stochastic superiority, conditional log-odds, in-sample explained variance) without shifting the question based on diagnostic outcomes.
2. **Contrast Direction and Group Ordering**:
   - Independent two-group contrasts must consistently present `'first' - 'second'` derived from declared `group_order` or `contrast` metadata.
   - Paired two-condition contrasts must consistently present `'cond1' - 'cond2'` derived from declared `condition_order`.
   - One-sample comparisons must consistently present `observed sample mean - reference value`.
   - Logistic regression must orient odds ratios toward `event_level` relative to `non_event_level`.
   - OLS regression coefficients must identify categorical dummy levels relative to declared baseline references.
   - ANOVA pairwise follow-ups must preserve contrast order identically across tables and figure forest plots.
   - No exporter may silently invert sign or swap contrast orientation.
3. **Primary Estimates and Uncertainty Bounds**:
   - Primary estimates ($t, F, U, W, \chi^2, r, \tau, \beta, OR, \alpha, \text{ICC}$) must match to the stored precision.
   - Confidence interval bounds must not be reordered, fabricated, or recalculated.
   - Confidence interval labels must dynamically reflect the stored confidence level (e.g. `90% CI`, `95% CI`, `99% CI`) and never default to an assumed 95% when another level was computed.
4. **P-Value Representation**:
   - Formatting must never display `p = 0.000`; values below 0.001 must be represented responsibly as `p < 0.001`.
   - Exporters must never apply significance-based ranking, traffic-light coloring, or asterisks that alter scientific meaning.
5. **Sample Accounting**:
   - Total rows, analyzed rows, excluded rows, group counts ($n_1, n_2$), complete pairs, complete units, and rater counts must agree across terminal, HTML, PDF, DOCX, and research bundles.
6. **Zero Recalculation**:
   - Presentation adapters and renderers must never call SciPy, NumPy, or Statsmodels to recompute hypothesis tests, model parameters, or bootstrap distributions. All formats render from immutable result objects.

---

## 3. Medium-by-Medium Detailed Audits

### 3.1 Terminal Presentation (`show`)
- **Public Entry Point**: `pyautostat.show(target, *, detail="standard", console=None)`
- **Accepted Targets**: `ResearchWorkflowResult`, `AnalysisResult`, dataset profile `dict`, `ResearchSessionSnapshot`, `StudyPlanningResult`, `StatisticalAnalysisPlan`, `PlanAdherenceResult`, `PresentationView`.
- **Return Type**: `None` (renders to stdout/Rich console).
- **Detail Modes**:
  - `compact`: Outputs a single focused card highlighting the contrast, primary estimate, dynamic CI, and p-value.
  - `standard`: Outputs the full primary result card, key metrics grid, group summaries, diagnostic indicators, and plain-language interpretation.
  - `full`: Adds deeper component tables, methodology rationale, formula documentation, and audit trail metadata.
- **Workflow Status**: Blocked workflows (`needs_input`, `data_limited`, `unsupported`, `failed`) render a dedicated status card showing exact missing input fields or diagnostic blockers.
- **Privacy Model**: Pure aggregate presentation; zero raw row-level records are displayed.
- **Verdict**: **KEEP** | **Priority**: P1.

### 3.2 Static HTML Export
- **Public Entry Point**: `pyautostat.to_html(target, ...)`, `pyautostat.save_html(target, path, ...)`
- **Accepted Targets**: `ResearchWorkflowResult`, `AnalysisResult`, `ResearchReport`, `PresentationView`.
- **Return Types**: `to_html` returns `str`; `save_html` returns `pathlib.Path`.
- **Dependencies**: Zero external dependencies. Operates purely with Python standard library.
- **Styling**: Modern, responsive CSS with accessible contrast ratios, print stylesheet (`@media print`), and semantic markup (`<html lang="en">`, `<th>`, `<section>`).
- **Style Modes**: `general`, `apa` (Times New Roman, 1.15 line spacing, academic borders), `ieee` (numbered sections, technical tables).
- **Security**: All user-supplied text (column names, string values, dataset labels) is HTML-escaped via `html.escape()`.
- **Blocked Workflows**: Renders clean, styled status reports explaining blockers or required fields.
- **Verdict**: **KEEP** | **Priority**: P1.

### 3.3 Interactive HTML Export
- **Public Entry Point**: `pyautostat.to_interactive_html(target, ...)`, `pyautostat.save_interactive_html(target, path, ...)`
- **Accepted Targets**: `ResearchWorkflowResult`, `AnalysisResult`, `ResearchReport`, `PresentationView`.
- **Return Types**: `to_interactive_html` returns `str`; `save_interactive_html` returns `pathlib.Path`.
- **Dependencies**: `plotly>=5` (`pyautostat[report]`). If `include_figures=True` is requested on a target with no visual specifications (e.g. table-only method), falls back cleanly without requiring Plotly.
- **Self-Contained & Offline**: Embeds Plotly JS inline (`include_plotlyjs=True`) ensuring complete offline functionality without external CDN requests.
- **Figures**: Forest plots, interval charts, and heatmaps render from stored `FigureSpec` objects.
- **Verdict**: **KEEP** | **Priority**: P1.

### 3.4 PDF Export
- **Public Entry Point**: `pyautostat.to_pdf(target, ...)`, `pyautostat.save_pdf(target, path, ...)`
- **Accepted Targets**: `ResearchWorkflowResult`, `AnalysisResult`, `ResearchReport`, `PresentationView`.
- **Return Types**: `to_pdf` returns `bytes`; `save_pdf` returns `pathlib.Path`.
- **Dependencies**: `playwright>=1.40` (`pyautostat[pdf]`) with headless Chromium. Raises actionable `ReportError` if missing.
- **Fidelity**: Prints the canonical HTML presentation via Chromium's printing engine. Guarantees 100% tabular, typographical, and numerical parity with HTML.
- **Page Setup**: Configurable page sizes (`"A4"`, `"Letter"`), landscape orientation (`landscape=True`), and running page numbers in footers (`page_numbers=True`).
- **Privacy & Security**: External network access is blocked during PDF generation (`networkidle` wait with isolated local page context). Raw data rows are never serialized.
- **Blocked Workflows**: Prints the canonical status report cleanly to PDF.
- **Verdict**: **KEEP** | **Priority**: P1.

### 3.5 Microsoft Word Export (DOCX)
- **Public Entry Point**: `pyautostat.to_docx(target, ...)`, `pyautostat.save_docx(target, path, ...)`
- **Accepted Targets**: `ResearchWorkflowResult`, `AnalysisResult`, `ResearchReport`, `PresentationView`.
- **Return Types**: `to_docx` returns `bytes`; `save_docx` returns `pathlib.Path`.
- **Dependencies**: `python-docx>=1.2` (`pyautostat[docx]`). If `include_figures=True`, also requires `pyautostat[figures]`.
- **Editable Structure**: Real Word documents with native OpenXML headings, paragraphs, and tables (`w:tbl`). Never embeds rasterized text or HTML approximations.
- **Advanced OpenXML Features**:
  - `w:tblHeader`: Table headers automatically repeat on subsequent pages.
  - `w:cantSplit`: Table rows do not break awkwardly across page splits.
  - `w:fldSimple`: Dynamic Word field codes (`PAGE` / `NUMPAGES`) in footers.
- **Figure Embedding**: Supported via `include_figures=True`. Generates static high-resolution PNG figures via Kaleido and embeds them with styled captions.
- **Blocked Workflows**: Renders editable status documents recording missing fields or diagnostic blockers.
- **Verdict**: **KEEP** | **Priority**: P1.

### 3.6 Static Scientific Figures
- **Public Entry Point**: `pyautostat.to_static_figures(target, ...)`, `pyautostat.save_static_figures(target, directory, ...)`
- **Accepted Targets**: `ResearchWorkflowResult`, `AnalysisResult`, `ResearchReport`.
- **Return Types**: `to_static_figures` returns `tuple[StaticFigureArtifact, ...]`; `save_static_figures` returns `tuple[Path, ...]`.
- **Dependencies**: `plotly>=6.1` and `kaleido>=1,<2` (`pyautostat[figures]`). If target has no figure specifications, returns `()` without requiring dependencies.
- **Supported Formats**: PNG (raster), SVG (vector), PDF (vector figure).
- **Supported Visual Types**:
  - `estimate_ci`: Contrast estimate with dynamic confidence interval and neutral zero reference line.
  - `forest`: OLS regression coefficient forest with zero reference line.
  - `odds_ratio_forest`: Logistic regression odds-ratio forest with 1.0 reference line and logarithmic x-axis.
  - `pairwise_forest`: ANOVA post-hoc pairwise comparisons with simultaneous confidence intervals.
  - `contingency_heatmap`: 2D categorical contingency count grid with marginal labels.
  - Reliability plots for ICC and Cronbach's alpha with confidence intervals and neutral non-null reference lines.
- **Scientific Safeguards**: Zero recalculation; neutral reference lines match mathematical quantities (0 for differences/correlations, 1 for odds ratios, None for ICC/reliability).
- **Verdict**: **KEEP** | **Priority**: P1.

### 3.7 ResearchReport Native Exports
- **Public Interface**: `report.to_dict()`, `report.to_json()`, `report.to_csv_tables()`, `report.to_markdown()`, `report.to_latex()`, `report.to_html()`, `report.to_pdf()`, `report.to_docx()`, `report.to_bundle()`
- **Accepted Target**: `ResearchReport` instance (effectively immutable snapshot).
- **Role**: Serves as the canonical structured report container with complete sections, diagnostic records, sensitivity comparisons, and governance metadata.
- **Verdict**: **KEEP** | **Priority**: P1.

### 3.8 Research Export Bundles
- **Public Entry Point**: `to_bundle(target, ...)`, `save_bundle(target, path, ...)`, `verify_bundle(source)`
- **Accepted Targets**: `ResearchWorkflowResult`, `ResearchReport`, `AnalysisResult`, `PresentationView`.
- **Return Types**: `to_bundle` returns `bytes` (ZIP); `save_bundle` returns `pathlib.Path`; `verify_bundle` returns `BundleVerificationResult`.
- **Archive Contents**:
  - `pyautostat_bundle/README.md`: Human-readable guide and explicit privacy notice.
  - `pyautostat_bundle/manifest.json`: Schema version 1 manifest with file roles, exact byte counts, and SHA-256 cryptographic digests.
  - `pyautostat_bundle/report/`: Rendered publication documents (HTML, PDF, DOCX, Markdown, LaTeX).
  - `pyautostat_bundle/tables/`: Sanitized CSV files of all presentation tables.
  - `pyautostat_bundle/figures/`: Static scientific figures (when requested).
  - `pyautostat_bundle/provenance/`: Execution records, decision ledgers, and reproducibility specifications.
- **Integrity Verification**: `verify_bundle` recalculates SHA-256 digests and validates member paths against traversal attacks. Explicitly documented: integrity verification detects corruption or tampering, but does not constitute a digital signature or PKI certificate.
- **Blocked Workflow Policy**: Bundles require a completed or partial workflow with computed statistical results. Blocked workflows raise a clear `ReportError` guiding users to status inspection via `show`, `save_html`, or `save_docx`.
- **Verdict**: **KEEP / POLISH** | **Priority**: P1.

---

## 4. Architectural Separation: ResearchReport vs PresentationView

| Dimension | `PresentationView` | `ResearchReport` |
|---|---|---|
| **Primary Role** | Normalized display model tailored for renderers | Canonical structured research report snapshot |
| **Origin** | Produced by `adapt(target, detail=...)` | Created by `ResearchWorkflowResult` or `ResearchAssistant.run()` |
| **Scope** | Display-ready metrics, tables, diagnostic badges, interpretation | Comprehensive sections (executive summary, methods, results, diagnostics, sensitivity, audit) |
| **Native Methods** | None (pure immutable dataclass) | `to_dict`, `to_json`, `to_csv_tables`, `to_markdown`, `to_latex`, `to_html`, `to_pdf`, `to_docx`, `to_bundle` |
| **Supported Renderers** | Terminal `show`, `HtmlRenderer`, `DocxRenderer` | All presentation renderers plus report-native serialization |
| **Target Audience** | Renderers, adapters, and UI integrations | Researchers, archival pipelines, and governance ledgers |
| **User Guidance** | Beginners rarely construct directly; use `show(workflow)` | Accessible via `workflow.report`; beginners call `save_html`, `save_pdf`, etc. |

Both classes serve distinct, non-overlapping responsibilities: `PresentationView` guarantees display decoupling, while `ResearchReport` guarantees scientific record permanence.

---

## 5. Detail Mode and Style Mode Parity

### 5.1 Detail Modes
- **`compact`**: Fast primary inspection. Highlights contrast, point estimate, dynamic CI, and p-value. Truncates long multi-group tables to 6 rows.
- **`standard`**: Default researcher review. Full primary results, key metrics, group summaries, diagnostic indicators, and interpretation. Truncates pairwise tables exceeding 10 rows to 6 rows.
- **`full`**: Comprehensive archival review. Zero table truncation, complete pairwise contrast matrices, diagnostic details, sensitivity evaluations, practical significance checks, and full audit trails.

### 5.2 Style Modes
Style modes modify presentation typography only; they never alter calculations, p-values, or decisions:
- **`general`** (default): Modern typography, clean layout, system fonts.
- **`apa`**: APA-oriented preset using Times New Roman typography, 1.15 line spacing, standard italicized symbols ($t, F, p, d, r, \alpha$), and restrained horizontal table rules.
- **`ieee`**: IEEE-oriented preset using Times New Roman typography, numbered sections, compact multi-column layouts, and technical column headers.

*(Note: Style presets provide publication-oriented typography; they do not certify formal journal or institutional compliance.)*

---

## 6. Action Items and Findings

### 6.1 P0 (Critical Correctness / Parity) — Resolved
1. **Wilcoxon Paired Terminology**: Replaced inaccurate "Wilcoxon rank sum" reference in paired comparison documentation with "Wilcoxon signed-rank".
2. **HC3 Covariance Clarification**: Corrected market benchmarking description to distinguish Breusch-Pagan (heteroscedasticity diagnostic) from HC3 (robust covariance estimator).
3. **Cronbach Alpha Standardized Alpha Contradiction**: Removed unverified claim that reliability reports provide standardized alpha; PyAutoStat provides sample-variance Cronbach's alpha with bootstrap CI.
4. **DOCX Figure Embedding Documentation**: Reconciled conflicting statements between `docs/DOCX_REPORTING.md` and `docs/STATIC_FIGURES.md`. Verified that `to_docx(..., include_figures=True)` and `DocxRenderer` embed static PNG figures; updated DOCX documentation accordingly.
5. **Categorical Parity Claims Scoped**: Narrowed unverified global "no contradictions remain" claims and formal compliance claims ("WCAG AA", "publication-ready") to accurate, verified statements.

### 6.2 P1 (High-Value Usability Polish) — Implemented
1. **Blocked Workflow Bundle Policy**: Hardened `BundleAssembler` to reject blocked/pending workflows (`needs_input`, `data_limited`, `unsupported`, `failed`) with an actionable `ReportError` directing users to `show`, `save_html`, or `save_docx`.
2. **Canonical Export API Table**: Documented the canonical compatibility table in `API_REFERENCE.md` contrasting beginner `save_*` APIs with in-memory `to_*` functions and exact return types.
3. **Documentation Navigation**: Linked `REPORTING_AUDIT.md` across `docs/README.md`.

### 6.3 P2 (Deferred to Future Work)
1. **Alternative Word Processor Layout Parity**: Microsoft Word, LibreOffice Writer, and Google Docs use distinct pagination engines; exact line-break equivalence remains out of scope.
2. **Interactive Jupyter Notebook Widgets**: Dedicated interactive IPyWidgets remain deferred to future UI-focused phases.
3. **Cryptographic PKI Digital Signatures**: Research bundles use SHA-256 integrity digests; formal public-key infrastructure (PKI) signing remains deferred.
