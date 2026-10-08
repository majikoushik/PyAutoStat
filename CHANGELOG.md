# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

-   Completed exhaustive result ergonomics consistency audit across all 24 registered statistical methods (`docs/RESULT_ERGONOMICS_AUDIT.md`), verifying zero-recalculation guarantees and documenting ambiguity rules for complex multi-parameter models.
-   Completed authoritative effect-size and uncertainty gap audit (`docs/EFFECT_SIZE_AND_UNCERTAINTY_AUDIT.md`), evaluating estimand alignment, runtime field locations, independent validation levels, candidate metrics, and non-blocking candidate prioritization (Hedges' $g$, standardized regression betas, and omega-squared classified as optional future metrics).
-   Added deterministic SHA-256 validation framework content fingerprinting (`validation_framework_content_sha256`) in `validation/manifest.py` and `validation/run_reference_validation.py` to truthfully record framework content provenance across commits.

### Fixed

-   Hardened result ergonomics convenience accessors in `pyautostat.result_access`:
    -   Corrected repeated-measures ANOVA statement formatting to read authoritative integrated schema `values["greenhouse_geisser"]["epsilon"]` and `values["greenhouse_geisser"]["applied"]`, preventing epsilon from displaying as unavailable when stored.
    -   Corrected `result.effect_size_confidence_interval` lookup to read authoritative nested `values["effect_size"]["confidence_interval"]`.
    -   Corrected `result.degrees_of_freedom` extraction to read canonical stored values (returning 2-element tuples for ANOVA variants and ICC, scalars for single-df tests, and `None` for ambiguous models).
    -   Corrected two-way factorial ANOVA statement formatting to extract residual denominator df, read partial $\eta^2$ from `term["effect_size"]["value"]`, format $F(\text{df}_{\text{num}}, \text{df}_{\text{resid}})$, and omit the residual row from inferential reporting.
    -   Corrected logistic regression statement formatting to read McFadden's pseudo-$R^2$ from `model_fit["mcfadden_r2"]`.
    -   Removed unstored $n-2$ degrees of freedom derivation from Pearson correlation statements, enforcing strict zero-recalculation guarantees.
    -   Corrected McNemar statement wording to reflect the exact paired binary procedure and avoided inaccurate "continuity-corrected" descriptions.
    -   Expanded `sample_accounting` to surface all tracked unit, pair, target, rater, and difference fields, using explicit `is not None` logic to preserve valid zero values.
    -   Preserved exact scalar, tuple, or `None` objects in `result.to_dataframe("primary")` for degrees of freedom rather than stringifying.
-   Reconciled audit and documentation surfaces against runtime source:
    -   Distinguished sample mean from primary estimate/estimand (mean difference from reference) in one-sample t-test audit.
    -   Corrected paired-t sample accounting audit to remove Wilcoxon zero-difference fields.
    -   Documented runtime availability of Friedman Kendall's W bootstrap confidence interval (`PRESENT_BUT_VALIDATION_GAP`).
    -   Scoped Pearson Fisher-z interval description as an analytical transformation-based interval rather than an exact finite-sample interval.
    -   Confirmed Welch-ANOVA effect-size documentation aligns with runtime schema (`name: "global standardized effect"`, `value: None`, `status: "not_applicable"`).

### Changed

-   Modernized the example suite to use the canonical Rich `show()` presentation for statistical, planning, governance, and workflow results.
-   Removed redundant hand-built terminal tables and duplicated statistical formatting from tutorial examples.
-   Aligned example terminal framing with the current presentation system while preserving business/research teaching commentary.

### Added

-   Implemented optional publication-quality static scientific figure export (`to_static_figures`, `save_static_figures`):
    -   Renders canonical `FigureSpec` objects directly to standalone PNG, SVG, and figure-PDF artifacts using modern Plotly and Kaleido v1+ headless Chromium rendering.
    -   Refactored shared Plotly converter (`spec_to_plotly_figure`) in `pyautostat.presentation.figures.plotly` used by both interactive HTML and static image export layers.
    -   Guarantees zero statistical recalculation and zero raw dataset row exposure; figures consume only authoritative stored estimates, intervals, and aggregates.
    -   Enforced centralized dimension defaults and rigorous parameter validation (positive integer dimensions, maximum dimension 10,000 px, scale limit 8.0, booleans rejected as integers).
    -   Preflights destination path collisions with overwrite protection (`overwrite=False` by default).
    -   Returns empty tuple `()` without requiring Kaleido or Chrome when targets have no figure specifications.
    -   Provides actionable setup errors distinguishing missing `pyautostat[figures]` package from missing Chrome browser (`plotly_get_chrome`).
    -   Demonstrated in `examples/18_static_scientific_figures.py` and documented in `docs/STATIC_FIGURES.md`.

-   Implemented optional scientific figure embedding in editable Microsoft Word reports (`to_docx`, `save_docx`, `ResearchReport.to_docx`, `ResearchReport.save_docx`):
    -   Added `include_figures: bool = False` parameter (defaulting to False to preserve zero-Kaleido DOCX usage).
    -   When enabled, embeds high-resolution PNG figures near their relevant semantic sections (`key_results`, `coefficients`, `pairwise`, `contingency`, `cell_summary`).
    -   Added numbered, editable captions styled with `PyAutoStat Figure Caption` (e.g. `Figure 1. ...`).
    -   Automatically scales image width to printable document dimensions across page formats (A4/Letter) and orientations (portrait/landscape) without rasterizing report text, tables, or diagnostics.

-   Implemented optional static figure packaging in research export bundles:
    -   Added `include_static_figures: bool = False` and `static_figure_format: str = "png"` to bundle options.
    -   Packages standalone scientific figures under `pyautostat_bundle/figures/` with manifest role `"scientific_figure"` and verified SHA-256 digests.

-   Implemented reproducible research export bundles (`pyautostat.to_bundle`, `pyautostat.save_bundle`, `pyautostat.verify_bundle`, and `ResearchReport.to_bundle` / `ResearchReport.save_bundle`):
    -   Introduced modern public bundle APIs: `to_bundle(target, *, formats=("html", "json", "csv"), detail="full", style="general", title=None, include_figures=False, page_size="A4", landscape=False, page_numbers=True) -> bytes` and `save_bundle(target, path, *, formats=("html", "json", "csv"), detail="full", style="general", title=None, include_figures=False, page_size="A4", landscape=False, page_numbers=True, overwrite=False) -> Path`.
    -   Implemented offline ZIP archive assembler packaging existing canonical reports (`html`, `interactive_html`, `pdf`, `docx`, `json`, `csv`, `markdown`, `latex`), presentation tables, and execution provenance without recalculating statistics.
    -   Introduced versioned manifest schema (`BUNDLE_SCHEMA_VERSION = 1`) recording bundle metadata, target type, method ID, detail/style options, and sorted member records with file roles, formats, exact byte sizes, and cryptographic SHA-256 digests.
    -   Implemented cryptographic integrity verification engine `verify_bundle(bundle_input) -> BundleVerificationResult` accepting bytes, paths, or file strings; verifies SHA-256 digests and file sizes, detects missing declared files, undeclared files, modified bytes, and structural archive corruptions.
    -   Enforced single-render efficiency: each requested format is rendered exactly once during bundle assembly.
    -   Packaged authoritative execution provenance: includes `provenance/analysis.json`, `provenance/audit.json`, `provenance/reproducibility.json`, and `provenance/session.json` where already recorded without triggering new scans.
    -   Enforced strict privacy guarantees: raw observations, participant identifiers, and source DataFrame rows are strictly excluded.
    -   Hardened ZIP path safety and resource bounds: validates forward-slash relative member paths, rejects `..` traversal, absolute paths, and Windows drive letters; sanitizes table filenames; enforces conservative limits on member count (500), single file size (50 MB), and total uncompressed size (100 MB).
    -   Added demonstration in `examples/17_research_export_bundle.py` and comprehensive documentation in `docs/RESEARCH_BUNDLES.md`.

-   DOCX export fidelity hardening:
    -   Replaced synthetic all-24 DOCX test with schema-faithful method fixtures verifying actual method labels and primary numerical statistics across all 24 registered methods.
    -   Aligned HTML and DOCX standard-mode table row selection via shared `resolve_table_row_limit` helper; standard mode bounds long tables to 6 rows in stable source order with an informative truncation notice, while full mode retains all rows.
    -   Eliminated raw Python `dict`/`list` string repr dumps from reader-facing DOCX report sections via structured text/bullet rendering helpers.

-   Implemented optional editable Microsoft Word (`.docx`) research reporting layer (`pyautostat.to_docx`, `pyautostat.save_docx`, and `ResearchReport.to_docx` / `ResearchReport.save_docx`):
    -   Introduced modern public DOCX export APIs: `to_docx(target, *, detail="standard", title=None, style="general", page_size="A4", landscape=False, page_numbers=True) -> bytes` and `save_docx(target, path, *, detail="standard", title=None, style="general", page_size="A4", landscape=False, page_numbers=True, overwrite=False) -> Path`.
    -   Implemented genuine OpenXML export architecture directly rendering from canonical `PresentationView` and `ResearchReport` structures via `python-docx`, eliminating HTML round-tripping or rasterized page captures.
    -   Provided flexible page setup: supports standard page sizes (`A4`, `Letter`) and portrait or landscape orientation with centralized margins (~20 mm).
    -   Added real dynamic Word footer field codes (`PAGE` and `NUMPAGES`) rendering `Page X of Y` when `page_numbers=True`.
    -   Created clean semantic named styles (`PyAutoStat Title`, `PyAutoStat Subtitle`, `PyAutoStat Body`, `PyAutoStat Metric Label`, `PyAutoStat Metric Value`, `PyAutoStat Table Header`, `PyAutoStat Diagnostic`, `PyAutoStat Warning`) with typographic style presets (`general`, `apa`, `ieee`).
    -   Structured OpenXML tables: converts statistical metrics and displays to true Word tables, applies `w:tblHeader` to repeat header rows across pages, applies `w:cantSplit` to individual rows to prevent awkward page splits, and right-aligns numerical estimates.
    -   Validated universal presentation coverage: renders all 24 registered statistical method families and all 14 non-analysis governance/planning objects without statistical adapters.
    -   Preserved research report fidelity: structured sections (Executive Summary, Research Question, Dataset, Methods, Results, Diagnostics, Interpretation, Practical Significance, Sensitivity Analysis, Limitations, Warnings, Analysis Record) matching standalone export values.
    -   Enforced zero-recalculation guarantee and privacy protections: never re-executes statistical models or leaks raw row-level data.
    -   Added actionable missing-dependency error handling when `python-docx` is not installed, preserving clean base imports.
    -   Demonstrated in `examples/16_docx_reporting.py` and documented in `docs/DOCX_REPORTING.md`.

### Fixed

-   Strengthened bundle manifest validation: strictly validates manifest file records before lookup map conversion, rejecting missing/malformed SHA-256 digests, invalid/negative/boolean sizes, non-mapping records, unsafe paths, manifest self-declarations, and duplicate path declarations with actionable `ReportError`.
-   Rejected duplicate generated bundle member paths: tracks generated internal bundle member paths in `BundleAssembler`, raising `ReportError` immediately upon collision (e.g. sanitized CSV table IDs) before ZIP archive creation.
-   Propagated figure options for direct AnalysisResult PDF bundle export: ensured `include_figures` is passed to `to_pdf(target_obj, ...)` when bundling direct analysis results.
-   Completed all-format bundle integration coverage: extended full-publication bundle tests to exercise `interactive_html` alongside `html`, `pdf`, `docx`, `json`, `csv`, `markdown`, and `latex`.
-   Removed numbered-phase terminology across all active repository documentation files to preserve clean, feature-based governance.
-   Corrected bundle documentation determinism wording: clarified that SHA-256 digests verify transmission integrity rather than cryptographic digital signatures or deterministic byte reproducibility across differing compression engines.
-   Fixed brittle case-sensitive string assertion in `test_partial_result_pdf_export`.
-   Narrowed figure-wait exception handling in PDF backend to distinguish `PlaywrightTimeoutError` from general `PlaywrightError`.
-   Corrected PDF documentation to state "native selectable text and structured table layout" rather than claiming formally tagged semantic PDF tables.

-   Implemented optional publication-oriented PDF export and print fidelity layer (`pyautostat.to_pdf`, `pyautostat.save_pdf`, and `ResearchReport.to_pdf` / `ResearchReport.save_pdf`):
    -   Introduced modern public PDF export APIs: `to_pdf(target, *, detail="standard", title=None, style="general", include_figures=False, page_size="A4", landscape=False, page_numbers=True)` and `save_pdf(target, path, *, detail="standard", title=None, style="general", include_figures=False, page_size="A4", landscape=False, page_numbers=True, overwrite=False)`.
    -   Implemented browser-print architecture reusing canonical HTML reports via headless Chromium and Playwright, preserving the same canonical stored statistical values, tabular content, and scientific meaning as the HTML presentation without recalculating statistics (pagination and print layout may differ under Chromium rendering).
    -   Enforced 100% offline generation: browser execution context aborts all external network requests (`http://`, `https://`, fonts, trackers).
    -   Provided flexible page formatting: supports standard paper sizes (`A4`, `Letter`) and portrait or landscape orientation for wide tables.
    -   Added running page numbers in the print footer (`"Page <current> of <total>"`) via Chromium print templates.
    -   Supported optional inclusion of existing scientific figures (`include_figures=True`), waiting deterministically for chart readiness promises (`data-pyautostat-rendered="true"`) with actionable timeout handling.
    -   Preserved privacy guarantees: never serializes or embeds raw row-level observations.
    -   Preserved native selectable text and structured table layout rather than raster screenshots.
    -   Added actionable error handling for missing Playwright or missing Chromium browser binary with exact installation guidance.
    -   Demonstrated in `examples/15_pdf_reporting.py` and documented in `docs/PDF_REPORTING.md`.
-   Implemented optional interactive HTML presentation layer and Plotly-backed scientific figures (`pyautostat.to_interactive_html`, `pyautostat.save_interactive_html`, and `ResearchReport.to_interactive_html` / `ResearchReport.save_interactive_html`):
    -   Introduced modern public interactive reporting APIs: `to_interactive_html(target, *, detail="standard", title=None, style="general", include_figures=True)` and `save_interactive_html(target, path, *, detail="standard", title=None, style="general", include_figures=True, overwrite=False)` supporting `ResearchWorkflowResult`, `AnalysisResult`, `ResearchReport`, and `PresentationView`.
    -   Added Plotly-independent neutral figure specification layer (`FigureSpec`, `FigureSeries`, `build_figure_spec` in `pyautostat.presentation.figures`), isolating visualization data structures from rendering backends and keeping figures strictly supplementary to canonical textual and tabular presentation.
    -   Enforced zero-recalculation guarantee: figure adapters extract authoritative stored estimates, confidence intervals, summaries, coefficient records, pairwise contrasts, contingency frequencies, and cell summaries without re-running models or recalculating inference.
    -   Implemented raw-data privacy guarantees: figures never serialize row-level observations, raw scatterplots, raw jitter plots, or participant hover clouds, preventing data leaks in shared standalone HTML files.
    -   Created reusable scientific figure renderers: point estimate with CI (differences, correlations, reliability), OLS coefficient forest plot (reference line at 0), binary logistic regression odds-ratio forest plot (reference line at 1.0, log x-scale), ANOVA pairwise comparison forest plot (Games-Howell and Tukey-Kramer), categorical contingency count heatmap (Pearson chi-square and Fisher exact), and factorial cell profile plot (two-way ANOVA).
    -   Intentionally omitted figures where authoritative inputs or intervals are absent (e.g. Kruskal-Wallis pairwise contrasts where Dunn lacks authoritative intervals) to avoid visual fabrication.
    -   Guaranteed 100% offline, self-contained interactive delivery by embedding the Plotly JavaScript runtime once per document in the `<head>`, eliminating external CDN dependencies, tracking, and redundant bundle duplication across multiple figures.
    -   Implemented restrained Plotly configuration: minimal hover modebar, disabled scroll zoom, and no external cloud links.
    -   Protected against XSS and script injection by safely Unicode-escaping user-controlled labels, titles, and JSON metadata (`\u003c`, `\u003e`, `\u0026`).
    -   Maintained full backward compatibility: default static HTML output (`to_html`, `save_html`, `ResearchReport.to_html`) remains 100% static, JavaScript-free, Plotly-free, and lightweight.
    -   Maintained distinction with legacy `ReportGenerator.to_interactive_html()`, which remains preserved for backward compatibility with the legacy `StatisticalAnalyzer` path.
    -   Audited and respected `ResearchReport._include_figures` flag, allowing runtime rendering overrides without mutating report payload.
    -   Added demonstration in `examples/14_interactive_html_reporting.py` and comprehensive tests in `tests/test_interactive_html.py`.
-   Unified `ResearchReport.to_html()` and completed static HTML presentation coverage (`pyautostat.to_html`, `pyautostat.save_html`, and `pyautostat.presentation.html`):
    -   Unified `ResearchReport.to_html()` to consume the exact same shared HTML component and rendering architecture (`ResearchReportHtmlRenderer` in `presentation/html/report_renderer.py`), eliminating the legacy separate hand-built HTML path while preserving canonical payload serialization.
    -   Enabled source `AnalysisResult` normalization via `adapt(report._source_result, detail=...)` without re-running statistical algorithms, ensuring standalone HTML and `ResearchReport` HTML agree on primary estimates, confidence intervals, effect sizes, p-values, table fields, and diagnostic statuses.
    -   Completed static HTML rendering and validation across all 24 registered statistical method IDs: one-sample t-test, Welch t-test, Student t-test, Mann-Whitney U, paired t-test, Wilcoxon signed-rank, Welch one-way ANOVA, classical one-way ANOVA, Kruskal-Wallis, Pearson correlation, Spearman rank correlation, Kendall's tau-b, point-biserial correlation, partial Pearson correlation, Pearson chi-square test of independence, Fisher's exact test, McNemar test, linear regression (OLS), logistic regression, Cronbach's alpha, repeated-measures ANOVA, Friedman test, two-way factorial ANOVA, and intraclass correlation (ICC).
    -   Validated static HTML rendering across all non-analysis and governance presentation objects: dataset profile, frequency table, cross-tabulation, `StudyPlanningResult`, `SensitivityResult`, `PracticalSignificanceResult`, `StatisticalAnalysisPlan`, `PlanAdherenceResult`, `ReportingCompletenessResult`, `AuditResult`, `ReproducibilityRecord`, `ReproductionOutcome`, `DecisionLedger`, and `ResearchSessionSnapshot`.
    -   Supported distinct rendering for non-completed workflow states: `needs_input`, `data_limited`, `unsupported`, and `failed` without raw tracebacks.
    -   Eliminated duplicate visible table titles by consolidating visible titles in section headers and adding accessible `<caption class="sr-only">` table captions.
    -   Expanded semantic table column alignment heuristics across library-wide text identifiers (Method, Scenario, Comparability, Reference, Severity, Rationale, Planning Status, Estimand, Adjustment, Reason, Event, Category, Level, etc.) to guarantee left-alignment.
    -   Enforced responsive table row bounding in standard mode (showing up to 6 rows with a pagination note) while rendering all rows in full mode.
    -   Preserved dynamic confidence levels (90%, 95%, 99%), falsey value fidelity (0, 0.0, False), and the zero-recalculation guarantee.
    -   Added comprehensive coverage test suite in `tests/test_html_presentation_coverage.py`.
-   Implemented the modern static HTML presentation system (`pyautostat.to_html`, `pyautostat.save_html`, and `pyautostat.presentation.html`) for reproducible research workflows:
    -   Added public `to_html(target, *, detail="standard", title=None, style="general")` and `save_html(target, path, *, detail="standard", title=None, style="general", overwrite=False)` APIs supporting `ResearchWorkflowResult`, `AnalysisResult`, `ResearchReport`, and `PresentationView`.
    -   Generalized the presentation architecture by introducing `PresentationView` (aliasing `TerminalView` for full backward compatibility), ensuring terminal and HTML renderers consume identical normalized semantic view models.
    -   Delivered optimized, validated initial HTML presentation for four representative analysis families: Welch independent-samples t-test (signed mean difference, dynamic effect CI, group summary, unassumed variance), Pearson linear correlation (r, dynamic CI, causality limitation), OLS linear regression (model fit, coefficient table in stable source order, Breusch-Pagan and VIF diagnostics), and Kruskal-Wallis rank test (H omnibus test, Dunn-Holm pairwise comparisons, unassumed equal variance, non-string zero group label preservation, no fabricated IQR).
    -   Upgraded canonical `ResearchReport.to_html()` and `ResearchReport.save_html()` with centralized CSS tokens, responsive layouts, print styles, and optional `detail` and `title` overrides while maintaining complete backward compatibility with existing tests and formats.
    -   Guaranteed 100% static, offline, and self-contained HTML output with zero external CDN dependencies, zero JavaScript requirement, and no charting library dependencies.
    -   Implemented security hardening with strict HTML escaping across all user-supplied text, variable names, group labels, and titles to prevent XSS.
    -   Enforced zero-recalculation guarantee where HTML generation never re-runs or re-estimates backend statistical models.
    -   Added documentation in `docs/HTML_REPORTING.md`, demonstration script in `examples/12_html_reporting.py`, and comprehensive test suite in `tests/test_html_presentation.py`.
-   Completed the Rich-based terminal presentation layer (`pyautostat.show` / `pyautostat.presentation`) across all supported analytical, descriptive, and governance families:
    -   Implemented dedicated presentation renderers and modular adapters covering all 24 registered statistical method IDs: one-sample t-test, Welch t-test, Student t-test, Mann-Whitney U, paired t-test, Wilcoxon signed-rank, Welch one-way ANOVA, classical one-way ANOVA, Kruskal-Wallis, Pearson correlation, Spearman rank correlation, Kendall's tau-b, point-biserial correlation, partial Pearson correlation, Pearson chi-square test of independence, Fisher's exact test, McNemar test, linear regression (OLS), logistic regression, Cronbach's alpha, repeated-measures ANOVA, Friedman test, two-way factorial ANOVA, and intraclass correlation (ICC).
    -   Added terminal presentation support for descriptive outputs: `frequency_table()` with level counts, valid/total/cumulative percentages, and cardinality diagnostics; and `cross_tab()` with 2D contingency counts and complete paired case accounting.
    -   Added terminal presentation support for governance and prospective planning records: `StudyPlanningResult`, `SensitivityResult`, `PracticalSignificanceResult`, `StatisticalAnalysisPlan`, `PlanAdherenceResult`, `ReportingCompletenessResult`, `AuditResult`, `ReproducibilityRecord`, `ReproductionOutcome`, `DecisionLedger`, and `ResearchSessionSnapshot`.
    -   Added direct `show(AnalysisResult)` presentation support for standalone analysis objects containing self-describing statistical values.
    -   Preserved scientific invariants across all renderers: mandatory orientation displays (contrast direction, paired conditions, binary levels, modeled events, factor interactions, and canonical ICC definition preceding estimate), negative ICC values preserved without clamping, no post-hoc power display in study planning, no p-value sorting in sensitivity analysis, and operational badges strictly decoupled from statistical significance.
    -   No statistical engine calculations, formulas, or result schemas were modified.
    -   Added demonstration gallery in `examples/11_rich_terminal_method_gallery.py` and comprehensive coverage tests in `tests/test_terminal_presentation_coverage.py`.

### Fixed

-   Remediated interactive HTML and presentation layer for scientific fidelity and CI stability:
    -   Fixed partial-result presentation crash across adapters (`means.py`, `paired.py`, `repeated.py`, `multigroup.py`, `governance.py`) when optional components such as `effect_size` are explicitly `None`, restoring clean rendering for valid partial `ResearchReport`s and passing audit checks.
    -   Removed silent 95% CI fallback in figure layer; unified on centralized `confidence_interval_phrase` helper with dynamic interval precedence (stored interval level -> spec fallback -> neutral "confidence interval" when unknown).
    -   Made source-less `ResearchReport` figure fallback method-aware and conservative, correctly mapping reference lines (e.g. 0.0 for differences/associations, 1.0 for odds ratios) and omitting figures on ambiguous payloads rather than guessing null values.
    -   Removed generic reliability threshold claims from figure notes (eliminated `0.70`, "acceptable internal consistency", and universal qualitative ratings for Cronbach's alpha and ICC).
    -   Made correlation reference-line notes method-accurate (explicitly referencing the null value for Pearson r, Spearman rho, Kendall tau-b, point-biserial, and partial Pearson).
    -   Implemented true multi-figure rendering infrastructure (`build_figure_specs`, `figure_specs` in `HtmlRenderer` and `ResearchReportHtmlRenderer`) with unique chart IDs and verified single Plotly bundle embedding.
    -   Avoided unnecessary Plotly requirement in `to_interactive_html`: checks if any figure can actually be built before checking Plotly availability, allowing table-only methods (e.g., Kruskal-Wallis) and direct `PresentationView` targets to render without Plotly installed.
    -   Strengthened figure numeric fidelity, privacy sentinels (extended across OLS, logistic, factorial), and XSS escaping protections.

### Changed

-   Hardened `ResearchReportHtmlRenderer` presentation adapter fallback:
    -   Narrowed broad exception handling to catch only `UnsupportedPresentationError`, ensuring unexpected programming or adapter failures surface clearly instead of silently downgrading to canonical payload-only presentation.
-   Final Rich terminal presentation fidelity pass (`pyautostat.show` / `pyautostat.presentation`):
    -   Implemented dynamic confidence-level labels across all presentation adapters via a centralized helper (`format_confidence_level_label`), resolving stored interval level and specification fallback (`confidence_level`) without assuming 95% or recalculating intervals (e.g., rendering "90% CI", "99% CI", "90% Simultaneous CI", "R-squared 90% CI", "Wald CI", or neutral "CI").
    -   Corrected Kruskal-Wallis group-summary schema mapping to render `Group | N | Median` using authoritative `sample_size` and stored `median` without fabricating an `IQR` column or displaying `Unavailable`.
    -   Preserved falsey categorical and group labels (including integer `0`) across pairwise comparisons and group tables using explicit non-None evaluation (`first_present`), preventing valid numeric zero categories from collapsing into missing defaults or invalid contrast strings.
-   Final presentation-fidelity cleanup in Rich terminal presentation adapters (`pyautostat.show` / `pyautostat.presentation`):
    -   Aligned `AuditResult` and `AuditFinding` field mappings to authoritative schemas: mapped top-level audit status and internal consistency diagnostic consistently to `PASSED` (`success`), `INCOMPLETE` (`warning`), and `FAILED` (`error`); aligned finding severity counting to real vocabulary (`error`, `warning`, `pass`); and surfaced stored `explanation` rather than nonexistent `message`.
    -   Aligned Cronbach's alpha presentation with authoritative reliability result schema: retrieved sample accounting from `metadata["sample"]`, supported list-of-dicts reversed-item configuration, added inter-item alignment diagnostic for negative correlations, and added item-level missingness table under `detail="full"`.
    -   Corrected Kruskal-Wallis equal-variance diagnostic to render `Not applicable` with neutral severity rather than ANOVA-style `Assumed`.
    -   Updated practical-significance threshold wording to respect stored `planning_status` (`planned`, `exploratory`, `unknown`), avoiding unverified a priori or prespecification claims.
    -   Removed blanket sensitivity-analysis comparability assertions, adopting neutral subtitles (`Specification sensitivity analysis for <primary_method>`) and diagnostic descriptions evaluated directly from stored records.
    -   Eliminated unsafe numeric `or` fallbacks across presentation adapters using explicit non-None fallback helpers (`first_present`), preserving legitimate `0.0` test statistics and counts.
    -   Removed presentation-layer chi-square expected-count threshold inference, strictly consuming backend `expected_count_status` with a neutral fallback.
-   Hardened Rich terminal presentation layer correctness and stored-result fidelity (`pyautostat.show` / `pyautostat.presentation`):
    -   Corrected logistic regression coefficient mapping to consume stored `odds_ratio` and `odds_ratio_ci` directly without falling back to log-odds intervals or exponentiating coefficients, and properly mapped `estimate`, `standard_error`, `statistic`, and `p_value`.
    -   Corrected OLS regression coefficient table mapping to display `estimate`, `standard_error`, `confidence_interval`, `statistic`, and `p_value` directly from stored records.
    -   Refactored OLS diagnostics to consume stored Breusch-Pagan status and VIF maximum/advisories without inventing presentation-layer alpha thresholds or rule-of-thumb cutoffs.
    -   Corrected repeated-measures ANOVA and Friedman follow-up tables to use stored condition names, contrast orientation, test statistics, effect sizes, adjusted p-values, and stored decisions without hardcoded alpha thresholds.
    -   Enforced stored decision consumption across multigroup and repeated-measures pairwise comparisons, eliminating presentation-layer `0.05` threshold evaluations.
    -   Implemented method-explicit multigroup follow-up branching ensuring Kruskal-Wallis (Dunn-Holm) follow-ups are never mislabeled as mean differences.
    -   Mapped `PracticalSignificanceResult.statistical_significance` as a semantic tri-state string (`evidence_against_null`, `no_evidence_against_null`, `unavailable`) rather than evaluating boolean truthiness.
    -   Consumed authoritative comparability and contrast orientation records in sensitivity analysis without unconditionally asserting preserved contrasts.
    -   Mapped two-way ANOVA interaction headline metric to authoritative `f_statistic`.
    -   Removed derived percentage-of-total calculations in ICC presentation to preserve raw unconstrained variance component estimates (including negative estimates).
    -   Corrected audit outcome status and role mapping (`passed` -> `PASSED`/`success`, `incomplete` -> `INCOMPLETE`/`warning`, `failed` -> `FAILED`/`error`).
    -   Mapped statistical analysis plan practical thresholds using stored `minimum_magnitude` instead of dataclass string representations.
    -   Cleaned two-group mean and median summary tables to avoid advertising unavailable descriptive statistics when not stored in the authoritative result.
    -   Surfaced stored Cronbach's alpha missingness, reverse-scoring configuration, and negative correlation diagnostics.
    -   Added comprehensive zero-recalculation fidelity and non-default alpha test suites.
-   Optimized participant-block bootstrap for Kendall's W rank concordance (`friedman_kendall_w_bootstrap_ci`) by precomputing within-unit ranks and unit tie penalties once across units, yielding mathematically identical replicate distributions in $O(B \times k)$ time.
-   Standardized paired Wilcoxon signed-rank asymptotic approximation policy (`paired_wilcoxon` and `friedman_test` pairwise follow-ups), selecting asymptotic approximation when non-zero paired differences exceed 50 to maintain consistent large-sample behavior and performance across SciPy versions.
-   Refined the minimum valid resample threshold for paired rank-biserial and Kendall's W bootstrap confidence intervals to `max(10, bootstrap_samples // 2)`, supporting low-resample fast test/CI runs ($B < 20$) with an empirical floor of 10 while maintaining the 50% validity requirement for standard runs ($B \ge 20$).

### Examples & Documentation

-   Remediated the public CustomerDataset.csv analytics showcase (examples 01 through 09):
    -   Preserved signed contrast orientation in Example 02 (`No` minus `Yes` = -$47.70, 95% CI [-$56.86, -$38.55]) without using `abs()` inversion.
    -   Replaced hard-coded inferential results throughout examples with dynamic extraction from structured PyAutoStat result records and verified cleaned data.
    -   Softened two-way ANOVA interaction interpretation in Example 08 to distinguish absence of statistically detectable interaction from proof of additivity.
    -   Clarified HC3 robust covariance claims in Example 05, emphasizing reduced reliance on equal-variance assumptions rather than universal validity, and noted "spend drivers" is business shorthand rather than causal.
    -   Refined target-leakage documentation in Example 06 to describe observed perfect separation by spend rather than unverified historical generation provenance.
    -   Added executable test enforcement of model predictor exclusions (leakage invariants) for linear and logistic regression examples.
    -   Dynamically calculated Product B and Product C zero-inflation percentages in Example 07.
    -   Standardized fast execution mode (`--fast`) with visible console disclosures and automated integration test coverage in `examples/run_all.py`.
    -   Isolated example artifact generation so that test execution does not dirty the working directory or write to untracked report locations.

## [0.5.0] - 2026-10-01

This release completes PyAutoStat's scientific-depth hardening and
expands the package with effect-size uncertainty, independent two-way
factorial ANOVA, and a design-aware Intraclass Correlation Coefficient
(ICC) reliability framework. It also completes the associated scientific
closure work across method contracts, validation, audit,
reproducibility, documentation, packaging, and compatibility. PyAutoStat
remains an alpha package; publishing or tagging a release remains a
separate owner decision.

### Added

-   Added Intraclass Correlation Coefficient (ICC) for quantitative
    rater reliability and agreement:

    -   Supports the six canonical Shrout & Fleiss / McGraw & Wong
        forms: `ICC(1,1)`, `ICC(1,k)`, `ICC(2,1)`, `ICC(2,k)`,
        `ICC(3,1)`, and `ICC(3,k)`, together with implemented
        McGraw-Wong aliases/configurations.
    -   Makes the rater model, absolute-agreement versus consistency
        definition, and single- versus average-measure estimand explicit
        instead of reporting an unlabeled generic ICC.
    -   Validates fully crossed target-by-rater designs with at least
        two targets and two raters, rejects duplicate target-rater
        cells, and applies transparent complete-target panel filtering
        for incomplete panels.
    -   Computes one-way and two-way ANOVA mean-square decompositions
        and method-of-moments target, rater, and residual variance
        components.
    -   Preserves negative sample ICC estimates and unconstrained
        method-of-moments variance-component estimates rather than
        silently clamping them to zero.
    -   Provides analytical F-inversion confidence intervals, including
        the dedicated Satterthwaite effective-degrees-of-freedom
        treatment for two-way random absolute-agreement ICCs.
    -   Provides applicable F tests, sample accounting, diagnostics,
        deterministic interpretation, reporting, semantic audit,
        serialization, and reproducibility replay.
    -   Added `docs/ICC_GUIDE.md` covering model selection, notation
        mapping, formulas, agreement versus consistency, single versus
        average measures, negative ICC behavior, missingness policy, and
        limitations.
    -   Added dedicated ICC reporting for summaries, ANOVA components,
        variance components, and variant results.

-   Added independent two-way factorial ANOVA using the full
    `A + B + A×B` model:

    -   Supports balanced and unbalanced fully crossed independent
        designs.
    -   Supports explicit Type II and Type III sums-of-squares policies.
    -   Uses sum-to-zero/deviation contrast coding for Type III
        inference.
    -   Reports factor A, factor B, and A×B interaction effects
        separately.
    -   Reports partial eta-squared for factorial terms with
        noncentral-F confidence intervals.
    -   Adds cell summaries and unweighted estimated marginal means.
    -   Adds planned simple effects, marginal comparisons, and 2×2
        difference-of-differences interaction contrasts with Holm
        multiplicity adjustment.
    -   Adds residual diagnostics and explicit empty-cell/design
        validation.
    -   Integrates factorial ANOVA with guided and direct APIs,
        deterministic interpretation, reporting, semantic audit,
        serialization, and reproducibility replay.

-   Added effect-size confidence intervals and uncertainty hardening
    across existing methods:

    -   Exact noncentral-t inversion confidence intervals for one-sample
        Cohen's d and paired Cohen's dz.
    -   Fisher-z asymptotic confidence intervals for Pearson
        correlation.
    -   Estimator-matched log-Wald confidence intervals for Fisher exact
        sample odds ratios.
    -   Deterministic participant/pair-level percentile-bootstrap
        confidence intervals for matched-pairs rank-biserial effects and
        Friedman follow-up effects.
    -   Participant-block bootstrap confidence intervals for Friedman
        Kendall's W.
    -   Exact noncentral-F confidence intervals for repeated-measures
        ANOVA partial eta-squared.
    -   Exact noncentral-t confidence intervals for repeated-measures
        paired Cohen's dz contrasts.
    -   Within-group bootstrap confidence intervals for
        Kruskal-Wallis/Dunn pairwise rank-biserial effects.
    -   Case-resampling bootstrap confidence intervals for OLS in-sample
        R-squared.
    -   Uses structured unavailable statuses when an interval is not
        scientifically or numerically defensible rather than
        manufacturing a value.

-   Added `src/pyautostat/uncertainty.py` with reusable deterministic
    root-finding and bootstrap helpers without increasing the declared
    numerical dependency floors.

-   Added a machine-checkable scientific method-contract architecture
    using `MethodContract` and `METHOD_CONTRACTS`, recording explicit
    estimands, hypotheses, estimates, effect quantities, uncertainty
    status, assumptions, diagnostics, missing-data policy,
    degenerate-data behavior, multiplicity policy, numerical provenance,
    interpretation limitations, audit invariants, and validation
    sources.

-   Added scientific-contract and uncertainty documentation:

    -   `docs/STATISTICAL_METHOD_CONTRACTS.md`
    -   `docs/EFFECT_SIZE_CI_GAPS.md`

-   Added final scientific-closure regression coverage spanning
    representative workflow families, including method execution,
    uncertainty, semantic audit, JSON-safe serialization, reporting, and
    reproducibility replay.

-   Extended installed-wheel and minimum-numerical-stack smoke coverage
    for the expanded statistical capability set.

### Changed

-   Hardened scientific narration and result metadata:
    -   Corrected Friedman non-rejection wording to avoid universal
        equality-of-medians claims.
    -   Clarified that Kendall's W represents within-unit rank
        concordance rather than percentage of variance explained.
    -   Canonicalized Mauchly sphericity status handling.
    -   Distinguished Fisher exact's inferential null hypothesis
        (`odds ratio = 1`) from the sample cross-product odds-ratio
        effect estimate.
    -   Clarified that logistic regression uses analytical Wald z
        intervals for coefficients and exponentiated Wald intervals for
        odds ratios rather than bootstrap intervals.
    -   Retained the scientific policy of not reporting
        observed/post-hoc power; prospective study planning remains
        supported.
-   Expanded semantic audit invariants and corruption-test coverage
    across existing and newly added methods, including:
    -   p-value bounds;
    -   confidence-interval ordering;
    -   mathematical bounds for bounded effect quantities;
    -   sample-size and contrast-order accounting;
    -   degrees-of-freedom identities;
    -   repeated-measures ANOVA identities;
    -   multiplicity consistency;
    -   logistic-regression coefficient/odds-ratio consistency;
    -   factorial-ANOVA identities;
    -   ICC model, notation, formula, ANOVA-component, and interval
        consistency.
-   Reconciled the roadmap with implemented capabilities:
    -   Two-way factorial ANOVA and ICC are described as current
        capabilities rather than future candidate work.
    -   Future statistical methods remain intentionally unscheduled and
        subject to complete scientific contracts and independent
        numerical validation.
    -   No additional statistical method was introduced solely to extend
        the roadmap.
-   Strengthened scientific-closure and release-readiness checks around
    scientific consistency, serialization, reproducibility, installed
    artifacts, minimum dependency compatibility, and end-to-end workflow
    execution.

### Fixed

-   Corrected `docs/ICC_GUIDE.md` to match the implementation's
    unconstrained method-of-moments variance-component calculations.
    Negative finite-sample target/rater component estimates are no
    longer documented as if they were truncated using `max(0, ...)`.

-   Added regression coverage confirming that negative ICC
    variance-component estimates are preserved rather than silently
    clamped.

-   Reconciled scientific documentation, method contracts, roadmap
    language, and implementation behavior identified during the
    scientific-closure audits.

### Validation and packaging

-   Maintains automated testing across Python 3.10, 3.11, 3.12, and 3.13
    on Ubuntu and Windows.
-   Maintains Ruff linting and formatting checks and mypy type checking.
-   Builds source and wheel distributions and validates distribution
    metadata.
-   Exercises an isolated installed-wheel smoke path rather than relying
    only on source-tree imports.
-   Exercises a minimum-supported numerical-stack compatibility path.
-   Keeps statistical execution local and deterministic where the method
    itself is deterministic.
-   Preserves PyAutoStat's bounded scientific scope: no mixed models,
    GEE, survival analysis, causal-inference framework, arbitrary
    incomplete-panel longitudinal modeling, or automatic
    statistical-model selection was added in this release.

## [0.4.0] - 2026-09-30

This release expands PyAutoStat with new descriptive, inferential,
multi-group, regression, reliability, binary/association, and
repeated-measures workflows while preserving deterministic method
selection, explicit study-design and estimand contracts, reproducible
reporting, and scientific guardrails. PyAutoStat remains an Alpha
release.

### Added

-   Added complete repeated-measures analysis for 3+ conditions on the
    same observational units: one-way repeated-measures ANOVA for
    continuous mean outcomes with full ANOVA tables, Mauchly's
    sphericity test, Greenhouse-Geisser epsilon and corrected degrees of
    freedom / p-values when sphericity is violated, partial eta-squared
    repeated-measures effect sizes, and complete pairwise paired t-test
    follow-up with per-contrast analytical confidence intervals and Holm
    multiplicity adjustment; and the Friedman rank-sum test for repeated
    rank/distribution targets with Kendall's W effect size and complete
    pairwise Wilcoxon signed-rank follow-up with matched-pairs
    rank-biserial correlations and Holm multiplicity adjustment.

-   Added five complete binary-outcome and extended-association
    workflows: binary logistic regression, exact unit-ID McNemar
    inference, point-biserial correlation, explicit inferential Kendall
    tau-b, and partial Pearson correlation for declared quantitative
    controls.

-   Added a complete researcher-declared survey and scale reliability
    workflow centered on Cronbach's alpha, with complete-case and
    per-item missingness accounting, deterministic respondent-row
    bootstrap intervals, corrected item-total correlations,
    alpha-if-deleted, inter-item diagnostics, optional explicit bounded
    reverse scoring, qualified non-inferential interpretation, dedicated
    report tables, audit, replay, sessions, examples, and
    installed-wheel coverage.

-   Added complete simple and multiple ordinary least-squares
    conditional-mean regression for continuous outcomes, including
    classical or explicit HC3 covariance inference, diagnostics,
    deterministic interpretation, canonical reports, audit, and replay.

-   Added guided Welch one-way ANOVA with Games-Howell comparisons,
    classical ANOVA with Tukey-Kramer comparisons, and Kruskal-Wallis
    with Dunn-Holm comparisons.

-   Added one-sample t inference, explicit unit-ID paired Wilcoxon
    signed-rank inference, inferential Spearman correlation, two-sided
    Fisher exact inference, percentile profiles, categorical frequency
    tables/cross-tabs, and safeguarded coefficient-of-variation
    metadata.

### Fixed

-   Corrected Mauchly sphericity p-value using the higher-order
    Box/Anderson asymptotic chi-square approximation.
-   Strengthened repeated-measures audit and validation checks.
-   Reconciled repeated-measures documentation and edge-case behavior.

## [0.3.0] - 2026-09-27

This release adds a deterministic, researcher-readable narration layer
across profiling, recommendation, interpretation, practical-significance
assessment, sensitivity analysis, and reporting. Structured statistical
records remain authoritative, and all narration remains local,
rule-based, reproducible, and independent of generative AI or external
services.

### Added

-   Added deterministic effect-size narratives, four-quadrant hypothesis
    explanations, graded assumption messages, practical-significance
    verdicts, dataset story mode, recommendation explanations, workflow
    explanations, and escaped executive summaries.
-   Added public end-to-end, determinism, boundary, HTML safety,
    serialization, example-execution, and installed-wheel validation
    coverage.

### Changed

-   Expanded researcher-facing prose while preserving numerical results
    and structured records.
-   Improved beginner-facing labels, warnings, report styling, examples,
    and documentation.

### Fixed

-   Hardened completeness, sensitivity, practical-significance,
    diagnostics, serialization, recommendation, report, and narration
    edge cases.

### Security

-   Continued escaping untrusted HTML and LaTeX text, protecting
    formula-like CSV cells, omitting raw DataFrames and participant
    identifiers from reports, and keeping narration offline.

## [0.2.0] - 2026-09-25

This release expands PyAutoStat from its initial analysis utilities into
an explainable and reproducible research-analysis assistant.

### Added

-   Added `ResearchAssistant` guided workflows, structured research
    questions, deterministic method recommendation, structured results,
    canonical reports, paired analysis, prospective planning, analysis
    plans, sensitivity analysis, practical-significance thresholds,
    audit/reproducibility, richer profiling, resource metadata, and
    session snapshots.

### Fixed

-   Hardened estimand preservation, Welch defaults, contrast
    orientation, numerical edge cases, diagnostics, bootstrap
    interpretation, paired-design safeguards, sensitivity identity,
    practical-significance directionality, and report security.

### Changed

-   Organized documentation around current capabilities, established
    `ROADMAP.md` as the forward-looking roadmap, required Python 3.10+,
    and strengthened quality/packaging automation.

### Quality and packaging

-   Added Ruff, mypy, coverage enforcement, build/Twine checks, GitHub
    Actions across Python 3.10-3.13 on Linux and Windows, isolated
    installed-wheel smoke testing, and a minimum-stack route.

## [0.1.0] - 2026-09-21

-   Initial statistical analysis, insight, and report-export package.
