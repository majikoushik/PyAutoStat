# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Phase 12 local implementation pending owner review: schema-versioned analysis plans and
  adherence comparison; standalone bounded Welch/paired prospective planning; an explicit-unit-ID
  paired two-condition t workflow; reporting completeness without a quality score; General,
  APA-oriented, and IEEE-oriented presentation; safe LaTeX text; styled auditing; a versioned
  UI-independent session snapshot; tests, examples, and readiness documentation. Specification
  schemas 1 and 2 remain readable and paired fields use additive schema 3. No dependency or
  package-version change was made; this is not a release or CI claim.

- Phase 11 explicit `ResearchAssistant.sensitivity_analysis()` with ordered, JSON-safe scenario specifications and results; same-estimand estimate/CI comparison, different-estimand labels, group-orientation normalization, complete attempt retention, local provenance, and no p-value ranking or robustness score.
- Phase 11 researcher-defined `MeaningfulEffectThreshold` and `ResearchAssistant.practical_significance()` with named metric, direction, unit, point-estimate relation, separate confidence-interval relation, partial status when uncertainty is unavailable, and explicit unsupported equivalence/noninferiority requests.
- Optional report schema 2 sensitivity/practical-significance sections, Phase 11 audit checks, reproducibility schema 2 configuration metadata, and a runnable sensitivity example. Base Phase 10 reports and reproducibility records remain schema version 1 when Phase 11 content is absent.
- Phase 10 `ResearchAssistant.run()` and JSON-safe `ResearchWorkflowResult` integrate question intake, design blocking, recommendation, one statistical execution, deterministic interpretation, canonical reporting, default consistency audit, and reproducibility metadata. Missing information, observed data limits, unsupported requests, partial findings, and audit failures retain distinct workflow statuses; no file export or replay occurs automatically.
- A controlled-MVP method matrix, failure-mode matrix, researcher responsibilities, and runnable profile/guided/continuation example.
- Phase 9 optional observed-event `DecisionLedger`, content and dataset fingerprints, read-only `StatisticalResultAuditor` with actual export checks, runtime `ReproducibilityRecord`, metadata-only ZIP package, and explicit same-data replay through the existing Phase 6 engine. No raw DataFrame is included in records or packages by default.
- Phase 8 `ResearchAssistant.report(result)` and canonical `ResearchReport` with structured Methods, Results, sample accounting, diagnostics, interpretation, limitations, optional histogram bin specifications, and deterministic HTML, Markdown, JSON, and CSV table exports. The legacy `ReportGenerator` API is retained.
- Phase 7 deterministic `ResearchAssistant.interpret(result)`, with structured `InterpretationResult` and coded findings for executable Phase 6 methods. It reads recorded numbers, qualification flags, and intervals without recalculating or changing the original result.

- Phase 6 `ResearchAssistant.analyze()` revalidates a question and executes ready recommended methods through existing numerical backends. Additive `AnalysisResult` links the specification and recommendation to exact statistics, effect measures, interval quantities, sample accounting, group order, diagnostics and warnings; known failures remain unavailable.
- Phase 5 deterministic `ResearchAssistant.recommend_test()` with a small method capability registry, design checks, data-feasibility rules, structured decision traces, alternatives, assumptions and GUI-ready clarification records. It recommends existing calculations without executing them or changing the declared target; unimplemented inference stays unavailable.
- Phase 4 research question intake with `ResearchAssistant.prepare_question()` and `update_question()`, a typed `QuestionDraft`, structured clarification questions, selected-variable availability counts, and Phase 3 metadata reuse. The builder records objectives, roles, targets, and design without selecting or executing a method.
- Analysis specification schema version 2 for optional Phase 3 `data_dictionary` declarations; version 1 text-only `variable_metadata` and its round trips remain supported.

- Phase 3 dataset intelligence through both profiling entry points: categorical frequencies and tied modes, analytical type/role evidence, optional validated data dictionary, row and pattern missingness, duplicate overlap, structured quality findings, per-pair correlation sample sizes, and explicit outlier/distribution metadata. Optional row positions and histogram bin count support future visual interfaces without changing source data.
- `ResearchAssistant.complete_case_count(columns)` reports available and excluded rows for specified columns without applying a missing-data treatment.
- Phase 1 architecture: `ResearchAssistant(df).profile()` delegates to existing profiling, and typed research configuration and future result contracts serialize to JSON-compatible dictionaries. See `docs/ARCHITECTURE.md`.

- Initial packaging as `pyautostat` (src layout, pyproject.toml, tests, CI).
- `PyAutoStatError` and friendly, specific error messages for invalid data,
  unknown columns, unknown `test_type`, and too few groups in `hypothesis_tests()`.
- Effect sizes and confidence intervals in `hypothesis_tests()`: Cohen's d + CI
  for t-tests, rank-biserial correlation for Mann-Whitney, eta-squared for
  ANOVA, epsilon-squared for Kruskal-Wallis, each with a small/medium/large
  interpretation.
- `pyautostat.detect_column_types()` and `pyautostat.suggest_column_roles()`:
  heuristics for numeric-but-categorical columns, date-like text, and
  name-based role suggestions (identifier/target/datetime/economic). Both
  are included automatically in `analyze_all()`.
- `ReportGenerator.to_interactive_html()`: Plotly-based report with a
  hoverable correlation heatmap and zoomable per-column histograms.
  Requires the `report` extra (`pip install pyautostat[report]`).
- Phase 1: contact-format and missingness hints in `detect_column_types()`;
  measurement and action hints in `suggest_column_roles()`.
- Detailed per-group normality and Levene assumption statuses, selection
  reasons, and configurable deterministic percentile bootstrap intervals for
  hypothesis effect sizes. Reports can include hypothesis results in JSON,
  HTML, interactive HTML, and CSV.
- Categorical chi-square association with Cramér's V, expected-count checks,
  and optional Cohen's h for a named success outcome in a 2x2 table; both
  effect sizes support bootstrap intervals.
- Collapsible interactive report sections, table filtering and sorting, and
  chart rendering when a section is opened.

### Fixed

- Treat Z-score outlier counts as unavailable when variation is below a reliable floating-point scale relative to a large offset, matching the existing near-constant correlation safeguard instead of reporting a misleading zero count.
- Phase 7 interpretation accepts valid percentile-bootstrap intervals outside the original point estimate, preserves raw mean-difference interpretation when Cohen's d is unavailable, and flags conflicting mean-difference/d directions.
- Phase 2 CI correction: retain finite D'Agostino-Pearson results under SciPy's advisory small-sample kurtosis warning, record approximation limits, and reject unreliable warnings or nonfinite results. Omit Anderson-Darling when its critical-value grid is invalid, including negative small-sample thresholds, with an explanatory warning.
- Phase 2: automatic group-test selection requires a stated estimand and no longer switches from mean to rank methods after normality or variance screens. Welch is the default two-group mean method; automatic multi-group mean comparison reports unsupported.
- Normality and Levene diagnostics now distinguish rejected, not rejected, and unknown states; reports and insights no longer call a nonrejection proof of normality.
- Explicitly request two-sided Mann-Whitney p-values and calculate first-group U consistently across supported SciPy versions. Undefined Z-score/MAD counts and all-missing outlier counts are unavailable rather than zero.
- Report bootstrap valid/requested counts and seed, including unavailable intervals; add sample and excluded-row counts to group and categorical results.

- `hypothesis_tests()` no longer treats a missing-value row as its own group.
- Validate DataFrame labels, scalar values, finite real numeric input, hypothesis
  test compatibility, usable group sizes, and constant or undefined test data.
- Skip normality tests that lack the required sample size or variation; preserve
  finite/undefined results explicitly and report reasons in `analysis_warnings`.
- Use pairwise observations for correlation p-values and represent undefined
  coefficients, outlier metrics, and descriptive statistics as `None`.
- Keep extreme finite values from aborting histogram peak analysis; flag
  numerical underflow and reject undefined effect sizes or confidence intervals.
- Escape untrusted HTML report text, emit standards-compliant JSON, protect
  written CSV files from spreadsheet formulas, and wrap file errors in
  `ReportError`. Removed global warning suppression.

### Changed

- Renamed the package from `autostat` to `pyautostat` everywhere (import
  path, PyPI distribution name, docs). `AutoStatError` is now `PyAutoStatError`.
- Dropped Python 3.9 support (EOL); the package now requires Python >=3.10.
- Earlier alpha behavior (superseded by Phase 2): three-or-more-group `auto`
  comparisons chose ANOVA after normality and variance screens, or Kruskal-Wallis
  otherwise.
- Rank-biserial correlation now follows the first-versus-second group direction;
  negative sample epsilon-squared estimates are truncated at zero.

### Chore

- Prepared the README for PyPI and removed generated report files from version
  control; the example script recreates them on demand.
- Consolidated repository documentation into a focused README, API reference,
  roadmap, changelog, contributor guide, and examples guide; removed redundant
  and unverified marketing documents.
- Reworked the runnable example to cover all public analysis and export
  features, optional Plotly output, and representative bad-data errors; added
  an examples guide and CLI smoke tests.
- Added `ruff check`, `ruff format`, and `mypy` (clean on `src/`) with a
  pre-commit config, and a GitHub Actions CI workflow that lints,
  type-checks, runs the test suite on Python 3.10-3.13 across Linux and
  Windows, and builds/validates the sdist and wheel with `twine check`.

## [0.1.0] - 2026-09-21

- Statistical analysis (descriptive stats, normality, outliers, correlation, hypothesis testing).
- Automated insights engine.
- Report export to JSON, HTML, CSV, and dict.
