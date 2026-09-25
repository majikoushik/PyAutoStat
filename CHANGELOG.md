# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

The established analysis, planning, and reporting implementation is checked into `main`, and its
GitHub CI matrix passed. Packaging a new release remains a separate owner decision.

### Added

- Added structured profiling resource metadata based on deep pandas memory usage, with advisory
  large-memory and wide-correlation warnings. The diagnostics explicitly record that no sampling,
  truncation, or source-data modification occurred.
- Added schema-versioned statistical analysis plans and adherence comparison, bounded prospective
  independent and paired mean planning, explicit unit-ID paired mean analysis, reporting
  completeness, General/APA-oriented/IEEE-oriented presentation, safe LaTeX text, styled auditing,
  and UI-independent session snapshots. Earlier specification schemas remain readable.
- Added explicit `ResearchAssistant.sensitivity_analysis()` with ordered scenario retention,
  estimand-aware comparison, paired identity and contrast protection, group-orientation handling,
  local provenance, and no p-value ranking or robustness score.
- Added researcher-defined `MeaningfulEffectThreshold` and
  `ResearchAssistant.practical_significance()` with named metrics, direction, unit, paired contrast
  orientation, point-estimate relation, separate interval relation, partial status when uncertainty
  is unavailable, and explicit unsupported equivalence/noninferiority requests.
- Added optional sensitivity and practical-significance report sections, audit checks,
  reproducibility configuration metadata, and a runnable sensitivity example. Base-only reports
  and reproducibility records retain their established schemas.
- Added `ResearchAssistant.run()` and JSON-safe `ResearchWorkflowResult`, connecting question
  intake, design blocking, recommendation, one statistical execution, deterministic
  interpretation, canonical reporting, default consistency audit, and reproducibility metadata.
- Added controlled method and failure-mode matrices, researcher responsibilities, and runnable
  profile, guided-analysis, and continuation examples.
- Added optional `DecisionLedger`, content and dataset fingerprints, read-only
  `StatisticalResultAuditor`, runtime `ReproducibilityRecord`, metadata-only ZIP packaging, and
  explicit supplied-data replay. Raw DataFrames are not included by default.
- Added `ResearchAssistant.report(result)` and canonical `ResearchReport` with Methods, Results,
  sample accounting, diagnostics, interpretation, limitations, optional histogram specifications,
  and deterministic HTML, Markdown, JSON, and CSV table exports.
- Added deterministic `ResearchAssistant.interpret(result)` with structured results and coded
  findings. Interpretation reads recorded values without recalculating or changing them.
- Added `ResearchAssistant.analyze()` with fresh question validation and dispatch through existing
  numerical backends. `AnalysisResult` ties statistics, estimates, intervals, sample accounting,
  group direction, diagnostics, and warnings to the specification and recommendation.
- Added deterministic `ResearchAssistant.recommend_test()` with a bounded method registry, design
  checks, data-feasibility rules, ordered decision traces, alternatives, assumptions, and
  clarification records without executing a test.
- Added research-question intake with `prepare_question()` and `update_question()`, immutable
  `QuestionDraft`, structured clarification questions, selected-variable availability, and dataset
  metadata reuse.
- Added dataset intelligence to both profiling entry points: categorical frequencies and tied
  modes, analytical type and role evidence, optional data dictionaries, row and pattern
  missingness, duplicate overlap, structured quality findings, pairwise correlation sample sizes,
  and explicit outlier and distribution metadata.
- Added `ResearchAssistant.complete_case_count(columns)` for transparent selected-column row
  availability without applying a missing-data treatment.
- Added `ResearchAssistant(df).profile()` as the high-level profiling facade and typed,
  JSON-compatible research configuration and result contracts.
- Added package exceptions with actionable errors for invalid data, unknown columns, test choices,
  and insufficient groups.
- Added effect sizes and confidence intervals for supported group comparisons, categorical
  association with Cramér's V and optional Cohen's h, and deterministic percentile bootstrap
  metadata.
- Added advisory column-type and role detection, optional Plotly interactive reports, collapsible
  report sections, and report tables with filtering and sorting.

### Fixed

- Directional paired practical-significance thresholds now require a matching first-minus-second
  contrast; two-sided magnitude thresholds remain orientation invariant.
- Paired sensitivity comparisons now include unit ID, condition variable, design, estimand, and
  contrast orientation in scientific identity.
- Declared missing codes in paired unit-ID columns now block pairing until callers normalize the
  source explicitly.
- Plan adherence can compare planned and performed sensitivity scenarios and meaningful-effect
  thresholds without making conduct judgments or inventing reasons.
- Percentile-bootstrap interpretation accepts valid intervals outside the original point estimate,
  preserves a valid raw mean difference when a standardized effect is unavailable, and flags
  conflicting signed effects.
- Finite D'Agostino-Pearson results are retained under SciPy's advisory small-sample warning, while
  unreliable warnings and nonfinite results remain unavailable. Anderson-Darling is omitted when
  SciPy supplies an invalid critical-value grid.
- Automatic independent-group selection requires a stated estimand and no longer switches a mean
  target to a rank target based on diagnostic p-values. Welch is the default two-group mean method;
  automatic multi-group mean comparison remains unsupported.
- Normality and variance diagnostics distinguish rejected, not rejected, and unknown states.
- Mann-Whitney explicitly requests a two-sided p-value and keeps first-group direction consistent.
- Undefined outlier counts, correlations, effects, intervals, and descriptive quantities remain
  unavailable rather than being reported as zero or valid results.
- Extreme finite values no longer abort histogram peak analysis; unreliable extreme-scale results
  are blocked or accompanied by warnings.
- Reports escape untrusted HTML, emit strict JSON, protect formula-like CSV cells, and wrap file
  errors in `ReportError`. Global warning suppression was removed.

### Changed

- Renamed the package from `autostat` to `pyautostat`; `AutoStatError` became `PyAutoStatError`.
- Dropped Python 3.9 support; the package requires Python 3.10 or newer.
- Rank-biserial correlation follows first-versus-second group direction, and negative sample
  epsilon-squared estimates are truncated at zero.
- Consolidated active documentation into a beginner README, detailed API reference, enduring
  product vision, single future roadmap, contributor guidance, changelog, and focused technical
  documents. Obsolete numbered-development planning documents were removed.
- Renamed examples and test modules with permanent capability-oriented names, added a focused
  documentation index, and consolidated overlapping workflow and support matrices into one
  capabilities document.

### Chore

- Added Ruff, formatting, mypy, coverage, build, Twine, installed-wheel smoke, and a GitHub Actions
  matrix across supported Python versions and Windows/Linux.
- Kept generated reports out of version control and made example file output explicit.

## [0.1.0] - 2026-09-21

- Initial statistical analysis, insight, and report-export package.
