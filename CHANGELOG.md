# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Added complete repeated-measures analysis for 3+ conditions on the same observational units:
  one-way repeated-measures ANOVA for continuous mean outcomes with full ANOVA tables, Mauchly's
  sphericity test, Greenhouse-Geisser epsilon and corrected degrees of freedom / p-values when
  sphericity is violated, partial eta-squared repeated-measures effect sizes, and complete pairwise
  paired t-test follow-up with per-contrast analytical confidence intervals and Holm multiplicity
  adjustment; and the Friedman rank-sum test for repeated rank/distribution targets with Kendall's W
  effect size and complete pairwise Wilcoxon signed-rank follow-up with matched-pairs rank-biserial
  correlations and Holm multiplicity adjustment. Both workflows require explicit unit identity,
  long-form panels, and 3+ ordered condition labels, enforce complete-case panels while auditing
  and reporting missing and excluded unit accounting, block duplicate unit-condition records,
  preserve deterministic contrast orientations, and integrate with explainable narration, canonical
  HTML/Markdown/JSON/CSV reports, reporting completeness, result auditing, replay reproducibility,
  analysis plans, session snapshots, examples, and installed-wheel smoke testing. Two-condition paired
  workflows remain completely unchanged. No mixed-effects models, GEE, mixed ANOVA, factorial
  repeated measures, or automatic imputation were added.

- Added five complete binary-outcome and extended-association workflows: binary logistic
  regression, exact unit-ID McNemar inference, point-biserial correlation, explicit inferential
  Kendall tau-b, and partial Pearson correlation for declared quantitative controls. The methods
  preserve event/positive-level and condition orientation, use paired-unit or complete-row
  bootstrap uncertainty where appropriate, integrate with deterministic interpretation,
  canonical reports and tables, semantic audit, analysis plans, replay, session snapshots, and a
  synthetic example, and explicitly block separation, duplicate pairs, rank-deficient adjustment,
  and ambiguous binary coding. No classification metrics, automatic thresholds, causal claims,
  control selection, or additional generalized models were added.

- Added a complete researcher-declared survey and scale reliability workflow centered on
  Cronbach's alpha, with complete-case and per-item missingness accounting, deterministic
  respondent-row bootstrap intervals, corrected item-total correlations, alpha-if-deleted,
  inter-item diagnostics, optional explicit bounded reverse scoring, qualified non-inferential
  interpretation, dedicated report tables, audit, replay, sessions, examples, and installed-wheel
  coverage. The workflow never discovers scales, deletes items, reverse-scores automatically,
  creates composite columns, or treats alpha as validity or a universal pass/fail threshold.

- Added complete simple and multiple ordinary least-squares conditional-mean regression for
  continuous outcomes, with ordered predictor lists, explicit treatment coding and reference
  levels for Boolean/nominal/ordinal predictors, complete-case accounting, rank validation,
  classical or explicit HC3 covariance inference, coefficient and model-fit records, continuous
  standardized betas, VIF, variance/residual/influence diagnostics, deterministic qualified
  interpretation, canonical report tables, audit, replay, sessions, examples, and installed-wheel
  coverage. Regression does not perform variable selection, row deletion, causal inference, or
  out-of-sample prediction validation.

- Added guided Welch one-way ANOVA with complete Games-Howell simultaneous comparisons,
  explicit classical ANOVA with Tukey-Kramer comparisons, and Kruskal-Wallis with complete
  Dunn-Holm comparisons. Pairwise families are calculated regardless of the omnibus decision and
  retain orientation, estimates, uncertainty where supported, raw/adjusted p-values, multiplicity
  metadata, sample sizes, standard errors, degrees of freedom, and decisions.
- Integrated multi-group summaries and pairwise records through deterministic narration, reports
  and safe exports, audit, completeness, sensitivity identity, replay, examples, installed-wheel
  smoke coverage, and a Python 3.10 minimum-SciPy 1.7.3 CI route.

- Added complete one-sample t inference against a finite researcher-declared reference, preserving
  observed-minus-reference orientation, analytical raw-difference intervals, and one-sample
  Cohen's d when defined.
- Added explicit unit-ID paired Wilcoxon signed-rank inference with recorded condition order,
  `wilcox` zero handling, matched-pairs rank-biserial effect, and complete/incomplete-pair
  accounting.
- Added inferential Spearman correlation for declared monotonic targets, including rho, p-value,
  ties metadata, and a deterministic paired-observation percentile bootstrap interval.
- Added two-sided Fisher exact inference for sparse 2x2 categorical tables using the shared
  contingency builder, ordered observed counts, and SciPy's sample odds ratio; adequate tables
  continue to use Pearson chi-square.
- Integrated all four methods with deterministic explanation, canonical reports, completeness,
  semantic audit checks, analysis planning, sensitivity identity, practical-significance
  boundaries, reproducibility/replay, session capabilities, examples, and installed-wheel tests.

- Added default and custom linear-interpolation percentile profiles, with P50 tied to the existing
  median calculation and deterministic percentile narration.
- Added JSON-safe categorical frequency tables and descriptive cross-tabs with explicit valid,
  missing, and excluded row accounting; separate valid/total, row, column, and total percentages;
  meaningful ordinal ordering; bounded narration; and shared contingency construction for the
  existing chi-square workflow.
- Added safeguarded coefficient-of-variation metadata and narration using sample SD divided by the
  absolute mean, with zero/near-zero and measurement-scale caveats.
- Integrated the new descriptions into profiles, story mode, insights, legacy/canonical reports,
  a synthetic example, and installed-wheel smoke coverage.

## [0.3.0] - 2026-09-27

This release adds a deterministic, researcher-readable narration layer across profiling,
recommendation, interpretation, practical-significance assessment, sensitivity analysis, and
reporting. Structured statistical records remain authoritative, and all narration remains local,
rule-based, reproducible, and independent of generative AI or external services.

### Added

- Added deterministic effect-size narratives for supported measures, including conventional
  magnitude labels, direction, sample context, and confidence-interval precision commentary.
- Added four-quadrant hypothesis explanations that combine the recorded `p < alpha` decision with
  effect magnitude without treating non-significance as equivalence or practical irrelevance.
- Added graded assumption messages for recorded normality, variance, independence, pairing, and
  other diagnostic states without changing method selection or the stated estimand.
- Added practical-significance verdicts for researcher-declared thresholds and sensitivity
  verdicts that preserve same-estimand, different-estimand, unavailable, and incompatible states.
- Added opt-in dataset story mode, descriptive column stories, connected InsightEngine narratives,
  and deterministic prioritized actions while retaining the existing structured outputs.
- Added recommendation explanations with why-this, relevant why-not, and researcher-verification
  sections, plus a complete `ResearchWorkflowResult.explain()` view with recorded group order,
  sample accounting, findings, assumptions, limitations, and warnings.
- Added escaped executive summaries to canonical and legacy HTML reports using only stored dataset,
  analysis, interpretation, diagnostic, practical-significance, and sensitivity records.
- Added a synthetic explainability example and public end-to-end, determinism, boundary, HTML
  safety, serialization, example-execution, and installed-wheel validation coverage.

### Changed

- Expanded researcher-facing profile, interpretation, insight, recommendation, and report prose
  while preserving numerical results, stable identifiers, structured decision traces, and schemas.
- Improved beginner-facing method labels, missing-information guidance, normality verdicts,
  printable report styling, and plain-text findings.
- Added advisory profiling resource metadata based on deep DataFrame memory use and analytical
  width without sampling, truncating, or changing computations.
- Reworked examples and documentation around permanent capability-oriented workflows and clarified
  the boundary between statistical evidence and human-readable narration.

### Fixed

- Preserved zero-percent completeness and unavailable values instead of displaying misleading
  defaults, duplicate punctuation, or duplicated confidence-interval labels.
- Restricted sensitivity decision summaries to completed comparable scenarios using the declared
  alpha; mixed estimands are no longer presented as directly numerically comparable.
- Required directional paired practical-significance thresholds to match the recorded contrast and
  strengthened paired sensitivity identity checks for unit, condition, design, and orientation.
- Blocked pairing when declared missing codes remain in the unit identifier until the caller
  explicitly normalizes the source data.
- Preserved valid raw mean differences when standardized effects are unavailable and accepted valid
  percentile-bootstrap intervals that do not contain the original point estimate.
- Retained finite D'Agostino-Pearson results under the recognized small-sample advisory and omitted
  Anderson-Darling results when SciPy supplies an invalid critical-value grid.
- Made workflow serialization JSON-safe when an included profile contains pandas dtype objects,
  without changing the in-memory profile contract.
- Ensured recommendation, explanation, report, and narration rendering remains deterministic and
  non-mutating across missing, nonfinite, minimal, and user-controlled inputs.

### Security

- Continued escaping untrusted HTML and LaTeX text, protecting formula-like CSV cells, omitting raw
  DataFrames and participant identifiers from reports, and keeping the narration layer offline.

## [0.2.0] - 2026-09-25

This release expands PyAutoStat from its initial analysis utilities into an explainable and
reproducible research-analysis assistant. It adds guided research workflows, study planning,
sensitivity and practical-significance analysis, paired-data support, reproducibility tooling,
richer reporting, and stronger statistical safeguards.

### Added

- Added `ResearchAssistant` guided workflows from research-question intake through method
  recommendation, analysis, interpretation, reporting, audit, and reproducibility metadata.
- Added structured research-question preparation with `prepare_question()` and
  `update_question()`, including explicit clarification when essential design information is
  missing.
- Added deterministic method recommendation with explicit study-design, estimand, variable-type,
  and data-feasibility checks.
- Added structured `AnalysisResult` and deterministic interpretation with effect estimates,
  confidence intervals, sample accounting, diagnostics, warnings, and limitations.
- Added canonical research reports with HTML, Markdown, JSON, CSV, and safe LaTeX output.
- Added General, APA-oriented, and IEEE-oriented report presentation styles.
- Added reporting-completeness assessment without converting reporting completeness into a
  study-quality score.
- Added explicit paired two-condition mean analysis using a researcher-supplied unit identifier,
  paired t-test, paired mean difference, confidence interval, and Cohen's \(d_z\).
- Added prospective study planning for independent and paired means, including power and
  confidence-interval precision planning.
- Added serializable statistical analysis plans and plan-adherence comparison.
- Added researcher-declared sensitivity analysis with estimand-aware comparison and retention of
  every attempted scenario.
- Added researcher-defined practical-significance thresholds with separate point-estimate and
  confidence-interval interpretation.
- Added optional decision-ledger tracking, dataset/content fingerprints, result auditing,
  reproducibility records, metadata-only reproducibility packages, and explicit supplied-data
  replay.
- Added richer dataset profiling with variable intelligence, categorical summaries, missingness
  patterns, duplicate information, pairwise correlation sample sizes, data-quality findings, and
  optional data dictionaries.
- Added advisory DataFrame resource metadata using deep pandas memory estimates, including
  large-memory and wide-correlation warnings without sampling or modifying source data.
- Added a JSON-safe session snapshot suitable for future notebook, CLI, or GUI integrations.
- Added explicit capability, architecture, statistical-validation, scientific-limitations,
  provenance, robustness, planning, and report-schema documentation.

### Fixed

- Prevented automatic switching from a mean estimand to a rank/distribution estimand based on
  diagnostic tests.
- Made Welch's t-test the default supported two-group independent mean comparison.
- Standardized first-versus-second group and paired-condition contrast direction across estimates
  and effects.
- Improved handling of undefined, nonfinite, extreme-scale, and numerically unreliable statistical
  results.
- Improved normality and variance diagnostic states so rejected, not rejected, and unknown remain
  distinct.
- Corrected percentile-bootstrap interpretation so valid intervals are not required to contain the
  observed point estimate.
- Preserved valid raw mean differences when standardized effects are unavailable.
- Added paired-design safeguards for unit identifiers, incomplete pairs, duplicate unit-condition
  observations, contrast orientation, and declared missing-value codes.
- Added paired sensitivity safeguards so analyses using different pairing definitions are not
  treated as directly comparable.
- Added directional practical-significance safeguards for reversed paired contrasts.
- Extended plan-adherence checks to planned sensitivity analyses and meaningful-effect thresholds.
- Strengthened report security with HTML escaping, strict JSON serialization, CSV formula
  protection, safe LaTeX escaping, and explicit file-write behavior.

### Changed

- Organized documentation around current capabilities rather than historical development stages.
- Established `ROADMAP.md` as the single forward-looking development roadmap.
- Renamed examples and tests with permanent capability-oriented names.
- Consolidated detailed scientific and technical documentation under `docs/`.
- Required Python 3.10 or newer.

### Quality and packaging

- Added Ruff formatting and linting, mypy type checking, coverage enforcement, package build
  validation, and Twine checks.
- Added GitHub Actions testing across Python 3.10-3.13 on Linux and Windows.
- Added isolated installed-wheel smoke testing.
- Expanded the automated test suite beyond 500 tests with greater than 90% code coverage.

## [0.1.0] - 2026-09-21

- Initial statistical analysis, insight, and report-export package.
