# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

- Added effect-size confidence intervals for existing shipped effect quantities across the package:
  - Exact noncentral-t inversion confidence intervals for paired Cohen's $d_z$ (`paired_t`) and one-sample Cohen's $d$ (`one_sample_t`), with exact bracketed root solving and degenerate handling.
  - Fisher-z asymptotic normal confidence interval for Pearson correlation ($r$), with bounds $[-1, 1]$ and explicit unavailable status for $n \le 3$.
  - Estimator-matched asymptotic log-Wald confidence interval for Fisher's exact sample odds ratio ($ad/bc$), with explicit unavailable status for zero-cell tables and strict refusal of silent continuity corrections.
  - Deterministic participant/pair-level percentile bootstrap confidence intervals for matched-pairs rank-biserial correlation (`wilcoxon_signed_rank`) and Friedman Wilcoxon-Holm contrasts.
  - Participant-block percentile bootstrap confidence interval for Friedman Kendall's $W$.
  - Exact noncentral-F inversion confidence interval for repeated-measures ANOVA partial eta-squared ($\eta_p^2$) based on uncorrected observed F and uncorrected condition/error degrees of freedom, preserving SS-based point effect.
  - Exact noncentral-t pairwise Cohen's $d_z$ confidence intervals on all repeated-measures pairwise paired-t contrasts, with explicit pointwise (`multiplicity_adjusted=False`) accounting.
  - Independent within-group percentile bootstrap confidence intervals for Kruskal-Wallis post-hoc Dunn pairwise rank-biserial correlations, with explicit pointwise (`multiplicity_adjusted=False`) accounting.
  - Case-resampling percentile bootstrap confidence interval for OLS in-sample $R^2$, preserving design matrix structure and categorical factor coding.
- Added dedicated `src/pyautostat/uncertainty.py` module providing reusable, numerically validated root-finding and bootstrap helpers without raising package dependency floors.
- Added machine-checkable authoritative method contract architecture (`MethodContract` dataclass and `METHOD_CONTRACTS` registry) covering all 22 executable statistical methods with 26 explicit scientific fields: estimand, hypotheses, primary estimate, effect size, uncertainty status, required assumptions, diagnostics, missing-data policy, degenerate-data behavior, multiplicity policy, numerical provenance, interpretation limitations, audit invariants, and independent validation sources.
- Added comprehensive documentation deliverables:
  - `docs/STATISTICAL_METHOD_CONTRACTS.md`: authoritative scientific contract specification for every shipped inferential method.
  - `docs/EFFECT_SIZE_CI_GAPS.md`: uncertainty gap classification, candidate defensible confidence interval methods, and milestone priorities.
- Added audit invariant hardening and extensive corruption test coverage (`tests/test_audit_hardening.py`, `tests/test_effect_size_confidence_intervals.py`, and `tests/test_uncertainty_hardening.py`):
  - Verification of finite p-values in [0, 1] across all results.
  - Verification of confidence interval bound ordering (`lower <= upper`) and analytical Student-t point estimate inclusion.
  - Verification of mathematical bounds on bounded effect sizes (rank-biserial, Pearson r, Spearman rho, Kendall tau-b, Cramér's V, Kendall's W, eta-squared, partial eta-squared, rank epsilon-squared, in-sample R-squared).
  - Two-group sample size and contrast order accounting.
  - Degrees of freedom invariants for Student-t ($df = n - 2$), one-sample t ($df = n - 1$), and paired t ($df = n_{\text{pairs}} - 1$).
  - Repeated-measures ANOVA sum-of-squares partition and partial eta-squared algebraic consistency.
  - Greenhouse-Geisser corrected degrees of freedom and p-value consistency.
  - Friedman Kendall's W consistency with Friedman Q.
  - Logistic regression odds ratio consistency with beta ($\text{OR} = \exp(\beta)$) and exponentiated Wald CI consistency.
  - Pairwise multiplicity count matching and Holm step-down invariant ($adjusted\_p \ge raw\_p$).

### Changed

- Hardened scientific narration and metadata across result interpretation:
  - Corrected Friedman non-rejection phrasing to avoid universal equality-of-medians claims.
  - Clarified that Kendall's W represents within-unit rank concordance rather than percentage of variance explained.
  - Canonicalized Mauchly sphericity status checks to strictly `"not_rejected"`.
  - Distinguished Fisher's exact inferential null hypothesis (`odds ratio = 1.0`) from the sample cross-product effect estimate.
- Documented intentional non-support of observed/post-hoc power by scientific policy in `docs/SCIENTIFIC_LIMITATIONS.md` and `ROADMAP.md` (Hoenig & Heisey, 2001).
- Clarified that logistic regression uses analytical Wald z-intervals for coefficients and exponentiated Wald intervals for odds ratios, not bootstrap intervals.

## [0.4.0] - 2026-09-30

This release expands PyAutoStat with new descriptive, inferential,
multi-group, regression, reliability, binary/association, and
repeated-measures workflows while preserving deterministic method
selection, explicit study-design and estimand contracts, reproducible
reporting, and scientific guardrails. PyAutoStat remains an Alpha
release.

### Added

- Added complete repeated-measures analysis for 3+ conditions on the
  same observational units: one-way repeated-measures ANOVA for
  continuous mean outcomes with full ANOVA tables, Mauchly’s sphericity
  test, Greenhouse-Geisser epsilon and corrected degrees of freedom /
  p-values when sphericity is violated, partial eta-squared
  repeated-measures effect sizes, and complete pairwise paired t-test
  follow-up with per-contrast analytical confidence intervals and Holm
  multiplicity adjustment; and the Friedman rank-sum test for repeated
  rank/distribution targets with Kendall’s W effect size and complete
  pairwise Wilcoxon signed-rank follow-up with matched-pairs
  rank-biserial correlations and Holm multiplicity adjustment. Both
  workflows require explicit unit identity, long-form panels, and 3+
  ordered condition labels, enforce complete-case panels while auditing
  and reporting missing and excluded unit accounting, block duplicate
  unit-condition records, preserve deterministic contrast orientations,
  and integrate with explainable narration, canonical
  HTML/Markdown/JSON/CSV reports, reporting completeness, result
  auditing, replay reproducibility, analysis plans, session snapshots,
  examples, and installed-wheel smoke testing. Two-condition paired
  workflows remain completely unchanged. No mixed-effects models, GEE,
  mixed ANOVA, factorial repeated measures, or automatic imputation were
  added.

- Added five complete binary-outcome and extended-association workflows:
  binary logistic regression, exact unit-ID McNemar inference,
  point-biserial correlation, explicit inferential Kendall tau-b, and
  partial Pearson correlation for declared quantitative controls. The
  methods preserve event/positive-level and condition orientation, use
  paired-unit or complete-row bootstrap uncertainty where appropriate,
  integrate with deterministic interpretation, canonical reports and
  tables, semantic audit, analysis plans, replay, session snapshots, and
  a synthetic example, and explicitly block separation, duplicate pairs,
  rank-deficient adjustment, and ambiguous binary coding. No
  classification metrics, automatic thresholds, causal claims, control
  selection, or additional generalized models were added.

- Added a complete researcher-declared survey and scale reliability
  workflow centered on Cronbach’s alpha, with complete-case and per-item
  missingness accounting, deterministic respondent-row bootstrap
  intervals, corrected item-total correlations, alpha-if-deleted,
  inter-item diagnostics, optional explicit bounded reverse scoring,
  qualified non-inferential interpretation, dedicated report tables,
  audit, replay, sessions, examples, and installed-wheel coverage. The
  workflow never discovers scales, deletes items, reverse-scores
  automatically, creates composite columns, or treats alpha as validity
  or a universal pass/fail threshold.

- Added complete simple and multiple ordinary least-squares
  conditional-mean regression for continuous outcomes, with ordered
  predictor lists, explicit treatment coding and reference levels for
  Boolean/nominal/ordinal predictors, complete-case accounting, rank
  validation, classical or explicit HC3 covariance inference,
  coefficient and model-fit records, continuous standardized betas, VIF,
  variance/residual/influence diagnostics, deterministic qualified
  interpretation, canonical report tables, audit, replay, sessions,
  examples, and installed-wheel coverage. Regression does not perform
  variable selection, row deletion, causal inference, or out-of-sample
  prediction validation.

- Added guided Welch one-way ANOVA with complete Games-Howell
  simultaneous comparisons, explicit classical ANOVA with Tukey-Kramer
  comparisons, and Kruskal-Wallis with complete Dunn-Holm comparisons.
  Pairwise families are calculated regardless of the omnibus decision
  and retain orientation, estimates, uncertainty where supported,
  raw/adjusted p-values, multiplicity metadata, sample sizes, standard
  errors, degrees of freedom, and decisions.

- Integrated multi-group summaries and pairwise records through
  deterministic narration, reports and safe exports, audit,
  completeness, sensitivity identity, replay, examples, installed-wheel
  smoke coverage, and a Python 3.10 minimum-supported numerical-stack CI
  route.

- Added complete one-sample t inference against a finite
  researcher-declared reference, preserving observed-minus-reference
  orientation, analytical raw-difference intervals, and one-sample
  Cohen’s d when defined.

- Added explicit unit-ID paired Wilcoxon signed-rank inference with
  recorded condition order, `wilcox` zero handling, matched-pairs
  rank-biserial effect, and complete/incomplete-pair accounting.

- Added inferential Spearman correlation for declared monotonic targets,
  including rho, p-value, ties metadata, and a deterministic
  paired-observation percentile bootstrap interval.

- Added two-sided Fisher exact inference for sparse 2x2 categorical
  tables using the shared contingency builder, ordered observed counts,
  and SciPy’s sample odds ratio; adequate tables continue to use Pearson
  chi-square.

- Integrated all four methods with deterministic explanation, canonical
  reports, completeness, semantic audit checks, analysis planning,
  sensitivity identity, practical-significance boundaries,
  reproducibility/replay, session capabilities, examples, and
  installed-wheel tests.

- Added default and custom linear-interpolation percentile profiles,
  with P50 tied to the existing median calculation and deterministic
  percentile narration.

- Added JSON-safe categorical frequency tables and descriptive
  cross-tabs with explicit valid, missing, and excluded row accounting;
  separate valid/total, row, column, and total percentages; meaningful
  ordinal ordering; bounded narration; and shared contingency
  construction for the existing chi-square workflow.

- Added safeguarded coefficient-of-variation metadata and narration
  using sample SD divided by the absolute mean, with zero/near-zero and
  measurement-scale caveats.

- Integrated the new descriptions into profiles, story mode, insights,
  legacy/canonical reports, a synthetic example, and installed-wheel
  smoke coverage.

### Fixed

- Corrected Mauchly sphericity p-value using the higher-order
  Box/Anderson asymptotic chi-square approximation, matching published
  SPSS and reference results.

- Replaced mislabeled repeated-measures reference fixture with genuine
  external reference data from Andy Field’s Bushtucker example,
  validating exact arithmetic and published rounded SPSS values.

- Strengthened Greenhouse-Geisser and repeated-measures audit checks to
  use actual emitted schema keys, enforcing corrected degrees-of-freedom
  identities, corrected p-value identities, primary inference
  consistency, and sphericity branch consistency with corruption
  detection tests.

- Reconciled repeated-measures documentation across README and docs to
  accurately describe supported one-way 3+ condition designs,
  complete-case matching, and explicit boundaries (including that
  Friedman requires ordered numeric outcomes and does not auto-encode
  textual labels).

- Formatted multiplicity assertions in installed-wheel smoke test within
  maximum line length constraints.

- Repeated-measures pairwise inference now reports mathematically
  unavailable zero-variance contrasts explicitly with descriptive mean
  differences and degenerate intervals instead of producing
  contradictory statistics.

- Repeated Friedman pairwise follow-up no longer converts signed-rank
  backend failures or all-zero differences into p=1, reusing the
  established paired-Wilcoxon contract.

- Repeated pairwise decisions and sphericity narration now consistently
  use the package-wide strict p-value decision convention.

- Repeated-measures validation now includes independent RM-ANOVA
  verification against statsmodels `AnovaRM`, sum-of-squares partition
  and partial eta-squared checks, and fixed external Mauchly and
  Greenhouse-Geisser reference checks.

- Tightened repeated-measures sphericity interpretation and structured
  metadata to state that no degrees-of-freedom correction was applied
  under the configured sphericity policy rather than implying that
  failing to reject Mauchly’s test proves sphericity or makes correction
  universally unrequired.

## [0.3.0] - 2026-09-27

This release adds a deterministic, researcher-readable narration layer
across profiling, recommendation, interpretation, practical-significance
assessment, sensitivity analysis, and reporting. Structured statistical
records remain authoritative, and all narration remains local,
rule-based, reproducible, and independent of generative AI or external
services.

### Added

- Added deterministic effect-size narratives for supported measures,
  including conventional magnitude labels, direction, sample context,
  and confidence-interval precision commentary.

- Added four-quadrant hypothesis explanations that combine the recorded
  `p < alpha` decision with effect magnitude without treating
  non-significance as equivalence or practical irrelevance.

- Added graded assumption messages for recorded normality, variance,
  independence, pairing, and other diagnostic states without changing
  method selection or the stated estimand.

- Added practical-significance verdicts for researcher-declared
  thresholds and sensitivity verdicts that preserve same-estimand,
  different-estimand, unavailable, and incompatible states.

- Added opt-in dataset story mode, descriptive column stories, connected
  InsightEngine narratives, and deterministic prioritized actions while
  retaining the existing structured outputs.

- Added recommendation explanations with why-this, relevant why-not, and
  researcher-verification sections, plus a complete
  `ResearchWorkflowResult.explain()` view with recorded group order,
  sample accounting, findings, assumptions, limitations, and warnings.

- Added escaped executive summaries to canonical and legacy HTML reports
  using only stored dataset, analysis, interpretation, diagnostic,
  practical-significance, and sensitivity records.

- Added a synthetic explainability example and public end-to-end,
  determinism, boundary, HTML safety, serialization, example-execution,
  and installed-wheel validation coverage.

### Changed

- Expanded researcher-facing profile, interpretation, insight,
  recommendation, and report prose while preserving numerical results,
  stable identifiers, structured decision traces, and schemas.

- Improved beginner-facing method labels, missing-information guidance,
  normality verdicts, printable report styling, and plain-text findings.

- Added advisory profiling resource metadata based on deep DataFrame
  memory use and analytical width without sampling, truncating, or
  changing computations.

- Reworked examples and documentation around permanent
  capability-oriented workflows and clarified the boundary between
  statistical evidence and human-readable narration.

### Fixed

- Preserved zero-percent completeness and unavailable values instead of
  displaying misleading defaults, duplicate punctuation, or duplicated
  confidence-interval labels.

- Restricted sensitivity decision summaries to completed comparable
  scenarios using the declared alpha; mixed estimands are no longer
  presented as directly numerically comparable.

- Required directional paired practical-significance thresholds to match
  the recorded contrast and strengthened paired sensitivity identity
  checks for unit, condition, design, and orientation.

- Blocked pairing when declared missing codes remain in the unit
  identifier until the caller explicitly normalizes the source data.

- Preserved valid raw mean differences when standardized effects are
  unavailable and accepted valid percentile-bootstrap intervals that do
  not contain the original point estimate.

- Retained finite D’Agostino-Pearson results under the recognized
  small-sample advisory and omitted Anderson-Darling results when SciPy
  supplies an invalid critical-value grid.

- Made workflow serialization JSON-safe when an included profile
  contains pandas dtype objects, without changing the in-memory profile
  contract.

- Ensured recommendation, explanation, report, and narration rendering
  remains deterministic and non-mutating across missing, nonfinite,
  minimal, and user-controlled inputs.

### Security

- Continued escaping untrusted HTML and LaTeX text, protecting
  formula-like CSV cells, omitting raw DataFrames and participant
  identifiers from reports, and keeping the narration layer offline.

## [0.2.0] - 2026-09-25

This release expands PyAutoStat from its initial analysis utilities into
an explainable and reproducible research-analysis assistant. It adds
guided research workflows, study planning, sensitivity and
practical-significance analysis, paired-data support, reproducibility
tooling, richer reporting, and stronger statistical safeguards.

### Added

- Added `ResearchAssistant` guided workflows from research-question
  intake through method recommendation, analysis, interpretation,
  reporting, audit, and reproducibility metadata.

- Added structured research-question preparation with
  `prepare_question()` and `update_question()`, including explicit
  clarification when essential design information is missing.

- Added deterministic method recommendation with explicit study-design,
  estimand, variable-type, and data-feasibility checks.

- Added structured `AnalysisResult` and deterministic interpretation
  with effect estimates, confidence intervals, sample accounting,
  diagnostics, warnings, and limitations.

- Added canonical research reports with HTML, Markdown, JSON, CSV, and
  safe LaTeX output.

- Added General, APA-oriented, and IEEE-oriented report presentation
  styles.

- Added reporting-completeness assessment without converting reporting
  completeness into a study-quality score.

- Added explicit paired two-condition mean analysis using a
  researcher-supplied unit identifier, paired t-test, paired mean
  difference, confidence interval, and Cohen’s (d_z).

- Added prospective study planning for independent and paired means,
  including power and confidence-interval precision planning.

- Added serializable statistical analysis plans and plan-adherence
  comparison.

- Added researcher-declared sensitivity analysis with estimand-aware
  comparison and retention of every attempted scenario.

- Added researcher-defined practical-significance thresholds with
  separate point-estimate and confidence-interval interpretation.

- Added optional decision-ledger tracking, dataset/content fingerprints,
  result auditing, reproducibility records, metadata-only
  reproducibility packages, and explicit supplied-data replay.

- Added richer dataset profiling with variable intelligence, categorical
  summaries, missingness patterns, duplicate information, pairwise
  correlation sample sizes, data-quality findings, and optional data
  dictionaries.

- Added advisory DataFrame resource metadata using deep pandas memory
  estimates, including large-memory and wide-correlation warnings
  without sampling or modifying source data.

- Added a JSON-safe session snapshot suitable for future notebook, CLI,
  or GUI integrations.

- Added explicit capability, architecture, statistical-validation,
  scientific-limitations, provenance, robustness, planning, and
  report-schema documentation.

### Fixed

- Prevented automatic switching from a mean estimand to a
  rank/distribution estimand based on diagnostic tests.

- Made Welch’s t-test the default supported two-group independent mean
  comparison.

- Standardized first-versus-second group and paired-condition contrast
  direction across estimates and effects.

- Improved handling of undefined, nonfinite, extreme-scale, and
  numerically unreliable statistical results.

- Improved normality and variance diagnostic states so rejected, not
  rejected, and unknown remain distinct.

- Corrected percentile-bootstrap interpretation so valid intervals are
  not required to contain the observed point estimate.

- Preserved valid raw mean differences when standardized effects are
  unavailable.

- Added paired-design safeguards for unit identifiers, incomplete pairs,
  duplicate unit-condition observations, contrast orientation, and
  declared missing-value codes.

- Added paired sensitivity safeguards so analyses using different
  pairing definitions are not treated as directly comparable.

- Added directional practical-significance safeguards for reversed
  paired contrasts.

- Extended plan-adherence checks to planned sensitivity analyses and
  meaningful-effect thresholds.

- Strengthened report security with HTML escaping, strict JSON
  serialization, CSV formula protection, safe LaTeX escaping, and
  explicit file-write behavior.

### Changed

- Organized documentation around current capabilities rather than
  historical development stages.

- Established `ROADMAP.md` as the single forward-looking development
  roadmap.

- Renamed examples and tests with permanent capability-oriented names.

- Consolidated detailed scientific and technical documentation under
  `docs/`.

- Required Python 3.10 or newer.

### Quality and packaging

- Added Ruff formatting and linting, mypy type checking, coverage
  enforcement, package build validation, and Twine checks.

- Added GitHub Actions testing across Python 3.10-3.13 on Linux and
  Windows.

- Added isolated installed-wheel smoke testing.

- Expanded the automated test suite beyond 500 tests with greater than
  90% code coverage.

## [0.1.0] - 2026-09-21

- Initial statistical analysis, insight, and report-export package.
