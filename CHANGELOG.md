# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

No changes recorded yet.

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
