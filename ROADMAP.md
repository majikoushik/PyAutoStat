# PyAutoStat roadmap

This is the single forward-looking roadmap for PyAutoStat. It describes improvement priorities,
not release promises or a record of past implementation order.

## Current product status

PyAutoStat is a stable Python package for deterministic, explainable analysis of pandas
DataFrames. Its guided workflow covers dataset profiling, structured research questions,
design-aware method recommendation, a bounded set of independent and paired analyses,
deterministic interpretation, research reports, consistency auditing, and reproducibility
metadata. It also supports explicit sensitivity scenarios, researcher-defined meaningful-effect
thresholds, analysis plans, prospective independent and paired mean planning, reporting
completeness, styled HTML/Markdown/LaTeX output, and UI-independent session snapshots.
It supports bounded simple and multiple ordinary least-squares conditional-mean regression with
explicit treatment coding, classical or HC3 covariance inference, and stored diagnostics.
It also supports researcher-declared multi-item internal-consistency analysis with Cronbach's
alpha, deterministic bootstrap uncertainty, item diagnostics, and explicit reverse scoring.
It supports binary logistic regression, exact paired-binary McNemar inference, point-biserial
correlation, explicit inferential Kendall tau-b, and partial Pearson correlation with declared
quantitative controls. Binary and paired orientations remain explicit, and these methods do not
add classification thresholds or causal adjustment claims. It also supports explicit repeated-measures
designs for three or more conditions, including one-way repeated-measures ANOVA with sphericity
evaluation and Greenhouse-Geisser correction, the Friedman rank-sum test, and planned Holm-adjusted
pairwise follow-up. It supports independent two-way factorial ANOVA with balanced or unbalanced
designs, Type II and Type III sums of squares, partial eta-squared effect sizes with exact noncentral-F
inversion confidence intervals, unweighted estimated marginal means, cell summaries, residual
diagnostics, and planned follow-ups with Holm multiplicity adjustment.
It supports rater reliability and agreement via Intraclass Correlation Coefficients (ICC) across all 6
canonical Shrout & Fleiss (1979) and McGraw & Wong (1996) variants (ICC(1,1), ICC(1,k), ICC(2,1),
ICC(2,k), ICC(3,1), ICC(3,k)) with ANOVA mean squares decomposition, method-of-moments variance
components, exact and Satterthwaite F-inversion confidence intervals, complete-target panel filtering,
and non-clamping of negative estimates.
The established analysis, planning, and reporting capabilities are checked into `main`, and their
GitHub CI matrix passed. A release remains a separate owner decision.

The method catalogue is intentionally limited. Mixed models, mixed ANOVA, factorial repeated-measures
ANOVA, clustered regression, incomplete-panel longitudinal models, generalized estimating equations (GEE),
generalized and regularized regression families, interaction or polynomial model construction, survival
analysis, causal inference, broad multiplicity procedures, observed or post-hoc power (intentionally
unsupported by scientific policy; prospective study planning is supported), and sparse exact-table
alternatives are not currently supported. Study design facts and scientific meaning remain researcher
responsibilities. See [statistical method contracts](docs/STATISTICAL_METHOD_CONTRACTS.md) for full
contracts of all supported methods.

## Near term: 1.x hardening

- Keep beginner onboarding centered on `ResearchAssistant(df).profile()` and
  `ResearchAssistant(df).run(...)`, with advanced contracts introduced progressively.
- Add methodology citations and independently check numerical examples against reference datasets.
- Exercise the package on varied real-world data shapes, missingness patterns, dtypes, and study
  designs without weakening validation.
- Validate declared dependency floors where compatible test environments exist and document the
  exact tested support matrix.
- Benchmark representative narrow, wide, and larger datasets; measure runtime, temporary memory,
  and copying before claiming performance improvements.
- Improve warning and error clarity, packaging checks, installed-artifact smoke coverage, and
  end-to-end examples.

## Stable-series maintenance

Stable-series work includes broader user testing, measured supported-data-size guidance,
dependency and Python support review, documentation maintenance, and repeatable performance
benchmarks. Packaging and release automation should continue to be exercised without changing
statistical contracts. Breaking public API or schema changes require the documented deprecation
process or a new major release, subject to the scientific-correctness and security exceptions.

## Candidate statistical capabilities

Future candidate statistical methods remain open and will be evaluated based on demonstrated
scientific demand, feasibility of complete contracts, and independent numerical verification.
Candidate methods will only be scheduled when the complete contract can be supported:

A method or uncertainty algorithm should be added only when the complete contract can be supported:

```text
design
-> estimand
-> validation
-> execution
-> uncertainty
-> interpretation
-> report
-> audit
-> reproducibility
```

Method count must not take priority over correct design handling, numerical validation, clear
limits, and stable serializable results.

## Research lifecycle

- **Design and planning:** improve prospective planning contracts, explicit assumptions, analysis
  plans, and design-specific blockers without deriving future effects from observed results.
- **Data:** strengthen profiling, declarations, missing-data transparency, privacy guidance, and
  measured large-data behavior while preserving source values.
- **Analysis:** add only methods with a defensible estimand, validation policy, numerical backend,
  uncertainty measure, and edge-case behavior.
- **Interpretation:** retain deterministic, quantity-aware explanations that separate statistical,
  practical, and causal claims.
- **Reporting:** keep every format linked to one canonical result and improve accessible,
  publication-oriented presentation.
- **Reproducibility:** extend consistency checks, environment records, and explicit replay while
  stating the limits of local provenance.

## Performance and scale

- Profile memory and runtime on representative datasets rather than relying on row count alone.
- Reduce unnecessary full-frame and intermediate copies after measuring their cost.
- Investigate scalable correlation and profile modes for wide data while preserving explicit
  behavior and complete provenance.
- Consider optional chunk-aware descriptive operations only where they preserve the exact target
  quantity and disclose their execution semantics.
- Provide explicit controls for expensive profiling components if benchmarks show a clear need.
- Never introduce automatic sampling, truncation, dtype conversion, or silent calculation skips.

## Interfaces

- A future GUI may render the existing session snapshot, clarification questions, and validated
  records. It must submit choices back to the core and must not duplicate statistical rules.
- Improve notebook presentation and discovery of structured warnings and continuation steps.
- Consider a CLI only where it can consume the same specifications and results without creating a
  second decision engine.

## Reporting and export

- Assess PDF and DOCX export for maintainability, accessibility, deterministic rendering, and
  security before adding dependencies.
- Add report customization and journal-specific extensions only when the same canonical values,
  limitations, and audit trail remain intact.
- Continue improving accessibility and privacy guidance for small aggregate cells and shared
  artifacts.

## Scientific validation

- Expand methodology references and trace each method to its assumptions and numerical source.
- Maintain reference datasets and independent calculations for supported methods and edge cases.
- Seek external statistical and subject-matter review of design contracts, interpretations, and
  reporting language.
- Use user feedback to identify ambiguous guidance and unsupported real-world designs.

## Release maintenance

```text
compatible 1.0.x fixes -> backward-compatible 1.x additions -> deliberate 2.0 changes
```

Stable maintenance focuses on correctness, bounded scope, usability, compatibility evidence,
dependable packaging, and honest limitations. A future major release would require a deliberate
owner decision, documented migration guidance, and evidence that the compatibility cost is
justified. No future release date or version is assigned by this roadmap.
