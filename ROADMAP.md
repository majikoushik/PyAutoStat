# PyAutoStat roadmap

This is the single forward-looking roadmap for PyAutoStat. It describes improvement priorities,
not release promises or a record of past implementation order.

## Current product status

PyAutoStat is an alpha Python package for deterministic, explainable analysis of pandas
DataFrames. Its guided workflow covers dataset profiling, structured research questions,
design-aware method recommendation, a bounded set of independent and paired analyses,
deterministic interpretation, research reports, consistency auditing, and reproducibility
metadata. It also supports explicit sensitivity scenarios, researcher-defined meaningful-effect
thresholds, analysis plans, prospective independent and paired mean planning, reporting
completeness, styled HTML/Markdown/LaTeX output, and UI-independent session snapshots.
The established analysis, planning, and reporting capabilities are checked into `main`, and their
GitHub CI matrix passed. A release remains a separate owner decision.

The method catalogue is intentionally limited. Repeated measures with more than two conditions,
clustered models, regression families, mixed models, survival analysis, causal inference, broad
multiplicity procedures, and sparse exact-table alternatives are not currently supported. Study
design facts and scientific meaning remain researcher responsibilities.

## Near term: alpha hardening

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

## Beta readiness

Beta-readiness work includes broader user testing, an API stability review, a documented
deprecation policy, supported-data-size guidance, dependency and Python support review,
documentation completeness, and repeatable performance benchmarks. Packaging and release
automation should be exercised without changing the statistical contracts. Beta is a maturity
target, not a scheduled next release.

## Candidate statistical capabilities

Potential additions include justified multi-group mean workflows, additional paired or
distributional methods, additional correlation inference, post-hoc and multiplicity procedures,
regression, and repeated-measures models. These are candidates rather than commitments.

A method should be added only when the complete contract can be supported:

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

## Release maturity

```text
alpha -> beta readiness -> stable
```

Alpha work focuses on correctness, bounded scope, usability, compatibility evidence, and honest
limitations. Beta readiness requires broader validation, clearer stability guarantees, mature
documentation, and measured scale behavior. Stable releases require a deliberate owner decision,
documented compatibility policy, dependable packaging, and sustained evidence from real use. No
date or version is assigned by this roadmap.
