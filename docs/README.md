# PyAutoStat documentation

Use this page as the documentation home. Begin with a research task; move to reference and
governance material only when you need it.

## Start here

- [Getting Started](GETTING_STARTED.md): install the package and run independent, paired, and
  correlation examples.
- [Task Guides](task_guides/README.md): choose a workflow by research question.
- [README](../README.md): one-minute project introduction and exact quickstart output.
- [Examples Guide](../examples/README.md): longer examples using the bundled customer dataset.

## Task guides

| Research task | Guide |
| --- | --- |
| Independent, paired, distribution, or reference comparisons | [Mean comparisons](task_guides/mean_comparisons.md) |
| Pearson, Spearman, Kendall, partial, point-biserial, or categorical association | [Associations](task_guides/associations.md) |
| 3+ groups, repeated conditions, or two factors | [Multigroup, repeated & factorial](task_guides/multigroup_repeated_factorial.md) |
| Conditional means or binary event odds | [Regression](task_guides/regression.md) |
| Scale internal consistency or rater agreement | [Reliability & agreement](task_guides/reliability_agreement.md) |

## Statistical reference

- [Capabilities](CAPABILITIES.md): user paths, workflow statuses, supported analyses, and explicit
  boundaries.
- [Statistical Method Contracts](STATISTICAL_METHOD_CONTRACTS.md): estimands, inputs, assumptions,
  exclusions, numerical backends, effects, intervals, and limits for every registered method.
- [Statistical Validation](STATISTICAL_VALIDATION.md): cross-method numerical and interpretive
  policies.
- [Numerical Validation](NUMERICAL_VALIDATION.md): reference cases, tolerances, and evidence levels.
- [Scientific Limitations](SCIENTIFIC_LIMITATIONS.md): boundaries on designs and claims.
- [ICC Guide](ICC_GUIDE.md): focused selection and interpretation for agreement/reliability.

## Results and interpretation

- [Statistical validation](STATISTICAL_VALIDATION.md): numerical definitions, method-selection
  rules, validation cases, dependency behavior, and interpretation policies.
- [Numerical validation](NUMERICAL_VALIDATION.md): evidence levels, tolerance policy, and
  independently checked reference cases.
- [API reference: results and statements](../API_REFERENCE.md#result-ergonomics-primary-views-and-apa-oriented-statements): typed result
  records, scalar safeguards, tabular views, and deterministic APA-oriented statements.
- [Robustness and practical significance](ROBUSTNESS_AND_PRACTICAL_SIGNIFICANCE.md): Sensitivity scenario comparability and researcher-defined meaningful-effect thresholds.
- [Scientific limitations](SCIENTIFIC_LIMITATIONS.md): Unsupported claims, design responsibilities, privacy limits, resource caveats, and dependency boundaries.

## Reporting and export

- [Terminal presentation](TERMINAL_PRESENTATION.md): Polished, Rich-powered console rendering for workflows, profiles, and governance records via `show(...)`.
- [HTML reporting](HTML_REPORTING.md): Offline static and interactive HTML presentation layers with zero statistical recalculation.
- [PDF reporting](PDF_REPORTING.md): Publication-oriented PDF export via headless Chromium print fidelity.
- [DOCX reporting](DOCX_REPORTING.md): Editable Microsoft Word export with native OpenXML tables and styled captions.
- [Static scientific figures](STATIC_FIGURES.md): Publication-quality vector (SVG, PDF) and raster (PNG) figure generation via Kaleido.
- [Research export bundles](RESEARCH_BUNDLES.md): Multi-format self-contained zip archives with cryptographic SHA-256 tamper verification.
- [Research report schema](RESEARCH_REPORT_SCHEMA.md): Canonical report structure, serialization, export protections, and presentation contracts.

### Reporting quick guide

| Need | Use |
| --- | --- |
| Quick inspection | `show` |
| Share in browser | `save_html` |
| Printable fixed report | `save_pdf` |
| Editable Word report | `save_docx` |
| Journal/slide figure | `save_static_figures` |
| Archive/share all artifacts | `save_bundle` |

Core installation provides terminal, static HTML, Markdown, JSON, CSV tables, and safe LaTeX.
Plotly, Playwright/Chromium, python-docx, and Kaleido/Chrome are optional and documented in their
respective guides.

## Reproducibility and advanced

Start with [Advanced Workflows](ADVANCED_WORKFLOWS.md), which routes planning, practical
significance, sensitivity, reporting completeness, audit, fingerprints, replay, research bundles,
governance, and direct APIs.

- [Advanced workflows](ADVANCED_WORKFLOWS.md): analysis plans, prospective study planning, plan
  adherence, result accessors and statements, reporting completeness, session snapshots, and
  direct APIs.
- [Robustness and practical significance](ROBUSTNESS_AND_PRACTICAL_SIGNIFICANCE.md): Same-estimand vs different-estimand sensitivity scenarios (`sensitivity_analysis`) and researcher-defined meaningful-effect thresholds (`practical_significance`).
- [Provenance and replay](PROVENANCE_AND_REPLAY.md): Decision tracking (`DecisionLedger`), dataset fingerprints (`dataset_fingerprint`), deterministic content references, report consistency audits (`audit`), reproducibility records (`ReproducibilityRecord`), and explicit re-execution (`reproduce`).
- [Research export bundles](RESEARCH_BUNDLES.md): Multi-format self-contained zip archives with cryptographic SHA-256 tamper verification.

### Advanced lifecycle quick guide

| Need | Use |
| --- | --- |
| Prospective sample-size planning | `StudyPlanner` / `assistant.study_planner()` |
| Record an immutable analysis plan snapshot | `StatisticalAnalysisPlan` / `assistant.analysis_plan(...)` |
| Check plan vs execution | `plan_adherence` / `assistant.plan_adherence(...)` |
| Explicit sensitivity scenarios | `sensitivity_analysis` / `assistant.sensitivity_analysis(...)` |
| Researcher-defined meaningful threshold | `practical_significance` / `assistant.practical_significance(...)` |
| Report-element completeness | `reporting_completeness` / `assistant.reporting_completeness(...)` |
| Record local decisions | `DecisionLedger` / `assistant.enable_tracking()` |
| Check record/report consistency | `audit` / `assistant.audit(...)` |
| Capture replay metadata | `reproducibility_record` / `assistant.reproducibility_record(...)` |
| Explicitly rerun | `reproduce(record, data=...)` |
| Snapshot current application state | `session_snapshot` / `assistant.session_snapshot(...)` |

These records preserve and compare recorded evidence. They do not authenticate data, prove
preregistration, certify study quality, or establish causality.

## Developer and API

- [Architecture](ARCHITECTURE.md): Module boundaries, public workflow, typed contracts, scientific safeguards, data ownership, and compatibility policy.
- [API Reference](../API_REFERENCE.md): Comprehensive public API documentation with parameter specifications and return types.
- [API Stability Policy](API_STABILITY.md): Stability tiers, SemVer policies, and deprecation process.
- [Public API 1.0](PUBLIC_API_1_0.md): audited top-level exports, stability tiers, and legacy-policy
  decisions.
- [Product Vision](../PRODUCT_VISION.md): Enduring product principles, estimand preservation, and scientific philosophy.
- [Roadmap](../ROADMAP.md): Forward-looking development roadmap and planned capabilities.
- [Citation Metadata](../CITATION.cff): Machine-readable academic citation metadata (CFF format).

## Migration and release

- [Migration to 1.0](MIGRATION_TO_1_0.md): canonical workflows, compatibility surfaces, and
  serialization guidance for existing users.
- [Release notes 1.0.0](RELEASE_NOTES_1_0_0.md): stable-release scope, validation, reporting, and
  known boundaries.
- [Changelog](../CHANGELOG.md): chronological software changes.

