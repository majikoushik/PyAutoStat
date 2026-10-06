# PyAutoStat Documentation

The root [README](../README.md) is the beginner introduction, the [API reference](../API_REFERENCE.md)
documents public calls and return contracts, [Product Vision](../PRODUCT_VISION.md) contains core
principles, and the [roadmap](../ROADMAP.md) lists future improvement priorities. The documents here
provide focused technical and scientific detail organized by research goal.

## Getting started

- [README](../README.md): Quick start, installation, core workflow, and overview.
- [Examples Guide](../examples/README.md): Walkthroughs using the bundled realistic 5,000-row customer dataset.
- [Capabilities](CAPABILITIES.md): Current user paths, workflow statuses, supported analyses, and boundaries.
- [Usability Audit](USABILITY_AUDIT.md): Systematic usability audit across all 13 supported feature families, input shapes, sample accounting, orientation clarity, and market benchmarks.

## Analysis workflows

- [Capabilities](CAPABILITIES.md): Exact supported analysis families, guided defaults, and explicit alternatives.
- [Statistical method contracts](STATISTICAL_METHOD_CONTRACTS.md): Authoritative scientific contracts, estimands, null hypotheses, uncertainty classifications, assumptions, degenerate data policies, and validation sources for every shipped inferential method.
- [ICC Guide](ICC_GUIDE.md): Dedicated guide to Intraclass Correlation Coefficients for quantitative inter-rater and test-retest reliability and agreement across the 6 canonical Shrout & Fleiss / McGraw & Wong configurations.
- [Advanced planning and presentation](ADVANCED_PLANNING_AND_PRESENTATION.md): Analysis plans, prospective study planning, explicit two-condition paired analysis, reporting completeness, and session snapshots.

## Understanding results

- [Statistical validation](STATISTICAL_VALIDATION.md): Numerical definitions, method-selection rules, validation cases, dependency behavior, and interpretation policies.
- [Effect-size confidence interval gaps](EFFECT_SIZE_CI_GAPS.md): Explicit classification of missing effect-size confidence intervals, candidate methods, validation requirements, and implementation priorities.
- [Robustness and practical significance](ROBUSTNESS_AND_PRACTICAL_SIGNIFICANCE.md): Sensitivity scenario comparability and researcher-defined meaningful-effect thresholds.
- [Scientific limitations](SCIENTIFIC_LIMITATIONS.md): Unsupported claims, design responsibilities, privacy limits, resource caveats, and dependency boundaries.

## Reporting and reproducibility

- [Terminal presentation](TERMINAL_PRESENTATION.md): Polished, Rich-powered console rendering for workflows, profiles, and governance records via `show(...)`.
- [HTML reporting](HTML_REPORTING.md): Offline static and interactive HTML presentation layers with zero statistical recalculation.
- [PDF reporting](PDF_REPORTING.md): Publication-oriented PDF export via headless Chromium print fidelity.
- [DOCX reporting](DOCX_REPORTING.md): Editable Microsoft Word export with native OpenXML tables and styled captions.
- [Static scientific figures](STATIC_FIGURES.md): Publication-quality vector (SVG, PDF) and raster (PNG) figure generation via Kaleido.
- [Research export bundles](RESEARCH_BUNDLES.md): Multi-format self-contained zip archives with cryptographic SHA-256 tamper verification.
- [Reporting audit](REPORTING_AUDIT.md): Systematic engineering and usability audit across all presentation media, target compatibilities, and semantic parity contracts.
- [Research report schema](RESEARCH_REPORT_SCHEMA.md): Canonical report structure, serialization, export protections, and presentation contracts.
- [Provenance and replay](PROVENANCE_AND_REPLAY.md): Decision ledger, content references, dataset fingerprints, auditing, reproducibility records, and explicit replay.

### Reporting quick guide

| Need | Use |
| --- | --- |
| Quick inspection | `show` |
| Share in browser | `save_html` |
| Printable fixed report | `save_pdf` |
| Editable Word report | `save_docx` |
| Journal/slide figure | `save_static_figures` |
| Archive/share all artifacts | `save_bundle` |

## Advanced research lifecycle

PyAutoStat supports an end-to-end governance and research lifecycle connecting prospective planning, execution, sensitivity, practical significance, reporting completeness, auditing, reproducibility records, and explicit replay:

- [Governance usability audit](GOVERNANCE_USABILITY_AUDIT.md): Systematic audit across all 16 lifecycle feature families, language contracts, status vocabularies, and scientific boundaries.
- [Advanced planning and presentation](ADVANCED_PLANNING_AND_PRESENTATION.md): Analysis plans (`StatisticalAnalysisPlan`), plan adherence (`plan_adherence`), prospective study planning (`StudyPlanner`), reporting completeness (`reporting_completeness`), and session snapshots (`session_snapshot`), including an end-to-end lifecycle script.
- [Robustness and practical significance](ROBUSTNESS_AND_PRACTICAL_SIGNIFICANCE.md): Same-estimand vs different-estimand sensitivity scenarios (`sensitivity_analysis`) and researcher-defined meaningful-effect thresholds (`practical_significance`).
- [Provenance and replay](PROVENANCE_AND_REPLAY.md): Decision tracking (`DecisionLedger`), dataset fingerprints (`dataset_fingerprint`), deterministic content references, report consistency audits (`audit`), reproducibility records (`ReproducibilityRecord`), and explicit re-execution (`reproduce`).
- [Research export bundles](RESEARCH_BUNDLES.md): Multi-format self-contained zip archives with cryptographic SHA-256 tamper verification.

### Advanced lifecycle quick guide

| Need | Use |
| --- | --- |
| Prospective sample-size planning | `StudyPlanner` / `assistant.study_planner()` |
| Freeze an analysis plan | `StatisticalAnalysisPlan` / `assistant.analysis_plan(...)` |
| Check plan vs execution | `plan_adherence` / `assistant.plan_adherence(...)` |
| Explicit sensitivity scenarios | `sensitivity_analysis` / `assistant.sensitivity_analysis(...)` |
| Researcher-defined meaningful threshold | `practical_significance` / `assistant.practical_significance(...)` |
| Report-element completeness | `reporting_completeness` / `assistant.reporting_completeness(...)` |
| Record local decisions | `DecisionLedger` / `assistant.enable_tracking()` |
| Check record/report consistency | `audit` / `assistant.audit(...)` |
| Capture replay metadata | `reproducibility_record` / `assistant.reproducibility_record(...)` |
| Explicitly rerun | `reproduce(record, data=...)` |
| Snapshot current application state | `session_snapshot` / `assistant.session_snapshot(...)` |

## Advanced / developer

- [Architecture](ARCHITECTURE.md): Module boundaries, public workflow, typed contracts, scientific safeguards, data ownership, and compatibility policy.
- [API Reference](../API_REFERENCE.md): Comprehensive public API documentation with parameter specifications and return types.
- [Product Vision](../PRODUCT_VISION.md): Enduring product principles, estimand preservation, and scientific philosophy.
- [Roadmap](../ROADMAP.md): Forward-looking development roadmap and planned capabilities.

