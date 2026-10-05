# PyAutoStat Documentation

The root [README](../README.md) is the beginner introduction, the [API reference](../API_REFERENCE.md)
documents public calls and return contracts, [Product Vision](../PRODUCT_VISION.md) contains core
principles, and the [roadmap](../ROADMAP.md) lists future improvement priorities. The documents here
provide focused technical and scientific detail organized by research goal.

## Getting started

- [README](../README.md): Quick start, installation, core workflow, and overview.
- [Examples Guide](../examples/README.md): Walkthroughs using the bundled realistic 5,000-row customer dataset.
- [Capabilities](CAPABILITIES.md): Current user paths, workflow statuses, supported analyses, and boundaries.

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
- [PDF reporting](PDF_REPORTING.md): Publication-ready PDF export via headless Chromium print fidelity.
- [DOCX reporting](DOCX_REPORTING.md): Editable Microsoft Word export with native OpenXML tables and styled captions.
- [Static scientific figures](STATIC_FIGURES.md): Publication-quality vector (SVG, PDF) and raster (PNG) figure generation via Kaleido.
- [Research export bundles](RESEARCH_BUNDLES.md): Multi-format self-contained zip archives with cryptographic SHA-256 tamper verification.
- [Research report schema](RESEARCH_REPORT_SCHEMA.md): Canonical report structure, serialization, export protections, and presentation contracts.
- [Provenance and replay](PROVENANCE_AND_REPLAY.md): Decision ledger, content references, dataset fingerprints, auditing, reproducibility records, and explicit replay.

## Advanced / developer

- [Architecture](ARCHITECTURE.md): Module boundaries, public workflow, typed contracts, scientific safeguards, data ownership, and compatibility policy.
- [API Reference](../API_REFERENCE.md): Comprehensive public API documentation with parameter specifications and return types.
- [Product Vision](../PRODUCT_VISION.md): Enduring product principles, estimand preservation, and scientific philosophy.
- [Roadmap](../ROADMAP.md): Forward-looking development roadmap and planned capabilities.
