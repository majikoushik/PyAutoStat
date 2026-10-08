# Advanced workflows

This page routes advanced research-lifecycle work. Start with
[Getting Started](GETTING_STARTED.md) or the [task guides](task_guides/README.md) for ordinary
analysis. Advanced records consume the same specifications and validated results; they do not
certify scientific truth, study quality, external preregistration, or data authenticity.

## Plan an analysis or study

- [Advanced planning and presentation](ADVANCED_PLANNING_AND_PRESENTATION.md) covers statistical
  analysis plans, prospective independent/paired mean planning, plan adherence, and an end-to-end
  lifecycle example.
- [Robustness and practical significance](ROBUSTNESS_AND_PRACTICAL_SIGNIFICANCE.md) covers
  researcher-declared sensitivity scenarios and meaningful-effect thresholds.
- [API reference: advanced feature routing](../API_REFERENCE.md#which-advanced-feature-do-i-need)
  lists the entry point and returned record for each feature.

Prospective planning requires researcher-supplied assumptions. PyAutoStat does not compute
observed post-hoc power.

## Assess and report recorded evidence

- [Research report schema](RESEARCH_REPORT_SCHEMA.md) documents the canonical report.
- [Result ergonomics and statements](RESULT_ERGONOMICS_AND_STATEMENTS.md) documents deterministic
  result accessors and APA-oriented statements.
- [Reporting audit](REPORTING_AUDIT.md) scopes format parity, privacy, and reporting claims.
- [HTML reporting](HTML_REPORTING.md), [PDF reporting](PDF_REPORTING.md),
  [DOCX reporting](DOCX_REPORTING.md), and [static figures](STATIC_FIGURES.md) cover individual
  outputs.

Reporting-completeness checks identify recorded elements as present, missing, partial, or not
applicable. They are not publication-readiness or journal-compliance scores.

## Audit, fingerprint, and replay

- [Provenance and replay](PROVENANCE_AND_REPLAY.md) covers decision records, dataset fingerprints,
  consistency audit, reproducibility metadata, and explicit replay with separately supplied data.
- [Research bundles](RESEARCH_BUNDLES.md) covers multi-format packages and SHA-256 manifest
  verification.
- [Governance usability audit](GOVERNANCE_USABILITY_AUDIT.md) records governance boundaries and
  status vocabularies.

An audit compares recorded values without rerunning the statistical test. A fingerprint detects
some dataset changes but does not authenticate identity, custody, or provenance. Replay is explicit
and never draws raw data from a metadata package.

## Direct and low-level APIs

Use `StatisticalAnalyzer` only when the exact calculation has already been selected.
`InsightEngine` and `ReportGenerator` remain supported compatibility surfaces. Typed
specifications, results, auditors, presentation models, and adapters support application
integration without creating a second statistical engine.

See the [API reference](../API_REFERENCE.md#level-2--direct-statistical-and-planning-api), the
[API stability policy](API_STABILITY.md), and the [architecture](ARCHITECTURE.md).

## Boundaries

Sensitivity results retain every declared attempt and identify estimand changes; they are not a
search for a favorable p-value. A nonsignificant superiority test is not an equivalence result.
Local planning records do not prove external preregistration. Reports and bundles do not include
raw participant rows by default, but small aggregates can still be sensitive.
