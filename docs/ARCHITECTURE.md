# Architecture

PyAutoStat separates research specification, numerical execution, interpretation, reporting, and
provenance so each layer can be validated independently. The Python API is the reference
interface. Other interfaces must use the same serializable records and core validators.

## Public workflows

`ResearchAssistant` owns a defensive copy of a pandas DataFrame and coordinates the supported
workflow:

1. `profile()` produces descriptive dataset metadata without requiring a research question.
2. `prepare_question()` and `update_question()` build an immutable research specification and
   return structured questions when essential information is missing.
3. `recommend_test()` validates design, target quantity, analytical types, and data feasibility
   without running an inferential calculation.
4. `analyze()` revalidates the request, dispatches one selected method, and returns an
   `AnalysisResult`.
5. `interpret()` converts that recorded result into deterministic qualified findings without
   recalculation.
6. `report()`, `audit()`, and `reproducibility_record()` render and inspect the same captured
   records.
7. `run()` coordinates the bounded guided path and returns `needs_input`, `data_limited`,
   `unsupported`, `failed`, `partial`, or `completed` as appropriate.

Presentation helpers such as `summarize()`, `explain()`, `findings_plain`, `verdict`, and
`compare()` read existing records. They do not select methods, rerun calculations, or change a
serialized contract. Display method names resolve from the capability registry.

Optional sensitivity, practical-significance, prospective-planning, analysis-plan, completeness,
and session-snapshot operations remain explicit. They do not silently replace the primary
analysis.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `analyzer.py`, `categorical.py` | Established descriptive and inferential numerical backends |
| `profiling.py`, `detection.py`, `insights.py` | Dataset profiling, advisory variable intelligence, data-quality summaries, and resource metadata |
| `specifications.py`, `question_builder.py` | Typed research configuration, completeness checks, and structured missing-information requests |
| `recommendation.py` | Capability registry, design safeguards, and deterministic method selection; no test execution |
| `execution.py` | Fresh validation, stable method dispatch, and conversion to `AnalysisResult` |
| `interpretation.py` | Deterministic result validation and qualified findings; no numerical backend calls |
| `research_report.py`, `report.py` | Canonical research reports and legacy profile exports |
| `sensitivity.py`, `practical_significance.py` | Explicit robustness scenarios, estimand comparison, and researcher-defined meaningful thresholds |
| `analysis_plan.py`, `study_planning.py` | Analysis-plan records and prospective power or precision calculations |
| `decision_ledger.py`, `provenance.py` | Optional observed-event records and stable local content or dataset references |
| `audit.py`, `reproducibility.py` | Read-only consistency checking and explicit supplied-data replay |
| `completeness.py`, `session.py` | Reporting checklists and interface-safe snapshots |
| `research_assistant.py` | Orchestration only; statistical decisions remain in their owning layers |

## Configuration and result contracts

`ResearchQuestion`, `AnalysisSpecification`, `QuestionDraft`, `Recommendation`,
`AnalysisResult`, `InterpretationResult`, `ResearchReport`, and the planning and provenance
records are typed, documented, and JSON serializable. Schema versions identify wire contracts.
Additive fields preserve existing meanings; an incompatible schema change requires a new version
and a documented migration.

The source result remains the numerical authority. Interpretation and reporting validate and
render recorded values instead of recomputing them. Auditing reconstructs deterministic report
content but does not rerun a statistical method. Replay is a separate explicit operation that
requires supplied data and checks the recorded method and fingerprint when available.

## Scientific boundaries

The workflow starts with objective, estimand, variables, and study design. Design facts such as
independence, pairing, clustering, and causal assignment are never inferred from values. Missing
facts produce structured questions or blockers. Diagnostics can qualify a method but do not
silently change the estimand. Unsupported designs remain explicit rather than being routed to a
convenient calculation.

Paired analyses require an explicit unit identifier and two-condition contrast. The pairing
identity, condition order, estimand, design, and unit-ID column participate in sensitivity
comparability. Declared missing codes observed in selected columns, including the paired unit ID,
block execution until the caller normalizes source data explicitly.

## Data ownership, resource information, and privacy

Construction and analysis preserve the caller's DataFrame. Profiling reports a best-effort deep
memory estimate and advisory warnings for large or wide inputs. It does not sample, truncate,
downcast, mutate, or skip requested calculations in response to those warnings. If memory
estimation itself fails, profiling continues and records that resource status is unknown.

Reports and session snapshots omit participant-level source rows. Dataset fingerprints cover the
full supplied data when supported but contain no observations. Exports escape untrusted text and
protect spreadsheet-readable cells from formula interpretation. Aggregate output can still be
sensitive and remains the researcher's responsibility.

## Compatibility

Existing public imports and established result dictionary keys remain available where
scientifically defensible. Correctness changes are additive when possible and are covered by
regression tests. Legacy dictionary-oriented analyzer and `ReportGenerator` entry points remain
separate from the guided typed workflow.

See [statistical validation](STATISTICAL_VALIDATION.md), [research report schema](RESEARCH_REPORT_SCHEMA.md),
and [provenance and replay](PROVENANCE_AND_REPLAY.md) for detailed contracts.
