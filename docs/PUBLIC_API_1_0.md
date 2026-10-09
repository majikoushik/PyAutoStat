# PyAutoStat 1.0 public API

This document defines the human-readable public compatibility boundary for PyAutoStat 1.x. The
machine-readable source of top-level names is `pyautostat.__all__`; the
[API reference](../API_REFERENCE.md) defines signatures and detailed behavior, and the
[API stability policy](API_STABILITY.md) defines how that behavior may evolve.

## Canonical beginner API

`ResearchAssistant` is the canonical entry point. Use `profile()` for dataset-only inspection,
`compare_means()` and `correlate()` for ordinary beginner tasks, and `run()` for explicit advanced
designs and estimands. `ResearchWorkflowResult.brief()`, `type_guidance()`, `explain()`, and
`apa_statement()` are stable result views over the same stored record.

`show`, `save_html`, `save_pdf`, and `save_docx` are `CORE_STABLE`. Optional exporters retain
actionable installation errors when their extras are absent.

## Export-by-export freeze record

Every name below was exported before 1.0, is documented here and in the API reference or a focused
guide, and remains exported in 1.0.0. “Used” means referenced by active documentation, examples,
tests, or package integration. No export was removed during 1.0 consolidation.

### Core stable

| Export | Category | Documented | Used | Intended user | Tier | 1.0 decision and rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `ResearchAssistant` | Integrated workflow | Yes | Yes | All researchers | `CORE_STABLE` | Keep; canonical beginner and guided API. |
| `show` | Terminal presentation | Yes | Yes | All users | `CORE_STABLE` | Keep; canonical inspection path. |
| `save_html` | Static report export | Yes | Yes | All users | `CORE_STABLE` | Keep; core, offline export. |
| `save_pdf` | PDF export | Yes | Yes | Report authors | `CORE_STABLE` | Keep; established optional export. |
| `save_docx` | DOCX export | Yes | Yes | Report authors | `CORE_STABLE` | Keep; established optional export. |

### Direct analysis, planning, and sensitivity

| Export | Category | Documented | Used | Intended user | Tier | 1.0 decision and rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `StatisticalAnalyzer` | Direct analysis | Yes | Yes | Advanced researchers | `ADVANCED_STABLE` | Keep; one engine behind direct and guided interfaces. |
| `StudyPlanner` | Prospective planning | Yes | Yes | Study planners | `ADVANCED_STABLE` | Keep; dataset-free planning API. |
| `StudyPlanningResult` | Planning result | Yes | Yes | Study planners/integrators | `ADVANCED_STABLE` | Keep; stable JSON-safe result. |
| `StatisticalAnalysisPlan` | Analysis-plan record | Yes | Yes | Advanced researchers | `ADVANCED_STABLE` | Keep; stable plan snapshot. |
| `AnalysisPlanStatus` | Plan status enum | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; serialized status vocabulary. |
| `PlanAdherenceResult` | Plan comparison | Yes | Yes | Advanced researchers | `ADVANCED_STABLE` | Keep; stable comparison record. |
| `compare_plan_to_result` | Plan comparison function | Yes | Yes | Advanced researchers | `ADVANCED_STABLE` | Keep; direct equivalent of assistant method. |
| `MeaningfulEffectThreshold` | Threshold specification | Yes | Yes | Researchers | `ADVANCED_STABLE` | Keep; researcher-supplied context contract. |
| `PracticalSignificanceResult` | Threshold result | Yes | Yes | Researchers/integrators | `ADVANCED_STABLE` | Keep; structured non-equivalence assessment. |
| `SensitivitySpecification` | Sensitivity specification | Yes | Yes | Advanced researchers | `ADVANCED_STABLE` | Keep; explicit scenario contract. |
| `SensitivityScenario` | Sensitivity scenario | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; stable scenario record. |
| `SensitivityScenarioResult` | Scenario result | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; preserves every attempt. |
| `SensitivityResult` | Sensitivity result | Yes | Yes | Researchers/integrators | `ADVANCED_STABLE` | Keep; canonical scenario collection. |
| `SensitivityStatus` | Sensitivity status enum | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; serialized status vocabulary. |
| `ScenarioStatus` | Scenario status enum | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; serialized status vocabulary. |
| `Comparability` | Comparability enum | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; protects estimand comparisons. |

### Stable specifications, recommendations, and results

| Export | Category | Documented | Used | Intended user | Tier | 1.0 decision and rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `ResearchWorkflowResult` | Workflow result | Yes | Yes | All programmatic users | `ADVANCED_STABLE` | Keep; canonical integrated record. |
| `WorkflowStatus` | Workflow status enum | Yes | Yes | All programmatic users | `ADVANCED_STABLE` | Keep; workflow state contract. |
| `AnalysisResult` | Analysis result | Yes | Yes | Advanced users/integrators | `ADVANCED_STABLE` | Keep; authoritative calculation record. |
| `AnalysisStatus` | Analysis status enum | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; execution state contract. |
| `AnalysisSpecification` | Analysis specification | Yes | Yes | Advanced users/integrators | `ADVANCED_STABLE` | Keep; serializable execution input. |
| `AnalysisOptions` | Analysis options | Yes | Yes | All programmatic users | `ADVANCED_STABLE` | Keep; immutable defaults and per-call options. |
| `ResearchQuestion` | Research question | Yes | Yes | Advanced users/integrators | `ADVANCED_STABLE` | Keep; explicit objective and estimand. |
| `StudyDesign` | Design enum | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; prevents inferred design facts. |
| `Objective` | Objective enum | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; stable question vocabulary. |
| `Recommendation` | Recommendation record | Yes | Yes | Advanced users/integrators | `ADVANCED_STABLE` | Keep; visible method decision. |
| `RecommendationStatus` | Recommendation status | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; stable state vocabulary. |
| `QuestionDraft` | Draft question | Yes | Yes | Advanced users/integrators | `ADVANCED_STABLE` | Keep; supports explicit continuation. |
| `QuestionStatus` | Draft status enum | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; stable clarification state. |
| `ClarificationQuestion` | Clarification record | Yes | Yes | UI/tool builders | `ADVANCED_STABLE` | Keep; library calls never prompt. |
| `MethodCapability` | Recommendation capability | Yes | Yes | Tool builders | `ADVANCED_STABLE` | Keep; registry-facing typed record. |
| `MethodContract` | Method contract | Yes | Yes | Tool builders/reviewers | `ADVANCED_STABLE` | Keep; documented scientific metadata. |
| `METHOD_CONTRACTS` | Contract registry | Yes | Yes | Tool builders/reviewers | `ADVANCED_STABLE` | Keep; established read-only registry surface. |

### Audit, reproducibility, completeness, and sessions

| Export | Category | Documented | Used | Intended user | Tier | 1.0 decision and rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `DecisionLedger` | Local decision record | Yes | Yes | Advanced users/integrators | `ADVANCED_STABLE` | Keep; explicit local provenance. |
| `AuditFinding` | Audit finding | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; typed consistency finding. |
| `AuditResult` | Audit result | Yes | Yes | Researchers/integrators | `ADVANCED_STABLE` | Keep; stable consistency result. |
| `StatisticalResultAuditor` | Auditor engine | Yes | Yes | Advanced users | `ADVANCED_STABLE` | Keep; direct audit access. |
| `ReproducibilityRecord` | Replay metadata | Yes | Yes | Researchers/integrators | `ADVANCED_STABLE` | Keep; versioned reproducibility record. |
| `ReproductionOutcome` | Replay result | Yes | Yes | Researchers/integrators | `ADVANCED_STABLE` | Keep; explicit comparison outcome. |
| `reproduce` | Explicit replay | Yes | Yes | Researchers/integrators | `ADVANCED_STABLE` | Keep; replay never occurs implicitly. |
| `CompletenessItem` | Completeness item | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; typed checklist item. |
| `ReportingCompletenessResult` | Completeness result | Yes | Yes | Report authors | `ADVANCED_STABLE` | Keep; structural reporting check. |
| `assess_reporting_completeness` | Completeness function | Yes | Yes | Report authors | `ADVANCED_STABLE` | Keep; direct equivalent of assistant method. |
| `ResearchSessionSnapshot` | Session snapshot | Yes | Yes | UI/tool builders | `ADVANCED_STABLE` | Keep; UI-independent state. |
| `build_session_snapshot` | Snapshot builder | Yes | Yes | UI/tool builders | `ADVANCED_STABLE` | Keep; direct construction API. |
| `capability_payload` | Capability metadata | Yes | Yes | UI/tool builders | `ADVANCED_STABLE` | Keep; established integration helper. |

### Presentation, export, and bundles

| Export | Category | Documented | Used | Intended user | Tier | 1.0 decision and rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `ResearchReport` | Canonical report | Yes | Yes | Researchers/integrators | `ADVANCED_STABLE` | Keep; structured report source. |
| `PresentationView` | Presentation model | Yes | Yes | Renderer/tool builders | `ADVANCED_STABLE` | Keep; UI-independent display record. |
| `UnsupportedPresentationError` | Presentation exception | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; targeted adapter error. |
| `to_html` | In-memory HTML | Yes | Yes | Report builders | `ADVANCED_STABLE` | Keep; canonical static rendering. |
| `to_pdf` | In-memory PDF | Yes | Yes | Report builders | `ADVANCED_STABLE` | Keep; optional binary rendering. |
| `to_docx` | In-memory DOCX | Yes | Yes | Report builders | `ADVANCED_STABLE` | Keep; optional binary rendering. |
| `to_interactive_html` | Interactive HTML | Yes | Yes | Report builders | `ADVANCED_STABLE` | Keep; optional Plotly rendering. |
| `save_interactive_html` | Interactive HTML save | Yes | Yes | Report authors | `ADVANCED_STABLE` | Keep; established file API. |
| `to_static_figures` | In-memory figures | Yes | Yes | Report builders | `ADVANCED_STABLE` | Keep; canonical figure artifacts. |
| `save_static_figures` | Figure save | Yes | Yes | Report authors | `ADVANCED_STABLE` | Keep; established file API. |
| `StaticFigureArtifact` | Figure record | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; typed binary artifact. |
| `BUNDLE_SCHEMA_VERSION` | Bundle schema constant | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; versioned archive contract. |
| `BundleOptions` | Bundle options | Yes | Yes | Report builders | `ADVANCED_STABLE` | Keep; typed archive configuration. |
| `BundleVerificationResult` | Bundle verification | Yes | Yes | Researchers/integrators | `ADVANCED_STABLE` | Keep; integrity result. |
| `save_bundle` | Bundle save | Yes | Yes | Report authors | `ADVANCED_STABLE` | Keep; archival file API. |
| `to_bundle` | In-memory bundle | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; archival bytes API. |
| `verify_bundle` | Bundle verification | Yes | Yes | Researchers/integrators | `ADVANCED_STABLE` | Keep; checks manifest integrity. |

### Interpretation and deterministic narration

| Export | Category | Documented | Used | Intended user | Tier | 1.0 decision and rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `InterpretationEngine` | Interpretation engine | Yes | Yes | Advanced users | `ADVANCED_STABLE` | Keep; independent rule engine. |
| `InterpretationFinding` | Finding record | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; stable coded finding. |
| `InterpretationResult` | Interpretation result | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; JSON-safe interpretation. |
| `InterpretationStatus` | Interpretation status | Yes | Yes | Integrators | `ADVANCED_STABLE` | Keep; stable state vocabulary. |
| `assumption_grade` | Narration helper | Yes | Yes | Advanced presenters | `ADVANCED_STABLE` | Keep; documented side-effect-free helper. |
| `coefficient_of_variation_narrative` | Narration helper | Yes | Yes | Advanced presenters | `ADVANCED_STABLE` | Keep; documented deterministic helper. |
| `column_story` | Narration helper | Yes | Yes | Advanced presenters | `ADVANCED_STABLE` | Keep; documented deterministic helper. |
| `crosstab_narrative` | Narration helper | Yes | Yes | Advanced presenters | `ADVANCED_STABLE` | Keep; documented deterministic helper. |
| `dataset_opening` | Narration helper | Yes | Yes | Advanced presenters | `ADVANCED_STABLE` | Keep; documented deterministic helper. |
| `effect_narrative` | Narration helper | Yes | Yes | Advanced presenters | `ADVANCED_STABLE` | Keep; reads stored effect values. |
| `executive_summary` | Narration helper | Yes | Yes | Report builders | `ADVANCED_STABLE` | Keep; canonical format-neutral summary. |
| `frequency_narrative` | Narration helper | Yes | Yes | Advanced presenters | `ADVANCED_STABLE` | Keep; documented deterministic helper. |
| `hypothesis_verdict` | Narration helper | Yes | Yes | Advanced presenters | `ADVANCED_STABLE` | Keep; documented stored-result verdict. |
| `insight_narrative` | Narration helper | Yes | Yes | Advanced presenters | `ADVANCED_STABLE` | Keep; documented deterministic helper. |
| `interval_verdict` | Narration helper | Yes | Yes | Advanced presenters | `ADVANCED_STABLE` | Keep; preserves interval semantics. |
| `percentile_narrative` | Narration helper | Yes | Yes | Advanced presenters | `ADVANCED_STABLE` | Keep; documented deterministic helper. |
| `recommendation_rationale` | Narration helper | Yes | Yes | Tool builders | `ADVANCED_STABLE` | Keep; explains stored recommendation. |
| `sensitivity_verdict` | Narration helper | Yes | Yes | Advanced presenters | `ADVANCED_STABLE` | Keep; preserves comparability rules. |

### Detection, compatibility, and exceptions

| Export | Category | Documented | Used | Intended user | Tier | 1.0 decision and rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `detect_column_types` | Advisory detection | Yes | Yes | Advanced users | `ADVANCED_STABLE` | Keep; explicit advisory helper. |
| `suggest_column_roles` | Advisory detection | Yes | Yes | Advanced users | `ADVANCED_STABLE` | Keep; does not transform data or select a test. |
| `InsightEngine` | Legacy interpretation | Yes | Yes | Existing 0.1 users | `COMPATIBILITY_STABLE` | Keep without warning; modern path documented. |
| `ReportGenerator` | Legacy reporting | Yes | Yes | Existing 0.1 users | `COMPATIBILITY_STABLE` | Keep without warning; modern path documented. |
| `PyAutoStatError` | Base exception | Yes | Yes | All programmatic users | `ADVANCED_STABLE` | Keep; catch-all package exception. |
| `ColumnNotFoundError` | Input exception | Yes | Yes | All programmatic users | `ADVANCED_STABLE` | Keep; actionable column failure. |
| `InsufficientDataError` | Data exception | Yes | Yes | All programmatic users | `ADVANCED_STABLE` | Keep; numerical insufficiency. |
| `InsufficientGroupsError` | Group exception | Yes | Yes | All programmatic users | `ADVANCED_STABLE` | Keep; explicit group insufficiency. |
| `InvalidDataError` | Validation exception | Yes | Yes | All programmatic users | `ADVANCED_STABLE` | Keep; established validation contract. |
| `InvalidTestError` | Method exception | Yes | Yes | Advanced users | `ADVANCED_STABLE` | Keep; incompatible method request. |
| `ReportError` | Reporting exception | Yes | Yes | Report authors | `ADVANCED_STABLE` | Keep; export/setup failures. |

## What is not public

Modules and names absent from `pyautostat.__all__` are implementation details unless explicitly
documented otherwise. This includes internal execution adapters, numerical inversion/bootstrap
helpers, presentation adapters/renderers, HTML/DOCX/PDF component functions, bundle assembly and
safety helpers, and private attributes on public objects. Importing them from a submodule does not
create a 1.x compatibility guarantee.

## Compatibility API status

`InsightEngine` and `ReportGenerator` are retained, tested, and supported as
`COMPATIBILITY_STABLE`; 1.0.0 emits no deprecation warning. See
[Migration to 1.0](MIGRATION_TO_1_0.md) for the recommended integrated replacements. Any future
deprecation must follow [the deprecation policy](API_STABILITY.md) and removal cannot occur before
2.0.0.
