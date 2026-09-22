# Architecture and Phase 1 contracts

PyAutoStat has two entry points for dataset profiling. `StatisticalAnalyzer(df).analyze_all()` is the established API. `ResearchAssistant(df).profile()` calls that same method and returns the same dictionary. Construction validates and copies the DataFrame through `StatisticalAnalyzer`; calculations run on `profile()`. Neither entry point infers a research design.

| Module | Responsibility |
| --- | --- |
| `analyzer.py` | Existing profiling, correlations, and independent-group analyses |
| `profiling.py` | Phase 3 dataset profile coordinator, optional data dictionary validation, and structured descriptive additions; delegates numerical calculations to the analyzer |
| `categorical.py` | Existing categorical association calculations |
| `detection.py` | Advisory column type and role hints |
| `insights.py` | Data quality findings from existing analysis dictionaries |
| `report.py` | Existing dictionary, JSON, CSV, and HTML exports |
| `exceptions.py` | Package exceptions; invalid contracts use `InvalidDataError` |
| `research_assistant.py` | Lightweight public facade over the existing analyzer |
| `specifications.py` | Data-independent research question, design, and options contracts |
| `results.py` | Records for missing information, recommendation, diagnostic, and method result |

`ResearchQuestion` records an optional objective (`descriptive`, `compare_groups`, `association`), outcome, predictor/grouping column, estimand, and description. `AnalysisSpecification` combines it with a design (`unknown`, `independent`, `paired`, `repeated`, `clustered`), optional metadata, and `AnalysisOptions`. An absent objective or outcome remains `null`; the design defaults to `unknown`. No design is inferred from column values. `alpha=0.05` and `confidence_level=0.95` are optional defaults for future inference, and `random_seed` defaults to `null`. These are configuration records only; they do not authorize or run analyses. Column existence and design compatibility will be checked when a future workflow evaluates a specification against data.

All configuration models provide `to_dict()` and `from_dict()`. The root `AnalysisSpecification` dictionary includes `schema_version: 1`; its nested records do not repeat the version. `from_dict()` accepts this version only. Records in `results.py` provide `to_dict()` only and also include `schema_version: 1`; they are contracts, not currently generated research outcomes. Enum values serialize as strings, absent values as `null`, and lists in the JSON form. `to_dict()` rejects non-finite and non-JSON values, including DataFrames, instead of emitting invalid JSON. Call `json.dumps(model.to_dict(), allow_nan=False)` for strict JSON. Display formatting is separate from underlying values.

The result contract uses a common envelope (`method_id`, status, sample and excluded-row counts, assumptions, warnings, metadata) and a `values` map for method-specific statistics, estimates, intervals, and directions. Unavailable values are `None`. `MissingInformation` names a field and a question. Recommendation statuses are `ready`, `needs_input`, and `unsupported`; result statuses are `available` and `unavailable`. `Diagnostic` carries an identifier, status, message, and optional values. Future reporting should consume validated results and keep its own format-specific rendering; the existing `ReportGenerator` still consumes legacy dictionaries. No new report generator or report format is defined here.

Future phases can add dataset-aware validation, recommendation, broader execution, interpretation, provenance, and report assembly around these contracts. A GUI should serialize these records and render them without duplicating statistical rules. Phase 2 corrected the legacy analyzer's automatic independent-group selection: it now requires a stated target quantity, and the new facade still does not call it. See [statistical validation](STATISTICAL_VALIDATION.md).

Phase 3 routes both profiling entry points through `DatasetProfiler`. The analyzer still computes the existing statistics, then the profiler attaches categorical summaries, column intelligence, missingness patterns, duplicate overlap, outlier/distribution details, pairwise correlation counts, and profile defaults. Optional declarations are validated before calculations. Numeric identifiers supported by name and observed uniqueness are omitted from profile calculations; `analyzer.numeric_cols` continues to describe raw numeric dtypes and hypothesis testing remains separate. User-declared types and roles can override profiling selection. `ResearchAssistant.complete_case_count(columns)` provides reusable row availability for future analysis intake. The profile remains a plain dictionary; `ReportGenerator.to_json()` is its JSON-safe serialization path, including pandas dtype conversion. No presentation layer recalculates statistics.

Compatibility policy: existing public imports and established result dictionary keys remain intact; Phase 2 adds fields and changes method behavior where needed for correctness. Schema version 1 is the first configuration wire format; a future incompatible schema change must use a new version and explicitly document migration. The new records do not retrofit legacy result dictionaries.
