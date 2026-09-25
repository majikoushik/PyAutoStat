# PyAutoStat API reference

This page describes the public API in `pyautostat`. The [README](README.md) has a short start-to-finish example; the [examples guide](examples/README.md) runs every feature and shows its output.

## Imports

```python
from pyautostat import (
    AnalysisOptions,
    AnalysisResult,
    AnalysisSpecification,
    AnalysisStatus,
    InsightEngine,
    MeaningfulEffectThreshold,
    Objective,
    QuestionDraft,
    Recommendation,
    RecommendationStatus,
    ReportGenerator,
    ResearchAssistant,
    ResearchWorkflowResult,
    ResearchQuestion,
    StatisticalAnalyzer,
    StatisticalAnalysisPlan,
    StudyPlanner,
    ReportingCompletenessResult,
    ResearchSessionSnapshot,
    StudyDesign,
    SensitivitySpecification,
    WorkflowStatus,
    detect_column_types,
    suggest_column_roles,
)
```

All documented exception classes are also exported from `pyautostat`.

## `ResearchAssistant` common workflows

```python
assistant = ResearchAssistant(df)
profile = assistant.profile()
print(profile["overview"])
print(profile["resource_info"])
```

The assistant validates and copies the DataFrame using `StatisticalAnalyzer`; `profile()` returns
the same statistical dictionary as `analyze_all()` plus structured profiling metadata. It needs no
research question. `resource_info` contains deep-memory and correlation-width performance
advisories without sampling, truncating, modifying, or skipping calculations.

### Integrated guided workflow

```python
workflow = assistant.run(
    objective="compare_groups", outcome="score", predictor="group",
    estimand="mean", design="independent",
    variable_types={"score": "continuous"},
)
```

`run()` accepts the same raw question fields as `prepare_question()` plus `draft` or
`specification` for continuation, `include_profile=False`, `audit=True`, `fingerprint=True`,
`title=None`, and `include_figures=False`. A draft/specification is mutually exclusive with raw
question arguments. Invalid API types and conflicting inputs raise `InvalidDataError`.

`ResearchWorkflowResult` exposes `status`, `specification`, `draft`, `recommendation`, `analysis`,
`interpretation`, `report`, `audit`, `reproducibility`, optional `profile`,
`missing_information`, `blockers`, and `warnings`. Statuses are `completed`, `partial`,
`needs_input`, `data_limited`, `unsupported`, and `failed`. A stage that did not run stays `None`.
`to_dict()` and `to_json()` use workflow schema version 1, reject nonfinite JSON values, and do
not embed the source DataFrame.

The default successful path performs one statistical execution. Later stages consume that result;
reproducibility-record creation does not replay it. Default audit renders and checks HTML,
Markdown, JSON, and CSV in memory. `audit=False` returns a partial workflow with `audit=None`.
`include_profile=True` requests one dataset profile for an inferential workflow; descriptive
execution reuses its existing profile. No files are written. See
[`docs/CAPABILITIES.md`](docs/CAPABILITIES.md) for the method and failure-mode matrices.

### Continue after `needs_input`

When an essential scientific fact is absent, `run()` returns a structured request and does not
execute a test:

```python
pending = assistant.run(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    estimand="mean",
)
assert pending.status == "needs_input"

revised = assistant.update_question(pending.draft, design="independent")
workflow = assistant.run(draft=revised)
```

The draft retains confirmed values. Its questions explain each missing field, and continuation
passes through the same validation and recommendation rules as a new request.

### Sensitivity analysis

```python
scenario = SensitivitySpecification(
    name="pooled variance",
    specification=workflow.analysis.specification,
    method_id="student_t",
    rationale="Assess the pooled-variance assumption.",
    planning_status="planned",
    assumptions=("Equal population variances",),
)
sensitivity = assistant.sensitivity_analysis(
    workflow.analysis,
    scenarios=[scenario],
)
```

`sensitivity_analysis(result, *, scenarios) -> SensitivityResult` requires an available base
result and a nonempty ordered list of uniquely named `SensitivitySpecification` records. A
scenario stores an `AnalysisSpecification`, optional explicit `method_id`, rationale,
`planning_status` (`planned`, `exploratory`, or `unknown`), and explicit assumptions. With no
method ID, the ordinary deterministic recommendation for that scenario specification is used.
Student t and standard ANOVA require an explicit equal-population-variance assumption. No method
is chosen from a p-value.

`SensitivityResult.status` is `complete`, `partial`, or `unavailable`. Its
`scenario_results` retain every supplied scenario in order with `completed`, `unavailable`,
`incompatible`, or `failed` execution status and `same_estimand`, `different_estimand`,
`incompatible`, or `unavailable` comparability. Each record includes actual method, stable
estimate quantity, estimate, effect quantity/value, CI, p-value, sample/exclusion counts, contrast,
warnings, error, and the aggregate `AnalysisResult` in memory. Serialization excludes the
in-memory result object and source DataFrame.

Same-estimand comparisons record estimate difference, direction, interval availability and
descriptive overlap, sample-size change, and a relative change only when the base estimate is not
near zero. Reversed group orientation is normalized in comparison fields with an explicit
transformation note; raw scenario values remain unchanged. Different-estimand analyses, including
Welch mean difference versus Mann–Whitney rank distribution, have no direct estimate-change
calculation. There is no p-value ordering, scenario selection, robustness score, or automatic
changed-data analysis.

### Practical significance

```python
threshold = MeaningfulEffectThreshold(
    quantity="mean_difference",
    minimum_magnitude=5,
    direction="two_sided",
    unit="points",
    rationale="Researcher-defined decision threshold.",
)
practical = assistant.practical_significance(
    workflow.analysis,
    threshold=threshold,
)
```

`MeaningfulEffectThreshold` supports signed `mean_difference`, `cohens_d`, `pearson_r`, and
`rank_biserial` quantities and nonnegative `cramers_v`, `eta_squared`, and `epsilon_squared`.
Magnitude must be finite and nonnegative. Directions are `two_sided`, `positive`, `negative`, or
`nonnegative` as appropriate. `equivalence` and `noninferiority` are recognized only to return an
explicit unsupported result; no TOST or noninferiority test is implemented. Unit and rationale are
optional preserved metadata. A known result unit must match the threshold unit.
For a directional paired mean-difference threshold, supply
`contrast_order=(first_condition, second_condition)` to identify the intended first-minus-second
contrast. A missing or mismatched orientation returns an unavailable assessment rather than
reusing the signed threshold. Two-sided magnitude thresholds remain orientation-invariant.

`practical_significance(result, *, threshold) -> PracticalSignificanceResult` selects only the
named existing quantity and its recorded interval. It reports `point_estimate_relation`,
`confidence_interval_relation`, `uncertainty_status`, and null-hypothesis
`statistical_significance` separately. Missing CIs return `partial`; unavailable analyses return
`unavailable`; formal equivalence/noninferiority requests return `unsupported`. An interval wholly
inside a two-sided negligible region is described without claiming formal equivalence. Generic
small/medium/large labels are not used as meaningful-effect criteria.

Both models serialize with schema version 1. `assistant.report(..., sensitivity=...,
practical_significance=...)`, `assistant.audit(...)`, and
`assistant.reproducibility_record(...)` accept the optional sensitivity and threshold records. See
[`docs/ROBUSTNESS_AND_PRACTICAL_SIGNIFICANCE.md`](docs/ROBUSTNESS_AND_PRACTICAL_SIGNIFICANCE.md).
`SensitivitySpecification.from_dict(...)` and `MeaningfulEffectThreshold.from_dict(...)` restore
their validated configuration records.

### Question intake and specification details

```python
draft = assistant.prepare_question(
    objective="compare_groups", outcome="score", predictor="group"
)
print(draft.status)  # needs_input
print([item.field for item in draft.questions])  # estimand, design
draft = assistant.update_question(draft, estimand="mean", design="independent")
assert draft.status == "ready"

saved = draft.specification.to_dict()
restored = AnalysisSpecification.from_dict(saved)
rechecked = assistant.prepare_question(specification=restored)
```

`prepare_question()` accepts optional `objective`, `outcome`, `predictor`, `design`, `estimand`, `description`, `options`, `data_dictionary`, `variable_types`, and `specification`. Objectives are `descriptive`, `compare_groups`, and `association`. For comparison, `predictor` names the group or condition column. For association it names the second variable. A descriptive request needs none of these optional inferential details. Comparison needs both variables, target, and design. Association needs two variables and the dependence structure across rows; this is distinct from two x/y values occupying one row. `unknown` design remains unresolved. `mean` and `distribution` are distinct comparison targets. Other explicitly named targets can be recorded but later support is not guaranteed.

The `QuestionDraft` has `specification`, `status` (`ready`, `needs_input`, `data_limited`, `unsupported`), `missing_information`, `questions`, `warnings`, `blockers`, `variable_suggestions`, and `availability`. `questions` carry `field`, `question`, `explanation`, `input_type`, `required`, and options with stable `value` and display `label`; `to_dict()` is JSON-compatible. A future GUI should render these records and send selected values through `update_question()`. `availability` reports total, complete, and missing-relevant rows using complete-case counts. It is an availability summary, not a missing-data treatment. Declared missing codes remain ordinary observed values until the caller normalizes them. All-missing selected columns, no complete rows, and a group with fewer than two observed categories produce `data_limited` with blockers. An unsupported objective raises an actionable error; `unsupported` is reserved for future requests that cannot be represented. Constant variables produce warnings. Invalid column names and incompatible declarations raise package errors.

`update_question(draft, **changes)` preserves confirmed answers, reconstructs a new draft, and leaves the old draft unchanged on invalid input. When the objective changes, it clears the previous predictor, target, and design; switching to descriptive also clears the old outcome. Supply new role selections explicitly. `profile(data_dictionary=...)` makes a copied declaration available to later question preparation; an explicit `data_dictionary` or `variable_types` correction takes precedence. The builder reuses dataset variable intelligence without running full profiling, hypothesis tests, or recommendations.

`ready` means question fields and basic selected-data checks are complete. It does not establish an appropriate method, valid assumptions, or a certified analysis plan. Paired designs require an explicit unit-ID column; repeated and clustered designs remain representable but unsupported for execution.

### Method recommendation

```python
draft = assistant.prepare_question(
    objective="compare_groups", outcome="score", predictor="group",
    estimand="mean", design="independent",
)
recommendation = assistant.recommend_test(draft)
# Alternatively: assistant.recommend_test(specification=draft.specification)
print(recommendation.status, recommendation.method_id, recommendation.rationale)
```

`recommend_test(draft=None, *, specification=None)` accepts exactly one `QuestionDraft` or `AnalysisSpecification`. A direct specification, and any supplied draft, are revalidated against the assistant's current DataFrame and question-availability rules. No full profile or hypothesis test is run. Unknown essential information returns `needs_input` with `missing_information` and GUI-ready `questions`; a data-limited draft returns `unsupported` with its blockers. An incompatible design, target, measurement type, or hard computational requirement returns `unsupported`. `ready` means a compatible existing operation can be called later; it does not certify uncheckable assumptions.

| Objective and target | Conditions | Result |
| --- | --- | --- |
| Description | Valid DataFrame | `dataset_profile` (`ResearchAssistant.profile()`), descriptive only |
| Group comparison, mean | Two independent groups, quantitative outcome, at least two usable outcomes per group, representable spread | `welch_t`; Student t is an explicit, equal-variance alternative |
| Group comparison, distribution | Two independent groups, ordered numeric outcome, at least two usable outcomes per group | `mann_whitney_u`; tied small samples have an approximation warning |
| Group comparison, distribution | Three or more independent groups, ordered numeric outcome, at least five usable outcomes per group | `kruskal_wallis`; no post-hoc comparisons |
| Group comparison, mean | Three or more independent groups | `unsupported`; standard one-way ANOVA is a conditional explicit option, while variance-robust Welch ANOVA is unavailable |
| Association, linear | Two quantitative variables, independent observational pairs, at least three complete varying pairs | `pearson_correlation`, including the existing pairwise p-value when numerically valid |
| Association, monotonic | Two varying ordered numeric variables | `unsupported` for inference; Spearman/Kendall coefficients are listed as `coefficient_only` alternatives |
| Association, categorical independence | Two categorical variables, independent observations, at least two categories per axis, expected counts at least five in every cell | `pearson_chi_square`; sparse tables return `unsupported` |

Numeric association with no specified relationship target requests one clarification. A numeric/categorical association requests confirmation before changing the research objective. A paired mean question without a unit ID returns `needs_input`; compatible two-condition paired data select `paired_t`. Repeated and clustered designs return `unsupported`; an unknown essential design returns `needs_input`. A declared missing code still present among selected values blocks a finalized recommendation until the caller normalizes the data and rebuilds the assistant. The engine never recodes or excludes those values itself.

`Recommendation` retains its version 1 envelope and original fields. Additive fields are `method_availability` (`runnable`, `coefficient_only`, or `unavailable`), `decision_trace` (ordered `{key, value, reason}` entries), `alternatives` (method ID, name, availability, reason), `context` (objective, target, design, selected analytical types, complete-case availability and relevant feasibility facts), and `questions` (question-builder-compatible clarification dictionaries). `required_assumptions` and `context.assumption_checks` disclose researcher-confirmed facts, checkable feasibility, and conditions requiring review. `to_dict()` is JSON-compatible and includes no raw rows or invented p-values. The small `METHOD_CAPABILITIES` registry in `pyautostat.recommendation` documents actual backend availability. An explicit preferred-method override is deferred to avoid changing the specification schema; existing explicit analyzer calls remain available to experts.

### Statistical execution

```python
result = assistant.analyze(draft)
# Or: result = assistant.analyze(specification=draft.specification)
print(result.status, result.method_id)
print(result.values["primary_estimate"], result.values["p_value"])
print(result.metadata["sample"], result.metadata["group_order"])
```

`analyze(draft=None, *, specification=None)` accepts exactly one completed `QuestionDraft` or `AnalysisSpecification`. It rebuilds the question against the assistant's copied DataFrame and obtains a fresh recommendation. Only `ready` and `runnable` selections dispatch. Incomplete, unsupported, or known backend numerical failures return `AnalysisResult(status="unavailable")` with a reason and no fabricated values. Invalid specification fields or nonexistent columns raise the existing package exceptions. Unexpected programming errors are not hidden.

| Selected method ID | Existing numerical source | Main outputs |
| --- | --- | --- |
| `dataset_profile` | `StatisticalAnalyzer.analyze_all()` | JSON-safe profile; no hypothesis statistic or p-value |
| `welch_t` | `hypothesis_tests(test_type="ttest", equal_var=False)` | Mean difference, Welch statistic/df/p, Cohen's d, analytical mean-difference CI, optional bootstrap d CI |
| `mann_whitney_u` | `hypothesis_tests(test_type="mannwhitney")` | First-group U, two-sided p, rank-biserial effect and optional bootstrap CI |
| `kruskal_wallis` | `hypothesis_tests(test_type="kruskal")` | H, df, p, rank epsilon-squared and optional bootstrap CI |
| `pearson_correlation` | Pair-only `analyze_all()` correlation profile | Pearson r and its pairwise p-value; no CI |
| `pearson_chi_square` | `categorical_association()` | Chi-square, df, p, observed/expected table, Cramer's V and optional bootstrap CI |

The registry also describes Student's pooled t-test and standard one-way ANOVA as runnable **legacy explicit calculations**, but the guided recommender does not select them automatically. They remain available through `StatisticalAnalyzer.hypothesis_tests()`; `analyze()` never substitutes them for a Welch or unsupported multi-group mean request. The guided engine supports `paired_t` only for an explicit unit ID and exactly two conditions. Spearman/Kendall inference, repeated designs with more than two conditions, clustered methods, Welch ANOVA, and sparse-table exact tests are unavailable.

`AnalysisResult` retains its version 1 common envelope (`method_id`, `status`, `sample_size`, `excluded_rows`, `values`, `assumptions`, `warnings`, `metadata`). Additive `specification` and `recommendation` fields retain the actual validated request and selected method; `to_dict()` serializes both. `values` uses `test_statistic`, `degrees_of_freedom`, `p_value`, `primary_estimate`, `estimate_name`, `estimate_unit`, `effect_size`, and `confidence_interval`. Each interval names its `quantity`, `method`, `level`, and bounds. `None` means the backend provided no supported value. The descriptive path uses `values.profile` and explicit `None` inferential fields. `metadata.sample` records original, analyzed and excluded rows, with group sizes or effective pair count where relevant. `metadata.group_order` follows the backend's first-observed order. For two-group tests, `metadata.contrast` defines first minus second; the mean difference, Cohen's d, U orientation and rank-biserial sign use this order. `metadata.diagnostics` preserves backend assumption results; `warnings` combines intake, recommendation and backend warnings without duplicates.

Group and categorical effect intervals use the existing 499-resample bootstrap with effective seed 0 when no seed was specified; this default and any explicit seed are recorded in metadata and diagnostics. The specification's alpha and confidence level remain available through `result.specification.options`; alpha is not used to alter the numerical p-value. No missing rows are imputed, no outliers are removed, and declared missing codes still block execution until normalized externally. JSON export is `json.dumps(result.to_dict(), allow_nan=False)`.

### Deterministic interpretation

```python
result = assistant.analyze(draft)
interpretation = assistant.interpret(result)
print(interpretation.summary)
print(interpretation.status, [finding.code for finding in interpretation.findings])
payload = interpretation.to_dict()  # json.dumps(payload, allow_nan=False)
```

`interpret(result: AnalysisResult) -> InterpretationResult` reads the completed analysis record only; it runs no statistical test or report writer. `InterpretationEngine().interpret(result)` is the independently usable rule engine. The result has `status` (`available`, `partial`, `unavailable`), `execution_status`, `method_id`, `summary`, `method_explanation`, `hypothesis_interpretation`, `effect_interpretation`, `uncertainty_interpretation`, `assumption_notes`, `limitations`, `conclusion`, `warnings`, `metadata`, and `findings`. Each `InterpretationFinding` has a stable `code`, display `message`, and `supporting_fields` pointing to the source result. `to_dict()` is JSON compatible and leaves the original `AnalysisResult` unchanged.

Supported guided results are `dataset_profile`, `welch_t`, `mann_whitney_u`, `kruskal_wallis`, `pearson_correlation`, and `pearson_chi_square`. Templates also accept valid `student_t` and `one_way_anova` adapter results; the guided selector does not choose these. Spearman and Kendall are coefficient-only in legacy profiling, so there is no successful guided inferential interpretation for them. Unrecognized or unavailable results return an unavailable interpretation. A missing p-value, effect, or primary confidence interval gives a partial interpretation, preserving factual components without inventing the missing result. Pearson's current result has no CI, so its interpretation is normally partial.

The decision rule is `p < result.specification.options.alpha`; equality does not reject. It uses the original numeric p-value and formats very small values separately; computational zero displays as `p < 0.001` with a numerical warning. The engine does not claim multiplicity adjustment, equivalence, causation, or practical importance. Two-group directions follow `metadata.contrast`; confidence intervals retain their own `quantity`, `method`, and `level`. Finite ordered percentile-bootstrap bounds may exclude the original point estimate; containment is required for the analytical t mean-difference interval. The analytical t interval is checked against a two-sided p-value only when its level matches `1-alpha`. Apparent disagreement produces a warning and partial status. A valid raw mean difference remains interpretable when Cohen's d is missing or inconsistent, with partial status and a warning; conflicting signs invalidate the d interpretation. Bootstrap effect intervals are not treated as interchangeable with the analytical mean-difference interval. Assumption diagnostics describe rejection or non-rejection, never proof; source warnings and excluded-row counts remain visible.

### Research reports

```python
result = assistant.analyze(draft)
report = assistant.report(result)  # matching interpretation is generated if omitted
print(report.status, report.to_html())
markdown = report.to_markdown()
json_text = report.to_json()
csv_tables = report.to_csv_tables()  # stable table-ID to CSV-text mapping
```

`report(result, *, interpretation=None, sensitivity=None, practical_significance=None, title=None, include_figures=False) -> ResearchReport` does not rerun an analysis or write a file. Supplied interpretations must exactly match the deterministic interpretation for this result; mismatches raise `ReportError`. The snapshot exposes `status` (`complete`, `partial`, `unavailable`), `title`, and `to_dict()`. It includes the source specification/result, matching interpretation, structured research question, dataset, Methods, diagnostics, Results, interpretation, source-linked tables, optional histogram-bin specifications, warnings, and limitations. Contradictory sample counts raise `ReportError`. An unavailable result has no displayed numerical findings, even if its envelope contains stale values. Invalid effect measures and intervals remain unavailable in reader-facing sections. Without sensitivity or practical-significance records the payload remains report schema version 1; optional follow-up content uses additive report schema version 2.

`to_html()` returns self-contained escaped static HTML; `to_markdown()` escapes user syntax; `to_json()` preserves JSON-safe raw numbers; `to_csv_tables()` returns a dictionary of independent CSV strings. Cells beginning with formula-like prefixes after whitespace are prefixed with an apostrophe only in CSV output; numeric cells remain numeric. `save_html(path)`, `save_markdown(path)`, `save_json(path)`, and `save_csv_tables(directory)` write only to explicit paths and reject existing files unless `overwrite=True`. Filenames for CSV tables are stable IDs, never derived from report titles or category labels. The report omits categorical identifier labels from descriptive profiles and does not include the complete input DataFrame. Small aggregate groups can still disclose information. See [the schema and method matrix](docs/RESEARCH_REPORT_SCHEMA.md) and [the runnable example](examples/research_report_example.py). The legacy `ReportGenerator` continues to accept its original dictionary inputs.

### Provenance, audit, and replay

```python
from pyautostat import reproduce

assistant.enable_tracking()  # only subsequent actions are recorded
draft = assistant.prepare_question(...)
result = assistant.analyze(draft)
report = assistant.report(result)
audit = assistant.audit(report)
record = assistant.reproducibility_record(result)
replay = reproduce(record, data=df)  # explicit supplied-data rerun
```

`decision_ledger` is `None` when tracking is off. Tracked events have sequence IDs, local software timestamps, before/after states for revisions, optional researcher-supplied reasons, and content references. `update_question(draft, reason="...")` accepts an optional decision reason. `declare_planning("planned" | "exploratory" | "unknown")` is an explicit declaration; the default is `unknown`. `DecisionLedger.import_result(result)` records only an import and marks prior history unavailable. No ledger or hash authenticates an external preregistration or timestamp.

`audit(report, *, result=None, sensitivity=None, practical_significance=None, exports=None) -> AuditResult` compares the report with its captured sources or explicitly supplied originals. Findings have stable codes and field paths, including threshold, relation, estimate, omission, status, and comparability mismatches. `passed`, `failed`, and `incomplete` distinguish agreement, contradiction, and checks that could not be performed. A directly reconstructed report without a supplied source result is incomplete. If `exports` is omitted, all four current formats are rendered and checked; supplied content is inspected as provided. HTML and Markdown checks compare the exact canonical rendering, not arbitrary edited prose semantics. The auditor never reruns a statistical test or sensitivity scenario. Its pass status does not establish scientific validity.

`reproducibility_record(result, *, fingerprint=True, sensitivity=None, practical_significance=None) -> ReproducibilityRecord` stores a JSON-safe specification, method, restricted expected-result projection, actual runtime versions, recorded seed and bootstrap configuration, and an optional hash of the assistant's current DataFrame. Base-only records remain schema version 1. Optional sensitivity configuration produces schema version 2 with ordered scenarios, actual methods/statuses/seeds, fingerprint metadata, and the meaningful threshold. `ReproducibilityRecord.from_dict(...)` reloads this metadata. `record.save_package(path, data_reference=None, overwrite=False)` writes a metadata-only ZIP to an explicit path; no raw data or script are included. `reproduce(record, *, data=df, allow_changed_data=False) -> ReproductionOutcome` checks the fingerprint, revalidates the recorded base method, executes only on explicit request, and compares actual numeric fields under the documented tolerance. It does not automatically replay sensitivity scenarios. A changed dataset is a mismatch by default; an allowed changed-data rerun remains labelled as such. A missing fingerprint cannot establish same-data reproduction. See [the precise fingerprint, comparison, privacy, and export policy](docs/PROVENANCE_AND_REPLAY.md).

`to_dict()` and `from_dict()` are supported by `ResearchQuestion`, `AnalysisOptions`, and `AnalysisSpecification`. The root specification uses `schema_version: 1` when it has no data dictionary and `schema_version: 2` when `data_dictionary` is present. Version 1 payloads round-trip unchanged; version 2 adds that field without overloading version 1's text-only `variable_metadata`. A legacy caller can keep using `variable_metadata` for descriptions. The data dictionary is authoritative for analytical types and roles. See [the architecture document](docs/ARCHITECTURE.md) for migration details. `pyautostat.results` exposes `MissingInformation`, `Recommendation`, `Diagnostic`, and `AnalysisResult` as serializable records. No recommendation, inferential result, or report is manufactured by question preparation.

## `StatisticalAnalyzer`

### Construction and full analysis

```python
analyzer = StatisticalAnalyzer(df)
analysis = analyzer.analyze_all()
# Both entry points also accept data_dictionary=..., histogram_bins=20,
# and include_row_positions=False as optional keyword arguments.
```

`df` must be a nonempty pandas DataFrame with unique, nonempty string column names and scalar values. Numeric values must be finite and real; missing values are allowed. The analyzer copies the DataFrame and exposes `df`, `numeric_cols`, `categorical_cols`, and `all_results`.

`analyze_all()` returns a dictionary with these sections:

| Key | Contents |
| --- | --- |
| `overview` | Shape, row and column counts, memory use, column names and dtypes |
| `descriptive` | Per-numeric-column count, center, spread, quantiles, skewness and kurtosis |
| `normality` | Shapiro-Wilk, D'Agostino-Pearson and Anderson-Darling when their preconditions are met |
| `outliers` | IQR, Z-score and median absolute deviation summaries |
| `correlation` | Pearson, Spearman and Kendall matrices; pairwise Pearson p-values |
| `missing_data` | Missing counts and percentages by column and overall |
| `data_quality` | Completeness, per-column uniqueness and duplicate rows |
| `distributions` | Skewness and kurtosis descriptions, range and a histogram-based bimodality hint |
| `column_roles` | Advisory roles derived from column names |
| `column_types` | Advisory types and missingness hints |
| `histograms` | Precomputed bin edges and counts for numeric columns |
| `analysis_warnings` | Records with `code`, `section`, `column` and `message` for skipped or undefined calculations |
| `categorical_summary` | Nonmissing/missing counts, observed categories, tied modes, top 20 frequencies and percentages of nonmissing observations |
| `variable_intelligence` | Observed dtype, advisory analytical type and role, evidence, declaration sources and warnings |
| `data_dictionary` | Validated, copied column declarations supplied by the caller; empty by default |
| `resource_info` | Deep memory estimate, analytical numeric width, advisory resource level, correlation size, and structured performance warnings |
| `profile_metadata` | Schema version and disclosed binning, missingness, row-position and outlier defaults |

An unavailable numeric result is `None`. Some tests are skipped for all-missing, constant or short columns; inspect `analysis_warnings`. For D'Agostino-Pearson at 8-19 observations, a finite result is retained with a small-sample approximation warning; other numerical warnings or nonfinite output make it unavailable. Anderson-Darling is omitted if SciPy supplies an unusable critical-value grid, including nonpositive values sometimes returned for very small samples. Correlation uses pairwise nonmissing observations. Only Pearson pairs include p-values.

### Profile details and optional declarations

`overview` retains original pandas dtype objects for Python callers and adds suggested-type counts, datetime/Boolean counts, memory bytes, missing cells and exact duplicates. Use `ReportGenerator(profile).to_json()` for JSON-safe export; it converts dtype objects and unavailable values. `missing_data` adds `rows_with_missing`, `completely_missing_rows`, `complete_rows`, top 10 `common_patterns`, column `available_count`, and co-missing pair counts. Cell counts and affected-row counts are distinct. `ResearchAssistant(df).complete_case_count(["score", "group"])` returns selected columns, available rows, excluded rows and total rows without changing stored data.

`resource_info` uses `DataFrame.memory_usage(index=True, deep=True).sum()` when available. The
default performance-advisory bands are 256 MiB (`large`), 1 GiB (`very_large`), and 100 analytical
numeric columns for a correlation-width warning. It reports matrix dimensions and distinct pair
count before all-pairs correlation work. These local policy categories are not scientific limits
or universal hardware limits. Warnings never sample, truncate, cast, mutate, or skip calculations.
If deep memory estimation fails, the estimate and MiB value are `None`, the level is `unknown`,
and profiling continues with an advisory record.

`data_quality.duplicate_rows` retains the count of rows that repeat an earlier entire row. `duplicate_group_rows` counts every row in a repeated group; `missing_duplicate_overlap_rows` counts rows with both flags. These are overlapping observations, not counts to subtract from a denominator. Repeated identifiers are reported separately when an identifier role is declared or suggested. `data_quality.issues` contains `code`, `severity`, `section`, `column`, `message`, `evidence`, and `recommendation`. Severity is a review priority, not a dataset-quality score.

Outlier methods retain their existing values and add `method`, `definition`, `threshold`, `usable_count`, `status`, and a limitation. A constant column may have an available IQR count of zero, while Z-score and MAD are unavailable when their denominators are zero. Optional `include_row_positions=True` adds zero-based `flagged_positions` to available outlier methods and `duplicate_group_positions`/`repeated_row_positions` to data quality; these are row offsets, regardless of DataFrame index labels. No rows are removed. `histogram_bins` defaults to 20 (valid 1-1000); histograms record binning method, requested bins, edges, counts, and sample size. `distributions.peak_heuristic` identifies the legacy 30-bin `is_bimodal` flag as an informal histogram cue, not proof of population bimodality.

Each of `correlation.pearson`, `.spearman`, and `.kendall` retains its coefficient `matrix` and adds a symmetric `sample_sizes` matrix and `undefined_pairs`. Counts use the rows where both variables are observed, separately for every pair. Only Pearson has `p_values`; an unavailable coefficient has no p-value. Pearson describes linear association, Spearman rank-based monotonic association, and Kendall pairwise concordance. None establishes causation. Identifier-like numeric columns supported by both a name hint and unique observed values are omitted from numerical profiling; low-cardinality integer suggestions alone do not change analysis selection.

`data_dictionary` maps existing column names to optional `label`, `description`, `type`, `role`, `unit`, `valid_range`, `allowed_values`, `missing_codes`, and `ordinal_order`. Types are `continuous`, `discrete`, `nominal`, `ordinal`, `datetime`, `boolean`, `identifier`, or `unknown`. Declarations are validated and inspectable in `variable_intelligence`; categorical and identifier declarations select descriptive methods, without recoding values. Valid-range and allowed-value violations become data-quality issues. **Declared missing codes are counted but not applied**: raw missingness and numerical denominators remain unchanged, and a structured issue calls out this limit. Numbers such as 0, 99, and 999 remain observed values even when declared as missing codes in this phase. No missing mechanism is inferred.

### Independent group comparisons

```python
result = analyzer.hypothesis_tests(
    group_col="group",
    value_col="outcome",
    test_type="auto",
    estimand="mean",
    confidence_level=0.95,
    bootstrap_samples=499,
    random_state=0,
)
```

`test_type` can be `auto`, `ttest`, `mannwhitney`, `anova` or `kruskal`. `auto` now requires the keyword-only `estimand="mean"` or `"distribution"`. For two groups, `mean` selects Welch's t-test and `distribution` selects Mann-Whitney U. For three or more groups, `distribution` selects Kruskal-Wallis; `mean` raises `InvalidTestError` because automatic Welch ANOVA is not available. Diagnostics never switch the estimand. Explicit methods remain available. For an explicit `ttest`, `equal_var=False` (default) uses Welch; `equal_var=True` requests Student's equal-variance version. Supplying an incompatible estimand with an explicit method raises `InvalidTestError`. The library cannot verify independence or decide whether a research design justifies a chosen test.

The result retains `test`, `statistic`, `p_value`, `groups`, `assumptions` and `effect_size`. Additions are `sample_size`, `excluded_rows`, `group_sizes`, degrees of freedom where applicable, and `mean_difference` for t-tests. Assumption diagnostics include `not_rejected`, `rejected` or `unknown` at the documented reference alpha 0.05; they do not certify population assumptions. The effect-size record contains its name, value, interpretation when supported, and a percentile bootstrap confidence interval or `None`. Rank-biserial correlation and rank epsilon-squared have no qualitative magnitude label. T-tests return an analytical `confidence_interval` for first minus second group's mean and a Boolean `equal_variance` indicating the method used.

Rows missing a group or outcome are excluded. Every group needs at least two usable numeric values. The Kruskal-Wallis minimum of five per group is this library's conservative chi-square approximation policy. Use `bootstrap_samples=0` to omit effect-size intervals, or an integer of at least 100 for resampling. `random_state` controls a local random generator. Bootstrap metadata records the requested and valid counts and seed; an interval is `None` with a warning if fewer than `max(50, floor(requested/2))` resamples are valid.

Mann-Whitney U tests a distributional null for independent samples; it is not universally a median test or a fallback for a rejected normality diagnostic. Its statistic is the first-group U, and rank-biserial equals `2U/(n1*n2)-1`. Small tied samples rely on an asymptotic p-value and receive a warning. The reported Cohen's d uses a pooled standard deviation even when Welch's t-test is used. Standard ANOVA assumes equal population variances; a nonrejected Levene result does not prove that assumption. Kruskal-Wallis uses a chi-square approximation and reports a zero-truncated rank epsilon-squared estimate. Bootstrap intervals do not adjust for multiple comparisons. See [statistical validation](docs/STATISTICAL_VALIDATION.md).

### Categorical association

```python
association = analyzer.categorical_association(
    group_col="group",
    outcome_col="response",
    success_value="yes",
    confidence_level=0.95,
    bootstrap_samples=499,
    random_state=0,
)
```

This method runs a Pearson chi-square test of independence without continuity correction. It excludes rows missing either selected value, preserves first-appearance category order, and applies this library's conservative policy requiring every expected cell count to be at least five.

The result includes `test`, `statistic`, `p_value`, `degrees_of_freedom`, `groups`, `outcomes`, `observed_counts`, `expected_counts`, `sample_size`, `excluded_rows`, `assumptions` and Cramér's V in `effect_size`. For a two-by-two table with an explicit `success_value`, it also includes `cohens_h`. Positive Cohen's h means the first group has the higher success proportion. Bootstrap intervals resample complete observed group/outcome rows; interval and assumption metadata report valid resample counts and seed. Set `bootstrap_samples=0` to omit intervals.

## Analysis planning, paired analysis, presentation, and adapters

```python
plan = assistant.analysis_plan(
    draft,
    sensitivity_scenarios=[scenario],
    meaningful_threshold=threshold,
    multiplicity_policy="none_planned",
    report_style="apa",
)
result = assistant.analyze(draft)
adherence = assistant.plan_adherence(
    plan,
    result,
    sensitivity=performed_sensitivity,
    practical_significance=performed_practical_significance,
)
```

`analysis_plan()` accepts one `QuestionDraft` or `AnalysisSpecification`. It performs validation
and recommendation but no analysis. Optional ordered sensitivity specifications and a meaningful
threshold are retained. `previous_plan=` plus optional `reason=` records a revision when tracking
is enabled. `StatisticalAnalysisPlan` has schema version 1, `to_dict()`, `to_json()`,
`from_dict()`, and `to_specification()`. A plan created after this assistant has executed an
analysis records `created_after_analysis=True`. `plan_adherence()` compares recorded fields with
a result as `matched`, `changed`, or `not_recorded` and makes no conduct inference. When supplied,
the optional performed sensitivity and practical-significance records are also compared with the
planned scenario specifications and meaningful-effect threshold. It does not invent a reason for
a change.

```python
planner = StudyPlanner()
independent = planner.independent_mean_power(
    target_difference=5, sd_group1=10, sd_group2=12,
    alpha=0.05, target_power=0.80, allocation_ratio=1,
)
paired = planner.paired_mean_precision(
    sd_difference=5, confidence_level=0.95, target_half_width=2,
)
```

`StudyPlanner` is standalone. It exposes `independent_mean_power`,
`independent_mean_precision`, `paired_mean_power(target_mean_difference=...)`, and
`paired_mean_precision`. Searches are bounded by `max_n` or `max_pairs` (default 100,000).
`allocation_ratio` means `n2/n1`. Paired results use `required_pairs`; `total_required_n` is left
unset so a pair count is not mistaken for raw rows. Inputs must be finite and scientifically
valid. These functions never inspect an observed result and there is no observed-power API.

For paired execution, pass `unit_id=` and optionally `condition_order=(first, second)` through
`prepare_question()`, `run()`, or schema-3 `AnalysisSpecification`. Exactly two observed
conditions are required. Duplicate usable unit/condition rows are blocked, incomplete pairs are
counted and excluded, and row order never establishes pairing. The result method is `paired_t`,
with first-minus-second mean paired difference, analytical paired t interval, Cohen's dz, and
aggregate complete/incomplete-pair counts. Identifier values are not exported.

```python
completeness = assistant.reporting_completeness(report, style="apa")
apa_html = report.to_html(style="apa")
ieee_markdown = report.to_markdown(style="ieee")
latex = report.to_latex(style="general")
report.save_latex("report.tex", overwrite=False)
snapshot = assistant.session_snapshot(
    workflow, analysis_plan=plan, reporting_completeness=completeness
)
```

Completeness item statuses are `present`, `missing`, `partial`, and `not_applicable`; there is no
quality score. Styles are `general`, `apa`, and `ieee` and change presentation only. LaTeX is
escaped inert text and is never compiled. `ResearchSessionSnapshot` schema version 1 contains
JSON-safe workflow state, machine-renderable questions, action identifiers, registry-derived
capabilities, warnings/blockers, and optional records; it embeds no DataFrame or callable.

See [advanced planning and presentation](docs/ADVANCED_PLANNING_AND_PRESENTATION.md), the
[capabilities and support matrix](docs/CAPABILITIES.md), and
[scientific limitations](docs/SCIENTIFIC_LIMITATIONS.md).

## Column detection

```python
types = detect_column_types(df)
roles = suggest_column_roles(df)
```

`detect_column_types()` returns one record per column with `detected_type`, `notes`, `missing_count` and `missing_percentage`. It recognizes numeric categories, continuous numbers, booleans, datetimes, date-like text and common email, URL and phone formats. Contact and date checks sample up to 20 nonmissing values.

`suggest_column_roles()` returns `role`, `reason` and `suggested_action` for each column. Roles include identifier, target, datetime, economic, measurement and unknown. Both helpers are advisory: they do not transform data or select test columns.

## `InsightEngine`

```python
engine = InsightEngine(analysis)
findings = engine.generate_insights()
summary = engine.get_summary()
```

`generate_insights()` returns a list of findings with category, severity, finding and recommendations. `get_summary()` generates findings if needed and returns `total_insights`, `high_severity`, `medium_severity`, `low_severity` and `insights`.

## `ReportGenerator`

```python
report = ReportGenerator(
    analysis_results=analysis,
    insights=summary,
    hypothesis_results=[result, association],
)

payload = report.to_dict()
json_text = report.to_json()
csv_tables = report.to_csv()
html_text = report.to_html()
interactive_html = report.to_interactive_html()  # requires the "report" extra
```

`insights` and `hypothesis_results` are optional. A comparison can be one result dictionary or a list of results. Pass a path to write a format instead of returning its content:

```python
report.to_json("analysis.json", pretty=True)
report.to_csv("csv_reports")
report.to_html("analysis.html", title="Analysis")
report.to_interactive_html("interactive.html", title="Interactive analysis")
```

| Method | Without path | With path |
| --- | --- | --- |
| `to_dict()` | Dictionary with timestamp, analysis, insights and optionally hypothesis tests | N/A |
| `to_json(filepath=None, pretty=True)` | JSON string | Writes JSON and returns a status message |
| `to_csv(output_dir=None)` | Dictionary of DataFrames | Writes CSV files and returns the tables |
| `to_html(filepath=None, title=...)` | Static HTML string | Writes HTML and returns a status message |
| `to_interactive_html(filepath=None, title=...)` | Plotly HTML string | Writes HTML and returns a status message |

CSV tables include descriptive statistics, outliers, missing data and insights. When comparisons are supplied, they also include hypothesis tests. Written CSV protects formula-like text for spreadsheet use; in-memory DataFrames preserve the original values. JSON represents unavailable or non-finite numbers as `null`. Report HTML escapes data-derived text. Interactive HTML requires Plotly and loads Plotly JavaScript from a CDN when opened.

## Errors

All package-specific errors derive from `PyAutoStatError`:

| Exception | Typical cause |
| --- | --- |
| `InvalidDataError` | Unsupported DataFrame shape, names, values or outcome type |
| `ColumnNotFoundError` | Requested column is absent |
| `InsufficientGroupsError` | Too few usable groups |
| `InsufficientDataError` | Too few usable observations or undefined test result |
| `InvalidTestError` | Unknown or incompatible test request |
| `ReportError` | A report file could not be created or written |

`analysis_warnings` records skipped descriptive calculations without aborting the full analysis.

For working code and output files, see [the feature showcase](examples/README.md).
