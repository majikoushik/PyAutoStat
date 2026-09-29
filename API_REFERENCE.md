# PyAutoStat API reference

This page describes the public API in `pyautostat`. The [README](README.md) has a short start-to-finish example; the [examples guide](examples/README.md) covers profiling, estimand-aware tests, and the complete research lifecycle.

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
    coefficient_of_variation_narrative,
    column_story,
    crosstab_narrative,
    dataset_opening,
    detect_column_types,
    executive_summary,
    frequency_narrative,
    insight_narrative,
    percentile_narrative,
    recommendation_rationale,
    suggest_column_roles,
)
```

All documented exception classes are also exported from `pyautostat`.

## `ResearchAssistant` common workflows

```python
assistant = ResearchAssistant(df)
profile = assistant.profile()  # Default quantiles: .05, .25, .50, .75, .95
print(profile["overview"])
print(profile["resource_info"])
print(assistant.summarize())
print(assistant.summarize(mode="story"))
```

The assistant validates and copies the DataFrame using `StatisticalAnalyzer`; `profile()` returns
the same statistical dictionary as `analyze_all()` plus structured profiling metadata. It needs no
research question. `resource_info` contains deep-memory and correlation-width performance
advisories without sampling, truncating, modifying, or skipping calculations.
`profile(data_dictionary=None, histogram_bins=20, include_row_positions=False,
quantiles=(.05, .25, .5, .75, .95))` stores percentiles under each numerical descriptive record.
Quantiles are finite values in `[0, 1]`, deduplicated and sorted, and use pandas `linear`
interpolation after per-column missing-value exclusion. P50 therefore matches the recorded median.

`summarize(data_dictionary=None, histogram_bins=20, mode="profile", quantiles=(...))` runs the same profile operation and formats
shape, missingness, distribution diagnostics, strong correlations, quality metrics, and warnings
as portable plain text. The established `mode="profile"` output remains the default.
`mode="story"` instead assembles an opt-in dataset narrative with a sample-size description, the
strongest eligible recorded Pearson pair, data-quality and distribution summaries, and at most
three deterministically prioritized first steps. It does not recalculate correlations or
diagnostics, infer domain causes, or mutate the DataFrame. Any other mode raises
`InvalidDataError`. The dictionary returned by `profile()` remains the structured source.

### Deterministic profile narration

`dataset_opening(profile) -> str` narrates the recorded row and column counts. Its sample-size
bands are 0-29 (very small), 30-99 (small), 100-499 (moderate), 500-4,999 (medium), and 5,000 or
more (large). These are communication categories, not guarantees that a method is suitable.

`column_story(column_name, stats, unit=None) -> str` consumes an existing numeric record such as
`profile["descriptive"][column_name]`. It reports available center, spread, range, and skewness;
handles constant, all-missing, nonfinite, and partially unavailable records; and never invents a
unit. Skewness retains the established `abs(skew) < 0.5` symmetric boundary and refines nonzero
shape into mild (`0.5` to `<1`), moderate (`1` to `<2`), and strong (`>=2`) left/right states.
Mean/median equality uses numerical closeness. Otherwise the gap is scaled by the first available
positive SD, IQR, or range; a gap at most 0.25 of that spread is described as small.
It also narrates stored quartiles and the safeguarded coefficient of variation. CV is reported as
`100 * sample SD / abs(mean)` only for a finite, numerically nonzero mean and finite sample SD.
Its metadata always notes that CV is most interpretable for ratio-scale measurements with a
meaningful zero; PyAutoStat does not infer that scale from a numeric dtype.

### Categorical description

`assistant.frequency_table(column, data_dictionary=None) -> dict` (also available on
`StatisticalAnalyzer`) accepts nominal, ordinal, and Boolean analytical variables. It returns
ordered `levels` with `level`, `count`, valid-observation `percent`, and `total_percent`, plus
`valid_n`, `missing_n`, `total_n`, exclusion policy, ordering metadata, and deterministic
`narrative`. Missing values are never turned into a level. Declared `ordinal_order` has priority,
then an ordered pandas categorical dtype, then descending frequency with first-observed tie
breaking (the established profile-summary policy). Cumulative percentages appear only for ordinal
variables with one of the first two complete meaningful orders; unordered or incompletely ordered
ordinal results record why they were omitted. Full tables are returned, while narration is capped
at ten levels and profile previews at twenty.

`assistant.cross_tab(row_variable, column_variable, data_dictionary=None) -> dict` returns JSON-safe
axis labels and separate `counts`, `row_percent`, `column_percent`, and `total_percent` matrices.
It uses complete cases for exactly the two columns and records original, valid, and excluded row
counts. Its narration identifies one largest row percentage deterministically and explicitly says
the table is descriptive. The Pearson chi-square backend reuses the same contingency-count
construction; descriptive cross-tabs do not calculate significance.

`percentile_narrative()`, `coefficient_of_variation_narrative()`, `frequency_narrative()`, and
`crosstab_narrative()` consume existing structured values and never recalculate them.

`insight_narrative(category, findings, context=None) -> str` provides deterministic connected
prose for existing insight details. `InsightEngine(results, objective=None).get_narrative()`
assembles these sections. `get_summary()` retains all established counts and the complete
structured `insights` list and adds JSON-safe `narrative` and `recommended_actions` keys. Actions
are ranked by the explicit legacy severity order `high`, `medium`, `low`, then category priority,
then original order; duplicate text is suppressed and output is capped at five actions. At most
three finding examples appear in prose, while the structured detail remains available.

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
Valid objectives are `descriptive`, `compare_groups`, `compare_reference`, `association`,
`regression`, and `reliability`.
`compare_reference` requires one continuous `outcome`, `estimand="mean"`, and a finite explicit
`reference_value`; it has no predictor. Accepted design values
are `independent`, `paired`, `repeated`, `clustered`, and `unknown`; only documented supported
designs execute. Comparison estimands are `mean`, `distribution`, and paired binary `proportion`.
Association targets include `linear`, `point_biserial`, `monotonic`, `partial_linear`, and
`categorical_independence`, with unavailable targets retained as
unsupported rather than replaced.

`ResearchWorkflowResult` exposes `status`, `specification`, `draft`, `recommendation`, `analysis`,
`interpretation`, `report`, `audit`, `reproducibility`, optional `profile`,
`missing_information`, `blockers`, and `warnings`. Statuses are `completed`, `partial`,
`needs_input`, `data_limited`, `unsupported`, and `failed`. A stage that did not run stays `None`.
`to_dict()` and `to_json()` use workflow schema version 1, reject nonfinite JSON values, and do
not embed the source DataFrame.
`workflow.explain() -> str` returns a deterministic portable plain-text view assembled from the
recorded specification, analysis, interpretation, limitations, warnings, and structured
clarification questions. For completed group analyses it includes the recorded group order and
sample accounting before the hypothesis, effect, interval, assumption, and limitation sections;
unsupported sections are omitted rather than manufactured. The method does not rerun profiling,
diagnostics, or statistics, does not mutate or extend the serialized workflow, and requires no
network or generative service.
`str(workflow)` delegates to this view. `analysis.method_label` and
`recommendation.method_label` expose human display names while `method_id` remains the stable
machine identifier. `workflow.interpretation.findings_plain` numbers the existing finding messages
without changing their meaning.

The default successful path performs one statistical execution. Later stages consume that result;
reproducibility-record creation does not replay it. Default audit renders and checks HTML,
Markdown, JSON, and CSV in memory. `audit=False` returns a partial workflow with `audit=None`.
`include_profile=True` requests one dataset profile for an inferential workflow; descriptive
execution reuses its existing profile. No files are written. See
[`docs/CAPABILITIES.md`](docs/CAPABILITIES.md) for the method and failure-mode matrices.

### OLS regression workflow

```python
workflow = assistant.run(
    objective="regression",
    outcome="exam_score",
    predictors=["study_hours", "attendance", "study_method"],
    estimand="conditional_mean",
    design="independent",
    variable_types={
        "exam_score": "continuous",
        "study_hours": "continuous",
        "attendance": "continuous",
        "study_method": "nominal",
    },
    reference_levels={"study_method": "standard"},
    covariance_type="HC3",
)
```

`objective="regression"` requires a continuous numerical outcome, an ordered nonempty
`predictors` list, `estimand="conditional_mean"`, and `design="independent"`. For simple
regression, the singular `predictor=` spelling is accepted as a one-item list. A predictor cannot
also be the outcome; duplicate predictors are invalid. Supported predictors are continuous or
discrete numerical, Boolean, nominal, and ordinal. Numerical terms stay in original units.
Boolean, nominal, and ordinal terms use treatment coding with `k-1` indicator terms; ordinal
categories are not assigned equal spacing.

`reference_levels` maps categorical predictor names to observed scalar levels. Explicit values
take priority, then declared data-dictionary order, pandas categorical order, and first-observed
complete-case order. The result records every reference, comparison level, readable term label,
and design-matrix mapping. An intercept is always included. Unsupported interactions,
transformations, offsets, and alternative intercept policies are not inferred.

One sample is created by excluding rows missing the outcome or any predictor. Original, analyzed,
and excluded counts and the participating columns are stored. Nonfinite values, constant outcomes
or predictors, rank-deficient matrices, and nonpositive residual degrees of freedom are blocked;
predictors are never silently removed.

`covariance_type` is exactly `"classical"` (default) or `"HC3"`. Diagnostics never switch it.
Each coefficient record contains term identity and meaning, estimate, covariance estimator,
standard error, t statistic, p-value, confidence interval, and decision. Standardized beta is
available only for continuous predictors and uses the same complete-case sample; categorical,
discrete, and intercept records mark it not applicable. Model results include F inference when
defined, R-squared, adjusted R-squared, residual sum of squares, residual standard error, and
RMSE.

Structured diagnostics include per-term VIF, Breusch-Pagan, Jarque-Bera residual normality,
Cook's distance/leverage/externally studentized-residual flag counts, and condition number. VIF
and influence thresholds are review heuristics. No diagnostic deletes a row, selects a variable,
or proves an assumption. Reports omit row identifiers and residual arrays. Regression
coefficients are conditional associations rather than causal effects, and R-squared is in-sample
fit rather than out-of-sample predictive accuracy.

The expert method
`StatisticalAnalyzer.linear_regression(outcome, predictors, *, variable_types,
covariance_type="classical", reference_levels=None, data_dictionary=None,
confidence_level=.95, alpha=.05)` returns the same validated numerical schema. The guided path is
preferred when reports, interpretation, audit, replay, and session records are needed.
The existing sensitivity comparison contract is scalar, so it does not currently compare a full
regression coefficient vector. Run separately declared classical and HC3 specifications instead;
their covariance estimator remains part of each coefficient and model record.

### Binary logistic regression and extended association

The guided binary logistic request uses `objective="regression"`, a nonempty ordered
`predictors` list, `estimand="event_probability"`, `design="independent"`, and an explicit
`event_level` when the binary orientation is not inherently Boolean or an analytically declared
`{0, 1}` coding. Exactly two outcome levels must remain in the complete-case sample. Predictor
treatment coding, reference levels, rank validation, VIF, condition number, and complete-case
accounting follow the linear-regression contract. Statsmodels `Logit` reports log likelihoods,
likelihood-ratio inference, AIC, BIC, McFadden pseudo-R-squared, convergence and iterations, plus
per-term `b`, SE, Wald z, p, coefficient CI, odds ratio, and odds-ratio CI. Classical covariance
is the default and explicit HC3 is supported. Nonconvergence and perfect or near separation block
ordinary coefficient inference; no penalized fallback, classification threshold, accuracy, ROC,
or AUC is generated.

Exact McNemar inference uses `objective="compare_groups"`, `estimand="proportion"`,
`design="paired"`, and explicit `unit_id`, `condition_order`, and `event_level`. The shared
long-format unit-pair builder rejects duplicate unit-condition rows and records complete and
incomplete units. The primary effect is first-condition minus second-condition event proportion;
the transition table, discordant `b` and `c`, exact two-sided binomial p-value, paired-unit
bootstrap interval, and status-coded matched odds ratio are retained.

Point-biserial association accepts one declared binary and one quantitative variable with
`estimand="point_biserial"` (or `linear`) and explicit positive `event_level`. Its sign follows the
0/1 coding, and its bootstrap resamples binary/continuous observation pairs. Kendall tau-b is an
explicit `association_measure="kendall"` preference for `estimand="monotonic"`; otherwise guided
monotonic association remains Spearman. Kendall uses SciPy `variant="b"` and a paired-observation
bootstrap interval.

Partial Pearson uses `estimand="partial_linear"` and an explicit ordered, nonempty `controls`
list. This release supports quantitative controls in their original units. It forms one complete-
case sample, fits both OLS adjustment models with an intercept, correlates their residuals, and
uses `df = n - k - 2`, where `k` is the effective non-intercept control-term count. Each bootstrap
draw resamples complete original rows and refits both models. Adjustment is a conditional
association description and does not establish removal of confounding or an independent causal
effect.

### Scale reliability workflow

`ResearchAssistant.reliability(items, *, confidence_level=.95, bootstrap_samples=499,
random_state=0, reverse_scoring=None, data_dictionary=None, title=None, audit=True,
fingerprint=True)` is the focused beginner path. `items` is an ordered sequence of at least two
distinct existing numeric columns. Scale membership is never inferred. The equivalent integrated
request uses `objective="reliability"`, `estimand="internal_consistency"`, and `items=[...]`.

Analysis uses respondents complete on every selected item and records original, analyzed, and
excluded rows plus per-item missing counts and percentages. Cronbach's alpha uses sample
variances (`ddof=1`) and is not clipped when negative. Its percentile interval resamples whole
respondent rows with a local deterministic generator. The interval records its seed and requested
and valid replicate counts; too few valid replicates make interpretation and reporting partial
without invalidating a finite point estimate.

Each item record contains analyzed descriptive statistics, its corrected item-total correlation
(the focal item is excluded from the total), alpha if deleted, and the change from full-scale
alpha. Two-item scales mark deletion diagnostics not applicable. The result also retains an
ordered complete-case inter-item matrix, mean unique off-diagonal correlation, and negative-pair
cues. Numeric items declared ordinal also receive reusable frequency-table records.

`reverse_scoring={"q4": (1, 5)}` explicitly applies `lower + upper - original` on an internal
copy after validating bounds and observed values. It is recorded in the specification, analysis,
report, audit, and replay metadata. No item is reverse-scored or deleted automatically, and no
composite column is created.

The expert method `StatisticalAnalyzer.scale_reliability(items, *, confidence_level=.95,
bootstrap_samples=499, random_state=0, reverse_scoring=None)` returns the validated numerical
record. Reliability output has no p-value, null hypothesis, or effect-size classification. Alpha
does not establish unidimensionality, measurement validity, invariance, test-retest stability, or
inter-rater reliability.

The focused method returns the ordinary `ResearchWorkflowResult`: `completed` when all diagnostics
and the interval are available, `partial` when a valid alpha remains but optional uncertainty or
diagnostics are unavailable, and `data_limited` for observed mathematical blockers. Its
`to_dict()`, `to_json()`, and `explain()` methods remain strict JSON-safe/readable workflow views.
Canonical reports add `reliability_summary`, `reliability_items`, and
`inter_item_correlations` tables and use the same stored values for audit and explicit replay.

### Repeated-measures analysis (3+ conditions)

Repeated-measures workflows evaluate three or more conditions or time points on the same units:

```python
workflow = ResearchAssistant(df).run(
    objective="compare_groups",
    outcome="score",
    predictor="session",
    design="repeated",
    unit_id="participant_id",
    condition_order=("baseline", "week4", "week8"),
    estimand="mean",              # or "distribution" for Friedman
    variable_types={"score": "continuous", "session": "ordinal"},
)
```

The long-form panel requires a unit identifier, condition column, outcome column, and explicit
`condition_order` with at least 3 labels. Duplicate unit-condition records block execution. The
analysis uses a complete-case panel across all declared conditions; missingness is not modeled.

- **One-way repeated-measures ANOVA** (`repeated_measures_anova`):
  Evaluates repeated condition means. The result includes a full ANOVA table (`ss_condition`,
  `ss_error`, `ss_subject`, `ss_total`, `ms_condition`, `ms_error`, `f_statistic`, `p_value`),
  condition means and SDs, and partial eta-squared (`SS_condition / (SS_condition + SS_error)`).
  Sphericity is evaluated using Mauchly's test on orthonormal Helmert contrast covariances.
  When sphericity is rejected ($p < \alpha$), Greenhouse-Geisser corrected degrees of freedom and
  p-values are reported as primary; the observed F statistic is unchanged. Pairwise follow-up
  computes all $k(k-1)/2$ paired t-tests on the omnibus panel with analytical mean-difference CIs,
  Cohen's dz, and Holm-adjusted p-values.

- **Friedman rank-sum test** (`friedman_test`):
  Evaluates within-unit rank distributions for rank or distribution targets. The result includes
  the omnibus $Q$ statistic with $df = k - 1$, $p$-value, condition medians and IQRs, and
  Kendall's coefficient of concordance $W = Q / (n(k - 1))$. Complete pairwise follow-up executes
  all $k(k-1)/2$ paired Wilcoxon signed-rank tests with matched-pairs rank-biserial correlations
  and Holm multiplicity adjustment.

The expert methods `StatisticalAnalyzer.repeated_measures_anova(...)` and
`StatisticalAnalyzer.friedman_test(...)` accept long-form DataFrames or pre-extracted panels and
return the same validated schemas. Two-condition paired designs continue to route through
`paired_t` or `wilcoxon_signed_rank`.

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
print(pending.explain())

revised = assistant.update_question(pending.draft, design="independent")
workflow = assistant.run(draft=revised)
```

The draft retains confirmed values. Its questions explain each missing field, and continuation
passes through the same validation and recommendation rules as a new request. The explanation
shows both supported continuation paths: call `run()` again with the missing raw fields, or update
the retained draft and pass it back through `run(draft=...)`.

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
print(sensitivity.compare())
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

`sensitivity.verdict` and `sensitivity.compare()` provide deterministic plain-language decision
consistency. The printable comparison retains every scenario and labels the comparable conclusion
as `ROBUST`, `CONSISTENT across methods`, `INCONSISTENT`, `DESCRIPTIVE`, or `UNAVAILABLE`.
Only completed `same_estimand` scenarios enter decision agreement, using the primary
specification's alpha and strict `p < alpha` rule. A mixed-estimand result explicitly prevents
direct numerical comparison. The `ROBUST` display label means only that the supplied comparable
methods have the same recorded hypothesis decision; the accompanying qualification states that
this does not establish general robustness, equivalence, or practical importance.

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
print(practical.verdict)
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
`practical.verdict` formats the already validated estimate, threshold, point/interval relations,
conclusion, statistical-significance field, uncertainty status, warnings, and rationale. It adds a
deterministic interval-region label and the display ratio `abs(estimate) / threshold` when the
researcher supplied a positive threshold; zero thresholds never produce infinite prose. Missing
intervals are labelled point-estimate-only, and unknown future relation values receive an explicit
unavailable fallback. This narration does not alter the stored comparison or merge null-hypothesis
significance with practical significance.

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

`prepare_question()` accepts optional `objective`, `outcome`, `predictor`, `predictors`, `items`,
`controls`, `design`, `estimand`, `event_level`, `association_measure`, `description`, `options`,
`data_dictionary`, `variable_types`, and `specification`. For comparison, `predictor` names the
group or condition column. For association it names the second variable. `controls` is restricted
to association requests, while `predictors` is restricted to regression. A descriptive request
needs none of these inferential details. `unknown` design remains unresolved; essential event and
pairing orientations produce structured `needs_input` rather than arbitrary level selection.

The `QuestionDraft` has `specification`, `status` (`ready`, `needs_input`, `data_limited`, `unsupported`), `missing_information`, `questions`, `warnings`, `blockers`, `variable_suggestions`, and `availability`. `questions` carry `field`, `question`, `explanation`, `input_type`, `required`, and options with stable `value` and display `label`; `to_dict()` is JSON-compatible. A future GUI should render these records and send selected values through `update_question()`. `availability` reports total, complete, and missing-relevant rows using complete-case counts. It is an availability summary, not a missing-data treatment. Declared missing codes remain ordinary observed values until the caller normalizes them. All-missing selected columns, no complete rows, and a group with fewer than two observed categories produce `data_limited` with blockers. An unsupported objective raises an actionable error; `unsupported` is reserved for future requests that cannot be represented. Constant variables produce warnings. Invalid column names and incompatible declarations raise package errors.

`update_question(draft, **changes)` preserves confirmed answers, reconstructs a new draft, and leaves the old draft unchanged on invalid input. When the objective changes, it clears the previous predictor, target, and design; switching to descriptive also clears the old outcome. Supply new role selections explicitly. `profile(data_dictionary=...)` makes a copied declaration available to later question preparation; an explicit `data_dictionary` or `variable_types` correction takes precedence. The builder reuses dataset variable intelligence without running full profiling, hypothesis tests, or recommendations.

`ready` means question fields and basic selected-data checks are complete. It does not establish an appropriate method, valid assumptions, or a certified analysis plan. Paired and repeated designs require an explicit unit-ID column and condition order; arbitrary clustered designs remain representable but unsupported for execution.

### Method recommendation

```python
draft = assistant.prepare_question(
    objective="compare_groups", outcome="score", predictor="group",
    estimand="mean", design="independent",
)
recommendation = assistant.recommend_test(draft)
# Alternatively: assistant.recommend_test(specification=draft.specification)
print(recommendation.status, recommendation.method_id, recommendation.rationale)
print(recommendation.rationale_text)
# After analysis, the same helper can narrate an actual recorded diagnostic mapping:
result = assistant.analyze(draft)
print(recommendation.explain(diagnostics=result.metadata["diagnostics"]))
```

`recommend_test(draft=None, *, specification=None)` accepts exactly one `QuestionDraft` or `AnalysisSpecification`. A direct specification, and any supplied draft, are revalidated against the assistant's current DataFrame and question-availability rules. No full profile or hypothesis test is run. Unknown essential information returns `needs_input` with `missing_information` and GUI-ready `questions`; a data-limited draft returns `unsupported` with its blockers. An incompatible design, target, measurement type, or hard computational requirement returns `unsupported`. `ready` means a compatible existing operation can be called later; it does not certify uncheckable assumptions.

| Objective and target | Conditions | Result |
| --- | --- | --- |
| Description | Valid DataFrame | `dataset_profile` (`ResearchAssistant.profile()`), descriptive only |
| Reference comparison, mean | One continuous outcome, independent units, finite explicit reference, at least two observations | `one_sample_t`; estimate is observed mean minus reference |
| Group comparison, mean | Two independent groups, quantitative outcome, at least two usable outcomes per group, representable spread | `welch_t`; Student t is an explicit, equal-variance alternative |
| Group comparison, distribution | Two independent groups, ordered numeric outcome, at least two usable outcomes per group | `mann_whitney_u`; tied small samples have an approximation warning |
| Group comparison, distribution | Three or more independent groups, ordered numeric outcome, at least five usable outcomes per group | `kruskal_wallis` with all Dunn comparisons and Holm adjustment |
| Group comparison, mean | Three or more independent groups, positive finite within-group variance | `welch_anova` with all Games-Howell comparisons; standard one-way ANOVA remains an explicit equal-variance option |
| Group comparison, distribution | Paired two-condition data with explicit unit ID and condition order, at least two nonzero differences | `wilcoxon_signed_rank`; zeros use the recorded `wilcox` policy |
| Group comparison, proportion | Paired two-condition binary outcome with explicit unit ID, condition order, and event | exact `mcnemar` with paired proportion difference |
| Association, linear | Two quantitative variables, independent observational pairs, at least three complete varying pairs | `pearson_correlation`, including the existing pairwise p-value when numerically valid |
| Association, monotonic | Two varying ordered numeric variables and at least three complete pairs | `spearman_correlation` with rho, p-value, and paired-observation bootstrap CI when available |
| Association, monotonic with explicit Kendall preference | Two varying ordered numeric variables and at least three complete pairs | `kendall_tau_b`; Spearman remains the default without the preference |
| Association, binary/continuous | Exactly one genuine binary variable with explicit positive level and one quantitative variable | `point_biserial_correlation` |
| Association, partial linear | Two quantitative variables and one or more explicit quantitative controls | `partial_pearson_correlation` with complete-row model-refitting bootstrap |
| Association, categorical independence | Two categorical variables, independent observations, at least two categories per axis | `pearson_chi_square` when every expected count is at least five; otherwise `fisher_exact` for 2x2 only |
| Regression, event probability | Exactly binary declared outcome/event, independent rows, estimable full-rank predictor design | `logistic_regression` with likelihood and odds-ratio inference |

Numeric association with no specified relationship target requests one clarification. A numeric/categorical association requests confirmation before changing the research objective. A paired mean question without a unit ID returns `needs_input`; compatible two-condition paired data select `paired_t`. Repeated and clustered designs return `unsupported`; an unknown essential design returns `needs_input`. A declared missing code still present among selected values blocks a finalized recommendation until the caller normalizes the data and rebuilds the assistant. The engine never recodes or excludes those values itself.

`Recommendation` retains its version 1 envelope and original fields. Additive fields are `method_availability` (`runnable`, `coefficient_only`, or `unavailable`), `decision_trace` (ordered `{key, value, reason}` entries), `alternatives` (method ID, name, availability, reason), `context` (objective, target, design, selected analytical types, complete-case availability and relevant feasibility facts), and `questions` (question-builder-compatible clarification dictionaries). `required_assumptions` and `context.assumption_checks` disclose researcher-confirmed facts, checkable feasibility, and conditions requiring review. `to_dict()` is JSON-compatible and includes no raw rows or invented p-values. The small `METHOD_CAPABILITIES` registry in `pyautostat.recommendation` documents actual backend availability. An explicit preferred-method override is deferred to avoid changing the specification schema; existing explicit analyzer calls remain available to experts.

`recommendation.rationale_text` and `recommendation.explain(diagnostics=None)` derive a readable
`RECOMMENDED TEST`, `WHY THIS TEST?`, context-specific `WHY NOT ...?`, and
`WHAT YOU NEED TO VERIFY` explanation from the same stored record. Coverage includes every
automatically recommended method: dataset profile, one-sample t, Welch t, paired t, paired
Wilcoxon, Mann-Whitney U, Welch ANOVA, Kruskal-Wallis, Pearson correlation, Spearman correlation, Pearson
chi-square, and Fisher exact. The prose distinguishes mean, rank-distribution, paired, linear,
monotonic, and categorical-independence targets. Independence,
representativeness, unit identity, and contrast order remain researcher-verification items.
Diagnostics are never run by narration: for example, Levene's result appears only when the caller
passes an actual recorded diagnostic mapping. The original `rationale`, decision trace, version 1
serialization, and recommendation calculation remain unchanged. The public
`recommendation_rationale(recommendation, diagnostics=None)` helper provides the same output.

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
| `one_sample_t` | `one_sample_t_test()` / `scipy.stats.ttest_1samp` | Sample mean, explicit reference, observed-minus-reference mean difference, SE, t/df/p, analytical raw-difference CI, and one-sample Cohen's d when defined |
| `welch_t` | `hypothesis_tests(test_type="ttest", equal_var=False)` | Mean difference, Welch statistic/df/p, Cohen's d, analytical mean-difference CI, optional bootstrap d CI |
| `mann_whitney_u` | `hypothesis_tests(test_type="mannwhitney")` | First-group U, two-sided p, rank-biserial effect and optional bootstrap CI |
| `welch_anova` | `welch_anova()` | Welch F and numerator/denominator df, group summaries, and every Games-Howell mean difference with simultaneous interval and adjusted p-value; no questionable global standardized effect |
| `kruskal_wallis` | `hypothesis_tests(test_type="kruskal")` | H, df, p, rank epsilon-squared, optional bootstrap CI, and every Dunn comparison with raw and Holm-adjusted p-values and pairwise rank-biserial effect |
| `pearson_correlation` | Pair-only `analyze_all()` correlation profile | Pearson r and its pairwise p-value; no CI |
| `paired_t` | `scipy.stats.ttest_rel` over explicit unit-ID pairs | First-minus-second paired mean, analytical CI, Cohen's dz, and complete/incomplete-pair accounting |
| `wilcoxon_signed_rank` | `paired_wilcoxon()` / `scipy.stats.wilcoxon` | Signed-rank statistic/p, matched-pairs rank-biserial correlation, zero policy and pair accounting; effect CI unavailable |
| `spearman_correlation` | `spearman_correlation()` / `scipy.stats.spearmanr` | Spearman rho/p and deterministic paired-observation bootstrap CI when enough valid resamples exist |
| `pearson_chi_square` | `categorical_association()` | Chi-square, df, p, observed/expected table, Cramer's V and optional bootstrap CI |
| `fisher_exact` | `fisher_exact()` / `scipy.stats.fisher_exact` | Ordered observed 2x2 table, two-sided p, and SciPy's unconditional sample odds ratio; CI unavailable |
| `logistic_regression` | `statsmodels.api.Logit` | Declared event/reference, model likelihood fit, convergence, coefficients and odds ratios with Wald intervals, VIF and condition diagnostics |
| `mcnemar` | shared unit-ID pair builder / `scipy.stats.binomtest` | Ordered transition table, exact two-sided p, event proportions, paired proportion difference and paired-unit bootstrap CI |
| `point_biserial_correlation` | `scipy.stats.pointbiserialr` | Explicit 0/1 coding, r_pb/p, group counts/means and paired-observation bootstrap CI |
| `kendall_tau_b` | `scipy.stats.kendalltau(variant="b")` | Tau-b/p, tie metadata and paired-observation bootstrap CI |
| `partial_pearson_correlation` | two statsmodels OLS residual models + Pearson r | Ordered controls, effective df, partial r/p and complete-row model-refitting bootstrap CI |

The registry also describes Student's pooled t-test and standard one-way ANOVA as runnable
**legacy explicit calculations**, but the guided recommender does not select them automatically.
Kendall remains available as a coefficient matrix in profiling and now also has an explicit
guided inferential route. Two-way repeated designs, mixed ANOVA, clustered methods, and
Fisher tests beyond 2x2 remain unavailable.

`AnalysisResult` retains its version 1 common envelope (`method_id`, `status`, `sample_size`, `excluded_rows`, `values`, `assumptions`, `warnings`, `metadata`). Additive `specification` and `recommendation` fields retain the actual validated request and selected method; `to_dict()` serializes both. The `method_label` property resolves a display name from the existing method metadata without changing serialization. `values` uses `test_statistic`, `degrees_of_freedom`, `p_value`, `primary_estimate`, `estimate_name`, `estimate_unit`, `effect_size`, and `confidence_interval`. Each interval names its `quantity`, `method`, `level`, and bounds. `None` means the backend provided no supported value. The descriptive path uses `values.profile` and explicit `None` inferential fields. `metadata.sample` records original, analyzed and excluded rows, with group sizes or effective pair count where relevant. `metadata.group_order` follows the backend's first-observed order. For two-group tests, `metadata.contrast` defines first minus second; the mean difference, Cohen's d, U orientation and rank-biserial sign use this order. `metadata.diagnostics` preserves backend assumption results; `warnings` combines intake, recommendation and backend warnings without duplicates.

Multi-group results add `values.group_summaries` and the complete
`values.pairwise_comparisons` family. Each pair records procedure, ordered groups and an explicit
first-minus-second contrast, estimate and name, statistic and name, raw p-value when defined,
adjusted p-value, adjustment method and family size, supported interval/effect information, both
sample sizes, standard error, degrees of freedom when applicable, alpha, decision, and warnings.
`metadata.pairwise_method`, `multiplicity_control`, and `pairwise_comparison_count` describe the
family. Pairwise calculation is unconditional on the omnibus p-value; only prose display is
bounded.

Direct expert methods are `StatisticalAnalyzer.welch_anova(...)`,
`games_howell(...)`, `tukey_hsd(...)`, and `dunn(..., adjustment="holm")`. Games-Howell and
Tukey-Kramer use `scipy.stats.studentized_range`; Dunn applies the pooled-rank tie correction and
the central Holm adjustment helper. All preserve first-observed group order and reject incomplete,
nonfinite, or unsupported variance cases explicitly.

Group and categorical effect intervals use the existing 499-resample bootstrap with effective seed 0 when no seed was specified; this default and any explicit seed are recorded in metadata and diagnostics. The specification's alpha and confidence level remain available through `result.specification.options`; alpha is not used to alter the numerical p-value. No missing rows are imputed, no outliers are removed, and declared missing codes still block execution until normalized externally. JSON export is `json.dumps(result.to_dict(), allow_nan=False)`.

### Deterministic interpretation

`pyautostat.effect_narrative(measure, value, n=None, ci=None, *, confidence_level=None,
orientation=None, definition=None) -> str` converts an already-computed effect value into
deterministic researcher-readable text. It accepts the public quantity identifiers `cohens_d`,
`rank_biserial`, `paired_rank_biserial`, `eta_squared`, `epsilon_squared`, `cramers_v`,
`pearson_r`, `spearman_rho`, and `odds_ratio`, their result-record display names, and the existing
paired effect `cohens_dz`/`Cohen's dz`. It performs no statistical
calculation, method selection, or interval construction. Unsupported measures and missing,
nonfinite, or out-of-range values return an explicit unavailable narrative rather than raising or
substituting zero.

Magnitude labels preserve PyAutoStat's established bands: Cohen's d/dz use negligible, small,
medium, and large, with the `very large` band beginning at an absolute value of 2;
eta-squared and Pearson r/Cramer's V retain their existing eta-squared and correlation bands.
Rank-biserial correlation and epsilon-squared remain deliberately unlabelled because the package
does not currently assign them a conventional qualitative magnitude. Signed effects use absolute
magnitude for classification while retaining their sign and optional recorded orientation in the
text.

`ci` may be a `(lower, upper)` tuple or an existing interval mapping. Valid finite bounds receive
relative-width commentary based on the recorded estimate. A stored interval `level` and bootstrap
method are reported when present; no confidence level is invented. A nonzero-width interval around
a zero estimate is described as wide because relative width cannot be divided by zero. The same
narration is used by `InterpretationEngine`; existing result and interpretation schemas are
unchanged.

`hypothesis_verdict(p, alpha, effect_val, measure, magnitude_label=None, n=None) -> str` combines
the existing strict `p < alpha` decision with the recorded effect magnitude in four deterministic
quadrants. `large`, `very large`, `moderate`, and `strong` labels are treated as non-trivial; an
unavailable or unlabelled effect is never silently treated as small. Very small finite p-values use
compact significant-digit formatting, and computational zero is shown as an inequality.

`assumption_grade(assumption, status, *, n=None, p_value=None, group=None, method_id=None)` returns
`(severity, message)` using the centralized `INFO`, `CAUTION`, `WARNING`, and `CRITICAL` scale.
Rejected normality is `WARNING` below 15 observations, `CAUTION` from 15 through 29, and `INFO`
from 30 onward. Missing group size is conservatively `WARNING`. A rejected equal-variance
diagnostic is `INFO` for Welch because equal variance is not assumed, but `WARNING` for pooled
methods. Diagnostic p-values are included only when recorded.

`interval_verdict(...)` narrates all current point/interval relation combinations and the safe
magnitude ratio described above. `sensitivity_verdict(...)` narrates comparable structured
hypothesis decisions while preserving mixed-estimand and unavailable states. These public helpers
are side-effect-free presentation functions; they do not run analyses or construct intervals.

```python
result = assistant.analyze(draft)
interpretation = assistant.interpret(result)
print(interpretation.summary)
print(interpretation.findings_plain)
print(interpretation.status, [finding.code for finding in interpretation.findings])
payload = interpretation.to_dict()  # json.dumps(payload, allow_nan=False)
```

`interpret(result: AnalysisResult) -> InterpretationResult` reads the completed analysis record only; it runs no statistical test or report writer. `InterpretationEngine().interpret(result)` is the independently usable rule engine. The result has `status` (`available`, `partial`, `unavailable`), `execution_status`, `method_id`, `summary`, `method_explanation`, `hypothesis_interpretation`, `effect_interpretation`, `uncertainty_interpretation`, `assumption_notes`, `limitations`, `conclusion`, `warnings`, `metadata`, and `findings`. Each `InterpretationFinding` has a stable `code`, display `message`, and `supporting_fields` pointing to the source result. `findings_plain` numbers the existing messages for display without changing codes or serialization. `to_dict()` is JSON compatible and leaves the original `AnalysisResult` unchanged.

Supported guided results include the four additions `one_sample_t`, `wilcoxon_signed_rank`,
`spearman_correlation`, and `fisher_exact` alongside the established methods. One-sample and
Spearman results are complete when their intervals are available. Wilcoxon and Fisher
interpretations are intentionally partial because no supported primary-effect interval is
fabricated. Constant one-sample data retain the raw mean contrast but leave t, p, and standardized
effect unavailable. Unrecognized or unavailable results remain unavailable.

The decision rule is `p < result.specification.options.alpha`; equality does not reject. It uses the original numeric p-value and formats very small values separately; computational zero displays as `p < 0.001` with a numerical warning. The hypothesis paragraph uses the four-quadrant verdict without changing the stable coded finding or conclusion. Assumption notes retain their recorded diagnostics and add bracketed severity; normality severity uses each diagnostic's stored group size, and Welch variance narration uses the recorded Levene result when available. The engine does not claim multiplicity adjustment, equivalence, causation, or practical importance. Two-group directions follow `metadata.contrast`; confidence intervals retain their own `quantity`, `method`, and `level`. Finite ordered percentile-bootstrap bounds may exclude the original point estimate; containment is required for the analytical t mean-difference interval. The analytical t interval is checked against a two-sided p-value only when its level matches `1-alpha`. Apparent disagreement produces a warning and partial status. A valid raw mean difference remains interpretable when Cohen's d is missing or inconsistent, with partial status and a warning; conflicting signs invalidate the d interpretation. Bootstrap effect intervals are not treated as interchangeable with the analytical mean-difference interval. Diagnostic non-rejection never proves an assumption; source warnings and excluded-row counts remain visible.

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

`to_html()` returns self-contained escaped static HTML; `to_markdown()` escapes user syntax; `to_json()` preserves JSON-safe raw numbers; `to_csv_tables()` returns a dictionary of independent CSV strings. Cells beginning with formula-like prefixes after whitespace are prefixed with an apostrophe only in CSV output; numeric cells remain numeric. `save_html(path)`, `save_markdown(path)`, `save_json(path)`, and `save_csv_tables(directory)` write only to explicit paths and reject existing files unless `overwrite=True`. Filenames for CSV tables are stable IDs, never derived from report titles or category labels. The report omits categorical identifier labels from descriptive profiles and does not include the complete input DataFrame. Small aggregate groups can still disclose information. See [the schema and method matrix](docs/RESEARCH_REPORT_SCHEMA.md) and [the complete workflow example](examples/03_advanced_workflow.py). The legacy `ReportGenerator` continues to accept its original dictionary inputs.

Static canonical and legacy HTML reports place an escaped semantic `Executive Summary` near the
top. Its format-neutral content is built by `executive_summary(...)` from stored dataset scale,
represented analyses, the existing hypothesis/effect narration, existing data-quality narration,
and recorded diagnostics or limitations. Canonical reports add practical-significance and sensitivity
verdicts only when those assessments were supplied. The builder does not execute a test,
reclassify an effect, infer causation, or add fields to the JSON report schema. Markdown, JSON,
CSV, and LaTeX contracts are unchanged.

`executive_summary(*, profile=None, dataset=None, analyses=(), limitations=(),
practical_significance=None, sensitivity=None) -> tuple[str, ...]` is the public format-neutral
builder used by those report renderers. Each item in `analyses` is an already-normalized mapping
with optional `method_name` and `finding` text; the function does not accept a DataFrame or run an
analysis. It returns ordered plain-text paragraphs, including practical-significance and
sensitivity text only when supplied. Report renderers remain responsible for escaping those
paragraphs for HTML.

```python
paragraphs = executive_summary(
    dataset={"original_rows": 40, "analyzed_rows": 38},
    analyses=[{"method_name": "Recorded analysis", "finding": interpretation.summary}],
    limitations=interpretation.limitations,
)
```

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

`to_dict()` and `from_dict()` are supported by `ResearchQuestion`, `AnalysisOptions`, and
`AnalysisSpecification`. The root specification uses `schema_version: 1` when it has no data
dictionary and `schema_version: 2` when `data_dictionary` is present. Existing payloads remain
readable; the optional `ResearchQuestion.reference_value` is additive within the current schema
and defaults to `None` when absent. Version 2 adds the data dictionary without overloading version
1's text-only `variable_metadata`. The data dictionary is authoritative for analytical types and
roles. See [the architecture document](docs/ARCHITECTURE.md) for migration details.
`pyautostat.results` exposes the serializable result records. Question preparation manufactures
no recommendation, inferential result, or report.

## `StatisticalAnalyzer`

### Construction and full analysis

```python
analyzer = StatisticalAnalyzer(df)
analysis = analyzer.analyze_all()
# Both entry points also accept data_dictionary=..., histogram_bins=20,
# and include_row_positions=False as optional keyword arguments.
```

Direct basic-inference methods are:

- `one_sample_t_test(value_col, reference_value, *, confidence_level=.95)`;
- `paired_wilcoxon(unit_id, condition_col, value_col, *, condition_order=None)`;
- `spearman_correlation(first, second, *, confidence_level=.95, bootstrap_samples=499,
  random_state=0)`; and
- `fisher_exact(row_variable, column_variable)` for 2x2 tables only.

All exclude missing rows transparently and never mutate the input. Paired Wilcoxon constructs
pairs by unit ID, never row order. Fisher retains level order with its observed table. The
profile-wide Spearman matrix remains descriptive and separate from two-variable inference.

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

An unavailable numeric result is `None`. Some tests are skipped for all-missing, constant or short columns; inspect `analysis_warnings`. Available normality records include a qualified `verdict`; non-rejection explicitly does not prove normality. For D'Agostino-Pearson at 8-19 observations, a finite result is retained with a small-sample approximation warning; other numerical warnings or nonfinite output make it unavailable. Anderson-Darling uses the validated five-percent critical value when present and is omitted if SciPy supplies an unusable critical-value grid, including nonpositive values sometimes returned for very small samples. Correlation uses pairwise nonmissing observations. Only Pearson pairs include p-values.

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
    alpha=0.05,
)
```

`test_type` can be `auto`, `ttest`, `mannwhitney`, `anova` or `kruskal`. `auto` requires the keyword-only `estimand="mean"` or `"distribution"`. For two groups, `mean` selects Welch's t-test and `distribution` selects Mann-Whitney U. For three or more groups, `mean` selects Welch ANOVA plus Games-Howell and `distribution` selects Kruskal-Wallis plus Dunn-Holm. Diagnostics never switch the estimand. Explicit `anova` adds Tukey-Kramer comparisons. For an explicit `ttest`, `equal_var=False` (default) uses Welch; `equal_var=True` requests Student's equal-variance version. `alpha` controls only recorded pairwise decisions, while `confidence_level` controls supported intervals. Supplying an incompatible estimand with an explicit method raises `InvalidTestError`. The library cannot verify independence or decide whether a research design justifies a chosen test.

The result retains `test`, `statistic`, `p_value`, `groups`, `assumptions` and `effect_size`. Additions are `sample_size`, `excluded_rows`, `group_sizes`, degrees of freedom where applicable, `mean_difference` for t-tests, and a display-only `interpretation` assembled from those recorded values. Assumption diagnostics include `not_rejected`, `rejected` or `unknown` at the documented reference alpha 0.05; they do not certify population assumptions. The effect-size record contains its name, value, interpretation when supported, and a percentile bootstrap confidence interval or `None`. Conventional effect labels are not practical-significance decisions. Rank-biserial correlation and rank epsilon-squared have no qualitative magnitude label. T-tests return an analytical `confidence_interval` for first minus second group's mean and a Boolean `equal_variance` indicating the method used.

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
a change. Binary and extended-association plans retain modeled event/positive level, condition
order, ordered predictors or controls, regression references, covariance, bootstrap settings,
and an explicit Kendall preference where applicable.

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

`generate_insights()` returns advisory records with `category`, `severity`, `finding`, `details`,
and `recommendation`; the last two are always lists. Distribution findings also preserve the
legacy `columns` alias. Recommendations identify relevant columns but do not switch estimands,
delete outliers, or impute values automatically. `get_summary()` generates findings if needed and
returns `total_insights`, `high_severity`, `medium_severity`, `low_severity` and `insights`.

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

The static legacy HTML uses escaped content, styled severity badges, responsive tables, and print
rules. Its missing-data section includes a plain-language summary derived from the same counts;
the structured analysis dictionary remains authoritative.

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

For working code and output files, see [the examples guide](examples/README.md).
