# PyAutoStat API reference

This page describes the public API in `pyautostat`. The [README](README.md) has a short start-to-finish example; the [examples guide](examples/README.md) runs every feature and shows its output.

## Imports

```python
from pyautostat import (
    AnalysisOptions,
    AnalysisSpecification,
    InsightEngine,
    Objective,
    ReportGenerator,
    ResearchAssistant,
    ResearchQuestion,
    StatisticalAnalyzer,
    StudyDesign,
    detect_column_types,
    suggest_column_roles,
)
```

All documented exception classes are also exported from `pyautostat`.

## `ResearchAssistant` and research configuration

```python
assistant = ResearchAssistant(df)
profile = assistant.profile()

spec = AnalysisSpecification(
    question=ResearchQuestion(
        objective=Objective.COMPARE_GROUPS,
        outcome="score",
        predictor="group",
        estimand="difference in means",
    ),
    design=StudyDesign.UNKNOWN,
    options=AnalysisOptions(alpha=0.05, confidence_level=0.95),
)
payload = spec.to_dict()
restored = AnalysisSpecification.from_dict(payload)
```

The assistant validates and copies the DataFrame using `StatisticalAnalyzer`; `profile()` returns its existing `analyze_all()` dictionary. It does not run group tests or use the research specification. `ResearchQuestion` fields may remain `None` while information is gathered. `StudyDesign.UNKNOWN` is explicit and never converted to independent. Invalid values raise `InvalidDataError` with the affected field. Data-dependent column and design checks are not part of these records yet.

`to_dict()` and `from_dict()` are supported by `ResearchQuestion`, `AnalysisOptions`, and `AnalysisSpecification`. The root specification uses `schema_version: 1`; see [the architecture document](docs/ARCHITECTURE.md) for fields, status values, and the future result contracts. `pyautostat.results` exposes `MissingInformation`, `Recommendation`, `Diagnostic`, and `AnalysisResult` as serializable records, but no Phase 1 workflow produces them. They have `to_dict()` only. No recommendation, inferential result, or report is manufactured from these records.

## `StatisticalAnalyzer`

### Construction and full analysis

```python
analyzer = StatisticalAnalyzer(df)
analysis = analyzer.analyze_all()
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

An unavailable numeric result is `None`. Some tests are skipped for all-missing, constant or short columns; inspect `analysis_warnings`. Correlation uses pairwise nonmissing observations. Only Pearson pairs include p-values.

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
