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
    confidence_level=0.95,
    bootstrap_samples=499,
    random_state=0,
)
```

`test_type` can be `auto`, `ttest`, `mannwhitney`, `anova` or `kruskal`. Automatic selection uses per-group normality and Levene variance screens. It chooses a t-test or Mann-Whitney U for two groups, and one-way ANOVA or Kruskal-Wallis for three or more groups. Explicitly selected tests are retained, with assumption warnings where appropriate.

The result contains `test`, `statistic`, `p_value`, `groups`, `assumptions` and `effect_size`. The assumptions include per-group normality, Levene results, selection reason and warnings. The effect-size record contains its name, value, interpretation and a percentile bootstrap confidence interval. T-tests also return an analytical `confidence_interval` for the first minus second group's mean; a t-test result includes `equal_variance`.

Rows missing a group or outcome are excluded. Every group needs at least two usable numeric values. Kruskal-Wallis requires at least five per group for its approximation. Use `bootstrap_samples=0` to omit effect-size intervals, or an integer of at least 100 for resampling. `random_state` controls a local random generator.

These tests assume independent observations. The automatic screens cannot verify study design, and bootstrap intervals do not adjust for multiple testing.

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

This method runs a Pearson chi-square test of independence without continuity correction. It excludes rows missing either selected value, preserves first-appearance category order, and requires every expected cell count to be at least five.

The result includes `test`, `statistic`, `p_value`, `degrees_of_freedom`, `groups`, `outcomes`, `observed_counts`, `expected_counts`, `sample_size`, `assumptions` and Cramér's V in `effect_size`. For a two-by-two table with an explicit `success_value`, it also includes `cohens_h`. Positive Cohen's h means the first group has the higher success proportion. Set `bootstrap_samples=0` to omit bootstrap intervals.

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
