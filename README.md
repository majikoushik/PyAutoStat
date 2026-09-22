# PyAutoStat

[![CI](https://github.com/majikoushik/pyautostat/actions/workflows/ci.yml/badge.svg)](https://github.com/majikoushik/pyautostat/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/majikoushik/PyAutoStat/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/majikoushik/PyAutoStat/blob/main/LICENSE)

PyAutoStat analyzes pandas DataFrames and returns structured statistical results, data quality findings, and reports. It supports exploratory analysis and independent group comparisons. The package is in **alpha**; review assumptions and results before using them in research or decisions.

## Features

- **Explore data:** descriptive statistics, missing values, duplicates, distributions, histograms, and advisory column type and role detection.
- **Check assumptions:** Shapiro-Wilk, D'Agostino-Pearson, and Anderson-Darling normality results; IQR, Z-score, and MAD outlier summaries.
- **Study relationships:** Pearson, Spearman, and Kendall correlations, with p-values for Pearson pairs.
- **Compare independent groups:** Welch or explicit Student t-test, Mann-Whitney U, one-way ANOVA, and Kruskal-Wallis; assumption diagnostics, effect sizes, and confidence intervals. Automatic selection requires a stated target quantity.
- **Test categorical association:** Pearson chi-square, Cramér's V, and Cohen's h for a two-by-two table with a named success outcome.
- **Plan an analysis:** turn a completed research question into a traceable method recommendation, clarification request, or unsupported result without running a test.
- **Execute a supported plan:** `ResearchAssistant.analyze()` rechecks the design and recommendation, calls the existing numerical backend, and returns a structured result tied to the research specification.
- **Share results:** severity-rated insights and dictionary, JSON, CSV, static HTML, or optional Plotly HTML reports.

## Installation

Requires Python 3.10 or newer:

```bash
python -m pip install pyautostat
```

For interactive HTML reports, install the optional Plotly dependency:

```bash
python -m pip install "pyautostat[report]"
```

## Quick start

For a dataset profile without a research question:

```python
import pandas as pd

from pyautostat import ResearchAssistant

df = pd.DataFrame({
    "group": ["A"] * 8 + ["B"] * 8,
    "score": [10.1, 11.2, 12.3, 13.4, 14.5, 15.6, 16.7, 17.8,
              14.1, 15.2, 16.3, 17.4, 18.5, 19.6, 20.7, 21.8],
})
profile = ResearchAssistant(df).profile()
print(profile["overview"])
```

`profile()` returns the same dictionary as `StatisticalAnalyzer(df).analyze_all()`. Guided question preparation, recommendation, and execution of supported methods are available. Interpretation and research-report assembly remain future work. See the [architecture and contracts](docs/ARCHITECTURE.md).

### Prepare a research question

```python
assistant = ResearchAssistant(df)
draft = assistant.prepare_question(
    objective="compare_groups", outcome="score", predictor="group",
    variable_types={"score": "continuous"},
)
print(draft.status, [question.to_dict() for question in draft.questions])
draft = assistant.update_question(draft, estimand="mean", design="independent")
saved = draft.specification.to_dict()
recommendation = assistant.recommend_test(draft)
print(recommendation.status, recommendation.method_name)
print(recommendation.rationale)
result = assistant.analyze(draft)
print(result.status, result.method_id, result.values["p_value"])
print(result.values["primary_estimate"], result.metadata["contrast"])
```

`descriptive` needs no design or target. `compare_groups` requires an outcome, group column, target (`mean` or `distribution` for the common path), and confirmed design. `association` requires two columns and the relationship between observations across rows; two values in one row do not establish a paired-group design. Unknown facts stay as `needs_input` questions with stable option values; unusable selected data produce `data_limited` blockers. A `ready` draft means only that Phase 4 intake is complete and the selected data pass basic availability checks. `recommend_test()` adds method and design checks. Its `ready` status means a compatible calculation exists, not that the study's assumptions have been proven or a test has run. Use `data_dictionary={"score": {"type": "continuous"}}` or `variable_types={"score": "continuous"}` to correct an ambiguous type suggestion. No source values are recoded.

For two independent quantitative groups targeting means, the recommendation is Welch's t-test. For ordered distribution comparisons it can recommend Mann-Whitney or Kruskal-Wallis. Linear numerical association can receive Pearson; adequate categorical tables can receive chi-square. Descriptive questions receive `dataset_profile`. A multi-group mean question remains unsupported without a justified explicit standard ANOVA choice. Paired, repeated, and clustered designs are unsupported by the current recommendation path. Spearman and Kendall have descriptive coefficients only, with no inferential p-values. The result records a structured decision trace, assumptions requiring review, relevant alternatives, and any missing information. [The API reference](API_REFERENCE.md) details the rules.

`analyze()` executes only a fresh, ready, runnable recommendation. It records the original specification, method ID, analyzed and excluded rows, numerical values, confidence-interval quantity, group order, diagnostics, and warnings in `AnalysisResult`. Incomplete or unsupported requests return `status="unavailable"` without running a test; invalid column names still raise a package error. A successful calculation does not verify the scientific design. The existing `StatisticalAnalyzer` and `ReportGenerator` APIs remain available separately.

The profile includes categorical frequencies and tied modes, missing rows and patterns, exact-duplicate overlap, advisory analytical types, outlier and distribution metadata, and pairwise observation counts for Pearson, Spearman, and Kendall. The DataFrame remains unchanged. Optional declarations and row positions stay out of the one-argument path:

```python
profile = ResearchAssistant(df).profile(
    data_dictionary={"score": {"type": "continuous", "valid_range": [0, 100]}},
    histogram_bins=20,
    include_row_positions=True,
)
print(profile["correlation"].get("pearson", {}).get("sample_sizes"))
```

Declared missing codes are counted and flagged but are not recoded or excluded. A reported outlier or duplicate is a review cue, not an automatic deletion or error verdict. Export with `ReportGenerator(profile).to_json()` for JSON-safe values such as dtype names.

This example runs without an input file:

```python
import pandas as pd

from pyautostat import InsightEngine, ReportGenerator, StatisticalAnalyzer

df = pd.DataFrame(
    {
        "group": ["control"] * 8 + ["treatment"] * 8,
        "outcome": [4, 5, 5, 6, 4, 5, 6, 5, 7, 8, 7, 9, 8, 7, 9, 8],
    }
)

analyzer = StatisticalAnalyzer(df)
analysis = analyzer.analyze_all()
comparison = analyzer.hypothesis_tests(
    "group", "outcome", test_type="auto", estimand="mean", bootstrap_samples=0
)
insights = InsightEngine(analysis).get_summary()

print(analysis["descriptive"]["outcome"])
print(comparison["test"], comparison["p_value"])
print(insights["total_insights"])

report = ReportGenerator(analysis, insights, hypothesis_results=comparison)
report.to_json("analysis.json")
report.to_html("analysis.html")
```

`analyze_all()` preserves `overview`, `descriptive`, `normality`, `outliers`, `correlation`, `missing_data`, `data_quality`, `distributions`, `column_roles`, `column_types`, `histograms`, and `analysis_warnings`. Phase 3 adds `categorical_summary`, `variable_intelligence`, `data_dictionary`, and `profile_metadata`. Group comparisons are requested separately; pass their results to `ReportGenerator` to include them in reports.

## More examples

From a repository checkout, run the [complete feature showcase](https://github.com/majikoushik/PyAutoStat/blob/main/examples/README.md) to see printed output for every public workflow and generated report files:

```bash
python examples/example_usage.py --output-dir reports
```

Add `--skip-interactive` if you want only JSON, CSV, and static HTML. For your own data, replace the sample DataFrame with `pd.read_csv("data.csv")` and select the appropriate group and outcome columns.

## Data and statistical limits

- Input must be a nonempty DataFrame with unique, nonempty string column names. Missing values are allowed; unsupported nested, complex, or non-finite numeric values raise `InvalidDataError`.
- The analyzer copies its input. Undefined or skipped analyses appear as `None` or in `analysis_warnings`. Review these warnings before interpreting output.
- Group tests require **independent observations**, which the library cannot verify from values. `auto` requires `estimand="mean"` or `"distribution"`; it never changes that target because of a diagnostic p-value. Automatic multi-group mean comparison is unavailable until a suitable procedure is implemented. An explicit `ttest` uses Welch by default; `equal_var=True` requests Student's pooled-variance test.
- Normality and Levene results say `rejected`, `not_rejected`, or `unknown` at an advisory 0.05 threshold. A failure to reject does not establish an assumption. The legacy `is_normal` flag is retained as a screening flag, not a distribution classification.
- Constant data and unrepresentable statistics return unavailable values or package errors. Z-score and modified Z-score outlier counts are unavailable when their denominators are zero; no observations are removed.
- The chi-square method applies a conservative policy requiring expected counts of at least five in every cell. Bootstrap intervals are exploratory and do not adjust for multiple comparisons.
- Column type and role suggestions are advisory and never alter the input. A numeric column suggested to be an identifier from both its name and observed uniqueness is excluded from numerical profiling; a declared measurement type and role can override this selection. Low-cardinality integer hints alone do not recode values or remove numeric analyses.
- Processing is in memory. The interactive HTML loads Plotly JavaScript from a CDN when opened in a browser.

## Documentation

- [API reference](https://github.com/majikoushik/PyAutoStat/blob/main/API_REFERENCE.md): public methods, parameters, return values, and errors.
- [Examples guide](https://github.com/majikoushik/PyAutoStat/blob/main/examples/README.md): complete runnable showcase and output files.
- [Changelog](https://github.com/majikoushik/PyAutoStat/blob/main/CHANGELOG.md): shipped changes.
- [Roadmap](https://github.com/majikoushik/PyAutoStat/blob/main/ROADMAP.md): product goals and current status.

## Development

Install development dependencies and run the checks used in CI:

```bash
python -m pip install -e ".[dev]"
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/pyautostat
python -m pytest -q --cov=pyautostat --cov-report=term-missing --cov-fail-under=90
```

Issues and contributions are welcome through the [GitHub issue tracker](https://github.com/majikoushik/pyautostat/issues). Include a small reproducible DataFrame when reporting a data-handling problem.

## License

PyAutoStat is distributed under the [MIT License](https://github.com/majikoushik/PyAutoStat/blob/main/LICENSE).
