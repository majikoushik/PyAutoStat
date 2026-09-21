# PyAutoStat

[![CI](https://github.com/majikoushik/pyautostat/actions/workflows/ci.yml/badge.svg)](https://github.com/majikoushik/pyautostat/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

PyAutoStat analyzes pandas DataFrames and returns structured statistical results, data quality findings, and reports. It supports exploratory analysis and independent group comparisons. The package is in **alpha**; review assumptions and results before using them in research or decisions.

## Features

- **Explore data:** descriptive statistics, missing values, duplicates, distributions, histograms, and advisory column type and role detection.
- **Check assumptions:** Shapiro-Wilk, D'Agostino-Pearson, and Anderson-Darling normality results; IQR, Z-score, and MAD outlier summaries.
- **Study relationships:** Pearson, Spearman, and Kendall correlations, with p-values for Pearson pairs.
- **Compare independent groups:** automatic or explicit t-test, Mann-Whitney U, one-way ANOVA, and Kruskal-Wallis; assumption checks, effect sizes, and confidence intervals.
- **Test categorical association:** Pearson chi-square, Cramér's V, and Cohen's h for a two-by-two table with a named success outcome.
- **Share results:** severity-rated insights and dictionary, JSON, CSV, static HTML, or optional Plotly HTML reports.

## Installation

Requires Python 3.10 or newer. From a checkout of this repository:

```bash
python -m pip install .
```

For interactive HTML reports, install the optional Plotly dependency:

```bash
python -m pip install ".[report]"
```

## Quick start

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
    "group", "outcome", test_type="auto", bootstrap_samples=0
)
insights = InsightEngine(analysis).get_summary()

print(analysis["descriptive"]["outcome"])
print(comparison["test"], comparison["p_value"])
print(insights["total_insights"])

report = ReportGenerator(analysis, insights, hypothesis_results=comparison)
report.to_json("analysis.json")
report.to_html("analysis.html")
```

`analyze_all()` returns sections named `overview`, `descriptive`, `normality`, `outliers`, `correlation`, `missing_data`, `data_quality`, `distributions`, `column_roles`, `column_types`, `histograms`, and `analysis_warnings`. Group comparisons are requested separately; pass their results to `ReportGenerator` to include them in reports.

## More examples

Run the [complete feature showcase](examples/README.md) to see printed output for every public workflow and generated report files:

```bash
python examples/example_usage.py --output-dir reports
```

Add `--skip-interactive` if you want only JSON, CSV, and static HTML. For your own data, replace the sample DataFrame with `pd.read_csv("data.csv")` and select the appropriate group and outcome columns.

## Data and statistical limits

- Input must be a nonempty DataFrame with unique, nonempty string column names. Missing values are allowed; unsupported nested, complex, or non-finite numeric values raise `InvalidDataError`.
- The analyzer copies its input. Undefined or skipped analyses appear as `None` or in `analysis_warnings`. Review these warnings before interpreting output.
- Group tests are for **independent observations**. The automatic choice uses normality and variance screens; it cannot establish that a study design or statistical model is appropriate.
- Chi-square association requires expected counts of at least five in every cell. Bootstrap intervals are exploratory and do not adjust for multiple comparisons.
- Column type and role suggestions are advisory; they do not alter the input or choose analysis columns.
- Processing is in memory. The interactive HTML loads Plotly JavaScript from a CDN when opened in a browser.

## Documentation

- [API reference](API_REFERENCE.md): public methods, parameters, return values, and errors.
- [Examples guide](examples/README.md): complete runnable showcase and output files.
- [Changelog](CHANGELOG.md): shipped changes.
- [Roadmap](ROADMAP.md): product goals and current status.

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

PyAutoStat is distributed under the [MIT License](LICENSE).
