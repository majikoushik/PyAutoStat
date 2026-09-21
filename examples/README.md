# Run the complete example

For a minimal profile using the Phase 1 facade:

```python
import pandas as pd
from pyautostat import ResearchAssistant

df = pd.DataFrame({"group": ["A", "A", "B", "B"], "score": [10, 12, 15, 17]})
profile = ResearchAssistant(df).profile()
print(profile["overview"])
```

This only profiles the DataFrame. It does not recommend or execute a group test.

From the repository root, install the package and run the showcase:

```bash
python -m pip install -e ".[report]"
python examples/example_usage.py --output-dir reports
```

The script creates its own reproducible dataset, prints the results in labeled
sections, and writes reports to the chosen directory. It needs no input file.
Plotly is optional; without it, the script prints a skip message and still
creates the other reports. To skip the interactive export explicitly:

```bash
python examples/example_usage.py --output-dir reports --skip-interactive
```

`--sample-size` changes the generated row count (minimum 60, default 300).
`--verbose` prints the full `analyze_all()` result after the guided tour.
The data is synthetic and the p-values are examples of API output, not findings
from a real study.

## What the script demonstrates

| Section | Features and output shown |
|---|---|
| 1. Analysis | DataFrame copying; overview and dtypes; descriptive statistics; three normality methods; IQR, Z-score and MAD outliers; Pearson, Spearman and Kendall correlations with Pearson p-values; missing data; quality metrics; distributions; column roles and types; histogram bins; analysis warnings |
| 2. Column intelligence | `detect_column_types()` for numeric categories, continuous numbers, booleans, datetimes, date-like strings, email, URL, phone, text and empty columns; `suggest_column_roles()` for identifier, target, datetime, economic, measurement and unknown roles |
| 3. Insights | `InsightEngine.generate_insights()` and `get_summary()` with severity counts, findings and recommendations |
| 4. Group comparisons | Automatic selection with an explicit mean or distribution target, plus explicit Welch t-test, Mann-Whitney U, ANOVA and Kruskal-Wallis; diagnostic statuses; effect sizes; bootstrap intervals; analytical t-test interval |
| 5. Categorical association | Chi-square, observed and expected counts, Cramér's V, Cohen's h for a named success outcome, and a multi-category table |
| 6. Reports | `to_dict()`, in-memory and file JSON, in-memory and written CSV, in-memory and file static HTML, and in-memory and file optional interactive HTML; hypothesis results included in exports |
| 7. Bad data | Unavailable results and warnings for missing, short and constant columns; examples of every public package exception subclass |

## Files produced

```text
reports/
├── analysis.json
├── analysis.html
├── interactive.html       # when Plotly is available
└── csv/
    ├── descriptive_stats.csv
    ├── outliers.csv
    ├── missing_data.csv
    ├── insights.csv
    └── hypothesis_tests.csv
```

Open `analysis.html` for the static report. The interactive report has
collapsible sections, searchable and sortable tables, and charts that render
when their section opens. Its browser view loads Plotly JavaScript from a CDN.

The source in [example_usage.py](example_usage.py) is organized into one
function per feature area, so a reader can copy a focused section into their
own project. The [API reference](../API_REFERENCE.md) documents the return
structures and limitations.
