# Run the complete example

For a minimal profile:

```python
import pandas as pd
from pyautostat import ResearchAssistant

df = pd.DataFrame({"group": ["A", "A", "B", "B"], "score": [10, 12, 15, 17]})
profile = ResearchAssistant(df).profile()
print(profile["overview"])
```

This only profiles the DataFrame. It does not recommend or execute a group test.
To prepare a group-comparison question without running a test:

```python
assistant = ResearchAssistant(df)
draft = assistant.prepare_question(
    objective="compare_groups", outcome="score", predictor="group",
    variable_types={"score": "continuous"},
)
print([item.field for item in draft.questions])  # estimand, design
draft = assistant.update_question(draft, estimand="mean", design="independent")
print(draft.status)  # ready for later design review, not an executed analysis
recommendation = assistant.recommend_test(draft)
print(recommendation.status, recommendation.method_id)  # ready, welch_t
print(recommendation.rationale)
print(recommendation.to_dict()["decision_trace"])
```

`recommend_test()` checks design and method compatibility without running a hypothesis test. Its `ready` result still requires researcher review of assumptions. The runnable showcase below covers the existing analyzer and report API; the short snippet above covers the Phase 5 recommendation API.

The showcase also demonstrates an optional data dictionary, row-level missingness,
categorical summaries, and pairwise correlation sample sizes.

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
| 1. Analysis | DataFrame copying; overview and dtypes; numerical and categorical summaries; normality methods when available; IQR, Z-score and MAD outliers; Pearson, Spearman and Kendall correlations with pairwise sample sizes and Pearson p-values; missing rows and cells; duplicate diagnostics; distributions; column roles and types; histogram bins; analysis warnings |
| 2. Column intelligence | `detect_column_types()` for numeric categories, continuous numbers, booleans, datetimes, date-like strings, email, URL, phone, text and empty columns; `suggest_column_roles()` for identifier, target, datetime, economic, measurement and unknown roles; optional validated score metadata |
| 3. Insights | `InsightEngine.generate_insights()` and `get_summary()` with severity counts, findings and recommendations |
| 3a. Method recommendation | Phase 5 question intake and a structured Welch recommendation with rationale and decision trace; no test executed by this section |
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
