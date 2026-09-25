# Run the complete example

For the shortest integrated workflow, run:

```bash
python examples/integrated_workflow_example.py
```

It uses a synthetic examination-score dataset to show a one-call profile, a `needs_input` design
question, immutable draft continuation, Welch execution, interpretation, report audit, and
reproducibility metadata. It writes no files. Use `workflow.report.save_html(path)` or another
`ResearchReport` save method when an explicit export is wanted. The exact support and failure-mode matrices
are in [`docs/CONTROLLED_MVP.md`](../docs/CONTROLLED_MVP.md).

For explicit sensitivity and a researcher-defined meaningful threshold, run:

```bash
python examples/phase11_sensitivity_example.py
```

It retains a pooled-variance same-estimand scenario and a Mann–Whitney different-estimand
scenario, classifies the point estimate and interval against a five-point threshold, builds an
optional schema 2 report, and audits it without rerunning scenarios. It writes no files and does
not select or rank scenarios by p-value. See
[`docs/ROBUSTNESS_AND_PRACTICAL_SIGNIFICANCE.md`](../docs/ROBUSTNESS_AND_PRACTICAL_SIGNIFICANCE.md).

For prospective planning, analysis-plan creation, an explicit-ID paired mean workflow,
report completeness, oriented presentation, and a JSON session snapshot, run:

```bash
python examples/phase12_planning_and_paired_example.py
```

The example writes no files, never derives planning inputs from the observed result, and reports
only aggregate pair counts. See
[`docs/ADVANCED_PLANNING_AND_PRESENTATION.md`](../docs/ADVANCED_PLANNING_AND_PRESENTATION.md).

For a minimal profile:

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
result = assistant.analyze(draft)
print(result.method_id, result.status)
print(result.values["primary_estimate"], result.values["p_value"])
print(result.metadata["sample"], result.metadata["contrast"])
interpretation = assistant.interpret(result)
print(interpretation.summary)
print([finding.code for finding in interpretation.findings])
```

`recommend_test()` checks design and method compatibility without running a hypothesis test. `analyze()` revalidates that decision and calls an existing backend, returning a structured result tied to the original specification. `interpret()` turns its recorded evidence into deterministic coded findings and qualified text without recalculating. An available result still requires researcher review of assumptions. The runnable showcase below also covers the existing analyzer and report API.

For the canonical research report, run `python examples/research_report_example.py`. It executes a Welch comparison and a Pearson association, then obtains HTML, Markdown, JSON, and CSV content in memory. The Pearson report is partial because the current backend has no correlation confidence interval. The example writes no files. Use explicit `report.save_html(path)` or the other save methods when a file is wanted. This new `ResearchReport` workflow is separate from the legacy `ReportGenerator` showcase.

For the observed-event ledger, report audit, fingerprint check, and explicit replay, run `python examples/phase9_reproducibility_example.py`. It also alters one report value and one observation to demonstrate failed consistency checks. It writes no files or raw-data package.

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
| 3a. Method recommendation and execution | Question intake and structured Welch recommendation, then execution with estimate, p-value and sample accounting |
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
