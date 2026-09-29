# PyAutoStat examples

These tutorials progress from dataset profiling to a traceable research
workflow. They use the bundled **CustomerDataset.xlsx** workbook (5,000 rows
and 40 columns). The workbook is demonstration data; this repository does not
establish that it represents a sampled population or supports causal claims.

The examples never print customer identifiers. The advanced paired example
uses **CustomerID** only to match within-customer measurements.

For a small synthetic example with no workbook dependency, run:

~~~bash
python examples/explainable_recommendation_and_report.py
python examples/data_understanding.py
python examples/basic_inference.py
python examples/multigroup_analysis.py
python examples/linear_regression.py
python examples/scale_reliability.py
python examples/binary_and_extended_association.py
~~~

It prints the dataset story, deterministic recommendation rationale (why this
method, why not relevant alternatives, and facts to verify), and the complete
researcher-facing workflow explanation, then verifies that the generated HTML
contains its executive summary. No file is written and no generative service
is used.

`data_understanding.py` demonstrates default percentiles, a safeguarded CV, categorical frequency
tables, ordered cumulative percentages, a descriptive cross-tab with four explicit table views,
and their deterministic narratives using synthetic data only.

`basic_inference.py` demonstrates the complete one-sample t, paired Wilcoxon signed-rank,
inferential Spearman, and sparse 2x2 Fisher paths. It prints researcher-readable workflow
explanations and requires no external files. Wilcoxon and Fisher explicitly disclose that their
primary-effect confidence intervals are unavailable.

`multigroup_analysis.py` demonstrates guided Welch ANOVA with Games-Howell, guided
Kruskal-Wallis with Dunn-Holm, and explicit classical ANOVA with Tukey-Kramer. Every structured
pair is retained even when the printed explanation is bounded.

`linear_regression.py` demonstrates simple and multiple OLS regression with a continuous
predictor, categorical and Boolean treatment coding, explicit reference levels, HC3 covariance,
coefficient interpretation, stored diagnostics, `workflow.explain()`, report CSV tables, and
audit. It uses synthetic data and writes no files.

`scale_reliability.py` demonstrates the beginner Cronbach-alpha workflow with an explicit item
set, declared reverse scoring, complete-case accounting, item diagnostics, deterministic
bootstrap uncertainty, audit, and replay. It uses synthetic data and writes no files.

`binary_and_extended_association.py` demonstrates binary logistic regression, exact long-format
McNemar inference, point-biserial correlation, explicit Kendall tau-b inference, and partial
Pearson correlation. Every binary direction, pair identity, condition order, and control variable
is declared; the example uses synthetic data and writes no files.

## Install

~~~bash
python -m pip install -e ".[examples]"
~~~

The **examples** extra installs openpyxl, which pandas needs to read the
workbook. Core PyAutoStat does not require Excel support.

## 1. Profile and understand a dataset

~~~bash
python examples/01_quick_start.py
~~~

[01_quick_start.py](01_quick_start.py) demonstrates:

- **ResearchAssistant.summarize()** for a compact beginner view;
- opt-in **ResearchAssistant.summarize(mode="story")** for a connected profile narrative;
- declared data meaning for numeric category codes;
- the structured **ResearchAssistant.profile()** result;
- normality and IQR diagnostics without automatic deletion or method switching;
- **InsightEngine** findings and recommendations; and
- safe handling of the unique identifier.

The profile also contains descriptive statistics, missingness, data quality,
histograms, distribution shape, column intelligence, and Pearson, Spearman,
and Kendall correlation summaries.

## 2. Run estimand-aware hypothesis tests

~~~bash
python examples/02_hypothesis_testing.py
~~~

[02_hypothesis_testing.py](02_hypothesis_testing.py) uses the compatibility
**StatisticalAnalyzer** API to expose the numerical result dictionaries:

| Research target | Declared estimand | Method shown |
|---|---|---|
| Spending difference by gender | population mean difference | Welch independent t test |
| Product A spending across job categories | rank distributions | Kruskal-Wallis |
| Home ownership and value category | categorical association | Pearson chi-square |
| Age by active-lifestyle category | rank distributions | Mann-Whitney U |

The output includes analyzed and excluded rows, assumption diagnostics,
selection rationale, statistics, p-values, effect estimates, supported
confidence intervals, warnings, and deterministic interpretation.

The automatic mean comparison preserves the declared mean estimand.
Normality and variance diagnostics remain visible, but a diagnostic p-value
does not silently change the scientific question. Bootstrap intervals are
reported only when the requested interval can be computed from enough valid
resamples.

## 3. Follow the complete research lifecycle

~~~bash
python examples/03_advanced_workflow.py --output-dir reports
~~~

[03_advanced_workflow.py](03_advanced_workflow.py) demonstrates the modern
**ResearchAssistant** workflow:

1. An omitted independence declaration returns structured **needs_input**.
2. The caller supplies the missing design fact explicitly.
3. A local statistical analysis plan is recorded before numerical execution.
4. The selected method preserves the design and mean-difference estimand.
5. Sensitivity scenarios distinguish:
   - a same-estimand pooled-variance comparison; and
   - a different-estimand Mann-Whitney comparison.
6. A researcher-defined meaningful-effect threshold is assessed separately
   from null-hypothesis significance.
7. Plan adherence compares the planned and performed analysis without making
   conduct judgments.
8. A canonical **ResearchReport** is rendered to APA-oriented HTML, Markdown,
   JSON, and CSV tables from the same validated values.
9. The report is audited, a reproducibility record is replayed against the
   same data, and a serializable session snapshot is created.
10. Prospective power planning uses researcher-supplied assumptions and does
    not consume the observed effect.
11. A second workflow performs explicit two-condition paired analysis using a
    unit identifier and ordered contrast.
12. A decision ledger records the actions observed by that assistant instance.

Generated files:

| Path | Contents |
|---|---|
| **customer_analysis.html** | Canonical styled research report |
| **customer_analysis.md** | Markdown report |
| **customer_analysis.json** | JSON-safe report record |
| **customer_analysis_tables/** | Stable CSV tables |
| **session_snapshot.json** | UI-independent workflow snapshot |
| **decision_ledger.json** | Locally observed decision events |

The example overwrites only these named tutorial outputs so it can be rerun.
The report omits the complete input DataFrame and customer identifier values.
Small aggregate cells can still disclose information and need contextual
review before sharing.

## What the examples establish

The examples show how the public interfaces work and verify that the bundled
data can exercise them. They do not establish:

- that the workbook is representative of a target population;
- independence, randomization, causal identification, or collection intent;
- that a statistically detectable result is practically important;
- that a different-estimand sensitivity result confirms the primary estimand;
- equivalence from a failure to reject a superiority null; or
- external preregistration from a local plan or decision ledger.

## Use your own DataFrame

Replace the workbook loader with any pandas DataFrame and update the declared
column names and meanings:

~~~python
import pandas as pd
from pyautostat import ResearchAssistant

frame = pd.read_csv("your_data.csv")
assistant = ResearchAssistant(frame)
print(assistant.summarize())
print(assistant.summarize(mode="story"))
~~~

For detailed signatures and supported method families, see the
[API reference](../API_REFERENCE.md) and
[capability matrix](../docs/CAPABILITIES.md).
