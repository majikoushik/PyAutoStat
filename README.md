# PyAutoStat

> From a pandas DataFrame to a defensible statistical analysis, with the question, assumptions,
> effect, uncertainty, interpretation, and audit trail kept together.

[![CI](https://github.com/majikoushik/PyAutoStat/actions/workflows/ci.yml/badge.svg)](https://github.com/majikoushik/PyAutoStat/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: Stable](https://img.shields.io/badge/Status-Stable-brightgreen.svg)](#project-status)

**PyAutoStat** is an open-source, deterministic research analysis assistant for Python and pandas.
It helps researchers state a question, preserve the intended target, run a validated calculation,
and inspect or export the same recorded result. It works offline and requires no generative AI,
cloud service, or GUI.

## Why PyAutoStat?

- Declare the scientific question and target quantity before selecting a method.
- Ask for essential design facts such as pairing rather than guess them from values.
- Report effects, uncertainty, sample exclusions, assumptions, warnings, and limitations together.
- Produce deterministic interpretation, audit records, and reproducible exports from one result.

PyAutoStat checks internal consistency; internal consistency does not prove scientific truth.

## Installation

PyAutoStat requires Python 3.10 or newer.

```bash
pip install pyautostat
```

Optional reporting extras are listed under [Reporting and export](#reporting-and-export).

## 60-second quickstart

This fictional example directly compares two independent group means. Profiling can help you
understand a dataset, but it is not required before analysis.

```python
import pandas as pd
from pyautostat import ResearchAssistant

df = pd.DataFrame({
    "group": ["standard"] * 6 + ["new"] * 6,
    "score": [62, 65, 68, 70, 72, 75, 67, 71, 74, 78, 80, 84],
})
assistant = ResearchAssistant(df)
workflow = assistant.compare_means("score", by="group")
print(workflow.brief())
```

Actual output from the code above:

```text
Welch independent-samples t-test: mean difference = -7.00 (standard minus new); 95% CI [-14.17, 0.17]; t(9.32) = -2.20; p = .055; Cohen's d = -1.27; N = 12, excluded rows = 0; Warning: Independence must be confirmed from the study design.
```

The group order defines the signed contrast. A non-significant p-value does not prove that the
means are equal, and the effect estimate and interval remain part of the result.

## Choose your task

| I want to... | Start with |
| --- | --- |
| Understand my dataset | `assistant.profile()` |
| Compare independent means | `assistant.compare_means("score", by="group")` |
| Compare paired means | `assistant.compare_means(..., paired_by="id", condition_order=(...))` |
| Study ordinary correlation | `assistant.correlate("x", "y", method="pearson")` |
| Run another design or estimand | `assistant.run(...)` |
| Read the shortest result | `workflow.brief()` |
| Inspect type guidance when useful | `workflow.type_guidance()` |
| See detailed reasoning | `workflow.explain()` |
| Render full output | `show(workflow)` |

See [Getting Started](docs/GETTING_STARTED.md) for runnable independent, paired, and correlation
examples, or browse the [task guides](docs/task_guides/README.md).

## Scientific safeguards and what happens under the hood

The beginner methods are thin routes into the same design-aware workflow used by `run()`:

```text
question and estimand -> declared design -> recommendation -> validated calculation
-> deterministic interpretation -> report -> audit and reproducibility record
```

PyAutoStat does not silently remove outliers, impute or recode values, change alpha, infer pairing
from row order, or switch a mean question to a rank question because a diagnostic rejected. If a
scientific fact is unresolved, the workflow returns `needs_input` and preserves a structured
request instead of prompting inside the library call.

Friendly diagnostics identify the supplied field and may offer one bounded suggestion:

```text
Column 'scroe' supplied for `outcome` was not found. Did you mean 'score'?
```

The suggestion is never applied automatically. Invalid event, reference, and condition levels
show observed valid choices; the researcher still decides their scientific meaning.

### Advisory type guidance

For ambiguous data, inspect guidance when it helps:

```python
print(workflow.type_guidance())
```

An inferred type is advisory, not a confirmed measurement type. A declared type is supplied by
the researcher. Ambiguity can produce `needs_input`; override it with public type names through
`variable_types`, for example `{"score": "continuous", "group": "nominal"}`. Not every workflow
requires this inspection.

## Result views

Every view consumes the stored workflow result; presentation does not choose a new method or
recalculate statistics.

| View | Purpose |
| --- | --- |
| `workflow.brief()` | Shortest result |
| `workflow.apa_statement()` | Reporting-oriented sentence |
| `workflow.explain()` | Detailed deterministic explanation |
| `show(workflow)` | Full structured terminal presentation |
| `save_*` / `to_*` | Export and report surfaces |

```python
from pyautostat import show

show(workflow)
```

`brief()` uses stored values only and includes bounded warnings by default. APA-oriented output is
a reporting aid, not certification of journal compliance.

## Beginner, advanced, and direct APIs

### Beginner

- `compare_means()` targets ordinary independent or explicitly paired means.
- `correlate()` requires an explicit `"pearson"`, `"spearman"`, or `"kendall"` method.
- `brief()` gives the shortest deterministic result view.

These conveniences delegate to the same validated workflow as an equivalent `run()` call. They do
not add methods, select a correlation method automatically, or change result schemas.

### Advanced and design-aware

Use `run()` when the research question needs an explicit estimand, a reference value, three or
more groups, repeated measures, factorial analysis, regression, partial or categorical
association, or another advanced design/model option. Essential design facts remain explicit.

### Low-level and retained public surfaces

`StatisticalAnalyzer` provides direct pre-specified calculations. `InsightEngine` and
`ReportGenerator` are retained as compatibility-stable APIs for existing dictionary-oriented
workflows. New work should use `ResearchAssistant` and the canonical presentation/export APIs.
See the [1.0 public API boundary](docs/PUBLIC_API_1_0.md), [API reference](API_REFERENCE.md), and
[API stability policy](docs/API_STABILITY.md).

## Supported methods and capabilities

PyAutoStat currently registers **24 supported statistical methods**. They are grouped into fewer
analysis families; method IDs are not themselves families.

| Research task | Recommended entry point | Notes |
| --- | --- | --- |
| Dataset profile | `assistant.profile()` | Descriptive; optional before analysis |
| Independent or paired means | `assistant.compare_means(...)` | Paired order defines first-minus-second |
| Distribution/rank comparison | `assistant.run(..., estimand="distribution")` | Keeps the distribution target explicit |
| One mean versus a reference | `assistant.run(..., objective="compare_reference")` | Reference is researcher supplied |
| Pearson, Spearman, Kendall | `assistant.correlate(..., method=...)` | Method is never auto-selected by the wrapper |
| Point-biserial, partial Pearson, categorical association | `assistant.run(...)` | Requires explicit roles/orientation where applicable |
| Three or more independent groups | `assistant.run(...)` | Welch ANOVA or explicit rank target; classical alternative is direct |
| Repeated measurements, 3+ conditions | `assistant.run(..., design="repeated")` | Requires unit ID and ordered conditions |
| Two-factor independent ANOVA | `assistant.two_way_anova(...)` or `run(...)` | Type II or Type III sums of squares |
| OLS or binary logistic regression | `assistant.run(..., objective="regression")` | Explicit predictors, coding, covariance/event choices |
| Cronbach's alpha | `assistant.reliability(...)` | Researcher-declared item set |
| Intraclass correlation | `assistant.intraclass_correlation(...)` | Explicit target, rater, model, definition, and unit |

For exact contracts, use [Capabilities](docs/CAPABILITIES.md) and the
[Statistical Method Contracts](docs/STATISTICAL_METHOD_CONTRACTS.md). Numerical validation is
scoped to specified methods, fields, and reference cases; it is not proof that a study is valid.

## Reporting and export

Core installation includes terminal presentation, static HTML, Markdown, JSON, CSV tables, and
safe LaTeX. File output is explicit; `run()` itself writes nothing.

```python
from pyautostat import save_html

save_html(workflow, "analysis.html")
```

Install only the extras you need:

| Need | Installation | Additional setup |
| --- | --- | --- |
| Interactive Plotly HTML | `pip install "pyautostat[report]"` | None |
| Publication-oriented PDF | `pip install "pyautostat[pdf]"` | `python -m playwright install chromium` |
| Editable Word (DOCX) | `pip install "pyautostat[docx]"` | None |
| Static PNG/SVG/figure PDF | `pip install "pyautostat[figures]"` | `plotly_get_chrome -y` |

See the [reporting guides](docs/README.md#reporting-and-export) for HTML, PDF, DOCX, terminal, and figure
details. Publication-oriented output still requires expert and journal-specific review.

## Advanced reproducibility and governance

PyAutoStat also supports analysis plans, prospective study planning, researcher-defined practical
significance, sensitivity scenarios, reporting-completeness checks, audits, dataset fingerprints,
explicit replay, session snapshots, and research bundles. These records organize and compare
recorded evidence; they do not authenticate data, prove preregistration, certify study quality, or
establish causality.

Start at [Advanced Workflows](docs/ADVANCED_WORKFLOWS.md) instead of assembling these APIs from the
README.

## Documentation map

| Goal | Document |
| --- | --- |
| First analysis | [Getting Started](docs/GETTING_STARTED.md) |
| Choose by research task | [Task Guides](docs/task_guides/README.md) |
| Public signatures and contracts | [API Reference](API_REFERENCE.md) |
| Supported methods and boundaries | [Capabilities](docs/CAPABILITIES.md) |
| Advanced lifecycle and reproducibility | [Advanced Workflows](docs/ADVANCED_WORKFLOWS.md) |
| Numerical evidence | [Numerical Validation](docs/NUMERICAL_VALIDATION.md) |
| Realistic customer examples | [Examples](examples/README.md) |
| All documentation | [Documentation Home](docs/README.md) |

## Scope and limitations

The bounded method catalogue intentionally does not include mixed-effects models or mixed ANOVA,
factorial repeated-measures ANOVA, generalized estimating equations (GEE), survival analysis,
causal inference, automatic imputation or outlier deletion, automatic model/predictor selection,
formal equivalence or noninferiority tests, or observed post-hoc power. These boundaries prevent an
unsupported design from being silently reinterpreted as a supported one.

See [Scientific Limitations](docs/SCIENTIFIC_LIMITATIONS.md) for details. Researchers remain
responsible for design, measurement meaning, sampling assumptions, and external review.

## Project status

PyAutoStat is version **1.0.0** and has a stable public 1.x compatibility commitment. Documented
top-level exports and stable serialized contracts follow the compatibility and deprecation policy;
compatible additions and scientifically necessary corrections remain possible. Stable software
does not imply universal statistical coverage, APA certification, guaranteed publication
readiness, or scientific validity.

See the [changelog](CHANGELOG.md), [roadmap](ROADMAP.md), and
[migration guide](docs/MIGRATION_TO_1_0.md).

## Compatibility

- Python 3.10+
- pandas DataFrames
- NumPy and SciPy numerical backends
- Rich terminal presentation in the core package
- Plotly, Playwright/Chromium, python-docx, and Kaleido/Chrome only for their optional features

## Contributing

Contributions are welcome. Before opening a pull request:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/pyautostat
python -m pytest -q --cov=pyautostat --cov-report=term-missing --cov-fail-under=90
```

Read the [product vision](PRODUCT_VISION.md), [roadmap](ROADMAP.md), and project-wide contributor
instructions before proposing statistical behavior changes.

## Citation

If you use PyAutoStat in research, cite the software version and repository. A generic form is:

```text
PyAutoStat contributors. PyAutoStat (version 1.0.0) [Computer software].
https://github.com/majikoushik/PyAutoStat
```

Also cite the original methodological sources appropriate to the methods used in your analysis.

## License

PyAutoStat is released under the [MIT License](LICENSE).
