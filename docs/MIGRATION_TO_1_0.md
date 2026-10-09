# Migrating to PyAutoStat 1.0

PyAutoStat 1.0 makes the documented 1.x compatibility commitment without changing the statistical
method catalogue or invalidating supported 0.5 workflows. This guide covers the recommended
surface and the compatibility status of older composition APIs.

## Canonical API

Use `ResearchAssistant` for new work:

```python
import pandas as pd
from pyautostat import ResearchAssistant

df = pd.DataFrame(
    {
        "group": ["standard"] * 6 + ["new"] * 6,
        "score": [62, 65, 68, 70, 72, 75, 67, 71, 74, 78, 80, 84],
    }
)

assistant = ResearchAssistant(df)
workflow = assistant.compare_means("score", by="group")
print(workflow.brief())
```

`profile()` remains the dataset-only beginner entry point. `run()` remains the complete guided
path for explicit estimands, reference comparisons, multigroup/repeated/factorial designs,
regression, reliability, specialized association, and governance options.

## Beginner conveniences

`compare_means()` supports independent means and explicit two-condition pairing:

```python
independent = assistant.compare_means("score", by="group")

paired_df = pd.DataFrame(
    {
        "participant": [1, 1, 2, 2, 3, 3, 4, 4],
        "condition": ["before", "after"] * 4,
        "score": [58, 63, 61, 65, 67, 70, 64, 69],
    }
)
paired = ResearchAssistant(paired_df).compare_means(
    "score",
    by="condition",
    paired_by="participant",
    condition_order=("before", "after"),
)
```

`correlate()` requires the researcher to choose `"pearson"`, `"spearman"`, or `"kendall"`:

```python
association_df = pd.DataFrame(
    {
        "hours": [1, 2, 3, 4, 5, 6],
        "score": [52, 55, 61, 64, 70, 74],
    }
)
association = ResearchAssistant(association_df).correlate(
    "hours",
    "score",
    method="pearson",
)
```

These methods delegate to `run()` and return the same `ResearchWorkflowResult`; they do not add a
second statistical engine or select a method after inspecting p-values.

## Constructor `AnalysisOptions`

Stable defaults can be supplied once:

```python
from pyautostat import AnalysisOptions, ResearchAssistant

assistant = ResearchAssistant(
    df,
    options=AnalysisOptions(confidence_level=0.95, random_seed=7),
)
```

A call-specific `options=` object replaces the complete constructor default for that call. Options
are not merged field by field.

## Reading results

- `workflow.brief()` returns the shortest deterministic stored-result summary.
- `workflow.type_guidance()` exposes declared and advisory variable-type information without
  rerunning profiling.
- `workflow.explain()` provides detailed deterministic interpretation.
- `workflow.apa_statement()` provides APA-oriented reporting text, not compliance certification.
- `show(workflow)` renders the complete terminal presentation.

## Advanced `run()` path

Existing explicit calls remain supported:

```python
workflow = assistant.run(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    estimand="mean",
    design="independent",
    variable_types={"score": "continuous", "group": "nominal"},
)
```

Essential design facts are still never guessed. An unresolved request returns structured
`needs_input` state instead of prompting inside the library call.

## Public export cleanup

No top-level exports were removed for 1.0. All 97 names in `pyautostat.__all__` are frozen at their
documented tier. Low-level narration, detection, contract, and capability helpers were retained
because they were already documented and used by tests or integration surfaces.

## Legacy composition APIs

`InsightEngine` and `ReportGenerator` are `COMPATIBILITY_STABLE`. They remain importable, tested,
and supported without `DeprecationWarning` in 1.0.0. Existing 0.1-style composition code continues
to run.

Preferred replacements for new work are:

| Existing API | Preferred 1.x path |
| --- | --- |
| `StatisticalAnalyzer(df).analyze_all()` for profiling | `ResearchAssistant(df).profile()` |
| `InsightEngine(...)` | `workflow.interpretation`, `brief()`, or `explain()` |
| `ReportGenerator(...)` | `ResearchReport`, `show`, `save_html`, `save_pdf`, `save_docx`, or `save_bundle` |

`StatisticalAnalyzer` is not legacy; it remains the advanced direct-calculation API.

## Serialization and reporting compatibility

Established result/specification keys and schema versions remain readable. The 1.0 consolidation
does not alter p-values, effects, intervals, recommendations, missing-data rules, outlier policy,
or report calculations. Presentation and exports continue to consume recorded validated values
without recalculation.

Minor 1.x releases may add optional fields or status values. Consumers should tolerate unknown
additive fields and use recorded schema versions. See [API stability](API_STABILITY.md),
[Public API 1.0](PUBLIC_API_1_0.md), and the [API reference](../API_REFERENCE.md).
