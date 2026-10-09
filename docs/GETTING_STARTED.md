# Getting started

This page takes a small fictional dataset through PyAutoStat's beginner analysis paths. Run the
Python blocks in order in one session. The core package is sufficient.

## Install

```bash
pip install pyautostat
```

PyAutoStat requires Python 3.10 or newer.

## Independent mean comparison

```python
import pandas as pd
from pyautostat import ResearchAssistant, show

df = pd.DataFrame({
    "group": ["standard"] * 6 + ["new"] * 6,
    "score": [62, 65, 68, 70, 72, 75, 67, 71, 74, 78, 80, 84],
})
assistant = ResearchAssistant(df)
workflow = assistant.compare_means("score", by="group")
print(workflow.brief())
```

`compare_means()` declares a mean target and routes through the same validated workflow as
`run()`. The independent form uses the guided Welch path. The first observed group minus the
second defines the signed contrast; inspect the recorded contrast rather than interpreting the
sign without its labels.

Profiling is useful but optional. To understand the dataset first:

```python
profile = assistant.profile()
print(profile["overview"])
```

## Paired mean comparison

Pairing is declared with a unit identifier. Row order is never used as pair identity, and
`condition_order` defines the signed first-minus-second contrast.

```python
paired_df = pd.DataFrame({
    "participant": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
    "condition": ["before", "after"] * 6,
    "score": [58, 63, 61, 65, 67, 70, 64, 69, 72, 75, 69, 74],
})
paired = ResearchAssistant(paired_df).compare_means(
    "score",
    by="condition",
    paired_by="participant",
    condition_order=("before", "after"),
)
print(paired.brief())
```

The estimate above is `before - after`. Reverse the order only when the scientific contrast is
truly `after - before`, not to obtain a preferred sign.

## Pearson, Spearman, and Kendall associations

The wrapper requires an explicit method; PyAutoStat does not choose among these targets
automatically.

```python
association_df = pd.DataFrame({
    "hours": [1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
    "score": [52, 55, 57, 61, 63, 68, 72, 74, 79, 83],
})
association = ResearchAssistant(association_df)

types = {"hours": "continuous", "score": "continuous"}
pearson = association.correlate("hours", "score", method="pearson", variable_types=types)
spearman = association.correlate("hours", "score", method="spearman", variable_types=types)
kendall = association.correlate("hours", "score", method="kendall", variable_types=types)

print(pearson.brief())
print(spearman.brief())
print(kendall.brief())
```

Pearson targets linear association, Spearman targets monotonic rank association, and Kendall
reports tau-b. None establishes causation.

## Read the result

Use the same hierarchy across workflows:

```python
print(workflow.brief())          # shortest result
print(workflow.apa_statement())  # reporting-oriented sentence
print(workflow.explain())        # detailed deterministic explanation
show(workflow)                   # full structured terminal presentation
```

These views consume stored values; they do not rerun the analysis. APA-oriented text is not a
claim of journal compliance.

## Type guidance

```python
print(workflow.type_guidance())
```

Inferred types are advisory. Declared types are researcher supplied. If scientific meaning is
ambiguous, the workflow can return `needs_input`; pass public type names such as `continuous`,
`discrete`, `nominal`, `ordinal`, `identifier`, or `boolean` through `variable_types`. You do not
need to inspect type guidance after every analysis.

## When a workflow needs input

Library calls never prompt unexpectedly. This intentionally incomplete paired request returns a
structured result without executing a test:

```python
incomplete = ResearchAssistant(paired_df).compare_means(
    "score",
    by="condition",
    paired_by="participant",
)
assert incomplete.status.value == "needs_input"
print(incomplete.missing_information)
```

Supply `condition_order=("before", "after")` after deciding which signed contrast answers the
research question. Similar requests preserve unknown event levels, reference levels, variable
meaning, and design facts rather than guessing.

## Next

- [Choose a task guide](task_guides/README.md).
- Read the [API reference](../API_REFERENCE.md) for signatures and contracts.
- Check [capabilities and limitations](CAPABILITIES.md).
- Learn about [reporting](README.md#reporting-and-export).
- Route planning, audit, replay, and bundles through [Advanced Workflows](ADVANCED_WORKFLOWS.md).
