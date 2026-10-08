# Mean comparisons

## Use this when

Use these paths when the scientific target is a population mean, a paired mean difference, an
explicit distribution/rank contrast, or one mean relative to a researcher-supplied reference.

Minimum inputs are the numeric outcome, grouping or condition column when applicable, the study
design, and enough context to define the contrast. Paired data also require a unit identifier and
ordered condition labels.

## Independent means: beginner path

```python
import pandas as pd
from pyautostat import ResearchAssistant

df = pd.DataFrame({
    "group": ["standard"] * 6 + ["new"] * 6,
    "score": [62, 65, 68, 70, 72, 75, 67, 71, 74, 78, 80, 84],
})
assistant = ResearchAssistant(df)
independent = assistant.compare_means("score", by="group")
print(independent.brief())
```

The wrapper targets means and uses the guided independent Welch path. Independence remains a
study-design claim, not something the values can prove.

## Paired means: beginner path

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

`condition_order` defines the signed `before - after` contrast. PyAutoStat never pairs rows by
position. Duplicate unit-condition observations or incomplete pairs remain visible in validation
and sample accounting.

## Distribution or rank questions: advanced path

`compare_means()` does not target distributions or ranks. Declare that estimand through `run()`:

```python
distribution = assistant.run(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    estimand="distribution",
    design="independent",
)
print(distribution.brief())
```

For two independent groups this routes to Mann-Whitney U; for three or more it routes to
Kruskal-Wallis with the documented follow-up. These are distribution/rank targets, not automatic
fallbacks after a normality diagnostic and not universal median tests.

## One sample versus a reference: advanced path

```python
reference = assistant.run(
    objective="compare_reference",
    outcome="score",
    estimand="mean",
    design="independent",
    reference_value=70.0,
    variable_types={"score": "continuous"},
)
print(reference.brief())
```

The primary difference is observed mean minus the supplied reference. PyAutoStat does not infer
the reference from the data.

## Read the result

Use `brief()` for the shortest stored-value view, `apa_statement()` for a reporting-oriented
sentence, `explain()` for detailed deterministic reasoning, and `show(workflow)` for the full
terminal presentation. Review group order, pair order, exclusions, effect estimate, interval,
assumptions, and warnings together.

## Scientific caveat

A diagnostic p-value does not silently change the estimand. Non-rejection is not proof of
normality, equal means, or equivalence. Pairing and independence require researcher knowledge.

For signatures and complete contracts, see [ResearchAssistant common workflows](../../API_REFERENCE.md#researchassistant-common-workflows)
and [independent group comparisons](../../API_REFERENCE.md#independent-group-comparisons).
