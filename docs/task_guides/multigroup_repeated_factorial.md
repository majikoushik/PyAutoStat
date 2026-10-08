# Multigroup, repeated, and factorial designs

These are different designs. Do not treat a repeated panel or a two-factor experiment as an
ordinary one-way independent-groups comparison.

## Independent groups: three or more

Use this when each observation belongs to one independent group. Minimum inputs are the outcome,
one grouping variable, an independent-design declaration, and the estimand.

```python
import pandas as pd
from pyautostat import ResearchAssistant, StatisticalAnalyzer

groups_df = pd.DataFrame({
    "group": ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
    "score": [12, 14, 15, 16, 18, 19, 15, 17, 18, 20, 21, 23, 20, 22, 23, 25, 27, 28],
})
welch = ResearchAssistant(groups_df).run(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    estimand="mean",
    design="independent",
)
print(welch.brief())
```

The guided mean path uses Welch one-way ANOVA with Games-Howell comparisons. If the research plan
justifies the classical equal-variance model, request the direct pre-specified alternative:

```python
classical = StatisticalAnalyzer(groups_df).hypothesis_tests(
    "group",
    "score",
    test_type="anova",
    estimand="mean",
)
print(classical["test"])
```

For a distribution/rank target, use `estimand="distribution"`; with three or more groups this
routes to Kruskal-Wallis plus documented Dunn-Holm follow-up.

## Repeated measurements: three or more conditions

Use this when the same units are observed in every ordered condition. Minimum inputs are a unit ID,
condition column, outcome, at least three condition labels, and a mean or distribution target.

```python
repeated_df = pd.DataFrame({
    "participant": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
    "condition": ["baseline", "week4", "week8"] * 6,
    "score": [62, 65, 69, 58, 61, 64, 71, 73, 77, 66, 68, 72, 75, 78, 81, 69, 72, 76],
})
repeated = ResearchAssistant(repeated_df).run(
    objective="compare_groups",
    outcome="score",
    predictor="condition",
    estimand="mean",
    design="repeated",
    unit_id="participant",
    condition_order=("baseline", "week4", "week8"),
    variable_types={"score": "continuous"},
)
print(repeated.brief())
```

The mean path is one-way repeated-measures ANOVA with sphericity evaluation,
Greenhouse-Geisser correction when recorded, and Holm-adjusted paired follow-up. The explicit
distribution path uses Friedman with paired Wilcoxon-Holm follow-up. Both use complete panels and
never infer unit identity from row order.

## Two-factor independent ANOVA

Use this when independent observations are cross-classified by exactly two categorical factors.

```python
factorial_df = pd.DataFrame({
    "score": [12, 14, 15, 18, 19, 22, 14, 16, 17, 21, 23, 26],
    "program": ["standard"] * 6 + ["new"] * 6,
    "site": ["north"] * 3 + ["south"] * 3 + ["north"] * 3 + ["south"] * 3,
})
factorial = ResearchAssistant(factorial_df).two_way_anova(
    "score",
    "program",
    "site",
    sum_of_squares="type2",
)
print(factorial.brief())
```

Type II and Type III sums of squares are supported. Choose based on the planned model and design;
the software does not make that scientific decision automatically.

## Read the result

`brief()` reports the primary omnibus result. Use `explain()` or `show(workflow)` for
condition/group summaries, effect sizes, sphericity or model diagnostics, interactions, and
adjusted follow-up comparisons.

## Scientific caveat

An omnibus result does not identify a particular pair or simple effect. Independence, repeated
unit identity, factor roles, and contrast order come from the study design. Mixed ANOVA,
mixed-effects models, factorial repeated-measures combinations, and incomplete-panel longitudinal
models are not supported.

For complete contracts, see [the guided workflow](../../API_REFERENCE.md#integrated-guided-workflow)
and [Capabilities](../CAPABILITIES.md#supported-statistical-analyses).
