# Associations

## Use this when

Use association workflows when the scientific question concerns how two variables vary together,
not a causal effect. Minimum inputs are the two variables, their scientific measurement meanings,
the independence/design declaration, and an explicit association target.

## Pearson, Spearman, and Kendall: beginner path

```python
import pandas as pd
from pyautostat import ResearchAssistant

df = pd.DataFrame({
    "hours": [1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
    "score": [52, 55, 57, 61, 63, 68, 72, 74, 79, 83],
})
assistant = ResearchAssistant(df)

types = {"hours": "continuous", "score": "continuous"}
pearson = assistant.correlate("hours", "score", method="pearson", variable_types=types)
spearman = assistant.correlate("hours", "score", method="spearman", variable_types=types)
kendall = assistant.correlate("hours", "score", method="kendall", variable_types=types)

print(pearson.brief())
print(spearman.brief())
print(kendall.brief())
```

- Pearson estimates linear association.
- Spearman estimates monotonic rank association.
- Kendall reports tau-b and accounts for ties through that definition.

The wrapper requires `method=`; it never auto-recommends Pearson versus Spearman versus Kendall.

## Advanced association paths

Use `run()` for targets that require additional roles or orientation:

```python
advanced_df = pd.DataFrame({
    "score": [42, 48, 51, 55, 61, 65, 70, 74, 77, 82],
    "group": ["control"] * 5 + ["treated"] * 5,
    "hours": [1, 2, 3, 4, 5, 3, 4, 5, 6, 7],
    "baseline": [40, 46, 50, 53, 58, 60, 63, 67, 71, 75],
    "exposure": ["low", "low", "high", "low", "high"] * 2,
    "response": ["no", "no", "yes", "no", "yes", "yes", "yes", "yes", "no", "no"],
})
advanced = ResearchAssistant(advanced_df)

point_biserial = advanced.run(
    objective="association",
    outcome="score",
    predictor="group",
    estimand="point_biserial",
    design="independent",
    event_level="treated",
    variable_types={"score": "continuous", "group": "nominal"},
)
partial = advanced.run(
    objective="association",
    outcome="score",
    predictor="hours",
    controls=["baseline"],
    estimand="partial_linear",
    design="independent",
    variable_types={"score": "continuous", "hours": "continuous", "baseline": "continuous"},
)
categorical = advanced.run(
    objective="association",
    outcome="response",
    predictor="exposure",
    estimand="categorical_independence",
    design="independent",
    variable_types={"response": "nominal", "exposure": "nominal"},
)

print(point_biserial.brief())
print(partial.brief())
print(categorical.brief())
```

Point-biserial orientation follows the explicit positive/event level. Partial Pearson conditions
on declared quantitative controls but does not establish causal deconfounding. Categorical
association uses Pearson chi-square when the expected-count policy passes and Fisher's exact test
only for a sparse 2x2 table. Fisher provides an exact two-sided p-value; when all cells are
positive, its sample odds-ratio interval is an asymptotic log-Wald interval, not an exact interval.

Repeated-measures correlation is not implemented.

## Read the result

Start with `brief()`, then inspect `explain()` or `show(workflow)` for orientation, paired
sample count, uncertainty method, assumptions, and warnings. Use `apa_statement()` only as a
reporting-oriented sentence.

## Scientific caveat

Association does not establish causation. An odds ratio is not a risk ratio. Select the target from
the research question rather than whichever method gives a preferred p-value.

For signatures and contracts, see [the guided workflow](../../API_REFERENCE.md#integrated-guided-workflow)
and [Statistical Method Contracts](../STATISTICAL_METHOD_CONTRACTS.md).
