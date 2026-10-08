# Reliability and agreement

Scale internal consistency and rater/measurement agreement are distinct research tasks. Neither
establishes validity.

## Cronbach's alpha

Use this for a researcher-declared set of scored items intended to measure a common reflective
construct. Minimum inputs are at least two numeric item columns and a defensible item set.

```python
import pandas as pd
from pyautostat import ResearchAssistant

scale_df = pd.DataFrame({
    "q1": [1, 2, 3, 4, 5, 2, 4, 3, 5, 1, 4, 2],
    "q2": [1, 2, 3, 4, 4, 2, 5, 3, 5, 1, 3, 2],
    "q3": [2, 2, 3, 5, 5, 2, 4, 4, 5, 1, 4, 1],
})
alpha = ResearchAssistant(scale_df).reliability(
    ["q1", "q2", "q3"],
    bootstrap_samples=199,
    random_state=7,
)
print(alpha.brief())
```

Reverse scoring is explicit: pass, for example,
`reverse_scoring={"q3": (1, 5)}` only when the item key and scale bounds justify it. The result
includes item-total correlations, alpha-if-deleted diagnostics, and the inter-item matrix.
PyAutoStat reports ordinary covariance-based alpha; standardized alpha is unavailable.

## Intraclass correlation coefficient

Use ICC for quantitative measurements in a fully crossed target-by-rater panel. Minimum inputs are
target ID, rater ID, outcome, and the researcher-chosen model, definition, and unit.

```python
icc_df = pd.DataFrame({
    "target": ["T1", "T1", "T2", "T2", "T3", "T3", "T4", "T4"],
    "rater": ["R1", "R2", "R1", "R2", "R1", "R2", "R1", "R2"],
    "score": [9.0, 2.0, 6.0, 1.0, 8.0, 4.0, 7.0, 2.0],
})
icc = ResearchAssistant(icc_df).intraclass_correlation(
    target="target",
    rater="rater",
    value="score",
    model="two_way_random",
    definition="absolute_agreement",
    unit="single",
)
print(icc.brief())
```

Supported choices map to the six canonical forms ICC(1,1), ICC(1,k), ICC(2,1), ICC(2,k),
ICC(3,1), and ICC(3,k):

- model: `one_way_random`, `two_way_random`, or `two_way_mixed`;
- definition: `absolute_agreement` or `consistency` where compatible;
- unit: `single` or `average`.

Absolute agreement penalizes systematic rater offsets; consistency can retain relative ordering
despite them. A single-measure ICC targets one rating; an average-measure ICC targets the mean of
the panel's ratings. Negative estimates are not clamped.

## Read the result

Use `brief()` for the shortest estimate and interval. Use `explain()` or `show(workflow)` for
item diagnostics, ICC definition, panel accounting, mean squares, variance components, and
warnings.

## Scientific caveat

Reliability is not validity. Alpha does not establish unidimensionality or a universal adequacy
threshold. ICC selection depends on how raters and measurement units generalize; the software
cannot infer that intent from values. Missing or unbalanced rater panels are not imputed.

For details, see [scale reliability](../../API_REFERENCE.md#scale-reliability-workflow),
[ICC workflow](../../API_REFERENCE.md#intraclass-correlation-coefficient-icc-workflow), and the
[ICC guide](../ICC_GUIDE.md).
