# Regression

## Use this when

Use regression when the question concerns a conditional mean (ordinary least squares) or the
probability/odds of an explicitly chosen binary event. Minimum inputs are the outcome, ordered
predictor set, variable meanings, independent-design declaration, and any categorical reference
levels. Logistic regression also requires the modeled event level.

## Ordinary least squares

```python
import numpy as np
import pandas as pd
from pyautostat import ResearchAssistant

x = np.arange(1.0, 21.0)
z = np.array([2, 1, 4, 2, 5, 3, 7, 6, 8, 5, 9, 7, 11, 8, 13, 9, 14, 12, 15, 13], dtype=float)
error = np.array([0.2, -0.4, 0.1, 0.3, -0.2, 0.5, -0.3, 0.1, 0.4, -0.5] * 2)
ols_df = pd.DataFrame({"outcome": 4 + 1.7 * x - 0.6 * z + error, "x": x, "z": z})

ols = ResearchAssistant(ols_df).run(
    objective="regression",
    outcome="outcome",
    predictors=["x", "z"],
    estimand="conditional_mean",
    design="independent",
    covariance_type="HC3",
    variable_types={"outcome": "continuous", "x": "continuous", "z": "continuous"},
)
print(ols.brief())
```

`covariance_type="classical"` uses classical covariance inference; `"HC3"` changes
coefficient uncertainty, not the fitted OLS coefficients. Categorical predictors use explicit
treatment coding. Supply `reference_levels={"group": "standard"}` when the baseline matters.

## Binary logistic regression

```python
rng = np.random.default_rng(42)
age = np.linspace(20, 70, 80)
program = np.where(np.arange(80) % 2 == 0, "standard", "new")
linear = -3.0 + 0.055 * age + 0.45 * (program == "new")
probability = 1 / (1 + np.exp(-linear))
event = np.where(rng.binomial(1, probability), "yes", "no")
logit_df = pd.DataFrame({"event": event, "age": age, "program": program})

logistic = ResearchAssistant(logit_df).run(
    objective="regression",
    outcome="event",
    predictors=["age", "program"],
    estimand="event_probability",
    design="independent",
    event_level="yes",
    reference_levels={"program": "standard"},
    variable_types={"event": "nominal", "age": "continuous", "program": "nominal"},
)
print(logistic.brief())
```

The event level defines the modeled probability and coefficient/odds-ratio orientation. Invalid
events and references show observed choices and are never silently replaced.

## Read the result

Model-fit statistics answer a model-level question; coefficient estimates answer conditional,
term-specific questions. Use `brief()` for the shortest primary view and `explain()` or
`show(workflow)` for coding, coefficients, intervals, diagnostics, fit, warnings, and exclusions.
For logistic models, coefficient exponentiation yields odds ratios with Wald intervals.

## Scientific caveat

PyAutoStat performs no automatic predictor selection, interaction construction, polynomial
construction, imputation, or outlier deletion. An odds ratio is not a risk ratio. In-sample fit is
not out-of-sample validation, and regression alone does not establish causality.

For authoritative signatures and fields, see [OLS regression workflow](../../API_REFERENCE.md#ols-regression-workflow)
and [binary logistic regression](../../API_REFERENCE.md#binary-logistic-regression-and-extended-association).
