"""Five bounded binary-outcome and extended-association workflows."""

import numpy as np
import pandas as pd

from pyautostat import ResearchAssistant

rng = np.random.default_rng(17)
n = 120
age = rng.normal(40, 9, n)
spend = rng.normal(70, 15, n)
probability = 1 / (1 + np.exp(-(-1.5 + 0.035 * age + 0.012 * spend)))
churned = np.where(rng.binomial(1, probability), "yes", "no")
customers = pd.DataFrame({"churned": churned, "age": age, "spend": spend})

logistic = ResearchAssistant(customers).run(
    objective="regression",
    outcome="churned",
    predictors=["age", "spend"],
    estimand="event_probability",
    design="independent",
    event_level="yes",
    variable_types={"churned": "nominal", "age": "continuous", "spend": "continuous"},
)
print(logistic.explain())

pairs = pd.DataFrame(
    [
        {"person": person, "condition": condition, "response": response}
        for person, (after, before) in enumerate(
            [("yes", "no")] * 8 + [("no", "yes")] * 3 + [("yes", "yes")] * 5
        )
        for condition, response in (("after", after), ("before", before))
    ]
)
mcnemar = ResearchAssistant(pairs).run(
    objective="compare_groups",
    outcome="response",
    predictor="condition",
    estimand="proportion",
    design="paired",
    unit_id="person",
    condition_order=("after", "before"),
    event_level="yes",
    variable_types={"response": "nominal", "condition": "nominal"},
)
print(mcnemar.explain())

association = pd.DataFrame(
    {
        "score": rng.normal(size=n),
        "exposed": np.where(np.arange(n) % 2, "yes", "no"),
        "rank_a": rng.integers(1, 6, n),
        "rank_b": rng.integers(1, 6, n),
        "age": age,
    }
)
association["adjusted_score"] = 0.5 * association["score"] + 0.03 * age + rng.normal(size=n)

point_biserial = ResearchAssistant(association).run(
    objective="association",
    outcome="score",
    predictor="exposed",
    estimand="point_biserial",
    design="independent",
    event_level="yes",
    variable_types={"score": "continuous", "exposed": "nominal"},
)
kendall = ResearchAssistant(association).run(
    objective="association",
    outcome="rank_a",
    predictor="rank_b",
    estimand="monotonic",
    association_measure="kendall",
    design="independent",
    variable_types={"rank_a": "discrete", "rank_b": "discrete"},
)
partial = ResearchAssistant(association).run(
    objective="association",
    outcome="score",
    predictor="adjusted_score",
    controls=["age"],
    estimand="partial_linear",
    design="independent",
    variable_types={"score": "continuous", "adjusted_score": "continuous", "age": "continuous"},
)
print(point_biserial.explain())
print(kendall.explain())
print(partial.explain())
