"""Four complete basic-inference workflows over small synthetic datasets."""

import pandas as pd

from pyautostat import ResearchAssistant


def show(title, workflow):
    print(f"\n{title}")
    print("-" * len(title))
    print(workflow.explain())


one_sample = ResearchAssistant(
    pd.DataFrame({"score": [48.0, 51.0, 54.0, 56.0, 52.0, 55.0]})
).run(
    objective="compare_reference",
    outcome="score",
    reference_value=50,
    estimand="mean",
    design="independent",
    variable_types={"score": "continuous"},
)
show("ONE-SAMPLE MEAN INFERENCE", one_sample)

paired = pd.DataFrame(
    {
        "student_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "condition": ["after", "before"] * 5,
        "score": [72, 66, 80, 73, 69, 68, 85, 75, 77, 74],
    }
)
wilcoxon = ResearchAssistant(paired).run(
    objective="compare_groups",
    outcome="score",
    predictor="condition",
    estimand="distribution",
    design="paired",
    unit_id="student_id",
    condition_order=("after", "before"),
    variable_types={"score": "continuous", "condition": "nominal"},
)
show("PAIRED SIGNED-RANK INFERENCE", wilcoxon)

monotonic = ResearchAssistant(
    pd.DataFrame(
        {
            "study_hours": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "score": [48.0, 55.0, 53.0, 67.0, 65.0, 78.0, 82.0],
        }
    )
).run(
    objective="association",
    outcome="study_hours",
    predictor="score",
    estimand="monotonic",
    design="independent",
    variable_types={"study_hours": "continuous", "score": "continuous"},
)
show("SPEARMAN MONOTONIC INFERENCE", monotonic)

sparse = ResearchAssistant(
    pd.DataFrame(
        {
            "treatment": ["new"] * 5 + ["standard"] * 7,
            "response": ["yes"] + ["no"] * 4 + ["yes"] * 5 + ["no"] * 2,
        }
    )
).run(
    objective="association",
    outcome="response",
    predictor="treatment",
    estimand="categorical_independence",
    design="independent",
    variable_types={"response": "nominal", "treatment": "nominal"},
)
show("FISHER EXACT 2x2 INFERENCE", sparse)
