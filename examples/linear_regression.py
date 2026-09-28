"""Complete OLS workflow with continuous and categorical predictors."""

import pandas as pd

from pyautostat import ResearchAssistant


data = pd.DataFrame(
    {
        "score": [58, 61, 65, 67, 70, 72, 75, 77, 80, 83, 85, 88],
        "study_hours": [2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11],
        "program": ["standard", "intensive", "standard"] * 4,
        "remote": [False, True] * 6,
    }
)

simple = ResearchAssistant(data).run(
    objective="regression",
    outcome="score",
    predictors=["study_hours"],
    estimand="conditional_mean",
    design="independent",
    variable_types={"score": "continuous", "study_hours": "continuous"},
)

workflow = ResearchAssistant(data).run(
    objective="regression",
    outcome="score",
    predictors=["study_hours", "program", "remote"],
    estimand="conditional_mean",
    design="independent",
    variable_types={
        "score": "continuous",
        "study_hours": "continuous",
        "program": "nominal",
        "remote": "boolean",
    },
    reference_levels={"program": "standard", "remote": False},
    covariance_type="HC3",
)

assert simple.analysis is not None
assert workflow.analysis is not None
assert workflow.report is not None
assert workflow.audit is not None
print("SIMPLE REGRESSION")
print(simple.explain())
print("\nMULTIPLE REGRESSION WITH HC3")
print(workflow.explain())
print("\nCoefficient terms:")
for coefficient in workflow.analysis.values["coefficients"]:
    print(
        coefficient["term_label"],
        coefficient["estimate"],
        coefficient["confidence_interval"],
    )
print("\nCSV tables:", sorted(workflow.report.to_csv_tables()))
print("Audit:", workflow.audit.status)
