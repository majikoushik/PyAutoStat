"""Explicit sensitivity and meaningful-effect threshold example."""

from dataclasses import replace

import pandas as pd

from pyautostat import (
    MeaningfulEffectThreshold,
    ResearchAssistant,
    SensitivitySpecification,
)


def main() -> None:
    frame = pd.DataFrame(
        {
            "teaching_method": ["new"] * 8 + ["standard"] * 8,
            "exam_score": [78, 82, 75, 86, 80, 84, 79, 83, 70, 74, 72, 77, 69, 76, 73, 71],
        }
    )
    assistant = ResearchAssistant(frame)
    assistant.enable_tracking()
    workflow = assistant.run(
        objective="compare_groups",
        outcome="exam_score",
        predictor="teaching_method",
        estimand="mean",
        design="independent",
        data_dictionary={"exam_score": {"type": "continuous", "unit": "points"}},
    )
    if workflow.analysis is None:
        raise RuntimeError(f"Base analysis unavailable: {workflow.blockers}")

    pooled = SensitivitySpecification(
        name="pooled variance",
        specification=workflow.analysis.specification,
        method_id="student_t",
        rationale="Assess sensitivity to the explicitly declared pooled-variance assumption.",
        planning_status="planned",
        assumptions=("Equal population variances",),
    )
    distribution_spec = replace(
        workflow.analysis.specification,
        question=replace(workflow.analysis.specification.question, estimand="distribution"),
    )
    ranks = SensitivitySpecification(
        name="rank distribution",
        specification=distribution_spec,
        method_id="mann_whitney_u",
        rationale="Supplementary rank-distribution question; not the mean-difference estimand.",
        planning_status="exploratory",
    )
    sensitivity = assistant.sensitivity_analysis(workflow.analysis, scenarios=[pooled, ranks])

    threshold = MeaningfulEffectThreshold(
        quantity="mean_difference",
        minimum_magnitude=5,
        direction="two_sided",
        unit="points",
        rationale="A difference below 5 points would not change the teaching decision.",
        planning_status="planned",
    )
    practical = assistant.practical_significance(workflow.analysis, threshold=threshold)
    report = assistant.report(
        workflow.analysis,
        sensitivity=sensitivity,
        practical_significance=practical,
    )
    audit = assistant.audit(report)

    print("Primary method:", workflow.analysis.method_id)
    for scenario in sensitivity.scenario_results:
        print(
            scenario.name,
            scenario.status.value,
            scenario.comparability.value,
            scenario.primary_estimate,
        )
    print("Point estimate relation:", practical.point_estimate_relation)
    print("Interval relation:", practical.confidence_interval_relation)
    print("Report schema:", report.to_dict()["schema_version"])
    print("Audit:", audit.status)


if __name__ == "__main__":
    main()
