"""Phase 12 planning, paired analysis, presentation, and adapter example."""

from __future__ import annotations

import pandas as pd

from pyautostat import ResearchAssistant, StudyPlanner


def main() -> None:
    prospective = StudyPlanner().paired_mean_power(
        target_mean_difference=2,
        sd_difference=5,
        alpha=0.05,
        target_power=0.80,
    )
    print("Required complete pairs:", prospective.required_pairs)

    frame = pd.DataFrame(
        {
            "participant_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            "condition": ["before", "after"] * 5 + ["before"],
            "score": [10.0, 8.0, 9.0, 7.0, 11.0, 10.0, 8.0, 5.0, 13.0, 10.0, 20.0],
        }
    )
    assistant = ResearchAssistant(frame)
    assistant.declare_planning("planned", reason="Declared for this local example")
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="paired",
        estimand="mean",
        unit_id="participant_id",
        condition_order=("before", "after"),
        variable_types={"score": "continuous"},
    )
    plan = assistant.analysis_plan(draft, report_style="apa")
    workflow = assistant.run(draft=draft)
    completeness = assistant.reporting_completeness(workflow.report, style="apa")
    snapshot = assistant.session_snapshot(
        workflow,
        analysis_plan=plan,
        study_planning=prospective,
        reporting_completeness=completeness,
    )

    print("Plan:", plan.status.value, plan.primary_method_id)
    print("Workflow:", workflow.status.value, workflow.analysis.method_id)
    print("Complete pairs:", workflow.analysis.metadata["sample"]["complete_pairs"])
    print("Mean paired difference:", workflow.analysis.values["primary_estimate"])
    print("Completeness:", completeness.status)
    print("Snapshot schema:", snapshot.to_dict()["schema_version"])
    print("APA-oriented HTML characters:", len(workflow.report.to_html(style="apa")))
    print("Escaped LaTeX characters:", len(workflow.report.to_latex(style="apa")))


if __name__ == "__main__":
    main()
