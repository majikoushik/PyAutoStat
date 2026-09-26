"""Profile, guided workflow, and clarification continuation."""

import pandas as pd

from pyautostat import ResearchAssistant


def example_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exam_score": [70.0, 72, 68, 71, 69, 74, 82, 85, 80, 84, 83, 81],
            "teaching_method": ["A"] * 6 + ["B"] * 6,
        }
    )


def main() -> None:
    frame = example_data()
    assistant = ResearchAssistant(frame)

    profile = assistant.profile()
    print("Profile rows:", profile["overview"]["shape"][0])
    print(assistant.summarize())

    incomplete = assistant.run(
        objective="compare_groups",
        outcome="exam_score",
        predictor="teaching_method",
        estimand="mean",
        variable_types={"exam_score": "continuous"},
    )
    print("Initial status:", incomplete.status.value)
    print("Needed fields:", [item.field for item in incomplete.missing_information])
    print(incomplete.explain())

    revised = assistant.update_question(incomplete.draft, design="independent")
    workflow = assistant.run(draft=revised)
    print("Final status:", workflow.status.value)
    print("Method ID:", workflow.analysis.method_id)
    print("Method:", workflow.analysis.method_label)
    print(workflow.explain())
    print(workflow.interpretation.findings_plain)
    print("Mean difference:", workflow.analysis.values["primary_estimate"])
    print("Analyzed rows:", workflow.analysis.sample_size)
    print("Audit:", workflow.audit.status)
    print("Reproducibility method:", workflow.reproducibility.method_id)
    print("HTML characters:", len(workflow.report.to_html()))


if __name__ == "__main__":
    main()
