"""In-memory research report example; no files are written."""

import pandas as pd

from pyautostat import ResearchAssistant


def main() -> None:
    frame = pd.DataFrame(
        {
            "teaching_method": ["A"] * 8 + ["B"] * 8,
            "exam_score": [61, 63, 65, 66, 68, 70, 72, 74, 66, 68, 70, 72, 74, 76, 78, 80],
        }
    )
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="exam_score",
        predictor="teaching_method",
        estimand="mean",
        design="independent",
        variable_types={"exam_score": "continuous"},
    )
    result = assistant.analyze(draft)
    interpretation = assistant.interpret(result)
    report = assistant.report(result, interpretation=interpretation)
    print(report.status, result.method_label)
    print(interpretation.findings_plain)
    print("HTML characters:", len(report.to_html()))
    print("Markdown characters:", len(report.to_markdown()))
    print("JSON characters:", len(report.to_json()))
    print("CSV tables:", list(report.to_csv_tables()))

    # Pearson currently has no guided confidence interval, so this report is partial.
    association = ResearchAssistant(
        pd.DataFrame(
            {
                "study_hours": [1.0, 2.0, 3.0, 4.0, 5.0],
                "exam_score": [55.0, 62.0, 60.0, 75.0, 79.0],
            }
        )
    )
    association_draft = association.prepare_question(
        objective="association",
        outcome="study_hours",
        predictor="exam_score",
        estimand="linear",
        design="independent",
        variable_types={"study_hours": "continuous", "exam_score": "continuous"},
    )
    partial = association.report(association.analyze(association_draft))
    print("Pearson report:", partial.status, partial.to_dict()["limitations"])


if __name__ == "__main__":
    main()
