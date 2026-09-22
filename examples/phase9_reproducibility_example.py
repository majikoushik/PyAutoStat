"""Observed decisions, a deliberate report mismatch, and explicit replay."""

from copy import deepcopy

import pandas as pd

from pyautostat import ResearchAssistant, ResearchReport, reproduce


def main() -> None:
    frame = pd.DataFrame(
        {
            "teaching_method": ["A"] * 8 + ["B"] * 8,
            "exam_score": [50.0, 52.0, 54.0, 55.0, 57.0, 58.0, 60.0, 61.0]
            + [55.0, 57.0, 59.0, 60.0, 62.0, 63.0, 65.0, 66.0],
        }
    )
    original = frame.copy(deep=True)
    assistant = ResearchAssistant(frame)
    assistant.enable_tracking()
    draft = assistant.prepare_question(
        objective="compare_groups", outcome="exam_score", predictor="teaching_method",
        estimand="mean", design="independent",
        variable_types={"exam_score": "continuous"},
    )
    recommendation = assistant.recommend_test(draft)
    result = assistant.analyze(draft)
    interpretation = assistant.interpret(result)
    report = assistant.report(result, interpretation=interpretation)
    audit = assistant.audit(report)
    record = assistant.reproducibility_record(result)
    replay = reproduce(record, data=frame)

    changed_report = deepcopy(report.to_dict())
    changed_report["sections"]["results"]["p_value"] = 0.9
    failed_audit = assistant.audit(ResearchReport(changed_report), result=result)
    changed_data = frame.copy(deep=True)
    changed_data.loc[0, "exam_score"] += 1
    mismatch = reproduce(record, data=changed_data)

    print("Selected method:", recommendation.method_id, result.method_id)
    print("Sample and order:", report.to_dict()["sections"]["dataset"])
    print("Report audit:", audit.status)
    print("Deliberate mutation:", failed_audit.status, failed_audit.findings[0].code)
    print("Same-data replay:", replay.status)
    print("Changed-data check:", mismatch.status, mismatch.data_status)
    print("Observed events:", [event["event_type"] for event in assistant.decision_ledger.events])
    pd.testing.assert_frame_equal(frame, original)


if __name__ == "__main__":
    main()
