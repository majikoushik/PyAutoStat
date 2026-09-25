"""Executable checks for the README's beginner-facing workflow."""

import pandas as pd

from pyautostat import ResearchAssistant


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "teaching_method": ["standard"] * 6 + ["new"] * 6,
            "exam_score": [62, 65, 68, 70, 72, 75, 67, 71, 74, 78, 80, 84],
        }
    )


def test_readme_profile_run_and_needs_input_paths():
    assistant = ResearchAssistant(_scores())

    profile = assistant.profile()
    assert profile["overview"]["total_rows"] == 12
    assert profile["resource_info"]["sampling_applied"] is False

    workflow = assistant.run(
        objective="compare_groups",
        outcome="exam_score",
        predictor="teaching_method",
        estimand="mean",
        design="independent",
        variable_types={"exam_score": "continuous"},
    )
    assert workflow.status == "completed"
    assert workflow.analysis is not None
    assert workflow.analysis.method_id == "welch_t"

    pending = assistant.run(
        objective="compare_groups",
        outcome="exam_score",
        predictor="teaching_method",
        estimand="mean",
        variable_types={"exam_score": "continuous"},
    )
    assert pending.status == "needs_input"
    assert pending.analysis is None
    assert pending.draft is not None

    revised = assistant.update_question(pending.draft, design="independent")
    continued = assistant.run(draft=revised)
    assert continued.status == "completed"
