"""Tests for integrated Cronbach's alpha sensitivity analysis."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pyautostat import (
    Comparability,
    ResearchAssistant,
    SensitivitySpecification,
    SensitivityStatus,
)


@pytest.fixture
def survey_data():
    return pd.DataFrame(
        {
            "q1": [1, 2, 3, 4, 5, 2, 4, 3, 5, 4, 2, 3],
            "q2": [1, 2, 3, 4, 4, 2, 5, 3, 4, 5, 1, 3],
            "q3": [2, 2, 3, 5, 5, 2, 4, 4, 5, 4, 2, 4],
            "q4": [5, 4, 3, 2, 1, 4, 2, 3, 1, 2, 4, 3],  # negatively keyed
        }
    )


def test_cronbach_alpha_integrated_sensitivity(survey_data):
    assistant = ResearchAssistant(survey_data)
    workflow = assistant.reliability(["q1", "q2", "q3"], bootstrap_samples=60)
    base = workflow.analysis
    assert base is not None
    assert base.method_id == "cronbach_alpha"

    scenario = SensitivitySpecification(
        "repeat reliability",
        base.specification,
        method_id="cronbach_alpha",
    )

    result = assistant.sensitivity_analysis(base, scenarios=[scenario])

    assert result.status == SensitivityStatus.COMPLETE
    item = result.scenario_results[0]
    assert item.status.value == "completed"
    assert item.comparability == Comparability.SAME_ESTIMAND
    assert item.estimate_quantity == "cronbach_alpha"
    assert item.primary_estimate == pytest.approx(base.values["cronbach_alpha"])
    assert item.p_value is None  # reliability is not a hypothesis test
    assert item.sample_size == len(survey_data)
    assert item.excluded_rows == 0

    comp = item.comparison
    assert comp is not None
    assert comp["available"] is True
    assert comp["comparison_quantity"] == "cronbach_alpha"
    assert comp["estimate_difference_from_base"] == pytest.approx(0.0)
    assert comp["interval_overlap"] is True

    # Narrative output
    text = result.compare()
    assert "Primary (Scale reliability (Cronbach's alpha)):" in text
    assert "Scenario (Scale reliability (Cronbach's alpha)):" in text
    assert "completed same-estimand scenario" in text
    assert "CONSISTENT" in result.verdict


def test_cronbach_alpha_same_scale_comparability(survey_data):
    assistant = ResearchAssistant(survey_data)
    workflow = assistant.reliability(["q1", "q2", "q3"], bootstrap_samples=40)
    base = workflow.analysis

    # Vary bootstrap sample count in specification options - same item set
    resample_spec = replace(
        base.specification,
        options=replace(base.specification.options, bootstrap_samples=80),
    )
    scenario = SensitivitySpecification("more resamples", resample_spec, method_id="cronbach_alpha")

    result = assistant.sensitivity_analysis(base, scenarios=[scenario])
    item = result.scenario_results[0]

    assert item.comparability == Comparability.SAME_ESTIMAND
    assert item.status.value == "completed"
    assert item.comparison["reverse_scoring_change"] is False


def test_cronbach_alpha_changed_items_different_estimand(survey_data):
    assistant = ResearchAssistant(survey_data)
    workflow = assistant.reliability(["q1", "q2", "q3", "q4"], bootstrap_samples=40)
    base = workflow.analysis

    # Item deletion scenario: drop q4
    deleted_item_spec = replace(
        base.specification,
        question=replace(base.specification.question, items=("q1", "q2", "q3")),
    )
    scenario = SensitivitySpecification("drop q4", deleted_item_spec, method_id="cronbach_alpha")

    result = assistant.sensitivity_analysis(base, scenarios=[scenario])
    item = result.scenario_results[0]

    # Changing item membership changes the scale definition -> DIFFERENT_ESTIMAND
    assert item.status.value == "completed"
    assert item.comparability == Comparability.DIFFERENT_ESTIMAND
    assert item.comparison is None
    assert any("different estimand" in w.lower() for w in item.warnings)
    assert "different estimand" in result.compare()
    assert "DESCRIPTIVE" in result.verdict


def test_cronbach_alpha_reverse_scoring_provenance(survey_data):
    assistant = ResearchAssistant(survey_data)
    workflow = assistant.reliability(["q1", "q2", "q3", "q4"], bootstrap_samples=40)
    base = workflow.analysis

    # Scenario with explicit declared reverse scoring for q4
    reversed_spec = replace(
        base.specification,
        options=replace(
            base.specification.options,
            reverse_scoring={"q4": (1.0, 5.0)},
        ),
    )
    scenario = SensitivitySpecification("q4 reversed", reversed_spec, method_id="cronbach_alpha")

    result = assistant.sensitivity_analysis(base, scenarios=[scenario])
    item = result.scenario_results[0]

    assert item.status.value == "completed"
    assert item.comparability == Comparability.SAME_ESTIMAND
    comp = item.comparison
    assert comp is not None
    assert comp["reverse_scoring_change"] is True
    assert comp["scoring_provenance"]["scenario_scoring"]["reverse_scoring_applied"] is True


def test_cronbach_alpha_sensitivity_serialization(survey_data):
    assistant = ResearchAssistant(survey_data)
    workflow = assistant.reliability(["q1", "q2", "q3"], bootstrap_samples=40)
    base = workflow.analysis

    scenario = SensitivitySpecification("repeat", base.specification, method_id="cronbach_alpha")
    result = assistant.sensitivity_analysis(base, scenarios=[scenario])

    payload = result.to_dict()
    assert payload["schema_version"] == 2
    scenario_dict = payload["scenario_results"][0]
    assert scenario_dict["estimate_quantity"] == "cronbach_alpha"
    assert scenario_dict["primary_estimate"] is not None
    assert scenario_dict["p_value"] is None

    # Deterministic roundtrip
    json_str = result.to_json()
    assert "DataFrame" not in json_str
    # Verify no raw respondent data is stored
    for q in ["q1", "q2", "q3"]:
        assert str(list(survey_data[q])) not in json_str
    parsed = json.loads(json_str)
    assert parsed == payload


def test_cronbach_alpha_deterministic_repeat_and_sample_accounting(survey_data):
    # Introduce some missing rows in a copy
    df_with_na = survey_data.copy()
    df_with_na.loc[0, "q1"] = np.nan
    df_with_na.loc[1, "q2"] = np.nan

    assistant = ResearchAssistant(df_with_na)
    workflow = assistant.reliability(["q1", "q2", "q3"], bootstrap_samples=40)
    base = workflow.analysis

    scenario = SensitivitySpecification("repeat", base.specification, method_id="cronbach_alpha")
    run1 = assistant.sensitivity_analysis(base, scenarios=[scenario])
    run2 = assistant.sensitivity_analysis(base, scenarios=[scenario])

    # Sample and exclusion accounting preserved
    assert run1.scenario_results[0].sample_size == 10
    assert run1.scenario_results[0].excluded_rows == 2

    # Deterministic repeat
    assert run1.to_dict() == run2.to_dict()
