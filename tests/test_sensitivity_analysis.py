"""Sensitivity semantics and result-driven selection safeguards."""

import json
import math
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest
from scipy import stats

from pyautostat import (
    AnalysisStatus,
    InvalidDataError,
    ResearchAssistant,
    SensitivityScenario,
    SensitivitySpecification,
)
from pyautostat.sensitivity import compare_same_estimand


@pytest.fixture
def sensitivity_case():
    frame = pd.DataFrame(
        {
            "group": ["Treatment"] * 8 + ["Control"] * 8,
            "score": [12.0, 15.0, 13.0, 18.0, 16.0, 14.0, 17.0, 19.0]
            + [8.0, 9.0, 11.0, 10.0, 12.0, 7.0, 13.0, 10.0],
        }
    )
    assistant = ResearchAssistant(frame)
    workflow = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        data_dictionary={"score": {"type": "continuous", "unit": "points"}},
    )
    assert workflow.analysis is not None
    return frame, assistant, workflow.analysis


def _student(base, name="pooled", *, planning_status="planned"):
    return SensitivitySpecification(
        name=name,
        specification=base.specification,
        method_id="student_t",
        rationale="Assess the explicit pooled-variance assumption.",
        planning_status=planning_status,
        assumptions=("Equal population variances",),
    )


def _mann_whitney(base, name="rank comparison"):
    question = replace(base.specification.question, estimand="distribution")
    return SensitivitySpecification(
        name=name,
        specification=replace(base.specification, question=question),
        method_id="mann_whitney_u",
        rationale="Supplementary comparison of rank distributions.",
    )


def test_explicit_scenarios_execute_once_in_declared_order_and_preserve_inputs(
    sensitivity_case, monkeypatch
):
    frame, assistant, base = sensitivity_case
    original_frame = frame.copy(deep=True)
    original_result = deepcopy(base.to_dict())
    scenarios = [_student(base, "second"), _mann_whitney(base, "first")]
    import pyautostat.research_assistant as module

    original = module.execute_selected_method
    calls = []

    def counted(analyzer, specification, method_id):
        calls.append(method_id)
        return original(analyzer, specification, method_id)

    monkeypatch.setattr(module, "execute_selected_method", counted)
    sensitivity = assistant.sensitivity_analysis(base, scenarios=scenarios)

    assert calls == ["student_t", "mann_whitney_u"]
    assert [item.name for item in sensitivity.scenario_results] == ["second", "first"]
    assert [item.comparability.value for item in sensitivity.scenario_results] == [
        "same_estimand",
        "different_estimand",
    ]
    assert sensitivity.base_result.to_dict() == original_result
    assert base.to_dict() == original_result
    pd.testing.assert_frame_equal(frame, original_frame)
    payload = sensitivity.to_dict()
    assert "robustness_score" not in json.dumps(payload)
    assert "smallest_p_value" not in json.dumps(payload)
    assert "average_estimate" not in json.dumps(payload)
    assert json.loads(sensitivity.to_json()) == payload

    repeated = assistant.sensitivity_analysis(base, scenarios=scenarios)
    assert repeated.to_dict() == payload
    assert calls == ["student_t", "mann_whitney_u", "student_t", "mann_whitney_u"]


def test_student_sensitivity_matches_scipy_and_compares_same_mean_difference(sensitivity_case):
    frame, assistant, base = sensitivity_case
    sensitivity = assistant.sensitivity_analysis(base, scenarios=[_student(base)])
    item = sensitivity.scenario_results[0]
    first = frame.loc[frame.group == "Treatment", "score"]
    second = frame.loc[frame.group == "Control", "score"]
    reference = stats.ttest_ind(first, second, equal_var=True)

    assert sensitivity.status.value == "complete"
    assert item.status.value == "completed"
    assert item.method_id == "student_t"
    assert item.comparability.value == "same_estimand"
    assert item.primary_estimate == pytest.approx(first.mean() - second.mean())
    assert item.analysis.values["test_statistic"] == pytest.approx(reference.statistic)
    assert item.analysis.values["p_value"] == pytest.approx(reference.pvalue)
    assert item.comparison["direction_consistent"] is True
    assert item.comparison["estimate_difference_from_base"] == pytest.approx(0.0)
    assert item.comparison["interval_overlap"] is True
    assert item.effect_size_quantity == "cohens_d"


def test_different_estimand_is_visible_without_direct_estimate_change(sensitivity_case):
    _, assistant, base = sensitivity_case
    result = assistant.sensitivity_analysis(base, scenarios=[_mann_whitney(base)])
    item = result.scenario_results[0]

    assert item.status.value == "completed"
    assert item.comparability.value == "different_estimand"
    assert item.estimate_quantity == "rank_biserial"
    assert item.comparison is None
    assert any("different estimand" in warning for warning in item.warnings)
    assert result.comparison_summary["different_estimand_scenarios"] == 1


def test_compare_reports_only_descriptive_same_estimand_decision_consistency(sensitivity_case):
    _, assistant, base = sensitivity_case
    result = assistant.sensitivity_analysis(
        base,
        scenarios=[_student(base), _mann_whitney(base)],
    )

    text = result.compare()

    assert "1 completed same-estimand scenario(s)" in text
    assert "different estimand" in text
    assert "does not by itself establish robustness" in text
    assert "appears robust" not in text
    text.encode("cp1252")


def test_sensitivity_verdict_exposes_methods_decisions_and_estimand_qualification(
    sensitivity_case,
):
    _, assistant, base = sensitivity_case
    result = assistant.sensitivity_analysis(
        base,
        scenarios=[_student(base), _mann_whitney(base)],
    )
    before = result.to_dict()
    text = result.compare()
    assert "Primary (Welch" in text
    assert "Scenario (Student" in text
    assert "Scenario (Mann-Whitney" in text
    assert "CONSISTENT across methods" in result.verdict
    assert "not all test the same estimand" in result.verdict
    assert "different estimand" in text
    assert result.to_dict() == before
    assert "verdict" not in before


def test_sensitivity_verdict_is_descriptive_when_no_completed_alternative_shares_estimand(
    sensitivity_case,
):
    _, assistant, base = sensitivity_case
    result = assistant.sensitivity_analysis(base, scenarios=[_mann_whitney(base)])
    assert result.verdict.startswith("DESCRIPTIVE")
    assert "cannot be directly compared" in result.verdict


def test_compare_uses_declared_alpha(sensitivity_case):
    _, assistant, base = sensitivity_case
    options = replace(base.specification.options, alpha=0.01)
    specification = replace(base.specification, options=options)
    adjusted_base = replace(base, specification=specification)
    scenario = _student(adjusted_base)

    result = assistant.sensitivity_analysis(adjusted_base, scenarios=[scenario])

    assert "alpha = 0.01" in result.compare()


def test_incompatible_unsupported_and_failed_scenarios_all_remain_visible(
    sensitivity_case, monkeypatch
):
    _, assistant, base = sensitivity_case
    ledger = assistant.enable_tracking(clock=lambda: "2026-01-01T00:00:00Z")
    paired = SensitivitySpecification(
        "paired",
        replace(base.specification, design="paired"),
        method_id="student_t",
        assumptions=("Equal population variances",),
    )
    unsupported = SensitivitySpecification(
        "coefficient only", base.specification, method_id="spearman_coefficient"
    )
    failed = _student(base, "numerical failure")
    import pyautostat.research_assistant as module

    original = module.execute_selected_method

    def selective_failure(analyzer, specification, method_id):
        if method_id == "student_t":
            raise ValueError("deliberate numerical failure")
        return original(analyzer, specification, method_id)

    monkeypatch.setattr(module, "execute_selected_method", selective_failure)
    result = assistant.sensitivity_analysis(base, scenarios=[paired, unsupported, failed])

    assert [item.name for item in result.scenario_results] == [
        "paired",
        "coefficient only",
        "numerical failure",
    ]
    assert [item.status.value for item in result.scenario_results] == [
        "incompatible",
        "unavailable",
        "failed",
    ]
    assert result.status.value == "unavailable"
    assert result.verdict.startswith("UNAVAILABLE")
    assert "deliberate numerical failure" in result.scenario_results[2].error
    assert [event["event_type"] for event in ledger.events] == [
        "sensitivity_plan_created",
        "sensitivity_scenario_attempted",
        "sensitivity_scenario_unavailable",
        "sensitivity_scenario_attempted",
        "sensitivity_scenario_unavailable",
        "sensitivity_scenario_attempted",
        "sensitivity_scenario_failed",
    ]
    assert ledger.events[0]["new_state"][0]["rationale"] is None


def test_student_requires_explicit_equal_variance_assumption(sensitivity_case):
    _, assistant, base = sensitivity_case
    scenario = SensitivitySpecification("pooled", base.specification, method_id="student_t")
    result = assistant.sensitivity_analysis(base, scenarios=[scenario])
    assert result.scenario_results[0].status.value == "incompatible"
    assert "equal-population-variance" in result.scenario_results[0].error


def test_unspecified_method_uses_declared_specification_recommendation(sensitivity_case):
    _, assistant, base = sensitivity_case
    scenario = SensitivityScenario("repeat Welch", base.specification)
    result = assistant.sensitivity_analysis(base, scenarios=[scenario])
    assert result.scenario_results[0].method_id == "welch_t"
    assert result.scenario_results[0].comparability.value == "same_estimand"


def test_reversed_group_contrast_is_normalized_and_documented(sensitivity_case):
    _, _, base = sensitivity_case
    metadata = deepcopy(base.metadata)
    metadata["contrast"] = {
        "definition": "first group minus second group",
        "first": "Control",
        "second": "Treatment",
    }
    values = deepcopy(base.values)
    values["primary_estimate"] *= -1
    interval = values["confidence_interval"]
    interval["lower"], interval["upper"] = -interval["upper"], -interval["lower"]
    reversed_result = replace(base, values=values, metadata=metadata)

    comparison = compare_same_estimand(base, reversed_result)
    assert comparison["contrast_orientation"] == "reversed"
    assert comparison["scenario_estimate_normalized"] == pytest.approx(
        base.values["primary_estimate"]
    )
    assert comparison["estimate_difference_from_base"] == pytest.approx(0.0)
    assert "signs were reversed" in comparison["orientation_transformation"]


def test_near_zero_base_omits_unstable_relative_change(sensitivity_case):
    _, _, base = sensitivity_case
    base_values = deepcopy(base.values)
    base_values["primary_estimate"] = 0.0
    scenario_values = deepcopy(base.values)
    scenario_values["primary_estimate"] = 0.1
    comparison = compare_same_estimand(
        replace(base, values=base_values), replace(base, values=scenario_values)
    )
    assert comparison["relative_estimate_change"] is None


def test_tracking_records_plan_attempts_and_every_outcome_without_preregistration_claim(
    sensitivity_case,
):
    _, assistant, base = sensitivity_case
    ledger = assistant.enable_tracking(clock=lambda: "2026-01-01T00:00:00Z")
    result = assistant.sensitivity_analysis(base, scenarios=[_student(base), _mann_whitney(base)])
    event_types = [event["event_type"] for event in ledger.events]
    assert event_types == [
        "sensitivity_plan_created",
        "sensitivity_scenario_attempted",
        "sensitivity_scenario_completed",
        "sensitivity_scenario_attempted",
        "sensitivity_scenario_completed",
    ]
    assert result.provenance["external_preregistration_verified"] is False
    assert "preregistered" not in ledger.to_json().lower()


def test_sensitivity_validates_api_and_base_dataset_identity(sensitivity_case):
    _, assistant, base = sensitivity_case
    with pytest.raises(InvalidDataError, match="non-empty"):
        assistant.sensitivity_analysis(base, scenarios=[])
    with pytest.raises(InvalidDataError, match="Every scenario"):
        assistant.sensitivity_analysis(base, scenarios=[object()])
    with pytest.raises(InvalidDataError, match="unique"):
        assistant.sensitivity_analysis(
            base, scenarios=[_student(base, "same"), _student(base, "same")]
        )
    changed_metadata = deepcopy(base.metadata)
    changed_metadata["sample"]["original_rows"] = 999
    with pytest.raises(InvalidDataError, match="row count"):
        assistant.sensitivity_analysis(
            replace(base, metadata=changed_metadata), scenarios=[_student(base)]
        )
    with pytest.raises(InvalidDataError, match="scenario name"):
        SensitivitySpecification(" ", base.specification)
    with pytest.raises(InvalidDataError, match="planning_status"):
        SensitivitySpecification("x", base.specification, planning_status="prespecified")


def test_scenario_serialization_contains_configuration_not_raw_dataframe(sensitivity_case):
    frame, _, base = sensitivity_case
    scenario = _student(base)
    text = json.dumps(scenario.to_dict(), allow_nan=False)
    assert "DataFrame" not in text
    assert str(frame.iloc[0].to_dict()) not in text
    assert scenario.to_dict()["assumptions"] == ["Equal population variances"]
    assert SensitivitySpecification.from_dict(scenario.to_dict()) == scenario
    invalid = scenario.to_dict()
    invalid["schema_version"] = 2
    with pytest.raises(InvalidDataError, match="schema_version"):
        SensitivitySpecification.from_dict(invalid)


def test_unavailable_base_and_nonfinite_scenario_output_are_not_successes(sensitivity_case):
    _, assistant, base = sensitivity_case
    unavailable = replace(base, status=AnalysisStatus.UNAVAILABLE)
    with pytest.raises(InvalidDataError, match="available base"):
        assistant.sensitivity_analysis(unavailable, scenarios=[_student(base)])

    import pyautostat.research_assistant as module

    original = module.execute_selected_method

    def nonfinite_output(analyzer, specification, method_id):
        analysis = original(analyzer, specification, method_id)
        values = deepcopy(analysis.values)
        values["primary_estimate"] = math.nan
        values["p_value"] = math.inf
        return replace(analysis, values=values)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(module, "execute_selected_method", nonfinite_output)
        result = assistant.sensitivity_analysis(base, scenarios=[_student(base)])

    scenario = result.scenario_results[0]
    assert result.status.value == "unavailable"
    assert scenario.status.value == "failed"
    assert scenario.comparability.value == "unavailable"
    assert scenario.primary_estimate is None
    assert scenario.p_value is None
    assert "no finite primary estimate or p-value" in scenario.error
    assert json.loads(result.to_json()) == result.to_dict()
