"""Phase 11 practical thresholds, reporting, audit, and reproducibility."""

import json
import math
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from pyautostat import (
    AnalysisStatus,
    InvalidDataError,
    MeaningfulEffectThreshold,
    ResearchAssistant,
    ResearchReport,
    SensitivitySpecification,
    StatisticalAnalyzer,
)


@pytest.fixture
def mean_case():
    frame = pd.DataFrame(
        {
            "group": ["Treatment"] * 10 + ["Control"] * 10,
            "score": [20.0, 21.0, 19.0, 23.0, 22.0, 24.0, 18.0, 25.0, 21.0, 22.0]
            + [14.0, 15.0, 16.0, 13.0, 17.0, 14.0, 16.0, 15.0, 13.0, 17.0],
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


def _mean_result(base, estimate, lower=None, upper=None, *, p_value=None):
    values = deepcopy(base.values)
    values["primary_estimate"] = float(estimate)
    if lower is None:
        values["confidence_interval"] = None
    else:
        values["confidence_interval"] = {
            "quantity": "mean difference",
            "lower": float(lower),
            "upper": float(upper),
            "level": 0.95,
            "method": "analytical t interval",
        }
    if p_value is not None:
        values["p_value"] = float(p_value)
    return replace(base, values=values)


@pytest.mark.parametrize(
    ("estimate", "lower", "upper", "point", "interval"),
    [
        (8, 6, 10, "positive_meaningful_region", "entirely_positive_meaningful"),
        (-8, -10, -6, "negative_meaningful_region", "entirely_negative_meaningful"),
        (1, -2, 3, "below_meaningful_magnitude", "entirely_within_negligible_region"),
        (8, 2, 12, "positive_meaningful_region", "crosses_meaningful_boundary"),
        (0, -7, 7, "below_meaningful_magnitude", "spans_both_directions"),
    ],
)
def test_two_sided_point_and_interval_relations(mean_case, estimate, lower, upper, point, interval):
    _, assistant, base = mean_case
    threshold = MeaningfulEffectThreshold(
        "mean_difference", 5, unit="points", rationale="Decision threshold."
    )
    result = assistant.practical_significance(
        _mean_result(base, estimate, lower, upper), threshold=threshold
    )
    assert result.status == "complete"
    assert result.point_estimate_relation == point
    assert result.confidence_interval_relation == interval
    assert result.threshold.rationale == "Decision threshold."
    assert "formal equivalence" in result.conclusion
    assert "small" not in result.to_json().lower()


@pytest.mark.parametrize(
    ("direction", "estimate", "lower", "upper", "point", "interval"),
    [
        ("positive", 6, 5, 8, "meets_positive_threshold", "entirely_above_positive_threshold"),
        ("positive", 2, 1, 4, "below_positive_threshold", "entirely_below_positive_threshold"),
        ("positive", 6, 2, 8, "meets_positive_threshold", "crosses_positive_threshold"),
        ("negative", -6, -8, -5, "meets_negative_threshold", "entirely_below_negative_threshold"),
        ("negative", -2, -4, 1, "above_negative_threshold", "entirely_above_negative_threshold"),
        ("negative", -6, -8, -2, "meets_negative_threshold", "crosses_negative_threshold"),
    ],
)
def test_directional_threshold_relations(
    mean_case, direction, estimate, lower, upper, point, interval
):
    _, assistant, base = mean_case
    threshold = MeaningfulEffectThreshold("mean_difference", 5, direction=direction)
    result = assistant.practical_significance(
        _mean_result(base, estimate, lower, upper), threshold=threshold
    )
    assert result.point_estimate_relation == point
    assert result.confidence_interval_relation == interval


def test_missing_interval_returns_partial_without_manufacturing_uncertainty(mean_case):
    _, assistant, base = mean_case
    result = assistant.practical_significance(
        _mean_result(base, 7, None),
        threshold=MeaningfulEffectThreshold("mean_difference", 5),
    )
    assert result.status == "partial"
    assert result.point_estimate_relation == "positive_meaningful_region"
    assert result.confidence_interval is None
    assert result.confidence_interval_relation == "unavailable"
    assert result.uncertainty_status == "confidence_interval_unavailable"


def test_pearson_threshold_is_partial_because_current_backend_has_no_interval():
    frame = pd.DataFrame({"hours": [1, 2, 3, 4, 5, 6], "score": [2, 3, 4, 8, 7, 10]})
    assistant = ResearchAssistant(frame)
    workflow = assistant.run(
        objective="association",
        outcome="hours",
        predictor="score",
        estimand="linear",
        design="independent",
        variable_types={"hours": "continuous", "score": "continuous"},
    )
    result = assistant.practical_significance(
        workflow.analysis,
        threshold=MeaningfulEffectThreshold("pearson_r", 0.3),
    )
    assert result.status == "partial"
    assert result.quantity == "pearson_r"
    assert result.confidence_interval_relation == "unavailable"


def test_cohens_d_threshold_selects_standardized_effect_not_raw_mean(mean_case):
    _, assistant, base = mean_case
    result = assistant.practical_significance(
        base, threshold=MeaningfulEffectThreshold("cohens_d", 0.5)
    )
    assert result.estimate == pytest.approx(base.values["effect_size"]["value"])
    assert result.confidence_interval["quantity"] == "Cohen's d"
    assert result.quantity == "cohens_d"


def test_nonnegative_cramers_v_threshold_and_interval_relation():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            "outcome": ["yes"] * 15 + ["no"] * 5 + ["yes"] * 5 + ["no"] * 15,
        }
    )
    assistant = ResearchAssistant(frame)
    workflow = assistant.run(
        objective="association",
        outcome="outcome",
        predictor="group",
        estimand="categorical_independence",
        design="independent",
        variable_types={"outcome": "nominal", "group": "nominal"},
    )
    result = assistant.practical_significance(
        workflow.analysis,
        threshold=MeaningfulEffectThreshold("cramers_v", 0.2, direction="nonnegative"),
    )
    assert result.status == "complete"
    assert result.point_estimate_relation == "meets_nonnegative_threshold"
    assert result.confidence_interval_relation in {
        "entirely_at_or_above_nonnegative_threshold",
        "crosses_nonnegative_threshold",
    }


def test_threshold_quantity_units_and_interval_metadata_are_validated(mean_case):
    _, assistant, base = mean_case
    no_unit = assistant.practical_significance(
        base, threshold=MeaningfulEffectThreshold("mean_difference", 5)
    )
    assert no_unit.threshold.unit is None
    assert "unit" not in " ".join(no_unit.warnings).lower()
    with pytest.raises(InvalidDataError, match="available result quantity"):
        assistant.practical_significance(
            base, threshold=MeaningfulEffectThreshold("pearson_r", 0.2)
        )
    with pytest.raises(InvalidDataError, match="does not match result unit"):
        assistant.practical_significance(
            base,
            threshold=MeaningfulEffectThreshold("mean_difference", 5, unit="percent"),
        )
    values = deepcopy(base.values)
    values["confidence_interval"]["quantity"] = "Cohen's d"
    with pytest.raises(InvalidDataError, match="interval quantity"):
        assistant.practical_significance(
            replace(base, values=values),
            threshold=MeaningfulEffectThreshold("mean_difference", 5),
        )


@pytest.mark.parametrize("value", [-1, math.nan, math.inf, -math.inf, True])
def test_invalid_threshold_magnitudes_are_rejected(value):
    with pytest.raises(InvalidDataError, match="finite nonnegative"):
        MeaningfulEffectThreshold("mean_difference", value)


def test_threshold_model_rejects_bad_quantity_direction_and_text():
    with pytest.raises(InvalidDataError, match="quantity"):
        MeaningfulEffectThreshold("odds_ratio", 1)
    with pytest.raises(InvalidDataError, match="nonnegative"):
        MeaningfulEffectThreshold("cramers_v", 0.2)
    with pytest.raises(InvalidDataError, match="cannot use"):
        MeaningfulEffectThreshold("mean_difference", 5, direction="nonnegative")
    with pytest.raises(InvalidDataError, match="direction"):
        MeaningfulEffectThreshold("mean_difference", 5, direction="favorable")
    with pytest.raises(InvalidDataError, match="rationale"):
        MeaningfulEffectThreshold("mean_difference", 5, rationale=" ")
    with pytest.raises(InvalidDataError, match="must not declare raw units"):
        MeaningfulEffectThreshold("cohens_d", 0.5, unit="points")
    threshold = MeaningfulEffectThreshold("mean_difference", 5, unit="points")
    assert MeaningfulEffectThreshold.from_dict(threshold.to_dict()) == threshold
    with pytest.raises(InvalidDataError, match="schema_version"):
        MeaningfulEffectThreshold.from_dict({"schema_version": 2})


@pytest.mark.parametrize("direction", ["equivalence", "noninferiority"])
def test_formal_equivalence_and_noninferiority_requests_are_explicitly_unsupported(
    mean_case, direction
):
    _, assistant, base = mean_case
    result = assistant.practical_significance(
        base,
        threshold=MeaningfulEffectThreshold("mean_difference", 5, direction=direction),
    )
    assert result.status == "unsupported"
    assert direction in result.conclusion
    assert result.point_estimate_relation == "unavailable"
    assert "ordinary two-sided analysis" in result.warnings[0]


def test_statistical_and_practical_dimensions_remain_separate(mean_case):
    _, assistant, base = mean_case
    significant_but_below = assistant.practical_significance(
        _mean_result(base, 2, 1, 3, p_value=0.001),
        threshold=MeaningfulEffectThreshold("mean_difference", 5),
    )
    nonsignificant_but_large = assistant.practical_significance(
        _mean_result(base, 8, 2, 12, p_value=0.20),
        threshold=MeaningfulEffectThreshold("mean_difference", 5),
    )
    significant_but_uncertain = assistant.practical_significance(
        _mean_result(base, 8, 2, 12, p_value=0.001),
        threshold=MeaningfulEffectThreshold("mean_difference", 5),
    )
    assert significant_but_below.statistical_significance == "evidence_against_null"
    assert significant_but_below.point_estimate_relation == "below_meaningful_magnitude"
    assert nonsignificant_but_large.statistical_significance == "no_evidence_against_null"
    assert nonsignificant_but_large.point_estimate_relation == "positive_meaningful_region"
    assert significant_but_uncertain.statistical_significance == "evidence_against_null"
    assert significant_but_uncertain.confidence_interval_relation == ("crosses_meaningful_boundary")


def test_threshold_provenance_records_revisions_without_inventing_timing(mean_case):
    _, assistant, base = mean_case
    ledger = assistant.enable_tracking(clock=lambda: "2026-01-01T00:00:00Z")
    first = MeaningfulEffectThreshold("mean_difference", 5, rationale="Initial decision threshold.")
    second = MeaningfulEffectThreshold("mean_difference", 3, rationale="Revised by the researcher.")
    assistant.practical_significance(base, threshold=first)
    assistant.practical_significance(base, threshold=second)
    events = ledger.to_dict()["events"]
    assert [event["event_type"] for event in events] == [
        "meaningful_threshold_declared",
        "meaningful_threshold_declared",
    ]
    assert events[0]["previous_state"] is None
    assert events[1]["previous_state"]["minimum_magnitude"] == 5
    assert events[1]["new_state"]["minimum_magnitude"] == 3
    assert events[0]["metadata"]["planning_status"] == "unknown"


def _phase11_outputs(assistant, base):
    scenario = SensitivitySpecification(
        "pooled",
        base.specification,
        method_id="student_t",
        rationale="Explicit same-estimand alternative.",
        assumptions=("Equal population variances",),
    )
    sensitivity = assistant.sensitivity_analysis(base, scenarios=[scenario])
    practical = assistant.practical_significance(
        base,
        threshold=MeaningfulEffectThreshold(
            "mean_difference", 5, unit="points", rationale="Decision threshold."
        ),
    )
    return sensitivity, practical


def test_optional_report_sections_render_and_default_phase10_report_stays_schema_one(mean_case):
    _, assistant, base = mean_case
    ordinary = assistant.report(base)
    sensitivity, practical = _phase11_outputs(assistant, base)
    report = assistant.report(base, sensitivity=sensitivity, practical_significance=practical)
    payload = report.to_dict()

    assert ordinary.to_dict()["schema_version"] == 1
    assert "sensitivity" not in ordinary.to_dict()
    assert payload["schema_version"] == 2
    assert payload["sections"]["sensitivity_analysis"]["same_estimand_scenarios"] == 1
    assert payload["sections"]["practical_significance"]["threshold"] == 5
    assert "Sensitivity analysis" in report.to_markdown()
    assert "Practical significance" in report.to_html()
    assert "sensitivity_scenarios" in report.to_csv_tables()
    assert "practical_significance" in report.to_csv_tables()
    assert assistant.audit(report).status == "passed"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("threshold", "PRACTICAL_THRESHOLD_MISMATCH"),
        ("relation", "PRACTICAL_RELATION_MISMATCH"),
        ("estimate", "SENSITIVITY_ESTIMATE_MISMATCH"),
        ("comparability", "SENSITIVITY_COMPARABILITY_MISMATCH"),
        ("omitted", "SENSITIVITY_SCENARIO_OMITTED"),
        ("status", "REPORT_VALUE_MISMATCH"),
    ],
)
def test_auditor_detects_deliberate_phase11_report_errors(mean_case, mutation, code):
    _, assistant, base = mean_case
    sensitivity, practical = _phase11_outputs(assistant, base)
    report = assistant.report(base, sensitivity=sensitivity, practical_significance=practical)
    payload = report.to_dict()
    if mutation == "threshold":
        payload["practical_significance"]["threshold"]["minimum_magnitude"] = 3
    elif mutation == "relation":
        payload["practical_significance"]["point_estimate_relation"] = "below_meaningful_magnitude"
    elif mutation == "estimate":
        payload["sensitivity"]["scenario_results"][0]["primary_estimate"] = 999
    elif mutation == "comparability":
        payload["sensitivity"]["scenario_results"][0]["comparability"] = "different_estimand"
    elif mutation == "omitted":
        payload["sensitivity"]["scenario_results"] = []
    else:
        payload["sensitivity"]["scenario_results"][0]["status"] = "unavailable"
    changed = ResearchReport(
        payload,
        source_result=base,
        source_sensitivity=sensitivity,
        source_practical_significance=practical,
    )
    audit = assistant.audit(changed)
    assert audit.status == "failed"
    assert any(finding.code == code for finding in audit.findings)


def test_reporting_and_audit_do_not_rerun_sensitivity(mean_case, monkeypatch):
    _, assistant, base = mean_case
    sensitivity, practical = _phase11_outputs(assistant, base)

    def forbidden(*args, **kwargs):
        raise AssertionError("downstream stages must not rerun statistical analyses")

    monkeypatch.setattr(StatisticalAnalyzer, "hypothesis_tests", forbidden)
    report = assistant.report(base, sensitivity=sensitivity, practical_significance=practical)
    assert assistant.audit(report).status == "passed"


def test_phase11_report_retains_noncomparable_and_unsupported_scenarios_safely(mean_case):
    frame, assistant, base = mean_case
    rank_specification = replace(
        base.specification,
        question=replace(base.specification.question, estimand="distribution"),
    )
    scenarios = [
        SensitivitySpecification(
            "rank | supplement",
            rank_specification,
            method_id="mann_whitney_u",
            rationale="<script>alert('comparison')</script>",
        ),
        SensitivitySpecification(
            "unsupported",
            base.specification,
            method_id="spearman_coefficient",
            rationale="=FORMULA()",
        ),
    ]
    sensitivity = assistant.sensitivity_analysis(base, scenarios=scenarios)
    report = assistant.report(base, sensitivity=sensitivity)
    payload = report.to_dict()
    rows = payload["tables"][-1]["rows"]

    assert [row[5]["value"] for row in rows] == ["different_estimand", "unavailable"]
    assert [row[4]["value"] for row in rows] == ["completed", "unavailable"]
    assert str(frame.iloc[0].to_dict()) not in json.dumps(payload, allow_nan=False)
    assert "<script>" not in report.to_html()
    assert "&lt;script&gt;" in report.to_html()
    assert "rank \\| supplement" in report.to_markdown()
    assert "'=FORMULA()" in report.to_csv_tables()["sensitivity_scenarios"]


def test_phase11_reproducibility_metadata_is_json_safe_and_replay_remains_explicit(mean_case):
    frame, assistant, base = mean_case
    sensitivity, practical = _phase11_outputs(assistant, base)
    record = assistant.reproducibility_record(
        base, sensitivity=sensitivity, practical_significance=practical
    )
    payload = record.to_dict()
    assert payload["schema_version"] == 2
    assert payload["phase11"]["automatic_replay"] is False
    assert payload["phase11"]["sensitivity"]["configuration"]["scenario_order"] == ["pooled"]
    assert payload["phase11"]["practical_significance"]["threshold"]["minimum_magnitude"] == 5
    assert "DataFrame" not in json.dumps(payload, allow_nan=False)
    from pyautostat import reproduce

    replay = reproduce(record, data=frame)
    assert replay.status == "reproduced"
    assert any("not automatically replayed" in warning for warning in replay.warnings)


def test_unavailable_analysis_returns_unavailable_practical_assessment(mean_case):
    _, assistant, base = mean_case
    unavailable = replace(base, status=AnalysisStatus.UNAVAILABLE)
    result = assistant.practical_significance(
        unavailable, threshold=MeaningfulEffectThreshold("mean_difference", 5)
    )
    assert result.status == "unavailable"
    assert result.estimate is None
    assert result.point_estimate_relation == "unavailable"


def test_integrated_run_does_not_execute_hidden_sensitivity_scenarios(mean_case, monkeypatch):
    frame, _, _ = mean_case
    assistant = ResearchAssistant(frame)
    import pyautostat.research_assistant as module

    def forbidden(*args, **kwargs):
        raise AssertionError("run() must not execute a sensitivity method")

    monkeypatch.setattr(module, "execute_selected_method", forbidden)
    workflow = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        data_dictionary={"score": {"type": "continuous"}},
    )
    assert workflow.status.value in {"completed", "partial"}
    assert workflow.analysis.method_id == "welch_t"
