"""Phase 7 rules consume real Phase 6 results without recomputing statistics."""

import json
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from pyautostat import (
    AnalysisOptions,
    AnalysisResult,
    AnalysisStatus,
    InterpretationEngine,
    InterpretationFinding,
    InterpretationResult,
    InterpretationStatus,
    InvalidDataError,
    ResearchAssistant,
    StatisticalAnalyzer,
)
from pyautostat.execution import _group_result


def _codes(interpretation):
    return {item.code for item in interpretation.findings}


@pytest.fixture(scope="module")
def group_case():
    frame = pd.DataFrame(
        {
            "group": ["B"] * 8 + ["A"] * 8 + ["A"],
            "score": list(range(1, 9)) + list(range(3, 11)) + [None],
        }
    )
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        data_dictionary={"score": {"type": "continuous", "unit": "points"}},
    )
    return frame, assistant, assistant.analyze(draft)


def _with_values(result, **changes):
    values = deepcopy(result.values)
    values.update(changes)
    return replace(result, values=values)


def test_welch_integration_preserves_contrast_units_and_source(group_case, monkeypatch):
    frame, assistant, result = group_case
    before_frame = frame.copy(deep=True)
    before_result = deepcopy(result.to_dict())

    def forbidden(*args, **kwargs):
        raise AssertionError("interpretation must not run a numerical backend")

    monkeypatch.setattr(StatisticalAnalyzer, "hypothesis_tests", forbidden)
    interpretation = assistant.interpret(result)
    assert interpretation.status is InterpretationStatus.AVAILABLE
    assert interpretation.execution_status is AnalysisStatus.AVAILABLE
    assert "'B' minus 'A'" in interpretation.effect_interpretation
    assert "points" in interpretation.effect_interpretation
    assert "Cohen's d" in interpretation.effect_interpretation
    assert "bootstrap interval" in interpretation.effect_interpretation
    assert "mean difference" in interpretation.uncertainty_interpretation
    assert "interval_includes_null" in _codes(interpretation)
    assert "insufficient_evidence_against_null" in _codes(interpretation)
    assert any("does not establish normality" in note for note in interpretation.assumption_notes)
    assert any("does not prove" in note for note in interpretation.assumption_notes)
    assert interpretation.metadata["p_value"] == result.values["p_value"]
    assert interpretation.metadata["primary_estimate"] == result.values["primary_estimate"]
    assert interpretation.metadata["excluded_rows"] == 1
    assert (
        json.loads(json.dumps(interpretation.to_dict(), allow_nan=False))
        == interpretation.to_dict()
    )
    assert interpretation.to_dict() == assistant.interpret(result).to_dict()
    assert result.to_dict() == before_result
    pd.testing.assert_frame_equal(frame, before_frame)


@pytest.mark.parametrize(
    ("p_value", "alpha", "expected"),
    [
        (0.049999, 0.05, "evidence_against_null"),
        (0.05, 0.05, "insufficient_evidence_against_null"),
        (0.050001, 0.05, "insufficient_evidence_against_null"),
        (0.009, 0.01, "evidence_against_null"),
        (0.02, 0.01, "insufficient_evidence_against_null"),
    ],
)
def test_threshold_uses_recorded_alpha_and_unrounded_p(group_case, p_value, alpha, expected):
    _, assistant, result = group_case
    spec = replace(result.specification, options=AnalysisOptions(alpha=alpha))
    altered = replace(_with_values(result, p_value=p_value), specification=spec)
    interpretation = assistant.interpret(altered)
    assert expected in _codes(interpretation)
    assert f"alpha = {alpha}" in interpretation.conclusion
    assert interpretation.metadata["p_value"] == p_value


@pytest.mark.parametrize("p_value", [None, -0.01, 1.01, float("nan"), float("inf")])
def test_invalid_or_missing_p_has_no_significance_decision(group_case, p_value):
    _, assistant, result = group_case
    interpretation = assistant.interpret(_with_values(result, p_value=p_value))
    assert interpretation.status is InterpretationStatus.PARTIAL
    assert interpretation.conclusion is None
    assert "p_value_unavailable" in _codes(interpretation)
    assert not _codes(interpretation) & {
        "evidence_against_null",
        "insufficient_evidence_against_null",
    }


@pytest.mark.parametrize(
    ("p_value", "phrase"),
    [(1e-12, "1e-12"), (0.0, "p < 0.001")],
)
def test_very_small_p_display_preserves_numeric_source(group_case, p_value, phrase):
    _, assistant, result = group_case
    interpretation = assistant.interpret(_with_values(result, p_value=p_value))
    assert phrase in interpretation.hypothesis_interpretation
    assert "p = 0.000" not in interpretation.hypothesis_interpretation
    assert interpretation.metadata["p_value"] == p_value
    if p_value == 0:
        assert any("machine precision" in warning for warning in interpretation.warnings)


def test_reversed_group_contrast_and_direction_follow_metadata(group_case):
    _, assistant, result = group_case
    values = deepcopy(result.values)
    values["primary_estimate"] *= -1
    values["effect_size"]["value"] *= -1
    values["confidence_interval"]["lower"], values["confidence_interval"]["upper"] = (
        -values["confidence_interval"]["upper"],
        -values["confidence_interval"]["lower"],
    )
    (
        values["effect_size"]["confidence_interval"]["lower"],
        values["effect_size"]["confidence_interval"]["upper"],
    ) = (
        -values["effect_size"]["confidence_interval"]["upper"],
        -values["effect_size"]["confidence_interval"]["lower"],
    )
    metadata = deepcopy(result.metadata)
    metadata["group_order"] = ["A", "B"]
    metadata["contrast"] = {
        "definition": "first group minus second group",
        "first": "A",
        "second": "B",
    }
    altered = replace(result, values=values, metadata=metadata)
    interpretation = assistant.interpret(altered)
    assert "'A' minus 'B'" in interpretation.effect_interpretation
    assert "estimate_positive" in _codes(interpretation)
    assert "B' minus 'A" not in interpretation.effect_interpretation


def test_interval_excludes_null_and_disagreement_is_flagged(group_case):
    _, assistant, result = group_case
    values = deepcopy(result.values)
    values["p_value"] = 0.01
    values["confidence_interval"]["lower"] = -3.0
    values["confidence_interval"]["upper"] = -1.0
    consistent = assistant.interpret(replace(result, values=values))
    assert "interval_excludes_null" in _codes(consistent)
    assert consistent.status is InterpretationStatus.AVAILABLE
    values["p_value"] = 0.2
    inconsistent = assistant.interpret(replace(result, values=values))
    assert inconsistent.status is InterpretationStatus.PARTIAL
    assert any("different threshold decisions" in item for item in inconsistent.warnings)


@pytest.mark.parametrize(
    "interval",
    [
        None,
        {"lower": 1.0, "upper": -1.0, "level": 0.95, "quantity": "mean difference"},
        {"lower": -3.0, "upper": -1.0, "level": 0.95, "quantity": "Cohen's d"},
        {"lower": -5.0, "upper": -3.0, "level": 0.95, "quantity": "mean difference"},
        {"lower": float("nan"), "upper": 1.0, "level": 0.95, "quantity": "mean difference"},
    ],
)
def test_missing_or_invalid_primary_interval_does_not_invent_bounds(group_case, interval):
    _, assistant, result = group_case
    interpretation = assistant.interpret(_with_values(result, confidence_interval=interval))
    assert interpretation.status is InterpretationStatus.PARTIAL
    assert interpretation.metadata["confidence_interval"] is None
    assert not _codes(interpretation) & {"interval_includes_null", "interval_excludes_null"}


def test_unavailable_execution_ignores_stale_numbers(group_case):
    _, assistant, result = group_case
    stale = replace(result, status=AnalysisStatus.UNAVAILABLE)
    interpretation = assistant.interpret(stale)
    assert interpretation.status is InterpretationStatus.UNAVAILABLE
    assert interpretation.execution_status is AnalysisStatus.UNAVAILABLE
    assert interpretation.conclusion is None
    assert interpretation.metadata.get("p_value") is None
    assert "interpretation_unavailable" in _codes(interpretation)


def test_unsupported_incomplete_question_is_unavailable(group_case):
    _, assistant, _ = group_case
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
    )
    result = assistant.analyze(draft)
    interpretation = assistant.interpret(result)
    assert result.status is AnalysisStatus.UNAVAILABLE
    assert interpretation.status is InterpretationStatus.UNAVAILABLE
    assert interpretation.conclusion is None


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"method_id": "spearman_correlation"}, "not supported"),
        ({"method_id": "unknown"}, "not supported"),
        ({"sample_size": None}, "sample size"),
    ],
)
def test_unsupported_or_incomplete_contract_is_unavailable(group_case, change, fragment):
    _, assistant, result = group_case
    interpretation = assistant.interpret(replace(result, **change))
    assert interpretation.status is InterpretationStatus.UNAVAILABLE
    assert fragment in interpretation.summary


def test_missing_group_order_or_contrast_is_unavailable(group_case):
    _, assistant, result = group_case
    for key in ("group_order", "contrast"):
        metadata = deepcopy(result.metadata)
        metadata.pop(key)
        interpretation = assistant.interpret(replace(result, metadata=metadata))
        assert interpretation.status is InterpretationStatus.UNAVAILABLE
        assert interpretation.conclusion is None


def test_missing_effect_is_partial_and_does_not_invent_magnitude(group_case):
    _, assistant, result = group_case
    interpretation = assistant.interpret(_with_values(result, effect_size=None))
    assert interpretation.status is InterpretationStatus.PARTIAL
    assert interpretation.effect_interpretation is None
    assert "effect_unavailable" in _codes(interpretation)


def test_mann_whitney_interprets_ranks_without_median_claim(group_case):
    frame, assistant, _ = group_case
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="distribution",
        design="independent",
        variable_types={"score": "continuous"},
    )
    result = assistant.analyze(draft)
    interpretation = assistant.interpret(result)
    assert result.method_id == "mann_whitney_u"
    assert interpretation.status is InterpretationStatus.AVAILABLE
    assert "rank-biserial" in interpretation.effect_interpretation.lower()
    assert "not a median difference" in interpretation.effect_interpretation
    assert frame.shape[0] == 17


def test_kruskal_is_omnibus_and_does_not_name_pairwise_difference():
    frame = pd.DataFrame(
        {"group": ["C"] * 5 + ["A"] * 5 + ["B"] * 5, "score": [float(i) for i in range(15)]}
    )
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="distribution",
        design="independent",
        variable_types={"score": "continuous"},
    )
    result = assistant.analyze(draft)
    interpretation = assistant.interpret(result)
    assert interpretation.method_id == "kruskal_wallis"
    assert "epsilon-squared" in interpretation.effect_interpretation.lower()
    assert any("does not identify specific" in item for item in interpretation.limitations)
    assert interpretation.status is InterpretationStatus.PARTIAL


@pytest.mark.parametrize("association", [1, -1, 0])
def test_pearson_direction_without_causation_or_independence_claim(association):
    frame = pd.DataFrame({"hours": [1.0, 2.0, 3.0, 4.0, 5.0], "score": [1.0, 3.0, 2.0, 5.0, 4.0]})
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="association",
        outcome="hours",
        predictor="score",
        estimand="linear",
        design="independent",
        variable_types={"hours": "continuous", "score": "continuous"},
    )
    result = assistant.analyze(draft)
    value = 0.0 if association == 0 else 0.4 * association
    altered = _with_values(result, primary_estimate=value, test_statistic=value)
    altered.values["effect_size"]["value"] = value
    interpretation = assistant.interpret(altered)
    label = "positive" if value > 0 else "negative" if value < 0 else "zero"
    assert label in interpretation.effect_interpretation
    assert "causation" in " ".join(interpretation.limitations)
    assert "independent" not in interpretation.effect_interpretation
    assert interpretation.status is InterpretationStatus.PARTIAL


def test_chi_square_effect_has_no_direction():
    frame = pd.DataFrame(
        {
            "group": ["B"] * 20 + ["A"] * 20,
            "response": ["yes"] * 15 + ["no"] * 5 + ["yes"] * 5 + ["no"] * 15,
        }
    )
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="association",
        outcome="response",
        predictor="group",
        design="independent",
    )
    result = assistant.analyze(draft)
    interpretation = assistant.interpret(result)
    assert result.method_id == "pearson_chi_square"
    assert interpretation.status is InterpretationStatus.AVAILABLE
    assert "nonnegative" in interpretation.effect_interpretation
    assert "estimate_nonnegative" in _codes(interpretation)
    assert "causation" in " ".join(interpretation.limitations)


def test_student_and_anova_templates_consume_actual_backend_adapters(group_case):
    _, assistant, welch = group_case
    student_recommendation = replace(
        welch.recommendation, method_id="student_t", method_name="Student t-test"
    )
    student = _group_result(assistant._analyzer, welch.specification, student_recommendation)
    student_interpretation = assistant.interpret(student)
    assert student_interpretation.status is InterpretationStatus.AVAILABLE
    assert "pooled-variance" in student_interpretation.method_explanation
    assert "equal population variances" in student_interpretation.assumption_notes[0]

    frame = pd.DataFrame({"group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5, "score": list(range(15))})
    multi = ResearchAssistant(frame)
    draft = multi.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )
    recommendation = replace(
        welch.recommendation, method_id="one_way_anova", method_name="One-way ANOVA"
    )
    anova = _group_result(multi._analyzer, draft.specification, recommendation)
    interpreted = multi.interpret(anova)
    assert interpreted.method_id == "one_way_anova"
    assert "Eta-squared" in interpreted.effect_interpretation
    assert any("specific group differences" in item for item in interpreted.limitations)


def test_descriptive_profile_has_no_inferential_conclusion_or_file_write(tmp_path):
    frame = pd.DataFrame({"score": [1.0, 2.0, 3.0, None], "group": ["A", "A", "B", "B"]})
    assistant = ResearchAssistant(frame)
    result = assistant.analyze(assistant.prepare_question(objective="descriptive"))
    interpretation = assistant.interpret(result)
    assert interpretation.status is InterpretationStatus.AVAILABLE
    assert interpretation.hypothesis_interpretation is None
    assert "descriptive_only" in _codes(interpretation)
    assert interpretation.metadata["numeric_summaries"][0]["mean"] == 2.0
    assert not list(tmp_path.iterdir())


def test_invalid_api_input_is_actionable():
    with pytest.raises(InvalidDataError, match="AnalysisResult"):
        InterpretationEngine().interpret({"method_id": "welch_t"})
    with pytest.raises(InvalidDataError, match="status"):
        AnalysisResult(method_id="x", status="bad")


def test_interpretation_model_rejects_invalid_records():
    with pytest.raises(InvalidDataError, match="finding code"):
        InterpretationFinding("", "missing code")
    base = {
        "status": "available",
        "execution_status": "available",
        "method_id": "welch_t",
        "summary": "summary",
    }
    with pytest.raises(InvalidDataError, match="status"):
        InterpretationResult(**{**base, "status": "wrong"})
    with pytest.raises(InvalidDataError, match="required"):
        InterpretationResult(**{**base, "summary": ""})
    with pytest.raises(InvalidDataError, match="findings"):
        InterpretationResult(**{**base, "findings": ("text",)})
    with pytest.raises(InvalidDataError, match="JSON"):
        InterpretationResult(**{**base, "metadata": {"bad": float("nan")}})


def test_missing_specification_and_sample_contradiction_are_unavailable(group_case):
    _, assistant, result = group_case
    assert assistant.interpret(replace(result, specification=None)).status == "unavailable"
    metadata = deepcopy(result.metadata)
    metadata["sample"]["analyzed_rows"] = 999
    interpretation = assistant.interpret(replace(result, metadata=metadata))
    assert interpretation.status is InterpretationStatus.UNAVAILABLE
    assert "sample counts" in interpretation.summary


def test_missing_question_variable_and_wrong_group_count_are_unavailable(group_case):
    _, assistant, result = group_case
    question = replace(result.specification.question, predictor=None)
    specification = replace(result.specification, question=question)
    assert assistant.interpret(replace(result, specification=specification)).status == "unavailable"
    metadata = deepcopy(result.metadata)
    metadata["group_order"] = ["A", "B", "C"]
    assert assistant.interpret(replace(result, metadata=metadata)).status == "unavailable"


def test_diagnostics_and_source_warnings_are_preserved(group_case):
    _, assistant, result = group_case
    metadata = deepcopy(result.metadata)
    metadata["diagnostics"]["normality"] = [
        "malformed",
        {"group": "B", "status": "rejected"},
        {"group": "A", "status": "unknown"},
    ]
    metadata["diagnostics"]["equal_variance_status"] = "rejected"
    altered = replace(
        result, metadata=metadata, warnings=("Declared missing code requires review.",)
    )
    interpretation = assistant.interpret(altered)
    notes = " ".join(interpretation.assumption_notes)
    assert "rejected normality" in notes
    assert "unknown" in notes
    assert "rejected equal variances" in notes
    assert interpretation.warnings[0] == "Declared missing code requires review."
    metadata["diagnostics"]["equal_variance_status"] = "unknown"
    assert "Equal-variance diagnostic: unknown" in " ".join(
        assistant.interpret(replace(result, metadata=metadata)).assumption_notes
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("test_statistic", None, "test statistic"),
        ("primary_estimate", float("inf"), "effect"),
        ("estimate_name", None, "named quantity"),
    ],
)
def test_incomplete_core_values_do_not_produce_full_interpretation(
    group_case, field, value, expected
):
    _, assistant, result = group_case
    interpretation = assistant.interpret(_with_values(result, **{field: value}))
    assert interpretation.status is InterpretationStatus.PARTIAL
    assert expected in " ".join(interpretation.warnings)


def test_invalid_alpha_and_hypothesis_metadata_prevent_threshold_claim(group_case):
    _, assistant, result = group_case
    options = replace(result.specification.options)
    object.__setattr__(options, "alpha", float("nan"))
    altered = replace(result, specification=replace(result.specification, options=options))
    interpretation = assistant.interpret(altered)
    assert interpretation.status is InterpretationStatus.PARTIAL
    assert interpretation.conclusion is None
    assert interpretation.metadata["alpha"] is None
    metadata = deepcopy(result.metadata)
    metadata.pop("null_hypothesis")
    interpretation = assistant.interpret(replace(result, metadata=metadata))
    assert interpretation.conclusion is None
    assert "p_value_without_decision" in _codes(interpretation)


def test_effect_range_and_cross_field_mismatch_are_flagged(group_case):
    _, assistant, result = group_case
    rank = assistant.analyze(
        assistant.prepare_question(
            objective="compare_groups",
            outcome="score",
            predictor="group",
            estimand="distribution",
            design="independent",
            variable_types={"score": "continuous"},
        )
    )
    invalid = _with_values(rank, primary_estimate=2.0)
    interpretation = assistant.interpret(invalid)
    assert interpretation.status is InterpretationStatus.PARTIAL
    assert interpretation.effect_interpretation is None
    mismatched = _with_values(rank, primary_estimate=0.25)
    interpreted = assistant.interpret(mismatched)
    assert interpreted.status is InterpretationStatus.PARTIAL
    assert any("disagrees" in warning for warning in interpreted.warnings)


@pytest.mark.parametrize(
    "effect_interval",
    [
        "invalid",
        {"quantity": "Cohen's d", "lower": 2.0, "upper": 3.0, "level": 0.95},
    ],
)
def test_invalid_standardized_effect_interval_is_separate_from_mean_ci(group_case, effect_interval):
    _, assistant, result = group_case
    values = deepcopy(result.values)
    values["effect_size"]["confidence_interval"] = effect_interval
    interpreted = assistant.interpret(replace(result, values=values))
    assert interpreted.status is InterpretationStatus.PARTIAL
    assert "mean difference" in interpreted.uncertainty_interpretation
    assert any("standardized-effect interval" in warning for warning in interpreted.warnings)


def test_malformed_profile_is_unavailable():
    frame = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
    assistant = ResearchAssistant(frame)
    result = assistant.analyze(assistant.prepare_question(objective="descriptive"))
    for profile in (None, {"descriptive": [], "categorical_summary": {}}):
        assert assistant.interpret(_with_values(result, profile=profile)).status == "unavailable"
    broken = deepcopy(result.values["profile"])
    broken["descriptive"]["score"]["mean"] = float("nan")
    interpreted = assistant.interpret(_with_values(result, profile=broken))
    assert interpreted.status is InterpretationStatus.UNAVAILABLE
