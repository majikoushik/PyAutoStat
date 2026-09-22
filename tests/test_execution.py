"""Phase 6 execution checks against genuine analyzer calculations."""

import json
import math
from copy import deepcopy

import pandas as pd
import pytest
from scipy import stats

from pyautostat import (
    AnalysisOptions,
    AnalysisResult,
    AnalysisSpecification,
    AnalysisStatus,
    ColumnNotFoundError,
    InsufficientDataError,
    ResearchAssistant,
    ResearchQuestion,
    StatisticalAnalyzer,
    execution,
)
from pyautostat.recommendation import METHOD_CAPABILITIES
from pyautostat.results import Recommendation


def _draft(assistant, *, target="mean", design="independent", **extra):
    return assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand=target,
        design=design,
        variable_types={"score": "continuous"},
        **extra,
    )


@pytest.fixture
def two_groups():
    frame = pd.DataFrame(
        {
            "group": ["B"] * 8 + ["A"] * 8 + ["A"],
            "score": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
            + [3.1, 4.2, 5.3, 6.4, 7.5, 8.6, 9.7, 10.8]
            + [None],
        }
    )
    return frame, ResearchAssistant(frame)


def test_welch_end_to_end_uses_backend_numbers_and_preserves_context(two_groups):
    frame, assistant = two_groups
    original = frame.copy(deep=True)
    draft = _draft(
        assistant,
        data_dictionary={"score": {"type": "continuous", "unit": "points"}},
    )
    recommendation = assistant.recommend_test(draft)
    result = assistant.analyze(draft)
    first = frame.loc[frame.group == "B", "score"].dropna().to_numpy()
    second = frame.loc[frame.group == "A", "score"].dropna().to_numpy()
    reference = stats.ttest_ind(first, second, equal_var=False)
    assert isinstance(result, AnalysisResult)
    assert result.status is AnalysisStatus.AVAILABLE
    assert result.method_id == recommendation.method_id == "welch_t"
    assert result.values["test_statistic"] == pytest.approx(reference.statistic)
    assert result.values["p_value"] == pytest.approx(reference.pvalue)
    assert result.values["primary_estimate"] == pytest.approx(first.mean() - second.mean())
    assert result.values["estimate_name"] == "mean difference"
    assert result.values["estimate_unit"] == "points"
    assert result.values["effect_size"]["name"] == "Cohen's d"
    assert result.values["effect_size"]["value"] < 0
    assert result.values["confidence_interval"]["quantity"] == "mean difference"
    assert result.values["confidence_interval"]["method"] == "analytical t interval"
    assert result.values["effect_size"]["confidence_interval"]["quantity"] == "Cohen's d"
    assert result.metadata["group_order"] == ["B", "A"]
    assert result.metadata["contrast"]["definition"] == "first group minus second group"
    assert result.metadata["sample"]["group_sizes"] == [
        {"group": "B", "size": 8},
        {"group": "A", "size": 8},
    ]
    assert result.metadata["sample"]["original_rows"] == 17
    assert result.sample_size == 16
    assert result.excluded_rows == 1
    assert result.specification.to_dict() == draft.specification.to_dict()
    assert result.recommendation.method_id == "welch_t"
    assert result.metadata["diagnostics"]["normality"]
    assert result.specification.options.alpha == 0.05
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False)) == result.to_dict()
    pd.testing.assert_frame_equal(frame, original)


def test_reversing_observed_group_order_reverses_welch_contrast(two_groups):
    frame, assistant = two_groups
    first = assistant.analyze(_draft(assistant))
    reversed_frame = pd.concat([frame.iloc[8:], frame.iloc[:8]], ignore_index=True)
    reversed_assistant = ResearchAssistant(reversed_frame)
    second = reversed_assistant.analyze(_draft(reversed_assistant))
    assert second.metadata["group_order"] == ["A", "B"]
    assert second.values["primary_estimate"] == pytest.approx(-first.values["primary_estimate"])
    assert second.values["effect_size"]["value"] == pytest.approx(
        -first.values["effect_size"]["value"]
    )


def test_direct_specification_preserves_options(two_groups):
    _, assistant = two_groups
    spec = AnalysisSpecification(
        question=ResearchQuestion(
            objective="compare_groups", outcome="score", predictor="group", estimand="mean"
        ),
        design="independent",
        options=AnalysisOptions(alpha=0.01, confidence_level=0.90, random_seed=12),
        data_dictionary={"score": {"type": "continuous"}},
    )
    result = assistant.analyze(specification=spec)
    assert result.status == "available"
    assert result.values["confidence_interval"]["level"] == 0.90
    assert result.metadata["effective_random_seed"] == 12
    assert result.specification.options.alpha == 0.01
    assert result.specification.to_dict() == spec.to_dict()


def test_incomplete_and_unsupported_requests_stop_before_backend(monkeypatch, two_groups):
    _, assistant = two_groups

    def forbidden(*args, **kwargs):
        raise AssertionError("The numerical backend must not run")

    monkeypatch.setattr(StatisticalAnalyzer, "hypothesis_tests", forbidden)
    incomplete = assistant.analyze(
        assistant.prepare_question(
            objective="compare_groups",
            outcome="score",
            predictor="group",
            estimand="mean",
            variable_types={"score": "continuous"},
        )
    )
    assert incomplete.status == "unavailable"
    assert incomplete.method_id == "unselected"
    assert incomplete.recommendation.status == "needs_input"
    assert incomplete.values == {}
    for design in ("paired", "repeated", "clustered"):
        blocked = assistant.analyze(_draft(assistant, design=design))
        assert blocked.status == "unavailable"
        assert blocked.recommendation.status == "unsupported"
        assert design in blocked.warnings[-1]


def test_direct_specification_is_revalidated_against_current_dataset(two_groups):
    _, assistant = two_groups
    invalid = AnalysisSpecification(
        question=ResearchQuestion(
            objective="compare_groups", outcome="absent", predictor="group", estimand="mean"
        ),
        design="independent",
    )
    with pytest.raises(ColumnNotFoundError, match="absent"):
        assistant.analyze(specification=invalid)


def test_distribution_target_executes_mann_whitney_with_correct_effect(two_groups):
    frame, assistant = two_groups
    result = assistant.analyze(_draft(assistant, target="distribution"))
    first = frame.loc[frame.group == "B", "score"].dropna().to_numpy()
    second = frame.loc[frame.group == "A", "score"].dropna().to_numpy()
    reference = stats.mannwhitneyu(first, second, alternative="two-sided")
    assert result.status == "available"
    assert result.method_id == "mann_whitney_u"
    assert result.values["p_value"] == pytest.approx(reference.pvalue)
    assert result.values["test_statistic"] == pytest.approx(reference.statistic)
    assert result.values["primary_estimate"] == pytest.approx(
        2 * reference.statistic / (len(first) * len(second)) - 1
    )
    assert result.values["estimate_name"] == "rank-biserial correlation"
    assert result.values["confidence_interval"]["quantity"] == "rank-biserial correlation"
    assert result.specification.question.estimand == "distribution"


def test_kruskal_executes_without_pairwise_or_mean_claim():
    frame = pd.DataFrame(
        {
            "group": ["C"] * 5 + ["A"] * 5 + ["B"] * 5,
            "score": [float(i) + 0.1 for i in range(15)],
        }
    )
    assistant = ResearchAssistant(frame)
    result = assistant.analyze(_draft(assistant, target="distribution"))
    reference = stats.kruskal(
        *(frame.loc[frame.group == name, "score"] for name in ("C", "A", "B"))
    )
    assert result.status == "available"
    assert result.method_id == "kruskal_wallis"
    assert result.values["test_statistic"] == pytest.approx(reference.statistic)
    assert result.values["p_value"] == pytest.approx(reference.pvalue)
    assert result.values["degrees_of_freedom"] == 2
    assert result.metadata["group_order"] == ["C", "A", "B"]
    assert result.metadata["contrast"] is None
    assert result.values["estimate_name"] == "epsilon-squared (rank)"
    assert result.sample_size == 15


def test_multi_group_mean_stays_unavailable():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "score": [float(i) + 0.1 for i in range(15)],
        }
    )
    assistant = ResearchAssistant(frame)
    result = assistant.analyze(_draft(assistant, target="mean"))
    assert result.status == "unavailable"
    assert result.method_id == "unselected"
    assert result.recommendation.status == "unsupported"
    assert result.specification.question.estimand == "mean"


def test_pearson_uses_existing_profile_inference_and_complete_pairs():
    frame = pd.DataFrame({"hours": [1.1, 2.2, 3.3, None, 5.5], "score": [2.1, 3.2, 5.3, 7.4, 11.5]})
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
    pairs = frame.dropna()
    reference = stats.pearsonr(pairs.hours, pairs.score)
    assert result.status == "available"
    assert result.method_id == "pearson_correlation"
    assert result.values["primary_estimate"] == pytest.approx(reference.statistic)
    assert result.values["p_value"] == pytest.approx(reference.pvalue)
    assert result.values["confidence_interval"] is None
    assert result.metadata["sample"]["effective_pair_count"] == 4
    assert result.sample_size == 4
    assert result.excluded_rows == 1
    assert result.metadata["variable_order"] == ["hours", "score"]


def test_chi_square_uses_backend_table_and_records_nonnegative_effect():
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
    reference = stats.chi2_contingency([[15, 5], [5, 15]], correction=False)
    assert result.status == "available"
    assert result.method_id == "pearson_chi_square"
    assert result.values["test_statistic"] == pytest.approx(reference.statistic)
    assert result.values["p_value"] == pytest.approx(reference.pvalue)
    assert result.values["primary_estimate"] == pytest.approx(math.sqrt(reference.statistic / 40))
    assert result.values["effect_size"]["name"] == "Cramer's V"
    assert result.values["confidence_interval"]["quantity"] == "Cramer's V"
    assert result.metadata["group_order"] == ["B", "A"]
    assert result.metadata["outcome_order"] == ["yes", "no"]
    assert result.metadata["sample"]["group_sizes"] == [
        {"group": "B", "size": 20},
        {"group": "A", "size": 20},
    ]
    assert result.metadata["observed_counts"] == [[15, 5], [5, 15]]


def test_sparse_chi_square_is_blocked_before_backend(monkeypatch):
    frame = pd.DataFrame({"group": ["A"] * 4 + ["B"] * 4, "response": ["yes", "no"] * 4})
    assistant = ResearchAssistant(frame)

    def forbidden(*args, **kwargs):
        raise AssertionError("Chi-square backend must not run")

    monkeypatch.setattr(StatisticalAnalyzer, "categorical_association", forbidden)
    result = assistant.analyze(
        assistant.prepare_question(
            objective="association", outcome="response", predictor="group", design="independent"
        )
    )
    assert result.status == "unavailable"
    assert result.recommendation.status == "unsupported"
    assert "expected cell count" in result.warnings[-1]


def test_descriptive_profile_has_no_invented_inference(two_groups):
    _, assistant = two_groups
    result = assistant.analyze(assistant.prepare_question(objective="descriptive"))
    assert result.status == "available"
    assert result.method_id == "dataset_profile"
    assert result.metadata["inference"] is False
    assert result.values["test_statistic"] is None
    assert result.values["p_value"] is None
    assert result.values["profile"]["overview"]["total_rows"] == 17
    json.dumps(result.to_dict(), allow_nan=False)


def test_unapplied_missing_codes_and_constant_outcome_are_unavailable(two_groups):
    _, assistant = two_groups
    coded = assistant.analyze(
        _draft(
            assistant,
            data_dictionary={"score": {"type": "continuous", "missing_codes": [1.1]}},
        )
    )
    assert coded.status == "unavailable"
    assert "missing-code" in coded.warnings[-1]
    constant = ResearchAssistant(pd.DataFrame({"group": ["A"] * 4 + ["B"] * 4, "score": [1.0] * 8}))
    blocked = constant.analyze(_draft(constant))
    assert blocked.status == "unavailable"
    assert blocked.values == {}


def test_known_backend_numerical_failure_returns_unavailable(monkeypatch, two_groups):
    _, assistant = two_groups

    def failure(*args, **kwargs):
        raise InsufficientDataError("Numerical result undefined")

    monkeypatch.setattr(StatisticalAnalyzer, "hypothesis_tests", failure)
    result = assistant.analyze(_draft(assistant))
    assert result.status == "unavailable"
    assert result.method_id == "welch_t"
    assert result.values == {}
    assert "Numerical result undefined" in result.warnings[-1]


def test_legacy_backend_and_result_records_remain_compatible(two_groups):
    frame, assistant = two_groups
    previous = StatisticalAnalyzer(frame).hypothesis_tests(
        "group", "score", test_type="auto", estimand="mean", bootstrap_samples=0
    )
    assert previous["test"] == "t-test"
    assert assistant.recommend_test(_draft(assistant)).method_id == "welch_t"
    assert assistant.profile()["overview"]["total_rows"] == 17
    legacy = AnalysisResult(method_id="legacy", status="available", values={"p_value": 0.5})
    assert legacy.to_dict()["specification"] is None
    assert legacy.to_dict()["recommendation"] is None
    assert METHOD_CAPABILITIES["student_t"].availability == "runnable"
    assert METHOD_CAPABILITIES["one_way_anova"].availability == "runnable"


def test_analysis_does_not_mutate_draft_or_assistant_data(two_groups):
    frame, assistant = two_groups
    draft = _draft(assistant)
    before = deepcopy(draft.specification.to_dict())
    first = assistant.analyze(draft).to_dict()
    second = assistant.analyze(draft).to_dict()
    assert first == second
    assert draft.specification.to_dict() == before
    pd.testing.assert_frame_equal(frame, assistant._analyzer.df)


def test_missing_effect_interval_is_explicit_without_losing_mean_interval(monkeypatch, two_groups):
    _, assistant = two_groups
    original = StatisticalAnalyzer.hypothesis_tests

    def no_effect_interval(self, *args, **kwargs):
        raw = original(self, *args, **kwargs)
        raw["effect_size"]["confidence_interval"] = None
        raw["assumptions"]["warnings"].append("Bootstrap interval unavailable.")
        return raw

    monkeypatch.setattr(StatisticalAnalyzer, "hypothesis_tests", no_effect_interval)
    result = assistant.analyze(_draft(assistant))
    assert result.status == "available"
    assert result.values["effect_size"]["confidence_interval"] is None
    assert result.values["confidence_interval"]["quantity"] == "mean difference"
    assert "Bootstrap interval unavailable." in result.warnings


@pytest.mark.parametrize("corruption", ["p_value", "interval_level", "degrees_of_freedom"])
def test_invalid_backend_numbers_cannot_become_available(monkeypatch, two_groups, corruption):
    _, assistant = two_groups
    original = StatisticalAnalyzer.hypothesis_tests

    def corrupt(self, *args, **kwargs):
        raw = original(self, *args, **kwargs)
        if corruption == "p_value":
            raw["p_value"] = float("nan")
        elif corruption == "interval_level":
            raw["confidence_interval"]["level"] = 0.5
        else:
            raw["degrees_of_freedom"] = float("inf")
        return raw

    monkeypatch.setattr(StatisticalAnalyzer, "hypothesis_tests", corrupt)
    result = assistant.analyze(_draft(assistant))
    assert result.status == "unavailable"
    assert result.values == {}
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("method_id", ["unknown_method", "spearman_coefficient"])
def test_unknown_or_coefficient_only_selection_cannot_bypass_dispatch(
    monkeypatch, two_groups, method_id
):
    _, assistant = two_groups
    fake = Recommendation(status="ready", method_id=method_id, method_availability="runnable")
    monkeypatch.setattr(execution, "recommend_from_draft", lambda *args: fake)
    result = assistant.analyze(_draft(assistant))
    assert result.status == "unavailable"
    assert result.method_id == method_id
    assert "Unknown selected method" in result.warnings[-1]


def test_skewed_mean_question_does_not_dispatch_rank_test():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "score": [0.1] * 9 + [100.2] + [0.2] * 9 + [200.3],
        }
    )
    assistant = ResearchAssistant(frame)
    result = assistant.analyze(_draft(assistant))
    assert result.status == "available"
    assert result.method_id == "welch_t"
    assert result.specification.question.estimand == "mean"


def test_execution_revalidates_a_draft_from_another_dataset(two_groups):
    _, first_assistant = two_groups
    draft = _draft(first_assistant)
    second = ResearchAssistant(pd.DataFrame({"group": ["A"] * 4 + ["B"] * 4, "score": [1.0] * 8}))
    result = second.analyze(draft)
    assert result.status == "unavailable"
    assert result.recommendation.status == "unsupported"


def test_analyze_requires_exactly_one_valid_request(two_groups):
    _, assistant = two_groups
    with pytest.raises(Exception, match="either one"):
        assistant.analyze()
    with pytest.raises(Exception, match="either one"):
        assistant.analyze(_draft(assistant), specification=_draft(assistant).specification)
    with pytest.raises(Exception, match="QuestionDraft"):
        assistant.analyze("not a draft")


@pytest.mark.parametrize(
    "field,value",
    [
        ("p_value", 2.0),
        ("effect_size", None),
        ("effect_name", None),
        ("sample_size", 99),
        ("group_order", ["A", "B"]),
        ("mean_interval", None),
        ("assumptions", []),
        ("malformed_interval", "not an interval"),
    ],
)
def test_inconsistent_group_backend_payload_is_unavailable(monkeypatch, two_groups, field, value):
    _, assistant = two_groups
    original = StatisticalAnalyzer.hypothesis_tests

    def corrupt(self, *args, **kwargs):
        raw = original(self, *args, **kwargs)
        if field == "effect_name":
            raw["effect_size"]["name"] = value
        elif field == "group_order":
            raw["group_sizes"][0]["group"] = value[0]
        elif field == "mean_interval":
            raw["confidence_interval"] = value
        elif field == "malformed_interval":
            raw["confidence_interval"] = value
        else:
            raw[field] = value
        return raw

    monkeypatch.setattr(StatisticalAnalyzer, "hypothesis_tests", corrupt)
    result = assistant.analyze(_draft(assistant))
    assert result.status == "unavailable"
    assert result.values == {}
    assert result.warnings


def test_pearson_unavailable_inference_and_inconsistent_pair_count(monkeypatch):
    frame = pd.DataFrame({"x": [1.1, 2.2, 3.3, 4.4], "y": [2.1, 3.2, 5.3, 7.4]})
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="linear",
        design="independent",
        variable_types={"x": "continuous", "y": "continuous"},
    )
    original = StatisticalAnalyzer.analyze_all

    def no_inference(self, *args, **kwargs):
        raw = original(self, *args, **kwargs)
        raw["correlation"]["p_values"]["x"]["y"] = None
        return raw

    monkeypatch.setattr(StatisticalAnalyzer, "analyze_all", no_inference)
    assert assistant.analyze(draft).status == "unavailable"

    def wrong_count(self, *args, **kwargs):
        raw = original(self, *args, **kwargs)
        raw["correlation"]["pearson"]["sample_sizes"]["x"]["y"] = 3
        return raw

    monkeypatch.setattr(StatisticalAnalyzer, "analyze_all", wrong_count)
    assert assistant.analyze(draft).status == "unavailable"


def test_categorical_backend_count_mismatch_is_unavailable(monkeypatch):
    frame = pd.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            "response": ["yes"] * 15 + ["no"] * 5 + ["yes"] * 5 + ["no"] * 15,
        }
    )
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="association", outcome="response", predictor="group", design="independent"
    )
    original = StatisticalAnalyzer.categorical_association

    def corrupt(self, *args, **kwargs):
        raw = original(self, *args, **kwargs)
        raw["sample_size"] = 39
        return raw

    monkeypatch.setattr(StatisticalAnalyzer, "categorical_association", corrupt)
    result = assistant.analyze(draft)
    assert result.status == "unavailable"
    assert "counts disagree" in result.warnings[-1]


def test_explicit_legacy_student_and_anova_adapters_preserve_backend_identity():
    two = pd.DataFrame({"group": [0] * 8 + [1] * 8, "score": [float(i) + 0.1 for i in range(16)]})
    assistant = ResearchAssistant(two)
    spec = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
    ).specification
    selected = Recommendation(
        status="ready",
        method_id="student_t",
        method_name="Student t",
        method_availability="runnable",
    )
    student = execution._group_result(assistant._analyzer, spec, selected)
    assert student.method_id == "student_t"
    assert student.metadata["group_order"] == [0, 1]
    assert student.values["degrees_of_freedom"] == 14
    assert student.values["p_value"] == pytest.approx(
        stats.ttest_ind(two.score.iloc[:8], two.score.iloc[8:], equal_var=True).pvalue
    )

    three = pd.DataFrame(
        {"group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5, "score": [float(i) + 0.1 for i in range(15)]}
    )
    multi = ResearchAssistant(three)
    multi_spec = multi.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    ).specification
    anova_rec = Recommendation(
        status="ready",
        method_id="one_way_anova",
        method_name="ANOVA",
        method_availability="runnable",
    )
    anova = execution._group_result(multi._analyzer, multi_spec, anova_rec)
    assert anova.method_id == "one_way_anova"
    assert anova.values["degrees_of_freedom"] == [2.0, 12.0]
    assert anova.values["effect_size"]["name"] == "eta-squared"
    assert anova.metadata["contrast"] is None
