"""Phase 5 decision scenarios; expected methods come from the published capability contract."""

import json

import pandas as pd
import pytest

from pyautostat import (
    AnalysisSpecification,
    ColumnNotFoundError,
    InvalidDataError,
    ResearchAssistant,
    ResearchQuestion,
    StatisticalAnalyzer,
)
from pyautostat.recommendation import METHOD_CAPABILITIES


@pytest.fixture
def comparison():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "score": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 3.1, 4.2, 5.3, 6.4, 7.5, 8.6],
        }
    )
    return ResearchAssistant(frame)


def question(assistant, **kwargs):
    return assistant.recommend_test(assistant.prepare_question(**kwargs))


def test_registry_matches_real_backend_capabilities():
    assert METHOD_CAPABILITIES["welch_t"].backend.endswith("equal_var=False)")
    assert METHOD_CAPABILITIES["student_t"].backend.endswith("equal_var=True)")
    assert METHOD_CAPABILITIES["one_way_anova"].availability == "runnable"
    assert METHOD_CAPABILITIES["pearson_correlation"].inferential
    for method_id in ("spearman_coefficient", "kendall_coefficient"):
        assert METHOD_CAPABILITIES[method_id].availability == "coefficient_only"
        assert not METHOD_CAPABILITIES[method_id].inferential
    assert "paired_t" not in METHOD_CAPABILITIES
    assert "welch_anova" not in METHOD_CAPABILITIES


def test_descriptive_recommends_profile_without_inference(comparison):
    result = question(comparison, objective="descriptive")
    assert result.status == "ready"
    assert result.method_id == "dataset_profile"
    assert result.method_availability == "runnable"
    assert not METHOD_CAPABILITIES[result.method_id].inferential
    assert result.context["design"] == "unknown"


def test_two_group_mean_targets_welch_and_explains_alternatives(comparison):
    result = question(
        comparison,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )
    assert result.status == "ready"
    assert result.method_id == "welch_t"
    assert result.method_availability == "runnable"
    assert result.context["estimand"] == "mean"
    assert result.context["group_sizes"] == [6, 6]
    assert "means" in result.rationale
    assert {item["method_id"] for item in result.alternatives} == {"student_t", "mann_whitney_u"}
    assert any(
        item["key"] == "method" and item["value"] == "welch_t" for item in result.decision_trace
    )
    assert result.context["assumption_checks"][0]["status"] == "confirmed"
    assert result.context["assumption_checks"][1]["status"] == "requires_review"


def test_skewed_data_do_not_change_a_mean_target():
    assistant = ResearchAssistant(
        pd.DataFrame(
            {
                "group": ["A"] * 10 + ["B"] * 10,
                "score": [0.1] * 9 + [100.2] + [0.2] * 9 + [200.3],
            }
        )
    )
    result = question(
        assistant,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
    )
    assert result.status == "ready"
    assert result.method_id == "welch_t"
    assert "mann_whitney_u" in {item["method_id"] for item in result.alternatives}


def test_two_group_distribution_uses_mann_whitney(comparison):
    result = question(
        comparison,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="distribution",
        design="independent",
    )
    assert result.status == "ready"
    assert result.method_id == "mann_whitney_u"
    assert "rank distributions" in result.rationale
    assert "median" not in result.rationale


@pytest.mark.parametrize("design", ["paired", "repeated", "clustered"])
def test_dependent_designs_block_independent_methods(comparison, design):
    result = question(
        comparison,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design=design,
    )
    assert result.status == "unsupported"
    assert result.method_id is None
    assert design in result.blockers[0]
    assert "independent" in result.blockers[0]


def test_unknown_design_preserves_phase4_question(comparison):
    draft = comparison.prepare_question(
        objective="compare_groups", outcome="score", predictor="group", estimand="mean"
    )
    result = comparison.recommend_test(draft)
    assert result.status == "needs_input"
    assert [item.field for item in result.missing_information] == ["design"]
    assert result.questions[0]["field"] == "design"
    assert result.method_id is None


def test_three_group_mean_does_not_switch_estimand():
    assistant = ResearchAssistant(
        pd.DataFrame(
            {
                "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
                "score": [float(i) + 0.1 for i in range(15)],
            }
        )
    )
    result = question(
        assistant,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
    )
    assert result.status == "unsupported"
    assert result.method_id is None
    assert "variance" in result.blockers[0]
    alternatives = {item["method_id"]: item for item in result.alternatives}
    assert alternatives["one_way_anova"]["availability"] == "runnable"
    assert "assumptions" in alternatives["one_way_anova"]["reason"]
    assert "kruskal_wallis" in alternatives
    assert result.context["estimand"] == "mean"


def test_three_group_distribution_uses_kruskal():
    assistant = ResearchAssistant(
        pd.DataFrame(
            {
                "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
                "score": [float(i) + 0.1 for i in range(15)],
            }
        )
    )
    result = question(
        assistant,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="distribution",
        design="independent",
    )
    assert result.status == "ready"
    assert result.method_id == "kruskal_wallis"
    assert "which groups" in result.rationale


def test_kruskal_minimum_follows_existing_backend():
    assistant = ResearchAssistant(
        pd.DataFrame(
            {
                "group": ["A"] * 4 + ["B"] * 5 + ["C"] * 5,
                "score": [float(i) + 0.1 for i in range(14)],
            }
        )
    )
    result = question(
        assistant,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="distribution",
        design="independent",
    )
    assert result.status == "unsupported"
    assert "five usable" in result.blockers[0]


@pytest.fixture
def numerical_association():
    return ResearchAssistant(
        pd.DataFrame(
            {
                "hours": [1.1, 2.2, 3.3, 4.4, 5.5],
                "score": [3.2, 4.1, 8.4, 7.1, 9.3],
            }
        )
    )


def test_linear_association_recommends_pearson(numerical_association):
    draft = numerical_association.prepare_question(
        objective="association",
        outcome="hours",
        predictor="score",
        estimand="linear",
        design="independent",
    )
    result = numerical_association.recommend_test(draft)
    assert result.status == "ready"
    assert result.method_id == "pearson_correlation"
    assert result.context["complete_pairs"] == 5
    assert result.method_availability == "runnable"


def test_unspecified_numerical_target_asks_one_question(numerical_association):
    result = question(
        numerical_association,
        objective="association",
        outcome="hours",
        predictor="score",
        design="independent",
    )
    assert result.status == "needs_input"
    assert [item.field for item in result.missing_information] == ["estimand"]
    assert [item["value"] for item in result.questions[0]["options"]] == [
        "linear",
        "monotonic",
        "unknown",
    ]


def test_monotonic_target_discloses_coefficient_only(numerical_association):
    result = question(
        numerical_association,
        objective="association",
        outcome="hours",
        predictor="score",
        estimand="monotonic",
        design="independent",
    )
    assert result.status == "unsupported"
    assert result.method_id is None
    assert {item["method_id"] for item in result.alternatives} == {
        "spearman_coefficient",
        "kendall_coefficient",
    }
    assert all(item["availability"] == "coefficient_only" for item in result.alternatives)
    assert "p-values" in result.blockers[0]


def test_pearson_requires_three_complete_varying_pairs():
    assistant = ResearchAssistant(pd.DataFrame({"x": [1.1, 2.2, None], "y": [2.1, 3.2, 4.3]}))
    result = question(
        assistant,
        objective="association",
        outcome="x",
        predictor="y",
        estimand="linear",
        design="independent",
    )
    assert result.status == "unsupported"
    assert "three complete" in result.blockers[0]


def test_constant_numeric_pair_is_blocked():
    assistant = ResearchAssistant(
        pd.DataFrame({"x": [1.1, 1.1, 1.1, 1.1], "y": [2.1, 3.2, 4.3, 5.4]})
    )
    result = question(
        assistant,
        objective="association",
        outcome="x",
        predictor="y",
        estimand="linear",
        design="independent",
    )
    assert result.status == "unsupported"
    assert "variation" in result.blockers[0]


def test_categorical_association_expected_counts_and_sparse_block():
    adequate = ResearchAssistant(
        pd.DataFrame(
            {
                "method": ["A"] * 20 + ["B"] * 20,
                "response": (["yes"] * 10 + ["no"] * 10) * 2,
            }
        )
    )
    result = question(
        adequate,
        objective="association",
        outcome="response",
        predictor="method",
        design="independent",
    )
    assert result.status == "ready"
    assert result.method_id == "pearson_chi_square"
    assert result.context["contingency_shape"] == [2, 2]
    assert result.context["minimum_expected_count"] == 10
    sparse = ResearchAssistant(
        pd.DataFrame(
            {
                "method": ["A"] * 4 + ["B"] * 4,
                "response": (["yes"] * 2 + ["no"] * 2) * 2,
            }
        )
    )
    blocked = question(
        sparse,
        objective="association",
        outcome="response",
        predictor="method",
        design="independent",
    )
    assert blocked.status == "unsupported"
    assert blocked.method_id is None
    assert blocked.context["minimum_expected_count"] == 2
    assert blocked.alternatives[0]["availability"] == "not_implemented"


def test_identifier_and_unordered_outcome_not_treated_as_quantitative():
    assistant = ResearchAssistant(
        pd.DataFrame(
            {
                "student_id": [101, 102, 103, 104],
                "group": ["A", "A", "B", "B"],
            }
        )
    )
    result = question(
        assistant,
        objective="compare_groups",
        outcome="student_id",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"student_id": "identifier"},
    )
    assert result.status == "unsupported"
    assert "identifier" in result.blockers[0]
    nominal = ResearchAssistant(
        pd.DataFrame({"group": ["A", "A", "B", "B"], "response": ["yes", "no", "yes", "no"]})
    )
    result = question(
        nominal,
        objective="compare_groups",
        outcome="response",
        predictor="group",
        estimand="distribution",
        design="independent",
    )
    assert result.status == "unsupported"
    assert "ordered numeric" in result.blockers[0]


def test_numeric_ordinal_can_be_ranked_but_ordered_labels_are_not_recoded():
    numeric = ResearchAssistant(
        pd.DataFrame(
            {
                "rating": [1, 2, 3, 1, 2, 3],
                "group": ["A"] * 3 + ["B"] * 3,
            }
        )
    )
    result = question(
        numeric,
        objective="compare_groups",
        outcome="rating",
        predictor="group",
        estimand="distribution",
        design="independent",
        variable_types={"rating": "ordinal"},
    )
    assert result.status == "ready"
    assert result.method_id == "mann_whitney_u"
    ordered = ResearchAssistant(
        pd.DataFrame(
            {
                "rating": pd.Categorical(
                    ["low", "mid", "high", "low", "mid", "high"], ordered=True
                ),
                "group": ["A"] * 3 + ["B"] * 3,
            }
        )
    )
    result = question(
        ordered,
        objective="compare_groups",
        outcome="rating",
        predictor="group",
        estimand="distribution",
        design="independent",
    )
    assert result.status == "unsupported"
    assert "unencoded ordered categories" in result.blockers[0]


def test_one_usable_outcome_group_is_blocked():
    assistant = ResearchAssistant(
        pd.DataFrame({"group": ["A", "A", "B", "B"], "score": [1.1, 2.2, 3.3, None]})
    )
    result = question(
        assistant,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
    )
    assert result.status == "unsupported"
    assert "two usable" in result.blockers[0]
    assert result.context["availability"]["available_rows"] == 3


def test_phase4_data_limit_is_preserved():
    assistant = ResearchAssistant(
        pd.DataFrame({"group": ["A", "A", "A"], "score": [1.1, 2.2, 3.3]})
    )
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
    )
    assert draft.status == "data_limited"
    result = assistant.recommend_test(draft)
    assert result.status == "unsupported"
    assert result.blockers == draft.blockers
    assert result.method_id is None


def test_overlapping_missing_rows_use_complete_pairs():
    assistant = ResearchAssistant(
        pd.DataFrame(
            {
                "x": [1.1, None, None, 4.4, 5.5],
                "y": [2.2, None, 3.3, 5.5, 6.6],
            }
        )
    )
    result = question(
        assistant,
        objective="association",
        outcome="x",
        predictor="y",
        estimand="linear",
        design="independent",
    )
    assert result.context["complete_pairs"] == 3
    assert result.context["availability"]["excluded_rows"] == 2


def test_unapplied_missing_codes_block_final_recommendation(comparison):
    result = question(
        comparison,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        data_dictionary={"score": {"type": "continuous", "missing_codes": [1.1]}},
    )
    assert result.status == "unsupported"
    assert "declared missing-code" in result.blockers[0]
    assert result.context["availability"]["available_rows"] == 12


def test_direct_specification_is_revalidated(comparison):
    spec = AnalysisSpecification(
        question=ResearchQuestion(
            objective="compare_groups", outcome="score", predictor="group", estimand="mean"
        ),
        design="independent",
    )
    assert comparison.recommend_test(specification=spec).method_id == "welch_t"
    bad = AnalysisSpecification(
        question=ResearchQuestion(
            objective="compare_groups", outcome="absent", predictor="group", estimand="mean"
        ),
        design="independent",
    )
    with pytest.raises(ColumnNotFoundError, match="absent"):
        comparison.recommend_test(specification=bad)
    with pytest.raises(InvalidDataError, match="either one"):
        comparison.recommend_test()


def test_mixed_types_request_objective_clarification():
    assistant = ResearchAssistant(
        pd.DataFrame({"score": [1.1, 2.2, 3.3, 4.4], "group": ["A", "A", "B", "B"]})
    )
    result = question(
        assistant,
        objective="association",
        outcome="score",
        predictor="group",
        design="independent",
    )
    assert result.status == "needs_input"
    assert [item.field for item in result.missing_information] == ["objective"]
    assert result.questions[0]["options"][0]["value"] == "compare_groups"
    assert result.method_id is None


def test_deterministic_json_and_unmodified_input(comparison):
    frame = comparison._analyzer.df.copy(deep=True)
    draft = comparison.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
    )
    first = comparison.recommend_test(draft).to_dict()
    second = comparison.recommend_test(draft).to_dict()
    assert first == second
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    assert "p_value" not in first
    pd.testing.assert_frame_equal(comparison._analyzer.df, frame)


def test_recommendation_never_executes_existing_tests(monkeypatch, comparison):
    def forbidden(*args, **kwargs):
        raise AssertionError("Statistical execution was called")

    monkeypatch.setattr(StatisticalAnalyzer, "hypothesis_tests", forbidden)
    monkeypatch.setattr(StatisticalAnalyzer, "categorical_association", forbidden)
    monkeypatch.setattr(StatisticalAnalyzer, "analyze_all", forbidden)
    assert (
        question(
            comparison,
            objective="compare_groups",
            outcome="score",
            predictor="group",
            estimand="mean",
            design="independent",
        ).method_id
        == "welch_t"
    )


def test_existing_public_apis_still_work(comparison):
    assert comparison.complete_case_count(["score", "group"])["available_rows"] == 12
    assert comparison.profile()["overview"]["total_rows"] == 12
    result = StatisticalAnalyzer(comparison._analyzer.df).hypothesis_tests(
        "group", "score", test_type="auto", estimand="mean", bootstrap_samples=0
    )
    assert result["test"] == "t-test"


def test_unsupported_targets_are_preserved(comparison, numerical_association):
    grouped = question(
        comparison,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="median",
        design="independent",
    )
    assert grouped.status == "unsupported"
    assert "median" in grouped.blockers[0]
    assert grouped.context["estimand"] == "median"
    association = question(
        numerical_association,
        objective="association",
        outcome="hours",
        predictor="score",
        estimand="causal_effect",
        design="independent",
    )
    assert association.status == "unsupported"
    assert "causal_effect" in association.blockers[0]


def test_mean_rejects_declared_ordinal_measurement(comparison):
    result = question(
        comparison,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "ordinal"},
    )
    assert result.status == "unsupported"
    assert "quantitative" in result.blockers[0]


def test_constant_rank_outcome_and_one_level_categorical_axis_are_blocked():
    grouped = ResearchAssistant(
        pd.DataFrame({"group": ["A"] * 3 + ["B"] * 3, "rating": [1, 1, 1, 1, 1, 1]})
    )
    ranked = question(
        grouped,
        objective="compare_groups",
        outcome="rating",
        predictor="group",
        estimand="distribution",
        design="independent",
        variable_types={"rating": "ordinal"},
    )
    assert ranked.status == "unsupported"
    assert "no observed variation" in ranked.blockers[0]
    categorical = ResearchAssistant(pd.DataFrame({"x": ["a"] * 12, "y": ["u"] * 6 + ["v"] * 6}))
    result = question(
        categorical,
        objective="association",
        outcome="x",
        predictor="y",
        design="independent",
    )
    assert result.status == "unsupported"
    assert "two observed categories" in result.blockers[0]


def test_ordinal_numeric_association_keeps_its_scale():
    assistant = ResearchAssistant(pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 3, 4, 5, 1]}))
    declarations = {"x": "ordinal", "y": "ordinal"}
    monotonic = question(
        assistant,
        objective="association",
        outcome="x",
        predictor="y",
        estimand="monotonic",
        design="independent",
        variable_types=declarations,
    )
    assert monotonic.status == "unsupported"
    assert {item["availability"] for item in monotonic.alternatives} == {"coefficient_only"}
    linear = question(
        assistant,
        objective="association",
        outcome="x",
        predictor="y",
        estimand="linear",
        design="independent",
        variable_types=declarations,
    )
    assert linear.status == "unsupported"
    assert "ordinal" in linear.blockers[0]


def test_extreme_mean_spread_and_unhandled_type_pair_are_blocked():
    extreme = ResearchAssistant(
        pd.DataFrame({"group": ["A", "A", "B", "B"], "score": [-1e308, 1e308, 1.0, 2.0]})
    )
    result = question(
        extreme,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )
    assert result.status == "unsupported"
    assert "not representable" in result.blockers[0]
    dates = ResearchAssistant(
        pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=6),
                "category": ["a", "a", "a", "b", "b", "b"],
            }
        )
    )
    result = question(
        dates,
        objective="association",
        outcome="date",
        predictor="category",
        design="independent",
    )
    assert result.status == "unsupported"
    assert "no compatible" in result.blockers[0]


def test_datetime_group_is_rejected_and_ordinal_chi_square_discloses_lost_order():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"] * 4 + ["2025-01-02"] * 4),
            "score": [float(i) for i in range(8)],
        }
    )
    grouped = question(
        ResearchAssistant(frame),
        objective="compare_groups",
        outcome="score",
        predictor="date",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )
    assert grouped.status == "unsupported"
    assert "datetime" in grouped.blockers[0]

    categorical = ResearchAssistant(
        pd.DataFrame(
            {
                "rating": (["low"] * 10 + ["high"] * 10) * 2,
                "group": ["A"] * 20 + ["B"] * 20,
            }
        )
    )
    associated = question(
        categorical,
        objective="association",
        outcome="rating",
        predictor="group",
        design="independent",
        variable_types={"rating": "ordinal", "group": "nominal"},
    )
    assert associated.method_id == "pearson_chi_square"
    assert any("ordered categories" in warning for warning in associated.warnings)
