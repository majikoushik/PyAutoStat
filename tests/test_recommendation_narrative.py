"""Researcher-readable explanations of existing recommendation records."""

import pandas as pd

from pyautostat import ResearchAssistant, recommendation_rationale


def _recommend(frame, **question):
    assistant = ResearchAssistant(frame)
    return assistant.recommend_test(assistant.prepare_question(**question))


def test_welch_rationale_preserves_mean_estimand_and_serialized_record():
    recommendation = _recommend(
        pd.DataFrame(
            {
                "group": ["A"] * 6 + ["B"] * 6,
                "score": [1.0, 2, 3, 4, 5, 6, 3.0, 4, 5, 6, 7, 8],
            }
        ),
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )
    serialized = recommendation.to_dict()

    text = recommendation.rationale_text

    assert recommendation.method_id == "welch_t"
    assert "RECOMMENDED TEST:" in text
    assert "WHY THIS TEST?" in text
    assert "WHY NOT STUDENT'S T-TEST?" in text
    assert "WHY NOT MANN-WHITNEY?" in text
    assert "rank distributions" in text
    assert "WHAT YOU NEED TO VERIFY" in text
    assert "independence" in text
    assert "representativeness" in text
    assert "Levene" not in text
    assert recommendation.to_dict() == serialized
    assert "rationale_text" not in serialized


def test_welch_mentions_only_explicitly_supplied_variance_diagnostic():
    frame = pd.DataFrame(
        {"group": ["A"] * 5 + ["B"] * 5, "score": [1, 2, 3, 4, 5, 8, 9, 12, 16, 22]}
    )
    recommendation = _recommend(
        frame,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )

    without = recommendation.explain()
    with_diagnostic = recommendation.explain(
        diagnostics={"equal_variance_status": "rejected", "levene_p_value": 0.003}
    )

    assert "Levene" not in without
    assert "Levene p = 0.003" in with_diagnostic
    assert "rejected" in with_diagnostic
    assert "0.003" not in without


def test_rank_recommendations_explain_different_estimands_and_group_scope():
    two_group = pd.DataFrame({"group": ["A"] * 6 + ["B"] * 6, "score": list(range(1, 13))})
    mann = _recommend(
        two_group,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="distribution",
        design="independent",
    )
    three_group = pd.DataFrame(
        {"group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5, "score": list(range(15))}
    )
    kruskal = _recommend(
        three_group,
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="distribution",
        design="independent",
    )

    assert mann.method_id == "mann_whitney_u"
    assert "ordered distribution estimand" in mann.rationale_text
    assert "WHY NOT WELCH'S T-TEST?" in mann.rationale_text
    assert "means" in mann.rationale_text
    assert "median difference" in mann.rationale_text
    assert kruskal.method_id == "kruskal_wallis"
    assert "3 independent groups" in kruskal.rationale_text
    assert "WHY NOT ONE-WAY ANOVA?" in kruskal.rationale_text
    assert "Dunn-Holm supplies every" in kruskal.rationale_text


def test_paired_rationale_requires_identity_order_and_does_not_equate_independent_tests():
    frame = pd.DataFrame(
        {
            "participant": [1, 1, 2, 2, 3, 3, 4, 4],
            "condition": ["before", "after"] * 4,
            "score": [10.0, 8.0, 9.0, 8.0, 12.0, 9.0, 8.0, 6.0],
        }
    )
    recommendation = _recommend(
        frame,
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        estimand="mean",
        design="paired",
        unit_id="participant",
        condition_order=("before", "after"),
        variable_types={"score": "continuous"},
    )
    text = recommendation.rationale_text

    assert recommendation.method_id == "paired_t"
    assert "within-unit difference" in text
    assert "'before' minus 'after'" in text
    assert "'participant' correctly identifies" in text
    assert "WHY NOT AN INDEPENDENT-SAMPLES TEST?" in text
    assert "different sampling structure" in text


def test_categorical_and_linear_association_rationales_are_context_specific():
    categorical = pd.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            "response": (["yes"] * 10 + ["no"] * 10) * 2,
        }
    )
    chi = _recommend(
        categorical,
        objective="association",
        outcome="response",
        predictor="group",
        design="independent",
    )
    numeric = pd.DataFrame({"x": [1.0, 2, 3, 4, 5], "y": [2.0, 4, 3, 7, 8]})
    pearson = _recommend(
        numeric,
        objective="association",
        outcome="x",
        predictor="y",
        estimand="linear",
        design="independent",
        variable_types={"x": "continuous", "y": "continuous"},
    )

    assert chi.method_id == "pearson_chi_square"
    assert "categorical independence" in chi.rationale_text
    assert "minimum expected count of 10" in chi.rationale_text
    assert "WHY NOT PEARSON CORRELATION?" in chi.rationale_text
    assert "T-TEST" not in chi.rationale_text
    assert pearson.method_id == "pearson_correlation"
    assert "linear association" in pearson.rationale_text
    assert "WHY NOT SPEARMAN CORRELATION?" in pearson.rationale_text
    assert "does not establish causation" in pearson.rationale_text
    assert "caused" not in pearson.rationale_text


def test_descriptive_and_not_ready_explanations_are_safe():
    assistant = ResearchAssistant(pd.DataFrame({"score": [1.0, 2.0, 3.0]}))
    descriptive = assistant.recommend_test(assistant.prepare_question(objective="descriptive"))
    unresolved = assistant.recommend_test(
        assistant.prepare_question(
            objective="compare_groups",
            outcome="score",
            predictor=None,
            estimand="mean",
        )
    )

    assert descriptive.method_id == "dataset_profile"
    assert "WHY NOT AN INFERENTIAL TEST?" in descriptive.rationale_text
    assert "missing-value codes" in descriptive.rationale_text
    assert recommendation_rationale(unresolved).startswith("RECOMMENDATION NOT READY")
