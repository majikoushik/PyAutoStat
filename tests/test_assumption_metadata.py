"""Tests for PR-4 assumption metadata architecture and diagnostic reconciliation."""

from unittest.mock import patch

import pandas as pd

from pyautostat.assumptions import (
    AssessmentMode,
    AssumptionCheck,
    AssumptionStatus,
    reconcile_assumptions,
)
from pyautostat.research_assistant import ResearchAssistant


def _recommend(assistant: ResearchAssistant, **kwargs):
    return assistant.recommend_test(assistant.prepare_question(**kwargs))


def test_recommendation_does_not_execute_inferential_diagnostics():
    """Verify recommendation does not run shapiro, levene, or inferential tests."""
    df = pd.DataFrame(
        {
            "group": ["A"] * 15 + ["B"] * 15,
            "score": [10.0 + i * 0.1 for i in range(15)] + [15.0 + i * 0.1 for i in range(15)],
        }
    )
    assistant = ResearchAssistant(df)

    with patch("scipy.stats.shapiro") as mock_shapiro, patch("scipy.stats.levene") as mock_levene:
        recommendation = _recommend(
            assistant,
            objective="compare_groups",
            outcome="score",
            predictor="group",
            estimand="mean",
            design="independent",
        )
        assert recommendation.status.value == "ready"
        mock_shapiro.assert_not_called()
        mock_levene.assert_not_called()


def test_independence_is_recorded_as_design_fact_not_statistical_proof():
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "score": list(range(20)),
        }
    )
    rec = _recommend(
        ResearchAssistant(df),
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
    )
    indep_checks = [c for c in rec.assumption_checks if "independent" in c.assumption_id]
    assert indep_checks, "Expected an independence assumption check"
    indep = indep_checks[0]
    assert indep.assessment_mode == AssessmentMode.RESEARCHER_DESIGN_FACT
    assert indep.status == AssumptionStatus.CONFIRMED
    assert (
        "not statistically proven" in indep.message.lower()
        or "not verified" in indep.message.lower()
    )


def test_normality_is_marked_post_execution_diagnostic_before_execution():
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "score": list(range(20)),
        }
    )
    rec = _recommend(
        ResearchAssistant(df),
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
    )
    norm_checks = [c for c in rec.assumption_checks if "normality" in c.assumption_id]
    assert norm_checks, "Expected a normality assumption check"
    norm = norm_checks[0]
    assert norm.assessment_mode == AssessmentMode.POST_EXECUTION_DIAGNOSTIC
    assert norm.status == AssumptionStatus.NOT_ASSESSED
    assert "shapiro" in norm.diagnostic_ids


def test_variance_assumption_is_marked_post_execution_diagnostic():
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "score": list(range(20)),
        }
    )
    # Student's t-test has the equal population variance assumption
    draft = ResearchAssistant(df).prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
    )
    ResearchAssistant(df).recommend_test(draft)
    # Even if welch_t is recommended by default, test student_t capability assumption records
    from pyautostat.assumptions import build_recommendation_assumptions
    from pyautostat.recommendation import METHOD_CAPABILITIES

    checks = build_recommendation_assumptions(
        "student_t", draft.specification, METHOD_CAPABILITIES["student_t"]
    )
    var_checks = [
        c for c in checks if "variance" in c.assumption_id or "homogeneity" in c.assumption_id
    ]
    assert var_checks, "Expected a variance homogeneity assumption check for student_t"
    var = var_checks[0]
    assert var.assessment_mode == AssessmentMode.POST_EXECUTION_DIAGNOSTIC
    assert var.status == AssumptionStatus.NOT_ASSESSED


def test_expected_frequency_is_pre_execution_data_check():
    df = pd.DataFrame(
        {
            "cat1": ["X"] * 10 + ["Y"] * 10,
            "cat2": ["A"] * 5 + ["B"] * 5 + ["A"] * 5 + ["B"] * 5,
        }
    )
    rec = _recommend(
        ResearchAssistant(df),
        objective="association",
        outcome="cat1",
        predictor="cat2",
        estimand="categorical_independence",
        design="independent",
        variable_types={"cat1": "nominal", "cat2": "nominal"},
    )
    freq_checks = [
        c
        for c in rec.assumption_checks
        if "expected_cell_counts" in c.assumption_id or "expected_frequencies" in c.assumption_id
    ]
    assert freq_checks, "Expected expected-cell-counts check"
    freq = freq_checks[0]
    assert freq.assessment_mode == AssessmentMode.PRE_EXECUTION_DATA_CHECK
    assert freq.status == AssumptionStatus.CHECKED


def test_execution_diagnostic_links_to_assumption_record():
    df = pd.DataFrame(
        {
            "group": ["A"] * 15 + ["B"] * 15,
            "score": [
                10.0,
                10.5,
                11.0,
                9.5,
                10.2,
                10.8,
                9.8,
                10.1,
                10.4,
                9.9,
                10.3,
                10.7,
                9.7,
                10.0,
                10.2,
            ]
            + [
                15.0,
                15.5,
                14.8,
                15.2,
                14.9,
                15.3,
                15.1,
                14.7,
                15.4,
                15.0,
                14.6,
                15.2,
                15.1,
                14.9,
                15.3,
            ],
        }
    )
    res = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
    )
    assert res.analysis is not None
    checks = res.analysis.assumption_checks
    norm_checks = [c for c in checks if "normality" in c.assumption_id]
    assert norm_checks
    assert norm_checks[0].status in {
        AssumptionStatus.NO_STRONG_EVIDENCE_AGAINST,
        AssumptionStatus.EVIDENCE_AGAINST,
        AssumptionStatus.DIAGNOSTIC_AVAILABLE,
    }


def test_nonsignificant_shapiro_does_not_claim_normality_proven():
    rec_check = AssumptionCheck(
        assumption_id="normality",
        description="Normality of outcome distributions",
        assessment_mode=AssessmentMode.POST_EXECUTION_DIAGNOSTIC,
        status=AssumptionStatus.NOT_ASSESSED,
        diagnostic_ids=("shapiro",),
    )
    diagnostics = {"shapiro": {"p_value": 0.45, "statistic": 0.98, "status": "pass"}}
    reconciled = reconcile_assumptions((rec_check,), diagnostics)
    assert reconciled[0].status == AssumptionStatus.NO_STRONG_EVIDENCE_AGAINST
    assert "normality proven" not in reconciled[0].message.lower()
    assert "normality satisfied" not in reconciled[0].message.lower()
    assert "assumption satisfied" not in reconciled[0].message.lower()
    assert "did not detect strong evidence against" in reconciled[0].message.lower()


def test_nonsignificant_levene_does_not_claim_equal_variance_proven():
    rec_check = AssumptionCheck(
        assumption_id="homogeneity_of_variance",
        description="Homogeneity of variance across groups",
        assessment_mode=AssessmentMode.POST_EXECUTION_DIAGNOSTIC,
        status=AssumptionStatus.NOT_ASSESSED,
        diagnostic_ids=("levene",),
    )
    diagnostics = {"levene": {"p_value": 0.32, "statistic": 1.15, "status": "pass"}}
    reconciled = reconcile_assumptions((rec_check,), diagnostics)
    assert reconciled[0].status == AssumptionStatus.NO_STRONG_EVIDENCE_AGAINST
    assert "equal variance proven" not in reconciled[0].message.lower()
    assert "assumption proven" not in reconciled[0].message.lower()
    assert "assumption satisfied" not in reconciled[0].message.lower()
    assert "did not detect strong evidence against" in reconciled[0].message.lower()


def test_significant_diagnostic_records_evidence_against_without_method_switch():
    df = pd.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            # Group A has extreme skew / outlier to trigger shapiro rejection
            "score": [1.0] * 19 + [100.0] + [5.0 + i * 0.1 for i in range(20)],
        }
    )
    wf = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
    )
    # Method should still be welch_t (the originally recommended method, no silent switch)
    assert wf.analysis.method_id == "welch_t"
    checks = wf.analysis.assumption_checks
    norm_checks = [c for c in checks if "normality" in c.assumption_id]
    assert norm_checks
    assert norm_checks[0].status == AssumptionStatus.EVIDENCE_AGAINST


def test_repeated_measures_sphericity_metadata_links_to_mauchly():
    rec_check = AssumptionCheck(
        assumption_id="sphericity",
        description="Sphericity (equal variances of pairwise differences)",
        assessment_mode=AssessmentMode.POST_EXECUTION_DIAGNOSTIC,
        status=AssumptionStatus.NOT_ASSESSED,
        diagnostic_ids=("mauchly",),
    )
    diagnostics = {"sphericity": {"p_value": 0.02, "statistic": 0.75, "status": "violated"}}
    reconciled = reconcile_assumptions((rec_check,), diagnostics)
    assert reconciled[0].status == AssumptionStatus.EVIDENCE_AGAINST
    assert "evidence against sphericity" in reconciled[0].message.lower()


def test_assumption_records_serialize_deterministically():
    check = AssumptionCheck(
        assumption_id="normality",
        description="Normality",
        assessment_mode=AssessmentMode.POST_EXECUTION_DIAGNOSTIC,
        status=AssumptionStatus.NOT_ASSESSED,
        diagnostic_ids=("shapiro",),
        message="Pending execution diagnostics.",
    )
    d = check.to_dict()
    assert d["schema_version"] == 1
    assert d["assumption_id"] == "normality"
    assert d["assessment_mode"] == "post_execution_diagnostic"
    assert d["status"] == "not_assessed"
    assert d["diagnostic_ids"] == ["shapiro"]
    assert d["message"] == "Pending execution diagnostics."


def test_existing_recommendation_context_compatibility():
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "score": list(range(20)),
        }
    )
    rec = _recommend(
        ResearchAssistant(df),
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
    )
    assert "assumption_checks" in rec.context
    legacy = rec.context["assumption_checks"]
    assert isinstance(legacy, (list, tuple))
    assert len(legacy) > 0
    for item in legacy:
        assert isinstance(item, dict)
        assert item["status"] in {
            "confirmed",
            "unresolved",
            "checked",
            "not_assessed",
            "requires_review",
            "not_applicable",
        }
