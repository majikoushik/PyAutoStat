"""Comprehensive tests for Phase 7 Repeated-Measures Analysis Workflow."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyautostat import (
    InvalidDataError,
    ResearchAssistant,
    StatisticalAnalyzer,
    StatisticalResultAuditor,
    assess_reporting_completeness,
    reproduce,
)
from pyautostat.analysis_plan import StatisticalAnalysisPlan, compare_plan_to_result
from pyautostat.repeated_measures import (
    friedman_test,
    repeated_measures_anova,
    repeated_panel,
)
from pyautostat.session import build_session_snapshot

# ---------------------------------------------------------------------------
# Fixtures & Reference Datasets
# ---------------------------------------------------------------------------


@pytest.fixture
def balanced_panel_df():
    """A balanced 3-condition repeated-measures dataset with 8 participants."""
    # Condition means: baseline ~ 50, week4 ~ 55, week8 ~ 62
    rng = np.random.default_rng(42)
    participants = []
    conditions = []
    scores = []
    for pid in range(1, 9):
        base = 50.0 + rng.normal(0, 4)
        for cond, delta in [("baseline", 0.0), ("week4", 5.0), ("week8", 12.0)]:
            participants.append(pid)
            conditions.append(cond)
            scores.append(round(base + delta + float(rng.normal(0, 1.5)), 2))
    return pd.DataFrame({"participant": participants, "condition": conditions, "score": scores})


@pytest.fixture
def non_spherical_df():
    """Dataset with strong sphericity violation across 4 conditions."""
    # Constructed with non-spherical covariance
    # Covariance structure where variances of pairwise differences differ greatly
    # e.g., T1 and T2 highly correlated, T1 and T4 weakly correlated with high variance
    np.random.seed(123)
    n = 20
    # True covariance matrix with heterogeneous variances and varying covariances
    cov = np.array(
        [
            [1.0, 0.9, 0.2, 0.1],
            [0.9, 2.0, 0.3, 0.2],
            [0.2, 0.3, 8.0, 0.4],
            [0.1, 0.2, 0.4, 15.0],
        ]
    )
    means = [10.0, 14.0, 18.0, 25.0]
    data = np.random.multivariate_normal(means, cov, size=n)
    rows = []
    for i in range(n):
        for j, cond in enumerate(["T1", "T2", "T3", "T4"]):
            rows.append({"subject": i + 1, "time": cond, "val": round(data[i, j], 3)})
    return pd.DataFrame(rows)


@pytest.fixture
def spherical_df():
    """Dataset with compound symmetry (satisfies sphericity)."""
    np.random.seed(456)
    n = 25
    # Equal variances and equal covariances
    cov = np.array(
        [
            [4.0, 2.5, 2.5],
            [2.5, 4.0, 2.5],
            [2.5, 2.5, 4.0],
        ]
    )
    means = [20.0, 22.0, 26.0]
    data = np.random.multivariate_normal(means, cov, size=n)
    rows = []
    for i in range(n):
        for j, cond in enumerate(["condA", "condB", "condC"]):
            rows.append({"subject": i + 1, "cond": cond, "val": round(data[i, j], 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Panel Builder Tests
# ---------------------------------------------------------------------------


def test_repeated_panel_balanced(balanced_panel_df):
    panel = repeated_panel(
        balanced_panel_df,
        unit_id="participant",
        condition_col="condition",
        value_col="score",
        condition_order=("baseline", "week4", "week8"),
    )
    assert panel["total_units"] == 8
    assert panel["complete_units"] == 8
    assert panel["incomplete_units"] == 0
    assert panel["excluded_rows"] == 0
    assert panel["analyzed_rows"] == 24
    assert panel["panel"].shape == (8, 3)
    assert len(panel["condition_summaries"]) == 3
    assert [s["condition"] for s in panel["condition_summaries"]] == ["baseline", "week4", "week8"]


def test_repeated_panel_shuffled_row_order(balanced_panel_df):
    shuffled = balanced_panel_df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    panel = repeated_panel(
        shuffled,
        unit_id="participant",
        condition_col="condition",
        value_col="score",
        condition_order=("baseline", "week4", "week8"),
    )
    assert panel["complete_units"] == 8
    # Matrix columns must strictly match condition_order
    ordered_panel = repeated_panel(
        balanced_panel_df,
        unit_id="participant",
        condition_col="condition",
        value_col="score",
        condition_order=("baseline", "week4", "week8"),
    )
    assert np.allclose(panel["panel"], ordered_panel["panel"])


def test_repeated_panel_explicit_nonalphabetical_order(balanced_panel_df):
    panel = repeated_panel(
        balanced_panel_df,
        unit_id="participant",
        condition_col="condition",
        value_col="score",
        condition_order=("week8", "baseline", "week4"),
    )
    assert [s["condition"] for s in panel["condition_summaries"]] == ["week8", "baseline", "week4"]
    # Check that first column corresponds to week8
    w8_vals = (
        balanced_panel_df[balanced_panel_df["condition"] == "week8"]
        .sort_values("participant")["score"]
        .values
    )
    assert np.allclose(panel["panel"][:, 0], w8_vals)


def test_repeated_panel_missing_repeated_observation(balanced_panel_df):
    # Remove one observation for participant 2
    drop_idx = balanced_panel_df[
        (balanced_panel_df["participant"] == 2) & (balanced_panel_df["condition"] == "week8")
    ].index
    modified = balanced_panel_df.drop(index=drop_idx)
    panel = repeated_panel(
        modified,
        unit_id="participant",
        condition_col="condition",
        value_col="score",
        condition_order=("baseline", "week4", "week8"),
    )
    assert panel["total_units"] == 8
    assert panel["complete_units"] == 7
    assert panel["incomplete_units"] == 1
    # Analyzed rows = 7 complete units * 3 conditions = 21
    assert panel["analyzed_rows"] == 21
    assert panel["panel"].shape == (7, 3)


def test_repeated_panel_missing_unit_id(balanced_panel_df):
    modified = balanced_panel_df.copy()
    modified.loc[0, "participant"] = None
    panel = repeated_panel(
        modified,
        unit_id="participant",
        condition_col="condition",
        value_col="score",
        condition_order=("baseline", "week4", "week8"),
    )
    assert panel["missing_unit_rows"] == 1
    # Participant 1 is now incomplete because one observation had null unit_id
    assert panel["complete_units"] == 7


def test_repeated_panel_duplicate_unit_condition_blocks(balanced_panel_df):
    # Add duplicate row for participant 1 at baseline
    dup_row = balanced_panel_df.iloc[[0]].copy()
    with_dup = pd.concat([balanced_panel_df, dup_row], ignore_index=True)
    with pytest.raises(
        InvalidDataError, match="one usable outcome per unit per declared condition"
    ):
        repeated_panel(
            with_dup,
            unit_id="participant",
            condition_col="condition",
            value_col="score",
            condition_order=("baseline", "week4", "week8"),
        )


def test_repeated_panel_requires_at_least_three_conditions(balanced_panel_df):
    with pytest.raises(InvalidDataError, match="at least 3 declared conditions"):
        repeated_panel(
            balanced_panel_df,
            unit_id="participant",
            condition_col="condition",
            value_col="score",
            condition_order=("baseline", "week4"),
        )


def test_repeated_panel_source_df_not_mutated(balanced_panel_df):
    orig_copy = balanced_panel_df.copy(deep=True)
    repeated_panel(
        balanced_panel_df,
        unit_id="participant",
        condition_col="condition",
        value_col="score",
        condition_order=("baseline", "week4", "week8"),
    )
    pd.testing.assert_frame_equal(balanced_panel_df, orig_copy)


# ---------------------------------------------------------------------------
# 2. Friedman Test & Kendall's W
# ---------------------------------------------------------------------------


def test_friedman_test_against_scipy(balanced_panel_df):
    panel = repeated_panel(
        balanced_panel_df,
        unit_id="participant",
        condition_col="condition",
        value_col="score",
        condition_order=("baseline", "week4", "week8"),
    )
    # Scipy reference
    ref = stats.friedmanchisquare(panel["panel"][:, 0], panel["panel"][:, 1], panel["panel"][:, 2])
    res = friedman_test(
        balanced_panel_df, "participant", "condition", "score", ("baseline", "week4", "week8")
    )

    assert res["statistic"] == pytest.approx(float(ref.statistic))
    assert res["p_value"] == pytest.approx(float(ref.pvalue))
    assert res["degrees_of_freedom"] == 2
    assert res["conditions"] == ["baseline", "week4", "week8"]
    assert res["complete_units"] == 8

    # Kendall's W = Q / (n * (k - 1)) = Q / (8 * 2) = Q / 16
    expected_w = float(ref.statistic) / (8 * 2)
    assert res["effect_size"]["value"] == pytest.approx(expected_w)
    assert 0.0 <= res["effect_size"]["value"] <= 1.0


def test_friedman_pairwise_wilcoxon_and_holm(balanced_panel_df):
    res = friedman_test(
        balanced_panel_df, "participant", "condition", "score", ("baseline", "week4", "week8")
    )
    pairwise = res["pairwise_comparisons"]
    # 3 conditions -> 3 * 2 / 2 = 3 pairs
    assert len(pairwise) == 3
    pairs = [(p["first_condition"], p["second_condition"]) for p in pairwise]
    assert pairs == [("baseline", "week4"), ("baseline", "week8"), ("week4", "week8")]

    # Check Holm adjustment properties
    raw_p = [p["raw_p_value"] for p in pairwise]
    adj_p = [p["adjusted_p_value"] for p in pairwise]
    for r, a in zip(raw_p, adj_p, strict=True):
        assert a >= r
        assert a <= 1.0
    for p in pairwise:
        assert p["multiplicity_adjustment"] == "holm"
        assert p["family_size"] == 3
        assert p["contrast"]["definition"] == "first condition minus second condition"


def test_friedman_reversed_condition_order(balanced_panel_df):
    res_fwd = friedman_test(
        balanced_panel_df, "participant", "condition", "score", ("baseline", "week4", "week8")
    )
    res_rev = friedman_test(
        balanced_panel_df, "participant", "condition", "score", ("week8", "week4", "baseline")
    )
    # Omnibus statistic and p-value invariant under permutation of conditions
    assert res_fwd["statistic"] == pytest.approx(res_rev["statistic"])
    assert res_fwd["p_value"] == pytest.approx(res_rev["p_value"])
    # Pairwise comparisons follow declared order
    pairs_rev = [
        (p["first_condition"], p["second_condition"]) for p in res_rev["pairwise_comparisons"]
    ]
    assert pairs_rev == [("week8", "week4"), ("week8", "baseline"), ("week4", "baseline")]


# ---------------------------------------------------------------------------
# 3. Repeated-Measures ANOVA & Sphericity
# ---------------------------------------------------------------------------


def test_rm_anova_numerics_and_effect_size(balanced_panel_df):
    res = repeated_measures_anova(
        balanced_panel_df, "participant", "condition", "score", ("baseline", "week4", "week8")
    )
    # Check ANOVA table
    table = res["anova_table"]
    k = 3
    n = 8
    assert table["df_condition"] == k - 1  # 2
    assert table["df_subject"] == n - 1  # 7
    assert table["df_error"] == (n - 1) * (k - 1)  # 14
    assert table["f_statistic"] > 0
    assert 0.0 <= table["p_value"] <= 1.0

    # Partial eta-squared = SS_condition / (SS_condition + SS_error)
    ss_cond = table["ss_condition"]
    ss_err = table["ss_error"]
    expected_p_eta2 = ss_cond / (ss_cond + ss_err)
    assert res["effect_size"]["value"] == pytest.approx(expected_p_eta2)
    assert 0.0 <= res["effect_size"]["value"] <= 1.0


def test_rm_anova_pairwise_paired_t_and_holm(balanced_panel_df):
    res = repeated_measures_anova(
        balanced_panel_df, "participant", "condition", "score", ("baseline", "week4", "week8")
    )
    pairwise = res["pairwise_comparisons"]
    assert len(pairwise) == 3

    # Validate against individual paired t-tests
    panel = repeated_panel(
        balanced_panel_df, "participant", "condition", "score", ("baseline", "week4", "week8")
    )
    c0 = panel["panel"][:, 0]
    c1 = panel["panel"][:, 1]
    ref_pair = stats.ttest_rel(c0, c1)

    p0 = pairwise[0]
    assert p0["first_condition"] == "baseline"
    assert p0["second_condition"] == "week4"
    assert p0["statistic"] == pytest.approx(float(ref_pair.statistic))
    assert p0["raw_p_value"] == pytest.approx(float(ref_pair.pvalue))
    assert p0["mean_difference"] == pytest.approx(float(np.mean(c0 - c1)))
    assert p0["confidence_interval"] is not None
    assert (
        p0["confidence_interval"]["lower"]
        <= p0["mean_difference"]
        <= p0["confidence_interval"]["upper"]
    )
    assert p0["effect_size"]["name"] in ("cohens_dz", "Cohen's dz")


def test_rm_anova_sphericity_not_rejected(spherical_df):
    res = repeated_measures_anova(
        spherical_df, "subject", "cond", "val", ("condA", "condB", "condC")
    )
    sphericity = res["sphericity"]
    assert sphericity["status"] == "not_rejected"
    assert sphericity["p_value"] >= 0.05
    # When sphericity not rejected, primary inference should be uncorrected
    assert res["primary_inference"] == "uncorrected"
    assert res["p_value"] == pytest.approx(res["anova_table"]["p_value"])
    # GG epsilon should be close to 1.0
    gg = res["greenhouse_geisser"]
    assert gg["epsilon"] >= 0.85
    assert gg["applied"] is False


def test_rm_anova_sphericity_rejected_applies_gg(non_spherical_df):
    res = repeated_measures_anova(
        non_spherical_df, "subject", "time", "val", ("T1", "T2", "T3", "T4")
    )
    sphericity = res["sphericity"]
    assert sphericity["status"] == "rejected"
    assert sphericity["p_value"] < 0.05
    # When sphericity rejected, primary inference should be greenhouse_geisser
    assert res["primary_inference"] == "greenhouse_geisser"
    gg = res["greenhouse_geisser"]
    assert gg["applied"] is True
    assert gg["epsilon"] < 0.90
    assert 1.0 / 3.0 <= gg["epsilon"] <= 1.0
    # Corrected df
    orig_num_df = res["anova_table"]["df_condition"]
    orig_den_df = res["anova_table"]["df_error"]
    assert gg["corrected_df_num"] == pytest.approx(gg["epsilon"] * orig_num_df)
    assert gg["corrected_df_den"] == pytest.approx(gg["epsilon"] * orig_den_df)
    # Primary p_value matches corrected p_value
    assert res["p_value"] == pytest.approx(gg["corrected_p_value"])
    # F statistic unchanged by GG
    assert res["statistic"] == pytest.approx(res["anova_table"]["f_statistic"])


# ---------------------------------------------------------------------------
# 4. Method Selection & 2-Condition vs 3+-Condition Boundary
# ---------------------------------------------------------------------------


def test_method_selection_boundaries():
    # 2 conditions paired mean -> paired_t
    df_2 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "time": ["pre"] * 5 + ["post"] * 5,
            "score": [10, 12, 14, 11, 13, 15, 16, 17, 14, 18],
        }
    )
    assistant_2 = ResearchAssistant(df_2)
    draft_2_mean = assistant_2.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="time",
        design="paired",
        estimand="mean",
        unit_id="id",
        condition_order=("pre", "post"),
        variable_types={"score": "continuous"},
    )
    rec_2_mean = assistant_2.recommend_test(draft_2_mean)
    assert rec_2_mean.method_id == "paired_t"

    # 2 conditions paired distribution -> wilcoxon_signed_rank
    draft_2_dist = assistant_2.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="time",
        design="paired",
        estimand="distribution",
        unit_id="id",
        condition_order=("pre", "post"),
        variable_types={"score": "continuous"},
    )
    rec_2_dist = assistant_2.recommend_test(draft_2_dist)
    assert rec_2_dist.method_id == "wilcoxon_signed_rank"

    # 3 conditions repeated mean -> repeated_measures_anova
    df_3 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5] * 3,
            "time": ["t1"] * 5 + ["t2"] * 5 + ["t3"] * 5,
            "score": [10, 12, 14, 11, 13, 15, 16, 17, 14, 18, 20, 21, 22, 19, 23],
        }
    )
    assistant_3 = ResearchAssistant(df_3)
    draft_3_mean = assistant_3.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="time",
        design="repeated",
        estimand="mean",
        unit_id="id",
        condition_order=("t1", "t2", "t3"),
        variable_types={"score": "continuous"},
    )
    rec_3_mean = assistant_3.recommend_test(draft_3_mean)
    assert rec_3_mean.method_id == "repeated_measures_anova"

    # 3 conditions repeated distribution -> friedman_test
    draft_3_dist = assistant_3.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="time",
        design="repeated",
        estimand="distribution",
        unit_id="id",
        condition_order=("t1", "t2", "t3"),
        variable_types={"score": "continuous"},
    )
    rec_3_dist = assistant_3.recommend_test(draft_3_dist)
    assert rec_3_dist.method_id == "friedman_test"


def test_repeated_design_with_only_two_conditions_blocked():
    df_2 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "time": ["pre"] * 5 + ["post"] * 5,
            "score": [10, 12, 14, 11, 13, 15, 16, 17, 14, 18],
        }
    )
    assistant = ResearchAssistant(df_2)
    # Repeated design requires at least 3 conditions
    with pytest.raises(InvalidDataError, match="at least three condition labels"):
        assistant.prepare_question(
            objective="compare_groups",
            outcome="score",
            predictor="time",
            design="repeated",
            estimand="mean",
            unit_id="id",
            condition_order=("pre", "post"),
            variable_types={"score": "continuous"},
        )


# ---------------------------------------------------------------------------
# 5. Integrated Workflow, explain(), Report & Completeness
# ---------------------------------------------------------------------------


def test_workflow_explain_rm_anova(balanced_panel_df):
    assistant = ResearchAssistant(balanced_panel_df)
    workflow = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="repeated",
        estimand="mean",
        unit_id="participant",
        condition_order=("baseline", "week4", "week8"),
        variable_types={"score": "continuous"},
    )
    assert workflow.status == "completed"
    assert workflow.analysis.method_id == "repeated_measures_anova"
    assert workflow.analysis.values["pairwise_comparisons"] is not None

    explanation = workflow.explain()
    assert "OMNIBUS TEST" in explanation
    assert "SPHERICITY / CORRECTION" in explanation
    assert "CONDITION SUMMARIES" in explanation
    assert "PAIRWISE FOLLOW-UP" in explanation
    assert "baseline" in explanation
    assert "week4" in explanation
    assert "week8" in explanation
    assert "Holm" in explanation or "holm" in explanation


def test_workflow_explain_friedman(balanced_panel_df):
    assistant = ResearchAssistant(balanced_panel_df)
    workflow = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="repeated",
        estimand="distribution",
        unit_id="participant",
        condition_order=("baseline", "week4", "week8"),
        variable_types={"score": "continuous"},
    )
    assert workflow.status == "completed"
    assert workflow.analysis.method_id == "friedman_test"

    explanation = workflow.explain()
    assert "OMNIBUS TEST" in explanation
    assert "Friedman" in explanation
    assert "Kendall" in explanation
    assert "PAIRWISE FOLLOW-UP" in explanation


def test_canonical_reports_and_exports(balanced_panel_df):
    assistant = ResearchAssistant(balanced_panel_df)
    workflow = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="repeated",
        estimand="mean",
        unit_id="participant",
        condition_order=("baseline", "week4", "week8"),
        variable_types={"score": "continuous"},
    )
    report = workflow.report
    assert report is not None

    # Check exports
    html = report.to_html()
    assert "Repeated-measures ANOVA" in html
    assert "Repeated condition summaries" in html
    assert "Repeated-measures pairwise comparisons" in html

    md = report.to_markdown()
    assert "Repeated-measures ANOVA" in md

    csv_tables = report.to_csv_tables()
    assert "repeated_anova_omnibus" in csv_tables
    assert "repeated_condition_summary" in csv_tables
    assert "repeated_sphericity" in csv_tables
    assert "repeated_pairwise" in csv_tables

    json_str = report.to_json()
    assert json.loads(json_str) is not None


def test_strict_json_serialization(balanced_panel_df, non_spherical_df):
    # rmANOVA with GG
    assistant_gg = ResearchAssistant(non_spherical_df)
    wf_gg = assistant_gg.run(
        objective="compare_groups",
        outcome="val",
        predictor="time",
        design="repeated",
        estimand="mean",
        unit_id="subject",
        condition_order=("T1", "T2", "T3", "T4"),
        variable_types={"val": "continuous"},
    )
    payload_gg = wf_gg.analysis.to_dict()
    # Must serialize with allow_nan=False
    json.dumps(payload_gg, allow_nan=False)

    # Friedman
    assistant_f = ResearchAssistant(balanced_panel_df)
    wf_f = assistant_f.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="repeated",
        estimand="distribution",
        unit_id="participant",
        condition_order=("baseline", "week4", "week8"),
        variable_types={"score": "continuous"},
    )
    payload_f = wf_f.analysis.to_dict()
    json.dumps(payload_f, allow_nan=False)


def test_reporting_completeness_rm_anova(balanced_panel_df):
    assistant = ResearchAssistant(balanced_panel_df)
    workflow = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="repeated",
        estimand="mean",
        unit_id="participant",
        condition_order=("baseline", "week4", "week8"),
        variable_types={"score": "continuous"},
    )
    completeness = assess_reporting_completeness(workflow.report)
    assert completeness.status == "complete"
    codes = [item.code for item in completeness.items]
    assert "SPHERICITY_REPORTED" in codes
    assert "CORRECTION_REPORTED" in codes
    assert "PAIRWISE_COMPARISONS_REPORTED" in codes


# ---------------------------------------------------------------------------
# 6. Audit, Reproducibility, Analysis Plans, and Sessions
# ---------------------------------------------------------------------------


def test_audit_passes_for_repeated_measures(balanced_panel_df, non_spherical_df):
    auditor = StatisticalResultAuditor()

    # Test rmANOVA uncorrected
    assistant_1 = ResearchAssistant(balanced_panel_df)
    wf_1 = assistant_1.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="repeated",
        estimand="mean",
        unit_id="participant",
        condition_order=("baseline", "week4", "week8"),
        variable_types={"score": "continuous"},
    )
    audit_1 = auditor.audit(wf_1.report, result=wf_1.analysis)
    assert audit_1.status == "passed", f"Audit failed: {audit_1.findings}"

    # Test rmANOVA GG corrected
    assistant_2 = ResearchAssistant(non_spherical_df)
    wf_2 = assistant_2.run(
        objective="compare_groups",
        outcome="val",
        predictor="time",
        design="repeated",
        estimand="mean",
        unit_id="subject",
        condition_order=("T1", "T2", "T3", "T4"),
        variable_types={"val": "continuous"},
    )
    audit_2 = auditor.audit(wf_2.report, result=wf_2.analysis)
    assert audit_2.status == "passed", f"Audit failed: {audit_2.findings}"

    # Test Friedman
    assistant_3 = ResearchAssistant(balanced_panel_df)
    wf_3 = assistant_3.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="repeated",
        estimand="distribution",
        unit_id="participant",
        condition_order=("baseline", "week4", "week8"),
        variable_types={"score": "continuous"},
    )
    audit_3 = auditor.audit(wf_3.report, result=wf_3.analysis)
    assert audit_3.status == "passed", f"Audit failed: {audit_3.findings}"


def test_reproducibility_replay(balanced_panel_df):
    assistant = ResearchAssistant(balanced_panel_df)
    wf = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="repeated",
        estimand="mean",
        unit_id="participant",
        condition_order=("baseline", "week4", "week8"),
        variable_types={"score": "continuous"},
    )
    replay = reproduce(wf.reproducibility, data=balanced_panel_df)
    assert replay.status == "reproduced"
    assert not replay.differing_fields


def test_analysis_plan_round_trip_and_adherence(balanced_panel_df):
    assistant = ResearchAssistant(balanced_panel_df)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="repeated",
        estimand="mean",
        unit_id="participant",
        condition_order=("baseline", "week4", "week8"),
        variable_types={"score": "continuous"},
    )
    plan = assistant.analysis_plan(draft)
    assert plan.status.value == "ready"
    assert plan.primary_method_id == "repeated_measures_anova"
    assert plan.effect_quantity == "partial_eta_squared"

    # Round trip dict / json
    plan_dict = plan.to_dict()
    restored = StatisticalAnalysisPlan.from_dict(plan_dict)
    assert restored.primary_method_id == plan.primary_method_id
    assert restored.specification.condition_order == ("baseline", "week4", "week8")

    # Compare plan to executed result
    result = assistant.analyze(draft)
    adherence = compare_plan_to_result(plan, result)
    assert adherence.status == "matched"


def test_session_snapshot(balanced_panel_df):
    assistant = ResearchAssistant(balanced_panel_df)
    workflow = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="repeated",
        estimand="mean",
        unit_id="participant",
        condition_order=("baseline", "week4", "week8"),
        variable_types={"score": "continuous"},
    )
    snapshot = build_session_snapshot(workflow)
    snap_dict = snapshot.to_dict()
    assert snap_dict["schema_version"] == 1
    assert "capabilities" in snap_dict
    assert "workflow" in snap_dict
    # Verify strict JSON
    json.dumps(snap_dict, allow_nan=False)


# ---------------------------------------------------------------------------
# 7. StatisticalAnalyzer API Integration
# ---------------------------------------------------------------------------


def test_statistical_analyzer_methods(balanced_panel_df):
    analyzer = StatisticalAnalyzer(balanced_panel_df)
    res_f = analyzer.friedman_test(
        unit_id="participant",
        condition_col="condition",
        value_col="score",
        condition_order=("baseline", "week4", "week8"),
    )
    assert "statistic" in res_f
    assert "p_value" in res_f

    res_a = analyzer.repeated_measures_anova(
        unit_id="participant",
        condition_col="condition",
        value_col="score",
        condition_order=("baseline", "week4", "week8"),
    )
    assert "anova_table" in res_a
    assert "sphericity" in res_a
