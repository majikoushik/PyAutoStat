"""Regression tests for public usability and presentation helpers."""

import numpy as np
import pandas as pd
import pytest

from pyautostat import (
    AnalysisResult,
    AnalysisStatus,
    InsightEngine,
    InterpretationFinding,
    InterpretationResult,
    InterpretationStatus,
    MeaningfulEffectThreshold,
    Recommendation,
    RecommendationStatus,
    ResearchAssistant,
    StatisticalAnalyzer,
    column_story,
)
from pyautostat.exceptions import InvalidDataError
from pyautostat.recommendation import METHOD_CAPABILITIES
from pyautostat.results import MissingInformation


@pytest.fixture()
def two_group_df():
    return pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "score": [10, 12, 11, 13, 9, 10, 18, 20, 19, 22, 17, 21],
            "age": [25, 30, 28, 35, 22, 31, 40, 38, 42, 45, 37, 39],
        }
    )


def _completed_workflow(frame):
    return ResearchAssistant(frame).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
        data_dictionary={"score": {"type": "continuous", "unit": "points"}},
    )


class TestFindingsPlain:
    def test_messages_are_numbered_without_changing_them(self):
        result = InterpretationResult(
            status=InterpretationStatus.AVAILABLE,
            method_id="welch_t",
            execution_status=AnalysisStatus.AVAILABLE,
            summary="Summary.",
            findings=(
                InterpretationFinding("first", "First finding."),
                InterpretationFinding("second", "Second finding."),
            ),
        )

        assert result.findings_plain == "1. First finding.\n2. Second finding."

    def test_empty_findings_have_clear_placeholder(self):
        result = InterpretationResult(
            status=InterpretationStatus.PARTIAL,
            method_id="welch_t",
            execution_status=AnalysisStatus.AVAILABLE,
            summary="Summary.",
        )

        assert result.findings_plain == "No findings recorded."


class TestMethodLabel:
    def test_analysis_result_uses_capability_registry(self):
        result = AnalysisResult(method_id="welch_t", status=AnalysisStatus.AVAILABLE)
        assert result.method_label == METHOD_CAPABILITIES["welch_t"].name

        # Phase 5 result ergonomics and convenience accessor verification
        res_welch = AnalysisResult(
            method_id="welch_t",
            status=AnalysisStatus.AVAILABLE,
            sample_size=30,
            excluded_rows=2,
            values={
                "test_statistic": 2.45,
                "degrees_of_freedom": 24.3,
                "p_value": 0.022,
                "primary_estimate": 4.5,
                "estimate_name": "mean difference",
                "confidence_interval": {"lower": 0.7, "upper": 8.3, "level": 0.95},
                "effect_size": {
                    "name": "cohen_d",
                    "value": 0.82,
                    "confidence_interval": {"lower": 0.12, "upper": 1.51, "level": 0.95},
                },
            },
            metadata={
                "sample": {"group_sizes": [{"group": "A", "size": 15}, {"group": "B", "size": 15}]}
            },
        )
        assert res_welch.test_name == METHOD_CAPABILITIES["welch_t"].name
        assert res_welch.statistic == 2.45
        assert res_welch.p_value == 0.022
        assert res_welch.degrees_of_freedom == 24.3
        assert res_welch.estimate == 4.5
        assert res_welch.confidence_interval == {"lower": 0.7, "upper": 8.3, "level": 0.95}
        assert res_welch.effect_size["value"] == 0.82
        assert res_welch.effect_size_confidence_interval == {
            "lower": 0.12,
            "upper": 1.51,
            "level": 0.95,
        }
        assert res_welch.sample_accounting["sample_size"] == 30
        assert res_welch.sample_accounting["excluded_rows"] == 2
        prim = res_welch.primary_result()
        assert prim["method_id"] == "welch_t"
        df_prim = res_welch.to_dataframe("primary")
        assert df_prim["degrees_of_freedom"].iloc[0] == 24.3
        stmt = res_welch.apa_statement()
        assert (
            "Welch's independent-samples t-test: t(24.3) = 2.45, p = .022, cohen_d = 0.82" in stmt
        )
        assert res_welch.statement("apa") == stmt
        assert res_welch.statement("generic").startswith(
            "Welch independent-samples t-test: statistic = 2.45"
        )

        # One-way ANOVA
        res_anova = AnalysisResult(
            method_id="one_way_anova",
            status=AnalysisStatus.AVAILABLE,
            sample_size=45,
            values={
                "test_statistic": 5.12,
                "degrees_of_freedom": [2, 42],
                "p_value": 0.009,
                "primary_estimate": None,
                "effect_size": {"name": "eta_squared", "value": 0.196},
                "pairwise_comparisons": [{"first": "A", "second": "B", "p_value": 0.01}],
                "group_summaries": [{"group": "A", "mean": 10.0}],
            },
        )
        assert res_anova.degrees_of_freedom == (2, 42)
        assert "omnibus F(2, 42) = 5.12, p = .009, eta_squared = 0.20" in res_anova.apa_statement()
        assert len(res_anova.to_dataframe("comparisons")) == 1

        # Repeated-measures ANOVA
        res_rm = AnalysisResult(
            method_id="repeated_measures_anova",
            status=AnalysisStatus.AVAILABLE,
            sample_size=30,
            values={
                "test_statistic": 8.44,
                "degrees_of_freedom": [1.6, 22.4],
                "p_value": 0.003,
                "primary_inference": "greenhouse_geisser",
                "greenhouse_geisser": {"applied": True, "epsilon": 0.8},
                "effect_size": {"name": "partial_eta_squared", "value": 0.38},
            },
        )
        assert res_rm.degrees_of_freedom == (1.6, 22.4)
        assert "Greenhouse-Geisser epsilon = .80 applied" in res_rm.apa_statement()

        # Two-way factorial ANOVA
        res_tw = AnalysisResult(
            method_id="two_way_anova",
            status=AnalysisStatus.AVAILABLE,
            sample_size=60,
            values={
                "sum_of_squares_type": "III",
                "terms": [
                    {
                        "term": "FactorA",
                        "term_type": "main_effect",
                        "df": 1,
                        "f_statistic": 4.5,
                        "p_value": 0.038,
                        "effect_size": {"value": 0.08},
                    },
                    {
                        "term": "FactorB",
                        "term_type": "main_effect",
                        "df": 2,
                        "f_statistic": 1.2,
                        "p_value": 0.31,
                        "effect_size": {"value": 0.02},
                    },
                    {
                        "term": "Residual",
                        "term_type": "residual",
                        "df": 54,
                    },
                ],
            },
        )
        assert res_tw.statistic is None
        assert res_tw.p_value is None
        assert res_tw.degrees_of_freedom is None
        tw_stmt = res_tw.apa_statement()
        assert "Type III SS" in tw_stmt
        assert "FactorA: F(1, 54) = 4.50, p = .038, partial eta^2 = .08" in tw_stmt
        assert "Residual" not in tw_stmt
        assert len(res_tw.to_dataframe("terms")) == 3

        # Linear regression
        res_ols = AnalysisResult(
            method_id="linear_regression",
            status=AnalysisStatus.AVAILABLE,
            sample_size=100,
            values={
                "model_fit": {
                    "model_degrees_of_freedom": 2,
                    "residual_degrees_of_freedom": 97,
                    "model_f_statistic": 12.3,
                    "model_f_p_value": 0.00005,
                    "r_squared": 0.202,
                    "adjusted_r_squared": 0.186,
                },
                "coefficients": [
                    {
                        "term": "x1",
                        "estimate": 1.5,
                        "standard_error": 0.4,
                        "statistic": 3.75,
                        "p_value": 0.0003,
                        "confidence_interval": {"lower": 0.7, "upper": 2.3, "level": 0.95},
                    }
                ],
            },
        )
        assert res_ols.statistic is None
        assert res_ols.degrees_of_freedom is None
        assert (
            "Linear regression model fit: F(2, 97) = 12.30, p < .001, R^2 = .20"
            in res_ols.apa_statement()
        )
        assert len(res_ols.to_dataframe("coefficients")) == 1

        # Logistic regression
        res_logit = AnalysisResult(
            method_id="logistic_regression",
            status=AnalysisStatus.AVAILABLE,
            sample_size=120,
            values={
                "event_level": "Yes",
                "model_fit": {
                    "lr_degrees_of_freedom": 2,
                    "lr_statistic": 15.6,
                    "lr_p_value": 0.0004,
                    "mcfadden_r2": 0.18,
                },
                "coefficients": [
                    {
                        "term": "x1",
                        "estimate": 0.693,
                        "standard_error": 0.25,
                        "statistic": 2.77,
                        "p_value": 0.0056,
                        "odds_ratio": 2.0,
                        "odds_ratio_ci": {"lower": 1.23, "upper": 3.25, "level": 0.95},
                    }
                ],
            },
        )
        assert res_logit.statistic is None
        assert res_logit.degrees_of_freedom is None
        assert (
            "Logistic regression model fit (event: Yes): LR chi^2(2) = 15.60, "
            "p < .001, McFadden pseudo-R^2 = .18" in res_logit.apa_statement()
        )
        assert "OR = 2.00" in res_logit.apa_statement()

        # Mann-Whitney & Wilcoxon
        res_mw = AnalysisResult(
            method_id="mann_whitney_u",
            status=AnalysisStatus.AVAILABLE,
            values={"test_statistic": 42.0, "p_value": 0.04, "effect_size": {"value": 0.35}},
        )
        assert (
            "Mann-Whitney U test: U = 42.00, p = .040, rank-biserial r = .35"
            in res_mw.apa_statement()
        )

        res_wx = AnalysisResult(
            method_id="wilcoxon_signed_rank",
            status=AnalysisStatus.AVAILABLE,
            values={"test_statistic": 15.0, "p_value": 0.03, "effect_size": {"value": -0.42}},
        )
        assert (
            "Wilcoxon signed-rank test: W = 15.00, p = .030, matched-pairs rank-biserial r = -.42"
            in res_wx.apa_statement()
        )

        # Kruskal-Wallis & Friedman
        res_kw = AnalysisResult(
            method_id="kruskal_wallis",
            status=AnalysisStatus.AVAILABLE,
            values={
                "test_statistic": 7.2,
                "degrees_of_freedom": 2,
                "p_value": 0.027,
                "effect_size": {"value": 0.22},
            },
        )
        assert (
            "Kruskal-Wallis test: H(2) = 7.20, p = .027, epsilon^2 = 0.22" in res_kw.apa_statement()
        )

        res_fr = AnalysisResult(
            method_id="friedman_test",
            status=AnalysisStatus.AVAILABLE,
            values={
                "test_statistic": 6.5,
                "degrees_of_freedom": 2,
                "p_value": 0.039,
                "effect_size": {"value": 0.33, "name": "Kendall's W"},
            },
        )
        assert "Friedman test: Q(2) = 6.50, p = .039, Kendall's W = 0.33" in res_fr.apa_statement()

        # Pearson & Spearman & Point-biserial
        res_pear = AnalysisResult(
            method_id="pearson_correlation",
            status=AnalysisStatus.AVAILABLE,
            values={
                "primary_estimate": 0.45,
                "p_value": 0.002,
                "confidence_interval": {"lower": 0.18, "upper": 0.66, "level": 0.95},
            },
        )
        assert res_pear.degrees_of_freedom is None
        assert (
            "Pearson correlation: r = .45, p = .002, 95% CI [.18, .66]" in res_pear.apa_statement()
        )

        res_spear = AnalysisResult(
            method_id="spearman_correlation",
            status=AnalysisStatus.AVAILABLE,
            values={
                "primary_estimate": 0.51,
                "p_value": 0.001,
                "confidence_interval": {"lower": 0.22, "upper": 0.71, "level": 0.95},
            },
        )
        assert (
            "Spearman rank correlation: r_s = .51, p = .001, 95% CI [.22, .71]"
            in res_spear.apa_statement()
        )

        res_pb = AnalysisResult(
            method_id="point_biserial_correlation",
            status=AnalysisStatus.AVAILABLE,
            values={"primary_estimate": 0.38, "degrees_of_freedom": 48, "p_value": 0.007},
        )
        assert "r_pb(48) = .38, p = .007." in res_pb.apa_statement()

        # Chi-Square, Fisher, McNemar
        res_chi = AnalysisResult(
            method_id="pearson_chi_square",
            status=AnalysisStatus.AVAILABLE,
            sample_size=100,
            values={
                "test_statistic": 9.4,
                "degrees_of_freedom": 2,
                "p_value": 0.009,
                "primary_estimate": 0.31,
            },
        )
        assert "chi^2(2, N = 100) = 9.40, p = .009, Cramer's V = .31" in res_chi.apa_statement()

        res_fish = AnalysisResult(
            method_id="fisher_exact",
            status=AnalysisStatus.AVAILABLE,
            values={
                "primary_estimate": 3.2,
                "p_value": 0.041,
                "confidence_interval": {"lower": 1.05, "upper": 9.8, "level": 0.95},
            },
        )
        assert (
            "Fisher's exact test: p = .041, sample OR = 3.20, 95% CI [1.05, 9.80]"
            in res_fish.apa_statement()
        )

        res_mcn = AnalysisResult(
            method_id="mcnemar",
            status=AnalysisStatus.AVAILABLE,
            values={
                "primary_estimate": 0.15,
                "p_value": 0.035,
                "confidence_interval": {"lower": 0.02, "upper": 0.28, "level": 0.95},
            },
        )
        assert (
            "Exact McNemar test: p = .035, paired proportion difference = .15, 95% CI [.02, .28]"
            in res_mcn.apa_statement()
        )

        # Cronbach alpha & ICC
        res_alpha = AnalysisResult(
            method_id="cronbach_alpha",
            status=AnalysisStatus.AVAILABLE,
            sample_size=80,
            values={"primary_estimate": 0.84, "item_count": 10},
        )
        assert res_alpha.statistic is None
        assert res_alpha.degrees_of_freedom is None
        assert (
            "Cronbach's alpha: alpha = .84, k = 10 items, N = 80 respondents"
            in res_alpha.apa_statement()
        )

        res_icc = AnalysisResult(
            method_id="intraclass_correlation",
            status=AnalysisStatus.AVAILABLE,
            values={
                "notation": "ICC(2,1)",
                "description": "Two-way random single-measure",
                "primary_estimate": 0.78,
                "degrees_of_freedom": [19, 38],
                "confidence_interval": {"lower": 0.55, "upper": 0.91, "level": 0.95},
                "f_test": {"statistic": 4.6, "p_value": 0.0001, "df1": 19, "df2": 38},
            },
        )
        assert res_icc.degrees_of_freedom == (19, 38)
        assert "Intraclass correlation ICC(2,1)" in res_icc.apa_statement()

        # Student t, One-sample t, Paired t
        res_stud = AnalysisResult(
            method_id="student_t",
            status=AnalysisStatus.AVAILABLE,
            values={
                "test_statistic": 2.1,
                "degrees_of_freedom": 28,
                "p_value": 0.045,
                "effect_size": {"name": "cohen_d", "value": 0.65},
            },
        )
        assert "t(28) = 2.10, p = .045, cohen_d = 0.65" in res_stud.apa_statement()

        res_os = AnalysisResult(
            method_id="one_sample_t",
            status=AnalysisStatus.AVAILABLE,
            values={
                "test_statistic": 3.1,
                "degrees_of_freedom": 19,
                "p_value": 0.006,
                "effect_size": {"name": "cohen_d", "value": 0.7},
            },
        )
        assert "One-sample t-test: t(19) = 3.10, p = .006, cohen_d = 0.70" in res_os.apa_statement()

        res_pt = AnalysisResult(
            method_id="paired_t",
            status=AnalysisStatus.AVAILABLE,
            values={
                "test_statistic": 2.8,
                "degrees_of_freedom": 14,
                "p_value": 0.014,
                "effect_size": {"name": "cohen_dz", "value": 0.72},
            },
        )
        assert (
            "Paired-samples t-test: t(14) = 2.80, p = .014, cohen_dz = 0.72"
            in res_pt.apa_statement()
        )

        # Kendall tau-b & Partial Pearson
        res_kend = AnalysisResult(
            method_id="kendall_tau_b",
            status=AnalysisStatus.AVAILABLE,
            values={"primary_estimate": 0.35, "p_value": 0.012},
        )
        assert "tau = .35, p = .012" in res_kend.apa_statement()

        res_part = AnalysisResult(
            method_id="partial_pearson_correlation",
            status=AnalysisStatus.AVAILABLE,
            values={"primary_estimate": 0.42, "degrees_of_freedom": 45, "p_value": 0.003},
        )
        assert "r_partial(45) = .42, p = .003" in res_part.apa_statement()

        # Sample accounting zero-value preservation
        res_zero = AnalysisResult(
            method_id="welch_t",
            status=AnalysisStatus.AVAILABLE,
            sample_size=10,
            excluded_rows=0,
            values={},
            metadata={
                "sample": {
                    "original_rows": 10,
                    "analyzed_rows": 10,
                    "excluded_rows": 0,
                    "nonzero_differences": 0,
                }
            },
        )
        assert res_zero.sample_accounting["excluded_rows"] == 0
        assert res_zero.sample_accounting["nonzero_differences"] == 0

        # Workflow statement generation helper
        from pyautostat.result_access import generate_workflow_statement

        class DummyWorkflowWithBlocker:
            analysis = None
            blockers = ("Missing outcome variable",)

        assert "Missing outcome variable" in generate_workflow_statement(DummyWorkflowWithBlocker())

        class DummyWorkflowEmpty:
            analysis = None
            blockers = ()

        assert "has not completed statistical execution" in generate_workflow_statement(
            DummyWorkflowEmpty()
        )

        class DummyWorkflowWithAnalysis:
            analysis = res_welch

        assert "Welch" in generate_workflow_statement(DummyWorkflowWithAnalysis())

        # primary_result across complex methods
        assert "terms" in res_tw.primary_result()
        assert "model_fit" in res_ols.primary_result()
        assert "model_fit" in res_logit.primary_result()
        assert res_rm.primary_result()["statistic"] == 8.44
        assert "pairwise_comparisons" in res_anova.primary_result()
        assert "all_variants" in res_icc.primary_result()
        assert res_icc.statistic == 4.6
        assert res_icc.p_value == 0.0001
        assert not res_alpha.to_dataframe("primary").empty
        assert not res_icc.to_dataframe("primary").empty

        # Unavailable and invalid cases
        res_unavail = AnalysisResult(method_id="welch_t", status=AnalysisStatus.UNAVAILABLE)
        assert res_unavail.primary_result()["status"] == "unavailable"
        with pytest.raises(ValueError, match="Unsupported section 'bogus'"):
            res_welch.to_dataframe("bogus")

    def test_unknown_analysis_method_falls_back_to_identifier(self):
        result = AnalysisResult(method_id="custom_method", status=AnalysisStatus.AVAILABLE)
        assert result.method_label == "custom_method"

    def test_recommendation_preserves_its_recorded_method_name(self):
        recommendation = Recommendation(
            status=RecommendationStatus.READY,
            method_id="custom_method",
            method_name="Custom method",
            rationale="Researcher supplied.",
        )
        assert recommendation.method_label == "Custom method"

    def test_invalid_empty_method_identifier_remains_rejected(self):
        with pytest.raises(InvalidDataError, match="non-empty"):
            AnalysisResult(method_id="", status=AnalysisStatus.AVAILABLE)


class TestNormalityVerdict:
    def test_all_available_normality_diagnostics_have_verdicts(self):
        normality = StatisticalAnalyzer(pd.DataFrame({"x": range(10)})).analyze_all()["normality"][
            "x"
        ]

        assert set(normality) == {
            "shapiro_wilk",
            "d_agostino_pearson",
            "anderson_darling",
        }
        assert all(test["verdict"] for test in normality.values())

    def test_rejected_verdict_is_explicit(self):
        values = [1, 1, 1, 1, 1, 100, 200, 300, 400, 500]
        shapiro = StatisticalAnalyzer(pd.DataFrame({"x": values})).analyze_all()["normality"]["x"][
            "shapiro_wilk"
        ]

        assert shapiro["status"] == "rejected"
        assert "rejected" in shapiro["verdict"].lower()

    def test_nonrejection_does_not_claim_normality(self):
        values = np.random.default_rng(42).normal(0, 1, 30)
        shapiro = StatisticalAnalyzer(pd.DataFrame({"x": values})).analyze_all()["normality"]["x"][
            "shapiro_wilk"
        ]

        assert shapiro["status"] == "not_rejected"
        assert "does not prove normality" in shapiro["verdict"].lower()


class TestHypothesisInterpretation:
    def test_interpretation_names_groups_and_avoids_duplicate_ci_label(self, two_group_df):
        result = StatisticalAnalyzer(two_group_df).hypothesis_tests(
            "group", "score", test_type="ttest"
        )
        text = result["interpretation"]

        assert "t-test" in text
        assert "p =" in text or "p <" in text
        assert "'A'" in text and "'B'" in text
        assert "confidence interval for 95% CI" not in text


class TestSummarize:
    def test_summary_contains_core_sections_and_shape(self, two_group_df):
        text = ResearchAssistant(two_group_df).summarize()

        assert "12 rows x 3 columns" in text
        assert "MISSING DATA" in text
        assert "CORRELATIONS" in text
        assert "DATA QUALITY" in text

    def test_zero_completeness_is_not_replaced_by_default(self):
        frame = pd.DataFrame({"x": [None, None], "y": [None, None]})
        text = ResearchAssistant(frame).summarize()

        assert "Completeness    : 0.0%" in text
        assert "Overall missing rate : 100.0%" in text

    def test_plain_text_summary_prints_on_cp1252_terminals(self, two_group_df):
        ResearchAssistant(two_group_df).summarize().encode("cp1252")

    def test_story_mode_is_opt_in_deterministic_readable_and_nonmutating(self):
        frame = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 100.0, 100.0],
                "y": [2.0, 4.0, 6.0, 8.0, 200.0, 200.0],
                "group": ["a", "a", "b", "b", None, None],
            }
        )
        before = frame.copy(deep=True)
        assistant = ResearchAssistant(frame)

        legacy = assistant.summarize()
        assert legacy == assistant.summarize(mode="profile")
        story = assistant.summarize(mode="story")

        assert "DATASET STORY" in story
        assert "6 rows" in story and "3 columns" in story
        assert "KEY FINDING" in story
        assert "DATA QUALITY" in story
        assert "DISTRIBUTION" in story
        assert "RECOMMENDED FIRST STEPS" in story
        assert "{'" not in story
        assert story == assistant.summarize(mode="story")
        pd.testing.assert_frame_equal(frame, before)

    def test_story_mode_rejects_unknown_mode(self, two_group_df):
        with pytest.raises(InvalidDataError, match="summary mode"):
            ResearchAssistant(two_group_df).summarize(mode="future")

    def test_profile_statistics_feed_public_column_story_without_recalculation(self):
        assistant = ResearchAssistant(pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]}))
        profile = assistant.profile()
        text = column_story("score", profile["descriptive"]["score"], unit="points")
        assert "Mean = 2.5 points" in text
        assert "median = 2.5 points" in text


class TestMissingInformationDisplay:
    def test_standalone_message_is_actionable_and_portable(self):
        text = str(MissingInformation("design", "Study design is required."))

        assert "design" in text
        assert "Study design is required." in text
        assert "assistant.run" in text
        text.encode("cp1252")

    def test_workflow_explanation_uses_structured_questions(self, two_group_df):
        result = ResearchAssistant(two_group_df).run(
            objective="compare_groups",
            outcome="score",
            predictor="group",
            estimand="mean",
            data_dictionary={"score": {"type": "continuous"}},
        )
        text = result.explain()

        assert result.status.value == "needs_input"
        assert "MISSING INFORMATION" in text
        assert "'independent'" in text
        assert "'paired'" in text
        assert "update_question" in text
        text.encode("cp1252")

    def test_recommendation_stage_questions_retain_choices(self):
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 7.0]})
        result = ResearchAssistant(frame).run(
            objective="association",
            outcome="x",
            predictor="y",
            design="independent",
            variable_types={"x": "continuous", "y": "continuous"},
        )

        text = result.explain()

        assert result.status.value == "needs_input"
        assert "'linear'" in text
        assert "'monotonic'" in text


class TestInsightShape:
    def test_every_generated_insight_has_normalized_keys(self, two_group_df):
        summary = StatisticalAnalyzer(two_group_df).analyze_all()
        insights = InsightEngine(summary).generate_insights()
        required = {"category", "severity", "finding", "details", "recommendation"}

        for insight in insights:
            assert required <= set(insight)
            assert isinstance(insight["details"], list)
            assert isinstance(insight["recommendation"], list)

    def test_outlier_recommendation_is_column_specific(self):
        frame = pd.DataFrame({"x": [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 200, 300, 400]})
        insights = InsightEngine(StatisticalAnalyzer(frame).analyze_all()).generate_insights()
        outlier = next(item for item in insights if item["category"] == "Outliers")

        assert any("'x'" in recommendation for recommendation in outlier["recommendation"])

    def test_distribution_retains_legacy_columns_alias(self):
        results = {"distributions": {"x": {"skewness_interpretation": "Positively skewed"}}}
        insight = InsightEngine(results).generate_insights()[0]

        assert insight["details"][0]["column"] == "x"
        assert insight["columns"] == ["x"]
        assert not any("non-parametric" in item for item in insight["recommendation"])


class TestPracticalSignificanceVerdict:
    def test_end_to_end_verdict_uses_validated_result(self, two_group_df):
        workflow = _completed_workflow(two_group_df)
        practical = ResearchAssistant(two_group_df).practical_significance(
            workflow.analysis,
            threshold=MeaningfulEffectThreshold(
                "mean_difference", 5, unit="points", rationale="Instructional relevance."
            ),
        )
        text = practical.verdict

        assert str(practical.estimate)[:3] in text
        assert "5" in text
        assert practical.conclusion in text
        assert "evidence against its recorded null hypothesis" in text
        assert "Threshold rationale" in text


class TestWorkflowExplain:
    def test_completed_explanation_has_no_duplicate_periods(self, two_group_df):
        result = _completed_workflow(two_group_df)
        payload = result.to_dict()
        text = result.explain()

        assert result.status.value == "completed"
        assert result.analysis.method_label in text
        assert "score" in text and "group" in text
        assert "Groups    : 'A', 'B'" in text
        assert " Sample    : 12 rows analysed" in text
        assert " HYPOTHESIS TEST" in text
        assert " EFFECT SIZE" in text
        assert " CONFIDENCE INTERVAL" in text
        assert " LIMITATIONS" in text
        assert ".." not in text
        assert text == result.explain()
        assert result.to_dict() == payload
        assert str(result) == text

    def test_assumption_notes_use_neutral_bullets(self, two_group_df):
        text = _completed_workflow(two_group_df).explain()

        assert " ASSUMPTIONS" in text
        assert "   - " in text
        assert "?" not in text
