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
        text = result.explain()

        assert result.status.value == "completed"
        assert result.analysis.method_label in text
        assert "score" in text and "group" in text
        assert ".." not in text
        assert str(result) == text

    def test_assumption_notes_use_neutral_bullets(self, two_group_df):
        text = _completed_workflow(two_group_df).explain()

        assert " ASSUMPTIONS" in text
        assert "   - " in text
        assert "?" not in text
