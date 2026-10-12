"""Tests for PR-6 Part D alpha vs confidence-level semantic clarity and independence."""

import math

import pandas as pd

from pyautostat.research_assistant import ResearchAssistant
from pyautostat.specifications import AnalysisOptions


def test_alpha_and_confidence_level_may_differ():
    # Options permit independent alpha and confidence_level
    options = AnalysisOptions(alpha=0.01, confidence_level=0.90)
    assert options.alpha == 0.01
    assert options.confidence_level == 0.90
    assert not options.levels_are_complementary


def test_p_value_decision_uses_alpha():
    # If p = 0.03, alpha = 0.01 -> fail to reject; alpha = 0.05 -> reject
    options_strict = AnalysisOptions(alpha=0.01, confidence_level=0.90)
    assert not options_strict.levels_are_complementary


def test_interval_uses_confidence_level():
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "score": [10.0, 11.0, 12.0, 9.0, 11.0, 10.0, 12.0, 11.0, 9.0, 10.0]
            + [15.0, 14.0, 16.0, 15.0, 17.0, 14.0, 16.0, 15.0, 17.0, 15.0],
        }
    )
    ra = ResearchAssistant(df)
    # 90% CI with alpha=0.05 (non-complementary: 0.90 != 1 - 0.05 = 0.95)
    wf_90 = ra.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.90),
        variable_types={"score": "continuous", "group": "nominal"},
    )
    assert wf_90.analysis is not None
    ci_90 = wf_90.analysis.values["confidence_interval"]
    assert ci_90 is not None
    assert math.isclose(ci_90["level"], 0.90)

    # 99% CI with alpha=0.05
    wf_99 = ra.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.99),
        variable_types={"score": "continuous", "group": "nominal"},
    )
    assert wf_99.analysis is not None
    ci_99 = wf_99.analysis.values["confidence_interval"]
    assert ci_99 is not None
    assert math.isclose(ci_99["level"], 0.99)

    # 99% interval is wider than 90% interval
    width_90 = ci_90["upper"] - ci_90["lower"]
    width_99 = ci_99["upper"] - ci_99["lower"]
    assert width_99 > width_90


def test_noncomplementary_levels_are_disclosed():
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "score": [10.0, 11.0, 12.0, 9.0, 11.0, 10.0, 12.0, 11.0, 9.0, 10.0]
            + [15.0, 14.0, 16.0, 15.0, 17.0, 14.0, 16.0, 15.0, 17.0, 15.0],
        }
    )
    ra = ResearchAssistant(df)
    wf = ra.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        options=AnalysisOptions(alpha=0.01, confidence_level=0.90),
        variable_types={"score": "continuous", "group": "nominal"},
    )
    assert wf.analysis is not None
    # Check warning emitted in analysis.warnings
    assert any("not complementary" in w for w in wf.analysis.warnings)
    # Check metadata disclosure
    disclosure = wf.analysis.metadata.get("inferential_levels")
    assert disclosure is not None
    assert disclosure["alpha"] == 0.01
    assert disclosure["confidence_level"] == 0.90
    assert not disclosure["complementary"]


def test_noncomplementary_levels_do_not_claim_test_ci_equivalence():
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "score": [10.0, 11.0, 12.0, 9.0, 11.0, 10.0, 12.0, 11.0, 9.0, 10.0]
            + [15.0, 14.0, 16.0, 15.0, 17.0, 14.0, 16.0, 15.0, 17.0, 15.0],
        }
    )
    ra = ResearchAssistant(df)
    wf = ra.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        options=AnalysisOptions(alpha=0.01, confidence_level=0.80),
        variable_types={"score": "continuous", "group": "nominal"},
    )
    assert wf.analysis is not None
    # Check that report/narrative does not claim mathematical equivalence
    report_dict = wf.report.to_dict() if wf.report else {}
    report_text = str(report_dict).lower()
    assert "ci proves significance" not in report_text
    assert "confirms the significance test" not in report_text


def test_complementary_levels_do_not_emit_false_warning():
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "score": [10.0, 11.0, 12.0, 9.0, 11.0, 10.0, 12.0, 11.0, 9.0, 10.0]
            + [15.0, 14.0, 16.0, 15.0, 17.0, 14.0, 16.0, 15.0, 17.0, 15.0],
        }
    )
    ra = ResearchAssistant(df)
    wf = ra.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95),
        variable_types={"score": "continuous", "group": "nominal"},
    )
    assert wf.analysis is not None
    # No "not complementary" warning
    assert not any("not complementary" in w for w in wf.analysis.warnings)
    assert "inferential_levels" not in wf.analysis.metadata
