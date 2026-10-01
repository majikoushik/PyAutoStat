"""Tests for statistical auditor verification of two-way factorial ANOVA results."""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from pyautostat.audit import StatisticalResultAuditor
from pyautostat.research_assistant import ResearchAssistant


@pytest.fixture
def clean_anova_setup():
    """Produce clean ResearchAssistant, AnalysisResult, and ResearchReport for 3x2 ANOVA."""
    df = pd.DataFrame(
        {
            "y": [
                10.0,
                11.0,
                12.0,
                14.0,
                15.0,
                16.0,
                13.0,
                12.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                25.0,
            ],
            "A": [
                "A1",
                "A1",
                "A1",
                "A2",
                "A2",
                "A2",
                "A2",
                "A3",
                "A3",
                "A3",
                "A3",
                "A3",
                "A3",
                "A3",
            ],
            "B": [
                "B1",
                "B1",
                "B2",
                "B1",
                "B1",
                "B2",
                "B2",
                "B1",
                "B1",
                "B1",
                "B2",
                "B2",
                "B2",
                "B2",
            ],
        }
    )
    assistant = ResearchAssistant(df)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="y",
        factor_a="A",
        factor_b="B",
        design="independent",
        estimand="mean",
    )
    result = assistant.analyze(draft)
    report = assistant.report(result)
    return assistant, result, report


def test_clean_two_way_anova_audit_passes(clean_anova_setup):
    """An unmodified two-way ANOVA workflow must pass audit with zero discrepancies."""
    assistant, result, report = clean_anova_setup
    audit = assistant.audit(report, result=result)

    assert audit.status == "passed"
    assert len(audit.findings) == 0

    serialized = json.dumps(audit.to_dict(), allow_nan=False)
    assert len(serialized) > 0


def test_audit_catches_tampered_term_df(clean_anova_setup):
    """Auditor catches degrees of freedom inconsistent with mean square."""
    _, result, report = clean_anova_setup
    tampered_result = copy.deepcopy(result)
    # Modify df for factor A from 2 to 99 without updating mean_square
    tampered_result.values["terms"][0]["df"] = 99.0

    auditor = StatisticalResultAuditor()
    audit = auditor.audit(report, result=tampered_result)

    assert audit.status == "failed"
    assert any("terms.A.mean_square" in f.field for f in audit.findings)


def test_audit_catches_tampered_f_statistic(clean_anova_setup):
    """Auditor catches F statistic inconsistent with term MS / residual MS."""
    _, result, report = clean_anova_setup
    tampered_result = copy.deepcopy(result)
    # Modify F-statistic for factor B
    tampered_result.values["terms"][1]["f_statistic"] = 999.0

    auditor = StatisticalResultAuditor()
    audit = auditor.audit(report, result=tampered_result)

    assert audit.status == "failed"
    assert any("terms.B.f_statistic" in f.field for f in audit.findings)


def test_audit_catches_invalid_p_value(clean_anova_setup):
    """Auditor catches p-values outside [0, 1]."""
    _, result, report = clean_anova_setup
    tampered_result = copy.deepcopy(result)
    tampered_result.values["terms"][0]["p_value"] = -0.05

    auditor = StatisticalResultAuditor()
    audit = auditor.audit(report, result=tampered_result)

    assert audit.status == "failed"
    assert any("terms.A.p_value" in f.field for f in audit.findings)


def test_audit_catches_tampered_partial_eta_squared(clean_anova_setup):
    """Auditor catches partial eta-squared inconsistent with term SS / (term SS + SS_error)."""
    _, result, report = clean_anova_setup
    tampered_result = copy.deepcopy(result)
    tampered_result.values["terms"][0]["effect_size"]["value"] = 0.999

    auditor = StatisticalResultAuditor()
    audit = auditor.audit(report, result=tampered_result)

    assert audit.status == "failed"
    assert any("terms.A.effect_size.value" in f.field for f in audit.findings)


def test_audit_catches_inverted_ci_bounds(clean_anova_setup):
    """Auditor catches partial eta-squared confidence interval where lower > upper."""
    _, result, report = clean_anova_setup
    tampered_result = copy.deepcopy(result)
    ci = tampered_result.values["terms"][0]["effect_size"]["confidence_interval"]
    ci["lower"] = 0.8
    ci["upper"] = 0.2

    auditor = StatisticalResultAuditor()
    audit = auditor.audit(report, result=tampered_result)

    assert audit.status == "failed"
    assert any("terms.A.effect_size.confidence_interval" in f.field for f in audit.findings)


def test_audit_catches_tampered_term_names(clean_anova_setup):
    """Auditor catches term list names that do not match factor specifications."""
    _, result, report = clean_anova_setup
    tampered_result = copy.deepcopy(result)
    tampered_result.values["terms"][0]["term"] = "TamperedTerm"

    auditor = StatisticalResultAuditor()
    audit = auditor.audit(report, result=tampered_result)

    assert audit.status == "failed"
    assert any("terms.names" in f.field for f in audit.findings)


def test_audit_catches_followup_p_value_corruption(clean_anova_setup):
    """Auditor catches followup contrast where adjusted p is strictly less than raw p."""
    _, result, report = clean_anova_setup
    tampered_result = copy.deepcopy(result)
    fup = tampered_result.values["followups"][0]
    fup["raw_p_value"] = 0.10
    fup["adjusted_p_value"] = 0.01  # Impossible for Holm step-down

    auditor = StatisticalResultAuditor()
    audit = auditor.audit(report, result=tampered_result)

    assert audit.status == "failed"
    assert any("followups[0].adjusted_p_value" in f.field for f in audit.findings)


def test_audit_catches_missing_cell_summaries(clean_anova_setup):
    """Auditor catches missing cells in factorial cell summaries."""
    _, result, report = clean_anova_setup
    tampered_result = copy.deepcopy(result)
    # Remove one cell from the 6 expected cells
    tampered_result.values["cell_summaries"] = tampered_result.values["cell_summaries"][:-1]

    auditor = StatisticalResultAuditor()
    audit = auditor.audit(report, result=tampered_result)

    assert audit.status == "failed"
    assert any("cell_summaries" in f.field for f in audit.findings)
