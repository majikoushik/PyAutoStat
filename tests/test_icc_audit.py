"""Tests for auditor verification of Intraclass Correlation Coefficient (ICC) results."""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from pyautostat.audit import StatisticalResultAuditor
from pyautostat.research_assistant import ResearchAssistant


@pytest.fixture
def clean_icc_setup():
    """Produce clean ResearchAssistant, AnalysisResult, and ResearchReport for ICC(2,1)."""
    raw = [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ]
    rows = []
    for t_idx, target_scores in enumerate(raw):
        for j_idx, score in enumerate(target_scores):
            rows.append(
                {
                    "target": f"Target_{t_idx + 1}",
                    "rater": f"Judge_{j_idx + 1}",
                    "score": float(score),
                }
            )
    df = pd.DataFrame(rows)
    assistant = ResearchAssistant(df)
    workflow = assistant.intraclass_correlation(
        target="target",
        rater="rater",
        value="score",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
    )
    return assistant, workflow.analysis, workflow.report


def test_clean_icc_audit_passes(clean_icc_setup):
    """An unmodified ICC workflow must pass audit with zero discrepancies."""
    assistant, result, report = clean_icc_setup
    audit = assistant.audit(report, result=result)

    assert audit.status == "passed"
    assert len(audit.findings) == 0

    serialized = json.dumps(audit.to_dict(), allow_nan=False)
    assert len(serialized) > 0


def test_audit_passes_on_negative_icc():
    """Negative sample ICC estimates must be accepted by the auditor without failing."""
    rows = []
    ratings = [
        [2.0, 10.0, 1.0, 7.0],
        [8.0, 3.0, 9.0, 4.0],
        [1.0, 9.0, 3.0, 7.0],
        [10.0, 2.0, 8.0, 4.0],
    ]
    for t_idx, row in enumerate(ratings):
        for r_idx, val in enumerate(row):
            rows.append({"target": f"T{t_idx}", "rater": f"R{r_idx}", "score": val})
    neg_df = pd.DataFrame(rows)
    assistant = ResearchAssistant(neg_df)
    workflow = assistant.intraclass_correlation(
        target="target",
        rater="rater",
        value="score",
        model="one_way_random",
        unit="single",
    )
    assert workflow.analysis.values["intraclass_correlation"] < 0.0

    audit = assistant.audit(workflow.report, result=workflow.analysis)
    assert audit.status == "passed"
    assert len(audit.findings) == 0


def test_audit_catches_tampered_estimate(clean_icc_setup):
    """Auditor catches estimate that does not match the recomputed value from ANOVA MS."""
    _, result, report = clean_icc_setup
    tampered_result = copy.deepcopy(result)
    # Alter the estimate from 0.2898 to 0.9999
    tampered_result.values["estimate"] = 0.9999
    tampered_result.values["intraclass_correlation"] = 0.9999

    auditor = StatisticalResultAuditor()
    audit = auditor.audit(report, result=tampered_result)

    assert audit.status == "failed"
    assert any("estimate" in f.field for f in audit.findings)


def test_audit_catches_tampered_targets_df(clean_icc_setup):
    """Auditor catches targets degrees of freedom inconsistent with target count."""
    _, result, report = clean_icc_setup
    tampered_result = copy.deepcopy(result)
    # Modify df_targets from 5 to 99
    tampered_result.values["anova_table"]["df_targets"] = 99

    auditor = StatisticalResultAuditor()
    audit = auditor.audit(report, result=tampered_result)

    assert audit.status == "failed"
    assert any("df_targets" in f.field for f in audit.findings)


def test_audit_catches_tampered_f_statistic(clean_icc_setup):
    """Auditor catches F statistic inconsistent with mean squares."""
    _, result, report = clean_icc_setup
    tampered_result = copy.deepcopy(result)
    # Modify target F statistic
    tampered_result.values["f_test"]["statistic"] = 999.0

    auditor = StatisticalResultAuditor()
    audit = auditor.audit(report, result=tampered_result)

    assert audit.status == "failed"
    assert any("f_test.statistic" in f.field for f in audit.findings)


def test_audit_catches_inverted_ci_bounds(clean_icc_setup):
    """Auditor catches confidence interval where lower > upper."""
    _, result, report = clean_icc_setup
    tampered_result = copy.deepcopy(result)
    tampered_result.values["confidence_interval"]["lower"] = 0.8
    tampered_result.values["confidence_interval"]["upper"] = 0.2

    auditor = StatisticalResultAuditor()
    audit = auditor.audit(report, result=tampered_result)

    assert audit.status == "failed"
    assert any("confidence_interval" in f.field for f in audit.findings)


def test_audit_catches_corrupted_sample_size(clean_icc_setup):
    """Auditor catches target count < 2 or rater count < 2."""
    _, result, report = clean_icc_setup
    tampered_result = copy.deepcopy(result)
    tampered_result.values["n_targets"] = 1

    auditor = StatisticalResultAuditor()
    audit = auditor.audit(report, result=tampered_result)

    assert audit.status == "failed"
    assert any("n_targets" in f.field for f in audit.findings)
