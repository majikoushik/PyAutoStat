"""Corruption tests verifying audit detects scientific defects across result families."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pyautostat import (
    ResearchAssistant,
    StatisticalResultAuditor,
    execution,
)
from pyautostat.audit import _method_contract_findings
from pyautostat.results import Recommendation


@pytest.fixture
def two_group_case():
    df = pd.DataFrame(
        {
            "group": ["T"] * 10 + ["C"] * 10,
            "score": [12.0, 14.0, 13.5, 15.0, 16.0, 14.5, 15.5, 13.0, 16.5, 17.0]
            + [10.0, 11.5, 9.5, 12.0, 10.5, 11.0, 12.5, 9.0, 10.0, 11.0],
        }
    )
    asst = ResearchAssistant(df)
    wf = asst.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
    )
    assert wf.status.value == "completed"
    res = wf.analysis
    rep = asst.report(res)
    return asst, res, rep


@pytest.fixture
def rm_case():
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
    df = pd.DataFrame({"participant": participants, "condition": conditions, "score": scores})
    asst = ResearchAssistant(df)
    wf = asst.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="repeated",
        estimand="mean",
        unit_id="participant",
        condition_order=("baseline", "week4", "week8"),
        variable_types={"score": "continuous"},
    )
    assert wf.status.value == "completed"
    res = wf.analysis
    rep = asst.report(res)
    return asst, res, rep


@pytest.fixture
def friedman_case():
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
    df = pd.DataFrame({"participant": participants, "condition": conditions, "score": scores})
    asst = ResearchAssistant(df)
    wf = asst.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        design="repeated",
        estimand="distribution",
        unit_id="participant",
        condition_order=("baseline", "week4", "week8"),
        variable_types={"score": "continuous"},
    )
    assert wf.status.value == "completed"
    res = wf.analysis
    rep = asst.report(res)
    return asst, res, rep


@pytest.fixture
def paired_case():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6] * 2,
            "cond": ["pre"] * 6 + ["post"] * 6,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 5.0, 5.0, 7.0, 8.0],
        }
    )
    asst = ResearchAssistant(df)
    wf = asst.run(
        objective="compare_groups",
        outcome="y",
        predictor="cond",
        design="paired",
        estimand="mean",
        unit_id="id",
        condition_order=("pre", "post"),
        variable_types={"y": "continuous"},
    )
    assert wf.status.value == "completed"
    res = wf.analysis
    rep = asst.report(res)
    return asst, res, rep


@pytest.fixture
def logit_case():
    rng = np.random.default_rng(42)
    x = rng.normal(size=100)
    group = np.where(np.arange(100) % 2 == 0, "premium", "standard")
    prob = 1 / (1 + np.exp(-(-0.3 + 0.8 * x + 0.5 * (group == "premium"))))
    event = np.where(rng.binomial(1, prob), "yes", "no")
    df = pd.DataFrame({"event": event, "x": x, "group": group})
    asst = ResearchAssistant(df)
    wf = asst.run(
        objective="regression",
        outcome="event",
        predictors=["x", "group"],
        estimand="event_probability",
        design="independent",
        event_level="yes",
        variable_types={"event": "nominal", "x": "continuous", "group": "nominal"},
        reference_levels={"group": "standard"},
    )
    assert wf.status.value == "completed"
    res = wf.analysis
    rep = asst.report(res)
    return asst, res, rep


def test_audit_detects_corrupted_p_value(two_group_case):
    asst, res, rep = two_group_case
    auditor = StatisticalResultAuditor()
    assert auditor.audit(rep).status == "passed"

    # Corrupt p-value < 0
    bad_vals = dict(res.values)
    bad_vals["p_value"] = -0.05
    corrupted = replace(res, values=bad_vals)
    audit = auditor.audit(rep, result=corrupted)
    assert audit.status == "failed"
    assert any(f.code == "PVALUE_MISMATCH" for f in audit.findings)
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "PVALUE_MISMATCH" for f in findings)

    # Corrupt p-value > 1
    bad_vals["p_value"] = 1.25
    corrupted = replace(res, values=bad_vals)
    audit = auditor.audit(rep, result=corrupted)
    assert audit.status == "failed"
    assert any(f.code == "PVALUE_MISMATCH" for f in audit.findings)
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "PVALUE_MISMATCH" for f in findings)


def test_audit_detects_reversed_confidence_interval(two_group_case):
    asst, res, rep = two_group_case
    auditor = StatisticalResultAuditor()

    bad_vals = deepcopy(res.values)
    ci = bad_vals["confidence_interval"]
    ci["lower"], ci["upper"] = ci["upper"], ci["lower"]
    corrupted = replace(res, values=bad_vals)
    audit = auditor.audit(rep, result=corrupted)
    assert audit.status == "failed"
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "INTERVAL_MISMATCH" for f in findings)


def test_audit_detects_point_estimate_outside_analytical_interval(two_group_case):
    asst, res, rep = two_group_case
    auditor = StatisticalResultAuditor()

    bad_vals = dict(res.values)
    bad_vals["primary_estimate"] = 999.0
    corrupted = replace(res, values=bad_vals)
    audit = auditor.audit(rep, result=corrupted)
    assert audit.status == "failed"
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "INTERVAL_MISMATCH" for f in findings)


def test_audit_detects_corrupted_group_order_and_contrast(two_group_case):
    asst, res, rep = two_group_case
    auditor = StatisticalResultAuditor()

    # Corrupt group order list length != 2
    bad_meta = deepcopy(res.metadata)
    bad_meta["group_order"] = ["T", "C", "Extra"]
    corrupted = replace(res, metadata=bad_meta)
    audit = auditor.audit(rep, result=corrupted)
    assert audit.status == "failed"
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "GROUP_ORDER_MISMATCH" for f in findings)

    # Corrupt contrast direction
    bad_meta = deepcopy(res.metadata)
    bad_meta["contrast"] = "C - T"
    corrupted = replace(res, metadata=bad_meta)
    audit = auditor.audit(rep, result=corrupted)
    assert audit.status == "failed"
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "GROUP_ORDER_MISMATCH" for f in findings)


def test_audit_detects_student_t_degrees_of_freedom_mismatch(two_group_case):
    asst, res, rep = two_group_case
    # Dispatch student_t directly
    spec = res.specification
    assert spec is not None
    rec = Recommendation(
        status="ready",
        method_id="student_t",
        method_name="Student t",
        method_availability="runnable",
    )
    student_res = execution._group_result(asst._analyzer, spec, rec)
    assert student_res.method_id == "student_t"
    student_rep = asst.report(student_res)

    auditor = StatisticalResultAuditor()
    assert auditor.audit(student_rep).status == "passed"

    bad_vals = dict(student_res.values)
    bad_vals["degrees_of_freedom"] = 55
    corrupted = replace(student_res, values=bad_vals)
    audit = auditor.audit(student_rep, result=corrupted)
    assert audit.status == "failed"
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "DEGREES_OF_FREEDOM_MISMATCH" for f in findings)


def test_audit_detects_paired_t_degrees_of_freedom_mismatch(paired_case):
    asst, res, rep = paired_case
    auditor = StatisticalResultAuditor()
    assert auditor.audit(rep).status == "passed"

    bad_vals = dict(res.values)
    bad_vals["degrees_of_freedom"] = 25
    corrupted = replace(res, values=bad_vals)
    audit = auditor.audit(rep, result=corrupted)
    assert audit.status == "failed"
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "DEGREES_OF_FREEDOM_MISMATCH" for f in findings)


def test_audit_detects_correlation_and_effect_size_out_of_bounds():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "y": [2.0, 3.5, 4.0, 6.5, 6.0, 8.0, 8.5, 10.0],
        }
    )
    asst = ResearchAssistant(df)
    wf = asst.run(
        objective="association",
        outcome="y",
        predictor="x",
        estimand="linear",
        design="independent",
        variable_types={"x": "continuous", "y": "continuous"},
    )
    assert wf.analysis is not None
    res = wf.analysis
    rep = asst.report(res)
    auditor = StatisticalResultAuditor()
    assert auditor.audit(rep).status == "passed"

    # Corrupt Pearson r > 1.0
    bad_vals = deepcopy(res.values)
    bad_vals["r"] = 1.45
    bad_vals["effect_size"]["value"] = 1.45
    corrupted = replace(res, values=bad_vals)
    audit = auditor.audit(rep, result=corrupted)
    assert audit.status == "failed"
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "EFFECT_SIZE_MISMATCH" for f in findings)


def test_audit_detects_rm_anova_ss_eta_squared_mismatch(rm_case):
    asst, res, rep = rm_case
    auditor = StatisticalResultAuditor()
    assert auditor.audit(rep).status == "passed"

    # Corrupt partial eta^2
    bad_vals = deepcopy(res.values)
    bad_vals["effect_size"]["value"] = 0.999
    corrupted = replace(res, values=bad_vals)
    audit = auditor.audit(rep, result=corrupted)
    assert audit.status == "failed"
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "EFFECT_SIZE_MISMATCH" for f in findings)


def test_audit_detects_rm_anova_gg_dfs_mismatch(rm_case):
    asst, res, rep = rm_case
    auditor = StatisticalResultAuditor()

    # Corrupt GG corrected numerator df
    bad_vals = deepcopy(res.values)
    bad_vals["greenhouse_geisser"]["corrected_numerator_df"] = 99.0
    corrupted = replace(res, values=bad_vals)
    audit = auditor.audit(rep, result=corrupted)
    assert audit.status == "failed"
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "DEGREES_OF_FREEDOM_MISMATCH" for f in findings)


def test_audit_detects_friedman_kendall_w_mismatch(friedman_case):
    asst, res, rep = friedman_case
    auditor = StatisticalResultAuditor()
    assert auditor.audit(rep).status == "passed"

    # Corrupt Kendall's W relative to Q / [n(k-1)]
    bad_vals = deepcopy(res.values)
    bad_vals["effect_size"]["value"] = 0.05
    corrupted = replace(res, values=bad_vals)
    audit = auditor.audit(rep, result=corrupted)
    assert audit.status == "failed"
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "EFFECT_SIZE_MISMATCH" for f in findings)


def test_audit_detects_logistic_odds_ratio_relative_to_beta(logit_case):
    asst, res, rep = logit_case
    auditor = StatisticalResultAuditor()
    assert auditor.audit(rep).status == "passed"

    # Corrupt OR != exp(beta)
    bad_vals = deepcopy(res.values)
    bad_vals["coefficients"][1]["odds_ratio"] = 999.0
    corrupted = replace(res, values=bad_vals)
    audit = auditor.audit(rep, result=corrupted)
    assert audit.status == "failed"
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "ODDS_RATIO_MISMATCH" for f in findings)


def test_audit_detects_pairwise_multiplicity_and_adjusted_p_corruption(rm_case):
    asst, res, rep = rm_case
    auditor = StatisticalResultAuditor()
    assert auditor.audit(rep).status == "passed"

    # Corrupt adjusted p < raw p
    bad_vals = deepcopy(res.values)
    pairwise = bad_vals["pairwise_comparisons"]
    pairwise[0]["adjusted_p_value"] = pairwise[0]["raw_p_value"] - 0.1
    corrupted = replace(res, values=bad_vals)
    audit = auditor.audit(rep, result=corrupted)
    assert audit.status == "failed"
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "PVALUE_MISMATCH" for f in findings)

    # Corrupt multiplicity count
    bad_vals = deepcopy(res.values)
    bad_vals["pairwise_comparisons"] = bad_vals["pairwise_comparisons"][:1]
    corrupted = replace(res, values=bad_vals)
    audit = auditor.audit(rep, result=corrupted)
    assert audit.status == "failed"
    findings = _method_contract_findings(corrupted)
    assert any(f.code == "MULTIPLICITY_MISMATCH" for f in findings)
