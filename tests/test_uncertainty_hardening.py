"""Audit hardening and corruption tests for effect-size confidence intervals."""

import copy

import pandas as pd
import pytest

from pyautostat import (
    AnalysisOptions,
    ResearchAssistant,
)
from pyautostat.audit import _method_contract_findings
from pyautostat.reproducibility import reproduce


@pytest.fixture
def paired_workflow():
    frame = pd.DataFrame(
        {
            "id": list(range(10)) * 2,
            "condition": ["after"] * 10 + ["before"] * 10,
            "score": [
                12,
                14,
                15,
                18,
                20,
                22,
                23,
                25,
                28,
                30,
                10,
                11,
                14,
                15,
                17,
                20,
                21,
                22,
                24,
                25,
            ],
        }
    )
    assistant = ResearchAssistant(frame)
    return assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        estimand="mean",
        design="paired",
        unit_id="id",
        condition_order=("after", "before"),
        variable_types={"score": "continuous", "condition": "nominal"},
    )


@pytest.fixture
def wilcoxon_workflow():
    frame = pd.DataFrame(
        {
            "id": list(range(10)) * 2,
            "condition": ["after"] * 10 + ["before"] * 10,
            "score": [
                12,
                14,
                15,
                18,
                20,
                22,
                23,
                25,
                28,
                30,
                10,
                11,
                14,
                15,
                17,
                20,
                21,
                22,
                24,
                25,
            ],
        }
    )
    assistant = ResearchAssistant(frame)
    return assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        estimand="distribution",
        design="paired",
        unit_id="id",
        condition_order=("after", "before"),
        variable_types={"score": "continuous", "condition": "nominal"},
        options=AnalysisOptions(random_seed=42, bootstrap_samples=199),
    )


@pytest.fixture
def pearson_workflow():
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0],
        }
    )
    assistant = ResearchAssistant(frame)
    return assistant.run(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="linear",
        design="independent",
        variable_types={"x": "continuous", "y": "continuous"},
    )


def test_corrupted_inverted_bounds_fails_audit(paired_workflow):
    """Audit must fail when confidence interval lower bound exceeds upper bound."""
    result = copy.deepcopy(paired_workflow.analysis)
    ci = result.values["effect_size"]["confidence_interval"]
    ci["lower"], ci["upper"] = ci["upper"], ci["lower"]  # Invert bounds
    findings = _method_contract_findings(result)
    assert any(f.code == "INTERVAL_MISMATCH" for f in findings)


def test_corrupted_out_of_bounds_effect_ci_fails_audit(pearson_workflow):
    """Audit must fail when Pearson r CI bounds exceed [-1, 1]."""
    result = copy.deepcopy(pearson_workflow.analysis)
    ci = result.values["effect_size"]["confidence_interval"]
    ci["upper"] = 1.05  # Out of bounds
    findings = _method_contract_findings(result)
    assert any(f.code == "INTERVAL_MISMATCH" for f in findings)


def test_corrupted_bootstrap_accounting_fails_audit(wilcoxon_workflow):
    """Audit must fail if bootstrap resample accounting is inconsistent."""
    result = copy.deepcopy(wilcoxon_workflow.analysis)
    ci = result.values["effect_size"]["confidence_interval"]
    ci["valid_resamples"] = ci["requested_resamples"] + 10  # valid > requested
    findings = _method_contract_findings(result)
    assert any("valid_resamples" in f.field for f in findings)

    # Invalid resamples mismatch
    result2 = copy.deepcopy(wilcoxon_workflow.analysis)
    ci2 = result2.values["effect_size"]["confidence_interval"]
    ci2["invalid_resamples"] = 999  # invalid != requested - valid
    findings2 = _method_contract_findings(result2)
    assert any("invalid_resamples" in f.field for f in findings2)


def test_corrupted_pearson_n_le_3_available_ci_fails_audit():
    """Audit must fail if Pearson correlation reports an available CI when n <= 3."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 4.0]})
    assistant = ResearchAssistant(frame)
    workflow = assistant.run(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="linear",
        design="independent",
        variable_types={"x": "continuous", "y": "continuous"},
    )
    result = copy.deepcopy(workflow.analysis)
    # Artificially mark CI as available
    ci = result.values["effect_size"]["confidence_interval"]
    ci["status"] = "available"
    ci["lower"] = 0.5
    ci["upper"] = 0.99
    findings = _method_contract_findings(result)
    assert any(
        f.code == "INTERVAL_MISMATCH"
        and f.field == "analysis.values.effect_size.confidence_interval"
        for f in findings
    )


def test_fisher_exact_audit_blocks_conditional_exact_label_on_sample_or():
    """Audit must fail if a log-Wald sample-OR CI is labeled as exact conditional."""
    frame = pd.DataFrame(
        {
            "treatment": ["A"] * 10 + ["B"] * 10,
            "response": ["yes"] * 7 + ["no"] * 3 + ["yes"] * 2 + ["no"] * 8,
        }
    )
    assistant = ResearchAssistant(frame)
    workflow = assistant.run(
        objective="association",
        outcome="response",
        predictor="treatment",
        estimand="categorical_independence",
        design="independent",
        variable_types={"response": "nominal", "treatment": "nominal"},
    )
    result = copy.deepcopy(workflow.analysis)
    ci = result.values["effect_size"]["confidence_interval"]
    ci["method"] = "exact conditional confidence interval"  # Mislabeled
    findings = _method_contract_findings(result)
    assert any("method" in f.field for f in findings)


def test_rm_anova_audit_verifies_uncorrected_eta_provenance():
    """Audit verifies RM ANOVA partial eta-squared matches SS formula."""
    scores = [
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        12,
        14,
        13,
        16,
        15,
        18,
        17,
        20,
        19,
        21,
        15,
        16,
        18,
        17,
        20,
        19,
        22,
        21,
        23,
        25,
    ]
    frame = pd.DataFrame(
        {
            "subject": list(range(10)) * 3,
            "condition": ["c1"] * 10 + ["c2"] * 10 + ["c3"] * 10,
            "score": scores,
        }
    )
    assistant = ResearchAssistant(frame)
    workflow = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        estimand="mean",
        design="repeated",
        unit_id="subject",
        condition_order=("c1", "c2", "c3"),
        variable_types={"score": "continuous", "condition": "nominal"},
    )
    # Legitimate result passes audit
    findings = _method_contract_findings(workflow.analysis)
    assert not any(f.severity == "critical" or f.severity == "error" for f in findings)

    # Corrupt point effect size value
    corrupt = copy.deepcopy(workflow.analysis)
    corrupt.values["effect_size"]["value"] = 0.999
    corrupt_findings = _method_contract_findings(corrupt)
    assert any("effect_size.value" in f.field for f in corrupt_findings)


def test_reproducibility_replays_exact_bootstrap_intervals(wilcoxon_workflow):
    """Reproducibility replay must reproduce exact bootstrap confidence limits."""
    frame = pd.DataFrame(
        {
            "id": list(range(10)) * 2,
            "condition": ["after"] * 10 + ["before"] * 10,
            "score": [
                12,
                14,
                15,
                18,
                20,
                22,
                23,
                25,
                28,
                30,
                10,
                11,
                14,
                15,
                17,
                20,
                21,
                22,
                24,
                25,
            ],
        }
    )
    record = wilcoxon_workflow.reproducibility
    assert record is not None
    replay = reproduce(record, data=frame)
    assert replay.status == "reproduced"
    assert len(replay.differing_fields) == 0
