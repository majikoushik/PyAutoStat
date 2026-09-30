"""Machine-checkable verification of statistical method contracts.

Every runnable method in PyAutoStat must define an explicit scientific contract:
question, estimand, hypotheses, effect size, uncertainty status, required
assumptions, missingness policy, numerical provenance, audit invariants,
and independent validation source.
"""

from __future__ import annotations

import json

import pytest

from pyautostat import METHOD_CONTRACTS, MethodContract
from pyautostat.recommendation import METHOD_CAPABILITIES

VALID_CI_STATUSES = {
    "available",
    "unavailable",
    "not_supported",
    "not_applicable",
    "uncomputable",
}


def test_every_runnable_method_has_contract():
    """Every runnable inferential/model/reliability method in capabilities must have a contract."""
    runnable_capabilities = {
        method_id
        for method_id, cap in METHOD_CAPABILITIES.items()
        if cap.availability == "runnable" and method_id != "dataset_profile"
    }
    contract_ids = set(METHOD_CONTRACTS.keys())

    missing_contracts = runnable_capabilities - contract_ids
    assert not missing_contracts, (
        f"Runnable methods lack an authoritative contract: {missing_contracts}"
    )

    extra_contracts = contract_ids - runnable_capabilities
    assert not extra_contracts, f"Contracts exist for non-runnable methods: {extra_contracts}"


def test_contract_keys_match_method_ids():
    """Dictionary keys must match the internal method_id attribute."""
    for method_id, contract in METHOD_CONTRACTS.items():
        assert contract.method_id == method_id, (
            f"Key '{method_id}' does not match contract method_id '{contract.method_id}'"
        )


@pytest.mark.parametrize("method_id", sorted(METHOD_CONTRACTS.keys()))
def test_contract_fields_are_populated_and_structured(method_id: str):
    """Every method contract must provide complete, non-empty metadata."""
    contract = METHOD_CONTRACTS[method_id]
    assert isinstance(contract, MethodContract)

    # Text fields must be substantial non-empty strings
    assert len(contract.name.strip()) >= 5
    assert len(contract.scientific_question.strip()) >= 15
    assert len(contract.study_design.strip()) >= 5
    assert len(contract.outcome_type.strip()) >= 3
    assert len(contract.predictor_type.strip()) >= 3
    assert len(contract.estimand.strip()) >= 10
    assert len(contract.primary_estimate.strip()) >= 5
    assert len(contract.null_hypothesis.strip()) >= 5
    assert len(contract.alternative_hypothesis.strip()) >= 5
    assert len(contract.test_statistic.strip()) >= 3
    assert len(contract.degrees_of_freedom.strip()) >= 3
    assert len(contract.effect_size_quantity.strip()) >= 3
    assert len(contract.effect_size_definition.strip()) >= 10
    assert len(contract.ci_method.strip()) >= 3
    assert len(contract.missing_data_policy.strip()) >= 10
    assert len(contract.degenerate_data_behavior.strip()) >= 10
    assert len(contract.multiplicity_policy.strip()) >= 5
    assert len(contract.numerical_provenance.strip()) >= 10
    assert len(contract.validation_source.strip()) >= 10

    # Uncertainty statuses must come from standard set
    assert contract.effect_size_ci_status in VALID_CI_STATUSES, (
        f"{method_id}: invalid effect_size_ci_status '{contract.effect_size_ci_status}'"
    )
    assert contract.primary_estimate_ci_status in VALID_CI_STATUSES, (
        f"{method_id}: invalid primary_estimate_ci_status '{contract.primary_estimate_ci_status}'"
    )

    # Collections must have at least one entry
    assert len(contract.assumptions) >= 1
    assert len(contract.diagnostics) >= 1
    assert len(contract.interpretation_limitations) >= 1
    assert len(contract.audit_invariants) >= 1

    # Verify dictionary export and strict JSON serializability
    d = contract.to_dict()
    assert isinstance(d, dict)
    dumped = json.dumps(d, allow_nan=False)
    assert len(dumped) > 100


def test_null_hypothesis_and_estimand_scientific_alignment():
    """Verify statistical alignment between estimands and null hypotheses."""
    # Independent t tests
    for m in ("welch_t", "student_t"):
        c = METHOD_CONTRACTS[m]
        assert "mean difference" in c.estimand.lower()
        assert c.null_value == 0.0
        assert c.null_quantity == "mean difference"

    # Paired t test
    paired_t = METHOD_CONTRACTS["paired_t"]
    assert "paired difference" in paired_t.estimand.lower()
    assert paired_t.null_value == 0.0

    # One-sample t test
    one_sample_t = METHOD_CONTRACTS["one_sample_t"]
    assert "reference" in one_sample_t.estimand.lower()
    assert one_sample_t.null_quantity is not None
    assert "mean difference" in one_sample_t.null_quantity.lower()

    # Correlations
    for m in ("pearson_correlation", "spearman_correlation", "kendall_tau_b"):
        c = METHOD_CONTRACTS[m]
        assert c.null_value == 0.0

    # Fisher exact
    fisher = METHOD_CONTRACTS["fisher_exact"]
    assert fisher.null_value == 1.0
    assert fisher.null_quantity == "odds ratio"

    # Repeated measures ANOVA
    rm_anova = METHOD_CONTRACTS["repeated_measures_anova"]
    assert "repeated-condition" in rm_anova.null_hypothesis.lower()
    assert "means" in rm_anova.null_hypothesis.lower()

    # Friedman test
    friedman = METHOD_CONTRACTS["friedman_test"]
    assert "rank distribution" in friedman.estimand.lower()
    assert "rank distributions" in friedman.null_hypothesis.lower()


def test_scientific_guardrails_in_contract_limitations():
    """Verify that vital scientific guardrails are explicitly recorded in limitations."""
    # Friedman must state it is NOT a universal median test
    friedman = METHOD_CONTRACTS["friedman_test"]
    assert any(
        "NOT a universal test of medians" in lim for lim in friedman.interpretation_limitations
    )
    assert any("variance explained" in lim.lower() for lim in friedman.interpretation_limitations)

    # Wilcoxon signed rank must state it is not a generic median test
    wilcoxon = METHOD_CONTRACTS["wilcoxon_signed_rank"]
    assert any(
        "NOT a universal test of medians" in lim for lim in wilcoxon.interpretation_limitations
    )

    # Logistic regression must NOT claim bootstrap
    logit = METHOD_CONTRACTS["logistic_regression"]
    assert "wald" in logit.ci_method.lower()
    assert "not bootstrap" in logit.ci_method.lower() or "bootstrap" not in logit.ci_method.lower()

    # Repeated measures ANOVA sphericity
    rm_anova = METHOD_CONTRACTS["repeated_measures_anova"]
    assert any(
        "Mauchly non-rejection does NOT prove sphericity" in lim
        for lim in rm_anova.interpretation_limitations
    )


def test_effect_size_confidence_interval_contract_status():
    """Verify that implemented effect-size CIs are marked available and Welch is not_applicable."""
    assert METHOD_CONTRACTS["paired_t"].effect_size_ci_status == "available"
    assert METHOD_CONTRACTS["one_sample_t"].effect_size_ci_status == "available"
    assert METHOD_CONTRACTS["wilcoxon_signed_rank"].effect_size_ci_status == "available"
    assert METHOD_CONTRACTS["friedman_test"].effect_size_ci_status == "available"
    assert METHOD_CONTRACTS["fisher_exact"].effect_size_ci_status == "available"
    assert METHOD_CONTRACTS["repeated_measures_anova"].effect_size_ci_status == "available"
    assert METHOD_CONTRACTS["linear_regression"].effect_size_ci_status == "available"

    # Welch ANOVA has no standardized global effect size by design
    assert METHOD_CONTRACTS["welch_anova"].effect_size_ci_status == "not_applicable"
