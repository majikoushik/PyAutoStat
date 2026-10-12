"""Audit hardening and corruption tests for effect-size confidence intervals."""

import copy
import math
import warnings

import pandas as pd
import pytest

from pyautostat import (
    AnalysisOptions,
    ResearchAssistant,
)
from pyautostat.audit import _method_contract_findings
from pyautostat.reproducibility import reproduce
from pyautostat.uncertainty import (
    noncentral_t_confidence_limits,
    one_sample_cohen_d_ci,
    paired_cohen_dz_ci,
)


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


def test_known_nct_internal_warning_does_not_escape():
    """Known SciPy internal NCT CDF warnings (e.g. divide by zero in _nct_cdf) do not leak."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        # Small df and moderate/high t_stat triggers root bracket expansion across wide NCPs
        limits = noncentral_t_confidence_limits(t_stat=3.5, df=2.0, confidence_level=0.95)
        assert math.isfinite(limits[0])
        assert math.isfinite(limits[1])
        assert limits[0] < limits[1]

        nct_warnings = [
            w
            for w in recorded
            if "_nct_cdf" in str(w.message) or "divide by zero" in str(w.message)
        ]
        assert len(nct_warnings) == 0


def test_unrelated_runtime_warning_is_not_globally_suppressed():
    """Unrelated RuntimeWarnings are preserved and never globally suppressed by pyautostat."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        noncentral_t_confidence_limits(t_stat=1.5, df=5.0, confidence_level=0.90)
        warnings.warn("unrelated division by zero in caller code", RuntimeWarning, stacklevel=2)

        msg = "unrelated division by zero in caller code"
        unrelated = [w for w in recorded if msg in str(w.message)]
        assert len(unrelated) == 1
        assert issubclass(unrelated[0].category, RuntimeWarning)


def test_small_sample_nct_ci_remains_finite_or_cleanly_uncomputable():
    """Small sample sizes produce finite, ordered CIs or cleanly uncomputable records."""
    # N = 3 paired test (df = 2)
    ci_res = paired_cohen_dz_ci(mean_diff=1.2, sd_diff=0.6, n=3, confidence_level=0.95)
    assert ci_res["status"] == "available"
    assert math.isfinite(ci_res["lower"])
    assert math.isfinite(ci_res["upper"])
    assert ci_res["lower"] < ci_res["upper"]

    # N = 1 (insufficient data)
    ci_underflow = paired_cohen_dz_ci(mean_diff=1.2, sd_diff=0.6, n=1, confidence_level=0.95)
    assert ci_underflow["status"] == "unavailable"
    assert ci_underflow["lower"] is None
    assert ci_underflow["upper"] is None

    # Zero variance (uncomputable)
    ci_zero_sd = paired_cohen_dz_ci(mean_diff=1.2, sd_diff=0.0, n=5, confidence_level=0.95)
    assert ci_zero_sd["status"] == "uncomputable"
    assert ci_zero_sd["lower"] is None

    # One sample small n
    ci_one = one_sample_cohen_d_ci(
        mean_val=5.0, ref_val=2.0, sd_val=1.5, n=3, confidence_level=0.95
    )
    assert ci_one["status"] == "available"
    assert math.isfinite(ci_one["lower"])
    assert math.isfinite(ci_one["upper"])
    assert ci_one["lower"] < ci_one["upper"]
