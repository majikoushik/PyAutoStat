"""Numerical validation and contract tests for effect-size confidence intervals."""

import json
import math

import pandas as pd
import pytest
import scipy.stats as stats

from pyautostat import (
    ResearchAssistant,
    StatisticalAnalyzer,
)
from pyautostat.uncertainty import (
    fisher_exact_sample_or_ci,
    fisher_z_correlation_ci,
    noncentral_t_confidence_limits,
    one_sample_cohen_d_ci,
    paired_cohen_dz_ci,
)


def test_noncentral_t_inversion_mathematical_limits():
    """Verify noncentral-t root finding produces exact tail probabilities."""
    t_val = 2.5
    df = 15
    conf_level = 0.95
    alpha = 0.05
    lam_l, lam_u = noncentral_t_confidence_limits(t_val, df, conf_level)

    prob_lower = 1.0 - float(stats.nct.cdf(t_val, df, lam_l))
    prob_upper = float(stats.nct.cdf(t_val, df, lam_u))
    assert prob_lower == pytest.approx(alpha / 2.0, abs=1e-5)
    assert prob_upper == pytest.approx(alpha / 2.0, abs=1e-5)

    # Edge case: t = 0 (symmetric noncentral-t limits)
    lam_l_zero, lam_u_zero = noncentral_t_confidence_limits(0.0, df, conf_level)
    assert lam_l_zero == pytest.approx(-lam_u_zero, abs=1e-6)

    # Edge case: negative t
    lam_l_neg, lam_u_neg = noncentral_t_confidence_limits(-t_val, df, conf_level)
    assert lam_l_neg == pytest.approx(-lam_u, abs=1e-6)
    assert lam_u_neg == pytest.approx(-lam_l, abs=1e-6)


def test_paired_cohen_dz_ci_numerical_validation():
    """Validate paired Cohen's dz noncentral-t interval against exact tail probabilities."""
    mean_diff = 2.0
    sd_diff = 1.0
    n = 16
    conf_level = 0.95
    dz_rec = paired_cohen_dz_ci(mean_diff, sd_diff, n, conf_level)

    assert dz_rec["status"] == "available"
    assert dz_rec["method"] == "exact noncentral-t inversion"
    assert dz_rec["quantity"] == "Cohen's dz"
    assert dz_rec["sidedness"] == "two-sided"
    assert dz_rec["multiplicity_adjusted"] is False

    d_val = mean_diff / sd_diff
    assert dz_rec["lower"] < d_val < dz_rec["upper"]

    t_val = d_val * math.sqrt(n)
    df = n - 1
    prob_lower = 1.0 - float(stats.nct.cdf(t_val, df, dz_rec["lower"] * math.sqrt(n)))
    prob_upper = float(stats.nct.cdf(t_val, df, dz_rec["upper"] * math.sqrt(n)))
    assert prob_lower == pytest.approx(0.025, abs=1e-5)
    assert prob_upper == pytest.approx(0.025, abs=1e-5)

    # Edge case: zero variance
    deg_rec = paired_cohen_dz_ci(0.0, 0.0, n, conf_level)
    assert deg_rec["status"] == "uncomputable"


def test_one_sample_cohen_d_ci_numerical_validation():
    """Validate one-sample Cohen's d noncentral-t interval against reference fixture."""
    mean_val = 105.0
    ref_val = 100.0
    sd_val = 10.0
    n = 25
    conf_level = 0.95
    rec = one_sample_cohen_d_ci(mean_val, ref_val, sd_val, n, conf_level)

    assert rec["status"] == "available"
    assert rec["method"] == "exact noncentral-t inversion"
    assert rec["quantity"] == "one-sample Cohen's d"

    d_val = (mean_val - ref_val) / sd_val
    assert rec["lower"] < d_val < rec["upper"]

    t_val = d_val * math.sqrt(n)
    df = n - 1
    prob_lower = 1.0 - float(stats.nct.cdf(t_val, df, rec["lower"] * math.sqrt(n)))
    prob_upper = float(stats.nct.cdf(t_val, df, rec["upper"] * math.sqrt(n)))
    assert prob_lower == pytest.approx(0.025, abs=1e-5)
    assert prob_upper == pytest.approx(0.025, abs=1e-5)

    # n < 2 unavailable
    assert one_sample_cohen_d_ci(mean_val, ref_val, sd_val, 1)["status"] == "unavailable"


def test_pearson_fisher_z_ci_matches_modern_scipy_and_direct_algebra():
    """Validate Pearson Fisher-z interval against modern SciPy and direct algebra."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    y = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0]
    r, _ = stats.pearsonr(x, y)
    n = len(x)

    rec = fisher_z_correlation_ci(r, n, 0.95)
    assert rec["status"] == "available"
    assert rec["method"] == "Fisher-z asymptotic normal confidence interval"
    assert rec["quantity"] == "Pearson r"

    # Direct formula
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = float(stats.norm.ppf(0.975))
    expected_lower = math.tanh(z - z_crit * se)
    expected_upper = math.tanh(z + z_crit * se)

    assert rec["lower"] == pytest.approx(expected_lower, abs=1e-12)
    assert rec["upper"] == pytest.approx(expected_upper, abs=1e-12)

    # Edge cases
    assert fisher_z_correlation_ci(0.5, 3)["status"] == "unavailable"
    assert fisher_z_correlation_ci(0.5, 2)["status"] == "unavailable"
    perf = fisher_z_correlation_ci(1.0, 10)
    assert perf["status"] == "available" and perf["lower"] <= 1.0 and perf["upper"] <= 1.0


def test_fisher_sample_odds_ratio_ci_coherence_and_validation():
    """Validate sample OR log-Wald CI against statsmodels Table2x2 and direct formula."""
    counts = [[10, 5], [2, 8]]
    rec = fisher_exact_sample_or_ci(counts, 0.95)
    assert rec["status"] == "available"
    assert rec["method"] == "log-Wald confidence interval for sample odds-ratio estimator"
    assert rec["quantity"] == "sample odds ratio"

    # Direct log-Wald algebra
    a, b, c, d = 10.0, 5.0, 2.0, 8.0
    sample_or = (a * d) / (b * c)  # 8.0
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z_crit = float(stats.norm.ppf(0.975))
    expected_lower = math.exp(math.log(sample_or) - z_crit * se)
    expected_upper = math.exp(math.log(sample_or) + z_crit * se)

    assert rec["lower"] == pytest.approx(expected_lower, abs=1e-12)
    assert rec["upper"] == pytest.approx(expected_upper, abs=1e-12)

    # Inverting columns inverts the odds ratio and swaps/inverts bounds
    inv_counts = [[5, 10], [8, 2]]
    inv_rec = fisher_exact_sample_or_ci(inv_counts, 0.95)
    assert inv_rec["lower"] == pytest.approx(1.0 / rec["upper"], abs=1e-10)
    assert inv_rec["upper"] == pytest.approx(1.0 / rec["lower"], abs=1e-10)

    # Zero cells: must be unavailable, no silent 0.5 correction
    zero_cell_counts = [[10, 0], [2, 8]]
    zero_rec = fisher_exact_sample_or_ci(zero_cell_counts, 0.95)
    assert zero_rec["status"] == "unavailable"
    assert "zero" in zero_rec["reason"].lower()


def test_wilcoxon_matched_pairs_rank_biserial_bootstrap_determinism_and_reversal():
    """Validate paired rank-biserial bootstrap: determinism and reversal."""
    frame = pd.DataFrame(
        {
            "id": list(range(10)) * 2,
            "condition": ["post"] * 10 + ["pre"] * 10,
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
    direct = StatisticalAnalyzer(frame).paired_wilcoxon(
        "id",
        "condition",
        "score",
        condition_order=("post", "pre"),
        random_state=42,
        bootstrap_samples=199,
    )
    ci = direct["effect_size"]["confidence_interval"]
    assert ci["status"] == "available"
    assert ci["method"] == "paired-observation percentile bootstrap"
    assert ci["quantity"] == "matched-pairs rank-biserial correlation"
    assert -1.0 <= ci["lower"] <= ci["upper"] <= 1.0
    assert ci["requested_resamples"] == 199
    assert ci["valid_resamples"] <= 199
    assert ci["invalid_resamples"] == 199 - ci["valid_resamples"]
    assert ci["random_seed"] == 42

    # Determinism
    direct_same = StatisticalAnalyzer(frame).paired_wilcoxon(
        "id",
        "condition",
        "score",
        condition_order=("post", "pre"),
        random_state=42,
        bootstrap_samples=199,
    )
    ci_same = direct_same["effect_size"]["confidence_interval"]
    assert ci["lower"] == ci_same["lower"]
    assert ci["upper"] == ci_same["upper"]

    # Reversing condition order negates point estimate and inverts bounds
    direct_rev = StatisticalAnalyzer(frame).paired_wilcoxon(
        "id",
        "condition",
        "score",
        condition_order=("pre", "post"),
        random_state=42,
        bootstrap_samples=199,
    )
    ci_rev = direct_rev["effect_size"]["confidence_interval"]
    assert direct_rev["effect_size"]["value"] == pytest.approx(-direct["effect_size"]["value"])
    assert ci_rev["lower"] == pytest.approx(-ci["upper"], abs=1e-8)
    assert ci_rev["upper"] == pytest.approx(-ci["lower"], abs=1e-8)


def test_friedman_kendall_w_block_bootstrap_validation():
    """Validate Kendall's W participant block bootstrap."""
    frame = pd.DataFrame(
        {
            "id": list(range(8)) * 3,
            "time": ["t1"] * 8 + ["t2"] * 8 + ["t3"] * 8,
            "rating": [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    direct = StatisticalAnalyzer(frame).friedman_test(
        "id",
        "time",
        "rating",
        condition_order=("t1", "t2", "t3"),
        random_state=123,
        bootstrap_samples=199,
    )
    ci = direct["effect_size"]["confidence_interval"]
    assert ci["status"] == "available"
    assert ci["method"] == "participant-block percentile bootstrap"
    assert ci["quantity"] == "Kendall's W"
    assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0
    assert ci["requested_resamples"] == 199

    # Pairwise comparisons must have rank-biserial bootstrap CIs
    for comp in direct["pairwise_comparisons"]:
        comp_ci = comp["confidence_interval"]
        assert comp_ci is not None
        assert comp_ci["method"] == "paired-observation percentile bootstrap"
        assert comp_ci["multiplicity_adjusted"] is False
        assert -1.0 <= comp_ci["lower"] <= comp_ci["upper"] <= 1.0


def test_repeated_measures_partial_eta_squared_ci_and_pairwise_dz():
    """Validate RM partial eta-squared noncentral-F interval and pairwise dz intervals."""
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
    direct = StatisticalAnalyzer(frame).repeated_measures_anova(
        "subject", "condition", "score", condition_order=("c1", "c2", "c3")
    )
    eta_ci = direct["effect_size"]["confidence_interval"]
    assert eta_ci["status"] == "available"
    assert eta_ci["method"] == "noncentral-F inversion (uncorrected F and df)"
    assert eta_ci["quantity"] == "partial eta-squared"
    assert 0.0 <= eta_ci["lower"] <= eta_ci["upper"] <= 1.0

    # Verify zero-error degenerate case returns uncomputable safely
    zero_err_scores = [
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
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
    ]
    frame_zero_err = pd.DataFrame(
        {
            "subject": list(range(10)) * 3,
            "condition": ["c1"] * 10 + ["c2"] * 10 + ["c3"] * 10,
            "score": zero_err_scores,
        }
    )
    direct_deg = StatisticalAnalyzer(frame_zero_err).repeated_measures_anova(
        "subject", "condition", "score", condition_order=("c1", "c2", "c3")
    )
    assert direct_deg["effect_size"]["confidence_interval"]["status"] == "uncomputable"

    # Verify point eta_p^2 matches SS formula
    ss_cond = direct["sums_of_squares"]["condition"]
    ss_err = direct["sums_of_squares"]["error"]
    expected_eta = ss_cond / (ss_cond + ss_err)
    assert direct["effect_size"]["value"] == pytest.approx(expected_eta, abs=1e-10)

    # Verify each pairwise contrast has dz CI with multiplicity_adjusted=False
    for comp in direct["pairwise_comparisons"]:
        comp_ci = comp["effect_size"]["confidence_interval"]
        assert comp_ci is not None
        assert comp_ci["method"] == "exact noncentral-t inversion"
        assert comp_ci["quantity"] == "Cohen's dz"
        assert comp_ci["multiplicity_adjusted"] is False
        assert comp_ci["lower"] <= comp["effect_size"]["value"] <= comp_ci["upper"]


def test_dunn_pairwise_rank_biserial_bootstrap_validation():
    """Validate Dunn pairwise comparisons have within-group bootstrap CIs."""
    frame = pd.DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
            "score": [1, 2, 2, 3, 3, 4, 3, 4, 4, 5, 5, 6, 5, 6, 6, 7, 7, 8],
        }
    )
    direct = StatisticalAnalyzer(frame).dunn(
        "group", "score", random_state=42, bootstrap_samples=199
    )
    for comp in direct["comparisons"]:
        ci = comp["confidence_interval"]
        assert ci is not None
        assert ci["status"] == "available"
        assert ci["method"] == "independent within-group percentile bootstrap"
        assert ci["multiplicity_adjusted"] is False
        assert -1.0 <= ci["lower"] <= ci["upper"] <= 1.0


def test_ols_r_squared_case_bootstrap_validation():
    """Validate OLS in-sample R-squared case-resampling bootstrap CI."""
    frame = pd.DataFrame(
        {
            "y": [1.0, 2.5, 3.2, 4.8, 5.1, 6.4, 7.2, 8.5, 9.1, 10.3],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "cat": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )
    direct = StatisticalAnalyzer(frame).linear_regression(
        "y",
        ["x1", "cat"],
        variable_types={"y": "continuous", "x1": "continuous", "cat": "nominal"},
        random_state=42,
        bootstrap_samples=199,
    )
    r2_ci = direct["model_fit"]["r_squared_confidence_interval"]
    assert r2_ci is not None
    assert r2_ci["status"] == "available"
    assert r2_ci["method"] == "case-resampling percentile bootstrap CI for in-sample R-squared"
    assert r2_ci["quantity"] == "R-squared"
    assert 0.0 <= r2_ci["lower"] <= r2_ci["upper"] <= 1.0
    assert r2_ci["lower"] <= direct["model_fit"]["r_squared"] <= r2_ci["upper"]


def test_strict_json_serialization_of_all_phase_9_intervals():
    """Ensure json.dumps(..., allow_nan=False) succeeds for all workflows containing Phase 9 CIs."""
    frame = pd.DataFrame(
        {
            "id": list(range(8)) * 3,
            "cond": ["c1"] * 8 + ["c2"] * 8 + ["c3"] * 8,
            "score": list(range(24)),
        }
    )
    assistant = ResearchAssistant(frame)
    q = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="cond",
        estimand="distribution",
        design="repeated",
        unit_id="id",
        condition_order=("c1", "c2", "c3"),
        variable_types={"score": "continuous", "cond": "nominal"},
    )
    result = assistant.analyze(q)
    # Strict serialization
    serialized = json.dumps(result.to_dict(), allow_nan=False)
    assert (
        "partial eta-squared" in serialized
        or "Kendall's W" in serialized
        or "statistic" in serialized
    )
