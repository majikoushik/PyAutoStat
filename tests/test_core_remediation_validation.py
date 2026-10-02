"""Scientific validation tests for core statistical optimizations and remediation policies.

Covers:
1. Section 12A: Wilcoxon large-sample asymptotic route and contracts.
2. Section 12B: Kendall's W participant-block bootstrap reference equivalence.
3. Section 12C: Bootstrap valid-resample threshold policy boundary transitions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from pyautostat.inference import paired_wilcoxon
from pyautostat.uncertainty import (
    friedman_kendall_w_bootstrap_ci,
    matched_pairs_rank_biserial_bootstrap_ci,
)


def _make_paired_df(first_vals: np.ndarray, second_vals: np.ndarray) -> pd.DataFrame:
    n = len(first_vals)
    return pd.DataFrame(
        {
            "unit": [f"u_{i:04d}" for i in range(n)] * 2,
            "cond": ["A"] * n + ["B"] * n,
            "val": np.concatenate([first_vals, second_vals]),
        }
    )


# =============================================================================
# 12A: WILCOXON LARGE-SAMPLE ROUTE VALIDATION
# =============================================================================


def test_wilcoxon_small_sample_no_ties() -> None:
    """Small sample without ties (n <= 50) uses auto routing matching SciPy."""
    rng = np.random.default_rng(42)
    first = rng.normal(loc=1.5, scale=1.0, size=20)
    second = rng.normal(loc=1.0, scale=1.0, size=20)
    diffs = first - second
    assert len(np.unique(diffs)) == 20

    df = _make_paired_df(first, second)
    res = paired_wilcoxon(df, "unit", "cond", "val", ("A", "B"))
    ref = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")

    assert math.isclose(float(res["statistic"]), float(ref.statistic), rel_tol=1e-9)
    assert math.isclose(float(res["p_value"]), float(ref.pvalue), rel_tol=1e-9)


def test_wilcoxon_small_sample_with_ties_and_zeros() -> None:
    """Small sample with ties and zero differences."""
    first = np.array([1.0, 2.0, 2.0, 1.0, 5.0, 3.0, 2.0, 6.0, 6.0, 4.0])
    second = np.array([1.0, 1.0, 1.0, 3.0, 2.0, 3.0, 3.0, 2.0, 2.0, 2.0])
    diffs = first - second

    df = _make_paired_df(first, second)
    res = paired_wilcoxon(df, "unit", "cond", "val", ("A", "B"))

    ref = stats.wilcoxon(diffs, zero_method="wilcox", correction=False, alternative="two-sided")

    assert math.isclose(float(res["statistic"]), float(ref.statistic), rel_tol=1e-9)
    assert math.isclose(float(res["p_value"]), float(ref.pvalue), rel_tol=1e-9)


def test_wilcoxon_large_sample_asymptotic_agreement() -> None:
    """Large sample (n > 50) selects asymptotic normal approximation ('approx')."""
    rng = np.random.default_rng(123)
    first = rng.normal(loc=5.3, scale=1.5, size=200)
    second = rng.normal(loc=5.0, scale=1.5, size=200)
    diffs = first - second

    df = _make_paired_df(first, second)
    res = paired_wilcoxon(df, "unit", "cond", "val", ("A", "B"))
    ref = stats.wilcoxon(
        diffs,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="approx",
    )

    assert math.isclose(float(res["statistic"]), float(ref.statistic), rel_tol=1e-9)
    assert math.isclose(float(res["p_value"]), float(ref.pvalue), rel_tol=1e-9)


def test_wilcoxon_zero_heavy_paired_data() -> None:
    """Zero-heavy paired data omits zeros via zero_method='wilcox'."""
    zeros = np.zeros(80)
    nonzeros = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -0.5] * 10)
    diffs = np.concatenate([zeros, nonzeros])
    first = diffs + 10.0
    second = np.full_like(diffs, 10.0)

    df = _make_paired_df(first, second)
    res = paired_wilcoxon(df, "unit", "cond", "val", ("A", "B"))
    ref = stats.wilcoxon(
        nonzeros,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="auto",
    )

    assert math.isclose(float(res["statistic"]), float(ref.statistic), rel_tol=1e-9)
    assert math.isclose(float(res["p_value"]), float(ref.pvalue), rel_tol=1e-9)


def test_wilcoxon_condition_order_reversal_and_effect_orientation() -> None:
    """Reversing condition order reverses sign of rank-biserial correlation."""
    rng = np.random.default_rng(99)
    cond_a = rng.normal(loc=10.0, scale=2.0, size=60)
    cond_b = rng.normal(loc=8.0, scale=2.0, size=60)

    df = _make_paired_df(cond_a, cond_b)
    res_ab = paired_wilcoxon(df, "unit", "cond", "val", ("A", "B"))
    res_ba = paired_wilcoxon(df, "unit", "cond", "val", ("B", "A"))

    assert res_ab["p_value"] == pytest.approx(res_ba["p_value"], rel=1e-9)
    eff_ab = res_ab["effect_size"]["value"]
    eff_ba = res_ba["effect_size"]["value"]
    assert eff_ab == pytest.approx(-eff_ba, rel=1e-9)
    assert eff_ab > 0.0


# =============================================================================
# 12B: KENDALL'S W BOOTSTRAP EQUIVALENCE (REFERENCE VS OPTIMIZED)
# =============================================================================


def _reference_kendall_w_bootstrap_ci(
    matrix: np.ndarray,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 499,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Test reference implementation using unoptimized stats.friedmanchisquare."""
    n, k = matrix.shape
    seed = 0 if random_state is None else random_state
    rng = np.random.default_rng(seed)
    reps: list[float] = []

    for _ in range(bootstrap_samples):
        indices = rng.integers(0, n, size=n)
        sample = matrix[indices, :]
        try:
            q_b, _ = stats.friedmanchisquare(*[sample[:, j] for j in range(k)])
            w_b = float(q_b / (n * (k - 1)))
            if math.isfinite(w_b) and 0.0 <= w_b <= 1.0:
                reps.append(w_b)
        except Exception:
            continue

    min_valid = max(10, bootstrap_samples // 2)
    if len(reps) < min_valid:
        return {"status": "unavailable", "valid_resamples": len(reps), "reps": reps}

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "status": "available",
        "lower": float(lower),
        "upper": float(upper),
        "valid_resamples": len(reps),
        "reps": reps,
    }


@pytest.mark.parametrize(
    "matrix_case",
    [
        "no_ties",
        "moderate_ties",
        "heavy_ties",
        "zero_heavy",
        "small_n",
        "medium_n",
    ],
)
def test_kendall_w_bootstrap_reference_equivalence(matrix_case: str) -> None:
    """Optimized Kendall's W bootstrap must be equivalent to reference implementation."""
    rng = np.random.default_rng(1234)

    if matrix_case == "no_ties":
        matrix = rng.normal(size=(30, 4))
    elif matrix_case == "moderate_ties":
        matrix = rng.integers(1, 6, size=(30, 4)).astype(float)
    elif matrix_case == "heavy_ties":
        matrix = rng.integers(0, 2, size=(40, 3)).astype(float)
    elif matrix_case == "zero_heavy":
        base = rng.normal(size=(35, 4))
        mask = rng.uniform(size=(35, 4)) < 0.6
        base[mask] = 0.0
        matrix = base
    elif matrix_case == "small_n":
        matrix = rng.uniform(size=(5, 3))
    elif matrix_case == "medium_n":
        matrix = rng.uniform(size=(50, 4))
    else:
        raise ValueError(matrix_case)

    seed = 42
    b_samples = 100

    optimized = friedman_kendall_w_bootstrap_ci(
        matrix, bootstrap_samples=b_samples, random_state=seed
    )
    reference = _reference_kendall_w_bootstrap_ci(
        matrix, bootstrap_samples=b_samples, random_state=seed
    )

    assert optimized["status"] == reference["status"]
    assert optimized["valid_resamples"] == reference["valid_resamples"]

    if optimized["status"] == "available":
        assert math.isclose(
            float(optimized["lower"]), float(reference["lower"]), rel_tol=1e-9, abs_tol=1e-9
        )
        assert math.isclose(
            float(optimized["upper"]), float(reference["upper"]), rel_tol=1e-9, abs_tol=1e-9
        )
        assert 0.0 <= optimized["lower"] <= optimized["upper"] <= 1.0


def test_kendall_w_bootstrap_reproducibility() -> None:
    """Identical seed produces identical intervals; different seed produces distinct samples."""
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 2.0, 1.0],
            [1.0, 3.0, 2.0],
            [2.0, 3.0, 1.0],
        ]
    )
    ci1 = friedman_kendall_w_bootstrap_ci(matrix, bootstrap_samples=50, random_state=101)
    ci2 = friedman_kendall_w_bootstrap_ci(matrix, bootstrap_samples=50, random_state=101)
    ci3 = friedman_kendall_w_bootstrap_ci(matrix, bootstrap_samples=50, random_state=202)

    assert ci1["lower"] == ci2["lower"]
    assert ci1["upper"] == ci2["upper"]
    assert ci1["lower"] != ci3["lower"] or ci1["upper"] != ci3["upper"]


# =============================================================================
# 12C: VALID-RESAMPLE THRESHOLD POLICY BOUNDARY TESTS
# =============================================================================


@pytest.mark.parametrize("requested", [1, 10, 19, 20, 50, 99, 100, 199])
def test_valid_resample_threshold_boundaries_kendall_w(requested: int) -> None:
    """Verify available/unavailable transitions across bootstrap sample sizes."""
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 4.0],
            [2.0, 3.0, 5.0],
            [1.0, 2.0, 4.0],
        ]
    )

    expected_min_valid = max(10, requested // 2)
    ci = friedman_kendall_w_bootstrap_ci(matrix, bootstrap_samples=requested, random_state=7)

    if requested < 10:
        assert ci["status"] == "unavailable"
        assert f"minimum {expected_min_valid}" in ci["reason"]
    else:
        assert ci["valid_resamples"] == requested
        assert ci["status"] == "available"
        assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0


@pytest.mark.parametrize("requested", [1, 10, 19, 20, 50, 99, 100, 199])
def test_valid_resample_threshold_boundaries_rank_biserial(requested: int) -> None:
    """Verify rank-biserial valid resample threshold boundaries."""
    diffs = np.array([1.5, -0.5, 2.0, 3.0, -1.0, 2.5, 1.0, -0.8, 1.2, 0.9])
    expected_min_valid = max(10, requested // 2)

    ci = matched_pairs_rank_biserial_bootstrap_ci(
        diffs, bootstrap_samples=requested, random_state=7
    )

    if requested < 10:
        assert ci["status"] == "unavailable"
        assert f"minimum {expected_min_valid}" in ci["reason"]
    else:
        assert ci["valid_resamples"] == requested
        assert ci["status"] == "available"
        assert -1.0 <= ci["lower"] <= ci["upper"] <= 1.0
