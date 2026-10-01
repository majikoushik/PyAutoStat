"""Tests for Intraclass Correlation Coefficient (ICC) analytical confidence intervals.

Validates:
- Exact F-inversion analytical intervals for ICC(1,1), ICC(1,k), ICC(3,1), ICC(3,k).
- Satterthwaite effective df analytical intervals for ICC(2,1) and ICC(2,k).
- Invariant bounds ordering (lower <= upper) across confidence levels (0.90, 0.95, 0.99).
- Interval widening as confidence level increases.
- Non-clamping of negative lower bounds.
- Point estimate containment within interval.
- Invalid confidence level parameter validation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pyautostat.exceptions import InvalidDataError
from pyautostat.icc import (
    ICC_VARIANTS,
    compute_icc_ci,
    compute_icc_estimate,
    icc_anova_components,
    icc_panel,
    intraclass_correlation,
)


@pytest.fixture
def panel_data() -> pd.DataFrame:
    """Standard 8 targets, 3 raters panel."""
    targets = [f"T{i}" for i in range(1, 9)]
    ratings = [
        [7.0, 6.0, 8.0],
        [4.0, 5.0, 4.0],
        [9.0, 8.0, 9.0],
        [3.0, 2.0, 3.0],
        [6.0, 7.0, 6.0],
        [8.0, 8.0, 7.0],
        [5.0, 4.0, 6.0],
        [7.0, 6.0, 7.0],
    ]
    rows = []
    for t_idx, t in enumerate(targets):
        for r_idx, val in enumerate(ratings[t_idx]):
            rows.append({"target": t, "rater": f"R{r_idx + 1}", "score": val})
    return pd.DataFrame(rows)


def test_analytical_ci_bounds_ordering_all_variants(panel_data: pd.DataFrame):
    """Every variant must produce lower <= upper with status='available'."""
    for variant in ("icc_1_1", "icc_1_k", "icc_2_1", "icc_2_k", "icc_3_1", "icc_3_k"):
        meta = ICC_VARIANTS[variant]
        res = intraclass_correlation(
            panel_data,
            target="target",
            rater="rater",
            outcome="score",
            model=meta["model"],
            definition=meta["definition"],
            unit=meta["unit"],
            confidence_level=0.95,
        )
        ci = res["confidence_interval"]
        assert ci["status"] == "available"
        assert isinstance(ci["lower"], float)
        assert isinstance(ci["upper"], float)
        assert ci["lower"] <= ci["upper"]
        assert ci["lower"] <= res["estimate"] <= ci["upper"]
        assert ci["level"] == 0.95


def test_ci_widens_with_higher_confidence_level(panel_data: pd.DataFrame):
    """Higher confidence level must produce strictly wider interval."""
    for variant in ("icc_1_1", "icc_2_1", "icc_3_1"):
        meta = ICC_VARIANTS[variant]
        ci_90 = intraclass_correlation(
            panel_data,
            target="target",
            rater="rater",
            outcome="score",
            model=meta["model"],
            definition=meta["definition"],
            unit=meta["unit"],
            confidence_level=0.90,
        )["confidence_interval"]

        ci_95 = intraclass_correlation(
            panel_data,
            target="target",
            rater="rater",
            outcome="score",
            model=meta["model"],
            definition=meta["definition"],
            unit=meta["unit"],
            confidence_level=0.95,
        )["confidence_interval"]

        ci_99 = intraclass_correlation(
            panel_data,
            target="target",
            rater="rater",
            outcome="score",
            model=meta["model"],
            definition=meta["definition"],
            unit=meta["unit"],
            confidence_level=0.99,
        )["confidence_interval"]

        width_90 = ci_90["upper"] - ci_90["lower"]
        width_95 = ci_95["upper"] - ci_95["lower"]
        width_99 = ci_99["upper"] - ci_99["lower"]

        assert width_90 < width_95 < width_99


def test_negative_lower_bound_preserved_not_clamped():
    """When analytical F-inversion yields a negative lower bound, it is preserved."""
    # Data with moderate target variation and noticeable noise
    rows = []
    ratings = [
        [3.0, 4.0, 5.0],
        [4.0, 3.0, 4.0],
        [3.0, 5.0, 3.0],
        [5.0, 4.0, 5.0],
    ]
    for t_idx, row in enumerate(ratings):
        for r_idx, val in enumerate(row):
            rows.append({"target": f"T{t_idx}", "rater": f"R{r_idx}", "score": val})
    df = pd.DataFrame(rows)

    res = intraclass_correlation(
        df,
        target="target",
        rater="rater",
        outcome="score",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
        confidence_level=0.99,
    )
    ci = res["confidence_interval"]
    assert ci["status"] == "available"
    # The lower bound should be negative and NOT clamped to zero
    assert ci["lower"] < 0.0
    assert ci["lower"] <= ci["upper"]


def test_compute_icc_ci_satterthwaite_effective_df(panel_data: pd.DataFrame):
    """Verify Satterthwaite effective degrees of freedom computation for ICC(2,1)."""
    panel = icc_panel(panel_data, "target", "rater", "score")
    anova = icc_anova_components(panel["matrix"])
    n = panel["n_targets"]
    k = panel["n_raters"]
    est = compute_icc_estimate("icc_2_1", n, k, anova)

    ci = compute_icc_ci("icc_2_1", n, k, anova, est, 0.95)
    assert ci["status"] == "available"
    assert ci["method"] == "satterthwaite_f_inversion"
    assert ci["lower"] < est < ci["upper"]


def test_invalid_confidence_level_rejected(panel_data: pd.DataFrame):
    for bad_conf in (0.0, 1.0, -0.05, 1.05, float("nan"), float("inf")):
        with pytest.raises(InvalidDataError, match="confidence_level must be a finite number"):
            intraclass_correlation(
                panel_data,
                target="target",
                rater="rater",
                outcome="score",
                confidence_level=bad_conf,
            )
