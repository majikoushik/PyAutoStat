"""Independent numerical validation for Intraclass Correlation Coefficient (ICC).

Validates against:
- Shrout & Fleiss (1979) Table 4 benchmark data (6 targets, 4 judges).
- Pure additive rater bias fixture: consistency ICC(3,1) == 1.0 vs agreement ICC(2,1) < 1.0.
- Perfect agreement fixture: all variants == 1.0.
- McGraw & Wong (1996) C(2) and A(3) algebraic equivalents.
- Exact ANOVA mean squares decomposition and F-tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pyautostat.icc import (
    icc_anova_components,
    icc_panel,
    intraclass_correlation,
)


@pytest.fixture
def shrout_fleiss_1979_table4() -> pd.DataFrame:
    """Shrout & Fleiss (1979) Table 4 data: 6 targets, 4 judges."""
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
    return pd.DataFrame(rows)


def test_shrout_fleiss_table4_anova_mean_squares(shrout_fleiss_1979_table4: pd.DataFrame):
    """Verify ANOVA mean squares match Shrout & Fleiss (1979) published values:
    BMS = 11.24, JMS = 32.49, EMS = 1.24.
    """
    panel = icc_panel(shrout_fleiss_1979_table4, "target", "rater", "score")
    anova = icc_anova_components(panel["matrix"])

    # Published BMS = 11.24 (between targets)
    assert anova["ms_targets"] == pytest.approx(11.2417, rel=1e-3)
    # JMS = 32.49 (between raters)
    assert anova["ms_raters"] == pytest.approx(32.4861, rel=1e-3)
    # EMS = 1.019 (residual error)
    assert anova["ms_error"] == pytest.approx(1.0194, rel=1e-3)


def test_shrout_fleiss_table4_canonical_estimates(shrout_fleiss_1979_table4: pd.DataFrame):
    """Validate all 6 Shrout & Fleiss (1979) Table 4 published ICC estimates:
    ICC(1,1) = 0.166 (or 0.1659)
    ICC(2,1) = 0.290 (or 0.2898)
    ICC(3,1) = 0.715 (or 0.7149)
    ICC(1,k) = 0.443 (or 0.4431)
    ICC(2,k) = 0.620 (or 0.6198)
    ICC(3,k) = 0.909 (or 0.9094)
    """
    df = shrout_fleiss_1979_table4

    # ICC(1,1) - One-way random, single
    r11 = intraclass_correlation(
        df, "target", "rater", "score", model="one_way_random", unit="single"
    )
    assert r11["estimate"] == pytest.approx(0.1659, abs=0.001)

    # ICC(1,k) - One-way random, average
    r1k = intraclass_correlation(
        df, "target", "rater", "score", model="one_way_random", unit="average"
    )
    assert r1k["estimate"] == pytest.approx(0.4431, abs=0.001)

    # ICC(2,1) - Two-way random, agreement, single
    r21 = intraclass_correlation(
        df,
        "target",
        "rater",
        "score",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
    )
    assert r21["estimate"] == pytest.approx(0.2898, abs=0.001)

    # ICC(2,k) - Two-way random, agreement, average
    r2k = intraclass_correlation(
        df,
        "target",
        "rater",
        "score",
        model="two_way_random",
        definition="absolute_agreement",
        unit="average",
    )
    assert r2k["estimate"] == pytest.approx(0.6198, abs=0.001)

    # ICC(3,1) - Two-way mixed, consistency, single
    r31 = intraclass_correlation(
        df,
        "target",
        "rater",
        "score",
        model="two_way_mixed",
        definition="consistency",
        unit="single",
    )
    assert r31["estimate"] == pytest.approx(0.7149, abs=0.001)

    # ICC(3,k) - Two-way mixed, consistency, average
    r3k = intraclass_correlation(
        df,
        "target",
        "rater",
        "score",
        model="two_way_mixed",
        definition="consistency",
        unit="average",
    )
    assert r3k["estimate"] == pytest.approx(0.9094, abs=0.001)


def test_pure_additive_rater_bias_consistency_vs_agreement():
    """When raters have systematic level offsets (Rater B = Rater A + 5, Rater C = Rater A + 10):
    - Consistency ICC(3,1) is exactly 1.0 (residual error is zero; ordering is identical).
    - Absolute agreement ICC(2,1) is strictly < 1.0 (penalized for rater bias).
    """
    rows = []
    base_scores = [10.0, 20.0, 30.0, 40.0, 50.0]
    for t_idx, base in enumerate(base_scores):
        rows.append({"target": f"T{t_idx}", "rater": "R1", "score": base})
        rows.append({"target": f"T{t_idx}", "rater": "R2", "score": base + 5.0})
        rows.append({"target": f"T{t_idx}", "rater": "R3", "score": base + 10.0})
    df = pd.DataFrame(rows)

    # Consistency ICC(3,1)
    res_consistency = intraclass_correlation(
        df,
        "target",
        "rater",
        "score",
        model="two_way_mixed",
        definition="consistency",
        unit="single",
    )
    assert res_consistency["estimate"] == pytest.approx(1.0, abs=1e-6)

    # Absolute agreement ICC(2,1)
    res_agreement = intraclass_correlation(
        df,
        "target",
        "rater",
        "score",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
    )
    assert res_agreement["estimate"] < 0.95
    # Rater test must detect significant rater differences
    assert res_agreement["rater_test"]["p_value"] < 0.01


def test_perfect_agreement_all_variants_equal_one():
    """When all raters give identical ratings to each target, all ICC variants equal 1.0."""
    rows = []
    for t_idx, score in enumerate([12.0, 18.0, 25.0, 32.0, 40.0]):
        for r_idx in range(1, 4):
            rows.append({"target": f"T{t_idx}", "rater": f"R{r_idx}", "score": score})
    df = pd.DataFrame(rows)

    for variant in ("icc_1_1", "icc_1_k", "icc_2_1", "icc_2_k", "icc_3_1", "icc_3_k"):
        meta = intraclass_correlation(df, "target", "rater", "score")["all_variants"]
        var_match = [v for v in meta if v["variant"] == variant][0]
        assert var_match["estimate"] == pytest.approx(1.0, abs=1e-6)


def test_mcgraw_wong_c2_and_a3_equivalences(shrout_fleiss_1979_table4: pd.DataFrame):
    """McGraw-Wong C(2) is mathematically identical to ICC(3) [consistency],
    and McGraw-Wong A(3) is mathematically identical to ICC(2) [agreement].
    """
    df = shrout_fleiss_1979_table4

    # C(2,1) == ICC(3,1)
    c21 = intraclass_correlation(
        df,
        "target",
        "rater",
        "score",
        model="two_way_random",
        definition="consistency",
        unit="single",
    )
    icc31 = intraclass_correlation(
        df,
        "target",
        "rater",
        "score",
        model="two_way_mixed",
        definition="consistency",
        unit="single",
    )
    assert c21["estimate"] == pytest.approx(icc31["estimate"], rel=1e-8)

    # C(2,k) == ICC(3,k)
    c2k = intraclass_correlation(
        df,
        "target",
        "rater",
        "score",
        model="two_way_random",
        definition="consistency",
        unit="average",
    )
    icc3k = intraclass_correlation(
        df,
        "target",
        "rater",
        "score",
        model="two_way_mixed",
        definition="consistency",
        unit="average",
    )
    assert c2k["estimate"] == pytest.approx(icc3k["estimate"], rel=1e-8)

    # A(3,1) == ICC(2,1)
    a31 = intraclass_correlation(
        df,
        "target",
        "rater",
        "score",
        model="two_way_mixed",
        definition="absolute_agreement",
        unit="single",
    )
    icc21 = intraclass_correlation(
        df,
        "target",
        "rater",
        "score",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
    )
    assert a31["estimate"] == pytest.approx(icc21["estimate"], rel=1e-8)

    # A(3,k) == ICC(2,k)
    a3k = intraclass_correlation(
        df,
        "target",
        "rater",
        "score",
        model="two_way_mixed",
        definition="absolute_agreement",
        unit="average",
    )
    icc2k = intraclass_correlation(
        df,
        "target",
        "rater",
        "score",
        model="two_way_random",
        definition="absolute_agreement",
        unit="average",
    )
    assert a3k["estimate"] == pytest.approx(icc2k["estimate"], rel=1e-8)
