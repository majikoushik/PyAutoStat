import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pyautostat.analyzer as analyzer_module
from pyautostat import StatisticalAnalyzer


def test_normality_reports_all_three_tests(numeric_df):
    profile = StatisticalAnalyzer(numeric_df).analyze_all()
    normality = profile["normality"]

    for col in ("linear", "with_outlier"):
        tests = normality[col]
        assert {"shapiro_wilk", "d_agostino_pearson"} <= set(tests)
        for name in ("shapiro_wilk", "d_agostino_pearson"):
            assert 0.0 <= tests[name]["p_value"] <= 1.0
            assert tests[name]["is_normal"] in (True, False)
            assert tests[name]["status"] == (
                "not_rejected" if tests[name]["is_normal"] else "rejected"
            )
            assert tests[name]["reference_alpha"] == 0.05
        if "anderson_darling" in tests:
            assert len(tests["anderson_darling"]["critical_values"]) == len(
                tests["anderson_darling"]["significance_levels"]
            )
            assert min(tests["anderson_darling"]["critical_values"]) > 0
        else:
            assert any(
                warning["section"] == "normality"
                and warning["column"] == col
                and "critical-value grid" in warning["message"]
                for warning in profile["analysis_warnings"]
            )


@pytest.mark.parametrize(
    "message",
    [
        "kurtosistest only valid for n>=20 ... continuing anyway, n=10",
        "`kurtosistest` p-value may be inaccurate with fewer than 20 observations; "
        "only n=10 observations were given.",
    ],
)
def test_normaltest_retains_only_advisory_small_sample_warning(monkeypatch, message):
    original = analyzer_module.normaltest

    def advisory(values):
        warnings.warn(message, UserWarning, stacklevel=2)
        return original(values)

    monkeypatch.setattr(analyzer_module, "normaltest", advisory)
    profile = StatisticalAnalyzer(pd.DataFrame({"x": list(range(10))})).analyze_all()
    assert np.isfinite(profile["normality"]["x"]["d_agostino_pearson"]["statistic"])
    assert any(w["code"] == "small_sample_approximation" for w in profile["analysis_warnings"])


def test_normaltest_rejects_numerical_warning_and_nonfinite_output(monkeypatch):
    def unreliable(_values):
        warnings.warn("catastrophic cancellation", RuntimeWarning, stacklevel=2)
        return 2.0, 0.4

    monkeypatch.setattr(analyzer_module, "normaltest", unreliable)
    profile = StatisticalAnalyzer(pd.DataFrame({"x": list(range(10))})).analyze_all()
    assert "d_agostino_pearson" not in profile["normality"]["x"]
    assert any(w["code"] == "undefined_result" for w in profile["analysis_warnings"])

    monkeypatch.setattr(analyzer_module, "normaltest", lambda _values: (np.nan, 0.4))
    profile = StatisticalAnalyzer(pd.DataFrame({"x": list(range(10))})).analyze_all()
    assert "d_agostino_pearson" not in profile["normality"]["x"]


@pytest.mark.parametrize(
    "critical_values,significance_levels",
    [([-0.2, -0.1, 0.1], [15, 10, 5]), ([], []), ([0.3, 0.4], [15])],
)
def test_anderson_rejects_unusable_small_sample_critical_grid(
    monkeypatch, critical_values, significance_levels
):
    monkeypatch.setattr(
        analyzer_module.stats,
        "anderson",
        lambda _values: SimpleNamespace(
            statistic=0.1,
            critical_values=critical_values,
            significance_level=significance_levels,
        ),
    )
    profile = StatisticalAnalyzer(pd.DataFrame({"x": [1.0, 2.0, 4.0]})).analyze_all()
    assert "anderson_darling" not in profile["normality"]["x"]
    assert any(
        warning["section"] == "normality" and "critical-value grid" in warning["message"]
        for warning in profile["analysis_warnings"]
    )


def test_normality_skipped_for_columns_with_fewer_than_three_values():
    df = pd.DataFrame({"tiny": [1, 2]})
    normality = StatisticalAnalyzer(df).analyze_all()["normality"]
    assert normality == {}
