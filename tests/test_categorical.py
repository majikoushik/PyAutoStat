"""Categorical association and proportion effect-size boundaries."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from pyautostat import InsufficientDataError, InvalidTestError, ReportGenerator, StatisticalAnalyzer


@pytest.fixture
def two_by_two():
    return pd.DataFrame(
        {
            "group": ["A"] * 40 + ["B"] * 40,
            "outcome": ["yes"] * 30 + ["no"] * 10 + ["yes"] * 10 + ["no"] * 30,
        }
    )


def test_chi_square_and_effect_sizes_match_reference(two_by_two):
    analyzer = StatisticalAnalyzer(two_by_two)
    result = analyzer.categorical_association(
        "group", "outcome", success_value="yes", bootstrap_samples=100
    )
    expected = chi2_contingency([[30, 10], [10, 30]], correction=False)

    assert result["statistic"] == pytest.approx(expected.statistic)
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["degrees_of_freedom"] == 1
    assert result["observed_counts"] == [[30, 10], [10, 30]]
    assert result["effect_size"]["value"] == pytest.approx(0.5)
    assert result["cohens_h"]["value"] == pytest.approx(
        2 * (np.arcsin(np.sqrt(0.75)) - np.arcsin(np.sqrt(0.25)))
    )
    assert result["effect_size"]["confidence_interval"]["valid_resamples"] == 100
    assert result["cohens_h"]["confidence_interval"]["valid_resamples"] == 100
    report = ReportGenerator(analyzer.analyze_all(), hypothesis_results=result)
    assert "Cohen's h interval" in report.to_html()
    assert (
        analyzer.categorical_association(
            "group", "outcome", success_value="yes", bootstrap_samples=100
        )
        == result
    )


def test_multicategory_association_and_report(two_by_two):
    frame = pd.concat(
        [two_by_two, pd.DataFrame({"group": ["C"] * 40, "outcome": ["yes"] * 20 + ["no"] * 20})],
        ignore_index=True,
    )
    analyzer = StatisticalAnalyzer(frame)
    result = analyzer.categorical_association("group", "outcome", bootstrap_samples=0)
    assert len(result["groups"]) == 3
    assert "cohens_h" not in result
    assert result["effect_size"]["confidence_interval"] is None
    report = ReportGenerator(analyzer.analyze_all(), hypothesis_results=result)
    assert "Minimum expected cell count" in report.to_html()
    assert "Levene equal variance" not in report.to_html()
    assert "Cramer's V" in report.to_json()


def test_categorical_association_rejects_sparse_and_degenerate_data(two_by_two):
    sparse = pd.DataFrame(
        {"group": ["A"] * 6 + ["B"] * 6, "outcome": ["yes"] * 5 + ["no"] + ["yes"] + ["no"] * 5}
    )
    with pytest.raises(InsufficientDataError, match="expected counts"):
        StatisticalAnalyzer(sparse).categorical_association("group", "outcome")

    one_outcome = two_by_two.assign(outcome="yes")
    with pytest.raises(InsufficientDataError, match="2 observed categories"):
        StatisticalAnalyzer(one_outcome).categorical_association("group", "outcome")

    missing = two_by_two.copy()
    missing.loc[0, "outcome"] = None
    result = StatisticalAnalyzer(missing).categorical_association(
        "group", "outcome", bootstrap_samples=0
    )
    assert result["sample_size"] == 79


def test_categorical_association_rejects_invalid_requests(two_by_two):
    analyzer = StatisticalAnalyzer(two_by_two)
    for kwargs in (
        {"success_value": "unknown"},
        {"success_value": pd.NA},
        {"success_value": ["yes"]},
        {"bootstrap_samples": 99},
        {"confidence_level": 1},
        {"random_state": -1},
    ):
        with pytest.raises(InvalidTestError):
            analyzer.categorical_association("group", "outcome", **kwargs)
    with pytest.raises(InvalidTestError, match="different"):
        analyzer.categorical_association("group", "group")
