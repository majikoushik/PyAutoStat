import pytest

from pyautostat.analyzer import StatisticalAnalyzer


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("cohens_d", 0.1, "negligible"),
        ("cohens_d", 0.3, "small"),
        ("cohens_d", 0.6, "medium"),
        ("cohens_d", 1.0, "large"),
        ("eta_squared", 0.005, "negligible"),
        ("eta_squared", 0.02, "small"),
        ("eta_squared", 0.10, "medium"),
        ("eta_squared", 0.20, "large"),
        ("correlation", 0.05, "negligible"),
        ("correlation", 0.2, "small"),
        ("correlation", 0.4, "medium"),
        ("correlation", 0.7, "large"),
    ],
)
def test_effect_size_interpretation_bands(name, value, expected):
    assert StatisticalAnalyzer._interpret_effect_size(name, value) == expected


def test_effect_size_interpretation_uses_magnitude_for_negative_values():
    assert StatisticalAnalyzer._interpret_effect_size("cohens_d", -1.0) == "large"
