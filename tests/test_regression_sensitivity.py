"""Tests for integrated linear and logistic regression sensitivity analysis."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pyautostat import (
    Comparability,
    ResearchAssistant,
    SensitivityScenario,
    SensitivitySpecification,
    SensitivityStatus,
)
from pyautostat.sensitivity import (
    _compare_same_estimand_regression,
    classify_comparability,
)


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    n = 30
    x = rng.normal(10, 2, size=n)
    z = rng.normal(5, 1, size=n)
    noise = rng.normal(0, 0.5, size=n)
    y = 2.0 + 1.5 * x - 0.8 * z + noise
    cat = np.array(["A", "B", "C"] * 10)
    # binary outcome for logistic
    prob = 1 / (1 + np.exp(-(0.5 * x - 5.0)))
    binary_y = (rng.uniform(0, 1, size=n) < prob).astype(int)

    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "z": z,
            "group": cat,
            "binary_y": binary_y,
        }
    )


def test_linear_regression_classical_vs_hc3_sensitivity(regression_data):
    assistant = ResearchAssistant(regression_data)
    workflow = assistant.run(
        objective="regression",
        outcome="y",
        predictors=["x", "z"],
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "continuous", "x": "continuous", "z": "continuous"},
        covariance_type="classical",
    )
    base = workflow.analysis
    assert base is not None

    hc3_spec = replace(
        base.specification,
        options=replace(base.specification.options, covariance_type="HC3"),
    )
    scenario = SensitivitySpecification("HC3 Robust SE", hc3_spec, method_id="linear_regression")

    result = assistant.sensitivity_analysis(base, scenarios=[scenario])

    assert result.status == SensitivityStatus.COMPLETE
    item = result.scenario_results[0]
    assert item.status.value == "completed"
    assert item.comparability == Comparability.SAME_ESTIMAND
    assert item.method_id == "linear_regression"

    # Regression results must carry structured term estimates, not a fake scalar
    assert item.primary_estimate is None
    assert len(item.estimates) == 3  # Intercept, x, z

    terms = {est.key: est for est in item.estimates}
    assert "Intercept" in terms
    assert "x" in terms
    assert "z" in terms

    for _key, est in terms.items():
        assert est.quantity == "regression_coefficient"
        assert est.estimate is not None
        assert est.standard_error is not None
        assert est.p_value is not None
        assert est.confidence_interval is not None
        assert est.metadata.get("covariance_type") == "HC3"

    # For classical vs HC3, OLS point estimates are identical
    comp = item.comparison
    assert comp is not None
    assert comp["available"] is True
    assert comp["comparison_quantity"] == "coefficient_vector"
    assert comp["all_matched_terms_sign_consistent"] is True
    assert comp["terms"]["x"]["estimate_difference"] == pytest.approx(0.0, abs=1e-10)
    assert comp["terms"]["z"]["estimate_difference"] == pytest.approx(0.0, abs=1e-10)

    # Standard errors and intervals differ between classical and HC3
    base_coeffs = {c["term"]: c for c in base.values["coefficients"]}
    assert comp["terms"]["x"]["base_standard_error"] == pytest.approx(
        base_coeffs["x"]["standard_error"]
    )
    assert comp["terms"]["x"]["scenario_standard_error"] == pytest.approx(terms["x"].standard_error)

    # Check compare text and verdict
    compare_text = result.compare()
    assert "Primary (Ordinary least-squares linear regression):" in compare_text
    assert "Scenario 'HC3 Robust SE'" in compare_text
    assert "completed same-estimand scenario" in compare_text
    assert "CONSISTENT" in result.verdict


def test_linear_regression_sensitivity_preserves_term_names(regression_data):
    assistant = ResearchAssistant(regression_data)
    workflow = assistant.run(
        objective="regression",
        outcome="y",
        predictors=["z", "x"],
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "continuous", "z": "continuous", "x": "continuous"},
    )
    base = workflow.analysis
    hc3_spec = replace(
        base.specification,
        options=replace(base.specification.options, covariance_type="HC3"),
    )
    scenario = SensitivitySpecification("hc3", hc3_spec, method_id="linear_regression")

    result = assistant.sensitivity_analysis(base, scenarios=[scenario])
    item = result.scenario_results[0]

    base_term_names = [c["term"] for c in base.values["coefficients"]]
    scenario_term_names = [est.key for est in item.estimates]
    assert scenario_term_names == base_term_names
    assert scenario_term_names == ["Intercept", "z", "x"]


def test_linear_regression_same_model_is_same_estimand(regression_data):
    assistant = ResearchAssistant(regression_data)
    workflow = assistant.run(
        objective="regression",
        outcome="y",
        predictors=["x"],
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "continuous", "x": "continuous"},
    )
    base = workflow.analysis
    scenario_analysis = assistant.run(
        objective="regression",
        outcome="y",
        predictors=["x"],
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "continuous", "x": "continuous"},
        covariance_type="HC3",
    ).analysis

    assert classify_comparability(base, scenario_analysis) == Comparability.SAME_ESTIMAND


def test_linear_regression_changed_predictors_are_different_estimand(regression_data):
    assistant = ResearchAssistant(regression_data)
    base_wf = assistant.run(
        objective="regression",
        outcome="y",
        predictors=["x"],
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "continuous", "x": "continuous"},
    )
    base = base_wf.analysis

    spec_with_controls = replace(
        base.specification,
        question=replace(base.specification.question, predictors=("x", "z")),
    )
    scenario = SensitivitySpecification(
        "add z control", spec_with_controls, method_id="linear_regression"
    )

    result = assistant.sensitivity_analysis(base, scenarios=[scenario])

    assert result.status == SensitivityStatus.COMPLETE
    item = result.scenario_results[0]
    assert item.status.value == "completed"
    assert item.comparability == Comparability.DIFFERENT_ESTIMAND
    assert item.comparison is None
    assert any("different estimand" in w.lower() for w in item.warnings)
    assert len(item.estimates) == 3


def test_linear_regression_changed_reference_level_not_silently_aligned(regression_data):
    assistant = ResearchAssistant(regression_data)
    base_wf = assistant.run(
        objective="regression",
        outcome="y",
        predictors=["group"],
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "continuous", "group": "nominal"},
        reference_levels={"group": "A"},
    )
    base = base_wf.analysis

    changed_ref_spec = replace(
        base.specification,
        options=replace(base.specification.options, reference_levels={"group": "B"}),
    )
    scenario = SensitivitySpecification(
        "change reference level", changed_ref_spec, method_id="linear_regression"
    )

    result = assistant.sensitivity_analysis(base, scenarios=[scenario])
    item = result.scenario_results[0]

    # Changing categorical reference level changes coefficient definitions
    assert item.comparability == Comparability.DIFFERENT_ESTIMAND
    assert item.status.value == "completed"


def test_linear_regression_missing_term_is_reported(regression_data):
    assistant = ResearchAssistant(regression_data)
    wf_base = assistant.run(
        objective="regression",
        outcome="y",
        predictors=["x", "z"],
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "continuous", "x": "continuous", "z": "continuous"},
    )
    wf_scenario = assistant.run(
        objective="regression",
        outcome="y",
        predictors=["x"],
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "continuous", "x": "continuous"},
    )

    comp = _compare_same_estimand_regression(wf_base.analysis, wf_scenario.analysis)
    assert comp["available"] is True
    assert "z" in comp["unmatched_base_terms"]
    assert comp["terms"]["z"]["comparison_available"] is False
    assert "absent from scenario" in comp["terms"]["z"]["reason"].lower()
    assert "x" in comp["matched_terms"]


def test_regression_structured_estimates_serialize_deterministically(regression_data):
    assistant = ResearchAssistant(regression_data)
    wf = assistant.run(
        objective="regression",
        outcome="y",
        predictors=["x"],
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "continuous", "x": "continuous"},
    )
    base = wf.analysis
    hc3_spec = replace(
        base.specification,
        options=replace(base.specification.options, covariance_type="HC3"),
    )
    scenario = SensitivitySpecification("hc3", hc3_spec, method_id="linear_regression")
    result = assistant.sensitivity_analysis(base, scenarios=[scenario])

    payload = result.to_dict()
    assert payload["schema_version"] == 2
    assert "estimates" in payload["scenario_results"][0]
    estimates_data = payload["scenario_results"][0]["estimates"]
    assert len(estimates_data) == 2
    assert estimates_data[0]["key"] == "Intercept"
    assert estimates_data[0]["quantity"] == "regression_coefficient"
    assert estimates_data[1]["key"] == "x"

    # Deterministic JSON roundtrip without nonfinite values or models
    json_str = result.to_json()
    assert "DataFrame" not in json_str
    assert "RegressionResults" not in json_str
    parsed = json.loads(json_str)
    assert parsed == payload


def test_logistic_regression_sensitivity(regression_data):
    assistant = ResearchAssistant(regression_data)
    wf = assistant.run(
        objective="regression",
        outcome="binary_y",
        predictors=["x"],
        estimand="event_probability",
        event_level=1,
        design="independent",
        variable_types={"binary_y": "nominal", "x": "continuous"},
        covariance_type="classical",
    )
    base = wf.analysis
    assert base is not None
    assert base.method_id == "logistic_regression"

    hc3_spec = replace(
        base.specification,
        options=replace(base.specification.options, covariance_type="HC3"),
    )
    scenario = SensitivitySpecification("HC3 Logistic", hc3_spec, method_id="logistic_regression")

    result = assistant.sensitivity_analysis(base, scenarios=[scenario])
    assert result.status == SensitivityStatus.COMPLETE
    item = result.scenario_results[0]
    assert item.status.value == "completed"
    assert item.comparability == Comparability.SAME_ESTIMAND
    assert len(item.estimates) == 2


def test_logistic_regression_preserves_log_odds_coefficient_identity(regression_data):
    assistant = ResearchAssistant(regression_data)
    wf = assistant.run(
        objective="regression",
        outcome="binary_y",
        predictors=["x"],
        estimand="event_probability",
        event_level=1,
        design="independent",
        variable_types={"binary_y": "nominal", "x": "continuous"},
    )
    base = wf.analysis
    scenario = SensitivityScenario("repeat", base.specification, method_id="logistic_regression")

    result = assistant.sensitivity_analysis(base, scenarios=[scenario])
    item = result.scenario_results[0]

    for est in item.estimates:
        # Canonical coefficient quantity is log-odds coefficient, not odds ratio
        assert est.quantity == "log_odds_coefficient"
        # Odds ratio is retained in metadata
        assert "odds_ratio" in est.metadata
        assert "odds_ratio_ci" in est.metadata
