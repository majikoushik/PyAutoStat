"""Complete OLS workflow, diagnostics, safety, and integration tests."""

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from pyautostat import ResearchAssistant, StatisticalAnalyzer
from pyautostat.exceptions import InsufficientDataError, InvalidDataError
from pyautostat.reproducibility import reproduce


def _run(frame, predictors, variable_types, **kwargs):
    return ResearchAssistant(frame).run(
        objective="regression",
        outcome="y",
        predictors=predictors,
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "continuous", **variable_types},
        **kwargs,
    )


@pytest.fixture
def continuous_frame():
    x = np.arange(1.0, 21.0)
    z = np.array([2, 1, 4, 2, 5, 3, 7, 6, 8, 5, 9, 7, 11, 8, 13, 9, 14, 12, 15, 13], dtype=float)
    error = np.array([0.2, -0.4, 0.1, 0.3, -0.2, 0.5, -0.3, 0.1, 0.4, -0.5] * 2)
    return pd.DataFrame({"y": 4 + 1.7 * x - 0.6 * z + error, "x": x, "z": z})


def test_simple_regression_matches_statsmodels_and_reports(continuous_frame):
    workflow = _run(continuous_frame, ["x"], {"x": "continuous"})
    expected = sm.OLS(continuous_frame["y"], sm.add_constant(continuous_frame[["x"]])).fit()
    assert workflow.status.value == "completed"
    assert workflow.analysis.values["coefficients"][1]["estimate"] == pytest.approx(
        expected.params["x"]
    )
    assert workflow.analysis.values["model_fit"]["r_squared"] == pytest.approx(expected.rsquared)
    assert workflow.analysis.values["model_fit"]["model_f_p_value"] == pytest.approx(
        expected.f_pvalue
    )
    assert workflow.audit.status == "passed"
    assert "MODEL FIT" in workflow.explain()
    csv_tables = workflow.report.to_csv_tables()
    assert "regression_coefficients" in csv_tables
    assert "regression_diagnostics" in csv_tables
    assert "regression_vif" in csv_tables
    assert "Regression coefficients" in workflow.report.to_markdown()
    coefficient = workflow.analysis.values["coefficients"][1]
    assert coefficient["term_id"] == "term_1"
    assert coefficient["term_label"] == "x"
    assert coefficient["confidence_interval"]["quantity"] == "regression coefficient"
    json.loads(workflow.to_json())


def test_multiple_regression_order_complete_cases_and_replay(continuous_frame):
    frame = continuous_frame.copy()
    frame.loc[3, "x"] = np.nan
    frame.loc[8, "z"] = np.nan
    workflow = _run(frame, ["z", "x"], {"z": "continuous", "x": "continuous"})
    assert workflow.analysis.values["predictors"] == ["z", "x"]
    assert workflow.analysis.values["design_matrix"]["term_names"] == ["Intercept", "z", "x"]
    assert workflow.analysis.sample_size == 18
    assert workflow.analysis.excluded_rows == 2
    replay = reproduce(workflow.reproducibility, data=frame)
    assert replay.status == "reproduced"


def test_categorical_boolean_and_ordinal_coding_is_explicit():
    frame = pd.DataFrame(
        {
            "y": [10, 11, 14, 15, 18, 17, 22, 21, 25, 24, 29, 28],
            "x": list(range(1, 13)),
            "method": ["A", "B", "C"] * 4,
            "flag": [False, True] * 6,
            "level": pd.Categorical(
                [
                    "low",
                    "medium",
                    "high",
                    "medium",
                    "low",
                    "high",
                    "high",
                    "low",
                    "medium",
                    "high",
                    "medium",
                    "low",
                ],
                categories=["low", "medium", "high"],
                ordered=True,
            ),
        }
    )
    workflow = _run(
        frame,
        ["x", "method", "flag", "level"],
        {"x": "continuous", "method": "nominal", "flag": "boolean", "level": "ordinal"},
        reference_levels={"method": "B", "flag": False, "level": "low"},
    )
    coding = {
        item["predictor"]: item for item in workflow.analysis.values["design_matrix"]["coding"]
    }
    assert coding["method"]["reference_level"] == "B"
    assert len(coding["method"]["terms"]) == 2
    assert coding["flag"]["reference_level"] is False
    assert coding["level"]["ordinal_policy"] == "treated categorically; no equal spacing assumed"
    assert coding["level"]["observed_levels"] == ["low", "medium", "high"]
    assert workflow.specification.options.reference_levels == {
        "method": "B",
        "flag": False,
        "level": "low",
    }
    categorical = [
        item for item in workflow.analysis.values["coefficients"] if item["kind"] == "categorical"
    ]
    assert all(item["standardized_beta_status"] == "not_applicable" for item in categorical)


def test_hc3_changes_inference_not_coefficients(continuous_frame):
    classical = _run(continuous_frame, ["x", "z"], {"x": "continuous", "z": "continuous"})
    hc3 = _run(
        continuous_frame,
        ["x", "z"],
        {"x": "continuous", "z": "continuous"},
        covariance_type="HC3",
    )
    left = classical.analysis.values["coefficients"]
    right = hc3.analysis.values["coefficients"]
    assert [item["estimate"] for item in left] == pytest.approx(
        [item["estimate"] for item in right]
    )
    assert [item["standard_error"] for item in left] != pytest.approx(
        [item["standard_error"] for item in right]
    )
    assert hc3.analysis.values["covariance_type"] == "HC3"
    assert "covariance=HC3" in hc3.explain()


def test_standardized_beta_uses_complete_case_sample(continuous_frame):
    result = StatisticalAnalyzer(continuous_frame).linear_regression(
        "y",
        ["x", "z"],
        variable_types={"y": "continuous", "x": "continuous", "z": "continuous"},
    )
    slope = result["coefficients"][1]["estimate"]
    expected = slope * continuous_frame["x"].std(ddof=1) / continuous_frame["y"].std(ddof=1)
    assert result["coefficients"][1]["standardized_beta"] == pytest.approx(expected)
    assert result["coefficients"][0]["standardized_beta_status"] == "not_applicable"


def test_diagnostics_are_complete_and_do_not_mutate_covariance(continuous_frame):
    workflow = _run(continuous_frame, ["x", "z"], {"x": "continuous", "z": "continuous"})
    diagnostics = workflow.analysis.values["diagnostics"]
    assert set(diagnostics) == {
        "vif",
        "breusch_pagan",
        "residual_normality",
        "influence",
        "condition_number",
    }
    assert diagnostics["influence"]["rows_removed"] == 0
    assert diagnostics["influence"]["row_identifiers_included"] is False
    assert diagnostics["breusch_pagan"]["status"] in {"rejected", "not_rejected"}
    assert diagnostics["residual_normality"]["status"] in {"rejected", "not_rejected"}
    assert workflow.analysis.values["covariance_type"] == "classical"


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"y": [1.0, 2, 3, 4], "x": [1.0, 1, 1, 1]}), "no variation"),
        (pd.DataFrame({"y": [2.0, 2, 2, 2], "x": [1.0, 2, 3, 4]}), "outcome is constant"),
        (
            pd.DataFrame({"y": [1.0, 2, 3, 4, 5], "x": [1.0, 2, 3, 4, 5], "z": [2.0, 4, 6, 8, 10]}),
            "not full rank",
        ),
    ],
)
def test_unidentifiable_models_are_blocked(frame, message):
    predictors = [column for column in frame if column != "y"]
    with pytest.raises(InsufficientDataError, match=message):
        StatisticalAnalyzer(frame).linear_regression(
            "y",
            predictors,
            variable_types={"y": "continuous", **{name: "continuous" for name in predictors}},
        )
    workflow = _run(
        frame,
        predictors,
        {name: "continuous" for name in predictors},
    )
    assert workflow.status.value == "data_limited"
    assert message in " ".join(workflow.blockers)


def test_invalid_reference_and_categorical_outcome_are_rejected():
    frame = pd.DataFrame({"y": [1.0, 2, 3, 4, 5, 6], "g": ["A", "B"] * 3})
    with pytest.raises(InvalidDataError, match="not observed"):
        StatisticalAnalyzer(frame).linear_regression(
            "y",
            ["g"],
            variable_types={"y": "continuous", "g": "nominal"},
            reference_levels={"g": "missing"},
        )
    blocked = ResearchAssistant(frame.assign(y=frame["g"])).run(
        objective="regression",
        outcome="y",
        predictors=["g"],
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "nominal", "g": "nominal"},
    )
    assert blocked.status.value == "unsupported"
    assert "continuous numerical outcome" in " ".join(blocked.blockers)


def test_singular_predictor_alias_and_conflicts(continuous_frame):
    workflow = ResearchAssistant(continuous_frame).run(
        objective="regression",
        outcome="y",
        predictor="x",
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "continuous", "x": "continuous"},
    )
    assert workflow.analysis.values["predictors"] == ["x"]
    with pytest.raises(InvalidDataError, match="conflicting"):
        ResearchAssistant(continuous_frame).prepare_question(
            objective="regression",
            outcome="y",
            predictor="x",
            predictors=["z"],
            estimand="conditional_mean",
            design="independent",
        )
    with pytest.raises(InvalidDataError, match="only for objective='regression'"):
        ResearchAssistant(continuous_frame).prepare_question(
            objective="association",
            outcome="y",
            predictor="x",
            estimand="linear",
            design="independent",
            covariance_type="HC3",
        )


def test_association_selection_remains_pearson(continuous_frame):
    workflow = ResearchAssistant(continuous_frame).run(
        objective="association",
        outcome="y",
        predictor="x",
        estimand="linear",
        design="independent",
        variable_types={"y": "continuous", "x": "continuous"},
    )
    assert workflow.analysis.method_id == "pearson_correlation"


def test_regression_language_is_noncausal_and_not_predictive_accuracy(continuous_frame):
    workflow = _run(continuous_frame, ["x", "z"], {"x": "continuous", "z": "continuous"})
    text = " ".join(
        [workflow.explain(), workflow.report.to_markdown(), workflow.interpretation.summary]
    ).lower()
    assert "does not establish causation" in text
    assert "out-of-sample predictive accuracy" in text
    assert "predicts future" not in text
    assert "causes y" not in text


def test_scalar_sensitivity_contract_rejects_regression_vector(continuous_frame):
    workflow = _run(continuous_frame, ["x", "z"], {"x": "continuous", "z": "continuous"})
    with pytest.raises(InvalidDataError, match="coefficient vector"):
        ResearchAssistant(continuous_frame).sensitivity_analysis(
            workflow.analysis,
            scenarios=(),
        )


def test_regression_plan_and_session_preserve_model_identity(continuous_frame):
    assistant = ResearchAssistant(continuous_frame)
    draft = assistant.prepare_question(
        objective="regression",
        outcome="y",
        predictors=["z", "x"],
        estimand="conditional_mean",
        design="independent",
        variable_types={"y": "continuous", "z": "continuous", "x": "continuous"},
        covariance_type="HC3",
    )
    plan = assistant.analysis_plan(draft)
    payload = plan.to_dict()
    assert payload["predictors"] == ["z", "x"]
    assert payload["effect_quantity"] == "coefficient_vector"
    assert payload["covariance_type"] == "HC3"
    assert payload["analytical_variable_types"] == {
        "y": "continuous_numerical",
        "z": "continuous_numerical",
        "x": "continuous_numerical",
    }
    assert "VIF" in payload["planned_diagnostics"]
    workflow = assistant.run(draft=draft)
    snapshot = assistant.session_snapshot(workflow).to_dict()
    assert snapshot["workflow"]["analysis"]["values"]["predictors"] == ["z", "x"]
    assert "run_sensitivity" not in snapshot["available_actions"]


def test_html_escapes_categorical_labels():
    dangerous = "<script>alert(1)</script>"
    frame = pd.DataFrame({"y": [1.0, 2, 3, 4, 5, 6, 7, 8], "group": [dangerous, "A&B"] * 4})
    workflow = _run(
        frame,
        ["group"],
        {"group": "nominal"},
        reference_levels={"group": "A&B"},
    )
    html = workflow.report.to_html()
    assert dangerous not in html
    assert "&lt;script&gt;" in html


def test_extreme_finite_values_fail_clearly_without_json_nonfinite():
    frame = pd.DataFrame({"y": [1e300, 1.1e300, 1.2e300, 1.3e300, 1.4e300], "x": [1, 2, 3, 4, 5]})
    workflow = _run(frame, ["x"], {"x": "continuous"})
    assert workflow.status.value in {"completed", "partial", "failed"}
    json.dumps(workflow.to_dict(), allow_nan=False)
