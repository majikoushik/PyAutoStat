import json
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from pyautostat import (
    AnalysisOptions,
    AnalysisSpecification,
    ResearchAssistant,
    StatisticalAnalysisPlan,
)
from pyautostat.exceptions import InvalidDataError
from pyautostat.reproducibility import reproduce


def _scipy_stat(res: object) -> float:
    return float(getattr(res, "statistic", getattr(res, "correlation", res[0])))  # type: ignore[index]


def test_logistic_matches_statsmodels_and_integrates_report_audit_replay():
    rng = np.random.default_rng(42)
    x = rng.normal(size=180)
    group = np.where(np.arange(180) % 3 == 0, "premium", "standard")
    probability = 1 / (1 + np.exp(-(-0.3 + 0.8 * x + 0.5 * (group == "premium"))))
    outcome = np.where(rng.binomial(1, probability), "yes", "no")
    frame = pd.DataFrame({"event": outcome, "x": x, "group": group})
    workflow = ResearchAssistant(frame).run(
        objective="regression",
        outcome="event",
        predictors=["x", "group"],
        estimand="event_probability",
        design="independent",
        event_level="yes",
        variable_types={"event": "nominal", "x": "continuous", "group": "nominal"},
        reference_levels={"group": "standard"},
    )
    expected_x = sm.add_constant(
        pd.DataFrame({"x": x, "group": (group == "premium").astype(float)})
    )
    expected = sm.Logit((outcome == "yes").astype(float), expected_x).fit(disp=0)
    assert workflow.status.value == "completed"
    assert workflow.analysis.values["coefficients"][1]["estimate"] == pytest.approx(
        expected.params["x"]
    )
    assert workflow.analysis.values["model_fit"]["log_likelihood"] == pytest.approx(expected.llf)
    coefficient = workflow.analysis.values["coefficients"][2]
    assert coefficient["odds_ratio"] == pytest.approx(math.exp(coefficient["estimate"]))
    assert workflow.analysis.values["event_level"] == "yes"
    assert workflow.analysis.values["non_event_level"] == "no"
    assert workflow.audit.status == "passed"
    assert "ODDS RATIOS" in workflow.explain()
    tables = workflow.report.to_csv_tables()
    assert {"logistic_coefficients", "logistic_model_fit", "logistic_diagnostics"} <= set(tables)
    assert reproduce(workflow.reproducibility, data=frame).status == "reproduced"
    json.loads(workflow.to_json())


def test_logistic_event_is_requested_and_separation_is_blocked():
    frame = pd.DataFrame({"event": ["no", "yes"] * 10, "x": np.linspace(-1, 1, 20)})
    needs_event = ResearchAssistant(frame).run(
        objective="regression",
        outcome="event",
        predictors=["x"],
        estimand="event_probability",
        design="independent",
        variable_types={"event": "nominal", "x": "continuous"},
    )
    assert needs_event.status.value == "needs_input"
    assert [item.field for item in needs_event.missing_information] == ["event_level"]
    separated = pd.DataFrame({"event": [0] * 15 + [1] * 15, "x": [0] * 15 + [1] * 15})
    blocked = ResearchAssistant(separated).run(
        objective="regression",
        outcome="event",
        predictors=["x"],
        estimand="event_probability",
        design="independent",
        event_level=1,
        variable_types={"event": "nominal", "x": "continuous"},
    )
    assert blocked.status.value in {"data_limited", "failed"}
    assert "separat" in " ".join((*blocked.blockers, *blocked.warnings)).lower()


def _mcnemar_frame() -> pd.DataFrame:
    pairs = [
        ("yes", "yes"),
        ("yes", "no"),
        ("yes", "no"),
        ("yes", "no"),
        ("no", "yes"),
        ("no", "no"),
    ]
    rows = []
    for unit, (after, before) in enumerate(pairs):
        rows.extend(
            [
                {"unit": unit, "condition": "before", "response": before},
                {"unit": unit, "condition": "after", "response": after},
            ]
        )
    rows.append({"unit": 99, "condition": "after", "response": "yes"})
    return pd.DataFrame(rows)


def test_mcnemar_exact_orientation_pair_accounting_and_replay():
    frame = _mcnemar_frame()
    workflow = ResearchAssistant(frame).run(
        objective="compare_groups",
        outcome="response",
        predictor="condition",
        estimand="proportion",
        design="paired",
        unit_id="unit",
        condition_order=("after", "before"),
        event_level="yes",
        variable_types={"response": "nominal", "condition": "nominal"},
    )
    values = workflow.analysis.values
    table = values["transition_table"]
    assert workflow.status.value == "completed"
    assert table["discordant_b"] == 3
    assert table["discordant_c"] == 1
    assert values["p_value"] == pytest.approx(stats.binomtest(3, 4, 0.5).pvalue)
    assert values["primary_estimate"] == pytest.approx((3 - 1) / 6)
    assert workflow.analysis.metadata["sample"]["complete_pairs"] == 6
    assert workflow.analysis.metadata["sample"]["incomplete_units"] == 1
    assert workflow.analysis.metadata["method"] == "exact binomial McNemar"
    assert workflow.audit.status == "passed"
    assert "mcnemar_transition_table" in workflow.report.to_csv_tables()
    assert reproduce(workflow.reproducibility, data=frame).status == "reproduced"


def test_mcnemar_reversing_order_reverses_difference_not_p_value():
    frame = _mcnemar_frame()
    common = dict(
        objective="compare_groups",
        outcome="response",
        predictor="condition",
        estimand="proportion",
        design="paired",
        unit_id="unit",
        event_level="yes",
        variable_types={"response": "nominal", "condition": "nominal"},
    )
    forward = ResearchAssistant(frame).run(condition_order=("after", "before"), **common)
    reverse = ResearchAssistant(frame).run(condition_order=("before", "after"), **common)
    assert reverse.analysis.values["primary_estimate"] == pytest.approx(
        -forward.analysis.values["primary_estimate"]
    )
    assert reverse.analysis.values["p_value"] == pytest.approx(forward.analysis.values["p_value"])


def test_point_biserial_matches_scipy_and_positive_level_reverses_sign():
    frame = pd.DataFrame({"score": [1.0, 2, 3, 5, 7, 8, 9, 11], "group": ["A"] * 4 + ["B"] * 4})
    common = dict(
        objective="association",
        outcome="score",
        predictor="group",
        estimand="point_biserial",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
    )
    positive_b = ResearchAssistant(frame).run(event_level="B", **common)
    positive_a = ResearchAssistant(frame).run(event_level="A", **common)
    expected = stats.pointbiserialr((frame["group"] == "B").astype(int), frame["score"])
    exp_stat = _scipy_stat(expected)
    exp_p = float(getattr(expected, "pvalue", expected[1]))
    assert positive_b.analysis.values["primary_estimate"] == pytest.approx(exp_stat)
    assert positive_b.analysis.values["p_value"] == pytest.approx(exp_p)
    assert positive_a.analysis.values["primary_estimate"] == pytest.approx(-exp_stat)
    assert positive_b.analysis.values["confidence_interval"]["method"] == (
        "paired-observation percentile bootstrap"
    )


def test_kendall_is_explicit_and_spearman_remains_default():
    frame = pd.DataFrame({"x": [1, 2, 2, 4, 5, 6], "y": [1, 3, 2, 4, 4, 7]})
    common = dict(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="monotonic",
        design="independent",
        variable_types={"x": "continuous", "y": "continuous"},
    )
    default = ResearchAssistant(frame).run(**common)
    kendall = ResearchAssistant(frame).run(association_measure="kendall", **common)
    expected = stats.kendalltau(frame["x"], frame["y"], variant="b")
    assert default.analysis.method_id == "spearman_correlation"
    assert kendall.analysis.method_id == "kendall_tau_b"
    assert kendall.analysis.values["primary_estimate"] == pytest.approx(_scipy_stat(expected))
    assert kendall.analysis.values["ties"]["variant"] == "b"

    assistant = ResearchAssistant(frame)
    variable_types = {"x": "continuous", "y": "continuous"}
    for method, estimand, association_measure, method_id in (
        ("pearson", "linear", None, "pearson_correlation"),
        ("spearman", "monotonic", "spearman", "spearman_correlation"),
        ("kendall", "monotonic", "kendall", "kendall_tau_b"),
    ):
        wrapper = assistant.correlate("x", "y", method=method, variable_types=variable_types)
        explicit = assistant.run(
            objective="association",
            outcome="x",
            predictor="y",
            estimand=estimand,
            design="independent",
            association_measure=association_measure,
            variable_types=variable_types,
        )
        assert wrapper.status is explicit.status
        assert wrapper.specification == explicit.specification
        assert wrapper.recommendation.method_id == method_id
        assert wrapper.analysis.method_id == explicit.analysis.method_id == method_id
        assert wrapper.analysis.to_dict() == explicit.analysis.to_dict()

    with pytest.raises(InvalidDataError, match="first must be a non-empty string"):
        assistant.correlate("", "y", method="pearson")
    with pytest.raises(InvalidDataError, match="second must be a non-empty string"):
        assistant.correlate("x", " ", method="pearson")
    with pytest.raises(InvalidDataError, match="different columns"):
        assistant.correlate("x", "x", method="pearson")
    with pytest.raises(InvalidDataError, match="pearson, spearman, kendall"):
        assistant.correlate("x", "y", method="automatic")


def test_partial_pearson_matches_residual_reference_and_uses_correct_df():
    rng = np.random.default_rng(8)
    n = 100
    age = rng.normal(size=n)
    attendance = rng.normal(size=n)
    x = 0.8 * age + rng.normal(size=n)
    y = 0.7 * x + 1.1 * attendance + rng.normal(size=n)
    frame = pd.DataFrame({"x": x, "y": y, "age": age, "attendance": attendance})
    workflow = ResearchAssistant(frame).run(
        objective="association",
        outcome="x",
        predictor="y",
        controls=["age", "attendance"],
        estimand="partial_linear",
        design="independent",
        variable_types={name: "continuous" for name in frame},
    )
    design = sm.add_constant(frame[["age", "attendance"]])
    residual_x = sm.OLS(frame["x"], design).fit().resid
    residual_y = sm.OLS(frame["y"], design).fit().resid
    expected = stats.pearsonr(residual_x, residual_y)
    assert workflow.status.value == "completed"
    assert workflow.analysis.values["primary_estimate"] == pytest.approx(_scipy_stat(expected))
    assert workflow.analysis.values["degrees_of_freedom"] == n - 2 - 2
    assert workflow.analysis.values["confidence_interval"]["method"] == (
        "complete-row percentile bootstrap with model refitting"
    )
    assert "does not establish that confounding has been removed" in " ".join(
        workflow.interpretation.limitations
    )


def test_phase6_specification_round_trip_preserves_scientific_choices():
    frame = pd.DataFrame({"x": range(10), "y": range(10), "age": range(10)})
    draft = ResearchAssistant(frame).prepare_question(
        objective="association",
        outcome="x",
        predictor="y",
        controls=["age"],
        estimand="partial_linear",
        design="independent",
        association_measure=None,
        options=AnalysisOptions(random_seed=7, bootstrap_samples=199),
        variable_types={name: "continuous" for name in frame},
    )
    payload = draft.specification.to_dict()
    restored = AnalysisSpecification.from_dict(payload)
    assert restored.question.controls == ("age",)
    assert restored.options.bootstrap_samples == 199
    json.dumps(payload, allow_nan=False)


def test_logistic_mixed_predictors_hc3_complete_contract_and_safe_rendering():
    rng = np.random.default_rng(2026)
    n = 320
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    flag = np.arange(n) % 2 == 0
    tier = np.resize(np.array(["base", "<script>", "gold"]), n)
    linear = -0.4 + 0.7 * x - 0.45 * z + 0.35 * flag + 0.5 * (tier == "gold")
    event = rng.binomial(1, 1 / (1 + np.exp(-linear))).astype(bool)
    frame = pd.DataFrame({"event": event, "x": x, "z": z, "flag": flag, "tier": tier})
    workflow = ResearchAssistant(frame).run(
        objective="regression",
        outcome="event",
        predictors=["x", "z", "flag", "tier"],
        estimand="event_probability",
        design="independent",
        variable_types={
            "event": "boolean",
            "x": "continuous",
            "z": "continuous",
            "flag": "boolean",
            "tier": "ordinal",
        },
        reference_levels={"tier": "base"},
        covariance_type="HC3",
    )
    assert workflow.status.value == "completed"
    values = workflow.analysis.values
    assert values["event_level"] is True
    assert values["event_count"] + values["non_event_count"] == n
    assert values["covariance_type"] == "HC3"
    assert values["diagnostics"]["covariance_type"] == "HC3"
    assert values["diagnostics"]["converged"] is True
    assert values["design_matrix"]["full_rank"] is True
    tier_coding = next(
        item for item in values["design_matrix"]["coding"] if item["predictor"] == "tier"
    )
    assert tier_coding["reference_level"] == "base"
    assert tier_coding["ordinal_policy"] == "treated categorically; no equal spacing assumed"
    for coefficient in values["coefficients"]:
        assert coefficient["odds_ratio"] == pytest.approx(math.exp(coefficient["estimate"]))
        assert coefficient["odds_ratio_ci"]["lower"] == pytest.approx(
            math.exp(coefficient["confidence_interval"]["lower"])
        )
    fit = values["model_fit"]
    assert all(
        fit[name] is not None
        for name in (
            "log_likelihood",
            "null_log_likelihood",
            "lr_statistic",
            "lr_p_value",
            "aic",
            "bic",
            "mcfadden_r2",
        )
    )
    explanation = workflow.explain()
    assert all(label in explanation for label in (" MODEL", " EVENT", " SAMPLE", " ODDS RATIOS"))
    assert "probability by" not in explanation.lower()
    html = workflow.report.to_html()
    assert "&lt;script&gt;" in html and "<script>" not in html
    json.dumps(workflow.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("frame", "event_level", "message"),
    [
        (pd.DataFrame({"event": ["yes"] * 12, "x": range(12)}), "yes", "two"),
        (
            pd.DataFrame({"event": np.resize(["a", "b", "c"], 15), "x": range(15)}),
            "a",
            "two",
        ),
        (pd.DataFrame({"event": ["yes", "no"] * 6, "x": range(12)}), "missing", "observed"),
        (pd.DataFrame({"event": [0, 1] * 6, "x": range(12)}), "1", "observed"),
        (pd.DataFrame({"event": ["yes", "no"] * 6, "x": [1.0] * 12}), "yes", "variation"),
        (
            pd.DataFrame({"event": ["yes", "no"] * 6, "x": range(12), "z": np.arange(12) * 2}),
            "yes",
            "full rank",
        ),
        (pd.DataFrame({"event": ["yes", "no"] * 4, "x": range(8)}), "yes", "ten"),
    ],
)
def test_logistic_invalid_models_are_not_presented_as_completed(frame, event_level, message):
    predictors = [name for name in ("x", "z") if name in frame]
    if message == "observed":
        with pytest.raises(InvalidDataError, match="Observed levels") as invalid_event:
            ResearchAssistant(frame).run(
                objective="regression",
                outcome="event",
                predictors=predictors,
                estimand="event_probability",
                design="independent",
                event_level=event_level,
                variable_types={
                    "event": "nominal",
                    **{name: "continuous" for name in predictors},
                },
            )
        event_message = str(invalid_event.value)
        for observed in pd.unique(frame["event"]):
            assert repr(observed) in event_message
        assert repr(event_level) in event_message
        assert "Change `event_level`" in event_message
        return
    workflow = ResearchAssistant(frame).run(
        objective="regression",
        outcome="event",
        predictors=predictors,
        estimand="event_probability",
        design="independent",
        event_level=event_level,
        variable_types={"event": "nominal", **{name: "continuous" for name in predictors}},
    )
    assert workflow.status.value not in {"completed", "partial"}
    assert message in " ".join(workflow.blockers).lower()


def test_logistic_nonfinite_source_data_is_rejected_before_analysis():
    frame = pd.DataFrame({"event": ["yes", "no"] * 6, "x": [*range(11), np.inf]})
    with pytest.raises(InvalidDataError, match="infinity"):
        ResearchAssistant(frame)


def test_logistic_numeric_binary_event_defaults_only_when_declared_binary():
    frame = pd.DataFrame({"event": [0, 1] * 30, "x": np.linspace(-2, 2, 60)})
    declared = ResearchAssistant(frame).run(
        objective="regression",
        outcome="event",
        predictors=["x"],
        estimand="event_probability",
        design="independent",
        variable_types={"event": "nominal", "x": "continuous"},
    )
    undeclared = ResearchAssistant(frame).run(
        objective="regression",
        outcome="event",
        predictors=["x"],
        estimand="event_probability",
        design="independent",
        variable_types={"event": "continuous", "x": "continuous"},
    )
    assert declared.status.value == "completed"
    assert declared.specification.question.event_level == 1
    assert undeclared.status.value != "completed"


def _mcnemar_from_pairs(pairs, *, order=(1, 0), event_level=1, seed=19):
    rows = [
        {"unit": unit, "condition": condition, "response": value}
        for unit, pair in enumerate(pairs)
        for condition, value in zip((0, 1), pair, strict=True)
    ]
    frame = pd.DataFrame(rows)
    workflow = ResearchAssistant(frame).run(
        objective="compare_groups",
        outcome="response",
        predictor="condition",
        estimand="proportion",
        design="paired",
        unit_id="unit",
        condition_order=order,
        event_level=event_level,
        options=AnalysisOptions(random_seed=seed, bootstrap_samples=100),
        variable_types={"response": "nominal", "condition": "nominal"},
    )
    return frame, workflow


def test_mcnemar_zero_discordance_numeric_order_and_matched_or_statuses():
    frame, unchanged = _mcnemar_from_pairs([(0, 0), (1, 1), (0, 0), (1, 1)])
    assert unchanged.status.value == "completed"
    assert unchanged.analysis.values["p_value"] == 1.0
    assert unchanged.analysis.values["primary_estimate"] == 0.0
    assert unchanged.analysis.values["matched_odds_ratio"]["status"] == "undefined"
    assert unchanged.analysis.values["condition_order"] == [1, 0]
    assert unchanged.analysis.values["confidence_interval"]["lower"] == 0.0
    assert reproduce(unchanged.reproducibility, data=frame).status == "reproduced"

    _, one_direction = _mcnemar_from_pairs([(0, 1), (0, 1), (0, 0), (1, 1)])
    assert one_direction.analysis.values["matched_odds_ratio"]["status"] == "positive_infinity"
    assert one_direction.analysis.values["matched_odds_ratio"]["value"] is None
    json.dumps(one_direction.to_dict(), allow_nan=False)


def test_mcnemar_event_reversal_and_duplicate_pairs_are_explicit():
    pairs = [(0, 1), (0, 1), (1, 0), (0, 0), (1, 1)]
    _, event_one = _mcnemar_from_pairs(pairs, event_level=1)
    _, event_zero = _mcnemar_from_pairs(pairs, event_level=0)
    assert event_zero.analysis.values["primary_estimate"] == pytest.approx(
        -event_one.analysis.values["primary_estimate"]
    )
    assert event_zero.analysis.values["p_value"] == pytest.approx(
        event_one.analysis.values["p_value"]
    )
    duplicated = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2],
            "condition": [0, 0, 1, 0, 1],
            "response": [0, 1, 1, 0, 1],
        }
    )
    blocked = ResearchAssistant(duplicated).run(
        objective="compare_groups",
        outcome="response",
        predictor="condition",
        estimand="proportion",
        design="paired",
        unit_id="unit",
        condition_order=(1, 0),
        event_level=1,
        variable_types={"response": "nominal", "condition": "nominal"},
    )
    assert blocked.status.value not in {"completed", "partial"}
    assert "multiple usable observations" in " ".join(blocked.blockers)


def test_point_biserial_auto_orientation_missing_rows_pearson_equivalence_and_replay():
    frame = pd.DataFrame(
        {
            "binary": [False, False, False, True, True, True, True, None],
            "score": [1.0, 2.0, 2.5, 4.0, 5.0, 7.0, 8.0, 100.0],
        }
    )
    options = AnalysisOptions(random_seed=13, bootstrap_samples=100)
    workflow = ResearchAssistant(frame).run(
        objective="association",
        outcome="binary",
        predictor="score",
        estimand="point_biserial",
        design="independent",
        options=options,
        variable_types={"binary": "nominal", "score": "continuous"},
        event_level=True,
    )
    complete = frame.dropna()
    expected = stats.pearsonr(complete["binary"].astype(int), complete["score"])
    assert workflow.status.value == "completed"
    assert workflow.analysis.values["primary_estimate"] == pytest.approx(_scipy_stat(expected))
    assert workflow.analysis.excluded_rows == 1
    assert workflow.analysis.metadata["continuous_variable"] == "score"
    assert workflow.analysis.metadata["binary_variable"] == "binary"
    assert reproduce(workflow.reproducibility, data=frame).status == "reproduced"
    assert workflow.report.to_dict()["status"] == "complete"


def test_point_biserial_ambiguous_text_and_constant_inputs_are_blocked():
    frame = pd.DataFrame({"group": ["A"] * 5 + ["B"] * 5, "score": range(10)})
    pending = ResearchAssistant(frame).run(
        objective="association",
        outcome="group",
        predictor="score",
        estimand="point_biserial",
        design="independent",
        variable_types={"group": "nominal", "score": "continuous"},
    )
    assert pending.status.value == "needs_input"
    assert [item.field for item in pending.missing_information] == ["event_level"]
    constant = frame.assign(score=1.0)
    blocked = ResearchAssistant(constant).run(
        objective="association",
        outcome="group",
        predictor="score",
        estimand="point_biserial",
        design="independent",
        event_level="B",
        variable_types={"group": "nominal", "score": "continuous"},
    )
    assert blocked.status.value not in {"completed", "partial"}


def test_kendall_negative_missing_deterministic_interval_and_replay():
    frame = pd.DataFrame({"x": [1, 2, 2, 4, 5, 6, np.nan], "y": [9, 7, 8, 5, 3, 1, 0]})
    common = dict(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="monotonic",
        association_measure="kendall",
        design="independent",
        options=AnalysisOptions(random_seed=23, bootstrap_samples=100),
        variable_types={"x": "continuous", "y": "continuous"},
    )
    first = ResearchAssistant(frame).run(**common)
    second = ResearchAssistant(frame).run(**common)
    expected = stats.kendalltau(frame["x"], frame["y"], nan_policy="omit", variant="b")
    assert first.analysis.values["primary_estimate"] == pytest.approx(_scipy_stat(expected))
    assert first.analysis.values["primary_estimate"] < 0
    assert first.analysis.excluded_rows == 1
    assert (
        first.analysis.values["confidence_interval"]
        == second.analysis.values["confidence_interval"]
    )
    assert reproduce(first.reproducibility, data=frame).status == "reproduced"
    constant = ResearchAssistant(frame.assign(x=1.0)).run(**common)
    assert constant.status.value not in {"completed", "partial"}


def test_partial_pearson_t_formula_determinism_report_audit_replay_and_plan():
    rng = np.random.default_rng(31)
    n = 90
    c1, c2 = rng.normal(size=(2, n))
    x = 0.8 * c1 - 0.3 * c2 + rng.normal(size=n)
    y = -0.5 * x + c1 + 0.7 * c2 + rng.normal(size=n)
    frame = pd.DataFrame({"x": x, "y": y, "c1": c1, "c2": c2})
    assistant = ResearchAssistant(frame)
    options = AnalysisOptions(random_seed=29, bootstrap_samples=100)
    kwargs = dict(
        objective="association",
        outcome="x",
        predictor="y",
        controls=["c2", "c1"],
        estimand="partial_linear",
        design="independent",
        options=options,
        variable_types={name: "continuous" for name in frame},
    )
    first = assistant.run(**kwargs)
    second = assistant.run(**kwargs)
    values = first.analysis.values
    partial_r = values["primary_estimate"]
    df = n - 2 - 2
    expected_t = partial_r * math.sqrt(df / (1 - partial_r**2))
    assert values["test_statistic"] == pytest.approx(expected_t)
    assert values["degrees_of_freedom"] == df
    assert values["controls"] == ["c2", "c1"]
    assert values["control_design"]["effective_control_terms"] == 2
    assert values["confidence_interval"] == second.analysis.values["confidence_interval"]
    assert first.audit.status == "passed"
    assert reproduce(first.reproducibility, data=frame).status == "reproduced"
    plan = assistant.analysis_plan(first.specification)
    assert plan.specification.question.controls == ("c2", "c1")
    assert plan.to_dict()["controls"] == ["c2", "c1"]
    completeness = assistant.reporting_completeness(first.report)
    assert completeness.status == "complete"
    snapshot = assistant.session_snapshot(first, analysis_plan=plan)
    json.loads(snapshot.to_json())


def test_partial_pearson_invalid_controls_and_residual_degeneracy_are_blocked():
    frame = pd.DataFrame(
        {
            "x": np.arange(12, dtype=float),
            "y": np.linspace(0, 3, 12),
            "c": np.arange(12, dtype=float),
            "duplicate": np.arange(12, dtype=float) * 2,
        }
    )
    common = dict(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="partial_linear",
        design="independent",
        options=AnalysisOptions(bootstrap_samples=100),
        variable_types={name: "continuous" for name in frame},
    )
    with pytest.raises(InvalidDataError, match="duplicates"):
        ResearchAssistant(frame).run(controls=["c", "c"], **common)
    with pytest.raises(InvalidDataError, match="differ"):
        ResearchAssistant(frame).run(controls=["x"], **common)
    collinear = ResearchAssistant(frame).run(controls=["c", "duplicate"], **common)
    assert collinear.status.value not in {"completed", "partial"}
    residual_zero = ResearchAssistant(frame).run(controls=["c"], **common)
    assert residual_zero.status.value in {"data_limited", "failed"}
    assert "zero variance" in " ".join((*residual_zero.blockers, *residual_zero.warnings)).lower()


def test_method_selection_boundaries_remain_distinct():
    numeric = pd.DataFrame({"x": np.arange(30, dtype=float), "y": np.arange(30) ** 1.2})
    common = dict(
        objective="association",
        outcome="x",
        predictor="y",
        design="independent",
        variable_types={"x": "continuous", "y": "continuous"},
    )
    linear = ResearchAssistant(numeric).run(estimand="linear", **common)
    monotonic = ResearchAssistant(numeric).run(estimand="monotonic", **common)
    assert linear.analysis.method_id == "pearson_correlation"
    assert monotonic.analysis.method_id == "spearman_correlation"

    frame = pd.DataFrame({"event": [0, 1] * 20, "x": np.linspace(-1, 1, 40)})
    regression = ResearchAssistant(frame).run(
        objective="regression",
        outcome="x",
        predictors=["event"],
        estimand="conditional_mean",
        design="independent",
        variable_types={"x": "continuous", "event": "continuous"},
    )
    assert regression.analysis.method_id == "linear_regression"


def test_logistic_near_separation_is_handled_safely():
    x = np.array([1] * 20 + [0] * 20)
    event = np.array([1] * 20 + [0] * 15 + [1] * 5)
    frame = pd.DataFrame({"event": event, "x": x})
    workflow = ResearchAssistant(frame).run(
        objective="regression",
        outcome="event",
        predictors=["x"],
        estimand="event_probability",
        design="independent",
        event_level=1,
        variable_types={"event": "nominal", "x": "continuous"},
    )
    assert workflow.status.value in {"data_limited", "failed"}
    all_reasons = " ".join((*workflow.blockers, *workflow.warnings)).lower()
    assert any(term in all_reasons for term in ("separat", "converge", "unreliable"))
    payload = workflow.to_dict()
    assert json.dumps(payload, allow_nan=False)
    if workflow.report is not None:
        report_payload = workflow.report.to_dict()
        assert json.dumps(report_payload, allow_nan=False)


def test_logistic_non_convergence_handling(monkeypatch):
    frame = pd.DataFrame({"event": [0, 1] * 20, "x": np.linspace(-1, 1, 40)})
    orig_fit = sm.Logit.fit

    def mock_fit(self, *args, **kwargs):
        res = orig_fit(self, *args, **kwargs)
        res.mle_retvals = dict(res.mle_retvals, converged=False)
        return res

    monkeypatch.setattr(sm.Logit, "fit", mock_fit)

    workflow = ResearchAssistant(frame).run(
        objective="regression",
        outcome="event",
        predictors=["x"],
        estimand="event_probability",
        design="independent",
        event_level=1,
        variable_types={"event": "nominal", "x": "continuous"},
    )
    assert workflow.status.value in {"data_limited", "failed"}
    assert "converge" in " ".join((*workflow.blockers, *workflow.warnings)).lower()
    assert workflow.analysis is None or workflow.analysis.status.value == "unavailable"
    assert json.dumps(workflow.to_dict(), allow_nan=False)


def test_phase6_analysis_plan_metadata_and_round_trip():
    rng = np.random.default_rng(99)
    n = 60
    log_df = pd.DataFrame(
        {
            "churn": rng.choice(["yes", "no"], size=n),
            "usage": rng.normal(size=n),
            "plan_type": rng.choice(["basic", "pro", "enterprise"], size=n),
        }
    )
    assistant_log = ResearchAssistant(log_df)
    draft_log = assistant_log.prepare_question(
        objective="regression",
        outcome="churn",
        predictors=["usage", "plan_type"],
        estimand="event_probability",
        design="independent",
        event_level="yes",
        covariance_type="HC3",
        reference_levels={"plan_type": "basic"},
        options=AnalysisOptions(alpha=0.01, confidence_level=0.99),
        variable_types={"churn": "nominal", "usage": "continuous", "plan_type": "nominal"},
    )
    plan_log = assistant_log.analysis_plan(draft_log)
    assert plan_log.status.value == "ready"
    assert plan_log.primary_method_id == "logistic_regression"
    assert plan_log.effect_quantity == "coefficient_vector"
    assert plan_log.confidence_interval_quantity == "coefficient_vector"
    plan_log_dict = plan_log.to_dict()
    assert plan_log_dict["objective"] == "regression"
    assert plan_log_dict["outcome"] == "churn"
    assert plan_log_dict["predictors"] == ["usage", "plan_type"]
    assert plan_log_dict["estimand"] == "event_probability"
    assert plan_log_dict["study_design"] == "independent"
    assert plan_log_dict["event_level"] == "yes"
    assert plan_log_dict["covariance_type"] == "HC3"
    assert plan_log_dict["reference_levels"] == {"plan_type": "basic"}
    assert plan_log_dict["alpha"] == 0.01
    assert plan_log_dict["confidence_level"] == 0.99
    restored_log = StatisticalAnalysisPlan.from_dict(plan_log_dict)
    assert restored_log.to_dict() == plan_log_dict
    assert restored_log.specification.question.event_level == "yes"
    assert restored_log.specification.options.covariance_type == "HC3"
    assert restored_log.specification.options.reference_levels == {"plan_type": "basic"}

    pairs = [
        {"unit": u, "condition": c, "resp": r}
        for u, (b, a) in enumerate([("no", "yes")] * 10 + [("yes", "no")] * 5)
        for c, r in (("before", b), ("after", a))
    ]
    mcnemar_df = pd.DataFrame(pairs)
    assistant_mc = ResearchAssistant(mcnemar_df)
    draft_mc = assistant_mc.prepare_question(
        objective="compare_groups",
        outcome="resp",
        predictor="condition",
        estimand="proportion",
        design="paired",
        unit_id="unit",
        condition_order=("after", "before"),
        event_level="yes",
        options=AnalysisOptions(
            random_seed=123, bootstrap_samples=150, alpha=0.01, confidence_level=0.99
        ),
        variable_types={"resp": "nominal", "condition": "nominal"},
    )
    plan_mc = assistant_mc.analysis_plan(draft_mc)
    assert plan_mc.status.value == "ready"
    assert plan_mc.primary_method_id == "mcnemar"
    assert plan_mc.effect_quantity == "paired_proportion_difference"
    plan_mc_dict = plan_mc.to_dict()
    assert plan_mc_dict["unit_id"] == "unit"
    assert plan_mc_dict["condition_order"] == ["after", "before"]
    assert plan_mc_dict["event_level"] == "yes"
    restored_mc = StatisticalAnalysisPlan.from_dict(plan_mc_dict)
    assert restored_mc.specification.condition_order == ("after", "before")
    assert restored_mc.specification.unit_id == "unit"
    assert restored_mc.specification.question.event_level == "yes"
    assert restored_mc.specification.options.random_seed == 123
    assert restored_mc.specification.options.bootstrap_samples == 150

    assoc_df = pd.DataFrame({"score": range(10), "flag": [True, False] * 5, "rank": range(10)})
    assistant_assoc = ResearchAssistant(assoc_df)
    draft_pb = assistant_assoc.prepare_question(
        objective="association",
        outcome="score",
        predictor="flag",
        estimand="point_biserial",
        design="independent",
        event_level=True,
        variable_types={"score": "continuous", "flag": "boolean"},
    )
    plan_pb = assistant_assoc.analysis_plan(draft_pb)
    assert plan_pb.primary_method_id == "point_biserial_correlation"
    assert plan_pb.to_dict()["event_level"] is True
    assert plan_pb.to_dict()["estimand"] == "point_biserial"

    draft_kt = assistant_assoc.prepare_question(
        objective="association",
        outcome="score",
        predictor="rank",
        estimand="monotonic",
        association_measure="kendall",
        design="independent",
        variable_types={"score": "continuous", "rank": "continuous"},
    )
    plan_kt = assistant_assoc.analysis_plan(draft_kt)
    assert plan_kt.primary_method_id == "kendall_tau_b"
    assert plan_kt.to_dict()["association_measure"] == "kendall"
    assert plan_kt.to_dict()["estimand"] == "monotonic"


def test_phase6_reporting_completeness_and_edge_cases():
    frame = pd.DataFrame(
        {
            "event": ["no", "yes"] * 25,
            "x": np.linspace(-2, 2, 50),
            "tier": (["base", "base", "gold", "gold"] * 12 + ["base", "gold"]),
        }
    )
    assistant = ResearchAssistant(frame)
    log_wf = assistant.run(
        objective="regression",
        outcome="event",
        predictors=["x", "tier"],
        estimand="event_probability",
        design="independent",
        event_level="yes",
        reference_levels={"tier": "base"},
        variable_types={"event": "nominal", "x": "continuous", "tier": "nominal"},
    )
    assert log_wf.status.value == "completed"
    comp = assistant.reporting_completeness(log_wf.report)
    assert comp.status == "complete"
    items_by_code = {item.code: item.status for item in comp.items}
    for required_code in (
        "LOGISTIC_EVENT_REPORTED",
        "LOGISTIC_MODEL_FIT_REPORTED",
        "LOGISTIC_COEFFICIENTS_REPORTED",
        "LOGISTIC_ODDS_RATIOS_REPORTED",
        "LOGISTIC_INTERVALS_REPORTED",
        "LOGISTIC_DIAGNOSTICS_REPORTED",
        "LOGISTIC_COVARIANCE_REPORTED",
        "LOGISTIC_SAMPLE_COUNTS_REPORTED",
    ):
        assert items_by_code[required_code] == "present"
    assert json.dumps(comp.to_dict(), allow_nan=False)

    pairs_zero = [(0, 0), (1, 1), (0, 0), (1, 1)] * 5
    rows_zero = [
        {"unit": u, "condition": c, "resp": r}
        for u, p in enumerate(pairs_zero)
        for c, r in ((0, p[0]), (1, p[1]))
    ]
    asst_zero = ResearchAssistant(pd.DataFrame(rows_zero))
    wf_zero = asst_zero.run(
        objective="compare_groups",
        outcome="resp",
        predictor="condition",
        estimand="proportion",
        design="paired",
        unit_id="unit",
        condition_order=(0, 1),
        event_level=1,
        options=AnalysisOptions(bootstrap_samples=100),
        variable_types={"resp": "nominal", "condition": "nominal"},
    )
    assert wf_zero.status.value == "completed"
    assert wf_zero.analysis.values["matched_odds_ratio"]["status"] == "undefined"
    comp_zero = asst_zero.reporting_completeness(wf_zero.report)
    assert comp_zero.status == "complete"
    assert json.dumps(comp_zero.to_dict(), allow_nan=False)

    pairs_inf = [(1, 0), (1, 0), (0, 0), (1, 1)] * 5
    rows_inf = [
        {"unit": u, "condition": c, "resp": r}
        for u, p in enumerate(pairs_inf)
        for c, r in ((0, p[0]), (1, p[1]))
    ]
    asst_inf = ResearchAssistant(pd.DataFrame(rows_inf))
    wf_inf = asst_inf.run(
        objective="compare_groups",
        outcome="resp",
        predictor="condition",
        estimand="proportion",
        design="paired",
        unit_id="unit",
        condition_order=(0, 1),
        event_level=1,
        options=AnalysisOptions(bootstrap_samples=100),
        variable_types={"resp": "nominal", "condition": "nominal"},
    )
    assert wf_inf.status.value == "completed"
    assert wf_inf.analysis.values["matched_odds_ratio"]["status"] == "positive_infinity"
    comp_inf = asst_inf.reporting_completeness(wf_inf.report)
    assert comp_inf.status == "complete"
    assert json.dumps(comp_inf.to_dict(), allow_nan=False)


def test_phase6_audit_detects_deliberate_contradictions():
    import copy

    frame = pd.DataFrame({"event": [0, 1] * 20, "x": np.linspace(-1, 1, 40)})
    assistant = ResearchAssistant(frame)
    wf_log = assistant.run(
        objective="regression",
        outcome="event",
        predictors=["x"],
        estimand="event_probability",
        design="independent",
        event_level=1,
        variable_types={"event": "nominal", "x": "continuous"},
    )
    assert wf_log.audit.status == "passed"

    tampered = copy.deepcopy(wf_log.analysis)
    tampered.values["coefficients"][1]["odds_ratio"] = 999.0
    audit_res = assistant.audit(wf_log.report, result=tampered)
    assert audit_res.status == "failed"
    assert any("odds_ratio" in f.field for f in audit_res.findings)

    tampered = copy.deepcopy(wf_log.analysis)
    tampered.values["event_count"] = 999
    audit_res = assistant.audit(wf_log.report, result=tampered)
    assert audit_res.status == "failed"
    assert any("event_count" in f.field for f in audit_res.findings)

    tampered = copy.deepcopy(wf_log.analysis)
    tampered.values["event_level"] = 0
    audit_res = assistant.audit(wf_log.report, result=tampered)
    assert audit_res.status == "failed"
    assert any("event_level" in f.field for f in audit_res.findings)

    tampered = copy.deepcopy(wf_log.analysis)
    tampered.values["coefficients"][1]["odds_ratio_ci"]["upper"] = 999.0
    audit_res = assistant.audit(wf_log.report, result=tampered)
    assert audit_res.status == "failed"
    assert any("odds_ratio_ci" in f.field for f in audit_res.findings)

    pairs = [(0, 1), (1, 0), (0, 0), (1, 1)] * 5
    m_rows = [
        {"unit": u, "condition": c, "resp": r}
        for u, p in enumerate(pairs)
        for c, r in ((0, p[0]), (1, p[1]))
    ]
    asst_m = ResearchAssistant(pd.DataFrame(m_rows))
    wf_m = asst_m.run(
        objective="compare_groups",
        outcome="resp",
        predictor="condition",
        estimand="proportion",
        design="paired",
        unit_id="unit",
        condition_order=(0, 1),
        event_level=1,
        options=AnalysisOptions(bootstrap_samples=100),
        variable_types={"resp": "nominal", "condition": "nominal"},
    )
    assert wf_m.audit.status == "passed"

    tampered_m = copy.deepcopy(wf_m.analysis)
    tampered_m.values["transition_table"]["first_event_second_event"] = 999
    audit_m = asst_m.audit(wf_m.report, result=tampered_m)
    assert audit_m.status == "failed"
    assert any("transition_table" in f.field for f in audit_m.findings)

    tampered_m = copy.deepcopy(wf_m.analysis)
    tampered_m.values["transition_table"]["discordant_b"] = 999
    audit_m = asst_m.audit(wf_m.report, result=tampered_m)
    assert audit_m.status == "failed"
    assert any("transition_table" in f.field for f in audit_m.findings)

    tampered_m = copy.deepcopy(wf_m.analysis)
    tampered_m.values["primary_estimate"] = 0.999
    audit_m = asst_m.audit(wf_m.report, result=tampered_m)
    assert audit_m.status == "failed"
    assert any("primary_estimate" in f.field for f in audit_m.findings)

    tampered_m = copy.deepcopy(wf_m.analysis)
    tampered_m.values["condition_order"] = [1, 0]
    audit_m = asst_m.audit(wf_m.report, result=tampered_m)
    assert audit_m.status == "failed"
    assert any("condition_order" in f.field for f in audit_m.findings)

    rng = np.random.default_rng(101)
    c1 = rng.normal(size=30)
    x = 0.5 * c1 + rng.normal(size=30)
    y = 0.3 * x + 0.4 * c1 + rng.normal(size=30)
    asst_p = ResearchAssistant(pd.DataFrame({"x": x, "y": y, "c1": c1}))
    wf_p = asst_p.run(
        objective="association",
        outcome="x",
        predictor="y",
        controls=["c1"],
        estimand="partial_linear",
        design="independent",
        options=AnalysisOptions(bootstrap_samples=100),
        variable_types={"x": "continuous", "y": "continuous", "c1": "continuous"},
    )
    assert wf_p.audit.status == "passed"

    tampered_p = copy.deepcopy(wf_p.analysis)
    tampered_p.values["degrees_of_freedom"] = 999
    audit_p = asst_p.audit(wf_p.report, result=tampered_p)
    assert audit_p.status == "failed"
    assert any("degrees_of_freedom" in f.field for f in audit_p.findings)

    tampered_p = copy.deepcopy(wf_p.analysis)
    tampered_p.values["controls"] = ["wrong_control"]
    audit_p = asst_p.audit(wf_p.report, result=tampered_p)
    assert audit_p.status == "failed"
    assert any("controls" in f.field for f in audit_p.findings)


def test_phase6_session_snapshot_preserves_scientific_choices_and_strict_json():
    frame = pd.DataFrame(
        {
            "event": ["no", "yes"] * 25,
            "x": np.linspace(-1, 1, 50),
            "grp": (["standard", "standard", "premium", "premium"] * 12 + ["standard", "premium"]),
        }
    )
    assistant = ResearchAssistant(frame)
    workflow = assistant.run(
        objective="regression",
        outcome="event",
        predictors=["x", "grp"],
        estimand="event_probability",
        design="independent",
        event_level="yes",
        covariance_type="HC3",
        reference_levels={"grp": "standard"},
        variable_types={"event": "nominal", "x": "continuous", "grp": "nominal"},
    )
    plan = assistant.analysis_plan(workflow.specification)
    snapshot = assistant.session_snapshot(workflow, analysis_plan=plan)
    raw_json = snapshot.to_json()
    parsed = json.loads(raw_json)
    assert parsed["schema_version"] == 1
    assert parsed["analysis_plan"]["specification"]["question"]["event_level"] == "yes"
    assert (
        parsed["analysis_plan"]["specification"]["options"]["reference_levels"]["grp"] == "standard"
    )
    assert parsed["analysis_plan"]["specification"]["options"]["covariance_type"] == "HC3"
    assert parsed["workflow"]["analysis"]["values"]["event_level"] == "yes"
    assert parsed["workflow"]["analysis"]["values"]["covariance_type"] == "HC3"
    assert json.dumps(snapshot.to_dict(), allow_nan=False)
