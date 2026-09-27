"""Scientific and integration tests for the four basic-inference additions."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyautostat import (
    AnalysisStatus,
    InvalidTestError,
    ResearchAssistant,
    SensitivitySpecification,
    StatisticalAnalyzer,
    WorkflowStatus,
    reproduce,
)


def _one_sample(frame, reference=10.0):
    return ResearchAssistant(frame).run(
        objective="compare_reference",
        outcome="score",
        reference_value=reference,
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )


def _paired(frame, order=("after", "before")):
    return ResearchAssistant(frame).run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        estimand="distribution",
        design="paired",
        unit_id="student",
        condition_order=order,
        variable_types={"score": "continuous", "condition": "nominal"},
    )


def _spearman(frame):
    return ResearchAssistant(frame).run(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="monotonic",
        design="independent",
        variable_types={"x": "continuous", "y": "continuous"},
    )


def _categorical(frame):
    return ResearchAssistant(frame).run(
        objective="association",
        outcome="response",
        predictor="treatment",
        estimand="categorical_independence",
        design="independent",
        variable_types={"response": "nominal", "treatment": "nominal"},
    )


@pytest.mark.parametrize(
    ("values", "reference", "direction"),
    [
        ([12.0, 13.0, 14.0, 15.0], 10.0, 1),
        ([5.0, 6.0, 7.0, 8.0], 10.0, -1),
        ([-1.0, 0.0, 1.0], 0.0, 0),
    ],
)
def test_one_sample_matches_scipy_and_preserves_orientation(values, reference, direction):
    frame = pd.DataFrame({"score": values})
    direct = StatisticalAnalyzer(frame).one_sample_t_test("score", reference)
    expected = stats.ttest_1samp(values, reference, alternative="two-sided")
    assert direct["statistic"] == pytest.approx(expected.statistic)
    assert direct["p_value"] == pytest.approx(expected.pvalue)
    assert direct["mean_difference"] == pytest.approx(np.mean(values) - reference)
    assert np.sign(direct["mean_difference"]) == direction
    assert np.sign(direct["effect_size"]["value"]) == direction
    assert direct["confidence_interval"]["lower"] <= direct["mean_difference"]
    assert direct["confidence_interval"]["upper"] >= direct["mean_difference"]


def test_one_sample_guided_contract_report_audit_and_replay():
    frame = pd.DataFrame({"score": [9.0, 11.0, 12.0, 13.0, None]})
    workflow = _one_sample(frame)
    assert workflow.status is WorkflowStatus.COMPLETED
    result = workflow.analysis
    assert result is not None and result.method_id == "one_sample_t"
    assert result.sample_size == 4 and result.excluded_rows == 1
    assert result.values["reference_value"] == 10.0
    assert result.values["primary_estimate"] == pytest.approx(1.25)
    assert result.values["estimate_name"] == "mean difference from reference"
    assert result.metadata["contrast"]["definition"].startswith("observed sample mean")
    assert workflow.audit is not None and workflow.audit.status == "passed"
    assert workflow.report is not None
    report = workflow.report.to_dict()
    assert report["sections"]["research_question"]["reference_value"] == 10.0
    assert report["sections"]["results"]["sample_mean"] == pytest.approx(11.25)
    assert "reference value" in workflow.explain().lower()
    assert workflow.reproducibility is not None
    replay = reproduce(workflow.reproducibility, data=frame)
    assert replay.status == "reproduced"
    json.dumps(workflow.to_dict(), allow_nan=False)


@pytest.mark.parametrize("reference", [math.nan, math.inf, -math.inf])
def test_one_sample_rejects_nonfinite_reference(reference):
    with pytest.raises(InvalidTestError, match="reference_value"):
        StatisticalAnalyzer(pd.DataFrame({"score": [1.0, 2.0]})).one_sample_t_test(
            "score", reference
        )


def test_one_sample_constant_samples_keep_raw_difference_without_invented_d():
    for reference, expected in [(5.0, 0.0), (4.0, 1.0)]:
        workflow = _one_sample(pd.DataFrame({"score": [5.0, 5.0, 5.0]}), reference)
        assert workflow.status is WorkflowStatus.PARTIAL
        result = workflow.analysis
        assert result is not None and result.status is AnalysisStatus.AVAILABLE
        assert result.values["primary_estimate"] == expected
        assert result.values["test_statistic"] is None
        assert result.values["p_value"] is None
        assert result.values["effect_size"]["value"] is None
        assert result.values["confidence_interval"]["lower"] == expected
        json.dumps(result.to_dict(), allow_nan=False)


def test_one_sample_wrong_type_all_missing_and_minimum_boundary():
    with pytest.raises(InvalidTestError, match="numeric"):
        StatisticalAnalyzer(pd.DataFrame({"score": ["low", "high"]})).one_sample_t_test("score", 0)
    all_missing = _one_sample(pd.DataFrame({"score": [np.nan, np.nan]}), 0)
    assert all_missing.status is WorkflowStatus.DATA_LIMITED
    two = StatisticalAnalyzer(pd.DataFrame({"score": [1.0, 3.0]})).one_sample_t_test("score", 0)
    assert two["degrees_of_freedom"] == 1


def test_one_sample_extreme_finite_values_remain_finite():
    values = 1e150 + np.array([-3e140, -1e140, 2e140, 4e140])
    result = StatisticalAnalyzer(pd.DataFrame({"score": values})).one_sample_t_test("score", 1e150)
    assert math.isfinite(result["mean_difference"])
    assert math.isfinite(result["statistic"])
    assert math.isfinite(result["p_value"])
    assert all(math.isfinite(result["confidence_interval"][bound]) for bound in ("lower", "upper"))


def test_wilcoxon_uses_unit_identity_matches_scipy_and_records_zeros():
    frame = pd.DataFrame(
        {
            "student": [3, 1, 4, 2, 1, 4, 2, 3, 5, 5],
            "condition": ["before", "after"] * 5,
            "score": [5.0, 5.0, 4.0, 4.0, 3.0, 8.0, 2.0, 6.0, 7.0, 7.0],
        }
    )
    direct = StatisticalAnalyzer(frame).paired_wilcoxon(
        "student", "condition", "score", condition_order=("after", "before")
    )
    differences = np.array([2.0, 2.0, 1.0, 4.0, 0.0])
    expected = stats.wilcoxon(
        differences,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="auto",
    )
    assert direct["statistic"] == pytest.approx(expected.statistic)
    assert direct["p_value"] == pytest.approx(expected.pvalue)
    assert direct["zero_method"] == "wilcox"
    assert direct["zero_differences"] == 1
    assert direct["effect_size"]["value"] > 0
    workflow = _paired(frame)
    assert workflow.status is WorkflowStatus.PARTIAL
    assert workflow.analysis is not None
    assert workflow.analysis.metadata["sample"]["complete_pairs"] == 5
    assert workflow.analysis.values["confidence_interval"] is None
    assert "not universally" in " ".join(workflow.interpretation.limitations).lower()
    assert "paired t-test models" not in workflow.explain().lower()


def test_wilcoxon_reversed_order_reverses_effect_and_keeps_pair_accounting():
    frame = pd.DataFrame(
        {
            "student": [1, 1, 2, 2, 3, 3, 4, 4, 5, None],
            "condition": ["after", "before"] * 4 + ["after", "before"],
            "score": [5.0, 2.0, 7.0, 3.0, 4.0, 3.0, 8.0, 2.0, 9.0, 100.0],
        }
    )
    forward = _paired(frame).analysis
    reverse = _paired(frame, ("before", "after")).analysis
    assert forward is not None and reverse is not None
    assert forward.values["effect_size"]["value"] == pytest.approx(
        -reverse.values["effect_size"]["value"]
    )
    sample = forward.metadata["sample"]
    assert sample["complete_pairs"] == 4
    assert sample["incomplete_units"] == 1
    assert sample["missing_unit_rows"] == 1
    assert forward.sample_size == 8 and forward.excluded_rows == 2


def test_wilcoxon_blocks_duplicate_pairs_all_zero_and_insufficient_nonzero():
    duplicate = pd.DataFrame(
        {
            "student": [1, 1, 1, 2, 2],
            "condition": ["after", "after", "before", "after", "before"],
            "score": [2.0, 3.0, 1.0, 4.0, 2.0],
        }
    )
    assert _paired(duplicate).status is WorkflowStatus.UNSUPPORTED
    for values in ([2.0, 2.0, 3.0, 3.0], [3.0, 2.0, 3.0, 3.0]):
        frame = pd.DataFrame(
            {
                "student": [1, 1, 2, 2],
                "condition": ["after", "before"] * 2,
                "score": values,
            }
        )
        assert _paired(frame).status is WorkflowStatus.UNSUPPORTED


def test_wilcoxon_sensitivity_preserves_paired_identity_and_different_estimand():
    frame = pd.DataFrame(
        {
            "student": [1, 1, 2, 2, 3, 3, 4, 4],
            "condition": ["after", "before"] * 4,
            "score": [5.0, 2.0, 7.0, 3.0, 4.0, 3.0, 8.0, 2.0],
        }
    )
    assistant = ResearchAssistant(frame)
    mean = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        estimand="mean",
        design="paired",
        unit_id="student",
        condition_order=("after", "before"),
        variable_types={"score": "continuous", "condition": "nominal"},
    ).analysis
    rank = _paired(frame).analysis
    assert mean is not None and rank is not None
    sensitivity = assistant.sensitivity_analysis(
        mean,
        scenarios=[
            SensitivitySpecification(
                "signed ranks", rank.specification, method_id="wilcoxon_signed_rank"
            )
        ],
    )
    scenario = sensitivity.scenario_results[0]
    assert scenario.status.value == "completed"
    assert scenario.comparability.value == "different_estimand"
    assert scenario.comparison is None


@pytest.mark.parametrize(
    ("y", "sign"),
    [
        ([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1),
        ([6.0, 3.0, 5.0, 2.0, 4.0, 1.0], -1),
    ],
)
def test_spearman_matches_scipy_and_bootstrap_is_deterministic(y, sign):
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": y})
    analyzer = StatisticalAnalyzer(frame)
    first = analyzer.spearman_correlation("x", "y", random_state=17)
    second = analyzer.spearman_correlation("x", "y", random_state=17)
    expected = stats.spearmanr(frame["x"], frame["y"], alternative="two-sided")
    assert first["statistic"] == pytest.approx(expected.statistic)
    assert first["p_value"] == pytest.approx(expected.pvalue)
    assert np.sign(first["statistic"]) == sign
    assert first["confidence_interval"] == second["confidence_interval"]
    assert first["confidence_interval"]["method"] == "paired-observation percentile bootstrap"


def test_spearman_guided_ties_missing_report_replay_and_no_linear_or_causal_claim():
    frame = pd.DataFrame(
        {"x": [1.0, 1.0, 2.0, 3.0, 4.0, None], "y": [2.0, 3.0, 3.0, 5.0, 6.0, 8.0]}
    )
    workflow = _spearman(frame)
    assert workflow.status is WorkflowStatus.COMPLETED
    result = workflow.analysis
    assert result is not None and result.method_id == "spearman_correlation"
    assert result.sample_size == 5 and result.excluded_rows == 1
    assert result.metadata["ties"]["first_has_ties"] is True
    text = workflow.explain().lower()
    assert "monotonic" in text and "causation" in text and "linear" in text
    assert workflow.audit is not None and workflow.audit.status == "passed"
    assert workflow.reproducibility is not None
    assert reproduce(workflow.reproducibility, data=frame).status == "reproduced"


def test_spearman_constant_too_few_and_profile_matrix_behavior():
    constant = pd.DataFrame({"x": [1.0, 1.0, 1.0], "y": [1.0, 2.0, 3.0]})
    assert _spearman(constant).status is WorkflowStatus.UNSUPPORTED
    too_few = pd.DataFrame({"x": [1.0, 2.0, None], "y": [2.0, 3.0, 4.0]})
    assert _spearman(too_few).status is WorkflowStatus.UNSUPPORTED
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 1.0, 3.0, 2.0]})
    profile = StatisticalAnalyzer(frame).analyze_all()
    assert profile["correlation"]["spearman"]["matrix"]["x"]["y"] == pytest.approx(
        frame["x"].corr(frame["y"], method="spearman")
    )


def test_spearman_bootstrap_unavailability_retains_inference_and_type_contract():
    frame = pd.DataFrame({"x": [0.0, 0.0, 1.0], "y": [0.0, 1.0, 0.0]})
    direct = StatisticalAnalyzer(frame).spearman_correlation(
        "x", "y", bootstrap_samples=100, random_state=0
    )
    assert direct["statistic"] == pytest.approx(-0.5)
    assert math.isfinite(direct["p_value"])
    assert direct["confidence_interval"] is None
    assert direct["bootstrap"]["valid_resamples"] == 47
    assert direct["warnings"]

    ordinal = ResearchAssistant(frame).run(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="monotonic",
        design="independent",
        variable_types={"x": "ordinal", "y": "ordinal"},
    )
    assert ordinal.analysis is not None
    assert ordinal.analysis.method_id == "spearman_correlation"
    categorical = ResearchAssistant(pd.DataFrame({"x": ["a", "b", "c"], "y": ["d", "e", "f"]})).run(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="monotonic",
        design="independent",
        variable_types={"x": "nominal", "y": "nominal"},
    )
    assert categorical.status is WorkflowStatus.UNSUPPORTED


def test_fisher_sparse_2x2_matches_scipy_and_shared_cross_tab_counts():
    frame = pd.DataFrame(
        {
            "treatment": ["A"] * 5 + ["B"] * 7 + [None],
            "response": ["yes"] + ["no"] * 4 + ["yes"] * 5 + ["no"] * 2 + ["yes"],
        }
    )
    workflow = _categorical(frame)
    assert workflow.status is WorkflowStatus.PARTIAL
    result = workflow.analysis
    assert result is not None and result.method_id == "fisher_exact"
    observed = result.metadata["observed_counts"]
    expected = stats.fisher_exact(observed, alternative="two-sided")
    assert result.values["primary_estimate"] == pytest.approx(expected.statistic)
    assert result.values["p_value"] == pytest.approx(expected.pvalue)
    assert result.sample_size == 12 and result.excluded_rows == 1
    cross_tab = StatisticalAnalyzer(frame).cross_tab(
        "response",
        "treatment",
        data_dictionary={"response": {"type": "nominal"}, "treatment": {"type": "nominal"}},
    )
    assert observed == cross_tab["counts"]
    assert workflow.report is not None
    assert any(
        table["id"] == "observed_contingency_table" for table in workflow.report.to_dict()["tables"]
    )
    assert workflow.audit is not None and workflow.audit.status == "passed"


def test_fisher_zero_cell_is_json_safe_and_ci_is_explicitly_unavailable():
    frame = pd.DataFrame(
        {
            "treatment": ["A"] * 4 + ["B"] * 4,
            "response": ["yes"] * 4 + ["no"] * 4,
        }
    )
    result = _categorical(frame).analysis
    assert result is not None and result.method_id == "fisher_exact"
    assert result.values["primary_estimate"] is None
    assert result.metadata["odds_ratio_status"] == "positive_infinity"
    assert result.values["confidence_interval"] is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_fisher_scope_and_chi_square_selection_policy():
    adequate = pd.DataFrame(
        {
            "treatment": ["A"] * 20 + ["B"] * 20,
            "response": (["yes"] * 10 + ["no"] * 10) * 2,
        }
    )
    assert _categorical(adequate).analysis.method_id == "pearson_chi_square"
    sparse_2x3 = pd.DataFrame(
        {
            "treatment": ["A", "A", "A", "B", "B", "B"],
            "response": ["yes", "no", "maybe", "yes", "no", "maybe"],
        }
    )
    blocked = _categorical(sparse_2x3)
    assert blocked.status in {WorkflowStatus.UNSUPPORTED, WorkflowStatus.DATA_LIMITED}
    assert "limited to 2x2" in blocked.blockers[0]


def test_fisher_direct_ordering_one_level_block_and_reproducibility():
    frame = pd.DataFrame(
        {
            "treatment": ["B", "A", "A", "A", "B", "B"],
            "response": ["no", "yes", "no", "no", "yes", "yes"],
        }
    )
    direct = StatisticalAnalyzer(frame).fisher_exact("treatment", "response")
    assert direct["row_levels"] == ["B", "A"]
    assert direct["column_levels"] == ["no", "yes"]
    expected = stats.fisher_exact(direct["observed_counts"], alternative="two-sided")
    assert direct["odds_ratio"] == pytest.approx(expected.statistic)
    assert direct["p_value"] == pytest.approx(expected.pvalue)

    workflow = _categorical(frame)
    assert workflow.reproducibility is not None
    assert reproduce(workflow.reproducibility, data=frame).status == "reproduced"
    one_level = frame.assign(response="yes")
    assert _categorical(one_level).status in {
        WorkflowStatus.UNSUPPORTED,
        WorkflowStatus.DATA_LIMITED,
    }


def test_declared_targets_not_normality_diagnostics_control_method_selection():
    paired = pd.DataFrame(
        {
            "student": [1, 1, 2, 2, 3, 3, 4, 4],
            "condition": ["after", "before"] * 4,
            "score": [1.0, 100.0, 2.0, 3.0, 4.0, 5.0, 20.0, 6.0],
        }
    )
    rank = _paired(paired)
    mean = ResearchAssistant(paired).run(
        objective="compare_groups",
        outcome="score",
        predictor="condition",
        estimand="mean",
        design="paired",
        unit_id="student",
        condition_order=("after", "before"),
        variable_types={"score": "continuous", "condition": "nominal"},
    )
    assert rank.analysis.method_id == "wilcoxon_signed_rank"
    assert mean.analysis.method_id == "paired_t"
    association = pd.DataFrame({"x": np.arange(1.0, 9.0), "y": [1, 4, 2, 8, 3, 7, 5, 6]})
    assert _spearman(association).analysis.method_id == "spearman_correlation"
    linear = ResearchAssistant(association).run(
        objective="association",
        outcome="x",
        predictor="y",
        estimand="linear",
        design="independent",
        variable_types={"x": "continuous", "y": "continuous"},
    )
    assert linear.analysis.method_id == "pearson_correlation"
