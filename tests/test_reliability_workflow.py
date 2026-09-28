"""Cronbach-alpha workflow, diagnostics, safety, and integration tests."""

import json

import numpy as np
import pandas as pd
import pytest

from pyautostat import (
    AnalysisOptions,
    ResearchAssistant,
    StatisticalAnalyzer,
    build_session_snapshot,
    reproduce,
)
from pyautostat.exceptions import InvalidDataError


@pytest.fixture
def scale_frame():
    return pd.DataFrame(
        {
            "q1": [1, 2, 3, 4, 5, 2, 4, 3, 5, 1, 4, 2],
            "q2": [1, 2, 3, 4, 4, 2, 5, 3, 5, 1, 3, 2],
            "q3": [2, 2, 3, 5, 5, 2, 4, 4, 5, 1, 4, 1],
            "unrelated": list("abcdefghijkl"),
        }
    )


def _expected_alpha(frame, items):
    matrix = frame[list(items)].dropna().to_numpy(dtype=float)
    k = matrix.shape[1]
    return (
        k
        / (k - 1)
        * (1 - np.var(matrix, axis=0, ddof=1).sum() / np.var(matrix.sum(axis=1), ddof=1))
    )


def test_beginner_workflow_matches_formula_and_integrates_outputs(scale_frame):
    workflow = ResearchAssistant(scale_frame).reliability(["q1", "q2", "q3"], bootstrap_samples=79)
    assert workflow.status.value == "completed"
    assert workflow.analysis.method_id == "cronbach_alpha"
    assert workflow.analysis.values["cronbach_alpha"] == pytest.approx(
        _expected_alpha(scale_frame, ["q1", "q2", "q3"])
    )
    assert "p_value" not in workflow.analysis.values
    assert workflow.analysis.metadata["inference"] is False
    assert workflow.audit.status == "passed"
    table_ids = {table["id"] for table in workflow.report.to_dict()["tables"]}
    assert {
        "reliability_summary",
        "reliability_items",
        "inter_item_correlations",
    } <= table_ids
    assert "RELIABILITY ESTIMATE" in workflow.explain()
    assert "ITEM DIAGNOSTICS" in workflow.explain()
    json.loads(workflow.to_json())


def test_bootstrap_is_deterministic_and_replayable(scale_frame):
    first = ResearchAssistant(scale_frame).reliability(
        ["q1", "q2", "q3"], bootstrap_samples=79, random_state=17
    )
    second = ResearchAssistant(scale_frame).reliability(
        ["q1", "q2", "q3"], bootstrap_samples=79, random_state=17
    )
    assert (
        first.analysis.values["confidence_interval"]
        == second.analysis.values["confidence_interval"]
    )
    interval = first.analysis.values["confidence_interval"]
    assert interval["requested_resamples"] == 79
    assert interval["valid_resamples"] <= 79
    assert interval["random_state"] == 17
    matrix = scale_frame[["q1", "q2", "q3"]].to_numpy(dtype=float)
    rng = np.random.default_rng(17)
    reference = []
    for _ in range(79):
        sampled_rows = matrix[rng.integers(0, len(matrix), size=len(matrix)), :]
        value = _expected_alpha(pd.DataFrame(sampled_rows), [0, 1, 2])
        if np.isfinite(value):
            reference.append(value)
    expected_bounds = np.quantile(reference, [0.025, 0.975])
    assert interval["lower"] == pytest.approx(expected_bounds[0])
    assert interval["upper"] == pytest.approx(expected_bounds[1])
    alternate = ResearchAssistant(scale_frame).reliability(
        ["q1", "q2", "q3"], bootstrap_samples=79, random_state=18
    )
    assert alternate.analysis.values["confidence_interval"] != interval
    assert reproduce(first.reproducibility, data=scale_frame).status == "reproduced"


def test_complete_case_counts_missingness_and_source_preservation(scale_frame):
    frame = scale_frame.copy()
    frame.loc[0, "q1"] = np.nan
    frame.loc[1, "q2"] = np.nan
    before = frame.copy(deep=True)
    workflow = ResearchAssistant(frame).reliability(["q1", "q2", "q3"], bootstrap_samples=59)
    assert workflow.analysis.sample_size == 10
    assert workflow.analysis.excluded_rows == 2
    assert workflow.analysis.metadata["sample"]["original_rows"] == 12
    by_item = {row["item"]: row for row in workflow.analysis.values["missingness"]}
    assert by_item["q1"]["missing_count"] == 1
    assert by_item["q2"]["missing_count"] == 1
    assert by_item["q3"]["missing_count"] == 0
    assert "16.7%" in workflow.explain()
    assert "'q1', 'q2'" in workflow.explain()
    pd.testing.assert_frame_equal(frame, before)


def test_corrected_item_total_and_alpha_if_deleted_are_independent_calculations(scale_frame):
    workflow = ResearchAssistant(scale_frame).reliability(["q1", "q2", "q3"], bootstrap_samples=59)
    stats = {row["item"]: row for row in workflow.analysis.values["item_statistics"]}
    expected_corrected = np.corrcoef(scale_frame["q1"], scale_frame["q2"] + scale_frame["q3"])[0, 1]
    assert stats["q1"]["corrected_item_total_correlation"] == pytest.approx(expected_corrected)
    naive = np.corrcoef(
        scale_frame["q1"], scale_frame["q1"] + scale_frame["q2"] + scale_frame["q3"]
    )[0, 1]
    assert stats["q1"]["corrected_item_total_correlation"] != pytest.approx(naive)
    assert stats["q1"]["corrected_total_excludes_focal_item"] is True
    for item in ("q1", "q2", "q3"):
        remaining = [name for name in ("q1", "q2", "q3") if name != item]
        expected_deleted = _expected_alpha(scale_frame, remaining)
        assert stats[item]["alpha_if_deleted"] == pytest.approx(expected_deleted)
        assert stats[item]["delta_from_full_alpha"] == pytest.approx(
            expected_deleted - workflow.analysis.values["cronbach_alpha"]
        )


def test_two_item_scale_keeps_alpha_but_marks_deletion_not_applicable(scale_frame):
    workflow = ResearchAssistant(scale_frame).reliability(["q1", "q2"], bootstrap_samples=59)
    assert workflow.status.value == "completed"
    assert all(
        row["alpha_if_deleted_status"] == "not_applicable" and row["alpha_if_deleted"] is None
        for row in workflow.analysis.values["item_statistics"]
    )
    assert any("only two items" in warning for warning in workflow.warnings)


def test_negative_alpha_is_preserved_without_automatic_rescoring():
    frame = pd.DataFrame(
        {
            "forward": np.arange(1.0, 9.0),
            "opposed": np.arange(8.0, 0.0, -1),
            "third": [1, 2, 1, 2, 1, 2, 1, 3],
        }
    )
    workflow = ResearchAssistant(frame).reliability(
        ["forward", "opposed", "third"], bootstrap_samples=59
    )
    assert workflow.analysis.values["cronbach_alpha"] < 0
    assert workflow.analysis.values["scoring"]["automatic_reverse_scoring"] is False
    assert workflow.analysis.values["scoring"]["reversed_items"] == []
    assert "negative" in " ".join(workflow.warnings).lower()


def test_explicit_reverse_scoring_is_validated_recorded_and_nonmutating():
    frame = pd.DataFrame(
        {
            "q1": [1, 2, 3, 4, 5, 4, 2, 3],
            "q2_reverse": [5, 4, 3, 2, 1, 2, 4, 3],
            "q3": [1, 2, 4, 4, 5, 4, 2, 3],
            "q4_reverse": [10, 9, 8, 7, 6, 7, 9, np.nan],
        }
    )
    before = frame.copy(deep=True)
    transformed = frame.assign(
        q2_reverse=6 - frame["q2_reverse"],
        q4_reverse=16 - frame["q4_reverse"],
    )
    workflow = ResearchAssistant(frame).reliability(
        ["q1", "q2_reverse", "q3", "q4_reverse"],
        bootstrap_samples=59,
        reverse_scoring={"q2_reverse": (1, 5), "q4_reverse": (6, 10)},
    )
    assert workflow.analysis.values["cronbach_alpha"] == pytest.approx(
        _expected_alpha(transformed, ["q1", "q2_reverse", "q3", "q4_reverse"])
    )
    scoring = workflow.analysis.values["scoring"]
    assert scoring["reverse_scoring_applied"] is True
    assert [item["item"] for item in scoring["reversed_items"]] == [
        "q2_reverse",
        "q4_reverse",
    ]
    assert scoring["reversed_items"][0]["formula"] == "lower + upper - original"
    assert "Explicit researcher-supplied reverse scoring" in workflow.explain()
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(InvalidDataError, match="outside reverse-scoring bounds"):
        StatisticalAnalyzer(frame).scale_reliability(
            ["q1", "q2_reverse", "q3"], reverse_scoring={"q2_reverse": (1, 4)}
        )


def test_constant_item_is_retained_when_alpha_defined_and_correlations_are_explicit(scale_frame):
    frame = scale_frame.assign(constant=3)
    workflow = ResearchAssistant(frame).reliability(["q1", "q2", "constant"], bootstrap_samples=59)
    assert workflow.analysis.status.value == "available"
    constant = workflow.analysis.values["item_statistics"][2]
    assert constant["corrected_item_total_status"] == "unavailable"
    matrix = workflow.analysis.values["inter_item_correlations"]["values"]
    assert matrix[2] == [None, None, None]
    assert workflow.status.value == "partial"


def test_zero_total_variance_and_too_few_complete_rows_are_data_limited():
    zero_total = pd.DataFrame({"q1": [1, 2, 3], "q2": [3, 2, 1]})
    result = ResearchAssistant(zero_total).reliability(["q1", "q2"])
    assert result.status.value == "data_limited"
    assert "zero variance" in " ".join(result.blockers).lower()
    sparse = pd.DataFrame({"q1": [1.0, np.nan], "q2": [1.0, 2.0]})
    result = ResearchAssistant(sparse).reliability(["q1", "q2"])
    assert result.status.value == "data_limited"
    assert "at least two respondents" in " ".join(result.blockers).lower()
    all_constant = pd.DataFrame({"q1": [2, 2, 2], "q2": [4, 4, 4], "q3": [1, 1, 1]})
    result = ResearchAssistant(all_constant).reliability(["q1", "q2", "q3"])
    assert result.status.value == "data_limited"
    json.dumps(result.to_dict(), allow_nan=False)


def test_low_bootstrap_count_makes_interval_partial_not_failed(scale_frame):
    workflow = ResearchAssistant(scale_frame).reliability(["q1", "q2", "q3"], bootstrap_samples=10)
    assert workflow.status.value == "partial"
    assert workflow.analysis.status.value == "available"
    assert workflow.analysis.values["confidence_interval"]["status"] == "unavailable"
    assert workflow.report.status == "partial"
    assert workflow.audit.status == "passed"


def test_numeric_ordinal_frequency_support_and_identifier_rejection(scale_frame):
    dictionary = {item: {"type": "ordinal"} for item in ("q1", "q2", "q3")}
    workflow = ResearchAssistant(scale_frame).reliability(
        ["q1", "q2", "q3"], data_dictionary=dictionary, bootstrap_samples=59
    )
    assert set(workflow.analysis.values["item_frequencies"]) == {"q1", "q2", "q3"}
    identifier = ResearchAssistant(scale_frame).reliability(
        ["q1", "q2", "q3"],
        data_dictionary={"q1": {"type": "identifier"}},
        bootstrap_samples=59,
    )
    assert identifier.status.value == "unsupported"
    assert "identifier" in " ".join(identifier.blockers).lower()


def test_invalid_item_inputs_and_non_numeric_items_are_rejected(scale_frame):
    assistant = ResearchAssistant(scale_frame)
    with pytest.raises(InvalidDataError, match="at least two"):
        assistant.reliability(["q1"])
    with pytest.raises(InvalidDataError, match="duplicates"):
        assistant.reliability(["q1", "q1"])
    result = assistant.reliability(["q1", "unrelated"])
    assert result.status.value == "unsupported"
    assert "numeric" in " ".join(result.blockers).lower()


def test_inter_item_matrix_order_symmetry_and_negative_pair_cues():
    frame = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [1, 3, 2, 5, 4, 6],
            "c": [6, 5, 4, 2, 3, 1],
        }
    )
    workflow = ResearchAssistant(frame).reliability(["b", "a", "c"], bootstrap_samples=59)
    correlations = workflow.analysis.values["inter_item_correlations"]
    assert correlations["items"] == ["b", "a", "c"]
    matrix = np.asarray(correlations["values"], dtype=float)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(matrix, np.corrcoef(frame[["b", "a", "c"]].to_numpy().T))
    negative = workflow.analysis.values["negative_inter_item_correlations"]
    assert negative["count"] >= 1
    assert negative["most_negative_pair"]["correlation"] == min(
        pair["correlation"] for pair in negative["pairs"]
    )


def test_language_has_no_universal_cutoff_or_validity_claim(scale_frame):
    workflow = ResearchAssistant(scale_frame).reliability(["q1", "q2", "q3"], bootstrap_samples=59)
    text = " ".join(
        [
            workflow.interpretation.summary,
            workflow.interpretation.method_explanation,
            *workflow.interpretation.limitations,
        ]
    ).lower()
    assert ".70" not in text
    assert "universal adequacy cutoff" in text
    assert "does not establish unidimensionality" in text
    assert "construct validity" in text
    assert workflow.interpretation.hypothesis_interpretation is None
    assert workflow.analysis.values["items"] == ["q1", "q2", "q3"]
    assert "not instructions to delete" in workflow.explain()


def test_report_escapes_item_names_and_session_is_json_safe():
    item = "q<script>alert(1)</script>"
    frame = pd.DataFrame(
        {
            item: [1, 2, 3, 4, 5],
            "Q&A": [1, 2, 3, 4, 4],
            "<Item 1>": [1, 2, 3, 5, 5],
        }
    )
    workflow = ResearchAssistant(frame).reliability([item, "Q&A", "<Item 1>"], bootstrap_samples=59)
    html = workflow.report.to_html()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "Q&amp;A" in html
    assert "&lt;Item 1&gt;" in html
    snapshot = build_session_snapshot(workflow)
    payload = json.loads(snapshot.to_json())
    assert "reliability" in payload["capabilities"]["objectives"]
    assert "run_sensitivity" not in payload["available_actions"]
    assert "assess_practical_significance" not in payload["available_actions"]


def test_options_are_reliability_specific(scale_frame):
    with pytest.raises(InvalidDataError, match="only for objective='reliability'"):
        ResearchAssistant(scale_frame).run(
            objective="descriptive", options=AnalysisOptions(bootstrap_samples=10)
        )
