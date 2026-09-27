"""Statistical coverage for descriptive data-understanding helpers."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from pyautostat import (
    ColumnNotFoundError,
    InsightEngine,
    InsufficientDataError,
    InvalidDataError,
    ReportGenerator,
    ResearchAssistant,
    StatisticalAnalyzer,
    coefficient_of_variation_narrative,
    crosstab_narrative,
    frequency_narrative,
    percentile_narrative,
)


def test_default_percentiles_use_linear_interpolation_and_match_median():
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]})
    before = frame.copy(deep=True)
    profile = ResearchAssistant(frame).profile()
    stats = profile["descriptive"]["value"]

    assert list(stats["percentiles"]) == ["p05", "p25", "p50", "p75", "p95"]
    assert stats["percentiles"] == pytest.approx(
        {"p05": 1.2, "p25": 2.0, "p50": 3.0, "p75": 4.0, "p95": 4.8}
    )
    assert stats["percentiles"]["p50"] == pytest.approx(stats["median"])
    assert stats["percentile_method"] == "pandas linear interpolation"
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    "values",
    ([1, 2, 3, 4], [-4, -2, -2, 8], [7, 7, 7], [1e-200, 2e-200, 3e-200]),
)
def test_percentiles_cover_even_negative_tied_constant_and_small_finite_values(values):
    stats = ResearchAssistant(pd.DataFrame({"value": values})).profile()["descriptive"]["value"]
    expected = pd.Series(values).quantile([0.05, 0.25, 0.5, 0.75, 0.95], interpolation="linear")
    assert list(stats["percentiles"].values()) == pytest.approx(expected.tolist())


def test_custom_quantiles_are_sorted_deduplicated_and_validated():
    assistant = ResearchAssistant(pd.DataFrame({"value": [0.0, 10.0, 20.0]}))
    profile = assistant.profile(quantiles=(0.9, 0.1, 0.5, 0.1))
    assert profile["profile_metadata"]["quantiles"] == [0.1, 0.5, 0.9]
    assert list(profile["descriptive"]["value"]["percentiles"]) == ["p10", "p50", "p90"]
    with pytest.raises(InvalidDataError, match="quantile"):
        assistant.profile(quantiles=(-0.1, 0.5))
    with pytest.raises(InvalidDataError, match="quantiles"):
        assistant.profile(quantiles=())


def test_all_missing_numeric_percentiles_and_cv_are_explicitly_unavailable():
    stats = ResearchAssistant(pd.DataFrame({"value": [np.nan, np.nan]})).profile()["descriptive"][
        "value"
    ]
    assert all(value is None for value in stats["percentiles"].values())
    assert stats["coefficient_of_variation"] is None
    assert stats["coefficient_of_variation_details"]["status"] == "unavailable"
    assert "unavailable" in coefficient_of_variation_narrative(stats).lower()


def test_cv_uses_absolute_mean_and_has_scientific_safeguards():
    positive = ResearchAssistant(pd.DataFrame({"value": [10.0, 20.0, 30.0]})).profile()[
        "descriptive"
    ]["value"]
    negative = ResearchAssistant(pd.DataFrame({"value": [-10.0, -20.0, -30.0]})).profile()[
        "descriptive"
    ]["value"]
    centered = ResearchAssistant(pd.DataFrame({"value": [-1.0, 1.0]})).profile()["descriptive"][
        "value"
    ]
    near_zero = ResearchAssistant(
        pd.DataFrame({"value": [-1.0, 1.0 + np.finfo(float).eps]})
    ).profile()["descriptive"]["value"]
    constant = ResearchAssistant(pd.DataFrame({"value": [4.0, 4.0, 4.0]})).profile()["descriptive"][
        "value"
    ]
    tiny = ResearchAssistant(pd.DataFrame({"value": [1e-200, 2e-200, 3e-200]})).profile()[
        "descriptive"
    ]["value"]

    assert positive["coefficient_of_variation"] == pytest.approx(50.0)
    assert negative["coefficient_of_variation"] == pytest.approx(50.0)
    assert centered["coefficient_of_variation"] is None
    assert near_zero["coefficient_of_variation"] is None
    assert constant["coefficient_of_variation"] == 0
    assert tiny["coefficient_of_variation"] is None
    assert tiny["coefficient_of_variation_details"]["reason"] == "numeric_underflow"
    assert "ratio-scale" in positive["coefficient_of_variation_details"]["applicability"]
    assert "good" not in coefficient_of_variation_narrative(positive).lower()
    singleton = ResearchAssistant(pd.DataFrame({"value": [5.0]})).profile()["descriptive"]["value"]
    assert singleton["coefficient_of_variation_details"]["reason"] == (
        "sample_standard_deviation_unavailable"
    )


def test_nominal_frequency_table_accounts_for_missing_values_and_ties():
    result = ResearchAssistant(pd.DataFrame({"group": ["B", "A", "B", "A", None]})).frequency_table(
        "group"
    )
    assert result["valid_n"] == 4
    assert result["missing_n"] == 1
    assert result["total_n"] == 5
    assert [row["level"] for row in result["levels"]] == ["B", "A"]
    assert sum(row["count"] for row in result["levels"]) == result["valid_n"]
    assert sum(row["percent"] for row in result["levels"]) == pytest.approx(100)
    assert all("cumulative_percent" not in row for row in result["levels"])
    assert "tied" in result["narrative"].lower()
    assert result["narrative"] == frequency_narrative(result)
    json.dumps(result, allow_nan=False)
    ranked = ResearchAssistant(pd.DataFrame({"group": ["B", "A", "A"]})).frequency_table("group")
    assert [row["level"] for row in ranked["levels"]] == ["A", "B"]
    assert ranked["ordering"] == "descending_frequency_first_observed_ties"


def test_ordinal_frequency_table_preserves_declared_and_dtype_order():
    frame = pd.DataFrame({"rating": ["high", "low", "medium", "high"]})
    declared = ResearchAssistant(frame).frequency_table(
        "rating",
        data_dictionary={"rating": {"type": "ordinal", "ordinal_order": ["low", "medium", "high"]}},
    )
    assert [row["level"] for row in declared["levels"]] == ["low", "medium", "high"]
    assert declared["levels"][-1]["cumulative_percent"] == pytest.approx(100)

    ordered = pd.DataFrame(
        {
            "rating": pd.Categorical(
                ["high", "low", "medium"],
                categories=["low", "medium", "high", "unused"],
                ordered=True,
            )
        }
    )
    dtype_result = ResearchAssistant(ordered).frequency_table("rating")
    assert [row["level"] for row in dtype_result["levels"]] == ["low", "medium", "high"]
    assert dtype_result["ordering"] == "ordered_categorical_dtype"


def test_unordered_ordinal_omits_cumulative_percentage_with_explanation():
    frame = pd.DataFrame({"rating": ["low", "high", "medium"]})
    result = ResearchAssistant(frame).frequency_table(
        "rating", data_dictionary={"rating": {"type": "ordinal"}}
    )
    assert result["cumulative_percentage_status"] == "omitted_unordered"
    assert all("cumulative_percent" not in row for row in result["levels"])
    assert "no complete ordinal order" in result["narrative"].lower()

    incomplete = ResearchAssistant(
        pd.DataFrame({"rating": ["low", "medium", "high"]})
    ).frequency_table(
        "rating",
        data_dictionary={"rating": {"type": "ordinal", "ordinal_order": ["low", "high"]}},
    )
    assert incomplete["ordering"] == "declared_order_incomplete"
    assert incomplete["cumulative_percentage_status"] == "omitted_unordered"


def test_frequency_table_supports_boolean_one_level_and_bounds_narration():
    boolean = ResearchAssistant(pd.DataFrame({"flag": [True, False, True]})).frequency_table("flag")
    assert boolean["analytical_type"] == "boolean"
    assert boolean["cumulative_percentage_status"] == "not_applicable"
    one = ResearchAssistant(pd.DataFrame({"group": ["A", "A"]})).frequency_table("group")
    assert one["levels"] == [{"level": "A", "count": 2, "percent": 100.0, "total_percent": 100.0}]
    many = ResearchAssistant(
        pd.DataFrame({"group": [f"level-{index}" for index in range(30)]})
    ).frequency_table("group")
    assert len(many["levels"]) == 30
    assert "20 additional level(s)" in many["narrative"]


def test_profile_story_and_insights_use_bounded_categorical_summaries():
    frame = pd.DataFrame(
        {
            "group": ["A", "A", "B", "A"] + [f"level-{index}" for index in range(21)],
            "score": list(range(25)),
        }
    )
    assistant = ResearchAssistant(frame)
    profile = assistant.profile()
    story = assistant.summarize(mode="story")
    insights = InsightEngine(profile).generate_insights()
    assert "CATEGORICAL DISTRIBUTIONS" in story
    assert "'A' is the most frequent recorded level" in story
    assert any(item["category"] == "Categorical Cardinality" for item in insights)
    assert len(profile["categorical_summary"]["group"]["frequencies"]) == 20


def test_frequency_table_rejects_invalid_uses():
    assistant = ResearchAssistant(
        pd.DataFrame(
            {
                "participant_id": ["p1", "p2", "p3"],
                "measure": [0.1, 0.2, 0.3],
                "empty": [None, None, None],
            }
        )
    )
    with pytest.raises(ColumnNotFoundError):
        assistant.frequency_table("unknown")
    with pytest.raises(InvalidDataError, match="identifier"):
        assistant.frequency_table("participant_id")
    with pytest.raises(InvalidDataError, match="not an analytical categorical"):
        assistant.frequency_table("measure")
    with pytest.raises(InsufficientDataError, match="no non-missing"):
        assistant.frequency_table("empty")


def test_cross_tabulation_percentages_and_complete_case_accounting():
    frame = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", None, "C"],
            "choice": ["X", "Y", "X", "X", "Y", None],
        }
    )
    before = frame.copy(deep=True)
    result = ResearchAssistant(frame).cross_tab("group", "choice")
    assert result["counts"] == [[1, 1], [2, 0]]
    assert result["valid_n"] == 4
    assert result["original_rows"] == 6
    assert result["excluded_rows"] == 2
    assert all(sum(row) == pytest.approx(100) for row in result["row_percent"])
    assert all(
        sum(result["column_percent"][row][column] for row in range(len(result["row_levels"])))
        == pytest.approx(100)
        for column in range(len(result["column_levels"]))
    )
    assert sum(map(sum, result["total_percent"])) == pytest.approx(100)
    assert "observed distribution only" in result["narrative"]
    assert "significant" not in result["narrative"].lower()
    assert result["narrative"] == crosstab_narrative(result)
    json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(frame, before)


def test_cross_tabulation_supports_declared_numeric_categories_and_single_level():
    frame = pd.DataFrame({"group": [1, 1, 1], "choice": [0, 1, 1]})
    dictionary = {
        "group": {"type": "nominal"},
        "choice": {"type": "nominal"},
    }
    result = ResearchAssistant(frame).cross_tab("group", "choice", data_dictionary=dictionary)
    assert result["shape"] == [1, 2]
    assert result["counts"] == [[1, 2]]


def test_cross_tabulation_retains_large_n_by_m_table_but_bounds_narration():
    frame = pd.DataFrame(
        {
            "row": [f"r{index % 11}" for index in range(121)],
            "column": [f"c{index // 11}" for index in range(121)],
        }
    )
    result = ResearchAssistant(frame).cross_tab("row", "column")
    assert result["shape"] == [11, 11]
    assert result["cell_count"] == 121
    assert result["large_table"] is True
    assert sum(map(sum, result["counts"])) == 121
    assert "full 11 by 11 table is retained" in result["narrative"].lower()


def test_cross_tabulation_rejects_empty_complete_cases_and_same_column():
    assistant = ResearchAssistant(pd.DataFrame({"a": ["x", None], "b": [None, "y"]}))
    with pytest.raises(InsufficientDataError, match="no complete rows"):
        assistant.cross_tab("a", "b")
    with pytest.raises(InvalidDataError, match="different columns"):
        assistant.cross_tab("a", "a")


def test_shared_contingency_backend_preserves_existing_chi_square_contrast_order():
    frame = pd.DataFrame(
        {
            "group": pd.Categorical(["A"] * 20 + ["B"] * 20, categories=["B", "A"], ordered=True),
            "choice": ["yes"] * 15 + ["no"] * 5 + ["yes"] * 5 + ["no"] * 15,
        }
    )
    analyzer = StatisticalAnalyzer(frame)
    inferred = analyzer.categorical_association("group", "choice", bootstrap_samples=0)
    described = analyzer.cross_tab("group", "choice")
    assert inferred["groups"] == ["A", "B"]
    assert inferred["observed_counts"] == [[15, 5], [5, 15]]
    assert described["row_levels"] == ["B", "A"]
    assert described["counts"] == [[5, 15], [15, 5]]


def test_narration_uses_stored_percentiles_and_is_deterministic():
    stats = ResearchAssistant(pd.DataFrame({"value": [1, 2, 3, 4, 5]})).profile()["descriptive"][
        "value"
    ]
    text = percentile_narrative(stats)
    assert "50th percentile" in text
    assert "middle 50%" in text
    assert "95th percentile" in text
    assert text == percentile_narrative(stats)
    assert not any(word in text.lower() for word in ("causes", "important", "representative"))


def test_outputs_never_expose_nonfinite_cv_values():
    stats = ResearchAssistant(pd.DataFrame({"value": [1e308, 1e308]})).profile()["descriptive"][
        "value"
    ]
    detail = stats["coefficient_of_variation_details"]
    assert detail["status"] in {"available", "unavailable"}
    assert detail["percent"] is None or math.isfinite(detail["percent"])
    json.dumps(stats, allow_nan=False)


def test_profile_reports_render_percentiles_cv_and_bounded_frequencies_safely():
    frame = pd.DataFrame({"score": [1.0, 2.0, 3.0], "group": ["<A>", "B", "<A>"]})
    assistant = ResearchAssistant(frame)
    profile = assistant.profile()
    legacy_html = ReportGenerator(profile).to_html()
    assert "<th>P5</th>" in legacy_html
    assert "<th>CV</th>" in legacy_html
    assert "Categorical Frequency Summaries" in legacy_html
    assert "&lt;A&gt;" in legacy_html

    report = assistant.run(objective="descriptive").report.to_dict()
    tables = {table["id"]: table for table in report["tables"]}
    assert "P95" in tables["descriptive_statistics"]["columns"]
    assert "CV (%)" in tables["descriptive_statistics"]["columns"]
    assert "frequency_1" in tables
