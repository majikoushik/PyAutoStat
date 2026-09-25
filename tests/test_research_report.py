"""Report assembly and export from genuine analysis results."""

import csv
import io
import json
import math
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from pyautostat import (
    AnalysisResult,
    AnalysisStatus,
    Recommendation,
    ReportError,
    ReportGenerator,
    ResearchAssistant,
    ResearchReport,
    StatisticalAnalyzer,
    execution,
)


@pytest.fixture(scope="module")
def welch_case():
    frame = pd.DataFrame(
        {
            "group": ["B"] * 8 + ["A"] * 8 + ["A"],
            "score": list(range(1, 9)) + list(range(3, 11)) + [None],
            "private_id": [f"secret-{i}" for i in range(17)],
        }
    )
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        description="Compare examination scores.",
        data_dictionary={"score": {"type": "continuous", "unit": "points"}},
    )
    return frame, assistant, assistant.analyze(draft)


def _table(report, identifier):
    return next(table for table in report.to_dict()["tables"] if table["id"] == identifier)


def _values(result, **changes):
    values = deepcopy(result.values)
    values.update(changes)
    return replace(result, values=values)


def test_welch_report_uses_one_record_and_is_deterministic(welch_case, tmp_path, monkeypatch):
    frame, assistant, result = welch_case
    source = deepcopy(result.to_dict())
    before_frame = frame.copy(deep=True)
    supplied = assistant.interpret(result)
    before_interpretation = deepcopy(supplied.to_dict())

    def forbidden(*args, **kwargs):
        raise AssertionError("reporting must not run a statistical backend")

    monkeypatch.setattr(StatisticalAnalyzer, "hypothesis_tests", forbidden)
    report = assistant.report(result, interpretation=supplied)
    assert isinstance(report, ResearchReport)
    assert report.status == "complete"
    assert report.title == "Statistical Research Report"
    payload = report.to_dict()
    assert payload["analysis"] == source
    assert payload["interpretation"] == before_interpretation
    assert payload["sections"]["methods"]["method_id"] == "welch_t"
    assert payload["sections"]["methods"]["declared_design"] == "independent"
    assert payload["sections"]["dataset"]["group_order"] == ["B", "A"]
    assert payload["sections"]["dataset"]["contrast"]["first"] == "B"
    assert payload["sections"]["dataset"]["original_rows"] == 17
    assert payload["sections"]["dataset"]["analyzed_rows"] == 16
    assert payload["sections"]["dataset"]["excluded_rows"] == 1
    assert payload["sections"]["results"]["p_value"] == result.values["p_value"]
    assert payload["sections"]["results"]["primary_estimate"] == result.values["primary_estimate"]
    assert (
        payload["sections"]["results"]["effect_size"]["value"]
        == result.values["effect_size"]["value"]
    )
    assert payload["sections"]["results"]["confidence_interval"]["quantity"] == "mean difference"
    assert _table(report, "confidence_intervals")["rows"][1][0]["value"] == "Cohen's d"
    assert json.loads(report.to_json()) == payload
    assert report.to_json() == assistant.report(result).to_json()
    assert report.to_html() == assistant.report(result).to_html()
    assert report.to_markdown() == assistant.report(result).to_markdown()
    assert not list(tmp_path.iterdir())
    assert "secret-0" not in report.to_json()
    assert source == result.to_dict()
    assert before_interpretation == supplied.to_dict()
    pd.testing.assert_frame_equal(frame, before_frame)


def test_supplied_interpretation_must_match_the_exact_result(welch_case):
    _, assistant, result = welch_case
    interpretation = assistant.interpret(result)
    altered = _values(result, p_value=0.001)
    with pytest.raises(ReportError, match="does not match"):
        assistant.report(altered, interpretation=interpretation)
    wrong_method = replace(interpretation, method_id="pearson_correlation")
    with pytest.raises(ReportError, match="does not match"):
        assistant.report(result, interpretation=wrong_method)


def test_partial_mean_difference_survives_missing_or_opposing_d(welch_case):
    _, assistant, result = welch_case
    for effect in (None, {**deepcopy(result.values["effect_size"]), "value": 1.5}):
        report = assistant.report(_values(result, effect_size=effect))
        assert report.status == "partial"
        data = report.to_dict()
        assert data["sections"]["results"]["primary_estimate"] == result.values["primary_estimate"]
        assert data["sections"]["results"]["effect_size"] is None
        assert "mean difference" in data["sections"]["interpretation"]["effect"]
        assert all(table["id"] != "effect_estimates" for table in data["tables"])
        assert "Cohen's d was 1.5" not in report.to_html()


def test_valid_percentile_interval_outside_estimate_is_reported(welch_case):
    _, assistant, result = welch_case
    values = deepcopy(result.values)
    d = values["effect_size"]["value"]
    values["effect_size"]["confidence_interval"]["lower"] = d + 0.5
    values["effect_size"]["confidence_interval"]["upper"] = d + 1
    report = assistant.report(replace(result, values=values))
    assert report.status == "complete"
    rows = _table(report, "confidence_intervals")["rows"]
    assert rows[1][1]["value"] == d + 0.5
    assert rows[1][0]["value"] == "Cohen's d"


def test_pearson_report_is_partial_without_fabricated_interval():
    frame = pd.DataFrame({"hours": [1.0, 2.0, 3.0, 4.0, 5.0], "score": [2.0, 4.0, 3.0, 6.0, 7.0]})
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="association",
        outcome="hours",
        predictor="score",
        estimand="linear",
        design="independent",
        variable_types={"hours": "continuous", "score": "continuous"},
    )
    result = assistant.analyze(draft)
    report = assistant.report(result)
    assert report.status == "partial"
    assert report.to_dict()["sections"]["dataset"]["effective_pair_count"] == 5
    assert report.to_dict()["sections"]["results"]["confidence_interval"] is None
    assert "confidence interval" in report.to_markdown().lower()
    assert all(table["id"] != "confidence_intervals" for table in report.to_dict()["tables"])
    assert "causation" in report.to_html()


def test_descriptive_report_has_genuine_profile_table_and_optional_figure():
    frame = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0], "group": ["A", "A", "B", "B"]})
    assistant = ResearchAssistant(frame)
    result = assistant.analyze(assistant.prepare_question(objective="descriptive"))
    report = assistant.report(result)
    assert report.status == "complete"
    assert report.to_dict()["figures"] == []
    assert report.to_dict()["sections"]["results"]["p_value"] is None
    assert _table(report, "descriptive_statistics")["rows"][0][2]["value"] == 2.5
    with_figures = assistant.report(result, include_figures=True)
    assert (
        with_figures.to_dict()["figures"][0]["counts"]
        == result.values["profile"]["histograms"]["score"]["counts"]
    )
    assert "histogram_1_bins" in with_figures.to_csv_tables()
    assert "Histogram bin data" in with_figures.to_html()
    assert "Histogram bin data" in with_figures.to_markdown()


def test_descriptive_report_omits_identifier_category_values():
    frame = pd.DataFrame(
        {
            "participant_id": ["secret-1", "secret-2", "secret-3"],
            "score": [1.0, 2.0, 3.0],
        }
    )
    assistant = ResearchAssistant(frame)
    result = assistant.analyze(assistant.prepare_question(objective="descriptive"))
    assert "secret-1" in json.dumps(result.to_dict())
    generic = ResearchAssistant(
        pd.DataFrame({"code": ["secret-a", "secret-b", "secret-c"], "score": [1.0, 2.0, 3.0]})
    )
    generic_result = generic.analyze(generic.prepare_question(objective="descriptive"))
    assert "secret-a" not in generic.report(generic_result).to_json()
    report = assistant.report(result)
    assert "secret-1" not in report.to_json()
    assert "secret-1" not in report.to_html()
    assert "secret-1" not in report.to_markdown()
    assert "secret-1" not in "".join(report.to_csv_tables().values())
    assert any("Identifier category labels" in item for item in report.to_dict()["limitations"])
    assert "secret-1" in json.dumps(result.to_dict())


@pytest.mark.parametrize(
    ("estimand", "method"),
    [("distribution", "mann_whitney_u"), ("mean", "welch_t")],
)
def test_group_method_tables_use_selected_method(welch_case, estimand, method):
    frame, _, _ = welch_case
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand=estimand,
        design="independent",
        variable_types={"score": "continuous"},
    )
    result = assistant.analyze(draft)
    report = assistant.report(result)
    assert result.method_id == method
    assert report.to_dict()["sections"]["methods"]["method_id"] == method
    assert report.to_dict()["sections"]["dataset"]["group_order"] == ["B", "A"]


def test_unavailable_analysis_cannot_look_successful(welch_case):
    _, assistant, result = welch_case
    unavailable = replace(result, status=AnalysisStatus.UNAVAILABLE)
    report = assistant.report(unavailable)
    assert report.status == "unavailable"
    assert report.to_dict()["sections"]["results"]["p_value"] is None
    assert all(table["id"] != "statistical_results" for table in report.to_dict()["tables"])
    assert "unavailable" in report.to_html().lower()
    assert "No successful statistical result" in report.to_markdown()
    assert report.to_dict()["analysis"]["values"] == {}
    assert report.to_dict()["sections"]["methods"]["execution_status"] == "unavailable"


def test_sample_arithmetic_and_metadata_contradictions_block_report(welch_case):
    _, assistant, result = welch_case
    with pytest.raises(ReportError, match="row counts disagree"):
        assistant.report(replace(result, excluded_rows=2))
    metadata = deepcopy(result.metadata)
    metadata["sample"]["group_sizes"][0]["size"] = 99
    with pytest.raises(ReportError, match="Per-group"):
        assistant.report(replace(result, metadata=metadata))


def test_raw_p_value_and_csv_numeric_cells_agree(welch_case):
    _, assistant, result = welch_case
    report = assistant.report(result)
    rows = list(csv.reader(io.StringIO(report.to_csv_tables()["statistical_results"])))
    p_row = next(row for row in rows if row[0] == "p_value")
    mean_row = next(row for row in rows if row[0] == "primary_estimate")
    assert float(p_row[1]) == result.values["p_value"]
    assert float(mean_row[1]) == result.values["primary_estimate"]
    assert mean_row[1].startswith("-")
    assert str(result.values["p_value"]) in report.to_json()
    assert "p = 0.000" not in report.to_html()


def test_small_p_value_display_preserves_underlying_zero(welch_case):
    _, assistant, result = welch_case
    report = assistant.report(_values(result, p_value=0.0))
    assert report.to_dict()["analysis"]["values"]["p_value"] == 0.0
    assert "<dt><strong>p_value</strong></dt><dd>p &lt; 0.001" in report.to_html()
    assert "**p\\_value:** p &lt; 0.001" in report.to_markdown()
    assert "p &lt; 0.001" in report.to_html()
    assert "p &lt; 0.001" in report.to_markdown()


def test_html_markdown_and_csv_escape_untrusted_text(tmp_path):
    dangerous_group = '=HYPERLINK("https://example.com","click")'
    other_group = "+SUM(1,1)|<script>alert(1)</script>"
    frame = pd.DataFrame(
        {
            "=SUM(1,1)": [dangerous_group] * 8 + [other_group] * 8,
            "score": list(range(8)) + list(range(3, 11)),
        }
    )
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="=SUM(1,1)",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
        description="<img src=x onerror=alert(1)>",
    )
    report = assistant.report(assistant.analyze(draft), title="../<script>alert(1)</script>|report")
    html = report.to_html()
    markdown = report.to_markdown()
    csv_text = report.to_csv_tables()["group_sizes"]
    assert "<script>" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img" in html
    assert "&lt;script&gt;" in markdown
    assert "\\|" in markdown
    csv_rows = list(csv.reader(io.StringIO(csv_text)))
    assert csv_rows[1][0].startswith("'=HYPERLINK")
    assert csv_rows[2][0].startswith("'+SUM")
    paths = report.save_csv_tables(tmp_path / "tables")
    assert {path.name for path in paths} == {
        "sample_accounting.csv",
        "group_sizes.csv",
        "statistical_results.csv",
        "effect_estimates.csv",
        "confidence_intervals.csv",
    }
    assert not (tmp_path / "script").exists()


def test_save_methods_require_explicit_paths_and_protect_existing_files(welch_case, tmp_path):
    _, assistant, result = welch_case
    report = assistant.report(result)
    html_path = report.save_html(tmp_path / "report.html")
    md_path = report.save_markdown(tmp_path / "report.md")
    json_path = report.save_json(tmp_path / "report.json")
    assert html_path.read_text(encoding="utf-8") == report.to_html()
    assert md_path.read_text(encoding="utf-8") == report.to_markdown()
    assert json.loads(json_path.read_text(encoding="utf-8")) == report.to_dict()
    with pytest.raises(ReportError, match="already exists"):
        report.save_html(html_path)
    report.save_html(html_path, overwrite=True)
    directory = tmp_path / "tables"
    report.save_csv_tables(directory)
    with pytest.raises(ReportError, match="file exists"):
        report.save_csv_tables(directory)
    with pytest.raises(ReportError):
        report.save_json(directory)


def test_invalid_inputs_are_actionable(welch_case):
    _, assistant, result = welch_case
    with pytest.raises(ReportError, match="AnalysisResult"):
        assistant.report({"values": {}})
    with pytest.raises(ReportError, match="title"):
        assistant.report(result, title=" ")
    with pytest.raises(ReportError, match="Boolean"):
        assistant.report(result, include_figures="yes")
    with pytest.raises(ReportError, match="specification"):
        assistant.report(replace(result, specification=None))


def test_legacy_report_generator_constructor_and_exports_unchanged():
    frame = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
    legacy = ReportGenerator(StatisticalAnalyzer(frame).analyze_all())
    assert "overview" in legacy.to_dict()["analysis"]
    assert "Dataset Overview" in legacy.to_html()
    assert isinstance(legacy.to_csv(), dict)


def test_kruskal_and_chi_square_report_genuine_omnibus_results():
    rank_frame = pd.DataFrame(
        {"group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5, "score": list(range(15))}
    )
    rank_assistant = ResearchAssistant(rank_frame)
    rank_result = rank_assistant.analyze(
        rank_assistant.prepare_question(
            objective="compare_groups",
            outcome="score",
            predictor="group",
            estimand="distribution",
            design="independent",
            variable_types={"score": "continuous"},
        )
    )
    rank_report = rank_assistant.report(rank_result)
    assert rank_report.to_dict()["sections"]["methods"]["method_id"] == "kruskal_wallis"
    assert "specific group differences" in rank_report.to_html()

    chi_frame = pd.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            "response": ["yes"] * 15 + ["no"] * 5 + ["yes"] * 5 + ["no"] * 15,
        }
    )
    chi_assistant = ResearchAssistant(chi_frame)
    chi_result = chi_assistant.analyze(
        chi_assistant.prepare_question(
            objective="association",
            outcome="response",
            predictor="group",
            design="independent",
        )
    )
    chi_report = chi_assistant.report(chi_result)
    assert chi_report.to_dict()["sections"]["methods"]["method_id"] == "pearson_chi_square"
    assert chi_report.to_dict()["sections"]["results"]["effect_size"]["name"] == "Cramer's V"
    assert "causation" in chi_report.to_markdown()


@pytest.mark.parametrize(
    ("method_id", "group_count", "group_size"),
    [("student_t", 2, 8), ("one_way_anova", 3, 5)],
)
def test_legacy_execution_adapters_have_compatible_reports(method_id, group_count, group_size):
    labels = [chr(ord("A") + i) for i in range(group_count) for _ in range(group_size)]
    frame = pd.DataFrame({"group": labels, "score": [float(i) + 0.1 for i in range(len(labels))]})
    assistant = ResearchAssistant(frame)
    specification = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    ).specification
    recommendation = Recommendation(
        status="ready", method_id=method_id, method_name=method_id, method_availability="runnable"
    )
    result = execution._group_result(assistant._analyzer, specification, recommendation)
    report = assistant.report(result)
    payload = report.to_dict()
    assert payload["sections"]["methods"]["method_id"] == method_id
    assert payload["sections"]["results"]["p_value"] == result.values["p_value"]
    assert payload["sections"]["dataset"]["group_order"] == labels[::group_size]
    assert "statistical_results" in report.to_csv_tables()
    if group_count == 3:
        assert "specific group differences" in report.to_html()


def test_report_dict_is_defensive_copy_and_invalid_group_sizes_are_blocked(welch_case):
    _, assistant, result = welch_case
    report = assistant.report(result)
    copied = report.to_dict()
    copied["tables"][0]["rows"][0][1]["value"] = 999
    assert report.to_dict()["tables"][0]["rows"][0][1]["value"] == 17
    metadata = deepcopy(result.metadata)
    metadata["sample"]["group_sizes"][0]["size"] = "eight"
    with pytest.raises(ReportError, match="Per-group"):
        assistant.report(replace(result, metadata=metadata))


def test_report_accepts_manual_available_analysis_result_only_with_context(welch_case):
    _, assistant, result = welch_case
    assert isinstance(result, AnalysisResult)
    assert assistant.report(result).to_dict()["analysis"]["method_id"] == "welch_t"


def test_report_rejects_nonfinite_analysis_and_malformed_sample_metadata(welch_case):
    _, assistant, result = welch_case
    with pytest.raises(ReportError, match="invalid JSON data"):
        assistant.report(_values(result, p_value=math.nan))

    metadata = deepcopy(result.metadata)
    metadata["sample"] = ["invalid"]
    with pytest.raises(ReportError, match="sample metadata is invalid"):
        assistant.report(replace(result, metadata=metadata))

    metadata = deepcopy(result.metadata)
    metadata["sample"]["analyzed_rows"] = 15
    with pytest.raises(ReportError, match="Sample metadata contradicts"):
        assistant.report(replace(result, metadata=metadata))

    metadata = deepcopy(result.metadata)
    metadata["sample"]["analyzed_rows"] = None
    with pytest.raises(ReportError, match="analyzed row count"):
        assistant.report(replace(result, sample_size=None, metadata=metadata))


def test_missing_exclusion_count_is_disclosed_as_partial(welch_case):
    _, assistant, result = welch_case
    metadata = deepcopy(result.metadata)
    metadata["sample"]["excluded_rows"] = None
    report = assistant.report(replace(result, excluded_rows=None, metadata=metadata))
    assert report.status == "partial"
    assert report.to_dict()["sections"]["dataset"]["excluded_rows"] is None
    assert any("row accounting is unavailable" in item for item in report.to_dict()["limitations"])


def test_export_paths_report_filesystem_failures(welch_case, tmp_path):
    _, assistant, result = welch_case
    report = assistant.report(result)
    missing_parent = tmp_path / "missing" / "report.html"
    with pytest.raises(ReportError, match="Could not write report"):
        report.save_html(missing_parent)

    file_destination = tmp_path / "already_a_file"
    file_destination.write_text("existing", encoding="utf-8")
    with pytest.raises(ReportError, match="not a directory"):
        report.save_csv_tables(file_destination)
    with pytest.raises(ReportError, match="Could not save CSV report tables"):
        report.save_csv_tables(file_destination / "nested")
    assert file_destination.read_text(encoding="utf-8") == "existing"
