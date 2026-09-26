import json

import pandas as pd
import pytest

from pyautostat import InvalidDataError, ReportGenerator, StatisticalAnalyzer
from pyautostat.insights import InsightEngine


@pytest.fixture
def sample_report(mixed_df):
    analyzer = StatisticalAnalyzer(mixed_df)
    results = analyzer.analyze_all()
    insights = InsightEngine(results).get_summary()
    return ReportGenerator(results, insights)


def test_to_dict_contains_analysis_and_insights(sample_report):
    report = sample_report.to_dict()
    assert set(report) == {"timestamp", "analysis", "insights"}
    assert "descriptive" in report["analysis"]


def test_to_json_round_trips_as_valid_json(sample_report):
    json_str = sample_report.to_json(pretty=False)
    parsed = json.loads(json_str)
    assert "analysis" in parsed


def test_to_json_writes_file(sample_report, tmp_path):
    path = tmp_path / "report.json"
    message = sample_report.to_json(str(path))
    assert path.exists()
    assert "saved" in message.lower()
    json.loads(path.read_text())


def test_to_csv_returns_dataframes_for_each_section(sample_report):
    csvs = sample_report.to_csv()
    assert set(csvs) >= {"descriptive_stats", "outliers", "missing_data", "insights"}
    assert isinstance(csvs["descriptive_stats"], pd.DataFrame)


def test_to_csv_writes_files(sample_report, tmp_path):
    sample_report.to_csv(str(tmp_path))
    assert (tmp_path / "descriptive_stats.csv").exists()
    assert (tmp_path / "outliers.csv").exists()
    assert (tmp_path / "missing_data.csv").exists()
    assert (tmp_path / "insights.csv").exists()


def test_to_html_contains_title_and_sections(sample_report):
    html = sample_report.to_html(title="Test Report")
    assert "Test Report" in html
    assert "Dataset Overview" in html
    assert "Descriptive Statistics" in html
    assert 'class="severity-badge medium"' in html
    assert 'class="severity-badge medium>' not in html
    assert "column(s) are complete" in html


def test_to_html_writes_file(sample_report, tmp_path):
    path = tmp_path / "report.html"
    message = sample_report.to_html(str(path))
    assert path.exists()
    assert "saved" in message.lower()


def test_hypothesis_results_appear_in_every_report_format(two_group_normal_df, tmp_path):
    analyzer = StatisticalAnalyzer(two_group_normal_df)
    hypothesis = analyzer.hypothesis_tests("group", "value", bootstrap_samples=100, estimand="mean")
    report = ReportGenerator(analyzer.analyze_all(), hypothesis_results=hypothesis)

    assert report.to_dict()["hypothesis_tests"] == [hypothesis]
    assert json.loads(report.to_json())["hypothesis_tests"][0]["test"] == hypothesis["test"]
    for html in (report.to_html(), report.to_interactive_html()):
        assert "Hypothesis Tests" in html
        assert "Usable N: 80; excluded rows: 0" in html
        assert "Levene equal variance" in html
        assert "Effect size interval" in html
    report.to_csv(tmp_path)
    assert (tmp_path / "hypothesis_tests.csv").exists()
    csv = (tmp_path / "hypothesis_tests.csv").read_text(encoding="utf-8")
    assert "sample_size" in csv and "excluded_rows" in csv


def test_report_rejects_malformed_hypothesis_container():
    with pytest.raises(InvalidDataError, match="hypothesis_results"):
        ReportGenerator({}, hypothesis_results=[{"test": "t"}, "bad"])
