import json
import re
import sys

import pandas as pd
import pytest

from pyautostat import PyAutoStatError, ReportGenerator, StatisticalAnalyzer
from pyautostat.insights import InsightEngine


@pytest.fixture
def sample_report(mixed_df):
    analyzer = StatisticalAnalyzer(mixed_df)
    results = analyzer.analyze_all()
    insights = InsightEngine(results).get_summary()
    return ReportGenerator(results, insights)


def test_to_interactive_html_contains_plotly_chart_and_sections(sample_report):
    html = sample_report.to_interactive_html(title="Interactive Test Report")

    assert "Interactive Test Report" in html
    assert "Correlation Heatmap" in html
    assert "Distributions" in html
    assert "plotly" in html.lower()


def test_to_interactive_html_only_loads_plotly_js_once(sample_report):
    html = sample_report.to_interactive_html()
    # cdn.plot.ly script tag should appear exactly once even though multiple
    # figures (1 heatmap + 2 histograms) are embedded.
    assert html.count("cdn.plot.ly") == 1


def test_to_interactive_html_writes_file(sample_report, tmp_path):
    path = tmp_path / "interactive.html"
    message = sample_report.to_interactive_html(str(path))
    assert path.exists()
    assert "saved" in message.lower()


def test_to_interactive_html_raises_friendly_error_without_plotly(sample_report, monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    monkeypatch.setitem(sys.modules, "plotly", None)

    with pytest.raises(PyAutoStatError, match=r"pip install pyautostat\[report\]"):
        sample_report.to_interactive_html()


def test_interactive_report_defers_charts_and_includes_table_controls(sample_report):
    html = sample_report.to_interactive_html()
    assert '<details class="report-section"' in html
    assert 'type="application/json"' in html
    assert "Plotly.newPlot" in html
    assert "Filter table rows" in html
    assert "Sort by" in html
    assert "Descriptive Statistics" in html
    assert "Normality Tests" in html

    figures = re.findall(
        r'<script type="application/json" id="data-chart-\d+">(.*?)</script>', html
    )
    assert len(figures) >= 2
    assert all("data" in json.loads(figure) for figure in figures)


def test_interactive_chart_data_cannot_close_script_element():
    label = "</script><script>alert(1)</script>"
    frame = pd.DataFrame({label: list(range(8)), "other": list(range(8, 16))})
    report = ReportGenerator(StatisticalAnalyzer(frame).analyze_all())
    html = report.to_interactive_html(title=label)

    assert label not in html
    figures = re.findall(
        r'<script type="application/json" id="data-chart-\d+">(.*?)</script>', html
    )
    assert figures
    assert all("<" not in figure for figure in figures)
    assert any("&lt;" in json.dumps(json.loads(figure)) for figure in figures)
