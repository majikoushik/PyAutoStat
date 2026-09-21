"""
Report Generator - Produces reports in multiple formats
"""

import json
import math
from collections.abc import Mapping
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .exceptions import InvalidDataError, PyAutoStatError, ReportError


def _html(value):
    """Escape untrusted report text and display unavailable values clearly."""
    return escape("N/A" if value is None else str(value), quote=True)


def _number(value, digits=4):
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.{digits}f}" if math.isfinite(number) else "N/A"


def _percent(value, digits=2):
    number = _number(value, digits)
    return f"{number}%" if number != "N/A" else number


def _json_safe(value):
    """Convert analysis values to standards-compliant JSON primitives."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _csv_safe(value):
    """Prevent spreadsheet formula execution in exported text cells."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _csv_safe_frame(frame):
    safe = frame.copy()
    safe.columns = [_csv_safe(column) for column in safe.columns]
    safe.index = safe.index.map(_csv_safe)
    for column in safe.columns:
        if pd.api.types.is_object_dtype(safe[column]) or pd.api.types.is_string_dtype(safe[column]):
            safe[column] = safe[column].map(_csv_safe)
    return safe


def _write_text(filepath, content):
    try:
        Path(filepath).write_text(content, encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        raise ReportError(f"Could not write report to '{filepath}': {exc}") from exc


_REPORT_CSS = """
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
            border-left: 4px solid #007bff;
            padding-left: 10px;
        }
        h3 {
            color: #777;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #007bff;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .insight {
            margin: 15px 0;
            padding: 15px;
            border-left: 4px solid #dc3545;
            background-color: #fff3cd;
        }
        .insight.high {
            border-left-color: #dc3545;
            background-color: #f8d7da;
        }
        .insight.medium {
            border-left-color: #ffc107;
            background-color: #fff3cd;
        }
        .insight.low {
            border-left-color: #28a745;
            background-color: #d4edda;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px;
            background-color: #e7f3ff;
            border-radius: 4px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
        }
        .recommendation {
            margin: 10px 0;
            padding: 8px;
            background-color: #f0f0f0;
            border-radius: 4px;
        }
        .timestamp {
            color: #999;
            font-size: 12px;
            margin-top: 20px;
            text-align: right;
        }
"""

_INTERACTIVE_CSS = """
        .report-section { border: 1px solid #ddd; border-radius: 6px; margin: 12px 0; }
        .report-section > summary { cursor: pointer; padding: 14px; font-weight: bold; }
        .section-body { padding: 0 14px 14px; }
        .table-filter { margin: 10px 0; padding: 8px; width: min(100%, 320px); }
        .sortable-header { cursor: pointer; }
        .plotly-chart { min-height: 300px; }
"""

_INTERACTIVE_JS = """
document.addEventListener('DOMContentLoaded', () => {
  for (const section of document.querySelectorAll('details.report-section')) {
    for (const table of section.querySelectorAll('table')) {
      const search = document.createElement('input');
      search.type = 'search';
      search.className = 'table-filter';
      search.setAttribute('aria-label', 'Filter table rows');
      search.placeholder = 'Filter table rows';
      table.parentNode.insertBefore(search, table);
      search.addEventListener('input', () => {
        const term = search.value.toLocaleLowerCase();
        for (const row of table.querySelectorAll('tr')) {
          if (row.querySelector('th')) continue;
          row.hidden = !row.textContent.toLocaleLowerCase().includes(term);
        }
      });
      const headers = table.querySelectorAll('tr:first-child th');
      headers.forEach((header, index) => {
        header.classList.add('sortable-header');
        header.tabIndex = 0;
        header.setAttribute('role', 'button');
        header.setAttribute('aria-label', `Sort by ${header.textContent}`);
        const sort = () => {
          const body = table.tBodies[0];
          if (!body) return;
          const ascending = header.dataset.ascending !== 'true';
          const rows = Array.from(body.rows);
          rows.sort((left, right) => {
            const a = left.cells[index]?.textContent.trim() ?? '';
            const b = right.cells[index]?.textContent.trim() ?? '';
            const x = Number(a.replace(/[% ,]/g, ''));
            const y = Number(b.replace(/[% ,]/g, ''));
            const comparison = a !== '' && b !== '' && Number.isFinite(x) &&
              Number.isFinite(y) ? x - y : a.localeCompare(b, undefined, {numeric: true});
            return ascending ? comparison : -comparison;
          });
          rows.forEach(row => body.appendChild(row));
          header.dataset.ascending = String(ascending);
        };
        header.addEventListener('click', sort);
        header.addEventListener('keydown', event => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            sort();
          }
        });
      });
    }
    section.addEventListener('toggle', () => {
      if (!section.open) return;
      for (const chart of section.querySelectorAll('.plotly-chart')) {
        if (chart.dataset.rendered === 'true') continue;
        if (typeof Plotly === 'undefined') {
          chart.textContent = 'Charts are unavailable: Plotly JavaScript did not load.';
          continue;
        }
        const data = document.getElementById(`data-${chart.id}`);
        const figure = JSON.parse(data.textContent);
        Plotly.newPlot(chart, figure.data, figure.layout, {displaylogo: false});
        chart.dataset.rendered = 'true';
      }
    });
    if (section.open) section.dispatchEvent(new Event('toggle'));
  }
});
"""


class ReportGenerator:
    """Generate reports in multiple formats (JSON, HTML, CSV, Dict)"""

    def __init__(self, analysis_results, insights=None, hypothesis_results=None):
        """
        Initialize report generator.

        Parameters:
        -----------
        analysis_results : dict
            Output from StatisticalAnalyzer.analyze_all()
        insights : dict
            Output from InsightEngine.get_summary()
        hypothesis_results : dict or list of dict, optional
            One or more hypothesis_tests() or categorical_association() results.
        """
        if not isinstance(analysis_results, Mapping):
            raise InvalidDataError("ReportGenerator expects analysis_results to be a mapping.")
        if insights is not None and not isinstance(insights, Mapping):
            raise InvalidDataError("ReportGenerator expects insights to be a mapping or None.")
        if hypothesis_results is None:
            hypotheses = []
        elif isinstance(hypothesis_results, Mapping):
            hypotheses = [hypothesis_results]
        elif isinstance(hypothesis_results, (list, tuple)) and all(
            isinstance(item, Mapping) for item in hypothesis_results
        ):
            hypotheses = list(hypothesis_results)
        else:
            raise InvalidDataError(
                "ReportGenerator expects hypothesis_results to be a mapping or a list of mappings."
            )
        self.analysis_results = analysis_results
        self.insights = insights or {}
        self.hypothesis_results = hypotheses
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        """Export as Python dictionary"""
        report = {
            "timestamp": self.timestamp,
            "analysis": self.analysis_results,
            "insights": self.insights,
        }
        if self.hypothesis_results:
            report["hypothesis_tests"] = self.hypothesis_results
        return report

    def to_json(self, filepath=None, pretty=True):
        """
        Export as JSON.

        Parameters:
        -----------
        filepath : str, optional
            Path to save JSON file. If None, returns string
        pretty : bool
            Pretty print JSON
        """
        report = self.to_dict()

        json_str = json.dumps(_json_safe(report), indent=2 if pretty else None, allow_nan=False)

        if filepath:
            _write_text(filepath, json_str)
            return f"Report saved to {filepath}"

        return json_str

    def to_csv(self, output_dir=None):
        """
        Export as multiple CSV files.

        Parameters:
        -----------
        output_dir : str, optional
            Directory to save CSV files

        Returns:
        --------
        dict of pd.DataFrame or filepaths
        """
        csvs = {}

        # Descriptive statistics
        if "descriptive" in self.analysis_results:
            desc_df = pd.DataFrame(self.analysis_results["descriptive"]).T
            csvs["descriptive_stats"] = desc_df

        # Outliers summary
        if "outliers" in self.analysis_results:
            outlier_data = []
            for col, methods in self.analysis_results["outliers"].items():
                for method, details in methods.items():
                    row = {"column": col, "method": method}
                    row.update(details)
                    outlier_data.append(row)
            outlier_df = pd.DataFrame(outlier_data)
            csvs["outliers"] = outlier_df

        # Missing data summary
        if "missing_data" in self.analysis_results:
            missing_cols = self.analysis_results["missing_data"].get("by_column", {})
            missing_df = pd.DataFrame(missing_cols).T
            csvs["missing_data"] = missing_df

        # Insights
        if self.insights and "insights" in self.insights:
            insights_data = []
            for insight in self.insights["insights"]:
                insights_data.append(
                    {
                        "category": insight.get("category"),
                        "severity": insight.get("severity"),
                        "finding": insight.get("finding"),
                        "recommendations": " | ".join(insight.get("recommendation", [])),
                    }
                )
            insights_df = pd.DataFrame(insights_data)
            csvs["insights"] = insights_df

        if self.hypothesis_results:
            csvs["hypothesis_tests"] = pd.DataFrame(
                [
                    {
                        "test": item.get("test"),
                        "groups": ", ".join(map(str, item.get("groups", []))),
                        "statistic": item.get("statistic"),
                        "p_value": item.get("p_value"),
                        "degrees_of_freedom": item.get("degrees_of_freedom"),
                        "effect_size": item.get("effect_size", {}).get("value"),
                        "cohens_h": item.get("cohens_h", {}).get("value"),
                        "selection_reason": item.get("assumptions", {}).get("selection_reason"),
                    }
                    for item in self.hypothesis_results
                ]
            )

        if output_dir:
            try:
                directory = Path(output_dir)
                directory.mkdir(parents=True, exist_ok=True)
                for name, frame in csvs.items():
                    _csv_safe_frame(frame).to_csv(
                        directory / f"{name}.csv",
                        index=name in ("descriptive_stats", "missing_data"),
                    )
            except (OSError, ValueError, TypeError) as exc:
                raise ReportError(f"Could not write CSV reports to '{output_dir}': {exc}") from exc

        return csvs

    def to_html(self, filepath=None, title="Statistical Analysis Report"):
        """
        Export as HTML report.

        Parameters:
        -----------
        filepath : str, optional
            Path to save HTML file
        title : str
            Report title
        """
        safe_title = _html(title)
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{safe_title}</title>
    <style>{_REPORT_CSS}</style>
</head>
<body>
    <div class="container">
        <h1>{safe_title}</h1>

        {self._generate_overview_html()}
        {self._generate_insights_html()}
        {self._generate_hypothesis_html()}
        {self._generate_descriptive_html()}
        {self._generate_normality_html()}
        {self._generate_outliers_html()}
        {self._generate_correlation_html()}
        {self._generate_missing_data_html()}
        {self._generate_warnings_html()}

        <div class="timestamp">Generated: {self.timestamp}</div>
    </div>
</body>
</html>
        """

        if filepath:
            _write_text(filepath, html_content)
            return f"HTML report saved to {filepath}"

        return html_content

    def to_interactive_html(self, filepath=None, title="Statistical Analysis Report (Interactive)"):
        """
        Export as an interactive HTML report with Plotly charts
        (a hoverable correlation heatmap and zoomable distribution histograms).

        Requires the optional 'report' extra: pip install pyautostat[report]

        Parameters:
        -----------
        filepath : str, optional
            Path to save HTML file
        title : str
            Report title
        """
        try:
            import plotly.graph_objects as go
            from plotly.offline import get_plotlyjs_version
        except ImportError as exc:
            raise PyAutoStatError(
                "to_interactive_html() requires Plotly. "
                "Install it with: pip install pyautostat[report]"
            ) from exc

        figure_count = 0

        def render_figure(fig):
            nonlocal figure_count
            chart_id = f"chart-{figure_count}"
            figure_count += 1
            # JSON lives in an inert script element. Escape HTML delimiters so
            # labels cannot terminate that element and inject report markup.
            figure_json = (
                fig.to_json()
                .replace("<", "\\u003c")
                .replace(">", "\\u003e")
                .replace("&", "\\u0026")
            )
            return (
                f'<div class="plotly-chart" id="{chart_id}"></div>'
                f'<script type="application/json" id="data-{chart_id}">{figure_json}</script>'
            )

        sections = [
            ("Overview", self._generate_overview_html()),
            ("Insights", self._generate_insights_html()),
            ("Hypothesis Tests", self._generate_hypothesis_html()),
            ("Descriptive Statistics", self._generate_descriptive_html()),
            ("Normality", self._generate_normality_html()),
            ("Correlations", self._interactive_correlation_html(go, render_figure)),
            ("Distributions", self._interactive_distributions_html(go, render_figure)),
            ("Outliers", self._generate_outliers_html()),
            ("Missing Data", self._generate_missing_data_html()),
            ("Analysis Warnings", self._generate_warnings_html()),
        ]
        body = "\n".join(
            f'<details class="report-section" {"open" if index == 0 else ""}>'
            f"<summary>{_html(label)}</summary>"
            f'<div class="section-body">{content}</div></details>'
            for index, (label, content) in enumerate(sections)
            if content
        )
        safe_title = _html(title)

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{safe_title}</title>
    <style>{_REPORT_CSS}{_INTERACTIVE_CSS}</style>
    <script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" defer></script>
</head>
<body>
    <div class="container">
        <h1>{safe_title}</h1>

        {body}

        <div class="timestamp">Generated: {self.timestamp}</div>
    </div>
    <script>{_INTERACTIVE_JS}</script>
</body>
</html>
        """

        if filepath:
            _write_text(filepath, html_content)
            return f"Interactive HTML report saved to {filepath}"

        return html_content

    def _interactive_correlation_html(self, go, render_figure):
        """Hoverable Pearson correlation heatmap."""
        pearson_matrix = (
            self.analysis_results.get("correlation", {}).get("pearson", {}).get("matrix", {})
        )
        if not pearson_matrix:
            return ""

        cols = list(pearson_matrix.keys())
        z = [[pearson_matrix[c1].get(c2) for c2 in cols] for c1 in cols]
        text = [[f"{v:.2f}" if v is not None else "" for v in row] for row in z]
        labels = [_html(col) for col in cols]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                text=text,
                texttemplate="%{text}",
                hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Pearson Correlation (hover for exact values)",
            height=450,
            margin=dict(l=40, r=40, t=60, b=40),
        )

        return "<h2>Correlation Heatmap</h2>" + render_figure(fig)

    def _interactive_distributions_html(self, go, render_figure):
        """Zoomable histogram for each numeric column."""
        histograms = self.analysis_results.get("histograms")
        if not histograms:
            return ""

        html = "<h2>Distributions (scroll to zoom)</h2>"
        for col, hist in histograms.items():
            edges = hist["bin_edges"]
            counts = hist["counts"]
            centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
            widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]

            fig = go.Figure(go.Bar(x=centers, y=counts, width=widths))
            fig.update_layout(
                title=f"Distribution of {_html(col)}",
                height=320,
                margin=dict(l=40, r=40, t=50, b=40),
                bargap=0.02,
            )
            html += render_figure(fig)

        return html

    @staticmethod
    def _metric_html(value, label):
        """Render a single metric tile (used across the overview/insights/missing-data sections)."""
        return (
            f'<div class="metric"><div class="metric-value">{_html(value)}</div>'
            f'<div class="metric-label">{_html(label)}</div></div>'
        )

    def _generate_overview_html(self):
        """Generate overview section HTML"""
        if "overview" not in self.analysis_results:
            return ""

        overview = self.analysis_results["overview"]

        html = "<h2>Dataset Overview</h2>"
        html += "<div>"
        html += self._metric_html(overview.get("total_rows"), "Rows")
        html += self._metric_html(overview.get("total_columns"), "Columns")
        html += self._metric_html(overview.get("numeric_columns"), "Numeric")
        html += self._metric_html(overview.get("categorical_columns"), "Categorical")
        html += "</div>"

        return html

    def _generate_insights_html(self):
        """Generate insights section HTML"""
        if not self.insights or "insights" not in self.insights:
            return ""

        insights = self.insights["insights"]

        html = "<h2>Key Insights & Recommendations</h2>"

        summary = self.insights
        html += self._metric_html(summary.get("total_insights", 0), "Total Issues")
        html += self._metric_html(summary.get("high_severity", 0), "High Severity")
        html += self._metric_html(summary.get("medium_severity", 0), "Medium Severity")

        for insight in insights:
            severity = insight.get("severity", "medium")
            severity_class = severity if severity in ("high", "medium", "low") else "medium"
            html += f'<div class="insight {severity_class}">'
            html += (
                f"<strong>{_html(insight.get('category'))}</strong> "
                f"({_html(str(severity).upper())})<br>"
            )
            html += f"{_html(insight.get('finding'))}<br>"

            if insight.get("recommendation"):
                html += '<div style="margin-top: 10px;"><strong>Recommendations:</strong>'
                for rec in insight["recommendation"]:
                    html += f'<div class="recommendation">• {_html(rec)}</div>'
                html += "</div>"

            html += "</div>"

        return html

    def _generate_hypothesis_html(self):
        """Render independent-group test results and their assumption screens."""
        if not self.hypothesis_results:
            return ""

        html = "<h2>Hypothesis Tests</h2>"
        for result in self.hypothesis_results:
            assumptions = result.get("assumptions", {})
            effect = result.get("effect_size", {})
            groups = ", ".join(map(str, result.get("groups", [])))
            html += f"<h3>{_html(result.get('test'))}: {_html(groups)}</h3>"
            html += (
                f"<p>Statistic: {_number(result.get('statistic'))}; "
                f"p-value: {_number(result.get('p_value'))}; "
                f"effect size ({_html(effect.get('name'))}): "
                f"{_number(effect.get('value'))}</p>"
            )
            if assumptions.get("selection_reason"):
                html += f"<p>Selection: {_html(assumptions['selection_reason'])}</p>"
            if "normality" in assumptions:
                html += "<table><thead><tr><th>Assumption</th><th>Group</th><th>N</th>"
                html += "<th>p-value</th><th>Status</th></tr></thead><tbody>"
                for entry in assumptions["normality"]:
                    html += (
                        f"<tr><td>{_html(entry.get('test') or 'Normality')}</td>"
                        f"<td>{_html(entry.get('group'))}</td>"
                        f"<td>{_html(entry.get('sample_size'))}</td>"
                        f"<td>{_number(entry.get('p_value'))}</td>"
                        f"<td>{_html(entry.get('status'))}</td></tr>"
                    )
                html += (
                    "<tr><td>Levene equal variance</td><td>All groups</td><td></td>"
                    f"<td>{_number(assumptions.get('levene_p_value'))}</td>"
                    f"<td>{_html(assumptions.get('equal_variance_status'))}</td></tr>"
                )
                html += "</tbody></table>"
            elif "minimum_expected_count" in assumptions:
                html += (
                    "<p>Minimum expected cell count: "
                    f"{_number(assumptions['minimum_expected_count'])} "
                    f"({_html(assumptions.get('expected_count_status'))}). "
                    f"{_html(assumptions.get('independent_observations'))}</p>"
                )
            for label, interval in (
                ("Mean difference interval", result.get("confidence_interval")),
                ("Effect size interval", effect.get("confidence_interval")),
            ):
                if interval:
                    html += (
                        f"<p>{_html(label)} ({_percent(interval.get('level') * 100, 0)}): "
                        f"[{_number(interval.get('lower'))}, "
                        f"{_number(interval.get('upper'))}]"
                        f"; {_html(interval.get('method')) if interval.get('method') else ''}</p>"
                    )
            if result.get("cohens_h"):
                cohen = result["cohens_h"]
                html += (
                    f"<p>Cohen's h for {_html(cohen.get('success_value'))}: "
                    f"{_number(cohen.get('value'))}</p>"
                )
                interval = cohen.get("confidence_interval")
                if interval:
                    html += (
                        f"<p>Cohen's h interval ({_percent(interval['level'] * 100, 0)}): "
                        f"[{_number(interval.get('lower'))}, "
                        f"{_number(interval.get('upper'))}]</p>"
                    )
            if assumptions.get("warnings"):
                html += "<ul>"
                for message in assumptions["warnings"]:
                    html += f"<li>{_html(message)}</li>"
                html += "</ul>"
        return html

    def _generate_descriptive_html(self):
        """Generate descriptive statistics HTML"""
        if "descriptive" not in self.analysis_results:
            return ""

        descriptive = self.analysis_results["descriptive"]

        html = "<h2>Descriptive Statistics</h2>"
        html += "<table><thead>"
        html += (
            "<tr><th>Variable</th><th>Mean</th><th>Median</th><th>Std Dev</th>"
            "<th>Min</th><th>Max</th><th>Skewness</th></tr></thead><tbody>"
        )

        for col, stats in descriptive.items():
            html += f"""<tr>
                <td>{_html(col)}</td>
                <td>{_number(stats.get("mean"))}</td>
                <td>{_number(stats.get("median"))}</td>
                <td>{_number(stats.get("std"))}</td>
                <td>{_number(stats.get("min"))}</td>
                <td>{_number(stats.get("max"))}</td>
                <td>{_number(stats.get("skewness"))}</td>
            </tr>"""

        html += "</tbody></table>"

        return html

    def _generate_normality_html(self):
        """Generate normality tests HTML"""
        if "normality" not in self.analysis_results:
            return ""

        normality = self.analysis_results["normality"]

        html = "<h2>Normality Tests</h2>"
        html += "<table><thead>"
        html += (
            "<tr><th>Variable</th><th>Test</th><th>Statistic</th>"
            "<th>P-Value</th><th>Normal?</th></tr></thead><tbody>"
        )

        for col, tests in normality.items():
            for test_name, test_result in tests.items():
                if isinstance(test_result, dict):
                    is_normal = test_result.get("is_normal")
                    status = "N/A" if is_normal is None else ("✓" if is_normal else "✗")
                    html += f"""<tr>
                        <td>{_html(col)}</td>
                        <td>{_html(test_name)}</td>
                        <td>{_number(test_result.get("statistic"))}</td>
                        <td>{_number(test_result.get("p_value"))}</td>
                        <td>{status}</td>
                    </tr>"""

        html += "</tbody></table>"

        return html

    def _generate_outliers_html(self):
        """Generate outliers section HTML"""
        if "outliers" not in self.analysis_results:
            return ""

        outliers = self.analysis_results["outliers"]

        html = "<h2>Outlier Detection</h2>"
        html += "<table><thead>"
        html += (
            "<tr><th>Variable</th><th>Method</th><th>Count</th>"
            "<th>Percentage</th></tr></thead><tbody>"
        )

        for col, methods in outliers.items():
            for method, details in methods.items():
                html += f"""<tr>
                    <td>{_html(col)}</td>
                    <td>{_html(method)}</td>
                    <td>{_html(details.get("count", 0))}</td>
                    <td>{_percent(details.get("percentage"))}</td>
                </tr>"""

        html += "</tbody></table>"

        return html

    def _generate_correlation_html(self):
        """Generate correlation section HTML"""
        if "correlation" not in self.analysis_results:
            return ""

        html = "<h2>Correlations</h2>"
        html += "<p>Multiple correlation methods calculated: Pearson, Spearman, Kendall</p>"
        html += "<p><em>See detailed JSON export for correlation matrices</em></p>"

        return html

    def _generate_missing_data_html(self):
        """Generate missing data section HTML"""
        if "missing_data" not in self.analysis_results:
            return ""

        missing = self.analysis_results["missing_data"]

        html = "<h2>Missing Data Analysis</h2>"
        overall_missing_pct = missing.get("overall_missing_percentage", 0)
        html += self._metric_html(f"{overall_missing_pct:.2f}%", "Overall Missing")

        html += "<table><thead>"
        html += "<tr><th>Variable</th><th>Missing Count</th><th>Missing %</th></tr></thead><tbody>"

        for col, info in missing.get("by_column", {}).items():
            html += f"""<tr>
                <td>{_html(col)}</td>
                <td>{_html(info.get("count", 0))}</td>
                <td>{_percent(info.get("percentage"))}</td>
            </tr>"""

        html += "</tbody></table>"

        return html

    def _generate_warnings_html(self):
        """Display skipped or undefined analyses without hiding the reason."""
        warnings_data = self.analysis_results.get("analysis_warnings", [])
        if not warnings_data:
            return ""
        rows = ["<h2>Analysis Warnings</h2><ul>"]
        for item in warnings_data:
            column = item.get("column")
            prefix = f"{_html(column)}: " if column is not None else ""
            rows.append(f"<li>{prefix}{_html(item.get('message'))}</li>")
        rows.append("</ul>")
        return "".join(rows)
