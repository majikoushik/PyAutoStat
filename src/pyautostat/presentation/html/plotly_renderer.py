"""Plotly HTML renderer for PyAutoStat scientific figures.

Transforms neutral FigureSpec models into restrained, self-contained interactive
Plotly figures with offline single-bundle embedding, responsive scaling, and
strict XSS sanitization.
"""

from __future__ import annotations

import json
from typing import Any

from ...exceptions import ReportError
from ..figures.formatting import format_hover_ci, format_hover_number
from ..figures.models import FigureSpec
from .formatting import escape_text


def check_plotly_available() -> None:
    """Verify that Plotly is installed, or raise an informative ReportError."""
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ReportError(
            "Interactive HTML figures require the 'plotly' package. "
            'Install it with: pip install "pyautostat[report]"'
        ) from exc


def get_plotly_bundle() -> str:
    """Return the offline Plotly JavaScript bundle wrapped in a script tag.

    This bundle must be embedded once per HTML document.
    """
    check_plotly_available()
    import plotly.offline as po

    bundle = po.get_plotlyjs()
    return f'<script type="text/javascript" id="pyautostat-plotly-bundle">\n{bundle}\n</script>'


def safe_json_for_script(data: Any) -> str:
    """Serialize a dictionary or list into a script-safe JSON string.

    Escapes '<', '>', and '&' to prevent '</script>' breakout and HTML injection.
    """
    raw = json.dumps(data, ensure_ascii=False)
    return raw.replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026")


def render_noscript_banner() -> str:
    """Render an accessible noscript notice when JavaScript is disabled."""
    return (
        "<noscript>\n"
        '  <div class="pyautostat-noscript-banner" role="status">\n'
        "    Interactive figures require JavaScript; all statistical results "
        "remain available in the report tables and text.\n"
        "  </div>\n"
        "</noscript>"
    )


def spec_to_plotly_figure(spec: FigureSpec) -> Any:
    """Convert an immutable FigureSpec into a plotly.graph_objects.Figure."""
    check_plotly_available()
    import plotly.graph_objects as go

    if spec.kind == "estimate_ci":
        y_labels = [s.label for s in spec.series]
        x_vals = [s.estimate for s in spec.series]
        err_plus = [
            (s.upper - s.estimate) if s.upper is not None and s.estimate is not None else 0
            for s in spec.series
        ]
        err_minus = [
            (s.estimate - s.lower) if s.lower is not None and s.estimate is not None else 0
            for s in spec.series
        ]
        has_ci = any(s.lower is not None and s.upper is not None for s in spec.series)

        hover_texts = []
        for s in spec.series:
            h_text = f"<b>{s.label}</b><br>Estimate: {format_hover_number(s.estimate)}"
            if s.lower is not None and s.upper is not None:
                h_text += f"<br>CI: {format_hover_ci(s.lower, s.upper)}"
            hover_texts.append(h_text)

        trace = go.Scatter(
            x=x_vals,
            y=y_labels,
            mode="markers",
            marker=dict(size=9, color="#1e3a8a"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=err_plus,
                arrayminus=err_minus,
                color="#1e3a8a",
                thickness=1.5,
                width=6,
            )
            if has_ci
            else None,
            hoverinfo="text",
            hovertext=hover_texts,
            showlegend=False,
        )
        fig = go.Figure(data=[trace])

    elif spec.kind in ("forest", "odds_ratio_forest", "pairwise_forest"):
        y_labels = [s.label for s in spec.series]
        x_vals = [s.estimate for s in spec.series]
        err_plus = [
            (s.upper - s.estimate) if s.upper is not None and s.estimate is not None else 0
            for s in spec.series
        ]
        err_minus = [
            (s.estimate - s.lower) if s.lower is not None and s.estimate is not None else 0
            for s in spec.series
        ]
        has_ci = any(s.lower is not None and s.upper is not None for s in spec.series)

        hover_texts = []
        for s in spec.series:
            h_text = f"<b>{s.label}</b><br>Estimate: {format_hover_number(s.estimate)}"
            if s.lower is not None and s.upper is not None:
                h_text += f"<br>CI: {format_hover_ci(s.lower, s.upper)}"
            for k, v in s.metadata.items():
                if v is not None:
                    h_text += f"<br>{k}: {format_hover_number(v)}"
            hover_texts.append(h_text)

        trace = go.Scatter(
            x=x_vals,
            y=y_labels,
            mode="markers",
            marker=dict(size=8, color="#2563eb", symbol="square"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=err_plus,
                arrayminus=err_minus,
                color="#2563eb",
                thickness=1.5,
                width=5,
            )
            if has_ci
            else None,
            hoverinfo="text",
            hovertext=hover_texts,
            showlegend=False,
        )
        fig = go.Figure(data=[trace])
        fig.update_layout(yaxis=dict(autorange="reversed"))

    elif spec.kind == "count_heatmap":
        x_labels = list(spec.series[0].categories) if spec.series else []
        y_labels = [s.label for s in spec.series]
        z_vals = [list(s.values) for s in spec.series]

        text_matrix = [
            [
                str(int(v)) if isinstance(v, (int, float)) and v == int(v) else str(v)
                for v in s.values
            ]
            for s in spec.series
        ]

        trace = go.Heatmap(
            x=x_labels,
            y=y_labels,
            z=z_vals,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(color="#0f172a", size=12),
            colorscale="Blues",
            showscale=True,
            hoverongaps=False,
        )
        fig = go.Figure(data=[trace])
        fig.update_layout(yaxis=dict(autorange="reversed"))

    elif spec.kind == "cell_profile":
        traces = []
        for s in spec.series:
            x_cats = list(s.categories)
            y_vals = list(s.values)
            h_texts = [
                f"<b>{s.label}</b><br>{x}: {format_hover_number(y)}"
                for x, y in zip(x_cats, y_vals, strict=False)
            ]
            trace = go.Scatter(
                x=x_cats,
                y=y_vals,
                mode="lines+markers",
                name=s.label,
                marker=dict(size=8),
                line=dict(width=2),
                hoverinfo="text",
                hovertext=h_texts,
            )
            traces.append(trace)
        fig = go.Figure(data=traces)
    else:
        return None

    # Reference line (neutral: 0 or 1)
    if spec.reference_value is not None:
        fig.add_vline(
            x=spec.reference_value,
            line_width=1.5,
            line_dash="dash",
            line_color="#64748b",
        )

    is_log_x = bool(spec.layout_hints.get("log_x", False))
    fig.update_layout(
        title=dict(
            text=spec.title,
            font=dict(size=14, color="#0f172a", family="inherit"),
            x=0.0,
            xanchor="left",
        ),
        xaxis=dict(
            title=spec.x_label,
            type="log" if is_log_x else "linear",
            showgrid=True,
            gridcolor="#e2e8f0",
            zeroline=False,
        ),
        yaxis=dict(
            title=spec.y_label,
            showgrid=True,
            gridcolor="#e2e8f0",
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#334155", size=12, family="inherit"),
        margin=dict(l=60, r=40, t=50, b=50),
        hovermode="closest",
        showlegend=len(fig.data) > 1
        and spec.kind not in ("forest", "odds_ratio_forest", "pairwise_forest"),
    )
    return fig


def render_figure_html(spec: FigureSpec, figure_idx: int = 1) -> str:
    """Render a FigureSpec into an accessible HTML container with Plotly initialization script."""
    fig = spec_to_plotly_figure(spec)
    if fig is None:
        return ""

    fig_id = f"pyautostat-chart-{figure_idx}"
    fig_json = safe_json_for_script(fig.to_dict())

    escaped_title = escape_text(spec.title)
    escaped_note = (
        f'  <figcaption class="pyautostat-figcaption">{escape_text(spec.note)}</figcaption>\n'
        if spec.note
        else ""
    )

    return (
        f'<div class="pyautostat-figure-container" id="container-{fig_id}">\n'
        f'  <figure class="pyautostat-figure" role="group" aria-label="{escaped_title}">\n'
        f'    <div id="{fig_id}" class="pyautostat-plotly-graph"></div>\n'
        '    <script type="text/javascript">\n'
        "      (function() {\n"
        f"        var figData = {fig_json};\n"
        f"        Plotly.newPlot('{fig_id}', figData.data, figData.layout, {{\n"
        "          responsive: true,\n"
        "          displaylogo: false,\n"
        "          scrollZoom: false,\n"
        "          modeBarButtonsToRemove: [\n"
        "            'lasso2d', 'select2d', 'sendDataToCloud', 'toggleSpikelines'\n"
        "          ]\n"
        "        }).then(function() {\n"
        f"          var el = document.getElementById('{fig_id}');\n"
        "          if (el) {\n"
        '            el.setAttribute("data-pyautostat-rendered", "true");\n'
        "          }\n"
        "          window.__pyautostatFiguresRendered = "
        "(window.__pyautostatFiguresRendered || 0) + 1;\n"
        "        }).catch(function(err) {\n"
        f"          var el = document.getElementById('{fig_id}');\n"
        "          if (el) {\n"
        '            el.setAttribute("data-pyautostat-error", String(err));\n'
        "          }\n"
        "        });\n"
        "      })();\n"
        "    </script>\n"
        f"{escaped_note}"
        "  </figure>\n"
        "</div>"
    )


__all__ = [
    "check_plotly_available",
    "get_plotly_bundle",
    "render_figure_html",
    "render_noscript_banner",
    "safe_json_for_script",
    "spec_to_plotly_figure",
]
