"""Plotly HTML renderer for PyAutoStat scientific figures.

Transforms neutral FigureSpec models into restrained, self-contained interactive
Plotly figures with offline single-bundle embedding, responsive scaling, and
strict XSS sanitization.
"""

from __future__ import annotations

import json
from typing import Any

from ...exceptions import ReportError
from ..figures.models import FigureSpec
from ..figures.plotly import spec_to_plotly_figure
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
