"""Tests for static figure SVG security, raw-data privacy sentinels, and zero recalculation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from unittest.mock import patch

import pandas as pd
from method_factories import make_welch_t_workflow

from pyautostat import (
    AnalysisOptions,
    ResearchAssistant,
    to_static_figures,
)
from pyautostat.presentation.figures.adapters import build_figure_specs
from pyautostat.presentation.figures.models import FigureSeries, FigureSpec


def test_svg_malicious_label_injection_defense():
    """Verify that malicious strings in FigureSpec labels do not inject executable script tags."""
    malicious_labels = [
        "</script><script>alert(1)</script>",
        '<img src="x" onerror="alert(1)">',
        "Group <A> & 'B' \"C\"",
        "javascript:alert(1)",
    ]

    for label in malicious_labels:
        spec = FigureSpec(
            kind="estimate_ci",
            title=f"Test {label}",
            subtitle=label,
            x_label=label,
            y_label=label,
            reference_value=0.0,
            series=(
                FigureSeries(
                    label=label,
                    estimate=1.5,
                    lower=0.5,
                    upper=2.5,
                ),
            ),
            note=label,
            placement="key_results",
        )

        from pyautostat.presentation.figures.static import generate_static_artifact

        # PNG rendering succeeds
        art_png = generate_static_artifact(spec, index=1, fmt="png")
        assert art_png.data.startswith(b"\x89PNG\r\n\x1a\n")

        # SVG rendering succeeds and contains no executable script tags
        art_svg = generate_static_artifact(spec, index=1, fmt="svg")
        svg_text = art_svg.data.decode("utf-8")

        # Parse SVG XML safely
        root = ET.fromstring(svg_text)
        # Check no script elements in any namespace
        assert not any(elem.tag.lower().endswith("script") for elem in root.iter())

        # Check no event handlers (onload, onerror, onclick)
        for elem in root.iter():
            for attr in elem.attrib:
                assert not attr.lower().startswith("on"), f"Found event handler attribute: {attr}"


def test_privacy_sentinels_not_leaked_in_svg_or_figures():
    """Row sentinel IDs in DataFrames must not leak into FigureSpec or static artifacts."""
    sentinel_id = "SECRET_PARTICIPANT_ROW_XYZ_12345"
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "score": [10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            "participant_id": [f"{sentinel_id}_{i}" for i in range(10)],
            "ssn_note": [f"Confidential_{sentinel_id}_{i}" for i in range(10)],
        }
    )

    assistant = ResearchAssistant(df)
    wf = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
        options=AnalysisOptions(confidence_level=0.95),
    )

    specs = build_figure_specs(wf)
    assert len(specs) >= 1

    # Check FigureSpecs do not contain sentinels in any field
    for spec in specs:
        spec_repr = repr(spec)
        assert sentinel_id not in spec_repr

    # Render static SVG and verify text does not contain sentinel
    figs = to_static_figures(wf, format="svg")
    assert len(figs) >= 1
    for fig in figs:
        svg_text = fig.data.decode("utf-8")
        assert sentinel_id not in svg_text

    # Render static PNG and verify
    png_figs = to_static_figures(wf, format="png")
    assert len(png_figs) >= 1
    for fig in png_figs:
        assert fig.data.startswith(b"\x89PNG\r\n\x1a\n")


def test_zero_recalculation_for_static_figures():
    """Generating static figures must not recalculate statistics or refit models."""
    wf = make_welch_t_workflow()
    assert wf.analysis is not None

    def exploding_statistic(*args, **kwargs):
        raise RuntimeError("Illegal recalculation detected during static figure rendering!")

    with (
        patch("scipy.stats.ttest_ind", side_effect=exploding_statistic),
        patch("scipy.stats.ttest_1samp", side_effect=exploding_statistic),
        patch("pyautostat.execution.execute_specification", side_effect=exploding_statistic),
        patch("pyautostat.execution.execute_selected_method", side_effect=exploding_statistic),
    ):
        figs = to_static_figures(wf, format="png")
        assert len(figs) >= 1
        assert figs[0].data.startswith(b"\x89PNG\r\n\x1a\n")
