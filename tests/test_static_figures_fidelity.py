"""Tests verifying scientific and numerical fidelity between FigureSpec and Plotly figures."""

from __future__ import annotations

import math

from method_factories import (
    make_chi_square_workflow,
    make_linear_regression_workflow,
    make_logistic_regression_workflow,
    make_one_way_anova_workflow,
    make_pearson_workflow,
    make_two_way_anova_workflow,
    make_welch_t_workflow,
)

from pyautostat.presentation.figures.adapters import build_figure_specs
from pyautostat.presentation.figures.plotly import spec_to_plotly_figure


def test_welch_t_estimate_ci_numerical_fidelity():
    wf = make_welch_t_workflow()
    specs = build_figure_specs(wf)
    assert len(specs) >= 1
    spec = specs[0]
    assert spec.kind == "estimate_ci"
    assert len(spec.series) == 1
    series = spec.series[0]

    fig = spec_to_plotly_figure(spec)
    assert fig is not None
    assert len(fig.data) >= 1

    scatter = fig.data[0]
    # Estimate match
    assert math.isclose(scatter.x[0], series.estimate, rel_tol=1e-7)

    # Confidence interval bounds match error bar arrays
    expected_error_plus = series.upper - series.estimate
    expected_error_minus = series.estimate - series.lower
    assert math.isclose(scatter.error_x.array[0], expected_error_plus, rel_tol=1e-7)
    assert math.isclose(scatter.error_x.arrayminus[0], expected_error_minus, rel_tol=1e-7)

    # Reference line at 0.0
    shapes = fig.layout.shapes
    assert any(s.x0 == 0.0 and s.x1 == 0.0 for s in shapes)


def test_pearson_correlation_estimate_ci_fidelity():
    wf = make_pearson_workflow()
    specs = build_figure_specs(wf)
    assert len(specs) >= 1
    spec = specs[0]
    assert spec.kind == "estimate_ci"
    series = spec.series[0]

    fig = spec_to_plotly_figure(spec)
    assert fig is not None
    scatter = fig.data[0]
    assert math.isclose(scatter.x[0], series.estimate, rel_tol=1e-7)
    if series.upper is not None and series.lower is not None:
        assert math.isclose(scatter.error_x.array[0], series.upper - series.estimate, rel_tol=1e-7)
        assert math.isclose(
            scatter.error_x.arrayminus[0], series.estimate - series.lower, rel_tol=1e-7
        )


def test_linear_regression_coefficient_forest_fidelity():
    wf = make_linear_regression_workflow()
    specs = build_figure_specs(wf)
    forest_specs = [s for s in specs if s.kind == "forest"]
    assert len(forest_specs) >= 1
    spec = forest_specs[0]

    fig = spec_to_plotly_figure(spec)
    assert fig is not None
    scatter = fig.data[0]

    # Verify every coefficient in series appears in plot data in order
    assert len(scatter.x) == len(spec.series)
    assert len(scatter.y) == len(spec.series)

    for i, s in enumerate(spec.series):
        assert scatter.y[i] == s.label
        assert math.isclose(scatter.x[i], s.estimate, rel_tol=1e-7)
        if s.upper is not None and s.lower is not None:
            assert math.isclose(scatter.error_x.array[i], s.upper - s.estimate, rel_tol=1e-7)
            assert math.isclose(scatter.error_x.arrayminus[i], s.estimate - s.lower, rel_tol=1e-7)


def test_logistic_odds_ratio_forest_fidelity():
    wf = make_logistic_regression_workflow()
    specs = build_figure_specs(wf)
    or_specs = [s for s in specs if s.kind == "odds_ratio_forest"]
    assert len(or_specs) >= 1
    spec = or_specs[0]

    fig = spec_to_plotly_figure(spec)
    assert fig is not None
    assert fig.layout.xaxis.type == "log"

    scatter = fig.data[0]
    for i, s in enumerate(spec.series):
        assert scatter.y[i] == s.label
        assert math.isclose(scatter.x[i], s.estimate, rel_tol=1e-7)
        # OR estimates must be positive
        assert scatter.x[i] > 0

    # Neutral reference line at OR = 1.0
    shapes = fig.layout.shapes
    assert any(s.x0 == 1.0 and s.x1 == 1.0 for s in shapes)


def test_one_way_anova_pairwise_forest_fidelity():
    wf = make_one_way_anova_workflow()
    specs = build_figure_specs(wf)
    pairwise_specs = [s for s in specs if s.kind == "pairwise_forest"]
    assert len(pairwise_specs) >= 1
    spec = pairwise_specs[0]

    fig = spec_to_plotly_figure(spec)
    assert fig is not None
    scatter = fig.data[0]

    for i, s in enumerate(spec.series):
        assert scatter.y[i] == s.label
        assert math.isclose(scatter.x[i], s.estimate, rel_tol=1e-7)


def test_chi_square_count_heatmap_fidelity():
    wf = make_chi_square_workflow()
    specs = build_figure_specs(wf)
    heatmap_specs = [s for s in specs if s.kind == "count_heatmap"]
    assert len(heatmap_specs) >= 1
    spec = heatmap_specs[0]

    fig = spec_to_plotly_figure(spec)
    assert fig is not None
    assert len(fig.data) >= 1
    heatmap = fig.data[0]

    # Verify matrix structure and counts match FigureSpec
    assert heatmap.type == "heatmap"
    assert list(heatmap.x) == list(spec.series[0].categories)
    assert list(heatmap.y) == [s.label for s in spec.series]

    for r_idx, s in enumerate(spec.series):
        for c_idx, val in enumerate(s.values):
            assert heatmap.z[r_idx][c_idx] == val


def test_two_way_anova_cell_profile_fidelity():
    wf = make_two_way_anova_workflow()
    specs = build_figure_specs(wf)
    profile_specs = [s for s in specs if s.kind == "cell_profile"]
    assert len(profile_specs) >= 1
    spec = profile_specs[0]

    fig = spec_to_plotly_figure(spec)
    assert fig is not None
    # Traces correspond to series
    assert len(fig.data) == len(spec.series)
    for i, s in enumerate(spec.series):
        trace = fig.data[i]
        assert trace.name == s.label


def test_no_traffic_light_colors_or_stars():
    """Ensure no green=significant or red=nonsignificant coloring or significance stars."""
    wf = make_linear_regression_workflow()
    specs = build_figure_specs(wf)
    for spec in specs:
        fig = spec_to_plotly_figure(spec)
        assert fig is not None
        # Check titles / annotations do not contain significance stars
        assert "*" not in (fig.layout.title.text or "")
        for annot in fig.layout.annotations or ():
            assert "*" not in (annot.text or "")
