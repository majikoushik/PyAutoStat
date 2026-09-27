"""Executive summaries consume stored results without rerunning analyses."""

import pandas as pd

from pyautostat import ReportGenerator, ResearchAssistant


def _profile(*, missing=0, duplicate_rows=0, dangerous_column=None):
    by_column = {}
    if missing:
        name = dangerous_column or "score"
        by_column[name] = {"count": missing, "percentage": 20.0}
    return {
        "overview": {"total_rows": 10, "total_columns": 2},
        "missing_data": {"total_missing_cells": missing, "by_column": by_column},
        "data_quality": {
            "completeness": 1 - missing / 20,
            "duplicate_rows": duplicate_rows,
            "duplicate_rows_percentage": duplicate_rows * 10.0,
        },
    }


def _hypothesis(name, p, effect_name="Cohen's d", effect=0.8, magnitude="large"):
    return {
        "test": name,
        "p_value": p,
        "sample_size": 40,
        "effect_size": {
            "name": effect_name,
            "value": effect,
            "interpretation": magnitude,
        },
        "assumptions": {"diagnostic_alpha": 0.05},
    }


def test_legacy_html_summary_classifies_multiple_stored_analyses_with_effects():
    analyses = [
        _hypothesis("Large comparison", 0.001, effect=1.2, magnitude="large"),
        _hypothesis("Small comparison", 0.01, effect=0.1, magnitude="negligible"),
        _hypothesis("Uncertain comparison", 0.4, effect=0.5, magnitude="medium"),
    ]
    report = ReportGenerator(_profile(), hypothesis_results=analyses)

    first = report.to_html()
    second = report.to_html()

    assert first == second
    assert '<section class="executive-summary">' in first
    assert "Executive Summary" in first
    assert "10 row(s) across 2 column(s)" in first
    assert "contains 3 statistical analyses" in first
    assert "Large comparison: p = 0.001" in first
    assert "effect size is large" in first
    assert "Small comparison: p = 0.01" in first
    assert "effect is negligible" in first
    assert "Uncertain comparison: p = 0.4" in first
    assert "does not reach significance" in first
    assert "[object Object]" not in first


def test_legacy_summary_handles_unavailable_effect_and_data_quality():
    no_effect = {
        "test": "Recorded test",
        "p_value": 0.02,
        "sample_size": 10,
        "effect_size": {},
        "assumptions": {"diagnostic_alpha": 0.05},
    }
    html = ReportGenerator(
        _profile(missing=4, duplicate_rows=1), hypothesis_results=no_effect
    ).to_html()

    assert "effect magnitude is unavailable" in html
    assert "Missingness is concentrated in &#x27;score&#x27;" in html
    assert "4 value(s)" in html
    assert "1 duplicate row(s)" in html
    assert "Practical significance was assessed" not in html
    assert "Sensitivity analysis was performed" not in html


def test_legacy_descriptive_summary_is_clean_and_interactive_summary_is_present():
    report = ReportGenerator(_profile())
    html = report.to_html()

    assert "No inferential test is represented" in html
    assert "No missing values were recorded" in html
    assert "No duplicate rows were recorded" in html
    interactive = report.to_interactive_html()
    assert "Executive Summary" in interactive
    assert "No inferential test is represented" in interactive


def test_summary_escapes_user_controlled_profile_and_method_text():
    dangerous = "<script>alert(1)</script>"
    html = ReportGenerator(
        _profile(missing=2, dangerous_column=dangerous),
        hypothesis_results=_hypothesis(dangerous, 0.3),
    ).to_html()

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_canonical_summary_uses_stored_interpretation_and_profile_data():
    frame = pd.DataFrame(
        {"group": ["A"] * 8 + ["B"] * 8, "score": list(range(8)) + list(range(4, 12))}
    )
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )
    result = assistant.analyze(draft)
    report = assistant.report(result)
    html = report.to_html()

    assert '<section class="executive-summary">' in html
    assert "contains 1 statistical analysis" in html
    assert "Welch independent-samples t-test" in html
    assert "for &#x27;score&#x27; and &#x27;group&#x27;" in html
    assert "effect size is large" in html
    assert "confidence interval" in html
    assert (
        str(result.values["p_value"]) not in html
    )  # display narration uses a stable rounded value
    assert report.to_dict()["schema_version"] == 1
    assert "executive_summary" not in report.to_dict()

    descriptive = assistant.analyze(assistant.prepare_question(objective="descriptive"))
    descriptive_html = assistant.report(descriptive).to_html()
    assert "16 row(s) across 2 column(s)" in descriptive_html
    assert "No inferential test is represented" in descriptive_html
    assert "Data quality:" in descriptive_html
