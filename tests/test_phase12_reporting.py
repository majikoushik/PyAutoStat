"""Phase 12 reporting completeness, styles, and safe LaTeX exports."""

import json
from pathlib import Path

import pandas as pd
import pytest

from pyautostat import (
    InvalidDataError,
    ReportError,
    ResearchAssistant,
    ResearchReport,
    assess_reporting_completeness,
)


@pytest.fixture
def report():
    assistant = ResearchAssistant(
        pd.DataFrame(
            {
                "group": ["a"] * 6 + ["b"] * 6,
                "score": [1, 2, 3, 4, 5, 7, 3, 4, 6, 7, 8, 10],
            }
        )
    )
    workflow = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        design="independent",
        estimand="mean",
        variable_types={"score": "continuous"},
    )
    return assistant, workflow.report


@pytest.mark.parametrize("style", ["general", "apa", "ieee"])
def test_styles_preserve_canonical_payload_and_visible_cautions(report, style):
    _, value = report
    before = value.to_dict()
    markdown = value.to_markdown(style=style)
    html = value.to_html(style=style)
    latex = value.to_latex(style=style)
    assert value.to_dict() == before
    assert (
        before["sections"]["results"]["primary_estimate"]
        == value.to_dict()["sections"]["results"]["primary_estimate"]
    )
    assert "Limitations" in markdown and "Warnings" in markdown
    assert "Limitations" in html and "Warnings" in html
    assert r"\section*{Limitations}" in latex and r"\section*{Warnings}" in latex


def test_oriented_styles_change_presentation_and_make_no_compliance_claim(report):
    _, value = report
    general = value.to_markdown()
    apa = value.to_markdown(style="apa")
    ieee = value.to_markdown(style="ieee")
    assert "APA-oriented summary" in apa
    assert "## 1. Research question and design" in ieee
    assert apa != general != ieee
    assert "compliant" not in apa.lower()
    assert "certified" not in ieee.lower()


def test_styled_exports_and_latex_remain_auditable(report):
    assistant, value = report
    for style in ("general", "apa", "ieee"):
        audit = assistant.audit(
            value,
            exports={
                "html": value.to_html(style=style),
                "markdown": value.to_markdown(style=style),
                "latex": value.to_latex(style=style),
            },
        )
        assert audit.status == "passed"
    altered = value.to_markdown(style="apa").replace("estimate =", "estimate = 999, ", 1)
    assert assistant.audit(value, exports={"markdown": altered}).status == "failed"


def test_latex_escapes_untrusted_user_text_and_save_is_explicit():
    assistant = ResearchAssistant(pd.DataFrame({"x": [1, 2, 3]}))
    result = assistant.analyze(
        assistant.prepare_question(objective="descriptive", description=r"A&B_#%${}~^\input{x}")
    )
    value = assistant.report(result, title=r"A&B_#%${}~^\input{x}")
    latex = value.to_latex(style="apa")
    for token in (r"\&", r"\_", r"\#", r"\%", r"\$", r"\{", r"\}"):
        assert token in latex
    assert r"\input{x}" not in latex
    destination = Path(".phase12_latex_save_test.tex")
    try:
        destination.unlink(missing_ok=True)
        assert value.save_latex(destination, style="ieee") == destination
        assert destination.read_text(encoding="utf-8") == value.to_latex(style="ieee")
        with pytest.raises(ReportError, match="already exists"):
            value.save_latex(destination)
        value.save_latex(destination, overwrite=True)
    finally:
        destination.unlink(missing_ok=True)


def test_completeness_is_machine_readable_deterministic_and_has_no_quality_score(report):
    _, value = report
    first = assess_reporting_completeness(value, style="apa")
    second = assess_reporting_completeness(value, style="apa")
    payload = json.loads(first.to_json())
    assert payload == json.loads(second.to_json())
    assert payload["status"] == "complete"
    assert "quality_score" not in payload
    assert {item["status"] for item in payload["items"]} <= {
        "present",
        "missing",
        "partial",
        "not_applicable",
    }
    assert any(item["status"] == "not_applicable" for item in payload["items"])


def test_completeness_distinguishes_report_defect_from_backend_limitation(report):
    _, value = report
    broken_payload = value.to_dict()
    broken_payload["sections"]["results"]["confidence_interval"] = None
    broken = ResearchReport(broken_payload)
    assessment = assess_reporting_completeness(broken)
    interval = next(
        item for item in assessment.items if item.code == "CONFIDENCE_INTERVAL_REPORTED"
    )
    assert interval.status == "missing"
    assert assessment.status == "incomplete"

    correlation = ResearchAssistant(pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 1, 4, 3, 6]})).run(
        objective="association",
        outcome="y",
        predictor="x",
        design="independent",
        estimand="linear",
        variable_types={"x": "continuous", "y": "continuous"},
    )
    limited = assess_reporting_completeness(correlation.report)
    interval = next(item for item in limited.items if item.code == "CONFIDENCE_INTERVAL_REPORTED")
    assert interval.status == "partial"
    assert limited.status == "partial"


def test_invalid_styles_and_inputs_are_rejected(report):
    _, value = report
    with pytest.raises(ReportError, match="style"):
        value.to_html(style="unknown")
    with pytest.raises(InvalidDataError, match="style"):
        assess_reporting_completeness(value, style="unknown")
    with pytest.raises(InvalidDataError, match="ResearchReport"):
        assess_reporting_completeness({})
