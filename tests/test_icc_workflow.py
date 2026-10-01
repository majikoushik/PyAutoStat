"""End-to-end guided workflow, reporting, replay, and completeness tests for ICC."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pyautostat import (
    ResearchAssistant,
    assess_reporting_completeness,
    build_session_snapshot,
    compare_plan_to_result,
    reproduce,
)


@pytest.fixture
def ratings_df() -> pd.DataFrame:
    """Ratings panel with 6 targets, 4 judges (Shrout-Fleiss Table 4)."""
    raw = [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ]
    rows = []
    for t_idx, target_scores in enumerate(raw):
        for j_idx, score in enumerate(target_scores):
            rows.append(
                {
                    "subject": f"S{t_idx + 1}",
                    "judge": f"J{j_idx + 1}",
                    "rating": float(score),
                }
            )
    return pd.DataFrame(rows)


def test_icc_clarification_questions_when_parameters_missing(ratings_df: pd.DataFrame):
    """When design roles are omitted, question_builder prompts with structured questions."""
    assistant = ResearchAssistant(ratings_df)
    # Start with target/rater specified, but model/definition/unit omitted
    draft = assistant.prepare_question(
        objective="reliability",
        target="subject",
        rater="judge",
        outcome="rating",
        estimand="intraclass_correlation",
    )
    # The question is created and specification holds values
    assert draft.specification.question.target == "subject"
    assert draft.specification.question.rater == "judge"
    assert draft.specification.question.outcome == "rating"

    # When target/rater/outcome are omitted, clarification questions are generated
    draft_missing = assistant.prepare_question(
        objective="reliability",
        estimand="intraclass_correlation",
    )
    assert draft_missing.status.value == "needs_input"
    field_names = [q.field for q in draft_missing.questions]
    assert "target" in field_names
    assert "rater" in field_names
    assert "outcome" in field_names


def test_icc_full_guided_workflow_execution(ratings_df: pd.DataFrame):
    """Full guided flow via ResearchAssistant.run."""
    assistant = ResearchAssistant(ratings_df)
    workflow = assistant.run(
        objective="reliability",
        target="subject",
        rater="judge",
        outcome="rating",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
    )
    assert workflow.status.value == "completed"
    assert workflow.analysis.method_id == "intraclass_correlation"
    assert workflow.analysis.values["variant"] == "icc_2_1"
    assert workflow.analysis.values["estimate"] == pytest.approx(0.2898, abs=0.001)
    assert workflow.audit.status == "passed"

    # Explain output checks
    explained = workflow.explain()
    assert "RELIABILITY ESTIMATE" in explained
    assert "ICC(2,1)" in explained
    assert "PANEL DESIGN" in explained
    assert "DIAGNOSTIC AND MISSINGNESS NOTES" in explained


def test_icc_report_tables_and_styled_exports(ratings_df: pd.DataFrame, tmp_path: Path):
    """Report payload contains all required ICC tables and renders to text/HTML/LaTeX/CSV."""
    assistant = ResearchAssistant(ratings_df)
    workflow = assistant.intraclass_correlation(
        target="subject",
        rater="judge",
        value="rating",
        model="two_way_mixed",
        definition="consistency",
        unit="average",
    )
    report = workflow.report
    payload = report.to_dict()

    table_ids = {t["id"] for t in payload["tables"]}
    assert "icc_summary" in table_ids
    assert "icc_anova_table" in table_ids
    assert "icc_variance_components" in table_ids
    assert "icc_all_variants" in table_ids

    # Export formats
    md_text = report.to_markdown()
    assert "Intraclass correlation summary" in md_text
    assert "ANOVA mean squares table" in md_text
    assert "Method-of-moments variance components" in md_text

    html_text = report.to_html()
    assert "<table" in html_text
    assert "Intraclass correlation summary" in html_text

    latex_text = report.to_latex()
    assert "\\documentclass{article}" in latex_text
    assert "\\section*{" in latex_text

    csv_tables = report.to_csv_tables()
    assert "icc_summary" in csv_tables
    assert "Variant" in csv_tables["icc_summary"] or "icc" in csv_tables["icc_summary"].lower()

    # File saving
    report_file = tmp_path / "icc_report.md"
    saved = report.save_markdown(report_file)
    assert saved.exists()


def test_icc_reproducibility_and_replay(ratings_df: pd.DataFrame):
    """Reproducibility record captures full calculation and replays deterministically."""
    assistant = ResearchAssistant(ratings_df)
    workflow = assistant.intraclass_correlation(
        target="subject",
        rater="judge",
        value="rating",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
    )
    rep_record = workflow.reproducibility
    assert rep_record.method_id == "intraclass_correlation"

    # Replay
    replay_res = reproduce(rep_record, data=ratings_df)
    assert replay_res.status == "reproduced"
    assert len(replay_res.differing_fields) == 0


def test_icc_completeness_check(ratings_df: pd.DataFrame):
    """Reporting completeness check validates that all required ICC items are present."""
    assistant = ResearchAssistant(ratings_df)
    workflow = assistant.intraclass_correlation(
        target="subject",
        rater="judge",
        value="rating",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
    )
    completeness = assess_reporting_completeness(workflow.report)
    assert completeness.status in {"complete", "acceptable", "pass"}
    item_codes = {item.code for item in completeness.items}
    assert {
        "ICC_ESTIMATE_REPORTED",
        "ICC_INTERVAL_REPORTED",
        "ICC_ANOVA_TABLE_REPORTED",
        "ICC_VARIANCE_COMPONENTS_REPORTED",
        "ICC_F_TEST_REPORTED",
    } <= item_codes


def test_icc_session_snapshot_and_analysis_plan(ratings_df: pd.DataFrame):
    """Session snapshot and statistical analysis plan correctly serialize ICC specifications."""
    assistant = ResearchAssistant(ratings_df)
    workflow = assistant.intraclass_correlation(
        target="subject",
        rater="judge",
        value="rating",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
    )
    snapshot = build_session_snapshot(workflow)
    assert snapshot.payload["workflow"]["status"] == "completed"
    assert snapshot.payload["workflow"]["analysis"]["method_id"] == "intraclass_correlation"
    json.loads(snapshot.to_json())

    # Analysis plan creation and adherence
    draft = assistant.prepare_question(
        objective="reliability",
        target="subject",
        rater="judge",
        outcome="rating",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
    )
    plan = assistant.analysis_plan(draft)
    assert plan.primary_method_id == "intraclass_correlation"
    assert plan.effect_quantity == "intraclass_correlation"
    adherence = compare_plan_to_result(plan, workflow.analysis)
    assert adherence.status == "matched"
