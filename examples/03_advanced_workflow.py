"""Run a traceable research workflow from clarification through replay.

This example demonstrates the modern ResearchAssistant API: unresolved design
facts, a prospective analysis plan, estimand-aware sensitivity scenarios,
researcher-defined practical significance, canonical reports, audit,
reproducibility, prospective planning, a paired analysis, and a serializable
session snapshot.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from pyautostat import (
    AnalysisOptions,
    MeaningfulEffectThreshold,
    ResearchAssistant,
    SensitivitySpecification,
    StudyPlanner,
    reproduce,
)

DATA_FILE = Path(__file__).with_name("CustomerDataset.xlsx")
RANDOM_SEED = 42


def load_customer_data() -> pd.DataFrame:
    try:
        return pd.read_excel(DATA_FILE)
    except ImportError as exc:
        raise SystemExit(
            'Reading the example workbook requires: python -m pip install -e ".[examples]"'
        ) from exc


def section(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete PyAutoStat workflow example.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("reports"),
        help="Directory for canonical report exports.",
    )
    return parser.parse_args()


def run_independent_workflow(frame: pd.DataFrame, output_dir: Path) -> None:
    section("1. STRUCTURED CLARIFICATION: UNKNOWN DESIGN STAYS UNKNOWN")
    analysis_frame = frame.drop(columns=["CustomerID"])
    assistant = ResearchAssistant(analysis_frame)
    assistant.enable_tracking()
    assistant.declare_planning("planned", reason="Declared before the numerical analysis.")

    incomplete = assistant.run(
        objective="compare_groups",
        outcome="TotalAvgMonthlySpend",
        predictor="Gender",
        estimand="mean",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary={
            "TotalAvgMonthlySpend": {"type": "continuous", "unit": "currency/month"},
            "Gender": {"type": "nominal"},
        },
    )
    print("Initial workflow status:", incomplete.status.value)
    print("Requested information:", [item.field for item in incomplete.missing_information])
    draft = assistant.update_question(
        incomplete.draft,
        design="independent",
        reason="Each row represents a different customer.",
    )

    section("2. PLAN THE ESTIMAND, SENSITIVITY SCENARIOS, AND MEANINGFUL EFFECT")
    pooled = SensitivitySpecification(
        name="Pooled-variance mean comparison",
        specification=draft.specification,
        method_id="student_t",
        rationale="Assess sensitivity of the mean contrast to an equal-variance assumption.",
        planning_status="planned",
        assumptions=("Equal population variances",),
    )
    distribution_specification = replace(
        draft.specification,
        question=replace(draft.specification.question, estimand="distribution"),
    )
    ranks = SensitivitySpecification(
        name="Supplementary rank-distribution comparison",
        specification=distribution_specification,
        method_id="mann_whitney_u",
        rationale=(
            "Ask a supplementary distribution question. This changes the estimand "
            "and cannot confirm robustness of the mean difference."
        ),
        planning_status="exploratory",
    )
    threshold = MeaningfulEffectThreshold(
        quantity="mean_difference",
        minimum_magnitude=20.0,
        direction="two_sided",
        unit="currency/month",
        rationale="A smaller monthly difference would not change the proposed business action",
        planning_status="planned",
    )
    plan = assistant.analysis_plan(
        draft,
        sensitivity_scenarios=[pooled, ranks],
        meaningful_threshold=threshold,
        report_style="apa",
    )
    print("Plan status:", plan.status.value)
    print("Planned method:", plan.primary_method_id)
    print("Plan created after analysis:", plan.created_after_analysis)

    section("3. EXECUTE ONCE AND INTERPRET RECORDED VALUES")
    workflow = assistant.run(draft=draft)
    if workflow.analysis is None or workflow.interpretation is None:
        raise RuntimeError(f"Analysis unavailable: {workflow.blockers}")
    result = workflow.analysis
    print(workflow.explain())
    print("Method ID:", result.method_id)
    print("Analyzed/excluded rows:", result.sample_size, "/", result.excluded_rows)
    print("Primary estimate:", result.values.get("primary_estimate"))
    print("Finding codes:", [item.code for item in workflow.interpretation.findings])

    section("4. COMPARE DECLARED SENSITIVITY SCENARIOS")
    sensitivity = assistant.sensitivity_analysis(result, scenarios=[pooled, ranks])
    print(sensitivity.compare())
    for scenario in sensitivity.scenario_results:
        print(
            scenario.name,
            "| status:",
            scenario.status.value,
            "| comparability:",
            scenario.comparability.value,
        )

    section("5. KEEP PRACTICAL IMPORTANCE SEPARATE FROM THE P-VALUE")
    practical = assistant.practical_significance(result, threshold=threshold)
    print(practical.verdict)
    print("Assessment status:", practical.status)
    print("Statistical significance:", practical.statistical_significance)
    print("Point estimate relation:", practical.point_estimate_relation)
    adherence = assistant.plan_adherence(
        plan,
        result,
        sensitivity=sensitivity,
        practical_significance=practical,
    )
    print("Plan adherence:", adherence.status)
    print(
        "Adherence fields:",
        {item["field"]: item["status"] for item in adherence.comparisons},
    )

    section("6. REPORT, AUDIT, REPLAY, AND SERIALIZE THE SESSION")
    report = assistant.report(
        result,
        interpretation=workflow.interpretation,
        sensitivity=sensitivity,
        practical_significance=practical,
        title="Customer spending mean comparison",
    )
    audit = assistant.audit(
        report,
        result=result,
        sensitivity=sensitivity,
        practical_significance=practical,
    )
    record = assistant.reproducibility_record(
        result,
        sensitivity=sensitivity,
        practical_significance=practical,
    )
    replay = reproduce(record, data=analysis_frame)
    completeness = assistant.reporting_completeness(report, style="apa")
    planning = StudyPlanner().independent_mean_power(
        target_difference=20.0,
        sd_group1=100.0,
        sd_group2=100.0,
        alpha=0.05,
        target_power=0.80,
    )
    snapshot = assistant.session_snapshot(
        workflow,
        sensitivity=sensitivity,
        practical_significance=practical,
        analysis_plan=plan,
        study_planning=planning,
        reporting_completeness=completeness,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = report.save_html(output_dir / "customer_analysis.html", style="apa", overwrite=True)
    markdown_path = report.save_markdown(
        output_dir / "customer_analysis.md", style="apa", overwrite=True
    )
    json_path = report.save_json(output_dir / "customer_analysis.json", overwrite=True)
    csv_paths = report.save_csv_tables(output_dir / "customer_analysis_tables", overwrite=True)
    (output_dir / "session_snapshot.json").write_text(snapshot.to_json(), encoding="utf-8")
    (output_dir / "decision_ledger.json").write_text(
        assistant.decision_ledger.to_json(), encoding="utf-8"
    )

    print("Audit:", audit.status)
    print("Same-data replay:", replay.status)
    print("Reporting completeness:", completeness.status)
    print("Prospective total sample size:", planning.total_required_n)
    print("Decision events:", len(assistant.decision_ledger.events))
    print("Snapshot schema:", snapshot.to_dict()["schema_version"])
    print("In-memory APA HTML characters:", len(report.to_html(style="apa")))
    print("Exports:", html_path, markdown_path, json_path)
    print("CSV tables:", len(csv_paths))


def run_paired_workflow(frame: pd.DataFrame) -> None:
    section("7. EXPLICIT PAIRED ANALYSIS AND PROSPECTIVE PAIRED PLANNING")
    paired_frame = frame[["CustomerID", "MonthlySpend_ProductA", "MonthlySpend_ProductB"]].melt(
        id_vars="CustomerID",
        var_name="Product",
        value_name="MonthlySpend",
    )
    paired_frame["Product"] = paired_frame["Product"].map(
        {
            "MonthlySpend_ProductA": "Product A",
            "MonthlySpend_ProductB": "Product B",
        }
    )
    paired = ResearchAssistant(paired_frame).run(
        objective="compare_groups",
        outcome="MonthlySpend",
        predictor="Product",
        design="paired",
        estimand="mean",
        unit_id="CustomerID",
        condition_order=("Product A", "Product B"),
        options=AnalysisOptions(random_seed=RANDOM_SEED),
        data_dictionary={"MonthlySpend": {"type": "continuous", "unit": "currency/month"}},
    )
    if paired.analysis is None:
        raise RuntimeError(f"Paired analysis unavailable: {paired.blockers}")
    sample = paired.analysis.metadata["sample"]
    print("Method:", paired.analysis.method_label)
    print("Contrast order:", paired.analysis.specification.condition_order)
    print("Complete pairs:", sample["complete_pairs"])
    print("Mean paired difference:", paired.analysis.values["primary_estimate"])
    print("Identifier values are used for matching and are not included in report output.")

    paired_planning = StudyPlanner().paired_mean_power(
        target_mean_difference=10.0,
        sd_difference=50.0,
        alpha=0.05,
        target_power=0.80,
    )
    print("Prospective complete pairs from supplied assumptions:", paired_planning.required_pairs)


def main() -> None:
    arguments = parse_arguments()
    frame = load_customer_data()
    print(f"Loaded bundled dataset: {len(frame):,} rows and {len(frame.columns)} columns.")
    run_independent_workflow(frame, arguments.output_dir.resolve())
    run_paired_workflow(frame)


if __name__ == "__main__":
    main()
