"""Example 09: Complete Research Lifecycle & Scientific Governance Showcase.

Business Question:
    "Do news subscribers and non-subscribers differ in average total monthly spend?"

Scientific Focus:
    The flagship demonstration of PyAutoStat's end-to-end research workflow:
    1. Structured clarification: unknown study design returns needs_input rather than guessing.
    2. Statistical analysis planning recorded BEFORE numerical execution.
    3. Method recommendation preserving the stated population-mean estimand.
    4. Declared sensitivity scenarios distinguishing same-estimand from different-estimand checks.
    5. Researcher-defined practical significance threshold evaluated separately from p-values.
    6. Plan adherence tracking without conduct judgment.
    7. Canonical multi-format report generation (HTML, Markdown, JSON, CSV tables).
    8. Consistency audit, reproducibility ledger, and supplied-data replay.
    9. Prospective power planning from researcher assumptions (never observed post-hoc power).
    10. Explicit within-unit paired analysis and UI-independent session snapshot serialization.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# Allow running directly from repository root or examples directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from _customer_data import (
    DATA_DICTIONARY,
    load_customer_data,
    section,
)

from pyautostat import (
    AnalysisOptions,
    MeaningfulEffectThreshold,
    ResearchAssistant,
    SensitivitySpecification,
    StudyPlanner,
    reproduce,
)

RANDOM_SEED = 42


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete PyAutoStat research workflow example."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Directory for canonical report exports.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run in fast mode with reduced bootstrap iterations for test/CI.",
    )
    return parser.parse_args()


def run_independent_workflow(frame: pd.DataFrame, output_dir: Path) -> None:
    section("1. STRUCTURED CLARIFICATION: UNKNOWN DESIGN STAYS UNKNOWN")
    assistant = ResearchAssistant(frame)
    assistant.enable_tracking()
    assistant.declare_planning("planned", reason="Declared before the numerical analysis.")

    # Incomplete specification: design omitted intentionally
    incomplete = assistant.run(
        objective="compare_groups",
        outcome="total_avg_monthly_spend",
        predictor="news_subscriber",
        estimand="mean",
        options=AnalysisOptions(alpha=0.05, confidence_level=0.95, random_seed=RANDOM_SEED),
        data_dictionary=DATA_DICTIONARY,
    )
    print("Initial workflow status:", incomplete.status.value)
    print("Requested information:", [item.field for item in incomplete.missing_information])

    # Caller explicitly declares the design
    draft = assistant.update_question(
        incomplete.draft,
        design="independent",
        reason="Each row represents a distinct customer; observational units are independent.",
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
        minimum_magnitude=25.0,
        direction="two_sided",
        unit="USD/month",
        rationale=(
            "Tutorial assumption: An illustrative $25/month minimum practical threshold "
            "to justify marketing intervention."
        ),
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
    replay = reproduce(record, data=frame)
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
    print("Exports:", html_path.name, markdown_path.name, json_path.name)
    print("CSV tables:", len(csv_paths))


def run_paired_workflow(full_data: pd.DataFrame) -> None:
    section("7. EXPLICIT PAIRED ANALYSIS AND PROSPECTIVE PAIRED PLANNING")
    # Reshape within-customer product spend for Product A and Product B
    paired_frame = full_data[
        ["customer_id", "monthly_spend_product_a", "monthly_spend_product_b"]
    ].melt(
        id_vars="customer_id",
        var_name="product",
        value_name="monthly_spend",
    )
    paired_frame["product"] = paired_frame["product"].map(
        {
            "monthly_spend_product_a": "Product A",
            "monthly_spend_product_b": "Product B",
        }
    )

    paired = ResearchAssistant(paired_frame).run(
        objective="compare_groups",
        outcome="monthly_spend",
        predictor="product",
        design="paired",
        estimand="mean",
        unit_id="customer_id",
        condition_order=("Product A", "Product B"),
        options=AnalysisOptions(random_seed=RANDOM_SEED),
        data_dictionary={"monthly_spend": {"type": "continuous", "unit": "USD/month"}},
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
    print(
        "Prospective complete pairs from supplied assumptions:",
        paired_planning.required_pairs,
    )


def main() -> None:
    arguments = parse_arguments()
    # 1. Load data without customer_id for main independent workflow
    analysis_frame = load_customer_data(include_customer_id=False)
    # 2. Load data with customer_id for explicit unit-matched paired analysis
    full_data = load_customer_data(include_customer_id=True)

    print(
        f"Loaded CustomerDataset.csv: {len(analysis_frame):,} rows and "
        f"{len(analysis_frame.columns)} analysis columns."
    )
    out_dir = arguments.output_dir.resolve()
    run_independent_workflow(analysis_frame, out_dir)
    run_paired_workflow(full_data)

    section("SUMMARY OF GENERATED ARTIFACTS")
    print(f"Artifact directory: {out_dir}")
    print("  - customer_analysis.html    : Canonical APA-style self-contained HTML report")
    print("  - customer_analysis.md      : Markdown report for documentation/PRs")
    print("  - customer_analysis.json    : Schema-versioned JSON report object")
    print("  - customer_analysis_tables/ : Directory of standalone CSV tables")
    print("  - decision_ledger.json      : Audit trail of decisions observed by assistant")
    print("  - session_snapshot.json     : Fully serializable session state for replay")
    print("\nComplete research workflow finished successfully.")


if __name__ == "__main__":
    main()
