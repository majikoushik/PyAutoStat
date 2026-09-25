"""Run every public PyAutoStat workflow with reproducible sample data.

From the repository root after ``python -m pip install -e ".[report]"``:

    python examples/example_usage.py --output-dir reports

Plotly is optional. Use ``--skip-interactive`` if it is not installed or if
only the static exports are needed. No external dataset or network is needed
to run the analyses; opening the interactive HTML loads Plotly from a CDN.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import pandas as pd

from pyautostat import (
    InsightEngine,
    PyAutoStatError,
    ReportGenerator,
    ResearchAssistant,
    StatisticalAnalyzer,
    detect_column_types,
    suggest_column_roles,
)

DEFAULT_SAMPLE_SIZE = 300
RANDOM_SEED = 42
BOOTSTRAP_SAMPLES = 100  # Keep the showcase quick; the library defaults to 499.


def create_sample_data(n: int = DEFAULT_SAMPLE_SIZE, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return a balanced, deterministic dataset with missing and skewed values."""
    if n < 60:
        raise ValueError("The showcase needs at least 60 rows for its categorical tests.")
    rng = np.random.default_rng(seed)
    control_count = n // 2
    groups = np.array(["Control"] * control_count + ["Treatment"] * (n - control_count))
    arms = np.resize(np.array(["Control", "Standard", "New"]), n)
    response = np.full(n, "no", dtype=object)
    for group, success_rate in (("Control", 0.30), ("Treatment", 0.70)):
        positions = np.flatnonzero(groups == group)
        response[positions[: round(len(positions) * success_rate)]] = "yes"

    variable_a = rng.normal(100, 12, n)
    frame = pd.DataFrame(
        {
            "record_id": [f"ID-{index:04d}" for index in range(1, n + 1)],
            "group": groups,
            "arm": arms,
            "age": rng.normal(45, 10, n),
            "outcome": rng.normal(50 + 5 * (groups == "Treatment"), 6, n),
            "measurement_1": rng.exponential(8, n),
            "measurement_2": rng.normal(100, 15, n),
            "variable_a": variable_a,
            "variable_b": 0.8 * variable_a + rng.normal(0, 4, n),
            "category": np.resize(np.array(["Type_A", "Type_B"]), n),
            "response": response,
            "signup_date": pd.date_range("2024-01-01", periods=n).astype(str),
        }
    )
    for column, fraction in (("measurement_1", 0.05), ("variable_a", 0.03)):
        count = max(1, round(n * fraction))
        positions = rng.choice(n, size=count, replace=False)
        frame.loc[positions, column] = np.nan
    return frame


def heading(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def show_analysis(analyzer: StatisticalAnalyzer, results: dict) -> None:
    """Visit every section returned by analyze_all()."""
    heading("1. analyze_all(): every analysis section")
    overview = results["overview"]
    print("Overview:", overview["shape"], "rows/columns; dtypes:")
    pprint({name: str(dtype) for name, dtype in overview["dtypes"].items()})
    print("Analyzer numeric columns:", analyzer.numeric_cols)
    print("Analyzer categorical columns:", analyzer.categorical_cols)

    print("\nDescriptive statistics for age (including quartiles and variation):")
    pprint(results["descriptive"]["age"])
    print("\nNormality results for age (Shapiro, D'Agostino, Anderson where applicable):")
    pprint(results["normality"].get("age", {}))
    print("\nOutlier methods for skewed measurement_1:")
    pprint(results["outliers"]["measurement_1"])

    print("\nCorrelation of variable_a and variable_b:")
    correlation = results["correlation"]
    for method in ("pearson", "spearman", "kendall"):
        print(f"  {method}: {correlation[method]['matrix']['variable_a']['variable_b']}")
    print("  Pearson p-value:", correlation["p_values"]["variable_a"]["variable_b"])
    print(
        "  Pairwise observed rows:",
        correlation["pearson"]["sample_sizes"]["variable_a"]["variable_b"],
    )

    print("\nMissing-data summary:")
    missing = results["missing_data"]
    print("  Overall missing percentage:", missing["overall_missing_percentage"])
    print("  Rows containing missing values:", missing["rows_with_missing"])
    print("  Complete rows:", missing["complete_rows"])
    pprint({name: info for name, info in missing["by_column"].items() if info["count"]})
    print("\nData-quality metrics (completeness, uniqueness, duplicates):")
    pprint(results["data_quality"])
    print("\nCategorical summary for group:")
    pprint(results["categorical_summary"]["group"])
    print("\nDistribution analysis for measurement_1:")
    pprint(results["distributions"]["measurement_1"])

    print("\nColumn role suggestions for ID, outcome, and date:")
    pprint(
        {name: results["column_roles"][name] for name in ("record_id", "outcome", "signup_date")}
    )
    print("\nDetected types for category text and date-like text:")
    pprint({name: results["column_types"][name] for name in ("category", "signup_date")})
    histogram = results["histograms"]["measurement_1"]
    print("\nPrecomputed histogram for measurement_1:")
    print("  First three bin edges:", histogram["bin_edges"][:3])
    print("  First three counts:", histogram["counts"][:3])
    print("  Total binned observations:", sum(histogram["counts"]))
    print("\nAnalysis warnings (skipped or undefined calculations):")
    pprint(results["analysis_warnings"])


def show_column_intelligence() -> None:
    """Show both public detection helpers on common input formats."""
    heading("2. detect_column_types() and suggest_column_roles()")
    contacts = pd.DataFrame(
        {
            "user_id": ["A1", "A2", "A3"],
            "target": [1, 0, 1],
            "contact_email": ["a@example.org", "b@example.org", None],
            "website": ["https://example.org/a", "http://example.net", None],
            "phone": ["+1 212 555 0123", "+1 212 555 0124", None],
            "created_at": ["2024-01-01", "2024-01-02", None],
            "purchase_price": [10.0, 12.5, 8.0],
            "score": [1.0, 2.0, 3.0],
            "is_active": pd.Series([True, False, True], dtype="boolean"),
            "event_at": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "all_missing": [None, None, None],
        }
    )
    print("Detected formats and missing percentages:")
    pprint(detect_column_types(contacts))
    print("Suggested roles and actions (advisory; data is never transformed):")
    pprint(suggest_column_roles(contacts))
    declared_profile = ResearchAssistant(contacts).profile(
        data_dictionary={"score": {"type": "continuous", "unit": "points", "valid_range": [0, 100]}}
    )
    print("\nOptional declared score metadata:")
    pprint(declared_profile["variable_intelligence"]["score"])


def show_insights(results: dict) -> dict:
    heading("3. InsightEngine: severity-rated findings and recommendations")
    engine = InsightEngine(results)
    generated = engine.generate_insights()
    summary = engine.get_summary()
    print("Generated insights:", len(generated))
    print(
        "Severity counts:",
        {name: summary[f"{name}_severity"] for name in ("high", "medium", "low")},
    )
    for insight in summary["insights"]:
        print(f"  [{insight['severity']}] {insight['category']}: {insight['finding']}")
        for recommendation in insight.get("recommendation", []):
            print("    -", recommendation)
    return summary


def show_recommendation(frame: pd.DataFrame) -> None:
    """Show design review without executing a hypothesis test."""
    heading("3a. ResearchAssistant: question and method recommendation")
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="outcome",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"outcome": "continuous"},
    )
    recommendation = assistant.recommend_test(draft)
    print("Status:", recommendation.status.value)
    print("Method:", recommendation.method_name)
    print("Reason:", recommendation.rationale)
    print("Decision trace:", recommendation.to_dict()["decision_trace"])
    result = assistant.analyze(draft)
    print("Executed method:", result.method_id, "status:", result.status.value)
    print("Estimate:", result.values.get("primary_estimate"))
    print("P-value:", result.values.get("p_value"))
    print("Sample:", result.metadata["sample"])
    interpretation = assistant.interpret(result)
    print("Interpretation:", interpretation.summary)
    print("Finding codes:", [finding.code for finding in interpretation.findings])


def show_hypothesis_tests(analyzer: StatisticalAnalyzer) -> list[dict]:
    """Demonstrate automatic and explicit independent-group comparisons."""
    heading("4. hypothesis_tests(): selection, assumptions, effect sizes, intervals")
    requests = (
        ("Two groups, auto mean", "group", "outcome", "auto", "mean"),
        ("Two groups, Welch t-test", "group", "outcome", "ttest", None),
        ("Two groups, Mann-Whitney U", "group", "measurement_1", "mannwhitney", None),
        ("Three groups, auto distribution", "arm", "measurement_1", "auto", "distribution"),
        ("Three groups, ANOVA", "arm", "outcome", "anova", None),
        ("Three groups, Kruskal-Wallis", "arm", "measurement_1", "kruskal", None),
    )
    comparisons = []
    for label, group_col, value_col, test_type, estimand in requests:
        result = analyzer.hypothesis_tests(
            group_col,
            value_col,
            test_type=test_type,
            confidence_level=0.95,
            bootstrap_samples=BOOTSTRAP_SAMPLES,
            random_state=RANDOM_SEED,
            estimand=estimand,
        )
        comparisons.append(result)
        print(f"\n{label}: {result['test']}")
        print("  Groups:", result["groups"])
        print("  Statistic / p-value:", result["statistic"], "/", result["p_value"])
        print("  Selected because:", result["assumptions"]["selection_reason"])
        print("  Per-group normality:", result["assumptions"]["normality"])
        print("  Levene p-value:", result["assumptions"]["levene_p_value"])
        print("  Equal-variance status:", result["assumptions"]["equal_variance_status"])
        print("  Assumption warnings:", result["assumptions"]["warnings"])
        print("  Effect size and bootstrap interval:")
        pprint(result["effect_size"])
        if "confidence_interval" in result:
            print("  Analytical mean-difference interval:")
            pprint(result["confidence_interval"])
    no_bootstrap = analyzer.hypothesis_tests(
        "group", "outcome", "ttest", confidence_level=0.90, bootstrap_samples=0
    )
    print("\nOptional configuration: 90% mean-difference interval and no bootstrap:")
    pprint(no_bootstrap["confidence_interval"])
    print("  Effect-size interval:", no_bootstrap["effect_size"]["confidence_interval"])
    return comparisons


def show_categorical_association(analyzer: StatisticalAnalyzer) -> list[dict]:
    heading("5. categorical_association(): chi-square, Cramer's V, Cohen's h")
    binary = analyzer.categorical_association(
        "group",
        "response",
        success_value="yes",
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        random_state=RANDOM_SEED,
    )
    print("Two-by-two observed counts:", binary["observed_counts"])
    print("Expected counts:", binary["expected_counts"])
    print("Chi-square / p-value / degrees of freedom:")
    print(binary["statistic"], binary["p_value"], binary["degrees_of_freedom"])
    print("Cramer's V with interval:")
    pprint(binary["effect_size"])
    print("Cohen's h for 'yes', first group minus second:")
    pprint(binary["cohens_h"])

    multi = analyzer.categorical_association(
        "arm", "category", bootstrap_samples=BOOTSTRAP_SAMPLES, random_state=RANDOM_SEED
    )
    print("\nThree-by-two association (Cramer's V only):")
    pprint({key: multi[key] for key in ("groups", "outcomes", "observed_counts", "effect_size")})
    return [binary, multi]


def show_reports(report: ReportGenerator, output_dir: Path, skip_interactive: bool) -> bool:
    """Show in-memory and file-based output for every report format."""
    heading("6. ReportGenerator: dict, JSON, CSV, HTML, interactive HTML")
    print("to_dict() top-level keys:", list(report.to_dict()))
    json_text = report.to_json(pretty=True)
    print("to_json() first hypothesis test:")
    pprint(json.loads(json_text)["hypothesis_tests"][0]["test"])
    json_path = output_dir / "analysis.json"
    print(report.to_json(json_path))

    tables = report.to_csv()
    print("to_csv() in-memory table names:", list(tables))
    print("Descriptive CSV preview (DataFrame returned without writing):")
    print(tables["descriptive_stats"].head(3).to_string())
    csv_dir = output_dir / "csv"
    report.to_csv(csv_dir)
    print("Written CSV directory:", csv_dir)

    html = report.to_html(title="PyAutoStat showcase")
    print("to_html() returned", len(html), "characters")
    html_path = output_dir / "analysis.html"
    print(report.to_html(html_path, title="PyAutoStat showcase"))

    if skip_interactive:
        print("Interactive HTML skipped by --skip-interactive.")
        return False
    if find_spec("plotly") is None:
        print("Interactive HTML skipped: install pyautostat[report] to enable Plotly.")
        return False
    interactive_html = report.to_interactive_html(title="Interactive showcase")
    print("to_interactive_html() returned", len(interactive_html), "characters")
    interactive_path = output_dir / "interactive.html"
    print(report.to_interactive_html(interactive_path, title="Interactive showcase"))
    print("  Expand sections to render charts; tables can be filtered and sorted.")
    return True


def show_error_handling(output_dir: Path, report: ReportGenerator) -> None:
    """Display representative actionable exceptions and unavailable results."""
    heading("7. Bad-data and error-handling examples")
    all_missing = StatisticalAnalyzer(pd.DataFrame({"empty": [np.nan] * 8})).analyze_all()
    print("All-missing mean:", all_missing["descriptive"]["empty"]["mean"])
    print("Reason:", all_missing["analysis_warnings"][0]["message"])
    small = StatisticalAnalyzer(
        pd.DataFrame({"short": [1.0, np.nan, np.nan], "constant": [7, 7, 7]})
    ).analyze_all()
    print("Small/constant normality results:", small["normality"])
    print("Small/constant warnings:")
    pprint(small["analysis_warnings"])

    examples: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("InvalidDataError", lambda: StatisticalAnalyzer(pd.DataFrame({"x": [1, np.inf]}))),
        (
            "ColumnNotFoundError",
            lambda: StatisticalAnalyzer(pd.DataFrame({"x": [1, 2]})).hypothesis_tests(
                "missing", "x"
            ),
        ),
        (
            "InsufficientGroupsError",
            lambda: StatisticalAnalyzer(
                pd.DataFrame({"group": ["A", "A"], "value": [1, 2]})
            ).hypothesis_tests("group", "value"),
        ),
        (
            "InsufficientDataError",
            lambda: StatisticalAnalyzer(
                pd.DataFrame({"group": ["A", "A", "B"], "value": [1, 2, 3]})
            ).hypothesis_tests("group", "value"),
        ),
        (
            "InvalidTestError",
            lambda: StatisticalAnalyzer(
                pd.DataFrame({"group": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})
            ).hypothesis_tests("group", "value", test_type="not_a_test"),
        ),
        ("ReportError", lambda: report.to_json(output_dir)),
    )
    for expected_name, action in examples:
        try:
            action()
        except PyAutoStatError as exc:
            if type(exc).__name__ != expected_name:
                raise AssertionError(f"Expected {expected_name}, got {type(exc).__name__}") from exc
            print(f"  {expected_name}: {exc}")
        else:
            raise AssertionError(f"Expected {expected_name} to be raised")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every PyAutoStat feature on sample data.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path.cwd() / "reports", help="Report directory."
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="At least 60 rows."
    )
    parser.add_argument("--skip-interactive", action="store_true", help="Skip the Plotly report.")
    parser.add_argument("--verbose", action="store_true", help="Print the complete result dict.")
    arguments = parser.parse_args(argv)
    if arguments.sample_size < 60:
        parser.error("--sample-size must be at least 60")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("PyAutoStat complete feature showcase")
    print("Sample size:", arguments.sample_size)
    print("Output directory:", output_dir)
    frame = create_sample_data(arguments.sample_size)
    analyzer = StatisticalAnalyzer(frame)
    results = analyzer.analyze_all()
    print("Analyzer uses an input copy:", analyzer.df is not frame)

    show_analysis(analyzer, results)
    show_column_intelligence()
    summary = show_insights(results)
    show_recommendation(frame)
    comparisons = show_hypothesis_tests(analyzer)
    associations = show_categorical_association(analyzer)
    report = ReportGenerator(results, summary, hypothesis_results=comparisons + associations)
    interactive_written = show_reports(report, output_dir, arguments.skip_interactive)
    show_error_handling(output_dir, report)
    if arguments.verbose:
        heading("Complete analyze_all() result")
        pprint(results, width=100, sort_dicts=False)
    heading("Done")
    print("Open analysis.html in", output_dir)
    if interactive_written:
        print("Open interactive.html in", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
