"""Run estimand-aware comparisons on the bundled customer dataset.

The example uses the compatibility StatisticalAnalyzer interface to show the
numerical result dictionaries. Method choice starts with the research target
and design. Diagnostics are reported, but they do not silently replace a mean
question with a rank-distribution question.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from pyautostat import StatisticalAnalyzer

DATA_FILE = Path(__file__).with_name("CustomerDataset.xlsx")
BOOTSTRAP_SAMPLES = 100  # Short demonstration; increase to reduce Monte Carlo error.
RANDOM_SEED = 42


def load_analysis_data() -> pd.DataFrame:
    try:
        frame = pd.read_excel(DATA_FILE)
    except ImportError as exc:
        raise SystemExit(
            'Reading the example workbook requires: python -m pip install -e ".[examples]"'
        ) from exc
    return frame.drop(columns=["CustomerID"])


def _number(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:.{digits}g}"
    return "unavailable"


def print_comparison(title: str, result: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("-" * 72)
    print("Method:", result["test"])
    print("Groups:", result.get("groups"))
    print(
        "Rows:",
        result.get("sample_size"),
        "analyzed;",
        result.get("excluded_rows", 0),
        "excluded",
    )
    print("Statistic:", _number(result.get("statistic")))
    print("P-value:", _number(result.get("p_value")))
    print("Selection rationale:", result.get("assumptions", {}).get("selection_reason"))

    effect = result.get("effect_size") or {}
    print(
        "Effect:",
        effect.get("name", "unavailable"),
        "=",
        _number(effect.get("value")),
        f"({effect.get('interpretation', 'unavailable')})",
    )
    effect_interval = effect.get("confidence_interval")
    if effect_interval:
        print(
            "Effect interval:",
            f"[{_number(effect_interval.get('lower'))}, {_number(effect_interval.get('upper'))}]",
            f"from {effect_interval.get('valid_resamples')} valid resamples",
        )
    mean_interval = result.get("confidence_interval")
    if mean_interval:
        print(
            "Mean-difference interval:",
            f"[{_number(mean_interval.get('lower'))}, {_number(mean_interval.get('upper'))}]",
        )
    for warning in result.get("assumptions", {}).get("warnings", ()):
        print("Warning:", warning)
    print("Interpretation:", result.get("interpretation"))


def main() -> None:
    frame = load_analysis_data()
    print(f"Loaded {len(frame):,} rows and {len(frame.columns)} analysis columns.")
    analyzer = StatisticalAnalyzer(frame)

    # Q1 asks about population means, so auto selection preserves the mean estimand.
    mean_result = analyzer.hypothesis_tests(
        "Gender",
        "TotalAvgMonthlySpend",
        test_type="auto",
        estimand="mean",
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        random_state=RANDOM_SEED,
    )
    print_comparison(
        "Q1. Mean monthly spending difference by gender (independent groups)",
        mean_result,
    )

    # Q2 is explicitly a rank-distribution question across more than two groups.
    job_result = analyzer.hypothesis_tests(
        "JobCategory",
        "MonthlySpend_ProductA",
        test_type="kruskal",
        estimand="distribution",
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        random_state=RANDOM_SEED,
    )
    print_comparison(
        "Q2. Product A spending distributions across job categories",
        job_result,
    )

    # Q3 treats coded binary columns as categories before testing association.
    categorical = frame.assign(
        HomeOwnerLabel=frame["HomeOwner"].map({1: "Owner", 0: "Renter"}),
        ValueLabel=frame["HighValueCustomer"].map({1: "High Value", 0: "Standard"}),
    )
    association = StatisticalAnalyzer(categorical).categorical_association(
        "HomeOwnerLabel",
        "ValueLabel",
        success_value="High Value",
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        random_state=RANDOM_SEED,
    )
    print("\n" + "=" * 72)
    print("Q3. Association between home ownership and customer value category")
    print("-" * 72)
    print("Method:", association["test"])
    print("Chi-square:", _number(association.get("statistic")))
    print("Degrees of freedom:", association.get("degrees_of_freedom"))
    print("P-value:", _number(association.get("p_value")))
    print("Observed counts:", association.get("observed_counts"))
    print("Cramer's V:", _number((association.get("effect_size") or {}).get("value")))
    cohens_h = association.get("cohens_h") or {}
    print("Cohen's h for the declared success category:", _number(cohens_h.get("value")))

    # Q4 deliberately targets a rank-distribution contrast. It is not selected
    # merely because a normality test rejects.
    age_result = analyzer.hypothesis_tests(
        "ActiveLifestyle",
        "Age",
        test_type="mannwhitney",
        estimand="distribution",
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        random_state=RANDOM_SEED,
    )
    print_comparison(
        "Q4. Age distributions by active-lifestyle category",
        age_result,
    )

    print("\n" + "=" * 72)
    print("TRACEABLE SUMMARY")
    print("=" * 72)
    for label, result in (
        ("Q1 mean difference", mean_result),
        ("Q2 rank distributions", job_result),
        ("Q3 categorical association", association),
        ("Q4 rank distributions", age_result),
    ):
        p_value = result.get("p_value")
        evidence = (
            "p < 0.05"
            if isinstance(p_value, (int, float)) and math.isfinite(p_value) and p_value < 0.05
            else "p >= 0.05 or unavailable"
        )
        print(f"{label:30} {result.get('test', 'unavailable'):22} {evidence}")
    print(
        "\nA p-value does not establish effect importance or causality. "
        "See example 3 for practical thresholds, planning, audit, and replay."
    )


if __name__ == "__main__":
    main()
