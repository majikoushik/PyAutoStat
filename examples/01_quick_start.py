"""Profile the bundled customer dataset without exposing customer identifiers.

Run from the repository root after installing the example dependency::

    python -m pip install -e ".[examples]"
    python examples/01_quick_start.py

The workbook is used only as a local demonstration dataset. The example does
not print or export row-level values.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pyautostat import InsightEngine, ResearchAssistant

DATA_FILE = Path(__file__).with_name("CustomerDataset.xlsx")
KEY_VARIABLES = ("Age", "Household_Income", "TotalAvgMonthlySpend", "CarValue")


def load_analysis_data() -> pd.DataFrame:
    """Load the workbook and remove the unique identifier from this profile."""
    try:
        frame = pd.read_excel(DATA_FILE)
    except ImportError as exc:
        raise SystemExit(
            'Reading the example workbook requires: python -m pip install -e ".[examples]"'
        ) from exc
    return frame.drop(columns=["CustomerID"])


def main() -> None:
    frame = load_analysis_data()
    print(f"Loaded {len(frame):,} rows and {len(frame.columns)} analysis columns.")
    print("CustomerID is intentionally excluded from descriptive output.\n")

    # Dtypes alone cannot establish the meaning of coded variables.
    data_dictionary = {
        "Region": {"type": "nominal"},
        "HomeOwner": {"type": "nominal"},
        "HighValueCustomer": {"type": "nominal"},
        "TotalAvgMonthlySpend": {"type": "continuous", "unit": "currency/month"},
    }
    assistant = ResearchAssistant(frame)
    print(assistant.summarize(data_dictionary=data_dictionary))

    print("\n" + "=" * 72)
    print("NORMALITY AND OUTLIER DIAGNOSTICS FOR SELECTED CONTINUOUS VARIABLES")
    print("=" * 72)
    profile = assistant.profile(data_dictionary=data_dictionary)
    for column in KEY_VARIABLES:
        tests = profile.get("normality", {}).get(column, {})
        available = next(
            (
                tests[name]
                for name in ("shapiro_wilk", "d_agostino_pearson")
                if tests.get(name, {}).get("available", True) and tests.get(name, {}).get("verdict")
            ),
            None,
        )
        verdict = (
            available["verdict"]
            if available is not None
            else "No usable normality test was available."
        )
        iqr = profile.get("outliers", {}).get(column, {}).get("iqr", {})
        count = iqr.get("count") or 0
        percentage = iqr.get("percentage") or 0.0
        print(f"\n{column}")
        print(f"  Normality: {verdict}")
        print(f"  IQR flag: {count} observations ({percentage:.1f}%)")

    print(
        "\nThese are diagnostics, not instructions to delete observations or change the estimand."
    )

    print("\n" + "=" * 72)
    print("PRIORITIZED INSIGHTS")
    print("=" * 72)
    summary = InsightEngine(profile).get_summary()
    print(
        "Counts:",
        {
            "high": summary["high_severity"],
            "medium": summary["medium_severity"],
            "low": summary["low_severity"],
        },
    )
    for insight in summary["insights"]:
        print(f"\n[{insight['severity'].upper()}] {insight['category']}")
        print(f"  {insight['finding']}")
        for recommendation in insight.get("recommendation", ())[:2]:
            print(f"  - {recommendation}")

    print("\nNext: examples/02_hypothesis_testing.py demonstrates explicit estimands.")


if __name__ == "__main__":
    main()
