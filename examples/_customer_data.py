"""Authoritative customer dataset loader and domain metadata for PyAutoStat examples.

This module provides a reproducible loader for `CustomerDataset.csv` (5,000 customers,
40 columns), ensuring consistent data types, snake_case identifiers, safe currency
conversion, and structural relationship checks across all public examples.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().with_name("CustomerDataset.csv")

# 1-to-1 mapping from raw CSV column headers to Pythonic snake_case identifiers
COLUMN_RENAME_MAP: dict[str, str] = {
    "CustomerID": "customer_id",
    "Region": "region",
    "Gender": "gender",
    "Age": "age",
    "Education Years": "education_years",
    "Employment Years": "employment_years",
    "Job Category": "job_category",
    "Retired": "retired",
    "Marital Status": "marital_status",
    "Household Size": "household_size",
    "Household Income": "household_income",
    "Number Pets": "number_pets",
    "Home Owner": "home_owner",
    "Car Ownership": "car_ownership",
    "Car Brand": "car_brand",
    "Car Value": "car_value",
    "Commute Distance": "commute_distance",
    "Political Party": "political_party",
    "Votes": "votes",
    "Credit Card": "credit_card",
    "Credit Card Tenure": "credit_card_tenure",
    "Active Lifestyle": "active_lifestyle",
    "TV Watching Hours": "tv_watching_hours",
    "Streaming Svcs": "streaming_services",
    "Wireless Internet": "wireless_internet",
    "Smart Phone": "smart_phone",
    "Twitter Acct": "twitter_account",
    "LinkedIn Acct": "linkedin_account",
    "Facebook Acct": "facebook_account",
    "News Subscriber": "news_subscriber",
    "Coupon Redemption": "coupon_redemption",
    "Brand Tenure Months": "brand_tenure_months",
    "Monthly Spend ProductA": "monthly_spend_product_a",
    "Cumulative Spend ProductA": "cumulative_spend_product_a",
    "Monthly Spend ProductB": "monthly_spend_product_b",
    "Cumulative Spend ProductB": "cumulative_spend_product_b",
    "Monthly Spend ProductC": "monthly_spend_product_c",
    "Cumulative Spend ProductC": "cumulative_spend_product_c",
    "Total Avg Monthly Spend": "total_avg_monthly_spend",
    "High Value Customer": "high_value_customer",
}

# Raw string currency columns formatted with '$' and thousands commas
CURRENCY_COLUMNS: tuple[str, ...] = (
    "household_income",
    "car_value",
    "monthly_spend_product_a",
    "cumulative_spend_product_a",
    "monthly_spend_product_b",
    "cumulative_spend_product_b",
    "monthly_spend_product_c",
    "cumulative_spend_product_c",
    "total_avg_monthly_spend",
)

# Numeric columns that may contain spreadsheet error tokens like '#NULL!'
NUMERIC_ERROR_TOKEN_COLUMNS: tuple[str, ...] = (
    "commute_distance",
    "cumulative_spend_product_a",
)

# Reusable variable metadata dictionary for PyAutoStat profiling and method selection
DATA_DICTIONARY: dict[str, dict[str, str]] = {
    "total_avg_monthly_spend": {"type": "continuous", "unit": "USD/month"},
    "monthly_spend_product_a": {"type": "continuous", "unit": "USD/month"},
    "monthly_spend_product_b": {"type": "continuous", "unit": "USD/month"},
    "monthly_spend_product_c": {"type": "continuous", "unit": "USD/month"},
    "cumulative_spend_product_a": {"type": "continuous", "unit": "USD"},
    "cumulative_spend_product_b": {"type": "continuous", "unit": "USD"},
    "cumulative_spend_product_c": {"type": "continuous", "unit": "USD"},
    "household_income": {"type": "continuous", "unit": "USD/year"},
    "car_value": {"type": "continuous", "unit": "USD"},
    "age": {"type": "continuous", "unit": "years"},
    "education_years": {"type": "continuous", "unit": "years"},
    "employment_years": {"type": "continuous", "unit": "years"},
    "brand_tenure_months": {"type": "continuous", "unit": "months"},
    "credit_card_tenure": {"type": "continuous", "unit": "years"},
    "commute_distance": {"type": "continuous", "unit": "miles"},
    "tv_watching_hours": {"type": "continuous", "unit": "hours/week"},
    "household_size": {"type": "discrete", "unit": "persons"},
    "number_pets": {"type": "discrete", "unit": "pets"},
    "region": {"type": "nominal"},
    "job_category": {"type": "nominal"},
    "gender": {"type": "nominal"},
    "retired": {"type": "nominal"},
    "marital_status": {"type": "nominal"},
    "home_owner": {"type": "nominal"},
    "car_ownership": {"type": "nominal"},
    "car_brand": {"type": "nominal"},
    "credit_card": {"type": "nominal"},
    "active_lifestyle": {"type": "nominal"},
    "streaming_services": {"type": "nominal"},
    "wireless_internet": {"type": "nominal"},
    "smart_phone": {"type": "nominal"},
    "twitter_account": {"type": "nominal"},
    "linkedin_account": {"type": "nominal"},
    "facebook_account": {"type": "nominal"},
    "news_subscriber": {"type": "nominal"},
    "coupon_redemption": {"type": "nominal"},
    "high_value_customer": {"type": "nominal"},
    "political_party": {"type": "nominal"},
    "votes": {"type": "nominal"},
}


def load_customer_data(
    include_customer_id: bool = False,
    labeled_categories: bool = False,
) -> pd.DataFrame:
    """Load and clean the bundled CustomerDataset.csv.

    Args:
        include_customer_id: Whether to retain 'customer_id'. Public examples default
            to False to prevent unintentional disclosure or display of identifiers.
            Set to True only when matching within-unit repeated measurements.
        labeled_categories: When True, creates human-readable category labels for
            binary flags (home_owner -> Owner/Non-owner, high_value_customer ->
            High Value/Standard, coupon_redemption -> Redeemed/Not redeemed).

    Returns:
        A cleaned pandas DataFrame with standardized snake_case column names,
        numeric currency fields, and validated integrity.
    """
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Customer dataset not found at expected path: {DATA_FILE}")

    # Read UTF-8/UTF-8-sig safely to accommodate potential BOM
    raw = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

    if raw.shape != (5000, 40):
        raise ValueError(f"Expected shape (5000, 40), found {raw.shape}")

    # Standardize column names
    frame = raw.rename(columns=COLUMN_RENAME_MAP).copy()

    # Convert currency strings ($ and commas) to numeric floats
    for column in CURRENCY_COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        )

    # Convert columns with potential spreadsheet error tokens ('#NULL!') to numeric
    for column in NUMERIC_ERROR_TOKEN_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Optionally label categorical flags for clear presentation
    if labeled_categories:
        frame["home_owner"] = frame["home_owner"].map({1: "Owner", 0: "Non-owner"})
        frame["high_value_customer"] = frame["high_value_customer"].map(
            {1: "High Value", 0: "Standard"}
        )
        frame["coupon_redemption"] = frame["coupon_redemption"].map(
            {1: "Redeemed", 0: "Not redeemed"}
        )

    if not include_customer_id:
        frame = frame.drop(columns=["customer_id"])

    return frame


def load_product_spend_long(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reshape monthly product spend into long format for within-unit repeated analysis.

    Columns:
        customer_id: Customer identifier for unit matching.
        product: Factor levels ('Product A', 'Product B', 'Product C').
        monthly_spend: Observed monthly spending amount (continuous numeric).
    """
    if frame is None:
        source = load_customer_data(include_customer_id=True)
    else:
        source = frame

    if "customer_id" not in source.columns:
        raise ValueError(
            "Source DataFrame must include 'customer_id' for repeated-measures reshaping."
        )

    spend_cols = ["monthly_spend_product_a", "monthly_spend_product_b", "monthly_spend_product_c"]
    long_df = source.melt(
        id_vars=["customer_id"],
        value_vars=spend_cols,
        var_name="product",
        value_name="monthly_spend",
    )
    long_df["product"] = long_df["product"].map(
        {
            "monthly_spend_product_a": "Product A",
            "monthly_spend_product_b": "Product B",
            "monthly_spend_product_c": "Product C",
        }
    )
    return long_df


def verify_dataset_integrity(frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Verify structural dataset invariants programmatically.

    Verifies:
    1. Total monthly spend is exactly the sum of Product A, B, and C spends.
    2. Product B spend is nonzero exactly when Streaming Svcs == 'Yes'.
    3. Product C spend is nonzero exactly when Wireless Internet == 'Yes'.
    4. High-value customer flag is deterministically separated at spend ~ 275.
    5. Customer IDs are unique across all rows.
    """
    if frame is None:
        data = load_customer_data(include_customer_id=True)
    else:
        data = frame

    spend_sum = (
        data["monthly_spend_product_a"]
        + data["monthly_spend_product_b"]
        + data["monthly_spend_product_c"]
    )
    spend_matches = bool(np.allclose(spend_sum, data["total_avg_monthly_spend"], atol=1e-4))

    streaming_matches = bool(
        ((data["monthly_spend_product_b"] > 0) == (data["streaming_services"] == "Yes")).all()
    )
    wireless_matches = bool(
        ((data["monthly_spend_product_c"] > 0) == (data["wireless_internet"] == "Yes")).all()
    )

    std_max = float(data.loc[data["high_value_customer"] == 0, "total_avg_monthly_spend"].max())
    hvc_min = float(data.loc[data["high_value_customer"] == 1, "total_avg_monthly_spend"].min())
    separation_holds = bool(std_max < hvc_min)

    unique_ids = bool(data["customer_id"].is_unique) if "customer_id" in data.columns else True

    return {
        "rows": len(data),
        "columns": len(data.columns),
        "spend_sum_matches": spend_matches,
        "streaming_structural_match": streaming_matches,
        "wireless_structural_match": wireless_matches,
        "high_value_spend_separation": separation_holds,
        "standard_customer_max_spend": std_max,
        "high_value_customer_min_spend": hvc_min,
        "unique_customer_ids": unique_ids,
        "product_b_zero_fraction": float((data["monthly_spend_product_b"] == 0).mean()),
        "product_c_zero_fraction": float((data["monthly_spend_product_c"] == 0).mean()),
        "high_value_customer_rate": float((data["high_value_customer"] == 1).mean()),
    }


def section(title: str) -> None:
    """Print a major section banner."""
    print("\n" + "=" * 76)
    print(title.upper())
    print("=" * 76)


def subsection(title: str) -> None:
    """Print a subsection divider."""
    print("\n" + "-" * 76)
    print(title)
    print("-" * 76)


def format_number(value: Any, digits: int = 4) -> str:
    """Format a numerical value safely with scientific fallback."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        if abs(value) >= 10000 or (0 < abs(value) < 0.001):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}g}"
    return "unavailable"
