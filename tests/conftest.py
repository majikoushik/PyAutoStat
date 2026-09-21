import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def numeric_df():
    """One clean linear column, one column with a single obvious outlier."""
    return pd.DataFrame(
        {
            "linear": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "with_outlier": [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000],
        }
    )


@pytest.fixture
def mixed_df():
    """Numeric + categorical columns, no missing values, no duplicates."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "income": [30000, 32000, 35000, 40000, 42000, 45000, 50000, 52000, 58000, 60000],
            "department": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )


@pytest.fixture
def missing_df():
    """Column 'a' has exactly 1/5 (20%) missing; column 'b' has none."""
    return pd.DataFrame(
        {
            "a": [1, 2, np.nan, 4, 5],
            "b": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def duplicate_df():
    """4 rows, one exact duplicate -> 25% duplicate rows."""
    return pd.DataFrame(
        {
            "a": [1, 2, 2, 3],
            "b": ["x", "y", "y", "z"],
        }
    )


@pytest.fixture
def two_group_normal_df():
    """Two groups drawn from normal distributions -> expect the t-test path."""
    rng = np.random.default_rng(42)
    group_a = rng.normal(loc=50, scale=5, size=40)
    group_b = rng.normal(loc=55, scale=5, size=40)
    return pd.DataFrame(
        {
            "group": ["A"] * 40 + ["B"] * 40,
            "value": np.concatenate([group_a, group_b]),
        }
    )


@pytest.fixture
def two_group_skewed_df():
    """Two groups drawn from exponential distributions -> expect the Mann-Whitney path."""
    rng = np.random.default_rng(7)
    group_a = rng.exponential(scale=2.0, size=40)
    group_b = rng.exponential(scale=4.0, size=40)
    return pd.DataFrame(
        {
            "group": ["A"] * 40 + ["B"] * 40,
            "value": np.concatenate([group_a, group_b]),
        }
    )


@pytest.fixture
def three_group_df():
    rng = np.random.default_rng(3)
    a = rng.normal(50, 5, 30)
    b = rng.normal(55, 5, 30)
    c = rng.normal(60, 5, 30)
    return pd.DataFrame(
        {
            "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
            "value": np.concatenate([a, b, c]),
        }
    )
