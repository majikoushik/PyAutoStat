import pandas as pd
import pytest

from pyautostat import (
    ColumnNotFoundError,
    InsufficientGroupsError,
    InvalidDataError,
    PyAutoStatError,
    StatisticalAnalyzer,
)


def test_non_dataframe_input_raises_invalid_data_error():
    with pytest.raises(InvalidDataError, match="pandas DataFrame"):
        StatisticalAnalyzer([1, 2, 3])


def test_empty_dataframe_raises_invalid_data_error():
    with pytest.raises(InvalidDataError, match="empty DataFrame"):
        StatisticalAnalyzer(pd.DataFrame())


def test_unknown_group_column_raises_column_not_found_error(mixed_df):
    analyzer = StatisticalAnalyzer(mixed_df)
    with pytest.raises(ColumnNotFoundError, match="not_a_column"):
        analyzer.hypothesis_tests("not_a_column", "income")


def test_unknown_value_column_raises_column_not_found_error(mixed_df):
    analyzer = StatisticalAnalyzer(mixed_df)
    with pytest.raises(ColumnNotFoundError, match="not_a_column"):
        analyzer.hypothesis_tests("department", "not_a_column")


def test_unknown_test_type_raises_pyautostat_error(mixed_df):
    analyzer = StatisticalAnalyzer(mixed_df)
    with pytest.raises(PyAutoStatError, match="Unknown test_type"):
        analyzer.hypothesis_tests("department", "income", test_type="bogus")


def test_single_group_raises_insufficient_groups_error():
    df = pd.DataFrame({"group": ["A"] * 5, "value": [1, 2, 3, 4, 5]})
    analyzer = StatisticalAnalyzer(df)
    with pytest.raises(InsufficientGroupsError, match="at least 2 groups"):
        analyzer.hypothesis_tests("group", "value")


def test_group_column_with_missing_values_ignores_nan_group():
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", None],
            "value": [1, 2, 3, 10, 11, 12, 99],
        }
    )
    result = StatisticalAnalyzer(df).hypothesis_tests("group", "value")
    assert set(result["groups"]) == {"A", "B"}
