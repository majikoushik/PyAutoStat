import pandas as pd
import pytest

from pyautostat import StatisticalAnalyzer, detect_column_types, suggest_column_roles


def test_suggest_column_roles_detects_identifier_and_target():
    df = pd.DataFrame({"user_id": [1, 2, 3], "target": [0, 1, 0], "score": [1.1, 2.2, 3.3]})
    roles = suggest_column_roles(df)

    assert roles["user_id"]["role"] == "identifier"
    assert roles["target"]["role"] == "target"
    assert roles["score"]["role"] == "measurement"
    assert "units" in roles["score"]["suggested_action"]


def test_suggest_column_roles_detects_datetime_and_economic_names():
    df = pd.DataFrame({"created_at": ["x"], "purchase_price": [1]})
    roles = suggest_column_roles(df)

    assert roles["created_at"]["role"] == "datetime"
    assert roles["purchase_price"]["role"] == "economic"


def test_detect_column_types_flags_low_cardinality_integers_as_categorical():
    df = pd.DataFrame({"flag": [0, 1, 0, 1, 1, 0, 1, 0]})
    types = detect_column_types(df)
    assert types["flag"]["detected_type"] == "categorical_numeric"


def test_detect_column_types_treats_wide_range_integers_as_continuous():
    df = pd.DataFrame({"age": list(range(20, 40))})
    types = detect_column_types(df)
    assert types["age"]["detected_type"] == "continuous"


def test_detect_column_types_recognizes_date_like_text():
    df = pd.DataFrame({"signup_date": ["2024-01-01", "2024-02-15", "2024-03-20"]})
    types = detect_column_types(df)
    assert types["signup_date"]["detected_type"] == "datetime_like"


def test_detect_column_types_leaves_free_text_as_text():
    df = pd.DataFrame({"comment": ["great product", "not bad at all", "would buy again"]})
    types = detect_column_types(df)
    assert types["comment"]["detected_type"] == "text"


def test_detect_column_types_handles_empty_column():
    df = pd.DataFrame({"all_missing": [None, None, None]})
    types = detect_column_types(df)
    assert types["all_missing"]["detected_type"] == "empty"


def test_analyze_all_includes_column_intelligence_sections(mixed_df):
    results = StatisticalAnalyzer(mixed_df).analyze_all()
    assert set(results["column_roles"]) == set(mixed_df.columns)
    assert set(results["column_types"]) == set(mixed_df.columns)


def test_detect_column_types_recognizes_common_contact_formats_and_missingness():
    df = pd.DataFrame(
        {
            "email": ["a@example.org", "b@example.org", None],
            "url": ["https://example.org/a", "http://example.net", None],
            "phone": ["+1 212 555 0123", "+1 212 555 0124", None],
            "text": ["a@example.org", "not an email", None],
        }
    )
    types = detect_column_types(df)
    assert [types[name]["detected_type"] for name in ("email", "url", "phone")] == [
        "email",
        "url",
        "phone",
    ]
    assert types["text"]["detected_type"] == "text"
    assert types["email"]["missing_count"] == 1
    assert types["email"]["missing_percentage"] == pytest.approx(100 / 3)


def test_detect_column_types_recognizes_datetime_and_empty_frame():
    df = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", None])})
    assert detect_column_types(df)["when"]["detected_type"] == "datetime"
    empty = detect_column_types(pd.DataFrame({"x": pd.Series(dtype="object")}))
    assert empty["x"]["detected_type"] == "empty"
    assert empty["x"]["missing_percentage"] is None
