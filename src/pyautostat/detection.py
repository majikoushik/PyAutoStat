"""
Column intelligence: smart data-type detection and column-role suggestions.

These are heuristics over column names and values -- they inform
preprocessing decisions but never change what StatisticalAnalyzer
computes or feed back into it automatically.
"""

import re
import warnings

import pandas as pd

from .exceptions import InvalidDataError

_ID_NAME_PATTERN = re.compile(r"(^id$|_id$|^id_|uuid|guid)", re.IGNORECASE)
_DATE_NAME_PATTERN = re.compile(r"(date|timestamp|_dt$|created_at|updated_at)", re.IGNORECASE)
_TARGET_NAME_PATTERN = re.compile(r"^(target|label|outcome|y|class)$", re.IGNORECASE)
_ECONOMIC_NAME_PATTERN = re.compile(r"(price|cost|revenue|salary|income|amount|fee)", re.IGNORECASE)
_MEASURE_NAME_PATTERN = re.compile(r"(^age$|_age$|score|height|weight|duration)", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
_URL_PATTERN = re.compile(r"https?://[^\s/]+(?:/[^\s]*)?", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"\+?[0-9][0-9 ()-]{6,}[0-9]")

_DATE_PARSE_SAMPLE_SIZE = 20
_DATE_PARSE_SUCCESS_THRESHOLD = 0.8
_LOW_CARDINALITY_MAX_UNIQUE = 10


def suggest_column_roles(df: pd.DataFrame) -> dict:
    """Suggest a semantic role for each column based on its name.

    Returns {column: {"role": str, "reason": str}}. These are suggestions
    only; analyze_all() does not act on them.
    """
    if not isinstance(df, pd.DataFrame):
        raise InvalidDataError("suggest_column_roles() expects a pandas DataFrame.")
    if not df.columns.is_unique:
        raise InvalidDataError("Column names must be unique before suggesting roles.")
    roles = {}
    for col in df.columns:
        name = str(col)
        if _ID_NAME_PATTERN.search(name):
            roles[col] = {
                "role": "identifier",
                "reason": (
                    "Name looks like a record identifier; "
                    "consider excluding it from statistical analysis."
                ),
                "suggested_action": "Review whether this identifier should be excluded.",
            }
        elif _TARGET_NAME_PATTERN.search(name):
            roles[col] = {
                "role": "target",
                "reason": "Name matches a common outcome/label naming convention.",
                "suggested_action": "Review this column as a possible outcome variable.",
            }
        elif _DATE_NAME_PATTERN.search(name):
            roles[col] = {
                "role": "datetime",
                "reason": (
                    "Name suggests a date or timestamp; consider parsing with pandas.to_datetime()."
                ),
                "suggested_action": "Parse and validate dates before time-based analysis.",
            }
        elif _ECONOMIC_NAME_PATTERN.search(name):
            roles[col] = {
                "role": "economic",
                "reason": (
                    "Name suggests a monetary value; "
                    "consider currency normalization before comparing across sources."
                ),
                "suggested_action": "Check currency and units before comparison.",
            }
        elif _MEASURE_NAME_PATTERN.search(name):
            roles[col] = {
                "role": "measurement",
                "reason": "Name suggests a quantitative measurement.",
                "suggested_action": "Check units and whether the values are truly continuous.",
            }
        else:
            roles[col] = {"role": "unknown", "reason": "", "suggested_action": ""}
    return roles


def detect_column_types(df: pd.DataFrame) -> dict:
    """Detect the effective data type of each column, beyond its pandas dtype.

    Flags numeric columns that behave like categories (few distinct integer
    values) and text columns that parse cleanly as dates.
    """
    if not isinstance(df, pd.DataFrame):
        raise InvalidDataError("detect_column_types() expects a pandas DataFrame.")
    if not df.columns.is_unique:
        raise InvalidDataError("Column names must be unique before detecting types.")
    types = {}
    for col in df.columns:
        series = df[col].dropna()
        missing_count = int(df[col].isna().sum())
        missing_info = {
            "missing_count": missing_count,
            "missing_percentage": float(missing_count / len(df) * 100) if len(df) else None,
        }

        def result(detected_type: str, notes: str = "", missing_info=missing_info) -> dict:
            return {"detected_type": detected_type, "notes": notes, **missing_info}

        if series.empty:
            types[col] = result("empty", "Column has no non-null values.")
            continue

        if pd.api.types.is_bool_dtype(series):
            types[col] = result("boolean")
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            types[col] = result("datetime")
            continue

        if pd.api.types.is_numeric_dtype(series):
            n_unique = series.nunique()
            is_integer_like = (series == series.round()).all() if series.notna().all() else False
            if is_integer_like and n_unique <= _LOW_CARDINALITY_MAX_UNIQUE:
                types[col] = result(
                    "categorical_numeric",
                    f"Only {n_unique} distinct integer values; "
                    "likely a category or flag, not a continuous measure.",
                )
            else:
                types[col] = result("continuous")
            continue

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            sample = series.head(_DATE_PARSE_SAMPLE_SIZE)
            text_sample = sample.map(str)
            for detected_type, pattern in (
                ("email", _EMAIL_PATTERN),
                ("url", _URL_PATTERN),
                ("phone", _PHONE_PATTERN),
            ):
                matches = text_sample.map(
                    lambda value, pattern=pattern, detected_type=detected_type: (
                        bool(pattern.fullmatch(value.strip()))
                        and (detected_type != "phone" or sum(ch.isdigit() for ch in value) >= 9)
                    )
                )
                if matches.mean() >= _DATE_PARSE_SUCCESS_THRESHOLD:
                    types[col] = result(
                        detected_type,
                        "Most sampled non-missing values match this format; validate before use.",
                    )
                    break
            if col in types:
                continue
            with warnings.catch_warnings():
                # pandas warns when it can't infer one consistent format across
                # the sample; that's expected here since we're only probing.
                warnings.simplefilter("ignore", UserWarning)
                try:
                    parsed = pd.to_datetime(text_sample, errors="coerce")
                except (TypeError, ValueError, OverflowError):
                    types[col] = result("text", "Could not parse as dates.")
                    continue
            if parsed.notna().mean() >= _DATE_PARSE_SUCCESS_THRESHOLD:
                types[col] = result(
                    "datetime_like",
                    "Most sampled values parse as dates; consider pandas.to_datetime().",
                )
            else:
                types[col] = result("text")
            continue

        types[col] = result(str(series.dtype))

    return types
