"""Dataset-only profiling additions shared by both public entry points.

The established analyzer remains the numerical backend. This module validates
optional declarations and assembles descriptive, JSON-friendly profile metadata;
it does not select inferential methods or mutate the source DataFrame.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import pandas as pd

from .detection import detect_column_types, suggest_column_roles
from .exceptions import InvalidDataError

_TYPE_NAMES = {
    "continuous": "continuous_numerical",
    "discrete": "discrete_numerical",
    "nominal": "nominal_categorical",
    "ordinal": "ordinal_categorical",
    "datetime": "datetime",
    "boolean": "boolean",
    "identifier": "identifier",
    "unknown": "unknown",
}
_ROLES = {
    "identifier",
    "target",
    "outcome",
    "predictor",
    "group",
    "measurement",
    "datetime",
    "economic",
    "unknown",
}
_TEXT_FIELDS = {"label", "description", "unit"}
_LIST_FIELDS = {"allowed_values", "missing_codes", "ordinal_order"}
_ENTRY_FIELDS = _TEXT_FIELDS | _LIST_FIELDS | {"type", "role", "valid_range"}
_TOP_CATEGORY_LIMIT = 20
_TOP_PATTERN_LIMIT = 10
_LARGE_DATAFRAME_BYTES = 256 * 1024**2
_VERY_LARGE_DATAFRAME_BYTES = 1024 * 1024**2
_WIDE_CORRELATION_COLUMN_COUNT = 100


def _dataset_resource_info(frame: pd.DataFrame, numeric_column_count: int) -> dict[str, Any]:
    """Estimate profiling resources without changing or sampling the DataFrame."""
    resource_warnings: list[dict[str, Any]] = []
    estimated_bytes: int | None
    try:
        estimated_bytes = int(frame.memory_usage(index=True, deep=True).sum())
        if estimated_bytes < 0:
            raise ValueError("negative memory estimate")
    except Exception:  # Resource advice must never block a valid statistical workflow.
        estimated_bytes = None
        resource_warnings.append(
            {
                "code": "memory_estimate_unavailable",
                "severity": "advisory",
                "message": (
                    "Deep DataFrame memory estimation was unavailable. Profiling continued "
                    "without a memory-size classification."
                ),
                "context": {"estimation_method": "memory_usage(index=True, deep=True)"},
            }
        )

    estimated_mib = estimated_bytes / 1024**2 if estimated_bytes is not None else None
    if estimated_bytes is None:
        resource_level = "unknown"
    elif estimated_bytes >= _VERY_LARGE_DATAFRAME_BYTES:
        resource_level = "very_large"
        resource_warnings.append(
            {
                "code": "very_large_dataframe_profile",
                "severity": "strong_advisory",
                "message": (
                    f"This DataFrame uses approximately {estimated_bytes} bytes "
                    f"({estimated_mib:.2f} MiB). Full profiling may require substantial "
                    "additional temporary memory."
                ),
                "context": {
                    "estimated_memory_bytes": estimated_bytes,
                    "estimated_memory_mib": estimated_mib,
                },
            }
        )
    elif estimated_bytes >= _LARGE_DATAFRAME_BYTES:
        resource_level = "large"
        resource_warnings.append(
            {
                "code": "large_dataframe_profile",
                "severity": "advisory",
                "message": (
                    f"This DataFrame uses approximately {estimated_bytes} bytes "
                    f"({estimated_mib:.2f} MiB). Full profiling may require additional "
                    "temporary memory for distributions, histograms, correlations, and "
                    "intermediate calculations."
                ),
                "context": {
                    "estimated_memory_bytes": estimated_bytes,
                    "estimated_memory_mib": estimated_mib,
                },
            }
        )
    else:
        resource_level = "normal"

    pair_count = numeric_column_count * (numeric_column_count - 1) // 2
    if numeric_column_count >= _WIDE_CORRELATION_COLUMN_COUNT:
        resource_warnings.append(
            {
                "code": "wide_correlation_profile",
                "severity": "advisory",
                "message": (
                    f"The profile includes {numeric_column_count} numerical columns; each "
                    f"all-pairs correlation matrix is {numeric_column_count} x "
                    f"{numeric_column_count} ({pair_count} distinct column pairs) and may be "
                    "computationally expensive."
                ),
                "context": {
                    "numeric_column_count": numeric_column_count,
                    "matrix_dimension": [numeric_column_count, numeric_column_count],
                    "distinct_pair_count": pair_count,
                },
            }
        )

    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "numeric_column_count": int(numeric_column_count),
        "estimated_memory_bytes": estimated_bytes,
        "estimated_memory_mib": estimated_mib,
        "memory_estimation_method": "memory_usage(index=True, deep=True).sum()",
        "resource_level": resource_level,
        "correlation_matrix_dimension": [numeric_column_count, numeric_column_count],
        "correlation_distinct_pair_count": pair_count,
        "resource_warnings": resource_warnings,
        "policy": {
            "large_dataframe_bytes": _LARGE_DATAFRAME_BYTES,
            "very_large_dataframe_bytes": _VERY_LARGE_DATAFRAME_BYTES,
            "wide_correlation_column_count": _WIDE_CORRELATION_COLUMN_COUNT,
            "categories_are_performance_advisories": True,
        },
        "sampling_applied": False,
        "truncation_applied": False,
        "source_data_modified": False,
    }


def _plain_scalar(value: Any) -> bool:
    return type(value) in (str, int, float, bool) and (
        not isinstance(value, float) or math.isfinite(value)
    )


def validate_data_dictionary(frame: pd.DataFrame, data_dictionary: Any) -> dict[str, dict]:
    """Validate and copy a small JSON-compatible column declaration mapping."""
    if data_dictionary is None:
        return {}
    if not isinstance(data_dictionary, Mapping):
        raise InvalidDataError("data_dictionary must map existing column names to metadata.")
    clean: dict[str, dict] = {}
    for column, raw_entry in data_dictionary.items():
        if not isinstance(column, str) or column not in frame.columns:
            raise InvalidDataError(f"data_dictionary references unknown column {column!r}.")
        if not isinstance(raw_entry, Mapping):
            raise InvalidDataError(f"data_dictionary[{column!r}] must be a metadata mapping.")
        unknown = set(raw_entry) - _ENTRY_FIELDS
        if unknown:
            raise InvalidDataError(
                f"data_dictionary[{column!r}] has unsupported fields: {sorted(map(str, unknown))}."
            )
        entry: dict[str, Any] = {}
        for field, value in raw_entry.items():
            if field in _TEXT_FIELDS:
                if not isinstance(value, str) or not value.strip():
                    raise InvalidDataError(f"{column}.{field} must be nonempty text.")
                entry[field] = value
            elif field == "type":
                if not isinstance(value, str) or value not in _TYPE_NAMES:
                    raise InvalidDataError(f"{column}.type must be one of {sorted(_TYPE_NAMES)}.")
                entry[field] = value
            elif field == "role":
                if not isinstance(value, str) or value not in _ROLES:
                    raise InvalidDataError(f"{column}.role must be one of {sorted(_ROLES)}.")
                entry[field] = value
            elif field == "valid_range":
                if (
                    not isinstance(value, (list, tuple))
                    or len(value) != 2
                    or any(type(bound) not in (int, float) for bound in value)
                    or any(not math.isfinite(bound) for bound in value)
                    or value[0] > value[1]
                    or not pd.api.types.is_numeric_dtype(frame[column])
                ):
                    raise InvalidDataError(
                        f"{column}.valid_range needs finite [minimum, maximum] "
                        "for a numeric column."
                    )
                entry[field] = list(value)
            elif field in _LIST_FIELDS:
                if (
                    not isinstance(value, (list, tuple))
                    or not value
                    or any(not _plain_scalar(item) for item in value)
                    or len({(type(item).__name__, item) for item in value}) != len(value)
                ):
                    raise InvalidDataError(
                        f"{column}.{field} needs a nonempty list of distinct JSON scalar values."
                    )
                entry[field] = list(value)
        if "ordinal_order" in entry and entry.get("type") not in (None, "ordinal"):
            raise InvalidDataError(f"{column}.ordinal_order conflicts with its declared type.")
        clean[column] = entry
    return clean


def complete_case_count(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    """Count rows with every requested column observed, without dropping data."""
    if not isinstance(columns, (list, tuple)) or not columns:
        raise InvalidDataError("columns must be a nonempty list of column names.")
    if any(not isinstance(column, str) or column not in frame.columns for column in columns):
        raise InvalidDataError("Every complete-case column must exist in the DataFrame.")
    if len(set(columns)) != len(columns):
        raise InvalidDataError("Complete-case column names must be distinct.")
    available = int(frame[list(columns)].notna().all(axis=1).sum())
    return {
        "columns": list(columns),
        "available_rows": available,
        "excluded_rows": int(len(frame) - available),
        "total_rows": int(len(frame)),
    }


def _display_value(value: Any) -> str:
    """Stable human-readable category label; source values remain in the DataFrame."""
    return str(value)


class DatasetProfiler:
    """Add dataset intelligence to the analyzer's existing calculation engine."""

    def __init__(
        self,
        analyzer: Any,
        data_dictionary: Any,
        histogram_bins: int,
        include_row_positions: bool,
    ) -> None:
        self.analyzer = analyzer
        self.frame = analyzer.df
        if type(histogram_bins) is not int or not 1 <= histogram_bins <= 1000:
            raise InvalidDataError("histogram_bins must be an integer from 1 to 1000.")
        if type(include_row_positions) is not bool:
            raise InvalidDataError("include_row_positions must be true or false.")
        self.histogram_bins = histogram_bins
        self.include_row_positions = include_row_positions
        self.dictionary = validate_data_dictionary(self.frame, data_dictionary)

    def run(self) -> dict:
        type_hints = detect_column_types(self.frame)
        role_hints = suggest_column_roles(self.frame)
        variables = self._variable_intelligence(type_hints, role_hints)
        self.analyzer._profile_numeric_cols = [
            col
            for col in self.analyzer.numeric_cols
            if (
                variables[col]["suggested_type"] in {"continuous_numerical", "discrete_numerical"}
                or (self.frame[col].dropna().empty and col not in self.dictionary)
            )
            and not (
                variables[col]["suggested_role"] == "identifier"
                and variables[col]["role_source"] == "declared"
            )
        ]
        self.analyzer._profile_include_positions = self.include_row_positions
        resource_info = _dataset_resource_info(self.frame, len(self.analyzer._profile_numeric_cols))
        self.analyzer._profile_resource_info = resource_info
        results = self.analyzer._base_profile(self.histogram_bins)
        results["resource_info"] = resource_info
        results["variable_intelligence"] = variables
        results["data_dictionary"] = self.dictionary
        results["categorical_summary"] = self._categorical_summary(variables)
        self._enrich_missingness(results)
        self._enrich_duplicates(results, variables)
        self._enrich_overview(results, variables)
        self._enrich_outliers_and_distributions(results)
        self._enrich_quality(results, variables)
        results["profile_metadata"] = {
            "schema_version": 1,
            "histogram_bins": self.histogram_bins,
            "histogram_binning": "equal_width",
            "top_category_limit": _TOP_CATEGORY_LIMIT,
            "top_missing_pattern_limit": _TOP_PATTERN_LIMIT,
            "missing_codes_applied": False,
            "source_data_modified": False,
            "row_positions_included": self.include_row_positions,
            "correlation_missing_policy": "pairwise_complete",
            "outlier_missing_policy": "per_column_nonmissing",
        }
        self.analyzer.all_results = results
        return results

    def _variable_intelligence(self, type_hints: dict, role_hints: dict) -> dict:
        variables = {}
        for col in self.frame.columns:
            series = self.frame[col]
            nonmissing = series.dropna()
            observed = type_hints[col]["detected_type"]
            role = role_hints[col]["role"]
            evidence = [f"pandas dtype: {series.dtype}", f"legacy hint: {observed}"]
            if nonmissing.empty:
                suggested = "unknown"
            elif pd.api.types.is_bool_dtype(series):
                suggested = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(series):
                suggested = "datetime"
            elif isinstance(series.dtype, pd.CategoricalDtype):
                suggested = "ordinal_categorical" if series.dtype.ordered else "nominal_categorical"
            elif pd.api.types.is_numeric_dtype(series):
                if role == "identifier" and len(nonmissing) >= 3 and nonmissing.is_unique:
                    suggested = "identifier"
                    evidence.append("identifier-like name and all observed values unique")
                elif observed == "categorical_numeric":
                    suggested = "discrete_numerical"
                    evidence.append("few integer values; a category or rating is also possible")
                elif pd.api.types.is_integer_dtype(series):
                    suggested = "discrete_numerical"
                else:
                    suggested = "continuous_numerical"
            elif observed == "datetime_like":
                suggested = "datetime"
                evidence.append("sampled text parses as dates; source is not recoded")
            elif role == "identifier" and len(nonmissing) >= 3 and nonmissing.is_unique:
                suggested = "identifier"
                evidence.append("identifier-like name and all observed values unique")
            else:
                suggested = "nominal_categorical"
            entry = self.dictionary.get(col, {})
            if "type" in entry:
                suggested = _TYPE_NAMES[entry["type"]]
                evidence.append("researcher-declared type")
                if "role" not in entry and role == "identifier" and suggested != "identifier":
                    role = "measurement"
            if "ordinal_order" in entry:
                suggested = "ordinal_categorical"
                evidence.append("researcher-declared ordinal order")
            if "role" in entry:
                role = entry["role"]
                evidence.append("researcher-declared role")
            variables[col] = {
                "observed_dtype": str(series.dtype),
                "suggested_type": suggested,
                "ambiguous_numeric_category": observed == "categorical_numeric"
                and "type" not in entry,
                "suggested_role": role,
                "evidence": evidence,
                "declared": entry.copy(),
                "type_source": "declared"
                if "type" in entry or "ordinal_order" in entry
                else "suggested",
                "role_source": "declared" if "role" in entry else "suggested",
                "warning": (
                    "Analytical type and role are advisory unless declared by the researcher."
                ),
            }
        return variables

    def _categorical_summary(self, variables: dict) -> dict:
        summaries = {}
        for col, info in variables.items():
            is_categorical_dtype = (
                pd.api.types.is_object_dtype(self.frame[col])
                or pd.api.types.is_string_dtype(self.frame[col])
                or isinstance(self.frame[col].dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(self.frame[col])
            )
            if not is_categorical_dtype and info["suggested_type"] not in {
                "nominal_categorical",
                "ordinal_categorical",
                "boolean",
                "identifier",
            }:
                continue
            observed = self.frame[col].dropna()
            counts = observed.value_counts(dropna=True)
            counts = counts[counts > 0]
            top = counts.head(_TOP_CATEGORY_LIMIT)
            modes = counts[counts == counts.iloc[0]] if len(counts) else counts
            summaries[col] = {
                "nonmissing_count": int(len(observed)),
                "missing_count": int(self.frame[col].isna().sum()),
                "observed_categories": int(len(counts)),
                "mode_values": [
                    _display_value(value) for value in modes.index[:_TOP_CATEGORY_LIMIT]
                ],
                "mode_tie_count": int(len(modes)),
                "mode_truncated": len(modes) > _TOP_CATEGORY_LIMIT,
                "frequencies": [
                    {
                        "value": _display_value(value),
                        "count": int(count),
                        "percentage": float(count / len(observed) * 100),
                    }
                    for value, count in top.items()
                ],
                "other_category_count": max(0, int(len(counts) - len(top))),
                "other_observation_count": int(len(observed) - top.sum()),
                "percentage_denominator": "nonmissing observations",
            }
        return summaries

    def _enrich_missingness(self, results: dict) -> None:
        missing = results["missing_data"]
        mask = self.frame.isna()
        row_missing = mask.any(axis=1)
        pattern_counts = mask.value_counts(dropna=False)
        patterns = []
        for pattern, count in pattern_counts.head(_TOP_PATTERN_LIMIT).items():
            bits = pattern if isinstance(pattern, tuple) else (pattern,)
            patterns.append(
                {
                    "missing_columns": [
                        col
                        for col, is_missing in zip(self.frame.columns, bits, strict=True)
                        if is_missing
                    ],
                    "row_count": int(count),
                    "percentage": float(count / len(self.frame) * 100),
                }
            )
        missing.update(
            {
                "rows_with_missing": int(row_missing.sum()),
                "completely_missing_rows": int(mask.all(axis=1).sum()),
                "complete_rows": int((~row_missing).sum()),
                "common_patterns": patterns,
                "patterns_truncated": len(pattern_counts) > _TOP_PATTERN_LIMIT,
                "missing_mechanism": "unknown",
            }
        )
        for col in self.frame.columns:
            missing["by_column"][col]["available_count"] = int((~mask[col]).sum())
            declared_codes = self.dictionary.get(col, {}).get("missing_codes", [])
            missing["by_column"][col]["declared_missing_code_count"] = (
                int(self.frame[col].isin(declared_codes).sum()) if declared_codes else 0
            )
        missing_columns = [col for col in self.frame.columns if mask[col].any()]
        missing["co_missing_pairs"] = [
            {"columns": [left, right], "row_count": int((mask[left] & mask[right]).sum())}
            for left, right in combinations(missing_columns[:20], 2)
            if (mask[left] & mask[right]).any()
        ]
        missing["co_missing_columns_limited"] = len(missing_columns) > 20

    def _enrich_duplicates(self, results: dict, variables: dict) -> None:
        quality = results["data_quality"]
        duplicate_group_rows = self.frame.duplicated(keep=False)
        quality["duplicate_group_rows"] = int(duplicate_group_rows.sum())
        quality["duplicate_definition"] = (
            "Exact equality of all columns; first row retained in duplicate_rows count."
        )
        quality["missing_duplicate_overlap_rows"] = int(
            (duplicate_group_rows & self.frame.isna().any(axis=1)).sum()
        )
        if self.include_row_positions:
            quality["duplicate_group_positions"] = [
                int(position) for position in duplicate_group_rows.to_numpy().nonzero()[0]
            ]
            quality["repeated_row_positions"] = [
                int(position) for position in self.frame.duplicated().to_numpy().nonzero()[0]
            ]
        quality["repeated_identifiers"] = {}
        for col, info in variables.items():
            if info["suggested_role"] == "identifier" or info["suggested_type"] == "identifier":
                nonmissing = self.frame[col].dropna()
                quality["repeated_identifiers"][col] = int(nonmissing.duplicated().sum())

    def _enrich_overview(self, results: dict, variables: dict) -> None:
        overview = results["overview"]
        missing = results["missing_data"]
        quality = results["data_quality"]
        counts = Counter(info["suggested_type"] for info in variables.values())
        overview.update(
            {
                "suggested_type_counts": dict(counts),
                "analytical_numeric_columns": len(self.analyzer._profile_numeric_cols),
                "datetime_columns": counts["datetime"],
                "boolean_columns": counts["boolean"],
                "missing_cells": missing["total_missing_cells"],
                "missing_cell_percentage": missing["overall_missing_percentage"],
                "duplicate_rows": quality["duplicate_rows"],
                "memory_usage_bytes": results["resource_info"]["estimated_memory_bytes"],
            }
        )

    def _enrich_outliers_and_distributions(self, results: dict) -> None:
        definitions = {
            "iqr": ("Outside Q1 - 1.5*IQR to Q3 + 1.5*IQR", 1.5),
            "z_score": ("Absolute sample Z-score greater than 3", 3.0),
            "mad": ("Absolute modified Z-score greater than 3.5", 3.5),
        }
        for col, methods in results["outliers"].items():
            usable = int(self.frame[col].notna().sum())
            for method, detail in methods.items():
                definition, threshold = definitions[method]
                detail.update(
                    {
                        "method": method,
                        "definition": definition,
                        "threshold": threshold,
                        "usable_count": usable,
                        "percentage_denominator": "nonmissing observations",
                        "status": "available" if detail["count"] is not None else "unavailable",
                        "limitation": "A flag is a review cue, not evidence of a data error.",
                    }
                )
        for col, distribution in results["distributions"].items():
            descriptive = results["descriptive"][col]
            distribution.update(
                {
                    "sample_size": descriptive["count"],
                    "skewness": descriptive["skewness"],
                    "kurtosis": descriptive["kurtosis"],
                    "normality_methods": list(results["normality"].get(col, {})),
                    "peak_heuristic": {
                        "method": "30-bin local histogram peaks",
                        "suggests_multiple_peaks": distribution["is_bimodal"],
                        "formal_test": False,
                        "limitation": (
                            "Depends on bins and sample size; it does not prove bimodality."
                        ),
                    },
                }
            )
        for col, histogram in results["histograms"].items():
            histogram.update(
                {
                    "binning_method": "equal_width",
                    "requested_bins": self.histogram_bins,
                    "sample_size": int(self.frame[col].notna().sum()),
                }
            )

    def _enrich_quality(self, results: dict, variables: dict) -> None:
        issues = []

        def add(
            code: str,
            severity: str,
            section: str,
            column: str | None,
            message: str,
            evidence: dict,
            recommendation: str,
        ) -> None:
            issues.append(
                {
                    "code": code,
                    "severity": severity,
                    "section": section,
                    "column": column,
                    "message": message,
                    "evidence": evidence,
                    "recommendation": recommendation,
                }
            )

        for col in self.frame.columns:
            missing = results["missing_data"]["by_column"][col]["count"]
            if missing:
                add(
                    "missing_values",
                    "review",
                    "missing_data",
                    col,
                    f"{missing} rows have a missing value in {col!r}.",
                    {"rows": missing},
                    "Review missingness before choosing an analysis policy.",
                )
            if missing == len(self.frame):
                add(
                    "all_missing",
                    "high",
                    "missing_data",
                    col,
                    f"{col!r} has no observed values.",
                    {"rows": missing},
                    "Review source data and column meaning.",
                )
            entry = self.dictionary.get(col, {})
            if "valid_range" in entry:
                lower, upper = entry["valid_range"]
                values = self.frame[col].dropna()
                violations = int(((values < lower) | (values > upper)).sum())
                if violations:
                    add(
                        "declared_range_violation",
                        "review",
                        "data_dictionary",
                        col,
                        f"{violations} observed values fall outside the declared range.",
                        {"count": violations, "valid_range": [lower, upper]},
                        "Review values and the declared range; no values were changed.",
                    )
            if "allowed_values" in entry:
                observed = self.frame[col].dropna()
                violations = int((~observed.isin(entry["allowed_values"])).sum())
                if violations:
                    add(
                        "undeclared_category",
                        "review",
                        "data_dictionary",
                        col,
                        f"{violations} observed values are outside declared categories.",
                        {"count": violations},
                        "Review values and declared categories.",
                    )
            if "missing_codes" in entry:
                code_count = results["missing_data"]["by_column"][col][
                    "declared_missing_code_count"
                ]
                add(
                    "missing_codes_not_applied",
                    "info",
                    "data_dictionary",
                    col,
                    "Declared missing codes were counted but not converted to missing values.",
                    {"code_count": code_count},
                    "Normalize these codes explicitly before profiling if they should be excluded.",
                )
            if col in self.analyzer.numeric_cols and col not in self.analyzer._profile_numeric_cols:
                identifier = (
                    variables[col]["suggested_type"] == "identifier"
                    or variables[col]["suggested_role"] == "identifier"
                )
                add(
                    "numeric_identifier_excluded" if identifier else "declared_non_numeric_type",
                    "info",
                    "variable_intelligence",
                    col,
                    "Numeric values were excluded from numerical profiling by the "
                    "selected analytical type or identifier role.",
                    {"observed_count": int(self.frame[col].notna().sum())},
                    "Declare a numerical measurement type and role to include this column.",
                )
            nonmissing = self.frame[col].dropna()
            if len(nonmissing) > 1 and nonmissing.nunique() == 1:
                add(
                    "constant_column",
                    "info",
                    "data_quality",
                    col,
                    f"{col!r} has one observed value.",
                    {"observed_count": int(len(nonmissing))},
                    "Do not interpret undefined spread or correlation calculations.",
                )
            if col in results["outliers"]:
                for method, detail in results["outliers"][col].items():
                    if detail["count"]:
                        add(
                            "potential_outliers",
                            "review",
                            "outliers",
                            col,
                            f"{detail['count']} observations were flagged by {method}.",
                            {"method": method, "count": detail["count"]},
                            "Review observations in context; no values were removed.",
                        )
        duplicate_rows = results["data_quality"]["duplicate_rows"]
        if duplicate_rows:
            add(
                "exact_duplicate_rows",
                "review",
                "data_quality",
                None,
                f"{duplicate_rows} rows repeat an earlier complete record.",
                {
                    "repeated_rows": duplicate_rows,
                    "rows_in_duplicate_groups": results["data_quality"]["duplicate_group_rows"],
                },
                "Review whether repeats are expected; no rows were removed.",
            )
        for col, repeated in results["data_quality"]["repeated_identifiers"].items():
            if repeated:
                add(
                    "repeated_identifier",
                    "review",
                    "data_quality",
                    col,
                    f"{repeated} nonmissing identifier values repeat an earlier value.",
                    {"repeated_values": repeated},
                    "Review whether repeated identifiers represent legitimate "
                    "repeated measurements.",
                )
        for warning in results["analysis_warnings"]:
            warning.setdefault("severity", "review")
            warning.setdefault("evidence", {})
            add(
                warning["code"],
                "review",
                warning["section"],
                warning["column"],
                warning["message"],
                warning["evidence"],
                "Inspect method availability before interpreting this result.",
            )
        results["data_quality"]["issues"] = issues


def variable_intelligence_only(frame: pd.DataFrame, data_dictionary: Any = None) -> dict:
    """Reuse variable hints without running full descriptive or inferential profiling."""

    class _FrameHolder:
        def __init__(self, value: pd.DataFrame) -> None:
            self.df = value

    profiler = DatasetProfiler(_FrameHolder(frame), data_dictionary, 20, False)
    return profiler._variable_intelligence(detect_column_types(frame), suggest_column_roles(frame))
