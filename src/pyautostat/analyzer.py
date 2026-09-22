"""
Core Statistical Analyzer - Performs comprehensive statistical analysis
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    f_oneway,
    kruskal,
    levene,
    mannwhitneyu,
    normaltest,
    shapiro,
    ttest_ind,
)

from .categorical import categorical_association as _categorical_association
from .detection import detect_column_types, suggest_column_roles
from .exceptions import (
    ColumnNotFoundError,
    InsufficientDataError,
    InsufficientGroupsError,
    InvalidDataError,
    InvalidTestError,
)

_VALID_TEST_TYPES = ("auto", "ttest", "mannwhitney", "anova", "kruskal")


def _finite_or_none(value):
    """Return a plain finite float or None for an undefined numeric result."""
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _advisory_normaltest_warning(item, sample_size):
    """Recognize SciPy's small-sample kurtosis approximation warning only."""
    message = str(item.message).lower()
    return (
        8 <= sample_size < 20
        and issubclass(item.category, UserWarning)
        and (
            message.startswith("kurtosistest only valid for n>=20")
            or message.startswith(
                "`kurtosistest` p-value may be inaccurate with fewer than 20 observations"
            )
        )
    )


class StatisticalAnalyzer:
    """
    Comprehensive statistical analyzer for academic research data.

    Features:
    - Descriptive statistics
    - Distribution analysis (normality tests)
    - Correlation analysis (multiple methods)
    - Hypothesis testing (parametric & non-parametric)
    - Outlier detection (multiple methods)
    - Missing data analysis
    - Data quality metrics
    """

    def __init__(self, df):
        """
        Initialize analyzer with a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for analysis
        """
        if not isinstance(df, pd.DataFrame):
            raise InvalidDataError(
                f"StatisticalAnalyzer expects a pandas DataFrame, got {type(df).__name__}. "
                "Load your data with pandas.read_csv() or wrap it in pandas.DataFrame() first."
            )
        if len(df) == 0:
            raise InvalidDataError(
                "StatisticalAnalyzer received an empty DataFrame (0 rows). "
                "Check your data source or filtering steps before running analysis."
            )
        if len(df.columns) == 0:
            raise InvalidDataError(
                "StatisticalAnalyzer received a DataFrame with 0 columns. "
                "Add at least one column before running analysis."
            )
        if not df.columns.is_unique:
            raise InvalidDataError(
                "DataFrame column names must be unique. Rename duplicate columns before analysis."
            )
        if any(not isinstance(col, str) or not col.strip() for col in df.columns):
            raise InvalidDataError(
                "DataFrame column names must be non-empty strings. Rename columns before analysis."
            )
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                values = series.dropna().to_numpy()
                if np.iscomplexobj(values):
                    raise InvalidDataError(
                        f"Column '{col}' contains complex numbers, which these analyses "
                        "do not support."
                    )
                try:
                    finite = np.isfinite(np.asarray(values, dtype=float))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise InvalidDataError(
                        f"Column '{col}' cannot be represented as finite real numbers."
                    ) from exc
                if not finite.all():
                    raise InvalidDataError(
                        f"Column '{col}' contains infinity or values outside the finite "
                        "float range. "
                        "Replace them with missing values or valid numbers before analysis."
                    )
            else:
                non_missing = series.dropna()
                if not non_missing.map(pd.api.types.is_scalar).all():
                    raise InvalidDataError(
                        f"Column '{col}' contains nested values such as lists or dictionaries. "
                        "Flatten or encode them before analysis."
                    )
                try:
                    non_missing.nunique()
                except TypeError as exc:
                    raise InvalidDataError(
                        f"Column '{col}' contains unhashable values. Encode them before analysis."
                    ) from exc
        self.df = df.copy()
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [
            col
            for col in self.df.columns
            if pd.api.types.is_object_dtype(self.df[col])
            or pd.api.types.is_string_dtype(self.df[col])
            or isinstance(self.df[col].dtype, pd.CategoricalDtype)
        ]
        self.all_results = {}
        self.analysis_warnings = []

    def _add_warning(self, code, section, column, message):
        warning = {"code": code, "section": section, "column": column, "message": message}
        if warning not in self.analysis_warnings:
            self.analysis_warnings.append(warning)

    def analyze_all(self):
        """Run complete analysis suite"""
        self.analysis_warnings = []
        results = {
            "overview": self._overview(),
            "descriptive": self._descriptive_stats(),
            "normality": self._normality_tests(),
            "outliers": self._outlier_detection(),
            "correlation": self._correlation_analysis(),
            "missing_data": self._missing_data_analysis(),
            "data_quality": self._data_quality_metrics(),
            "distributions": self._distribution_analysis(),
            "column_roles": suggest_column_roles(self.df),
            "column_types": detect_column_types(self.df),
            "histograms": self._histogram_analysis(),
        }
        results["analysis_warnings"] = self.analysis_warnings.copy()
        self.all_results = results
        return results

    def _histogram_analysis(self, bins=20):
        """Precompute histogram bins for each numeric column (used by interactive reports)."""
        histograms = {}
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            if col_data.empty:
                continue
            try:
                counts, edges = np.histogram(np.asarray(col_data, dtype=float), bins=bins)
            except (ValueError, OverflowError) as exc:
                self._add_warning(
                    "undefined_result",
                    "histograms",
                    col,
                    f"Histogram could not be calculated: {exc}",
                )
                continue
            histograms[col] = {
                "bin_edges": [float(e) for e in edges],
                "counts": [int(c) for c in counts],
            }
        return histograms

    def _overview(self):
        """Basic dataset overview"""
        return {
            "shape": self.df.shape,
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "numeric_columns": len(self.numeric_cols),
            "categorical_columns": len(self.categorical_cols),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
            "columns": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.to_dict(),
        }

    def _descriptive_stats(self):
        """Comprehensive descriptive statistics"""
        stats_dict = {}

        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            mean = _finite_or_none(col_data.mean())
            std = _finite_or_none(col_data.std())

            stats_dict[col] = {
                "count": len(col_data),
                "mean": mean,
                "median": _finite_or_none(col_data.median()),
                "std": std,
                "variance": _finite_or_none(col_data.var()),
                "min": _finite_or_none(col_data.min()),
                "q1": _finite_or_none(col_data.quantile(0.25)),
                "q3": _finite_or_none(col_data.quantile(0.75)),
                "max": _finite_or_none(col_data.max()),
                "iqr": _finite_or_none(col_data.quantile(0.75) - col_data.quantile(0.25)),
                "skewness": _finite_or_none(col_data.skew()),
                "kurtosis": _finite_or_none(col_data.kurtosis()),
                "coefficient_of_variation": (
                    _finite_or_none(std / mean * 100) if std is not None and mean else None
                ),
            }
            if col_data.empty:
                self._add_warning(
                    "all_missing",
                    "descriptive",
                    col,
                    "No non-missing numeric values are available.",
                )
            elif len(col_data) >= 2 and std is None:
                self._add_warning(
                    "numeric_overflow",
                    "descriptive",
                    col,
                    "Some descriptive statistics are undefined at this numeric scale.",
                )
            elif len(col_data) >= 2 and col_data.nunique() > 1 and std == 0:
                stats_dict[col]["std"] = None
                stats_dict[col]["variance"] = None
                stats_dict[col]["coefficient_of_variation"] = None
                self._add_warning(
                    "numeric_underflow",
                    "descriptive",
                    col,
                    "Variation is too small to represent accurately at this numeric scale.",
                )

        return stats_dict

    def _normality_tests(self):
        """Test for normality using multiple methods"""
        normality_results = {}

        for col in self.numeric_cols:
            col_data = self.df[col].dropna()

            if len(col_data) < 3:
                self._add_warning(
                    "insufficient_data",
                    "normality",
                    col,
                    "Normality tests require at least 3 values.",
                )
                continue
            if col_data.nunique() < 2:
                self._add_warning(
                    "constant_data", "normality", col, "Normality tests require variation."
                )
                continue

            tests = {}

            # Shapiro-Wilk Test (best for sample size < 5000)
            if len(col_data) <= 5000:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    stat, p_value = shapiro(col_data)
                if not caught and np.isfinite(stat) and np.isfinite(p_value):
                    tests["shapiro_wilk"] = {
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "is_normal": bool(p_value > 0.05),  # Legacy screening flag.
                        "status": "not_rejected" if p_value > 0.05 else "rejected",
                        "reference_alpha": 0.05,
                    }
                else:
                    self._add_warning(
                        "undefined_result",
                        "normality",
                        col,
                        "Shapiro-Wilk was unavailable or issued a numerical warning.",
                    )

            if len(col_data) >= 8:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    stat, p_value = normaltest(col_data)
                unreliable_warning = any(
                    not _advisory_normaltest_warning(item, len(col_data)) for item in caught
                )
                if not unreliable_warning and np.isfinite(stat) and np.isfinite(p_value):
                    tests["d_agostino_pearson"] = {
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "is_normal": bool(p_value > 0.05),  # Legacy screening flag.
                        "status": "not_rejected" if p_value > 0.05 else "rejected",
                        "reference_alpha": 0.05,
                    }
                    if len(col_data) < 20:
                        self._add_warning(
                            "small_sample_approximation",
                            "normality",
                            col,
                            "D'Agostino-Pearson's p-value uses a chi-square approximation "
                            "that may be inaccurate with fewer than 20 observations.",
                        )
                else:
                    self._add_warning(
                        "undefined_result",
                        "normality",
                        col,
                        "D'Agostino-Pearson returned a nonfinite result or an "
                        "unrecognized numerical warning; its result is unavailable.",
                    )
            else:
                self._add_warning(
                    "insufficient_data",
                    "normality",
                    col,
                    "D'Agostino-Pearson's test requires at least 8 values.",
                )

            # Anderson-Darling Test
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = stats.anderson(col_data)
                critical_values = np.asarray(getattr(result, "critical_values", []), dtype=float)
                significance_levels = np.asarray(
                    getattr(result, "significance_level", []), dtype=float
                )
            unreliable_warning = any(
                not (
                    issubclass(item.category, FutureWarning)
                    and str(item.message).startswith("As of SciPy 1.17")
                )
                for item in caught
            )
            if (
                not unreliable_warning
                and np.isfinite(result.statistic)
                and result.statistic >= 0
                and critical_values.ndim == 1
                and significance_levels.ndim == 1
                and len(critical_values) == len(significance_levels) > 0
                and np.isfinite(critical_values).all()
                and (critical_values > 0).all()
                and np.isfinite(significance_levels).all()
                and ((significance_levels > 0) & (significance_levels < 100)).all()
                and (np.diff(significance_levels) < 0).all()
                and (np.diff(critical_values) > 0).all()
            ):
                tests["anderson_darling"] = {
                    "statistic": float(result.statistic),
                    "critical_values": [float(x) for x in critical_values],
                    "significance_levels": [float(x) for x in significance_levels],
                }
            else:
                self._add_warning(
                    "undefined_result",
                    "normality",
                    col,
                    "Anderson-Darling returned an unreliable statistic or unusable "
                    "critical-value grid (which can occur with very small samples); "
                    "its result is unavailable.",
                )

            normality_results[col] = tests

        return normality_results

    def _correlation_analysis(self):
        """Multiple correlation analysis methods"""
        if len(self.numeric_cols) < 2:
            return {}

        correlation_data = self.df[self.numeric_cols]

        correlations = {}

        for method, description in (
            ("pearson", "Linear correlation coefficient"),
            ("spearman", "Rank-based correlation"),
            ("kendall", "Ordinal correlation"),
        ):
            matrix = correlation_data.corr(method=method)
            correlations[method] = {
                "matrix": {
                    col: {other: _finite_or_none(value) for other, value in values.items()}
                    for col, values in matrix.to_dict().items()
                },
                "description": description,
            }

        # P-values for Pearson
        from scipy.stats import pearsonr as pearson_test

        p_values = {}
        for col1 in self.numeric_cols:
            p_values[col1] = {}
            for col2 in self.numeric_cols:
                if col1 != col2:
                    valid_data = self.df[[col1, col2]].dropna()
                    if (
                        len(valid_data) > 2
                        and valid_data[col1].nunique() > 1
                        and valid_data[col2].nunique() > 1
                    ):
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            _, p_val = pearson_test(valid_data[col1], valid_data[col2])
                        p_values[col1][col2] = None if caught else _finite_or_none(p_val)
                        if caught:
                            correlations["pearson"]["matrix"][col1][col2] = None
                            correlations["pearson"]["matrix"][col2][col1] = None
                            self._add_warning(
                                "numerical_warning",
                                "correlation",
                                col1,
                                f"Pearson inference for '{col1}' and '{col2}' was unreliable.",
                            )
                    else:
                        p_values[col1][col2] = None

        correlations["p_values"] = p_values

        return correlations

    def _outlier_detection(self):
        """Multiple outlier detection methods"""
        outliers = {}

        for col in self.numeric_cols:
            col_data = self.df[col].dropna()

            methods = {}
            if col_data.empty:
                outliers[col] = {
                    "iqr": {
                        "count": None,
                        "percentage": None,
                        "lower_bound": None,
                        "upper_bound": None,
                    },
                    "z_score": {"count": None, "percentage": None, "threshold": 3.0},
                    "mad": {"count": None, "percentage": None, "threshold": 3.5},
                }
                self._add_warning(
                    "all_missing", "outliers", col, "Outlier detection needs numeric values."
                )
                continue

            # IQR Method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            bounds_valid = np.isfinite(lower_bound) and np.isfinite(upper_bound)
            iqr_outliers = (
                int(((col_data < lower_bound) | (col_data > upper_bound)).sum())
                if bounds_valid
                else None
            )
            if not bounds_valid:
                self._add_warning("numeric_overflow", "outliers", col, "IQR bounds are not finite.")
            methods["iqr"] = {
                "count": iqr_outliers,
                "percentage": float(iqr_outliers / len(col_data) * 100) if bounds_valid else None,
                "lower_bound": _finite_or_none(lower_bound),
                "upper_bound": _finite_or_none(upper_bound),
            }

            # Z-score Method
            z_scores = None
            z_warning = False
            if col_data.nunique() > 1:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    z_scores = np.abs(stats.zscore(col_data))
                z_warning = bool(caught)
            z_valid = not z_warning and z_scores is not None and bool(np.isfinite(z_scores).all())
            z_outliers = int(np.sum(z_scores > 3)) if z_valid else None
            if not z_valid:
                self._add_warning(
                    "undefined_result", "outliers", col, "Z-scores could not be calculated."
                )
            methods["z_score"] = {
                "count": z_outliers,
                "percentage": float(z_outliers / len(col_data) * 100) if z_valid else None,
                "threshold": 3.0,
            }

            # Modified Z-score (using MAD)
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))
            mad_valid = mad != 0 and np.isfinite(mad)
            modified_z = 0.6745 * (col_data - median) / mad if mad_valid else None
            mad_outliers = int((np.abs(modified_z) > 3.5).sum()) if mad_valid else None
            if not mad_valid:
                self._add_warning(
                    "undefined_result",
                    "outliers",
                    col,
                    "MAD is zero or non-finite; modified Z-scores are unavailable.",
                )
            methods["mad"] = {
                "count": mad_outliers,
                "percentage": float(mad_outliers / len(col_data) * 100) if mad_valid else None,
                "threshold": 3.5,
            }

            outliers[col] = methods

        return outliers

    def _missing_data_analysis(self):
        """Comprehensive missing data analysis"""
        missing = {}

        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = missing_count / len(self.df) * 100

            missing[col] = {
                "count": int(missing_count),
                "percentage": float(missing_pct),
                "data_type": str(self.df[col].dtype),
            }

        total_missing = self.df.isnull().sum().sum()

        return {
            "by_column": missing,
            "total_missing_cells": int(total_missing),
            "total_cells": int(self.df.shape[0] * self.df.shape[1]),
            "overall_missing_percentage": float(
                total_missing / (self.df.shape[0] * self.df.shape[1]) * 100
            ),
        }

    def _data_quality_metrics(self):
        """Overall data quality assessment"""
        quality = {}

        # Completeness
        completeness = 1 - (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]))
        quality["completeness"] = float(completeness)

        # Uniqueness
        quality["uniqueness"] = {}
        for col in self.df.columns:
            non_missing = self.df[col].dropna()
            quality["uniqueness"][col] = (
                float(non_missing.nunique() / len(non_missing)) if len(non_missing) else None
            )

        # Duplicate rows
        duplicate_rows = self.df.duplicated().sum()
        quality["duplicate_rows"] = int(duplicate_rows)
        quality["duplicate_rows_percentage"] = float(duplicate_rows / len(self.df) * 100)

        return quality

    def _distribution_analysis(self):
        """Analyze distribution characteristics"""
        distributions = {}

        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            if col_data.empty:
                distributions[col] = {
                    "skewness_interpretation": "Unavailable",
                    "kurtosis_interpretation": "Unavailable",
                    "range": None,
                    "is_bimodal": None,
                }
                continue
            if col_data.nunique() < 2:
                distributions[col] = {
                    "skewness_interpretation": "Unavailable",
                    "kurtosis_interpretation": "Unavailable",
                    "range": 0.0,
                    "is_bimodal": False,
                }
                continue

            try:
                bimodal = self._test_bimodality(col_data) if len(col_data) >= 3 else None
            except (ValueError, OverflowError) as exc:
                bimodal = None
                self._add_warning(
                    "undefined_result",
                    "distributions",
                    col,
                    f"Histogram peak heuristic could not be calculated: {exc}",
                )

            distributions[col] = {
                "skewness_interpretation": self._interpret_skewness(col_data.skew()),
                "kurtosis_interpretation": self._interpret_kurtosis(col_data.kurtosis()),
                "range": _finite_or_none(col_data.max() - col_data.min()),
                "is_bimodal": bimodal,
            }

        return distributions

    @staticmethod
    def _interpret_skewness(skewness):
        """Interpret skewness value"""
        skewness = float(skewness)
        if not np.isfinite(skewness):
            return "Unavailable"
        if abs(skewness) < 0.5:
            return "Approximately symmetric"
        elif skewness > 0.5:
            return "Positively skewed (right-tailed)"
        else:
            return "Negatively skewed (left-tailed)"

    @staticmethod
    def _interpret_kurtosis(kurtosis):
        """Interpret kurtosis value"""
        kurtosis = float(kurtosis)
        if not np.isfinite(kurtosis):
            return "Unavailable"
        if abs(kurtosis) < 0.5:
            return "Mesokurtic (similar to normal)"
        elif kurtosis > 0.5:
            return "Leptokurtic (heavier tails)"
        else:
            return "Platykurtic (lighter tails)"

    @staticmethod
    def _test_bimodality(data):
        """Simple histogram peak heuristic; not a formal bimodality test."""
        hist, _ = np.histogram(data, bins=30)
        peaks = sum(
            1 for i in range(1, len(hist) - 1) if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]
        )
        return peaks >= 2

    def categorical_association(
        self,
        group_col: str,
        outcome_col: str,
        *,
        success_value=None,
        confidence_level: float = 0.95,
        bootstrap_samples: int = 499,
        random_state: int | None = 0,
    ) -> dict:
        """Test categorical independence and report Cramer's V.

        For a 2x2 table, pass success_value to also report Cohen's h for the
        difference in success proportions (first group minus second group).
        """
        return _categorical_association(
            self.df,
            group_col,
            outcome_col,
            success_value=success_value,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            random_state=random_state,
        )

    def hypothesis_tests(
        self,
        group_col: str,
        value_col: str,
        test_type: str = "auto",
        confidence_level: float = 0.95,
        bootstrap_samples: int = 499,
        random_state: int | None = 0,
        *,
        estimand: str | None = None,
        equal_var: bool = False,
    ) -> dict:
        """
        Perform hypothesis tests comparing groups.

        Parameters:
        -----------
        group_col : str
            Column containing group labels
        value_col : str
            Column containing values to compare
        test_type : str
            'auto', 'ttest', 'mannwhitney', 'anova', 'kruskal'
        confidence_level : float
            Interval confidence level in (0, 1), default 0.95.
        bootstrap_samples : int
            Number of within-group bootstrap resamples for effect-size intervals.
            Use 0 to omit these intervals, or at least 100 resamples.
        random_state : int or None
            Local random seed for reproducible bootstrap intervals.
        estimand : str or None
            Required for 'auto': 'mean' or 'distribution'. Never inferred from diagnostics.
        equal_var : bool
            For explicit 'ttest', use Student's pooled-variance test when True;
            otherwise use Welch's unequal-variance test. Defaults to False.
        """
        return self._hypothesis_tests_impl(
            group_col,
            value_col,
            test_type,
            confidence_level,
            bootstrap_samples,
            random_state,
            estimand,
            equal_var,
        )

    def _hypothesis_tests_impl(
        self,
        group_col,
        value_col,
        test_type,
        confidence_level,
        bootstrap_samples,
        random_state,
        estimand,
        equal_var,
    ):
        if not isinstance(group_col, str) or not group_col.strip():
            raise InvalidTestError("group_col must be a non-empty column name string.")
        if not isinstance(value_col, str) or not value_col.strip():
            raise InvalidTestError("value_col must be a non-empty column name string.")
        if not isinstance(test_type, str):
            raise InvalidTestError("test_type must be one of: " + ", ".join(_VALID_TEST_TYPES))
        if test_type not in _VALID_TEST_TYPES:
            raise InvalidTestError(
                f"Unknown test_type '{test_type}'. Choose one of: {', '.join(_VALID_TEST_TYPES)}."
            )
        if estimand not in (None, "mean", "distribution"):
            raise InvalidTestError("estimand must be 'mean', 'distribution', or None.")
        if not isinstance(equal_var, bool):
            raise InvalidTestError("equal_var must be a boolean.")
        if equal_var and test_type != "ttest":
            raise InvalidTestError("equal_var=True applies only to explicit test_type='ttest'.")
        expected_estimand = {
            "ttest": "mean",
            "anova": "mean",
            "mannwhitney": "distribution",
            "kruskal": "distribution",
        }
        if (
            test_type != "auto"
            and estimand is not None
            and estimand != expected_estimand[test_type]
        ):
            raise InvalidTestError(
                f"test_type='{test_type}' does not target estimand='{estimand}'. "
                f"Use estimand='{expected_estimand[test_type]}' or omit estimand "
                "for an explicit test."
            )
        if (
            isinstance(confidence_level, bool)
            or not isinstance(confidence_level, (int, float, np.integer, np.floating))
            or not np.isfinite(confidence_level)
            or not 0 < confidence_level < 1
        ):
            raise InvalidTestError("confidence_level must be a finite number between 0 and 1.")
        if (
            isinstance(bootstrap_samples, bool)
            or not isinstance(bootstrap_samples, (int, np.integer))
            or (bootstrap_samples != 0 and bootstrap_samples < 100)
        ):
            raise InvalidTestError("bootstrap_samples must be 0 or an integer of at least 100.")
        if random_state is not None and (
            isinstance(random_state, bool)
            or not isinstance(random_state, (int, np.integer))
            or random_state < 0
        ):
            raise InvalidTestError("random_state must be a nonnegative integer or None.")
        if group_col not in self.df.columns:
            raise ColumnNotFoundError(
                f"'{group_col}' is not a column in this DataFrame. "
                f"Available columns: {list(self.df.columns)}"
            )
        if value_col not in self.df.columns:
            raise ColumnNotFoundError(
                f"'{value_col}' is not a column in this DataFrame. "
                f"Available columns: {list(self.df.columns)}"
            )

        if group_col == value_col:
            raise InvalidTestError("group_col and value_col must name different columns.")
        if value_col not in self.numeric_cols:
            raise InvalidDataError(
                f"'{value_col}' must be a real numeric column for hypothesis testing. "
                "Convert it with pandas.to_numeric() if appropriate."
            )

        usable = self.df[[group_col, value_col]].dropna()
        groups = usable[group_col].unique()

        if len(groups) < 2:
            raise InsufficientGroupsError(
                f"'{group_col}' has {len(groups)} group(s) with usable '{value_col}' values; "
                "hypothesis_tests() needs at least 2 groups to compare."
            )

        group_data = [
            np.asarray(usable.loc[usable[group_col] == group, value_col], dtype=float)
            for group in groups
        ]
        if any(len(values) < 2 for values in group_data):
            raise InsufficientDataError(
                "Each group needs at least 2 non-missing numeric values. "
                "Check group labels and missing outcome values."
            )
        if test_type == "auto" and estimand is None:
            raise InvalidTestError(
                "test_type='auto' needs estimand='mean' or 'distribution'. "
                "Normality results cannot determine the research target."
            )

        def normality_p(values):
            if len(values) < 3 or np.unique(values).size < 2:
                return None, None
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                if len(values) <= 5000:
                    name, p = "Shapiro-Wilk", _finite_or_none(shapiro(values).pvalue)
                else:
                    name, p = "D'Agostino-Pearson", _finite_or_none(normaltest(values).pvalue)
            return name, None if caught else p

        normality_results = [normality_p(values) for values in group_data]
        normality_p_values = [p for _, p in normality_results]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            p_levene = _finite_or_none(levene(*group_data).pvalue)
        if caught:
            p_levene = None
        assumptions = {
            "normality_p_values": dict(
                zip((str(g) for g in groups), normality_p_values, strict=True)
            ),
            "levene_p_value": p_levene,
            "normality": [
                {
                    "group": group,
                    "sample_size": len(values),
                    "test": test_name,
                    "p_value": p,
                    "status": "unknown"
                    if p is None
                    else ("not_rejected" if p > 0.05 else "rejected"),
                }
                for group, values, (test_name, p) in zip(
                    groups, group_data, normality_results, strict=True
                )
            ],
            "equal_variance_status": (
                "unknown"
                if p_levene is None
                else ("not_rejected" if p_levene > 0.05 else "rejected")
            ),
            "diagnostic_alpha": 0.05,
            "independent_observations": "Required; cannot be verified from numerical values.",
            "warnings": ["Independence must be confirmed from the study design."],
        }
        requested_test_type = test_type
        selection_reason = "Selected explicitly by the caller."

        if len(groups) == 2:
            if test_type in ("anova", "kruskal"):
                raise InvalidTestError(
                    f"'{test_type}' compares 3 or more groups. Use 'ttest' or 'mannwhitney' "
                    "for 2 groups."
                )
            group1, group2 = group_data

            if test_type == "auto":
                test_type = "ttest" if estimand == "mean" else "mannwhitney"
                selection_reason = (
                    "Selected for the caller's stated target quantity; diagnostics did not "
                    "change the estimand."
                )

            if test_type == "ttest":
                if np.var(group1) == 0 and np.var(group2) == 0:
                    raise InsufficientDataError(
                        "A t-test needs numerically representable within-group variation. "
                        "Review constant or extremely small-scale observations."
                    )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    stat, p_val = ttest_ind(group1, group2, equal_var=equal_var)
                if caught:
                    assumptions["warnings"].append(
                        "The t-test backend reported numerical precision loss; verify this result."
                    )
                effect_size = self._cohens_d(group1, group2)
                mean_interval = self._mean_difference_ci(
                    group1, group2, equal_var, confidence_level
                )
                results = {
                    "test": "t-test",
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "groups": [groups[0], groups[1]],
                    "equal_variance": bool(equal_var),
                    "degrees_of_freedom": mean_interval["degrees_of_freedom"],
                    "mean_difference": float(np.mean(group1) - np.mean(group2)),
                    "assumptions": assumptions,
                    "effect_size": {
                        "name": "Cohen's d",
                        "value": effect_size,
                        "interpretation": self._interpret_effect_size("cohens_d", effect_size),
                    },
                    "confidence_interval": mean_interval,
                }
                if equal_var and (p_levene is None or p_levene <= 0.05):
                    assumptions["warnings"].append(
                        "Student's t-test assumes equal population variances; Levene's "
                        "diagnostic rejected this or was unavailable."
                    )
            else:
                _, p_val = mannwhitneyu(group1, group2, alternative="two-sided")
                stat = self._mann_whitney_u1(group1, group2)
                effect_size = self._rank_biserial(stat, len(group1), len(group2))
                results = {
                    "test": "Mann-Whitney U",
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "groups": [groups[0], groups[1]],
                    "assumptions": assumptions,
                    "effect_size": {
                        "name": "rank-biserial correlation",
                        "value": effect_size,
                        "interpretation": None,
                    },
                }
                if min(len(group1), len(group2)) < 10 and (
                    np.unique(np.concatenate(group_data)).size < len(usable)
                ):
                    assumptions["warnings"].append(
                        "Small tied samples use an asymptotic Mann-Whitney p-value; "
                        "the approximation may be unreliable."
                    )

        else:
            if test_type in ("ttest", "mannwhitney"):
                raise InvalidTestError(
                    f"'{test_type}' compares exactly 2 groups. Use 'anova' or 'kruskal' "
                    "for 3 or more groups."
                )
            n_total = sum(len(g) for g in group_data)

            if test_type == "auto":
                if estimand == "mean":
                    raise InvalidTestError(
                        "Automatic multi-group mean comparison is unsupported: Welch ANOVA is not "
                        "available. Choose test_type='anova' only when its assumptions "
                        "are justified."
                    )
                test_type = "kruskal"
                selection_reason = "Selected for the caller's stated distribution comparison."

            if test_type == "anova":
                if all(np.var(values) == 0 for values in group_data):
                    raise InsufficientDataError(
                        "ANOVA needs numerically representable within-group variation. "
                        "Review constant or extremely small-scale observations."
                    )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    stat, p_val = f_oneway(*group_data)
                if caught:
                    assumptions["warnings"].append(
                        "ANOVA reported numerical precision loss; verify this result."
                    )
                effect_size = self._eta_squared_anova(group_data)
                results = {
                    "test": "One-way ANOVA",
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "groups": list(groups),
                    "degrees_of_freedom": [len(groups) - 1, n_total - len(groups)],
                    "assumptions": assumptions,
                    "effect_size": {
                        "name": "eta-squared",
                        "value": effect_size,
                        "interpretation": self._interpret_effect_size("eta_squared", effect_size),
                    },
                }
            else:
                if np.unique(np.concatenate(group_data)).size < 2:
                    raise InsufficientDataError(
                        "Kruskal-Wallis needs at least 2 distinct outcome values. "
                        "All observations are identical."
                    )
                if any(len(values) < 5 for values in group_data):
                    raise InsufficientDataError(
                        "Kruskal-Wallis needs at least 5 usable observations per group "
                        "for its chi-square approximation. Add observations or choose a "
                        "different method."
                    )
                stat, p_val = kruskal(*group_data)
                effect_size = self._epsilon_squared_kruskal(stat, n_total, len(groups))
                results = {
                    "test": "Kruskal-Wallis",
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "groups": list(groups),
                    "degrees_of_freedom": len(groups) - 1,
                    "assumptions": assumptions,
                    "effect_size": {
                        "name": "epsilon-squared (rank)",
                        "value": effect_size,
                        "interpretation": None,
                    },
                }

        if not np.isfinite(stat) or not np.isfinite(p_val):
            raise InsufficientDataError(
                "The selected test returned an undefined result for these groups. "
                "Check sample sizes, constant values, and extreme values."
            )
        effect = results["effect_size"]["value"]
        interval = results.get("confidence_interval")
        if not np.isfinite(effect) or (
            interval is not None
            and (not np.isfinite(interval["lower"]) or not np.isfinite(interval["upper"]))
        ):
            raise InsufficientDataError(
                "The selected test produced an undefined effect size or confidence interval. "
                "Rescale extreme values before comparing groups."
            )
        assumptions["requested_test_type"] = requested_test_type
        assumptions["selected_test_type"] = test_type
        assumptions["selection_reason"] = selection_reason
        assumptions["estimand"] = estimand or expected_estimand[test_type]
        results["sample_size"] = len(usable)
        results["excluded_rows"] = len(self.df) - len(usable)
        results["group_sizes"] = [
            {"group": group, "size": len(values)}
            for group, values in zip(groups, group_data, strict=True)
        ]
        if test_type in ("ttest", "anova"):
            if any(p is None or p <= 0.05 for p in normality_p_values):
                assumptions["warnings"].append(
                    "A normality diagnostic was rejected or unavailable for at least one group; "
                    "review the parametric model."
                )
            if test_type == "anova" and (p_levene is None or p_levene <= 0.05):
                assumptions["warnings"].append(
                    "Levene's test rejected equal variance or was unavailable; standard ANOVA "
                    "may be unsuitable."
                )
        effect_interval, valid_resamples = self._effect_size_ci(
            group_data, test_type, float(confidence_level), int(bootstrap_samples), random_state
        )
        results["effect_size"]["confidence_interval"] = effect_interval
        assumptions["bootstrap"] = {
            "method": "independent within-group percentile bootstrap",
            "requested_resamples": int(bootstrap_samples),
            "valid_resamples": valid_resamples,
            "random_seed": random_state,
        }
        if bootstrap_samples and effect_interval is None:
            assumptions["warnings"].append(
                "Effect-size interval unavailable: fewer than half of requested bootstrap "
                "resamples were valid (minimum 50)."
            )
        return results

    @staticmethod
    def _cohens_d(group1, group2):
        """Cohen's d using the pooled standard deviation."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return float("nan")
        return float((np.mean(group1) - np.mean(group2)) / pooled_std)

    @classmethod
    def _effect_size_ci(cls, group_data, test_type, confidence_level, samples, random_state):
        """Percentile bootstrap interval, resampling independently within each group."""
        if samples == 0:
            return None, 0
        rng = np.random.default_rng(random_state)
        estimates = []
        for _ in range(samples):
            draw = [rng.choice(values, size=len(values), replace=True) for values in group_data]
            if test_type == "ttest":
                estimate = cls._cohens_d(*draw)
            elif test_type == "mannwhitney":
                u_stat = cls._mann_whitney_u1(*draw)
                estimate = cls._rank_biserial(u_stat, len(draw[0]), len(draw[1]))
            elif test_type == "anova":
                estimate = cls._eta_squared_anova(draw)
            else:
                if np.unique(np.concatenate(draw)).size < 2:
                    continue
                h_stat = kruskal(*draw).statistic
                estimate = cls._epsilon_squared_kruskal(h_stat, sum(map(len, draw)), len(draw))
            if np.isfinite(estimate):
                estimates.append(float(estimate))
        if len(estimates) < max(50, samples // 2):
            return None, len(estimates)
        alpha = (1 - confidence_level) / 2
        lower, upper = np.quantile(estimates, [alpha, 1 - alpha])
        if not np.isfinite(lower) or not np.isfinite(upper):
            return None, len(estimates)
        return {
            "level": confidence_level,
            "lower": float(lower),
            "upper": float(upper),
            "method": "independent within-group percentile bootstrap",
            "requested_resamples": samples,
            "random_seed": random_state,
            "valid_resamples": len(estimates),
        }, len(estimates)

    @staticmethod
    def _mean_difference_ci(group1, group2, equal_var, confidence=0.95):
        """Confidence interval for the difference in means (group1 - group2)."""
        n1, n2 = len(group1), len(group2)
        mean_diff = float(np.mean(group1) - np.mean(group2))
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        alpha = 1 - confidence

        if equal_var:
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1 / n1 + 1 / n2))
            df = n1 + n2 - 2
        else:
            se = np.sqrt(var1 / n1 + var2 / n2)
            df = (var1 / n1 + var2 / n2) ** 2 / (
                (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
            )

        margin = stats.t.ppf(1 - alpha / 2, df) * se
        return {
            "level": confidence,
            "lower": float(mean_diff - margin),
            "upper": float(mean_diff + margin),
            "degrees_of_freedom": float(df),
            "description": (
                f"{int(confidence * 100)}% CI for the difference in means (group1 - group2)"
            ),
        }

    @staticmethod
    def _mann_whitney_u1(group1, group2):
        """First-group U from average ranks, stable across supported SciPy versions."""
        n1 = len(group1)
        ranks = stats.rankdata(np.concatenate((group1, group2)))
        return float(np.sum(ranks[:n1]) - n1 * (n1 + 1) / 2)

    @staticmethod
    def _rank_biserial(u_statistic, n1, n2):
        """Rank-biserial correlation, positive when group 1 tends to be larger."""
        return float((2 * u_statistic) / (n1 * n2) - 1)

    @staticmethod
    def _eta_squared_anova(group_data):
        """Eta-squared: proportion of total variance explained by group membership."""
        all_values = np.concatenate(group_data)
        grand_mean = all_values.mean()
        ss_total = float(np.sum((all_values - grand_mean) ** 2))
        if ss_total == 0:
            return float("nan")
        ss_between = float(sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data))
        return ss_between / ss_total

    @staticmethod
    def _epsilon_squared_kruskal(h_statistic, n_total, n_groups):
        """Rank epsilon-squared: non-parametric effect size for Kruskal-Wallis."""
        denominator = n_total - n_groups
        if denominator <= 0:
            return float("nan")
        return float(max(0.0, (h_statistic - n_groups + 1) / denominator))

    @staticmethod
    def _interpret_effect_size(name, value):
        """Label effect-size magnitude using standard small/medium/large benchmarks."""
        magnitude = abs(value)
        bands = {
            "cohens_d": [(0.2, "negligible"), (0.5, "small"), (0.8, "medium")],
            "eta_squared": [(0.01, "negligible"), (0.06, "small"), (0.14, "medium")],
            "correlation": [(0.1, "negligible"), (0.3, "small"), (0.5, "medium")],
        }[name]
        for cutoff, label in bands:
            if magnitude < cutoff:
                return label
        return "large"
