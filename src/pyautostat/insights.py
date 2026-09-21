"""
Insight Engine - Generates actionable insights and recommendations
"""

from collections.abc import Mapping

from .exceptions import InvalidDataError


class InsightEngine:
    """Generate actionable insights and recommendations from analysis results"""

    def __init__(self, analysis_results):
        """
        Initialize with analysis results.

        Parameters:
        -----------
        analysis_results : dict
            Output from StatisticalAnalyzer.analyze_all()
        """
        if not isinstance(analysis_results, Mapping):
            raise InvalidDataError("InsightEngine expects analysis results to be a mapping.")
        self.results = analysis_results
        self.insights = []
        self._generated = False

    def generate_insights(self):
        """Generate comprehensive insights"""
        self.insights = []

        self._check_normality()
        self._check_outliers()
        self._check_missing_data()
        self._check_correlations()
        self._check_distributions()
        self._check_data_quality()

        self._generated = True

        return self.insights

    def _check_normality(self):
        """Check and report on normality"""
        if "normality" not in self.results:
            return

        normality = self.results["normality"]
        rejected = []

        for col, tests in normality.items():
            # Check multiple tests
            failed_tests = sum(
                1
                for test, result in tests.items()
                if isinstance(result, dict)
                and (
                    result.get("status") == "rejected"
                    or ("status" not in result and result.get("is_normal") is False)
                )
            )

            if failed_tests >= 2:
                rejected.append(col)

        if rejected:
            self.insights.append(
                {
                    "category": "Normality",
                    "severity": "medium",
                    "finding": f"Multiple normality diagnostics reject for columns {rejected}",
                    "recommendation": [
                        "Review distribution shape, outliers, sample size, and the estimand",
                        "Choose a method that matches the research target and study design",
                    ],
                }
            )

    def _check_outliers(self):
        """Check and report on outliers"""
        if "outliers" not in self.results:
            return

        outliers = self.results["outliers"]
        high_outlier_cols = []

        for col, methods in outliers.items():
            iqr_pct = methods.get("iqr", {}).get("percentage") or 0

            if iqr_pct > 5:
                high_outlier_cols.append(
                    {
                        "column": col,
                        "percentage": iqr_pct,
                        "count": methods.get("iqr", {}).get("count", 0),
                    }
                )

        if high_outlier_cols:
            self.insights.append(
                {
                    "category": "Outliers",
                    "severity": "medium",
                    "finding": (
                        f"High outlier presence detected in {len(high_outlier_cols)} columns"
                    ),
                    "details": high_outlier_cols,
                    "recommendation": [
                        "Review outliers for data entry errors or legitimate extreme values",
                        "Consider robust statistical methods (median, IQR-based)",
                        "Document and potentially exclude confirmed errors",
                        "Use Winsorization to cap extreme values",
                    ],
                }
            )

    def _check_missing_data(self):
        """Check and report on missing data"""
        if "missing_data" not in self.results:
            return

        missing = self.results["missing_data"]
        overall_missing = missing.get("overall_missing_percentage", 0)
        high_missing_cols = []

        for col, info in missing.get("by_column", {}).items():
            if info.get("percentage", 0) > 10:
                high_missing_cols.append(
                    {"column": col, "percentage": info["percentage"], "count": info["count"]}
                )

        if overall_missing > 1 or high_missing_cols:
            recommendations = []

            if overall_missing > 10:
                recommendations.append(
                    "Overall missing data is high (>10%) - consider data collection review"
                )

            if high_missing_cols:
                recommendations.extend(
                    [
                        "Columns with >10% missing: Consider deletion or imputation strategy",
                        "Perform Missing Completely at Random (MCAR) or "
                        "Missing at Random (MAR) analysis",
                        "Consider multiple imputation methods (KNN, EM, MICE)",
                    ]
                )

            self.insights.append(
                {
                    "category": "Missing Data",
                    "severity": "high" if overall_missing > 20 else "medium",
                    "finding": f"Missing data found: {overall_missing:.2f}% overall",
                    "details": high_missing_cols,
                    "recommendation": recommendations,
                }
            )

    def _check_correlations(self):
        """Check and report on correlations"""
        if "correlation" not in self.results:
            return

        corr = self.results["correlation"]
        pearson_matrix = corr.get("pearson", {}).get("matrix", {})

        high_corr_pairs = []

        for col1, correlations in pearson_matrix.items():
            for col2, r_value in correlations.items():
                if col1 < col2 and isinstance(r_value, (int, float)):  # Avoid duplicates
                    if 0.7 < abs(r_value) < 1.0:
                        high_corr_pairs.append({"pair": [col1, col2], "correlation": r_value})

        if high_corr_pairs:
            self.insights.append(
                {
                    "category": "Multicollinearity",
                    "severity": "medium",
                    "finding": (
                        f"High correlations detected between {len(high_corr_pairs)} variable pairs"
                    ),
                    "details": high_corr_pairs[:5],  # Show top 5
                    "recommendation": [
                        "Consider removing one variable from highly correlated pairs for modeling",
                        "Use dimension reduction (PCA) if many correlated variables exist",
                        "Check for variable redundancy",
                        "Document relationships for interpretation",
                    ],
                }
            )

    def _check_distributions(self):
        """Check and report on distributions"""
        if "distributions" not in self.results:
            return

        distributions = self.results["distributions"]
        skewed_cols = []

        for col, info in distributions.items():
            interpretation = info.get("skewness_interpretation", "")
            if "skewed" in interpretation.lower():
                skewed_cols.append(col)

        if skewed_cols:
            self.insights.append(
                {
                    "category": "Distribution Shape",
                    "severity": "low",
                    "finding": f"Skewed distributions detected in {len(skewed_cols)} columns",
                    "columns": skewed_cols,
                    "recommendation": [
                        "Explore log or power transformations",
                        "Consider Box-Cox transformation for optimization",
                        "Use non-parametric methods if parametric assumptions needed",
                        "Document domain knowledge about skewness",
                    ],
                }
            )

    def _check_data_quality(self):
        """Check overall data quality"""
        if "data_quality" not in self.results:
            return

        quality = self.results["data_quality"]
        completeness = quality.get("completeness", 1)
        duplicate_pct = quality.get("duplicate_rows_percentage", 0)

        issues = []

        if completeness < 0.9:
            issues.append("Low completeness (<90%)")

        if duplicate_pct > 1:
            issues.append(f"Significant duplicates ({duplicate_pct:.2f}%)")

        if issues:
            self.insights.append(
                {
                    "category": "Data Quality",
                    "severity": "high" if completeness < 0.8 else "medium",
                    "finding": ", ".join(issues),
                    "recommendation": [
                        "Review data collection and entry processes",
                        "Implement data validation rules",
                        "Remove confirmed duplicates",
                        "Document data quality issues for reporting",
                    ],
                }
            )

    def get_summary(self):
        """Get summary of insights"""
        if not self._generated:
            self.generate_insights()

        summary = {
            "total_insights": len(self.insights),
            "high_severity": sum(1 for i in self.insights if i.get("severity") == "high"),
            "medium_severity": sum(1 for i in self.insights if i.get("severity") == "medium"),
            "low_severity": sum(1 for i in self.insights if i.get("severity") == "low"),
            "insights": self.insights,
        }

        return summary
