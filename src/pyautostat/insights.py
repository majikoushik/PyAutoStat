"""
Insight Engine - Generates actionable insights and recommendations
"""

from collections.abc import Mapping

from .exceptions import InvalidDataError


def _normalise(
    *,
    category: str,
    severity: str,
    finding: str,
    details: list | None = None,
    recommendation: list[str] | None = None,
) -> dict:
    """Return an insight dict with a guaranteed stable shape.

    Every insight contains these keys so callers can iterate without checking
    for optional fields. A legacy category-specific alias may also be present.

    Keys
    ----
    category       : str    — topic area
    severity       : str    — 'high' | 'medium' | 'low'
    finding        : str    — plain-English description of what was found
    details        : list   — column-level or pair-level raw data (may be [])
    recommendation : list   — actionable plain-English suggestions (may be [])
    """
    return {
        "category": category,
        "severity": severity,
        "finding": finding,
        "details": details if details is not None else [],
        "recommendation": recommendation if recommendation is not None else [],
    }


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
            col_list = ", ".join(f"'{c}'" for c in rejected)
            self.insights.append(
                _normalise(
                    category="Normality",
                    severity="medium",
                    finding=(
                        f"Normality diagnostics rejected for {len(rejected)} column(s): {col_list}."
                    ),
                    details=[{"column": c} for c in rejected],
                    recommendation=[
                        "Review distribution shape, outliers, and sample size for the "
                        "flagged column(s).",
                        "Choose a method that matches the research target and study design; "
                        "rejecting normality does not automatically require a non-parametric test.",
                        "Non-rejection does not prove normality; report the diagnostic alongside "
                        "your chosen test.",
                    ],
                )
            )

    def _check_outliers(self):
        """Check and report on outliers"""
        if "outliers" not in self.results:
            return

        outliers = self.results["outliers"]
        high_outlier_cols = []

        for col, methods in outliers.items():
            iqr_info = methods.get("iqr", {})
            iqr_pct = iqr_info.get("percentage") or 0

            if iqr_pct > 5:
                lower = iqr_info.get("lower_bound")
                upper = iqr_info.get("upper_bound")
                count = iqr_info.get("count", 0) or 0
                fence_text = ""
                if lower is not None and upper is not None:
                    fence_text = f" (IQR fences: {lower:.4g} to {upper:.4g})"
                high_outlier_cols.append(
                    {
                        "column": col,
                        "percentage": iqr_pct,
                        "count": count,
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "note": (
                            f"'{col}': {count} values ({iqr_pct:.1f}%) outside IQR bounds"
                            f"{fence_text}."
                        ),
                    }
                )

        if high_outlier_cols:
            col_summary = "; ".join(d["note"] for d in high_outlier_cols)
            recs = [
                f"'{d['column']}': verify whether values outside "
                f"{d.get('lower_bound', '?'):.4g} to {d.get('upper_bound', '?'):.4g} "
                f"are legitimate measurements or data-entry errors."
                if isinstance(d.get("lower_bound"), (int, float))
                and isinstance(d.get("upper_bound"), (int, float))
                else f"'{d['column']}': review the {d['count']} flagged values."
                for d in high_outlier_cols
            ]
            recs.append(
                "If outliers are confirmed errors, document and exclude them explicitly. "
                "Do not silently remove values without recording the decision."
            )
            self.insights.append(
                _normalise(
                    category="Outliers",
                    severity="medium",
                    finding=(
                        f"High outlier presence (IQR method, >5%) in "
                        f"{len(high_outlier_cols)} column(s): {col_summary}"
                    ),
                    details=high_outlier_cols,
                    recommendation=recs,
                )
            )

    def _check_missing_data(self):
        """Check and report on missing data"""
        if "missing_data" not in self.results:
            return

        missing = self.results["missing_data"]
        overall_missing = missing.get("overall_missing_percentage", 0)
        high_missing_cols = []

        for col, info in missing.get("by_column", {}).items():
            pct = info.get("percentage", 0) or 0
            cnt = info.get("count", 0) or 0
            if pct > 10:
                high_missing_cols.append(
                    {
                        "column": col,
                        "percentage": pct,
                        "count": cnt,
                        "note": f"'{col}': {cnt} values missing ({pct:.1f}%)",
                    }
                )

        if overall_missing > 1 or high_missing_cols:
            recommendations = []

            if overall_missing > 10:
                recommendations.append(
                    f"Overall missing rate is {overall_missing:.1f}%; review data collection."
                )

            for d in high_missing_cols:
                recommendations.append(
                    f"'{d['column']}' has {d['count']} missing values ({d['percentage']:.1f}%). "
                    "Decide whether to exclude rows or use a documented imputation strategy."
                )

            recommendations.append(
                "Record all exclusion or imputation decisions in your analysis plan "
                "to ensure reproducibility."
            )

            self.insights.append(
                _normalise(
                    category="Missing Data",
                    severity="high" if overall_missing > 20 else "medium",
                    finding=(
                        f"Missing data present: {overall_missing:.2f}% overall"
                        + (
                            f"; {len(high_missing_cols)} column(s) exceed 10%."
                            if high_missing_cols
                            else "."
                        )
                    ),
                    details=high_missing_cols,
                    recommendation=recommendations,
                )
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
                        direction = "positive" if r_value > 0 else "negative"
                        high_corr_pairs.append(
                            {
                                "pair": [col1, col2],
                                "correlation": r_value,
                                "direction": direction,
                                "note": (f"'{col1}' <-> '{col2}': r = {r_value:.3f} ({direction})"),
                            }
                        )

        if high_corr_pairs:
            pair_summary = "; ".join(d["note"] for d in high_corr_pairs[:5])
            recs = [
                f"'{d['pair'][0]}' and '{d['pair'][1]}' share a strong {d['direction']} "
                f"correlation (r = {d['correlation']:.3f}). "
                "Verify they are not measuring the same construct."
                for d in high_corr_pairs[:5]
            ]
            recs.append(
                "Correlation indicates co-variation, not causation. "
                "Document the relationship and its interpretation."
            )
            self.insights.append(
                _normalise(
                    category="Multicollinearity",
                    severity="medium",
                    finding=(
                        f"Strong Pearson correlations (|r| > 0.70) detected between "
                        f"{len(high_corr_pairs)} variable pair(s): {pair_summary}"
                    ),
                    details=high_corr_pairs[:5],  # Show top 5
                    recommendation=recs,
                )
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
                skewed_cols.append(
                    {
                        "column": col,
                        "skewness_interpretation": interpretation,
                        "note": f"'{col}': {interpretation.lower()}",
                    }
                )

        if skewed_cols:
            col_summary = "; ".join(d["note"] for d in skewed_cols)
            recs = [
                f"'{d['column']}': the {d['skewness_interpretation'].lower()} shape may "
                "affect model assumptions. Review the chosen method against the stated "
                "estimand, design, sample size, and residual behavior."
                for d in skewed_cols
            ]
            recs.append(
                "Transformations (e.g. log, square root) can reduce skew but change the "
                "estimand. Document the decision and its justification."
            )
            insight = _normalise(
                category="Distribution Shape",
                severity="low",
                finding=(
                    f"Skewed distributions detected in {len(skewed_cols)} column(s): {col_summary}"
                ),
                details=skewed_cols,
                recommendation=recs,
            )
            # Preserve the legacy category-specific key while ``details`` provides
            # the normalized structure for new callers.
            insight["columns"] = [item["column"] for item in skewed_cols]
            self.insights.append(insight)

    def _check_data_quality(self):
        """Check overall data quality"""
        if "data_quality" not in self.results:
            return

        quality = self.results["data_quality"]
        completeness = quality.get("completeness", 1)
        duplicate_pct = quality.get("duplicate_rows_percentage", 0)
        duplicate_n = quality.get("duplicate_rows", 0)

        issues = []
        details = []

        if completeness < 0.9:
            pct = completeness * 100
            issues.append(f"Low completeness ({pct:.1f}%)")
            details.append({"issue": "completeness", "value": pct})

        if duplicate_pct > 1:
            issues.append(f"Significant duplicates ({duplicate_pct:.2f}%)")
            details.append(
                {"issue": "duplicates", "count": duplicate_n, "percentage": duplicate_pct}
            )

        if issues:
            recs = []
            if completeness < 0.9:
                recs.append(
                    f"Completeness is {completeness * 100:.1f}%; review data collection "
                    "and entry processes for systematic gaps."
                )
            if duplicate_pct > 1:
                recs.append(
                    f"{duplicate_n} duplicate rows ({duplicate_pct:.2f}%) detected. "
                    "Confirm they are genuine duplicates before removing them, and document "
                    "the decision."
                )
            recs.append("Implement data validation rules to prevent future quality issues.")
            self.insights.append(
                _normalise(
                    category="Data Quality",
                    severity="high" if completeness < 0.8 else "medium",
                    finding=", ".join(issues) + ".",
                    details=details,
                    recommendation=recs,
                )
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
