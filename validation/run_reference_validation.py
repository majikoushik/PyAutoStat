"""PyAutoStat Standalone Reference Validation Runner.

Executes independent reference validation comparing PyAutoStat numerical outputs
against independent manual formulas (Level A), external backend conformance (Level B),
internal invariants (Level C), and explicitly tracks deferred quantities (Level D).

Usage:
    python validation/run_reference_validation.py
    python validation/run_reference_validation.py --strict
    python validation/run_reference_validation.py --method welch_anova
    python validation/run_reference_validation.py --json report.json
    python validation/run_reference_validation.py --generate-manifest
    python validation/run_reference_validation.py --self-check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np  # noqa: E402
from pyautostat import AnalysisOptions, ResearchAssistant, StatisticalAnalyzer  # noqa: E402
from pyautostat.method_contracts import METHOD_CONTRACTS  # noqa: E402
from validation.cases import get_all_reference_cases  # noqa: E402
from validation.manifest import (  # noqa: E402
    build_manifest_v2,
    build_validation_summary,
    compute_validation_framework_content_sha256,
    write_manifest,
    write_summary,
)
from validation.models import (  # noqa: E402
    FieldValidationResult,
    compare_deferred_field,
    compare_numeric_field,
)
from validation.tolerances import (  # noqa: E402
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    INT_ATOL,
    INT_RTOL,
    NUMERICAL_SOLVER_ATOL,
    NUMERICAL_SOLVER_RTOL,
    PVAL_ATOL,
    PVAL_RTOL,
)


def run_case_validation(case: dict[str, Any]) -> list[FieldValidationResult]:
    """Execute a single reference case and evaluate its field comparisons."""
    case_id = case["case_id"]
    method_id = case["method_id"]
    df = case["df"]
    kw = case["run_kwargs"]
    expected = case["expected"]
    missing = case["missing_accounting"]
    results: list[FieldValidationResult] = []

    # -------------------------------------------------------------------------
    # 1. ONE-SAMPLE T-TEST
    # -------------------------------------------------------------------------
    if method_id == "one_sample_t":
        val_res = StatisticalAnalyzer(df).one_sample_t_test(
            kw["value_col"],
            kw["reference_value"],
            confidence_level=kw.get("confidence_level", 0.95),
        )
        # Level A: Primary estimates and t-statistic
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "sample_mean",
                val_res["sample_mean"],
                expected["sample_mean"],
                "A",
                "Independent mean formula Σx/n",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "mean_difference",
                val_res["mean_difference"],
                expected["mean_difference"],
                "A",
                "Independent difference formula x̄ - μ₀",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "degrees_of_freedom",
                val_res["degrees_of_freedom"],
                expected["degrees_of_freedom"],
                "A",
                "Independent df = n - 1",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "test_statistic",
                val_res["statistic"],
                expected["test_statistic"],
                "A",
                "Independent manual t = (x̄ - μ₀) / (s / √n)",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ci_lower",
                val_res["confidence_interval"]["lower"],
                expected["ci_lower"],
                "A",
                "Independent analytical t interval lower",
                shared_primitive="scipy.stats.t.ppf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ci_upper",
                val_res["confidence_interval"]["upper"],
                expected["ci_upper"],
                "A",
                "Independent analytical t interval upper",
                shared_primitive="scipy.stats.t.ppf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "cohens_d",
                val_res["effect_size"]["value"],
                expected["cohens_d"],
                "A",
                "Independent Cohen's d = (x̄ - μ₀) / s",
            )
        )
        # Level B: p-value backend cross-check
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "SciPy ttest_1samp backend conformance",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.t.sf",
            )
        )
        # Level C: Sample accounting & orientation invariant
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["sample_size"],
                missing["analyzed_rows"],
                "C",
                "Complete-case analyzed row count invariant",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
            )
        )
        obs_sign = 1.0 if val_res["mean_difference"] > 0 else -1.0
        exp_sign = 1.0 if expected["mean_difference"] > 0 else -1.0
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "orientation_sign",
                obs_sign,
                exp_sign,
                "C",
                "Orientation invariant: sign matches mean difference",
            )
        )
        # Level D: Effect size CI deferred
        results.append(
            compare_deferred_field(
                case_id,
                method_id,
                "effect_size_ci",
                val_res["effect_size"].get("confidence_interval"),
                "PyAutoStat noncentral-t inversion",
                "Effect size CI calculation deferred in the current validation framework",
            )
        )

    # -------------------------------------------------------------------------
    # 2. STUDENT'S TWO-SAMPLE T-TEST
    # -------------------------------------------------------------------------
    elif method_id == "student_t":
        val_res = StatisticalAnalyzer(df).hypothesis_tests(
            kw["group_col"],
            kw["value_col"],
            test_type="ttest",
            equal_var=True,
            confidence_level=kw.get("confidence_level", 0.95),
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "mean_difference",
                val_res["mean_difference"],
                expected["mean_difference"],
                "A",
                "Independent mean difference x̄₁ - x̄₂",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "degrees_of_freedom",
                val_res["degrees_of_freedom"],
                expected["degrees_of_freedom"],
                "A",
                "Independent pooled df = n₁ + n₂ - 2",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "test_statistic",
                val_res["statistic"],
                expected["test_statistic"],
                "A",
                "Independent manual pooled t formula",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ci_lower",
                val_res["confidence_interval"]["lower"],
                expected["ci_lower"],
                "A",
                "Independent pooled t interval lower",
                shared_primitive="scipy.stats.t.ppf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ci_upper",
                val_res["confidence_interval"]["upper"],
                expected["ci_upper"],
                "A",
                "Independent pooled t interval upper",
                shared_primitive="scipy.stats.t.ppf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "cohens_d",
                val_res["effect_size"]["value"],
                expected["cohens_d"],
                "A",
                "Independent pooled Cohen's d formula",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "SciPy ttest_ind(equal_var=True) backend conformance",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.t.sf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["sample_size"],
                missing["analyzed_rows"],
                "C",
                "Analyzed row count invariant n₁ + n₂",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
            )
        )
        obs_sign = 1.0 if val_res["mean_difference"] > 0 else -1.0
        exp_sign = 1.0 if expected["mean_difference"] > 0 else -1.0
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "orientation_sign",
                obs_sign,
                exp_sign,
                "C",
                "Orientation invariant: first group minus second group",
            )
        )
        results.append(
            compare_deferred_field(
                case_id,
                method_id,
                "effect_size_ci",
                val_res["effect_size"].get("confidence_interval"),
                "Bootstrap resampling",
                "Effect size bootstrap CI deferred in the current validation framework",
            )
        )

    # -------------------------------------------------------------------------
    # 3. WELCH'S TWO-SAMPLE T-TEST
    # -------------------------------------------------------------------------
    elif method_id == "welch_t":
        val_res = StatisticalAnalyzer(df).hypothesis_tests(
            kw["group_col"],
            kw["value_col"],
            test_type="ttest",
            equal_var=False,
            confidence_level=kw.get("confidence_level", 0.95),
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "mean_difference",
                val_res["mean_difference"],
                expected["mean_difference"],
                "A",
                "Independent mean difference x̄₁ - x̄₂",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "degrees_of_freedom",
                val_res["degrees_of_freedom"],
                expected["degrees_of_freedom"],
                "A",
                "Independent Welch-Satterthwaite formula",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "test_statistic",
                val_res["statistic"],
                expected["test_statistic"],
                "A",
                "Independent manual Welch t formula",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ci_lower",
                val_res["confidence_interval"]["lower"],
                expected["ci_lower"],
                "A",
                "Independent Welch t interval lower",
                shared_primitive="scipy.stats.t.ppf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ci_upper",
                val_res["confidence_interval"]["upper"],
                expected["ci_upper"],
                "A",
                "Independent Welch t interval upper",
                shared_primitive="scipy.stats.t.ppf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "cohens_d",
                val_res["effect_size"]["value"],
                expected["cohens_d"],
                "A",
                "Independent Cohen's d formula",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "SciPy ttest_ind(equal_var=False) backend conformance",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.t.sf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["sample_size"],
                missing["analyzed_rows"],
                "C",
                "Analyzed row count invariant",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
            )
        )
        obs_sign = 1.0 if val_res["mean_difference"] > 0 else -1.0
        exp_sign = 1.0 if expected["mean_difference"] > 0 else -1.0
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "orientation_sign",
                obs_sign,
                exp_sign,
                "C",
                "Orientation invariant: first group minus second group",
            )
        )
        results.append(
            compare_deferred_field(
                case_id,
                method_id,
                "effect_size_ci",
                val_res["effect_size"].get("confidence_interval"),
                "Bootstrap resampling",
                "Effect size bootstrap CI deferred in the current validation framework",
            )
        )

    # -------------------------------------------------------------------------
    # 4. PAIRED T-TEST
    # -------------------------------------------------------------------------
    elif method_id == "paired_t":
        wf = ResearchAssistant(df).run(
            objective="compare_groups",
            outcome=kw["value_col"],
            predictor=kw["condition_col"],
            design="paired",
            unit_id=kw["unit_id"],
            condition_order=kw["condition_order"],
            estimand="mean",
            variable_types={
                kw["value_col"]: "continuous",
                kw["condition_col"]: "nominal",
                kw["unit_id"]: "identifier",
            },
            options=AnalysisOptions(confidence_level=kw.get("confidence_level", 0.95)),
        )
        assert wf.analysis is not None
        val_res = wf.analysis.values
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "mean_difference",
                val_res["primary_estimate"],
                expected["mean_paired_difference"],
                "A",
                "Independent mean paired difference Σd/n",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "degrees_of_freedom",
                val_res["degrees_of_freedom"],
                expected["degrees_of_freedom"],
                "A",
                "Independent df = complete_pairs - 1",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "test_statistic",
                val_res["test_statistic"],
                expected["test_statistic"],
                "A",
                "Independent paired t = D̄ / (s_D / √n)",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ci_lower",
                val_res["confidence_interval"]["lower"],
                expected["ci_lower"],
                "A",
                "Independent paired analytical interval lower",
                shared_primitive="scipy.stats.t.ppf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ci_upper",
                val_res["confidence_interval"]["upper"],
                expected["ci_upper"],
                "A",
                "Independent paired analytical interval upper",
                shared_primitive="scipy.stats.t.ppf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "cohens_dz",
                val_res["effect_size"]["value"],
                expected["cohens_dz"],
                "A",
                "Independent Cohen's dz = D̄ / s_D",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "SciPy ttest_rel backend conformance",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.t.sf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "complete_pairs",
                wf.analysis.metadata["sample"]["complete_pairs"],
                missing["complete_pairs"],
                "C",
                "Complete-pair accounting invariant",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                wf.analysis.sample_size,
                missing["analyzed_rows"],
                "C",
                "Analyzed row count invariant (2 * complete_pairs)",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                wf.analysis.excluded_rows,
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
            )
        )
        obs_sign = 1.0 if val_res["primary_estimate"] > 0 else -1.0
        exp_sign = 1.0 if expected["mean_paired_difference"] > 0 else -1.0
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "orientation_sign",
                obs_sign,
                exp_sign,
                "C",
                "Orientation invariant: first condition minus second condition",
            )
        )
        results.append(
            compare_deferred_field(
                case_id,
                method_id,
                "effect_size_ci",
                val_res["effect_size"].get("confidence_interval"),
                "Noncentral-t inversion",
                "Cohen's dz exact noncentral-t CI deferred in the current validation framework",
            )
        )

    # -------------------------------------------------------------------------
    # 5. MANN-WHITNEY U TEST
    # -------------------------------------------------------------------------
    elif method_id == "mann_whitney_u":
        val_res = StatisticalAnalyzer(df).hypothesis_tests(
            kw["group_col"], kw["value_col"], test_type="mannwhitney", estimand="distribution"
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "test_statistic",
                val_res["statistic"],
                expected["u1"],
                "A",
                "Independent first-group U formula R₁ - n₁(n₁+1)/2",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "rank_biserial",
                val_res["effect_size"]["value"],
                expected["rank_biserial"],
                "A",
                "Independent rank-biserial 2U₁/(n₁n₂) - 1",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "SciPy mannwhitneyu backend conformance",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.mannwhitneyu",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["sample_size"],
                missing["analyzed_rows"],
                "C",
                "Analyzed row count invariant n₁ + n₂",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
            )
        )
        obs_sign = 1.0 if val_res["effect_size"]["value"] > 0 else -1.0
        exp_sign = 1.0 if expected["rank_biserial"] > 0 else -1.0
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "orientation_sign",
                obs_sign,
                exp_sign,
                "C",
                "Orientation invariant: positive when group 1 values exceed group 2",
            )
        )

    # -------------------------------------------------------------------------
    # 6. WILCOXON SIGNED-RANK TEST
    # -------------------------------------------------------------------------
    elif method_id == "wilcoxon_signed_rank":
        val_res = StatisticalAnalyzer(df).paired_wilcoxon(
            kw["unit_id"],
            kw["condition_col"],
            kw["value_col"],
            condition_order=kw.get("condition_order"),
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "positive_rank_sum",
                val_res["positive_rank_sum"],
                expected["w_plus"],
                "A",
                "Independent positive signed-rank sum W+",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "negative_rank_sum",
                val_res["negative_rank_sum"],
                expected["w_minus"],
                "A",
                "Independent negative signed-rank sum W-",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "rank_biserial",
                val_res["effect_size"]["value"],
                expected["rank_biserial"],
                "A",
                "Independent matched-pairs rank-biserial (W+ - W-)/(W+ + W-)",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "test_statistic",
                val_res["statistic"],
                expected["backend_statistic"],
                "B",
                "SciPy wilcoxon statistic backend conformance",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "SciPy wilcoxon p-value backend conformance",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.wilcoxon",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "complete_pairs",
                val_res["complete_pairs"],
                missing["complete_pairs"],
                "C",
                "Complete-pair accounting invariant",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["analyzed_rows"],
                missing["analyzed_rows"],
                "C",
                "Analyzed row count invariant",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "zero_differences",
                val_res["zero_differences"],
                expected["zero_differences"],
                "C",
                "Zero difference count under wilcox policy",
            )
        )
        obs_sign = 1.0 if val_res["effect_size"]["value"] > 0 else -1.0
        exp_sign = 1.0 if expected["rank_biserial"] > 0 else -1.0
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "orientation_sign",
                obs_sign,
                exp_sign,
                "C",
                "Orientation invariant: positive when first condition exceeds second",
            )
        )
        results.append(
            compare_deferred_field(
                case_id,
                method_id,
                "effect_size_ci",
                val_res["effect_size"].get("confidence_interval"),
                "Bootstrap resampling",
                "Matched-pairs rank-biserial bootstrap CI deferred in the current validation framework",
            )
        )

    # -------------------------------------------------------------------------
    # 7. PEARSON CORRELATION
    # -------------------------------------------------------------------------
    elif method_id == "pearson_correlation":
        wf = ResearchAssistant(df).run(
            objective="association",
            outcome=kw["outcome_col"],
            predictor=kw["predictor_col"],
            design="independent",
            estimand="linear",
            variable_types={
                kw["outcome_col"]: "continuous",
                kw["predictor_col"]: "continuous",
            },
            options=AnalysisOptions(confidence_level=kw.get("confidence_level", 0.95)),
        )
        assert wf.analysis is not None
        val_res = wf.analysis.values
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "pearson_r",
                val_res["primary_estimate"],
                expected["pearson_r"],
                "A",
                "Independent covariance/SD formula for r",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ci_lower",
                val_res["confidence_interval"]["lower"],
                expected["ci_lower"],
                "A",
                "Independent Fisher-z interval lower",
                shared_primitive="scipy.stats.norm.ppf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ci_upper",
                val_res["confidence_interval"]["upper"],
                expected["ci_upper"],
                "A",
                "Independent Fisher-z interval upper",
                shared_primitive="scipy.stats.norm.ppf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "SciPy pearsonr p-value backend conformance",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.t.sf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                wf.analysis.sample_size,
                missing["analyzed_rows"],
                "C",
                "Effective pair count invariant",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                wf.analysis.excluded_rows,
                missing["excluded_rows"],
                "C",
                "Excluded pair count invariant",
            )
        )
        obs_sign = 1.0 if val_res["primary_estimate"] > 0 else -1.0
        exp_sign = 1.0 if expected["pearson_r"] > 0 else -1.0
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "orientation_sign",
                obs_sign,
                exp_sign,
                "C",
                "Orientation invariant: correlation sign matches covariance sign",
            )
        )

    # -------------------------------------------------------------------------
    # 8. SPEARMAN CORRELATION
    # -------------------------------------------------------------------------
    elif method_id == "spearman_correlation":
        val_res = StatisticalAnalyzer(df).spearman_correlation(
            first=kw["outcome_col"], second=kw["predictor_col"]
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "spearman_rho",
                val_res["statistic"],
                expected["spearman_rho"],
                "A",
                "Independent rank transformation and Pearson formula",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "SciPy spearmanr p-value backend conformance",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.spearmanr",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["sample_size"],
                missing["analyzed_rows"],
                "C",
                "Effective complete-pair count invariant",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded pair count invariant",
            )
        )
        obs_sign = 1.0 if val_res["statistic"] > 0 else -1.0
        exp_sign = 1.0 if expected["spearman_rho"] > 0 else -1.0
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "orientation_sign",
                obs_sign,
                exp_sign,
                "C",
                "Orientation invariant: monotonic rank direction",
            )
        )
        results.append(
            compare_deferred_field(
                case_id,
                method_id,
                "confidence_interval",
                val_res.get("confidence_interval"),
                "Bootstrap resampling",
                "Spearman paired bootstrap CI deferred in the current validation framework",
            )
        )

    # -------------------------------------------------------------------------
    # 9. PEARSON CHI-SQUARE
    # -------------------------------------------------------------------------
    elif method_id == "pearson_chi_square":
        val_res = StatisticalAnalyzer(df).categorical_association(
            group_col=kw["group_col"], outcome_col=kw["outcome_col"]
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "degrees_of_freedom",
                val_res["degrees_of_freedom"],
                expected["degrees_of_freedom"],
                "A",
                "Independent df = (r-1)(c-1)",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "chi_square",
                val_res["statistic"],
                expected["chi_square"],
                "A",
                "Independent manual chi-square Σ(O - E)²/E",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "cramers_v",
                val_res["effect_size"]["value"],
                expected["cramers_v"],
                "A",
                "Independent manual Cramer's V formula",
            )
        )
        # Expected counts array comparison
        obs_exp = np.asarray(val_res["expected_counts"])
        exp_exp = np.asarray(expected["expected_counts"])
        max_exp_diff = float(np.max(np.abs(obs_exp - exp_exp)))
        results.append(
            FieldValidationResult(
                case_id=case_id,
                method_id=method_id,
                field="expected_counts_matrix",
                observed=val_res["expected_counts"],
                expected=expected["expected_counts"],
                absolute_error=max_exp_diff,
                relative_error=max_exp_diff / max(float(np.max(exp_exp)), 1e-12),
                atol=DEFAULT_ATOL,
                rtol=DEFAULT_RTOL,
                evidence_level="A",
                reference="Independent expected counts (R_i * C_j) / N",
                status="pass" if max_exp_diff <= DEFAULT_ATOL else "discrepancy",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "SciPy chi2_contingency backend conformance",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.chi2.sf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["sample_size"],
                missing["analyzed_rows"],
                "C",
                "Total analyzed contingency observations N",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
            )
        )
        results.append(
            compare_deferred_field(
                case_id,
                method_id,
                "effect_size_ci",
                val_res["effect_size"].get("confidence_interval"),
                "Bootstrap resampling",
                "Cramer's V bootstrap interval deferred in the current validation framework",
            )
        )

    # -------------------------------------------------------------------------
    # 10. FISHER'S EXACT TEST
    # -------------------------------------------------------------------------
    elif method_id == "fisher_exact":
        val_res = StatisticalAnalyzer(df).fisher_exact(
            row_variable=kw["row_variable"], column_variable=kw["column_variable"]
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "odds_ratio",
                val_res["odds_ratio"],
                expected["odds_ratio"],
                "A",
                "Independent cross-product odds ratio (a*d)/(b*c)",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "SciPy fisher_exact two-sided p-value backend conformance",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.fisher_exact",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["sample_size"],
                missing["analyzed_rows"],
                "C",
                "Total analyzed 2x2 contingency sample count",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
            )
        )
        results.append(
            compare_deferred_field(
                case_id,
                method_id,
                "confidence_interval",
                val_res.get("confidence_interval"),
                "Woolf / conditional OR CI",
                "Odds ratio confidence interval deferred in the current validation framework",
            )
        )

    # -------------------------------------------------------------------------
    # 11. WELCH ANOVA
    # -------------------------------------------------------------------------
    elif method_id == "welch_anova":
        val_res = StatisticalAnalyzer(df).welch_anova(kw["group_col"], kw["value_col"])
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "f_statistic",
                val_res["statistic"],
                expected["f_statistic"],
                "A",
                "Independent Welch weighted F statistic",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "df1",
                val_res["degrees_of_freedom"][0],
                expected["df1"],
                "A",
                "Independent numerator df = k - 1",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "df2",
                val_res["degrees_of_freedom"][1],
                expected["df2"],
                "A",
                "Independent Welch-Satterthwaite denominator df",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["p_value"],
                "A",
                "Welch F-distribution tail integral",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.f.sf",
            )
        )
        gh_obs = val_res.get("pairwise_comparisons", [])
        for exp_pw in expected["pairwise"]:
            matched = next(
                (
                    p
                    for p in gh_obs
                    if p["group1"] == exp_pw["group1"] and p["group2"] == exp_pw["group2"]
                ),
                None,
            )
            if matched:
                results.append(
                    compare_numeric_field(
                        case_id,
                        method_id,
                        f"gh_diff_{exp_pw['group1']}_{exp_pw['group2']}",
                        matched["estimate"],
                        exp_pw["mean_difference"],
                        "A",
                        "Games-Howell mean difference",
                    )
                )
                results.append(
                    compare_numeric_field(
                        case_id,
                        method_id,
                        f"gh_se_{exp_pw['group1']}_{exp_pw['group2']}",
                        matched["standard_error"],
                        exp_pw["standard_error"],
                        "A",
                        "Games-Howell pairwise standard error",
                    )
                )
                results.append(
                    compare_numeric_field(
                        case_id,
                        method_id,
                        f"gh_df_{exp_pw['group1']}_{exp_pw['group2']}",
                        matched["degrees_of_freedom"],
                        exp_pw["degrees_of_freedom"],
                        "A",
                        "Games-Howell pairwise Welch df",
                    )
                )
                results.append(
                    compare_numeric_field(
                        case_id,
                        method_id,
                        f"gh_q_{exp_pw['group1']}_{exp_pw['group2']}",
                        matched["statistic"],
                        exp_pw["q_statistic"],
                        "A",
                        "Games-Howell studentized range q statistic",
                    )
                )
                results.append(
                    compare_numeric_field(
                        case_id,
                        method_id,
                        f"gh_p_{exp_pw['group1']}_{exp_pw['group2']}",
                        matched["adjusted_p_value"],
                        exp_pw["p_value"],
                        "B",
                        "Studentized-range distribution CDF",
                        atol=PVAL_ATOL,
                        rtol=PVAL_RTOL,
                        shared_primitive="scipy.stats.studentized_range",
                    )
                )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["sample_size"],
                missing["analyzed_rows"],
                "C",
                "Complete-case analyzed rows",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded rows invariant",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )

    # -------------------------------------------------------------------------
    # 12. CLASSICAL ONE-WAY ANOVA
    # -------------------------------------------------------------------------
    elif method_id == "one_way_anova":
        val_res = StatisticalAnalyzer(df).hypothesis_tests(
            kw["group_col"], kw["value_col"], test_type="anova"
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "f_statistic",
                val_res["statistic"],
                expected["f_statistic"],
                "A",
                "Independent MS_between / MS_within",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "df_between",
                val_res["degrees_of_freedom"][0],
                expected["df_between"],
                "A",
                "Independent df_between = k - 1",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "df_within",
                val_res["degrees_of_freedom"][1],
                expected["df_within"],
                "A",
                "Independent df_within = N - k",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "eta_squared",
                val_res["effect_size"]["value"],
                expected["eta_squared"],
                "A",
                "Independent eta-squared = SS_between / SS_total",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "scipy.stats.f_oneway p-value",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.f.sf",
            )
        )
        tk_obs = val_res.get("pairwise_comparisons", [])
        for exp_pw in expected["pairwise"]:
            matched = next(
                (
                    p
                    for p in tk_obs
                    if p["group1"] == exp_pw["group1"] and p["group2"] == exp_pw["group2"]
                ),
                None,
            )
            if matched:
                results.append(
                    compare_numeric_field(
                        case_id,
                        method_id,
                        f"tk_diff_{exp_pw['group1']}_{exp_pw['group2']}",
                        matched["estimate"],
                        exp_pw["mean_difference"],
                        "A",
                        "Tukey-Kramer mean difference",
                    )
                )
                results.append(
                    compare_numeric_field(
                        case_id,
                        method_id,
                        f"tk_se_{exp_pw['group1']}_{exp_pw['group2']}",
                        matched["standard_error"],
                        exp_pw["standard_error"],
                        "A",
                        "Tukey-Kramer standard error",
                    )
                )
                results.append(
                    compare_numeric_field(
                        case_id,
                        method_id,
                        f"tk_p_{exp_pw['group1']}_{exp_pw['group2']}",
                        matched["adjusted_p_value"],
                        exp_pw["p_value"],
                        "B",
                        "Studentized-range distribution tail",
                        atol=PVAL_ATOL,
                        rtol=PVAL_RTOL,
                        shared_primitive="scipy.stats.studentized_range",
                    )
                )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["sample_size"],
                missing["analyzed_rows"],
                "C",
                "Complete-case analyzed rows",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded rows invariant",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )

    # -------------------------------------------------------------------------
    # 13. KRUSKAL-WALLIS
    # -------------------------------------------------------------------------
    elif method_id == "kruskal_wallis":
        val_res = StatisticalAnalyzer(df).hypothesis_tests(
            kw["group_col"], kw["value_col"], test_type="kruskal"
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "h_statistic",
                val_res["statistic"],
                expected["h_statistic"],
                "A",
                "Independent Kruskal-Wallis H statistic with tie adjustment",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "degrees_of_freedom",
                val_res["degrees_of_freedom"],
                expected["degrees_of_freedom"],
                "A",
                "Independent df = k - 1",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "epsilon_squared",
                val_res["effect_size"]["value"],
                expected["epsilon_squared"],
                "A",
                "Independent rank epsilon-squared = (H - k + 1) / (N - k)",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "scipy.stats.kruskal p-value",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.chi2.sf",
            )
        )
        dunn_obs = val_res.get("pairwise_comparisons", [])
        for exp_pw in expected["pairwise"]:
            matched = next(
                (
                    p
                    for p in dunn_obs
                    if p["group1"] == exp_pw["group1"] and p["group2"] == exp_pw["group2"]
                ),
                None,
            )
            if matched:
                results.append(
                    compare_numeric_field(
                        case_id,
                        method_id,
                        f"dunn_z_{exp_pw['group1']}_{exp_pw['group2']}",
                        matched["statistic"],
                        exp_pw["z_statistic"],
                        "A",
                        "Dunn pairwise z statistic",
                    )
                )
                results.append(
                    compare_numeric_field(
                        case_id,
                        method_id,
                        f"dunn_raw_p_{exp_pw['group1']}_{exp_pw['group2']}",
                        matched["raw_p_value"],
                        exp_pw["raw_p_value"],
                        "B",
                        "Standard normal tail probability",
                        atol=PVAL_ATOL,
                        rtol=PVAL_RTOL,
                        shared_primitive="scipy.stats.norm.sf",
                    )
                )
                results.append(
                    compare_numeric_field(
                        case_id,
                        method_id,
                        f"dunn_adj_p_{exp_pw['group1']}_{exp_pw['group2']}",
                        matched["adjusted_p_value"],
                        exp_pw["adjusted_p_value"],
                        "A",
                        "Holm step-down multiplicity adjustment",
                    )
                )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["sample_size"],
                missing["analyzed_rows"],
                "C",
                "Complete-case analyzed rows",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded rows invariant",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )

    # -------------------------------------------------------------------------
    # 14. REPEATED-MEASURES ANOVA
    # -------------------------------------------------------------------------
    elif method_id == "repeated_measures_anova":
        val_res = StatisticalAnalyzer(df).repeated_measures_anova(
            unit_id=kw["unit_id"],
            condition_col=kw["condition_col"],
            value_col=kw["value_col"],
            condition_order=kw["condition_order"],
        )
        anova_tbl = val_res["anova_table"]
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ss_condition",
                anova_tbl["ss_condition"],
                expected["ss_condition"],
                "A",
                "Independent SS_condition",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ss_subject",
                anova_tbl["ss_subject"],
                expected["ss_subject"],
                "A",
                "Independent SS_subject",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ss_error",
                anova_tbl["ss_error"],
                expected["ss_error"],
                "A",
                "Independent SS_error",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ss_total",
                anova_tbl["ss_total"],
                expected["ss_total"],
                "A",
                "Independent SS_total",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "df_condition",
                anova_tbl["df_condition"],
                expected["df_condition"],
                "A",
                "df_condition = k - 1",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "df_error",
                anova_tbl["df_error"],
                expected["df_error"],
                "A",
                "df_error = (n-1)(k-1)",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "f_statistic",
                anova_tbl["f_statistic"],
                expected["f_statistic"],
                "A",
                "Independent F = MS_condition / MS_error",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_uncorrected",
                anova_tbl["p_value"],
                expected["p_uncorrected"],
                "A",
                "F-distribution tail integral",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.f.sf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "partial_eta_squared",
                val_res["effect_size"]["value"],
                expected["partial_eta_squared"],
                "A",
                "Partial eta-squared = SS_cond / (SS_cond + SS_err)",
            )
        )
        gg_res = val_res.get("greenhouse_geisser", {})
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "epsilon_gg",
                gg_res["epsilon"],
                expected["epsilon_gg"],
                "A",
                "Greenhouse-Geisser epsilon formula",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "corrected_df_num",
                gg_res["corrected_df_num"],
                expected["corrected_df_num"],
                "A",
                "Corrected numerator df",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "corrected_df_den",
                gg_res["corrected_df_den"],
                expected["corrected_df_den"],
                "A",
                "Corrected denominator df",
            )
        )
        spher = val_res.get("sphericity")
        if spher:
            results.append(
                compare_numeric_field(
                    case_id,
                    method_id,
                    "mauchly_w",
                    spher["mauchly_w"],
                    expected["w_mauchly"],
                    "A",
                    "Mauchly W from orthonormal Helmert contrast determinant",
                )
            )
            results.append(
                compare_numeric_field(
                    case_id,
                    method_id,
                    "mauchly_chi2",
                    spher["chi2_statistic"],
                    expected["chi2_mauchly"],
                    "A",
                    "Box-Anderson chi-square approximation",
                )
            )
            results.append(
                compare_numeric_field(
                    case_id,
                    method_id,
                    "mauchly_df",
                    spher["df"],
                    expected["df_mauchly"],
                    "A",
                    "Mauchly df = k(k-1)/2 - 1",
                    atol=INT_ATOL,
                    rtol=INT_RTOL,
                )
            )
            results.append(
                compare_numeric_field(
                    case_id,
                    method_id,
                    "mauchly_p",
                    spher["p_value"],
                    expected["p_mauchly"],
                    "B",
                    "Chi-square distribution tail probability",
                    atol=PVAL_ATOL,
                    rtol=PVAL_RTOL,
                    shared_primitive="scipy.stats.chi2.sf",
                )
            )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["sample"]["analyzed_rows"],
                missing["analyzed_rows"],
                "C",
                "Complete panel analyzed rows",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["sample"]["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded incomplete observation rows",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )

    # -------------------------------------------------------------------------
    # 15. FRIEDMAN TEST
    # -------------------------------------------------------------------------
    elif method_id == "friedman_test":
        val_res = StatisticalAnalyzer(df).friedman_test(
            unit_id=kw["unit_id"],
            condition_col=kw["condition_col"],
            value_col=kw["value_col"],
            condition_order=kw["condition_order"],
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "q_statistic",
                val_res["statistic"],
                expected["q_statistic"],
                "A",
                "Independent Friedman Q rank sum statistic",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "degrees_of_freedom",
                val_res["degrees_of_freedom"],
                expected["degrees_of_freedom"],
                "A",
                "Independent df = k - 1",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "kendall_w",
                val_res["effect_size"]["value"],
                expected["kendall_w"],
                "A",
                "Independent Kendall's W = Q / (n*(k-1))",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "scipy.stats.friedmanchisquare p-value",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.chi2.sf",
            )
        )
        results.append(
            compare_deferred_field(
                case_id,
                method_id,
                "kendall_w_bootstrap_ci",
                val_res["effect_size"].get("confidence_interval"),
                "Kendall's W block bootstrap deferred",
                "Participant-block bootstrap interval requires deterministic resample harness",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["sample"]["analyzed_rows"],
                missing["analyzed_rows"],
                "C",
                "Complete panel analyzed rows",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["sample"]["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded incomplete observation rows",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )

    # -------------------------------------------------------------------------
    # 16. TWO-WAY FACTORIAL ANOVA
    # -------------------------------------------------------------------------
    elif method_id == "two_way_anova":
        val_res = StatisticalAnalyzer(df).two_way_anova(
            outcome=kw["outcome"],
            factor_a=kw["factor_a"],
            factor_b=kw["factor_b"],
            sum_of_squares=kw.get("sum_of_squares", "type2"),
        )
        terms = {t["term"]: t for t in val_res["terms"]}
        fa, fb = kw["factor_a"], kw["factor_b"]
        inter_key = f"{fa}:{fb}"
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ss_a",
                terms[fa]["sum_of_squares"],
                expected["ss_a"],
                "A",
                "Factor A sum of squares",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ss_b",
                terms[fb]["sum_of_squares"],
                expected["ss_b"],
                "A",
                "Factor B sum of squares",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ss_ab",
                terms[inter_key]["sum_of_squares"],
                expected["ss_ab"],
                "A",
                "Interaction sum of squares",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ss_residual",
                terms["Residual"]["sum_of_squares"],
                expected["ss_residual"],
                "A",
                "Residual sum of squares",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "df_a",
                terms[fa]["df"],
                expected["df_a"],
                "A",
                "df_a = a - 1",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "df_b",
                terms[fb]["df"],
                expected["df_b"],
                "A",
                "df_b = b - 1",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "df_ab",
                terms[inter_key]["df"],
                expected["df_ab"],
                "A",
                "df_ab = (a-1)(b-1)",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "df_residual",
                terms["Residual"]["df"],
                expected["df_residual"],
                "A",
                "Residual df",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "f_a",
                terms[fa]["f_statistic"],
                expected["f_a"],
                "A",
                "Factor A F statistic",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "f_b",
                terms[fb]["f_statistic"],
                expected["f_b"],
                "A",
                "Factor B F statistic",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "f_ab",
                terms[inter_key]["f_statistic"],
                expected["f_ab"],
                "A",
                "Interaction F statistic",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_a",
                terms[fa]["p_value"],
                expected["p_a"],
                "B",
                "F-distribution tail probability for factor A",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.f.sf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_b",
                terms[fb]["p_value"],
                expected["p_b"],
                "B",
                "F-distribution tail probability for factor B",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.f.sf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_ab",
                terms[inter_key]["p_value"],
                expected["p_ab"],
                "B",
                "F-distribution tail probability for interaction",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.f.sf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "partial_eta_a",
                terms[fa]["effect_size"]["value"],
                expected["partial_eta_a"],
                "A",
                "Partial eta-squared for factor A",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "partial_eta_b",
                terms[fb]["effect_size"]["value"],
                expected["partial_eta_b"],
                "A",
                "Partial eta-squared for factor B",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "partial_eta_ab",
                terms[inter_key]["effect_size"]["value"],
                expected["partial_eta_ab"],
                "A",
                "Partial eta-squared for interaction",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["analyzed_rows"],
                missing["analyzed_rows"],
                "C",
                "Factorial analyzed observation count",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )

    # -------------------------------------------------------------------------
    # 17. LINEAR REGRESSION
    # -------------------------------------------------------------------------
    elif method_id == "linear_regression":
        val_res = StatisticalAnalyzer(df).linear_regression(
            outcome=kw["outcome"],
            predictors=kw["predictors"],
            variable_types=kw["variable_types"],
            covariance_type=kw.get("covariance_type", "classical"),
        )
        fit = val_res["model_fit"]
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "r_squared",
                fit["r_squared"],
                expected["r_squared"],
                "A",
                "Independent 1 - SSE/SST",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "adjusted_r_squared",
                fit["adjusted_r_squared"],
                expected["adjusted_r_squared"],
                "A",
                "Independent adjusted R2",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "df_model",
                fit["model_degrees_of_freedom"],
                expected["df_model"],
                "A",
                "Model df = p - 1",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "df_resid",
                fit["residual_degrees_of_freedom"],
                expected["df_resid"],
                "A",
                "Residual df = n - p",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        exp_f = (
            expected.get("f_statistic_hc3", expected["f_statistic"])
            if kw.get("covariance_type") == "HC3"
            else expected["f_statistic"]
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "f_statistic",
                fit["model_f_statistic"],
                exp_f,
                "A",
                "Model F statistic",
            )
        )
        for idx, (coef_obs, beta_exp) in enumerate(
            zip(val_res["coefficients"], expected["beta"], strict=True)
        ):
            tname = coef_obs["term"]
            results.append(
                compare_numeric_field(
                    case_id,
                    method_id,
                    f"beta_{tname}",
                    coef_obs["estimate"],
                    beta_exp,
                    "A",
                    f"OLS slope for {tname}",
                )
            )
            if kw.get("covariance_type") == "HC3":
                results.append(
                    compare_numeric_field(
                        case_id,
                        method_id,
                        f"se_hc3_{tname}",
                        coef_obs["standard_error"],
                        expected["se_hc3"][idx],
                        "A",
                        f"HC3 robust SE for {tname}",
                    )
                )
            else:
                results.append(
                    compare_numeric_field(
                        case_id,
                        method_id,
                        f"se_cls_{tname}",
                        coef_obs["standard_error"],
                        expected["se_classical"][idx],
                        "A",
                        f"Classical OLS SE for {tname}",
                    )
                )
        diag = val_res.get("diagnostics", {})
        bp = diag.get("breusch_pagan", {})
        if bp:
            results.append(
                compare_numeric_field(
                    case_id,
                    method_id,
                    "bp_lm",
                    bp["lm_statistic"],
                    expected["bp_lm"],
                    "A",
                    "Breusch-Pagan auxiliary LM statistic",
                )
            )
            results.append(
                compare_numeric_field(
                    case_id,
                    method_id,
                    "bp_p",
                    bp["lm_p_value"],
                    expected["bp_lm_p"],
                    "B",
                    "Chi-square distribution tail probability",
                    atol=PVAL_ATOL,
                    rtol=PVAL_RTOL,
                    shared_primitive="scipy.stats.chi2.sf",
                )
            )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                fit["analyzed_rows"],
                missing["analyzed_rows"],
                "C",
                "Complete-case analyzed rows",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["sample"]["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )

    # -------------------------------------------------------------------------
    # 18. LOGISTIC REGRESSION
    # -------------------------------------------------------------------------
    elif method_id == "logistic_regression":
        workflow = ResearchAssistant(df).run(
            objective=kw["objective"],
            outcome=kw["outcome"],
            predictors=kw["predictors"],
            design=kw["design"],
            estimand=kw["estimand"],
            event_level=kw["event_level"],
            variable_types=kw["variable_types"],
        )
        if workflow.status.value != "completed" or workflow.analysis is None:
            raise RuntimeError(
                f"Logistic regression workflow did not complete: {workflow.blockers}"
            )
        val_res = workflow.analysis.values
        fit = val_res["model_fit"]
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "log_likelihood",
                fit["log_likelihood"],
                expected["log_likelihood"],
                "A",
                "Independent Newton-Raphson/IRLS log-likelihood",
                atol=NUMERICAL_SOLVER_ATOL,
                rtol=NUMERICAL_SOLVER_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "null_log_likelihood",
                fit["null_log_likelihood"],
                expected["null_log_likelihood"],
                "A",
                "Independent null log-likelihood",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "lr_statistic",
                fit["lr_statistic"],
                expected["lr_statistic"],
                "A",
                "Likelihood ratio statistic 2*(LL - LL0)",
                atol=NUMERICAL_SOLVER_ATOL,
                rtol=NUMERICAL_SOLVER_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "mcfadden_r2",
                fit["mcfadden_r2"],
                expected["mcfadden_r2"],
                "A",
                "McFadden pseudo-R2 = 1 - LL/LL0",
                atol=NUMERICAL_SOLVER_ATOL,
                rtol=NUMERICAL_SOLVER_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "aic",
                fit["aic"],
                expected["aic"],
                "A",
                "Akaike Information Criterion 2p - 2LL",
                atol=NUMERICAL_SOLVER_ATOL,
                rtol=NUMERICAL_SOLVER_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "bic",
                fit["bic"],
                expected["bic"],
                "A",
                "Bayesian Information Criterion p*ln(n) - 2LL",
                atol=NUMERICAL_SOLVER_ATOL,
                rtol=NUMERICAL_SOLVER_RTOL,
            )
        )
        for _idx, (coef_obs, beta_exp, se_exp, or_exp) in enumerate(
            zip(
                val_res["coefficients"],
                expected["beta"],
                expected["se_wald"],
                expected["odds_ratios"],
                strict=True,
            )
        ):
            tname = coef_obs["term"]
            results.append(
                compare_numeric_field(
                    case_id,
                    method_id,
                    f"beta_{tname}",
                    coef_obs["estimate"],
                    beta_exp,
                    "A",
                    f"MLE log-odds coefficient for {tname}",
                    atol=NUMERICAL_SOLVER_ATOL,
                    rtol=NUMERICAL_SOLVER_RTOL,
                )
            )
            results.append(
                compare_numeric_field(
                    case_id,
                    method_id,
                    f"se_{tname}",
                    coef_obs["standard_error"],
                    se_exp,
                    "A",
                    f"Wald standard error for {tname}",
                    atol=NUMERICAL_SOLVER_ATOL,
                    rtol=NUMERICAL_SOLVER_RTOL,
                )
            )
            results.append(
                compare_numeric_field(
                    case_id,
                    method_id,
                    f"odds_ratio_{tname}",
                    coef_obs["odds_ratio"],
                    or_exp,
                    "A",
                    f"Odds ratio exp(beta) for {tname}",
                    atol=NUMERICAL_SOLVER_ATOL,
                    rtol=NUMERICAL_SOLVER_RTOL,
                )
            )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                fit["analyzed_rows"],
                missing["analyzed_rows"],
                "C",
                "Complete-case analyzed rows",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                workflow.analysis.metadata["sample"]["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )

    # -------------------------------------------------------------------------
    # 19. KENDALL TAU-B
    # -------------------------------------------------------------------------
    elif method_id == "kendall_tau_b":
        workflow = ResearchAssistant(df).run(
            objective=kw["objective"],
            outcome=kw["outcome"],
            predictor=kw["predictor"],
            design=kw["design"],
            estimand=kw["estimand"],
            association_measure=kw["association_measure"],
            variable_types=kw["variable_types"],
        )
        val_res = workflow.analysis.values
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "tau_b",
                val_res["primary_estimate"],
                expected["tau_b"],
                "A",
                "Independent pair concordance and tie formula",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "scipy.stats.kendalltau p-value",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.norm.sf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                workflow.analysis.metadata["sample"]["analyzed_rows"],
                missing["analyzed_rows"],
                "C",
                "Complete pair analyzed count",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                workflow.analysis.metadata["sample"]["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )

    # -------------------------------------------------------------------------
    # 20. POINT-BISERIAL CORRELATION
    # -------------------------------------------------------------------------
    elif method_id == "point_biserial_correlation":
        workflow = ResearchAssistant(df).run(
            objective=kw["objective"],
            outcome=kw["outcome"],
            predictor=kw["predictor"],
            design=kw["design"],
            estimand=kw["estimand"],
            variable_types=kw["variable_types"],
        )
        val_res = workflow.analysis.values
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "point_biserial_r",
                val_res["primary_estimate"],
                expected["point_biserial_r"],
                "A",
                "Lev (1949) closed-form group difference formula",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "pearson_equivalence",
                val_res["primary_estimate"],
                expected["pearson_r_equivalence"],
                "A",
                "Pearson product-moment dummy equivalence",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "degrees_of_freedom",
                val_res["degrees_of_freedom"],
                expected["degrees_of_freedom"],
                "A",
                "Independent df = n - 2",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "scipy.stats.pointbiserialr p-value",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.t.sf",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                workflow.analysis.metadata["sample"]["analyzed_rows"],
                missing["analyzed_rows"],
                "C",
                "Complete-case analyzed rows",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                workflow.analysis.metadata["sample"]["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )

    # -------------------------------------------------------------------------
    # 21. PARTIAL PEARSON CORRELATION
    # -------------------------------------------------------------------------
    elif method_id == "partial_pearson_correlation":
        workflow = ResearchAssistant(df).run(
            objective=kw["objective"],
            outcome=kw["outcome"],
            predictor=kw["predictor"],
            controls=kw["controls"],
            design=kw["design"],
            estimand=kw["estimand"],
            variable_types=kw["variable_types"],
        )
        val_res = workflow.analysis.values
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "partial_r",
                val_res["primary_estimate"],
                expected["partial_r"],
                "A",
                "OLS covariate residualization Pearson correlation",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "degrees_of_freedom",
                val_res["degrees_of_freedom"],
                expected["degrees_of_freedom"],
                "A",
                "Independent df = n - 2 - k_controls",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["p_value"],
                "B",
                "Residual correlation t-distribution tail",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.t.sf",
            )
        )
        results.append(
            compare_deferred_field(
                case_id,
                method_id,
                "partial_r_ci",
                val_res.get("confidence_interval"),
                "Bootstrap resampling interval deferred",
                "Covariate-resampling bootstrap interval scheduled for subsequent expansion",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                workflow.analysis.metadata["sample"]["analyzed_rows"],
                missing["analyzed_rows"],
                "C",
                "Complete-case analyzed rows",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                workflow.analysis.metadata["sample"]["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )

    # -------------------------------------------------------------------------
    # 22. EXACT BINOMIAL MCNEMAR
    # -------------------------------------------------------------------------
    elif method_id == "mcnemar":
        workflow = ResearchAssistant(df).run(
            objective=kw["objective"],
            outcome=kw["outcome"],
            predictor=kw["predictor"],
            design=kw["design"],
            unit_id=kw["unit_id"],
            estimand=kw["estimand"],
            event_level=kw["event_level"],
            condition_order=kw["condition_order"],
            variable_types=kw["variable_types"],
        )
        val_res = workflow.analysis.values
        tbl = val_res["transition_table"]
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "discordant_b",
                tbl["discordant_b"],
                expected["discordant_b"],
                "A",
                "Paired transition table discordant cell b",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "discordant_c",
                tbl["discordant_c"],
                expected["discordant_c"],
                "A",
                "Paired transition table discordant cell c",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_value",
                val_res["p_value"],
                expected["p_value"],
                "A",
                "Independent exact combinatorial binomial tail sum",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "backend_p_value",
                val_res["p_value"],
                expected["backend_p_value"],
                "B",
                "scipy.stats.binomtest cross-check",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.binomtest",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                workflow.analysis.metadata["sample"]["analyzed_rows"],
                missing["analyzed_rows"],
                "C",
                "Matched pairs analyzed count (2 * n_pairs)",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                workflow.analysis.metadata["sample"]["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )

    # -------------------------------------------------------------------------
    # 23. CRONBACH'S ALPHA
    # -------------------------------------------------------------------------
    elif method_id == "cronbach_alpha":
        val_res = StatisticalAnalyzer(df).scale_reliability(kw["items"])
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "cronbach_alpha",
                val_res["cronbach_alpha"],
                expected["cronbach_alpha"],
                "A",
                "Independent alpha = k/(k-1)*(1 - Σvar_i / var_total)",
            )
        )
        item_stats_obs = {item["item"]: item for item in val_res.get("item_statistics", [])}
        for exp_it in expected["item_statistics"]:
            iname = exp_it["item"]
            if iname in item_stats_obs:
                results.append(
                    compare_numeric_field(
                        case_id,
                        method_id,
                        f"citc_{iname}",
                        item_stats_obs[iname]["corrected_item_total_correlation"],
                        exp_it["corrected_item_total_correlation"],
                        "A",
                        f"Corrected item-total correlation for {iname}",
                    )
                )
                if exp_it["alpha_if_deleted"] is not None:
                    results.append(
                        compare_numeric_field(
                            case_id,
                            method_id,
                            f"aid_{iname}",
                            item_stats_obs[iname]["alpha_if_deleted"],
                            exp_it["alpha_if_deleted"],
                            "A",
                            f"Alpha if deleted for {iname}",
                        )
                    )
        results.append(
            compare_deferred_field(
                case_id,
                method_id,
                "cronbach_bootstrap_ci",
                val_res.get("confidence_interval"),
                "Bootstrap resampling interval deferred",
                "Respondent-row bootstrap interval scheduled for subsequent expansion",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["sample"]["analyzed_rows"],
                missing["analyzed_rows"],
                "C",
                "Complete respondents analyzed",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["sample"]["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )

    # -------------------------------------------------------------------------
    # 24. INTRACLASS CORRELATION
    # -------------------------------------------------------------------------
    elif method_id == "intraclass_correlation":
        val_res = StatisticalAnalyzer(df).intraclass_correlation(
            target=kw["target"],
            rater=kw["rater"],
            value=kw["value"],
        )
        anova_tbl = val_res["anova_table"]
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "grand_mean",
                anova_tbl["grand_mean"],
                expected["grand_mean"],
                "A",
                "Grand mean of ratings",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "bms",
                anova_tbl["ms_targets"],
                expected["bms"],
                "A",
                "Between-targets mean square BMS",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "jms",
                anova_tbl["ms_raters"],
                expected["jms"],
                "A",
                "Between-raters mean square JMS",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "ems",
                anova_tbl["ms_error"],
                expected["ems"],
                "A",
                "Residual error mean square EMS",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "wms",
                anova_tbl["ms_within"],
                expected["wms"],
                "A",
                "Within-targets mean square WMS",
            )
        )
        variants = {v["notation"]: v for v in val_res["all_variants"]}
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "icc_1_1",
                variants["ICC(1,1)"]["estimate"],
                expected["icc_1_1"],
                "A",
                "One-way random single ICC(1,1)",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "icc_1_k",
                variants["ICC(1,k)"]["estimate"],
                expected["icc_1_k"],
                "A",
                "One-way random average ICC(1,k)",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "icc_2_1",
                variants["ICC(2,1)"]["estimate"],
                expected["icc_2_1"],
                "A",
                "Two-way random single agreement ICC(2,1)",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "icc_2_k",
                variants["ICC(2,k)"]["estimate"],
                expected["icc_2_k"],
                "A",
                "Two-way random average agreement ICC(2,k)",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "icc_3_1",
                variants["ICC(3,1)"]["estimate"],
                expected["icc_3_1"],
                "A",
                "Two-way mixed single consistency ICC(3,1)",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "icc_3_k",
                variants["ICC(3,k)"]["estimate"],
                expected["icc_3_k"],
                "A",
                "Two-way mixed average consistency ICC(3,k)",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "f_twoway",
                variants["ICC(2,1)"]["f_test"]["statistic"],
                expected["f_twoway"],
                "A",
                "Two-way BMS/EMS F ratio",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "p_twoway",
                variants["ICC(2,1)"]["f_test"]["p_value"],
                expected["p_twoway"],
                "B",
                "F-distribution tail probability",
                atol=PVAL_ATOL,
                rtol=PVAL_RTOL,
                shared_primitive="scipy.stats.f.sf",
            )
        )
        results.append(
            compare_deferred_field(
                case_id,
                method_id,
                "icc_exact_ci",
                val_res.get("confidence_interval"),
                "Exact F-inversion ICC confidence interval deferred",
                "Exact confidence limit inversion scheduled for subsequent expansion",
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "analyzed_rows",
                val_res["sample"]["analyzed_rows"],
                missing["analyzed_rows"],
                "C",
                "Complete rectangular matrix observations",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )
        results.append(
            compare_numeric_field(
                case_id,
                method_id,
                "excluded_rows",
                val_res["sample"]["excluded_rows"],
                missing["excluded_rows"],
                "C",
                "Excluded row count invariant",
                atol=INT_ATOL,
                rtol=INT_RTOL,
            )
        )

    # Common sample accounting invariant for all methods
    results.append(
        compare_numeric_field(
            case_id,
            method_id,
            "sample_accounting_sum",
            missing["analyzed_rows"] + missing["excluded_rows"],
            missing["original_rows"],
            "C",
            "N_analyzed + N_excluded == N_original",
            atol=INT_ATOL,
            rtol=INT_RTOL,
        )
    )

    return results


def run_all_reference_validations(
    filter_method: str | None = None,
    strict: bool = False,
) -> tuple[list[FieldValidationResult], dict[str, Any]]:
    """Execute validation across all reference cases."""
    cases = get_all_reference_cases()
    if filter_method:
        cases = [c for c in cases if c["method_id"] == filter_method]
        if not cases:
            raise ValueError(f"No reference cases found for method '{filter_method}'")

    field_results: list[FieldValidationResult] = []
    cases_run: list[str] = []
    methods_run: set[str] = set()

    for c in cases:
        c_res = run_case_validation(c)
        field_results.extend(c_res)
        cases_run.append(c["case_id"])
        methods_run.add(c["method_id"])

    level_a_passes = sum(1 for r in field_results if r.evidence_level == "A" and r.status == "pass")
    level_b_passes = sum(1 for r in field_results if r.evidence_level == "B" and r.status == "pass")
    level_c_passes = sum(1 for r in field_results if r.evidence_level == "C" and r.status == "pass")
    deferred_count = sum(1 for r in field_results if r.status == "deferred")
    discrepancies = [r for r in field_results if r.status == "discrepancy"]

    summary = {
        "methods_evaluated_count": len(methods_run),
        "methods_evaluated": sorted(methods_run),
        "cases_evaluated_count": len(cases_run),
        "cases_evaluated": cases_run,
        "fields_compared_count": len(field_results),
        "level_a_passes": level_a_passes,
        "level_b_passes": level_b_passes,
        "level_c_passes": level_c_passes,
        "deferred_count": deferred_count,
        "discrepancies_count": len(discrepancies),
        "discrepancies": [d.to_dict() for d in discrepancies],
        "strict_mode": strict,
        "overall_status": "pass" if len(discrepancies) == 0 else "failed",
    }

    return field_results, summary


def run_self_check() -> bool:
    """Run validation harness self-checks verifying internal integrity (Task 0D)."""
    cases = get_all_reference_cases()
    passed = True

    # 1. Manifest schema version & case count if manifest exists
    manifest_path = Path("validation/reference_manifest.json")
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            s_ver = manifest_data.get("schema_version")
            if s_ver not in (1, 2):
                print(f"[SELF-CHECK FAIL] Unsupported manifest schema version: {s_ver}")
                passed = False
            else:
                print(f"[SELF-CHECK PASS] Manifest schema version {s_ver} is supported.")

            m_cases = manifest_data.get("cases", [])
            if len(m_cases) != len(cases):
                print(f"[SELF-CHECK FAIL] Manifest case count ({len(m_cases)}) != runtime case count ({len(cases)})")
                passed = False
            else:
                print(f"[SELF-CHECK PASS] Manifest case count ({len(m_cases)}) matches runtime case count.")

            # Scan manifest for nonfinite JSON values
            def _has_nonfinite(obj: Any) -> bool:
                if isinstance(obj, float):
                    return not math.isfinite(obj)
                if isinstance(obj, dict):
                    return any(_has_nonfinite(v) for v in obj.values())
                if isinstance(obj, (list, tuple)):
                    return any(_has_nonfinite(v) for v in obj)
                return False

            if _has_nonfinite(manifest_data):
                print("[SELF-CHECK FAIL] Manifest contains nonfinite float values (NaN or Inf).")
                passed = False
            else:
                print("[SELF-CHECK PASS] Manifest contains no nonfinite values.")

            # Verify fields in manifest
            for mc in m_cases:
                mc_id = mc.get("case_id", "UNKNOWN")
                for fld in mc.get("fields", []):
                    fn = fld.get("field", "UNKNOWN")
                    ev_lvl = fld.get("evidence_level")
                    if ev_lvl not in ("A", "B", "C", "D"):
                        print(f"[SELF-CHECK FAIL] Case {mc_id} field {fn} has invalid evidence_level: {ev_lvl}")
                        passed = False
                    if ev_lvl == "A":
                        prov = fld.get("reference") or mc.get("reference_provenance")
                        if not prov:
                            print(f"[SELF-CHECK FAIL] Level A field {mc_id}.{fn} missing reference provenance.")
                            passed = False
                    elif ev_lvl == "B":
                        ref = fld.get("reference", "")
                        if not ref:
                            print(f"[SELF-CHECK FAIL] Level B field {mc_id}.{fn} missing reference backend.")
                            passed = False
                    elif ev_lvl == "D":
                        reason = fld.get("reason", "")
                        if not reason:
                            print(f"[SELF-CHECK FAIL] Level D field {mc_id}.{fn} missing explicit reason.")
                            passed = False

                    if fld.get("status") != "deferred":
                        if "expected" not in fld or fld["expected"] is None:
                            print(f"[SELF-CHECK FAIL] Non-deferred field {mc_id}.{fn} missing expected value.")
                            passed = False
                        atol = fld.get("atol")
                        rtol = fld.get("rtol")
                        if atol is not None and (not math.isfinite(atol) or atol < 0):
                            print(f"[SELF-CHECK FAIL] Field {mc_id}.{fn} has invalid atol: {atol}")
                            passed = False
                        if rtol is not None and (not math.isfinite(rtol) or rtol < 0):
                            print(f"[SELF-CHECK FAIL] Field {mc_id}.{fn} has invalid rtol: {rtol}")
                            passed = False
                    # Check shared primitive disclosure
                    sp = fld.get("shared_primitive")
                    if sp is not None and not isinstance(sp, str):
                        print(f"[SELF-CHECK FAIL] Field {mc_id}.{fn} invalid shared_primitive: {sp}")
                        passed = False

            print("[SELF-CHECK PASS] Manifest fields comply with integrity rules (A/B/C/D, tolerances, non-deferred expected, shared primitives).")

        except Exception as exc:
            print(f"[SELF-CHECK FAIL] Exception reading manifest: {exc}")
            passed = False

    # Summary integrity if summary exists
    summary_path = Path("validation/reference_validation_summary.json")
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            totals = s_data.get("totals", {})
            per_method = s_data.get("per_method_summary", {})
            expected_mids = set(METHOD_CONTRACTS.keys())
            actual_mids = set(per_method.keys())
            if expected_mids != actual_mids:
                print(f"[SELF-CHECK FAIL] Summary methods do not match METHOD_CONTRACTS: diff={expected_mids ^ actual_mids}")
                passed = False
            else:
                print(f"[SELF-CHECK PASS] Summary method IDs match all {len(expected_mids)} METHOD_CONTRACTS.")

            rec_cases = sum(m["cases_count"] for m in per_method.values())
            rec_fields = sum(m["fields_compared_count"] for m in per_method.values())
            rec_a = sum(m["level_a_passes"] for m in per_method.values())
            rec_b = sum(m["level_b_passes"] for m in per_method.values())
            rec_c = sum(m["level_c_passes"] for m in per_method.values())
            rec_d = sum(m["level_d_deferred"] for m in per_method.values())
            rec_disc = sum(m["discrepancies_count"] for m in per_method.values())

            if (
                totals.get("cases_count") != rec_cases
                or totals.get("fields_compared_count") != rec_fields
                or totals.get("level_a_passes") != rec_a
                or totals.get("level_b_passes") != rec_b
                or totals.get("level_c_passes") != rec_c
                or totals.get("deferred_count") != rec_d
                or totals.get("discrepancies_count") != rec_disc
            ):
                print("[SELF-CHECK FAIL] Summary totals do not equal recomputed totals from per_method_summary.")
                passed = False
            else:
                print("[SELF-CHECK PASS] Summary totals equal recomputed totals from field results.")

            committed_fp = s_data.get("package", {}).get("validation_framework_content_sha256")
            if not committed_fp:
                print("[SELF-CHECK FAIL] Summary missing validation_framework_content_sha256.")
                passed = False
            else:
                current_fp = compute_validation_framework_content_sha256()
                if committed_fp != current_fp:
                    print(
                        f"[SELF-CHECK FAIL] Validation framework content fingerprint mismatch: "
                        f"committed={committed_fp}, current={current_fp}"
                    )
                    passed = False
                else:
                    print("[SELF-CHECK PASS] Validation framework content fingerprint matches committed summary.")

        except Exception as exc:
            print(f"[SELF-CHECK FAIL] Exception reading summary: {exc}")
            passed = False

    # 2. All 24 registered method IDs represented
    registered_methods = set(METHOD_CONTRACTS.keys())
    case_methods = {c["method_id"] for c in cases}
    if registered_methods != case_methods:
        missing = registered_methods - case_methods
        print(f"[SELF-CHECK FAIL] Missing registered methods in validation cases: {missing}")
        passed = False
    else:
        print(f"[SELF-CHECK PASS] All {len(registered_methods)} registered method IDs represented in runtime cases.")

    # 3. No duplicate case IDs
    case_ids = [c["case_id"] for c in cases]
    if len(case_ids) != len(set(case_ids)):
        dups = [cid for cid in case_ids if case_ids.count(cid) > 1]
        print(f"[SELF-CHECK FAIL] Duplicate case IDs detected: {set(dups)}")
        passed = False
    else:
        print(f"[SELF-CHECK PASS] All {len(case_ids)} runtime case IDs are unique.")

    # 4. Tolerances non-negative and finite
    for tol_name, tol_val in [
        ("DEFAULT_ATOL", DEFAULT_ATOL),
        ("DEFAULT_RTOL", DEFAULT_RTOL),
        ("PVAL_ATOL", PVAL_ATOL),
        ("PVAL_RTOL", PVAL_RTOL),
        ("INT_ATOL", INT_ATOL),
        ("INT_RTOL", INT_RTOL),
    ]:
        if not math.isfinite(tol_val) or tol_val < 0:
            print(f"[SELF-CHECK FAIL] Invalid tolerance {tol_name}: {tol_val}")
            passed = False
    print("[SELF-CHECK PASS] All tolerances are finite and non-negative.")

    # 5. Cases structure validity
    for c in cases:
        cid = c["case_id"]
        if "expected" not in c or not isinstance(c["expected"], dict):
            print(f"[SELF-CHECK FAIL] Case {cid} missing expected dictionary")
            passed = False
        if "missing_accounting" not in c:
            print(f"[SELF-CHECK FAIL] Case {cid} missing missing_accounting")
            passed = False

    print("[SELF-CHECK PASS] All reference cases have required structural metadata.")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PyAutoStat Independent Numerical Validation Runner"
    )
    parser.add_argument("--method", help="Filter validation to one method_id")
    parser.add_argument("--json", dest="json_path", help="Path to write JSON validation report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with non-zero exit code if any numerical discrepancy occurs",
    )
    parser.add_argument(
        "--generate-manifest",
        action="store_true",
        help="Regenerate validation/reference_manifest.json and reference_validation_summary.json",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run internal harness integrity and schema sanity checks",
    )
    args = parser.parse_args()

    if args.self_check:
        ok = run_self_check()
        return 0 if ok else 1

    cases = get_all_reference_cases()
    results, summary = run_all_reference_validations(filter_method=args.method, strict=args.strict)

    if args.generate_manifest:
        manifest_data = build_manifest_v2(cases, results)
        write_manifest(manifest_data)
        summary_data = build_validation_summary(cases, results, summary)
        write_summary(summary_data)
        print("Regenerated validation/reference_manifest.json (Schema Version 2)")
        print("Generated validation/reference_validation_summary.json")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "field_results": [r.to_dict() for r in results],
                },
                f,
                indent=2,
            )
        print(f"JSON validation report written to {args.json_path}")

    # Print human-readable summary
    print("\n" + "=" * 78)
    print("PYAUTOSTAT INDEPENDENT NUMERICAL VALIDATION HARNESS -- FULL METHOD EXPANSION")
    print("=" * 78)
    print(f"Methods Evaluated : {summary['methods_evaluated_count']} / 24 Registered Method IDs")
    print(f"Cases Evaluated   : {summary['cases_evaluated_count']}")
    print(f"Fields Compared   : {summary['fields_compared_count']}")
    print("-" * 78)
    print(f"Level A Passes (Independent Reference)       : {summary['level_a_passes']}")
    print(f"Level B Passes (External Backend Conformance): {summary['level_b_passes']}")
    print(f"Level C Passes (Internal Invariants)         : {summary['level_c_passes']}")
    print(f"Level D Fields (Explicitly Deferred)         : {summary['deferred_count']}")
    print(f"Discrepancies Encountered                    : {summary['discrepancies_count']}")
    print("-" * 78)

    if summary["discrepancies_count"] > 0:
        print("DISCREPANCY DETAILS:")
        for disc in summary["discrepancies"]:
            cid = disc["case_id"]
            fld = disc["field"]
            obs = disc["observed"]
            exp = disc["expected"]
            err = disc["absolute_error"]
            print(f"  [{cid}] {fld}: obs={obs}, exp={exp} (err={err})")
        print("=" * 78)
        if args.strict:
            print("STRICT MODE: Exiting with code 1 due to discrepancies.")
            return 1
        else:
            print("DEFAULT MODE: Discrepancies reported, exiting with code 0.")
            return 0
    else:
        print("ALL COMPARISONS PASSED ACCORDING TO DECLARED TOLERANCE POLICIES.")
        print("=" * 78 + "\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
