"""PyAutoStat Standalone Reference Validation Runner.

Executes independent reference validation comparing PyAutoStat numerical outputs
against independent manual formulas (Level A), external backend conformance (Level B),
internal invariants (Level C), and explicitly tracks deferred quantities (Level D).

Usage:
    python validation/run_reference_validation.py
    python validation/run_reference_validation.py --method welch_t
    python validation/run_reference_validation.py --json report.json
    python validation/run_reference_validation.py --strict
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np  # noqa: E402

# PyAutoStat public APIs used for validation execution
from pyautostat import AnalysisOptions, ResearchAssistant, StatisticalAnalyzer  # noqa: E402

try:
    from validation.reference_cases import get_reference_cases  # noqa: E402
except ImportError:
    from reference_cases import get_reference_cases  # noqa: E402

DEFAULT_ATOL = 1e-12
DEFAULT_RTOL = 1e-10

# Slightly wider tolerances for floating-point p-values or asymptotic approximations
PVAL_ATOL = 1e-8
PVAL_RTOL = 1e-6


@dataclass
class FieldValidationResult:
    case_id: str
    method_id: str
    field: str
    observed: Any
    expected: Any
    absolute_error: float | None
    relative_error: float | None
    atol: float
    rtol: float
    evidence_level: str  # "A" | "B" | "C" | "D"
    reference: str
    status: str  # "pass" | "deferred" | "not_applicable" | "discrepancy"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_numeric_field(
    case_id: str,
    method_id: str,
    field: str,
    observed: Any,
    expected: Any,
    evidence_level: str,
    reference: str,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    notes: str = "",
) -> FieldValidationResult:
    """Compare a single numerical field against expected reference value."""
    if observed is None and expected is None:
        return FieldValidationResult(
            case_id=case_id,
            method_id=method_id,
            field=field,
            observed=None,
            expected=None,
            absolute_error=0.0,
            relative_error=0.0,
            atol=atol,
            rtol=rtol,
            evidence_level=evidence_level,
            reference=reference,
            status="pass",
            notes=notes,
        )

    if observed is None or expected is None:
        return FieldValidationResult(
            case_id=case_id,
            method_id=method_id,
            field=field,
            observed=observed,
            expected=expected,
            absolute_error=None,
            relative_error=None,
            atol=atol,
            rtol=rtol,
            evidence_level=evidence_level,
            reference=reference,
            status="discrepancy",
            notes=f"Missing value discrepancy: observed={observed}, expected={expected}",
        )

    obs_val = float(observed)
    exp_val = float(expected)

    if math.isnan(obs_val) and math.isnan(exp_val):
        return FieldValidationResult(
            case_id=case_id,
            method_id=method_id,
            field=field,
            observed=obs_val,
            expected=exp_val,
            absolute_error=0.0,
            relative_error=0.0,
            atol=atol,
            rtol=rtol,
            evidence_level=evidence_level,
            reference=reference,
            status="pass",
            notes=notes,
        )

    abs_err = abs(obs_val - exp_val)
    denom = max(abs(exp_val), atol)
    rel_err = abs_err / denom

    is_pass = abs_err <= atol or rel_err <= rtol
    status = "pass" if is_pass else "discrepancy"

    return FieldValidationResult(
        case_id=case_id,
        method_id=method_id,
        field=field,
        observed=obs_val,
        expected=exp_val,
        absolute_error=abs_err,
        relative_error=rel_err,
        atol=atol,
        rtol=rtol,
        evidence_level=evidence_level,
        reference=reference,
        status=status,
        notes=notes,
    )


def compare_deferred_field(
    case_id: str,
    method_id: str,
    field: str,
    observed: Any,
    reference: str,
    notes: str,
) -> FieldValidationResult:
    """Record an explicitly deferred quantity (Level D)."""
    return FieldValidationResult(
        case_id=case_id,
        method_id=method_id,
        field=field,
        observed=observed,
        expected=None,
        absolute_error=None,
        relative_error=None,
        atol=0.0,
        rtol=0.0,
        evidence_level="D",
        reference=reference,
        status="deferred",
        notes=notes,
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
                "Effect size CI calculation deferred in Phase 2",
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
                "Effect size bootstrap CI deferred in Phase 2",
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
                "Effect size bootstrap CI deferred in Phase 2",
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
                "Cohen's dz exact noncentral-t CI deferred in Phase 2",
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
                "Matched-pairs rank-biserial bootstrap CI deferred in Phase 2",
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
                "Spearman paired bootstrap CI deferred in Phase 2",
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
                "Cramer's V bootstrap interval deferred in Phase 2",
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
                "Odds ratio confidence interval deferred in Phase 2",
            )
        )

    return results


def run_all_reference_validations(
    filter_method: str | None = None,
    strict: bool = False,
) -> tuple[list[FieldValidationResult], dict[str, Any]]:
    """Run validation across all reference cases and summarize."""
    all_cases = get_reference_cases()
    if filter_method:
        all_cases = [c for c in all_cases if c["method_id"] == filter_method]
        if not all_cases:
            raise ValueError(f"No reference cases found for method: {filter_method}")

    field_results: list[FieldValidationResult] = []
    cases_run: list[str] = []
    methods_run: set[str] = set()

    for case in all_cases:
        case_res = run_case_validation(case)
        field_results.extend(case_res)
        cases_run.append(case["case_id"])
        methods_run.add(case["method_id"])

    # Aggregate metrics
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


def generate_manifest(output_path: str = "validation/reference_manifest.json") -> None:
    """Generate the structured reference manifest from verified cases."""
    cases = get_reference_cases()
    manifest_entries = []
    for c in cases:
        manifest_entries.append(
            {
                "case_id": c["case_id"],
                "method_id": c["method_id"],
                "scientific_target": c["scientific_target"],
                "dataset_description": c["description"],
                "reference_source_type": c["source_type"],
                "reference_provenance": c["reference_citation"],
                "sample_rows_original": c["missing_accounting"]["original_rows"],
                "sample_rows_analyzed": c["missing_accounting"]["analyzed_rows"],
                "sample_rows_excluded": c["missing_accounting"]["excluded_rows"],
                "orientation_definition": c.get("orientation", {}),
                "tolerances": {
                    "default_atol": DEFAULT_ATOL,
                    "default_rtol": DEFAULT_RTOL,
                    "p_value_atol": PVAL_ATOL,
                    "p_value_rtol": PVAL_RTOL,
                },
                "validation_evidence_levels": {
                    "primary_quantities": "Level A (Independent Reference)",
                    "p_values": "Level B (External Backend Conformance)",
                    "sample_accounting_and_orientation": "Level C (Internal Invariant)",
                    "effect_size_intervals": "Level D (Deferred)",
                },
                "notes": "Verified in PyAutoStat Phase 2 independent validation program",
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"schema_version": 1, "cases": manifest_entries}, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="PyAutoStat Reference Validation Harness")
    parser.add_argument("--method", help="Filter validation to one method_id")
    parser.add_argument("--json", dest="json_path", help="Path to write JSON validation report")
    parser.add_argument(
        "--strict", action="store_true", help="Fail with non-zero exit if any discrepancy"
    )
    parser.add_argument(
        "--generate-manifest",
        action="store_true",
        help="Regenerate validation/reference_manifest.json",
    )
    args = parser.parse_args()

    if args.generate_manifest:
        generate_manifest()
        print("Regenerated validation/reference_manifest.json")
        return 0

    results, summary = run_all_reference_validations(filter_method=args.method, strict=args.strict)

    # Print human-readable summary
    print("\n" + "=" * 78)
    print("PYAUTOSTAT INDEPENDENT NUMERICAL VALIDATION HARNESS -- PHASE 2")
    print("=" * 78)
    print(f"Methods Evaluated : {summary['methods_evaluated_count']} / 10 First-Tranche Methods")
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
        return 1
    else:
        print("ALL COMPARISONS PASSED ACCORDING TO DECLARED TOLERANCE POLICIES.")
        print("=" * 78 + "\n")

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
