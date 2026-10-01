"""Authoritative structured statistical method contracts.

Every inferential, model, and reliability method supported by PyAutoStat defines
an explicit scientific contract: question, estimand, hypotheses, effect size,
uncertainty status, required assumptions, missingness policy, numerical provenance,
audit invariants, and independent validation source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MethodContract:
    """Explicit scientific contract for a statistical method."""

    method_id: str
    name: str
    scientific_question: str
    study_design: str
    outcome_type: str
    predictor_type: str
    estimand: str
    primary_estimate: str
    null_hypothesis: str
    alternative_hypothesis: str
    null_value: float | None
    null_quantity: str | None
    test_statistic: str
    degrees_of_freedom: str
    effect_size_quantity: str
    effect_size_definition: str
    effect_size_ci_status: (
        str  # available | unavailable | not_supported | not_applicable | uncomputable
    )
    primary_estimate_ci_status: (
        str  # available | unavailable | not_supported | not_applicable | uncomputable
    )
    ci_method: str
    assumptions: tuple[str, ...]
    diagnostics: tuple[str, ...]
    missing_data_policy: str
    degenerate_data_behavior: str
    multiplicity_policy: str
    numerical_provenance: str
    interpretation_limitations: tuple[str, ...]
    audit_invariants: tuple[str, ...]
    validation_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "name": self.name,
            "scientific_question": self.scientific_question,
            "study_design": self.study_design,
            "outcome_type": self.outcome_type,
            "predictor_type": self.predictor_type,
            "estimand": self.estimand,
            "primary_estimate": self.primary_estimate,
            "null_hypothesis": self.null_hypothesis,
            "alternative_hypothesis": self.alternative_hypothesis,
            "null_value": self.null_value,
            "null_quantity": self.null_quantity,
            "test_statistic": self.test_statistic,
            "degrees_of_freedom": self.degrees_of_freedom,
            "effect_size_quantity": self.effect_size_quantity,
            "effect_size_definition": self.effect_size_definition,
            "effect_size_ci_status": self.effect_size_ci_status,
            "primary_estimate_ci_status": self.primary_estimate_ci_status,
            "ci_method": self.ci_method,
            "assumptions": list(self.assumptions),
            "diagnostics": list(self.diagnostics),
            "missing_data_policy": self.missing_data_policy,
            "degenerate_data_behavior": self.degenerate_data_behavior,
            "multiplicity_policy": self.multiplicity_policy,
            "numerical_provenance": self.numerical_provenance,
            "interpretation_limitations": list(self.interpretation_limitations),
            "audit_invariants": list(self.audit_invariants),
            "validation_source": self.validation_source,
        }


METHOD_CONTRACTS: dict[str, MethodContract] = {
    "welch_t": MethodContract(
        method_id="welch_t",
        name="Welch independent-samples t-test",
        scientific_question=(
            "Do two independent populations differ in their mean continuous outcome?"
        ),
        study_design="Independent groups, exactly 2 groups",
        outcome_type="Continuous quantitative",
        predictor_type="Binary categorical factor",
        estimand=("Population mean difference (first group mean minus second group mean)"),
        primary_estimate="Sample mean difference (x̄₁ - x̄₂)",
        null_hypothesis="The two population means are equal.",
        alternative_hypothesis="The two population means are unequal (two-sided).",
        null_value=0.0,
        null_quantity="mean difference",
        test_statistic="Welch's t statistic",
        degrees_of_freedom="Welch-Satterthwaite approximation",
        effect_size_quantity="Cohen's d",
        effect_size_definition="First minus second group mean, divided by the pooled sample SD.",
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method=(
            "Analytical Student-t interval for mean difference; percentile bootstrap for Cohen's d."
        ),
        assumptions=(
            "Independent observations within and between groups.",
            "Continuous outcome scale.",
            ("Approximate normality of group sampling distributions or sufficient sample size."),
            "Equal population variances are NOT assumed.",
        ),
        diagnostics=(
            "Shapiro-Wilk normality per group",
            "Group sample sizes",
            "Levene variance ratio (advisory)",
        ),
        missing_data_policy=("Complete-case analysis on (group, outcome). Excluded rows recorded."),
        degenerate_data_behavior=(
            "Zero variance in either group or n < 2 in any group returns unavailable status."
        ),
        multiplicity_policy="Not applicable (single contrast).",
        numerical_provenance="scipy.stats.ttest_ind(equal_var=False)",
        interpretation_limitations=(
            ("Mean difference is descriptive and noncausal without random assignment."),
            ("Non-rejection of the null hypothesis does not prove equal population means."),
            "Diagnostic non-rejection does not prove normality.",
        ),
        audit_invariants=(
            "n₁ + n₂ == analyzed_rows",
            "analyzed_rows + excluded_rows == original_rows",
            "contrast definition is first group minus second group",
            "p_value in [0, 1]",
            "analytical confidence interval contains sample mean difference",
            "CI lower <= upper",
            ("Cohen's d sign matches mean difference sign when difference is nonzero"),
        ),
        validation_source=(
            "Welch (1947); SciPy reference cross-check; verified against "
            "manual fixtures in tests/test_statistical_correctness.py."
        ),
    ),
    "student_t": MethodContract(
        method_id="student_t",
        name="Student independent-samples t-test",
        scientific_question=(
            "Do two independent populations with equal variance differ in "
            "their mean continuous outcome?"
        ),
        study_design="Independent groups, exactly 2 groups",
        outcome_type="Continuous quantitative",
        predictor_type="Binary categorical factor",
        estimand=("Population mean difference under equal variance assumption (μ₁ - μ₂)"),
        primary_estimate="Sample mean difference (x̄₁ - x̄₂)",
        null_hypothesis="The two population means are equal.",
        alternative_hypothesis="The two population means are unequal (two-sided).",
        null_value=0.0,
        null_quantity="mean difference",
        test_statistic="Student's t statistic",
        degrees_of_freedom="Pooled df = n₁ + n₂ - 2",
        effect_size_quantity="Cohen's d",
        effect_size_definition="First minus second group mean, divided by the pooled sample SD.",
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method=(
            "Analytical Student-t interval for mean difference; percentile bootstrap for Cohen's d."
        ),
        assumptions=(
            "Independent observations within and between groups.",
            "Continuous outcome scale.",
            "Equal population variances (homoscedasticity).",
            ("Approximate normality of group distributions or sufficient sample size."),
        ),
        diagnostics=(
            "Levene test for equality of variance",
            "Shapiro-Wilk normality per group",
        ),
        missing_data_policy=("Complete-case analysis on (group, outcome). Excluded rows recorded."),
        degenerate_data_behavior=(
            "Zero pooled variance or n < 2 in any group returns unavailable status."
        ),
        multiplicity_policy="Not applicable (single contrast).",
        numerical_provenance="scipy.stats.ttest_ind(equal_var=True)",
        interpretation_limitations=(
            ("Requires explicit scientific justification for equal population variance."),
            "Levene test non-rejection does not prove equal variance.",
            "Non-rejection does not prove equal population means.",
        ),
        audit_invariants=(
            "n₁ + n₂ == analyzed_rows",
            "df == n₁ + n₂ - 2",
            "analyzed_rows + excluded_rows == original_rows",
            "analytical confidence interval contains sample mean difference",
            "CI lower <= upper",
        ),
        validation_source=(
            "Student (1908); SciPy cross-check; verified in tests/test_statistical_correctness.py."
        ),
    ),
    "paired_t": MethodContract(
        method_id="paired_t",
        name="Paired-samples t-test",
        scientific_question=(
            "Is the mean of paired differences in a two-condition within-unit design zero?"
        ),
        study_design=(
            "Paired / repeated measures with exactly 2 conditions and explicit unit identifier"
        ),
        outcome_type="Continuous quantitative",
        predictor_type="Condition variable (2 levels) + Unit ID",
        estimand="Population mean paired difference (μ_D = μ₁ - μ₂)",
        primary_estimate="Sample mean paired difference (D̄ = (1/n) Σ dᵢ)",
        null_hypothesis="The population mean paired difference is zero.",
        alternative_hypothesis="The population mean paired difference is nonzero (two-sided).",
        null_value=0.0,
        null_quantity="mean paired difference",
        test_statistic="Paired t statistic: t = D̄ / (s_D / √n)",
        degrees_of_freedom="df = complete_pairs - 1",
        effect_size_quantity="Cohen's dz",
        effect_size_definition=(
            "Sample mean paired difference divided by the standard deviation of paired differences."
        ),
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method=(
            "Analytical Student-t interval for mean paired difference; "
            "exact noncentral-t inversion for Cohen's dz."
        ),
        assumptions=(
            "Explicit paired or matched observational units.",
            "Independent pairs across units.",
            "One usable measurement per unit and condition.",
            ("Paired differences approximately normally distributed or sufficient sample size."),
            "Nonzero finite variance of paired differences.",
        ),
        diagnostics=(
            "Shapiro-Wilk normality on paired differences",
            "Complete-pair accounting",
        ),
        missing_data_policy=(
            "Incomplete units (missing either condition) are excluded; "
            "incomplete unit count recorded. No row-adjacency pairing."
        ),
        degenerate_data_behavior=(
            "Zero variance of paired differences or complete_pairs < 2 returns unavailable status."
        ),
        multiplicity_policy="Not applicable (single contrast).",
        numerical_provenance="scipy.stats.ttest_rel",
        interpretation_limitations=(
            ("Mean paired difference is noncausal without condition randomization."),
            (
                "Cohen's dz CI uses exact noncentral-t inversion targeting the "
                "population standardized paired difference."
            ),
            "Non-rejection does not prove zero difference.",
        ),
        audit_invariants=(
            "2 * complete_pairs == analyzed_rows",
            "df == complete_pairs - 1",
            "contrast condition order matches declared condition order",
            "analytical CI contains sample mean paired difference",
            "CI lower <= upper",
            "Cohen's dz CI lower <= upper",
        ),
        validation_source=(
            "Fisher (1925); SciPy cross-check; verified in tests/test_paired_analysis.py."
        ),
    ),
    "one_sample_t": MethodContract(
        method_id="one_sample_t",
        name="One-sample t-test",
        scientific_question=(
            "Does the population mean of a single continuous variable differ "
            "from a declared reference value?"
        ),
        study_design="Single sample against a fixed external scalar reference value",
        outcome_type="Continuous quantitative",
        predictor_type="None (declared reference value)",
        estimand="Population mean minus declared reference value (μ - μ₀)",
        primary_estimate="Sample mean difference from reference (x̄ - μ₀)",
        null_hypothesis="The population mean equals the declared reference value.",
        alternative_hypothesis=(
            "The population mean does not equal the declared reference value (two-sided)."
        ),
        null_value=0.0,
        null_quantity="mean difference from reference",
        test_statistic="Student's one-sample t statistic",
        degrees_of_freedom="df = n - 1",
        effect_size_quantity="One-sample Cohen's d",
        effect_size_definition=(
            "Sample mean minus reference value, divided by the sample standard deviation."
        ),
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method=(
            "Analytical Student-t interval for mean difference from "
            "reference; exact noncentral-t inversion for one-sample Cohen's d."
        ),
        assumptions=(
            "Independent observations.",
            "Continuous outcome scale.",
            "Finite, researcher-declared reference value.",
            ("Approximate normality of population or sufficient sample size for CLT."),
            "Finite positive sample variance.",
        ),
        diagnostics=(
            "Shapiro-Wilk normality on outcome",
            "Sample size check",
        ),
        missing_data_policy="Complete-case analysis on outcome. Excluded rows recorded.",
        degenerate_data_behavior="Zero variance or n < 2 returns unavailable status.",
        multiplicity_policy="Not applicable.",
        numerical_provenance="scipy.stats.ttest_1samp",
        interpretation_limitations=(
            (
                "Validity depends on the scientific relevance and "
                "justification of the reference value."
            ),
            (
                "One-sample Cohen's d CI uses exact noncentral-t inversion targeting "
                "the population standardized difference."
            ),
        ),
        audit_invariants=(
            "df == analyzed_rows - 1",
            "metadata reference_value matches question specification",
            "analytical CI contains sample mean difference from reference",
            "CI lower <= upper",
            "one-sample Cohen's d CI lower <= upper",
        ),
        validation_source=(
            "Student (1908); SciPy cross-check; verified in tests/test_basic_inference.py."
        ),
    ),
    "mann_whitney_u": MethodContract(
        method_id="mann_whitney_u",
        name="Mann-Whitney U test",
        scientific_question=(
            "Do two independent populations differ in their underlying rank "
            "distributions (stochastic dominance)?"
        ),
        study_design="Independent groups, exactly 2 groups",
        outcome_type="Ordered numeric (ordinal or continuous)",
        predictor_type="Binary categorical factor",
        estimand=("Rank distribution separation / stochastic superiority (P(X > Y) - P(Y > X))"),
        primary_estimate="Matched rank-biserial correlation (r_rb = 2U/(n₁n₂) - 1)",
        null_hypothesis="The two underlying rank distributions are equal.",
        alternative_hypothesis="The two underlying rank distributions differ (two-sided).",
        null_value=0.0,
        null_quantity="rank-biserial correlation",
        test_statistic="Mann-Whitney U statistic (ties handled via mid-ranks)",
        degrees_of_freedom="None (not applicable to rank-sum test)",
        effect_size_quantity="Rank-biserial correlation",
        effect_size_definition="2 times first-group U divided by n1*n2, minus 1; ties count half.",
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method="Independent within-group percentile bootstrap.",
        assumptions=(
            "Independent observations within and between groups.",
            "Ordinal or continuous outcome supporting meaningful ranking.",
            ("Equal distribution shapes are NOT assumed unless specifically testing medians."),
        ),
        diagnostics=(
            "Group sample sizes",
            "Tie counts and tie proportions",
        ),
        missing_data_policy=("Complete-case analysis on (group, outcome). Excluded rows recorded."),
        degenerate_data_behavior=(
            "Completely tied data (zero rank variance) or n < 2 per group "
            "returns unavailable status."
        ),
        multiplicity_policy="Not applicable.",
        numerical_provenance="scipy.stats.mannwhitneyu(method='auto')",
        interpretation_limitations=(
            ("Test of stochastic dominance / rank distributions, NOT a universal test of medians."),
            ("Median-difference interpretation strictly requires identical distribution shapes."),
            "Does not establish causality.",
        ),
        audit_invariants=(
            "n₁ + n₂ == analyzed_rows",
            "degrees_of_freedom is None",
            "p_value in [0, 1]",
            "rank_biserial in [-1, 1]",
            "bootstrap CI bounds in [-1, 1]",
            "CI lower <= upper",
        ),
        validation_source=(
            "Mann & Whitney (1947); SciPy cross-check; verified in "
            "tests/test_statistical_correctness.py."
        ),
    ),
    "wilcoxon_signed_rank": MethodContract(
        method_id="wilcoxon_signed_rank",
        name="Paired Wilcoxon signed-rank test",
        scientific_question=(
            "Is the paired-difference distribution centered at zero under the signed-rank model?"
        ),
        study_design=(
            "Paired / repeated measures with exactly 2 conditions and explicit unit identifier"
        ),
        outcome_type="Ordered numeric (ordinal or continuous)",
        predictor_type="Condition variable (2 levels) + Unit ID",
        estimand=("Matched-pairs signed-rank distribution center / location shift under symmetry"),
        primary_estimate="Matched-pairs rank-biserial correlation ((W⁺ - W⁻) / (W⁺ + W⁻))",
        null_hypothesis=(
            "The paired-difference distribution is centered at zero under the signed-rank model."
        ),
        alternative_hypothesis=(
            "The paired-difference distribution is not centered at zero (two-sided location shift)."
        ),
        null_value=0.0,
        null_quantity="matched-pairs rank-biserial correlation",
        test_statistic="Wilcoxon signed-rank statistic W (sum of positive signed ranks)",
        degrees_of_freedom="None (not applicable)",
        effect_size_quantity="Matched-pairs rank-biserial correlation",
        effect_size_definition=(
            "Difference of positive and negative rank sums divided by total rank sum."
        ),
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method=(
            "Paired-observation percentile bootstrap for matched-pairs rank-biserial correlation."
        ),
        assumptions=(
            "Explicit paired or matched observational units.",
            "Independent pairs across units.",
            "Meaningful ordering and signed paired differences.",
            (
                "Symmetric paired-difference distribution for a "
                "location-shift / median interpretation."
            ),
            ("Zero differences handled via SciPy 'wilcox' policy (omitted from ranking)."),
        ),
        diagnostics=(
            "Count of zero differences",
            "Positive and negative rank sums",
            "Tie presence",
        ),
        missing_data_policy=(
            "Incomplete units (missing either condition) are excluded; "
            "incomplete unit count recorded."
        ),
        degenerate_data_behavior="Fewer than 2 nonzero differences returns unavailable status.",
        multiplicity_policy="Not applicable.",
        numerical_provenance="scipy.stats.wilcoxon(zero_method='wilcox')",
        interpretation_limitations=(
            (
                "NOT a universal test of medians; location shift strictly "
                "requires symmetry of paired differences."
            ),
            (
                "Matched-pairs rank-biserial CI uses unit-level paired percentile bootstrap; "
                "does not assume asymptotic normality."
            ),
        ),
        audit_invariants=(
            "2 * complete_pairs == analyzed_rows",
            "contrast condition order matches declared condition order",
            "degrees_of_freedom is None",
            "p_value in [0, 1]",
            "rank_biserial in [-1, 1]",
            "CI bounds in [-1, 1]",
            "CI lower <= upper",
        ),
        validation_source=(
            "Wilcoxon (1945); SciPy cross-check; verified in tests/test_paired_analysis.py."
        ),
    ),
    "welch_anova": MethodContract(
        method_id="welch_anova",
        name="Welch one-way ANOVA with Games-Howell comparisons",
        scientific_question=(
            "Do three or more independent population means differ, without "
            "assuming equal population variances?"
        ),
        study_design="Independent groups, 3 or more groups",
        outcome_type="Continuous quantitative",
        predictor_type="Categorical grouping (3+ levels)",
        estimand=(
            "Heteroscedastic multi-group population mean equality; pairwise "
            "population mean differences"
        ),
        primary_estimate="Group sample means; pairwise mean differences (x̄ⱼ - x̄ₘ)",
        null_hypothesis="All population means are equal.",
        alternative_hypothesis="At least one group population mean differs from another.",
        null_value=0.0,
        null_quantity="group means",
        test_statistic="Welch's omnibus F statistic",
        degrees_of_freedom="Two dfs: [k - 1, Welch-Satterthwaite denominator df]",
        effect_size_quantity=(
            "Not applicable globally; unstandardized pairwise mean differences reported"
        ),
        effect_size_definition=(
            "Not applicable; group means and pairwise mean differences are reported."
        ),
        effect_size_ci_status="not_applicable",
        primary_estimate_ci_status="available",
        ci_method=(
            "Games-Howell simultaneous studentized-range confidence intervals "
            "for all pairwise contrasts."
        ),
        assumptions=(
            "Independent observations within and between groups.",
            "Continuous outcome scale.",
            ("Outcome approximately normal within each group or large group sizes."),
            "Equal population variances are NOT assumed.",
            "Positive finite sample variance in every group.",
        ),
        diagnostics=(
            "Per-group sample sizes",
            "Per-group means and SDs",
            "Levene test (advisory)",
        ),
        missing_data_policy=("Complete-case analysis on (group, outcome). Excluded rows recorded."),
        degenerate_data_behavior=(
            "Zero variance in any group or n < 2 in any group returns unavailable status."
        ),
        multiplicity_policy=(
            "Complete family of k(k-1)/2 Games-Howell comparisons with "
            "simultaneous studentized-range CIs and adjusted p-values."
        ),
        numerical_provenance=(
            "Explicit PyAutoStat formula in multigroup.py (Welch 1951, Games & Howell 1976)."
        ),
        interpretation_limitations=(
            ("Omnibus significance does not identify which specific groups differ."),
            ("No global standardized effect size is reported under unequal variances."),
            "Does not establish causality.",
        ),
        audit_invariants=(
            "degrees_of_freedom has length 2",
            "df₁ == k - 1",
            "df₂ > 0",
            "complete family of k*(k-1)//2 pairwise records present",
            "pairwise adjusted p_values in [0, 1]",
            "pairwise Games-Howell CI contains point estimate",
            "CI lower <= upper",
        ),
        validation_source=(
            "Welch (1951); Games & Howell (1976); cross-checked with pingouin "
            "in tests/test_multigroup_analysis.py."
        ),
    ),
    "one_way_anova": MethodContract(
        method_id="one_way_anova",
        name="Standard classical one-way ANOVA with Tukey-Kramer comparisons",
        scientific_question=(
            "Do three or more independent population means differ under an "
            "assumed equal population variance?"
        ),
        study_design="Independent groups, 3 or more groups",
        outcome_type="Continuous quantitative",
        predictor_type="Categorical grouping (3+ levels)",
        estimand=(
            "Homoscedastic population mean equality (μ₁ = ... = μ_k); pairwise mean differences"
        ),
        primary_estimate="Omnibus F; pairwise mean differences",
        null_hypothesis="All group population means are equal.",
        alternative_hypothesis="At least one group population mean differs.",
        null_value=0.0,
        null_quantity="group means",
        test_statistic="Snedecor's F = MS_between / MS_within",
        degrees_of_freedom="Two dfs: [k - 1, N - k]",
        effect_size_quantity="Eta-squared (η²)",
        effect_size_definition="Between-group sum of squares divided by total sum of squares.",
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method=(
            "Percentile bootstrap for eta-squared; Tukey-Kramer "
            "studentized-range intervals for pairwise contrasts."
        ),
        assumptions=(
            "Independent observations within and between groups.",
            "Continuous outcome scale.",
            "Equal population variances (homoscedasticity).",
            "Within-group approximate normality or large sample sizes.",
        ),
        diagnostics=(
            "Levene test for equality of variance",
            "Shapiro-Wilk normality per group",
        ),
        missing_data_policy=("Complete-case analysis on (group, outcome). Excluded rows recorded."),
        degenerate_data_behavior=(
            "Zero total variance or n < 2 in any group returns unavailable status."
        ),
        multiplicity_policy=(
            "Complete family of k(k-1)/2 Tukey-Kramer studentized-range "
            "comparisons with familywise-adjusted p-values."
        ),
        numerical_provenance="scipy.stats.f_oneway for omnibus; multigroup.py for Tukey-Kramer.",
        interpretation_limitations=(
            "Requires explicit equal-variance justification.",
            "Eta-squared is an in-sample descriptive effect.",
            "Does not establish causality.",
        ),
        audit_invariants=(
            "degrees_of_freedom has length 2",
            "df₁ == k - 1",
            "df₂ == analyzed_rows - k",
            "eta_squared in [0, 1]",
            "complete family of k*(k-1)//2 pairwise records present",
            "pairwise adjusted p_values in [0, 1]",
        ),
        validation_source=(
            "Fisher (1925); Tukey (1949); SciPy cross-check; verified in "
            "tests/test_multigroup_analysis.py."
        ),
    ),
    "kruskal_wallis": MethodContract(
        method_id="kruskal_wallis",
        name="Kruskal-Wallis test with Dunn-Holm comparisons",
        scientific_question=(
            "Do three or more independent populations differ in their rank distributions?"
        ),
        study_design="Independent groups, 3 or more groups",
        outcome_type="Ordered numeric (ordinal or continuous)",
        predictor_type="Categorical grouping (3+ levels)",
        estimand="Multi-group rank distribution equality / stochastic dominance",
        primary_estimate="Rank epsilon-squared (ε²); pairwise mean-rank contrasts",
        null_hypothesis="All group rank distributions are equal.",
        alternative_hypothesis="At least one group rank distribution differs.",
        null_value=0.0,
        null_quantity="rank distributions",
        test_statistic="Kruskal-Wallis H statistic (tie-adjusted)",
        degrees_of_freedom="df = k - 1",
        effect_size_quantity="Rank epsilon-squared",
        effect_size_definition=(
            "Truncated rank epsilon-squared from H, group count, and sample size."
        ),
        effect_size_ci_status="available",
        primary_estimate_ci_status="unavailable",
        ci_method=(
            "Percentile bootstrap for omnibus epsilon-squared; pairwise "
            "rank-biserial CIs unavailable in this release."
        ),
        assumptions=(
            "Independent observations within and between groups.",
            "Ordinal or continuous outcome supporting ranking.",
            ("At least 5 usable values per group for chi-square approximation validity."),
        ),
        diagnostics=(
            "Per-group sample sizes",
            "Tie counts",
        ),
        missing_data_policy=("Complete-case analysis on (group, outcome). Excluded rows recorded."),
        degenerate_data_behavior=(
            "Fewer than 5 values in any group or completely tied outcome "
            "returns unavailable status."
        ),
        multiplicity_policy=(
            "Complete family of k(k-1)/2 Dunn pairwise comparisons with Holm "
            "step-down multiplicity adjustment."
        ),
        numerical_provenance="scipy.stats.kruskal for omnibus; multigroup.py for Dunn-Holm.",
        interpretation_limitations=(
            "NOT a universal test of medians; evaluates mean ranks.",
            ("Pairwise rank-biserial confidence intervals are currently unavailable."),
        ),
        audit_invariants=(
            "df == k - 1",
            "epsilon_squared in [0, 1]",
            "complete family of k*(k-1)//2 pairwise records present",
            "pairwise adjusted p_values in [0, 1]",
            "pairwise rank_biserial in [-1, 1]",
        ),
        validation_source=(
            "Kruskal & Wallis (1952); Dunn (1964); Holm (1979); verified in "
            "tests/test_multigroup_analysis.py."
        ),
    ),
    "pearson_correlation": MethodContract(
        method_id="pearson_correlation",
        name="Pearson product-moment correlation",
        scientific_question=(
            "Is there a nonzero linear association between two quantitative "
            "variables in the target population?"
        ),
        study_design="Single sample, bivariate continuous pairs",
        outcome_type="Continuous quantitative",
        predictor_type="Continuous quantitative",
        estimand="Population Pearson linear correlation coefficient (ρ)",
        primary_estimate="Sample Pearson correlation coefficient (r)",
        null_hypothesis="The population Pearson linear correlation is zero.",
        alternative_hypothesis="The population Pearson linear correlation is nonzero (two-sided).",
        null_value=0.0,
        null_quantity="Pearson r",
        test_statistic="Pearson r",
        degrees_of_freedom="None stored in result; inferential df = n - 2",
        effect_size_quantity="Pearson r",
        effect_size_definition="Signed linear correlation between the two quantitative variables.",
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method="Fisher-z asymptotic normal confidence interval.",
        assumptions=(
            "Independent observational pairs.",
            "Linear relationship between variables.",
            "Both variables continuous quantitative with nonzero variance.",
        ),
        diagnostics=(
            "Pairwise sample size",
            "Variance of each variable",
        ),
        missing_data_policy="Complete observation pairs. Excluded rows recorded.",
        degenerate_data_behavior=(
            "Zero variance in either variable or n < 3 pairs returns unavailable status."
        ),
        multiplicity_policy="Not applicable (single bivariate pair).",
        numerical_provenance=(
            "StatisticalAnalyzer.analyze_all correlation profile (SciPy backend)."
        ),
        interpretation_limitations=(
            ("Measures linear association only; sensitive to outliers and nonlinear curvature."),
            "r² is shared linear variance, not causal effect.",
            "Confidence interval uses Fisher-z transform; valid for n > 3.",
        ),
        audit_invariants=(
            "r in [-1, 1]",
            "p_value in [0, 1]",
            "analyzed_rows == effective_pair_count",
            "analyzed_rows + excluded_rows == original_rows",
            "CI bounds in [-1, 1]",
            "CI lower <= upper",
        ),
        validation_source=(
            "Pearson (1895); SciPy cross-check; verified in tests/test_analyzer_correlation.py."
        ),
    ),
    "spearman_correlation": MethodContract(
        method_id="spearman_correlation",
        name="Spearman rank correlation",
        scientific_question=(
            "Is there a nonzero monotonic rank association between two variables in the population?"
        ),
        study_design="Single sample, bivariate ordered pairs",
        outcome_type="Ordered numeric (ordinal or continuous)",
        predictor_type="Ordered numeric (ordinal or continuous)",
        estimand="Population Spearman rank correlation coefficient (ρ_s)",
        primary_estimate="Sample Spearman rank correlation (r_s)",
        null_hypothesis="The population Spearman monotonic correlation is zero.",
        alternative_hypothesis=(
            "The population Spearman monotonic correlation is nonzero (two-sided)."
        ),
        null_value=0.0,
        null_quantity="Spearman rho",
        test_statistic="Spearman rho",
        degrees_of_freedom="None stored; inferential df = n - 2",
        effect_size_quantity="Spearman rho",
        effect_size_definition=(
            "Signed monotonic rank association between the two ordered variables."
        ),
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method="Deterministic paired-observation percentile bootstrap.",
        assumptions=(
            "Independent observational pairs.",
            "Ordinal or continuous scale supporting ranking.",
            "Monotonic relationship under evaluation.",
        ),
        diagnostics=(
            "Number of complete pairs",
            "Tie presence",
        ),
        missing_data_policy="Complete observation pairs. Excluded rows recorded.",
        degenerate_data_behavior="Completely tied data or n < 3 pairs returns unavailable status.",
        multiplicity_policy="Not applicable.",
        numerical_provenance="scipy.stats.spearmanr",
        interpretation_limitations=(
            "Measures monotonic association, not linearity.",
            "Does not establish causality.",
        ),
        audit_invariants=(
            "r_s in [-1, 1]",
            "p_value in [0, 1]",
            "bootstrap CI bounds in [-1, 1]",
            "CI lower <= upper",
        ),
        validation_source=(
            "Spearman (1904); SciPy cross-check; verified in tests/test_basic_inference.py."
        ),
    ),
    "kendall_tau_b": MethodContract(
        method_id="kendall_tau_b",
        name="Kendall's tau-b with inference",
        scientific_question=(
            "Is there a nonzero ordinal concordance between two ordered "
            "variables, adjusting for tied pairs?"
        ),
        study_design="Single sample, bivariate ordered pairs",
        outcome_type="Ordered numeric",
        predictor_type="Ordered numeric",
        estimand="Population Kendall's tau-b parameter (τ_b)",
        primary_estimate="Sample Kendall tau-b (τ_b)",
        null_hypothesis="The population Kendall’s tau-b is zero.",
        alternative_hypothesis="The population Kendall’s tau-b is nonzero (two-sided).",
        null_value=0.0,
        null_quantity="Kendall tau-b",
        test_statistic="Kendall tau-b test statistic",
        degrees_of_freedom="None (not applicable)",
        effect_size_quantity="Kendall's tau-b",
        effect_size_definition="Pairwise ordinal concordance adjusted for ties.",
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method="Paired-observation percentile bootstrap.",
        assumptions=(
            "Independent observational pairs.",
            "Ordinal scale supporting ranking.",
            "Tie correction via tau-b formulation.",
        ),
        diagnostics=(
            "Tie count in each variable",
            "Concordant and discordant counts",
        ),
        missing_data_policy="Complete observation pairs. Excluded rows recorded.",
        degenerate_data_behavior=(
            "Fewer than 3 complete pairs or constant variable returns unavailable status."
        ),
        multiplicity_policy="Not applicable.",
        numerical_provenance="scipy.stats.kendalltau(variant='b')",
        interpretation_limitations=(
            "Measures ordinal concordance, NOT percentage variance explained.",
            "Does not establish causality.",
        ),
        audit_invariants=(
            "tau_b in [-1, 1]",
            "p_value in [0, 1]",
            "ties variant is 'b'",
            "bootstrap CI bounds in [-1, 1]",
            "CI lower <= upper",
        ),
        validation_source=(
            "Kendall (1938); SciPy cross-check; verified in "
            "tests/test_binary_extended_association.py."
        ),
    ),
    "point_biserial_correlation": MethodContract(
        method_id="point_biserial_correlation",
        name="Point-biserial correlation with inference",
        scientific_question=(
            "Is there a nonzero linear association between a continuous "
            "variable and an explicitly coded binary indicator?"
        ),
        study_design=("Independent observations with one continuous and one binary variable"),
        outcome_type="Continuous quantitative",
        predictor_type="Binary categorical factor (explicit 0/1 coding)",
        estimand="Population point-biserial correlation (r_pb)",
        primary_estimate="Sample point-biserial correlation (r_pb)",
        null_hypothesis="The population point-biserial correlation is zero.",
        alternative_hypothesis="The population point-biserial correlation is nonzero (two-sided).",
        null_value=0.0,
        null_quantity="point-biserial correlation",
        test_statistic="Point-biserial r",
        degrees_of_freedom="df = n - 2",
        effect_size_quantity="Point-biserial r",
        effect_size_definition="Pearson r between continuous values and the 0/1 binary indicator.",
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method="Paired-observation percentile bootstrap.",
        assumptions=(
            "Independent observational pairs.",
            "Continuous distribution for the quantitative variable.",
            ("Exactly two levels in binary variable with researcher-declared positive level."),
        ),
        diagnostics=(
            "Binary class frequencies",
            "Continuous variable spread",
        ),
        missing_data_policy=(
            "Complete-case analysis on (binary, continuous). Excluded rows recorded."
        ),
        degenerate_data_behavior=(
            "Single class observed or zero continuous variance returns unavailable status."
        ),
        multiplicity_policy="Not applicable.",
        numerical_provenance="scipy.stats.pointbiserialr",
        interpretation_limitations=(
            "Symmetric association measure; positive level determines sign.",
            "Not a substitute for a group mean difference test.",
            "Does not establish causality.",
        ),
        audit_invariants=(
            "r_pb in [-1, 1]",
            "p_value in [0, 1]",
            "binary_encoding records positive and negative levels",
            "bootstrap CI bounds in [-1, 1]",
            "CI lower <= upper",
        ),
        validation_source=(
            "Lev (1949); SciPy cross-check; verified in tests/test_binary_extended_association.py."
        ),
    ),
    "partial_pearson_correlation": MethodContract(
        method_id="partial_pearson_correlation",
        name="Partial Pearson correlation",
        scientific_question=(
            "Is there a nonzero linear association between two continuous "
            "variables after linearly controlling for declared covariates?"
        ),
        study_design=(
            "Independent observations with two focal continuous variables and "
            "k >= 1 quantitative controls"
        ),
        outcome_type="Continuous quantitative",
        predictor_type="Continuous quantitative + declared quantitative controls",
        estimand=("Population partial Pearson correlation controlling for covariates (ρ_XY.Z)"),
        primary_estimate="Sample partial Pearson correlation (r_XY.Z)",
        null_hypothesis=(
            "The partial population Pearson correlation is zero, controlling "
            "for the declared covariates."
        ),
        alternative_hypothesis="The partial population Pearson correlation is nonzero (two-sided).",
        null_value=0.0,
        null_quantity="partial Pearson correlation",
        test_statistic="Student's t = r_partial * sqrt((n - k - 2) / (1 - r_partial²))",
        degrees_of_freedom="df = n - k - 2",
        effect_size_quantity="Partial Pearson r",
        effect_size_definition=(
            "Pearson correlation of OLS residuals after regressing focal "
            "variables on declared controls."
        ),
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method="Complete-row model-refitting percentile bootstrap.",
        assumptions=(
            "Independent observational units.",
            "Linear relationship between focal variables and controls.",
            "Full-rank control design matrix.",
            "All focal and control variables quantitative.",
        ),
        diagnostics=(
            "Residualization rank",
            "VIF of controls",
            "Sample size relative to k + 3",
        ),
        missing_data_policy=(
            "Complete-case across focal variables and all controls. Excluded rows recorded."
        ),
        degenerate_data_behavior=(
            "Rank-deficient control matrix or n < k + 3 returns unavailable status."
        ),
        multiplicity_policy="Not applicable.",
        numerical_provenance=(
            "statsmodels.api.OLS residualization + NumPy correlation + SciPy t reference."
        ),
        interpretation_limitations=(
            ("Conditioning on controls does not establish that confounding has been removed."),
            "Control choice must be theoretically justified.",
            "Does not establish causality.",
        ),
        audit_invariants=(
            "r_partial in [-1, 1]",
            "p_value in [0, 1]",
            "df == analyzed_rows - len(controls) - 2",
            "test_statistic matches r * sqrt(df / (1 - r²))",
            "bootstrap CI bounds in [-1, 1]",
            "CI lower <= upper",
        ),
        validation_source=(
            "Yule (1907); statsmodels/pingouin cross-check; verified in "
            "tests/test_binary_extended_association.py."
        ),
    ),
    "pearson_chi_square": MethodContract(
        method_id="pearson_chi_square",
        name="Pearson chi-square test of independence",
        scientific_question=("Are two categorical variables independent in the target population?"),
        study_design="Independent observations, cross-classification table (r x c)",
        outcome_type="Categorical (nominal or ordinal)",
        predictor_type="Categorical (nominal or ordinal)",
        estimand=("Categorical independence / departure from multinomial product probabilities"),
        primary_estimate="Cramér's V",
        null_hypothesis="The two categorical variables are independent.",
        alternative_hypothesis="The two categorical variables are dependent / associated.",
        null_value=0.0,
        null_quantity="Cramér's V",
        test_statistic="Pearson chi-square statistic (X²)",
        degrees_of_freedom="df = (r - 1) * (c - 1)",
        effect_size_quantity="Cramér's V",
        effect_size_definition="Square root of X² divided by (N * min(r-1, c-1)).",
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method="Complete-row observation percentile bootstrap.",
        assumptions=(
            "Independent observations.",
            "Mutually exclusive and exhaustive categories.",
            ("Adequate expected cell counts (all expected cells >= 5 under PyAutoStat policy)."),
        ),
        diagnostics=(
            "Table of expected frequencies",
            "Percentage of cells with expected count < 5",
        ),
        missing_data_policy=(
            "Complete-case analysis on the two categorical variables. Excluded rows recorded."
        ),
        degenerate_data_behavior=(
            "Sparse tables (expected count < 5) fall back to Fisher's exact "
            "if 2x2, or return blocked status."
        ),
        multiplicity_policy="Not applicable.",
        numerical_provenance="scipy.stats.chi2_contingency(correction=False)",
        interpretation_limitations=(
            "Nonnegative measure without sign or direction.",
            "Cramér's V² is NOT percentage of variance explained.",
            "Does not establish causality.",
        ),
        audit_invariants=(
            "test_statistic >= 0",
            "p_value in [0, 1]",
            "df == (r - 1) * (c - 1)",
            "cramers_v in [0, 1]",
            "bootstrap CI bounds in [0, 1]",
            "CI lower <= upper",
        ),
        validation_source=(
            "Pearson (1900); Cramér (1946); SciPy cross-check; verified in "
            "tests/test_categorical.py."
        ),
    ),
    "fisher_exact": MethodContract(
        method_id="fisher_exact",
        name="Fisher's exact test",
        scientific_question=(
            "Are two binary categorical variables independent under fixed margins?"
        ),
        study_design="Independent observations, exactly 2x2 contingency table",
        outcome_type="Binary categorical",
        predictor_type="Binary categorical",
        estimand="Binary independence / odds ratio under hypergeometric distribution",
        primary_estimate="Sample odds ratio (ad / bc)",
        null_hypothesis="The two binary categorical variables are independent.",
        alternative_hypothesis=(
            "The two binary categorical variables are dependent (two-sided odds ratio != 1)."
        ),
        null_value=1.0,
        null_quantity="odds ratio",
        test_statistic="Hypergeometric exact probability / sample odds ratio",
        degrees_of_freedom="None (exact test)",
        effect_size_quantity="Sample odds ratio",
        effect_size_definition=(
            "Cross-product ratio (ad / bc) based on first-observed row and column order."
        ),
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method=(
            "log-Wald confidence interval for sample odds-ratio estimator "
            "(unavailable when any cell is zero)."
        ),
        assumptions=(
            "Independent observations.",
            "Exactly 2x2 contingency table.",
            "Fixed or conditioned marginal totals.",
        ),
        diagnostics=(
            "Observed 2x2 contingency counts",
            "Marginal totals",
            "Zero-cell detection",
        ),
        missing_data_policy=(
            "Complete-case analysis on the two binary variables. Excluded rows recorded."
        ),
        degenerate_data_behavior=(
            "Empty margin returns unavailable status; zero cells recorded "
            "with explicit status (zero, positive_infinity)."
        ),
        multiplicity_policy="Not applicable.",
        numerical_provenance="scipy.stats.fisher_exact(alternative='two-sided')",
        interpretation_limitations=(
            "Supported strictly for 2x2 tables.",
            "Odds ratio direction depends on category level order.",
            (
                "Odds-ratio CI uses asymptotic log-Wald method matching the sample "
                "OR estimator; unavailable when any cell count is zero (no silent "
                "continuity corrections)."
            ),
            "Does not establish causality.",
        ),
        audit_invariants=(
            "observed_counts is a 2x2 matrix",
            "degrees_of_freedom is None",
            "p_value in [0, 1]",
            "sample odds ratio >= 0 when finite",
            "sample odds ratio CI lower > 0 and finite when available",
            "CI lower <= upper",
        ),
        validation_source=(
            "Fisher (1935); SciPy cross-check; verified in tests/test_basic_inference.py."
        ),
    ),
    "mcnemar": MethodContract(
        method_id="mcnemar",
        name="McNemar's test for paired binary outcomes",
        scientific_question=(
            "Do paired binary marginal event probabilities differ across two "
            "conditions on the same units?"
        ),
        study_design=(
            "Paired design with explicit unit ID, exactly 2 conditions, and binary outcome"
        ),
        outcome_type="Binary (with declared event level)",
        predictor_type="Condition variable (2 levels) + Unit ID",
        estimand=("Paired marginal event probability difference (P(Y₁ = 1) - P(Y₂ = 1))"),
        primary_estimate="Sample paired event proportion difference ((b - c) / n)",
        null_hypothesis=("The paired binary outcome proportions are equal (marginal homogeneity)."),
        alternative_hypothesis="The paired binary outcome proportions are unequal (two-sided).",
        null_value=0.0,
        null_quantity="paired proportion difference",
        test_statistic="Exact binomial test on discordant pairs (b ~ Binomial(b + c, 0.5))",
        degrees_of_freedom="None (exact test)",
        effect_size_quantity="Paired proportion difference; matched odds ratio (b / c)",
        effect_size_definition="First minus second condition event proportion difference.",
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method=("Paired-unit resampling percentile bootstrap for proportion difference."),
        assumptions=(
            "Explicit unit identity across conditions.",
            "Independent paired units.",
            ("Exactly two conditions and binary outcome with declared event level."),
            "Discordant pairs are informative for marginal homogeneity.",
        ),
        diagnostics=(
            "2x2 transition table",
            "Discordant pair counts b and c",
            "Total discordant pairs",
        ),
        missing_data_policy=(
            "Incomplete units (missing either condition) are excluded; "
            "incomplete unit count recorded."
        ),
        degenerate_data_behavior="Zero discordant pairs (b + c = 0) returns unavailable status.",
        multiplicity_policy="Not applicable.",
        numerical_provenance="scipy.stats.binomtest(n=b+c, p=0.5, alternative='two-sided')",
        interpretation_limitations=(
            "Evaluates marginal homogeneity based on discordant pairs only.",
            "Concordant pairs provide no information about the contrast.",
            "Does not establish causality.",
        ),
        audit_invariants=(
            "2 * complete_pairs == analyzed_rows",
            "transition table counts sum to complete_pairs",
            "primary_estimate == (b - c) / complete_pairs",
            "p_value in [0, 1]",
            "bootstrap CI bounds in [-1, 1]",
            "CI lower <= upper",
        ),
        validation_source=(
            "McNemar (1947); SciPy cross-check; verified in "
            "tests/test_binary_extended_association.py."
        ),
    ),
    "linear_regression": MethodContract(
        method_id="linear_regression",
        name="Ordinary least-squares linear regression",
        scientific_question=(
            "How is the conditional mean of a continuous outcome associated "
            "with a set of predictors under a linear model?"
        ),
        study_design=("Independent observations, continuous outcome, one or more predictors"),
        outcome_type="Continuous quantitative",
        predictor_type="One or more continuous, binary, nominal, or ordinal predictors",
        estimand=(
            "Conditional mean coefficients (β_j) and in-sample explained "
            "variance (R²) under declared model"
        ),
        primary_estimate="Sample R²; sample OLS slope coefficients (b_j)",
        null_hypothesis="All non-intercept population slope coefficients are zero.",
        alternative_hypothesis="At least one non-intercept slope coefficient is nonzero.",
        null_value=0.0,
        null_quantity="R-squared",
        test_statistic="Model omnibus F statistic; individual coefficient t statistics",
        degrees_of_freedom="Two dfs: [model df = p, residual df = n - p - 1]",
        effect_size_quantity="In-sample R²; continuous standardized betas",
        effect_size_definition=(
            "Observed outcome variance accounted for by the fitted in-sample model."
        ),
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method=(
            "Analytical Student-t intervals for coefficients (classical or "
            "HC3 covariance); case-resampling percentile bootstrap CI for in-sample R-squared."
        ),
        assumptions=(
            "Independent observational units.",
            "Linear conditional-mean specification.",
            ("Full-rank design matrix with positive residual degrees of freedom."),
            ("Appropriate residual behavior for the selected covariance (classical or HC3)."),
        ),
        diagnostics=(
            "VIF for multicollinearity",
            "Breusch-Pagan test",
            "Residual normality",
            "Cook's distance",
        ),
        missing_data_policy=(
            "Complete-case analysis across outcome and all predictors. Excluded rows recorded."
        ),
        degenerate_data_behavior="Rank-deficient design or n <= p + 1 returns unavailable status.",
        multiplicity_policy=(
            "Omnibus model test; individual coefficient inferences reported "
            "without automatic multiplicity adjustment."
        ),
        numerical_provenance="statsmodels.api.OLS",
        interpretation_limitations=(
            "Additive linear main effects only; associations are noncausal.",
            ("R² is an in-sample descriptive fit, NOT validated out-of-sample prediction."),
            (
                "In-sample R² CI uses case-resampling percentile bootstrap; "
                "reflects in-sample fit uncertainty, not predictive performance."
            ),
        ),
        audit_invariants=(
            "r_squared in [0, 1]",
            "p_value in [0, 1]",
            "residual_df == analyzed_rows - parameter_count",
            "coefficient analytical CI contains estimate",
            "CI lower <= upper",
            "r_squared CI bounds in [0, 1]",
            "r_squared CI lower <= upper",
            "design_matrix.full_rank is True",
        ),
        validation_source=(
            "Legendre (1805); Gauss (1809); statsmodels cross-check; verified "
            "in tests/test_regression_workflow.py."
        ),
    ),
    "logistic_regression": MethodContract(
        method_id="logistic_regression",
        name="Binary logistic regression",
        scientific_question=(
            "How is the log-odds (and odds ratio) of a binary event "
            "associated with predictors under a logistic model?"
        ),
        study_design="Independent observations, binary outcome, one or more predictors",
        outcome_type="Binary (with declared event level)",
        predictor_type="One or more continuous, binary, nominal, or ordinal predictors",
        estimand=(
            "Conditional log-odds coefficients (β_j) and odds ratios (OR_j = "
            "exp(β_j)) under declared model"
        ),
        primary_estimate=(
            "Maximum likelihood log-odds coefficients (b_j) and odds ratios (exp(b_j))"
        ),
        null_hypothesis="All non-intercept population log-odds coefficients are zero.",
        alternative_hypothesis="At least one non-intercept log-odds coefficient is nonzero.",
        null_value=0.0,
        null_quantity="log-odds coefficients",
        test_statistic=("Model likelihood ratio test statistic; coefficient Wald z statistics"),
        degrees_of_freedom="Two dfs: [model df = p, residual df = n - p - 1]",
        effect_size_quantity="Odds ratios per predictor; McFadden pseudo-R²",
        effect_size_definition=(
            "Odds ratios exp(b_j); McFadden pseudo-R² = 1 - ln L_full / ln L_null."
        ),
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method=(
            "Analytical Wald z intervals on log-odds scale, exponentiated for "
            "odds ratios (NOT bootstrap intervals)."
        ),
        assumptions=(
            "Independent observational units.",
            "Binary outcome with researcher-declared event level.",
            "Linear relationship between predictors and log-odds.",
            "Absence of complete or quasi-complete separation.",
            "Full-rank design matrix; at least 10 complete cases.",
        ),
        diagnostics=(
            "Model convergence",
            "Event and non-event counts",
            "Condition number",
            "Likelihood ratio test",
        ),
        missing_data_policy=(
            "Complete-case analysis across outcome and all predictors. Excluded rows recorded."
        ),
        degenerate_data_behavior=(
            "Complete separation, single outcome class, or non-convergence "
            "returns unavailable status."
        ),
        multiplicity_policy=(
            "Likelihood ratio omnibus test; individual Wald z tests per coefficient."
        ),
        numerical_provenance="statsmodels.api.Logit (MLE fit, Wald inference).",
        interpretation_limitations=(
            ("Odds ratios describe relative odds, NOT absolute probabilities or causal effects."),
            ("McFadden pseudo-R² is a likelihood fit index, not directly comparable with OLS R²."),
            "No classification cutoff or prediction metric is applied.",
        ),
        audit_invariants=(
            "odds_ratio == exp(beta)",
            "odds_ratio_ci == [exp(lower), exp(upper)]",
            "mcfadden_r2 in [0, 1]",
            "event_count + non_event_count == analyzed_rows",
            "AIC and BIC match likelihood formulas",
            "design_matrix.full_rank is True",
        ),
        validation_source=(
            "Cox (1958); statsmodels cross-check; verified in tests/test_regression_workflow.py."
        ),
    ),
    "cronbach_alpha": MethodContract(
        method_id="cronbach_alpha",
        name="Cronbach's alpha scale reliability",
        scientific_question=(
            "What is the internal consistency of a researcher-declared set of multi-item responses?"
        ),
        study_design="Multi-item survey or psychometric scale (2+ numeric/ordinal items)",
        outcome_type="Multiple numeric or ordinal items",
        predictor_type="None (scale measurement model)",
        estimand="Scale internal consistency coefficient (α)",
        primary_estimate="Sample Cronbach's alpha (α̂)",
        null_hypothesis="None (descriptive measurement property; non-inferential).",
        alternative_hypothesis=(
            "None (non-inferential scale reliability; no alternative hypothesis tested)"
        ),
        null_value=None,
        null_quantity=None,
        test_statistic="None (descriptive index)",
        degrees_of_freedom="None",
        effect_size_quantity="Cronbach's alpha",
        effect_size_definition="k/(k-1) * (1 - Σ Var(Xᵢ) / Var(Σ Xᵢ)).",
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method="Respondent-row percentile bootstrap.",
        assumptions=(
            "Researcher-declared scale membership and item set.",
            ("Items scored in positive direction (or explicit reverse scoring declared)."),
            "At least two numeric items and two complete respondents.",
            "Positive finite total-score sample variance.",
            "Unidimensionality is assumed by the index, NOT verified.",
        ),
        diagnostics=(
            "Corrected item-total correlations",
            "Alpha-if-item-deleted",
            "Inter-item correlation matrix",
        ),
        missing_data_policy=(
            "Complete-case analysis across all scale items; per-item missing counts reported."
        ),
        degenerate_data_behavior=(
            "Zero total score variance, < 2 items, or < 2 complete "
            "respondents returns unavailable status."
        ),
        multiplicity_policy="Not applicable.",
        numerical_provenance=(
            "Explicit NumPy sample covariance and correlation formula in reliability.py."
        ),
        interpretation_limitations=(
            "Non-inferential; does NOT produce a hypothesis test or p-value.",
            ("Alpha does NOT prove unidimensionality, validity, or temporal stability."),
            ("No universal adequacy cutoff; negative alpha is preserved when observed."),
        ),
        audit_invariants=(
            "p_value is None",
            "test_statistic is None",
            "alpha matches covariance formula",
            "item_statistics has one record per declared item",
            "corrected_total excludes focal item",
            "bootstrap CI lower <= upper",
        ),
        validation_source=(
            "Cronbach (1951); Nunnally & Bernstein (1994); verified in "
            "tests/test_reliability_workflow.py."
        ),
    ),
    "repeated_measures_anova": MethodContract(
        method_id="repeated_measures_anova",
        name="One-way repeated-measures ANOVA with Greenhouse-Geisser correction",
        scientific_question=(
            "Do repeated-condition population means differ across three or "
            "more conditions on the same units?"
        ),
        study_design=(
            "Repeated measures with explicit unit ID, 3+ conditions, and long-format panel"
        ),
        outcome_type="Continuous quantitative",
        predictor_type="Condition variable (3+ levels) + Unit ID",
        estimand=(
            "Repeated-condition population mean equality; pairwise condition mean differences"
        ),
        primary_estimate=(
            "Condition sample means; partial eta-squared (η_p²); pairwise mean differences"
        ),
        null_hypothesis="All repeated-condition population means are equal.",
        alternative_hypothesis="At least one repeated-condition population mean differs.",
        null_value=0.0,
        null_quantity="condition means",
        test_statistic="Repeated-measures F = MS_condition / MS_error",
        degrees_of_freedom="Uncorrected [k-1, (k-1)(n-1)]; GG-corrected [ε(k-1), ε(k-1)(n-1)]",
        effect_size_quantity="Partial eta-squared (η_p²)",
        effect_size_definition=(
            "Condition sum of squares divided by condition sum of squares "
            "plus error sum of squares (partial eta-squared)."
        ),
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method=(
            "Exact noncentral-F inversion confidence interval for partial eta-squared "
            "(using uncorrected F and df); analytical paired-t intervals and exact "
            "noncentral-t dz intervals for pairwise contrasts."
        ),
        assumptions=(
            "Explicit repeated observational units across conditions.",
            "Units independent of other units.",
            "Continuous outcome / mean target.",
            "One observation per unit-condition.",
            "Complete panel across declared conditions.",
            (
                "Within-subject sphericity (Greenhouse-Geisser correction "
                "applied when rejected or uncomputable)."
            ),
        ),
        diagnostics=(
            "Mauchly's test of sphericity (W, χ², p)",
            "Greenhouse-Geisser epsilon",
            "Condition means and SDs",
        ),
        missing_data_policy=(
            "Incomplete units (missing any condition) are excluded; "
            "incomplete unit count recorded. No wide-to-long guessing."
        ),
        degenerate_data_behavior=(
            "Fewer than 3 complete units, < 3 conditions, or zero "
            "within-subject variance returns unavailable status."
        ),
        multiplicity_policy=(
            "Complete family of k(k-1)/2 pairwise paired-t tests with Holm "
            "step-down multiplicity adjustment."
        ),
        numerical_provenance=(
            "Explicit matrix decomposition in repeated_measures.py (Mauchly "
            "1940, Greenhouse & Geisser 1959)."
        ),
        interpretation_limitations=(
            "Mauchly non-rejection does NOT prove sphericity.",
            ("Greenhouse-Geisser correction applied when sphericity is rejected or uncomputable."),
            (
                "Partial eta-squared CI uses exact noncentral-F inversion on uncorrected F "
                "and degrees of freedom; GG correction modifies inferential p-values but not "
                "the observed SS-based effect."
            ),
            "Does not establish causality.",
        ),
        audit_invariants=(
            "ss_total == ss_condition + ss_subject + ss_error",
            "partial_eta_squared == ss_condition / (ss_condition + ss_error)",
            "partial_eta_squared in [0, 1]",
            "partial_eta_squared CI in [0, 1]",
            "partial_eta_squared CI lower <= upper",
            "gg_epsilon in [1/(k-1), 1]",
            "corrected dfs == epsilon * uncorrected dfs",
            "corrected p matches F and corrected dfs",
            "complete family of k*(k-1)//2 pairwise records present",
            "pairwise adjusted p_values in [0, 1]",
        ),
        validation_source=(
            "Andy Field (2012) textbook Bushtucker benchmark; statsmodels "
            "AnovaRM cross-check; verified in "
            "tests/test_repeated_measures.py."
        ),
    ),
    "friedman_test": MethodContract(
        method_id="friedman_test",
        name="Friedman rank-sum test with Wilcoxon-Holm follow-up",
        scientific_question=(
            "Do within-unit rank distributions differ across three or more "
            "repeated conditions on the same units?"
        ),
        study_design=(
            "Repeated measures with explicit unit ID, 3+ conditions, and long-format panel"
        ),
        outcome_type="Ordered numeric (ordinal or continuous)",
        predictor_type="Condition variable (3+ levels) + Unit ID",
        estimand=(
            "Within-unit rank distribution differences across conditions; "
            "Kendall's W rank concordance"
        ),
        primary_estimate=(
            "Kendall's W; condition rank medians; pairwise rank-biserial correlations"
        ),
        null_hypothesis="The repeated-condition within-unit rank distributions are equal.",
        alternative_hypothesis=(
            "At least one condition tends to yield higher/lower within-unit ranks than another."
        ),
        null_value=0.0,
        null_quantity="within-unit rank distributions",
        test_statistic="Friedman's Q statistic (chi-square approximation)",
        degrees_of_freedom="df = k - 1",
        effect_size_quantity="Kendall's W",
        effect_size_definition="Friedman Q divided by n*(k - 1) (Kendall's W rank concordance).",
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method=(
            "Participant-block percentile bootstrap for Kendall's W; "
            "paired-observation percentile bootstrap for pairwise rank-biserial contrasts."
        ),
        assumptions=(
            "Explicit repeated observational units across conditions.",
            "Units independent of other units.",
            "Meaningful rank ordering across conditions.",
            "One observation per unit-condition.",
            "Complete panel across declared conditions.",
        ),
        diagnostics=(
            "Condition medians and IQRs",
            "Tie presence across conditions",
        ),
        missing_data_policy=(
            "Complete-case panel across all declared conditions; incomplete "
            "units excluded and recorded."
        ),
        degenerate_data_behavior=(
            "Fewer than 3 complete units, < 3 conditions, or zero within-unit "
            "rank variance returns unavailable status."
        ),
        multiplicity_policy=(
            "Complete family of k(k-1)/2 pairwise paired Wilcoxon signed-rank "
            "tests with Holm step-down multiplicity adjustment."
        ),
        numerical_provenance=(
            "Explicit within-unit rank algorithm in repeated_measures.py "
            "(Friedman 1937, Kendall 1939)."
        ),
        interpretation_limitations=(
            ("Friedman evaluates within-unit rank distributions, NOT a universal test of medians."),
            ("Kendall's W is rank concordance, NOT percentage of variance explained."),
            (
                "Kendall's W and pairwise rank-biserial intervals use participant/paired "
                "percentile bootstrap; approximate uncertainty intervals, not significance tests."
            ),
        ),
        audit_invariants=(
            "df == k - 1",
            "kendalls_w == Q / (n * (k - 1))",
            "kendalls_w in [0, 1]",
            "kendalls_w CI in [0, 1]",
            "kendalls_w CI lower <= upper",
            "complete family of k*(k-1)//2 pairwise records present",
            "pairwise adjusted p_values in [0, 1]",
            "pairwise rank_biserial in [-1, 1]",
        ),
        validation_source=(
            "Friedman (1937); Kendall & Babington Smith (1939); SciPy "
            "cross-check; verified in tests/test_repeated_measures.py."
        ),
    ),
    "two_way_anova": MethodContract(
        method_id="two_way_anova",
        name="Two-way factorial ANOVA",
        scientific_question=(
            "Do the marginal means of a continuous outcome differ across levels of Factor A "
            "or Factor B, or does the effect of Factor A depend on the level of Factor B "
            "(interaction)?"
        ),
        study_design=(
            "Independent observations across exactly two categorical factors "
            "(fully crossed factorial design)"
        ),
        outcome_type="Continuous quantitative",
        predictor_type="Two categorical factors (at least 2 levels each)",
        estimand="Factorial mean model parameters, marginal means, and contrast differences",
        primary_estimate="Model marginal means, cell means, and partial eta-squared",
        null_hypothesis=(
            "Factor A, Factor B, and their interaction have zero effect on the outcome."
        ),
        alternative_hypothesis="At least one factor main effect or interaction effect is nonzero.",
        null_value=0.0,
        null_quantity="main effects and interaction",
        test_statistic="F-statistic for Factor A, Factor B, and Factor A x Factor B interaction",
        degrees_of_freedom=(
            "Numerator df for each term: (I-1), (J-1), (I-1)(J-1); Denominator df: N - I*J"
        ),
        effect_size_quantity="partial_eta_squared",
        effect_size_definition="SS_term / (SS_term + SS_error) for each model term",
        effect_size_ci_status="available",
        primary_estimate_ci_status="available",
        ci_method="Exact inversion of cumulative noncentral F distribution via SciPy",
        assumptions=(
            "Independent observations across all units.",
            "Exactly two declared categorical grouping factors.",
            "Continuous quantitative outcome measurement.",
            "Fully crossed design with all factor combination cells populated (no empty cells).",
            "Approximately normal model errors / residuals.",
            "Homoscedastic residual variance across factor cells.",
        ),
        diagnostics=(
            "Residual sample size, degrees of freedom, MSE, and RMSE",
            "Shapiro-Wilk and D'Agostino-Pearson residual normality tests",
            "Levene / Brown-Forsythe median-centered test for equal cell variances",
            "Cell counts and empty cell detection",
            "Design matrix rank",
        ),
        missing_data_policy=(
            "Complete-case analysis across outcome, factor_a, and factor_b; "
            "rows with missing values in any of these three variables are excluded and recorded."
        ),
        degenerate_data_behavior=(
            "Fewer than 2 levels in either factor, empty cells, nonpositive residual degrees of "
            "freedom, constant outcome, or nonfinite values returns unsupported/blocked status."
        ),
        multiplicity_policy=(
            "Simple effects of A within B, simple effects of B within A, and marginal mean "
            "pairwise contrasts are adjusted within their respective families using Holm "
            "step-down procedure. Pointwise Student-t confidence intervals have "
            "multiplicity_adjusted=False."
        ),
        numerical_provenance=(
            "Sum-to-zero contrast coding with ordinary least squares via statsmodels.api.OLS "
            "and scipy.stats F-distribution."
        ),
        interpretation_limitations=(
            "Factorial ANOVA evaluates mean differences and interaction under the declared model; "
            "it does NOT establish causality.",
            "Partial eta-squared measures variance accounted for relative to term plus residual "
            "variance, NOT total variance explained.",
            "A nonsignificant interaction does NOT prove that effects are identical across levels.",
            "Significant main effects cannot be assumed uniform across the other factor when "
            "interaction is present.",
            "Follow-up confidence intervals are pointwise, not simultaneous.",
        ),
        audit_invariants=(
            "terms include factor_a, factor_b, interaction, and residual",
            "df are positive and coherent: df_A == I - 1, df_B == J - 1, "
            "df_AB == (I - 1) * (J - 1), df_error == N - I * J",
            "SS values are finite and non-negative",
            "MS == SS / df",
            "F == MS_term / MS_error",
            "p-values in [0, 1]",
            "partial_eta_squared == SS_term / (SS_term + SS_error)",
            "partial_eta_squared in [0, 1]",
            "partial_eta_squared CI in [0, 1] with lower <= upper",
            "cell_summaries present for all I * J cells",
            "followup adjusted p_values >= raw p_values",
            "pointwise followup CI multiplicity_adjusted is False",
        ),
        validation_source=(
            "Validated against statsmodels OLS, car::Anova Type II / Type III in R, "
            "manual linear algebra, and SciPy noncentral-F distribution in "
            "tests/test_two_way_anova_validation.py."
        ),
    ),
}
