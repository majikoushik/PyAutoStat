# Statistical Method Contracts

This document establishes the authoritative, binding scientific contracts for every
inferential, model, and reliability method shipped in PyAutoStat.

Each method defines an explicit estimand, null/alternative hypotheses, primary estimate,
effect size, uncertainty/confidence-interval status, design and distributional assumptions,
missing-data policy, degenerate-data behavior, multiplicity policy, numerical backend,
audit invariants, and independent reference validation sources.

## Scientific Contract Principles

1. **Estimand Integrity**: The estimand is the scientific quantity under investigation, not the test name. Assumption diagnostics never silently redefine or substitute the estimand.
2. **Design Transparency**: Independence, pairing, clustering, and conditions cannot be inferred from numerical values and require explicit researcher declaration.
3. **Uncertainty Classification**: Uncertainty status is explicit across all methods:
   - `available`: Validated confidence interval is computed and reported.
   - `unavailable`: Interval is mathematically meaningful for the estimand/effect but not yet implemented.
   - `not_supported`: Interval is intentionally outside the package scope.
   - `not_applicable`: No meaningful interval exists for this quantity (e.g. omnibus tests with non-standardized statistics).
   - `uncomputable`: Numerical conditions (e.g., zero variance, singular covariance) prevent calculation.
4. **No Silent Alterations**: No automatic row deletion, outlier elimination, data imputation, alpha alteration, or test switching.
5. **Audit Invariants**: Every executed result is verified against mathematical and architectural invariants.

## Method Contract Summary Table

| Method ID | Method Name | Design | Estimand | Primary Estimate | Effect Size | Effect CI Status | Estimate CI Status |
|---|---|---|---|---|---|---|---|
| `welch_t` | Welch independent-samples t-test | Independent groups, exactly 2 groups | Population mean difference (first group mean minus second group mean) | Sample mean difference (x̄₁ - x̄₂) | Cohen's d | `available` | `available` |
| `student_t` | Student independent-samples t-test | Independent groups, exactly 2 groups | Population mean difference under equal variance assumption (μ₁ - μ₂) | Sample mean difference (x̄₁ - x̄₂) | Cohen's d | `available` | `available` |
| `paired_t` | Paired-samples t-test | Paired / repeated measures with exactly 2 conditions and explicit unit identifier | Population mean paired difference (μ_D = μ₁ - μ₂) | Sample mean paired difference (D̄ = (1/n) Σ dᵢ) | Cohen's dz | `available` | `available` |
| `one_sample_t` | One-sample t-test | Single sample against a fixed external scalar reference value | Population mean minus declared reference value (μ - μ₀) | Sample mean difference from reference (x̄ - μ₀) | One-sample Cohen's d | `available` | `available` |
| `mann_whitney_u` | Mann-Whitney U test | Independent groups, exactly 2 groups | Rank distribution separation / stochastic superiority (P(X > Y) - P(Y > X)) | Matched rank-biserial correlation (r_rb = 2U/(n₁n₂) - 1) | Rank-biserial correlation | `available` | `available` |
| `wilcoxon_signed_rank` | Paired Wilcoxon signed-rank test | Paired / repeated measures with exactly 2 conditions and explicit unit identifier | Matched-pairs signed-rank distribution center / location shift under symmetry | Matched-pairs rank-biserial correlation ((W⁺ - W⁻) / (W⁺ + W⁻)) | Matched-pairs rank-biserial correlation | `available` | `available` |
| `welch_anova` | Welch one-way ANOVA with Games-Howell comparisons | Independent groups, 3 or more groups | Heteroscedastic multi-group population mean equality; pairwise population mean differences | Group sample means; pairwise mean differences (x̄ⱼ - x̄ₘ) | Not applicable globally; unstandardized pairwise mean differences reported | `not_applicable` | `available` |
| `one_way_anova` | Standard classical one-way ANOVA with Tukey-Kramer comparisons | Independent groups, 3 or more groups | Homoscedastic population mean equality (μ₁ = ... = μ_k); pairwise mean differences | Omnibus F; pairwise mean differences | Eta-squared (η²) | `available` | `available` |
| `kruskal_wallis` | Kruskal-Wallis test with Dunn-Holm comparisons | Independent groups, 3 or more groups | Multi-group rank distribution equality / stochastic dominance | Rank epsilon-squared (ε²); pairwise mean-rank contrasts | Rank epsilon-squared | `available` | `unavailable` |
| `pearson_correlation` | Pearson product-moment correlation | Single sample, bivariate continuous pairs | Population Pearson linear correlation coefficient (ρ) | Sample Pearson correlation coefficient (r) | Pearson r | `available` | `available` |
| `spearman_correlation` | Spearman rank correlation | Single sample, bivariate ordered pairs | Population Spearman rank correlation coefficient (ρ_s) | Sample Spearman rank correlation (r_s) | Spearman rho | `available` | `available` |
| `kendall_tau_b` | Kendall's tau-b with inference | Single sample, bivariate ordered pairs | Population Kendall's tau-b parameter (τ_b) | Sample Kendall tau-b (τ_b) | Kendall's tau-b | `available` | `available` |
| `point_biserial_correlation` | Point-biserial correlation with inference | Independent observations with one continuous and one binary variable | Population point-biserial correlation (r_pb) | Sample point-biserial correlation (r_pb) | Point-biserial r | `available` | `available` |
| `partial_pearson_correlation` | Partial Pearson correlation | Independent observations with two focal continuous variables and k >= 1 quantitative controls | Population partial Pearson correlation controlling for covariates (ρ_XY.Z) | Sample partial Pearson correlation (r_XY.Z) | Partial Pearson r | `available` | `available` |
| `pearson_chi_square` | Pearson chi-square test of independence | Independent observations, cross-classification table (r x c) | Categorical independence / departure from multinomial product probabilities | Cramér's V | Cramér's V | `available` | `available` |
| `fisher_exact` | Fisher's exact test | Independent observations, exactly 2x2 contingency table | Binary independence / odds ratio under hypergeometric distribution | Sample odds ratio (ad / bc) | Sample odds ratio | `available` | `available` |
| `mcnemar` | McNemar's test for paired binary outcomes | Paired design with explicit unit ID, exactly 2 conditions, and binary outcome | Paired marginal event probability difference (P(Y₁ = 1) - P(Y₂ = 1)) | Sample paired event proportion difference ((b - c) / n) | Paired proportion difference; matched odds ratio (b / c) | `available` | `available` |
| `linear_regression` | Ordinary least-squares linear regression | Independent observations, continuous outcome, one or more predictors | Conditional mean coefficients (β_j) and in-sample explained variance (R²) under declared model | Sample R²; sample OLS slope coefficients (b_j) | In-sample R²; continuous standardized betas | `available` | `available` |
| `logistic_regression` | Binary logistic regression | Independent observations, binary outcome, one or more predictors | Conditional log-odds coefficients (β_j) and odds ratios (OR_j = exp(β_j)) under declared model | Maximum likelihood log-odds coefficients (b_j) and odds ratios (exp(b_j)) | Odds ratios per predictor; McFadden pseudo-R² | `available` | `available` |
| `cronbach_alpha` | Cronbach's alpha scale reliability | Multi-item survey or psychometric scale (2+ numeric/ordinal items) | Scale internal consistency coefficient (α) | Sample Cronbach's alpha (α̂) | Cronbach's alpha | `available` | `available` |
| `repeated_measures_anova` | One-way repeated-measures ANOVA with Greenhouse-Geisser correction | Repeated measures with explicit unit ID, 3+ conditions, and long-format panel | Repeated-condition population mean equality; pairwise condition mean differences | Condition sample means; partial eta-squared (η_p²); pairwise mean differences | Partial eta-squared (η_p²) | `available` | `available` |
| `friedman_test` | Friedman rank-sum test with Wilcoxon-Holm follow-up | Repeated measures with explicit unit ID, 3+ conditions, and long-format panel | Within-unit rank distribution differences across conditions; Kendall's W rank concordance | Kendall's W; condition rank medians; pairwise rank-biserial correlations | Kendall's W | `available` | `available` |

---

## Detailed Method Contracts

### `welch_t` — Welch independent-samples t-test

- **Scientific Question**: Do two independent populations differ in their mean continuous outcome?
- **Study Design**: Independent groups, exactly 2 groups
- **Outcome Type**: Continuous quantitative
- **Predictor / Factor Type**: Binary categorical factor
- **Estimand**: Population mean difference (first group mean minus second group mean)
- **Primary Estimate**: Sample mean difference (x̄₁ - x̄₂)
- **Null Hypothesis (H₀)**: The two population means are equal.
- **Alternative Hypothesis (H₁)**: The two population means are unequal (two-sided).
- **Null Value**: `0.0` (mean difference)
- **Test Statistic**: Welch's t statistic
- **Degrees of Freedom**: Welch-Satterthwaite approximation
- **Effect Size Quantity**: Cohen's d
- **Effect Size Definition**: First minus second group mean, divided by the pooled sample SD.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Analytical Student-t interval for mean difference; percentile bootstrap for Cohen's d.
- **Required Assumptions**:
  - Independent observations within and between groups.
  - Continuous outcome scale.
  - Approximate normality of group sampling distributions or sufficient sample size.
  - Equal population variances are NOT assumed.
- **Diagnostics**:
  - Shapiro-Wilk normality per group
  - Group sample sizes
  - Levene variance ratio (advisory)
- **Missing Data Policy**: Complete-case analysis on (group, outcome). Excluded rows recorded.
- **Degenerate Data Behavior**: Zero variance in either group or n < 2 in any group returns unavailable status.
- **Multiplicity Policy**: Not applicable (single contrast).
- **Numerical Provenance**: scipy.stats.ttest_ind(equal_var=False)
- **Interpretation Limitations**:
  - Mean difference is descriptive and noncausal without random assignment.
  - Non-rejection of the null hypothesis does not prove equal population means.
  - Diagnostic non-rejection does not prove normality.
- **Audit Invariants**:
  - n₁ + n₂ == analyzed_rows
  - analyzed_rows + excluded_rows == original_rows
  - contrast definition is first group minus second group
  - p_value in [0, 1]
  - analytical confidence interval contains sample mean difference
  - CI lower <= upper
  - Cohen's d sign matches mean difference sign when difference is nonzero
- **Independent Validation Source**: Welch (1947); SciPy reference cross-check; verified against manual fixtures in tests/test_statistical_correctness.py.

### `student_t` — Student independent-samples t-test

- **Scientific Question**: Do two independent populations with equal variance differ in their mean continuous outcome?
- **Study Design**: Independent groups, exactly 2 groups
- **Outcome Type**: Continuous quantitative
- **Predictor / Factor Type**: Binary categorical factor
- **Estimand**: Population mean difference under equal variance assumption (μ₁ - μ₂)
- **Primary Estimate**: Sample mean difference (x̄₁ - x̄₂)
- **Null Hypothesis (H₀)**: The two population means are equal.
- **Alternative Hypothesis (H₁)**: The two population means are unequal (two-sided).
- **Null Value**: `0.0` (mean difference)
- **Test Statistic**: Student's t statistic
- **Degrees of Freedom**: Pooled df = n₁ + n₂ - 2
- **Effect Size Quantity**: Cohen's d
- **Effect Size Definition**: First minus second group mean, divided by the pooled sample SD.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Analytical Student-t interval for mean difference; percentile bootstrap for Cohen's d.
- **Required Assumptions**:
  - Independent observations within and between groups.
  - Continuous outcome scale.
  - Equal population variances (homoscedasticity).
  - Approximate normality of group distributions or sufficient sample size.
- **Diagnostics**:
  - Levene test for equality of variance
  - Shapiro-Wilk normality per group
- **Missing Data Policy**: Complete-case analysis on (group, outcome). Excluded rows recorded.
- **Degenerate Data Behavior**: Zero pooled variance or n < 2 in any group returns unavailable status.
- **Multiplicity Policy**: Not applicable (single contrast).
- **Numerical Provenance**: scipy.stats.ttest_ind(equal_var=True)
- **Interpretation Limitations**:
  - Requires explicit scientific justification for equal population variance.
  - Levene test non-rejection does not prove equal variance.
  - Non-rejection does not prove equal population means.
- **Audit Invariants**:
  - n₁ + n₂ == analyzed_rows
  - df == n₁ + n₂ - 2
  - analyzed_rows + excluded_rows == original_rows
  - analytical confidence interval contains sample mean difference
  - CI lower <= upper
- **Independent Validation Source**: Student (1908); SciPy cross-check; verified in tests/test_statistical_correctness.py.

### `paired_t` — Paired-samples t-test

- **Scientific Question**: Is the mean of paired differences in a two-condition within-unit design zero?
- **Study Design**: Paired / repeated measures with exactly 2 conditions and explicit unit identifier
- **Outcome Type**: Continuous quantitative
- **Predictor / Factor Type**: Condition variable (2 levels) + Unit ID
- **Estimand**: Population mean paired difference (μ_D = μ₁ - μ₂)
- **Primary Estimate**: Sample mean paired difference (D̄ = (1/n) Σ dᵢ)
- **Null Hypothesis (H₀)**: The population mean paired difference is zero.
- **Alternative Hypothesis (H₁)**: The population mean paired difference is nonzero (two-sided).
- **Null Value**: `0.0` (mean paired difference)
- **Test Statistic**: Paired t statistic: t = D̄ / (s_D / √n)
- **Degrees of Freedom**: df = complete_pairs - 1
- **Effect Size Quantity**: Cohen's dz
- **Effect Size Definition**: Sample mean paired difference divided by the standard deviation of paired differences.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Analytical Student-t interval for mean paired difference; exact noncentral-t inversion for Cohen's dz.
- **Required Assumptions**:
  - Explicit paired or matched observational units.
  - Independent pairs across units.
  - One usable measurement per unit and condition.
  - Paired differences approximately normally distributed or sufficient sample size.
  - Nonzero finite variance of paired differences.
- **Diagnostics**:
  - Shapiro-Wilk normality on paired differences
  - Complete-pair accounting
- **Missing Data Policy**: Incomplete units (missing either condition) are excluded; incomplete unit count recorded. No row-adjacency pairing.
- **Degenerate Data Behavior**: Zero variance of paired differences or complete_pairs < 2 returns unavailable status.
- **Multiplicity Policy**: Not applicable (single contrast).
- **Numerical Provenance**: scipy.stats.ttest_rel
- **Interpretation Limitations**:
  - Mean paired difference is noncausal without condition randomization.
  - Cohen's dz CI uses exact noncentral-t inversion targeting the population standardized paired difference.
  - Non-rejection does not prove zero difference.
- **Audit Invariants**:
  - 2 * complete_pairs == analyzed_rows
  - df == complete_pairs - 1
  - contrast condition order matches declared condition order
  - analytical CI contains sample mean paired difference
  - CI lower <= upper
  - Cohen's dz CI lower <= upper
- **Independent Validation Source**: Fisher (1925); SciPy cross-check; verified in tests/test_paired_analysis.py.

### `one_sample_t` — One-sample t-test

- **Scientific Question**: Does the population mean of a single continuous variable differ from a declared reference value?
- **Study Design**: Single sample against a fixed external scalar reference value
- **Outcome Type**: Continuous quantitative
- **Predictor / Factor Type**: None (declared reference value)
- **Estimand**: Population mean minus declared reference value (μ - μ₀)
- **Primary Estimate**: Sample mean difference from reference (x̄ - μ₀)
- **Null Hypothesis (H₀)**: The population mean equals the declared reference value.
- **Alternative Hypothesis (H₁)**: The population mean does not equal the declared reference value (two-sided).
- **Null Value**: `0.0` (mean difference from reference)
- **Test Statistic**: Student's one-sample t statistic
- **Degrees of Freedom**: df = n - 1
- **Effect Size Quantity**: One-sample Cohen's d
- **Effect Size Definition**: Sample mean minus reference value, divided by the sample standard deviation.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Analytical Student-t interval for mean difference from reference; exact noncentral-t inversion for one-sample Cohen's d.
- **Required Assumptions**:
  - Independent observations.
  - Continuous outcome scale.
  - Finite, researcher-declared reference value.
  - Approximate normality of population or sufficient sample size for CLT.
  - Finite positive sample variance.
- **Diagnostics**:
  - Shapiro-Wilk normality on outcome
  - Sample size check
- **Missing Data Policy**: Complete-case analysis on outcome. Excluded rows recorded.
- **Degenerate Data Behavior**: Zero variance or n < 2 returns unavailable status.
- **Multiplicity Policy**: Not applicable.
- **Numerical Provenance**: scipy.stats.ttest_1samp
- **Interpretation Limitations**:
  - Validity depends on the scientific relevance and justification of the reference value.
  - One-sample Cohen's d CI uses exact noncentral-t inversion targeting the population standardized difference.
- **Audit Invariants**:
  - df == analyzed_rows - 1
  - metadata reference_value matches question specification
  - analytical CI contains sample mean difference from reference
  - CI lower <= upper
  - one-sample Cohen's d CI lower <= upper
- **Independent Validation Source**: Student (1908); SciPy cross-check; verified in tests/test_basic_inference.py.

### `mann_whitney_u` — Mann-Whitney U test

- **Scientific Question**: Do two independent populations differ in their underlying rank distributions (stochastic dominance)?
- **Study Design**: Independent groups, exactly 2 groups
- **Outcome Type**: Ordered numeric (ordinal or continuous)
- **Predictor / Factor Type**: Binary categorical factor
- **Estimand**: Rank distribution separation / stochastic superiority (P(X > Y) - P(Y > X))
- **Primary Estimate**: Matched rank-biserial correlation (r_rb = 2U/(n₁n₂) - 1)
- **Null Hypothesis (H₀)**: The two underlying rank distributions are equal.
- **Alternative Hypothesis (H₁)**: The two underlying rank distributions differ (two-sided).
- **Null Value**: `0.0` (rank-biserial correlation)
- **Test Statistic**: Mann-Whitney U statistic (ties handled via mid-ranks)
- **Degrees of Freedom**: None (not applicable to rank-sum test)
- **Effect Size Quantity**: Rank-biserial correlation
- **Effect Size Definition**: 2 times first-group U divided by n1*n2, minus 1; ties count half.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Independent within-group percentile bootstrap.
- **Required Assumptions**:
  - Independent observations within and between groups.
  - Ordinal or continuous outcome supporting meaningful ranking.
  - Equal distribution shapes are NOT assumed unless specifically testing medians.
- **Diagnostics**:
  - Group sample sizes
  - Tie counts and tie proportions
- **Missing Data Policy**: Complete-case analysis on (group, outcome). Excluded rows recorded.
- **Degenerate Data Behavior**: Completely tied data (zero rank variance) or n < 2 per group returns unavailable status.
- **Multiplicity Policy**: Not applicable.
- **Numerical Provenance**: scipy.stats.mannwhitneyu(method='auto')
- **Interpretation Limitations**:
  - Test of stochastic dominance / rank distributions, NOT a universal test of medians.
  - Median-difference interpretation strictly requires identical distribution shapes.
  - Does not establish causality.
- **Audit Invariants**:
  - n₁ + n₂ == analyzed_rows
  - degrees_of_freedom is None
  - p_value in [0, 1]
  - rank_biserial in [-1, 1]
  - bootstrap CI bounds in [-1, 1]
  - CI lower <= upper
- **Independent Validation Source**: Mann & Whitney (1947); SciPy cross-check; verified in tests/test_statistical_correctness.py.

### `wilcoxon_signed_rank` — Paired Wilcoxon signed-rank test

- **Scientific Question**: Is the paired-difference distribution centered at zero under the signed-rank model?
- **Study Design**: Paired / repeated measures with exactly 2 conditions and explicit unit identifier
- **Outcome Type**: Ordered numeric (ordinal or continuous)
- **Predictor / Factor Type**: Condition variable (2 levels) + Unit ID
- **Estimand**: Matched-pairs signed-rank distribution center / location shift under symmetry
- **Primary Estimate**: Matched-pairs rank-biserial correlation ((W⁺ - W⁻) / (W⁺ + W⁻))
- **Null Hypothesis (H₀)**: The paired-difference distribution is centered at zero under the signed-rank model.
- **Alternative Hypothesis (H₁)**: The paired-difference distribution is not centered at zero (two-sided location shift).
- **Null Value**: `0.0` (matched-pairs rank-biserial correlation)
- **Test Statistic**: Wilcoxon signed-rank statistic W (sum of positive signed ranks)
- **Degrees of Freedom**: None (not applicable)
- **Effect Size Quantity**: Matched-pairs rank-biserial correlation
- **Effect Size Definition**: Difference of positive and negative rank sums divided by total rank sum.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Paired-observation percentile bootstrap for matched-pairs rank-biserial correlation.
- **Required Assumptions**:
  - Explicit paired or matched observational units.
  - Independent pairs across units.
  - Meaningful ordering and signed paired differences.
  - Symmetric paired-difference distribution for a location-shift / median interpretation.
  - Zero differences handled via SciPy 'wilcox' policy (omitted from ranking).
- **Diagnostics**:
  - Count of zero differences
  - Positive and negative rank sums
  - Tie presence
- **Missing Data Policy**: Incomplete units (missing either condition) are excluded; incomplete unit count recorded.
- **Degenerate Data Behavior**: Fewer than 2 nonzero differences returns unavailable status.
- **Multiplicity Policy**: Not applicable.
- **Numerical Provenance**: scipy.stats.wilcoxon(zero_method='wilcox')
- **Interpretation Limitations**:
  - NOT a universal test of medians; location shift strictly requires symmetry of paired differences.
  - Matched-pairs rank-biserial CI uses unit-level paired percentile bootstrap; does not assume asymptotic normality.
- **Audit Invariants**:
  - 2 * complete_pairs == analyzed_rows
  - contrast condition order matches declared condition order
  - degrees_of_freedom is None
  - p_value in [0, 1]
  - rank_biserial in [-1, 1]
  - CI bounds in [-1, 1]
  - CI lower <= upper
- **Independent Validation Source**: Wilcoxon (1945); SciPy cross-check; verified in tests/test_paired_analysis.py.

### `welch_anova` — Welch one-way ANOVA with Games-Howell comparisons

- **Scientific Question**: Do three or more independent population means differ, without assuming equal population variances?
- **Study Design**: Independent groups, 3 or more groups
- **Outcome Type**: Continuous quantitative
- **Predictor / Factor Type**: Categorical grouping (3+ levels)
- **Estimand**: Heteroscedastic multi-group population mean equality; pairwise population mean differences
- **Primary Estimate**: Group sample means; pairwise mean differences (x̄ⱼ - x̄ₘ)
- **Null Hypothesis (H₀)**: All population means are equal.
- **Alternative Hypothesis (H₁)**: At least one group population mean differs from another.
- **Null Value**: `0.0` (group means)
- **Test Statistic**: Welch's omnibus F statistic
- **Degrees of Freedom**: Two dfs: [k - 1, Welch-Satterthwaite denominator df]
- **Effect Size Quantity**: Not applicable globally; unstandardized pairwise mean differences reported
- **Effect Size Definition**: Not applicable; group means and pairwise mean differences are reported.
- **Effect Size CI Status**: `not_applicable`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Games-Howell simultaneous studentized-range confidence intervals for all pairwise contrasts.
- **Required Assumptions**:
  - Independent observations within and between groups.
  - Continuous outcome scale.
  - Outcome approximately normal within each group or large group sizes.
  - Equal population variances are NOT assumed.
  - Positive finite sample variance in every group.
- **Diagnostics**:
  - Per-group sample sizes
  - Per-group means and SDs
  - Levene test (advisory)
- **Missing Data Policy**: Complete-case analysis on (group, outcome). Excluded rows recorded.
- **Degenerate Data Behavior**: Zero variance in any group or n < 2 in any group returns unavailable status.
- **Multiplicity Policy**: Complete family of k(k-1)/2 Games-Howell comparisons with simultaneous studentized-range CIs and adjusted p-values.
- **Numerical Provenance**: Explicit PyAutoStat formula in multigroup.py (Welch 1951, Games & Howell 1976).
- **Interpretation Limitations**:
  - Omnibus significance does not identify which specific groups differ.
  - No global standardized effect size is reported under unequal variances.
  - Does not establish causality.
- **Audit Invariants**:
  - degrees_of_freedom has length 2
  - df₁ == k - 1
  - df₂ > 0
  - complete family of k*(k-1)//2 pairwise records present
  - pairwise adjusted p_values in [0, 1]
  - pairwise Games-Howell CI contains point estimate
  - CI lower <= upper
- **Independent Validation Source**: Welch (1951); Games & Howell (1976); cross-checked with pingouin in tests/test_multigroup_analysis.py.

### `one_way_anova` — Standard classical one-way ANOVA with Tukey-Kramer comparisons

- **Scientific Question**: Do three or more independent population means differ under an assumed equal population variance?
- **Study Design**: Independent groups, 3 or more groups
- **Outcome Type**: Continuous quantitative
- **Predictor / Factor Type**: Categorical grouping (3+ levels)
- **Estimand**: Homoscedastic population mean equality (μ₁ = ... = μ_k); pairwise mean differences
- **Primary Estimate**: Omnibus F; pairwise mean differences
- **Null Hypothesis (H₀)**: All group population means are equal.
- **Alternative Hypothesis (H₁)**: At least one group population mean differs.
- **Null Value**: `0.0` (group means)
- **Test Statistic**: Snedecor's F = MS_between / MS_within
- **Degrees of Freedom**: Two dfs: [k - 1, N - k]
- **Effect Size Quantity**: Eta-squared (η²)
- **Effect Size Definition**: Between-group sum of squares divided by total sum of squares.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Percentile bootstrap for eta-squared; Tukey-Kramer studentized-range intervals for pairwise contrasts.
- **Required Assumptions**:
  - Independent observations within and between groups.
  - Continuous outcome scale.
  - Equal population variances (homoscedasticity).
  - Within-group approximate normality or large sample sizes.
- **Diagnostics**:
  - Levene test for equality of variance
  - Shapiro-Wilk normality per group
- **Missing Data Policy**: Complete-case analysis on (group, outcome). Excluded rows recorded.
- **Degenerate Data Behavior**: Zero total variance or n < 2 in any group returns unavailable status.
- **Multiplicity Policy**: Complete family of k(k-1)/2 Tukey-Kramer studentized-range comparisons with familywise-adjusted p-values.
- **Numerical Provenance**: scipy.stats.f_oneway for omnibus; multigroup.py for Tukey-Kramer.
- **Interpretation Limitations**:
  - Requires explicit equal-variance justification.
  - Eta-squared is an in-sample descriptive effect.
  - Does not establish causality.
- **Audit Invariants**:
  - degrees_of_freedom has length 2
  - df₁ == k - 1
  - df₂ == analyzed_rows - k
  - eta_squared in [0, 1]
  - complete family of k*(k-1)//2 pairwise records present
  - pairwise adjusted p_values in [0, 1]
- **Independent Validation Source**: Fisher (1925); Tukey (1949); SciPy cross-check; verified in tests/test_multigroup_analysis.py.

### `kruskal_wallis` — Kruskal-Wallis test with Dunn-Holm comparisons

- **Scientific Question**: Do three or more independent populations differ in their rank distributions?
- **Study Design**: Independent groups, 3 or more groups
- **Outcome Type**: Ordered numeric (ordinal or continuous)
- **Predictor / Factor Type**: Categorical grouping (3+ levels)
- **Estimand**: Multi-group rank distribution equality / stochastic dominance
- **Primary Estimate**: Rank epsilon-squared (ε²); pairwise mean-rank contrasts
- **Null Hypothesis (H₀)**: All group rank distributions are equal.
- **Alternative Hypothesis (H₁)**: At least one group rank distribution differs.
- **Null Value**: `0.0` (rank distributions)
- **Test Statistic**: Kruskal-Wallis H statistic (tie-adjusted)
- **Degrees of Freedom**: df = k - 1
- **Effect Size Quantity**: Rank epsilon-squared
- **Effect Size Definition**: Truncated rank epsilon-squared from H, group count, and sample size.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `unavailable`
- **Confidence Interval Method**: Percentile bootstrap for omnibus epsilon-squared; pairwise rank-biserial CIs unavailable in this release.
- **Required Assumptions**:
  - Independent observations within and between groups.
  - Ordinal or continuous outcome supporting ranking.
  - At least 5 usable values per group for chi-square approximation validity.
- **Diagnostics**:
  - Per-group sample sizes
  - Tie counts
- **Missing Data Policy**: Complete-case analysis on (group, outcome). Excluded rows recorded.
- **Degenerate Data Behavior**: Fewer than 5 values in any group or completely tied outcome returns unavailable status.
- **Multiplicity Policy**: Complete family of k(k-1)/2 Dunn pairwise comparisons with Holm step-down multiplicity adjustment.
- **Numerical Provenance**: scipy.stats.kruskal for omnibus; multigroup.py for Dunn-Holm.
- **Interpretation Limitations**:
  - NOT a universal test of medians; evaluates mean ranks.
  - Pairwise rank-biserial confidence intervals are currently unavailable.
- **Audit Invariants**:
  - df == k - 1
  - epsilon_squared in [0, 1]
  - complete family of k*(k-1)//2 pairwise records present
  - pairwise adjusted p_values in [0, 1]
  - pairwise rank_biserial in [-1, 1]
- **Independent Validation Source**: Kruskal & Wallis (1952); Dunn (1964); Holm (1979); verified in tests/test_multigroup_analysis.py.

### `pearson_correlation` — Pearson product-moment correlation

- **Scientific Question**: Is there a nonzero linear association between two quantitative variables in the target population?
- **Study Design**: Single sample, bivariate continuous pairs
- **Outcome Type**: Continuous quantitative
- **Predictor / Factor Type**: Continuous quantitative
- **Estimand**: Population Pearson linear correlation coefficient (ρ)
- **Primary Estimate**: Sample Pearson correlation coefficient (r)
- **Null Hypothesis (H₀)**: The population Pearson linear correlation is zero.
- **Alternative Hypothesis (H₁)**: The population Pearson linear correlation is nonzero (two-sided).
- **Null Value**: `0.0` (Pearson r)
- **Test Statistic**: Pearson r
- **Degrees of Freedom**: None stored in result; inferential df = n - 2
- **Effect Size Quantity**: Pearson r
- **Effect Size Definition**: Signed linear correlation between the two quantitative variables.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Fisher-z asymptotic normal confidence interval.
- **Required Assumptions**:
  - Independent observational pairs.
  - Linear relationship between variables.
  - Both variables continuous quantitative with nonzero variance.
- **Diagnostics**:
  - Pairwise sample size
  - Variance of each variable
- **Missing Data Policy**: Complete observation pairs. Excluded rows recorded.
- **Degenerate Data Behavior**: Zero variance in either variable or n < 3 pairs returns unavailable status.
- **Multiplicity Policy**: Not applicable (single bivariate pair).
- **Numerical Provenance**: StatisticalAnalyzer.analyze_all correlation profile (SciPy backend).
- **Interpretation Limitations**:
  - Measures linear association only; sensitive to outliers and nonlinear curvature.
  - r² is shared linear variance, not causal effect.
  - Confidence interval uses Fisher-z transform; valid for n > 3.
- **Audit Invariants**:
  - r in [-1, 1]
  - p_value in [0, 1]
  - analyzed_rows == effective_pair_count
  - analyzed_rows + excluded_rows == original_rows
  - CI bounds in [-1, 1]
  - CI lower <= upper
- **Independent Validation Source**: Pearson (1895); SciPy cross-check; verified in tests/test_analyzer_correlation.py.

### `spearman_correlation` — Spearman rank correlation

- **Scientific Question**: Is there a nonzero monotonic rank association between two variables in the population?
- **Study Design**: Single sample, bivariate ordered pairs
- **Outcome Type**: Ordered numeric (ordinal or continuous)
- **Predictor / Factor Type**: Ordered numeric (ordinal or continuous)
- **Estimand**: Population Spearman rank correlation coefficient (ρ_s)
- **Primary Estimate**: Sample Spearman rank correlation (r_s)
- **Null Hypothesis (H₀)**: The population Spearman monotonic correlation is zero.
- **Alternative Hypothesis (H₁)**: The population Spearman monotonic correlation is nonzero (two-sided).
- **Null Value**: `0.0` (Spearman rho)
- **Test Statistic**: Spearman rho
- **Degrees of Freedom**: None stored; inferential df = n - 2
- **Effect Size Quantity**: Spearman rho
- **Effect Size Definition**: Signed monotonic rank association between the two ordered variables.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Deterministic paired-observation percentile bootstrap.
- **Required Assumptions**:
  - Independent observational pairs.
  - Ordinal or continuous scale supporting ranking.
  - Monotonic relationship under evaluation.
- **Diagnostics**:
  - Number of complete pairs
  - Tie presence
- **Missing Data Policy**: Complete observation pairs. Excluded rows recorded.
- **Degenerate Data Behavior**: Completely tied data or n < 3 pairs returns unavailable status.
- **Multiplicity Policy**: Not applicable.
- **Numerical Provenance**: scipy.stats.spearmanr
- **Interpretation Limitations**:
  - Measures monotonic association, not linearity.
  - Does not establish causality.
- **Audit Invariants**:
  - r_s in [-1, 1]
  - p_value in [0, 1]
  - bootstrap CI bounds in [-1, 1]
  - CI lower <= upper
- **Independent Validation Source**: Spearman (1904); SciPy cross-check; verified in tests/test_basic_inference.py.

### `kendall_tau_b` — Kendall's tau-b with inference

- **Scientific Question**: Is there a nonzero ordinal concordance between two ordered variables, adjusting for tied pairs?
- **Study Design**: Single sample, bivariate ordered pairs
- **Outcome Type**: Ordered numeric
- **Predictor / Factor Type**: Ordered numeric
- **Estimand**: Population Kendall's tau-b parameter (τ_b)
- **Primary Estimate**: Sample Kendall tau-b (τ_b)
- **Null Hypothesis (H₀)**: The population Kendall’s tau-b is zero.
- **Alternative Hypothesis (H₁)**: The population Kendall’s tau-b is nonzero (two-sided).
- **Null Value**: `0.0` (Kendall tau-b)
- **Test Statistic**: Kendall tau-b test statistic
- **Degrees of Freedom**: None (not applicable)
- **Effect Size Quantity**: Kendall's tau-b
- **Effect Size Definition**: Pairwise ordinal concordance adjusted for ties.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Paired-observation percentile bootstrap.
- **Required Assumptions**:
  - Independent observational pairs.
  - Ordinal scale supporting ranking.
  - Tie correction via tau-b formulation.
- **Diagnostics**:
  - Tie count in each variable
  - Concordant and discordant counts
- **Missing Data Policy**: Complete observation pairs. Excluded rows recorded.
- **Degenerate Data Behavior**: Fewer than 3 complete pairs or constant variable returns unavailable status.
- **Multiplicity Policy**: Not applicable.
- **Numerical Provenance**: scipy.stats.kendalltau(variant='b')
- **Interpretation Limitations**:
  - Measures ordinal concordance, NOT percentage variance explained.
  - Does not establish causality.
- **Audit Invariants**:
  - tau_b in [-1, 1]
  - p_value in [0, 1]
  - ties variant is 'b'
  - bootstrap CI bounds in [-1, 1]
  - CI lower <= upper
- **Independent Validation Source**: Kendall (1938); SciPy cross-check; verified in tests/test_binary_extended_association.py.

### `point_biserial_correlation` — Point-biserial correlation with inference

- **Scientific Question**: Is there a nonzero linear association between a continuous variable and an explicitly coded binary indicator?
- **Study Design**: Independent observations with one continuous and one binary variable
- **Outcome Type**: Continuous quantitative
- **Predictor / Factor Type**: Binary categorical factor (explicit 0/1 coding)
- **Estimand**: Population point-biserial correlation (r_pb)
- **Primary Estimate**: Sample point-biserial correlation (r_pb)
- **Null Hypothesis (H₀)**: The population point-biserial correlation is zero.
- **Alternative Hypothesis (H₁)**: The population point-biserial correlation is nonzero (two-sided).
- **Null Value**: `0.0` (point-biserial correlation)
- **Test Statistic**: Point-biserial r
- **Degrees of Freedom**: df = n - 2
- **Effect Size Quantity**: Point-biserial r
- **Effect Size Definition**: Pearson r between continuous values and the 0/1 binary indicator.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Paired-observation percentile bootstrap.
- **Required Assumptions**:
  - Independent observational pairs.
  - Continuous distribution for the quantitative variable.
  - Exactly two levels in binary variable with researcher-declared positive level.
- **Diagnostics**:
  - Binary class frequencies
  - Continuous variable spread
- **Missing Data Policy**: Complete-case analysis on (binary, continuous). Excluded rows recorded.
- **Degenerate Data Behavior**: Single class observed or zero continuous variance returns unavailable status.
- **Multiplicity Policy**: Not applicable.
- **Numerical Provenance**: scipy.stats.pointbiserialr
- **Interpretation Limitations**:
  - Symmetric association measure; positive level determines sign.
  - Not a substitute for a group mean difference test.
  - Does not establish causality.
- **Audit Invariants**:
  - r_pb in [-1, 1]
  - p_value in [0, 1]
  - binary_encoding records positive and negative levels
  - bootstrap CI bounds in [-1, 1]
  - CI lower <= upper
- **Independent Validation Source**: Lev (1949); SciPy cross-check; verified in tests/test_binary_extended_association.py.

### `partial_pearson_correlation` — Partial Pearson correlation

- **Scientific Question**: Is there a nonzero linear association between two continuous variables after linearly controlling for declared covariates?
- **Study Design**: Independent observations with two focal continuous variables and k >= 1 quantitative controls
- **Outcome Type**: Continuous quantitative
- **Predictor / Factor Type**: Continuous quantitative + declared quantitative controls
- **Estimand**: Population partial Pearson correlation controlling for covariates (ρ_XY.Z)
- **Primary Estimate**: Sample partial Pearson correlation (r_XY.Z)
- **Null Hypothesis (H₀)**: The partial population Pearson correlation is zero, controlling for the declared covariates.
- **Alternative Hypothesis (H₁)**: The partial population Pearson correlation is nonzero (two-sided).
- **Null Value**: `0.0` (partial Pearson correlation)
- **Test Statistic**: Student's t = r_partial * sqrt((n - k - 2) / (1 - r_partial²))
- **Degrees of Freedom**: df = n - k - 2
- **Effect Size Quantity**: Partial Pearson r
- **Effect Size Definition**: Pearson correlation of OLS residuals after regressing focal variables on declared controls.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Complete-row model-refitting percentile bootstrap.
- **Required Assumptions**:
  - Independent observational units.
  - Linear relationship between focal variables and controls.
  - Full-rank control design matrix.
  - All focal and control variables quantitative.
- **Diagnostics**:
  - Residualization rank
  - VIF of controls
  - Sample size relative to k + 3
- **Missing Data Policy**: Complete-case across focal variables and all controls. Excluded rows recorded.
- **Degenerate Data Behavior**: Rank-deficient control matrix or n < k + 3 returns unavailable status.
- **Multiplicity Policy**: Not applicable.
- **Numerical Provenance**: statsmodels.api.OLS residualization + NumPy correlation + SciPy t reference.
- **Interpretation Limitations**:
  - Conditioning on controls does not establish that confounding has been removed.
  - Control choice must be theoretically justified.
  - Does not establish causality.
- **Audit Invariants**:
  - r_partial in [-1, 1]
  - p_value in [0, 1]
  - df == analyzed_rows - len(controls) - 2
  - test_statistic matches r * sqrt(df / (1 - r²))
  - bootstrap CI bounds in [-1, 1]
  - CI lower <= upper
- **Independent Validation Source**: Yule (1907); statsmodels/pingouin cross-check; verified in tests/test_binary_extended_association.py.

### `pearson_chi_square` — Pearson chi-square test of independence

- **Scientific Question**: Are two categorical variables independent in the target population?
- **Study Design**: Independent observations, cross-classification table (r x c)
- **Outcome Type**: Categorical (nominal or ordinal)
- **Predictor / Factor Type**: Categorical (nominal or ordinal)
- **Estimand**: Categorical independence / departure from multinomial product probabilities
- **Primary Estimate**: Cramér's V
- **Null Hypothesis (H₀)**: The two categorical variables are independent.
- **Alternative Hypothesis (H₁)**: The two categorical variables are dependent / associated.
- **Null Value**: `0.0` (Cramér's V)
- **Test Statistic**: Pearson chi-square statistic (X²)
- **Degrees of Freedom**: df = (r - 1) * (c - 1)
- **Effect Size Quantity**: Cramér's V
- **Effect Size Definition**: Square root of X² divided by (N * min(r-1, c-1)).
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Complete-row observation percentile bootstrap.
- **Required Assumptions**:
  - Independent observations.
  - Mutually exclusive and exhaustive categories.
  - Adequate expected cell counts (all expected cells >= 5 under PyAutoStat policy).
- **Diagnostics**:
  - Table of expected frequencies
  - Percentage of cells with expected count < 5
- **Missing Data Policy**: Complete-case analysis on the two categorical variables. Excluded rows recorded.
- **Degenerate Data Behavior**: Sparse tables (expected count < 5) fall back to Fisher's exact if 2x2, or return blocked status.
- **Multiplicity Policy**: Not applicable.
- **Numerical Provenance**: scipy.stats.chi2_contingency(correction=False)
- **Interpretation Limitations**:
  - Nonnegative measure without sign or direction.
  - Cramér's V² is NOT percentage of variance explained.
  - Does not establish causality.
- **Audit Invariants**:
  - test_statistic >= 0
  - p_value in [0, 1]
  - df == (r - 1) * (c - 1)
  - cramers_v in [0, 1]
  - bootstrap CI bounds in [0, 1]
  - CI lower <= upper
- **Independent Validation Source**: Pearson (1900); Cramér (1946); SciPy cross-check; verified in tests/test_categorical.py.

### `fisher_exact` — Fisher's exact test

- **Scientific Question**: Are two binary categorical variables independent under fixed margins?
- **Study Design**: Independent observations, exactly 2x2 contingency table
- **Outcome Type**: Binary categorical
- **Predictor / Factor Type**: Binary categorical
- **Estimand**: Binary independence / odds ratio under hypergeometric distribution
- **Primary Estimate**: Sample odds ratio (ad / bc)
- **Null Hypothesis (H₀)**: The two binary categorical variables are independent.
- **Alternative Hypothesis (H₁)**: The two binary categorical variables are dependent (two-sided odds ratio != 1).
- **Null Value**: `1.0` (odds ratio)
- **Test Statistic**: Hypergeometric exact probability / sample odds ratio
- **Degrees of Freedom**: None (exact test)
- **Effect Size Quantity**: Sample odds ratio
- **Effect Size Definition**: Cross-product ratio (ad / bc) based on first-observed row and column order.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: log-Wald confidence interval for sample odds-ratio estimator (unavailable when any cell is zero).
- **Required Assumptions**:
  - Independent observations.
  - Exactly 2x2 contingency table.
  - Fixed or conditioned marginal totals.
- **Diagnostics**:
  - Observed 2x2 contingency counts
  - Marginal totals
  - Zero-cell detection
- **Missing Data Policy**: Complete-case analysis on the two binary variables. Excluded rows recorded.
- **Degenerate Data Behavior**: Empty margin returns unavailable status; zero cells recorded with explicit status (zero, positive_infinity).
- **Multiplicity Policy**: Not applicable.
- **Numerical Provenance**: scipy.stats.fisher_exact(alternative='two-sided')
- **Interpretation Limitations**:
  - Supported strictly for 2x2 tables.
  - Odds ratio direction depends on category level order.
  - Odds-ratio CI uses asymptotic log-Wald method matching the sample OR estimator; unavailable when any cell count is zero (no silent continuity corrections).
  - Does not establish causality.
- **Audit Invariants**:
  - observed_counts is a 2x2 matrix
  - degrees_of_freedom is None
  - p_value in [0, 1]
  - sample odds ratio >= 0 when finite
  - sample odds ratio CI lower > 0 and finite when available
  - CI lower <= upper
- **Independent Validation Source**: Fisher (1935); SciPy cross-check; verified in tests/test_basic_inference.py.

### `mcnemar` — McNemar's test for paired binary outcomes

- **Scientific Question**: Do paired binary marginal event probabilities differ across two conditions on the same units?
- **Study Design**: Paired design with explicit unit ID, exactly 2 conditions, and binary outcome
- **Outcome Type**: Binary (with declared event level)
- **Predictor / Factor Type**: Condition variable (2 levels) + Unit ID
- **Estimand**: Paired marginal event probability difference (P(Y₁ = 1) - P(Y₂ = 1))
- **Primary Estimate**: Sample paired event proportion difference ((b - c) / n)
- **Null Hypothesis (H₀)**: The paired binary outcome proportions are equal (marginal homogeneity).
- **Alternative Hypothesis (H₁)**: The paired binary outcome proportions are unequal (two-sided).
- **Null Value**: `0.0` (paired proportion difference)
- **Test Statistic**: Exact binomial test on discordant pairs (b ~ Binomial(b + c, 0.5))
- **Degrees of Freedom**: None (exact test)
- **Effect Size Quantity**: Paired proportion difference; matched odds ratio (b / c)
- **Effect Size Definition**: First minus second condition event proportion difference.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Paired-unit resampling percentile bootstrap for proportion difference.
- **Required Assumptions**:
  - Explicit unit identity across conditions.
  - Independent paired units.
  - Exactly two conditions and binary outcome with declared event level.
  - Discordant pairs are informative for marginal homogeneity.
- **Diagnostics**:
  - 2x2 transition table
  - Discordant pair counts b and c
  - Total discordant pairs
- **Missing Data Policy**: Incomplete units (missing either condition) are excluded; incomplete unit count recorded.
- **Degenerate Data Behavior**: Zero discordant pairs (b + c = 0) returns unavailable status.
- **Multiplicity Policy**: Not applicable.
- **Numerical Provenance**: scipy.stats.binomtest(n=b+c, p=0.5, alternative='two-sided')
- **Interpretation Limitations**:
  - Evaluates marginal homogeneity based on discordant pairs only.
  - Concordant pairs provide no information about the contrast.
  - Does not establish causality.
- **Audit Invariants**:
  - 2 * complete_pairs == analyzed_rows
  - transition table counts sum to complete_pairs
  - primary_estimate == (b - c) / complete_pairs
  - p_value in [0, 1]
  - bootstrap CI bounds in [-1, 1]
  - CI lower <= upper
- **Independent Validation Source**: McNemar (1947); SciPy cross-check; verified in tests/test_binary_extended_association.py.

### `linear_regression` — Ordinary least-squares linear regression

- **Scientific Question**: How is the conditional mean of a continuous outcome associated with a set of predictors under a linear model?
- **Study Design**: Independent observations, continuous outcome, one or more predictors
- **Outcome Type**: Continuous quantitative
- **Predictor / Factor Type**: One or more continuous, binary, nominal, or ordinal predictors
- **Estimand**: Conditional mean coefficients (β_j) and in-sample explained variance (R²) under declared model
- **Primary Estimate**: Sample R²; sample OLS slope coefficients (b_j)
- **Null Hypothesis (H₀)**: All non-intercept population slope coefficients are zero.
- **Alternative Hypothesis (H₁)**: At least one non-intercept slope coefficient is nonzero.
- **Null Value**: `0.0` (R-squared)
- **Test Statistic**: Model omnibus F statistic; individual coefficient t statistics
- **Degrees of Freedom**: Two dfs: [model df = p, residual df = n - p - 1]
- **Effect Size Quantity**: In-sample R²; continuous standardized betas
- **Effect Size Definition**: Observed outcome variance accounted for by the fitted in-sample model.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Analytical Student-t intervals for coefficients (classical or HC3 covariance); case-resampling percentile bootstrap CI for in-sample R-squared.
- **Required Assumptions**:
  - Independent observational units.
  - Linear conditional-mean specification.
  - Full-rank design matrix with positive residual degrees of freedom.
  - Appropriate residual behavior for the selected covariance (classical or HC3).
- **Diagnostics**:
  - VIF for multicollinearity
  - Breusch-Pagan test
  - Residual normality
  - Cook's distance
- **Missing Data Policy**: Complete-case analysis across outcome and all predictors. Excluded rows recorded.
- **Degenerate Data Behavior**: Rank-deficient design or n <= p + 1 returns unavailable status.
- **Multiplicity Policy**: Omnibus model test; individual coefficient inferences reported without automatic multiplicity adjustment.
- **Numerical Provenance**: statsmodels.api.OLS
- **Interpretation Limitations**:
  - Additive linear main effects only; associations are noncausal.
  - R² is an in-sample descriptive fit, NOT validated out-of-sample prediction.
  - In-sample R² CI uses case-resampling percentile bootstrap; reflects in-sample fit uncertainty, not predictive performance.
- **Audit Invariants**:
  - r_squared in [0, 1]
  - p_value in [0, 1]
  - residual_df == analyzed_rows - parameter_count
  - coefficient analytical CI contains estimate
  - CI lower <= upper
  - r_squared CI bounds in [0, 1]
  - r_squared CI lower <= upper
  - design_matrix.full_rank is True
- **Independent Validation Source**: Legendre (1805); Gauss (1809); statsmodels cross-check; verified in tests/test_regression_workflow.py.

### `logistic_regression` — Binary logistic regression

- **Scientific Question**: How is the log-odds (and odds ratio) of a binary event associated with predictors under a logistic model?
- **Study Design**: Independent observations, binary outcome, one or more predictors
- **Outcome Type**: Binary (with declared event level)
- **Predictor / Factor Type**: One or more continuous, binary, nominal, or ordinal predictors
- **Estimand**: Conditional log-odds coefficients (β_j) and odds ratios (OR_j = exp(β_j)) under declared model
- **Primary Estimate**: Maximum likelihood log-odds coefficients (b_j) and odds ratios (exp(b_j))
- **Null Hypothesis (H₀)**: All non-intercept population log-odds coefficients are zero.
- **Alternative Hypothesis (H₁)**: At least one non-intercept log-odds coefficient is nonzero.
- **Null Value**: `0.0` (log-odds coefficients)
- **Test Statistic**: Model likelihood ratio test statistic; coefficient Wald z statistics
- **Degrees of Freedom**: Two dfs: [model df = p, residual df = n - p - 1]
- **Effect Size Quantity**: Odds ratios per predictor; McFadden pseudo-R²
- **Effect Size Definition**: Odds ratios exp(b_j); McFadden pseudo-R² = 1 - ln L_full / ln L_null.
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Analytical Wald z intervals on log-odds scale, exponentiated for odds ratios (NOT bootstrap intervals).
- **Required Assumptions**:
  - Independent observational units.
  - Binary outcome with researcher-declared event level.
  - Linear relationship between predictors and log-odds.
  - Absence of complete or quasi-complete separation.
  - Full-rank design matrix; at least 10 complete cases.
- **Diagnostics**:
  - Model convergence
  - Event and non-event counts
  - Condition number
  - Likelihood ratio test
- **Missing Data Policy**: Complete-case analysis across outcome and all predictors. Excluded rows recorded.
- **Degenerate Data Behavior**: Complete separation, single outcome class, or non-convergence returns unavailable status.
- **Multiplicity Policy**: Likelihood ratio omnibus test; individual Wald z tests per coefficient.
- **Numerical Provenance**: statsmodels.api.Logit (MLE fit, Wald inference).
- **Interpretation Limitations**:
  - Odds ratios describe relative odds, NOT absolute probabilities or causal effects.
  - McFadden pseudo-R² is a likelihood fit index, not directly comparable with OLS R².
  - No classification cutoff or prediction metric is applied.
- **Audit Invariants**:
  - odds_ratio == exp(beta)
  - odds_ratio_ci == [exp(lower), exp(upper)]
  - mcfadden_r2 in [0, 1]
  - event_count + non_event_count == analyzed_rows
  - AIC and BIC match likelihood formulas
  - design_matrix.full_rank is True
- **Independent Validation Source**: Cox (1958); statsmodels cross-check; verified in tests/test_regression_workflow.py.

### `cronbach_alpha` — Cronbach's alpha scale reliability

- **Scientific Question**: What is the internal consistency of a researcher-declared set of multi-item responses?
- **Study Design**: Multi-item survey or psychometric scale (2+ numeric/ordinal items)
- **Outcome Type**: Multiple numeric or ordinal items
- **Predictor / Factor Type**: None (scale measurement model)
- **Estimand**: Scale internal consistency coefficient (α)
- **Primary Estimate**: Sample Cronbach's alpha (α̂)
- **Null Hypothesis (H₀)**: None (descriptive measurement property; non-inferential).
- **Alternative Hypothesis (H₁)**: None (non-inferential scale reliability; no alternative hypothesis tested)
- **Null Value**: `None` (None)
- **Test Statistic**: None (descriptive index)
- **Degrees of Freedom**: None
- **Effect Size Quantity**: Cronbach's alpha
- **Effect Size Definition**: k/(k-1) * (1 - Σ Var(Xᵢ) / Var(Σ Xᵢ)).
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Respondent-row percentile bootstrap.
- **Required Assumptions**:
  - Researcher-declared scale membership and item set.
  - Items scored in positive direction (or explicit reverse scoring declared).
  - At least two numeric items and two complete respondents.
  - Positive finite total-score sample variance.
  - Unidimensionality is assumed by the index, NOT verified.
- **Diagnostics**:
  - Corrected item-total correlations
  - Alpha-if-item-deleted
  - Inter-item correlation matrix
- **Missing Data Policy**: Complete-case analysis across all scale items; per-item missing counts reported.
- **Degenerate Data Behavior**: Zero total score variance, < 2 items, or < 2 complete respondents returns unavailable status.
- **Multiplicity Policy**: Not applicable.
- **Numerical Provenance**: Explicit NumPy sample covariance and correlation formula in reliability.py.
- **Interpretation Limitations**:
  - Non-inferential; does NOT produce a hypothesis test or p-value.
  - Alpha does NOT prove unidimensionality, validity, or temporal stability.
  - No universal adequacy cutoff; negative alpha is preserved when observed.
- **Audit Invariants**:
  - p_value is None
  - test_statistic is None
  - alpha matches covariance formula
  - item_statistics has one record per declared item
  - corrected_total excludes focal item
  - bootstrap CI lower <= upper
- **Independent Validation Source**: Cronbach (1951); Nunnally & Bernstein (1994); verified in tests/test_reliability_workflow.py.

### `repeated_measures_anova` — One-way repeated-measures ANOVA with Greenhouse-Geisser correction

- **Scientific Question**: Do repeated-condition population means differ across three or more conditions on the same units?
- **Study Design**: Repeated measures with explicit unit ID, 3+ conditions, and long-format panel
- **Outcome Type**: Continuous quantitative
- **Predictor / Factor Type**: Condition variable (3+ levels) + Unit ID
- **Estimand**: Repeated-condition population mean equality; pairwise condition mean differences
- **Primary Estimate**: Condition sample means; partial eta-squared (η_p²); pairwise mean differences
- **Null Hypothesis (H₀)**: All repeated-condition population means are equal.
- **Alternative Hypothesis (H₁)**: At least one repeated-condition population mean differs.
- **Null Value**: `0.0` (condition means)
- **Test Statistic**: Repeated-measures F = MS_condition / MS_error
- **Degrees of Freedom**: Uncorrected [k-1, (k-1)(n-1)]; GG-corrected [ε(k-1), ε(k-1)(n-1)]
- **Effect Size Quantity**: Partial eta-squared (η_p²)
- **Effect Size Definition**: Condition sum of squares divided by condition sum of squares plus error sum of squares (partial eta-squared).
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Exact noncentral-F inversion confidence interval for partial eta-squared (using uncorrected F and df); analytical paired-t intervals and exact noncentral-t dz intervals for pairwise contrasts.
- **Required Assumptions**:
  - Explicit repeated observational units across conditions.
  - Units independent of other units.
  - Continuous outcome / mean target.
  - One observation per unit-condition.
  - Complete panel across declared conditions.
  - Within-subject sphericity (Greenhouse-Geisser correction applied when rejected or uncomputable).
- **Diagnostics**:
  - Mauchly's test of sphericity (W, χ², p)
  - Greenhouse-Geisser epsilon
  - Condition means and SDs
- **Missing Data Policy**: Incomplete units (missing any condition) are excluded; incomplete unit count recorded. No wide-to-long guessing.
- **Degenerate Data Behavior**: Fewer than 3 complete units, < 3 conditions, or zero within-subject variance returns unavailable status.
- **Multiplicity Policy**: Complete family of k(k-1)/2 pairwise paired-t tests with Holm step-down multiplicity adjustment.
- **Numerical Provenance**: Explicit matrix decomposition in repeated_measures.py (Mauchly 1940, Greenhouse & Geisser 1959).
- **Interpretation Limitations**:
  - Mauchly non-rejection does NOT prove sphericity.
  - Greenhouse-Geisser correction applied when sphericity is rejected or uncomputable.
  - Partial eta-squared CI uses exact noncentral-F inversion on uncorrected F and degrees of freedom; GG correction modifies inferential p-values but not the observed SS-based effect.
  - Does not establish causality.
- **Audit Invariants**:
  - ss_total == ss_condition + ss_subject + ss_error
  - partial_eta_squared == ss_condition / (ss_condition + ss_error)
  - partial_eta_squared in [0, 1]
  - partial_eta_squared CI in [0, 1]
  - partial_eta_squared CI lower <= upper
  - gg_epsilon in [1/(k-1), 1]
  - corrected dfs == epsilon * uncorrected dfs
  - corrected p matches F and corrected dfs
  - complete family of k*(k-1)//2 pairwise records present
  - pairwise adjusted p_values in [0, 1]
- **Independent Validation Source**: Andy Field (2012) textbook Bushtucker benchmark; statsmodels AnovaRM cross-check; verified in tests/test_repeated_measures.py.

### `friedman_test` — Friedman rank-sum test with Wilcoxon-Holm follow-up

- **Scientific Question**: Do within-unit rank distributions differ across three or more repeated conditions on the same units?
- **Study Design**: Repeated measures with explicit unit ID, 3+ conditions, and long-format panel
- **Outcome Type**: Ordered numeric (ordinal or continuous)
- **Predictor / Factor Type**: Condition variable (3+ levels) + Unit ID
- **Estimand**: Within-unit rank distribution differences across conditions; Kendall's W rank concordance
- **Primary Estimate**: Kendall's W; condition rank medians; pairwise rank-biserial correlations
- **Null Hypothesis (H₀)**: The repeated-condition within-unit rank distributions are equal.
- **Alternative Hypothesis (H₁)**: At least one condition tends to yield higher/lower within-unit ranks than another.
- **Null Value**: `0.0` (within-unit rank distributions)
- **Test Statistic**: Friedman's Q statistic (chi-square approximation)
- **Degrees of Freedom**: df = k - 1
- **Effect Size Quantity**: Kendall's W
- **Effect Size Definition**: Friedman Q divided by n*(k - 1) (Kendall's W rank concordance).
- **Effect Size CI Status**: `available`
- **Primary Estimate CI Status**: `available`
- **Confidence Interval Method**: Participant-block percentile bootstrap for Kendall's W; paired-observation percentile bootstrap for pairwise rank-biserial contrasts.
- **Required Assumptions**:
  - Explicit repeated observational units across conditions.
  - Units independent of other units.
  - Meaningful rank ordering across conditions.
  - One observation per unit-condition.
  - Complete panel across declared conditions.
- **Diagnostics**:
  - Condition medians and IQRs
  - Tie presence across conditions
- **Missing Data Policy**: Complete-case panel across all declared conditions; incomplete units excluded and recorded.
- **Degenerate Data Behavior**: Fewer than 3 complete units, < 3 conditions, or zero within-unit rank variance returns unavailable status.
- **Multiplicity Policy**: Complete family of k(k-1)/2 pairwise paired Wilcoxon signed-rank tests with Holm step-down multiplicity adjustment.
- **Numerical Provenance**: Explicit within-unit rank algorithm in repeated_measures.py (Friedman 1937, Kendall 1939).
- **Interpretation Limitations**:
  - Friedman evaluates within-unit rank distributions, NOT a universal test of medians.
  - Kendall's W is rank concordance, NOT percentage of variance explained.
  - Kendall's W and pairwise rank-biserial intervals use participant/paired percentile bootstrap; approximate uncertainty intervals, not significance tests.
- **Audit Invariants**:
  - df == k - 1
  - kendalls_w == Q / (n * (k - 1))
  - kendalls_w in [0, 1]
  - kendalls_w CI in [0, 1]
  - kendalls_w CI lower <= upper
  - complete family of k*(k-1)//2 pairwise records present
  - pairwise adjusted p_values in [0, 1]
  - pairwise rank_biserial in [-1, 1]
- **Independent Validation Source**: Friedman (1937); Kendall & Babington Smith (1939); SciPy cross-check; verified in tests/test_repeated_measures.py.

## Cross-References

- [Effect-Size Confidence Interval Gaps](EFFECT_SIZE_CI_GAPS.md)
- [Statistical Validation Baseline](STATISTICAL_VALIDATION.md)
- [Scientific Limitations](SCIENTIFIC_LIMITATIONS.md)
- [Capabilities](CAPABILITIES.md)
