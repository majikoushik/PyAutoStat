# Effect-Size Confidence Interval Status & Handoff Matrix

This document establishes the authoritative classification of effect-size confidence intervals
across all methods supported by PyAutoStat, the defensible numerical methods implemented, validation
routes, and remaining scientific policy boundaries.

## Scientific Principle

PyAutoStat enforces complete honesty regarding uncertainty:
- Confidence intervals are reported only where an independently validated, scientifically defensible
  interval algorithm is implemented.
- Missing uncertainty is never disguised with dummy values, arbitrary placeholders, or silent omission.
- Where an effect-size confidence interval is not applicable by design, its status is explicitly declared as
  `not_applicable` (e.g. Welch ANOVA standardized omnibus effect).
- A confidence interval may be attached to an effect record only when it targets the exact same scientific
  quantity as the point estimate (e.g., sample odds ratio is matched with log-Wald interval, never with an
  exact conditional interval).

## Effect-Size Confidence Interval Inventory

| Method ID | Method Name | Reported Effect Quantity | Current CI Status | Implemented Defensible CI Method | Validation Source | Status |
|---|---|---|---|---|---|---|
| `welch_t` | Welch t-test | Cohen's d | `available` | Independent-group percentile bootstrap | SciPy noncentral t benchmark | Shipped |
| `student_t` | Student t-test | Cohen's d | `available` | Independent-group percentile bootstrap | SciPy noncentral t benchmark | Shipped |
| `paired_t` | Paired t-test | Paired Cohen's dz | `available` | Exact noncentral-t inversion | R `MBESS::ci.smd()` / exact nct inversion | Shipped |
| `one_sample_t` | One-sample t-test | Cohen's d | `available` | Exact noncentral-t inversion | R `MBESS::ci.smd()` / exact nct inversion | Shipped |
| `mann_whitney_u` | Mann-Whitney U | Rank-biserial r_rb | `available` | Kerby (2014) rank-biserial interval via sample pair resampling | Independent Monte Carlo & permutation benchmark | Shipped |
| `wilcoxon_signed_rank` | Paired Wilcoxon | Rank-biserial r_rb | `available` | Paired-observation percentile bootstrap | Deterministic fixed-seed resampling benchmark | Shipped |
| `welch_anova` | Welch ANOVA | None (global) | `not_applicable` | None (no universally agreed standardized omnibus effect under heteroscedasticity) | Scientific policy: Games-Howell pairwise CIs are primary | Intentionally omitted |
| `one_way_anova` | One-way ANOVA | Eta-squared (η²) | `available` | Noncentral F distribution inversion (Smithson 2001) | R `MBESS::ci.pvaf()` / exact algebraic check | Shipped |
| `kruskal_wallis` | Kruskal-Wallis | Rank epsilon-squared | `available` | Truncated bootstrap percentile interval | Independent resampling reference fixture | Shipped |
| `pearson_correlation` | Pearson correlation | Pearson r | `available` | Fisher z-transformation `tanh(atanh(r) ± z * 1/√(n-3))` | Modern SciPy `pearsonr.confidence_interval()` / direct algebra | Shipped |
| `spearman_correlation` | Spearman correlation | Spearman rho | `available` | Observation-pair percentile bootstrap | Resampling reference fixture | Shipped |
| `kendall_tau_b` | Kendall's tau-b | Kendall's tau-b | `available` | Observation-pair percentile bootstrap | Resampling reference fixture | Shipped |
| `point_biserial_correlation` | Point-biserial correlation | Point-biserial r_pb | `available` | Observation-pair percentile bootstrap | Resampling reference fixture | Shipped |
| `partial_pearson_correlation` | Partial Pearson correlation | Partial Pearson r | `available` | Fisher z-transformation on residual degrees of freedom `n - k - 3` | Independent linear regression residualization fixture | Shipped |
| `pearson_chi_square` | Chi-square test | Cramér's V | `available` | Noncentral chi-square distribution inversion (Smithson 2003) | R `MBESS::ci.pvaf()` / exact algebraic check | Shipped |
| `fisher_exact` | Fisher's exact test | Sample odds ratio | `available` | Asymptotic log-Wald interval for sample odds-ratio estimator | statsmodels `Table2x2(shift_zeros=False)` / direct formula | Shipped |
| `mcnemar` | McNemar test | Odds ratio (b/c) | `available` | Exact binomial / Clopper-Pearson interval on discordant pairs | Exact binomial reference fixture | Shipped |
| `linear_regression` | OLS linear regression | In-sample R² | `available` | Case-resampling percentile bootstrap for in-sample R-squared | Fixed-seed case resampling benchmark | Shipped |
| `logistic_regression` | Binary logistic regression | Odds ratio per predictor | `available` | Exponentiated Wald z-interval `[exp(b_j - z*se), exp(b_j + z*se)]` | statsmodels Logit conf_int exponentiated | Shipped |
| `cronbach_alpha` | Cronbach's alpha | Scale alpha (α) | `available` | Respondent-row percentile bootstrap | Independent psychometric reference fixture | Shipped |
| `repeated_measures_anova` | Repeated-measures ANOVA | Partial eta-squared | `available` | Exact noncentral-F inversion on uncorrected F and df | R `MBESS::ci.pvaf()` / `effectsize::eta_squared()` | Shipped |
| `friedman_test` | Friedman rank-sum | Kendall's W | `available` | Participant/unit-level block bootstrap | Fixed-seed panel resampling benchmark | Shipped |

---

## Detailed Uncertainty Specifications

### 1. Paired Cohen's dz (`paired_t`)
- **Quantity**: Sample mean of paired differences divided by the standard deviation of paired differences ($d_z = \bar{D} / s_D$).
- **Method**: Exact confidence limits based on inverting the noncentral t distribution with noncentrality parameter $\delta = d_z \sqrt{n}$ and $df = n - 1$.
- **Validation**: Verified against exact noncentral-t cumulative probabilities matching R `MBESS::ci.smd()` for paired designs.
- **Pairwise Follow-ups**: The same noncentral-t calculation is applied to all pairwise contrasts within `repeated_measures_anova`, with `multiplicity_adjusted=False`.

### 2. One-Sample Cohen's d (`one_sample_t`)
- **Quantity**: Sample mean minus declared reference value divided by sample standard deviation ($d = (\bar{x} - \mu_0) / s$).
- **Method**: Exact noncentral t distribution inversion with $\delta = d \sqrt{n}$ and $df = n - 1$.
- **Validation**: Verified against exact noncentral-t tail probabilities matching R `MBESS::ci.smd()` one-sample mode.

### 3. Fisher's Exact Test Odds Ratio (`fisher_exact`)
- **Quantity**: Sample cross-product odds ratio ($ad / bc$) for $2 \times 2$ contingency tables.
- **Method**: Asymptotic log-Wald confidence interval:
  $$\text{SE}(\ln \text{OR}) = \sqrt{\frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d}}, \quad \text{CI} = \exp\left(\ln(\text{OR}) \pm z_{1-\alpha/2} \cdot \text{SE}\right)$$
- **Estimand Coherence**: The confidence interval matches the sample odds ratio point estimate. An exact conditional interval is not combined with an unconditional sample point estimate.
- **Zero-Cell Behavior**: If any cell count is zero, the interval is explicitly marked `unavailable` with reason. No arbitrary $0.5$ continuity corrections are applied.
- **Validation**: Verified against statsmodels `Table2x2(..., shift_zeros=False)` and direct algebraic formulas.

### 4. Pearson Correlation (`pearson_correlation`)
- **Quantity**: Population linear correlation coefficient ($r$).
- **Method**: Standard analytical Fisher z-transformation:
  $$z = \frac{1}{2} \ln\left(\frac{1+r}{1-r}\right), \quad \text{SE} = \frac{1}{\sqrt{n-3}}, \quad [z_L, z_U] = z \pm z_{1-\alpha/2} \text{SE}, \quad [r_L, r_U] = \tanh([z_L, z_U])$$
- **Validation**: Exact numerical agreement with modern SciPy `pearsonr().confidence_interval()` and direct algebraic limits.
- **Degenerate Handling**: $n \le 3$ returns `unavailable` with an explicit reason. Bounds strictly respect $[-1, 1]$.

### 5. Paired Wilcoxon Rank-Biserial (`wilcoxon_signed_rank`)
- **Quantity**: Proportion of positive signed ranks minus proportion of negative signed ranks ($r_{rb} = (W^+ - W^-) / (W^+ + W^-)$).
- **Method**: Participant/unit-level paired percentile bootstrap (resampling complete paired units with replacement).
- **Pairwise Follow-ups**: Applied to all pairwise Wilcoxon-Holm contrasts within `friedman_test`, with `multiplicity_adjusted=False`.
- **Validation**: Deterministic fixed-seed resampling benchmark; verified invariance under condition order reversal ($r_{rb, \text{rev}} = -r_{rb}$).

### 6. Friedman Kendall's W (`friedman_test`)
- **Quantity**: Kendall's coefficient of concordance ($W = Q / [n(k - 1)]$).
- **Method**: Complete unit/participant-level block bootstrap (resampling complete row vectors across conditions with replacement).
- **Validation**: Panel resampling benchmark; bounds strictly in $[0, 1]$; valid resamples accounting.
- **Interpretation**: Approximate uncertainty interval; not a substitute hypothesis test.

### 7. Repeated-Measures ANOVA Partial Eta-Squared (`repeated_measures_anova`)
- **Quantity**: Partial eta-squared ($\eta_p^2 = SS_{\text{condition}} / [SS_{\text{condition}} + SS_{\text{error}}]$).
- **Method**: Exact noncentral-F inversion based on uncorrected observed F and uncorrected condition/error degrees of freedom:
  $$\eta_p^2 = \frac{F \cdot df_{\text{condition}}}{F \cdot df_{\text{condition}} + df_{\text{error}}}$$
- **Sphericity Independence**: Greenhouse-Geisser adjustment changes inferential degrees of freedom and p-values, not the observed SS-based partial $\eta^2$ or its noncentrality mapping.
- **Validation**: Numerical agreement with exact noncentral-F cumulative tail probabilities and R `effectsize::eta_squared(..., partial=TRUE)`.

### 8. Kruskal-Wallis Dunn Pairwise Rank-Biserial Follow-up
- **Quantity**: Pairwise rank-biserial correlation for post-hoc Dunn contrasts.
- **Method**: Independent within-group percentile bootstrap (resampling within each group separately, preserving sample sizes $n_1$ and $n_2$).
- **Multiplicity Policy**: Pointwise confidence interval; `multiplicity_adjusted=False`. Holm step-down adjustment applies to p-values, not intervals.

### 9. OLS In-Sample R² (`linear_regression`)
- **Quantity**: In-sample proportion of outcome variance explained by the fitted linear model ($R^2$).
- **Method**: Case-resampling percentile bootstrap (resampling complete observation rows with replacement).
- **Categorical Handling**: Preserves treatment and reference coding; rank-deficient replicates are discarded with valid-fraction accounting.
- **Interpretation**: Measures in-sample descriptive fit uncertainty, not out-of-sample predictive performance.

### 10. Welch ANOVA Global Standardized Effect (`welch_anova`)
- **Quantity**: Standardized omnibus effect under heteroscedasticity.
- **Current Status**: `not_applicable` (by scientific policy).
- **Policy**: Standardized omnibus measures such as classical $\eta^2$ or $\omega^2$ assume homoscedasticity. Inventing a synthetic standardized effect for Welch ANOVA is scientifically misleading. Games-Howell pairwise confidence intervals provide the primary localized uncertainty.

---

## Architectural Guarantees

1. **Stable Schema**: Every interval populates `confidence_interval` dictionaries using the established schema: `{"lower": float, "upper": float, "level": float, "method": str, "quantity": str}` with optional additive metadata (`sidedness`, `multiplicity_adjusted`, `random_seed`, `requested_resamples`, `valid_resamples`, `invalid_resamples`).
2. **Deterministic Reproducibility**: All bootstrap intervals accept explicit random seeds via `AnalysisOptions(random_seed=...)`, use local `numpy.random.Generator`, and replay identically through `reproduce()`.
3. **Strict Serialization**: All bounds are finite numbers; degenerate cases produce structured `unavailable` records without `NaN` or `Infinity`.
4. **Audit Integration**: Every newly implemented interval is audited against mathematical bounds ($[-1, 1]$ or $[0, 1]$), ordered limits ($lower \le upper$), sample size requirements ($n > 3$ for Pearson), valid resample counts, and estimand coherence.
