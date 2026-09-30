# Effect-Size Confidence Interval Gaps & Handoff Matrix

This document establishes the explicit classification of missing effect-size confidence intervals
across all methods supported by PyAutoStat, the candidate defensible numerical methods, validation
requirements, and their implementation priority for future uncertainty-hardening milestones.

## Scientific Principle

PyAutoStat enforces complete honesty regarding uncertainty:
- Confidence intervals are reported only where an independently validated, scientifically defensible
  interval algorithm is implemented.
- Missing uncertainty is never disguised with dummy values, arbitrary placeholders, or silent omission.
- Where an effect-size confidence interval is not yet implemented, its status is explicitly declared as
  `unavailable`, `not_applicable`, or `not_supported`.
- Implementation of new confidence-interval algorithms is explicitly bounded to dedicated
  uncertainty-enhancement milestones and prohibited during pure contract-hardening phases.

## Effect-Size Confidence Interval Inventory & Gap Matrix

| Method ID | Method Name | Reported Effect Quantity | Current CI Status | Candidate Defensible CI Method | Validation Requirement | Milestone Priority |
|---|---|---|---|---|---|---|
| `welch_t` | Welch t-test | Cohen's d | `available` | Hedges-Olkin analytical Student-t noncentral inversion | SciPy noncentral t benchmark | Shipped |
| `student_t` | Student t-test | Cohen's d | `available` | Hedges-Olkin analytical Student-t noncentral inversion | SciPy noncentral t benchmark | Shipped |
| `paired_t` | Paired t-test | Paired Cohen's dz | `unavailable` | Analytical noncentral t inversion or pair-level percentile bootstrap | R `MBESS::ci.smd()` reference fixture | High |
| `one_sample_t` | One-sample t-test | Cohen's d | `unavailable` | Analytical noncentral t inversion (Cumming & Finch 2001) | R `MBESS::ci.smd()` reference fixture | High |
| `mann_whitney_u` | Mann-Whitney U | Rank-biserial r_rb | `available` | Kerby (2014) rank-biserial interval via sample pair resampling | Independent Monte Carlo & permutation benchmark | Shipped |
| `wilcoxon_signed_rank` | Paired Wilcoxon | Rank-biserial r_rb | `unavailable` | Pair-level percentile bootstrap or Kerby signed-rank interval | Non-parametric bootstrap reference fixture | Medium |
| `welch_anova` | Welch ANOVA | None (global) | `not_applicable` | None (no universally agreed standardized omnibus effect under heteroscedasticity) | Scientific policy: Games-Howell pairwise CIs are primary | Intentionally omitted |
| `one_way_anova` | One-way ANOVA | Eta-squared (η²) | `available` | Noncentral F distribution inversion (Smithson 2001) | R `MBESS::ci.pvaf()` / exact algebraic check | Shipped |
| `kruskal_wallis` | Kruskal-Wallis | Rank epsilon-squared | `available` | Truncated bootstrap percentile interval | Independent resampling reference fixture | Shipped |
| `pearson_correlation` | Pearson correlation | Pearson r | `unavailable` | Fisher z-transformation `tanh(atanh(r) ± z * 1/√(n-3))` | SciPy / statsmodels correlation interval cross-check | High |
| `spearman_correlation` | Spearman correlation | Spearman rho | `available` | Observation-pair percentile bootstrap | Resampling reference fixture | Shipped |
| `kendall_tau_b` | Kendall's tau-b | Kendall's tau-b | `available` | Observation-pair percentile bootstrap | Resampling reference fixture | Shipped |
| `point_biserial_correlation` | Point-biserial correlation | Point-biserial r_pb | `available` | Observation-pair percentile bootstrap | Resampling reference fixture | Shipped |
| `partial_pearson_correlation` | Partial Pearson correlation | Partial Pearson r | `available` | Fisher z-transformation on residual degrees of freedom `n - k - 3` | Independent linear regression residualization fixture | Shipped |
| `pearson_chi_square` | Chi-square test | Cramér's V | `available` | Noncentral chi-square distribution inversion (Smithson 2003) | R `MBESS::ci.pvaf()` / exact algebraic check | Shipped |
| `fisher_exact` | Fisher's exact test | Sample odds ratio | `unavailable` | Exact conditional maximum likelihood interval (Fisher/Clopper-Pearson) | R `fisher.test()$conf.int` reference fixture | High |
| `mcnemar` | McNemar test | Odds ratio (b/c) | `available` | Exact binomial / Clopper-Pearson interval on discordant pairs | Exact binomial reference fixture | Shipped |
| `linear_regression` | OLS linear regression | In-sample R² | `unavailable` | Noncentral F distribution inversion (Olkin & Finn 1995, Smithson 2001) | R `MBESS::ci.R2()` reference fixture | Low |
| `logistic_regression` | Binary logistic regression | Odds ratio per predictor | `available` | Exponentiated Wald z-interval `[exp(b_j - z*se), exp(b_j + z*se)]` | statsmodels Logit conf_int exponentiated | Shipped |
| `cronbach_alpha` | Cronbach's alpha | Scale alpha (α) | `available` | Respondent-row percentile bootstrap | Independent psychometric reference fixture | Shipped |
| `repeated_measures_anova` | Repeated-measures ANOVA | Partial eta-squared | `unavailable` | Noncentral F distribution inversion for partial eta-squared | R `MBESS::ci.pvaf()` / `effectsize::eta_squared()` | Medium |
| `friedman_test` | Friedman rank-sum | Kendall's W | `unavailable` | Participant/unit-level block bootstrap or transformation from Friedman Q | Non-parametric panel resampling fixture | Medium |

---

## Detailed Gap Analysis & Proposed Milestone Treatment

### 1. Paired Cohen's dz (`paired_t`)
- **Quantity**: Sample mean of paired differences divided by the standard deviation of paired differences ($d_z = \bar{D} / s_D$).
- **Current Status**: `unavailable` (point estimate and mean difference CI are reported; $d_z$ CI is omitted).
- **Candidate Method**: Exact confidence limits based on inverting the noncentral t distribution with noncentrality parameter $\delta = d_z \sqrt{n}$ and $df = n - 1$.
- **Validation**: Compare against R package `MBESS::ci.smd()` for paired designs across balanced fixtures.
- **Priority**: High.

### 2. One-Sample Cohen's d (`one_sample_t`)
- **Quantity**: Sample mean minus declared reference value divided by sample standard deviation ($d = (\bar{x} - \mu_0) / s$).
- **Current Status**: `unavailable` (mean contrast CI is reported; standardized $d$ CI is omitted).
- **Candidate Method**: Exact noncentral t distribution inversion with $\delta = d \sqrt{n}$ and $df = n - 1$.
- **Validation**: Compare against R `MBESS::ci.smd()` one-sample mode.
- **Priority**: High.

### 3. Fisher's Exact Test Odds Ratio (`fisher_exact`)
- **Quantity**: Sample cross-product odds ratio ($ad / bc$) for $2 \times 2$ contingency tables.
- **Current Status**: `unavailable` (Fisher exact p-value is reported; sample OR is reported without uncertainty).
- **Candidate Method**: Exact conditional odds-ratio interval via inversion of the hypergeometric distribution (Fisher's exact confidence interval) or Agresti-Min score interval.
- **Validation**: Compare against R `fisher.test(..., conf.int=TRUE)$conf.int` and `statsmodels.stats.contingency_tables.Table2x2`.
- **Priority**: High.

### 4. Pearson Correlation Execution Dispatch (`pearson_correlation`)
- **Quantity**: Population linear correlation coefficient ($r$).
- **Current Status**: `unavailable` in `execution.py` unified dispatch envelope (though computed in `StatisticalAnalyzer` legacy profile).
- **Candidate Method**: Standard analytical Fisher z-transformation:
  $$z = \frac{1}{2} \ln\left(\frac{1+r}{1-r}\right), \quad SE = \frac{1}{\sqrt{n-3}}, \quad [z_L, z_U] = z \pm z_{1-\alpha/2} SE, \quad [r_L, r_U] = \tanh([z_L, z_U])$$
- **Validation**: Exact numerical check against SciPy `stats.pearsonr` confidence interval (available in SciPy $\ge 1.9$).
- **Priority**: High.

### 5. Paired Wilcoxon Rank-Biserial (`wilcoxon_signed_rank`)
- **Quantity**: Proportion of positive signed ranks minus proportion of negative signed ranks ($r_{rb} = (W^+ - W^-) / T$).
- **Current Status**: `unavailable`.
- **Candidate Method**: Bounded participant-level percentile bootstrap interval (resampling pair differences with replacement).
- **Validation**: Benchmark against R `effectsize::rank_biserial()` bootstrap mode.
- **Priority**: Medium.

### 6. Friedman Kendall's W (`friedman_test`)
- **Quantity**: Kendall's coefficient of concordance ($W = Q / [n(k - 1)]$).
- **Current Status**: `unavailable`.
- **Candidate Method**: Complete unit/participant-level block bootstrap (resampling complete row vectors across conditions).
- **Validation**: Benchmark against R `DescTools::KendallW(..., boot=TRUE)`.
- **Priority**: Medium.

### 7. Repeated-Measures ANOVA Partial Eta-Squared (`repeated_measures_anova`)
- **Quantity**: Partial eta-squared ($\eta_p^2 = SS_{\text{condition}} / [SS_{\text{condition}} + SS_{\text{error}}]$).
- **Current Status**: `unavailable`.
- **Candidate Method**: Noncentral F inversion based on the observed repeated-measures F statistic and condition/error degrees of freedom.
- **Validation**: Compare against R `effectsize::eta_squared(..., partial=TRUE)` and `MBESS::ci.pvaf()`.
- **Priority**: Medium.

### 8. Kruskal-Wallis Dunn Pairwise Rank-Biserial Follow-up
- **Quantity**: Contrast-level rank-biserial correlation for post-hoc pairwise rank comparisons.
- **Current Status**: `unavailable`.
- **Candidate Method**: Pairwise bootstrap percentile intervals with Holm multiplicity disclaimer.
- **Validation**: Independent non-parametric resampling fixture.
- **Priority**: Low.

### 9. OLS In-Sample R² (`linear_regression`)
- **Quantity**: Proportion of sample outcome variance explained by the fitted linear model ($R^2$).
- **Current Status**: `unavailable` (model-level F and predictor-level Wald intervals are reported).
- **Candidate Method**: Inversion of the noncentral F distribution (Smithson 2001) or case-based bootstrap.
- **Validation**: Compare against R `MBESS::ci.R2()`.
- **Priority**: Low (scientifically, in-sample $R^2$ is descriptive; predictor coefficient CIs are primary).

### 10. Welch ANOVA Global Standardized Effect (`welch_anova`)
- **Quantity**: Standardized omnibus effect for heteroscedastic multi-group designs.
- **Current Status**: `not_applicable` (by scientific policy).
- **Candidate Method**: None. Standardized omnibus measures such as classical $\eta^2$ or $\omega^2$ assume homoscedasticity. Inventing a synthetic standardized effect for Welch ANOVA is scientifically misleading.
- **Validation**: Pairwise Games-Howell simultaneous confidence intervals are already provided and validated.
- **Priority**: None (intentionally excluded from scope).

---

## Architectural Constraints for Future Uncertainty Milestones

1. **No Breaking Return Schemas**: New intervals must populate `confidence_interval` dictionaries within the existing `effect_size` structure following the established schema: `{"lower": float, "upper": float, "level": float, "method": str, "quantity": str}`.
2. **Deterministic Reproducibility**: All bootstrap-based intervals must accept and record explicit random seeds (`AnalysisOptions(random_seed=...)`) and default to reproducible deterministic seeds in unseeded contexts.
3. **Strict Serialization**: All calculated bounds must be finite numbers; degenerate bounds must return structured unavailable payloads without `NaN` or `Infinity`.
4. **Audit Integration**: Every newly implemented interval must be accompanied by audit invariant checks (`lower <= upper`, finite probability bounds, contrast direction alignment).
