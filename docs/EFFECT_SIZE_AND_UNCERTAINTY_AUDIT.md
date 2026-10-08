# PyAutoStat Effect-Size & Uncertainty Gap Audit

Authoritative scientific audit of effect sizes, point estimates, uncertainty intervals, runtime field locations, numerical validation levels, and prioritization across all 24 registered statistical methods in PyAutoStat.

---

## 1. Methodology & Classification Criteria

This document provides a disciplined, audit-first assessment of PyAutoStat's effect-size and uncertainty coverage. To maintain scientific integrity, this audit strictly distinguishes between:

1. **Product Gap vs. Validation Gap**:
   - A quantity computed and exposed at runtime that currently lacks an independent benchmark in the standalone validation framework is classified as `PRESENT_BUT_VALIDATION_GAP`, **not** a missing feature.
   - Only quantities that are scientifically justified for the stated estimand but entirely absent from runtime computation are evaluated as potential product gaps.

2. **Estimand Consistency**:
   - Alternative metrics that change the scientific estimand (e.g. Risk Ratio vs. Odds Ratio) are explicitly identified and barred from being conflated with the existing method's completeness.

### Product Gap Classifications
- `COMPLETE_FOR_CURRENT_SCOPE`: Point estimate, effect size, and uncertainty intervals are fully implemented, ergonomically accessible, and aligned with standard scientific reporting.
- `PRESENT_BUT_ERGONOMICS_GAP`: Computed at runtime but previously difficult to access (all 0 following ergonomics hardening).
- `PRESENT_BUT_VALIDATION_GAP`: Computed at runtime (e.g. bootstrap CIs, exact Fisher OR CIs, exact ICC CIs) but currently classified as Level D (deferred) in the independent validation framework.
- `TRUE_HIGH_VALUE_GAP`: High scientific value, identical estimand, absent from runtime, and actionable for subsequent implementation.
- `OPTIONAL_ALTERNATIVE_METRIC`: Valid alternative formula or correction factor that may be added in the future without altering the estimand.
- `NOT_APPLICABLE`: Estimand does not define an inferential effect size or standardized magnitude (e.g., omnibus tests without post-hocs).
- `OUT_OF_SCOPE`: Incompatible with design-aware non-generative paradigm or changes research question.

---

## 2. Exhaustive 24-Method Audit

### 1. `welch_t` (Welch's Independent-Samples t-Test)
- **Primary Estimand**: Unstandardized difference in population means ($\Delta\mu = \mu_1 - \mu_2$) under unequal variances.
- **Primary Estimate**: Sample mean difference ($\bar{x}_1 - \bar{x}_2$).
- **Primary Estimate CI**: Available (95% Welch-Satterthwaite analytical $t$-interval).
- **Effect Size Quantity**: Cohen's $d$ ($(\bar{x}_1 - \bar{x}_2) / s_{\text{pooled}}$).
- **Effect Size Definition**: Standardized mean difference.
- **Effect Size CI Status**: Available (percentile/BCa bootstrap interval).
- **Runtime Field Location**: `values["effect_size"]["value"]`, `values["effect_size"]["confidence_interval"]`.
- **CI Method**: Analytical for primary estimate; non-parametric bootstrap for Cohen's $d$.
- **Validation Evidence (Estimate)**: Level A (SciPy/R).
- **Validation Evidence (Interval)**: Level A for primary CI; Level D for bootstrap effect CI.
- **Interpretation Caveat**: Cohen's $d$ pools standard deviations even when Welch's $t$ assumes variance heterogeneity.
- **Alternative Candidates**: Hedges' $g$ (same estimand, small-sample correction); Glass's $\Delta$ (same estimand, control SD standardization).
- **Classification**: `PRESENT_BUT_VALIDATION_GAP`.

### 2. `student_t` (Student's Independent-Samples t-Test)
- **Primary Estimand**: Unstandardized difference in population means ($\Delta\mu = \mu_1 - \mu_2$) under equal variances.
- **Primary Estimate**: Sample mean difference ($\bar{x}_1 - \bar{x}_2$).
- **Primary Estimate CI**: Available (95% Student $t$-interval).
- **Effect Size Quantity**: Cohen's $d$.
- **Effect Size Definition**: Standardized mean difference with pooled standard deviation.
- **Effect Size CI Status**: Available (bootstrap interval).
- **Runtime Field Location**: `values["effect_size"]["value"]`, `values["effect_size"]["confidence_interval"]`.
- **CI Method**: Analytical Student-$t$ for primary; bootstrap for effect size.
- **Validation Evidence (Estimate)**: Level A.
- **Validation Evidence (Interval)**: Level A for primary; Level D for bootstrap effect CI.
- **Interpretation Caveat**: Highly sensitive to variance heterogeneity and skewness in small samples.
- **Alternative Candidates**: Hedges' $g$ (same estimand, small-sample correction).
- **Classification**: `PRESENT_BUT_VALIDATION_GAP`.

### 3. `paired_t` (Paired-Samples t-Test)
- **Primary Estimand**: Mean of within-subject pairwise differences ($\mu_D$).
- **Primary Estimate**: Sample mean difference ($\bar{D}$).
- **Primary Estimate CI**: Available (95% paired $t$-interval).
- **Effect Size Quantity**: Cohen's $d_z$ ($\bar{D} / s_D$).
- **Effect Size Definition**: Standardized mean difference of paired observations.
- **Effect Size CI Status**: Available (bootstrap interval).
- **Runtime Field Location**: `values["effect_size"]["value"]`, `values["effect_size"]["confidence_interval"]`.
- **CI Method**: Analytical Student-$t$ for primary; bootstrap for $d_z$.
- **Validation Evidence (Estimate)**: Level A.
- **Validation Evidence (Interval)**: Level A for primary; Level D for bootstrap effect CI.
- **Interpretation Caveat**: $d_z$ depends strongly on pre-post correlation; cannot be directly compared to independent-groups $d$.
- **Alternative Candidates**: Cohen's $d_{\text{av}}$ (standardized by average variance of original variables).
- **Classification**: `PRESENT_BUT_VALIDATION_GAP`.

### 4. `one_sample_t` (One-Sample t-Test)
- **Primary Estimand**: Difference between population mean and reference benchmark ($\Delta\mu = \mu - \mu_0$).
- **Primary Estimate**: Mean difference from reference ($\bar{x} - \mu_0$, stored in `primary_estimate` with `estimate_name: "mean difference from reference"`; sample mean $\bar{x}$ is reported as supporting descriptive quantity).
- **Primary Estimate CI**: Available (95% analytical $t$-interval for mean difference from reference).
- **Effect Size Quantity**: One-sample Cohen's $d$ ($(\bar{x} - \mu_0) / s$).
- **Effect Size Definition**: Standardized distance from reference benchmark.
- **Effect Size CI Status**: Available (bootstrap interval).
- **Runtime Field Location**: `values["effect_size"]["value"]`, `values["effect_size"]["confidence_interval"]`.
- **CI Method**: Analytical Student-$t$ for primary; bootstrap for $d$.
- **Validation Evidence (Estimate)**: Level A.
- **Validation Evidence (Interval)**: Level A for primary; Level D for bootstrap effect CI.
- **Interpretation Caveat**: Assumes normal distribution of population scores.
- **Alternative Candidates**: Hedges' $g$ (small-sample correction; optional future metric).
- **Classification**: `PRESENT_BUT_VALIDATION_GAP`.

### 5. `mann_whitney_u` (Mann-Whitney U Test)
- **Primary Estimand**: Stochastic dominance / probability that a randomly chosen observation from Group 1 exceeds Group 2 ($P(X > Y)$).
- **Primary Estimate**: Rank-biserial correlation $r_{\text{rb}} = 1 - \frac{2U}{n_1 n_2}$.
- **Primary Estimate CI**: Available (bootstrap interval on $r_{\text{rb}}$).
- **Effect Size Quantity**: Rank-biserial correlation ($r_{\text{rb}}$).
- **Effect Size Definition**: Proportionate difference in favorable vs. unfavorable pairwise ranks.
- **Effect Size CI Status**: Available (bootstrap interval).
- **Runtime Field Location**: `values["effect_size"]["value"]`, `values["effect_size"]["confidence_interval"]`.
- **CI Method**: Non-parametric bootstrap resample.
- **Validation Evidence (Estimate)**: Level A for $U$; Level B for $r_{\text{rb}}$.
- **Validation Evidence (Interval)**: Level C.
- **Interpretation Caveat**: Tests stochastic superiority; does NOT test differences in population medians unless distributions have identical shapes.
- **Alternative Candidates**: Common language effect size (CLES) / Area under ROC curve (AUC = $(r_{\text{rb}} + 1)/2$, same estimand).
- **Classification**: `COMPLETE_FOR_CURRENT_SCOPE`.

### 6. `wilcoxon_signed_rank` (Wilcoxon Signed-Rank Test)
- **Primary Estimand**: Symmetry and median of paired differences around zero.
- **Primary Estimate**: Matched-pairs rank-biserial correlation ($r_{\text{rb}} = \frac{W_+ - W_-}{W_+ + W_-}$).
- **Primary Estimate CI**: Available (bootstrap interval on $r_{\text{rb}}$).
- **Effect Size Quantity**: Matched-pairs rank-biserial correlation.
- **Effect Size Definition**: Difference between positive and negative signed rank sums over total rank sum.
- **Effect Size CI Status**: Available (bootstrap interval).
- **Runtime Field Location**: `values["effect_size"]["value"]`, `values["effect_size"]["confidence_interval"]`.
- **CI Method**: Non-parametric bootstrap.
- **Validation Evidence (Estimate)**: Level A for $W$; Level B for $r_{\text{rb}}$.
- **Validation Evidence (Interval)**: Level D for bootstrap interval.
- **Interpretation Caveat**: Zero differences are excluded by default (Pratt/Wilcoxon handling documented in audit).
- **Alternative Candidates**: None required.
- **Classification**: `PRESENT_BUT_VALIDATION_GAP`.

### 7. `welch_anova` (Welch's One-Way ANOVA)
- **Primary Estimand**: Equality of $k \ge 3$ population means under variance heterogeneity.
- **Primary Estimate**: Omnibus Welch $F$-statistic; pairwise Games-Howell mean differences.
- **Primary Estimate CI**: Unavailable for omnibus test; available for pairwise Games-Howell comparisons.
- **Effect Size Quantity**: Global standardized effect (`name: "global standardized effect"`, `value: None`, `status: "not_applicable"`).
- **Effect Size Definition**: Global omnibus effect size is not applicable for Welch's ANOVA in PyAutoStat; pairwise Games-Howell unstandardized mean differences serve as primary local contrasts.
- **Effect Size CI Status**: Not applicable (pairwise Games-Howell post-hoc CIs provided).
- **Runtime Field Location**: `values["effect_size"]["value"]` (`None`), `values["post_hoc"]["comparisons"]`.
- **CI Method**: Games-Howell Welch-t intervals for post-hoc pairs.
- **Validation Evidence (Estimate)**: Level A for omnibus Welch $F$.
- **Validation Evidence (Interval)**: Level C for post-hoc intervals.
- **Interpretation Caveat**: Global omnibus effect size is not defined/emitted; pairwise Games-Howell comparisons locate group differences directly.
- **Alternative Candidates**: Pairwise Games-Howell CIs already provide unstandardized pairwise contrasts.
- **Classification**: `COMPLETE_FOR_CURRENT_SCOPE`.

### 8. `one_way_anova` (Fisher's One-Way ANOVA)
- **Primary Estimand**: Equality of $k \ge 3$ population means under homogeneity of variance.
- **Primary Estimate**: Omnibus $F$-statistic (Tukey HSD pairwise differences).
- **Primary Estimate CI**: Unavailable for omnibus; available for Tukey HSD comparisons.
- **Effect Size Quantity**: Eta-squared ($\eta^2 = \text{SS}_{\text{between}} / \text{SS}_{\text{total}}$).
- **Effect Size Definition**: Sample proportion of variance accounted for by group membership.
- **Effect Size CI Status**: Unavailable.
- **Runtime Field Location**: `values["effect_size"]["value"]`.
- **CI Method**: None for omnibus $\eta^2$.
- **Validation Evidence (Estimate)**: Level A for $F$ and $\eta^2$.
- **Validation Evidence (Interval)**: Level A for Tukey post-hoc CIs.
- **Interpretation Caveat**: $\eta^2$ has positive sample bias; omega-squared provides an unbiased alternative.
- **Alternative Candidates**: Omega-squared ($\omega^2$, same estimand, less biased).
- **Classification**: `OPTIONAL_ALTERNATIVE_METRIC`.

### 9. `kruskal_wallis` (Kruskal-Wallis Test)
- **Primary Estimand**: Equality of mean rank distributions across $k \ge 3$ independent groups.
- **Primary Estimate**: Omnibus $H$-statistic (Dunn-Bonferroni rank comparisons).
- **Primary Estimate CI**: Unavailable for omnibus test.
- **Effect Size Quantity**: Epsilon-squared ($\epsilon^2 = \frac{H - k + 1}{N - k}$ or $\frac{H}{(N^2 - 1)/(N + 1)}$).
- **Effect Size Definition**: Proportion of rank variance explained.
- **Effect Size CI Status**: Unavailable.
- **Runtime Field Location**: `values["effect_size"]["value"]`.
- **CI Method**: None for omnibus $\epsilon^2$.
- **Validation Evidence (Estimate)**: Level A for $H$; Level B for $\epsilon^2$.
- **Validation Evidence (Interval)**: Not applicable.
- **Interpretation Caveat**: Does not demonstrate equal group shifts unless shapes are identical.
- **Alternative Candidates**: Rank-biserial correlations for post-hoc pairwise Dunn tests.
- **Classification**: `COMPLETE_FOR_CURRENT_SCOPE`.

### 10. `pearson_correlation` (Pearson Product-Moment Correlation)
- **Primary Estimand**: Linear bivariate association parameter ($\rho$).
- **Primary Estimate**: Sample Pearson correlation ($r$).
- **Primary Estimate CI**: Available (95% Fisher $z$-transformation interval).
- **Effect Size Quantity**: Correlation coefficient ($r$).
- **Effect Size Definition**: Standardized bivariate linear covariance.
- **Effect Size CI Status**: Available (Fisher $z$ CI).
- **Runtime Field Location**: `values["primary_estimate"]`, `values["confidence_interval"]`.
- **CI Method**: Analytical Fisher $z$-transformation interval.
- **Validation Evidence (Estimate)**: Level A.
- **Validation Evidence (Interval)**: Level A.
- **Interpretation Caveat**: Sensitive to non-normality and outliers; bounded $[-1, 1]$. Strictly stored values reported without derived $n-2$ df.
- **Alternative Candidates**: $r^2$ (coefficient of determination, shared variance).
- **Classification**: `COMPLETE_FOR_CURRENT_SCOPE`.

### 11. `spearman_correlation` (Spearman Rank Correlation)
- **Primary Estimand**: Monotonic bivariate association parameter ($\rho_s$).
- **Primary Estimate**: Sample Spearman rank correlation ($r_s$).
- **Primary Estimate CI**: Available (paired non-parametric bootstrap interval).
- **Effect Size Quantity**: Rank correlation ($r_s$).
- **Effect Size Definition**: Pearson correlation calculated over ranked observations.
- **Effect Size CI Status**: Available (bootstrap interval).
- **Runtime Field Location**: `values["primary_estimate"]`, `values["confidence_interval"]`.
- **CI Method**: Paired row bootstrap.
- **Validation Evidence (Estimate)**: Level A for $r_s$ and $p$.
- **Validation Evidence (Interval)**: Level D for bootstrap interval.
- **Interpretation Caveat**: Measures monotonic association; non-linear monotonic relationships produce $|r_s| = 1.0$.
- **Alternative Candidates**: Fisher $z$ approximation on ranks (analytical).
- **Classification**: `PRESENT_BUT_VALIDATION_GAP`.

### 12. `kendall_tau_b` (Kendall's Tau-b)
- **Primary Estimand**: Population rank concordance parameter ($\tau_b$).
- **Primary Estimate**: Sample Kendall's $\tau_b$.
- **Primary Estimate CI**: Available (asymptotic analytical normal approximation).
- **Effect Size Quantity**: Concordance index ($\tau_b$).
- **Effect Size Definition**: Relative difference between concordant and discordant pairs adjusted for ties.
- **Effect Size CI Status**: Available (analytical interval).
- **Runtime Field Location**: `values["primary_estimate"]`, `values["confidence_interval"]`.
- **CI Method**: Asymptotic standard error using Kendall's variance estimator.
- **Validation Evidence (Estimate)**: Level A.
- **Validation Evidence (Interval)**: Level C.
- **Interpretation Caveat**: Values are numerically smaller than Spearman $r_s$ on the same data.
- **Alternative Candidates**: None required.
- **Classification**: `COMPLETE_FOR_CURRENT_SCOPE`.

### 13. `point_biserial_correlation` (Point-Biserial Correlation)
- **Primary Estimand**: Linear association between continuous variable and natural binary indicator.
- **Primary Estimate**: Sample point-biserial correlation ($r_{\text{pb}}$).
- **Primary Estimate CI**: Available (analytical confidence interval).
- **Effect Size Quantity**: Correlation ($r_{\text{pb}}$).
- **Effect Size Definition**: Equivalent to Pearson $r$ with $\{0, 1\}$ binary coding.
- **Effect Size CI Status**: Available.
- **Runtime Field Location**: `values["primary_estimate"]`, `values["confidence_interval"]`.
- **CI Method**: Analytical Fisher $z$-transformation.
- **Validation Evidence (Estimate)**: Level A.
- **Validation Evidence (Interval)**: Level C.
- **Interpretation Caveat**: Severely attenuated if binary group split is unbalanced (e.g. 90/10 split).
- **Alternative Candidates**: Cohen's $d$ algebraically converted from $r_{\text{pb}}$.
- **Classification**: `COMPLETE_FOR_CURRENT_SCOPE`.

### 14. `partial_pearson_correlation` (Partial Pearson Correlation)
- **Primary Estimand**: Bivariate linear association conditional on controlling for $k \ge 1$ continuous covariates.
- **Primary Estimate**: Sample partial correlation ($r_{\text{part}}$).
- **Primary Estimate CI**: Available (Fisher $z$-transformation with adjusted error $\text{SE} = 1/\sqrt{n - 3 - k}$).
- **Effect Size Quantity**: Partial correlation ($r_{\text{part}}$).
- **Effect Size Definition**: Correlation between residuals after projecting out covariates.
- **Effect Size CI Status**: Available.
- **Runtime Field Location**: `values["primary_estimate"]`, `values["confidence_interval"]`.
- **CI Method**: Analytical Fisher $z$ adjusted for covariate degrees of freedom.
- **Validation Evidence (Estimate)**: Level A.
- **Validation Evidence (Interval)**: Level D for standalone verification.
- **Interpretation Caveat**: Assumes linear relationship between controlled variables and target pairs.
- **Alternative Candidates**: None required.
- **Classification**: `PRESENT_BUT_VALIDATION_GAP`.

### 15. `pearson_chi_square` (Pearson's Chi-Square Test of Independence)
- **Primary Estimand**: Independence between two categorical classifications.
- **Primary Estimate**: Cramer's $V = \sqrt{\frac{\chi^2}{N \cdot \min(r-1, c-1)}}$.
- **Primary Estimate CI**: Available (bootstrap interval on Cramer's $V$).
- **Effect Size Quantity**: Cramer's $V$.
- **Effect Size Definition**: Scale-adjusted measure of nominal contingency association.
- **Effect Size CI Status**: Available (bootstrap interval).
- **Runtime Field Location**: `values["primary_estimate"]`, `values["effect_size"]["confidence_interval"]`.
- **CI Method**: Contingency table bootstrap.
- **Validation Evidence (Estimate)**: Level A for $\chi^2$; Level B for $V$.
- **Validation Evidence (Interval)**: Level D for bootstrap interval.
- **Interpretation Caveat**: Does not imply directional association; influenced by table dimensionality.
- **Alternative Candidates**: Phi coefficient ($\phi$) for $2 \times 2$ tables.
- **Classification**: `PRESENT_BUT_VALIDATION_GAP`.

### 16. `fisher_exact` (Fisher's Exact Test)
- **Primary Estimand**: Non-random association in a $2 \times 2$ table under fixed marginals.
- **Primary Estimate**: Sample odds ratio ($\text{OR} = \frac{a \cdot d}{b \cdot c}$).
- **Primary Estimate CI**: Available (exact conditional inversion interval).
- **Effect Size Quantity**: Odds ratio ($\text{OR}$).
- **Effect Size Definition**: Ratio of odds of exposure in cases vs. controls.
- **Effect Size CI Status**: Available.
- **Runtime Field Location**: `values["primary_estimate"]`, `values["confidence_interval"]`.
- **CI Method**: Exact hypergeometric inversion.
- **Validation Evidence (Estimate)**: Level A for $p$; Level B for sample OR.
- **Validation Evidence (Interval)**: Level D in validation framework benchmark.
- **Interpretation Caveat**: Sample OR differs from unconditional MLE; odds ratios are not relative risks.
- **Alternative Candidates**: Risk Ratio (CHANGES ESTIMAND; only valid in prospective cohorts).
- **Classification**: `PRESENT_BUT_VALIDATION_GAP`.

### 17. `mcnemar` (McNemar's Paired Binary Test)
- **Primary Estimand**: Equality of marginal probabilities in paired binary observations.
- **Primary Estimate**: Paired proportion difference ($\Delta p = \frac{b - c}{N}$).
- **Primary Estimate CI**: Available (Wilson-Newcombe score interval for paired proportions).
- **Effect Size Quantity**: Paired proportion difference.
- **Effect Size Definition**: Difference between discordant response proportions.
- **Effect Size CI Status**: Available.
- **Runtime Field Location**: `values["primary_estimate"]`, `values["confidence_interval"]`.
- **CI Method**: Wilson-Newcombe score method.
- **Validation Evidence (Estimate)**: Level A for exact binomial $p$; Level B for $\Delta p$.
- **Validation Evidence (Interval)**: Level C.
- **Interpretation Caveat**: Supported procedure is the exact paired binomial test; concordant pairs do not contribute to inference.
- **Alternative Candidates**: Discordant odds ratio ($b / c$).
- **Classification**: `COMPLETE_FOR_CURRENT_SCOPE`.

### 18. `linear_regression` (Multiple Linear Regression)
- **Primary Estimand**: Conditional expectation surface and partial linear regression slopes ($\beta_j$).
- **Primary Estimate**: Unstandardized regression coefficients ($B_j$).
- **Primary Estimate CI**: Available (95% analytical intervals for each predictor using HC3 robust standard errors).
- **Effect Size Quantity**: Coefficient of determination ($R^2$, adjusted $R^2$) in `model_fit`.
- **Effect Size Definition**: Proportion of variance in the outcome explained by the linear predictor set.
- **Effect Size CI Status**: Unavailable for $R^2$.
- **Runtime Field Location**: `values["model_fit"]["r_squared"]`, `values["coefficients"]`.
- **CI Method**: HC3 heteroskedasticity-robust covariance matrix.
- **Validation Evidence (Estimate)**: Level A for $B_j$, $\text{SE}$, $t$, $p$, and $R^2$.
- **Validation Evidence (Interval)**: Level A for coefficient CIs.
- **Interpretation Caveat**: Unstandardized coefficients depend on measurement units; omnibus scalars return `None` to prevent single-number distortion.
- **Alternative Candidates**: Standardized regression coefficients ($\beta^*_j$, scale-independent rescaled magnitude, optional future enhancement).
- **Classification**: `COMPLETE_FOR_CURRENT_SCOPE` (Unstandardized coefficients and HC3 robust CIs are complete; standardized betas are optional future metrics).

### 19. `logistic_regression` (Binary Logistic Regression)
- **Primary Estimand**: Conditional log-odds of binary outcome event ($P(Y = 1 \mid X)$).
- **Primary Estimate**: Predictor odds ratios ($\text{OR}_j = \exp(B_j)$).
- **Primary Estimate CI**: Available (95% Wald confidence intervals for each $\text{OR}_j$).
- **Effect Size Quantity**: Predictor odds ratios and McFadden's pseudo-$R^2$.
- **Effect Size Definition**: Multiplicative change in event odds per unit increase in predictor; log-likelihood ratio index.
- **Effect Size CI Status**: Available for all predictor odds ratios.
- **Runtime Field Location**: `values["coefficients"][i]["odds_ratio"]`, `values["coefficients"][i]["odds_ratio_ci"]`, `values["model_fit"]["mcfadden_r2"]`.
- **CI Method**: Exponentiated Wald standard errors.
- **Validation Evidence (Estimate)**: Level A.
- **Validation Evidence (Interval)**: Level A for OR CIs.
- **Interpretation Caveat**: Odds ratios describe relative odds, not probabilities or risk ratios; event level disclosure is mandatory.
- **Alternative Candidates**: Average marginal effects (changes estimand to probability differences).
- **Classification**: `COMPLETE_FOR_CURRENT_SCOPE`.

### 20. `cronbach_alpha` (Cronbach's Alpha Internal Consistency)
- **Primary Estimand**: Lower-bound internal consistency reliability coefficient for an additive composite scale ($\alpha$).
- **Primary Estimate**: Sample Cronbach's $\alpha$.
- **Primary Estimate CI**: Available (when uncertainty bootstrap is selected).
- **Effect Size Quantity**: Reliability coefficient ($\alpha$).
- **Effect Size Definition**: Proportion of total test score variance attributable to common covariance among items.
- **Effect Size CI Status**: Available via bootstrap when requested.
- **Runtime Field Location**: `values["primary_estimate"]`, `values["confidence_interval"]`.
- **CI Method**: Respondent-level bootstrap.
- **Validation Evidence (Estimate)**: Level A.
- **Validation Evidence (Interval)**: Level D for bootstrap validation.
- **Interpretation Caveat**: Descriptive measurement index, not a significance test; sensitive to scale length.
- **Alternative Candidates**: Feldt analytical exact $F$-interval (same estimand).
- **Classification**: `PRESENT_BUT_VALIDATION_GAP`.

### 21. `repeated_measures_anova` (One-Way Repeated-Measures ANOVA)
- **Primary Estimand**: Differences across within-subject conditions over repeated exposures.
- **Primary Estimate**: Omnibus $F$-statistic (with Greenhouse-Geisser or Huynh-Feldt epsilon correction).
- **Primary Estimate CI**: Unavailable for omnibus.
- **Effect Size Quantity**: Partial eta-squared ($\eta_p^2 = \frac{\text{SS}_{\text{condition}}}{\text{SS}_{\text{condition}} + \text{SS}_{\text{error}}}$).
- **Effect Size Definition**: Proportion of variance associated with repeated factor excluding individual subject variance.
- **Effect Size CI Status**: Unavailable.
- **Runtime Field Location**: `values["effect_size"]["value"]`.
- **CI Method**: None for $\eta_p^2$.
- **Validation Evidence (Estimate)**: Level A.
- **Validation Evidence (Interval)**: Level A for Bonferroni post-hoc pairwise CIs.
- **Interpretation Caveat**: Sphericity violations require epsilon adjustment of degrees of freedom.
- **Alternative Candidates**: Generalized eta-squared ($\eta_g^2$).
- **Classification**: `COMPLETE_FOR_CURRENT_SCOPE`.

### 22. `friedman_test` (Friedman Test)
- **Primary Estimand**: Equality of rank distributions across $k \ge 3$ repeated/matched conditions.
- **Primary Estimate**: Omnibus $Q$-statistic.
- **Primary Estimate CI**: Unavailable for omnibus test.
- **Effect Size Quantity**: Kendall's $W = \frac{Q}{N(k - 1)}$.
- **Effect Size Definition**: Coefficient of concordance across repeated rankings ($0 \le W \le 1$).
- **Effect Size CI Status**: Available (bootstrap interval).
- **Runtime Field Location**: `values["effect_size"]["value"]`, `values["effect_size"]["confidence_interval"]`.
- **CI Method**: Non-parametric bootstrap resample (`friedman_kendall_w_bootstrap_ci`).
- **Validation Evidence (Estimate)**: Level A for $Q$; Level B for $W$.
- **Validation Evidence (Interval)**: Level D for deferred bootstrap interval.
- **Interpretation Caveat**: Non-parametric analog of repeated ANOVA; ranks within each participant block.
- **Alternative Candidates**: None required.
- **Classification**: `PRESENT_BUT_VALIDATION_GAP`.

### 23. `two_way_anova` (Two-Way Factorial ANOVA)
- **Primary Estimand**: Factor A main effect, Factor B main effect, and Factor $A \times B$ interaction.
- **Primary Estimate**: Term-specific $F$-statistics and partial eta-squared values.
- **Primary Estimate CI**: Unavailable for omnibus terms.
- **Effect Size Quantity**: Partial eta-squared per term ($\eta_p^2 = \frac{\text{SS}_{\text{term}}}{\text{SS}_{\text{term}} + \text{SS}_{\text{residual}}}$).
- **Effect Size Definition**: Proportion of variance explained by term relative to term plus residual error.
- **Effect Size CI Status**: Unavailable.
- **Runtime Field Location**: `values["terms"][i]["effect_size"]["value"]`.
- **CI Method**: None for $\eta_p^2$.
- **Validation Evidence (Estimate)**: Level A.
- **Validation Evidence (Interval)**: Level B for cell summaries.
- **Interpretation Caveat**: Sum of squares type (Type II for unbalanced non-interaction, Type III for interactions) dictates hypothesis tested.
- **Alternative Candidates**: Partial omega-squared ($\omega_p^2$).
- **Classification**: `COMPLETE_FOR_CURRENT_SCOPE`.

### 24. `intraclass_correlation` (Intraclass Correlation Coefficient)
- **Primary Estimand**: Reliability, agreement, or consistency across raters/measurements.
- **Primary Estimate**: Sample ICC coefficient for selected form (e.g. ICC(2,1), ICC(3,k)).
- **Primary Estimate CI**: Available (95% analytical exact $F$-inversion interval).
- **Effect Size Quantity**: ICC coefficient.
- **Effect Size Definition**: Proportion of variance attributable to rated targets relative to total variance.
- **Effect Size CI Status**: Available (exact $F$-inversion).
- **Runtime Field Location**: `values["primary_estimate"]`, `values["confidence_interval"]`.
- **CI Method**: Exact analytical McGraw & Wong $F$-distribution inversion.
- **Validation Evidence (Estimate)**: Level A.
- **Validation Evidence (Interval)**: Level D for standalone verification harness.
- **Interpretation Caveat**: Form must match experimental design: one-way vs. two-way, random vs. mixed, single vs. average, agreement vs. consistency.
- **Alternative Candidates**: None; all 6 canonical forms are implemented.
- **Classification**: `PRESENT_BUT_VALIDATION_GAP`.

---

## 3. Candidate Effect-Size Review Without Auto-Implementation

| Domain / Method | Candidate Metric | Answers Same Estimand? | Scientific Value | Recommendation | Justification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean comparisons** | Hedges' $g$ | Yes (standardized mean diff) | High | **OPTIONAL_FUTURE** | Small-sample bias-corrected standardized mean difference ($N < 20$); non-blocking future enhancement since raw mean differences, Cohen's d, and CIs are fully reported. |
| **Mean comparisons** | Glass's $\Delta$ | Yes (standardized mean diff) | Medium | **OPTIONAL_FUTURE** | Useful when control and intervention variances differ substantially, but requires explicit designation of control group. |
| **ANOVA** | Omega-squared ($\omega^2$) | Yes (variance explained) | High | **OPTIONAL_FUTURE** | Unbiased estimator of population variance explained; non-blocking optional future metric. |
| **ANOVA** | Partial omega-squared ($\omega_p^2$) | Yes (variance explained) | Medium | **OPTIONAL_FUTURE** | Less biased for factorial/repeated designs; algebraic complexity higher under unbalanced designs. |
| **Rank tests** | Rank-biserial variants | Yes (stochastic dominance) | High | **KEEP CURRENT** | Already present in runtime for Mann-Whitney and Wilcoxon with bootstrap uncertainty. |
| **Rank tests** | Epsilon-squared ($\epsilon^2$) | Yes (rank variance explained) | High | **KEEP CURRENT** | Already present in runtime for Kruskal-Wallis. |
| **Categorical** | Cramer's $V$ | Yes (nominal association) | High | **KEEP CURRENT** | Already present in runtime with bootstrap confidence interval. |
| **Categorical** | Odds Ratio ($\text{OR}$) | Yes (odds association) | High | **KEEP CURRENT** | Already present in Fisher's exact test and logistic regression. |
| **Categorical** | Risk Ratio (Relative Risk) | **NO (changes estimand)** | N/A | **DO_NOT_ADD** | Relative risk requires a prospective cohort/clinical trial design. Computing RR in case-control/cross-sectional data is scientifically invalid. |
| **Regression** | Standardized beta ($\beta^*$) | Yes (relative predictor strength) | High | **OPTIONAL_FUTURE** | Scale-independent standardized regression coefficient; rescales original parameters. Non-blocking optional future metric. |
| **Regression** | $R^2$ / Adjusted $R^2$ | Yes (model variance explained) | High | **KEEP CURRENT** | Already present in `model_fit`. |
| **Regression** | $R^2$ Confidence Interval | Yes (fit uncertainty) | Medium | **OPTIONAL_FUTURE** | Can be calculated via noncentral $F$ inversion; moderate implementation complexity. |
| **Reliability** | Cronbach $\alpha$ Bootstrap CI | Yes (internal consistency) | High | **KEEP CURRENT** | Already supported at runtime via uncertainty settings. |
| **Reliability** | Feldt Analytical $\alpha$ CI | Yes (internal consistency) | Medium | **OPTIONAL_FUTURE** | Analytical alternative to bootstrap based on compound symmetry assumption. |
| **Reliability** | Standardized $\alpha$ | **NO (changes weighting)** | Low | **DO_NOT_ADD** | Standardized alpha forces equal item variances, artificially masking item quality. Psychometrically discouraged. |
| **Agreement** | ICC Estimate & Exact CI | Yes (rater agreement) | High | **KEEP CURRENT** | All 6 Shrout & Fleiss / McGraw & Wong forms fully supported with exact $F$-inversion CIs. |

---

## 4. Audit of Deferred Numerical Validation Quantities (Level D)

The standalone validation framework (`validation/reference_validation_summary.json`) records 12 deferred quantities (24 field comparisons across 2 cases). These are analyzed as follows:

| Method ID | Deferred Field | Runtime Status | Classification |
| :--- | :--- | :--- | :--- |
| `one_sample_t` | `effect_size_ci` | Runtime quantity exists (bootstrap CI) | A (Runtime exists; only independent reference deferred) |
| `student_t` | `effect_size_ci` | Runtime quantity exists (bootstrap CI) | A (Runtime exists; only independent reference deferred) |
| `welch_t` | `effect_size_ci` | Runtime quantity exists (bootstrap CI) | A (Runtime exists; only independent reference deferred) |
| `paired_t` | `effect_size_ci` | Runtime quantity exists (bootstrap CI) | A (Runtime exists; only independent reference deferred) |
| `wilcoxon_signed_rank` | `effect_size_ci` | Runtime quantity exists (bootstrap CI) | A (Runtime exists; only independent reference deferred) |
| `spearman_correlation` | `confidence_interval` | Runtime quantity exists (bootstrap CI) | A (Runtime exists; only independent reference deferred) |
| `pearson_chi_square` | `effect_size_ci` | Runtime quantity exists (bootstrap CI) | A (Runtime exists; only independent reference deferred) |
| `fisher_exact` | `confidence_interval` | Runtime quantity exists (exact conditional CI) | A (Runtime exists; only independent reference deferred) |
| `friedman_test` | `kendall_w_bootstrap_ci`| Runtime quantity exists (bootstrap CI) | A (Runtime exists; only independent reference deferred) |
| `partial_pearson_correlation` | `partial_r_ci` | Runtime quantity exists (Fisher $z$ CI) | A (Runtime exists; only independent reference deferred) |
| `cronbach_alpha` | `cronbach_bootstrap_ci`| Runtime quantity exists (bootstrap CI) | A (Runtime exists; only independent reference deferred) |
| `intraclass_correlation` | `icc_exact_ci` | Runtime quantity exists (exact $F$-inversion) | A (Runtime exists; only independent reference deferred) |

**Key Finding**: All 12 deferred fields already exist and execute accurately in PyAutoStat runtime; their Level D status reflects the deliberate omission of stochastic bootstrap harnesses or complex exact inversions from the lightweight standalone NumPy reference script.

---

## 5. Prioritization Model for True Candidate Gaps

Every candidate gap is evaluated against 5 objective criteria:
1. **Scientific Value**: high / medium / low
2. **User Frequency**: high / medium / low
3. **Implementation Risk**: high / medium / low
4. **Estimand Risk**: same estimand / changes estimand
5. **Validation Feasibility**: straightforward / moderate / difficult

| Candidate Metric | Scientific Value | User Frequency | Implementation Risk | Estimand Risk | Validation Feasibility | Recommended Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hedges' $g$ (Unbiased SMD)** | High | High | Low | Same estimand | Straightforward | **OPTIONAL_FUTURE** |
| **Standardized Regression Betas** | High | High | Low | Same estimand | Straightforward | **OPTIONAL_FUTURE** |
| **Omega-Squared ($\omega^2$) for ANOVA** | Medium | Medium | Low | Same estimand | Straightforward | **OPTIONAL_FUTURE** |
| **Feldt Exact CI for Cronbach's Alpha** | Medium | Medium | Low | Same estimand | Straightforward | **OPTIONAL_FUTURE** |
| **Glass's Delta ($\Delta$)** | Medium | Low | Medium | Same estimand | Moderate | **OPTIONAL_FUTURE** |
| **Partial Omega-Squared ($\omega_p^2$)** | Medium | Low | Medium | Same estimand | Moderate | **OPTIONAL_FUTURE** |
| **$R^2$ Confidence Interval** | Low | Low | Medium | Same estimand | Moderate | **OPTIONAL_FUTURE** |
| **Risk Ratio (Relative Risk)** | N/A | High | High | **Changes estimand**| N/A | **DO_NOT_ADD** |
| **Standardized Cronbach's Alpha** | Low | Medium | Low | **Changes estimand**| N/A | **DO_NOT_ADD** |

### Synthesis and Scope Boundaries:
- **Optional future metrics are not release blockers.** PyAutoStat's 24 statistical methods are fully interpretable, reproducible, and complete for their stated estimands with existing point estimates, raw/standardized effect sizes, and uncertainty intervals.
- **True High-Value Gaps in Current Scope**: 0 (None). All 24 methods provide complete effect-size and/or uncertainty coverage for their supported estimands.
- **Optional Future Enhancements**: Hedges' $g$, standardized regression coefficients ($\beta^*$), omega-squared ($\omega^2$), partial omega-squared ($\omega_p^2$), and Feldt's alpha CI are classified as non-blocking candidate enhancements for future development.
- **Excluded**: Risk Ratio and Standardized Alpha remain permanently excluded as scientifically inappropriate for the supported study designs.
