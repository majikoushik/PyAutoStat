# PyAutoStat Independent Numerical Validation Foundation

*Authoritative technical documentation for the PyAutoStat independent numerical validation programme.*

---

## 1. Purpose

This document defines the methodology, mathematical basis, evidence classification, and results for the independent numerical validation of PyAutoStat's statistical calculation engine.

PyAutoStat is an explainable and reproducible research assistant for pandas DataFrames. To ensure that research conclusions and reported effect sizes are statistically defensible, PyAutoStat maintains an inspectable validation programme designed to answer, field by field:

1. **Does PyAutoStat return the expected numerical result?**
2. **What independent mathematical reference or external calculation supports that conclusion?**
3. **Which quantities are validated independently versus backend-conformance checked?**
4. **Which quantities remain explicitly unvalidated or deferred?**

---

## 2. Difference Between Unit Testing and Numerical Validation

Unit tests and numerical validation serve distinct, complementary functions:

| Dimension | Unit Testing (`tests/`) | Independent Numerical Validation (`validation/`) |
| :--- | :--- | :--- |
| **Primary Goal** | Software stability, branch coverage, regression prevention, exception handling. | Scientific correctness, mathematical provenance, and estimand fidelity. |
| **Truth Source** | Codebase expectations, past behavior, regression snapshots. | Independent mathematical derivations, textbook formulas, external benchmarks. |
| **Backend Relation** | Validates that calling PyAutoStat produces expected dictionary shapes and values. | Distinguishes whether the calculation is verified by an independent algorithm or merely wraps a shared backend. |
| **Failure Mode** | Test crashes, syntax errors, changed keys, type violations. | Numerical drift, bias, inverted contrast orientations, incorrect degrees of freedom, tolerance breaches. |
| **Execution Policy** | Automated on every pull request via pytest CI. | Standalone inspectable harness executed for formal validation releases. |

A standard unit test confirms that current code behaves as programmed. Independent numerical validation establishes the scientific validity and origin of the number.

---

## 3. Validation Evidence Levels A/B/C/D

To prevent misleading claims of "validation" when an analysis merely delegates to an existing library, PyAutoStat strictly enforces four evidence levels:

### Level A — Independent Reference
- Manual analytical formula implemented completely independently of PyAutoStat internal helpers.
- Published numerical benchmarks or textbook canonical examples.
- Direct implementation of sample sums, pooled variances, Welch-Satterthwaite formulas, rank transformations, covariance matrices, and Fisher transformations.

### Level B — External Backend Conformance
- Direct call to SciPy (`scipy.stats`) verifying that PyAutoStat correctly parameterizes, delegates to, and extracts quantities from backend libraries.
- Verifies wrapper conformance and absence of data mangling during delegation.
- *Critical Scientific Principle*: **Level B alone is NEVER labeled as "independently validated."**

### Level C — Internal Invariant / Reproducibility
- Algebraic relationships that must hold mathematically:
  - $n_1 + n_2 = \text{analyzed\_rows}$
  - $\text{analyzed\_rows} + \text{excluded\_rows} = \text{original\_rows}$
  - $2 \times \text{complete\_pairs} = \text{analyzed\_rows}$ (for paired designs)
  - Contrast orientation: sign of Cohen's $d$ matches sign of mean difference.
- Deterministic seed reproduction and complete-case sample accounting.

### Level D — Deferred
- Quantities whose independent reference calculation is not implemented in the current tranche.
- Examples: Percentile bootstrap confidence intervals, exact noncentral-t inversion intervals, and complex iterative post-hoc procedures.
- These quantities remain visibly labeled as `deferred` rather than omitted or falsely certified.

---

## 4. Reference-Source Hierarchy

When establishing expected quantities, PyAutoStat adheres to the following priority hierarchy:

1. **Closed-Form Analytical Formula (Level A)**: Independent mathematical evaluation from raw sample moments (e.g. sample mean, standard error, pooled standard deviation, Welch degrees of freedom, Pearson covariance).
2. **Deterministic Combinatorial / Rank Calculation (Level A)**: Exact rank summation with fractional average ranks for ties (e.g. Mann-Whitney $U_1$, Wilcoxon signed-rank sums $W^+$ and $W^-$).
3. **Established Published Benchmarks (Level A)**: Verified textbook datasets with exact precision specifications (e.g. Student 1908, Fisher 1925, Wilcoxon 1945).
4. **Direct Library Cross-Check (Level B)**: Direct execution of backend library functions (`scipy.stats.ttest_ind`, `scipy.stats.fisher_exact`, `scipy.stats.chi2_contingency`) to verify interface fidelity.

---

## 5. Numerical Tolerance Policy

Deterministic analytical calculations are compared using dual absolute and relative error criteria:

$$\text{Error} \le \max(\text{atol}, \text{rtol} \times |\text{expected}|)$$

Centralized tolerances used by the validation harness:
- **Primary Analytical Statistics** (means, differences, test statistics, standard errors, degrees of freedom, effect sizes):
  - $\text{atol} = 10^{-12}$
  - $\text{rtol} = 10^{-10}$
- **Probabilities & p-values** (numerical tail integration):
  - $\text{atol} = 10^{-8}$
  - $\text{rtol} = 10^{-6}$
- **Integer Counts & Invariants** (sample sizes, pair counts, table totals, degrees of freedom):
  - Exact match ($\text{atol} = 0$, $\text{rtol} = 0$)

Comparisons are strictly evaluated on raw floating-point numbers; string formatting or rounded representations are never used as validation criteria.

---

## 6. Orientation / Sign Policy

Contrast direction is a core scientific estimand. Agreement in absolute value is insufficient.

The validation harness explicitly verifies signed orientation across all contrast types:
- **Independent Two-Group Means** (`welch_t`, `student_t`): Group 1 minus Group 2 ($\bar{x}_1 - \bar{x}_2$). Verified for both positive ($\bar{x}_1 > \bar{x}_2$) and negative ($\bar{x}_1 < \bar{x}_2$) directions.
- **One-Sample Comparison** (`one_sample_t`): Sample mean minus reference value ($\bar{x} - \mu_0$). Verified for both positive and negative contrasts.
- **Paired Two-Condition Means** (`paired_t`): Condition 1 minus Condition 2 ($\bar{d} = \frac{1}{n}\sum (x_{1,i} - x_{2,i})$). Verified for both directions.
- **Mann-Whitney Rank-Biserial** (`mann_whitney_u`): $r_{rb} = \frac{2 U_1}{n_1 n_2} - 1$. Positive when Group 1 systematically exceeds Group 2.
- **Wilcoxon Matched-Pairs Rank-Biserial** (`wilcoxon_signed_rank`): $r_{rb} = \frac{W^+ - W^-}{W^+ + W^-}$. Positive when Condition 1 exceeds Condition 2.
- **Bivariate Correlation** (`pearson_correlation`, `spearman_correlation`): Signed association direction ($r > 0$ vs. $r < 0$).
- **Contingency Tables & Odds Ratios** (`fisher_exact`, `pearson_chi_square`): Table orientation preserves declared row and column categories; odds ratio computed as cross-product $\frac{a \cdot d}{b \cdot c}$.

---

## 7. Missing-Data / Sample-Accounting Policy

PyAutoStat enforces complete-case analysis for declared analytical variables and never imputes missing values. The validation harness evaluates sample accounting across every method family:

- **Original Row Count ($N_{\text{orig}}$)**: Verified to equal analyzed rows plus excluded rows ($N_{\text{orig}} = N_{\text{analyzed}} + N_{\text{excluded}}$).
- **Independent Groups**: Rows with missing outcome or missing group labels are excluded; remaining rows sum exactly to group sizes ($n_1 + n_2 = N_{\text{analyzed}}$).
- **Paired Observations**: Matched unit filtering requires valid observations in both conditions. Incomplete units are excluded; row count satisfies $2 \times N_{\text{pairs}} = N_{\text{analyzed}}$.
- **Bivariate Associations**: Incomplete pairs $(x_i, \text{NaN})$ or $(\text{NaN}, y_i)$ are excluded; effective pair count equals analyzed rows.
- **Contingency Tables**: Missing values in row or column factors are excluded; table sum equals total analyzed observations.

---

## 8. Validation-Case Design Principles

Reference cases in `validation/reference_cases.py` adhere to the following principles:

1. **Non-Degenerate Designs**: At least two distinct cases per method covering diverse sample sizes and parameter ranges.
2. **Contrast Diversity**: Every difference and correlation method includes both positive and negative association/difference cases.
3. **Missingness Invariants**: Every method family contains at least one case with missing values to stress complete-case accounting.
4. **Boundary & Special Conditions**:
   - Mann-Whitney includes a no-ties case and a tied-rank case.
   - Wilcoxon signed-rank includes a no-zero-difference case and a case with zero differences and ties under the `wilcox` exclusion policy.
   - Chi-square includes a 2x2 table and a 3x2 RxC table (both satisfying PyAutoStat's conservative $E_{ij} \ge 5$ requirement).
   - Fisher's exact test includes both $OR > 1.0$ and $OR < 1.0$ orientations.

---

## 9. Phase-2 Validated Method Matrix

The matrix below summarizes the validation coverage for the First Tranche of 10 foundational methods:

| Method ID | Test Statistic | p-value | df | Primary Estimate | Primary CI | Effect Size | Effect CI | Missingness | Orientation | Evidence Levels |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `one_sample_t` | Passed | Passed | Passed | Passed | Passed | Passed | *Deferred* | Passed | Passed | Level A / B / C (CI Level D) |
| `student_t` | Passed | Passed | Passed | Passed | Passed | Passed | *Deferred* | Passed | Passed | Level A / B / C (CI Level D) |
| `welch_t` | Passed | Passed | Passed | Passed | Passed | Passed | *Deferred* | Passed | Passed | Level A / B / C (CI Level D) |
| `paired_t` | Passed | Passed | Passed | Passed | Passed | Passed | *Deferred* | Passed | Passed | Level A / B / C (CI Level D) |
| `mann_whitney_u` | Passed | Passed | N/A | Passed | N/A | Passed | N/A | Passed | Passed | Level A / B / C |
| `wilcoxon_signed_rank` | Passed | Passed | N/A | Passed | N/A | Passed | *Deferred* | Passed | Passed | Level A / B / C (CI Level D) |
| `pearson_correlation` | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Level A / B / C |
| `spearman_correlation` | Passed | Passed | N/A | Passed | *Deferred* | Passed | *Deferred* | Passed | Passed | Level A / B / C (CI Level D) |
| `pearson_chi_square` | Passed | Passed | Passed | Passed | N/A | Passed | *Deferred* | Passed | Passed | Level A / B / C (CI Level D) |
| `fisher_exact` | Passed | Passed | N/A | Passed | *Deferred* | Passed | *Deferred* | Passed | Passed | Level A / B / C (CI Level D) |

---

## 10. Fields Validated per Method

### 1. `one_sample_t`
- **Level A**: Sample mean ($\bar{x}$), mean difference ($\bar{x} - \mu_0$), standard error, Student's $t$ statistic, degrees of freedom ($n - 1$), analytical confidence interval, Cohen's $d$.
- **Level B**: Two-sided p-value via `scipy.stats.ttest_1samp`.
- **Level C**: Analyzed rows ($n$), excluded rows, sign orientation.
- **Level D**: Cohen's $d$ exact noncentral-t confidence interval.

### 2. `student_t`
- **Level A**: Sample mean difference ($\bar{x}_1 - \bar{x}_2$), pooled variance, pooled standard error, Student's pooled $t$ statistic, pooled df ($n_1 + n_2 - 2$), analytical Student-t interval, Cohen's $d$.
- **Level B**: Two-sided p-value via `scipy.stats.ttest_ind(equal_var=True)`.
- **Level C**: Analyzed rows, group sample sizes, excluded rows, contrast direction.
- **Level D**: Cohen's $d$ bootstrap confidence interval.

### 3. `welch_t`
- **Level A**: Sample mean difference ($\bar{x}_1 - \bar{x}_2$), Welch standard error, Welch $t$ statistic, Welch-Satterthwaite degrees of freedom, analytical Welch interval, Cohen's $d$.
- **Level B**: Two-sided p-value via `scipy.stats.ttest_ind(equal_var=False)`.
- **Level C**: Analyzed rows, group sample sizes, excluded rows, contrast direction.
- **Level D**: Cohen's $d$ bootstrap confidence interval.

### 4. `paired_t`
- **Level A**: Sample mean paired difference ($\bar{d}$), standard deviation of paired differences ($s_d$), standard error, paired $t$ statistic, df ($ complete\_pairs - 1$), analytical paired $t$ interval, Cohen's $d_z$.
- **Level B**: Two-sided p-value via `scipy.stats.ttest_rel`.
- **Level C**: Complete pairs count, analyzed rows ($2 \times \text{complete\_pairs}$), excluded rows, incomplete units, condition orientation.
- **Level D**: Cohen's $d_z$ noncentral-t inversion confidence interval.

### 5. `mann_whitney_u`
- **Level A**: Independent first-group rank sum ($R_1$), Mann-Whitney $U_1$ statistic, rank-biserial correlation ($r_{rb}$).
- **Level B**: Two-sided asymptotic p-value via `scipy.stats.mannwhitneyu(alternative="two-sided")`.
- **Level C**: Analyzed sample sizes, excluded rows, rank-biserial sign direction.

### 6. `wilcoxon_signed_rank`
- **Level A**: Positive signed-rank sum ($W^+$), negative signed-rank sum ($W^-$), matched-pairs rank-biserial correlation ($r_{rb}$).
- **Level B**: Test statistic and two-sided p-value via `scipy.stats.wilcoxon(zero_method="wilcox")`.
- **Level C**: Complete pairs, zero-difference count under `wilcox` policy, analyzed rows, excluded rows, sign direction.
- **Level D**: Matched-pairs rank-biserial bootstrap confidence interval.

### 7. `pearson_correlation`
- **Level A**: Sample covariance, sample standard deviations, Pearson $r$ coefficient, $t$ transformation statistic, Fisher $z$ transform, Fisher $z$ analytical confidence interval.
- **Level B**: Two-sided p-value via `scipy.stats.pearsonr`.
- **Level C**: Effective pair count, excluded pair count, correlation sign.

### 8. `spearman_correlation`
- **Level A**: Fractional average ranking on both variables, Pearson correlation on ranks (Spearman $\rho$).
- **Level B**: Two-sided p-value via `scipy.stats.spearmanr`.
- **Level C**: Complete pair count, excluded pair count, monotonic rank direction.
- **Level D**: Spearman paired-observation bootstrap confidence interval.

### 9. `pearson_chi_square`
- **Level A**: Expected counts matrix ($E_{ij} = R_i C_j / N$), uncorrected Pearson chi-square statistic ($\sum (O-E)^2/E$), degrees of freedom ($(r-1)(c-1)$), Cramer's $V$.
- **Level B**: Chi-square p-value via `scipy.stats.chi2_contingency(correction=False)`.
- **Level C**: Total contingency count $N$, excluded row count, table dimensions.
- **Level D**: Cramer's $V$ bootstrap confidence interval.

### 10. `fisher_exact`
- **Level A**: Sample cross-product odds ratio ($\frac{a \cdot d}{b \cdot c}$).
- **Level B**: Two-sided hypergeometric p-value via `scipy.stats.fisher_exact`.
- **Level C**: 2x2 contingency count $N$, excluded row count, table orientation.
- **Level D**: Odds ratio confidence interval.

---

## 11. Known Limitations / Deferred Fields

The following quantities are intentionally **deferred (Level D)** in Phase 2:
1. **Resampling / Bootstrap Intervals**: Confidence intervals for Cohen's $d$, Spearman $\rho$, Cramer's $V$, and rank-biserial correlations rely on pseudo-random bootstrap resampling. An independent deterministic bootstrap harness with reproducible random generation is scheduled for subsequent phases.
2. **Noncentral-t Exact Inversion Intervals**: Analytical confidence intervals for one-sample Cohen's $d$ and paired Cohen's $d_z$ rely on iterative root-finding over the noncentral Student-t cumulative distribution function. An independent root-finding reference solver is deferred.
3. **Woolf / Exact Conditional Odds Ratio Intervals**: Confidence intervals for sample odds ratios in Fisher's exact test are deferred.

No deferred quantity is certified as independently validated in Phase 2.

---

## 12. Environment Metadata

The Phase 2 foundation was verified under the following local environment:

- **Repository**: `majikoushik/PyAutoStat`
- **Package Version**: `0.5.0`
- **Verified Commit**: `817bb9c5454c266462a457690f3673099546b49f` (Phase 1 head)
- **Python Version**: `3.12.7` (64-bit AMD64)
- **NumPy Version**: `2.2.6`
- **pandas Version**: `3.0.6`
- **SciPy Version**: `1.13.0`
- **statsmodels Version**: `0.15.0`
- **External Packages**: `rpy2` (Not installed), `pingouin` (Not installed)
- **Verification Date**: `2026-10-07`

---

## 13. How to Reproduce the Validation Run

To execute the complete reference validation suite:

```bash
# Execute validation across all 10 first-tranche methods
python validation/run_reference_validation.py

# Execute in strict mode (fails with non-zero exit if any discrepancy occurs)
python validation/run_reference_validation.py --strict

# Filter to an individual method
python validation/run_reference_validation.py --method welch_t
python validation/run_reference_validation.py --method paired_t

# Export full machine-readable JSON report
python validation/run_reference_validation.py --json validation_report.json

# Regenerate machine-readable case manifest
python validation/run_reference_validation.py --generate-manifest
```

---

## 14. Interpretation of "Validated"

> **IMPORTANT SCIENTIFIC DISCLAIMER**:
> The designation **"Validated"** applies strictly and solely to the exact fields, cases, datasets, tolerances, and reference sources documented in this framework.
> It does **not** prove mathematical or computational correctness for all possible datasets, arbitrary distributions, degenerate matrices, or extreme scales beyond the tested parameters.
> PyAutoStat makes no claims of formal mathematical verification across arbitrary infinite input spaces.

---

## 15. Future Expansion Plan (Phase 3 Methods)

Phase 3 of the Post-Release Roadmap will expand the independent reference validation framework to the remaining 14 registered method IDs:

1. **Multi-Group & Post-Hoc**:
   - `welch_anova` + Games-Howell pairwise
   - `one_way_anova` + Tukey-Kramer HSD
   - `kruskal_wallis` + Dunn-Holm pairwise
2. **Repeated Measures & Factorial**:
   - `repeated_measures_anova` + Mauchly sphericity + Greenhouse-Geisser correction
   - `friedman_test` + Kendall's $W$ + pairwise Wilcoxon-Holm
   - `two_way_anova` (Type II and Type III sums of squares)
3. **Regression & Association**:
   - `linear_regression` (OLS + HC3 robust covariance + diagnostics)
   - `logistic_regression` (binary Logit + odds ratios + McFadden pseudo-$R^2$)
   - `kendall_tau_b`
   - `point_biserial_correlation`
   - `partial_pearson_correlation`
4. **Categorical & Reliability**:
   - `mcnemar` test of paired nominal proportions
   - `cronbach_alpha` internal consistency
   - `intraclass_correlation` (all 6 Shrout & Fleiss / McGraw & Wong configurations)

Phase 3 will maintain the standing constraint against adding tests to pytest until reference cases are mature, inspectable, and formally approved.
