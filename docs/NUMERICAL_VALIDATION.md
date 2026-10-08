# PyAutoStat Independent Numerical Validation Framework

*Authoritative technical documentation for the PyAutoStat independent numerical validation programme.*

---

## 1. Purpose

This document defines the methodology, mathematical basis, evidence classification, and verified results for the independent numerical validation of PyAutoStat's statistical calculation engine.

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
| **Backend Relation** | Validates that calling PyAutoStat produces expected dictionary shapes and values. | Distinguishes whether the calculation is verified by an independent algorithm or wraps a backend. |
| **Failure Mode** | Test crashes, syntax errors, changed keys, type violations. | Numerical drift, bias, inverted contrast orientations, incorrect degrees of freedom, tolerance breaches. |
| **Execution Policy** | Automated on every pull request via pytest CI. | Standalone inspectable harness executed for formal validation releases. |

A standard unit test confirms that current code behaves as programmed. Independent numerical validation provides numerical evidence for specified methods, fields, cases, and tolerances, establishing mathematical provenance and implementation fidelity.

Numerical validation does **not** establish:
- study-design validity;
- measurement validity;
- causal validity;
- sampling representativeness;
- universal correctness for all datasets.

---

## 3. Validation Evidence Levels A/B/C/D

To prevent misleading claims of "validation" when an analysis merely delegates to an existing library, PyAutoStat strictly enforces four evidence levels:

### Level A — Independent Reference
- Manual analytical formula implemented completely independently of PyAutoStat internal helpers.
- Published numerical benchmarks or textbook canonical examples.
- Direct implementation of sample moments, pooled variances, Welch-Satterthwaite formulas, rank transformations, covariance matrices, Fisher-z transforms, OLS matrix solves, HC3 sandwich covariance, Newton-Raphson/IRLS logistic solvers, combinatorial McNemar calculations, and Shrout & Fleiss / McGraw & Wong ICC formulations.
- *Shared Primitives Disclosure*: When a Level A implementation utilizes a standard distribution tail integral (e.g. `scipy.stats.f.sf`, `scipy.stats.t.sf`, `scipy.stats.chi2.sf`, `scipy.stats.norm.sf`, `scipy.stats.studentized_range`), the shared numerical primitive is explicitly recorded.

### Level B — External Backend Conformance
- Direct call to established numerical libraries (`scipy.stats`, `statsmodels`) matching PyAutoStat's underlying execution engine.
- Confirms that PyAutoStat correctly parameterizes, delegates to, and extracts quantities from backend libraries.
- *Critical Scientific Principle*: **Level B alone is NEVER labeled as "independently validated."**

### Level C — Internal Invariant / Reproducibility
- Algebraic relationships that must hold mathematically:
  - $n_1 + n_2 = \text{analyzed\_rows}$
  - $\text{analyzed\_rows} + \text{excluded\_rows} = \text{original\_rows}$
  - $2 \times \text{complete\_pairs} = \text{analyzed\_rows}$ (for paired designs)
  - Contrast orientation: sign of Cohen's $d$ matches sign of mean difference.
  - Decomposition identities: $\text{SS}_{\text{total}} = \text{SS}_{\text{between}} + \text{SS}_{\text{within}}$ (One-Way ANOVA); $\text{SS}_{\text{total}} = \text{SS}_{\text{cond}} + \text{SS}_{\text{subj}} + \text{SS}_{\text{error}}$ (RM-ANOVA).
- Complete-case sample accounting and row filtering.

### Level D — Deferred
- Quantities whose independent reference calculation is not implemented in the current validation framework.
- Examples: Percentile bootstrap confidence intervals, exact noncentral-t inversion intervals, and complex iterative post-hoc procedures.
- These quantities remain visibly labeled as `deferred` rather than omitted or falsely certified.

---

## 4. Reference-Source Hierarchy

When establishing expected quantities, PyAutoStat adheres to the following priority hierarchy:

1. **Closed-Form Analytical Formula (Level A)**: Independent mathematical evaluation from raw sample moments (e.g. sample mean, standard error, pooled standard deviation, Welch degrees of freedom, Pearson covariance, ICC variance components).
2. **Deterministic Combinatorial / Rank Calculation (Level A)**: Exact rank summation with fractional average ranks for ties (e.g. Mann-Whitney $U_1$, Wilcoxon signed-rank sums $W^+$ and $W^-$, Kruskal-Wallis $H$, Friedman $Q$, exact binomial probability sums for McNemar).
3. **Established Published Benchmarks (Level A)**: Verified textbook datasets with exact precision specifications (e.g. Student 1908, Fisher 1925, Wilcoxon 1945, Shrout & Fleiss 1979).
4. **Direct Library Cross-Check (Level B)**: Direct execution of backend library functions (`scipy.stats.ttest_ind`, `scipy.stats.fisher_exact`, `scipy.stats.chi2_contingency`, `scipy.stats.f_oneway`, `statsmodels.api.OLS`) to verify interface fidelity.

---

## 5. Numerical Tolerance Policy

Deterministic analytical calculations are compared using dual absolute and relative error criteria defined in `validation/tolerances.py`:

$$\text{Error} \le \max(\text{atol}, \text{rtol} \times |\text{expected}|)$$

Centralized tolerances:
- **Primary Analytical Statistics** (means, differences, test statistics, standard errors, degrees of freedom, effect sizes):
  - $\text{atol} = 10^{-12}$
  - $\text{rtol} = 10^{-10}$
- **Probabilities & p-values** (numerical tail integration):
  - $\text{atol} = 10^{-8}$
  - $\text{rtol} = 10^{-6}$
- **Numerical Solvers** (logistic Newton-Raphson MLE coefficients and probabilities):
  - $\text{atol} = 10^{-6}$
  - $\text{rtol} = 10^{-5}$
- **Integer Counts & Invariants** (sample sizes, pair counts, table totals, degrees of freedom):
  - Exact match ($\text{atol} = 0$, $\text{rtol} = 0$)

Comparisons are evaluated on raw floating-point numbers; string formatting or rounded representations are never used as validation criteria.

---

## 6. Orientation and Sign Policy

Contrast direction is a core scientific estimand. Agreement in absolute value is insufficient.

The validation harness explicitly verifies signed orientation across all contrast types:
- **Independent Two-Group Means** (`welch_t`, `student_t`): Group 1 minus Group 2 ($\bar{x}_1 - \bar{x}_2$). Verified for both positive and negative directions.
- **One-Sample Comparison** (`one_sample_t`): Sample mean minus reference value ($\bar{x} - \mu_0$). Verified for both directions.
- **Paired Two-Condition Means** (`paired_t`): Condition 1 minus Condition 2 ($\bar{d} = \frac{1}{n}\sum (x_{1,i} - x_{2,i})$). Verified for both directions.
- **Mann-Whitney Rank-Biserial** (`mann_whitney_u`): $r_{rb} = \frac{2 U_1}{n_1 n_2} - 1$. Positive when Group 1 systematically exceeds Group 2.
- **Wilcoxon Matched-Pairs Rank-Biserial** (`wilcoxon_signed_rank`): $r_{rb} = \frac{W^+ - W^-}{W^+ + W^-}$. Positive when Condition 1 exceeds Condition 2.
- **Bivariate Association** (`pearson_correlation`, `spearman_correlation`, `kendall_tau_b`, `point_biserial_correlation`): Signed association direction ($r > 0$ vs. $r < 0$).
- **Multigroup Pairwise Comparisons** (`welch_anova` Games-Howell, `one_way_anova` Tukey-Kramer): First group minus second group ($\bar{x}_i - \bar{x}_j$).
- **Contingency Tables & Odds Ratios** (`fisher_exact`, `pearson_chi_square`, `mcnemar`): Category order preserved; odds ratio computed as cross-product $\frac{a \cdot d}{b \cdot c}$.

---

## 7. Missing-Data and Sample-Accounting Policy

PyAutoStat enforces complete-case analysis for declared analytical variables and never imputes missing values. The validation harness evaluates sample accounting across every method family:

- **Row Invariant**: $N_{\text{orig}} = N_{\text{analyzed}} + N_{\text{excluded}}$ across all 24 methods.
- **Independent Groups**: Incomplete rows excluded; remaining rows satisfy $\sum n_k = N_{\text{analyzed}}$.
- **Paired Observations**: Matched unit filtering requires valid observations in both conditions; incomplete units excluded.
- **Panel Rectangularity**: Repeated-measures designs require complete panels across all conditions for each participant. Incomplete participant profiles are excluded.
- **Multivariate Regression**: Complete cases across outcome and all declared predictor variables.

---

## 8. Complete Method Coverage Matrix (All 24 Registered Method IDs)

| # | Method ID | Primary Statistic | p-value | df | Primary Estimate | Primary CI | Effect Size | Missingness | Orientation | Evidence Levels |
| -: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `one_sample_t` | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Level A / B / C (CI Level D) |
| 2 | `student_t` | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Level A / B / C (CI Level D) |
| 3 | `welch_t` | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Level A / B / C (CI Level D) |
| 4 | `paired_t` | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Level A / B / C (CI Level D) |
| 5 | `mann_whitney_u` | Passed | Passed | N/A | Passed | N/A | Passed | Passed | Passed | Level A / B / C |
| 6 | `wilcoxon_signed_rank` | Passed | Passed | N/A | Passed | N/A | Passed | Passed | Passed | Level A / B / C (CI Level D) |
| 7 | `pearson_correlation` | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Level A / B / C |
| 8 | `spearman_correlation` | Passed | Passed | N/A | Passed | *Deferred* | Passed | Passed | Passed | Level A / B / C (CI Level D) |
| 9 | `pearson_chi_square` | Passed | Passed | Passed | Passed | N/A | Passed | Passed | Passed | Level A / B / C (CI Level D) |
| 10 | `fisher_exact` | Passed | Passed | N/A | Passed | *Deferred* | Passed | Passed | Passed | Level A / B / C (CI Level D) |
| 11 | `welch_anova` | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Level A / B / C |
| 12 | `one_way_anova` | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Level A / B / C |
| 13 | `kruskal_wallis` | Passed | Passed | Passed | Passed | N/A | Passed | Passed | Passed | Level A / B / C |
| 14 | `repeated_measures_anova` | Passed | Passed | Passed | Passed | N/A | Passed | Passed | Passed | Level A / B / C |
| 15 | `friedman_test` | Passed | Passed | Passed | Passed | N/A | Passed | Passed | Passed | Level A / B / C (CI Level D) |
| 16 | `two_way_anova` | Passed | Passed | Passed | Passed | N/A | Passed | Passed | Passed | Level A / B / C |
| 17 | `linear_regression` | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Level A / B / C |
| 18 | `logistic_regression` | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Level A / C |
| 19 | `kendall_tau_b` | Passed | Passed | N/A | Passed | N/A | Passed | Passed | Passed | Level A / B / C |
| 20 | `point_biserial_correlation` | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Passed | Level A / B / C |
| 21 | `partial_pearson_correlation` | Passed | Passed | Passed | Passed | *Deferred* | Passed | Passed | Passed | Level A / B / C (CI Level D) |
| 22 | `mcnemar` | Passed | Passed | N/A | Passed | N/A | Passed | Passed | Passed | Level A / B / C |
| 23 | `cronbach_alpha` | Passed | N/A | N/A | Passed | *Deferred* | Passed | Passed | Passed | Level A / C (CI Level D) |
| 24 | `intraclass_correlation` | Passed | Passed | Passed | Passed | *Deferred* | Passed | Passed | Passed | Level A / B / C (CI Level D) |

---

## 9. Fields Validated per Method Family

### Foundational Tranche (Methods 1–10)
- **`one_sample_t`**: Mean, mean difference, standard error, $t$ statistic, df, analytical Student-t CI, Cohen's $d$; two-sided SciPy p-value.
- **`student_t`**: Mean difference, pooled variance, pooled standard error, $t$ statistic, pooled df, analytical Student-t CI, Cohen's $d$; two-sided SciPy p-value.
- **`welch_t`**: Mean difference, Welch standard error, $t$ statistic, Welch-Satterthwaite df, analytical Welch CI, Cohen's $d$; two-sided SciPy p-value.
- **`paired_t`**: Mean paired difference, SD of differences, standard error, paired $t$ statistic, df, analytical paired $t$ CI, Cohen's $d_z$; two-sided SciPy p-value.
- **`mann_whitney_u`**: Rank sum $R_1$, Mann-Whitney $U_1$, rank-biserial correlation $r_{rb}$; two-sided SciPy p-value.
- **`wilcoxon_signed_rank`**: Positive rank sum $W^+$, negative rank sum $W^-$, rank-biserial $r_{rb}$; test statistic and two-sided SciPy p-value under `wilcox` zero policy.
- **`pearson_correlation`**: Sample covariance, SDs, Pearson $r$, $t$ transformation statistic, Fisher $z$ transform, analytical Fisher $z$ CI; two-sided SciPy p-value.
- **`spearman_correlation`**: Fractional average ranking, Pearson correlation on ranks (Spearman $\rho$); two-sided SciPy p-value.
- **`pearson_chi_square`**: Expected counts matrix, uncorrected Pearson $\chi^2$, df, Cramer's $V$; SciPy p-value.
- **`fisher_exact`**: Sample cross-product odds ratio; two-sided hypergeometric SciPy p-value.

### Multi-Group Independent Tranche (Methods 11–13)
- **`welch_anova`**: Welch weights, weighted grand mean, Welch $F$ statistic, numerator df ($k-1$), Welch-Satterthwaite denominator df, tail probability integral (`scipy.stats.f.sf`); complete Games-Howell pairwise family: mean differences, pairwise SEs, Welch dfs, studentized-range $q$, familywise adjusted p-values (`scipy.stats.studentized_range`).
- **`one_way_anova`**: Grand mean, $\text{SS}_{\text{between}}$, $\text{SS}_{\text{within}}$, $\text{SS}_{\text{total}}$, MS components, Fisher $F$ statistic, $\eta^2$; SciPy `f_oneway` p-value; complete Tukey-Kramer pairwise family: mean differences, standard errors, studentized-range $q$, familywise adjusted p-values (`scipy.stats.studentized_range`).
- **`kruskal_wallis`**: Pooled fractional average ranks, group rank sums, tie correction factor, Kruskal-Wallis $H$ statistic, df, rank $\epsilon^2$; SciPy `kruskal` p-value; complete Dunn pairwise comparisons with $z$ statistics, two-sided normal tail probabilities (`scipy.stats.norm.sf`), and Holm step-down adjusted p-values.

### Within-Subject & Factorial Tranche (Methods 14–16)
- **`repeated_measures_anova`**: Complete panel construction, condition means, subject means, grand mean, $\text{SS}_{\text{condition}}$, $\text{SS}_{\text{subject}}$, $\text{SS}_{\text{error}}$, $\text{SS}_{\text{total}}$, MS components, uncorrected $F$, uncorrected p-value (`scipy.stats.f.sf`), partial $\eta^2$; Greenhouse-Geisser $\epsilon$, corrected numerator and denominator dfs; Mauchly sphericity test $W$ from orthonormal Helmert contrast covariance determinant, Box-Anderson $\chi^2$ approximation, df, and tail probability (`scipy.stats.chi2.sf`).
- **`friedman_test`**: Within-subject fractional ranks, condition rank sums, Friedman $Q$ statistic, df, Kendall's $W$; SciPy `friedmanchisquare` p-value; pairwise Wilcoxon signed-rank comparisons with Holm adjustment.
- **`two_way_anova`**: Cell means, marginal means, Type II and Type III sums of squares, residual SS, dfs, $F$ statistics, p-values (`scipy.stats.f.sf`), partial $\eta^2$ for main factors and interaction term.

### Regression & Association Tranche (Methods 17–22)
- **`linear_regression`**: Design matrix $X$, coefficient solve $\hat{\beta} = (X'X)^{-1}X'y$, fitted values, residuals, SSE, SST, $R^2$, adjusted $R^2$, classical covariance matrix, classical SEs, classical $t$ statistics; HC3 sandwich covariance $\hat{\Omega} = (X'X)^{-1} X' \text{diag}(e_i^2 / (1-h_{ii})^2) X (X'X)^{-1}$, HC3 robust SEs, HC3 $t$ statistics, HC3 Wald robust $F$ statistic; Breusch-Pagan auxiliary LM statistic, condition number, statsmodels $R^2$ cross-check.
- **`logistic_regression`**: Event encoding, design matrix, independent Newton-Raphson / IRLS binary Logit MLE convergence, log-likelihood, null log-likelihood, likelihood-ratio $\chi^2$ test, McFadden pseudo-$R^2$, odds ratios $\exp(\hat{\beta})$, AIC, BIC, Hessian-based asymptotic Wald standard errors, Wald $z$ statistics, coefficient CIs, and odds-ratio CIs.
- **`kendall_tau_b`**: Concordant pairs count ($P$), discordant pairs count ($Q$), ties in $x$ only ($T_x$), ties in $y$ only ($T_y$), Kendall $\tau_b$ denominator and coefficient; SciPy `kendalltau` p-value.
- **`point_biserial_correlation`**: Closed-form formula from group means, pooled SD, and proportions; exact equivalence with Pearson $r$ on binary 0/1 indicators; $t$ statistic, df, analytical CI; SciPy `pointbiserialr` p-value.
- **`partial_pearson_correlation`**: Complete-case matrix, covariate residualization via independent OLS, Pearson correlation between residuals, residual df ($n - 2 - k$), $t$ statistic; statsmodels residualization cross-check.
- **`mcnemar`**: Paired 2x2 transition table, discordant cells $b$ and $c$, discordant total $n_{\text{disc}} = b + c$, exact two-sided binomial p-value using independent combinatorial binomial summation; SciPy `binomtest` p-value.

### Reliability & Agreement Tranche (Methods 23–24)
- **`cronbach_alpha`**: Item sample variances, total scale variance, Cronbach's $\alpha = \frac{k}{k-1}(1 - \sum s_i^2 / s_{\text{total}}^2)$, corrected item-total correlations, $\alpha$-if-item-deleted for all items, inter-item correlation matrix.
- **`intraclass_correlation`**: Grand mean, target/subject means, rater means, BMS, JMS, EMS, WMS mean squares; all six standard configurations:
  - $\text{ICC}(1,1)$ & $\text{ICC}(1,k)$: One-way random effects model
  - $\text{ICC}(2,1)$ & $\text{ICC}(2,k)$: Two-way random effects model (absolute agreement)
  - $\text{ICC}(3,1)$ & $\text{ICC}(3,k)$: Two-way mixed effects model (consistency)
  - Omnibus $F$ ratios and degrees of freedom.

---

## 10. Known Limitations and Deferred Fields

The following quantities are intentionally **deferred (Level D)** in the current validation framework:
1. **Resampling / Bootstrap Confidence Intervals**: Intervals for Cohen's $d$, Spearman $\rho$, Cramer's $V$, Kendall's $W$, partial $r$, and Cronbach's $\alpha$ rely on pseudo-random case resampling. An independent deterministic bootstrap harness with reproducible random generation is scheduled for subsequent expansion.
2. **Noncentral-t Exact Inversion Intervals**: Analytical confidence intervals for one-sample Cohen's $d$ and paired Cohen's $d_z$ rely on iterative root-finding over the noncentral Student-t cumulative distribution function.
3. **Exact ICC Inversion Confidence Intervals**: Noncentral-F confidence limits for two-way random absolute agreement ICC variants.
4. **Fisher Sample Odds Ratio Intervals**: Asymptotic log-Wald confidence intervals for the
   unconditional sample odds-ratio estimator in Fisher's exact test; intervals are unavailable
   when any observed cell is zero, without continuity correction.

No deferred quantity is certified as independently validated.

---

## 11. Environment and Provenance Metadata

The validation foundation was verified under the following authoritative environment:

- **Repository**: `majikoushik/PyAutoStat`
- **Package Version**: `0.5.0`
- **Numerical Implementation Baseline Commit**: `817bb9c5454c266462a457690f3673099546b49f` (head commit establishing authoritative statistical calculation source)
- **Validation Framework Baseline Commit**: `8fce4e9087236446eb4238387ffc04a9e33930c4` (initial validation foundation head)
- **Validation Framework Current Revision**: `2113a7e6a0bc589ce2b3c51d922238bc354eb702` (head commit of expanded validation framework)
- **Source Code Invariance Status**: Evaluated dynamically via Git worktree inspection against `numerical_source_baseline_sha`; confirmed `unchanged` across all 13 statistical runtime source files.
- **Python Version**: `3.12.7` (64-bit AMD64)
- **NumPy Version**: `2.2.6`
- **pandas Version**: `3.0.6`
- **SciPy Version**: `1.13.0`
- **statsmodels Version**: `0.15.0`
- **External Packages**: `rpy2` (Not installed), `pingouin` (Not installed)
- **Verification Date**: `2026-10-08`

---

## 12. Reproduction and CLI Instructions

To execute the complete reference validation suite:

```bash
# Execute validation across all 24 methods (default mode: reports discrepancies, exits 0)
python validation/run_reference_validation.py

# Execute in strict mode (fails with non-zero exit if any discrepancy occurs)
python validation/run_reference_validation.py --strict

# Filter to an individual method
python validation/run_reference_validation.py --method welch_anova
python validation/run_reference_validation.py --method intraclass_correlation

# Export full machine-readable JSON report
python validation/run_reference_validation.py --json validation_report.json

# Regenerate machine-readable case manifest (Schema v2) and summary artifact under strict validation
python validation/run_reference_validation.py --generate-manifest --strict

# Run harness self-checks (verifies registry integrity, unique IDs, tolerances, shared primitives)
python validation/run_reference_validation.py --self-check
```

### CLI Strict-Mode Policy
- **Default Mode (`python validation/run_reference_validation.py`)**: Executes all reference cases, displays detailed comparison diagnostics, and exits with code 0 even if discrepancies exist. This enables exploratory use and interactive inspection.
- **Strict Mode (`python validation/run_reference_validation.py --strict` or `--generate-manifest --strict`)**: Returns non-zero exit code if one or more numerical discrepancies are encountered. Manifest generation writes artifacts and JSON output before exit so discrepancy evidence is preserved.

---

## 13. Latest Validation Results Summary

As generated by `python validation/run_reference_validation.py --generate-manifest` and recorded in `validation/reference_validation_summary.json`:

- **Methods Evaluated**: 24 / 24 registered method IDs
- **Cases Evaluated**: 49
- **Fields Compared**: 592
- **Level A Passes (Independent Reference)**: 332
- **Level B Passes (External Backend Conformance)**: 69
- **Level C Passes (Internal Invariants)**: 167
- **Level D Fields (Explicitly Deferred)**: 24
- **Discrepancies Encountered**: 0
- **Overall Status**: `PASS` across all 24 methods and all 49 reference cases.

---

## 14. Interpretation of "Validated"

> **IMPORTANT SCIENTIFIC DISCLAIMER**:
> The designation **"Validated"** applies strictly and solely to the exact fields, cases, datasets, tolerances, and reference sources documented in this framework.
> It does **not** prove mathematical or computational correctness for all possible datasets, arbitrary distributions, degenerate matrices, or extreme scales beyond the tested parameters.
> PyAutoStat makes no claims of formal mathematical verification across arbitrary infinite input spaces.
