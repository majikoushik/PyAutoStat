# PyAutoStat Independent Numerical Validation Framework

This directory contains the standalone, inspectable numerical validation framework for PyAutoStat.

---

## 1. Purpose and Philosophy

A standard unit test confirms that the codebase runs without crashing and that refactoring has not unintentionally modified existing outputs. However, a unit test alone cannot establish the scientific provenance of a number.

**Independent numerical validation** answers:
1. *Does PyAutoStat calculate the expected scientific quantity?*
2. *What independent mathematical or external reference establishes that expectation?*
3. *Which fields are validated independently versus backend-conformance checked?*
4. *Which quantities remain explicitly unvalidated or deferred?*

---

## 2. Evidence Classification Hierarchy

In accordance with PyAutoStat's scientific safeguards, comparisons with external libraries must never be conflated with independent formula validation. Evidence is categorized into four explicit levels:

### LEVEL A — Independent Reference
- Manual mathematical formulas implemented independently of PyAutoStat internal helpers.
- Published numerical benchmarks or textbook reference values.
- Direct implementation of sample moments, pooled variances, Welch-Satterthwaite formulas, rank transformations, covariance matrices, Fisher-z transforms, OLS matrix solves, HC3 sandwich covariance, Newton-Raphson/IRLS logistic solvers, combinatorial McNemar calculations, and Shrout & Fleiss / McGraw & Wong ICC formulations.
- *Shared Primitives Disclosure*: When a Level A implementation utilizes a standard distribution tail integral (e.g. `scipy.stats.f.sf`, `scipy.stats.t.sf`, `scipy.stats.chi2.sf`, `scipy.stats.norm.sf`, `scipy.stats.studentized_range`), the shared numerical primitive is explicitly recorded.

### LEVEL B — External Backend Conformance
- Direct calls to established numerical libraries (`scipy.stats`, `statsmodels`) matching PyAutoStat's underlying execution engine.
- Confirms that PyAutoStat correctly parameterizes, delegates to, and extracts quantities from backend libraries.
- *Critical Scientific Principle*: **Level B alone is NEVER labeled as "independently validated."**

### LEVEL C — Internal Invariant / Reproducibility
- Algebraic invariants (e.g. $n_1 + n_2 = \text{analyzed\_rows}$, $\text{analyzed} + \text{excluded} = \text{original}$).
- Sample-accounting invariants (complete pair filtering, panel completeness, zero-difference tracking).
- Sign and orientation consistency (contrast definitions: first condition minus second condition).

### LEVEL D — Deferred
- Quantities whose independent calculation is not implemented in the current validation framework.
- Examples: Bootstrap confidence intervals, noncentral-t exact inversion intervals.
- These fields are visibly recorded as `deferred` rather than omitted or falsely certified.

---

## 3. Standing Architectural Constraints

To preserve repository safety and honor standing scientific constraints:
- **No Test Frameworks**: This harness does NOT use `pytest`, `unittest.TestCase`, or files named `test_*.py`.
- **No Package Pollution**: This directory is NOT imported by `src/pyautostat` and is never bundled into runtime distributions.
- **No Dependency Additions**: The validation harness relies only on packages already declared in PyAutoStat's standard dependencies (`numpy`, `scipy`, `pandas`, `statsmodels`).
- **Standalone Execution**: The runner is executed manually from the repository root.

---

## 4. Directory Structure

```
validation/
├── README.md                          # Framework overview and execution instructions
├── tolerances.py                      # Centralized numerical tolerances
├── models.py                          # FieldValidationResult and comparison routines
├── manifest.py                        # Manifest v2 builder and validation summary artifact
├── reference_cases.py                 # Backward-compatibility case re-export
├── reference_math.py                  # Backward-compatibility formula re-export
├── reference_manifest.json            # Machine-readable case manifest (Schema v2)
├── reference_validation_summary.json  # Machine-readable deterministic validation summary
├── run_reference_validation.py        # Standalone CLI validation runner
├── cases/                             # Modular case definitions across 6 tranches
│   ├── foundational.py                # Methods 1-10 (20 cases)
│   ├── multigroup.py                  # Methods 11-13 (6 cases)
│   ├── repeated_factorial.py          # Methods 14-16 (7 cases)
│   ├── regression.py                  # Methods 17-18 (4 cases)
│   ├── association_categorical.py     # Methods 19-22 (8 cases)
│   └── reliability.py                 # Methods 23-24 (4 cases)
└── references/                        # Modular Level A reference implementations
    ├── foundational.py                # Reference formulas for methods 1-10
    ├── multigroup.py                  # Welch ANOVA, One-Way ANOVA, Kruskal-Wallis, Holm
    ├── repeated_factorial.py          # RM-ANOVA, Mauchly, GG, Friedman, Two-Way ANOVA
    ├── regression.py                  # OLS, HC3 robust covariance, IRLS/Newton binary Logit
    ├── association_categorical.py     # Kendall tau-b, point-biserial, partial Pearson, McNemar
    └── reliability.py                 # Cronbach alpha, ICC (all 6 configurations)
```

---

## 5. Method Coverage (All 24 Registered Method IDs)

The validation harness covers all 24 registered PyAutoStat method IDs:

### Tranche 1: Foundational Univariate & Bivariate (10 Methods)
1. `one_sample_t`: One-sample mean inference against scalar reference
2. `student_t`: Two-group independent means with pooled equal variance
3. `welch_t`: Two-group independent means with Welch-Satterthwaite unequal variance
4. `paired_t`: Two-condition paired-samples mean difference
5. `mann_whitney_u`: Two-group distribution comparison and rank-biserial correlation
6. `wilcoxon_signed_rank`: Two-condition paired Wilcoxon with `wilcox` zero policy
7. `pearson_correlation`: Bivariate linear correlation with Fisher-z interval
8. `spearman_correlation`: Bivariate monotonic rank correlation
9. `pearson_chi_square`: Categorical independence with Cramer's V (expected $\ge 5$)
10. `fisher_exact`: 2x2 contingency table sample odds ratio and exact p-value

### Tranche 2: Multi-Group Independent Comparisons (3 Methods)
11. `welch_anova`: Welch one-way ANOVA with Games-Howell studentized-range pairwise family
12. `one_way_anova`: Classical Fisher ANOVA with Tukey-Kramer HSD all-pairs comparisons
13. `kruskal_wallis`: Non-parametric Kruskal-Wallis ANOVA with Dunn-Holm pairwise family

### Tranche 3: Within-Subject & Factorial Designs (3 Methods)
14. `repeated_measures_anova`: One-way RM-ANOVA, Mauchly sphericity test, Greenhouse-Geisser correction
15. `friedman_test`: Non-parametric repeated-measures ANOVA, Kendall's $W$, Wilcoxon-Holm pairwise
16. `two_way_anova`: Factorial ANOVA with Type II and Type III sums of squares

### Tranche 4: Linear & Generalized Regression (2 Methods)
17. `linear_regression`: OLS multiple regression, HC3 robust sandwich covariance, Breusch-Pagan, VIF
18. `logistic_regression`: Binary logistic regression with Newton-Raphson MLE, Wald tests, McFadden pseudo-$R^2$

### Tranche 5: Extended Association & Paired Proportions (4 Methods)
19. `kendall_tau_b`: Monotonic association accounting for ties in $x$ and $y$
20. `point_biserial_correlation`: Continuous vs. binary association with directional coding
21. `partial_pearson_correlation`: Linear association controlling for continuous covariates
22. `mcnemar`: Exact binomial test for paired nominal 2x2 contingency tables

### Tranche 6: Scale Reliability & Inter-Rater Agreement (2 Methods)
23. `cronbach_alpha`: Internal consistency, corrected item-total correlation, alpha-if-deleted
24. `intraclass_correlation`: All six Shrout & Fleiss / McGraw & Wong ICC variants:
    - $\text{ICC}(1,1)$ & $\text{ICC}(1,k)$: One-way random effects
    - $\text{ICC}(2,1)$ & $\text{ICC}(2,k)$: Two-way random effects (absolute agreement)
    - $\text{ICC}(3,1)$ & $\text{ICC}(3,k)$: Two-way mixed effects (consistency)

---

## 6. How to Run

From the repository root:

```bash
# Run validation across all 24 methods (default mode: reports discrepancies, exits 0)
python validation/run_reference_validation.py

# Strict mode: returns non-zero exit code if any discrepancy exists
python validation/run_reference_validation.py --strict

# Filter to a specific method
python validation/run_reference_validation.py --method welch_anova
python validation/run_reference_validation.py --method intraclass_correlation

# Generate JSON validation report artifact
python validation/run_reference_validation.py --json report.json

# Regenerate machine-readable case manifest (Schema v2) and summary artifact
python validation/run_reference_validation.py --generate-manifest

# Run harness self-checks (verifies registry integrity, unique IDs, finite tolerances)
python validation/run_reference_validation.py --self-check
```

---

## 7. Numerical Tolerance Policy

Deterministic analytical quantities are evaluated against centralized tolerances defined in `validation/tolerances.py`:
- **Default analytical values**: $\text{atol} = 10^{-12}$, $\text{rtol} = 10^{-10}$
- **Floating-point p-values**: $\text{atol} = 10^{-8}$, $\text{rtol} = 10^{-6}$
- **Numerical iterative solvers** (logistic MLE): $\text{atol} = 10^{-6}$, $\text{rtol} = 10^{-5}$
- **Integer counts & sample accounting**: $\text{atol} = 0$, $\text{rtol} = 0$

---

## 8. Discrepancy Governance

If a numerical discrepancy is detected:
1. Re-run and reproduce the discrepancy.
2. Confirm contrast orientation and missing-data filtering rules.
3. Classify the cause: documentation mismatch, reference error, tolerance rounding, or numerical implementation defect.
4. **Do NOT silently patch statistical code** to force validation to pass. Document the finding in `docs/NUMERICAL_VALIDATION.md` and recommend a bounded remediation phase.
