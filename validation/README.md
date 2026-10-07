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
- Independent rank algorithms, sum of squares, Fisher transformations, and Welch-Satterthwaite calculations.

### LEVEL B — External Backend Conformance
- Direct calls to established numerical libraries (`scipy.stats`) matching PyAutoStat's underlying execution engine.
- Confirms that PyAutoStat correctly parameterizes, delegates to, and extracts quantities from backend libraries.
- *Critical Principle*: Level B alone is NEVER labeled as "independently validated."

### LEVEL C — Internal Invariant / Reproducibility
- Algebraic invariants (e.g. $n_1 + n_2 = \text{analyzed\_rows}$, $\text{analyzed} + \text{excluded} = \text{original}$).
- Sample-accounting invariants (complete pair filtering, zero-difference tracking).
- Sign and orientation consistency (contrast definitions: first minus second).

### LEVEL D — Deferred
- Quantities whose independent calculation is not implemented in this phase.
- Examples: Bootstrap confidence intervals, noncentral-t exact inversion intervals.
- These fields are visibly recorded as `deferred` rather than omitted or falsely certified.

---

## 3. Standing Architectural Constraints

To preserve repository safety and honor the user's standing constraints:
- **No Test Frameworks**: This harness does NOT use `pytest`, `unittest.TestCase`, or files named `test_*.py`.
- **No Package Pollution**: This directory is NOT imported by `src/pyautostat` and is never bundled into runtime distributions.
- **No CI Intrusion**: This harness is not part of the standard CI test job in Phase 2.
- **Standalone Execution**: The runner is executed manually from the repository root.

---

## 4. Directory Structure

```
validation/
├── README.md                    # This document
├── reference_math.py            # Independent Level A manual formulas and Level B cross-checks
├── reference_cases.py           # Reproducible validation datasets and case expectations
├── reference_manifest.json      # Machine-readable case inventory with metadata and tolerances
└── run_reference_validation.py  # Standalone CLI validation runner and reporter
```

---

## 5. First-Tranche Methods (10 Foundational Methods)

Phase 2 focuses on 10 high-use, foundational statistical methods:

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

---

## 6. How to Run

From the repository root:

```bash
# Run all first-tranche validation cases
python validation/run_reference_validation.py

# Filter to a specific method
python validation/run_reference_validation.py --method welch_t
python validation/run_reference_validation.py --method pearson_correlation

# Generate JSON validation artifact
python validation/run_reference_validation.py --json validation_report.json

# Strict mode (fails with non-zero exit code on any discrepancy)
python validation/run_reference_validation.py --strict
```

---

## 7. Numerical Tolerance Policy

Deterministic analytical quantities are evaluated against centralized tolerances:
- **Default analytical values**: $\text{atol} = 10^{-12}$, $\text{rtol} = 10^{-10}$
- **Floating-point p-values**: $\text{atol} = 10^{-8}$, $\text{rtol} = 10^{-6}$

---

## 8. Discrepancy Policy

If a numerical discrepancy is detected:
1. Re-run and reproduce the discrepancy.
2. Confirm contrast orientation and missing-data filtering rules.
3. Classify the cause: documentation mismatch, reference error, tolerance rounding, or numerical implementation defect.
4. **Do NOT silently patch statistical code** to force validation to pass. Document the finding in `docs/NUMERICAL_VALIDATION.md` and recommend a bounded remediation phase.
