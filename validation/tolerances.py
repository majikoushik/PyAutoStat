"""Centralized numerical tolerances for PyAutoStat reference validation."""

from __future__ import annotations

# Primary Analytical Statistics (means, differences, test statistics, standard errors,
# degrees of freedom, effect sizes)
DEFAULT_ATOL = 1e-12
DEFAULT_RTOL = 1e-10

# Probabilities & p-values (numerical tail integration)
PVAL_ATOL = 1e-8
PVAL_RTOL = 1e-6

# Integer counts, table dimensions, sample accounting (exact match)
INT_ATOL = 0.0
INT_RTOL = 0.0

# Slightly wider tolerance for complex iterative optimization or matrix inversions
# (e.g. IRLS logistic)
NUMERICAL_SOLVER_ATOL = 1e-8
NUMERICAL_SOLVER_RTOL = 1e-6
