# PyAutoStat Result Ergonomics Consistency Audit

This document records the exhaustive audit of all 24 registered statistical methods in PyAutoStat against the reviewer-style result ergonomics layer, public properties, DataFrame adapters, and deterministic APA-oriented reporting statements.

---

## 1. Executive Summary & Verification State

- **Audited Scope**: All 24 registered method IDs.
- **Underlying Principle**: **Zero recalculation**; the convenience layer derives numbers strictly from authoritative stored values in `AnalysisResult.values`, `metadata`, or `specification`.
- **Status Classification**:
  - `correct`: Public property or statement returns the exact authoritative stored quantity.
  - `intentionally_none`: Scientifically ambiguous or undefined scalar; strictly returns `None` to prevent misleading summaries.
  - `structured_only`: Complex multi-parameter model; results accessed through dedicated sub-tables or dictionaries.
  - `unavailable_by_design`: Quantity is not defined or not computed for the stated scientific estimand.
  - `needs_fix`: Defect in extraction or presentation (all resolved in ergonomics hardening; 0 rows remaining).

---

## 2. Ambiguity Rules for Complex Statistical Models

Four statistical method families have ambiguous scalar summaries. To protect researchers from reporting flattened or misleading numbers, PyAutoStat strictly enforces `None` for scalar properties (`statistic`, `p_value`, `estimate`, `degrees_of_freedom`):

1. **Two-Way Factorial ANOVA (`two_way_anova`)**:
   - Contains distinct $F$-tests, degrees of freedom, and $p$-values for Factor A, Factor B, and the interaction ($A \times B$).
   - Returning Factor A's statistic or $p$-value would silently misrepresent the omnibus factorial design.
   - **Resolution**: `statistic = None`, `p_value = None`, `degrees_of_freedom = None`. All inferential tests are structured under `result.primary_result()["terms"]` and `result.to_dataframe("terms")`.
   - **Statement Rule**: The APA-oriented statement reports each non-residual term with its numerator df and the model's residual denominator df: $F(\text{df}_{\text{num}}, \text{df}_{\text{resid}}) = \dots, p = \dots, \text{partial } \eta^2 = \dots$. The residual row is excluded from inferential narration.

2. **Multiple Linear Regression (`linear_regression`)**:
   - Contains an omnibus model-fit $F$-test ($R^2$, adjusted $R^2$, omnibus $F$, $p$) and individual predictor $t$-tests ($B$, $\text{SE}$, $t$, $p$, 95% CI).
   - Selecting a single predictor's $p$-value or the omnibus $p$-value as `result.p_value` is scientifically ambiguous.
   - **Resolution**: `statistic = None`, `p_value = None`, `degrees_of_freedom = None`. Model fit is accessed via `result.primary_result()["model_fit"]`, coefficients via `result.to_dataframe("coefficients")`.
   - **Statement Rule**: Omnibus fit is reported first ($F(\text{df}_1, \text{df}_2) = \dots, p = \dots, R^2 = \dots$), followed by each predictor coefficient ($B = \dots, \text{SE} = \dots, t = \dots, p = \dots, 95\%\text{ CI}$).

3. **Logistic Regression (`logistic_regression`)**:
   - Contains an omnibus likelihood-ratio $\chi^2$ test (LR statistic, df, $p$, McFadden's pseudo-$R^2$) and multiple predictor Wald $z$-tests ($\text{OR}$, 95% CI, $z$, $p$).
   - Selecting a single coefficient's $p$-value or the LR test as the universal scalar is scientifically ambiguous.
   - **Resolution**: `statistic = None`, `p_value = None`, `degrees_of_freedom = None`. Access via `model_fit` and `coefficients`.
   - **Statement Rule**: Omnibus LR test is reported with event-level disclosure and McFadden pseudo-$R^2$ (from `model_fit["mcfadden_r2"]`), followed by predictor odds ratios and Wald tests ($\text{OR} = \dots, 95\%\text{ CI}, z = \dots, p = \dots$).

4. **Scale Reliability (`cronbach_alpha`)**:
   - Cronbach's $\alpha$ is a descriptive internal-consistency metric, not a null-hypothesis test statistic. It has no null value, test statistic, or inferential $p$-value.
   - **Resolution**: `statistic = None`, `p_value = None`, `degrees_of_freedom = None`. `result.estimate` returns sample $\alpha$.
   - **Statement Rule**: Reports $\alpha = \dots, k = \dots\text{ items}, N = \dots\text{ respondents}$.

---

## 3. Comprehensive 24-Method Ergonomics Consistency Matrix

| Method ID | statistic | p_value | df | estimate | primary CI | effect | effect CI | sample accounting | primary_result sections | APA statement | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `welch_t` | float ($t$) | float ($p$) | float ($\text{df}_{\text{ws}}$) | float ($\Delta\mu$) | dict (95% CI) | dict (Cohen's $d$) | dict (boot CI) | original, analyzed, excluded, group_sizes | primary | $t(\text{df}) = \dots, p = \dots, d = \dots, \text{CI}$ | correct |
| `student_t` | float ($t$) | float ($p$) | float ($n_1+n_2-2$) | float ($\Delta\mu$) | dict (95% CI) | dict (Cohen's $d$) | dict (boot CI) | original, analyzed, excluded, group_sizes | primary | $t(\text{df}) = \dots, p = \dots, d = \dots, \text{CI}$ | correct |
| `paired_t` | float ($t$) | float ($p$) | int ($n_{\text{pairs}}-1$) | float ($\Delta\mu$) | dict (95% CI) | dict (Cohen's $d_z$) | dict (boot CI) | original, analyzed, excluded, total_units, complete_pairs, incomplete_units, excluded_units | primary | $t(\text{df}) = \dots, p = \dots, d_z = \dots, \text{CI}$ | correct |
| `one_sample_t` | float ($t$) | float ($p$) | int ($n-1$) | float ($\Delta\mu$) | dict (95% CI) | dict (Cohen's $d$) | dict (boot CI) | original, analyzed, excluded | primary | $t(\text{df}) = \dots, p = \dots, d = \dots, \text{CI}$ | correct |
| `mann_whitney_u` | float ($U$) | float ($p$) | `None` (intentionally_none) | float ($r_{\text{rb}}$) | `None` (unavailable) | dict ($r_{\text{rb}}$) | dict (boot CI) | sample_size, group_sizes | primary | $U = \dots, p = \dots, r_{\text{rb}} = \dots$ | correct |
| `wilcoxon_signed_rank` | float ($W$) | float ($p$) | `None` (intentionally_none) | float ($r_{\text{rb}}$) | `None` (unavailable) | dict ($r_{\text{rb}}$) | dict (boot CI) | complete_pairs, nonzero/zero_diffs | primary | $W = \dots, p = \dots, r_{\text{rb}} = \dots$ | correct |
| `welch_anova` | float ($F$) | float ($p$) | tuple ($k-1, \text{df}_{\text{error}}$) | `None` (omnibus) | `None` (unavailable) | `None` (not applicable) | `None` (not applicable) | sample_size, group_sizes | primary, comparisons, group_summaries | omnibus $F(\text{df}_1, \text{df}_2) = \dots, p = \dots$ | correct |
| `one_way_anova` | float ($F$) | float ($p$) | tuple ($k-1, N-k$) | `None` (omnibus) | `None` (unavailable) | dict ($\eta^2$) | `None` (unavailable) | sample_size, group_sizes | primary, comparisons, group_summaries | omnibus $F(\text{df}_1, \text{df}_2) = \dots, p = \dots$ | correct |
| `kruskal_wallis` | float ($H$) | float ($p$) | int ($k-1$) | `None` (omnibus) | `None` (unavailable) | dict ($\epsilon^2$) | `None` (unavailable) | sample_size, group_sizes | primary, comparisons, group_summaries | $H(\text{df}) = \dots, p = \dots, \epsilon^2 = \dots$ | correct |
| `pearson_correlation` | float ($r$) | float ($p$) | `None` (intentionally_none) | float ($r$) | dict (Fisher $z$) | dict ($r$) | dict (Fisher $z$) | original, analyzed, excluded | primary | $r = \dots, p = \dots, 95\%\text{ CI}$ (no derived df) | correct |
| `spearman_correlation` | float ($r_s$) | float ($p$) | `None` (intentionally_none) | float ($r_s$) | dict (boot CI) | dict ($r_s$) | dict (boot CI) | original, analyzed, excluded | primary | $r_s = \dots, p = \dots, 95\%\text{ CI}$ | correct |
| `kendall_tau_b` | float ($\tau$) | float ($p$) | `None` (intentionally_none) | float ($\tau$) | dict (asymptotic) | dict ($\tau$) | dict (asymptotic) | original, analyzed, excluded | primary | $\tau = \dots, p = \dots, 95\%\text{ CI}$ | correct |
| `point_biserial_correlation` | float ($r_{\text{pb}}$) | float ($p$) | int ($n-2$) | float ($r_{\text{pb}}$) | dict (analytical) | dict ($r_{\text{pb}}$) | dict (analytical) | sample_size, group_sizes | primary | $r_{\text{pb}}(\text{df}) = \dots, p = \dots, 95\%\text{ CI}$ | correct |
| `partial_pearson_correlation` | float ($r_{\text{part}}$) | float ($p$) | int ($n-2-k$) | float ($r_{\text{part}}$) | dict (Fisher $z$) | dict ($r_{\text{part}}$) | dict (Fisher $z$) | original, analyzed, excluded | primary | $r_{\text{part}}(\text{df}) = \dots, p = \dots, 95\%\text{ CI}$ | correct |
| `pearson_chi_square` | float ($\chi^2$) | float ($p$) | int ($(r-1)(c-1)$) | float ($V$) | `None` (unavailable) | dict (Cramer's $V$) | dict (boot CI) | sample_size, cell_counts | primary | $\chi^2(\text{df}, N = \dots) = \dots, p = \dots, V = \dots$ | correct |
| `fisher_exact` | float ($\text{OR}$) | float ($p$) | `None` (intentionally_none) | float ($\text{OR}$) | dict (exact conditional) | dict ($\text{OR}$) | dict (exact conditional) | original, analyzed, excluded | primary | $p = \dots, \text{sample OR} = \dots, 95\%\text{ CI}$ | correct |
| `mcnemar` | float ($\Delta p$) | float ($p$) | `None` (intentionally_none) | float ($\Delta p$) | dict (Wilson/Newcombe) | dict ($\Delta p$) | dict (Wilson/Newcombe) | complete_pairs, analyzed, excluded | primary | Exact McNemar test: $p = \dots, \Delta p = \dots, \text{CI}$ | correct |
| `linear_regression` | `None` (intentionally_none) | `None` (intentionally_none) | `None` (intentionally_none) | `None` (structured_only) | `None` (structured_only) | `None` (structured_only) | `None` (structured_only) | sample_size, analyzed, excluded | primary, model_fit, coefficients, diagnostics | Omnibus $F(\text{df}_1, \text{df}_2) = \dots, R^2 = \dots$; coefficients | correct |
| `logistic_regression` | `None` (intentionally_none) | `None` (intentionally_none) | `None` (intentionally_none) | `None` (structured_only) | `None` (structured_only) | `None` (structured_only) | `None` (structured_only) | sample_size, analyzed, excluded | primary, model_fit, coefficients, diagnostics | LR $\chi^2(\text{df}) = \dots$, McFadden $R^2 = \dots$; ORs | correct |
| `cronbach_alpha` | `None` (intentionally_none) | `None` (intentionally_none) | `None` (intentionally_none) | float ($\alpha$) | dict (boot CI if run) | dict ($\alpha$) | dict (boot CI if run) | sample_size, item_count | primary | $\alpha = \dots, k = \dots\text{ items}, N = \dots\text{ respondents}$ | correct |
| `repeated_measures_anova` | float ($F$) | float ($p$) | tuple ($\text{df}_1, \text{df}_2$) | `None` (omnibus) | `None` (unavailable) | dict (partial $\eta^2$) | `None` (unavailable) | total_units, complete_units, condition_count | primary, comparisons, group_summaries | $F(\text{df}_1, \text{df}_2) = \dots, p = \dots, \text{partial } \eta^2 = \dots$ (GG if applied) | correct |
| `friedman_test` | float ($Q$) | float ($p$) | int ($k-1$) | `None` (omnibus) | `None` (unavailable) | dict (Kendall's $W$) | dict (boot CI) | total_units, complete_units, condition_count | primary, comparisons, group_summaries | $Q(\text{df}) = \dots, p = \dots, W = \dots$ | correct |
| `two_way_anova` | `None` (intentionally_none) | `None` (intentionally_none) | `None` (intentionally_none) | `None` (structured_only) | `None` (structured_only) | `None` (structured_only) | `None` (structured_only) | sample_size, cell_counts | primary, terms, cell_summaries, followups | Type SS: Factor $F(\text{df}_{\text{num}}, \text{df}_{\text{resid}}) = \dots, p = \dots, \text{partial } \eta^2 = \dots$ | correct |
| `intraclass_correlation` | float ($F$) | float ($p$) | tuple ($\text{df}_1, \text{df}_2$) | float ($\text{ICC}$) | dict (exact F-inversion) | dict ($\text{ICC}$) | dict (exact F-inversion) | sample_size, targets, raters | primary, f_test, all_variants | $\text{ICC}(m,k): \text{ICC} = \dots, \text{CI}, F(\text{df}_1, \text{df}_2) = \dots, p = \dots$ | correct |

---

## 4. Remediation Highlights from Ergonomics Hardening

1. **Effect-Size CI Lookup (`result.effect_size_confidence_interval`)**:
   - Resolved key-path bug: authoritative location is `values["effect_size"]["confidence_interval"]`.
   - Returns defensive deep copies for methods including t-tests, Wilcoxon, Spearman, Cramer's V, repeated-measures ANOVA, and ICC.

2. **Authoritative Degrees of Freedom Extraction (`result.degrees_of_freedom`)**:
   - Reads canonical `values["degrees_of_freedom"]`.
   - Returns scalar for single-df tests (t-tests, chi-square, Kruskal-Wallis, Friedman, point-biserial, partial-r).
   - Returns 2-element tuple for ANOVA variants (`one_way_anova`, `welch_anova`, `repeated_measures_anova`, `intraclass_correlation`).
   - Returns `None` for ambiguous complex models (`two_way_anova`, `linear_regression`, `logistic_regression`, `cronbach_alpha`) and non-parametric tests that do not store inferential df.

3. **Strict Zero-Recalculation in Statements**:
   - Pearson correlation statement no longer derives unstored $n-2$ df; formats strictly stored quantities: $r = \dots, p = \dots, 95\%\text{ CI } [\dots]$.
   - Two-way ANOVA statement accurately extracts residual error df and reports $F(\text{df}_{\text{num}}, \text{df}_{\text{resid}})$, partial $\eta^2$ from `term["effect_size"]["value"]`, and completely excludes the residual term from inferential reporting.
   - Logistic regression statement extracts McFadden's pseudo-$R^2$ from `model_fit["mcfadden_r2"]`.
   - McNemar statement explicitly labels the method as "Exact McNemar test" and avoids inaccurate "continuity-corrected" descriptions.

4. **Preserved Data Types in `to_dataframe("primary")`**:
   - Degrees of freedom in the primary DataFrame view preserves exact integer, float, tuple, or `None` objects rather than converting to formatted display strings.
