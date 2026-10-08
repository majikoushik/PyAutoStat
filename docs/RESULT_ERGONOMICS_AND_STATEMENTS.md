# PyAutoStat Result Ergonomics & Deterministic Scientific Statements

Authoritative documentation for PyAutoStat's reviewer-style convenience accessors,
primary result extraction, tabular views, and deterministic APA-oriented statistical statements.

---

## 1. Why the Convenience Layer Exists

PyAutoStat produces structured, immutable scientific records encapsulated in
`AnalysisResult` and `ResearchWorkflowResult`. While the underlying `.values` dictionary
preserves complete, unrounded numerical precision and audit artifacts, extracting key
inferential results (e.g. test statistic, canonical $p$-value, effect size, and confidence
intervals) historically required knowledge of internal dictionary schemas across 24 distinct
method families.

The result ergonomics layer provides clean, readable convenience accessors directly on
`AnalysisResult` and `ResearchWorkflowResult`:

```python
workflow = assistant.run(outcome="score", predictor="group", design="independent", estimand="mean")
result = workflow.analysis

print(result.test_name)          # "Welch independent-samples t-test"
print(result.statistic)          # -10.98...
print(result.p_value)            # 3.76e-10
print(result.degrees_of_freedom) # 20.97...
print(result.estimate)           # -3.30
print(result.confidence_interval)# {'lower': -3.93, 'upper': -2.67, 'level': 0.95, ...}
print(result.apa_statement())    # "Welch's independent-samples t-test: t(21.0) = -10.98, p < .001, Cohen's d = -4.48, 95% CI [-3.93, -2.67] for the mean difference."
```

---

## 2. Stored Values Remain Authoritative

**Non-Negotiable Scientific Principle**:
The convenience layer **never recomputes statistics**.

Every convenience property, summary dictionary, tabular DataFrame view, and APA-oriented
statement is derived strictly by reading values already computed and recorded in
`AnalysisResult.values`, `AnalysisResult.metadata`, or `AnalysisResult.specification`.

The layer will **never**:
- Invoke SciPy or statsmodels;
- Call PyAutoStat internal calculation or inference engines;
- Re-run bootstrap resamples;
- Change an alpha level or hypothesis orientation;
- Fabricate a confidence interval when none was recorded;
- Convert an ambiguous result into a misleading single number.

---

## 3. Convenience Properties on `AnalysisResult`

The following read-only convenience properties are available on `AnalysisResult`:

| Property | Return Type | Description |
| :--- | :--- | :--- |
| `result.test_name` | `str` | Human-readable scientific method name (delegates to `result.method_label`). |
| `result.statistic` | `float \| None` | Primary scalar test statistic, or `None` if scientifically ambiguous. |
| `result.p_value` | `float \| None` | Primary scalar inferential $p$-value, or `None` if ambiguous. |
| `result.degrees_of_freedom` | `int \| float \| tuple \| None` | Analytical degrees of freedom, tuple for multigroup/repeated $F$-tests, or `None`. |
| `result.estimate` | `float \| None` | Primary scalar point estimate (e.g. mean difference, correlation $r$, reliability $\alpha$), or `None`. |
| `result.confidence_interval` | `dict[str, Any] \| None` | Defensive copy of stored primary estimate confidence interval dictionary, or `None`. |
| `result.effect_size` | `dict[str, Any] \| None` | Defensive copy of stored effect size record, or `None` if unavailable or ambiguous. |
| `result.effect_size_confidence_interval` | `dict[str, Any] \| None` | Defensive copy of stored effect size confidence interval dictionary, or `None`. |
| `result.sample_accounting` | `dict[str, Any]` | Defensive dictionary summarizing analyzed rows, excluded rows, and group/pair sizes. |

---

## 4. When a Scalar Property Returns `None` (Ambiguity Safeguards)

To protect researchers against reporting flattened or misleading numbers, scalar properties
(`statistic`, `p_value`, `estimate`, `confidence_interval`) strictly return `None` whenever
a single primary scalar is scientifically ambiguous or undefined:

- **Two-Way Factorial ANOVA (`two_way_anova`)**: Contains distinct $F$-statistics and $p$-values
  for Factor A, Factor B, and the interaction term ($A \times B$). Silently returning Factor A's
  $p$-value would misrepresent the omnibus factorial model. Therefore, `result.statistic` and
  `result.p_value` return `None`. Complete statistics are accessed via `result.primary_result()["terms"]`
  or `result.to_dataframe("terms")`.
- **Multiple Linear Regression (`linear_regression`)**: Contains both an omnibus model-fit $F$-test
  and multiple individual coefficient $t$-tests ($B$, $\text{SE}$, $t$, $p$, $\text{CI}$).
  Silently picking the first predictor's $p$-value is scientifically invalid. Therefore, `result.statistic`
  and `result.p_value` return `None`. The omnibus fit and coefficient table are available via
  `result.primary_result()["model_fit"]`, `result.primary_result()["coefficients"]`, or `result.to_dataframe("coefficients")`.
- **Logistic Regression (`logistic_regression`)**: Contains an omnibus likelihood-ratio $\chi^2$ test
  and multiple coefficient Wald $z$-tests. Scalar properties return `None`; structured sections are accessed
  via `model_fit` and `coefficients`.
- **Scale Reliability (`cronbach_alpha`)**: Cronbach's $\alpha$ is a descriptive internal-consistency
  coefficient without a null-hypothesis significance test. `result.estimate` returns the sample $\alpha$,
  while `result.statistic` and `result.p_value` return `None`.

No unavailable scalar is ever replaced with a placeholder zero.

---

## 5. `result.primary_result()`

`AnalysisResult.primary_result()` returns a deterministic, non-serialized convenience dictionary:

```python
summary = result.primary_result()
```

### Shape for Canonical Single-Test Methods
```python
{
    "method_id": "welch_t",
    "test_name": "Welch independent-samples t-test",
    "statistic": -10.9754,
    "statistic_label": "t",
    "degrees_of_freedom": 20.9735,
    "p_value": 3.76e-10,
    "estimate": -3.30,
    "estimate_label": "mean difference",
    "confidence_interval": {"lower": -3.93, "upper": -2.67, "level": 0.95, ...},
    "effect_size": {"name": "Cohen's d", "value": -4.48, ...},
    "effect_size_confidence_interval": {"lower": -6.33, "upper": -3.61, "level": 0.95, ...},
    "sample_accounting": {"analyzed_rows": 24, "excluded_rows": 0, ...},
}
```

### Shape for Complex Models
For complex models, `primary_result()` provides structured sub-sections:
- `two_way_anova`: includes `"terms"`, `"cell_summaries"`, `"estimated_marginal_means"`, `"followups"`.
- `linear_regression`: includes `"model_fit"`, `"coefficients"`, `"diagnostics"`.
- `logistic_regression`: includes `"model_fit"`, `"coefficients"`, `"diagnostics"`.
- `welch_anova` / `one_way_anova` / `kruskal_wallis` / `repeated_measures_anova` / `friedman_test`: includes `"pairwise_comparisons"`.

All returned mappings and lists are defensive deep copies; mutations by the caller cannot alter the underlying frozen result.

---

## 6. DataFrame View: `result.to_dataframe()`

For workflows requiring direct pandas DataFrame integration, `AnalysisResult.to_dataframe()` extracts
structured tabular sections without statistical rerun:

```python
# Single-row summary of primary result scalars
df_primary = result.to_dataframe(section="primary")

# Coefficients table for linear or logistic regression
df_coefs = reg_result.to_dataframe(section="coefficients")

# Factorial ANOVA summary table for two-way ANOVA
df_terms = anova_2w_result.to_dataframe(section="terms")

# Pairwise post-hoc comparisons table
df_pw = anova_result.to_dataframe(section="comparisons")
```

If an unsupported or unavailable section is requested, a clear `ValueError` is raised listing the valid sections for that method.

---

## 7. Deterministic APA-Oriented Statistical Statements

PyAutoStat generates narrative statistical statements following APA-oriented formatting conventions:

```python
# Directly from an AnalysisResult
statement = result.apa_statement()

# Or from an integrated ResearchWorkflowResult
statement = workflow.apa_statement()
```

### Terminology Policy: "APA-Oriented"
Statements follow standard American Psychological Association (APA) 7th Edition style conventions for reporting statistical tests in scientific manuscripts.
PyAutoStat uses the descriptor **"APA-oriented statement"** or **"APA-style formatting conventions"** and **does not claim formal compliance certification**.

---

## 8. Formatting Conventions

The statement formatter applies deterministic rules to already-stored numbers:

1. **$p$-Values**:
   - $p < .001$ is formatted as `p < .001`.
   - Other $p$-values are formatted to three decimal places without leading zeros: e.g. `p = .032`.
   - Stored non-zero floating-point values are never reported as exact zero.
2. **Quantities Bounded by 1**:
   - Quantities strictly bounded by 1 (e.g. correlations $r$, $r_s$, $\tau$, $\eta^2$, partial $\eta^2$, Cramer's $V$, Kendall's $W$, McFadden pseudo-$R^2$, Cronbach's $\alpha$) omit the leading zero: e.g. `r = .43`, `partial eta^2 = .75`, `alpha = .94`.
3. **Unbounded Quantities**:
   - Statistics that can exceed 1 (e.g. $t$, $F$, $U$, $W$, $\chi^2$, odds ratios, unstandardized regression slopes $B$, standard errors $\text{SE}$) retain leading zeros: e.g. `t(21.0) = -10.98`, `F(2, 14.0) = 91.58`, `OR = 3.70`.
4. **Confidence Intervals**:
   - Formatted in APA bracket notation: `95% CI [-3.93, -2.67]`.
5. **Degrees of Freedom**:
   - Integer degrees of freedom are formatted as integers: `t(21) = 2.81`.
   - Fractional degrees of freedom (e.g. Welch-Satterthwaite approximations) retain one decimal place: `t(21.0) = -10.98`.

---

## 9. Examples by Method Family

### A. Independent Two-Group $t$-Test (`welch_t`)
```python
workflow = assistant.run(objective="compare_groups", outcome="score", predictor="group", design="independent", estimand="mean")
print(workflow.apa_statement())
# Welch's independent-samples t-test: t(21.0) = -10.98, p < .001, Cohen's d = -4.48, 95% CI [-3.93, -2.67] for the mean difference.
```

### B. Non-Parametric Two-Group Rank Comparison (`mann_whitney_u`)
```python
workflow = assistant.run(objective="compare_groups", outcome="score", predictor="group", design="independent", estimand="distribution")
print(workflow.apa_statement())
# Mann-Whitney U test: U = 0.0, p < .001, rank-biserial r = -1.00.
```

### C. Multi-Group Independent ANOVA (`welch_anova`)
```python
workflow = assistant.run(objective="compare_groups", outcome="val", predictor="grp", design="independent", estimand="mean")
print(workflow.apa_statement())
# Welch one-way ANOVA with Games-Howell comparisons: F(2, 14.0) = 91.58, p < .001.
```

### D. Repeated-Measures ANOVA (`repeated_measures_anova`)
```python
workflow = assistant.run(objective="compare_groups", outcome="measure", predictor="time", unit_id="subject", design="repeated", condition_order=["T1", "T2", "T3"], estimand="mean")
print(workflow.apa_statement())
# Repeated-measures ANOVA: F(2, 14) = 276.09, p < .001, partial eta^2 = .98.
```

### E. Factorial Two-Way ANOVA (`two_way_anova`)
```python
workflow = assistant.two_way_anova(outcome="y", factor_a="A", factor_b="B", sum_of_squares="type2")
print(workflow.apa_statement())
# Two-way factorial ANOVA (TYPE2):
#   - A: F(1, 20) = 1706.37, p < .001, partial eta^2 = .99.
#   - B: F(1, 20) = 600.49, p < .001, partial eta^2 = .97.
#   - A:B: F(1, 20) = 59.31, p < .001, partial eta^2 = .75.
```

### F. Multiple Linear Regression (`linear_regression`)
```python
workflow = assistant.run(objective="regression", outcome="y", predictors=["x1", "x2"], design="independent", estimand="conditional_mean", data_dictionary=dict_spec)
print(workflow.apa_statement())
# Linear regression model fit: F(2, 7) = 17224.66, p < .001, R^2 = 1.00, adjusted R^2 = 1.00.
#   - Intercept: B = -0.18, SE = 0.13, t = -1.43, p = .195, 95% CI [-0.48, 0.12].
#   - x1: B = 2.73, SE = 0.29, t = 9.36, p < .001, 95% CI [2.04, 3.42].
#   - x2: B = -1.05, SE = 0.42, t = -2.50, p = .041, 95% CI [-2.05, -0.06].
```

### G. Binary Logistic Regression (`logistic_regression`)
```python
workflow = assistant.run(objective="regression", outcome="y", predictors=["x"], design="independent", estimand="event_probability", event_level="yes", data_dictionary=dict_spec)
print(workflow.apa_statement())
# Binary logistic regression (event = 'yes'): Likelihood-ratio chi^2(1) = 11.61, p < .001, McFadden pseudo-R^2 = .70.
#   - Intercept: B = -8.50, SE = 5.53, z = -1.54, p = .124, OR = 0.00.
#   - x: B = 1.31, SE = 0.83, z = 1.57, p = .116, OR = 3.70.
```

### H. Pearson Correlation (`pearson_correlation`)
```python
workflow = assistant.run(objective="association", outcome="y", predictor="x1", design="independent", estimand="linear", data_dictionary=dict_spec)
print(workflow.apa_statement())
# Pearson correlation: r(8) = 1.00, p < .001, 95% CI [1.00, 1.00].
```

### I. Contingency Independence (`pearson_chi_square`)
```python
workflow = assistant.run(objective="association", outcome="cat1", predictor="cat2", design="independent", estimand="categorical_independence", data_dictionary=dict_spec)
print(workflow.apa_statement())
# Pearson's chi-square test: chi^2(1, N = 60) = 6.67, p = .010, Cramer's V = .33.
```

### J. Scale Reliability (`cronbach_alpha`)
```python
workflow = assistant.run(objective="reliability", items=["item1", "item2", "item3"], design="independent", estimand="internal_consistency", data_dictionary=dict_spec)
print(workflow.apa_statement())
# Cronbach's alpha: alpha = .94, k = 3 items, N = 10 respondents, 95% CI [.87, .98].
```

### K. Inter-Rater Agreement (`intraclass_correlation`)
```python
workflow = assistant.run(objective="reliability", target="subject", rater="judge", outcome="rating", model="two_way_random", definition="absolute_agreement", unit="single")
print(workflow.apa_statement())
# Intraclass correlation ICC(2,1) (Two-way random-effects, absolute-agreement, single-measure ICC (ICC(2,1))): ICC = .90, 95% CI [.44, .99], F(5, 5) = 16.50, p = .004.
```

---

## 10. Relationship to `workflow.explain()` and Research Reports

PyAutoStat maintains distinct presentation layers for different communication contexts:

| Surface | Function | Content |
| :--- | :--- | :--- |
| `workflow.apa_statement()` | Concise narrative statement | 1–3 sentence statistical results summary for direct copy-paste into scientific manuscript text. |
| `workflow.explain()` | Multi-section terminal summary | Rich explanation of research question, design, sample accounting, assumptions, diagnostics, primary inference, effect size, and scientific limitations. |
| `show(workflow)` | Interactive Rich rendering | Full-color terminal presentation with stylized panels, diagnostic tables, and visual warning badges. |
| `ResearchReport` | Comprehensive document export | Archival research report exportable to Markdown, HTML, LaTeX, PDF, DOCX, and reproducibility research packages. |

---

## 11. Serialization and Reproducibility Safety

Convenience accessors do not modify result serialization schemas:
- `result.to_dict()` produces the exact same keys and structure as prior versions.
- Dataset cryptographic fingerprints, audit ledgers, and replay packages remain unchanged.
- Calling convenience properties or statement generators is purely read-only and free of side effects.
