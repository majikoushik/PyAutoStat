# Intraclass Correlation Coefficient (ICC) Guide

This guide describes the statistical methodology, mathematical definitions, design assumptions,
computational implementation, and reporting recommendations for the Intraclass Correlation
Coefficient (ICC) in PyAutoStat.

---

## 1. Overview and Scientific Purpose

The Intraclass Correlation Coefficient (ICC) assesses the reliability, reproducibility, and agreement
of quantitative measurements obtained across multiple raters, judges, observers, devices, or test
occasions. Unlike inter-item internal consistency metrics such as Cronbach's alpha (which evaluate
item covariance on a scale), ICC explicitly decomposes measurement variance into target-level,
rater-level, and residual components using Analysis of Variance (ANOVA).

PyAutoStat implements all six canonical variants formulated by Shrout and Fleiss (1979) and later
clarified and generalized by McGraw and Wong (1996).

---

## 2. Experimental Designs and Models

The choice of ICC depends on three fundamental design decisions:

1. **Model** (how raters/judges were selected):
   - **One-Way Random (`one_way_random`)**: Each target is rated by a different set of $k$ raters
     randomly selected from a larger population of raters.
   - **Two-Way Random (`two_way_random`)**: A fixed set of $k$ raters is randomly selected from a
     larger population of raters, and every target is evaluated by all $k$ raters (fully crossed).
     Both target effects and rater effects are treated as random.
   - **Two-Way Mixed (`two_way_mixed`)**: A specific, fixed set of $k$ raters evaluates every
     target (fully crossed). Target effects are random, but rater effects are fixed (results do
     not generalize beyond the specific raters in the study).

2. **Definition** (what constitutes agreement):
   - **Absolute Agreement (`absolute_agreement`)**: Systematic differences between raters count as
     disagreement. Both rater bias and residual variance reduce reliability.
   - **Consistency (`consistency`)**: Systematic rater differences (e.g., one rater scoring
     consistently 1 point higher than another) are treated as additive bias and partitioned out.
     Only the relative ranking or pattern across targets matters.
   - *Note*: In the one-way random model, rater variance cannot be separated from residual variance,
     so the distinction between agreement and consistency does not apply.

3. **Unit** (how measurements will be applied in practice):
   - **Single (`single`)**: The reliability of an individual rating from a single rater.
   - **Average (`average`)**: The reliability of the mean of ratings across all $k$ raters.

---

## 3. The Six Canonical Variants

| Shrout & Fleiss (1979) | McGraw & Wong (1996) | PyAutoStat Identifier | Model | Definition | Unit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ICC(1,1)** | ICC(1) | `icc_1_1` | `one_way_random` | N/A | `single` |
| **ICC(1,k)** | ICC(k) | `icc_1_k` | `one_way_random` | N/A | `average` |
| **ICC(2,1)** | ICC(A,1) | `icc_2_1` | `two_way_random` | `absolute_agreement` | `single` |
| **ICC(2,k)** | ICC(A,k) | `icc_2_k` | `two_way_random` | `absolute_agreement` | `average` |
| **ICC(3,1)** | ICC(C,1) | `icc_3_1` | `two_way_mixed` | `consistency` | `single` |
| **ICC(3,k)** | ICC(C,k) | `icc_3_k` | `two_way_mixed` | `consistency` | `average` |

### McGraw-Wong Aliases

McGraw and Wong (1996) noted that:
- Evaluating consistency in a two-way random model ($C(2,1)$ and $C(2,k)$) yields mathematical
  formulas identical to ICC(3,1) and ICC(3,k), but with broader generalization across random raters.
- Evaluating absolute agreement in a two-way mixed model ($A(3,1)$ and $A(3,k)$) yields formulas
  identical to ICC(2,1) and ICC(2,k), but with inference restricted to the fixed raters.

PyAutoStat supports these configurations and accurately documents their mathematical and
inferential interpretations.

---

## 4. ANOVA Mean Squares and Variance Components

For a balanced panel with $n$ targets and $k$ raters ($N = n \times k$ observations):

- **Between-Targets Mean Square ($BMS$)**:
  $$BMS = \frac{k}{n - 1} \sum_{i=1}^n (\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})^2, \quad df_B = n - 1$$

- **Between-Raters Mean Square ($JMS$)**:
  $$JMS = \frac{n}{k - 1} \sum_{j=1}^k (\bar{y}_{\cdot j} - \bar{y}_{\cdot\cdot})^2, \quad df_J = k - 1$$

- **Residual / Error Mean Square ($EMS$)**:
  $$EMS = \frac{1}{(n - 1)(k - 1)} \sum_{i=1}^n \sum_{j=1}^k (y_{ij} - \bar{y}_{i\cdot} - \bar{y}_{\cdot j} + \bar{y}_{\cdot\cdot})^2, \quad df_E = (n - 1)(k - 1)$$

- **Within-Targets Mean Square ($WMS$)**:
  $$WMS = \frac{1}{n(k - 1)} \sum_{i=1}^n \sum_{j=1}^k (y_{ij} - \bar{y}_{i\cdot})^2 = \frac{SS_J + SS_E}{n(k - 1)}, \quad df_W = n(k - 1)$$

### Method-of-Moments Variance Components

PyAutoStat reports unconstrained ANOVA method-of-moments variance component estimates without artificial clamping to zero:

- **Target Variance**: $\sigma_T^2 = \frac{BMS - EMS}{k}$ (or $\frac{BMS - WMS}{k}$ for one-way random)
- **Rater Variance**: $\sigma_R^2 = \frac{JMS - EMS}{n}$
- **Residual Variance**: $\sigma_e^2 = EMS$
- **Total Variance (Absolute Agreement)**: $\sigma_{\text{total, agreement}}^2 = \sigma_T^2 + \sigma_R^2 + \sigma_e^2$
- **Total Variance (Consistency)**: $\sigma_{\text{total, consistency}}^2 = \sigma_T^2 + \sigma_e^2$

In finite samples, if $BMS < EMS$ or $JMS < EMS$, the unconstrained method-of-moments estimate of target variance or rater variance can be negative. PyAutoStat deliberately preserves these negative estimates rather than clamping them to zero, because a negative variance component estimate provides critical diagnostic evidence that within-target residual variability or rater variability exceeds between-target variability, indicating a potential violation of the additive random-effects model.


---

## 5. Mathematical Formulas for Estimates

### One-Way Random
$$\text{ICC}(1,1) = \frac{BMS - WMS}{BMS + (k - 1)WMS}$$
$$\text{ICC}(1,k) = \frac{BMS - WMS}{BMS}$$

### Two-Way Random / Absolute Agreement
$$\text{ICC}(2,1) = \frac{BMS - EMS}{BMS + (k - 1)EMS + \frac{k}{n}(JMS - EMS)}$$
$$\text{ICC}(2,k) = \frac{BMS - EMS}{BMS + \frac{JMS - EMS}{n}}$$

### Two-Way Mixed / Consistency
$$\text{ICC}(3,1) = \frac{BMS - EMS}{BMS + (k - 1)EMS}$$
$$\text{ICC}(3,k) = \frac{BMS - EMS}{BMS}$$

---

## 6. Analytical Confidence Intervals

Confidence intervals ($100(1 - \alpha)\%$) are derived by analytical F-inversion:

### Exact F-Inversion: ICC(1) and ICC(3)
Let $F = BMS / WMS$ for ICC(1) and $F = BMS / EMS$ for ICC(3). The exact $(1 - \alpha)$ CI is computed
by inverting the central F-distribution at percentiles $F_{\text{lower}} = F / F_{1 - \alpha/2}(\nu_1, \nu_2)$
and $F_{\text{upper}} = F \times F_{1 - \alpha/2}(\nu_2, \nu_1)$.

### Satterthwaite Effective Degrees of Freedom: ICC(2)
For ICC(2), the denominator is a linear combination of mean squares. Following McGraw and Wong
(1996), effective degrees of freedom are approximated using Satterthwaite's method:
$$\nu = \frac{(a \cdot JMS + b \cdot EMS)^2}{\frac{(a \cdot JMS)^2}{k - 1} + \frac{(b \cdot EMS)^2}{(n - 1)(k - 1)}}$$
PyAutoStat implements these exact analytical bounds without resorting to arbitrary approximations
or simulations.

---

## 7. Non-Clamping of Negative Estimates

In finite samples, it is possible for $BMS < EMS$ or $BMS < WMS$, yielding a negative ICC point
estimate.

**Scientific Safeguard**: PyAutoStat **never clamps** negative sample ICC estimates to zero.
A negative sample ICC is mathematically valid and carries essential diagnostic information:
it signals that within-target variance or rater variance exceeds between-target variance,
violating the assumption of positive target intraclass correlation. Clamping would artificially
truncate uncertainty and produce misleadingly optimistic reliability estimates.

---

## 8. Missing Data and Sample Accounting

ICC requires a balanced, fully crossed panel ($n \ge 2$ targets, $k \ge 2$ raters).
PyAutoStat applies **complete-target panel filtering**:
- Targets with any missing rating across the active rater set are excluded.
- Targets with valid ratings from all active raters are retained.
- The report explicitly discloses total input rows, complete targets retained, raters evaluated,
  and excluded incomplete targets.

---

## 9. Usage Examples

### Guided Workflow

```python
from pyautostat import ResearchAssistant

assistant = ResearchAssistant(ratings_df)
workflow = assistant.run(
    objective="reliability",
    estimand="intraclass_correlation",
    target="subject_id",
    rater="rater_id",
    outcome="score",
    model="two_way_random",
    definition="absolute_agreement",
    unit="single",
)

print(workflow.explain())
```

### Direct Analyzer API

```python
from pyautostat import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(ratings_df)
result = analyzer.intraclass_correlation(
    target="subject_id",
    rater="rater_id",
    value="score",
    model="two_way_random",
    definition="absolute_agreement",
    unit="single",
)

print("ICC(2,1):", result["estimate"])
print("95% CI:", result["confidence_interval"])
```

### All Variants at Once

```python
from pyautostat.icc import intraclass_correlation

result = intraclass_correlation(
    ratings_df,
    target="subject_id",
    rater="rater_id",
    outcome="score",
    include_all_variants=True,
)

for var_name, var_info in result["all_variants"].items():
    print(f"{var_info['notation']}: {var_info['estimate']:.4f} "
          f"[{var_info['confidence_interval']['lower']:.4f}, "
          f"{var_info['confidence_interval']['upper']:.4f}]")
```

---

## 10. References

1. Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in assessing rater reliability.
   *Psychological Bulletin*, 86(2), 420–428.
2. McGraw, K. O., & Wong, S. P. (1996). Forming inferences about some intraclass correlation
   coefficients. *Psychological Methods*, 1(1), 30–46.
3. Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation
   coefficients for reliability research. *Journal of Chiropractic Medicine*, 15(2), 155–163.
