# PyAutoStat Customer Analytics Examples

> **One dataset. Multiple research questions. One traceable statistical workflow.**

These tutorials demonstrate how PyAutoStat guides a complete customer-analytics
investigation using only the bundled `CustomerDataset.csv` (5,000 customers and
40 variables). Rather than presenting isolated mathematical functions over toy synthetic
arrays, this gallery addresses real business questions while adhering strictly to scientific
guardrails.

Individual customer identifiers (`customer_id`) are never printed or disclosed.

---

## Start here

Install PyAutoStat:

```bash
python -m pip install -e .
```

Run the introductory profiling and hero research workflow:

```bash
python examples/01_customer_360_profile.py
python examples/09_complete_research_workflow.py --output-dir reports
```

Or execute the complete suite in one command:

```bash
python examples/run_all.py
```

---

## What you will learn

- **Data understanding first**: Turn a raw DataFrame into structured distribution summaries, percentile views, and data-quality diagnostics before hypothesis testing.
- **Ask rather than guess**: When study design, pairing, or estimands are missing, PyAutoStat pauses with structured clarification requests instead of guessing from values.
- **Keep estimands stable**: Normality and variance diagnostics are reported openly, but a diagnostic p-value never silently changes a mean question into a rank test.
- **Effect sizes and uncertainty**: Every inferential result provides effect estimates (Cohen's d, Cramer's V, partial eta-squared, odds ratios, rank-biserial correlations) and confidence intervals.
- **Prevent circular leakage**: Structurally derived targets (such as high-value flags based on spend) are protected against circular predictor inclusion.
- **End-to-end scientific governance**: Statistical analysis planning, same-estimand vs different-estimand sensitivity scenarios, researcher-defined practical thresholds, report generation, consistency audits, and supplied-data replay.

---

## The demonstration dataset

All public examples use the bundled:

```text
examples/CustomerDataset.csv
```

- **Scope**: 5,000 customer records across 40 demographic, behavioral, and spending variables.
- **Authoritative shared loader**: [`_customer_data.py`](_customer_data.py) standardizes snake_case names and currency strings without modifying the raw CSV.
- **Demonstration data disclaimer**: This dataset is provided solely for demonstration. The repository does not establish population representativeness, random sampling, or causal identification.
- **Privacy preservation**: Individual `customer_id` strings are used internally only for within-unit paired matching and are never exposed in console output or generated reports.

### Key dataset structural facts
- **Spend balance**: `total_avg_monthly_spend` is exactly equal to `monthly_spend_product_a + monthly_spend_product_b + monthly_spend_product_c`.
- **Zero inflation**: Product B monthly spend is nonzero exactly for customers with `streaming_services == "Yes"` (65.9% zero spend); Product C monthly spend is nonzero exactly for customers with `wireless_internet == "Yes"` (73.1% zero spend).
- **Segment separation**: `high_value_customer` is deterministically separated around a total monthly spend of approximately $275 (max standard customer: $274.95; min high-value customer: $275.20).

---

## Example gallery

| # | Business question | PyAutoStat capability | What the example proves | Command |
|---|---|---|---|---|
| **01** | *"What does this customer base look like, and what should we investigate first?"* | Data understanding & profiling | One call turns raw data into percentiles, skewness cues, and prioritized screening insights. Outliers are review cues, not deletion orders. | `python examples/01_customer_360_profile.py` |
| **02** | *"Do news subscribers and non-subscribers differ in average total monthly spend?"* | Two independent groups (Welch t) | Preserves the declared mean estimand without assuming equal variances. Reports Cohen's d with bootstrap CI alongside the p-value. | `python examples/02_compare_customer_segments.py` |
| **03** | *"Does average monthly spending differ across customer job categories?"* | Multi-group ANOVA (Welch + Games-Howell) | Handles omnibus mean testing across 6 groups and delivers all 15 multiplicity-controlled pairwise follow-up contrasts with simultaneous CIs. | `python examples/03_multigroup_customer_spending.py` |
| **04** | *"Is home-ownership status associated with high-value-customer segment membership?"* | Categorical association (Chi-Square & Cramer's V) | 2x2 categorical independence without spend leakage. Checks expected cell counts, reports Cramer's V effect size, and verifies Fisher fallback status. | `python examples/04_customer_value_association.py` |
| **05** | *"Which non-product customer characteristics are conditionally associated with total spend?"* | Multiple OLS regression (HC3 covariance) | Excludes spend components to prevent circularity. Uses HC3 robust covariance for heteroscedastic residuals; reports standardized betas and VIF. | `python examples/05_spend_drivers_regression.py` |
| **06** | *"Which non-spend customer characteristics are associated with the high-value segment?"* | Binary logistic regression | Programmatically verifies that high-value status is derived from spend, excluding spend fields to prevent circularity. Reports odds ratios and McFadden pseudo-R2. | `python examples/06_high_value_customer_logistic.py` |
| **07** | *"How do monthly spending distributions differ across Products A, B, and C within customers?"* | Repeated-measures distribution (Friedman & Wilcoxon) | Reshapes within-unit product spending into long format. Handles heavy zero-inflation cleanly using rank distributions, Kendall's W, and Holm-adjusted Wilcoxon pairs. | `python examples/07_product_portfolio_repeated_measures.py` |
| **08** | *"How are two customer-segmentation factors jointly associated with average monthly spend?"* | Two-way factorial ANOVA (Type II SS) | Evaluates home ownership and news subscription main effects and interaction. Exact partial eta-squared CIs and unweighted EMMs demonstrate additive structure. | `python examples/08_factorial_customer_segments.py` |
| **09** | *"End-to-end customer spending research lifecycle and scientific governance"* | Full research assistant lifecycle | The flagship hero workflow: structured clarification (`needs_input`), analysis planning, sensitivity scenarios, practical thresholds, reports, audit, and replay. | `python examples/09_complete_research_workflow.py` |

---

## Recommended learning paths

- **New to PyAutoStat**:
  [`01_customer_360_profile.py`](01_customer_360_profile.py) &rarr;
  [`02_compare_customer_segments.py`](02_compare_customer_segments.py) &rarr;
  [`09_complete_research_workflow.py`](09_complete_research_workflow.py)
- **Business / Marketing Analyst**:
  [`01_customer_360_profile.py`](01_customer_360_profile.py) &rarr;
  [`03_multigroup_customer_spending.py`](03_multigroup_customer_spending.py) &rarr;
  [`05_spend_drivers_regression.py`](05_spend_drivers_regression.py) &rarr;
  [`06_high_value_customer_logistic.py`](06_high_value_customer_logistic.py) &rarr;
  [`09_complete_research_workflow.py`](09_complete_research_workflow.py)
- **Statistical Researcher**:
  [`02_compare_customer_segments.py`](02_compare_customer_segments.py) &rarr;
  [`03_multigroup_customer_spending.py`](03_multigroup_customer_spending.py) &rarr;
  [`07_product_portfolio_repeated_measures.py`](07_product_portfolio_repeated_measures.py) &rarr;
  [`08_factorial_customer_segments.py`](08_factorial_customer_segments.py) &rarr;
  [`09_complete_research_workflow.py`](09_complete_research_workflow.py)
- **Developer / Platform Integrator**:
  [`01_customer_360_profile.py`](01_customer_360_profile.py) &rarr;
  [`05_spend_drivers_regression.py`](05_spend_drivers_regression.py) &rarr;
  [`09_complete_research_workflow.py`](09_complete_research_workflow.py)

---

## One-command run

Run all public examples in order through isolated subprocesses:

```bash
python examples/run_all.py
```

For fast CI execution:

```bash
python examples/run_all.py --fast
```

---

## What the examples deliberately do not claim

- **Representativeness**: The customer dataset is demonstration data; it does not claim to represent any national or commercial population.
- **Causation**: Every association (such as news subscription or home ownership with spending) is strictly observational and conditional; no causal intervention is established.
- **Automatic scientific truth**: Diagnostic tests and method recommendations assist analysis; they do not replace study-design validity or substantive expertise.
- **Out-of-sample prediction**: Regression R-squared (23.4%) and McFadden pseudo-R2 (14.6%) summarize in-sample fit; they are not machine learning benchmark scores.
- **Universal practical thresholds**: The $25/month threshold demonstrated in Example 09 is an illustrative tutorial input declared by the researcher, not a discovered economic constant.

---

## Why some PyAutoStat methods are not forced into this dataset

PyAutoStat includes extensive support for psychometric scale reliability (Cronbach's alpha with bootstrap CIs and item diagnostics) and rater reliability (Intraclass Correlation Coefficients ICC(1,1) through ICC(3,k)).

These methods are **intentionally not demonstrated with CustomerDataset.csv**:

1. **Cronbach's alpha** requires a researcher-declared set of items constructed to measure a shared reflective psychological construct. Converting arbitrary lifestyle or social media checkboxes into a pseudo-scale would encourage poor psychometric practice.
2. **ICC** requires a genuine target-by-rater measurement panel under a specified random or mixed rater model. Product spend streams are distinct revenue lines, not exchangeable raters evaluating a subject.

Rather than compromising scientific integrity for artificial feature coverage, PyAutoStat demonstrates reliability methods using dedicated test suites and documentation:
- See [`docs/ICC_GUIDE.md`](../docs/ICC_GUIDE.md) for rater reliability workflows.
- See the [API reference](../API_REFERENCE.md) and [`docs/CAPABILITIES.md`](../docs/CAPABILITIES.md) for `ResearchAssistant.reliability()` contracts.

---

## Use your own DataFrame

The same workflow applies to any pandas DataFrame:

```python
import pandas as pd
from pyautostat import ResearchAssistant

df = pd.read_csv("your_data.csv")
assistant = ResearchAssistant(df)

# 1. Profile and inspect data quality
print(assistant.summarize())

# 2. Run a design-aware research workflow
workflow = assistant.run(
    objective="compare_groups",
    outcome="revenue",
    predictor="campaign_tier",
    estimand="mean",
    design="independent",
    variable_types={"revenue": "continuous", "campaign_tier": "nominal"},
)

# 3. Print deterministic explanation and export canonical report
print(workflow.explain())
report = assistant.report(workflow.analysis)
report.save_html("research_report.html", style="apa")
```

For complete API signatures and configuration options, see the [API reference](../API_REFERENCE.md).
