# PyAutoStat

> **From pandas DataFrame to defensible statistical analysis — with the design, assumptions, effect sizes, uncertainty, interpretation, audit trail, and report kept together.**

[![CI](https://github.com/majikoushik/PyAutoStat/actions/workflows/ci.yml/badge.svg)](https://github.com/majikoushik/PyAutoStat/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange.svg)](#project-status)

**PyAutoStat** is an open-source statistical research assistant for Python and pandas.

It is built for researchers, analysts, students, educators, and scientific programmers who know
**what they want to investigate** but do not want every analysis to begin with manually deciding
which statistical procedure to run, reconstructing assumption checks, calculating effect sizes,
writing interpretation logic, and assembling a reproducible report.

PyAutoStat connects those pieces into one structured workflow:

```text
DataFrame
   ↓
Profile the data
   ↓
Declare the research question + study design
   ↓
Validate what is known — ask for what cannot be inferred safely
   ↓
Recommend a supported statistical method
   ↓
Run the analysis
   ↓
Effect size + confidence interval + diagnostics
   ↓
Deterministic interpretation
   ↓
Scientific consistency audit
   ↓
Reproducible research report
```

**The goal is not to replace statistical judgment. The goal is to make good statistical practice
easier to execute consistently.**

---

## Why PyAutoStat?

Most statistical Python libraries are excellent **calculation engines**. They assume you already
know the exact procedure, assumptions, estimand, contrast direction, missing-data policy, effect
size, post-hoc procedure, and reporting requirements you need.

PyAutoStat works one level above that.

Instead of starting with:

```python
# Which test should I import?
# Which assumptions matter?
# Welch or pooled t?
# Mean comparison or rank comparison?
# Which effect size?
# Which confidence interval?
# What should I report?
```

you can start with the scientific question:

```python
workflow = ResearchAssistant(df).run(
    objective="compare_groups",
    outcome="score",
    predictor="treatment",
    estimand="mean",
    design="independent",
    variable_types={"score": "continuous"},
)
```

PyAutoStat then keeps the **question, design, method, numerical result, uncertainty, assumptions,
interpretation, reporting, and audit metadata connected**.

That distinction is the core idea behind the library.

---

## Who is PyAutoStat for?

### PyAutoStat is a good fit if you are...

| You are... | PyAutoStat helps you... |
| --- | --- |
| **Researcher / scientist** | Move from a declared research question to a reproducible, reportable statistical workflow without rebuilding the analysis pipeline every time. |
| **Data analyst** | Add design-aware inferential statistics, effect sizes, diagnostics, and structured reporting around pandas workflows. |
| **Student learning applied statistics** | See not only the result, but why a method was selected, what it estimates, what assumptions matter, and what the result does *not* establish. |
| **Educator** | Demonstrate complete statistical workflows rather than isolated test functions. |
| **Python developer building research software** | Use structured, JSON-safe statistical results and session state as a foundation for notebooks, applications, or future GUIs. |
| **Statistician / advanced analyst** | Use direct APIs when the method is already known while retaining standardized result contracts, uncertainty, reporting, and audit infrastructure. |

### PyAutoStat is probably **not** the right tool if you need...

- a system that guesses study design, independence, pairing, clustering, randomization, or causal meaning from the numbers alone;
- mixed-effects models, mixed ANOVA, factorial repeated-measures ANOVA, GEE, survival analysis, broad generalized/regularized model families, or arbitrary model construction;
- automated causal inference;
- automated feature selection or p-value-driven model search;
- automatic deletion of outliers;
- automatic missing-value imputation;
- a machine-learning AutoML system for maximizing predictive accuracy;
- a black-box "upload data → publishable conclusion" button;
- a replacement for a statistician, domain expert, study-design review, or scientific judgment.

**PyAutoStat automates statistical workflow mechanics. It does not automate scientific responsibility.**

---

## Installation

PyAutoStat requires **Python 3.10 or newer**.

```bash
python -m pip install pyautostat
```

For optional interactive Plotly reporting:

```bash
python -m pip install "pyautostat[report]"
```

For development from the repository:

```bash
git clone https://github.com/majikoushik/PyAutoStat.git
cd PyAutoStat
python -m pip install -e .
```

---

## Quick start: profile a DataFrame

```python
import pandas as pd
from pyautostat import ResearchAssistant

df = pd.DataFrame(
    {
        "group": ["standard"] * 6 + ["new"] * 6,
        "score": [62, 65, 68, 70, 72, 75, 67, 71, 74, 78, 80, 84],
    }
)

assistant = ResearchAssistant(df)
profile = assistant.profile()
```

---

## Guided analysis

```python
result = assistant.run(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    estimand="mean",
    design="independent",
    variable_types={"score": "continuous"},
)

print(result.analysis.method_label)
print(result.interpretation.findings_plain)
print(result.audit.status)

html = result.report.to_html()
```

The important part is not merely that PyAutoStat can execute a t-test. It records **why that method
matches the declared question**, preserves the contrast direction and sample accounting, returns
effect and uncertainty information, generates qualified interpretation, and audits the structured
result for internal consistency.

---

## Beautiful terminal results

PyAutoStat includes a Rich-powered terminal presentation layer for inspecting dataset profiles, statistical workflows, and governance records directly in your console:

```python
from pyautostat import ResearchAssistant, show

assistant = ResearchAssistant(df)
show(assistant.profile())

# All supported analysis families are Rich-renderable
workflow = assistant.run(...)
show(workflow)

# Standalone results, descriptives, and planning
show(assistant.frequency_table("category"))
show(assistant.audit(workflow.report, result=workflow.analysis))
```

- **Universal coverage:** Renders dataset profiles, all 24 statistical method families (mean comparisons, ANOVA, associations, categorical tables, regression, reliability, repeated measures, and ICC), descriptive tables, and governance/planning results.
- **Clean visual hierarchy:** Panels, formatted metrics, and diagnostic tables without raw dictionary dumps.
- **Three detail levels:** `detail="compact"` (one-line summaries), `detail="standard"` (default), and `detail="full"` (complete diagnostics and metadata).
- **Presentation-only:** Consumes existing structured results without altering or recalculating any statistical values.
- **Terminal-aware:** Adapts to narrow/wide terminal widths and respects non-interactive environments and the `NO_COLOR` standard.

---

## Modern HTML and Research Reports

PyAutoStat also provides a self-contained, offline HTML presentation layer (`to_html`, `save_html`) and upgraded `ResearchReport` HTML exports:

```python
from pyautostat import ResearchAssistant, save_html, to_html

workflow = assistant.run(...)

# Generate standalone HTML string or save to file
html = to_html(workflow, detail="standard")
save_html(workflow, "reports/analysis.html", detail="full", overwrite=True)

# Upgraded canonical research report
report = workflow.report
report.save_html("reports/research_report.html", style="general", overwrite=True)
```

- **Unified presentation architecture:** Standalone results and canonical `ResearchReport` exports share the exact same component system, styling, and escaping rules.
- **Zero recalculation:** Consumes authoritative normalized presentation view models without re-running statistical algorithms.
- **Offline & self-contained:** Inline CSS with system fonts; no CDN dependencies, no external JavaScript, and no tracking.
- **Responsive & print-ready:** Responsive metric grids, overflow-wrapped semantic tables, and dedicated `@media print` stylesheets.
- **Security-hardened:** All user text, variable names, table cells, and group labels are safely HTML-escaped.
- **Universal coverage:** Complete static HTML rendering for all 24 statistical method families (mean comparisons, ANOVA, associations, categorical tables, regression, reliability, repeated measures, and ICC), dataset profiles, and planning/governance objects. See [`docs/HTML_REPORTING.md`](docs/HTML_REPORTING.md) for details.

---

## When information is missing

A DataFrame cannot tell you whether observations are truly independent, which measurements belong
to the same participant, what your scientific estimand is, or which category should represent an
event.

PyAutoStat does not pretend otherwise.

```python
pending = assistant.run(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    estimand="mean",
    variable_types={"score": "continuous"},
)

print(pending.status)  # needs_input
print(pending.missing_information)
```

Supply the missing design information and continue:

```python
revised = assistant.update_question(pending.draft, design="independent")
result = assistant.run(draft=revised)
```

This **ask rather than guess** behavior is a deliberate scientific safeguard.

---

## Core capabilities

| Area | Supported capabilities |
| --- | --- |
| **Data profiling** | Descriptive statistics, percentiles, categorical frequencies, cross-tabs, missingness, duplicates, distribution summaries, histograms, outlier review cues, correlations, advisory type/role suggestions, resource metadata |
| **Two independent groups** | Welch t, explicit Student t, Mann-Whitney U |
| **One sample** | One-sample t with signed difference, CI, Cohen's d and effect-size uncertainty |
| **Paired two-condition** | Paired t, Wilcoxon signed-rank, explicit unit-ID matching |
| **Independent 3+ groups** | Welch ANOVA + Games-Howell, classical ANOVA + Tukey-Kramer, Kruskal-Wallis + Dunn-Holm |
| **Two-factor independent designs** | Two-way factorial ANOVA, Type II/III SS, main effects, interaction, EMMs, simple effects, Holm-adjusted follow-ups |
| **Repeated 3+ conditions** | One-way repeated-measures ANOVA + sphericity/GG correction; Friedman + Kendall's W; complete pairwise follow-up |
| **Correlation** | Pearson, Spearman, Kendall tau-b, point-biserial, partial Pearson |
| **Categorical association** | Pearson chi-square, Fisher exact 2×2, exact McNemar |
| **Linear regression** | Simple/multiple OLS, categorical coding, classical/HC3 covariance, diagnostics, R² uncertainty |
| **Binary regression** | Logistic regression, odds ratios, Wald intervals, fit/convergence/design diagnostics |
| **Scale reliability** | Cronbach's alpha, bootstrap uncertainty, item-total diagnostics, alpha-if-deleted, explicit reverse scoring |
| **Rater reliability** | ICC(1,1), ICC(1,k), ICC(2,1), ICC(2,k), ICC(3,1), ICC(3,k) |
| **Research workflow** | Method recommendation, deterministic interpretation, practical significance, sensitivity analysis, analysis plans |
| **Reproducibility** | Fingerprints, replay metadata, decision ledger, session snapshots |
| **Reporting** | HTML, Markdown, JSON, CSV tables, safe LaTeX, optional Plotly HTML |
| **Scientific safeguards** | Method contracts, explicit missingness, effect-size uncertainty, semantic result audit, structured unsupported/unavailable states |

---

## More than p-values

PyAutoStat treats effect sizes and uncertainty as first-class results.

Depending on the method, supported quantities include:

- Cohen's d and Cohen's dz;
- rank-biserial correlation;
- Pearson r, Spearman rho and Kendall tau-b;
- Cramér's V;
- odds ratio;
- Kendall's W;
- eta-squared / partial eta-squared;
- rank epsilon-squared;
- R-squared;
- standardized regression coefficients.

Implemented uncertainty approaches include:

- noncentral-t inversion;
- noncentral-F inversion;
- Fisher-z transformation;
- analytical log-Wald intervals;
- deterministic percentile bootstrap;
- participant/block bootstrap;
- case-resampling bootstrap.

If a defensible interval is unavailable, PyAutoStat records that state instead of manufacturing a
number.

**Missing uncertainty is preferable to invented certainty.**

---

## Advanced workflows

PyAutoStat supports advanced study designs, regression modeling, scale reliability, and complete research governance.

### Two-way factorial ANOVA

For independent observations with two categorical factors and a continuous outcome:

```python
factorial = ResearchAssistant(df).run(
    objective="compare_groups",
    outcome="score",
    factor_a="teaching_method",
    factor_b="class_size",
    estimand="mean",
    design="independent",
    sum_of_squares="type2",  # or "type3"
)
```

PyAutoStat evaluates the full `A + B + A×B` model and supports balanced/unbalanced designs, Type II
and Type III sums of squares, partial eta-squared with noncentral-F intervals, cell summaries,
unweighted estimated marginal means, simple effects, marginal comparisons, 2×2
difference-of-differences contrasts, Holm multiplicity adjustment, and residual/design diagnostics.

---

## Repeated-measures analysis

For three or more conditions measured on the same units:

```python
repeated = ResearchAssistant(df).run(
    objective="compare_groups",
    outcome="score",
    predictor="visit",
    estimand="mean",
    design="repeated",
    unit_id="participant_id",
    condition_order=("baseline", "week4", "week8"),
    variable_types={"score": "continuous", "visit": "ordinal"},
)
```

For mean targets, PyAutoStat supports one-way repeated-measures ANOVA with Mauchly's sphericity
assessment, Greenhouse-Geisser correction where applicable, partial eta-squared and complete
Holm-adjusted paired-t follow-up.

For rank/distribution targets, it supports Friedman inference with Kendall's W and complete
Holm-adjusted paired Wilcoxon follow-up.

Complete and incomplete repeated panels remain visible in sample accounting.

---

## Linear and logistic regression

OLS regression supports:

- simple and multiple regression;
- continuous and categorical predictors;
- explicit reference coding;
- classical or HC3 covariance;
- coefficient intervals;
- R-squared and adjusted R-squared;
- bootstrap uncertainty for in-sample R-squared;
- standardized betas for continuous predictors;
- VIF, residual, heteroscedasticity, condition and influence diagnostics;
- rank-deficiency safeguards.

Binary logistic regression supports:

- explicit event orientation;
- coefficient and Wald inference;
- odds ratios with exponentiated Wald intervals;
- likelihood fit information;
- McFadden pseudo-R-squared;
- convergence, separation, VIF and condition safeguards.

Neither workflow performs automatic variable selection or converts association into a causal claim.

---

## Reliability: Cronbach's alpha and ICC

### Scale reliability

```python
reliability = ResearchAssistant(df).reliability(
    items=["q1", "q2", "q3", "q4"],
)
```

The workflow reports Cronbach's alpha, bootstrap uncertainty, corrected item-total correlations,
alpha-if-item-deleted, inter-item diagnostics, missingness and optional explicit reverse scoring.

It does not automatically discover scales, remove items, reverse-score items, or treat alpha as
proof of validity.

### Inter-rater / test-retest reliability

```python
icc = ResearchAssistant(df).intraclass_correlation(
    target="subject_id",
    rater="rater_id",
    value="score",
    model="two_way_random",
    definition="absolute_agreement",
    unit="single",
)
```

PyAutoStat supports the six canonical ICC configurations and requires model, agreement/consistency,
and single/average meaning to be explicit. Results include ANOVA mean-square decomposition,
method-of-moments variance components, analytical uncertainty, applicable F tests, and complete
target accounting.

Negative finite-sample ICC and unconstrained variance-component estimates are preserved rather
than silently clamped to zero.

---

## Explainable method recommendation

The recommendation layer considers the declared research objective, outcome, predictor/factors,
variable types, number of groups/conditions, independent/paired/repeated design, unit identity,
estimand, requested association/model target, and data feasibility.

This is intentionally different from simplistic rules such as:

```text
normal → t-test
not normal → Mann-Whitney
```

A diagnostic result does not silently redefine the scientific question.

---

## Deterministic interpretation — no LLM required

PyAutoStat can create qualified human-readable findings from stored statistical results without
calling a generative AI service.

Interpretation can incorporate direction, statistical decision, effect magnitude, uncertainty,
sample context, assumptions, limitations, practical significance and sensitivity information.

The structured numerical result remains authoritative.

---

## Practical significance

Researchers can declare what magnitude would be scientifically meaningful:

```python
from pyautostat import MeaningfulEffectThreshold

threshold = MeaningfulEffectThreshold(
    quantity="mean_difference",
    minimum_magnitude=5,
    direction="two_sided",
    unit="points",
)

practical = assistant.practical_significance(
    result.analysis,
    threshold=threshold,
)
```

Statistical significance and practical importance remain separate. PyAutoStat does not invent a
universal threshold for "important."

---

## Sensitivity analysis

Researchers can declare alternative defensible scenarios and retain every attempted result.
PyAutoStat records estimand and contrast comparability instead of blindly comparing p-values across
scientifically different questions.

It does not search across methods for the smallest p-value.

---

## Prospective study planning

`StudyPlanner` supports prospective power and confidence-interval precision planning for selected
independent and paired mean designs.

```python
from pyautostat import StudyPlanner

plan = StudyPlanner().independent_mean_power(
    target_difference=5,
    sd_group1=10,
    sd_group2=12,
    target_power=0.80,
)
```

Planning inputs are researcher-supplied assumptions. Observed/post-hoc power is intentionally not
reported.

---

## Analysis plans and reproducibility

PyAutoStat can keep analytical decisions connected to results through:

- serializable statistical analysis plans;
- plan-adherence comparison;
- decision ledgers;
- dataset/content fingerprints;
- reproducibility metadata;
- explicit supplied-data replay;
- session snapshots;
- deterministic bootstrap random-number handling.

These features support reproducibility but do not claim to authenticate source data or replace a
formal preregistration system.

---

## Scientific result auditing

PyAutoStat can check its structured results for internal consistency, including p-value bounds,
confidence-interval ordering, effect-size bounds, sample accounting, degrees-of-freedom identities,
multiplicity invariants, regression coefficient/odds-ratio consistency, ANOVA identities and ICC
model consistency.

An audit pass means the stored result is internally coherent. It does **not** prove that the study
design, source data or scientific conclusion is correct.

---

## Reporting and export

Supported outputs include:

- HTML;
- Markdown;
- JSON;
- CSV tables;
- safe LaTeX;
- optional interactive Plotly HTML.

Presentation styles include General, APA-oriented and IEEE-oriented layouts.

Reporting-completeness checks identify whether expected implemented elements are represented. They
are not study-quality or publication-readiness scores.

---

## Safe by design

PyAutoStat includes safeguards such as:

- HTML and LaTeX escaping;
- CSV formula protection;
- strict JSON-safe serialization;
- no automatic data upload;
- no automatic file writing from the guided workflow;
- reports that avoid embedding raw DataFrames and participant identifier values;
- explicit save operations;
- profiling that does not silently modify the source DataFrame.

Researchers should still review aggregate outputs for disclosure risk, especially small cells.

---

## Scientific contracts, not just functions

Supported inferential methods are designed around explicit contracts:

```text
scientific question
→ study design
→ estimand
→ null / alternative
→ primary estimate
→ effect size
→ uncertainty
→ assumptions
→ diagnostics
→ missing-data policy
→ degenerate-data behavior
→ multiplicity policy
→ numerical implementation
→ interpretation limits
→ audit invariants
→ validation evidence
```

This keeps the **meaning of the result** connected to the computation.

See:

- [`docs/STATISTICAL_METHOD_CONTRACTS.md`](docs/STATISTICAL_METHOD_CONTRACTS.md)
- [`docs/STATISTICAL_VALIDATION.md`](docs/STATISTICAL_VALIDATION.md)
- [`docs/SCIENTIFIC_LIMITATIONS.md`](docs/SCIENTIFIC_LIMITATIONS.md)
- [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md)

---

## Direct API for experienced users

`ResearchAssistant` is the recommended entry point when you want the full design → recommendation →
analysis → interpretation → report workflow.

Experienced users can use `StatisticalAnalyzer` when the required procedure is already known:

```python
from pyautostat import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(df)
```

The direct API provides control without changing the scientific contract of the method.

---

## Design principles

**Ask rather than guess.** Essential design facts that cannot be inferred safely are requested.

**Preserve the estimand.** Diagnostics do not silently turn a mean question into a rank question.

**Never hide missingness.** Analysis-specific exclusions, incomplete pairs and incomplete panels
remain visible.

**Never delete outliers automatically.** Outliers are review cues, not automatic deletion rules.

**Never manufacture uncertainty.** Unsupported or undefined intervals remain explicitly unavailable.

**Separate statistical significance from practical importance.** A small p-value does not define a
meaningful effect.

**Keep narration subordinate to structured results.** Explanations do not alter the numerical result.

**Prefer bounded scope over false comprehensiveness.** Unsupported designs remain unsupported until
they can be implemented with a defensible scientific contract.

---

## Current scope and important limitations

PyAutoStat does **not currently implement**:

- mixed-effects models;
- mixed ANOVA;
- factorial repeated-measures ANOVA;
- generalized estimating equations (GEE);
- survival analysis;
- broad multinomial, ordinal, count or regularized regression families;
- arbitrary model construction outside documented workflows;
- causal-inference frameworks;
- automatic imputation;
- automatic feature/model selection;
- automatic outlier removal;
- broad exact-table methods beyond supported Fisher 2×2 inference;
- formal equivalence/noninferiority workflows;
- automatic observed/post-hoc power.

See [`docs/SCIENTIFIC_LIMITATIONS.md`](docs/SCIENTIFIC_LIMITATIONS.md) before using PyAutoStat for
high-stakes or publication-critical analyses.

---

## Project status

PyAutoStat is currently an **alpha** project.

The development pipeline includes supported-version/platform CI, scientific regression tests,
static analysis, package builds, minimum numerical-stack compatibility checks, and installed-wheel
smoke testing. Alpha status means the public API and capability set may continue to evolve.

For clinical, regulatory, legal, safety-critical or other high-stakes decisions, independent expert
statistical review is strongly recommended.

---

## Compatibility

- **Python:** 3.10, 3.11, 3.12, 3.13
- **Primary data interface:** pandas DataFrames
- **CI platforms:** Ubuntu and Windows

Core numerical dependencies include pandas, NumPy, SciPy and statsmodels. See
[`pyproject.toml`](pyproject.toml) for authoritative dependency constraints.

---

## Documentation

| Resource | Purpose |
| --- | --- |
| [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md) | Exact supported analysis families and boundaries |
| [`API_REFERENCE.md`](API_REFERENCE.md) | Public API reference |
| [`docs/STATISTICAL_METHOD_CONTRACTS.md`](docs/STATISTICAL_METHOD_CONTRACTS.md) | Scientific contracts |
| [`docs/STATISTICAL_VALIDATION.md`](docs/STATISTICAL_VALIDATION.md) | Numerical/statistical validation policy |
| [`docs/SCIENTIFIC_LIMITATIONS.md`](docs/SCIENTIFIC_LIMITATIONS.md) | Scientific boundaries |
| [`docs/ICC_GUIDE.md`](docs/ICC_GUIDE.md) | ICC model and interpretation guide |
| [`ROADMAP.md`](ROADMAP.md) | Forward-looking priorities |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |

---

## See PyAutoStat on a realistic customer dataset

The repository ships a realistic 5,000-row customer demonstration dataset ([`examples/CustomerDataset.csv`](examples/CustomerDataset.csv)) and a complete customer-analytics example gallery.

Every public example uses this same bundled dataset — no external downloads or network calls are required. Customer IDs are never displayed, and observational associations are never presented as causal claims.

Try three representative workflows:

```bash
# 1. Customer 360 profiling and data understanding
python examples/01_customer_360_profile.py

# 2. Within-customer repeated-measures product portfolio analysis
python examples/07_product_portfolio_repeated_measures.py

# 3. Complete research lifecycle: clarification, planning, sensitivity, reporting, audit, and replay
python examples/09_complete_research_workflow.py --output-dir reports
```

Run the entire suite in one command:

```bash
python examples/run_all.py
# Or run in fast CI mode:
python examples/run_all.py --fast
```

Explore the full gallery and business scenarios in the [Examples Guide](examples/README.md).

---

## Quality and validation

The development pipeline includes:

- automated tests and coverage enforcement;
- Ruff linting and formatting checks;
- mypy type checking;
- Python 3.10–3.13 CI;
- Ubuntu and Windows CI;
- minimum numerical-stack compatibility checks;
- source-distribution and wheel builds;
- distribution metadata checks;
- isolated installed-wheel smoke tests;
- statistical regression and edge-case tests;
- semantic audit corruption tests.

Passing software tests does not by itself establish scientific truth. Numerical validation,
scientific contracts, explicit limitations and researcher judgment remain separate requirements.

---

## Contributing

Contributions are welcome, particularly for statistical validation, independent reference datasets,
numerical edge cases, scientific documentation, usability, reproducibility, reporting, performance
benchmarks and accessibility.

For a new statistical method, implementation alone is not enough. The contribution should support:

```text
design
→ estimand
→ validation
→ execution
→ uncertainty
→ interpretation
→ report
→ audit
→ reproducibility
```

Please open an issue before beginning a substantial new statistical capability so its scientific
contract and scope can be discussed first.

---

## Citation

If you use PyAutoStat in academic work, record the exact package version and analysis environment.

A formal project citation file may be added as the project matures. Until then, cite at minimum:

- PyAutoStat;
- the exact package version;
- the project repository;
- the version/date of the analysis environment.

---

## License

PyAutoStat is released under the **MIT License**. See [`LICENSE`](LICENSE).

---

## A final note

Statistical software can calculate an answer. Good research requires knowing **which question that
answer belongs to**.

PyAutoStat is built around that distinction.

If you want a Python library that helps you move from a pandas DataFrame to a **structured,
explainable, auditable and reproducible statistical workflow** — while refusing to guess the
scientific facts only you can provide — PyAutoStat is designed for you.
