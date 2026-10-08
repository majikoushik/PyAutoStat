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

For most analyses, start with `ResearchAssistant`.

---

## What PyAutoStat does

Most statistical Python libraries are excellent **calculation engines**. They assume you already
know the exact procedure, assumptions, estimand, contrast direction, missing-data policy, effect
size, post-hoc procedure, and reporting requirements you need.

PyAutoStat works one level above that. Instead of hand-assembling an analysis pipeline:

1. **Profile the data**: Discover distributions, missingness, and data quality cues without modifying the source DataFrame.
2. **Declare the scientific question**: Specify objective, outcome, predictor, estimand, and study design.
3. **Ask rather than guess**: If an essential design fact (such as pairing, clustering, or independence) cannot be inferred from the data values alone, PyAutoStat pauses with a structured clarification request rather than guessing.
4. **Recommend defensible methods**: Selects methods based on declared estimands and designs—never silently switching a mean question to a rank test based on a diagnostic p-value.
5. **Execute and quantify**: Returns primary estimates, effect sizes, and supported confidence intervals.
6. **Interpret deterministically**: Produces qualified, rule-based findings without requiring a generative AI model or cloud service.
7. **Audit for consistency**: Verifies internal coherence across degrees of freedom, sample accounting, and intervals.
8. **Export reproducible reports**: Outputs self-contained terminal views, HTML, PDF, Word (DOCX), or multi-format research bundles.

**The goal is not to replace statistical judgment. The goal is to make good statistical practice
easier to execute consistently.**

---

## Installation

PyAutoStat requires **Python 3.10 or newer**.

Install the core package:

```bash
pip install pyautostat
```

### Optional extras

PyAutoStat provides optional extras for extended reporting and visualization (configured in `pyproject.toml`):

| Need | Installation | Notes |
| --- | --- | --- |
| **Core statistics & terminal** | `pip install pyautostat` | Full profiling, all 24 registered statistical methods, Rich terminal presentation, static HTML, and Markdown |
| **Interactive Plotly reports** | `pip install "pyautostat[report]"` | Adds interactive Plotly charts (forest plots, CIs) to HTML exports |
| **Publication-oriented PDF** | `pip install "pyautostat[pdf]"`<br>`python -m playwright install chromium` | Generates high-fidelity PDF documents via headless Chromium |
| **Figure-enabled PDF** | `pip install "pyautostat[report,pdf]"`<br>`python -m playwright install chromium` | Combines interactive figures with headless Chromium PDF generation |
| **Editable Word (.docx)** | `pip install "pyautostat[docx]"` | Generates native Microsoft Word documents with styled tables and captions |
| **Static scientific figures** | `pip install "pyautostat[figures]"`<br>`plotly_get_chrome -y` | Exports standalone publication-quality PNG, SVG, or vector figure-PDF files |

For local development from the repository:

```bash
git clone https://github.com/majikoushik/PyAutoStat.git
cd PyAutoStat
pip install -e .
```

---

## Quick start: profile a DataFrame

Start by importing `ResearchAssistant` and `show`, then profile your dataset:

```python
import pandas as pd
from pyautostat import ResearchAssistant, show

df = pd.DataFrame(
    {
        "group": ["standard"] * 6 + ["new"] * 6,
        "score": [62, 65, 68, 70, 72, 75, 67, 71, 74, 78, 80, 84],
    }
)

assistant = ResearchAssistant(df)
profile = assistant.profile()
show(profile)
```

`assistant.profile()` returns descriptive statistics, percentiles, distribution summaries, and
data-quality checks without sampling or mutating the source DataFrame. `show(profile)` renders a
polished summary directly in your terminal.

---

## Guided analysis

Once you understand your data, run a design-aware research workflow:

For common mean comparisons and correlations, the beginner conveniences are
`assistant.compare_means(...)` and `assistant.correlate(..., method=...)`. They delegate to the
same guided workflow shown below.

```python
workflow = assistant.compare_means("score", by="group")
print(workflow.brief())
```

`workflow.brief()` gives a concise, single-paragraph result from the stored analysis. Use
`workflow.apa_statement()` for a reporting sentence, `workflow.explain()` for the detailed
plain-text explanation, `workflow.type_guidance()` to inspect declared versus advisory variable
types, or `show(workflow)` for the full structured presentation. Type suggestions are never
scientific confirmations, and typo suggestions in validation errors are never applied silently.

```python
workflow = assistant.run(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    estimand="mean",
    design="independent",
    variable_types={"score": "continuous"},
)

show(workflow)
```

Save a self-contained HTML research report:

```python
from pyautostat import save_html

save_html(workflow, "reports/analysis.html")
```

Every result provides direct ergonomic access to underlying values, structured primary views,
and deterministic APA-oriented statistical statements without recalculating statistics:

```python
result = workflow.analysis

# Direct ergonomic accessors
print(result.test_name)  # "Welch independent-samples t-test"
print(result.statistic)  # 2.81
print(result.p_value)  # 0.008
print(result.effect_size)  # {"name": "cohens_d", "value": 0.84}

# Concise primary result dictionary or deterministic statement
print(result.primary_result())
print(result.apa_statement())
# "The first group had a higher mean than the second, Welch's t(37.4) = 2.81, p = .008, Cohen's d = 0.84, 95% CI for the mean difference [1.12, 6.88]."
```

For multi-test designs (such as factorial ANOVA or multiple regression), scalar `statistic` and
`p_value` safely return `None` to prevent ambiguous single-number extraction; use `primary_result()`
or structured statement generators instead. See [Result Ergonomics and Statements](docs/RESULT_ERGONOMICS_AND_STATEMENTS.md).

The important part is not merely that PyAutoStat can execute a t-test. It records **why that method
matches the declared question**, preserves the contrast direction and sample accounting, returns
effect and uncertainty information, generates qualified interpretation, and audits the structured
result for internal consistency.

---

## When information is missing

A DataFrame cannot tell you whether observations are truly independent, which measurements belong
to the same participant, what your scientific estimand is, or which category should represent an
event.

PyAutoStat does not pretend otherwise:

```python
pending = assistant.run(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    estimand="mean",
    variable_types={"score": "continuous"},
)

print(pending.status.value)                               # "needs_input"
print([item.field for item in pending.missing_information])  # ["design"]
```

Supply the missing design information and continue:

```python
revised = assistant.update_question(pending.draft, design="independent")
workflow = assistant.run(draft=revised)
show(workflow)
```

This **ask rather than guess** behavior is a deliberate scientific safeguard.

### Understanding workflow status: Analysis success vs. workflow completeness

`workflow.status` describes the completeness of the full research workflow, not only whether the statistical calculation succeeded:

- **`completed`**: Statistical calculation ran successfully, and downstream interpretation, reporting, and audit verification completed.
- **`partial`**: The statistical calculation was executed and remains available via `workflow.analysis`, but an optional downstream step was intentionally omitted (e.g., `audit=False`) or incomplete.
- **`needs_input`**: Analysis did not run because required design information (such as independence, pairing, or rater model) is missing.
- **`data_limited`**: Analysis did not run because observed data violated basic mathematical requirements (such as constant values or zero variance).
- **`unsupported`**: Analysis did not run because the declared design combination has no valid implementation; no substitute method was run.
- **`failed`**: The workflow stopped due to an internal error or unresolved audit contradiction.

For example, when report auditing is explicitly disabled:

```python
workflow = assistant.run(..., audit=False)

assert workflow.analysis is not None  # Analysis succeeded
assert workflow.status.value == "partial"  # Workflow is partial because audit was skipped
```

The computed statistical result is unchanged and remains available; the overall workflow is partial because post-analysis audit verification was not performed.

---

## Choose your task

For most analyses, start with `ResearchAssistant`. Use this guide to select the right entry point for your goal:

| I want to... | Recommended entry point | What it evaluates |
| --- | --- | --- |
| **Understand my dataset** | `assistant.profile()` | Summary statistics, percentiles, missingness, and data quality cues |
| **Inspect categorical distributions** | `assistant.frequency_table("col")`<br>`assistant.cross_tab("row", "col")` | Frequency tables and two-way cross-tabulations with percentage breakdowns |
| **Compare two independent groups** | `assistant.run(objective="compare_groups", ...)` | Welch t-test (guided default for mean) or Mann-Whitney U (distribution target) |
| **Compare three or more groups** | `assistant.run(objective="compare_groups", ...)` | Welch ANOVA + Games-Howell (guided default) or Kruskal-Wallis + Dunn-Holm |
| **Compare paired observations** | `assistant.run(design="paired", ...)` | Paired t-test (mean) or Wilcoxon signed-rank (rank target) with explicit `unit_id` |
| **Compare repeated measures (3+)** | `assistant.run(design="repeated", ...)` | One-way RM-ANOVA with Mauchly sphericity check or Friedman rank test |
| **Analyze factorial experiments** | `assistant.two_way_anova(...)` | Two-way ANOVA with Type II/III sums of squares, simple effects, and marginal means |
| **Study relationships / correlations** | `assistant.run(objective="association", ...)` | Pearson, Spearman, Kendall tau-b, point-biserial, partial Pearson, or Chi-square |
| **Fit linear or logistic regression** | `assistant.run(objective="regression", ...)` | OLS with classical or HC3 robust errors, or binary logistic regression with odds ratios |
| **Assess scale reliability** | `assistant.reliability(items=[...])` | Cronbach's alpha with bootstrap CI, item-total correlations, and alpha-if-deleted |
| **Assess rater agreement** | `assistant.intraclass_correlation(...)` | 6 canonical ICC configurations with ANOVA variance decomposition and F-tests |
| **Inspect results in terminal** | `show(target)` | Polished Rich console presentation for workflows, profiles, and audit records |
| **Export research reports** | `save_html(...)`, `save_pdf(...)`, `save_docx(...)` | Writes self-contained, publication-oriented reports to disk |

---

## Reporting and export

PyAutoStat separates viewing results from saving files through a clear conceptual distinction:

- **`show(...)`**: Renders a polished Rich terminal presentation directly in your console.
- **`save_*`**: Writes a standalone report file to disk (`save_html`, `save_pdf`, `save_docx`, `save_bundle`).
- **`to_*`**: Returns an in-memory representation or byte stream (`to_html`, `to_pdf`, `to_docx`, `to_bundle`).

### Terminal inspection

```python
from pyautostat import show

# Inspect dataset profiles
show(assistant.profile())

# Inspect workflow results at standard, compact, or full detail
show(workflow)
show(workflow, detail="compact")
show(workflow, detail="full")
```

### Common file outputs: HTML, PDF, and DOCX

```python
from pyautostat import save_html, save_pdf, save_docx

# Static self-contained HTML report (offline, no CDN dependencies)
save_html(workflow, "reports/analysis.html", detail="standard")

# Publication-oriented PDF (requires [pdf] extra and Playwright Chromium)
save_pdf(workflow, "reports/analysis.pdf", detail="full")

# Editable Microsoft Word document (requires [docx] extra)
save_docx(workflow, "reports/analysis.docx", detail="full")
```

The canonical `ResearchReport` object also exposes matching save methods:

```python
report = workflow.report
report.save_html("reports/research_report.html", style="apa")
report.save_pdf("reports/research_report.pdf", style="general")
report.save_docx("reports/research_report.docx", style="ieee")
```

### Advanced presentation and export formats

For specialized and developer workflows, PyAutoStat provides advanced presentation utilities:

- **In-memory representations (`to_*`)**: `to_html(workflow)` returns an HTML string; `to_pdf(workflow)` and `to_docx(workflow)` return raw bytes for web applications or streaming pipelines.
- **Interactive Plotly HTML (`save_interactive_html`)**: Adds interactive forest plots, CI charts, and contingency heatmaps (requires `[report]` extra).
- **Static scientific figures (`save_static_figures`, `to_static_figures`)**: Generates publication-quality standalone vector (SVG, PDF) or high-DPI raster (PNG) figures from authoritative models (requires `[figures]` extra).
- **Reproducible research bundles (`save_bundle`, `verify_bundle`)**: Packages reports, machine-readable JSON results, tabular CSVs, and cryptographic SHA-256 manifests into an integrity-verifiable ZIP archive.

For details, see the dedicated presentation guides:
- [Terminal Presentation Guide](docs/TERMINAL_PRESENTATION.md)
- [HTML Reporting Guide](docs/HTML_REPORTING.md)
- [PDF Reporting Guide](docs/PDF_REPORTING.md)
- [DOCX Reporting Guide](docs/DOCX_REPORTING.md)
- [Static Figures Guide](docs/STATIC_FIGURES.md)
- [Research Bundles Guide](docs/RESEARCH_BUNDLES.md)

---

## Supported analysis families

PyAutoStat supports 24 inferential method families. Where multiple procedures exist, PyAutoStat
distinguishes **guided defaults** (recommended and selected automatically when data criteria are
met) from **explicitly supported alternatives** (available when specified by the researcher):

| Objective | Target & Design | Guided default | Explicitly supported alternative | Effect size & uncertainty |
| --- | --- | --- | --- | --- |
| **Two independent groups** | Continuous outcome (mean) | **Welch t-test** (robust to unequal variances) | **Student's pooled t-test** (requires equal-variance justification) | Cohen's d with noncentral-t CI |
| **Two independent groups** | Rank / distribution target | **Mann-Whitney U** (`estimand="distribution"`) | — | Rank-biserial correlation with bootstrap CI |
| **One sample** | Continuous vs reference | **One-sample t-test** | — | Mean difference, Cohen's d with noncentral-t CI |
| **Paired two-condition** | Continuous mean difference | **Paired t-test** (with explicit `unit_id`) | — | Paired mean difference, Cohen's dz with noncentral-t CI |
| **Paired two-condition** | Rank / distribution target | **Wilcoxon signed-rank** (`estimand="distribution"`) | — | Matched-pairs rank-biserial effect with bootstrap CI |
| **Multi-group (3+)** | Continuous outcome (mean) | **Welch ANOVA + Games-Howell** | **Classical ANOVA + Tukey-Kramer** (requires equal-variance justification) | Pairwise mean differences with simultaneous CIs |
| **Multi-group (3+)** | Rank / distribution target | **Kruskal-Wallis + Dunn-Holm** (`estimand="distribution"`) | — | Epsilon-squared; pairwise contrasts with adjusted p |
| **Two-factor independent** | 2 categorical factors | **Two-way factorial ANOVA** (Type II / III SS) | — | Partial eta-squared with noncentral-F CI, EMMs |
| **Repeated measures (3+)** | Continuous outcome (mean) | **One-way RM-ANOVA** (Mauchly sphericity & GG correction) | — | Partial eta-squared, Holm-adjusted paired t-tests |
| **Repeated measures (3+)** | Rank / distribution target | **Friedman test** (with Kendall's W) | — | Kendall's W with bootstrap CI, Holm-adjusted Wilcoxon pairs |
| **Linear association** | Continuous pairs | **Pearson correlation** | **Partial Pearson** (with explicit quantitative controls) | Pearson r with Fisher-z normal CI |
| **Monotonic association** | Continuous / ordinal pairs | **Spearman rank correlation** | **Kendall tau-b** (for tied ordinal data) | rho with bootstrap CI |
| **Categorical association** | 2 categorical variables | **Pearson Chi-square** (contingency tables) | **Fisher's exact test** (for sparse 2×2 tables) | Cramer's V; odds ratio with Wald CI |
| **Linear regression** | Continuous outcome | **OLS regression** (classical or HC3 robust errors) | — | Standardized betas, coefficient CIs, R² bootstrap CI |
| **Binary regression** | Binary outcome | **Binary logistic regression** (with explicit event level) | — | Odds ratios with Wald CIs, McFadden pseudo-R² |
| **Scale reliability** | Researcher-declared items | **Cronbach's alpha** | — | Alpha with bootstrap CI, item-total correlations |
| **Rater agreement** | Crossed target × rater | **Intraclass Correlation (ICC)** (6 canonical forms) | — | F-tests, exact/Satterthwaite CIs, variance components |

> [!IMPORTANT]
> **Estimand preservation:** Diagnostics (such as Shapiro-Wilk normality tests or Levene variance
> tests) never silently switch a declared mean target to a rank test. If normality is rejected,
> PyAutoStat reports the diagnostic violation while evaluating the declared estimand.

---

## More than p-values

PyAutoStat treats effect sizes and uncertainty as first-class results.

Depending on the method, supported quantities include:

- Cohen's d and Cohen's dz;
- rank-biserial correlation;
- Pearson r, Spearman rho, and Kendall tau-b;
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

PyAutoStat supports advanced study designs, regression modeling, scale reliability, and complete
research governance.

### Two-way factorial ANOVA

For independent observations with two categorical factors and a continuous outcome:

```python
two_way = assistant.two_way_anova(
    outcome="score",
    factor_a="teaching_method",
    factor_b="class_size",
    sum_of_squares="type2",  # or "type3"
)
show(two_way)
```

PyAutoStat evaluates the full `A + B + A×B` model and supports balanced/unbalanced designs, Type II
and Type III sums of squares, partial eta-squared with noncentral-F intervals, cell summaries,
unweighted estimated marginal means, simple effects, marginal comparisons, 2×2
difference-of-differences contrasts, Holm multiplicity adjustment, and residual/design diagnostics.

---

### Repeated-measures analysis

For three or more conditions measured on the same units:

```python
repeated = assistant.run(
    objective="compare_groups",
    outcome="score",
    predictor="visit",
    estimand="mean",
    design="repeated",
    unit_id="participant_id",
    condition_order=("baseline", "week4", "week8"),
    variable_types={"score": "continuous", "visit": "ordinal"},
)
show(repeated)
```

For mean targets, PyAutoStat supports one-way repeated-measures ANOVA with Mauchly's sphericity
assessment, Greenhouse-Geisser correction where applicable, partial eta-squared, and complete
Holm-adjusted paired-t follow-up.

For rank/distribution targets, it supports Friedman inference with Kendall's W and complete
Holm-adjusted paired Wilcoxon follow-up. Complete and incomplete repeated panels remain visible in
sample accounting.

---

### Linear and logistic regression

OLS multiple linear regression supports:

```python
regression = assistant.run(
    objective="regression",
    outcome="score",
    predictors=["study_hours", "attendance"],
    covariance_type="HC3",
)
show(regression)
```

Features include simple and multiple regression, continuous and categorical predictors, explicit
reference coding, classical or HC3 covariance, coefficient intervals, R-squared and adjusted
R-squared, bootstrap uncertainty for in-sample R-squared, standardized betas for continuous
predictors, VIF, residual, heteroscedasticity, condition and influence diagnostics, and
rank-deficiency safeguards.

Binary logistic regression supports:

```python
logistic = assistant.run(
    objective="regression",
    outcome="admitted",
    predictors=["gpa", "test_score"],
    event_level="admitted",
)
show(logistic)
```

Features include explicit event orientation, coefficient and Wald inference, odds ratios with
exponentiated Wald intervals, likelihood fit information, McFadden pseudo-R-squared, and
convergence, separation, VIF, and condition safeguards.

Neither workflow performs automatic variable selection or converts association into a causal claim.

---

### Reliability: Cronbach's alpha and ICC

#### Scale reliability

```python
reliability = assistant.reliability(
    items=["q1", "q2", "q3", "q4"],
)
show(reliability)
```

The workflow reports Cronbach's alpha, bootstrap uncertainty, corrected item-total correlations,
alpha-if-item-deleted, inter-item diagnostics, missingness, and optional explicit reverse scoring.

It does not automatically discover scales, remove items, reverse-score items, or treat alpha as
proof of validity.

#### Inter-rater / test-retest reliability

```python
icc = assistant.intraclass_correlation(
    target="subject_id",
    rater="rater_id",
    value="score",
    model="two_way_random",
    definition="absolute_agreement",
    unit="single",
)
show(icc)
```

PyAutoStat supports the six canonical ICC configurations and requires model, agreement/consistency,
and single/average meaning to be explicit. Results include ANOVA mean-square decomposition,
method-of-moments variance components, analytical uncertainty, applicable F tests, and complete
target accounting.

Negative finite-sample ICC and unconstrained variance-component estimates are preserved rather
than silently clamped to zero.

---

### Practical significance

Researchers can declare what magnitude would be scientifically meaningful:

```python
from pyautostat import MeaningfulEffectThreshold

threshold = MeaningfulEffectThreshold(
    quantity="mean_difference",
    minimum_magnitude=5.0,
    direction="two_sided",
    unit="points",
)

practical = assistant.practical_significance(
    workflow.analysis,
    threshold=threshold,
)
show(practical)
```

Statistical significance and practical importance remain separate. PyAutoStat does not invent a
universal threshold for "important."

---

### Sensitivity analysis

Researchers can declare alternative defensible scenarios and retain every attempted result.
PyAutoStat records estimand and contrast comparability instead of blindly comparing p-values across
scientifically different questions. It does not search across methods for the smallest p-value.

---

### Prospective study planning

`StudyPlanner` supports prospective power and confidence-interval precision planning for selected
independent and paired mean designs:

```python
from pyautostat import StudyPlanner

plan = StudyPlanner().independent_mean_power(
    target_difference=5.0,
    sd_group1=10.0,
    sd_group2=12.0,
    target_power=0.80,
)
show(plan)
```

Planning inputs are researcher-supplied assumptions. Observed/post-hoc power is intentionally not
reported.

---

### Analysis plans and reproducibility

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

### Scientific result auditing

PyAutoStat can check its structured results for internal consistency, including p-value bounds,
confidence-interval ordering, effect-size bounds, sample accounting, degrees-of-freedom identities,
multiplicity invariants, regression coefficient/odds-ratio consistency, ANOVA identities, and ICC
model consistency:

```python
audit = assistant.audit(workflow.report, result=workflow.analysis)
show(audit)
```

An audit pass means the stored result is internally coherent. It does **not** prove that the study
design, source data, or scientific conclusion is correct.

---

## Direct API for experienced users

`ResearchAssistant` is the recommended entry point when you want the full design → recommendation →
analysis → interpretation → report workflow.

Experienced users can use `StatisticalAnalyzer` when the required procedure is already known:

```python
from pyautostat import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(df)
result = analyzer.welch_t_test("score", "group")
```

The direct API provides control without changing the scientific contract of the method.

---

## Scientific safeguards and design principles

**Ask rather than guess.** Essential design facts that cannot be inferred safely are requested.

**Preserve the estimand.** Diagnostics do not silently turn a mean question into a rank question.

**Never hide missingness.** Analysis-specific exclusions, incomplete pairs, and incomplete panels
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
- broad multinomial, ordinal, count, or regularized regression families;
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

For clinical, regulatory, legal, safety-critical, or other high-stakes decisions, independent expert
statistical review is strongly recommended.

---

## Compatibility

- **Python:** 3.10, 3.11, 3.12, 3.13
- **Primary data interface:** pandas DataFrames
- **CI platforms:** Ubuntu and Windows

Core numerical dependencies include pandas, NumPy, SciPy, and statsmodels. See
[`pyproject.toml`](pyproject.toml) for authoritative dependency constraints.

---

## Documentation

| Resource | Purpose |
| --- | --- |
| [`docs/README.md`](docs/README.md) | Documentation index organized by research goal |
| [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md) | Exact supported analysis families and boundaries |
| [`API_REFERENCE.md`](API_REFERENCE.md) | Public API reference |
| [`docs/STATISTICAL_METHOD_CONTRACTS.md`](docs/STATISTICAL_METHOD_CONTRACTS.md) | Scientific contracts |
| [`docs/STATISTICAL_VALIDATION.md`](docs/STATISTICAL_VALIDATION.md) | Numerical/statistical validation policy |
| [`docs/SCIENTIFIC_LIMITATIONS.md`](docs/SCIENTIFIC_LIMITATIONS.md) | Scientific boundaries |
| [`docs/ICC_GUIDE.md`](docs/ICC_GUIDE.md) | ICC model and interpretation guide |
| [`docs/TERMINAL_PRESENTATION.md`](docs/TERMINAL_PRESENTATION.md) | Rich console display guide |
| [`docs/HTML_REPORTING.md`](docs/HTML_REPORTING.md) | HTML presentation guide |
| [`docs/PDF_REPORTING.md`](docs/PDF_REPORTING.md) | PDF export guide |
| [`docs/DOCX_REPORTING.md`](docs/DOCX_REPORTING.md) | Word export guide |
| [`docs/STATIC_FIGURES.md`](docs/STATIC_FIGURES.md) | Static figure export guide |
| [`docs/RESEARCH_BUNDLES.md`](docs/RESEARCH_BUNDLES.md) | Research export bundle guide |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Architecture and module boundaries |
| [`PRODUCT_VISION.md`](PRODUCT_VISION.md) | Enduring product principles |
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

## Compatibility and migration from 0.1.x

PyAutoStat maintains backwards compatibility for existing pipelines built with the 0.1.x composition classes:
- **`ResearchAssistant`**: Recommended canonical API for exploratory profiling and inferential research workflows.
- **`StatisticalAnalyzer`**: Direct advanced calculation engine for pre-specified tests (accompanied by `StudyPlanner` for prospective planning and dedicated sensitivity analysis APIs).
- **`InsightEngine`** and **`ReportGenerator`**: Legacy standalone engines preserved for backwards compatibility with 0.1.x pipelines.

All legacy classes remain importable directly from `pyautostat`, no runtime deprecation warnings are emitted, and no public APIs have been removed.

> [!NOTE]
> Modern `ResearchAssistant` workflows provide design safeguards, estimand preservation, and Welch defaults. They do not necessarily produce identical output to legacy or unconstrained pipelines; migrating analyses should review study design and estimand declarations.

See the [0.1.x to 0.5.x Migration Guide](docs/MIGRATING_FROM_0_1.md) and [API Stability Policy](docs/API_STABILITY.md) for full details.

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
scientific contracts, explicit limitations, and researcher judgment remain separate requirements.
See the [Independent Numerical Validation Guide](docs/NUMERICAL_VALIDATION.md) for PyAutoStat's
evidence hierarchy (Levels A–D), tolerance policies, and first-tranche reference results.

---

## Contributing

Contributions are welcome, particularly for statistical validation, independent reference datasets,
numerical edge cases, scientific documentation, usability, reproducibility, reporting, performance
benchmarks, and accessibility.

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

If you use PyAutoStat in academic research or teaching, please cite it using the metadata in [`CITATION.cff`](CITATION.cff):

```bibtex
@software{pyautostat2026,
  author = {Maji, Koushik Chandra},
  title = {PyAutoStat},
  version = {0.5.0},
  url = {https://github.com/majikoushik/pyautostat},
  year = {2026}
}
```

Or record at minimum:
- PyAutoStat (version 0.5.0);
- Repository: `https://github.com/majikoushik/pyautostat`;
- Exact package version and analysis environment details (Python, pandas, SciPy versions).

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
