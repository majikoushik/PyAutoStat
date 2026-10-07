# PyAutoStat 0.5.0 Release Notes

PyAutoStat 0.5.0 represents a major expansion of the library from exploratory analysis utilities
into an explainable, reproducible, and design-aware research analysis assistant for pandas
DataFrames.

---

## 1. Overview

PyAutoStat is an open-source statistical research assistant built for researchers, scientific
programmers, analysts, educators, and students. It bridges dataset profiling, research question
specification, method recommendation, statistical execution, uncertainty estimation, deterministic
interpretation, publication-oriented reporting, internal consistency audits, and reproducibility
records.

In version 0.5.0:
- `ResearchAssistant` is established as the canonical entry point for all beginner and standard
  research analyses.
- The statistical capability expands from 4 basic checks to **24 registered statistical method IDs**
  across 9 distinct analysis families.
- Multi-format presentation and export support is introduced, including Rich terminal rendering,
  offline static and interactive HTML, publication-oriented PDF, editable Microsoft Word (.docx), static
  scientific figures, and multi-format research bundles with SHA-256 integrity verification.
- Advanced governance tools provide prospective study planning, pre-analysis plans, sensitivity
  scenarios, practical-significance threshold evaluations, and replayable execution records.
- Legacy classes (`StatisticalAnalyzer`, `InsightEngine`, `ReportGenerator`) remain supported for
  backwards compatibility.

---

## 2. Canonical `ResearchAssistant` Workflow

`ResearchAssistant` is the recommended entry point for research analyses. It eliminates the need to
manually pipe data between disparate classes or hand-assemble reporting pipelines.

```python
import pandas as pd
from pyautostat import ResearchAssistant, save_html, show

df = pd.DataFrame(
    {
        "teaching_method": ["standard"] * 6 + ["new"] * 6,
        "exam_score": [62, 65, 68, 70, 72, 75, 67, 71, 74, 78, 80, 84],
    }
)

assistant = ResearchAssistant(df)

# 1. Profile dataset quality and distributions
profile = assistant.profile()
show(profile)

# 2. Execute a design-aware research workflow
workflow = assistant.run(
    objective="compare_groups",
    outcome="exam_score",
    predictor="teaching_method",
    estimand="mean",
    design="independent",
    variable_types={"exam_score": "continuous"},
)

# 3. Inspect results in console and export an offline HTML report
show(workflow)
save_html(workflow, "reports/analysis.html")
```

### Key Workflow Features
- **Design-first specification**: Declares objective, estimand, and study design (independent,
  paired, repeated).
- **"Ask rather than guess"**: Pauses with `WorkflowStatus.NEEDS_INPUT` if required design facts
  (e.g., pairing, clustering, event levels) cannot be safely deduced from data values alone.
- **Estimand preservation**: Recommends methods matching your scientific question—never silently
  substituting a rank test for a mean comparison because of a normality diagnostic p-value.
- **Deterministic interpretation**: Generates local, rule-based findings and executive summaries
  without requiring a generative AI model or cloud service.

---

## 3. Statistical Capability Expansion (24 Methods)

PyAutoStat supports 24 registered statistical method IDs across 9 analytical families:

1. **Two-Group Independent Means**:
   - Welch independent-samples t-test (`welch_t`) — default mean comparison without assuming equal
     variance
   - Student independent-samples t-test (`student_t`) — explicit equal-variance mean comparison
   - Mann-Whitney U test (`mann_whitney_u`) — rank-distribution comparison
2. **One-Sample & Reference Comparison**:
   - One-sample t-test (`one_sample_t`) — mean comparison against a reference threshold
3. **Paired & Two-Condition Comparisons**:
   - Paired-samples t-test (`paired_t`) — within-unit mean difference with unit ID tracking
   - Wilcoxon signed-rank test (`wilcoxon_signed_rank`) — within-unit rank-distribution comparison with
     matched-pairs rank-biserial correlation
4. **Multi-Group Comparisons (3+ Groups)**:
   - Welch one-way ANOVA (`welch_anova`) with Games-Howell pairwise post-hoc contrasts
   - Classical one-way ANOVA (`one_way_anova`) with Tukey-Kramer pairwise post-hoc contrasts
   - Kruskal-Wallis rank test (`kruskal_wallis`) with Dunn-Holm pairwise post-hoc contrasts
5. **Factorial ANOVA**:
   - Two-way factorial ANOVA (`two_way_anova`) with Type II (default) and Type III sums of squares, simple
     main effects, and marginal means
6. **Repeated Measures (3+ Conditions)**:
   - One-way repeated-measures ANOVA (`repeated_measures_anova`) with Mauchly sphericity test,
     Greenhouse-Geisser correction, partial eta-squared, and paired t follow-ups
   - Friedman rank test (`friedman_test`) with Kendall's W concordance and Wilcoxon follow-ups
7. **Regression**:
   - Ordinary least-squares regression (`linear_regression`) with classical or HC3 robust standard
     errors, VIF, Breusch-Pagan, and standardized coefficients
   - Binary logistic regression (`logistic_regression`) with odds ratios, Wald confidence intervals,
     and McFadden pseudo-R-squared
8. **Bivariate Association & Categorical**:
   - Pearson correlation (`pearson_correlation`) — linear association with Fisher-z confidence intervals ($n > 3$)
   - Spearman rank correlation (`spearman_correlation`) — monotonic association
   - Kendall's tau-b (`kendall_tau_b`) — rank concordance with tie handling
   - Point-biserial correlation (`point_biserial_correlation`) — continuous outcome with binary predictor
   - Partial Pearson correlation (`partial_pearson_correlation`) with declared quantitative controls
   - Pearson chi-square test of independence (`pearson_chi_square`) with Cramer's V
   - Fisher's exact test (`fisher_exact`) — 2x2 contingency test for sparse expected counts
   - McNemar test (`mcnemar`) — exact two-sided test of paired binary proportions using binomial inference on discordant pairs
9. **Reliability & Agreement**:
   - Cronbach's alpha (`cronbach_alpha`) — internal consistency scale reliability with respondent-row bootstrap confidence intervals,
     corrected item-total correlations, alpha-if-deleted, and optional explicit reverse scoring
   - Intraclass Correlation Coefficient (`intraclass_correlation`) — quantitative rater reliability across all 6 canonical Shrout & Fleiss / McGraw & Wong
     models: ICC(1,1), ICC(1,k), ICC(2,1), ICC(2,k), ICC(3,1), ICC(3,k)

---

## 4. Reporting and Export Expansion

PyAutoStat introduces a unified presentation system adhering to a strict **zero-recalculation
guarantee** and **data privacy contract**:

- **Rich Terminal Presentation**: `show(target)` provides clear, colored console summaries for
  workflows, profiles, planning results, sensitivity analyses, and audit records.
- **Offline Static HTML**: `save_html(...)` / `to_html(...)` outputs self-contained HTML reports with
  structured tables and XSS protection, requiring no JavaScript runtime.
- **Interactive HTML**: `save_interactive_html(...)` generates Plotly-powered interactive charts
  (forest plots, difference intervals, contingency heatmaps) bundled offline without CDN calls.
- **Publication-Oriented PDF**: `save_pdf(...)` / `to_pdf(...)` produces print-fidelity PDF documents via
  headless Chromium.
- **Editable Word Documents**: `save_docx(...)` / `to_docx(...)` exports native Microsoft Word (.docx)
  files with structured OpenXML tables, headings, and optional embedded figures.
- **Static Scientific Figures**: `save_static_figures(...)` exports standalone vector (SVG, PDF) or
  raster (PNG) figures via Kaleido.
- **Research Export Bundles**: `save_bundle(...)` packages multi-format reports, CSV tables, and
  provenance records into a single ZIP archive with SHA-256 integrity verification (`verify_bundle`).

---

## 5. Governance, Audit, and Reproducibility

PyAutoStat provides tools to support sound scientific workflows throughout the research lifecycle:

- **Prospective Study Planning**: `StudyPlanner` calculates required sample sizes and power/precision
  curves before data collection.
- **Statistical Analysis Plans**: `StatisticalAnalysisPlan` captures immutable pre-analysis snapshots;
  `compare_plan_to_result` evaluates adherence without judging research conduct.
- **Sensitivity Analysis**: `SensitivitySpecification` evaluates outcomes across explicit
  same-estimand and different-estimand alternative scenarios.
- **Practical Significance**: `MeaningfulEffectThreshold` evaluates empirical effects against
  researcher-declared minimum thresholds of practical interest.
- **Report Audit**: `StatisticalResultAuditor` checks internal consistency across sample counts,
  degrees of freedom, effect sizes, and p-values without recalculation.
- **Reproducibility & Replay**: `ReproducibilityRecord` captures analysis configuration, environment
  metadata, and dataset fingerprints, enabling explicit re-execution via `reproduce()`.

---

## 6. Compatibility with 0.1.x

- **Import Compatibility**: `StatisticalAnalyzer`, `InsightEngine`, and `ReportGenerator` remain
  importable from `pyautostat`.
- **Existing Workflows Supported**: Code written for 0.1.x continues to execute.
- **No Deprecation Warnings**: No runtime deprecation warnings are raised in this release.
- **Clear Separation**: New users are encouraged to start with `ResearchAssistant`. `StatisticalAnalyzer`
  is maintained as the direct advanced computation engine.

---

## 7. Migration Guide

For detailed instructions on transitioning from 0.1.x pipelines to the modern `ResearchAssistant`
architecture, see [`docs/MIGRATING_FROM_0_1.md`](MIGRATING_FROM_0_1.md).

For the API stability tier classification and evolution policy, see
[`docs/API_STABILITY.md`](API_STABILITY.md).

---

## 8. Installation and Optional Extras

PyAutoStat requires **Python 3.10 or newer**.

```bash
# Core package (profiling, 24 statistical methods, Rich terminal, static HTML)
pip install pyautostat

# Optional extras:
pip install "pyautostat[report]"   # Interactive Plotly HTML reports
pip install "pyautostat[pdf]"      # Publication-oriented PDF export (requires playwright)
pip install "pyautostat[docx]"     # Microsoft Word (.docx) export
pip install "pyautostat[figures]"  # Standalone static PNG/SVG/PDF figures (requires kaleido)
pip install "pyautostat[dev]"      # Developer tools, linters, test runner
```

---

## 9. Known Limitations

- **Bounded Method Scope**: PyAutoStat intentionally does not support mixed-effects models (LMM/GLMM),
  generalized estimating equations (GEE), survival analysis (Cox/Kaplan-Meier), time-series
  forecasting, or Bayesian inference.
- **No Automated Outlier Deletion**: Outliers are flagged for researcher review but never silently
  pruned or imputed.
- **Researcher Responsibility**: Study design facts, estimand selection, sampling validity, and
  measurement meaning cannot be deduced by software and remain the responsibility of the researcher.
- **Offline Library**: PyAutoStat does not provide a graphical user interface (GUI) or run web servers.

---

## 10. Alpha Status

PyAutoStat 0.5.0 is an **Alpha** release (`Development Status :: 3 - Alpha`). While the public API,
statistical engines, and export layers have undergone rigorous automated testing and auditing, the
interfaces may continue to receive additive refinements prior to a future 1.0.0 release.
