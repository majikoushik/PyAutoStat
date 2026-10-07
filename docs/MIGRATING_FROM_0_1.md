# Migrating from PyAutoStat 0.1.x to 0.5.x

This guide assists users transitioning from the legacy 0.1.x composition API (`StatisticalAnalyzer`,
`InsightEngine`, `ReportGenerator`) to the modern integrated research workflow centered on
`ResearchAssistant`.

---

## 1. Why the recommended API changed

In PyAutoStat 0.1.x, analyses were assembled by manually piping data across three separate classes:
running descriptive profiling in `StatisticalAnalyzer`, passing output dictionaries to
`InsightEngine` for rule-based heuristics, and passing those results into `ReportGenerator`.

While functional for initial exploration, this loose composition had substantial scientific and
usability limitations:

1. **Separation of profiling from the research question**: Profiling and hypothesis testing were
   treated as disconnected operations rather than parts of a coherent research design.
2. **Silent estimand shifts**: Early statistical tooling often defaulted to switching a user's
   scientific question (such as testing group means) to a non-parametric rank test based purely on a
   normality diagnostic p-value, silently changing the scientific estimand.
3. **Unstated study design facts**: Essential design properties—such as whether observations are
   independent, paired, or repeated measures, or which level represents a modeled event—cannot be
   reliably inferred from data values alone. Legacy pipelines often made implicit assumptions.
4. **Disjointed governance and reproducibility**: Calculating statistics, interpreting findings,
   checking report consistency, and preserving provenance required disparate ad-hoc steps.

`ResearchAssistant` addresses these challenges by providing one unified, design-aware interface. It
coordinates data profiling, question intake, method recommendation, statistical execution,
deterministic interpretation, publication reporting, internal consistency audits, and replayable
reproducibility records under a single contract.

---

## 2. Old 0.1.x quick-start pattern

In PyAutoStat 0.1.x, users wrote:

```python
import pandas as pd
from pyautostat import InsightEngine, ReportGenerator, StatisticalAnalyzer

df = pd.read_csv("study_data.csv")

# 1. Profile dataset and run standalone tests
analyzer = StatisticalAnalyzer(df)
analysis_results = analyzer.analyze_all()
t_test_results = analyzer.hypothesis_tests(
    test_type="t_test",
    column="exam_score",
    group_column="teaching_method",
)

# 2. Feed results into InsightEngine
engine = InsightEngine(analysis_results)
insights = engine.get_summary()

# 3. Feed results into ReportGenerator
report = ReportGenerator(
    analysis_results=analysis_results,
    insights=insights,
    hypothesis_results=t_test_results,
)
report.to_html("report.html")
```

---

## 3. Current `ResearchAssistant` pattern

In PyAutoStat 0.5.x, `ResearchAssistant` is the canonical entry point:

```python
import pandas as pd
from pyautostat import ResearchAssistant, save_html, show

df = pd.read_csv("study_data.csv")
assistant = ResearchAssistant(df)

# Step 1: Profile the dataset (no research question required)
profile = assistant.profile()
show(profile)

# Step 2: Execute a design-aware research workflow
workflow = assistant.run(
    objective="compare_groups",
    outcome="exam_score",
    predictor="teaching_method",
    estimand="mean",
    design="independent",
    variable_types={"exam_score": "continuous"},
)

# Step 3: Inspect results in the terminal and export a self-contained report
show(workflow)
save_html(workflow, "analysis.html")
```

---

## 4. Mapping table

| 0.1.x Concept / Call | Current Recommended Path | Semantics & Notes |
| :--- | :--- | :--- |
| `StatisticalAnalyzer(df).analyze_all()` | `ResearchAssistant(df).profile()` | Profiling returns comprehensive descriptive statistics, percentiles (P5 to P95), missingness, distributions, and data quality cues without modifying the DataFrame. |
| `StatisticalAnalyzer(df).summarize()` | `assistant.summarize()` or `show(assistant.profile())` | Structured terminal summary; also supports `assistant.summarize(mode="story")` for connected narrative. |
| `StatisticalAnalyzer(...).hypothesis_tests(...)` | `ResearchAssistant(df).run(...)` | Integrated workflow requiring explicit objective, estimand, and design; guards against silent estimand changes and provides effect sizes, intervals, and audit. |
| `InsightEngine(...).generate_insights()` | `workflow.interpretation` / `workflow.explain()` | Rule-based, deterministic interpretation is integrated directly into workflow results; no separate engine instantiation needed. |
| `InsightEngine(...).get_summary()` | `workflow.interpretation.findings_plain` | Accessible directly from the workflow result. |
| `ReportGenerator(...).to_html()` | `save_html(workflow, "path.html")` | Exports self-contained, offline HTML with zero statistical recalculation, structured tables, and XSS protection. |
| `ReportGenerator(...).to_dict()` | `workflow.to_dict()` | Structured, JSON-serializable dictionary capturing specifications, results, diagnostics, interpretation, and audit. |
| `ReportGenerator(...).to_json()` | `workflow.report.to_json()` | JSON-serialized report payload. |
| `ReportGenerator(...).to_csv()` | `save_bundle(...)` or `workflow.report.to_csv()` | Generates multi-format research bundles or CSV tables with formula-injection protection. |

> [!NOTE]
> `StatisticalAnalyzer` remains available as the direct, lower-level calculation engine for users
> who need pre-specified statistical calculations without guided intake or recommendation.

---

## 5. Reporting and export migration

PyAutoStat 0.5.x introduces a clear conceptual separation between viewing and saving:

- **Terminal inspection**: `show(target)` renders polished Rich console representations of
  workflows, profiles, and audit records.
- **Saving to disk**: `save_html(...)`, `save_pdf(...)`, `save_docx(...)`, `save_bundle(...)`, and
  `save_static_figures(...)` write self-contained files directly to disk.
- **In-memory representations**: `to_html(...)`, `to_pdf(...)`, `to_docx(...)`, `to_bundle(...)`,
  and `to_static_figures(...)` return strings, bytes, or data structures.

### Format comparison

| Media / Format | 0.1.x Pattern | 0.5.x Recommended Pattern | Extra Required |
| :--- | :--- | :--- | :--- |
| **Terminal view** | `print(analyzer.summarize())` | `show(workflow)` or `show(profile)` | None (included in core) |
| **Static HTML** | `ReportGenerator(...).to_html("r.html")` | `save_html(workflow, "r.html")` | None (included in core) |
| **Interactive HTML**| `ReportGenerator(...).to_interactive_html("r.html")` | `save_interactive_html(workflow, "r.html")` | `pyautostat[report]` |
| **Publication-oriented PDF** | Not available | `save_pdf(workflow, "r.pdf")` | `pyautostat[pdf]` |
| **Word (.docx)** | Not available | `save_docx(workflow, "r.docx")` | `pyautostat[docx]` |
| **Static Figures** | Not available | `save_static_figures(workflow, "dir")`| `pyautostat[figures]` |
| **Research Bundle** | Not available | `save_bundle(workflow, "bundle.zip")` | None (core) |

All 0.5.x export layers strictly enforce:
- **Zero recalculation guarantee**: Presentation renderers extract authoritative stored estimates,
  confidence intervals, and diagnostics without re-executing statistical models.
- **Privacy preservation**: Raw observation rows and participant identifiers are never serialized
  into reports, exports, or bundles.

---

## 6. What remains backwards compatible

1. **Imports**: `StatisticalAnalyzer`, `InsightEngine`, and `ReportGenerator` remain importable
   directly from `pyautostat`:
   ```python
   from pyautostat import InsightEngine, ReportGenerator, StatisticalAnalyzer
   ```
2. **0.1.x code continues to function**: Existing scripts using `StatisticalAnalyzer.analyze_all()`,
   `InsightEngine.generate_insights()`, and `ReportGenerator.to_html()` continue to run.
3. **No runtime deprecation warnings**: No code-level deprecation warnings or warnings are raised
   in 0.5.0 for using legacy classes.
4. **No removals**: No public class, function, or parameter from 0.1.x has been removed.

---

## 7. Major capability additions since 0.1.x

PyAutoStat has expanded from 4 basic statistical checks in 0.1.0 to **24 registered statistical method
IDs** across **9 distinct analytical families**:

1. **Two-Group Independent Means**:
   - Welch independent t-test (`welch_t`) — default mean comparison without assuming equal variance
   - Student independent t-test (`student_t`) — explicit equal-variance mean comparison
   - Mann-Whitney U test (`mann_whitney_u`) — rank-distribution comparison
2. **One-Sample & Reference Comparison**:
   - One-sample t-test (`one_sample_t`) — single-group mean against reference threshold
3. **Paired & Two-Condition Comparisons**:
   - Paired-samples t-test (`paired_t`) — within-unit mean difference with unit ID tracking
   - Wilcoxon signed-rank test (`wilcoxon_signed_rank`) — within-unit rank-distribution comparison with
     matched-pairs rank-biserial correlation
4. **Multi-Group Comparisons (3+ Groups)**:
   - Welch one-way ANOVA (`welch_anova`) with Games-Howell post-hoc contrasts
   - Classical one-way ANOVA (`one_way_anova`) with Tukey-Kramer post-hoc contrasts
   - Kruskal-Wallis rank test (`kruskal_wallis`) with Dunn-Holm post-hoc contrasts
5. **Factorial ANOVA**:
   - Two-way factorial ANOVA (`two_way_anova`) with Type II (default) and Type III sums of squares, simple
     main effects, and marginal means
6. **Repeated Measures (3+ Conditions)**:
   - One-way repeated-measures ANOVA (`repeated_measures_anova`) with Mauchly sphericity test,
     Greenhouse-Geisser correction, and paired t follow-ups
   - Friedman rank test (`friedman_test`) with Kendall's W concordance and Wilcoxon follow-ups
7. **Regression**:
   - OLS linear regression (`linear_regression`) with classical or HC3 robust standard errors, VIF,
     Breusch-Pagan, and standardized coefficients
   - Binary logistic regression (`logistic_regression`) with odds ratios, Wald confidence intervals,
     and McFadden pseudo-R-squared
8. **Bivariate Association & Categorical**:
   - Pearson correlation (`pearson_correlation`) with Fisher-z confidence intervals ($n > 3$)
   - Spearman rank correlation (`spearman_correlation`) — monotonic association
   - Kendall's tau-b (`kendall_tau_b`) — rank concordance with tie handling
   - Point-biserial correlation (`point_biserial_correlation`) — continuous outcome with binary predictor
   - Partial Pearson correlation (`partial_pearson_correlation`) with declared quantitative controls
   - Pearson chi-square test of independence (`pearson_chi_square`) with Cramer's V
   - Fisher's exact test (`fisher_exact`) — 2x2 contingency test for sparse expected counts
   - McNemar test (`mcnemar`) — exact two-sided test of paired binary proportions using binomial inference
9. **Reliability & Agreement**:
   - Cronbach's alpha (`cronbach_alpha`) with respondent-row bootstrap confidence intervals,
     corrected item-total correlations, alpha-if-deleted, and optional explicit reverse scoring
   - Intraclass Correlation Coefficient (`intraclass_correlation`) across all 6 canonical Shrout & Fleiss /
     McGraw & Wong models: ICC(1,1), ICC(1,k), ICC(2,1), ICC(2,k), ICC(3,1), ICC(3,k)

### Additional Research Lifecycle & Governance Capabilities

Beyond inferential methods, PyAutoStat provides integrated research lifecycle capabilities:

- **Prospective study planning**: `StudyPlanner` for sample-size and power/precision planning prior
  to data collection.
- **Statistical Analysis Plans**: `StatisticalAnalysisPlan` snapshots and adherence comparison
  (`compare_plan_to_result`).
- **Sensitivity analysis**: `SensitivitySpecification` for evaluating same-estimand and
  different-estimand alternative scenarios.
- **Practical significance**: `MeaningfulEffectThreshold` for evaluating empirical effects against
  researcher-defined practical thresholds.
- **Audit & Replay**: `StatisticalResultAuditor` for verifying internal consistency, and
  `ReproducibilityRecord` with `reproduce()` for explicit execution replay.

---

## 8. Behavioral differences users must understand

Users migrating analyses should be aware of key scientific differences:

### 8.1 Do not assume legacy and modern result equivalence

> [!WARNING]
> Do NOT assume that `ResearchAssistant.run(...)` and legacy 0.1.x pipelines will always produce
> identical results.

While shared underlying numerical primitives (such as the standard Student's t-test calculation)
use established numerical implementations, the orchestration, defaults, and safeguards differ:

- **Default two-group mean test**: `ResearchAssistant` defaults to **Welch's t-test** (which does
  not assume equal population variances) when comparing two independent group means. In 0.1.x,
  scripts often ran Student's t-test assuming equal variance.
- **No silent estimand changes**: If you ask for a mean comparison, `ResearchAssistant` runs a mean
  test with appropriate variance handling; it will not silently switch to Mann-Whitney U because
  Shapiro-Wilk rejected normality.
- **Explicit study design**: `ResearchAssistant` requires specifying whether data are independent,
  paired, or repeated. It does not infer pairing from row ordering.
- **Sample accounting**: `ResearchAssistant` enforces complete-case sample accounting across the
  outcome and all predictors, reporting initial N, excluded N, and effective N explicitly.
- **Contrast orientation**: Differences preserve signed contrast direction (`Group A - Group B`)
  rather than reporting absolute magnitudes.

Migrating analyses should explicitly verify study design, estimand choice, contrast direction, and
missing-data policies.

### 8.2 "Ask rather than guess"

If required design facts cannot be safely deduced from data values alone, `ResearchAssistant.run()`
pauses and returns `WorkflowStatus.NEEDS_INPUT` with structured clarification questions. It will not
silently assume independent observations, random assignment, or event levels.

### 8.3 Qualified interpretation, not automated claims

Interpretation in 0.5.x is deterministic, rule-based, and local. It does not use generative AI or
external APIs. Findings distinguish statistical significance from practical magnitude and
explicitly surface assumptions, limitations, and caveats.

---

## 9. What remains intentionally unsupported

PyAutoStat maintains explicit scientific boundaries:

- **No mixed-effects models**: Linear mixed models (LMM), generalized linear mixed models (GLMM),
  and GEE are not supported.
- **No survival analysis**: Cox proportional hazards, Kaplan-Meier curves, and log-rank tests are
  not supported.
- **No automated causal inference**: PyAutoStat does not infer causal direction without an explicit
  experimental design.
- **No automated outlier pruning or imputation**: Data values are never silently altered, recoded,
  or imputed.
- **No automated p-hacking**: Automated model search, stepwise selection, or alpha-hunting is
  strictly prohibited.
- **No GUI or cloud service**: Core operations remain local, offline, and UI-independent.

---

## 10. Suggested migration strategy

For teams with existing 0.1.x codebases:

1. **Step 1: Adopt `ResearchAssistant(df).profile()` first**:
   Replace `StatisticalAnalyzer(df).analyze_all()` with `ResearchAssistant(df).profile()`. This
   provides richer percentiles, distribution summaries, and quality cues immediately without
   affecting downstream logic.
2. **Step 2: Build new analyses using `ResearchAssistant.run()`**:
   For any new research question, declare the objective, outcome, predictor, estimand, and study
   design explicitly using `ResearchAssistant.run(...)`.
3. **Step 3: Incrementally migrate legacy hypothesis tests**:
   When updating legacy test scripts, clarify the scientific question:
   - If comparing means between two groups without equal variance assumptions, use
     `objective="compare_groups", estimand="mean", design="independent"` (runs Welch's t-test).
   - If evaluating a paired pre/post design, supply `design="paired"` with `unit_id`.
   - If evaluating ranks, explicitly declare `estimand="distribution"` or `estimand="rank"`.
4. **Step 4: Adopt modern export and presentation**:
   Replace `ReportGenerator` calls with `show(workflow)` for console inspection and `save_html(workflow, ...)`
   or `save_bundle(workflow, ...)` for archival reporting.
