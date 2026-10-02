# Terminal Presentation Layer

PyAutoStat provides a Rich-powered terminal presentation layer that transforms structured,
authoritative statistical results into clear, accessible, and readable terminal reports.

The presentation layer is strictly a **view** over existing results. It never recomputes
statistics, selects methods, alters estimands, reverses contrasts, or invents evidence.

---

## 1. Architecture

```
Existing PyAutoStat statistical engine
                │
                ▼
         structured result
                │
                ▼
       presentation adapter
                │
                ▼
        terminal view model
                │
                ▼
           Rich renderer
                │
                ▼
     polished terminal output
```

The presentation layer enforces clean separation of concerns:
- **`models.py`**: Normalized dataclasses (`TerminalView`, `DisplayMetric`, `DisplayTable`, `DisplayRow`, `DisplayDiagnostic`) that hold display-ready values.
- **`adapters/`**: Reads domain records (`ResearchWorkflowResult`, `AnalysisResult`, `DatasetProfile`, planning, governance, or descriptives) without modifying them, formatting values for presentation.
- **`theme.py`**: Centralized semantic Rich theme (`PYAUTOSTAT_THEME`).
- **`renderers/`**: Visual layout components that render normalized `TerminalView` instances to a Rich `Console`.
- **`formatting.py`**: Pure display helpers for numbers, p-values, sample sizes, effects, and intervals.
- **`api.py`**: Public entry point `show()`.

---

## 2. Public API

```python
from pyautostat import show

# 1. Dataset profile
profile = assistant.profile()
show(profile)

# 2. Workflow results (all 24 statistical methods)
workflow = assistant.run(...)
show(workflow)

# 3. Direct analysis results
analysis = workflow.analysis
show(analysis)

# 4. Descriptives
freq = assistant.frequency_table("category")
show(freq)
ct = assistant.cross_tab("status", "group")
show(ct)

# 5. Study planning & Sensitivity
plan = planner.independent_mean_power(...)
show(plan)
sens = assistant.sensitivity(...)
show(sens)

# 6. Governance & Audit
sap = assistant.analysis_plan(...)
show(sap)
adherence = assistant.plan_adherence(...)
show(adherence)
completeness = assistant.reporting_completeness(...)
show(completeness)
audit = assistant.audit(...)
show(audit)
repro = assistant.reproducibility_record(...)
show(repro)
outcome = reproduce(repro, data=df)
show(outcome)
ledger = assistant.enable_tracking()
show(ledger)
snapshot = assistant.session_snapshot(workflow)
show(snapshot)

# Detail levels: "compact", "standard" (default), "full"
show(result, detail="compact")
show(result, detail="standard")
show(result, detail="full")
```

### Detail Levels

- **`compact`**: Single-line concise summary suitable for batch logs, CI, pipelines, or quick console checks.
- **`standard`** (default): Complete yet clean report with title panel, design/estimand metrics, key results, essential summary tables, deterministic interpretation, diagnostic context, and limitations. Long tables truncate gracefully to the primary rows with explicit guidance.
- **`full`**: Everything in standard, plus recommendation rationale, audit verification status, reproducibility metadata (seeds, bootstrap resamples, specification references), untruncated tables, reference levels, and extended diagnostic context.

---

## 3. Color and Accessibility Semantics

Color is **never** the sole carrier of meaning:
- Every status uses explicit text: `PASS`, `REVIEW`, `NOT ASSUMED`, `REQUIRED`, `FAILED`.
- Statistical significance is **never** color-coded green (for significant) or red (for non-significant), because scientific significance is not inherently "good" or "bad".
- Green (`status.success`) is reserved for operational successes (e.g. audit `PASSED`).
- Amber (`status.warning` / `status.review`) indicates review cues, advisory warnings, or missing inputs.
- Red (`status.error`) indicates computational failures or audit failures.
- Blue/cyan tones represent analysis families and sections (`family.mean`, `family.association`, `family.profile`, `family.planning`, `family.governance`, `family.audit`).
- Background colors are avoided by default.
- Automatically respects `NO_COLOR` and non-TTY redirected output.

---

## 4. Responsive Terminal Widths

Outputs adapt cleanly across terminal widths:
- **70 columns (narrow)**: Sections stack vertically, labels wrap cleanly, secondary columns are omitted in standard mode, and primary estimates, intervals, and orientation fields remain visible without horizontal clipping.
- **90 columns (standard)**: Balanced tabular presentation with full metrics and diagnostics.
- **120 columns (wide)**: Extended layouts with wide confidence intervals, standard errors, and detailed diagnostics side-by-side.

---

## 5. Statistical Method Coverage Matrix

PyAutoStat supports terminal presentation across all 24 registered statistical method contracts:

| Method ID | Method Name | Renderer Family | Primary Display Elements |
|---|---|---|---|
| `one_sample_t` | One-sample t-test | `OneSampleRenderer` | Reference value, mean, difference, CI, Cohen's d, t, df, p |
| `welch_t` | Welch's independent-samples t-test | `TwoGroupRenderer` | Signed difference, CI, Cohen's d, unequal variance context, t, df, p |
| `student_t` | Student's independent-samples t-test | `TwoGroupRenderer` | Signed difference, CI, Cohen's d, equal-variance assumption context |
| `mann_whitney_u` | Mann-Whitney U test | `TwoGroupRenderer` | Group medians/IQRs, rank-biserial effect, CI, U statistic, p |
| `paired_t` | Paired-samples t-test | `PairedRenderer` | Condition order, paired difference, CI, Cohen's dz, complete pairs |
| `wilcoxon_signed_rank` | Wilcoxon signed-rank test | `PairedRenderer` | Condition medians/IQRs, matched rank-biserial, CI, zero-diff policy |
| `welch_anova` | Welch's one-way ANOVA | `MultiGroupRenderer` | Omnibus Welch F, df, p, Games-Howell pairwise table with simultaneous CIs |
| `one_way_anova` | One-way ANOVA (equal variance) | `MultiGroupRenderer` | Omnibus F, df, p, eta-squared, Tukey-Kramer pairwise table |
| `kruskal_wallis` | Kruskal-Wallis rank test | `MultiGroupRenderer` | Omnibus H, df, p, epsilon-squared, Dunn-Holm pairwise table with rank-biserial CIs |
| `pearson_correlation` | Pearson correlation | `AssociationRenderer` | Pearson r, 95% CI, p, complete pairs, linear target context |
| `spearman_correlation` | Spearman rank correlation | `AssociationRenderer` | Spearman rho, 95% CI, p, monotonic target context |
| `kendall_tau_b` | Kendall's tau-b concordance | `AssociationRenderer` | Kendall tau-b, 95% CI, p, tie context; explicit note that tau-b is not variance explained |
| `point_biserial_correlation`| Point-biserial correlation | `AssociationRenderer` | Point-biserial r, CI, p, explicit positive-level (+1 coding) orientation |
| `partial_pearson_correlation`| Partial Pearson correlation | `AssociationRenderer` | Partial r, CI, t, df, p, control covariates list; non-causal limitation |
| `pearson_chi_square` | Pearson chi-square independence | `CategoricalRenderer` | Contingency table, Chi-square, df, p, Cramer's V, expected count diagnostics |
| `fisher_exact` | Fisher's exact test | `CategoricalRenderer` | 2x2 contingency table, category orientation, sample odds ratio, CI |
| `mcnemar` | McNemar paired test | `PairedCategoricalRenderer` | 2x2 transition table, discordant pair counts, paired proportion difference, CI, exact p |
| `linear_regression` | Linear regression (OLS) | `RegressionRenderer` | R², adj R², R² CI, model F, p, coefficients table, HC3 covariance, full diagnostics (VIF, condition number, influence) |
| `logistic_regression` | Logistic regression | `LogisticRenderer` | Modeled event, reference category, LR statistic, pseudo-R², OR-first table, Wald CI, z, p |
| `cronbach_alpha` | Cronbach's alpha | `ReliabilityRenderer` | Cronbach alpha, CI, item count, item-total correlations, alpha-if-deleted, inter-item matrix |
| `repeated_measures_anova` | Repeated-measures ANOVA | `RepeatedMeasuresRenderer` | Condition summaries, omnibus F, df, partial eta², Mauchly sphericity, Greenhouse-Geisser correction, pairwise t |
| `friedman_test` | Friedman rank test | `RepeatedMeasuresRenderer` | Condition medians/IQRs, Kendall W, Q statistic, df, pairwise Wilcoxon with Holm adjustment |
| `two_way_anova` | Two-way factorial ANOVA | `FactorialRenderer` | Factors A and B, A×B interaction distinctly displayed, SS type, ANOVA effects table, cell summaries |
| `intraclass_correlation` | Intraclass correlation (ICC) | `ICCRenderer` | Canonical definition displayed BEFORE estimate (model, agreement vs consistency, single vs average), ICC estimate (never clamped if negative), CI, F, df, p, ANOVA mean squares, variance components |

---

## 6. Descriptive Direct Outputs

PyAutoStat supports terminal presentation for descriptive analysis outputs:
- **`frequency_table(column)`**: Displays level/category, count, valid percentage, total percentage, cumulative percentage (when applicable), valid N, missing N, and high-cardinality diagnostics.
- **`cross_tab(row, col)`**: Displays 2D contingency counts, row/column categories, complete paired cases, and missing paired rows.

---

## 7. Governance and Study Planning Support

PyAutoStat governance and planning results render via specialized presentation families:
- **`StudyPlanningResult`**: Prospective sample-size and power requirements, target precision, alpha, and researcher-supplied design assumptions. Explicitly notes that prospective planning is NOT observed post-hoc power.
- **`SensitivityResult`**: Robustness scenario comparison table showing alternative methods, estimand comparability, contrast comparability, and decision status. Scenarios are never sorted or ranked by p-value.
- **`PracticalSignificanceResult`**: Evaluates observed estimates and confidence intervals against researcher-supplied practical significance thresholds, keeping practical verdicts distinct from statistical significance.
- **`StatisticalAnalysisPlan`**: Renders as an a priori scientific study plan (question, design, planned method, alpha, confidence level, missing data rules, multiplicity controls, sensitivity scenarios, and revision status) rather than an empirical result.
- **`PlanAdherenceResult`**: Planned versus performed comparison table across all study dimensions, reporting deviations objectively without disciplinary or misconduct rhetoric.
- **`ReportingCompletenessResult`**: Summarizes present, partial, and missing structural reporting items under target guidelines (e.g. APA), explicitly stating it is not a study-quality or publication-readiness score.
- **`AuditResult`**: Scientific consistency audit verifying internal numerical invariants, contracts, and record coherence with operational status badges (`PASS`, `FAIL`, `REVIEW`). Explicitly notes it verifies internal consistency, not empirical truth.
- **`ReproducibilityRecord` & `ReproductionOutcome`**: Displays runtime, package version, deterministic random seeds, bootstrap metadata, and dataset fingerprint verification without raw dictionary dumps.
- **`DecisionLedger`**: Ordered sequential ledger of analytical decisions, audit events, and user specifications, noting that local ledgers do not claim external authenticated provenance.
- **`ResearchSessionSnapshot`**: Overview of active session state, completed analyses, governance records, and active capabilities without unstructured JSON dumps.

---

## 8. Direct `AnalysisResult` Support

Direct calls to `show(analysis_result)` are supported for standalone analysis objects when they contain safe, self-describing statistical values. When invoked directly on an `AnalysisResult`, presentation displays only stored analysis fields without fabricating questions, recommendations, or extraneous metadata.
