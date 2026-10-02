# Terminal Presentation Layer

PyAutoStat provides a Rich-powered terminal presentation layer that transforms structured,
authoritative statistical results into clear, accessible, and readable terminal reports.

The presentation layer is strictly a **view** over existing results. It never recomputes
statistics, selects methods, alters estimands, or invents evidence.

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
- **`adapters.py`**: Reads domain records (`ResearchWorkflowResult` or dataset profile dictionaries) without modifying them, formatting values for presentation.
- **`theme.py`**: Centralized semantic Rich theme (`PYAUTOSTAT_THEME`).
- **`renderers/`**: Visual layout components that render normalized `TerminalView` instances to a Rich `Console`.
- **`api.py`**: Public entry point `show()`.

---

## 2. Public API

```python
from pyautostat import show

# Dataset profile
profile = assistant.profile()
show(profile)

# Workflow results
workflow = assistant.run(...)
show(workflow)

# Detail levels: "compact", "standard" (default), "full"
show(workflow, detail="compact")
show(workflow, detail="standard")
show(workflow, detail="full")
```

### Detail Levels

- **`compact`**: Single-line concise output suitable for pipelines, batch runs, or quick terminal checks.
  - Profile: `Dataset Profile | 5,000 rows x 39 vars | missing=502 (0.3%) | duplicates=0`
  - Welch: `Welch t-test | N=5,000 | diff=-$47.70 | 95% CI -56.86...-38.55 | d=-0.29 | p=<0.001`
  - Pearson: `Pearson r | N=5,000 | r=-0.036 | 95% CI [-0.063, -0.008] | p=0.012`
- **`standard`** (default): Balanced report with title panel, research question/design, key result metrics, essential summary tables, deterministic interpretation, diagnostic context, and limitations.
- **`full`**: Everything in standard, plus recommendation rationale, audit verification status, reproducibility metadata (seed, bootstrap resamples, specification reference), and extended tables/diagnostics.

---

## 3. Color and Accessibility Semantics

Color is **never** the sole carrier of meaning:
- Every status uses explicit text: `PASS`, `REVIEW`, `NOT ASSUMED`, `REQUIRED`, `FAILED`.
- Statistical significance is **never** color-coded green (for significant) or red (for non-significant), because scientific significance is not inherently "good" or "bad".
- Green (`status.success`) is reserved for operational successes (e.g. audit `PASSED`).
- Amber (`status.warning` / `status.review`) indicates review cues, advisory warnings, or missing inputs.
- Red (`status.error`) indicates computational or audit failures.
- Blue/cyan tones represent analysis families and sections (`family.mean`, `family.association`, `family.profile`).
- Background colors are avoided by default.
- Automatically respects `NO_COLOR` and non-TTY redirected output.

---

## 4. Supported Pilot Objects

In this pilot implementation, `show()` supports:
1. **Dataset Profile**: Returned by `ResearchAssistant(df).profile()` or `StatisticalAnalyzer(df).analyze_all()`.
2. **Welch Independent-Samples t-test**: `ResearchWorkflowResult` with `method_id="welch_t"`.
3. **Pearson Correlation**: `ResearchWorkflowResult` with `method_id="pearson_correlation"`.
4. **Workflow Statuses**: `WorkflowStatus.NEEDS_INPUT`, `WorkflowStatus.DATA_LIMITED`, `WorkflowStatus.UNSUPPORTED`, and `WorkflowStatus.FAILED`.

Unsupported object types raise `TypeError`. Methods outside the pilot raise `UnsupportedPresentationError`.

---

## 5. Future Presentation Blueprint

Future phases will extend `RENDERER_REGISTRY` to cover all PyAutoStat methods using the following specification:

### ONE-SAMPLE T-TEST
- **Primary**: observed-minus-reference difference, CI, Cohen's d, effect CI, p-value
- **Additional**: reference value, sample mean, N, SD

### STUDENT T / WELCH T
- **Primary**: signed mean difference, CI, Cohen's d, effect CI, p-value
- **Additional**: group summary, contrast direction, variance assumption status

### PAIRED T
- **Primary**: paired mean difference, CI, Cohen's dz, dz CI, p-value
- **Additional**: complete pairs, incomplete units, condition order

### MANN-WHITNEY U
- **Primary**: U statistic, p-value, rank-biserial effect, CI
- **Additional**: group medians/IQRs, explicit distribution/rank estimand

### WILCOXON SIGNED-RANK
- **Primary**: statistic, p-value, matched-pairs rank-biserial, bootstrap CI
- **Additional**: complete pairs, zero-difference policy, condition order

### WELCH ANOVA
- **Primary**: Welch F, df, p-value
- **Additional**: group means, group Ns, Games-Howell pairwise table

### ONE-WAY ANOVA
- **Primary**: F, df, p-value, eta-squared
- **Additional**: group means, Tukey-Kramer follow-up table

### KRUSKAL-WALLIS
- **Primary**: H, df, p-value, epsilon-squared
- **Additional**: medians/IQRs, Dunn-Holm follow-up, pairwise rank-biserial effects

### SPEARMAN / KENDALL / POINT-BISERIAL / PARTIAL PEARSON
- Reuses `AssociationRenderer` family.
- **Method-specific fields**: ties, positive-level orientation, controls, df, bootstrap CI

### PEARSON CHI-SQUARE
- **Primary**: chi-square, df, p-value, Cramer's V
- **Additional**: contingency table, expected-count diagnostics

### FISHER EXACT
- **Primary**: odds ratio, CI, p-value
- **Additional**: ordered 2x2 table, category orientation

### MCNEMAR
- **Primary**: paired proportion difference, CI, exact p-value
- **Additional**: 2x2 transition table, discordant pair counts

### LINEAR REGRESSION
- **Primary**: R-squared, adjusted R-squared, R-squared CI, model F / p-value
- **Additional**: coefficient table, standardized beta, VIF, residual diagnostics, influence diagnostics, covariance type

### LOGISTIC REGRESSION
- **Primary**: likelihood ratio test, McFadden pseudo-R2, event N
- **Additional**: OR table first, raw beta only in full mode, convergence, VIF/condition diagnostics, event orientation

### CRONBACH ALPHA
- **Primary**: alpha, CI
- **Additional**: item count, respondent N, corrected item-total correlations, alpha-if-deleted, inter-item summary, explicit limitation: alpha != validity/unidimensionality

### REPEATED-MEASURES ANOVA
- **Primary**: F, df, corrected p-value if applicable, partial eta-squared + CI
- **Additional**: complete/incomplete units, condition summaries, Mauchly test, Greenhouse-Geisser epsilon, pairwise t + dz + adjusted p-value

### FRIEDMAN
- **Primary**: Q, df, p-value, Kendall's W + CI
- **Additional**: condition medians/IQRs, complete units, pairwise Wilcoxon, rank-biserial CI, Holm p-value

### TWO-WAY FACTORIAL ANOVA
- **Primary table**: Factor A, Factor B, A×B interaction, F, df, p-value, partial eta-squared, CI
- **Additional**: cell summaries, estimated marginal means, simple effects, interaction contrasts, multiplicity policy
- Interaction visually distinct but not red/green-coded by significance.

### INTRACLASS CORRELATION (ICC)
- ALWAYS display definition before estimate.
- **Definition**: canonical ICC form, model, agreement vs consistency, single vs average
- **Primary**: ICC estimate, CI, F, df, p-value
- **Additional**: target/rater counts, BMS/JMS/EMS/WMS, variance components, complete-target accounting, negative estimate policy

### STUDY PLANNER
- **Display**: planning objective, supplied assumptions, target power / precision, alpha, required N, allocation / pairs
- Never imply observed post-hoc power.

### SENSITIVITY
- **Table**: scenario, method, estimand comparability, contrast comparability, status, decision
- Do not rank by p-value.

### PRACTICAL SIGNIFICANCE
- **Display**: researcher threshold, observed estimate, CI, threshold relationship, statistical significance separately.

### AUDIT
- **Operational semantics**: PASS = green, INCOMPLETE = amber, FAIL = red.
- These colors represent the audit verification process outcome, not scientific significance.

### REPRODUCIBILITY
- **Display**: package version, Python runtime, fingerprint availability, replay status, deterministic seed metadata. Full environment only in `detail="full"`.
