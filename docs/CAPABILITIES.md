# Capabilities and workflow status

PyAutoStat provides a bounded, deterministic path from a pandas DataFrame and a declared research
question to an audited report and reproducibility metadata. Availability always depends on the
researcher supplying the correct design and on the selected data meeting the documented numerical
conditions.

## Supported user paths

### Profile a dataset

```python
profile = ResearchAssistant(df).profile()
```

Profiling needs no research question and performs no inferential test. It returns descriptive,
quality, variable-intelligence, and advisory resource metadata without sampling or changing the
source DataFrame.

### Run a guided analysis

```python
workflow = ResearchAssistant(df).run(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    estimand="mean",
    design="independent",
    variable_types={"score": "continuous"},
)
```

`run()` validates the question, recommends one compatible method, executes it once, interprets the
same `AnalysisResult`, builds the canonical report, audits its in-memory formats by default, and
creates metadata-only reproducibility information. It writes no files and does not replay the
analysis.

### Continue after missing information

When a required scientific fact is absent, `run()` returns `needs_input` without executing a test.
Confirmed values remain in the immutable draft:

```python
revised = assistant.update_question(incomplete.draft, design="independent")
workflow = assistant.run(draft=revised)
```

A supplied draft or specification cannot be combined with raw question arguments.

### Use expert interfaces

Advanced callers can prepare and serialize specifications, request a recommendation, execute a
validated specification, and invoke supported legacy analyzer methods directly. Explicit access
does not bypass numerical validation or turn an unsupported guided design into a supported one.
Student's pooled t-test and standard one-way ANOVA remain explicitly runnable but are not selected
automatically by the guided workflow.

## Workflow status contract

`ResearchWorkflowResult` uses these statuses:

| Status | Meaning |
| --- | --- |
| `completed` | Calculation, interpretation, complete report, and audit completed. |
| `partial` | Valid findings remain, but interpretation, reporting, or audit is incomplete, or audit was explicitly disabled. |
| `needs_input` | An essential researcher answer is missing; no test ran. |
| `data_limited` | The selected observations cannot meet the current numerical policy; no test ran. |
| `unsupported` | The stated design, target, or variable combination has no compatible guided method; no substitute ran. |
| `failed` | A selected calculation was unusable, interpretation/reporting was unavailable, or audit found a contradiction. |

Component records retain their own documented status vocabulary. For example, Pearson correlation
can have an available analysis and partial interpretation because no confidence interval is
implemented. `audit=False` also makes an otherwise successful workflow partial.

## Supported statistical analyses

| Objective and target | Design | Method | Access | Estimate and uncertainty | Main limit |
| --- | --- | --- | --- | --- | --- |
| Dataset description | Not applicable | Dataset profile | Guided and direct | Descriptive records; no inferential p-value | Descriptive only |
| Two-group population means | Independent | Welch t-test | Automatically guided | First-minus-second mean, analytical CI, Cohen's d | At least two usable values per group and representable spread |
| Pooled two-group population means | Independent | Student t-test | Explicitly runnable and available to sensitivity scenarios | First-minus-second mean, analytical CI, Cohen's d | Requires an explicit equal-variance justification |
| Two-condition paired population mean difference | Paired with explicit unit ID | Paired t-test | Automatically guided | First-minus-second paired mean, analytical CI, Cohen's dz | Unique unit/condition rows, at least two complete pairs, nonzero difference variance |
| Two-group rank distributions | Independent | Mann-Whitney U | Automatically guided | U, rank-biserial effect, optional bootstrap CI | Distribution target; no universal median claim |
| Standard multi-group population means | Independent | One-way ANOVA | Explicitly runnable and available to sensitivity scenarios | Omnibus F and eta-squared | Guided selector does not choose it; equal-variance assumptions require justification |
| Three-or-more rank distributions | Independent | Kruskal-Wallis | Automatically guided | Omnibus H, epsilon-squared, optional bootstrap CI | At least five usable values per group; no post-hoc comparisons |
| Linear numerical association | Independent rows | Pearson correlation | Automatically guided | r and p-value | Confidence interval unavailable, so interpretation is partial |
| Categorical independence | Independent rows | Pearson chi-square | Automatically guided when the expected-count policy passes | Chi-square, Cramer's V, optional bootstrap CI | At least two levels per axis and every expected cell at least five |
| Monotonic numerical association | Independent rows | Spearman and Kendall coefficients | Descriptive coefficient matrices only | Coefficients | Guided inferential p-values and intervals unavailable |

Diagnostics never silently change a declared mean target into a rank-distribution target. Group
and paired directions follow the recorded contrast. Missing rows use the documented
analysis-specific complete-case rules; no values are imputed and no outliers are deleted.

## Prospective study planning

`StudyPlanner` supports two-sided prospective planning:

| Family | Power planning | Precision planning | Required researcher assumptions |
| --- | --- | --- | --- |
| Independent means | Bounded Welch noncentral-t approximation | Bounded Welch t half-width | Target difference for power, two SDs, alpha or confidence, and allocation ratio |
| Paired means | Bounded paired noncentral-t calculation | Bounded paired t half-width | Target paired difference for power, paired-difference SD, and complete pairs |

Planning does not derive anticipated effects from observed results and does not report automatic
observed post-hoc power.

## Research workflow capabilities

| Capability | Current contract | Main limit |
| --- | --- | --- |
| Sensitivity analysis | Ordered researcher-declared scenarios; every attempt retained; estimand, pairing, and contrast comparability recorded | No p-value ranking, automatic scenario selection, or universal robustness score |
| Practical significance | Researcher-defined quantity, magnitude, direction, unit, and paired contrast where required | No universal threshold and no formal equivalence or noninferiority inference |
| Statistical analysis plan | JSON-safe plan, method rationale, missing-data rule, sensitivity scenarios, threshold, multiplicity state, reporting settings, and local revision record | Local record is not verified preregistration |
| Plan adherence | Structured comparison of recorded plan, result, performed sensitivity scenarios, and practical assessment | Makes no misconduct judgment and invents no reason |
| Reporting completeness | Machine-readable `present`, `missing`, `partial`, and `not_applicable` items | Not a study-quality or publication-readiness score |
| Research report | Canonical HTML, Markdown, JSON, CSV tables, and safe LaTeX; General, APA-oriented, and IEEE-oriented styles | No universal journal-compliance claim; no PDF or DOCX output |
| Audit | Canonical report/export consistency checks against captured source records | Checks consistency, not scientific truth |
| Reproducibility | Runtime metadata, optional dataset fingerprint, metadata-only package, and explicit supplied-data replay | Does not authenticate data and does not automatically replay follow-up scenarios |
| Decision ledger | Optional local observed-event sequence with content references | Not authenticated history or external preregistration |
| Session snapshot | JSON-safe questions, actions, capabilities, blockers, warnings, and supplied workflow records | No GUI and no statistical logic in adapters |

## Failure and unsupported boundaries

Common blockers remain explicit:

- Missing objective, variables, estimand, design, unit ID, or relationship target returns
  structured missing information rather than a guessed answer.
- All-missing selections, too few observations, constant outcomes, nonrepresentable spread,
  duplicate paired unit/condition rows, unusable pairs, or sparse contingency tables return a
  data-limited or unsupported result as documented.
- Repeated measures with more than two conditions, clustered models, mixed models, generic
  regression, survival analysis, causal inference, Welch ANOVA, exact sparse-table alternatives,
  broad multiplicity or post-hoc families, and formal equivalence or noninferiority tests are not
  implemented.
- Numerical backend failure never becomes a successful result. Invalid report metadata or audit
  contradictions remain visible and do not trigger an automatic rerun.

See [statistical validation](STATISTICAL_VALIDATION.md) for numerical policies and
[scientific limitations](SCIENTIFIC_LIMITATIONS.md) for the boundaries of research claims.

## Export, privacy, and researcher responsibility

Explicit `save_*` calls write reports and protect existing destinations by default. The guided
workflow does not upload data, save files, or include participant-level rows in reports. Aggregate
small cells can still be sensitive.

Researchers remain responsible for study design, measurement meaning, sampling assumptions,
scientific suitability, and appropriate external review. An audit pass, matching fingerprint, or
passing test suite does not establish scientific truth or data authenticity.
