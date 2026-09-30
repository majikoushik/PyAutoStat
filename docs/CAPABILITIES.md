# Capabilities and workflow status

PyAutoStat provides a bounded, deterministic path from a pandas DataFrame and a declared research
question to an audited report and reproducibility metadata. Availability always depends on the
researcher supplying the correct design and on the selected data meeting the documented numerical
conditions.

## Structured evidence and readable narration

Structured profiles, specifications, recommendations, analysis results, interpretations,
practical-significance assessments, sensitivity results, and reports remain the authoritative
records. Human-readable story, rationale, verdict, explanation, and executive-summary text is an
additive presentation layer over those stored values. It is deterministic, rule-based,
template-driven, reproducible for identical inputs, and entirely local; no generative AI, network
service, or cloud API is required. Narration does not recalculate statistics, select a different
method, alter an estimand, or create new evidence.

## Supported user paths

### Profile a dataset

```python
profile = ResearchAssistant(df).profile()
print(ResearchAssistant(df).summarize())
print(ResearchAssistant(df).summarize(mode="story"))
```

Profiling needs no research question and performs no inferential test. It returns descriptive,
quality, variable-intelligence, and advisory resource metadata without sampling or changing the
source DataFrame.
Numerical descriptions include linear-interpolation P5/P25/P50/P75/P95 values and a safeguarded
CV record. `frequency_table()` provides full categorical counts and valid/total percentages;
`cross_tab()` provides count, row-percent, column-percent, and total-percent matrices using paired
complete cases. Both retain explicit row accounting and bounded deterministic narration.
`summarize()` is the established plain-text view of the same profile and does not create a second
analysis. Opt-in `summarize(mode="story")` assembles recorded dimensions, strongest eligible
Pearson pair, data quality, normality-diagnostic coverage, and up to three explicitly prioritized
actions into connected prose. It does not infer causes, choose a method, or modify data.

`column_story()` narrates one existing descriptive-statistics mapping with scale-aware
mean/median comparison and seven deterministic skewness states. `InsightEngine.get_narrative()`
adds connected category prose; `get_summary()` preserves the full structured insight records and
adds JSON-safe narrative and ranked-action fields. Insight actions use the existing
`high`/`medium`/`low` severities without merging them with assumption severity terminology.
Story mode adds at most two categorical distribution observations. Profile frequency previews are
bounded to twenty levels; the dedicated API retains the full requested table. Declared ordinal
order, then ordered categorical dtype, governs cumulative percentages. Nominal and Boolean tables
do not receive cumulative percentages.

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
`workflow.explain()` formats the recorded workflow for console review, including recorded group
order and sample accounting where available; `method_label` and `findings_plain` provide readable
views while stable IDs and structured records remain available.

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
| One population mean versus reference | Independent | One-sample t-test | Automatically guided and direct | Observed-minus-reference mean difference, analytical CI, one-sample Cohen's d when defined | Explicit finite reference and continuous outcome required; zero variance leaves inferential quantities unavailable |
| Two-group population means | Independent | Welch t-test | Automatically guided | First-minus-second mean, analytical CI, Cohen's d | At least two usable values per group and representable spread |
| Pooled two-group population means | Independent | Student t-test | Explicitly runnable and available to sensitivity scenarios | First-minus-second mean, analytical CI, Cohen's d | Requires an explicit equal-variance justification |
| Two-condition paired population mean difference | Paired with explicit unit ID | Paired t-test | Automatically guided | First-minus-second paired mean, analytical CI, Cohen's dz | Unique unit/condition rows, at least two complete pairs, nonzero difference variance |
| Two-condition paired rank/distribution target | Paired with explicit unit ID | Wilcoxon signed-rank | Automatically guided and direct | Signed-rank statistic, matched-pairs rank-biserial effect; CI explicitly unavailable | `wilcox` zero policy; not universally a median test; location-shift reading needs symmetry |
| Two-group rank distributions | Independent | Mann-Whitney U | Automatically guided | U, rank-biserial effect, optional bootstrap CI | Distribution target; no universal median claim |
| Multi-group population means | Independent | Welch one-way ANOVA + Games-Howell | Automatically guided | Omnibus Welch F/df/p, group means, every pairwise mean difference with simultaneous CI and adjusted p | Positive finite variance required in every group; no global standardized effect claimed |
| Standard multi-group population means | Independent | One-way ANOVA + Tukey-Kramer | Explicitly runnable and available to sensitivity scenarios | Omnibus F/eta-squared and every pairwise mean difference with simultaneous CI and adjusted p | Guided selector does not choose it; equal-variance assumptions require justification |
| Three-or-more rank distributions | Independent | Kruskal-Wallis + Dunn-Holm | Automatically guided | Omnibus H/epsilon-squared and every pairwise mean-rank contrast with adjusted p and rank-biserial effect | At least five usable values per group; pairwise effect intervals unavailable |
| Linear numerical association | Independent rows | Pearson correlation | Automatically guided | r and p-value | Confidence interval unavailable, so interpretation is partial |
| Continuous conditional mean | Independent rows | Ordinary least-squares linear regression | Automatically guided for `objective="regression"` and direct | Coefficients/SE/t/p/CI, model F, R-squared/adjusted R-squared, residual error, continuous-predictor standardized beta | Additive main effects only; associations are noncausal and fit is not out-of-sample validation |
| Binary event probability/odds | Independent rows | Binary logistic regression | Guided `objective="regression"` | Likelihood fit, coefficients/SE/z/p/CI, odds ratios/CI, convergence and collinearity diagnostics | Exactly two outcome levels and explicit orientation; no classification metrics or causal claim |
| Paired binary marginal event proportions | Paired with explicit unit ID | Exact two-sided McNemar | Automatically guided for `estimand="proportion"` | Transition table, exact p, paired proportion difference and paired-unit bootstrap CI | Exactly two conditions and one usable binary response per unit/condition |
| Binary/continuous symmetric association | Independent rows | Point-biserial correlation | Automatically guided for an explicit association target | r_pb/p, explicit 0/1 coding and paired-observation bootstrap CI | Positive level determines sign; not substituted for a group mean test |
| Partial linear association | Independent rows | Partial Pearson correlation | Guided with explicit controls | Partial r/t/df/p and complete-row refitting bootstrap CI | Quantitative controls only; conditioning is not causal deconfounding |
| Multi-item internal consistency | Researcher-declared item set | Cronbach's alpha | Focused `reliability()` facade, guided `objective="reliability"`, and direct | Alpha, deterministic respondent-row bootstrap CI, item/deletion diagnostics, inter-item matrix | No scale discovery, dimensionality/validity claim, or universal adequacy cutoff |
| Categorical independence | Independent rows | Pearson chi-square | Automatically guided when the expected-count policy passes | Chi-square, Cramer's V, optional bootstrap CI | At least two levels per axis and every expected cell at least five |
| Monotonic numerical association | Independent rows | Spearman rank correlation | Automatically guided and direct | rho, p-value, deterministic paired-observation bootstrap CI | Ties allowed; does not establish linearity or causation |
| Sparse 2x2 categorical independence | Independent rows | Fisher's exact test | Automatically guided fallback and direct | Two-sided p and ordered sample odds ratio; CI explicitly unavailable | Exactly 2x2; no category collapsing or RxC extension |
| Kendall ordinal concordance | Independent rows | Kendall tau-b | Descriptive matrix and explicit guided inferential preference | Tau-b/p, ties and paired-observation bootstrap CI | Spearman remains the monotonic default; tau is not variance explained |
| Repeated continuous mean outcome (3+ conditions) | Repeated with explicit unit ID | One-way repeated-measures ANOVA | Automatically guided for estimand="mean" | Omnibus F/df/p, condition means/SDs, partial eta-squared, Mauchly sphericity, Greenhouse-Geisser corrected df/p, and every pairwise mean difference with analytical CI and Holm-adjusted p | At least three conditions, at least three complete units, one observation per unit-condition; missingness is complete-case only |
| Repeated rank/distribution target (3+ conditions) | Repeated with explicit unit ID | Friedman rank-sum test | Automatically guided for rank/distribution targets | Omnibus Q/df/p, condition medians/IQRs, Kendall's W concordance, and every pairwise paired Wilcoxon signed-rank test with rank-biserial effect and Holm-adjusted p | At least three conditions, at least three complete units, ordered numeric outcome required (no auto-encoding of text labels); does not universally test medians |

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
| Sensitivity analysis | Ordered researcher-declared scenarios; every attempt retained; estimand, pairing, and contrast comparability recorded; deterministic consistency verdict and descriptive `compare()` output | A `ROBUST` label means only matching decisions among supplied comparable methods; no p-value ranking, automatic scenario selection, or universal robustness score |
| Practical significance | Researcher-defined quantity, magnitude, direction, unit, and paired contrast where required; deterministic interval-region and ratio narration | No universal threshold and no formal equivalence or noninferiority inference |
| Statistical analysis plan | JSON-safe plan, method rationale, missing-data rule, sensitivity scenarios, threshold, multiplicity state, reporting settings, and local revision record | Local record is not verified preregistration |
| Plan adherence | Structured comparison of recorded plan, result, performed sensitivity scenarios, and practical assessment | Makes no misconduct judgment and invents no reason |
| Reporting completeness | Machine-readable `present`, `missing`, `partial`, and `not_applicable` items | Not a study-quality or publication-readiness score |
| Research report | Canonical HTML, Markdown, JSON, CSV tables, and safe LaTeX; General, APA-oriented, and IEEE-oriented styles; deterministic HTML executive summary from stored scale, analyses, effect narration, quality, diagnostics, and supplied follow-up assessments | No universal journal-compliance claim; no PDF or DOCX output |
| Recommendation explanation | On-demand `rationale_text`/`explain()` with method-specific why-this, why-not, and researcher-verification sections while retaining the structured decision trace | Optional diagnostics are narrated only when explicitly supplied; prose does not certify design facts or change method selection |
| Audit | Canonical report/export consistency checks against captured source records | Checks consistency, not scientific truth |
| Reproducibility | Runtime metadata, optional dataset fingerprint, metadata-only package, and explicit supplied-data replay | Does not authenticate data and does not automatically replay follow-up scenarios |
| Decision ledger | Optional local observed-event sequence with content references | Not authenticated history or external preregistration |
| Session snapshot | JSON-safe questions, actions, capabilities, blockers, warnings, and supplied workflow records | No GUI and no statistical logic in adapters |

## Failure and unsupported boundaries

Common blockers remain explicit:

- Missing objective, variables, estimand, design, unit ID, or relationship target returns
  structured missing information rather than a guessed answer.
- All-missing selections, too few observations, constant outcomes, nonrepresentable spread,
  duplicate paired unit/condition rows, unusable pairs, or unsupported sparse tables return a
  data-limited or unsupported result as documented. Sparse 2x2 tables use Fisher; larger sparse
  tables remain unsupported.
- Clustered models, mixed-effects models, mixed ANOVA, factorial repeated measures, multinomial,
  ordinal, count, regularized, interaction, and polynomial regression, survival analysis, causal
  inference, Fisher extensions beyond 2x2,
  broader multiplicity families, and formal equivalence or noninferiority tests are not
  implemented.
- Reliability is limited to ordinary covariance-based Cronbach's alpha. Omega, factor analysis,
  PCA, ordinal/polychoric alpha, split-half, test-retest, inter-rater, and measurement-invariance
  workflows are not implemented.
- Numerical backend failure never becomes a successful result. Invalid report metadata or audit
  contradictions remain visible and do not trigger an automatic rerun.

See [statistical method contracts](STATISTICAL_METHOD_CONTRACTS.md) for full scientific contracts,
[effect-size confidence interval gaps](EFFECT_SIZE_CI_GAPS.md) for uncertainty classifications,
[statistical validation](STATISTICAL_VALIDATION.md) for numerical policies, and
[scientific limitations](SCIENTIFIC_LIMITATIONS.md) for the boundaries of research claims.

## Export, privacy, and researcher responsibility

Explicit `save_*` calls write reports and protect existing destinations by default. The guided
workflow does not upload data, save files, or include participant-level rows in reports. Aggregate
small cells can still be sensitive.

Researchers remain responsible for study design, measurement meaning, sampling assumptions,
scientific suitability, and appropriate external review. An audit pass, matching fingerprint, or
passing test suite does not establish scientific truth or data authenticity.
