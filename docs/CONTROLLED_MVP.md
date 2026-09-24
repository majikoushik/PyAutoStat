# Phase 10 controlled MVP

PyAutoStat provides a bounded, deterministic path from a pandas DataFrame and a declared
research question to an audited report and reproducibility metadata. It supports the cases in
the matrix below. It does not determine study design from values or choose among arbitrary
statistical models.

## Three user paths

### Profile a dataset

```python
profile = ResearchAssistant(df).profile()
```

This path needs no research question and performs no inferential test.

### Run a guided analysis

```python
assistant = ResearchAssistant(df)
workflow = assistant.run(
    objective="compare_groups",
    outcome="exam_score",
    predictor="teaching_method",
    estimand="mean",
    design="independent",
    variable_types={"exam_score": "continuous"},
)

print(workflow.status)
print(workflow.analysis.values["primary_estimate"])
print(workflow.report.to_html())
```

`run()` prepares and validates the question, recommends one compatible method, executes it once,
interprets the same `AnalysisResult`, builds the canonical report, audits all four in-memory
formats by default, and creates metadata-only reproducibility information. It writes no files and
does not call `reproduce()`.

### Continue after a clarification

```python
incomplete = assistant.run(
    objective="compare_groups",
    outcome="exam_score",
    predictor="teaching_method",
    estimand="mean",
    variable_types={"exam_score": "continuous"},
)
print(incomplete.status, incomplete.draft.questions)

revised = assistant.update_question(incomplete.draft, design="independent")
workflow = assistant.run(draft=revised)
```

Confirmed values stay in the immutable draft. A supplied `draft` or `specification` cannot be
combined with raw question arguments. This prevents silent precedence rules.

## Workflow result contract

`ResearchWorkflowResult` has schema version 1. Its object fields reuse the existing Phase 4–9
types: `specification`, `draft`, `recommendation`, `analysis`, `interpretation`, `report`, `audit`,
and `reproducibility`. It also provides optional `profile`, plus `missing_information`, `blockers`,
and `warnings`. Downstream fields remain `None` when their stage did not run. `to_dict()` and
`to_json()` are JSON safe and do not include the source DataFrame.

| Status | Meaning |
| --- | --- |
| `completed` | Calculation, interpretation, complete report, and audit all completed. |
| `partial` | Valid findings remain, but interpretation/report/audit is incomplete or audit was explicitly disabled. |
| `needs_input` | An essential researcher answer is missing; no test ran. |
| `data_limited` | The observed selected data cannot meet the method's current numerical policy; no test ran. |
| `unsupported` | The stated design, target, or variable combination has no compatible guided method; no substitute ran. |
| `failed` | A selected calculation did not yield a usable result, interpretation/report was unavailable, or audit found a contradiction. |

Component statuses retain their own vocabulary. For example, Pearson can have an available
analysis, partial interpretation and report, a passed consistency audit, and an overall partial
workflow because its guided result has no confidence interval.

By default `audit=True` and `fingerprint=True`. `audit=False` avoids rendering the four formats,
sets `workflow.audit` to `None`, and makes the workflow partial. `fingerprint=False` keeps the
reproducibility record but omits the data fingerprint. `include_profile=True` adds one Phase 3
profile to an inferential workflow. A descriptive workflow reuses its one executed profile.

## Optional Phase 11 follow-up

`run()` remains the base workflow and does not accept or generate sensitivity scenarios. After a
successful result, callers may explicitly invoke `sensitivity_analysis()` with an ordered list of
scenario specifications and `practical_significance()` with a researcher-defined threshold.
These calls do not replace `workflow.analysis`. Their recorded results may be supplied to
`report()`, `audit()`, and `reproducibility_record()` for additive Phase 11 content. The ordinary
Phase 10 path still performs one statistical execution and produces report and reproducibility
schema version 1.

See [the Phase 11 contract](ROBUSTNESS_AND_PRACTICAL_SIGNIFICANCE.md) for same versus different
estimands, supported quantities, provenance, and formal-equivalence limits.

## Guided method support

| Question | Required declaration | Executed method | Interpretation/report/audit | Main limit |
| --- | --- | --- | --- | --- |
| Dataset description | `objective="descriptive"` | `dataset_profile` | Complete when the profile is usable | Descriptive only; no inferential conclusion |
| Two-group mean comparison | quantitative outcome, exactly two groups, independent design, `mean` target | Welch independent t-test | Available | At least two values per group and representable variance/effect |
| Two-group distribution comparison | ordered numeric outcome, independent design, `distribution` target | Mann–Whitney U | Available | Rank-distribution target; no universal median claim |
| Three-or-more-group distribution comparison | ordered numeric outcome, independent design, `distribution` target | Kruskal–Wallis | Available | At least five values per group; omnibus only |
| Linear numerical association | two quantitative variables, independent rows, `linear` target | Pearson correlation | Partial because no CI is implemented | At least three complete varying pairs; no causal claim |
| Categorical association | two categorical variables, independent rows, categorical-independence target | Pearson chi-square | Available | At least two levels per axis and every expected count at least five |

Student t-test and standard one-way ANOVA retain execution adapters for existing advanced callers,
but the guided selector does not choose them. Spearman and Kendall remain descriptive
coefficients without guided inference. Paired, repeated, clustered, multi-group robust mean,
Fisher exact, regression, mixed model, and survival workflows are unavailable. Diagnostics never
change a declared mean target into a rank-distribution target.

## Failure mode matrix

| Condition | Detected at | Outcome | Next action | Test/report produced |
| --- | --- | --- | --- | --- |
| Empty DataFrame | construction | `InvalidDataError` | Supply a nonempty DataFrame | No / no |
| Unknown column or wrong declaration dtype | question intake | package exception | Correct the API input or declaration | No / no |
| Missing objective, roles, estimand, design, or ambiguous numeric type | question intake or recommendation | `needs_input` | Answer `draft.questions` and call `update_question()` | No / no |
| All-missing selected variable or one observed group | question intake | `data_limited` | Correct/filter source data and rebuild | No / no |
| Too few values, constant outcome, nonrepresentable spread | recommendation | `data_limited` | Review source values/sample and rescale when justified | No / no |
| Sparse contingency table | recommendation | `data_limited` | Use an appropriate externally validated sparse-table method | No / no |
| Paired, repeated, or clustered design | recommendation | `unsupported` | Use a design-compatible method outside this guided engine | No / no |
| Three-group mean target | recommendation | `unsupported` | Use a justified supported external mean model | No / no |
| Unsupported target/type combination or method preference | recommendation/API validation | `unsupported` or exception | Preserve the question and choose a compatible supported contract | No / no |
| Numerical backend failure | execution | `failed`, unavailable analysis exposed | Inspect warnings and data scale | Yes / no successful report |
| Pearson confidence interval unavailable | interpretation | `partial` | Report the available coefficient/p-value and limitation | Yes / partial |
| Invalid result metadata | interpretation/report | unavailable result or package exception | Correct the producing code/result | No fabricated report |
| Report mismatch | report construction | `ReportError` | Regenerate from the canonical result | No new calculation |
| Audit contradiction | audit | `failed` with findings | Inspect source-linked fields and regenerate | Existing report exposed; no rerun |
| Audit disabled | orchestration | `partial`, `audit=None` | Run with default audit for consistency checks | Yes / yes |
| Fingerprint unavailable | reproducibility metadata | warning in record | Supply supported scalar data or disable fingerprint explicitly | No replay |
| Changed-data replay | explicit `reproduce()` | mismatch unless explicitly allowed | Supply original data or acknowledge a changed-data rerun | Only explicit allowed replay |

## Export, privacy, and researcher responsibility

Use `workflow.report.save_html(path)`, `save_markdown`, `save_json`, or `save_csv_tables` to write
explicitly. Existing overwrite, HTML/Markdown escaping, CSV formula protection, and stable
filenames remain in force. `run()` never writes a file, uploads data, includes participant-level
rows in reports, or saves a ZIP. Reports can still disclose sensitive small aggregates.

The researcher must correctly declare design, measurement meaning, analysis target, and whether
the analysis is scientifically appropriate. An audit pass establishes internal consistency for
the implemented checks. A local ledger is not authenticated preregistration, a matching hash is
not proof of data authenticity, and a passing test suite is not universal scientific validation.

The declared dependency floors (`pandas>=1.0`, `numpy>=1.19`, `scipy>=1.5`) are older than the
local Phase 10 validation environment. Modern-version tests alone do not verify those floors;
minimum-dependency testing remains a release-readiness requirement.
