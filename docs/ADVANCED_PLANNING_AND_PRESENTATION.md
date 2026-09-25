# Advanced planning and presentation

Phase 12 adds planning and presentation contracts around the existing deterministic workflow. It
does not add a GUI, observed post-hoc power, automatic model selection, or journal certification.

## Statistical analysis plans

```python
plan = assistant.analysis_plan(
    draft,
    sensitivity_scenarios=[scenario],
    meaningful_threshold=threshold,
    multiplicity_policy="none_planned",
    report_style="apa",
)
```

`StatisticalAnalysisPlan` schema version 1 records the question, design, selected method and
rationale, alpha, confidence level, target quantities, complete-case rule, no-automatic-outlier
rule, ordered sensitivity scenarios, optional researcher threshold, multiplicity state, report
style, audit/reproducibility settings, planning declaration, warnings, limitations, and local
provenance. Status is `ready`, `needs_input`, or `unsupported`. Creation performs intake and
recommendation only; it executes no hypothesis test and reads no eventual p-value.

Multiplicity is `not_applicable`, `none_planned`, `unknown`, or `planned_method`. The last state
requires `multiplicity_method` text. That procedure is recorded as unsupported for execution;
PyAutoStat never substitutes or silently applies a correction.

The assistant records whether an analysis had already executed in that assistant instance.
`created_after_analysis=True` prevents a retrospective plan from being described as prospective.
When tracking is enabled, passing `previous_plan=` records immutable old/new payloads, changed
fields, and an optional researcher reason. A local event is never evidence of externally
authenticated preregistration. `assistant.plan_adherence(plan, result)` labels recorded fields
`matched`, `changed`, or `not_recorded`; it makes no misconduct inference.

## Prospective study planning

`StudyPlanner` is standalone and does not require a DataFrame:

```python
from pyautostat import StudyPlanner

planner = StudyPlanner()
power = planner.independent_mean_power(
    target_difference=5,
    sd_group1=10,
    sd_group2=12,
    alpha=0.05,
    target_power=0.80,
    allocation_ratio=1.0,  # n2 / n1
)
precision = planner.paired_mean_precision(
    sd_difference=5,
    confidence_level=0.95,
    target_half_width=2,
)
```

Independent planning uses `SE = sqrt(sd1**2/n1 + sd2**2/n2)`, Welch-Satterthwaite degrees of
freedom, and a noncentral t approximation for two-sided power. Precision uses the corresponding t
critical value. Paired planning uses the supplied paired-difference SD, `df=n-1`, and returns
`required_pairs`; it does not call a pair count total raw rows. Searches test integer sizes from
two through the explicit `max_n` or `max_pairs` bound (default 100,000). Infeasible bounded
requests return `status="unavailable"` with an actionable warning.

All anticipated differences and SDs are explicit researcher inputs. Negative anticipated
differences are treated by magnitude for two-sided power. Zero, Boolean, NaN, infinity, invalid
probabilities, nonpositive SD/precision/allocation, and invalid bounds are rejected. No API derives
an anticipated effect from an observed result, and there is no observed-power helper.

## Explicit paired mean analysis

Paired execution requires `design="paired"`, `estimand="mean"`, an explicit `unit_id`, and
exactly two observed conditions. `condition_order=(first, second)` fixes the contrast; otherwise
first observed condition order is recorded. Row order is never used to create pairs.

Duplicate usable unit/condition observations are blocked rather than averaged. Units missing one
condition are excluded as incomplete pairs and counted. At least two complete finite pairs and a
finite nonzero paired-difference SD are required. `scipy.stats.ttest_rel` supplies the test;
PyAutoStat reports first-minus-second mean paired difference, an analytical paired t interval, and
Cohen's dz (mean paired difference divided by the sample SD of paired differences). Aggregate
pair counts and the unit-ID column name are reported; identifier values are not published.

## Completeness and presentation

`assistant.reporting_completeness(report, style=...)` returns a deterministic, machine-readable
checklist. Item statuses are `present`, `missing`, `partial`, and `not_applicable`. For example, a
missing report field whose canonical analysis contains a value is a reporting defect, while the
currently unavailable Pearson interval is a backend limitation (`partial`). No numerical quality
score is produced. Completeness does not assess sampling, design truth, bias, or publication
quality.

`ResearchReport.to_html()`, `to_markdown()`, and `to_latex()` accept `style="general"`, `"apa"`,
or `"ieee"`. The style changes headings and concise presentation only. The report's JSON payload,
raw values, alpha, interval, warnings, and limitations remain identical. These are oriented
templates, not claims of universal APA/IEEE or journal compliance.

LaTeX is generated as inert text and is never compiled. User text escapes backslash, braces,
dollar, ampersand, hash, underscore, percent, tilde, and caret. `save_latex(path, overwrite=False)`
writes only after an explicit call and rejects an existing destination by default.

## Adapter snapshot

`assistant.session_snapshot(workflow, ...)` returns `ResearchSessionSnapshot` schema version 1.
It contains the current workflow, machine-renderable questions, state-valid action identifiers,
capabilities derived from the method registry, blockers/warnings, and any supplied plan, study
planning, sensitivity, practical-significance, completeness, report, audit, and reproducibility
records. It contains no DataFrame, participant IDs, callables, credentials, or environment
variables. A future interface must still submit choices to the core validators.

See [the capability matrix](FINAL_CAPABILITY_MATRIX.md) and
[scientific limitations](SCIENTIFIC_LIMITATIONS.md).
