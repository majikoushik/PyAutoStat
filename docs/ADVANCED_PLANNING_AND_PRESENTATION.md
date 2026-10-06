# Advanced planning and presentation

PyAutoStat provides planning and presentation contracts around the existing deterministic workflow. It
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
rationale, event/positive level, condition order, ordered predictors or controls, categorical
references, covariance choice, alpha, confidence level, target quantities, complete-case rule,
no-automatic-outlier rule, ordered sensitivity scenarios, optional researcher threshold,
multiplicity state, report style, audit/reproducibility settings, planning declaration, warnings,
limitations, and local provenance. Status is `ready`, `needs_input`, or `unsupported`. Creation
performs intake and recommendation only; it executes no hypothesis test and reads no eventual
p-value.

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
missing report field whose canonical analysis contains a value is a reporting defect, while an
inherently unavailable interval (such as small-sample Pearson correlation with $n \le 3$, where the
asymptotic Fisher-z transformation is mathematically unavailable) is a backend limitation (`partial`).
For $n > 3$, Pearson correlation reports an analytical Fisher-z asymptotic normal confidence interval.
No numerical quality score is produced. Completeness does not assess sampling, design truth, bias,
or publication quality.

`ResearchReport.to_html()`, `to_markdown()`, and `to_latex()` accept `style="general"`, `"apa"`,
or `"ieee"`. The style changes headings and concise presentation only. The report's JSON payload,
raw values, alpha, interval, warnings, and limitations remain identical. These are oriented
templates, not claims of universal APA/IEEE or journal compliance.

Console helpers (`ResearchAssistant.summarize()`, `ResearchWorkflowResult.explain()`,
`InterpretationResult.findings_plain`, practical `verdict`, and sensitivity `compare()`) are
presentation views over existing records. They do not alter plans, execute methods, or certify
scientific conclusions.

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

## End-to-end advanced research lifecycle example

Below is a complete, coherent example illustrating how all advanced lifecycle components connect in Python.
All steps beyond `assistant.run(...)` are optional progressive disclosures:

```python
import pandas as pd
from pyautostat import (
    MeaningfulEffectThreshold,
    ResearchAssistant,
    SensitivitySpecification,
    reproduce,
    save_bundle,
    show,
)

# 1. Initialize assistant and optionally enable local decision tracking
assistant = ResearchAssistant(df)
assistant.enable_tracking()
assistant.declare_planning("planned", reason="Protocol pre-specified before analysis")

# 2. Prospective study planning (sample size / precision before data collection)
planner = assistant.study_planner()
power_plan = planner.independent_mean_power(
    target_difference=2.5,
    sd_group1=3.0,
    sd_group2=3.0,
    target_power=0.80,
)
show(power_plan)

# 3. Formulate research question draft and record Statistical Analysis Plan
draft = assistant.prepare_question(
    objective="compare_groups",
    outcome="score",
    predictor="treatment",
    design="independent",
    estimand="mean",
)
threshold = MeaningfulEffectThreshold(
    quantity="mean_difference",
    minimum_magnitude=1.5,
    unit="points",
    rationale="Clinical minimum clinically important difference",
    planning_status="planned",
)
scenario = SensitivitySpecification(
    name="equal_variance_student_t",
    specification=draft.specification,
    method_id="student_t",
    rationale="Compare Welch t against equal-variance Student t assumption",
    assumptions=("equal population variance",),
    planning_status="planned",
)
plan = assistant.analysis_plan(
    draft,
    sensitivity_scenarios=[scenario],
    meaningful_threshold=threshold,
    multiplicity_policy="none_planned",
    report_style="apa",
)
show(plan)

# 4. Execute guided research workflow
workflow = assistant.run(draft=draft)
show(workflow)

# 5. Evaluate plan adherence
adherence = assistant.plan_adherence(plan, workflow.analysis)
show(adherence)

# 6. Run explicit sensitivity analysis
sensitivity = assistant.sensitivity_analysis(
    workflow.analysis,
    scenarios=[scenario],
)
show(sensitivity)

# 7. Evaluate practical significance against declared threshold
practical = assistant.practical_significance(
    workflow.analysis,
    threshold=threshold,
)
show(practical)

# 8. Assemble canonical research report with attached follow-ups
report = assistant.report(
    workflow.analysis,
    sensitivity=sensitivity,
    practical_significance=practical,
)

# 9. Verify reporting completeness and audit internal consistency
completeness = assistant.reporting_completeness(report, style="apa")
show(completeness)

audit = assistant.audit(report)
show(audit)

# 10. Capture reproducibility record, test explicit replay, and export package
record = assistant.reproducibility_record(
    workflow.analysis,
    sensitivity=sensitivity,
    practical_significance=practical,
)
show(record)

# Explicit replay with supplied data (zero automatic replay)
outcome = reproduce(record, data=df)
show(outcome)

# Optional reproducibility metadata package (replay metadata only; no raw data)
record.save_package("exports/reproducibility_package.zip", overwrite=True)

# 11. Capture UI-independent session snapshot
snapshot = assistant.session_snapshot(
    workflow,
    analysis_plan=plan,
    study_planning=power_plan,
    sensitivity=sensitivity,
    practical_significance=practical,
    reporting_completeness=completeness,
)
show(snapshot)

# 12. Package research bundle containing the enriched report with SHA-256 manifest
# Uses default lightweight formats ("html", "json", "csv"); pass formats=("html", "pdf", "docx", ...)
# to include optional PDF (requires Chromium) and DOCX (requires python-docx) exports.
save_bundle(report, "exports/research_bundle.zip", formats=("html", "json", "csv"), overwrite=True)
```

See [the capabilities and support matrix](CAPABILITIES.md),
[governance usability audit](GOVERNANCE_USABILITY_AUDIT.md),
and [scientific limitations](SCIENTIFIC_LIMITATIONS.md).
