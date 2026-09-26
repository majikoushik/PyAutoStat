# Sensitivity analysis and practical significance

PyAutoStat provides two optional follow-up operations for a validated `AnalysisResult`. It does not
change `ResearchAssistant.run()` and does not run hidden alternatives.

## Explicit sensitivity scenarios

```python
from pyautostat import SensitivitySpecification

pooled = SensitivitySpecification(
    name="pooled variance",
    specification=workflow.analysis.specification,
    method_id="student_t",
    rationale="Assess sensitivity to the pooled-variance assumption.",
    planning_status="planned",
    assumptions=("Equal population variances",),
)
sensitivity = assistant.sensitivity_analysis(
    workflow.analysis,
    scenarios=[pooled],
)
print(sensitivity.compare())
```

The supplied order is preserved and each scenario runs at most once. Completed, unavailable,
incompatible, and failed scenarios all remain in `scenario_results`. The base result remains the
primary analysis. PyAutoStat does not sort by p-value, select a lowest p-value, stop after a
significant result, or calculate a robustness score.

Student's pooled-variance t-test is available only as an explicit mean-difference sensitivity
scenario with an independent design and a declared equal-population-variance assumption. Its
statistic is calculated by the existing execution adapter. A Mann–Whitney scenario must use a
distribution estimand and is labelled `different_estimand` relative to a Welch mean comparison.
Its estimate is displayed, but no estimate change or direct robustness conclusion is calculated.
Paired scenarios require the same explicit unit-ID and two-condition contract as the primary
paired workflow. Repeated, clustered, coefficient-only, and unknown methods remain incompatible
or unavailable; no independent method is substituted.

For `same_estimand` scenarios, the result records point-estimate difference, direction, effective
sample-size change, interval availability, and descriptive interval overlap. A reversed two-group
contrast is normalized only in the comparison fields, with the sign transformation recorded;
the scenario's raw result is retained. Relative change is omitted when the base estimate is too
close to zero. Interval overlap is descriptive and is not a hypothesis test.

`verdict` and `compare()` narrate the stored primary and scenario values. Decision consistency uses
the declared primary alpha and only completed `same_estimand` scenarios. Other attempts remain
visible but are excluded from that decision. Mixed-estimand analyses receive an explicit warning
against direct numerical comparison; no-same-estimand and no-completed-scenario states are
descriptive or unavailable. The `ROBUST` display label means only matching reject/fail-to-reject
decisions among the supplied comparable methods and is always qualified: it does not establish
general robustness, equivalence, effect stability, or practical importance.

The first implementation supports same-DataFrame specification sensitivity. It does not create
changed datasets, remove outliers, winsorize, impute, merge categories, or alter missing-data
rules. Every result stores a dataset fingerprint and ordered scenario configuration without raw
observations.

## Researcher-defined meaningful thresholds

```python
from pyautostat import MeaningfulEffectThreshold

threshold = MeaningfulEffectThreshold(
    quantity="mean_difference",
    minimum_magnitude=5,
    direction="two_sided",
    unit="points",
    rationale="A smaller difference would not change the educational decision.",
)
practical = assistant.practical_significance(
    workflow.analysis,
    threshold=threshold,
)
print(practical.verdict)
```

Supported signed quantities are `mean_difference`, `cohens_d`, `pearson_r`, and
`rank_biserial`. Supported nonnegative quantities are `cramers_v`, `eta_squared`, and
`epsilon_squared`. The named quantity must exist in the result. Raw and standardized effects are
never converted or substituted. A known raw unit must match the threshold unit; when result-unit
metadata are absent, a supplied unit is preserved with a verification warning.

The result reports the point-estimate relation separately from the confidence-interval relation.
For a two-sided threshold of ±5, an interval entirely inside (-5, 5) is described as lying inside
the researcher-defined negligible region. This is not a formal equivalence conclusion. An
interval crossing a threshold produces an uncertain relation even if its point estimate exceeds
the threshold. A missing interval yields a `partial` assessment and no invented uncertainty.
Null-hypothesis significance is recorded separately and never defines practical importance.
`verdict` formats this validated assessment with a complete interval-region narrative and a ratio
of the recorded estimate magnitude to a positive declared threshold. It does not recalculate the
stored relations or turn statistical significance into a meaningful-effect decision. Zero
thresholds omit the ratio, missing intervals are labelled point-estimate-only, and unknown future
relation values receive an unavailable fallback.

`direction="positive"`, `"negative"`, and `"nonnegative"` provide one-sided descriptive
relations for appropriate metrics. Requests using `"equivalence"` or `"noninferiority"` return
`unsupported`: PyAutoStat does not implement TOST or noninferiority inference.
For a positive or negative paired mean-difference threshold, declare
`contrast_order=(first_condition, second_condition)`. The order identifies the intended signed
first-minus-second contrast. A missing or reversed declaration leaves that directional assessment
unavailable; a two-sided magnitude threshold can be used across contrast reversal.

## Provenance, reporting, audit, and reproduction metadata

When tracking is enabled before a sensitivity call, the ledger records the local plan, every
attempt, and every outcome. `planning_status` is caller supplied and defaults to `unknown`; a
local event is never described as external preregistration. Threshold declarations and revisions
retain previous and new values and an optional researcher rationale.

```python
report = assistant.report(
    workflow.analysis,
    sensitivity=sensitivity,
    practical_significance=practical,
)
audit = assistant.audit(report)
record = assistant.reproducibility_record(
    workflow.analysis,
    sensitivity=sensitivity,
    practical_significance=practical,
)
```

An ordinary integrated workflow report remains schema version 1. Supplying either follow-up analysis component creates
report schema version 2 with optional sections and source-linked tables. The auditor rebuilds the
canonical report without rerunning scenarios and detects changed thresholds, estimates,
relations, scenario status, omissions, and comparability. Reproducibility schema version 2 stores
scenario order, specifications, actual methods, statuses, seeds, fingerprint, and threshold.
`reproduce()` still replays only the base execution analysis on explicit request; it reports that
follow-up configurations were recorded but were not automatically replayed.

## Limits

- Sensitivity results describe only the scenarios supplied by the researcher.
- Different-estimand scenarios are supplementary analyses, not direct replications.
- Pearson uncertainty remains partial because the current backend has no Pearson interval.
- Changed-data sensitivity, cross-design paired/independent substitutions, exact sparse-table methods, and rank-correlation
  inference are not implemented.
- Thresholds express researcher context. PyAutoStat does not supply universal practical-importance
  cutoffs or generic small, medium, and large labels for this decision.
- Formal equivalence and noninferiority tests remain unsupported.
