# PyAutoStat 1.0.0 release notes

PyAutoStat 1.0.0 is the stable API milestone for its bounded, deterministic research-analysis
workflow. It promotes the already implemented and validated product surface to a documented 1.x
compatibility commitment; it does not add statistical methods or broaden scientific claims.

## Why 1.0

The package now has a coherent beginner path, explicit advanced contracts, 24 registered
statistical methods, deterministic interpretation, multi-format reporting, audit and replay
records, independent numerical validation, a cross-platform CI matrix, and a defined public API
and deprecation policy. Stable status means documented compatibility, not universal statistical
or scientific validity.

## Canonical API

```python
from pyautostat import ResearchAssistant

assistant = ResearchAssistant(df)
workflow = assistant.compare_means("score", by="group")
print(workflow.brief())
```

Use `profile()` to understand a DataFrame, `compare_means()` and `correlate()` for ordinary
beginner tasks, and `run()` for explicit advanced designs and estimands. Constructor
`AnalysisOptions` provides reusable immutable defaults. `brief()`, `type_guidance()`, `explain()`,
`apa_statement()`, and `show()` are views of the same stored result.

## Supported scope

The 24 registered method IDs remain:

`welch_t`, `student_t`, `paired_t`, `one_sample_t`, `mann_whitney_u`,
`wilcoxon_signed_rank`, `welch_anova`, `one_way_anova`, `kruskal_wallis`,
`pearson_correlation`, `spearman_correlation`, `kendall_tau_b`,
`point_biserial_correlation`, `partial_pearson_correlation`, `pearson_chi_square`,
`fisher_exact`, `mcnemar`, `linear_regression`, `logistic_regression`, `cronbach_alpha`,
`repeated_measures_anova`, `friedman_test`, `two_way_anova`, and
`intraclass_correlation`.

Exact estimands, assumptions, effects, intervals, exclusions, and limits are documented in
[Statistical Method Contracts](STATISTICAL_METHOD_CONTRACTS.md).

## Stability guarantees

- All 97 documented top-level exports are retained for 1.x.
- `InsightEngine` and `ReportGenerator` remain `COMPATIBILITY_STABLE` without runtime deprecation
  warnings.
- Patch releases provide compatible fixes, including documented scientific-correctness and
  security corrections.
- Minor 1.x releases may add backward-compatible capabilities and formal deprecations.
- Intentional breaking API or stable-schema changes require 2.0.0.

See [Public API 1.0](PUBLIC_API_1_0.md) and [API stability](API_STABILITY.md).

## Validation scope

The independent validation framework represents all 24 registered methods with evidence levels,
reference provenance, tolerances, orientation checks, and sample-accounting invariants. Deferred
Level D quantities are disclosed rather than treated as independently validated. Passing the
framework is evidence for the recorded cases and fields; it is not proof of scientific truth or
of validity for a particular study.

## Reporting and reproducibility

The core package provides terminal, static HTML, Markdown, JSON, CSV-table, and safe LaTeX output.
Interactive HTML, PDF, DOCX, and static figures remain optional. Research bundles use SHA-256
manifest integrity checks; those checks are not digital signatures. Reports, audits, and exports
consume recorded values without statistical recalculation, and raw DataFrames are excluded by
default.

Analysis plans, prospective planning, sensitivity scenarios, meaningful-effect thresholds,
reporting completeness, decision ledgers, reproducibility records, explicit replay, and session
snapshots remain advanced stable APIs with their documented evidentiary limits.

## Known limitations

The catalogue intentionally excludes mixed-effects and clustered models, mixed or factorial
repeated-measures ANOVA, GEE, survival and causal inference, automated imputation/outlier removal,
automatic predictor/model selection, formal equivalence/noninferiority tests, and observed
post-hoc power. Design facts and scientific meaning remain researcher responsibilities. Optional
exporters may require browser or document dependencies described in their focused guides.

## Install

```bash
pip install pyautostat==1.0.0
```

Optional extras are installed only when needed, for example `pyautostat[pdf]`,
`pyautostat[docx]`, `pyautostat[report]`, or `pyautostat[figures]`.

For upgrade guidance, see [Migration to 1.0](MIGRATION_TO_1_0.md).
