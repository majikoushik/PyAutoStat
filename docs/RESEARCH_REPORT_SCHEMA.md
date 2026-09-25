# Research report schema

The integrated workflow exposes this schema through `workflow.report`. The report is built once from
`workflow.analysis` and `workflow.interpretation`; `run()` does not introduce another report
format. Default workflow auditing renders the four existing formats in memory and compares them
with the canonical report. An audit failure changes the workflow status to `failed` while leaving
the report and findings inspectable. `audit=False` leaves `workflow.audit` as `None` and makes the
workflow partial. File output still requires an explicit `ResearchReport.save_*` call.

`ResearchAssistant.report(result, interpretation=None, sensitivity=None, practical_significance=None, title=None, include_figures=False)` returns a `ResearchReport`. It captures a JSON-safe snapshot without rerunning a test or writing a file. `report.to_dict()` returns a defensive copy. A report without follow-up analysis content remains schema version 1. Supplying sensitivity or practical-significance results uses additive schema version 2; existing version 1 fields retain their meanings.

| Field | Source and meaning |
| --- | --- |
| `status` | `complete`, `partial`, or `unavailable` from execution and interpretation completeness |
| `title` | Neutral default or caller-provided title; never used as a filename |
| `analysis` | execution result and specification; unavailable numerical values are omitted |
| `interpretation` | Matching interpretation structured interpretation and coded findings |
| `sections.research_question` | Declared objective, variables, estimand, design, description, and data dictionary |
| `sections.dataset` | Original, analyzed, and excluded row counts; available group order/sizes, contrast, and effective pair count |
| `sections.methods` | Actual selected method, rationale, null/alternative, alpha, confidence level, assumptions, and recorded bootstrap settings |
| `sections.diagnostics` | Recorded diagnostic metadata and interpretation assumption notes |
| `sections.results` | Reader-facing validated numbers; invalid or unavailable quantities are `null` |
| `sections.interpretation` | interpretation summary and qualified conclusions; no new reporting interpretation rules |
| `tables` | Stable IDs, titles, column names, rows of `{value, source}` cells |
| `figures` | Optional histogram-bin data specifications from an existing descriptive profile |
| `limitations`, `warnings` | Visible qualifications and source warnings |

Schema version 2 may also contain `sensitivity`, `practical_significance`, matching optional
`sections.sensitivity_analysis` and `sections.practical_significance`, and tables named
`sensitivity_scenarios` and `practical_significance`. The sensitivity table retains declared order
and shows rationale, planning status, method, execution status, comparability, estimate, secondary
p-value, analyzed rows, and warnings. Same- and different-estimand labels remain visible;
unsupported and failed scenarios are not removed. The practical section shows the named quantity,
estimate, researcher threshold/direction/unit/rationale, recorded CI, separate point and interval
relations, null-hypothesis significance, and qualified conclusion.

A partial or unavailable optional follow-up analysis section makes an otherwise usable combined report
partial, while preserving the base analysis. Follow-up report generation never executes a scenario
or recalculates a test. The auditor regenerates schema 2 from captured independent follow-up analysis source
records and detects altered scenario counts, status, comparability, estimates, threshold, and
relations.

Tables may include `sample_accounting`, `group_sizes`, `statistical_results`, `effect_estimates`, `confidence_intervals`, `descriptive_statistics`, and optional `histogram_N_bins`. CSV export returns one UTF-8 string per table ID. HTML and Markdown render the same table cells; JSON retains raw numeric values. Display formatting is centralized and never changes the underlying p-value. A computational p-value of zero displays as a qualified inequality.

| Method | Report content | Limit |
| --- | --- | --- |
| `dataset_profile` | Descriptive summaries and optional histogram bins | No inferential p-value; detected identifier labels omitted |
| `welch_t`, `student_t` | Mean difference, t/df/p, separate Cohen's d, separate intervals | Student is not automatically selected; equal-variance condition requires review |
| `mann_whitney_u` | U/p and rank-biserial effect/interval | No universal median-difference claim |
| `one_way_anova`, `kruskal_wallis` | Omnibus statistic/p and named effect | No pairwise conclusion; ANOVA is not automatically selected |
| `pearson_correlation` | r/p and effective pair count | Current guided result has no CI, so report is partial |
| `pearson_chi_square` | Chi-square/df/p and Cramer's V/interval | No cell-specific or causal conclusion |

Unavailable analyses yield `status="unavailable"` and no reader-facing numerical result table. Partial interpretations yield `status="partial"` with available numbers and explicit missing information. Contradictory row accounting and mismatched supplied interpretations raise `ReportError`. Supplied interpretations must match the current deterministic result exactly. provenance content references are separate from both report schema versions.

The static HTML has embedded CSS and no network dependency. All user text is HTML escaped; Markdown syntax and HTML-like text are escaped; CSV formula-like text is apostrophe-prefixed after leading whitespace. This is a defensive CSV policy, not a guarantee for every spreadsheet application. Save methods require an explicit destination, refuse overwrite by default, and derive CSV names from stable table IDs.

The report excludes the raw DataFrame and omits detected identifier category values in descriptive profiles, including unique-value categorical columns. Aggregate small cells can still be sensitive. No package/dependency version, execution timestamp, randomization, ethics approval, or preregistration is fabricated inside the report. The recorded method ID, seed, bootstrap count, specification, tables, and source references support the separate provenance consistency auditor; the optional Decision Ledger records only observed local events and is not authenticated history. The legacy `ReportGenerator` remains separate.
## Presentation and completeness

The canonical report payload and its schema version remain style-neutral. `to_html`,
`to_markdown`, and `to_latex` accept `general`, `apa`, or `ieee`; all read the same stored values.
The oriented styles are presentation aids and do not claim journal compliance. LaTeX output
escapes user-controlled special characters and is never compiled. `save_latex` follows the same
explicit-write and `overwrite=False` policy as other save methods.

Paired reports add aggregate `complete_pairs`, `total_units`, `incomplete_units`,
`excluded_units`, the complete-pair rule, condition order, and signed contrast. The unit-ID column
name may be stated in Methods, but identifier values are omitted.

`ReportingCompletenessResult` schema version 1 is separate from the report. It labels applicable
items `present`, `missing`, `partial`, or `not_applicable`, including the distinction between an
omitted available interval and a backend-unavailable Pearson interval. It has no quality score.
