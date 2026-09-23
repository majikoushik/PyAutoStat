# Phase 8 research report schema

Phase 10 exposes this unchanged schema through `workflow.report`. The report is built once from
`workflow.analysis` and `workflow.interpretation`; `run()` does not introduce another report
format. Default workflow auditing renders the four existing formats in memory and compares them
with the canonical report. An audit failure changes the workflow status to `failed` while leaving
the report and findings inspectable. `audit=False` leaves `workflow.audit` as `None` and makes the
workflow partial. File output still requires an explicit `ResearchReport.save_*` call.

`ResearchAssistant.report(result, interpretation=None, title=None, include_figures=False)` returns a `ResearchReport`. It captures a JSON-safe snapshot without rerunning a test or writing a file. `report.to_dict()` returns a defensive copy. Schema version is 1.

| Field | Source and meaning |
| --- | --- |
| `status` | `complete`, `partial`, or `unavailable` from execution and Phase 7 interpretation completeness |
| `title` | Neutral default or caller-provided title; never used as a filename |
| `analysis` | Phase 6 result and specification; unavailable numerical values are omitted |
| `interpretation` | Matching Phase 7 structured interpretation and coded findings |
| `sections.research_question` | Declared objective, variables, estimand, design, description, and data dictionary |
| `sections.dataset` | Original, analyzed, and excluded row counts; available group order/sizes, contrast, and effective pair count |
| `sections.methods` | Actual selected method, rationale, null/alternative, alpha, confidence level, assumptions, and recorded bootstrap settings |
| `sections.diagnostics` | Recorded diagnostic metadata and Phase 7 assumption notes |
| `sections.results` | Reader-facing validated numbers; invalid or unavailable quantities are `null` |
| `sections.interpretation` | Phase 7 summary and qualified conclusions; no new reporting interpretation rules |
| `tables` | Stable IDs, titles, column names, rows of `{value, source}` cells |
| `figures` | Optional histogram-bin data specifications from an existing descriptive profile |
| `limitations`, `warnings` | Visible qualifications and source warnings |

Tables may include `sample_accounting`, `group_sizes`, `statistical_results`, `effect_estimates`, `confidence_intervals`, `descriptive_statistics`, and optional `histogram_N_bins`. CSV export returns one UTF-8 string per table ID. HTML and Markdown render the same table cells; JSON retains raw numeric values. Display formatting is centralized and never changes the underlying p-value. A computational p-value of zero displays as a qualified inequality.

| Method | Report content | Limit |
| --- | --- | --- |
| `dataset_profile` | Descriptive summaries and optional histogram bins | No inferential p-value; detected identifier labels omitted |
| `welch_t`, `student_t` | Mean difference, t/df/p, separate Cohen's d, separate intervals | Student is not automatically selected; equal-variance condition requires review |
| `mann_whitney_u` | U/p and rank-biserial effect/interval | No universal median-difference claim |
| `one_way_anova`, `kruskal_wallis` | Omnibus statistic/p and named effect | No pairwise conclusion; ANOVA is not automatically selected |
| `pearson_correlation` | r/p and effective pair count | Current guided result has no CI, so report is partial |
| `pearson_chi_square` | Chi-square/df/p and Cramer's V/interval | No cell-specific or causal conclusion |

Unavailable analyses yield `status="unavailable"` and no reader-facing numerical result table. Partial interpretations yield `status="partial"` with available numbers and explicit missing information. Contradictory row accounting and mismatched supplied interpretations raise `ReportError`. Supplied interpretations must match the current deterministic result exactly. Phase 9 content references are separate from this version 1 report schema.

The static HTML has embedded CSS and no network dependency. All user text is HTML escaped; Markdown syntax and HTML-like text are escaped; CSV formula-like text is apostrophe-prefixed after leading whitespace. This is a defensive CSV policy, not a guarantee for every spreadsheet application. Save methods require an explicit destination, refuse overwrite by default, and derive CSV names from stable table IDs.

The report excludes the raw DataFrame and omits detected identifier category values in descriptive profiles, including unique-value categorical columns. Aggregate small cells can still be sensitive. No package/dependency version, execution timestamp, randomization, ethics approval, or preregistration is fabricated inside the report. The recorded method ID, seed, bootstrap count, specification, tables, and source references support the separate Phase 9 consistency auditor; the optional Decision Ledger records only observed local events and is not authenticated history. The legacy `ReportGenerator` remains separate.
