# PyAutoStat — Phased Development Roadmap

**Status:** Locked planning baseline, 2026-09-21. This roadmap defines **planned work and acceptance criteria**, not shipped features. See [`PRODUCT_VISION.md`](PRODUCT_VISION.md) for mission, guardrails and intended experience; [`AGENTS.md`](AGENTS.md) for Codex working rules. `ROADMAP.md` is an existing older summary; reconcile it deliberately as implementation progresses rather than treating both as contradictory feature claims.

## 1. Delivery model

**Phase 0: Documentation and agent setup (this bundle).** Update root `AGENTS.md`, add product vision and this roadmap. No statistical API change, version bump or PyPI publication. Documentation does not establish that subsequent features are implemented.

**12 sequential development phases:** ask for a Codex prompt for **one phase at a time**, run code/tests, inspect the diff and completion report, address blockers, then authorize the next phase. Do not let Codex treat a future phase as permission to implement it early. Review phase interfaces when necessary, but protect the simple user-facing API.

**Milestones:** Phases 1–3 reliable foundation; 4–5 research decision engine; 6–8 analysis, interpretation and reporting; 9–10 audited, integrated MVP; 11–12 advanced research and GUI-ready expansion.

**Release control:** Phase completion does not automatically imply PyPI publication, a GitHub Release, a version bump or a stability claim. Those require a separate user-approved release decision.

### Every phase must deliver

- Focused implementation and an explanation of what changed; document deferred/out-of-scope functionality.
- Automated tests: numerical reference checks for statistical changes; edge cases, invalid requests, schema/compatibility tests as relevant.
- Updated public documentation/examples/changelog for **shipped** behavior, not aspirational claims.
- Verified lint, format, type, test, coverage and build checks where relevant and available; disclose blocked checks honestly.
- A Codex completion report with changed files, design decisions, test commands/results, limitations and acceptance-criterion status.

### UX constraints across all phases

1. DataFrame-only **profiling** needs no research question.
2. For inference, require objective, variable roles and unknown *essential* design details; infer only reliable observed metadata.
3. Use optional, surfaced defaults for alpha/report style/seeds where appropriate; do not default independence or causal validity.
4. Scripts return structured blockers/missing-information requests instead of interactive questions; GUI can render them later.
5. Prefer one small high-level API and serializable result models; advanced configuration stays optional.
6. No hidden mutation, outlier deletion, imputation, test switching or estimand change.

## 2. Phase-by-phase contract

### Phase 1 — Architecture, contracts and public API

**Goal:** Create the structural foundation without altering unrequested statistical behavior.

**Implement:** Audit current imports, modules and result schemas; define light typed/serializable question, design, recommendation, diagnostics, result, warning and missing-information contracts; specify high-level `ResearchAssistant` API (provisional naming may be finalized in this phase); separate UI-agnostic core from future presentation adapters; map existing analyzer/insight/report components to the future architecture. Add only enough scaffold to exercise models and construction; avoid feature-heavy implementations.

**Input/UX:** `ResearchAssistant(df)` and a minimal construction/configuration path should be intuitive. Profiling need not demand question input. Missing essential data should have a structured representation.

**Acceptance:** Existing API and test suite remain usable; schema objects validate and serialize predictably; one simple example demonstrates scaffolding honestly; no new advanced test selection is advertised. Document the final API contracts and migration policy.

**Exclude:** Rewriting statistical algorithms, generating full research reports, a GUI, additional method families.

### Phase 2 — Statistical correctness and regression baseline

**Goal:** Correct foundational methodological and numerical issues *before* broadening the feature set.

**Implement:** Review `hypothesis_tests(auto)` semantics and normality/Levene labels; prevent silent changes of estimand; specify method-specific prerequisites; validate existing statistics, effect-size directions, interval formulas, group order and numerical warnings. Distinguish unknown, violated and not-rejected assumptions. Define safe behavior for constant/tied/tiny/sparse/extreme datasets. Maintain compatibility where feasible; document corrections.

**Acceptance:** Every currently supported method has independently checked numerical examples and failure-case tests; unsuitable automatic selections do not silently proceed; existing regression tests pass or documented corrections update expectations. No claim of proof of normality from nonsignificant diagnostics.

**Exclude:** Large new method catalogue, GUI or reporting redesign.

### Phase 3 — Dataset Intelligence

**Goal:** Produce a useful general profile from a DataFrame alone.

**Implementation record (2026-09-22):** Dataset-only profiling additions are implemented on the `phase3-dataset-intelligence` branch; see [Phase 3 development record](docs/PHASE3_DEVELOPMENT_REPORT.md). Local validation and the documented Python 3.10 CI baseline must be considered separately before merge or release.

**Implement:** Consolidate dataset overview, descriptive statistics, missingness, duplicates, distribution summaries, histograms, outlier flags and correlations. Return advisory type/role hints including integer-coded categories, identifiers and dates; per-pair correlation sample sizes and method-appropriate diagnostics; optional data dictionary for names, types, units, valid ranges and missing codes. Track overlaps in quality flags and avoid misleading sample counts. Keep original input unchanged.

**Acceptance:** One-call profiling returns coherent structured results; all-missing, mixed-type, constant and nonstandard numeric columns are tested; no automatic outlier removal, imputation or variable reclassification; profile works without a research objective.

**Exclude:** Research-question interpretation or test recommendation.

### Phase 4 — Research Question Builder

**Goal:** Capture the smallest scientifically adequate analysis specification.

**Local implementation record (2026-09-22):** The Phase 4 question builder is implemented in the local `main` working tree, pending owner review and check-in. It captures objectives, roles, target, design, declarations, missing questions, and selected-data availability without method recommendation. See `API_REFERENCE.md` for the current API. This record does not imply a release or CI validation.

**Implement:** Objectives (descriptive, group comparison, association initially); explicit outcome/group/predictor roles; estimand and unit of analysis when material; design types (independent, paired, clustered/unsupported); alpha and planned/exploratory status; configuration save/load; user-friendly validation. Use DataFrame metadata to suggest, never certify, roles. Return concise machine-readable requests for unresolved essentials and preserve expert overrides.

**Acceptance:** Common specifications need only a few arguments; unknown pairing/design is not invented; invalid combinations give actionable feedback; round-trip configuration is stable; no terminal prompt required.

**Exclude:** Running all statistical methods or natural-language AI question parsing.

### Phase 5 — Research Design Guardian and Recommendation Engine

**Goal:** Recommend compatible analytical methods transparently or decline safely.

**Local implementation record (2026-09-22):** `ResearchAssistant.recommend_test()` and its deterministic capability registry are implemented in the local `main` working tree, pending owner review and check-in. It revalidates Phase 4 drafts, checks design and method feasibility, and returns structured recommendations without executing tests. See `API_REFERENCE.md` and `docs/STATISTICAL_VALIDATION.md`. This record does not imply a release or CI validation.

**Implement:** Explicit rule registry/decision tree keyed by objective, estimand, variable scale, group count and confirmed design. Return candidate/selected method, rationale, alternatives, diagnostics, blockers and warnings. Cover common independent means, paired means, multi-group means, rank-oriented questions, correlations and categorical association where corresponding execution exists or clearly flag planned-only execution. Define block/warn/unknown semantics and researcher override validation.

**Acceptance:** Expert-reviewed table-driven scenario tests cover valid, invalid and ambiguous cases; no normality-p-value-only method switching; tests never silently substitute a different estimand; recommendations are reproducible and explainable.

**Exclude:** Automatically interpreting unrestricted free text; claiming support for methods not executable.

### Phase 6 — Statistical Execution Engine

**Goal:** Implement the prioritized, recommended methods correctly under common result contracts.

**Local completion record (2026-09-22):** `ResearchAssistant.analyze()` now executes freshly validated runnable recommendations and returns a populated `AnalysisResult` linked to the specification and recommendation. It adapts the existing profile, independent-group, Pearson and chi-square backends; no new statistical algorithm or interpretation was added. See `API_REFERENCE.md` and `docs/STATISTICAL_VALIDATION.md`. Local implementation does not imply a release or CI validation.

**Implement:** Independent and paired comparisons, supported multi-group comparisons, justified rank-based methods, correlation inference and categorical association, selected post-hoc tests and multiple-comparison controls **only when their method contracts and backends are validated**. Use established scientific libraries and check support across declared dependency versions; add dependencies deliberately. Include primary estimate, effect measure, CI where valid, test statistic/df/p-value, group ordering, per-analysis sample sizes and assumptions. Clearly reject unsupported designs. Preserve existing public API via wrappers where sensible.

**Acceptance:** Numerical reference tests, effect-direction/CI tests, multiplicity tests, small/unbalanced/tied/missing cases, and end-to-end result-schema tests pass; same seed yields reproducible stochastic intervals; undefined results are explicit.

**Exclude:** Unvalidated complex mixed models, survival models or broad method proliferation.

### Phase 7 — Deterministic Interpretation Engine

**Goal:** Explain validated results without inventing scientific meaning.

**Local completion record (2026-09-22):** `ResearchAssistant.interpret(result)` now produces deterministic, JSON-safe findings from Phase 6 results for currently executable methods. Missing or contradictory essential fields are partial or unavailable; no statistical backend, report writer, or generative service runs. This local work is uncommitted and has not been validated by CI.

**Implement:** Test-specific narrative rules for question, methods, null/alternative, numeric finding, effect direction/magnitude, CI/uncertainty, assumption flags and limits. Keep current `InsightEngine` data-quality role distinct from inferential interpretation; both may contribute to reports. Structure narrative components before formatting into natural language. Allow style variants without changing statistical meaning.

**Acceptance:** Golden example tests for each method and p-value/CI/undefined branches; no “accept null,” p-value-as-probability, automatic causality, or fabricated practical importance; every displayed number matches canonical result values.

**Exclude:** LLM prose generation, fabricated citations or unsupported domain claims.

### Phase 8 — General Research Report Engine

**Goal:** Deliver a clear, consistent report from a validated analysis.

**Local completion record (2026-09-22):** `ResearchAssistant.report(result)` builds a canonical `ResearchReport` from the recorded Phase 6 analysis and matching Phase 7 interpretation. It provides structured sections and four in-memory export formats with explicit save methods, escaping, formula-text protection, and no new statistical calculations. This local work has not been committed or validated by CI.

**Implement:** One canonical analysis object feeds structured Methods/Results, dataset and inclusion/exclusion accounting, diagnostics, estimates, CIs, figures/tables, interpretation, limitations and warnings. General research style first; HTML, Markdown, JSON and CSV exports, with safe escaping and format-specific validation. Provide optional tables/figures by method and data availability; no invented study metadata. Keep legacy `ReportGenerator` working where practical.

**Acceptance:** Output numbers agree across JSON, tables and prose; HTML injection/spreadsheet-formula cases tested; unsupported/unknown information remains explicit; reference research example renders correctly.

**Exclude:** Guaranteed journal submission compliance; full APA/IEEE/DOCX/LaTeX/PDF machinery yet.

### Phase 9 — Decision Ledger, Result Auditor and Reproducibility

**Goal:** Make analyses inspectable, repeatable and resistant to reporting inconsistencies.

**Implement:** Complete the provenance structures introduced in Phase 1; record configurations, justified changes, preprocessing/exclusions, versions, seeds, fingerprints and result IDs. User-declared planned vs exploratory decisions remain distinguishable. Add an auditor checking canonical results against narrative/tables/export for names, groups, N, statistics, intervals and warnings. Reproduction package references authorized data and saves executable config/script; do not bundle raw data by default.

**Acceptance:** Deliberate mismatches are caught; replay matches computed results on same data/environment within documented numerical tolerance; ledger reflects changes honestly; no false preregistration or raw-data-sharing claim.

**Exclude:** Third-party proof of preregistration or guarantees of identical results across all future dependency releases.

### Phase 10 — Integrated Research Assistant: controlled MVP

**Goal:** Turn the components into an easy, cohesive user journey.

**Implement:** Refine `ResearchAssistant` quick-profile, guided `recommend`/`analyze` and `report` workflows, informative blockers, examples, docs, error messages, release checks, and consistent interfaces. Validate whole workflows, compatibility and audit/report integration. Define supported cases clearly.

**Acceptance scenario:** Given scores, a grouping variable, objective to compare means and confirmed independent design, the assistant profiles data, recommends a justified method, validates inputs, executes, explains effect/uncertainty, exports a consistent report and reproducibility record—without manual module orchestration. Ambiguous paired/clustered designs produce a blocker instead of a plausible but invalid result.

**Milestone:** First complete assistant suitable for controlled expert evaluation; not an unconditional claim of scientific validation or approval to publish a stable PyPI version.

### Phase 11 — Robustness and Practical Significance

**Goal:** Examine findings across defensible choices and meaningful-effect criteria.

**Implement:** Prespecified sensitivity specifications with provenance; compare common-estimand estimates/CIs and warn when alternatives answer different questions; researcher-defined meaningful-effect threshold and uncertainty comparison; correctly specified equivalence/noninferiority only when separately validated. Record all attempted alternatives rather than selecting significant ones.

**Acceptance:** Example scenarios demonstrate transparent variations, correct threshold logic, stable report audit and no p-value shopping or inappropriate equivalence claims.

**Exclude:** Automatic claims of robustness based on a proportion of significant tests.

### Phase 12 — Advanced research, reporting and GUI preparation

**Goal:** Extend into study planning, further methods and presentation options without compromising the core.

**Implement:** Statistical analysis plan and sample-size/precision planning; prioritized regression/repeated-measures/other method families only with complete design contracts; reporting completeness assessment; APA- and IEEE-oriented templates; optional DOCX/LaTeX/PDF export if justified and tested; stable GUI-facing schema serialization, documentation and examples. GUI implementation belongs to a distinct track.

**Acceptance:** Each new method has reference/edge-case/interpretation tests; templates do not imply generic journal compliance; CLI/notebook/GUI adapters can consume the same configuration and outputs without reimplementing statistics.

**Exclude:** Mandatory GUI, AI/cloud, arbitrary domain-specific causal claims or journal acceptance guarantee.

## 3. Cross-phase acceptance rubric

A phase is accepted only when the following relevant questions have evidence:

| Dimension | Acceptance question |
| --- | --- |
| Scientific correctness | Does the method answer the explicitly stated estimand and respect the design? |
| Numerical verification | Do independently referenced calculations, edge cases and interval/seed tests pass? |
| Safe uncertainty | Are assumptions, missing facts and unsupported scenarios represented honestly? |
| UX | Is the common path short, are unnecessary inputs avoided, and are errors actionable? |
| Architecture | Are contracts stable/serializable and statistical decisions independent of UI? |
| Compatibility | Are current public interfaces preserved or changed with documented migration? |
| Reproducibility | Are settings, input exclusions, random state and dependency context captured appropriately? |
| Reporting | Are every displayed number and claim supported by the canonical results/known facts? |
| Verification | Are actual test/lint/type/build results documented; are blocked checks disclosed? |
| Scope | Is the phase implemented without opportunistic future-phase work or unauthorized release? |

No single CI badge proves statistical correctness. Use reviewed scenario benchmarks, mathematical/reference validation and qualified claims before public stability announcements.

## 4. Future GUI contract

GUI is **not** a prerequisite. When built, it should:

1. Accept the same schema as the Python API, not re-implement recommendation rules.
2. Render `missing_information`, `blockers`, `warnings`, `recommendation`, `diagnostics`, `result`, `interpretation` and report metadata from structured objects.
3. Use progressive disclosure: brief, accessible questions for beginners; expert settings optional.
4. Present suggested variable types as editable suggestions rather than definitive truth.
5. Never start calculations when blockers remain; always preserve provenance on changes.
6. Treat user-provided text/data as untrusted; respect opt-in data export/privacy choices.

Exact field names will be established in Phase 1; these are semantic requirements, not a final schema.

## 5. How to request a Codex phase

For each phase request, provide Codex with (a) this file and `PRODUCT_VISION.md`, (b) current `AGENTS.md`, (c) the **single active phase number**, and (d) a phase-specific prompt. Ask it to review actual repository code before editing, detail any conflicts, implement only in-scope work, run checks, and return a completion report. Human review of code and test evidence precedes the next phase.

**Suggested completion report:** scope delivered; API/behavior changes; affected files; statistical validation; tests and exact commands/results; documentation changes; compatibility/migrations; open issues; out-of-scope items; no version or release changes unless expressly requested.

**Project state note:** Documentation phase 0 is not evidence that phases 1–12 have started. Update completion status only after implementation and review, not after writing a plan.
