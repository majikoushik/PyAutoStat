# PyAutoStat — Locked Product Vision

**Status:** Locked destination as agreed on 2026-09-21. This describes the **intended product**, not the capabilities already shipped. Development sequence and acceptance gates are in [`DEVELOPMENT_ROADMAP.md`](DEVELOPMENT_ROADMAP.md). Agent implementation rules are in [`AGENTS.md`](AGENTS.md).

## 1. Identity and mission

**Product:** PyAutoStat — An Explainable and Reproducible Research Analysis Assistant.

**Promise:** *From Research Question to Reproducible Statistical Report.*

PyAutoStat helps a researcher select a method appropriate to a defined question and study design, execute validated statistical calculations, interpret evidence and uncertainty accurately, and produce an auditable, publication-oriented report. It begins as an open-source, Python-first, deterministic library **without generative AI**. The later GUI is a separate interface over the same core engine.

**Defining principle:** Do not merely calculate statistics; help researchers make, justify, verify and reproduce analytical decisions.

**Audience:** Students, researchers, analysts and methodologically experienced users who need a guided yet inspectable workflow. A guided tool supports—not replaces—researcher judgment, statistical expertise, study-design knowledge or editorial review.

## 2. Primary workflow

```text
DataFrame / optional analysis plan
          |
          v
Dataset profile and variable-type suggestions
          |
          v
Research objective + outcome/predictors + study-design facts
          |
          v
Research Design Guardian + explainable recommendation
          |
          v
Method-specific diagnostics -> validated execution
          |
          v
Structured estimates + uncertainty + warnings
          |
          v
Rule-based interpretation + results audit
          |
          v
Research report + decision ledger + reproducibility manifest
```

Provenance accompanies the entire pipeline, not just export. Researchers can revise a configuration and rerun; revisions remain distinguishable from the original analysis. The same validated result record feeds all output formats.

## 3. Three user modes, one statistical engine

| Mode | Minimum input | Output and behavior |
| --- | --- | --- |
| Quick profile | A valid pandas DataFrame | Dataset summary, missingness, duplicates, descriptive statistics, distributions, outlier flags, appropriate correlations and advisory variable roles. No research question required. |
| Guided research | DataFrame + research objective + relevant variables; essential study-design facts when missing | Structured recommendation with rationale, blockers, diagnostics, validated test, interpretation and report. Ask only questions that materially affect validity. |
| Expert | Guided inputs + explicitly selected advanced options | Transparent method overrides, estimands, alpha, adjustments, meaningful-effect thresholds, sensitivity specifications and advanced reporting when supported. Invalid combinations remain blocked. |

### Proposed ease-of-use experience — not current API

```python
from pyautostat import ResearchAssistant

assistant = ResearchAssistant(df)
profile = assistant.profile()

result = assistant.analyze(
    outcome="exam_score",
    group="teaching_method",
    objective="compare_means",
    design="independent",
)
result.summary()
result.report("research_report.html")
```

Phase 1 defines the precise API and naming. Profiling should be effortless; hypothesis testing cannot responsibly be reduced to `analyze(df)` when the question or design is unknown. In noninteractive Python code, unresolved essentials are returned as structured requirements or actionable errors, **not surprise terminal prompts**. A future GUI renders those same requirements as concise controls.

### Input policy

- **Infer safely:** row count, column names, pandas dtypes, observed group count, numerical summaries, missingness, duplicate candidates and advisory type/role suggestions.
- **Request when necessary:** research objective, outcome and explanatory variables, quantity of interest (estimand), independence/paired/clustered structure, unit of analysis and relevant study-design facts. Ask in ordinary language and only when material.
- **Default transparently:** optional parameters such as alpha=0.05 when appropriate, report style, RNG seed and bootstrap count; surface defaults in results and let users override supported settings.
- **Never guess:** randomization, causal identification, independence, measurement validity, a scientifically meaningful effect, missing-data mechanism, verified errors, or a journal's exact formatting requirements.
- **Never alter silently:** original data, outlier exclusions, imputation, scale, alpha, chosen outcome, estimand or method. Document analysis-specific row exclusions and request explicit authorization for discretionary preprocessing.

## 4. Product capabilities

### 4.1 Dataset intelligence

Dataset shape and schema; continuous/categorical/ordinal/identifier/date suggestions; descriptive summaries; missing-data patterns and per-analysis usable counts; duplicate warnings; IQR/Z-score/MAD flags; distributions and diagnostic visualizations; Pearson/Spearman/Kendall matrices with appropriate pairwise sample sizes and inferential support where implemented; optional data dictionary (units, descriptions, missing codes and valid ranges). No detected category or outlier causes automatic deletion or transformation.

### 4.2 Question and design intake

Capture objective, hypotheses where needed, variable roles, estimand, study design, sampling unit, data arrangement and exploratory/confirmatory status. Keep a minimal user path and allow structured configurations to be saved/reloaded. Never infer pairing, nesting or causal design from column values alone.

### 4.3 Research Design Guardian

Identify known-invalid test/design combinations, check required fields and method preconditions, and distinguish confirmed, violated, unresolved and inapplicable assumptions. Block known-invalid analyses, ask for genuinely necessary unknown information, and report qualified warnings when verification is impossible. Explain the nature of a design problem and possible next steps without inventing facts.

### 4.4 Explainable Recommendation Engine

Use research question and estimand **before** choosing a statistical test. Return candidate/recommended method, rationale, alternatives, diagnostic requirements, limitations and any blocker. A p-value from a normality screen must never silently change the estimand (e.g., mean difference to rank-based tendency). Do not claim a single universally correct test where design or objective admits several justified methods.

Initial method families, subject to implementation/validation: independent-group mean comparisons (e.g., Welch t), paired mean differences (paired t), multi-group mean comparisons (appropriate ANOVA variants), rank-based comparisons when aligned with the estimand, numerical association (Pearson/Spearman/Kendall), categorical association (chi-square or validated exact alternatives). Complex models remain explicitly unsupported until added.

### 4.5 Validated execution

Use established numerical backends where appropriate. Report test/statistic/df/p-value as applicable, primary estimate, confidence interval when supported, effect-size definition, sample/group sizes, missing-row policy, group ordering, diagnostic results and limitations. Support justified multiple-comparison adjustments and post-hoc tests within the planned scope. Missing or undefined computations are explicit, never made up.

### 4.6 Interpretation Engine

Deterministic method-specific templates turn structured results into qualified plain-language, research-oriented and business-oriented descriptions. Explain what the hypothesis test addresses, direction/magnitude, uncertainty, assumptions and practical interpretation when justified. No universal `p < 0.05 = meaningful` template. No “accept the null,” proof of normality, or causation from correlation. Unknown information stays unknown.

### 4.7 Research report engine

One canonical result record produces HTML, Markdown, CSV/JSON outputs first; APA-oriented, IEEE-oriented, general research and business presentation templates later. Standard sections: research question, design and sample, dataset and data-quality accounting, methods/reasons, diagnostics, numerical results, effect estimates/uncertainty, interpretation, limitations, provenance. DOCX/LaTeX/PDF are advanced optional exports; journal-specific compliance is never guaranteed.

### 4.8 Decision Ledger and reproducibility

Record configuration, modifications and justifications, preprocessing, exclusions, software/dependency versions, RNG seeds, data fingerprint, diagnostics, warnings, and analyses performed. Distinguish user-declared planned work from subsequent exploratory work. A local timestamp does not prove externally registered preregistration. Generated scripts/configurations should support reruns with authorized data access; no raw-data export without opt-in.

### 4.9 Result Auditor and completeness checker

Validate correspondence between the canonical numerical results, tables, narratives and exported reports. Use independent tests against reference calculations and method-specific validation of output. Auditing numerical consistency is distinct from validating study-design appropriateness. A reporting-completeness checker identifies missing methodological elements; it does not assign a universal research-quality grade.

### 4.10 Advanced extensions

Defensible sensitivity/multiverse comparisons (not p-value shopping); practical-significance thresholds; correctly specified equivalence/noninferiority where supported; before-data-collection study planning/sample-size calculations; additional statistical models and study designs; educational explanation mode; GUI over stable, serializable core contracts. No mandatory generative-AI or cloud account.

## 5. Scientific safety contract

1. **Preserve the question:** Define the estimand before selecting methods. Mann–Whitney/Kruskal–Wallis are not blanket replacements for tests of means or generic tests of median differences.
2. **Respect design:** Independence, pairing, clustering and causal identification require substantive information. A correct numerical calculation does not establish a valid design.
3. **Describe diagnostics honestly:** Failing to reject normality or equal variance does not prove either assumption. Separate what the dataset reveals from what the researcher confirms.
4. **Interpret uncertainty properly:** A p-value is not the probability the null is true. Nonsignificance does not prove no effect; significance does not establish effect size, practical importance or causality.
5. **Document missingness and exclusions:** Maintain original data, account for overlap between data-quality flags and preserve analysis-specific sample sizes.
6. **Avoid selective analysis:** No searching across tests or exclusions for significance. Report performed alternatives and account for multiple comparisons as needed.
7. **Do not hallucinate report content:** Generate Methods/Results statements only from recorded facts and verified computed results. Indicate missing study details explicitly.
8. **Protect privacy:** Local-first by default; treat labels and input data as untrusted in exports; do not bundle raw data by default.
9. **Decline unsafe automation:** Missing critical design facts or unsupported scenarios produce explainable blockers, not a fabricated recommended test.
10. **Claim scope accurately:** Publication-oriented drafts and reproducible calculations require expert/journal review. Tests passing are not proof of complete statistical correctness.

## 6. Differentiation strategy

PyAutoStat does not seek to outnumber SciPy's methods or claim that guidance, reports or auditing are individually novel. The differentiator is the **integrated, transparent and programmable research workflow**, in which design validation, recommendation, execution, interpretation, decision history and report auditing share one source of truth. Favor depth of support, clearly handled unsupported cases and reproducibility over a large checkbox catalogue.

## 7. Technical and product boundaries

- Python-first, in-memory pandas DataFrames initially, cross-platform and consistent with the supported Python floor.
- The statistical engine is independent of CLI, notebook widgets, GUI, HTML and report templates.
- A simple high-level API is an adapter over small testable modules. Validated serializable input/output schemas make a future GUI feasible.
- Keep established public API functional where practical; document unavoidable corrections and migration paths.
- No mandatory LLM, cloud service, database, GUI framework or unnecessary runtime dependency.
- Every new supported method needs preconditions, numerical reference tests, edge cases, interpretation, report mapping and documentation.

## 8. Success criteria

A user can profile a DataFrame without configuration; express a common research question with minimal additional input; receive a justified method recommendation or clear blocker; run a validated analysis; inspect estimates and uncertainty; receive a defensible narrative; and export a consistent, reproducible report. Expert-reviewed benchmark scenarios must include ambiguous designs and deliberately invalid requests, not only success cases.

The first integrated milestone is **Phase 10**, contingent on acceptance gates and review, not a promise of release date or readiness for every research design. Phases 11–12 add advanced functionality. Exact supported capabilities are those documented and tested in the shipping code, never those merely listed here.
