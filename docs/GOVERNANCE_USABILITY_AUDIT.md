# PyAutoStat Advanced Research Lifecycle, Governance, Reproducibility, Planning, and Audit Usability Audit

This document provides a systematic engineering, scientific safeguards, and usability audit of PyAutoStat's advanced research lifecycle features. It establishes the governance language contract, catalogs all sixteen lifecycle feature families, provides a decision guide, documents status vocabularies by object, and specifies usability actions.

---

## 1. Executive Summary and Architecture

PyAutoStat is an explainable and reproducible research analysis assistant for pandas DataFrames. It connects dataset profiling, explicit research specifications, defensible recommendations, validated calculations, qualified interpretation, reporting, audit, and reproducibility into one coherent, deterministic system without external cloud dependencies or generative AI.

The complete research lifecycle proceeds as:

```text
    research question
           │
           ▼
    optional planning declaration ("planned" | "exploratory" | "unknown")
           │
           ▼
    optional StatisticalAnalysisPlan (immutable plan snapshot)
           │
           ▼
    statistical analysis execution (single statistical engine)
           │
           ▼
    optional sensitivity analysis (researcher-supplied scenarios)
           │
           ▼
    optional practical-significance assessment (meaningful effect threshold)
           │
           ▼
    research report assembly (multi-format rendering)
           │
           ▼
    reporting completeness evaluation (guideline checklist)
           │
           ▼
    scientific consistency audit (invariants & cross-export consistency)
           │
           ▼
    reproducibility record capture (provenance & dataset fingerprint)
           │
           ▼
    optional explicit replay (reproduce() with supplied data)
           │
           ▼
    optional research bundle / session snapshot
```

---

## 2. Governance Language Contract

To protect scientific integrity and avoid overclaiming, PyAutoStat enforces strict conceptual boundaries across its governance vocabulary:

### 2.1 Audit
- **What it means**: Implemented internal consistency checks, mathematical invariants, and cross-format verifications found no discrepancies in the inspected records and exports.
- **What it does NOT mean**: Statistically correct in every possible sense, scientifically valid, causally identified, free of confounding, authentic raw data, valid preregistration, or peer reviewed.

### 2.2 Dataset Fingerprint
- **What it means**: Deterministic same-content consistency checking under the documented `pyautostat-dataframe-sha256-v1` algorithm over column names, dtypes, categories, index metadata, row count, and serialized values without storing raw observations.
- **What it does NOT mean**: Authorship, data ownership, participant identity, collection timestamp, data authenticity, absence of fabrication, or participant consent.

### 2.3 Decision Ledger
- **What it means**: In-memory, opt-in chronological record of software-observed local events and revisions occurring in the current Python process after `enable_tracking()` was activated.
- **What it does NOT mean**: Reconstruction of unobserved prior decisions, external reasoning history, preregistration, or intent before tracking was activated.

### 2.4 Planning Status
- **What it means**: Researcher-declared local metadata (`"planned"`, `"exploratory"`, or `"unknown"`) recorded in the local session.
- **What it does NOT mean**: Independently verified preregistration status in an external registry (such as OSF, ClinicalTrials.gov, or PROSPERO).

### 2.5 Reproducibility
- **What it means**: The reproducibility record captures sufficient configuration, environment metadata, and expected numerical projections to support explicit replay when data are separately supplied. Successful replay means the recomputed results matched recorded expectations within numerical tolerances ($\text{rtol}=10^{-10}, \text{atol}=10^{-12}$).
- **What it does NOT mean**: Scientific truth, external replication on new populations, study design validity, or raw-data authenticity.

### 2.6 Sensitivity Analysis
- **What it means**: Descriptive comparison of hypothesis decisions and effect estimates across only the explicitly supplied scenarios.
- **What it does NOT mean**: Universal model robustness, mathematical proof of model correctness, or absence of omitted-variable bias. The label `ROBUST` describes reject/fail-to-reject decision consistency only among the supplied comparable same-estimand scenarios.

### 2.7 Practical Significance
- **What it means**: Contextual relation of an observed effect estimate and confidence interval to a researcher-declared substantive threshold.
- **What it does NOT mean**: Universal effect-size cutoffs, clinical efficacy certification, or formal statistical equivalence / noninferiority testing.

### 2.8 Reporting Completeness
- **What it means**: Expected structural reporting fields (objective, variables, sample sizes, estimates, CIs, p-values, limitations) are present, partial, missing, or not applicable under a chosen reporting guideline (`general`, `apa`, `ieee`).
- **What it does NOT mean**: Study quality, methodological rigor, low risk of bias, statistical correctness, or publication worthiness.

### 2.9 Bundle Verification
- **What it means**: SHA-256 manifest verification checks byte-level integrity against transmission errors, disk corruption, or file alteration.
- **What it does NOT mean**: Cryptographic digital signatures, identity authentication, or legal non-repudiation.

---

## 3. "Which Advanced Feature Do I Need?" Guide

| I want to... | Recommended Entry Point | Returned Object |
| :--- | :--- | :--- |
| Record subsequent workflow decisions in this Python session | `assistant.enable_tracking(clock=...)` | `DecisionLedger` |
| Declare whether an upcoming analysis is planned or exploratory | `assistant.declare_planning("planned" \| "exploratory" \| "unknown")` | `None` (updates assistant) |
| Record an immutable pre-analysis plan snapshot before or after execution | `assistant.analysis_plan(draft, ...)` | `StatisticalAnalysisPlan` |
| Compare an executed analysis against a recorded plan | `assistant.plan_adherence(plan, result, ...)` | `PlanAdherenceResult` |
| Prospective sample size or precision planning before data collection | `assistant.study_planner()` / `StudyPlanner()` | `StudyPlanningResult` |
| Evaluate stability under explicit alternative specifications | `assistant.sensitivity_analysis(result, scenarios=[...])` | `SensitivityResult` |
| Compare an observed effect against my domain meaningful threshold | `assistant.practical_significance(result, threshold=...)` | `PracticalSignificanceResult` |
| Check whether required reporting items are present in a report | `assistant.reporting_completeness(report, style=...)` | `ReportingCompletenessResult` |
| Verify internal consistency of result, report, and export records | `assistant.audit(report, ...)` | `AuditResult` |
| Capture replay provenance and dataset fingerprint metadata | `assistant.reproducibility_record(result, ...)` | `ReproducibilityRecord` |
| Explicitly rerun an analysis with supplied data | `pyautostat.reproduce(record, data=df, ...)` | `ReproductionOutcome` |
| Export a lightweight replay metadata package (no raw data) | `record.save_package("repro_pkg.zip", ...)` | `pathlib.Path` |
| Export an archival multi-format research package with checksum manifest | `pyautostat.save_bundle(workflow, "bundle.zip", ...)` | `pathlib.Path` |
| Capture UI-independent state for notebooks or client applications | `assistant.session_snapshot(workflow, ...)` | `ResearchSessionSnapshot` |

---

## 4. Status Vocabularies by Object

| Object | Allowed Status Values | Primary Meaning | Execution Occurs? | Researcher Action Required? |
| :--- | :--- | :--- | :--- | :--- |
| `ResearchWorkflowResult` | `completed`, `partial`, `needs_input`, `data_limited`, `unsupported`, `failed` | Lifecycle workflow outcome | Yes if completed/partial/failed | Yes if needs_input or blocked |
| `Recommendation` | `ready`, `needs_input`, `unsupported` | Method recommendation status | No | Yes if needs_input/unsupported |
| `AnalysisResult` | `available`, `data_limited`, `unsupported`, `failed` | Numerical execution outcome | Yes | Yes if unavailable or failed |
| `StatisticalAnalysisPlan` | `ready`, `needs_input`, `unsupported` | Protocol readiness | No | Yes if needs_input/unsupported |
| `PlanAdherenceResult` | `matched`, `changed` | Plan vs execution agreement | No | Review discrepancies if changed |
| `StudyPlanningResult` | `available`, `unavailable` | Sample size search outcome | Prospective search | Adjust bounds if unavailable |
| `SensitivityResult` | `complete`, `partial`, `unavailable` | Overall scenario execution | Yes (per scenario) | Inspect individual scenarios |
| `SensitivityScenarioResult` | `completed`, `unavailable`, `incompatible`, `failed` | Single scenario outcome | Yes | Review warnings if failed/incompatible |
| `PracticalSignificanceResult` | `complete`, `partial`, `unsupported`, `unavailable` | Threshold relation assessment | No | Supply missing CI or valid threshold |
| `ReportingCompletenessResult` | `complete`, `acceptable`, `partial`, `incomplete` | Overall checklist assessment | No | Add missing fields if incomplete |
| `AuditResult` | `passed`, `failed`, `incomplete` | Internal consistency outcome | In-memory verification | Inspect findings if failed/incomplete |
| `ReproductionOutcome` | `reproduced`, `mismatch`, `unavailable` (`data_status`: `same_data`, `changed_data`, `fingerprint_unavailable`) | Replay execution agreement | Yes (on supplied data) | Review differing fields if mismatch |
| `BundleVerificationResult` | `valid: True` / `valid: False` | SHA-256 archive checksum validity | Archive verification | Re-export bundle if invalid |

---

## 5. Detailed Audit of the 16 Feature Families

### 5.1 Decision Tracking / DecisionLedger
- **User Goal**: Record local, sequential analytical events and revisions in an in-memory chronological ledger during the active Python session.
- **When to Use**: When maintaining an audit trail of user questions, draft updates, plan revisions, method recommendations, and exports for local scientific transparency.
- **When NOT to Use**: Do not use expecting automatic filesystem logging, background daemon capture without `enable_tracking()`, or external legal/regulatory audit logging.
- **Recommended Entry Point**: `assistant.enable_tracking(clock=...)` or `assistant.decision_ledger`.
- **Required Input**: Optional custom clock callable `Callable[[], datetime | str]`.
- **Returned Object**: `DecisionLedger` (property `events`, `history_status`).
- **Status Vocabulary**: `history_status`: `"recorded"`, `"unavailable"`. 22 distinct event types (`question_prepared`, `specification_updated`, `method_recommended`, `analysis_executed`, `interpretation_generated`, `report_generated`, `report_exported`, `audit_performed`, `planning_declared`, `existing_analysis_imported`, etc.).
- **Runs Statistics**: No.
- **Writes Files**: No.
- **Accesses Raw Data**: No.
- **What It Proves**: Software-observed local sequential actions in the active session after tracking was enabled.
- **What It Explicitly Does NOT Prove**: Unobserved prior decisions, external reasoning history, preregistration, researcher intent before activation, or authenticated time.
- **Presentation Support**: `show(ledger)` (`adapt_decision_ledger`).
- **Serialization/Export Support**: `ledger.to_dict()`, `ledger.to_json()` (schema version 1), included in session snapshot and research bundle provenance.
- **Likely User Confusion**: Believing `enable_tracking()` can retroactively discover what the user did before calling it, or that event sequence IDs constitute legally binding timestamps.
- **Current Usability Verdict**: `KEEP`.
- **Priority**: P1.
- **Proposed Action**: Document opt-in boundary clearly and emphasize that imported analyses set `history_status="unavailable"` for prior unobserved actions.

### 5.2 Planning Declaration
- **User Goal**: Record researcher-declared analytical intent ("planned", "exploratory", or "unknown") for reporting and provenance.
- **When to Use**: At the start of a study or analysis workflow to indicate whether hypotheses and methods were prespecified or exploratory.
- **When NOT to Use**: Do not use to claim external preregistration on OSF, ClinicalTrials.gov, or other public registries.
- **Recommended Entry Point**: `assistant.declare_planning("planned" | "exploratory" | "unknown", reason=...)`.
- **Required Input**: One of `"planned"`, `"exploratory"`, `"unknown"`; optional non-empty `reason` string.
- **Returned Object**: `None` (updates `assistant.planning_status` and emits ledger event if tracking enabled).
- **Status Vocabulary**: `"planned"`, `"exploratory"`, `"unknown"`.
- **Runs Statistics**: No.
- **Writes Files**: No.
- **Accesses Raw Data**: No.
- **What It Proves**: Researcher-declared local metadata in this Python session.
- **What It Explicitly Does NOT Prove**: Independent verification, preregistration in an external registry, or timestamped protocol freezing.
- **Presentation Support**: Surfaces in `show(plan)`, `show(practical)`, `show(workflow)`, report metadata.
- **Serialization/Export Support**: Serialized in `AnalysisSpecification`, `StatisticalAnalysisPlan`, `ReproducibilityRecord`, and `ResearchReport`.
- **Likely User Confusion**: Equating `"planned"` with legally verified preregistration.
- **Current Usability Verdict**: `DOCUMENT BETTER`.
- **Priority**: P0.
- **Proposed Action**: Enforce clear documentation that planning declarations are local self-reports, not external registry attestations.

### 5.3 StatisticalAnalysisPlan Creation and Revision
- **User Goal**: Record an immutable, structured analysis plan snapshot of planned methods, estimands, exclusions, outlier rules, multiplicity policies, report styles, and planned follow-ups before or after execution.
- **When to Use**: Before running an analysis to define analytical constraints; or retrospectively to record intended protocol with transparent retrospective disclosure.
- **When NOT to Use**: Do not use as a mechanism to execute tests or automatically submit a study protocol to an external trial registry.
- **Recommended Entry Point**: `assistant.analysis_plan(draft, ...)` or `StatisticalAnalysisPlan(...)`.
- **Required Input**: `QuestionDraft` (or `AnalysisSpecification`), policies, optional sensitivity scenarios, optional practical threshold.
- **Returned Object**: `StatisticalAnalysisPlan`.
- **Status Vocabulary**: `status`: `ready`, `needs_input`, `unsupported`.
- **Runs Statistics**: No.
- **Writes Files**: No.
- **Accesses Raw Data**: No.
- **What It Proves**: Explicitly frozen analytical protocol rules; records whether created before or after analysis (`created_after_analysis: bool`).
- **What It Explicitly Does NOT Prove**: Trial registry submission, mathematical correctness of the planned test, or absence of post-hoc revisions.
- **Presentation Support**: `show(plan)` (`adapt_statistical_analysis_plan`, `AnalysisPlanRenderer`).
- **Serialization/Export Support**: `plan.to_dict()`, `plan.to_json()` (schema version 1), included in session snapshot and research bundle provenance.
- **Likely User Confusion**: Assuming that creating a plan runs the test, or that post-hoc plan creation can masquerade as prospective planning.
- **Current Usability Verdict**: `POLISH`.
- **Priority**: P1.
- **Proposed Action**: Keep `created_after_analysis` prominent in presentation; document that multiplicity policies record intent rather than executing automated family-wise adjustments.

### 5.4 Plan Adherence
- **User Goal**: Objectively compare an executed `AnalysisResult` against a previously frozen `StatisticalAnalysisPlan`.
- **When to Use**: After running an analysis, to verify whether objective, variables, design, estimand, alpha, confidence level, and methods match the planned protocol.
- **When NOT to Use**: Do not use to detect fraud, infer scientific misconduct, or score protocol compliance.
- **Recommended Entry Point**: `assistant.plan_adherence(plan, result, reason=...)` or `compare_plan_to_result(...)`.
- **Required Input**: `StatisticalAnalysisPlan`, `AnalysisResult`, optional `reason`, optional `sensitivity`, optional `practical_significance`.
- **Returned Object**: `PlanAdherenceResult`.
- **Status Vocabulary**: Overall `status`: `matched`, `changed`. Comparison item status: `matched`, `changed`, `not_recorded`.
- **Runs Statistics**: No.
- **Writes Files**: No.
- **Accesses Raw Data**: No.
- **What It Proves**: Descriptive agreement or difference between planned protocol fields and executed result fields.
- **What It Explicitly Does NOT Prove**: Scientific truth, data authenticity, or research misconduct (`misconduct_inference: False` is hardcoded).
- **Presentation Support**: `show(adherence)` (`adapt_plan_adherence`, `PlanAdherenceRenderer`).
- **Serialization/Export Support**: `adherence.to_dict()`, `adherence.to_json()` (schema version 1), ledger event `plan_adherence_compared`.
- **Likely User Confusion**: Interpreting `status="changed"` as an accusation of misconduct rather than a neutral comparison state.
- **Current Usability Verdict**: `DOCUMENT BETTER`.
- **Priority**: P1.
- **Proposed Action**: Reinforce in presentation and docs that `matched` and `changed` are neutral comparison states, not moral or regulatory verdicts.

### 5.5 Prospective StudyPlanner
- **User Goal**: Compute prospective sample size and precision requirements under researcher-supplied assumptions before data collection.
- **When to Use**: During study design before collecting data to plan sample size for independent two-group or paired-means designs under power or half-width targets.
- **When NOT to Use**: Do not use after data collection to compute observed post-hoc power; do not use for ANOVA, regression, or correlation planning (unsupported).
- **Recommended Entry Point**: `assistant.study_planner()` or `StudyPlanner()`.
- **Required Input**: `target_difference`/`target_half_width`, `sd_group1`, `sd_group2` or `sd_difference`, `alpha`, `target_power`/`confidence_level`, `allocation_ratio`, `max_n`/`max_pairs`.
- **Returned Object**: `StudyPlanningResult`.
- **Status Vocabulary**: `status`: `available`, `unavailable`. `planning_type`: `power`, `precision`. `method_family`: `welch_independent_means`, `paired_means`.
- **Runs Statistics**: Prospective analytical formulas (noncentral-t and Student-t search), not empirical data tests.
- **Writes Files**: No.
- **Accesses Raw Data**: No (strictly dataset-free).
- **What It Proves**: Mathematical sample size required under idealized assumptions.
- **What It Explicitly Does NOT Prove**: Observed post-hoc power, guaranteed real-world power, or valid sample size if assumed standard deviations are wrong.
- **Presentation Support**: `show(planning_result)` (`adapt_study_planning`, `StudyPlanningRenderer`).
- **Serialization/Export Support**: `result.to_dict()`, `result.to_json()` (schema version 1), ledger event `study_planning_completed`.
- **Likely User Confusion**: Attempting to feed empirical dataset results to calculate "observed power" (strictly forbidden).
- **Current Usability Verdict**: `KEEP`.
- **Priority**: P0.
- **Proposed Action**: Maintain strict prohibition of observed post-hoc power, document allocation ratio definition and complete pairs vs raw rows.

### 5.6 Sensitivity Analysis
- **User Goal**: Test hypothesis decision stability across explicitly supplied alternative analytical specifications (e.g., Welch t vs Student t, standard vs robust).
- **When to Use**: To assess whether conclusions change under alternative distributional assumptions, covariance structures, or estimators.
- **When NOT to Use**: Do not use to hunt for the lowest p-value, claim universal model robustness, or perform automated sensitivity sweeps.
- **Recommended Entry Point**: `assistant.sensitivity_analysis(result, scenarios=[...])`.
- **Required Input**: Completed `AnalysisResult` and non-empty list of `SensitivitySpecification`s.
- **Returned Object**: `SensitivityResult` (containing `SensitivityScenarioResult`s).
- **Status Vocabulary**: Overall `status`: `complete`, `partial`, `unavailable`. Scenario `status`: `completed`, `unavailable`, `incompatible`, `failed`. Comparability: `same_estimand`, `different_estimand`, `incompatible`, `unavailable`.
- **Runs Statistics**: Yes, executes the explicitly declared scenarios.
- **Writes Files**: No.
- **Accesses Raw Data**: Yes, executes backend on assistant's DataFrame.
- **What It Proves**: Decision and estimate consistency across the explicitly provided scenarios only.
- **What It Explicitly Does NOT Prove**: Universal robustness across unsupplied models, absence of confounding, or model correctness.
- **Presentation Support**: `show(sensitivity)` (`adapt_sensitivity`, `SensitivityRenderer`), `sensitivity.compare()`.
- **Serialization/Export Support**: `result.to_dict()`, `result.to_json()` (schema version 1), schema version 2 reports and reproducibility records.
- **Likely User Confusion**: Believing `ROBUST` means universally true or that scenarios targeting different estimands (e.g., paired t vs Wilcoxon) are direct replications.
- **Current Usability Verdict**: `POLISH`.
- **Priority**: P0.
- **Proposed Action**: Clarify that `ROBUST` describes decision consistency only among the supplied comparable same-estimand scenarios.

### 5.7 Practical Significance
- **User Goal**: Contextualize observed effect estimates and confidence intervals against researcher-defined minimum meaningful thresholds.
- **When to Use**: When statistical significance ($p < \alpha$) must be distinguished from substantive or clinical importance.
- **When NOT to Use**: Do not use as a substitute for formal equivalence/noninferiority testing; do not use generic small/medium/large rules without domain justification.
- **Recommended Entry Point**: `assistant.practical_significance(result, threshold=...)` or `assess_practical_significance(...)`.
- **Required Input**: Completed `AnalysisResult` and `MeaningfulEffectThreshold`.
- **Returned Object**: `PracticalSignificanceResult`.
- **Status Vocabulary**: Overall `status`: `complete`, `partial`, `unsupported`, `unavailable`. Point relation: `positive_meaningful_region`, `negative_meaningful_region`, `below_meaningful_magnitude`, `meets_positive_threshold`, `below_positive_threshold`, etc. CI relation: `entirely_positive_meaningful`, `entirely_within_negligible_region`, `crosses_meaningful_boundary`, etc. Statistical evidence: `evidence_against_null`, `no_evidence_against_null`, `unavailable`.
- **Runs Statistics**: No (compares stored estimates and CI bounds to declared threshold).
- **Writes Files**: No.
- **Accesses Raw Data**: No.
- **What It Proves**: Relationship between stored estimate/CI and the researcher's declared threshold.
- **What It Explicitly Does NOT Prove**: Formal equivalence/noninferiority, causal clinical utility, or universal effect importance.
- **Presentation Support**: `show(practical)` (`adapt_practical_significance`, `PracticalSignificanceRenderer`), `practical.verdict`.
- **Serialization/Export Support**: `result.to_dict()`, `result.to_json()` (schema version 1), schema version 2 reports and reproducibility records.
- **Likely User Confusion**: Believing that a CI inside the threshold certifies equivalence (formal equivalence tests require specific two-one-sided-tests procedures not implemented here).
- **Current Usability Verdict**: `POLISH`.
- **Priority**: P0.
- **Proposed Action**: Reiterate distinction between descriptive threshold contextualization and formal equivalence tests.

### 5.8 Reporting Completeness
- **User Goal**: Produce a deterministic, machine-readable checklist verifying that standard reporting elements (objective, sample size, estimates, CIs, p-values, limitations) are present in a report.
- **When to Use**: Pre-publication or reporting audit to verify structural completeness against guidelines (`general`, `apa`, `ieee`).
- **When NOT to Use**: Do not use to evaluate study design quality, methodological validity, risk of bias, or publication worthiness.
- **Recommended Entry Point**: `assistant.reporting_completeness(report, style=...)` or `assess_reporting_completeness(...)`.
- **Required Input**: `ResearchReport`, style string (`"general"`, `"apa"`, `"ieee"`).
- **Returned Object**: `ReportingCompletenessResult`.
- **Status Vocabulary**: Overall `status`: `complete`, `acceptable`, `partial`, `incomplete`. Item `status`: `present`, `missing`, `partial`, `not_applicable`.
- **Runs Statistics**: No.
- **Writes Files**: No.
- **Accesses Raw Data**: No.
- **What It Proves**: Structural presence or absence of required reporting fields.
- **What It Explicitly Does NOT Prove**: Scientific quality, absence of bias, validity of conclusions, or peer-review readiness.
- **Presentation Support**: `show(completeness)` (`adapt_reporting_completeness`, `ReportingCompletenessRenderer`).
- **Serialization/Export Support**: `result.to_dict()`, `result.to_json()` (schema version 1), session snapshot.
- **Likely User Confusion**: Mistaking high completeness for high study quality or proof of valid science.
- **Current Usability Verdict**: `KEEP`.
- **Priority**: P1.
- **Proposed Action**: Keep explicit limitation visible that completeness evaluates formatting and reporting structure, not empirical truth.

### 5.9 Statistical Result/Report Audit
- **User Goal**: Verify internal consistency, contract invariants, and zero recalculation between primary analysis results, generated report, and exports.
- **When to Use**: As a pre-export or pre-bundle verification step to confirm that reported values match primary calculation records.
- **When NOT to Use**: Do not use expecting verification of real-world scientific validity, causal truth, or data collection integrity.
- **Recommended Entry Point**: `assistant.audit(report, ...)` or `StatisticalResultAuditor().audit(...)`.
- **Required Input**: `ResearchReport`, optional `AnalysisResult`, optional `SensitivityResult`, optional `PracticalSignificanceResult`, optional `exports` dict.
- **Returned Object**: `AuditResult`.
- **Status Vocabulary**: `status`: `passed`, `failed`, `incomplete`. Finding severity: `error`, `warning`, `info`.
- **Runs Statistics**: Reconstructs canonical report structures and verifies exact values and invariant constraints; zero statistical recalculation.
- **Writes Files**: No.
- **Accesses Raw Data**: No.
- **What It Proves**: Internal numerical and methodological consistency between stored records and serialized representations.
- **What It Explicitly Does NOT Prove**: Scientific validity, causal correctness, study quality, or authentic data collection.
- **Presentation Support**: `show(audit)` (`adapt_audit`, `AuditRenderer`).
- **Serialization/Export Support**: `audit.to_dict()`, `audit.to_json()` (schema version 1), included in workflow results and bundles.
- **Likely User Confusion**: Confusing `status="passed"` with "this study is scientifically valid and proven true".
- **Current Usability Verdict**: `DOCUMENT BETTER`.
- **Priority**: P0.
- **Proposed Action**: Clarify the exact scope of audit findings and the 3 status values (`passed`, `failed`, `incomplete`).

### 5.10 Content References
- **User Goal**: Deterministically reference and detect mutation of structured data objects without storing entire duplicate payloads.
- **When to Use**: Within provenance records, session ledgers, and reproducibility packages to bind an event to a specific immutable state snapshot.
- **When NOT to Use**: Do not use as a digital signature, legal non-repudiation tool, or secret cryptographic key.
- **Recommended Entry Point**: `pyautostat.content_reference(kind, value)`.
- **Required Input**: ASCII identifier `kind` (e.g. `"analysis"`, `"specification"`), JSON-serializable `value`.
- **Returned Object**: String formatted as `sha256:<kind>:<hex_digest>`.
- **Status Vocabulary**: Deterministic hash string format.
- **Runs Statistics**: No.
- **Writes Files**: No.
- **Accesses Raw Data**: No.
- **What It Proves**: Content-based equality and tamper detection for that specific serialized JSON snapshot.
- **What It Explicitly Does NOT Prove**: Cryptographic identity, legal non-repudiation, timestamp authority, or authorship.
- **Presentation Support**: Displayed in diagnostic metadata and audit findings.
- **Serialization/Export Support**: String tokens embedded in ledgers, reports, and reproducibility payloads.
- **Likely User Confusion**: Believing `sha256:...` references are public-key digital signatures.
- **Current Usability Verdict**: `KEEP`.
- **Priority**: P1.
- **Proposed Action**: Document formatting and distinction from digital signatures.

### 5.11 Dataset Fingerprint
- **User Goal**: Compute a SHA-256 fingerprint over DataFrame schema, columns, dtypes, categories, indices, and values without embedding or retaining raw observations.
- **When to Use**: Before and after analysis or during reproducibility record creation to verify that replayed data matches the original data structure and values.
- **When NOT to Use**: Do not use as participant identity verification, proof of legal data ownership, or copyright attestation.
- **Recommended Entry Point**: `pyautostat.dataset_fingerprint(df)`.
- **Required Input**: `pd.DataFrame`.
- **Returned Object**: `dict` with `algorithm`, `columns`, `index_type`, `index_dtype`, `index_names`, `rows`, `digest`.
- **Status Vocabulary**: Algorithm `"pyautostat-dataframe-sha256-v1"`.
- **Runs Statistics**: No (hashes values).
- **Writes Files**: No.
- **Accesses Raw Data**: Yes (iterates over DataFrame rows and columns).
- **What It Proves**: Same tabular content, types, and values under the documented hashing algorithm.
- **What It Explicitly Does NOT Prove**: Data authenticity, consent, ownership, acquisition date, or absence of fabricated data.
- **Presentation Support**: Surfaces in `show(record)`, `show(reproduction_outcome)`.
- **Serialization/Export Support**: Serialized in `ReproducibilityRecord`, `manifest.json`.
- **Likely User Confusion**: Believing the fingerprint contains secret raw rows (it does not), or that a matching fingerprint proves the data is authentic real-world data.
- **Current Usability Verdict**: `KEEP`.
- **Priority**: P0.
- **Proposed Action**: Clearly document fingerprint boundaries, performance trade-offs, and `fingerprint=False` bypass option.

### 5.12 ReproducibilityRecord
- **User Goal**: Capture complete configuration, expected projections, runtime environment, package versions, random seeds, and optional dataset fingerprint for subsequent explicit replay.
- **When to Use**: At the completion of an analysis to record replay metadata for distribution or archival.
- **When NOT to Use**: Do not use expecting it to contain the raw DataFrame.
- **Recommended Entry Point**: `assistant.reproducibility_record(result, ...)` or `ReproducibilityRecord.from_result(...)`.
- **Required Input**: Completed `AnalysisResult`, optional DataFrame, optional sensitivity and practical significance results.
- **Returned Object**: `ReproducibilityRecord`.
- **Status Vocabulary**: Schema versions `1` (base analysis) and `2` (includes recorded follow-up analysis configurations).
- **Runs Statistics**: No (extracts metadata and projection).
- **Writes Files**: No.
- **Accesses Raw Data**: Only if `fingerprint=True` and DataFrame provided.
- **What It Proves**: Complete specification and expected numerical projections recorded by PyAutoStat.
- **What It Explicitly Does NOT Prove**: Scientific validity, external reproducibility on new data, or study quality.
- **Presentation Support**: `show(record)` (`adapt_reproducibility`, `ReproducibilityRenderer`).
- **Serialization/Export Support**: `record.to_dict()`, `record.to_json()`, `record.save_package(...)`.
- **Likely User Confusion**: Expecting the record to contain raw data or to run replay automatically upon creation.
- **Current Usability Verdict**: `POLISH`.
- **Priority**: P1.
- **Proposed Action**: Clarify that record creation is passive and replay is always explicit.

### 5.13 Explicit reproduce() / Replay
- **User Goal**: Rerun an analysis from a `ReproducibilityRecord` using explicitly supplied data and check if recomputed values match recorded expectations within numerical tolerances.
- **When to Use**: In verification scripts, CI pipelines, or peer-review workflows to confirm computational repeatability.
- **When NOT to Use**: Do not expect automatic execution during report creation; do not use without supplying the dataset.
- **Recommended Entry Point**: `pyautostat.reproduce(record, data=df, allow_changed_data=False)`.
- **Required Input**: `ReproducibilityRecord`, `pd.DataFrame`, optional `allow_changed_data: bool`.
- **Returned Object**: `ReproductionOutcome`.
- **Status Vocabulary**: Overall `status`: `reproduced`, `mismatch`, `unavailable`. `data_status`: `same_data`, `changed_data`, `fingerprint_unavailable`. (Note: `reproduced` indicates same-data numerical agreement within tolerances; `mismatch` indicates differing numerical fields or changed-data reruns; `unavailable` indicates fingerprint was unavailable or the method could not run).
- **Runs Statistics**: Yes, re-runs the recommended analysis on the supplied DataFrame.
- **Writes Files**: No.
- **Accesses Raw Data**: Yes.
- **What It Proves**: Computational reproducibility within comparison tolerances ($\text{rtol}=10^{-10}, \text{atol}=10^{-12}$) on the supplied data.
- **What It Explicitly Does NOT Prove**: Scientific truth, external validity, or empirical correctness.
- **Presentation Support**: `show(outcome)` (`adapt_reproducibility`).
- **Serialization/Export Support**: `outcome.to_dict()`.
- **Likely User Confusion**: Expecting automatic follow-up scenario execution or assuming changed data can reproduce an exact match without `allow_changed_data=True`.
- **Current Usability Verdict**: `KEEP`.
- **Priority**: P0.
- **Proposed Action**: Document the three data-status scenarios and tolerance rules.

### 5.14 Reproducibility Package Export
- **User Goal**: Export a metadata-only ZIP archive containing reproducible specifications and expectations for external replay.
- **When to Use**: When sharing verification specifications with collaborators or repositories without sharing raw data.
- **When NOT to Use**: Do not use as a multi-format publication report bundle (use `save_bundle` for that).
- **Recommended Entry Point**: `record.save_package(path, overwrite=False, data_reference=None)`.
- **Required Input**: Destination `.zip` path, optional relative `data_reference` text.
- **Returned Object**: `pathlib.Path` to written archive.
- **Status Vocabulary**: Member files: `README.md`, `analysis_specification.json`, `analysis_reference.json`, `reproducibility_record.json`.
- **Runs Statistics**: No.
- **Writes Files**: Yes, creates metadata-only ZIP archive.
- **Accesses Raw Data**: No.
- **What It Proves**: Preserved configuration and expected results package.
- **What It Explicitly Does NOT Prove**: Contains no dataset, executes no automatic code.
- **Presentation Support**: Documented in README and reproduction guides.
- **Serialization/Export Support**: ZIP archive.
- **Likely User Confusion**: Confusing a reproducibility package with a research bundle.
- **Current Usability Verdict**: `DOCUMENT BETTER`.
- **Priority**: P1.
- **Proposed Action**: Explicitly document the contrast between Reproducibility Packages (replay metadata only) and Research Bundles (multi-format publication archive).

### 5.15 ResearchSessionSnapshot
- **User Goal**: Capture a complete, UI-independent serializable state snapshot of the active research session for future notebook, CLI, or GUI clients.
- **When to Use**: When serializing the current workflow state, available actions, diagnostic blockers, questions, and attached governance objects.
- **When NOT to Use**: Do not use as a live database session or persistent background process.
- **Recommended Entry Point**: `assistant.session_snapshot(workflow, ...)` or `build_session_snapshot(...)`.
- **Required Input**: `ResearchWorkflowResult`, optional attached governance objects.
- **Returned Object**: `ResearchSessionSnapshot`.
- **Status Vocabulary**: Reflects workflow status (`completed`, `partial`, `needs_input`, `data_limited`, `unsupported`, `failed`); `available_actions` list.
- **Runs Statistics**: No.
- **Writes Files**: No.
- **Accesses Raw Data**: No.
- **What It Proves**: Exact state snapshot of software workflow at a single point in time.
- **What It Explicitly Does NOT Prove**: Persisted session continuity or external validation.
- **Presentation Support**: `show(snapshot)` (`adapt_session_snapshot`, `SessionSnapshotRenderer`).
- **Serialization/Export Support**: `snapshot.to_dict()`, `snapshot.to_json()` (schema version 1), included in bundles (`provenance/session.json`).
- **Likely User Confusion**: Believing the snapshot can execute actions without passing through core validators.
- **Current Usability Verdict**: `KEEP`.
- **Priority**: P1.
- **Proposed Action**: Document role as UI-independent client adapter.

### 5.16 Relationship with Research Bundles
- **User Goal**: Package all publication documents, data tables, static figures, and execution provenance into an authoritative, integrity-verified ZIP archive.
- **When to Use**: At the completion of research to create an archival artifact for distribution or publication.
- **When NOT to Use**: Do not use on blocked or pending workflows (`needs_input`, `data_limited`, `unsupported`, `failed`); use `show(workflow)` or `save_html(workflow)` instead.
- **Recommended Entry Point**: `pyautostat.save_bundle(target, path, ...)` or `to_bundle(target, ...)`.
- **Required Input**: `ResearchWorkflowResult` (must be `completed` or `partial`), `ResearchReport`, `AnalysisResult`, or `PresentationView`.
- **Returned Object**: `pathlib.Path` (file) or `bytes` (in-memory).
- **Status Vocabulary**: Bundle manifest schema version 1; member integrity verified via `verify_bundle(...)` returning `BundleVerificationResult(valid, errors)`.
- **Runs Statistics**: No (assembles already computed results).
- **Writes Files**: Yes (`save_bundle`).
- **Accesses Raw Data**: No.
- **What It Proves**: Byte-level file integrity and complete cross-format alignment.
- **What It Explicitly Does NOT Prove**: Digital signature, identity authentication, or legal non-repudiation.
- **Presentation Support**: `show` on target objects; `verify_bundle` returns structured verification result.
- **Serialization/Export Support**: ZIP archive containing HTML, PDF, DOCX, JSON, CSV, SVG/PNG figures, provenance JSONs, manifest.
- **Likely User Confusion**: Expecting blocked workflows to generate bundles, or expecting bundles to contain raw source datasets.
- **Current Usability Verdict**: `POLISH`.
- **Priority**: P0.
- **Proposed Action**: Keep the distinction between Research Bundles and Reproducibility Packages prominent; document workflow status constraints.

---

## 6. High-Value Action Items

### P0 Actions (Completed / Current Shipped State)
1. **Bundle Workflow-Status Restrictions**: Update `docs/RESEARCH_BUNDLES.md` to reflect `BundleAssembler` validation enforcing `completed` or `partial` workflow status. Blocked workflows direct users to `show(workflow)`, `save_html(workflow)`, or `save_docx(workflow)`.
2. **Reconcile Stale Pearson Uncertainty Statements**: Document that Pearson correlation provides an analytical Fisher-z asymptotic normal confidence interval when $n > 3$. For $n \le 3$, sample size is insufficient, yielding `unavailable` CI status and `partial` completeness/uncertainty status with an explicit small-sample limitation.
3. **PDF Parity Overclaim**: Narrow PDF claims from "100% typographical parity" to "Uses the same canonical HTML content and stored statistical values (zero recalculation); pagination and print layout may differ according to Chromium's print rendering."
4. **Scope Privacy Safeguards**: Explicitly note that while raw DataFrame rows and participant identifiers are excluded by default, when dataset profiling is explicitly configured with `include_row_positions=True`, integer row index offsets are stored in the profile dictionary for diagnostic navigation, but raw row values are never copied into exports.
5. **Establish Governance Language Safeguards**: Formulate unambiguous boundaries distinguishing audit from scientific validation, fingerprint from authentication, decision ledger from preregistration, and sensitivity from universal robustness.

### P1 Actions (Completed / Current Shipped State)
1. **Reporting Audit Target Matrix**: Align `docs/REPORTING_AUDIT.md` matrices with actual `adapt()` behavior, clarifying that profile dictionaries, frequency tables, cross-tabs, and governance/planning objects can be passed to `to_html`, `to_pdf`, and `to_docx`.
2. **"Which Advanced Feature Do I Need?" Decision Matrix**: Publish this guide in `API_REFERENCE.md` and `docs/README.md`.
3. **End-to-End Coherent Lifecycle Example**: Provide a complete, runnable end-to-end advanced lifecycle workflow example in `docs/ADVANCED_PLANNING_AND_PRESENTATION.md` and link it across documentation.
4. **Distinguish Reproducibility Packages vs Research Bundles**: Explicitly document that Reproducibility Packages contain replay metadata only, whereas Research Bundles contain full publication documents, tables, and figures.
5. **Status Vocabulary Reference**: Provide the complete cross-object status vocabulary table in docs.

### P2 Actions (Deferred)
- Formal external preregistration submission integrations (e.g. OSF API client).
- Cryptographic PKI public-key signing of bundles.
- Automatic scenario generation for sensitivity analysis.
- Multi-group / regression / ANOVA prospective power planning families.
- Observed post-hoc power calculation (intentionally rejected for scientific validity).
- Cloud storage adapters and collaborative multi-user ledgers.
