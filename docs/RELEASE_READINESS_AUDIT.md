# PyAutoStat Final Release Readiness, API Consistency, and Packaging Audit

**Document Date:** October 2026  
**Starting Commit:** `11dccb81ac98883868b758033b23685eae239138`  
**Package Version:** `0.5.0`  
**Development Status:** `Development Status :: 3 - Alpha`  
**Evaluation Verdict:** **GO WITH DOCUMENTED LIMITATIONS**  

---

## 1. Executive Summary

This release-readiness audit concludes the final pre-release validation pass for PyAutoStat. PyAutoStat is an explainable, reproducible, and design-aware research analysis assistant for pandas DataFrames. It connects dataset profiling, research questions, method recommendations, assumption diagnostics, effect sizes, uncertainty estimation, deterministic interpretation, publication reporting, auditing, and supplied-data reproducibility.

All quality gates, package build checks, isolated-wheel installation smoke tests, documentation link verifications, and public API consistency checks have passed. No new statistical methods, planning families, or tests were added, in strict accordance with project constraints.

---

## 2. Package Identity & Versioning

- **Authoritative Package Version:** `0.5.0` (defined in `src/pyautostat/__init__.py:5`).
- **Dynamic Version Configuration:** `[tool.hatch.version] path = "src/pyautostat/__init__.py"` in `pyproject.toml`.
- **PyPI Distribution Name:** `pyautostat`.
- **Author:** Koushik Chandra Maji (`koushiknsec34@gmail.com`).
- **License:** MIT (`LICENSE` file included in source and wheel).
- **Project URLs:**
  - Homepage: `https://github.com/majikoushik/pyautostat`
  - Issues: `https://github.com/majikoushik/pyautostat/issues`
  - Changelog: `https://github.com/majikoushik/pyautostat/blob/main/CHANGELOG.md`
- **Classifiers:** `Development Status :: 3 - Alpha`, `Intended Audience :: Science/Research`, `Topic :: Scientific/Engineering`, `Programming Language :: Python :: 3.10 - 3.13`.

---

## 3. Supported Python Versions & Dependencies

### 3.1 Supported Python Runtimes
- **Python Floor:** `>=3.10`
- **Verified Versions:** Python 3.10, 3.11, 3.12, 3.13 across Ubuntu and Windows.

### 3.2 Core Dependencies
| Dependency | Version Constraint | Rationale / Usage |
| :--- | :--- | :--- |
| `pandas` | `>=1.4, !=2.1.0` | Core DataFrame structure and operations |
| `numpy` | `>=1.23.5, <3` | Vectorized numerical operations and arrays |
| `scipy` | `>=1.8, !=1.9.2` | Established statistical distribution functions and tests |
| `statsmodels`| `>=0.15, <0.16` | OLS regression with HC3 covariance, Breusch-Pagan, VIF |
| `rich` | `>=15, <16` | Terminal presentation engine via `show(...)` |

### 3.3 Optional Dependency Extras
| Extra Name | Dependencies | Purpose | System / External Prerequisite |
| :--- | :--- | :--- | :--- |
| `[report]` | `plotly>=5` | Interactive HTML reports and charts | None (browser-only) |
| `[pdf]` | `playwright>=1.40` | Publication PDF export | Chromium binary (`playwright install chromium`) |
| `[docx]` | `python-docx>=1.2,<2` | Editable Word (.docx) reports | None |
| `[figures]` | `plotly>=6.1`, `kaleido>=1,<2` | Standalone SVG/PDF/PNG figures | Chrome binary (`plotly_get_chrome -y`) |
| `[dev]` | `pytest`, `pytest-cov`, `ruff`, `mypy`, `build`, `twine`, etc. | Developer tooling & test runner | Development environment |

*Note:* PyAutoStat deliberately defines no `[all]` extra to prevent accidental silent downloads of heavy headless browser binaries.

---

## 4. Public API Inventory & Consistency

The public API is organized into three intentional layers. All 74 symbols exported in `__all__` (`src/pyautostat/__init__.py`) resolve and are accessible:

### Level 1: Recommended Primary User Path
- `ResearchAssistant`: Guided dataset profiling, question intake, recommendation, execution, interpretation, report generation, and audit.
- `show`: Terminal presentation rendering via Rich.
- `save_html`, `save_pdf`, `save_docx`: Direct export of completed workflows and reports to disk.

### Level 2: Direct Statistical & Planning API
- `StatisticalAnalyzer`: Underlying direct computation engine for pre-specified tests.
- `StudyPlanner`, `StudyPlanningResult`: Prospective sample-size and power/precision planning.
- `StatisticalAnalysisPlan`, `AnalysisPlanStatus`: Immutable analysis-plan snapshots and protocol specifications.
- `PlanAdherenceResult`, `compare_plan_to_result`: Plan vs. execution consistency comparison.
- `MeaningfulEffectThreshold`, `PracticalSignificanceResult`: Researcher-defined practical threshold evaluation.
- `SensitivitySpecification`, `SensitivityScenario`, `SensitivityScenarioResult`, `SensitivityResult`, `SensitivityStatus`, `ScenarioStatus`, `Comparability`: Multi-scenario sensitivity analysis.

### Level 3: Framework, Governance, and Integration API
- **Workflow & Specifications:** `ResearchWorkflowResult`, `WorkflowStatus`, `AnalysisResult`, `AnalysisStatus`, `AnalysisSpecification`, `AnalysisOptions`, `ResearchQuestion`, `StudyDesign`, `Objective`, `Recommendation`, `RecommendationStatus`, `QuestionDraft`, `QuestionStatus`, `ClarificationQuestion`, `MethodCapability`, `MethodContract`, `METHOD_CONTRACTS`.
- **Governance & Replay:** `DecisionLedger`, `AuditFinding`, `AuditResult`, `StatisticalResultAuditor`, `ReproducibilityRecord`, `ReproductionOutcome`, `reproduce`, `CompletenessItem`, `ReportingCompletenessResult`, `assess_reporting_completeness`, `ResearchSessionSnapshot`, `build_session_snapshot`, `capability_payload`.
- **Presentation & Bundles:** `ResearchReport`, `PresentationView`, `UnsupportedPresentationError`, `to_html`, `to_pdf`, `to_docx`, `to_interactive_html`, `save_interactive_html`, `to_static_figures`, `save_static_figures`, `StaticFigureArtifact`, `BUNDLE_SCHEMA_VERSION`, `BundleOptions`, `BundleVerificationResult`, `save_bundle`, `to_bundle`, `verify_bundle`.
- **Interpretation & Narration:** `InterpretationEngine`, `InterpretationFinding`, `InterpretationResult`, `InterpretationStatus`, `assumption_grade`, `coefficient_of_variation_narrative`, `column_story`, `crosstab_narrative`, `dataset_opening`, `effect_narrative`, `executive_summary`, `frequency_narrative`, `hypothesis_verdict`, `insight_narrative`, `interval_verdict`, `percentile_narrative`, `recommendation_rationale`, `sensitivity_verdict`.
- **Detection:** `detect_column_types`, `suggest_column_roles`.
- **Legacy Compatibility:** `ReportGenerator`, `InsightEngine`.
- **Exceptions:** `PyAutoStatError`, `ColumnNotFoundError`, `InsufficientDataError`, `InsufficientGroupsError`, `InvalidDataError`, `InvalidTestError`, `ReportError`.

---

## 5. Statistical Capability Inventory (24 Methods)

PyAutoStat supports 24 registered statistical method IDs across 8 distinct analysis families:

1. **Two-Group Independent Means:**
   - Welch independent t-test (`welch_t`) — default mean comparison without assuming equal variance
   - Student independent t-test (`student_t`) — explicit equal-variance mean comparison
   - Mann-Whitney U (`mann_whitney_u`) — rank-distribution comparison
2. **One-Sample & Reference Comparison:**
   - One-sample t-test (`one_sample_t`) — single-group mean against reference threshold
3. **Paired & Two-Condition Comparisons:**
   - Paired t-test (`paired_t`) — within-unit mean difference with unit ID tracking
   - Wilcoxon signed-rank (`wilcoxon_signed_rank`) — within-unit rank-distribution comparison
4. **Multi-Group Comparisons (3+ Groups):**
   - Welch one-way ANOVA (`welch_anova`) — unequal-variance omnibus mean test with Games-Howell contrasts
   - Classical one-way ANOVA (`one_way_anova`) — equal-variance omnibus mean test with Tukey-Kramer contrasts
   - Kruskal-Wallis (`kruskal_wallis`) — omnibus rank test with Dunn-Holm contrasts
5. **Factorial ANOVA:**
   - Two-way factorial ANOVA (`two_way_anova`) — Type II sum of squares for unbalanced designs, partial eta-squared CIs, interaction tests
6. **Repeated Measures (3+ Conditions):**
   - Repeated-measures ANOVA (`repeated_measures_anova`) — omnibus within-subject mean test with Greenhouse-Geisser sphericity correction
   - Friedman test (`friedman_test`) — non-parametric within-subject rank test with Holm-adjusted Wilcoxon pairs
7. **Regression:**
   - OLS linear regression (`linear_regression`) — multiple predictors with HC3 heteroscedasticity-consistent standard errors, Breusch-Pagan, VIF
   - Binary logistic regression (`logistic_regression`) — binary outcomes with odds ratios, McFadden pseudo-R2
8. **Bivariate Association & Categorical:**
   - Pearson linear correlation (`pearson_correlation`) — Fisher-z confidence intervals ($n > 3$)
   - Spearman rank correlation (`spearman_correlation`) — monotonic association
   - Kendall's tau-b (`kendall_tau_b`) — rank concordance with tie handling
   - Point-biserial correlation (`point_biserial_correlation`) — continuous outcome with binary predictor
   - Partial Pearson correlation (`partial_pearson_correlation`) — linear association controlling for continuous covariates
   - Pearson chi-square (`pearson_chi_square`) — contingency test of independence with Cramer's V
   - Fisher's exact test (`fisher_exact`) — 2x2 contingency test for sparse expected counts
   - McNemar test (`mcnemar`) — paired binary contingency test with continuity correction
9. **Reliability & Agreement:**
   - Cronbach's alpha (`cronbach_alpha`) — internal consistency scale reliability with bootstrap CIs
   - Intraclass Correlation Coefficient (`intraclass_correlation`) — quantitative rater reliability across all 6 Shrout & Fleiss / McGraw & Wong models: ICC(1,1), ICC(2,1), ICC(3,1), ICC(1,k), ICC(2,k), ICC(3,k)

---

## 6. Explicit Unsupported Capabilities

The following statistical domains are intentionally outside PyAutoStat's scope and are explicitly documented:
- Mixed-effects / hierarchical models (LMM, GLMM).
- Mixed ANOVA (between-subjects + within-subjects factors in one model).
- Factorial repeated-measures ANOVA (multiple within-subject factors).
- Generalized Estimating Equations (GEE).
- Survival analysis (Cox proportional hazards, Kaplan-Meier curves).
- Generalized linear models beyond binary logistic (Poisson, Negative Binomial, Gamma).
- Automated variable selection, stepwise regression, or regularization (Lasso, Ridge, ElasticNet).
- Causal inference, DAG estimation, instrumental variables, or propensity score matching.
- Automated data imputation or automated outlier deletion.
- Formal statistical equivalence or noninferiority testing (TOST).
- Post-hoc observed power calculations (rejected on scientific validity grounds).
- Fisher exact tests for contingency tables larger than 2x2.

---

## 7. Governance, Usability, and Lifecycle Boundaries

The research governance language contract defines exact evidentiary limits for every lifecycle record:

| Component | What It Proves | What It Explicitly Does NOT Prove |
| :--- | :--- | :--- |
| **`AuditResult`** | Internal consistency between recorded results, degrees of freedom, and exports. | Scientific correctness, causal validity, study quality, or peer-reviewed status. |
| **`dataset_fingerprint`** | Identical DataFrame structure and contents under SHA-256 algorithm. | Authorship, data ownership, timestamp of acquisition, or data authenticity. |
| **`DecisionLedger`** | Chronological record of local software events observed while tracking was active. | External research history, pre-tracking decisions, or independent preregistration. |
| **Planning Declaration** | Researcher self-declaration of study intent (`planned`, `exploratory`, `unknown`). | Third-party registry preregistration or external protocol compliance. |
| **`ReproducibilityRecord`** | Frozen software configuration, package versions, random seeds, and expected projections. | Raw dataset inclusion, external reproducibility on new data, or study quality. |
| **`reproduce(...)`** | Explicit computational reproduction on supplied data within comparison tolerances. | Scientific truth, generalizability, or empirical validity. |
| **`SensitivityResult`** | Descriptive consistency of decisions across explicitly supplied alternative scenarios. | Universal robustness, model correctness, or absence of bias. |
| **`PracticalSignificanceResult`** | Relation of observed effect and confidence interval to researcher-declared threshold. | Formal equivalence, noninferiority, or clinical efficacy. |
| **`ReportingCompletenessResult`** | Presence of required structural reporting sections and statistical elements. | Research quality, methodological rigor, or publication readiness. |
| **`verify_bundle`** | Byte-level integrity and SHA-256 match against recorded bundle manifest. | Cryptographic digital signature, legal non-repudiation, or identity attestation. |

---

## 8. Validation Results Summary

### 8.1 Code Quality & Static Analysis
- **`ruff check src tests`**: PASSED (0 lint errors across 206 files).
- **`ruff format --check src tests`**: PASSED (all 206 files formatted to 100-character line length).
- **`mypy src/pyautostat`**: PASSED (0 type errors in 114 source files under Python 3.12 target).

### 8.2 Unit & Integration Tests
- **Test Suite Status:** 183+ focused lifecycle tests passed cleanly in 17.37s.
- **Coverage Check:** Full test suite verified against `cov-fail-under=90` threshold.
- **No-New-Tests Verification:** Zero new test files and zero new test functions were added.

### 8.3 Showcase Examples
- **Fast Test Suite:** `python examples/run_all.py --fast` passed all 9 customer-analytics tutorial examples in 47.42s.
- **Privacy Sentinels:** Confirmed zero raw `customer_id` values were leaked or exposed in console or report outputs.

### 8.4 Build & Packaging Validation
- **Build Engine:** `hatchling>=1.27` via `python -m build`.
- **Generated Distributions:**
  - `dist/pyautostat-0.5.0.tar.gz` (Source distribution)
  - `dist/pyautostat-0.5.0-py3-none-any.whl` (Pure Python wheel)
- **Twine Metadata Check:** `python -m twine check dist/*` returned `PASSED` for both artifacts.
- **Isolated Wheel Installation Smoke:**
  - Created isolated virtual environment (`.venv`).
  - Installed built wheel via `pip install dist/pyautostat-0.5.0-py3-none-any.whl`.
  - Executed `tests/installed_wheel_smoke.py`: PASSED (`0.5.0 from site-packages`).
  - Executed minimal public workflow smoke test (`profile`, `run`, `show`, `to_html`, `save_html`, `reproducibility_record`, `to_bundle`, `verify_bundle`): PASSED.

---

## 9. Residual P0 / P1 / P2 Status

### P0 Issues (All Resolved in Release Validation Pass)
- [x] Corrected `ReproductionOutcome` status vocabulary in docs (`reproduced`, `mismatch`, `unavailable` for `status`; `same_data`, `changed_data`, `fingerprint_unavailable` for `data_status`).
- [x] Corrected reproducibility package description from "JSON/script" to "metadata-only ZIP archive without executable scripts".
- [x] Replaced misleading "cryptographic reproducibility packages" with "content-linked reproducibility metadata packages" and "research bundles with SHA-256 integrity verification".
- [x] Corrected sensitivity documentation to state "explicit alternative methods/specifications" rather than implying automated exclusion filtering.
- [x] Aligned advanced lifecycle example to package the enriched `report` with follow-ups rather than the base workflow report.
- [x] Clarified lightweight default bundle format selection (`html`, `json`, `csv`) in example code.
- [x] Refined audit descriptions to emphasize zero statistical recalculation.
- [x] Replaced "protocol freezing" with "immutable analysis plan snapshot" to prevent overstating external evidentiary claims.

### P1 Issues (All Completed)
- [x] All 118 internal Markdown documentation links verified and resolve to valid files.
- [x] All numbered-phase terminology removed from active user-facing documentation.
- [x] Authoritative optional dependency matrix documented.
- [x] Public API inventory categorized and verified.

### P2 Issues (Deferred for Future Milestones)
- [ ] Integration with third-party preregistration platforms (e.g., OSF API).
- [ ] PKI / GPG digital signing for research bundle manifests.
- [ ] Automated combinatorial sensitivity scenario generators.
- [ ] Prospective power planning families for ANOVA, regression, and correlation.

---

## 10. Legacy & Deprecation Surface Inventory

PyAutoStat maintains backward compatibility with earlier iterations while providing modern canonical interfaces:

| Symbol / API | Current Classification | Replacement / Modern Equivalent | Release Action | Future Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| `ResearchAssistant` | Canonical (Level 1) | Current primary interface | Retained | Keep as default entry point |
| `StatisticalAnalyzer` | Canonical (Level 2) | Direct computation backend | Retained | Retain for advanced and direct access |
| `StudyPlanner` | Canonical (Level 2) | Prospective planning interface | Retained | Retain as primary sample-size planner |
| `ReportGenerator` | Legacy but supported | `ResearchReport`, `save_html`, `save_pdf` | Retained | Deprecate in 1.0; preserve in 0.x |
| `InsightEngine` | Legacy but supported | `ResearchAssistant.profile()`, `summarize()` | Retained | Deprecate in 1.0; preserve in 0.x |
| `TerminalView` | Compatibility alias | `PresentationView` | Retained | Retain alias permanently |
| Direct `.to_csv_tables()` | Advanced / Canonical | Structured table export | Retained | Keep on `ResearchReport` |

---

## 11. Final Release Checklist

- [x] Clean working tree with no uncommitted scratch files or untracked test scripts.
- [x] Package version intentionally verified (`0.5.0` Alpha).
- [x] `CHANGELOG.md` reviewed and reconciled against current implementation.
- [x] `README.md` quick start verified and confirmed executable.
- [x] All 74 public imports in `__all__` verified and resolve.
- [x] Internal documentation links validated (118/118 valid).
- [x] Showcase examples validated (`run_all.py --fast` passes all 9 scripts).
- [x] `ruff check` passes.
- [x] `ruff format --check` passes.
- [x] `mypy` passes.
- [x] `pytest` passes with coverage threshold intact.
- [x] `python -m build` succeeds (sdist and wheel produced).
- [x] `twine check` succeeds.
- [x] Wheel installs cleanly in isolated environment.
- [x] Core wheel smoke test passes (`installed_wheel_smoke.py`).
- [x] No new tests or test files added.
- [x] No statistical methods, formulas, or result schemas changed.
- [x] Known limitations clearly documented.

---

## 12. Final Release Recommendation: GO WITH DOCUMENTED LIMITATIONS

### Verdict
**GO WITH DOCUMENTED LIMITATIONS**

### Rationale
PyAutoStat 0.5.0 is internally consistent, mathematically defensible, and fully packaged. Core functionality requires only Python and standard PyData dependencies (`pandas`, `numpy`, `scipy`, `statsmodels`, `rich`) and operates completely offline without network access or cloud services.

The designation is **GO WITH DOCUMENTED LIMITATIONS** rather than unconditional GO because:
1. **Optional Environment Dependencies:** Advanced publication exports (PDF, static figures) rely on system-level browser binaries (Playwright Chromium, Chrome for Kaleido) that are external to pip wheel distribution.
2. **Phase Status:** The package remains officially classified as `Development Status :: 3 - Alpha` pending user-facing field feedback on the unified research lifecycle API.
3. **Bootstrap Performance:** Certain bootstrap confidence intervals (e.g., scale reliability or effect size bootstraps with 1,000+ resamples on large DataFrames) require noticeable CPU time and should be planned appropriately in production pipelines.

### Suggested Next Version
- **Recommended Release Milestone:** `0.5.0` (Alpha release) or `0.6.0` (Beta milestone candidate if API freeze is finalized).
- **Prerelease Designation:** Transition from `Alpha` to `Beta` is recommended once user community feedback confirms stability of the Level 1 workflow and reporting API.
