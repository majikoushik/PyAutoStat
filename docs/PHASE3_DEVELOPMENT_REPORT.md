# Phase 3 development record

**Status:** Phase 3 implementation complete; overall validation blocked by documented pre-existing failures and an unrun Phase 3 CI matrix.

## Baseline (before Phase 3 edits)

- Base commit: `a3df7b89dd1a0fc4f9bf77d6aa1f4d16df187edb` (merged Phase 2 correction).
- Local environment: Python 3.12.7, SciPy 1.18.1. Full suite: **156 passed**, 27 warnings, **94.49% coverage**. No local failing test identifiers.
- Latest `main` CI: [run 35676099846](https://github.com/majikoushik/PyAutoStat/actions/runs/35676099846). Lint and build passed; Python 3.11-3.13 tests passed on both operating systems. Python 3.10 on Ubuntu and Windows: **3 failed, 153 passed**, 19 warnings. Both jobs installed SciPy 1.15.3.
- CI failures: `test_normality_reports_all_three_tests` and `test_normaltest_retains_only_advisory_small_sample_warning` drop a finite D'Agostino-Pearson result because SciPy 1.15.3 uses a different small-sample advisory text than the merged warning classifier recognizes. `test_small_and_constant_samples_do_not_emit_undefined_normality_results` expects Anderson-Darling for three observations, while the merged grid validator correctly omits the invalid negative critical grid returned by this SciPy version. These are pre-existing Phase 2 failures; they are tracked separately from Phase 3.
- A separate Phase 2 follow-up commit recognizing the SciPy 1.15 wording is pushed on `phase2-ci-scipy115`; the user requested that its CI publication be skipped while Phase 3 proceeds. This Phase 3 branch starts from `origin/main` and does not include that follow-up.

## Final comparison

### Implementation and architecture

Both `ResearchAssistant.profile()` and `StatisticalAnalyzer.analyze_all()` call the same `DatasetProfiler` in `profiling.py`. The existing analyzer remains the backend for numerical descriptions, normality, outliers, distributions, histograms, and correlations. The profiler validates optional declarations, selects profile-appropriate numeric columns, assembles categorical and missingness summaries, and records structured defaults. It never selects a hypothesis test or mutates source values. The legacy top-level keys remain; the additive keys are `categorical_summary`, `variable_intelligence`, `data_dictionary`, and `profile_metadata`. `ResearchAssistant.complete_case_count(columns)` reports analysis-specific row availability without dropping rows. The existing `ReportGenerator` remains the JSON-safe export path because `overview.dtypes` retains pandas dtype objects for Python compatibility.

The one-argument workflow remains:

```python
profile = ResearchAssistant(df).profile()
```

Optional `data_dictionary`, `histogram_bins`, and `include_row_positions` keyword arguments expose advanced controls. Recognized declarations include text labels, analytical type and role, unit, valid range, allowed values, missing codes, and ordinal order. Declared missing codes are counted but deliberately **not applied** to denominators or values; a structured issue explains this. An explicitly declared categorical or identifier type can change profile method selection, without recoding the data. A numeric identifier is automatically excluded only when both an identifier-like name and unique observed values support the suggestion; a name alone does not establish meaning.

### Numerical checks

- `[1, 2, 3, 4, 20]`: mean 6, median 3, sample variance 62.5, Q1 2, Q3 4, IQR outlier count 1/5 = 20%; zero MAD remains unavailable.
- Four rows arranged in two exact duplicate pairs with alternating missing columns: 4 missing cells among 12 cells (33.33%), 4 affected rows, 0 complete rows, 2 repeated rows, 4 rows in duplicate groups, and 4 rows with both missingness and duplicate-group flags. These overlap and are never subtracted as disjoint counts.
- Three numeric columns with staggered missingness produce pairwise counts 4, 5, and 3; known positive and negative perfect Pearson, Spearman, and Kendall coefficients remain ±1. A pair with only one complete observation has unavailable coefficients and Pearson p-value without a backend small-sample warning.
- Constant and nearly constant pairs remain unavailable. Numerical backend warnings prevent misleading Pearson inference. Extreme-scale overflow and undefined outlier counts remain covered by Phase 2 regression tests.

### Final local checks and regression comparison

| Check | Result |
| --- | --- |
| Phase 3 tests (`tests/test_profiling.py`, `tests/test_profiling_metadata.py`) | 34 passed |
| Full suite | 190 passed, 27 warnings; 94.92% coverage; 90% gate passed |
| Ruff check and format, mypy | Passed |
| Isolated build and Twine metadata | Passed |
| Runnable example with static exports | Passed |

Commands used from the repository root (the first isolated build attempt was blocked by sandbox network access; an approved retry succeeded):

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m ruff format --check src tests
.\.venv\Scripts\python.exe -m mypy src/pyautostat
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_profiling.py tests/test_profiling_metadata.py --disable-warnings
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --cov=pyautostat --cov-report=term-missing --cov-fail-under=90 --disable-warnings
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe -m twine check dist/*
.\.venv\Scripts\python.exe examples/example_usage.py --skip-interactive --output-dir "$env:TEMP\pyautostat-phase3-example-final"
```

Local baseline: 156 passed, no failures, 94.49% coverage. Final local: 190 passed, no failures, 94.92% coverage. Newly introduced local failures: **none**. The three Python 3.10 failures in the baseline CI are still tracked as pre-existing Phase 2 blockers. A final Phase 3 CI matrix has **not** run, so its result and the final Python 3.10 failure set cannot be asserted. Only Python 3.12 is installed locally. Do not interpret local success as proof that the entire CI matrix passed.

### Files and dependencies

Created `src/pyautostat/profiling.py`, `tests/test_profiling.py`, `tests/test_profiling_metadata.py`, and this record. Updated `src/pyautostat/analyzer.py`, `src/pyautostat/research_assistant.py`, `README.md`, `API_REFERENCE.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md`, `DEVELOPMENT_ROADMAP.md`, `examples/example_usage.py`, and `examples/README.md`. No runtime or development dependency, package version, release artifact, or publishing configuration changed.

### Known limits and Phase 4 handoff

Declared missing codes are visible but not normalized. High-cardinality categorical frequency tables show only the top 20 values while preserving total observed categories and omitted observation counts. Missingness patterns show the top 10 and co-missing pairs consider at most 20 columns with missingness. The legacy 30-bin peak flag is descriptive, sensitive to bins and sample size, and cannot prove population bimodality. Statistical design facts, missingness mechanisms, and causal meaning remain unknown. The Phase 2 Python 3.10 warning/Anderson expectations are separate blockers.

Phase 4 can reuse `variable_intelligence`, validated declarations, `complete_case_count()`, pairwise correlation sample sizes, and structured issues when collecting a research question and design facts. Those records are advisory; they do not supply an estimand, independence, pairing, or a method recommendation.

### Acceptance checklist

| Requirement group | Status |
| --- | --- |
| DataFrame-only profile, legacy dictionary keys, correct numerical descriptions, categorical summaries, type/role evidence, optional validated metadata | PASS locally |
| Missing cells versus affected rows, patterns, complete-case counts, exact duplicates and overlap | PASS locally |
| Outlier availability, thresholds, optional row offsets, distributions, histogram metadata | PASS locally |
| Pearson/Spearman/Kendall coefficients, pairwise counts, unavailable pairs, Pearson p-value guards | PASS locally |
| Structured warnings, source preservation, existing ReportGenerator and InsightEngine compatibility | PASS locally |
| New Phase 3 tests and complete local quality gates | PASS locally |
| Complete supported-Python CI matrix | INCOMPLETE: Phase 3 branch has no CI run; baseline Python 3.10 jobs were red |
| No Phase 4 or later functionality and no dependency/version change | PASS |
